import torch
import chess
import numpy as np
import os
import time
import pygame
import sys

from cgan_model import Generator
from chess_logic import ChessAnalyzer
from gui import ChessGUI

# --- НАСТРОЙКИ ---
STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "models/G_ttur_ep_19.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 64

def get_board_from_output(output):
    board = chess.Board(None)
    piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    prediction = np.argmax(output, axis=0)
    probabilities = np.max(output, axis=0)
    for r in range(8):
        for c in range(8):
            if probabilities[r, c] > 0.4:
                idx = prediction[r, c]
                symbol = piece_symbols[idx]
                board.set_piece_at(chess.square(c, 7 - r), chess.Piece.from_symbol(symbol))
    # форсируем королей
    for color, king_idx, king_sym in [(chess.WHITE, 5, 'K'), (chess.BLACK, 11, 'k')]:
        if not board.pieces(chess.KING, color):
            king_layer = output[king_idx]
            idx = np.argmax(king_layer)
            r_k, c_k = divmod(idx, 8)
            board.set_piece_at(chess.square(c_k, 7 - r_k), chess.Piece.from_symbol(king_sym))
    return board

def get_material_score(board):
    values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = values[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    return score

def is_logic_valid(board):
    balance = get_material_score(board)
    if abs(balance) > 3:
        return False
    if len(board.pieces(chess.QUEEN, chess.WHITE)) > 1: return False
    if len(board.pieces(chess.QUEEN, chess.BLACK)) > 1: return False
    if len(board.piece_map()) < 6: return False
    return True

def is_interesting_move(board, move, analyzer):
    to_square = move.to_square
    # Слишком простое взятие
    if board.is_capture(move):
        if not board.attackers(not board.turn, to_square):
            return False
    board.push(move)
    info = analyzer.engine.analyse(board, chess.engine.Limit(depth=12))
    board.pop()
    mate = info.get("score").relative.mate()
    if mate and 1 <= abs(mate) <= 2:
        return True
    score = info.get("score").relative.score(mate_score=10000)
    if score is not None and score >= 200:
        return True
    return False

def generate_puzzle_by_rating(model, analyzer, target_rating, gui):
    model.eval()
    attempts = 0
    max_attempts = 2000
    while attempts < max_attempts:
        attempts += 1
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            r_norm = torch.tensor([[(target_rating - 500) / 2500]]).to(DEVICE).float()
            output = model(z, r_norm).cpu().numpy()[0]
            board = get_board_from_output(output)
            if not is_logic_valid(board):
                continue
            fen = board.fen()
            if analyzer.is_puzzle_material(fen, target_rating):
                best_move, score = analyzer.get_best_move(fen, depth=16)
                if best_move:
                    if not is_interesting_move(board, best_move, analyzer):
                        continue
                    # multipv проверка разницы в линиях
                    info = analyzer.engine.analyse(board, chess.engine.Limit(depth=14), multipv=2)
                    if len(info) > 1:
                        score1 = info[0]['score'].relative.score(mate_score=10000)
                        score2 = info[1]['score'].relative.score(mate_score=10000)
                        if abs(score1 - score2) < 200:
                            continue
                    print(f"🔥 Найдена реальная комбинация! Разница в линиях: {abs(score1 - score2)}")
                    return fen, best_move, score
    return None, None, None

def main():
    print("🤖 Запуск шахматного генератора (CGAN Edition)...")
    model = Generator(LATENT_DIM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Веса CGAN {MODEL_PATH} загружены.")
    else:
        print(f"❌ Ошибка: Файл {MODEL_PATH} не найден!")
        return

    analyzer = ChessAnalyzer(STOCKFISH_PATH)
    analyzer.start_engine()
    gui = ChessGUI()

    user_rating = gui.get_rating_input()
    if user_rating is None:
        analyzer.stop_engine()
        return

    while True:
        gui.show_waiting_screen(user_rating)
        fen, move, score = generate_puzzle_by_rating(model, analyzer, user_rating, gui)

        if fen == "QUIT": break

        if fen:
            result = gui.run_puzzle(fen, analyzer, user_rating)
            if result == "QUIT": break
        else:
            print("Не удалось найти комбинацию. Пробуем еще раз...")
            time.sleep(0.1)

    analyzer.stop_engine()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()