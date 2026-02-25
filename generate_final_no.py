import torch
import chess
import numpy as np
import os
import time
import pygame
from model import CVAE
from chess_logic import ChessAnalyzer
from gui import ChessGUI

STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "cvae_chess.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_piece_from_probs(probs):
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    # Снизил порог до 0.35, чтобы доска была насыщеннее
    if np.max(probs) < 0.35:
        return None
    idx = np.argmax(probs)
    p_type = piece_types[idx % 6]
    color = chess.WHITE if idx < 6 else chess.BLACK
    return chess.Piece(p_type, color)

def generate_puzzle_by_rating(model, analyzer, target_rating):
    model.eval()
    attempts = 0
    max_attempts = 1000 # Увеличено, так как фильтр стал строже

    while attempts < max_attempts:
        attempts += 1
        if attempts % 50 == 0:
            print(f"Поиск тактики... Попытка {attempts}")

        with torch.no_grad():
            z = torch.randn(1, 64).to(DEVICE)
            r_norm = torch.tensor([[(target_rating - 1000) / 1500]]).to(DEVICE).float()
            output = model.decode(z, r_norm).cpu().numpy()[0]

            board = chess.Board(None)
            for r_idx in range(8):
                for c_idx in range(8):
                    piece = get_piece_from_probs(output[:, r_idx, c_idx])
                    if piece:
                        board.set_piece_at(chess.square(c_idx, 7 - r_idx), piece)

            if len(board.pieces(chess.KING, chess.WHITE)) != 1 or \
               len(board.pieces(chess.KING, chess.BLACK)) != 1:
                continue

            num_pieces = len(board.piece_map())
            if num_pieces < 8 or num_pieces > 26: # Расширил диапазон для миттельшпиля
                continue

            fen = board.fen()
            if analyzer.is_puzzle_material(fen, target_rating):
                best_move, score = analyzer.get_best_move(fen, depth=16)
                if best_move:
                    print(f"Найдена комбинация! Попыток: {attempts}")
                    return fen, best_move, score

    return None, None, None

def main():
    print("Запуск системы генерации тактических задач...")
    model = CVAE().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Веса модели загружены.")
    else:
        print(f"Файл {MODEL_PATH} не найден!")
        return

    analyzer = ChessAnalyzer(STOCKFISH_PATH)
    analyzer.start_engine()
    gui = ChessGUI()

    user_rating = gui.get_rating_input()
    if not user_rating:
        analyzer.stop_engine()
        return

    while True:
        gui.screen.fill((30, 30, 30))
        txt = gui.font.render(f"Нейросеть ищет тактику {user_rating} Elo...", True, (200, 200, 200))
        gui.screen.blit(txt, (100, 300))
        pygame.display.flip()

        fen, move, score = generate_puzzle_by_rating(model, analyzer, user_rating)

        if fen:
            result = gui.run_puzzle(fen, analyzer, user_rating)
            if result == "QUIT": break
        else:
            print("Не удалось найти сложную задачу. Попробуем еще раз...")
            time.sleep(1)

    analyzer.stop_engine()
    pygame.quit()

if __name__ == "__main__":
    main()