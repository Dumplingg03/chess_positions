import torch
import chess
import chess.engine
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


# ==========================================
# 1. BOARD
# ==========================================

def get_board_from_output(output):
    board = chess.Board(None)

    piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K',
                     'p', 'n', 'b', 'r', 'q', 'k']

    prediction = np.argmax(output, axis=0)
    probabilities = np.max(output, axis=0)

    for r in range(8):
        for c in range(8):
            if probabilities[r, c] > 0.4:
                idx = prediction[r, c]
                symbol = piece_symbols[idx]
                board.set_piece_at(
                    chess.square(c, 7 - r),
                    chess.Piece.from_symbol(symbol)
                )

    # короли
    for color, king_idx, king_sym in [
        (chess.WHITE, 5, 'K'),
        (chess.BLACK, 11, 'k')
    ]:
        if board.king(color) is None:
            king_layer = output[king_idx]
            idx = np.argmax(king_layer)
            r_k, c_k = divmod(idx, 8)
            board.set_piece_at(
                chess.square(c_k, 7 - r_k),
                chess.Piece.from_symbol(king_sym)
            )

    return board


def is_logic_valid(board):
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return False

    if len(board.piece_map()) < 6:
        return False

    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    score = sum(
        values[p.piece_type] * (1 if p.color == chess.WHITE else -1)
        for p in board.piece_map().values()
    )

    return abs(score) <= 5


# ==========================================
# 2. SDC FORMULA (НАША ВЕРСИЯ)
# ==========================================

def cp(score_obj):
    if score_obj is None:
        return 0
    s = score_obj.score(mate_score=10000)
    return np.clip(s if s is not None else 0, -2000, 2000)


def calculate_sdc(board, analyzer):
    try:
        # PV depth
        info = analyzer.engine.analyse(
            board,
            chess.engine.Limit(depth=14),
            multipv=1
        )

        pv = info[0].get("pv", [])
        real_moves = max(len(pv), 2)

        # GAP
        info_gap = analyzer.engine.analyse(
            board,
            chess.engine.Limit(depth=10),
            multipv=2
        )

        if len(info_gap) < 2:
            gap = 0
        else:
            s1 = cp(info_gap[0]["score"].pov(board.turn))
            s2 = cp(info_gap[1]["score"].pov(board.turn))
            gap = abs(s1 - s2)

        # mobility
        mobility = board.legal_moves.count()

        # Сначала считаем чистый коэффициент сложности
        m4 = (real_moves * 3.90 ) + (np.log1p(gap) * 0.5) - (mobility * 0.1)

        # Затем переводим его в шахматный рейтинг (Scale + Offset)
        sdc_rating = (m4 * 85.0) + 220

        return int(np.clip(sdc_rating, 600, 3200))

    except:
        return 0


# ==========================================
# 3. GENERATION
# ==========================================

def generate_puzzle(model, analyzer, target_rating):
    model.eval()

    for _ in range(15000):
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            r_norm = torch.tensor(
                [[(target_rating - 500) / 2500]]
            ).float().to(DEVICE)

            output = model(z, r_norm).cpu().numpy()[0]
            board = get_board_from_output(output)

            if not is_logic_valid(board):
                continue

            fen = board.fen()

            if not analyzer.is_puzzle_material(fen, target_rating):
                continue

            best_move, _ = analyzer.get_best_move(fen, depth=16)

            if not best_move:
                continue

            # gap check
            info = analyzer.engine.analyse(
                board,
                chess.engine.Limit(depth=14),
                multipv=2
            )

            if len(info) > 1:
                s1 = cp(info[0]["score"].relative)
                s2 = cp(info[1]["score"].relative)

                if abs(s1 - s2) < 100:
                    continue

            sdc = calculate_sdc(board, analyzer)

            if abs(sdc - target_rating) <= 200:
                print(f"🔥 FOUND SDC: {sdc} target: {target_rating}")
                return fen, best_move, sdc

    return None, None, None


# ==========================================
# 4. MAIN
# ==========================================

def main():
    print("🤖 SDC Generator started")

    model = Generator(LATENT_DIM).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    analyzer = ChessAnalyzer(STOCKFISH_PATH)
    analyzer.start_engine()

    gui = ChessGUI()

    target_rating = gui.get_rating_input()
    if target_rating is None:
        analyzer.stop_engine()
        return

    while True:
        gui.show_waiting_screen(target_rating)

        fen, move, sdc = generate_puzzle(model, analyzer, target_rating)

        if fen:
            res = gui.run_puzzle(fen, analyzer, sdc)
            if res == "QUIT":
                break
        else:
            print("retry...")
            time.sleep(0.1)

    analyzer.stop_engine()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()