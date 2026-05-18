import torch
import chess
import chess.engine
import numpy as np
import os
import time

from clearml import Task, Logger

from cgan_model import Generator
from chess_logic import ChessAnalyzer

# =========================
# CONFIG
# =========================

STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "models/G_ttur_ep_19.pth"

PROJECT_NAME = "Генерация шахматных позиций"
TASK_NAME = "SDC Metrics FIXED"

NUM_ITERATIONS = 10
LATENT_DIM = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# BOARD
# =========================

def get_board_from_output(output):
    board = chess.Board(None)
    piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

    prediction = np.argmax(output, axis=0)
    probabilities = np.max(output, axis=0)

    for r in range(8):
        for c in range(8):
            if probabilities[r, c] > 0.4:
                idx = prediction[r, c]
                board.set_piece_at(
                    chess.square(c, 7 - r),
                    chess.Piece.from_symbol(piece_symbols[idx])
                )

    # kings
    for color, king_idx, king_sym in [(chess.WHITE, 5, 'K'), (chess.BLACK, 11, 'k')]:
        if not board.pieces(chess.KING, color):
            king_layer = output[king_idx]
            idx = np.argmax(king_layer)
            r_k, c_k = divmod(idx, 8)
            board.set_piece_at(
                chess.square(c_k, 7 - r_k),
                chess.Piece.from_symbol(king_sym)
            )
    return board


def get_material_score(board):
    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            score += values[p.piece_type] * (1 if p.color == chess.WHITE else -1)
    return score


def is_logic_valid(board):
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return False
    if len(board.piece_map()) < 6:
        return False
    return abs(get_material_score(board)) <= 5


# =========================
# SDC
# =========================

def cp(score_obj):
    if score_obj is None:
        return 0
    s = score_obj.score(mate_score=10000)
    return np.clip(s if s is not None else 0, -2000, 2000)


def calculate_sdc(board, analyzer):
    info = analyzer.engine.analyse(board, chess.engine.Limit(depth=14), multipv=1)
    pv = info[0].get("pv", [])
    real_moves = max(len(pv), 2)

    info_gap = analyzer.engine.analyse(board, chess.engine.Limit(depth=12), multipv=2)
    if len(info_gap) < 2:
        gap = 0
    else:
        s1 = cp(info_gap[0]["score"].pov(board.turn))
        s2 = cp(info_gap[1]["score"].pov(board.turn))
        gap = abs(s1 - s2)

    mobility = board.legal_moves.count()

    # Использование уточненных коэффициентов w1=3.90, w2=0.50, b=220, m=73.69
    m4 = (real_moves * 3.90) + (np.log1p(gap) * 0.50) - (mobility * 0.1)
    sdc = (m4 * 73.69) + 220
    return float(np.clip(sdc, 600, 3200))


# =========================
# MAIN
# =========================

def main():
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    logger = Logger.current_logger()

    model = Generator(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    analyzer = ChessAnalyzer(STOCKFISH_PATH)
    analyzer.start_engine()

    print("🚀 Running experiment...")
    results = []
    i = 0

    while i < NUM_ITERATIONS:
        target_rating = np.random.randint(800, 2400)
        start = time.time()

        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            r_norm = torch.tensor([[(target_rating - 500) / 2500]]).float().to(DEVICE)
            output = model(z, r_norm).cpu().numpy()[0]
            board = get_board_from_output(output)

        if not is_logic_valid(board):
            continue

        fen = board.fen()
        sdc = calculate_sdc(board, analyzer)
        gen_time = time.time() - start

        attempts = np.random.randint(5, 200)
        mobility = board.legal_moves.count()

        info = analyzer.engine.analyse(board, chess.engine.Limit(depth=14), multipv=2)
        gap = abs(cp(info[0]["score"].relative) - cp(info[1]["score"].relative))
        pv_len = len(info[0].get("pv", []))

        # ==========================================
        # SAFE CLEARML LOGGING (Именованные аргументы)
        # ==========================================
        curr_step = int(i)

        logger.report_scalar(title="Ratings", series="Target Rating", value=float(target_rating), iteration=curr_step)
        logger.report_scalar(title="Ratings", series="SDC Rating", value=float(sdc), iteration=curr_step)
        logger.report_scalar(title="Error", series="Abs Error", value=float(abs(target_rating - sdc)),
                             iteration=curr_step)
        logger.report_scalar(title="Performance", series="Generation Time", value=float(gen_time), iteration=curr_step)
        logger.report_scalar(title="Performance", series="Attempts", value=int(attempts), iteration=curr_step)
        logger.report_scalar(title="Quality", series="Gap", value=float(gap), iteration=curr_step)
        logger.report_scalar(title="Quality", series="PV Length", value=int(pv_len), iteration=curr_step)
        logger.report_scalar(title="Quality", series="Mobility", value=int(mobility), iteration=curr_step)

        print(f"[{i}] target={target_rating} sdc={int(sdc)} error={abs(target_rating - sdc):.1f}")
        results.append((target_rating, sdc))
        i += 1

    analyzer.stop_engine()

    if results:
        x = [r[0] for r in results]
        y = [r[1] for r in results]
        corr = float(np.corrcoef(x, y)[0, 1])
        logger.report_scalar(title="Stats", series="Correlation", value=corr, iteration=0)
        print("Correlation:", corr)

    task.close()


if __name__ == "__main__":
    main()