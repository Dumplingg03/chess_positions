import torch
import chess
import numpy as np
import os
import time
import sys

from clearml import Task

from cgan_model import Generator
from chess_logic import ChessAnalyzer
from gui import ChessGUI


# --- НАСТРОЙКИ ---
STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "models/G_ttur_ep_19.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 64


# =========================
# LOG STORAGE
# =========================

log_data = {
    "target": [],
    "sdc": [],
    "error": [],
    "gap": [],
    "mobility": [],
    "real_moves": [],
    "gen_time": []
}


# =========================
# BOARD
# =========================

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


# =========================
# VALIDATION
# =========================

def is_logic_valid(board):
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return False

    if len(board.piece_map()) < 6:
        return False

    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    score = sum(
        values[p.piece_type] * (1 if p.color == chess.WHITE else -1)
        for p in board.piece_map().values()
    )

    return abs(score) <= 5


# =========================
# SDC CALCULATION
# =========================

def cp(score_obj):
    if score_obj is None:
        return 0
    s = score_obj.score(mate_score=10000)
    return np.clip(s if s is not None else 0, -2000, 2000)


def calculate_sdc(board, analyzer):
    try:
        info = analyzer.engine.analyse(
            board,
            chess.engine.Limit(depth=14),
            multipv=1
        )

        pv = info[0].get("pv", [])
        real_moves = max(len(pv), 2)

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

        mobility = board.legal_moves.count()

        m4 = (real_moves * 3.9) + (np.log1p(gap) * 0.5) - (mobility * 0.1)
        sdc = (m4 * 85.0) + 220

        return int(np.clip(sdc, 600, 3200)), gap, mobility, real_moves

    except:
        return 0, 0, 0, 0


# =========================
# LOGGING (5 GROUPS)
# =========================

def log_metrics(step):
    if len(log_data["sdc"]) == 0:
        return

    # =====================
    # Ratings
    # =====================
    logger.report_scalar("Ratings", "Target", iteration=step,
                         value=float(np.mean(log_data["target"])))

    logger.report_scalar("Ratings", "SDC", iteration=step,
                         value=float(np.mean(log_data["sdc"])))

    # =====================
    # Error
    # =====================
    logger.report_scalar("Error", "Abs Error", iteration=step,
                         value=float(np.mean(log_data["error"])))

    # =====================
    # Performance
    # =====================
    logger.report_scalar("Performance", "Generation Time", iteration=step,
                         value=float(np.mean(log_data["gen_time"])) if log_data["gen_time"] else 0)

    # =====================
    # Analysis
    # =====================
    logger.report_scalar("Analysis", "Gap", iteration=step,
                         value=float(np.mean(log_data["gap"])))

    logger.report_scalar("Analysis", "Mobility", iteration=step,
                         value=float(np.mean(log_data["mobility"])))

    logger.report_scalar("Analysis", "Real Moves", iteration=step,
                         value=float(np.mean(log_data["real_moves"])))

    # =====================
    # Stats
    # =====================
    if len(log_data["target"]) > 1:
        corr = np.corrcoef(log_data["target"], log_data["sdc"])[0, 1]
    else:
        corr = 0

    logger.report_scalar("Stats", "Correlation", iteration=step,
                         value=float(corr))

    success_rate = len(log_data["sdc"]) / max(1, step)

    logger.report_scalar("Stats", "Success Rate", iteration=step,
                         value=float(success_rate))


# =========================
# GENERATION
# =========================

def generate_puzzle(model, analyzer, target_rating):
    model.eval()

    start_time = time.time()

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

            sdc, gap, mobility, real_moves = calculate_sdc(board, analyzer)

            if abs(sdc - target_rating) <= 200:

                gen_time = time.time() - start_time

                log_data["target"].append(target_rating)
                log_data["sdc"].append(sdc)
                log_data["error"].append(abs(sdc - target_rating))
                log_data["gap"].append(gap)
                log_data["mobility"].append(mobility)
                log_data["real_moves"].append(real_moves)
                log_data["gen_time"].append(gen_time)

                print(f"🔥 SDC {sdc} | target {target_rating}")

                return fen, best_move, sdc

    return None, None, None


# =========================
# MAIN
# =========================

def main():
    print("🤖 FULL SDC RESEARCH PIPELINE STARTED")

    model = Generator(LATENT_DIM).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    analyzer = ChessAnalyzer(STOCKFISH_PATH)
    analyzer.start_engine()

    target_ratings = np.linspace(800, 2000, 20).astype(int)

    found = 0
    MAX = 20

    while found < MAX:

        target_rating = int(np.random.choice(target_ratings))

        fen, move, sdc = generate_puzzle(model, analyzer, target_rating)

        if fen:
            found += 1
            log_metrics(found)
            print(f"✔ {found}/{MAX}")

        else:
            time.sleep(0.05)

    analyzer.stop_engine()

    print("✅ DONE - FULL ANALYTICS IN CLEARML")
    sys.exit()


if __name__ == "__main__":
    main()