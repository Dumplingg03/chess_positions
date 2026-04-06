import os
import torch
import chess
import numpy as np
import pygame
import time
from clearml import Task, Logger

from cgan_model import Generator
from chess_logic import ChessAnalyzer

# --- НАСТРОЙКИ ---
STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "models/G_ttur_ep_19.pth"
PROJECT_NAME = "Генерация шахматных позиций"
TASK_NAME = "CGAN + Логика Stockfish"
NUM_ITERATIONS = 50  # Можно ставить 50-100
LATENT_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIECES_PATH = "pieces/"
TILE_SIZE = 60
PIECE_SYMBOLS = ['P','N','B','R','Q','K','p','n','b','r','q','k']

# ---------- Функции ----------
def get_board_from_output(output):
    board = chess.Board(None)
    prediction = np.argmax(output, axis=0)
    probabilities = np.max(output, axis=0)
    for r in range(8):
        for c in range(8):
            if probabilities[r, c] > 0.4:
                idx = prediction[r, c]
                symbol = PIECE_SYMBOLS[idx]
                board.set_piece_at(chess.square(c, 7 - r), chess.Piece.from_symbol(symbol))
    # Форсируем королей
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
            score += values[piece.piece_type] if piece.color == chess.WHITE else -values[piece.piece_type]
    return score

def is_logic_valid(board):
    balance = get_material_score(board)
    if abs(balance) > 3: return False
    if len(board.pieces(chess.QUEEN, chess.WHITE)) > 1: return False
    if len(board.pieces(chess.QUEEN, chess.BLACK)) > 1: return False
    if len(board.piece_map()) < 6: return False
    return True

def is_interesting_move(board, move, analyzer):
    to_square = move.to_square
    if board.is_capture(move) and not board.attackers(not board.turn, to_square):
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

def generate_puzzle(model, analyzer, target_rating):
    model.eval()
    attempts = 0
    max_attempts = 2000
    start_time = time.time()
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
            if best_move and is_interesting_move(board, best_move, analyzer):
                # multipv проверка
                info = analyzer.engine.analyse(board, chess.engine.Limit(depth=14), multipv=2)
                if len(info) > 1:
                    score1 = info[0]['score'].relative.score(mate_score=10000)
                    score2 = info[1]['score'].relative.score(mate_score=10000)
                    if abs(score1 - score2) < 200:
                        continue
                return fen, best_move, score, attempts, time.time()-start_time
    return None, None, None, attempts, time.time()-start_time

def load_piece_images():
    images = {}
    piece_map = {
        'P':'wp','N':'wn','B':'wb','R':'wr','Q':'wq','K':'wk',
        'p':'bp','n':'bn','b':'bb','r':'br','q':'bq','k':'bk'
    }
    for sym,file in piece_map.items():
        path = os.path.join(PIECES_PATH,f"{file}.png")
        if os.path.exists(path):
            img = pygame.image.load(path)
            images[sym] = pygame.transform.smoothscale(img,(TILE_SIZE,TILE_SIZE))
    return images

def draw_board_pieces(screen, board, piece_images):
    colors = [(240,217,181),(181,136,99)]
    screen.fill((255,255,255))
    for r in range(8):
        for c in range(8):
            pygame.draw.rect(screen, colors[(r+c)%2], (c*TILE_SIZE,r*TILE_SIZE,TILE_SIZE,TILE_SIZE))
    for square,piece in board.piece_map().items():
        symbol = piece.symbol()
        if symbol in piece_images:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            screen.blit(piece_images[symbol], (col*TILE_SIZE,row*TILE_SIZE))

# ---------- MAIN ----------
def main():
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    logger = Logger.current_logger()

    pygame.init()
    screen = pygame.Surface((TILE_SIZE*8,TILE_SIZE*8))
    piece_images = load_piece_images()

    model = Generator(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model.eval()

    analyzer = ChessAnalyzer(STOCKFISH_PATH)
    analyzer.start_engine()

    print(f"🚀 Генерация {NUM_ITERATIONS} случайных рейтингов и шахматных задач...")

    for i in range(1, NUM_ITERATIONS+1):
        rating = np.random.randint(800, 2400)
        fen, move, score, attempts, gen_time = generate_puzzle(model, analyzer, rating)

        logger.report_scalar("Performance", "Generation Time (s)", iteration=i, value=gen_time)
        logger.report_scalar("Performance", "Attempts", iteration=i, value=attempts)
        logger.report_scalar("Performance", "Target Rating", iteration=i, value=rating)

        if fen:
            board = chess.Board(fen)
            draw_board_pieces(screen, board, piece_images)
            img_name = f"board_{i}.png"
            pygame.image.save(screen,img_name)
            logger.report_image("Boards", f"Board {i}", iteration=i, local_path=img_name)
            if score is not None:
                logger.report_scalar("Quality", "Tactical Score", iteration=i, value=score/100.0)

    analyzer.stop_engine()
    pygame.quit()
    task.close()
    print("✅ Генерация завершена.")

if __name__=="__main__":
    main()