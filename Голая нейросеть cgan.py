import os
import torch
import chess
import chess.engine
import numpy as np
import pygame
from clearml import Task, Logger

os.environ['CLEARML_API_DEFAULT_TIMEOUT'] = '60'

# --- НАСТРОЙКИ ---
STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "models/G_ttur_ep_19.pth"
PROJECT_NAME = "Генерация шахматных позиций"
TASK_NAME = "CGAN_голая нейросеть"
NUM_SAMPLES = 100
LATENT_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIECES_PATH = "pieces/"
TILE_SIZE = 60

from cgan_model import Generator

PIECE_SYMBOLS = ['P','N','B','R','Q','K','p','n','b','r','q','k']

def get_board_from_output(output):
    """Преобразует выход GAN [12,8,8] в шахматную доску chess.Board"""
    board = chess.Board(None)
    prediction = np.argmax(output, axis=0)
    probabilities = np.max(output, axis=0)

    for r in range(8):
        for c in range(8):
            if probabilities[r,c] > 0.4:
                idx = prediction[r,c]
                symbol = PIECE_SYMBOLS[idx]
                board.set_piece_at(chess.square(c, 7-r), chess.Piece.from_symbol(symbol))

    # Форсированный ремонт королей
    for color, king_idx, king_sym in [(chess.WHITE, 5,'K'),(chess.BLACK,11,'k')]:
        if not board.pieces(chess.KING, color):
            king_layer = output[king_idx]
            idx = np.argmax(king_layer)
            r_k,c_k = divmod(idx,8)
            board.set_piece_at(chess.square(c_k,7-r_k), chess.Piece.from_symbol(king_sym))

    return board

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
        else:
            print(f"⚠️ Файл не найден: {path}")
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

def main():
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    logger = Logger.current_logger()

    pygame.init()
    screen = pygame.Surface((TILE_SIZE*8,TILE_SIZE*8))
    piece_images = load_piece_images()

    model = Generator(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model.eval()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    valid_count = 0

    print(f"🚀 Генерация {NUM_SAMPLES} позиций и логирование графиков...")

    for i in range(1, NUM_SAMPLES+1):
        with torch.no_grad():
            z = torch.randn(1,LATENT_DIM).to(DEVICE)
            r = torch.rand(1,1).to(DEVICE)
            output = model(z,r).cpu().numpy()[0]
            board = get_board_from_output(output)

        is_legal = board.is_valid()
        if is_legal:
            valid_count += 1
            try:
                info = engine.analyse(board, chess.engine.Limit(time=0.05))
                score = info["score"].relative.score(mate_score=10000)
                if score is not None:
                    score = score / 100.0  # масштабируем
                    logger.report_scalar("Quality","Tactical Score",iteration=i,value=score)
            except:
                pass

        # Легальность позиций
        logger.report_scalar("Performance","Legal Rate (%)",iteration=i,value=(valid_count/i)*100)

        # Визуализация первых 10 досок
        if i <= 10:
            draw_board_pieces(screen, board, piece_images)
            img_name = f"sample_{i}.png"
            pygame.image.save(screen,img_name)
            logger.report_image("Visual Check",f"Board {i}",iteration=i,local_path=img_name)

    engine.quit()
    pygame.quit()
    task.close()
    print("Готово.")

if __name__=="__main__":
    main()