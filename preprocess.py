import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def fen_to_matrix(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    # 12 слоев: 6 для белых, 6 для черных
    matrix = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            # Индекс: 0-5 для белых, 6-11 для черных
            idx = PIECE_TYPES.index(piece.piece_type)
            if piece.color == chess.BLACK:
                idx += 6

            matrix[idx, row, col] = 1.0

    return matrix


def matrix_to_fen(matrix: np.ndarray) -> str:
    board = chess.Board(None)
    for idx in range(12):
        piece_type = PIECE_TYPES[idx % 6]
        color = chess.WHITE if idx < 6 else chess.BLACK
        rows, cols = np.where(matrix[idx] > 0.5)
        for r, c in zip(rows, cols):
            square = chess.square(c, 7 - r)
            board.set_piece_at(square, chess.Piece(piece_type, color))
    return board.fen()