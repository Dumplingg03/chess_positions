import chess
import chess.engine


class PuzzleValidator:
    def __init__(self, path_to_stockfish):
        self.engine = chess.engine.SimpleEngine.popen_uci(path_to_stockfish)

    def is_valid_puzzle(self, fen):
        board = chess.Board(fen)
        # 1. Позиция должна быть близка к равной ДО хода игрока (не больше 1.0 пешки)
        info_before = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        score_before = info_before["score"].relative.score(mate_score=10000)
        if score_before is None or abs(score_before) > 100:
            return False

        # 2. Проверяем, есть ли один "взрывной" ход (MultiPV=2)
        info = self.engine.analyse(board, chess.engine.Limit(time=0.3), multipv=2)
        if len(info) < 2: return False

        score1 = info[0]["score"].relative.score(mate_score=10000)
        score2 = info[1]["score"].relative.score(mate_score=10000)

        # 3. Разрыв должен быть огромным (лучший ход дает +4, остальные - около нуля)
        if score1 > 300 and score2 < 100:
            return True  # Это похоже на тактику!

        return False

    def close(self):
        self.engine.quit()