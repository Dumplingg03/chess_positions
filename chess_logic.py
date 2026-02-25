import chess
import chess.engine

class ChessAnalyzer:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None

    def start_engine(self):
        if self.engine:
            try: self.engine.quit()
            except: pass
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        except Exception as e:
            print(f"Ошибка запуска Stockfish: {e}")

    def stop_engine(self):
        if self.engine:
            try: self.engine.quit()
            except: pass
            self.engine = None

    def get_best_move(self, fen, depth=14):
        try:
            board = chess.Board(fen)
            result = self.engine.analyse(board, chess.engine.Limit(time=0.1, depth=depth))
            best_move = result["pv"][0] if "pv" in result else None
            score = result["score"].relative.score(mate_score=10000)
            return best_move, score
        except:
            return None, 0

    def is_puzzle_material(self, fen, target_rating):
        try:
            board = chess.Board(fen)
            if not board.is_valid() or board.is_game_over(): return False

            # Настройки под рейтинг
            if target_rating < 1400:
                depth, min_gap, max_init = 12, 350, 150
            elif target_rating < 1900:
                depth, min_gap, max_init = 16, 500, 100
            else:
                depth, min_gap, max_init = 20, 750, 80

            # 1. СТРОГИЙ ОТБОР: Позиция должна быть почти равной
            init_info = self.engine.analyse(board, chess.engine.Limit(depth=depth, time=0.1))
            init_score = init_info["score"].relative.score(mate_score=10000)
            if init_score is None or abs(init_score) > max_init: return False

            # 2. АНАЛИЗ ХОДОВ (MultiPV=2)
            info = self.engine.analyse(board, chess.engine.Limit(depth=depth, time=0.3), multipv=2)
            if len(info) < 2: return False

            score1 = info[0]["score"].relative.score(mate_score=10000)
            score2 = info[1]["score"].relative.score(mate_score=10000)
            gap = abs(score1 - (score2 if score2 is not None else -10000))

            # Единственность решения
            if gap < min_gap: return False

            best_move = info[0]["pv"][0]
            is_capture = board.is_capture(best_move)
            is_check = board.gives_check(best_move)

            # 3. АНТИ-ЗЕВОК: Если мы просто рубим незащищенную фигуру — это не задача
            if is_capture and not is_check:
                target_sq = best_move.to_square
                defenders = board.attackers(not board.turn, target_sq)
                if not defenders and score1 < 1000: # Если перевес после взятия не матовый
                    return False

            # 4. ТАКТИЧЕСКИЙ ПРИЗНАК: Проверяем длину линии (глубина комбинации)
            if len(info[0]["pv"]) < 3: return False

            return True
        except:
            return False

    def _count_material(self, board):
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        balance = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                val = values.get(piece.piece_type, 0)
                balance += val if piece.color == chess.WHITE else -val
        return abs(balance)