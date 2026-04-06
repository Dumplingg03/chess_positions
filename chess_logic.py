# chess_logic.py
import chess
import chess.engine


class ChessAnalyzer:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None

    def start_engine(self):
        """Запуск Stockfish"""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        except Exception as e:
            print(f"❌ Ошибка запуска Stockfish: {e}")

    def stop_engine(self):
        """Остановка Stockfish"""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None

    def get_best_move(self, fen, depth=12, time_limit=0.2):
        """
        Возвращает лучший ход и оценку позиции
        :param fen: FEN позиции
        :param depth: глубина анализа
        :param time_limit: время на анализ
        :return: tuple(best_move, score)
        """
        try:
            board = chess.Board(fen)
            result = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth, time=time_limit)
            )
            best_move = result["pv"][0] if "pv" in result else None
            score = result["score"].relative.score(mate_score=10000)
            return best_move, score if score is not None else 0
        except Exception:
            return None, 0

    def is_puzzle_material(self, fen, target_rating=1500, depth=12, time_limit=0.1):
        """
        Проверяет, подходит ли позиция для шахматной задачи (такт. позиция)
        :param fen: FEN позиции
        :param target_rating: примерная сложность позиции
        :return: True/False
        """
        try:
            board = chess.Board(fen)

            # Базовая проверка позиции
            if not board.is_valid() or board.is_check() or board.is_game_over():
                return False

            # Анализ через движок с multipv=2
            info = self.engine.analyse(board,
                                       chess.engine.Limit(depth=depth, time=time_limit),
                                       multipv=2)
            if len(info) < 2:
                return False

            # Считаем оценки первого и второго ходов
            score1_obj = info[0]["score"].relative
            score2_obj = info[1]["score"].relative

            s1 = score1_obj.score(mate_score=10000)
            s2 = score2_obj.score(mate_score=10000)
            if s1 is None or s2 is None:
                return False

            # Проверка на быстрый мат
            if score1_obj.is_mate():
                if score2_obj.is_mate():
                    # Если оба ведут к мату почти одинаково → не уникальная задача
                    if abs(s1 - s2) < 100:
                        return False
                return True

            # Материальный баланс (P=1,N/B=3,R=5,Q=9)
            def get_material_sum(b):
                return sum(
                    (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK))) * val
                    for pt, val in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                                    (chess.ROOK, 5), (chess.QUEEN, 9)]
                )

            balance = abs(get_material_sum(board))

            # Если слишком большое преимущество → это не тактика
            if balance > 5:
                return False

            # Разрыв между первым и вторым ходом должен быть значительным
            if s1 < 300 or (s1 - s2) < 250:
                return False

            # Минимальная длина варианта (не один ход)
            if len(info[0]["pv"]) < 3:
                return False

            return True

        except Exception:
            return False