# chess_logic.py

import chess
import chess.engine
import random



class ChessAnalyzer:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None

    # ==========================================
    # ENGINE
    # ==========================================

    def start_engine(self):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass

        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        print("✅ Stockfish started")

    def stop_engine(self):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None

    # ==========================================
    # BASIC
    # ==========================================

    def cp(self, score_obj):
        if score_obj is None:
            return 0

        score = score_obj.score(mate_score=10000)

        if score is None:
            return 0

        if score > 3000:
            score = 3000
        if score < -3000:
            score = -3000

        return score

    def get_material_balance(self, board):
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        score = 0

        for piece_type, val in values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * val
            score -= len(board.pieces(piece_type, chess.BLACK)) * val

        return score

    # ==========================================
    # BEST MOVE
    # ==========================================

    def get_best_move(self, fen, depth=14, time_limit=0.15):
        try:
            board = chess.Board(fen)

            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth, time=time_limit)
            )

            move = info["pv"][0]
            score = self.cp(info["score"].relative)

            return move, score

        except:
            return None, 0

    # ==========================================
    # MAIN FILTER
    # ==========================================

    def is_puzzle_material(self, fen, target_rating=1500, depth=10, time_limit=0.08):
        """
        Главный фильтр тактических задач
        Быстрый и сбалансированный
        """

        try:
            board = chess.Board(fen)

            # ----------------------------------
            # БАЗОВАЯ ВАЛИДАЦИЯ
            # ----------------------------------

            if not board.is_valid():
                return False

            if board.is_game_over():
                return False

            if board.king(chess.WHITE) is None:
                return False

            if board.king(chess.BLACK) is None:
                return False

            if len(board.piece_map()) < 6:
                return False

            # ----------------------------------
            # МАТЕРИАЛ
            # ----------------------------------

            balance = abs(self.get_material_balance(board))

            if balance > 5:
                return False

            # ----------------------------------
            # ANALYSE
            # ----------------------------------

            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth, time=time_limit),
                multipv=2
            )

            if len(info) < 2:
                return False

            best = info[0]
            second = info[1]

            score1_obj = best["score"].relative
            score2_obj = second["score"].relative

            s1 = self.cp(score1_obj)
            s2 = self.cp(score2_obj)

            gap = abs(s1 - s2)

            # ----------------------------------
            # МАТ В 1 ХОД ИНОГДА ОСТАВЛЯЕМ
            # ----------------------------------

            if score1_obj.is_mate():
                mate = score1_obj.mate()

                if mate is None:
                    return False

                mate = abs(mate)

                # мат в 1 ход только иногда
                if mate == 1:
                    return random.random() < 0.25

                # мат 2-4 допускаем
                if mate <= 4:
                    return True

                return False

            # ----------------------------------
            # РЕЙТИНГОВЫЕ ПОРОГИ
            # ----------------------------------

            if target_rating <= 1000:
                min_score = 120
                min_gap = 80

            elif target_rating <= 1600:
                min_score = 180
                min_gap = 120

            elif target_rating <= 2200:
                min_score = 250
                min_gap = 180

            else:
                min_score = 320
                min_gap = 250

            if s1 < min_score:
                return False

            if gap < min_gap:
                return False

            # ----------------------------------
            # ДЛИНА ЛИНИИ
            # ----------------------------------

            pv = best.get("pv", [])

            if len(pv) < 2:
                return False

            # ----------------------------------
            # МОБИЛЬНОСТЬ
            # ----------------------------------

            moves = board.legal_moves.count()

            if moves < 2:
                return False

            return True

        except:
            return False