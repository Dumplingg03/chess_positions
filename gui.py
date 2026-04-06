import pygame
import chess
import os

WIDTH, HEIGHT = 600, 650
SQ_SIZE = 600 // 8
MAX_PUZZLE_MOVES = 5


class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Chess Puzzle Generator")
        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.pieces = {}
        self.selected_sq = None
        self.flipped = False
        self.error_flash = 0
        self.load_assets()

    def load_assets(self):
        # Папка 'pieces' должна содержать файлы wP.png, wN.png и т.д.
        pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
        for p in pieces:
            path = os.path.join("pieces", p + ".png")
            try:
                img = pygame.image.load(path)
                self.pieces[p] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
            except:
                # Если картинок нет, рисуем красные квадраты
                surf = pygame.Surface((SQ_SIZE, SQ_SIZE))
                surf.fill((200, 0, 0))
                self.pieces[p] = surf

    def get_rating_input(self):
        input_box = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 25, 200, 50)
        text = '1500'  # Значение по умолчанию
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return int(text) if text else 1500
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif event.unicode.isdigit():
                        if len(text) < 4: text += event.unicode

            self.screen.fill((30, 30, 30))
            prompt = self.font.render("Введите рейтинг (1000-2600):", True, (255, 255, 255))
            self.screen.blit(prompt, (WIDTH // 2 - 140, HEIGHT // 2 - 70))
            pygame.draw.rect(self.screen, (100, 100, 100), input_box, 2)
            val_surf = self.font.render(text + "|", True, (255, 255, 255))
            self.screen.blit(val_surf, (input_box.x + 10, input_box.y + 10))
            pygame.display.flip()

    def show_waiting_screen(self, rating):
        """ Окно ожидания, пока Stockfish фильтрует задачи """
        self.screen.fill((40, 44, 52))
        text = self.font.render(f"ИИ ищет тактику {rating} Elo...", True, (200, 200, 200))
        sub = pygame.font.SysFont("Arial", 16).render("Это может занять 10-30 секунд", True, (150, 150, 150))
        self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 20))
        self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 20))
        pygame.display.flip()

    def draw_board(self, solved=False):
        colors = [pygame.Color("#eeeed2"), pygame.Color("#769656")]
        for r in range(8):
            for c in range(8):
                color = colors[((r + c) % 2)]
                pygame.draw.rect(self.screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
                if self.selected_sq == (r, c):
                    pygame.draw.rect(self.screen, (186, 202, 43, 150),
                                     pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        if self.error_flash > 0:
            s = pygame.Surface((600, 600))
            s.set_alpha(min(self.error_flash, 150))
            s.fill((255, 0, 0))
            self.screen.blit(s, (0, 0))
            self.error_flash -= 10

    def draw_pieces(self, board):
        for r in range(8):
            for c in range(8):
                # Корректный маппинг координат с учетом flipped
                display_r = 7 - r if self.flipped else r
                display_c = 7 - c if self.flipped else c
                piece = board.piece_at(chess.square(display_c, 7 - display_r))
                if piece:
                    name = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
                    if name in self.pieces:
                        self.screen.blit(self.pieces[name], (c * SQ_SIZE, r * SQ_SIZE))

    def show_message(self, message, sub_message=""):
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 600, WIDTH, 50))
        text_surf = self.font.render(message, True, (255, 255, 255))
        self.screen.blit(text_surf, (20, 605))
        if sub_message:
            sub_surf = pygame.font.SysFont("Arial", 14).render(sub_message, True, (180, 180, 180))
            self.screen.blit(sub_surf, (20, 630))

    def draw_button(self, text, rect, color):
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        pygame.draw.rect(self.screen, (200, 200, 200), rect, 2, border_radius=5)
        txt = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(txt,
                         (rect.x + (rect.width - txt.get_width()) // 2, rect.y + (rect.height - txt.get_height()) // 2))

    def run_puzzle(self, fen, analyzer, target_rating):
        board = chess.Board(fen)
        self.flipped = (board.turn == chess.BLACK)
        self.moves_made = 0
        running = True
        solved = False
        hint_sqs = None
        msg = "Ваш ход! Найдите выигрыш."

        # Координаты кнопок
        new_puzzle_btn = pygame.Rect(WIDTH - 130, 605, 120, 40)
        hint_btn = pygame.Rect(WIDTH - 260, 605, 120, 40)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return "QUIT"

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos

                    # 1. Проверка нажатия кнопок (приоритет)
                    if new_puzzle_btn.collidepoint(pos):
                        return "NEW"

                    if hint_btn.collidepoint(pos) and not solved:
                        best_move, _ = analyzer.get_best_move(board.fen())
                        if best_move:
                            # Функция перевода логической клетки в экранную
                            def sql_to_gui(sq):
                                f, r = chess.square_file(sq), chess.square_rank(sq)
                                if self.flipped:
                                    return 7 - f, r
                                return f, 7 - r

                            f_c, f_r = sql_to_gui(best_move.from_square)
                            t_c, t_r = sql_to_gui(best_move.to_square)
                            hint_sqs = [(f_r, f_c), (t_r, t_c)]
                            continue  # Пропускаем остальную логику клика

                    # 2. Логика клика по доске (только если клик в пределах 600x600)
                    if not solved and pos[1] <= 600:
                        c, r = pos[0] // SQ_SIZE, pos[1] // SQ_SIZE

                        # Координаты для chess.Board
                        if self.flipped:
                            logic_c, logic_r = 7 - c, r
                        else:
                            logic_c, logic_r = c, 7 - r

                        clicked_sq = chess.square(logic_c, logic_r)

                        if self.selected_sq == (r, c):
                            self.selected_sq = None
                        elif self.selected_sq is None:
                            piece = board.piece_at(clicked_sq)
                            if piece and piece.color == board.turn:
                                self.selected_sq = (r, c)
                        else:
                            # Пытаемся сделать ход
                            prev_r, prev_c = self.selected_sq
                            if self.flipped:
                                from_sq = chess.square(7 - prev_c, prev_r)
                            else:
                                from_sq = chess.square(prev_c, 7 - prev_r)

                            move = chess.Move(from_sq, clicked_sq)
                            if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN:
                                if chess.square_rank(clicked_sq) in [0, 7]:
                                    move.promotion = chess.QUEEN

                            if move in board.legal_moves:
                                best_m, _ = analyzer.get_best_move(board.fen())
                                if move == best_m:
                                    board.push(move)
                                    self.moves_made += 1
                                    hint_sqs = None  # Сбрасываем подсказку при верном ходе

                                    if board.is_game_over() or self.moves_made >= MAX_PUZZLE_MOVES:
                                        solved = True
                                        msg = "ЗАДАЧА РЕШЕНА!"
                                    else:
                                        # Ответ ИИ
                                        msg = "Верно! Компьютер думает..."
                                        self.draw_board();
                                        self.draw_pieces(board);
                                        pygame.display.flip()
                                        pygame.time.delay(500)
                                        opp_move, _ = analyzer.get_best_move(board.fen())
                                        if opp_move:
                                            board.push(opp_move)
                                            msg = "Ваш следующий ход!"
                                        else:
                                            solved = True;
                                            msg = "ПОБЕДА!"
                                else:
                                    self.error_flash = 150
                                    msg = "Неверно. Попробуйте еще раз."
                            self.selected_sq = None

            # --- Отрисовка ---
            self.draw_board(solved)

            # Отрисовка подсказки (желтые рамки)
            if hint_sqs:
                for sq_r, sq_c in hint_sqs:
                    pygame.draw.rect(self.screen, (255, 255, 0),
                                     (sq_c * SQ_SIZE, sq_r * SQ_SIZE, SQ_SIZE, SQ_SIZE), 4)

            self.draw_pieces(board)
            sub = f"Ходы: {self.moves_made} | Цель: {target_rating} Elo"
            self.show_message(msg, sub)
            self.draw_button("Hint", hint_btn, (100, 80, 0))
            self.draw_button("Next", new_puzzle_btn, (50, 50, 50))
            if solved: pygame.draw.rect(self.screen, (0, 255, 0), (0, 0, 600, 600), 6)
            pygame.display.flip()