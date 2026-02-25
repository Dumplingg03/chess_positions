import pygame
import chess
import os

WIDTH, HEIGHT = 600, 650
SQ_SIZE = 600 // 8
MAX_PUZZLE_MOVES = 5  # Максимальное количество ходов игрока в одной задаче


class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Chess Puzzle Generator")
        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.pieces = {}
        self.selected_sq = None
        self.flipped = False
        self.error_flash = 0  # Таймер для красной вспышки при ошибке
        self.load_assets()

    def load_assets(self):
        pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
        for p in pieces:
            path = os.path.join("pieces", p + ".png")
            try:
                img = pygame.image.load(path)
                self.pieces[p] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
            except:
                surf = pygame.Surface((SQ_SIZE, SQ_SIZE))
                surf.fill((200, 0, 0))
                self.pieces[p] = surf

    def get_rating_input(self):
        input_box = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 25, 200, 50)
        text = ''
        active = True
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return int(text) if text else 1500
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif event.unicode.isdigit():
                        text += event.unicode

            self.screen.fill((30, 30, 30))
            prompt = self.font.render("Введите целевой рейтинг (1000-2500):", True, (255, 255, 255))
            self.screen.blit(prompt, (WIDTH // 2 - 160, HEIGHT // 2 - 70))
            pygame.draw.rect(self.screen, (100, 100, 100), input_box, 2)
            val_surf = self.font.render(text + "|", True, (255, 255, 255))
            self.screen.blit(val_surf, (input_box.x + 10, input_box.y + 10))
            pygame.display.flip()

    def draw_board(self, solved=False):
        colors = [pygame.Color("#eeeed2"), pygame.Color("#769656")]
        for r in range(8):
            for c in range(8):
                color = colors[((r + c) % 2)]
                pygame.draw.rect(self.screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
                if self.selected_sq == (r, c):
                    pygame.draw.rect(self.screen, (186, 202, 43, 100),
                                     pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        # Вспышка при ошибке
        if self.error_flash > 0:
            s = pygame.Surface((600, 600))
            s.set_alpha(self.error_flash)
            s.fill((255, 0, 0))
            self.screen.blit(s, (0, 0))
            self.error_flash -= 15

    def draw_pieces(self, board):
        for r in range(8):
            for c in range(8):
                display_r = 7 - r if self.flipped else r
                display_c = 7 - c if self.flipped else c
                piece = board.piece_at(chess.square(display_c, 7 - display_r))
                if piece:
                    name = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
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
        msg = "Ваш ход! Найдите лучшее продолжение."

        new_puzzle_btn = pygame.Rect(WIDTH - 150, 605, 140, 40)
        hint_btn = pygame.Rect(WIDTH - 300, 605, 140, 40)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return "QUIT"
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if new_puzzle_btn.collidepoint(event.pos): return "NEW"
                    if hint_btn.collidepoint(event.pos) and not solved:
                        best_move, _ = analyzer.get_best_move(board.fen())
                        if best_move:
                            f_sq, t_sq = best_move.from_square, best_move.to_square
                            f_r, f_c = 7 - chess.square_rank(f_sq), chess.square_file(f_sq)
                            t_r, t_c = 7 - chess.square_rank(t_sq), chess.square_file(t_sq)
                            if self.flipped: f_r, f_c, t_r, t_c = 7 - f_r, 7 - f_c, 7 - t_r, 7 - t_c
                            hint_sqs = [(f_r, f_c), (t_r, t_c)]

                    if not solved:
                        c, r = event.pos[0] // SQ_SIZE, event.pos[1] // SQ_SIZE
                        if r > 7: continue
                        logic_r, logic_c = (7 - r if self.flipped else r), (7 - c if self.flipped else c)

                        if self.selected_sq == (r, c):
                            self.selected_sq = None
                        else:
                            if self.selected_sq is None:
                                if board.piece_at(chess.square(logic_c, 7 - logic_r)): self.selected_sq = (r, c)
                            else:
                                prev_r, prev_c = self.selected_sq
                                l_prev_r, l_prev_c = (7 - prev_r if self.flipped else prev_r), (
                                    7 - prev_c if self.flipped else prev_c)
                                from_sq, to_sq = chess.square(l_prev_c, 7 - l_prev_r), chess.square(logic_c,
                                                                                                    7 - logic_r)
                                move = chess.Move(from_sq, to_sq)
                                if board.piece_at(from_sq) and board.piece_at(
                                        from_sq).piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
                                    move.promotion = chess.QUEEN

                                if move in board.legal_moves:
                                    best_m, _ = analyzer.get_best_move(board.fen())
                                    if move == best_m:
                                        board.push(move)
                                        self.moves_made += 1
                                        hint_sqs = None
                                        if board.is_game_over() or self.moves_made >= MAX_PUZZLE_MOVES:
                                            solved = True;
                                            msg = "ЗАДАЧА РЕШЕНА!"
                                        else:
                                            self.draw_board();
                                            self.draw_pieces(board);
                                            self.show_message("Компьютер думает...");
                                            pygame.display.flip()
                                            pygame.time.delay(400)
                                            opp_move, _ = analyzer.get_best_move(board.fen())
                                            board.push(opp_move)
                                            # Проверка на досрочный выигрыш
                                            _, score = analyzer.get_best_move(board.fen())
                                            win_threshold = 500 if target_rating < 1800 else 700
                                            if (not self.flipped and score > win_threshold) or (
                                                    self.flipped and score < -win_threshold):
                                                solved = True;
                                                msg = "ПРЕИМУЩЕСТВО ДОБЫТО!"
                                            else:
                                                msg = "Верно! Продолжайте."
                                    else:
                                        self.error_flash = 120  # Красная вспышка
                                        msg = "Неверный ход. Попробуйте еще раз."
                                self.selected_sq = None

            self.draw_board(solved)
            if hint_sqs:
                for sq in hint_sqs: pygame.draw.rect(self.screen, (255, 255, 0),
                                                     (sq[1] * SQ_SIZE, sq[0] * SQ_SIZE, SQ_SIZE, SQ_SIZE), 4)
            self.draw_pieces(board)
            sub = f"Ход: {self.moves_made}/{MAX_PUZZLE_MOVES} | Цель: {target_rating} Elo"
            self.show_message(msg, sub)
            self.draw_button("Подсказка", hint_btn, (100, 80, 0))
            self.draw_button("Новая", new_puzzle_btn, (50, 50, 50))
            if solved: pygame.draw.rect(self.screen, (0, 255, 0), (0, 0, 600, 600), 6)
            pygame.display.flip()