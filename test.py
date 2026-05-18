import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import chess

# Импортируем твои функции из файла main.py (или как он у тебя называется)
# Если файлы в одной папке, это сработает
from mainsdc import (
    Generator, ChessAnalyzer, get_board_from_output,
    is_logic_valid, calculate_sdc, cp,
    STOCKFISH_PATH, MODEL_PATH, DEVICE, LATENT_DIM
)

# --- ПАРАМЕТРЫ ТЕСТА ---
TEST_RATINGS = [1000, 1500, 2000, 2500]
SAMPLES_PER_LEVEL = 30  # Сколько УДАЧНЫХ пазлов собрать для каждого уровня
MAX_ATTEMPTS = 500  # Макс. попыток на один уровень, чтобы не ждать вечность


def run_sdc_analysis():
    print(f"🚀 Начинаем стресс-тест на устройстве: {DEVICE}")

    analyzer = ChessAnalyzer(STOCKFISH_PATH)
    analyzer.start_engine()

    model = Generator(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    stats = defaultdict(list)
    efficiency = {}  # Процент прохождения фильтров

    for target in TEST_RATINGS:
        print(f"\n📊 Тестируем Target: {target}...")
        found = 0
        total_attempts = 0
        start_time = time.time()

        while found < SAMPLES_PER_LEVEL and total_attempts < MAX_ATTEMPTS:
            total_attempts += 1

            with torch.no_grad():
                z = torch.randn(1, LATENT_DIM).to(DEVICE)
                r_norm = torch.tensor([[(target - 500) / 2500]]).float().to(DEVICE)
                output = model(z, r_norm).cpu().numpy()[0]

                board = get_board_from_output(output)

                # 1. Проверка валидности (твои "гайки")
                if not is_logic_valid(board):
                    continue

                fen = board.fen()

                # 2. Проверка материала
                if not analyzer.is_puzzle_material(fen, target):
                    continue

                # 3. Лучший ход и Gap
                best_move, _ = analyzer.get_best_move(fen, depth=14)
                if not best_move:
                    continue

                info = analyzer.engine.analyse(board, chess.engine.Limit(depth=12), multipv=2)
                if len(info) > 1:
                    s1 = cp(info[0]["score"].relative)
                    s2 = cp(info[1]["score"].relative)
                    if abs(s1 - s2) < 100:  # Твой фильтр на единственность
                        continue

                # 4. Считаем SDC по твоей формуле
                sdc = calculate_sdc(board, analyzer)

                # Сохраняем ВСЕ найденные SDC, даже если они не попали в +-400,
                # чтобы увидеть реальный разброс модели
                stats[target].append(sdc)
                found += 1

                if found % 5 == 0:
                    print(f"  Найдено {found}/{SAMPLES_PER_LEVEL}...")

        efficiency[target] = (found / total_attempts) * 100 if total_attempts > 0 else 0
        print(f"✅ Готово для {target}. КПД фильтров: {efficiency[target]:.2f}%")

    analyzer.stop_engine()

    # === ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ===
    if not stats:
        print("Данные не собраны. Проверь модель или Stockfish.")
        return

    plt.figure(figsize=(12, 6))

    # 1. Boxplot распределения
    plt.subplot(1, 2, 1)
    data = [stats[t] for t in TEST_RATINGS if stats[t]]
    labels = [t for t in TEST_RATINGS if stats[t]]
    plt.boxplot(data, labels=labels)
    plt.plot(range(1, len(labels) + 1), labels, 'ro', label='Target (Ideal)', alpha=0.5)
    plt.title("Распределение реального SDC")
    plt.xlabel("Запрошенный рейтинг")
    plt.ylabel("Рассчитанный SDC (M4)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. КПД генерации
    plt.subplot(1, 2, 2)
    plt.bar([str(t) for t in TEST_RATINGS], [efficiency[t] for t in TEST_RATINGS], color='skyblue')
    plt.title("КПД генерации (%)")
    plt.ylabel("% прохождения фильтров")
    plt.xlabel("Target")

    plt.tight_layout()
    plt.show()

    # Текстовый отчет
    print("\n" + "=" * 30)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 30)
    for target in TEST_RATINGS:
        if stats[target]:
            avg = np.mean(stats[target])
            std = np.std(stats[target])
            print(f"Target {target}: Средний SDC = {int(avg)} (±{int(std)}) | КПД: {efficiency[target]:.1f}%")


if __name__ == "__main__":
    run_sdc_analysis()