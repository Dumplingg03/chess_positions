import pandas as pd
import numpy as np
import chess
import chess.engine
from tqdm import tqdm
from scipy.stats import pearsonr
from clearml import Task, Logger

# --- НАСТРОЙКИ ---
STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
INPUT_CSV = "lichess_puzzles.csv"
SAMPLE_SIZE = 20000

# Инициализация ClearML
task = Task.init(project_name="Генерация шахматных позиций", task_name="Grid Search Coefficients")
logger = Logger.current_logger()


def collect_features():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    # Загружаем данные и перемешиваем
    df_raw = pd.read_csv(INPUT_CSV)
    df = df_raw.sample(n=min(SAMPLE_SIZE, len(df_raw)), random_state=42)

    data_for_opt = []
    print(f"Анализируем {SAMPLE_SIZE} задач для поиска идеальной формулы...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            board = chess.Board(row['FEN'])
            info = engine.analyse(board, chess.engine.Limit(depth=10), multipv=2)
            if len(info) < 2:
                gap = 0
            else:
                s1 = info[0]["score"].pov(board.turn).score(mate_score=10000)
                s2 = info[1]["score"].pov(board.turn).score(mate_score=10000)
                gap = abs(np.clip(s1, -2000, 2000) - np.clip(s2, -2000, 2000))

            data_for_opt.append({
                'Rating': row['Rating'],
                'moves_count': len(row['Moves'].split()),
                'gap': gap,
                'mobility': board.legal_moves.count()
            })
        except:
            continue

    engine.quit()
    return pd.DataFrame(data_for_opt)


def run_grid_search(df):
    print("\nЗапуск перебора коэффициентов...")
    best_corr = -1
    best_p = {}
    iteration = 0

    # Перебираем веса w1 (ходы) и w2 (разрыв)
    for w1 in np.arange(1.5, 4.0, 0.2):  # Расширили диапазон для точности
        for w2 in np.arange(0.5, 2.5, 0.2):
            iteration += 1

            # Считаем M4
            m4 = (df['moves_count'] * w1) + (np.log1p(df['gap']) * w2) - (df['mobility'] * 0.1)
            corr, _ = pearsonr(df['Rating'], m4)

            # --- ЛОГИРОВАНИЕ В CLEARML ---
            # График зависимости корреляции от веса ходов (w1) и веса разрыва (w2)
            logger.report_scalar(title="Correlation Search", series="Pearson R", value=corr, iteration=iteration)
            logger.report_scalar(title="Weights Tracking", series="w1 (Moves)", value=w1, iteration=iteration)
            logger.report_scalar(title="Weights Tracking", series="w2 (Gap)", value=w2, iteration=iteration)

            if corr > best_corr:
                best_corr = corr
                best_p = {'w1': w1, 'w2': w2}
                # Логируем лучший результат как отдельный график
                logger.report_scalar(title="Best Result", series="Max Correlation", value=best_corr,
                                     iteration=iteration)

    w1, w2 = best_p['w1'], best_p['w2']
    final_m4 = (df['moves_count'] * w1) + (np.log1p(df['gap']) * w2) - (df['mobility'] * 0.1)

    m = df['Rating'].std() / final_m4.std()
    b = df['Rating'].mean() - (final_m4.mean() * m)

    # Логируем финальные гиперпараметры
    task.set_user_properties(
        best_w1=w1,
        best_w2=w2,
        multiplier_m=m,
        bias_b=b,
        max_correlation=best_corr
    )

    print("\n" + "=" * 40)
    print(f"РЕЗУЛЬТАТ (Корреляция: {best_corr:.4f})")
    print(f"w1 (ходы): {w1:.2f}")
    print(f"w2 (разрыв): {w2:.2f}")
    print(f"m (множитель): {m:.2f}")
    print(f"b (база): {int(b)}")
    print("=" * 40)

    # Отправляем итоговую таблицу в ClearML
    results_summary = {
        'Metric': ['w1', 'w2', 'm', 'b', 'Correlation'],
        'Value': [w1, w2, m, b, best_corr]
    }
    logger.report_table("Optimization Results", "Final Params", table_plot=pd.DataFrame(results_summary))


if __name__ == "__main__":
    feature_df = collect_features()
    if not feature_df.empty:
        run_grid_search(feature_df)

    task.close()