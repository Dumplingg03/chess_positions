import pandas as pd
import numpy as np
import chess
import chess.engine
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task, Logger

# ==========================================
# 1. CLEARML & CONFIG
# ==========================================
task = Task.init(project_name="Chess_CGAN", task_name="Dataset_SDC_Processing")
logger = Logger.current_logger()

CSV_INPUT = "lichess_puzzles.csv"
CSV_OUTPUT = "lichess_puzzles_with_sdc.csv"
STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"

SAMPLE_SIZE = 1500  # Ограничение выборки для адекватного времени расчета
DEPTH = 10

# Логируем параметры в ClearML
task.connect({
    "sample_size": SAMPLE_SIZE,
    "stockfish_depth": DEPTH,
    "sdc_formula": "m4 = (moves*2.0) + (log_gap*1.5) - (mobility*0.1)"
})


# ==========================================
# 2. ФУНКЦИЯ РАСЧЕТА SDC
# ==========================================
def calculate_sdc_logic(fen, moves_str, engine):
    try:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=2)

        if len(info) < 2:
            gap = 0
        else:
            def to_cp(score):
                s = score.score(mate_score=10000)
                return np.clip(s, -2000, 2000) if s is not None else 0

            gap = abs(to_cp(info[0]["score"].pov(board.turn)) - to_cp(info[1]["score"].pov(board.turn)))

        moves_count = len(moves_str.split())
        mobility = board.legal_moves.count()

        # Формула на базе твоей корреляции 0.42
        m4 = (moves_count * 2.0) + (np.log1p(gap) * 1.5) - (mobility * 0.1)
        sdc_rating = (m4 * 51.2) + 780

        return int(np.clip(sdc_rating, 600, 3000))
    except:
        return None


# ==========================================
# 3. ОСНОВНОЙ ПРОЦЕСС
# ==========================================
def main():
    # Загружаем данные и делаем выборку
    print(f"Загрузка {CSV_INPUT}...")
    df_full = pd.read_csv(CSV_INPUT)
    df = df_full.sample(n=min(SAMPLE_SIZE, len(df_full)), random_state=42).reset_index(drop=True)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sdc_results = []

    print(f"Начинаю расчет SDC для {len(df)} позиций...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        sdc = calculate_sdc_logic(row['FEN'], row['Moves'], engine)
        sdc_results.append(sdc)

        # Логируем прогресс в ClearML каждые 100 строк
        if i % 100 == 0 and i > 0:
            current_sdc = sdc if sdc else 0
            logger.report_scalar("Processing", "Current_SDC", iteration=i, value=current_sdc)

        # Промежуточное сохранение и графики каждые 2000 строк
        if i % 2000 == 0 and i > 0:
            current_batch = [x for x in sdc_results if x is not None]

            # Сохраняем файл
            temp_df = df.iloc[:len(sdc_results)].copy()
            temp_df['SDC_Rating'] = sdc_results
            temp_df.to_csv(CSV_OUTPUT, index=False)

            # Обновляем гистограмму в ClearML
            plt.figure(figsize=(10, 6))
            sns.histplot(current_batch, bins=30, kde=True, color='blue')
            plt.title(f"SDC Distribution (Progress: {i}/{len(df)})")
            plt.xlabel("SDC Rating")
            logger.report_matplotlib_figure("Distribution", "SDC_Histogram", figure=plt, iteration=i)
            plt.close()

    # Финальная обработка
    df['SDC_Rating'] = sdc_results
    df = df.dropna(subset=['SDC_Rating'])
    df.to_csv(CSV_OUTPUT, index=False)

    # Итоговый график
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['Rating'], y=df['SDC_Rating'], alpha=0.1)
    plt.title("Lichess Rating vs SDC Rating (Final)")
    plt.xlabel("Original Lichess Rating")
    plt.ylabel("Our SDC Rating")
    logger.report_matplotlib_figure("Analysis", "Correlation_Final", figure=plt)

    engine.quit()
    print(f"\nГотово! Обработано {len(df)} строк. Файл: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()