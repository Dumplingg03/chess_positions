import pandas as pd
import numpy as np
import chess
import chess.engine
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task, Logger

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ И НАСТРОЙКИ
# ==========================================
task = Task.init(project_name="Chess_CGAN", task_name="Custom_Metrics_Comparison")
logger = Logger.current_logger()

CSV_PATH = "lichess_puzzles.csv"
STOCKFISH_PATH = r"C:\Users\Koshk\PycharmProjects\Chess\Lab 4\chessgen\stockfish\stockfish-windows-x86-64-avx2.exe"
SAMPLE_SIZE = 20000  # Для теста возьми 1000, потом увеличь
DEPTH = 10


# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_engine_gap(board, engine):
    """Считает разрыв между 1-й и 2-й линией (Sharpness)"""
    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=2)
    if len(info) < 2:
        return 0

    def to_cp(score):
        s = score.score(mate_score=10000)
        return np.clip(s, -2000, 2000) if s is not None else 0

    best = to_cp(info[0]["score"].pov(board.turn))
    second = to_cp(info[1]["score"].pov(board.turn))
    return abs(best - second)


def get_board_tension(board):
    """Считает количество защищенных и атакованных фигур (Entropy)"""
    tension = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            attackers = len(board.attackers(not piece.color, sq))
            defenders = len(board.attackers(piece.color, sq))
            tension += (attackers + defenders)
    return tension


# ==========================================
# 3. ОСНОВНОЙ ЦИКЛ РАСЧЕТА
# ==========================================
df = pd.read_csv(CSV_PATH).sample(SAMPLE_SIZE, random_state=42)
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

results = []

print("Анализ позиций...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        board = chess.Board(row["FEN"])

        # Базовые параметры
        moves_count = len(row["Moves"].split())
        eval_gap = get_engine_gap(board, engine)
        tension = get_board_tension(board)
        mobility = board.legal_moves.count()

        # --- ФОРМУЛИРОВКА МЕТРИК ---

        # 1. Метрика сложности хода (Gap-based)
        m1_sharpness = np.log1p(eval_gap)

        # 2. Метрика глубины (Moves-based)
        m2_depth = moves_count * np.log1p(eval_gap)

        # 3. Метрика хаоса (Tension-based)
        m3_chaos = tension / (mobility + 1)

        # 4. Композитная метрика (Weighted)
        # Подбираем веса логически: ходы важны больше всего
        m4_composite = (moves_count * 2.0) + (np.log1p(eval_gap) * 1.5) - (mobility * 0.1)

        results.append({
            "Rating": row["Rating"],
            "M1_Sharpness": m1_sharpness,
            "M2_Depth": m2_depth,
            "M3_Chaos": m3_chaos,
            "M4_Composite": m4_composite
        })
    except Exception as e:
        continue

engine.quit()
res_df = pd.DataFrame(results)

# ==========================================
# 4. СРАВНЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ==========================================
metrics = ["M1_Sharpness", "M2_Depth", "M3_Chaos", "M4_Composite"]
correlations = {}

print("\n===== КОРРЕЛЯЦИЯ СПИРМЕНА С РЕЙТИНГОМ =====")
for m in metrics:
    corr, _ = spearmanr(res_df["Rating"], res_df[m])
    correlations[m] = corr
    print(f"{m}: {corr:.4f}")
    logger.report_scalar("Correlations", m, iteration=0, value=corr)

# Построение графиков
plt.figure(figsize=(15, 10))
for i, m in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.regplot(data=res_df, x=m, y="Rating", scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title(f"{m} (Corr: {correlations[m]:.2f})")

plt.tight_layout()
plt.show()
logger.report_matplotlib_figure("Metrics_Analysis", "Scatter_Plots", figure=plt)