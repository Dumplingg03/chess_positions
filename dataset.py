import pandas as pd
import numpy as np
from preprocess import fen_to_matrix
from sklearn.model_selection import train_test_split


class LichessPuzzleDataset:
    def __init__(self, csv_path, max_samples=None, train=True, test_size=0.2):
        """
        train=True  -> обучающая выборка
        train=False -> тестовая выборка
        """
        # читаем CSV
        df = pd.read_csv(csv_path, usecols=["FEN", "Rating"])

        # если нужно ограничить количество примеров
        if max_samples:
            df = df.sample(max_samples, random_state=42)

        # делим на train/test
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        self.df = train_df if train else test_df
        self.fens = self.df["FEN"].values
        self.ratings = self.df["Rating"].values

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        board = fen_to_matrix(self.fens[idx])
        rating = self.ratings[idx]

        return {
            "board": board.astype(np.float32),
            "rating": rating
        }


# -------------------
# тестирование
# -------------------
if __name__ == "__main__":
    train_dataset = LichessPuzzleDataset("lichess_puzzles.csv", max_samples=10, train=True)
    test_dataset = LichessPuzzleDataset("lichess_puzzles.csv", max_samples=10, train=False)

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    sample = train_dataset[0]
    print("Board shape:", sample["board"].shape)
    print("Rating:", sample["rating"])
    print(sample["board"])