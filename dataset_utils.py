import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ChessDataset(Dataset):
    def __init__(self, data_path, num_samples=1000000):
        # Берем ровно миллион задач
        print(f"Loading {num_samples} puzzles from CSV...")
        self.df = pd.read_csv(data_path, nrows=num_samples)

        self.min_rating = self.df['Rating'].min()
        self.max_rating = self.df['Rating'].max()
        print(f"Loaded {len(self.df)} puzzles. Rating range: {self.min_rating}-{self.max_rating}")

    def fen_to_tensor(self, fen):
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        rows = fen.split(' ')[0].split('/')
        for i, row in enumerate(rows):
            col = 0
            for char in row:
                if char.isdigit():
                    col += int(char)
                else:
                    piece_idx = piece_map[char]
                    tensor[piece_idx, i, col] = 1.0
                    col += 1
        return torch.tensor(tensor)

    def normalize_rating(self, rating):
        norm = (rating - self.min_rating) / (self.max_rating - self.min_rating)
        return np.clip(norm, 0, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board = self.fen_to_tensor(row['FEN'])
        rating = torch.tensor([self.normalize_rating(row['Rating'])], dtype=torch.float32)
        return board, rating