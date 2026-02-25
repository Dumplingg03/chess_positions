import torch
from torch.utils.data import DataLoader
from dataset import LichessPuzzleDataset
from model import CVAE
import torch.nn.functional as F


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LichessPuzzleDataset("lichess_puzzles.csv", max_samples=100000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        for batch in loader:
            boards = batch["board"].to(device)  # [Batch, 12, 8, 8]
            ratings = (batch["rating"].to(device).float() - 1500) / 1000  # Нормализация

            optimizer.zero_grad()
            recon, mu, logvar = model(boards, ratings)

            recon_loss = F.binary_cross_entropy(recon, boards, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} complete")

    torch.save(model.state_dict(), "cvae_chess.pth")


if __name__ == "__main__":
    train()