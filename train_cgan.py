import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from clearml import Task, Logger
import os
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

from cgan_model import Generator, Discriminator
from dataset_utils import ChessDataset

# Настройка окружения
os.environ['CLEARML_API_DEFAULT_TIMEOUT'] = '60'
os.makedirs('models', exist_ok=True)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
# TTUR: Генератор (0.0004) учится быстрее Дискриминатора (0.00005)
LR_G = 0.0004
LR_D = 0.00005
EPOCHS = 20
LATENT_DIM = 64


def main():
    # --- CLEARML ---
    task = Task.init(project_name="Генерация шахматных позиций", task_name="trainCGAN")
    logger = Logger.current_logger()

    # --- DATA ---
    print("Загрузка датасета (1 млн примеров)...")
    full_dataset = ChessDataset("lichess_puzzles.csv", num_samples=1000000)
    train_idx, _ = train_test_split(list(range(len(full_dataset))), test_size=0.1)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --- MODELS ---
    netG = Generator(LATENT_DIM).to(DEVICE)
    netD = Discriminator().to(DEVICE)

    criterion = nn.BCELoss()

    # Оптимизаторы с разными LR для баланса сил
    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    print(f"Старт обучения на {DEVICE}. Балансировка TTUR включена.")

    for epoch in range(EPOCHS):
        netG.train()
        netD.train()

        running_loss_G = 0.0
        running_loss_D = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Эпоха {epoch}/{EPOCHS}")

        for i, (real_boards, ratings) in pbar:
            real_boards, ratings = real_boards.to(DEVICE), ratings.to(DEVICE).float()
            b_size = real_boards.size(0)

            # --- ШАГ 1: ОБУЧЕНИЕ ДИСКРИМИНАТОРА ---
            netD.zero_grad()

            # Реальные доски (метка 0.9 вместо 1.0)
            out_real = netD(real_boards, ratings)
            label_real = torch.full((b_size, 1), 0.9, device=DEVICE)
            lossD_real = criterion(out_real, label_real)

            # Фейковые доски (метка 0.1 вместо 0.0)
            noise = torch.randn(b_size, LATENT_DIM, device=DEVICE)
            fake = netG(noise, ratings)
            out_fake = netD(fake.detach(), ratings)
            label_fake = torch.full((b_size, 1), 0.1, device=DEVICE)
            lossD_fake = criterion(out_fake, label_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # --- ШАГ 2: ОБУЧЕНИЕ ГЕНЕРАТОРА ---
            netG.zero_grad()
            # Генератор пытается убедить Дискриминатора, что это реал (1.0)
            out_fake_g = netD(fake, ratings)
            label_g = torch.full((b_size, 1), 1.0, device=DEVICE)
            lossG = criterion(out_fake_g, label_g)

            lossG.backward()
            optG.step()

            # Сбор статистики
            running_loss_G += lossG.item()
            running_loss_D += lossD.item()

            if i % 10 == 0:
                pbar.set_postfix({
                    'Avg_D': f"{running_loss_D / (i + 1):.3f}",
                    'Avg_G': f"{running_loss_G / (i + 1):.3f}"
                })

            if i % 200 == 0:
                iter_count = int(epoch * len(train_loader) + i)
                logger.report_scalar("Iteration_Loss", "Generator", iteration=iter_count, value=lossG.item())
                logger.report_scalar("Iteration_Loss", "Discriminator", iteration=iter_count, value=lossD.item())

        # --- КОНЕЦ ЭПОХИ: СРЕДНИЕ ЗНАЧЕНИЯ ---
        epoch_avg_g = running_loss_G / len(train_loader)
        epoch_avg_d = running_loss_D / len(train_loader)
        logger.report_scalar("Epoch_Metrics", "Avg_G", iteration=epoch, value=epoch_avg_g)
        logger.report_scalar("Epoch_Metrics", "Avg_D", iteration=epoch, value=epoch_avg_d)

        # --- ВИЗУАЛИЗАЦИЯ (СЕТКА ВСЕХ ФИГУР) ---
        netG.eval()
        with torch.no_grad():
            test_r = torch.tensor([[0.75]], device=DEVICE)  # Тестовый рейтинг
            noise = torch.randn(1, LATENT_DIM, device=DEVICE)
            sample_tensor = netG(noise, test_r)  # [1, 12, 8, 8]

            # Разбираем 12 каналов на отдельные изображения для сетки
            # sample_tensor[0] имеет размер [12, 8, 8]. Добавляем размерность канала [12, 1, 8, 8]
            channels = sample_tensor[0].unsqueeze(1)

            # nrow=4 создаст сетку 4x3 (всего 12 каналов)
            grid = vutils.make_grid(channels, nrow=4, normalize=True, padding=2)
            grid_np = grid.cpu().numpy().transpose(1, 2, 0)

            logger.report_image(
                "Board_Analysis",
                f"Full_Channels_Ep_{epoch}",
                iteration=epoch,
                image=grid_np
            )

        # Сохранение
        torch.save(netG.state_dict(), f"models/G_ttur_ep_{epoch}.pth")
        print(f"\nЭпоха {epoch} завершена. Avg_G: {epoch_avg_g:.4f}, Avg_D: {epoch_avg_d:.4f}")


if __name__ == '__main__':
    main()