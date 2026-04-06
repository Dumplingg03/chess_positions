import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Вход: шум + рейтинг (64 + 1 = 65)
        self.init_size = 4  # Начинаем с сетки 4x4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + 1, 256 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            # Слой 1: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU часто лучше ReLU в GAN

            # Слой 2: Финальная отрисовка 12 каналов (8x8)
            # Используем свертку 3x3 для уточнения деталей
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 12, kernel_size=3, padding=1),
            nn.Sigmoid()  # Значения от 0 до 1 (вероятность фигуры в клетке)
        )

    def forward(self, z, r):
        # z: [B, 64], r: [B, 1]
        z_cond = torch.cat([z, r], dim=1)
        out = self.l1(z_cond)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 12 каналов доски + 1 канал рейтинга = 13
        self.model = nn.Sequential(
            # Свертка 1: 8x8 -> 4x4
            # УБИРАЕМ BatchNorm здесь - это делает D менее "уверенным"
            nn.Conv2d(13, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Добавляем Dropout пораньше

            # Свертка 2: 4x4 -> 2x2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, r):
        # r_layer: расширяем рейтинг до размера доски 8x8
        r_layer = r.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        x_input = torch.cat([x, r_layer], dim=1)
        return self.model(x_input)