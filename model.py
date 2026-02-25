import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder: Conv layers
        self.enc_conv = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),  # 12 фигур + 1 слой рейтинга
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 4x4
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim + 1, 128 * 4 * 4)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 12, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, r):
        # Добавляем рейтинг как 13-й слой
        r_layer = r.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        x = torch.cat([x, r_layer], dim=1)
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, r):
        z_cond = torch.cat([z, r.view(-1, 1)], dim=1)
        h = self.dec_fc(z_cond).view(-1, 128, 4, 4)
        return self.dec_conv(h)

    def forward(self, x, r):
        mu, logvar = self.encode(x, r)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z, r), mu, logvar