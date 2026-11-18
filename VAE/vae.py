import torch
import torch.nn as nn
import torch.nn.funtional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, channels=64, image_size=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size

        self.encoder = nn.Sequencial(
            nn.Conv2d(3, channels, 4, 2, 1),    # (B, 3, 64, 64) -> (B, 64, 32, 32)
            nn.ReLU(True),

            nn.Conv2d(channels, channels * 2, 4, 2, 1),     # (B, 64, 32, 32) -> (B, 128, 16, 16)
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),

            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1),     # (B, 128, 16, 16) -> (B, 256, 8, 8)
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),

            nn.Conv2d(channels * 4, channels * 8, 4, 2, 1),     # (B, 256, 8, 8) -> (B, 512, 4, 4)
            nn.BatchNorm2d(channels * 8),
            nn.ReLU(True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_dim = self.encoder(dummy).view(1, -1).size(1)

        self.fc_mu = nn.Linear(feat_dim, latent_dim)    # (B, 8192) -> (B, 128)
        self.fc_sigma = nn.Linear(feat_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, feat_dim)

        self._feat_shape = self.encoder(dummy).shape[1:]    # (512, 4, 4)

        self.decoder = nn.Sequencial(
            nn.ConvTranspose2d(channels * 8, channels * 4, 4, 2, 1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),

            nn.ConvTranspose2d(channels, 3, 4, 2, 1),
            nn.Sigmoid(),   # 将输出像素值归一化到 [0, 1]
        )

    def encode(self, x):
        hidden = self.encoder(x)
        hidden = hidden.view(hidden.size(0), -1)
        mu = self.fc_mu(hidden)
        log_var = self.fc_sigma(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        hidden = self.fc_decoder(z)     # (B, 128) -> (B, 8192)
        hidden = hidden.view(hidden.size(0), *self._feat_shape)     # (B, 8192) -> (B, 512, 4, 4)
        x_rec = self.decoder(hidden)        # (B, 512, 4, 4) -> (B, 3, 64, 64)
        return x_rec

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)
        return x_rec, mu, log_var

    def loss(x_rec, x, mu, log_var):
        rec_loss = F.mse_loss(x_rec, x)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return rec_loss, kld