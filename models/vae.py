import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 128


class Encodeur(nn.Module):
    """Encodeur convolutif : image → (mu, log_var)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),    # 32×32 → 16×16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16×16 → 8×8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8×8 → 4×4
            nn.ReLU()
        )
        self.fc_mu      = nn.Linear(256 * 4 * 4, LATENT_DIM)
        self.fc_log_var = nn.Linear(256 * 4 * 4, LATENT_DIM)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decodeur(nn.Module):
    """Décodeur convolutif transposé : z → image reconstruite."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4×4 → 8×8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8×8 → 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),    # 16×16 → 32×32
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """VAE complet avec reparameterization trick."""

    def __init__(self):
        super().__init__()
        self.encodeur = Encodeur()
        self.decodeur = Decodeur()

    def reparametrize(self, mu, log_var):
        """z = mu + epsilon * sigma."""
        sigma   = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma

    def forward(self, x):
        mu, log_var = self.encodeur(x)
        z = self.reparametrize(mu, log_var)
        return self.decodeur(z), mu, log_var

    def encode(self, x):
        mu, _ = self.encodeur(x)
        return mu

    @staticmethod
    def loss(x, x_reconstruit, mu, log_var):
        """ELBO = reconstruction (MSE) + KL divergence."""
        recon = F.mse_loss(x_reconstruit, x, reduction='sum') / x.size(0)
        kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return recon + kl, recon.item(), kl.item()