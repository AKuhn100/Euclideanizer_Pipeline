"""
DistMap VAE (ChromVAE_Conv) for distance map training.
Uses shared pipeline utils.
"""
from torch import nn
import torch
import torch.nn.functional as F

from .. import utils

class ChromVAE_Conv(nn.Module):
    def __init__(self, num_atoms, latent_space_dim):
        super().__init__()
        self.num_atoms = num_atoms
        self.tri_dim = num_atoms * (num_atoms - 1) // 2
        self._padded_size = ((num_atoms + 7) // 8) * 8

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self._latent_space_dim = latent_space_dim
        self._build_linear()

        self.decoder_fc = nn.Linear(latent_space_dim, self._enc_flat_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Softplus(),
        )

    def _build_linear(self):
        dummy = torch.zeros(1, 1, self._padded_size, self._padded_size)
        out = self.encoder_conv(dummy)
        self._conv_out_shape = out.shape[1:]
        self._enc_flat_dim = int(out.numel())
        self._to_mu = nn.Linear(self._enc_flat_dim, self._latent_space_dim)
        self._to_logvar = nn.Linear(self._enc_flat_dim, self._latent_space_dim)

    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _decode_to_matrix(self, z):
        recon = self.decoder_fc(z)
        recon = recon.reshape(-1, *self._conv_out_shape)
        recon = self.decoder_conv(recon)
        recon = recon.squeeze(1)
        recon = recon[:, :self.num_atoms, :self.num_atoms]
        recon = 0.5 * (recon + recon.transpose(-1, -2))
        return recon

    def forward(self, x):
        N = self.num_atoms
        x = torch.log1p(x)
        x = x.unsqueeze(1)
        pad = self._padded_size - N
        if pad > 0:
            x = F.pad(x, (0, pad, 0, pad))
        x = self.encoder_conv(x)
        x = x.flatten(1)
        mu = self._to_mu(x)
        logvar = self._to_logvar(x)
        if self.training:
            z = self.reparameterization_trick(mu, logvar)
        else:
            z = mu
        recon = self._decode_to_matrix(z)
        reconstructions = utils.get_upper_tri(recon)
        return mu, logvar, z, reconstructions

    def decode(self, z):
        recon = self._decode_to_matrix(z)
        recon = torch.expm1(recon)
        return recon.detach().cpu().numpy()
