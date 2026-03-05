"""
Euclideanizer model: FrozenDistMapVAE (for loading pretrained DistMap) + Euclideanizer.
Uses shared pipeline utils.
"""
from torch import nn
import torch
import torch.nn.functional as F

from .. import utils


class FrozenDistMapVAE(nn.Module):
    """Minimal copy of DistMap ChromVAE_Conv for loading pre-trained weights."""

    def __init__(self, num_atoms, latent_space_dim):
        super().__init__()
        self.num_atoms = num_atoms
        self._padded_size = ((num_atoms + 7) // 8) * 8
        self._latent_space_dim = latent_space_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
        )
        self._build_linear()
        self.decoder_fc = nn.Linear(latent_space_dim, self._enc_flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), nn.Softplus(),
        )

    def _build_linear(self):
        dummy = torch.zeros(1, 1, self._padded_size, self._padded_size)
        out = self.encoder_conv(dummy)
        self._conv_out_shape = out.shape[1:]
        self._enc_flat_dim = int(out.numel())
        self._to_mu = nn.Linear(self._enc_flat_dim, self._latent_space_dim)
        self._to_logvar = nn.Linear(self._enc_flat_dim, self._latent_space_dim)

    def _decode_to_matrix(self, z):
        h = self.decoder_fc(z)
        h = h.reshape(-1, *self._conv_out_shape)
        h = self.decoder_conv(h)
        h = h.squeeze(1)[:, :self.num_atoms, :self.num_atoms]
        return 0.5 * (h + h.transpose(-1, -2))

    def encode(self, x):
        N = self.num_atoms
        h = x.unsqueeze(1)
        pad = self._padded_size - N
        if pad > 0:
            h = F.pad(h, (0, pad, 0, pad))
        h = self.encoder_conv(h).flatten(1)
        return self._to_mu(h)


def load_frozen_vae(checkpoint_path, num_atoms, latent_dim, device):
    vae = FrozenDistMapVAE(num_atoms, latent_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"Loaded frozen VAE from {checkpoint_path}")
    return vae


class Euclideanizer(nn.Module):
    """Converts non-Euclidean distance maps to 3D coordinates (Euclidean by construction)."""

    def __init__(self, num_atoms):
        super().__init__()
        self.num_atoms = num_atoms
        self._padded_size = ((num_atoms + 7) // 8) * 8

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
        )
        self._pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, num_atoms * 3),
        )

    def forward(self, D_log):
        N = self.num_atoms
        x = D_log.unsqueeze(1)
        pad = self._padded_size - N
        if pad > 0:
            x = F.pad(x, (0, pad, 0, pad))
        x = self.conv(x)
        x = self._pool(x)
        x = x.flatten(1)
        flat = self.fc(x)
        coords = flat.reshape(-1, N, 3)
        return coords - coords.mean(dim=1, keepdim=True)

    def forward_to_distmap(self, D_log):
        coords = self.forward(D_log)
        return torch.log1p(utils.get_distmaps(coords))
