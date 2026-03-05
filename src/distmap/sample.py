"""Latent sampling for DistMap (e.g. for generation plots and training video)."""
import torch


def generate_samples(num_samples, latent_dimension, device, variance=1.0):
    """Draw Gaussian latent samples with optional scale (variance**0.5). Returns (num_samples, latent_dimension) tensor."""
    scale = variance ** 0.5
    samples = torch.randn(size=(num_samples, latent_dimension)).to(device) * scale
    return samples
