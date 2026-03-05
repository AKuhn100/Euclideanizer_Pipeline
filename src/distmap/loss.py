"""
DistMap VAE loss: MSE + KL + position-wise Wasserstein on reconstructions and generations.
All weights are passed from config (no defaults).
"""
import torch
import torch.nn.functional as F


def _kl_div(mu, logvar):
    var = torch.exp(logvar)
    return 0.5 * torch.mean(var + mu.pow(2) - 1 - logvar)


def _mse_full(gts, reconstructions):
    return F.mse_loss(reconstructions, gts, reduction="mean")


def _positionwise_wasserstein(gts, generated):
    """Position-wise W1 between two (B, N, N) batches: sort at each (i,j), mean absolute difference."""
    B, N, _ = gts.shape
    i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=gts.device)
    gt_vals = gts[:, i_idx, j_idx]
    gen_vals = generated[:, i_idx, j_idx]
    gt_sorted = torch.sort(gt_vals, dim=0).values
    gen_sorted = torch.sort(gen_vals, dim=0).values
    return torch.mean(torch.abs(gt_sorted - gen_sorted))


def distmap_vae_loss(
    mu,
    logvar,
    gts,
    reconstructions,
    model,
    latent_dim,
    device,
    beta_kl,
    lambda_mse,
    lambda_w_gen,
    lambda_w_recon,
):
    """
    VAE loss: beta_kl * KL + lambda_mse * MSE(recon, gt) + lambda_w_recon * W1(recon, gt) + lambda_w_gen * W1(gen, gt).

    Position-wise Wasserstein is computed at each upper-triangle position across the batch.
    Generation samples are drawn from N(0,1) and decoded. All weights come from config.

    Returns:
        (loss, kl, recon_total, mse, gen_wasserstein, recon_wasserstein)
    """
    B = gts.shape[0]
    kl = _kl_div(mu, logvar)
    mse = _mse_full(gts, reconstructions)
    recon_w = _positionwise_wasserstein(gts, reconstructions)
    z = torch.randn(B, latent_dim, device=device)
    generated = model._decode_to_matrix(z)
    gen_w = _positionwise_wasserstein(gts, generated)
    recon_total = lambda_mse * mse + lambda_w_gen * gen_w + lambda_w_recon * recon_w
    loss = beta_kl * kl + recon_total
    return loss, kl, recon_total, mse, gen_w, recon_w
