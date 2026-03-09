import torch
import torch.nn.functional as F

DEFAULT_NUM_DIAGS = 50


def calc_MSE_loss_full(gts, reconstructions):
    return F.mse_loss(reconstructions, gts, reduction="mean")


def calc_positionwise_wasserstein(gts, generated):
    B, N, _ = gts.shape
    i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=gts.device)
    gt_vals = gts[:, i_idx, j_idx]
    gen_vals = generated[:, i_idx, j_idx]
    gt_sorted = torch.sort(gt_vals, dim=0).values
    gen_sorted = torch.sort(gen_vals, dim=0).values
    return torch.mean(torch.abs(gt_sorted - gen_sorted))


def calc_diagonal_wasserstein(gts, generated, num_diags=DEFAULT_NUM_DIAGS):
    """W1 loss computed per genomic separation (diagonal), then averaged.

    For each separation k = 1..num_diags, we sort the k-th diagonal
    values across the batch dimension and compute W1.  This directly
    penalises deviations in the genomic scaling curve P(s).
    When num_diags is 0, returns 0.0 (no diagonal penalty).
    """
    B, N, _ = gts.shape
    num_diags = min(max(0, num_diags), N - 1)
    if num_diags <= 0:
        return torch.tensor(0.0, device=gts.device)
    total_w = torch.tensor(0.0, device=gts.device)

    for k in range(1, num_diags + 1):
        idx = torch.arange(N - k, device=gts.device)
        gt_diag  = gts[:, idx, idx + k]       # (B, N-k)
        gen_diag = generated[:, idx, idx + k]  # (B, N-k)
        gt_s  = torch.sort(gt_diag.flatten()).values
        gen_s = torch.sort(gen_diag.flatten()).values
        n = min(gt_s.shape[0], gen_s.shape[0])
        total_w = total_w + torch.mean(torch.abs(gt_s[:n] - gen_s[:n]))

    return total_w / num_diags


def euclideanizer_loss(
    gts,
    D_euclid_recon,
    D_euclid_gen,
    lambda_mse,
    lambda_w_recon,
    lambda_w_gen,
    lambda_w_diag_recon,
    lambda_w_diag_gen,
    num_diags,
):
    """Distributional loss for the Euclideanizer. All weights and num_diags from config.

    Components: MSE(recon, GT) + λ_w_recon * W1(recon, GT) + λ_w_gen * W1(gen, GT)
    + λ_w_diag_recon * DiagW1(recon, GT) + λ_w_diag_gen * DiagW1(gen, GT).
    num_diags: number of diagonals (genomic separations) used for diagonal Wasserstein.
    No KL (latent is frozen VAE). Returns (loss, mse, recon_w, gen_w, diag_recon_w, diag_gen_w).
    """
    mse = calc_MSE_loss_full(gts, D_euclid_recon)
    recon_w = calc_positionwise_wasserstein(gts, D_euclid_recon)
    gen_w = calc_positionwise_wasserstein(gts, D_euclid_gen)

    diag_recon_w = torch.tensor(0.0, device=gts.device)
    diag_gen_w = torch.tensor(0.0, device=gts.device)
    if lambda_w_diag_recon != 0:
        diag_recon_w = calc_diagonal_wasserstein(gts, D_euclid_recon, num_diags=num_diags)
    if lambda_w_diag_gen != 0:
        diag_gen_w = calc_diagonal_wasserstein(gts, D_euclid_gen, num_diags=num_diags)

    loss = (lambda_mse * mse
            + lambda_w_recon * recon_w
            + lambda_w_gen * gen_w
            + lambda_w_diag_recon * diag_recon_w
            + lambda_w_diag_gen * diag_gen_w)
    return loss, mse, recon_w, gen_w, diag_recon_w, diag_gen_w
