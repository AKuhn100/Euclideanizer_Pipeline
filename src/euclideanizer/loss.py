import torch
import torch.nn.functional as F
from typing import Optional


def kabsch_align(recon_coords: torch.Tensor, gt_coords: torch.Tensor) -> torch.Tensor:
    """Optimal rotation (Kabsch) to align recon_coords to gt_coords. All differentiable.

    recon_coords, gt_coords: (B, N, 3). Both are centered inside; returns aligned recon (B, N, 3)
    in the same frame as gt_centered.
    """
    gt_c = gt_coords - gt_coords.mean(dim=1, keepdim=True)
    recon_c = recon_coords - recon_coords.mean(dim=1, keepdim=True)
    # H = recon_c^T @ gt_c per batch -> (B, 3, 3)
    H = recon_c.transpose(1, 2) @ gt_c
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    # Reflection fix: ensure det(R) = 1 (proper rotation); flip last column of V when det < 0 (no in-place for grad)
    R = V @ U.transpose(1, 2)
    det = torch.linalg.det(R)
    sign = torch.where(
        det < 0,
        -torch.ones(1, device=H.device, dtype=H.dtype).expand(H.size(0)),
        torch.ones(1, device=H.device, dtype=H.dtype).expand(H.size(0)),
    )
    scale = torch.ones(H.size(0), 1, 3, device=H.device, dtype=H.dtype)
    scale[:, 0, 2] = sign
    V_fixed = V * scale
    R = V_fixed @ U.transpose(1, 2)
    # R maps columns of recon_c^T to gt_c^T; for row vectors we need aligned = recon_c @ R^T
    aligned = recon_c @ R.transpose(1, 2)
    return aligned, gt_c


def calc_MSE_kabsch(gt_coords: torch.Tensor, recon_coords: torch.Tensor) -> torch.Tensor:
    """MSE between gt coords and recon coords after aligning recon to gt via Kabsch.

    gt_coords, recon_coords: (B, N, 3). Returns scalar.
    """
    aligned, gt_c = kabsch_align(recon_coords, gt_coords)
    return F.mse_loss(aligned, gt_c, reduction="mean")


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


def calc_diagonal_wasserstein(gts, generated, num_diags: int, per_sample: bool = False):
    """W1 loss computed per genomic separation (diagonal), then averaged.

    For each separation k = 1..num_diags, compute W1 between the k-th
    diagonal values of gts and generated.

    When per_sample=False (generation mode): pool values across the whole
    batch before sorting.  This matches the *distribution* of distances at
    each genomic lag across all generated structures against the ground-truth
    distribution — the intended behaviour for generative quality.

    When per_sample=True (reconstruction mode): sort and compare within each
    sample independently, then average across samples.  This ensures that
    structure b's reconstruction is compared only against structure b's ground
    truth, not against other structures in the batch.  Pooling across the
    batch for reconstruction is incorrect because a good recon on sample A
    can mask a bad recon on sample B when the values are ranked together.

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

        if per_sample:
            # Reconstruction: compare each sample to its own ground truth.
            # Sort along the diagonal axis (dim=1) so sample b's values are
            # only ranked against sample b's ground-truth values.
            gt_s  = torch.sort(gt_diag,  dim=1).values  # (B, N-k)
            gen_s = torch.sort(gen_diag, dim=1).values  # (B, N-k)
            total_w = total_w + torch.mean(torch.abs(gt_s - gen_s))
        else:
            # Generation: match the marginal distribution across the batch.
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
    *,
    gt_coords: Optional[torch.Tensor] = None,
    recon_coords: Optional[torch.Tensor] = None,
    lambda_kabsch_mse: float,
):
    """Distributional loss for the Euclideanizer. All weights and num_diags from config.

    Components: MSE(recon, GT) + λ_w_recon * W1(recon, GT) + λ_w_gen * W1(gen, GT)
    + λ_w_diag_recon * DiagW1_per_sample(recon, GT) + λ_w_diag_gen * DiagW1_pooled(gen, GT)
    + λ_kabsch_mse * MSE(gt_coords, Kabsch_aligned(recon_coords)) when coords provided.

    Diagonal Wasserstein for reconstruction uses per_sample=True (compare each
    reconstructed structure only to its own ground truth). Diagonal Wasserstein for
    generation uses per_sample=False (pool across the batch to match the distribution).

    num_diags: number of diagonals (genomic separations) used for diagonal Wasserstein.
    No KL (latent is frozen VAE). Returns (loss, mse, recon_w, gen_w, diag_recon_w, diag_gen_w, kabsch_mse).
    """
    mse = calc_MSE_loss_full(gts, D_euclid_recon)
    recon_w = calc_positionwise_wasserstein(gts, D_euclid_recon)
    gen_w = calc_positionwise_wasserstein(gts, D_euclid_gen)

    diag_recon_w = torch.tensor(0.0, device=gts.device)
    diag_gen_w = torch.tensor(0.0, device=gts.device)
    if lambda_w_diag_recon != 0:
        # per_sample=True: each reconstruction is compared to its own ground truth only.
        diag_recon_w = calc_diagonal_wasserstein(gts, D_euclid_recon, num_diags=num_diags, per_sample=True)
    if lambda_w_diag_gen != 0:
        # per_sample=False: pool across the batch to match the generative distribution.
        diag_gen_w = calc_diagonal_wasserstein(gts, D_euclid_gen, num_diags=num_diags, per_sample=False)

    kabsch_mse = torch.tensor(0.0, device=gts.device)
    if lambda_kabsch_mse != 0 and gt_coords is not None and recon_coords is not None:
        kabsch_mse = calc_MSE_kabsch(gt_coords, recon_coords)

    loss = (lambda_mse * mse
            + lambda_w_recon * recon_w
            + lambda_w_gen * gen_w
            + lambda_w_diag_recon * diag_recon_w
            + lambda_w_diag_gen * diag_gen_w
            + lambda_kabsch_mse * kabsch_mse)
    return loss, mse, recon_w, gen_w, diag_recon_w, diag_gen_w, kabsch_mse