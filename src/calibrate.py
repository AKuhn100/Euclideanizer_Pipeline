"""
Auto-calibration of batch sizes to stay under a GPU memory threshold.

- Training: distmap.batch_size, euclideanizer.batch_size (when null in config).
- Inference: gen_decode_batch_size and query_batch_size (decode-only path; when null,
  one value is used for both). Used for plotting and analysis.
"""
from __future__ import annotations

import warnings

import torch

from . import utils
from .distmap.loss import distmap_vae_loss
from .euclideanizer.loss import euclideanizer_loss

# Conservative default when CUDA is not available.
FALLBACK_BATCH_SIZE_NO_CUDA = 32

# Upper bound for search when not capped by dataset size.
DEFAULT_HIGH = 512
# Upper bound for inference/decode batch size (no dataset cap).
DEFAULT_HIGH_DECODE = 4096


def _get_train_size(coords: torch.Tensor, training_split: float, split_seed: int) -> int:
    train_ds, _ = utils.get_train_test_split(coords, training_split, split_seed)
    return len(train_ds)


def _run_distmap_step(
    model,
    batch_dm: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_atoms: int,
    dm_cfg: dict,
) -> None:
    latent_dim = dm_cfg["latent_dim"]
    model.train()
    mu, logvar, z, recon_tri = model(batch_dm)
    gt_full = torch.log1p(batch_dm)
    recon_full = utils.upper_tri_to_symmetric(recon_tri, num_atoms)
    loss, *_ = distmap_vae_loss(
        mu, logvar, gt_full, recon_full,
        model, latent_dim, device,
        dm_cfg["beta_kl"], dm_cfg["lambda_mse"], dm_cfg["lambda_w_gen"], dm_cfg["lambda_w_recon"],
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()


def _run_euclideanizer_step(
    embed,
    frozen_vae,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    latent_dim: int,
    eu_cfg: dict,
) -> None:
    embed.train()
    B = batch.shape[0]
    batch_dm = utils.get_distmaps(batch)
    gt_log = torch.log1p(batch_dm)
    with torch.no_grad():
        mu = frozen_vae.encode(gt_log)
        D_noneuclid = frozen_vae._decode_to_matrix(mu)
    recon_coords = embed.forward(D_noneuclid)
    D_euclid_recon = torch.log1p(utils.get_distmaps(recon_coords))
    z_gen = torch.randn(B, latent_dim, device=device)
    with torch.no_grad():
        D_noneuclid_gen = frozen_vae._decode_to_matrix(z_gen)
    D_euclid_gen = embed.forward_to_distmap(D_noneuclid_gen)
    loss, *_ = euclideanizer_loss(
        gt_log, D_euclid_recon, D_euclid_gen,
        gt_coords=batch, recon_coords=recon_coords,
        lambda_mse=eu_cfg["lambda_mse"],
        lambda_w_recon=eu_cfg["lambda_w_recon"],
        lambda_w_gen=eu_cfg["lambda_w_gen"],
        lambda_w_diag_recon=eu_cfg["lambda_w_diag_recon"],
        lambda_w_diag_gen=eu_cfg["lambda_w_diag_gen"],
        num_diags=eu_cfg["num_diags"],
        lambda_kabsch_mse=eu_cfg["lambda_kabsch_mse"],
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(embed.parameters(), max_norm=10.0)
    optimizer.step()


def calibrate_distmap_batch_size(
    model,
    dm_cfg: dict,
    coords: torch.Tensor,
    device: torch.device,
    threshold: float = 0.85,
    training_split: float = 0.8,
    split_seed: int = 0,
) -> int:
    """
    Find the largest batch size for DistMap training that stays under the given
    GPU memory fraction. Uses binary search with full training steps (forward,
    loss, backward, step). Caps at train split size.

    Returns a positive integer. If CUDA is not available, returns
    FALLBACK_BATCH_SIZE_NO_CUDA with a warning. If even batch_size=1 exceeds
    threshold, warns and returns 1.
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        warnings.warn(
            f"CUDA not available for calibration; using fallback batch_size={FALLBACK_BATCH_SIZE_NO_CUDA}.",
            UserWarning,
            stacklevel=2,
        )
        return FALLBACK_BATCH_SIZE_NO_CUDA

    num_atoms = coords.size(1)
    train_size = _get_train_size(coords, training_split, split_seed)
    max_cap = min(DEFAULT_HIGH, train_size)
    if max_cap < 1:
        return 1

    total_mem = torch.cuda.get_device_properties(device).total_memory
    limit = int(total_mem * threshold)

    def _probe(bs: int) -> bool | None:
        """Return True if under limit, False if over, None if OOM."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt = None
        try:
            opt = torch.optim.Adam(model.parameters(), lr=dm_cfg["learning_rate"])
            dummy_batch = torch.rand(bs, num_atoms, 3, device=device, dtype=coords.dtype)
            batch_dm = utils.get_distmaps(dummy_batch)
            _run_distmap_step(model, batch_dm, opt, device, num_atoms, dm_cfg)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return peak <= limit
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            if opt is not None:
                opt.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Double high until OOM or over threshold
    low, high = 1, 1
    while high <= max_cap:
        result = _probe(high)
        if result is True:
            low = high
            high = min(high * 2, max_cap)
            if high == low:
                break
        else:
            break

    if low == 1 and _probe(1) not in (True, None):
        # batch_size=1 still over threshold
        warnings.warn(
            "Calibration: batch_size=1 already exceeds memory threshold; using 1. Consider lowering calibration_memory_fraction.",
            UserWarning,
            stacklevel=2,
        )
        return 1

    # Binary search in [low, high]
    while low < high:
        mid = (low + high + 1) // 2
        result = _probe(mid)
        if result is True:
            low = mid
        else:
            high = mid - 1

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return low


def calibrate_euclideanizer_batch_size(
    embed,
    frozen_vae,
    eu_cfg: dict,
    coords: torch.Tensor,
    device: torch.device,
    threshold: float = 0.85,
    training_split: float = 0.8,
    split_seed: int = 0,
) -> int:
    """
    Find the largest batch size for Euclideanizer training that stays under the
    given GPU memory fraction. frozen_vae must already be on device in eval mode.
    Caps at train split size.
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        warnings.warn(
            f"CUDA not available for calibration; using fallback batch_size={FALLBACK_BATCH_SIZE_NO_CUDA}.",
            UserWarning,
            stacklevel=2,
        )
        return FALLBACK_BATCH_SIZE_NO_CUDA

    num_atoms = coords.size(1)
    latent_dim = frozen_vae._latent_space_dim
    train_size = _get_train_size(coords, training_split, split_seed)
    max_cap = min(DEFAULT_HIGH, train_size)
    if max_cap < 1:
        return 1

    total_mem = torch.cuda.get_device_properties(device).total_memory
    limit = int(total_mem * threshold)

    def _probe(bs: int) -> bool | None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt = None
        try:
            opt = torch.optim.Adam(embed.parameters(), lr=eu_cfg["learning_rate"])
            dummy_batch = torch.rand(bs, num_atoms, 3, device=device, dtype=coords.dtype)
            _run_euclideanizer_step(embed, frozen_vae, dummy_batch, opt, device, latent_dim, eu_cfg)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return peak <= limit
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            if opt is not None:
                opt.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    low, high = 1, 1
    while high <= max_cap:
        result = _probe(high)
        if result is True:
            low = high
            high = min(high * 2, max_cap)
            if high == low:
                break
        else:
            break

    if low == 1 and _probe(1) not in (True, None):
        warnings.warn(
            "Calibration: Euclideanizer batch_size=1 already exceeds memory threshold; using 1.",
            UserWarning,
            stacklevel=2,
        )
        return 1

    while low < high:
        mid = (low + high + 1) // 2
        result = _probe(mid)
        if result is True:
            low = mid
        else:
            high = mid - 1

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return low


def _run_decode_only(
    frozen_vae,
    embed,
    batch_size: int,
    device: torch.device,
    latent_dim: int,
) -> None:
    """Decode a batch of latents to coords (no gradients). VAE decode + Euclideanizer forward."""
    frozen_vae.eval()
    embed.eval()
    z = torch.randn(batch_size, latent_dim, device=device)
    with torch.no_grad():
        D_ne = frozen_vae._decode_to_matrix(z)
        embed(D_ne)


def calibrate_gen_decode_batch_size(
    frozen_vae,
    embed,
    device: torch.device,
    threshold: float = 0.85,
    latent_dim: int | None = None,
    num_atoms: int | None = None,
    max_cap: int = DEFAULT_HIGH_DECODE,
) -> int:
    """
    Find the largest decode batch size (inference: z -> VAE decode -> embed -> coords) under the
    given GPU memory fraction. Used for gen_decode_batch_size and query_batch_size (same profile).
    frozen_vae and embed must already be on device. Returns a positive int; on CPU returns fallback.
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        warnings.warn(
            f"CUDA not available for decode calibration; using fallback batch_size={FALLBACK_BATCH_SIZE_NO_CUDA}.",
            UserWarning,
            stacklevel=2,
        )
        return FALLBACK_BATCH_SIZE_NO_CUDA
    if latent_dim is None:
        latent_dim = frozen_vae._latent_space_dim
    total_mem = torch.cuda.get_device_properties(device).total_memory
    limit = int(total_mem * threshold)
    max_cap = min(max_cap, DEFAULT_HIGH_DECODE)

    def _probe(bs: int) -> bool | None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _run_decode_only(frozen_vae, embed, bs, device, latent_dim)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return peak <= limit
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    low, high = 1, 1
    while high <= max_cap:
        result = _probe(high)
        if result is True:
            low = high
            high = min(high * 2, max_cap)
            if high == low:
                break
        else:
            break

    if low == 1 and _probe(1) not in (True, None):
        warnings.warn(
            "Calibration: gen_decode_batch_size=1 already exceeds memory threshold; using 1.",
            UserWarning,
            stacklevel=2,
        )
        return 1

    while low < high:
        mid = (low + high + 1) // 2
        result = _probe(mid)
        if result is True:
            low = mid
        else:
            high = mid - 1

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return low
