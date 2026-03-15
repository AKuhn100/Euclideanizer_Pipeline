"""
Auto-calibration of batch sizes to stay under a GPU memory limit.

- Training: distmap.batch_size, euclideanizer.batch_size (when null in config).
- Inference: gen_decode_batch_size only (decode path; when null, calibrated from VRAM).
  Used for plotting and for decoding generated structures in analysis. query_batch_size
  is set in config (CPU RAM limit for analysis steps like RMSD/Q) and is not calibrated.

Uses fixed GB safety margin + min fraction reserved; true binary search refinement
and final verification.
"""
from __future__ import annotations

import warnings

import torch

from . import utils
from .distmap.loss import distmap_vae_loss
from .euclideanizer.loss import euclideanizer_loss

# Conservative default when CUDA is not available (used only when calibration is skipped).
FALLBACK_BATCH_SIZE_NO_CUDA = 32


def _compute_memory_limit(total_mem: int, safety_margin_gb: float) -> int:
    """
    Compute the usable memory ceiling for calibration.
    Uses a fixed GB margin. Hard floor: always allow at least 50% of VRAM to be used.
    """
    safety_bytes = int(safety_margin_gb * 1024**3)
    limit = total_mem - safety_bytes
    return max(limit, int(total_mem * 0.50))


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
    safety_margin_gb: float = 2.0,
    training_split: float = 0.8,
    split_seed: int = 0,
    training_batch_cap: int = 512,
    binary_search_steps: int = 5,
) -> int:
    """
    Find the largest batch size for DistMap training that stays under the computed
    GPU memory limit. Uses doubling, then binary search refinement, then final verification.
    Caps at min(training_batch_cap, train split size).
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
    max_cap = min(training_batch_cap, train_size)
    if max_cap < 1:
        return 1

    total_mem = torch.cuda.get_device_properties(device).total_memory
    limit = _compute_memory_limit(total_mem, safety_margin_gb)
    print(f"  Calibrating DistMap batch_size (safety_margin_gb={safety_margin_gb}, max={max_cap})...")

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
            peak = torch.cuda.max_memory_reserved()
            return peak <= limit
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # --- Phase 1: Doubling ---
    low = 0
    oom_high = None
    current = 1
    while current <= max_cap:
        result = _probe(current)
        if result is True:
            print(f"    batch_size={current} under limit")
            low = current
            next_size = min(current * 2, max_cap)
            if next_size == current:
                break
            current = next_size
        else:
            status = "OOM" if result is None else "over limit"
            print(f"    batch_size={current} {status}")
            oom_high = current
            break

    if low == 0:
        warnings.warn(
            "Calibration: batch_size=1 already exceeds memory threshold; using 1.",
            UserWarning,
            stacklevel=2,
        )
        return 1

    if oom_high is None:
        low = max_cap

    # --- Phase 2: Binary search refinement ---
    if oom_high is not None and oom_high - low > 1 and binary_search_steps > 0:
        search_low = low
        search_high = oom_high - 1
        steps_taken = 0
        while search_low < search_high and steps_taken < binary_search_steps:
            mid = (search_low + search_high + 1) // 2
            result = _probe(mid)
            if result is True:
                print(f"    refinement mid={mid} under limit")
                search_low = mid
            else:
                status = "OOM" if result is None else "over limit"
                print(f"    refinement mid={mid} {status}")
                search_high = mid - 1
            steps_taken += 1
        low = search_low
        print(f"    refinement complete after {steps_taken} step(s): batch_size={low}")

    # --- Phase 3: Final verification ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    final_ok = _probe(low)
    if final_ok is not True:
        backed_off = max(1, int(low * 0.75))
        print(f"    final verification failed for {low}; backing off to {backed_off}")
        low = backed_off

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Auto-calibrated batch_size: {low}")
    return low


def calibrate_euclideanizer_batch_size(
    embed,
    frozen_vae,
    eu_cfg: dict,
    coords: torch.Tensor,
    device: torch.device,
    safety_margin_gb: float = 2.0,
    training_split: float = 0.8,
    split_seed: int = 0,
    training_batch_cap: int = 512,
    binary_search_steps: int = 5,
) -> int:
    """
    Find the largest batch size for Euclideanizer training that stays under the
    computed GPU memory limit. frozen_vae must already be on device in eval mode.
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
    max_cap = min(training_batch_cap, train_size)
    if max_cap < 1:
        return 1

    total_mem = torch.cuda.get_device_properties(device).total_memory
    limit = _compute_memory_limit(total_mem, safety_margin_gb)
    print(f"  Calibrating Euclideanizer batch_size (safety_margin_gb={safety_margin_gb}, max={max_cap})...")

    def _probe(bs: int) -> bool | None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt = None
        try:
            opt = torch.optim.Adam(embed.parameters(), lr=eu_cfg["learning_rate"])
            dummy_batch = torch.rand(bs, num_atoms, 3, device=device, dtype=coords.dtype)
            _run_euclideanizer_step(embed, frozen_vae, dummy_batch, opt, device, latent_dim, eu_cfg)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            return peak <= limit
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    low = 0
    oom_high = None
    current = 1
    while current <= max_cap:
        result = _probe(current)
        if result is True:
            print(f"    batch_size={current} under limit")
            low = current
            next_size = min(current * 2, max_cap)
            if next_size == current:
                break
            current = next_size
        else:
            status = "OOM" if result is None else "over limit"
            print(f"    batch_size={current} {status}")
            oom_high = current
            break

    if low == 0:
        warnings.warn(
            "Calibration: Euclideanizer batch_size=1 already exceeds memory threshold; using 1.",
            UserWarning,
            stacklevel=2,
        )
        return 1

    if oom_high is None:
        low = max_cap

    if oom_high is not None and oom_high - low > 1 and binary_search_steps > 0:
        search_low = low
        search_high = oom_high - 1
        steps_taken = 0
        while search_low < search_high and steps_taken < binary_search_steps:
            mid = (search_low + search_high + 1) // 2
            result = _probe(mid)
            if result is True:
                print(f"    refinement mid={mid} under limit")
                search_low = mid
            else:
                status = "OOM" if result is None else "over limit"
                print(f"    refinement mid={mid} {status}")
                search_high = mid - 1
            steps_taken += 1
        low = search_low
        print(f"    refinement complete after {steps_taken} step(s): batch_size={low}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    final_ok = _probe(low)
    if final_ok is not True:
        backed_off = max(1, int(low * 0.75))
        print(f"    final verification failed for {low}; backing off to {backed_off}")
        low = backed_off

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Auto-calibrated batch_size: {low}")
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


def _run_decode_only_distmap(
    model,
    batch_size: int,
    device: torch.device,
    latent_dim: int,
) -> None:
    """Decode a batch of latents to distance maps (no gradients). DistMap VAE decode only."""
    model.eval()
    z = torch.randn(batch_size, latent_dim, device=device)
    with torch.no_grad():
        model._decode_to_matrix(z)


def calibrate_gen_decode_batch_size_distmap_only(
    model,
    device: torch.device,
    *,
    latent_dim: int,
    safety_margin_gb: float = 2.0,
    decode_batch_cap: int = 4096,
    binary_search_steps: int = 5,
) -> int:
    """
    Find the largest decode batch size for DistMap-only inference (z -> VAE decode to distance map)
    under the computed GPU memory limit. Used when plotting.gen_decode_batch_size is null and we
    need a value before any Euclideanizer exists (e.g. DistMap gen_variance plotting).
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        warnings.warn(
            f"CUDA not available for DistMap decode calibration; using fallback batch_size={FALLBACK_BATCH_SIZE_NO_CUDA}.",
            UserWarning,
            stacklevel=2,
        )
        return FALLBACK_BATCH_SIZE_NO_CUDA
    total_mem = torch.cuda.get_device_properties(device).total_memory
    limit = _compute_memory_limit(total_mem, safety_margin_gb)
    max_cap = decode_batch_cap
    print(f"  Calibrating gen_decode_batch_size (DistMap decode only, safety_margin_gb={safety_margin_gb}, max={max_cap})...")

    def _probe(bs: int) -> bool | None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _run_decode_only_distmap(model, bs, device, latent_dim)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            return peak <= limit
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    low = 0
    oom_high = None
    current = 1
    while current <= max_cap:
        result = _probe(current)
        if result is True:
            print(f"    batch_size={current} under limit")
            low = current
            next_size = min(current * 2, max_cap)
            if next_size == current:
                break
            current = next_size
        else:
            status = "OOM" if result is None else "over limit"
            print(f"    batch_size={current} {status}")
            oom_high = current
            break

    if low == 0:
        warnings.warn(
            "Calibration: gen_decode_batch_size=1 already exceeds memory threshold; using 1.",
            UserWarning,
            stacklevel=2,
        )
        return 1

    if oom_high is None:
        low = max_cap

    if oom_high is not None and oom_high - low > 1 and binary_search_steps > 0:
        search_low = low
        search_high = oom_high - 1
        steps_taken = 0
        while search_low < search_high and steps_taken < binary_search_steps:
            mid = (search_low + search_high + 1) // 2
            result = _probe(mid)
            if result is True:
                print(f"    refinement mid={mid} under limit")
                search_low = mid
            else:
                status = "OOM" if result is None else "over limit"
                print(f"    refinement mid={mid} {status}")
                search_high = mid - 1
            steps_taken += 1
        low = search_low
        print(f"    refinement complete after {steps_taken} step(s): batch_size={low}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    final_ok = _probe(low)
    if final_ok is not True:
        backed_off = max(1, int(low * 0.75))
        print(f"    final verification failed for {low}; backing off to {backed_off}")
        low = backed_off

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Auto-calibrated gen_decode_batch_size (DistMap decode): {low}")
    return low


def calibrate_gen_decode_batch_size(
    frozen_vae,
    embed,
    device: torch.device,
    safety_margin_gb: float = 2.0,
    latent_dim: int | None = None,
    num_atoms: int | None = None,
    decode_batch_cap: int = 4096,
    binary_search_steps: int = 5,
) -> int:
    """
    Find the largest decode batch size (inference: z -> VAE decode -> embed -> coords) under the
    computed GPU memory limit. Used for gen_decode_batch_size only (VRAM). query_batch_size
    is set in config (CPU RAM) and is not calibrated here.
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
    limit = _compute_memory_limit(total_mem, safety_margin_gb)
    max_cap = decode_batch_cap
    print(f"  Calibrating gen_decode_batch_size (safety_margin_gb={safety_margin_gb}, max={max_cap})...")

    def _probe(bs: int) -> bool | None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _run_decode_only(frozen_vae, embed, bs, device, latent_dim)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            return peak <= limit
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    low = 0
    oom_high = None
    current = 1
    while current <= max_cap:
        result = _probe(current)
        if result is True:
            print(f"    batch_size={current} under limit")
            low = current
            next_size = min(current * 2, max_cap)
            if next_size == current:
                break
            current = next_size
        else:
            status = "OOM" if result is None else "over limit"
            print(f"    batch_size={current} {status}")
            oom_high = current
            break

    if low == 0:
        warnings.warn(
            "Calibration: gen_decode_batch_size=1 already exceeds memory threshold; using 1.",
            UserWarning,
            stacklevel=2,
        )
        return 1

    if oom_high is None:
        low = max_cap

    if oom_high is not None and oom_high - low > 1 and binary_search_steps > 0:
        search_low = low
        search_high = oom_high - 1
        steps_taken = 0
        while search_low < search_high and steps_taken < binary_search_steps:
            mid = (search_low + search_high + 1) // 2
            result = _probe(mid)
            if result is True:
                print(f"    refinement mid={mid} under limit")
                search_low = mid
            else:
                status = "OOM" if result is None else "over limit"
                print(f"    refinement mid={mid} {status}")
                search_high = mid - 1
            steps_taken += 1
        low = search_low
        print(f"    refinement complete after {steps_taken} step(s): batch_size={low}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    final_ok = _probe(low)
    if final_ok is not True:
        backed_off = max(1, int(low * 0.75))
        print(f"    final verification failed for {low}; backing off to {backed_off}")
        low = backed_off

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Auto-calibrated batch_size: {low}")
    return low
