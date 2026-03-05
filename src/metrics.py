"""
Distance-map metrics and experimental statistics.
"""
from typing import Optional, Tuple

import numpy as np
import torch


def distmap_bond_lengths(distmaps: np.ndarray) -> np.ndarray:
    """(B, N, N) -> bond lengths d(i, i+1) flattened."""
    N = distmaps.shape[-1]
    idx = np.arange(N - 1)
    return distmaps[:, idx, idx + 1].flatten()


def distmap_rg(distmaps: np.ndarray) -> np.ndarray:
    """(B, N, N) -> Rg per structure."""
    N = distmaps.shape[-1]
    rg_sq = np.sum(distmaps ** 2, axis=(1, 2)) / (2.0 * N * N)
    return np.sqrt(rg_sq)


def distmap_scaling(distmaps: np.ndarray, max_sep: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """(B, N, N) -> genomic_distances, mean_spatial_distances."""
    N = distmaps.shape[-1]
    if max_sep is None:
        max_sep = min(N - 1, 999)
    genomic_distances = np.arange(1, max_sep + 1)
    mean_dists = []
    for s in genomic_distances:
        idx_i = np.arange(N - s)
        diag_vals = distmaps[:, idx_i, idx_i + s]
        mean_dists.append(np.mean(diag_vals))
    return genomic_distances, np.array(mean_dists)


def compute_exp_statistics(
    coords_np: np.ndarray,
    device: torch.device,
    get_distmaps_fn,
    indices: Optional[np.ndarray] = None,
) -> dict:
    """Pre-compute experimental distance-map statistics. If indices is given, use only coords_np[indices]."""
    if indices is not None:
        coords_np = np.asarray(coords_np)[np.asarray(indices)]
    coords_tensor = torch.tensor(coords_np, dtype=torch.float32).to(device)
    chunk_size = 100
    all_dm = []
    for start in range(0, len(coords_tensor), chunk_size):
        chunk = coords_tensor[start : start + chunk_size]
        all_dm.append(get_distmaps_fn(chunk).cpu().numpy())
    exp_dm = np.concatenate(all_dm, axis=0)
    s, exp_sc = distmap_scaling(exp_dm)
    n_sample = min(100, len(exp_dm))
    return {
        "exp_distmaps": exp_dm,
        "exp_bonds": distmap_bond_lengths(exp_dm),
        "exp_rg": distmap_rg(exp_dm),
        "genomic_distances": s,
        "exp_scaling": exp_sc,
        "avg_exp_map": np.mean(exp_dm[:n_sample], axis=0),
    }
