"""
Generative capacity analysis (RMSD and Q) for Euclideanizer runs.

For each metric:
- generate n_max structures once
- compute and persist full n_max x n_max pairwise matrix on disk
- evaluate nested subsamples for each n in n_structures
- save one figure + optional per-n data files
"""
from __future__ import annotations

import os
import shutil
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
import torch

from .plotting import _save_pdf_copy
from .plot_config import PLOT_DPI
from .distmap.sample import generate_samples
from . import rmsd as rmsd_module
from . import q_analysis as q_module


def _as_sorted_unique_ints(values) -> list[int]:
    vals = values if isinstance(values, list) else [values]
    out = sorted(set(int(v) for v in vals))
    return out


def build_nested_subsample_indices(n_max: int, n_values: list[int], seed: int) -> dict[int, np.ndarray]:
    """
    Build deterministic nested index sets by taking prefixes of one permutation.
    Used by tests to validate monotonic nested-subset behavior.
    """
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_max))
    return {int(n): perm[: int(n)] for n in sorted(n_values)}


def _generate_coords(
    *,
    n_gen: int,
    variance: float,
    latent_dim: int,
    device: torch.device,
    frozen_vae,
    embed,
    gen_decode_batch_size: int,
) -> np.ndarray:
    embed.eval()
    with torch.no_grad():
        z = generate_samples(n_gen, latent_dim, device, variance=variance)
        chunks = []
        for start in range(0, n_gen, gen_decode_batch_size):
            end = min(start + gen_decode_batch_size, n_gen)
            d_noneuclid = frozen_vae._decode_to_matrix(z[start:end])
            coords_chunk = embed(d_noneuclid)
            chunks.append(coords_chunk.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _compute_pairwise_matrix_to_disk(
    *,
    coords: np.ndarray,
    out_path: str,
    query_batch_size: int,
    pairwise_block_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    diagonal_fill: float,
) -> np.ndarray:
    n = coords.shape[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mat = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, n))
    mat[:] = 0.0
    for i0 in range(0, n, query_batch_size):
        i1 = min(i0 + query_batch_size, n)
        a = coords[i0:i1]
        for j0 in range(i0, n, query_batch_size):
            j1 = min(j0 + query_batch_size, n)
            b = coords[j0:j1]
            block = pairwise_block_fn(a, b).astype(np.float32, copy=False)
            mat[i0:i1, j0:j1] = block
            if i0 != j0:
                mat[j0:j1, i0:i1] = block.T
    np.fill_diagonal(mat, diagonal_fill)
    mat.flush()
    return mat


def _distribution_panel(ax, by_n: dict[int, np.ndarray], x_label: str) -> None:
    n_vals = sorted(by_n.keys())
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=np.log10(min(n_vals)), vmax=np.log10(max(n_vals)))
    all_vals = np.concatenate([by_n[n] for n in n_vals], axis=0)
    lo = float(np.percentile(all_vals, 1))
    hi = float(np.percentile(all_vals, 99))
    x = np.linspace(lo, hi, 400)
    for n in n_vals:
        vals = by_n[n].astype(np.float64)
        if vals.size < 2:
            continue
        kde = gaussian_kde(vals)
        y = kde(x)
        color = cmap(norm(np.log10(n)))
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.fill_between(x, 0.0, y, color=color, alpha=0.12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.08, pad=0.2)
    cbar.set_label("Number Of Generated Structures")
    cbar.set_ticks([np.log10(n) for n in n_vals])
    cbar.set_ticklabels([str(n) for n in n_vals])


def _curve_panel(ax, by_n: dict[int, np.ndarray], y_label: str) -> None:
    n_vals = sorted(by_n.keys())
    med = np.array([np.median(by_n[n]) for n in n_vals], dtype=np.float32)
    p25 = np.array([np.percentile(by_n[n], 25) for n in n_vals], dtype=np.float32)
    p75 = np.array([np.percentile(by_n[n], 75) for n in n_vals], dtype=np.float32)
    ax.plot(n_vals, med, marker="o", linewidth=2.0, color="#2a6fbb")
    ax.fill_between(n_vals, p25, p75, color="#2a6fbb", alpha=0.2)
    ax.set_xscale("log")
    ax.set_xticks(n_vals)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _p: str(int(v)) if v in n_vals else ""))
    ax.set_xlabel("Number Of Generated Structures")
    ax.set_ylabel(y_label)


def _save_per_n_npz(out_data_dir: str, filename_fn: Callable[[int], str], by_n: dict[int, np.ndarray], seed: int) -> None:
    os.makedirs(out_data_dir, exist_ok=True)
    for n, vals in by_n.items():
        out = os.path.join(out_data_dir, filename_fn(n))
        np.savez_compressed(
            out,
            values=vals.astype(np.float32),
            n=int(n),
            median=float(np.median(vals)),
            p25=float(np.percentile(vals, 25)),
            p75=float(np.percentile(vals, 75)),
            seed=int(seed),
        )


def run_generative_capacity_rmsd(
    *,
    run_dir: str,
    seed: int,
    latent_dim: int,
    device: torch.device,
    frozen_vae,
    embed,
    cfg_block: dict,
    display_root: str | None = None,
) -> str:
    n_values = _as_sorted_unique_ints(cfg_block["n_structures"])
    n_max = max(n_values)
    gen_decode_batch_size = int(cfg_block["gen_decode_batch_size"])
    query_batch_size = int(cfg_block["query_batch_size"])
    save_data = bool(cfg_block["save_data"])
    save_pdf_copy = bool(cfg_block["save_pdf_copy"])
    out_dir = os.path.join(run_dir, "analysis", "generative_capacity", "rmsd")
    data_dir = os.path.join(out_dir, "data")
    matrix_path = os.path.join(data_dir, "pairwise_matrix.npy")

    coords = _generate_coords(
        n_gen=n_max,
        variance=1.0,
        latent_dim=latent_dim,
        device=device,
        frozen_vae=frozen_vae,
        embed=embed,
        gen_decode_batch_size=gen_decode_batch_size,
    )
    mat = _compute_pairwise_matrix_to_disk(
        coords=coords,
        out_path=matrix_path,
        query_batch_size=query_batch_size,
        pairwise_block_fn=rmsd_module._rmsd_matrix_batch,
        diagonal_fill=np.inf,
    )

    nested_idx = build_nested_subsample_indices(n_max, n_values, seed)
    by_n: dict[int, np.ndarray] = {}
    for n in n_values:
        idx = nested_idx[n]
        sub = mat[np.ix_(idx, idx)]
        by_n[n] = np.min(sub, axis=1).astype(np.float32)

    if save_data:
        _save_per_n_npz(data_dir, lambda n: f"n{n}_min_rmsd.npz", by_n, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _distribution_panel(axes[0], by_n, "Min RMSD To Nearest Generated Structure (A)")
    _curve_panel(axes[1], by_n, "Median Min RMSD (A)")
    fig.suptitle(f"Generative Capacity - Min RMSD | Seed={seed}")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generative_capacity_rmsd.png")
    fig.savefig(out_path, dpi=PLOT_DPI)
    if save_pdf_copy:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close(fig)

    del mat
    if not save_data and os.path.isdir(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    return out_path


def run_generative_capacity_q(
    *,
    run_dir: str,
    seed: int,
    latent_dim: int,
    device: torch.device,
    frozen_vae,
    embed,
    cfg_block: dict,
    display_root: str | None = None,
) -> str:
    n_values = _as_sorted_unique_ints(cfg_block["n_structures"])
    n_max = max(n_values)
    gen_decode_batch_size = int(cfg_block["gen_decode_batch_size"])
    query_batch_size = int(cfg_block["query_batch_size"])
    delta = float(cfg_block["delta"])
    save_data = bool(cfg_block["save_data"])
    save_pdf_copy = bool(cfg_block["save_pdf_copy"])
    out_dir = os.path.join(run_dir, "analysis", "generative_capacity", "q")
    data_dir = os.path.join(out_dir, "data")
    matrix_path = os.path.join(data_dir, "pairwise_matrix.npy")

    coords = _generate_coords(
        n_gen=n_max,
        variance=1.0,
        latent_dim=latent_dim,
        device=device,
        frozen_vae=frozen_vae,
        embed=embed,
        gen_decode_batch_size=gen_decode_batch_size,
    )
    mat = _compute_pairwise_matrix_to_disk(
        coords=coords,
        out_path=matrix_path,
        query_batch_size=query_batch_size,
        pairwise_block_fn=lambda a, b: q_module._q_matrix_batch(a, b, delta),
        diagonal_fill=-np.inf,
    )

    nested_idx = build_nested_subsample_indices(n_max, n_values, seed)
    by_n: dict[int, np.ndarray] = {}
    for n in n_values:
        idx = nested_idx[n]
        sub = mat[np.ix_(idx, idx)]
        by_n[n] = np.max(sub, axis=1).astype(np.float32)

    if save_data:
        _save_per_n_npz(data_dir, lambda n: f"n{n}_max_q.npz", by_n, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _distribution_panel(axes[0], by_n, "Max Q To Nearest Generated Structure")
    _curve_panel(axes[1], by_n, "Median Max Q")
    fig.suptitle(f"Generative Capacity - Max Q | Seed={seed}")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generative_capacity_q.png")
    fig.savefig(out_path, dpi=PLOT_DPI)
    if save_pdf_copy:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close(fig)

    del mat
    if not save_data and os.path.isdir(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    return out_path

