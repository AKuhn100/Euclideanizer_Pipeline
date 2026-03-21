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
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import Normalize
import torch

from .plotting import _save_pdf_copy
from .plot_config import (
    PLOT_DPI,
    FONT_FAMILY,
    FONT_SIZE_AXIS,
    FONT_SIZE_TICK,
    HIST_BINS_DEFAULT,
    HIST_FILLED_EDGE_COLOR,
    LINEWIDTH_HIST_STEP,
    GEN_CAP_STACKED_FIGWIDTH,
    GEN_CAP_STACKED_ROW_HEIGHT,
)
from .distmap.sample import generate_samples
from . import rmsd as rmsd_module
from . import q_analysis as q_module


def _get_cmap(name: str = "viridis"):
    reg = getattr(matplotlib, "colormaps", None)
    if reg is not None:
        return reg[name]
    return cm.get_cmap(name)


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


def _apply_plain_number_axes(ax) -> None:
    """Avoid ``1e7``-style offsets and scientific tick notation on value axes."""
    for axis in (ax.xaxis, ax.yaxis):
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def _add_horizontal_n_colorbar(fig, cbar_ax, *, norm, cmap, n_min: int, n_max: int) -> None:
    """Horizontal colorbar for step-overlay tests / legacy layout."""
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label(
        "Number Of Generated Structures",
        fontsize=FONT_SIZE_AXIS,
        labelpad=4,
        family=FONT_FAMILY,
    )
    lo = float(np.log10(n_min))
    hi = float(np.log10(n_max))
    cbar.set_ticks([lo, hi])
    cbar.set_ticklabels([str(n_min), str(n_max)])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)


def _global_xlim_bins(by_n: dict[int, np.ndarray], n_bins: int) -> tuple[float, float, np.ndarray]:
    pieces = [by_n[n].ravel() for n in sorted(by_n.keys()) if by_n[n].size > 0]
    if not pieces:
        bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
        return 0.0, 1.0, bins
    all_vals = np.concatenate(pieces, axis=0)
    lo = float(np.percentile(all_vals, 1))
    hi = float(np.percentile(all_vals, 99))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if hi <= lo:
        span = max(abs(lo), 1.0) * 1e-6 + 1e-9
        lo, hi = lo - span, hi + span
    bins = np.linspace(lo, hi, int(n_bins) + 1)
    return lo, hi, bins


def _add_n_colorbar_right_of_stacked(
    fig,
    axes: list,
    *,
    norm,
    cmap,
    n_min: int,
    n_max: int,
    cbar_width: float = 0.02,
    gap: float = 0.01,
) -> None:
    fig.canvas.draw()
    p_top = axes[0].get_position()
    p_bot = axes[-1].get_position()
    x0 = p_top.x1 + gap
    y0 = p_bot.y0
    h = p_top.y1 - p_bot.y0
    cbar_ax = fig.add_axes([x0, y0, cbar_width, h])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.set_label(
        "Number Of Generated Structures",
        fontsize=FONT_SIZE_TICK,
        family=FONT_FAMILY,
        labelpad=-6,
        rotation=270,
    )
    cbar.ax.yaxis.set_label_position("right")
    lo = float(np.log10(n_min))
    hi = float(np.log10(n_max))
    cbar.set_ticks([lo, hi])
    cbar.set_ticklabels([str(n_min), str(n_max)])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)


def _generative_capacity_stacked_filled_figure(
    by_n: dict[int, np.ndarray],
    *,
    x_label: str,
    n_bins: int,
) -> plt.Figure:
    """One row per ``n`` (largest ``n`` at top); shared x; vertical ``log10(n)`` colorbar."""
    n_vals = sorted(by_n.keys())
    lo, hi, bins = _global_xlim_bins(by_n, n_bins)
    cmap = _get_cmap("viridis")
    n_min, n_max = min(n_vals), max(n_vals)
    norm = Normalize(vmin=np.log10(n_min), vmax=np.log10(n_max))

    fig_h = max(3.0, GEN_CAP_STACKED_ROW_HEIGHT * len(n_vals))
    fig, axes = plt.subplots(len(n_vals), 1, figsize=(GEN_CAP_STACKED_FIGWIDTH, fig_h), sharex=True)
    if len(n_vals) == 1:
        axes = [axes]
    for ax, n in zip(axes, reversed(n_vals)):
        vals = by_n[n].astype(np.float64).ravel()
        color = cmap(norm(np.log10(n)))
        ax.hist(
            vals,
            bins=bins,
            density=True,
            alpha=0.75,
            color=color,
            edgecolor=HIST_FILLED_EDGE_COLOR,
        )
        ax.set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        ax.set_xlim(lo, hi)
        _apply_plain_number_axes(ax)

    axes[-1].set_xlabel(x_label, fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.05)
    margin = 0.03
    for ax, n in zip(axes, reversed(n_vals)):
        ax.text(
            margin,
            1.0 - margin,
            f"N = {n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_TICK,
            fontweight="normal",
            family=FONT_FAMILY,
            color="0.15",
            zorder=8,
        )
    right = 0.88 if len(n_vals) >= 2 else 0.97
    fig.subplots_adjust(left=0.12, right=right, top=0.98, bottom=0.12)
    fig.canvas.draw()
    if len(n_vals) >= 2:
        _add_n_colorbar_right_of_stacked(fig, axes, norm=norm, cmap=cmap, n_min=n_min, n_max=n_max)
    return fig


def _distribution_panel(
    ax,
    by_n: dict[int, np.ndarray],
    x_label: str,
    *,
    n_bins: int,
    cbar_ax=None,
) -> None:
    """Overlapping step density histograms per `n` (``LINEWIDTH_HIST_STEP``); viridis by ``log10(n)``; optional top horizontal colorbar on ``cbar_ax``."""
    n_vals = sorted(by_n.keys())
    cmap = _get_cmap("viridis")
    n_min, n_max = min(n_vals), max(n_vals)
    norm = Normalize(vmin=np.log10(n_min), vmax=np.log10(n_max))
    fig = ax.figure

    pieces = [by_n[n].ravel() for n in n_vals if by_n[n].size > 0]
    if not pieces:
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        ax.set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        _apply_plain_number_axes(ax)
        if cbar_ax is not None:
            if len(n_vals) >= 2:
                _add_horizontal_n_colorbar(fig, cbar_ax, norm=norm, cmap=cmap, n_min=n_min, n_max=n_max)
            else:
                cbar_ax.set_visible(False)
        return

    all_vals = np.concatenate(pieces, axis=0)
    lo = float(np.percentile(all_vals, 1))
    hi = float(np.percentile(all_vals, 99))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if hi <= lo:
        span = max(abs(lo), 1.0) * 1e-6 + 1e-9
        lo, hi = lo - span, hi + span

    bins = np.linspace(lo, hi, int(n_bins) + 1)

    for zi, n in enumerate(n_vals):
        vals = by_n[n].astype(np.float64).ravel()
        if vals.size == 0:
            continue
        color = cmap(norm(np.log10(n)))
        ax.hist(
            vals,
            bins=bins,
            density=True,
            color=color,
            histtype="step",
            lw=LINEWIDTH_HIST_STEP,
            zorder=zi + 1,
        )

    ax.set_xlabel(x_label, fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax.set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    _apply_plain_number_axes(ax)
    if cbar_ax is not None:
        if len(n_vals) >= 2:
            _add_horizontal_n_colorbar(fig, cbar_ax, norm=norm, cmap=cmap, n_min=n_min, n_max=n_max)
        else:
            cbar_ax.set_visible(False)


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

    fig = _generative_capacity_stacked_filled_figure(
        by_n,
        x_label="Min RMSD To Nearest Generated Structure (Å)",
        n_bins=HIST_BINS_DEFAULT,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generative_capacity_rmsd.png")
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.15)
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

    fig = _generative_capacity_stacked_filled_figure(
        by_n,
        x_label="Max Q To Nearest Generated Structure",
        n_bins=HIST_BINS_DEFAULT,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generative_capacity_q.png")
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.15)
    if save_pdf_copy:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close(fig)

    del mat
    if not save_data and os.path.isdir(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    return out_path

