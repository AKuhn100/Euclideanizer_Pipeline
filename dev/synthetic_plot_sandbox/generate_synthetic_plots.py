#!/usr/bin/env python3
"""
Synthetic sufficiency meta-analysis + generative capacity figures.

Writes a minimal fake ``seed_*`` tree with **recon** NPZ (``test_recon_rmsd`` /
``test_recon_q``), then builds sufficiency figures using the **same layout rules** as
``Pipeline/src/meta_analysis.py`` (constants from ``plot_config`` when importable).

**Curve figures** (mirrors the main pipeline):

- **Sufficiency:** median test recon RMSD (or Q) vs **training split**, one line per
  ``max_data`` — family of curves for how much data / split affects recon saturation.
- **Generative capacity:** median per-structure **min RMSD** / **max Q** vs **N**
  (linear axes); curve color matches **`plot_config.COLOR_GEN`** (generated structures).

Generative capacity density/stacked plots are synthetic (no torch); stacked rows use
``GEN_CAP_STACKED_*`` from ``plot_config`` when available.

Use ``--no-clean --reuse-fake-base`` to skip rewriting ``fake_base/`` when
``.synthetic_sufficiency_fingerprint.yaml`` matches the requested grid.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if (_PIPELINE_ROOT / "src" / "plot_config.py").is_file():
    sys.path.insert(0, str(_PIPELINE_ROOT))

try:
    from src.plot_config import (
        COLOR_GEN,
        FONT_FAMILY,
        FONT_SIZE_AXIS,
        FONT_SIZE_SMALL,
        FONT_SIZE_TITLE,
        FONT_SIZE_TICK,
        GEN_CAP_STACKED_FIGWIDTH,
        GEN_CAP_STACKED_ROW_HEIGHT,
        GEN_CAP_CONVERGENCE_FIGSIZE,
        GEN_CAP_CONVERGENCE_LINEWIDTH,
        GEN_CAP_CONVERGENCE_MARKER_SIZE,
        GEN_CAP_CONVERGENCE_MARKER_EDGELINEWIDTH,
        GEN_CAP_CONVERGENCE_MARKER_FACE_COLOR,
        HIST_BINS_DEFAULT,
        META_CBAR_HEIGHT_FRAC,
        META_CBAR_WIDTH_FRAC,
        META_DIST_CB_GAP,
        META_HEATMAP_CB_GAP,
        PLOT_DPI,
        SUFFICIENCY_DIST_CB_GAP_EXTRA,
        SUFFICIENCY_DIST_FIG_WIDTH,
        SUFFICIENCY_CURVES_FIG_HEIGHT,
        SUFFICIENCY_DIST_FIGH_EXTRA,
        SUFFICIENCY_DIST_ROW_HEIGHT,
        SUFFICIENCY_HEATMAP_CELL_IN,
        SUFFICIENCY_HEATMAP_FIGH_EXTRA,
        sufficiency_dist_bottom_frac,
        sufficiency_dist_cbar_gap_frac,
        sufficiency_heatmap_bottom_frac,
        sufficiency_heatmap_cbar_gap_frac,
        sufficiency_heatmap_wspace,
        sufficiency_heatmap_ytick_fontsize,
    )

    CBAR_META_HEIGHT_FRAC = META_CBAR_HEIGHT_FRAC
    CBAR_META_WIDTH_FRAC = META_CBAR_WIDTH_FRAC
    META_DIST_CBAR_GAP_BELOW_AXIS = META_DIST_CB_GAP
    HEATMAP_CBAR_BELOW_AXES_FRAC = META_HEATMAP_CB_GAP
except ImportError:
    COLOR_GEN = "#D42248"
    PLOT_DPI = 150
    HIST_BINS_DEFAULT = 50
    FONT_FAMILY = "sans-serif"
    FONT_SIZE_AXIS = 10
    FONT_SIZE_TITLE = 11
    FONT_SIZE_TICK = 9
    CBAR_META_WIDTH_FRAC = 0.75
    CBAR_META_HEIGHT_FRAC = 0.028
    META_DIST_CBAR_GAP_BELOW_AXIS = 0.075
    HEATMAP_CBAR_BELOW_AXES_FRAC = 0.21
    SUFFICIENCY_DIST_FIG_WIDTH = 14.0
    SUFFICIENCY_DIST_ROW_HEIGHT = 3.05
    SUFFICIENCY_DIST_BOTTOM_BASE = 0.22
    SUFFICIENCY_DIST_BOTTOM_EXTRA = 0.042
    SUFFICIENCY_DIST_BOTTOM_CAP = 0.40
    SUFFICIENCY_DIST_CB_GAP_EXTRA = 0.030
    SUFFICIENCY_DIST_FIGH_EXTRA = 0.22
    SUFFICIENCY_HEATMAP_CELL_IN = 0.82
    SUFFICIENCY_HEATMAP_BOTTOM_BASE = 0.24
    SUFFICIENCY_HEATMAP_BOTTOM_EXTRA = 0.048
    SUFFICIENCY_HEATMAP_BOTTOM_CAP = 0.42
    SUFFICIENCY_HEATMAP_CB_GAP_EXTRA = 0.036
    SUFFICIENCY_HEATMAP_FIGH_EXTRA = 0.28
    FONT_SIZE_SMALL = 8
    FONT_SIZE_TINY = 6
    GEN_CAP_STACKED_FIGWIDTH = 11.0
    GEN_CAP_STACKED_ROW_HEIGHT = 3.05
    GEN_CAP_CONVERGENCE_FIGSIZE = (12.0, 4.8)
    GEN_CAP_CONVERGENCE_LINEWIDTH = 2.2
    GEN_CAP_CONVERGENCE_MARKER_SIZE = 6
    GEN_CAP_CONVERGENCE_MARKER_EDGELINEWIDTH = 1.2
    GEN_CAP_CONVERGENCE_MARKER_FACE_COLOR = "white"
    SUFFICIENCY_CURVES_FIG_HEIGHT = 5.2
    _INCH_H_LO, _INCH_H_SPAN = 16.0, 26.0
    _INCH_R_LO, _INCH_R_SPAN = 8.0, 8.0

    def _sufficiency_layout_inch_weight(fig_height_in, n_stack_rows):
        h = float(fig_height_in)
        n = max(int(n_stack_rows), 1)
        w_h = 0.0 if _INCH_H_SPAN <= 0 else (h - _INCH_H_LO) / _INCH_H_SPAN
        w_r = 0.0 if _INCH_R_SPAN <= 0 else (float(n) - _INCH_R_LO) / _INCH_R_SPAN
        return float(min(1.0, max(0.0, max(w_h, w_r))))

    def sufficiency_dist_bottom_frac(fig_height_in, small_split_steps, n_split_rows):
        h = max(float(fig_height_in), 1e-6)
        s = int(small_split_steps)
        legacy = min(SUFFICIENCY_DIST_BOTTOM_CAP, SUFFICIENCY_DIST_BOTTOM_BASE + s * SUFFICIENCY_DIST_BOTTOM_EXTRA)
        reserve_in = 0.92 + s * 0.24
        inch_frac = max(0.045, reserve_in / h)
        w = _sufficiency_layout_inch_weight(h, n_split_rows)
        blended = (1.0 - w) * legacy + w * inch_frac
        return min(SUFFICIENCY_DIST_BOTTOM_CAP, max(0.045, blended))

    def sufficiency_dist_cbar_gap_frac(fig_height_in, small_split_steps, n_split_rows):
        h = max(float(fig_height_in), 1e-6)
        s = int(small_split_steps)
        legacy = META_DIST_CBAR_GAP_BELOW_AXIS + s * SUFFICIENCY_DIST_CB_GAP_EXTRA
        gap_in = 0.38 + s * 0.055
        inch_frac = max(0.028, gap_in / h)
        w = _sufficiency_layout_inch_weight(h, n_split_rows)
        blended = (1.0 - w) * legacy + w * inch_frac
        return min(0.38, max(0.028, blended))

    def sufficiency_heatmap_bottom_frac(fig_height_in, small_grid_steps, n_heatmap_rows):
        h = max(float(fig_height_in), 1e-6)
        s = int(small_grid_steps)
        legacy = min(
            SUFFICIENCY_HEATMAP_BOTTOM_CAP,
            SUFFICIENCY_HEATMAP_BOTTOM_BASE + s * SUFFICIENCY_HEATMAP_BOTTOM_EXTRA,
        )
        reserve_in = 1.2 + s * 0.26
        inch_frac = max(0.10, reserve_in / h)
        w = _sufficiency_layout_inch_weight(h, n_heatmap_rows)
        blended = (1.0 - w) * legacy + w * inch_frac
        return min(SUFFICIENCY_HEATMAP_BOTTOM_CAP, max(0.10, blended))

    def sufficiency_heatmap_cbar_gap_frac(fig_height_in, small_grid_steps, n_heatmap_rows):
        h = max(float(fig_height_in), 1e-6)
        s = int(small_grid_steps)
        legacy = HEATMAP_CBAR_BELOW_AXES_FRAC + s * SUFFICIENCY_HEATMAP_CB_GAP_EXTRA
        gap_in = 0.72 + s * 0.065
        inch_frac = max(0.058, gap_in / h)
        w = _sufficiency_layout_inch_weight(h, n_heatmap_rows)
        blended = (1.0 - w) * legacy + w * inch_frac
        return min(0.36, max(0.058, blended))

    def sufficiency_heatmap_ytick_fontsize(n_rows):
        n = max(int(n_rows), 1)
        return float(max(FONT_SIZE_TINY, min(float(FONT_SIZE_TICK), 96.0 / n)))

    def sufficiency_heatmap_wspace(n_rows):
        n = max(int(n_rows), 1)
        return float(max(0.09, min(0.26, 0.062 + 0.0142 * n)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml
from matplotlib import cm
from matplotlib.colors import Normalize


def _get_cmap(name: str = "viridis"):
    """Resolve colormap without deprecated ``cm.get_cmap`` (Matplotlib ≥3.7)."""
    reg = getattr(matplotlib, "colormaps", None)
    if reg is not None:
        return reg[name]
    return cm.get_cmap(name)


# ---------------------------------------------------------------------------
# Generative capacity sandbox-only layout (overlay / legacy gridspec)
# ---------------------------------------------------------------------------
LINEWIDTH_HIST_STEP = 2.4
HIST_FILLED_EDGE_COLOR = "none"
GEN_CAP_FIGSIZE = (10.5, 4.6)
GEN_CAP_GRIDSPEC_HEIGHT_RATIOS = (1, 6)
GEN_CAP_GRIDSPEC_HSPACE = 0.22
GEN_CAP_SUBPLOT_MARGINS = dict(left=0.11, right=0.96, top=0.86, bottom=0.14)
GEN_CBAR_HEIGHT_FRAC = 0.014
GEN_CBAR_GAP_BELOW_AXES = 0.03

# Default synthetic sufficiency grid (5 × 6 heatmap per seed).
SYNTHETIC_SEEDS = (1, 2)
SYNTHETIC_TRAINING_SPLITS = (0.5, 0.58, 0.66, 0.74, 0.82)
SYNTHETIC_MAX_DATA = (400, 800, 1600, 3200, 5000, 10000)
SYNTHETIC_N_STRUCTURES_PER_RUN = 480

# Stress grid: many splits × many max_data (``--large-grid``).
LARGE_GRID_SEEDS = (1, 2, 3, 4)
LARGE_GRID_N_SPLITS = 14
LARGE_GRID_N_MAX_DATA = 12
LARGE_GRID_N_STRUCTURES = 640
LARGE_GRID_GEN_CAP_N = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]


def _large_grid_training_splits() -> tuple[float, ...]:
    return tuple(np.round(np.linspace(0.5, 0.92, LARGE_GRID_N_SPLITS), 4))


def _large_grid_max_data() -> tuple[int, ...]:
    raw = np.logspace(np.log10(250), np.log10(30_000), LARGE_GRID_N_MAX_DATA)
    return tuple(sorted({int(round(x)) for x in raw}))


# =============================================================================
# Sufficiency: on-disk scan (same NPZ layout as pipeline meta_analysis)
# =============================================================================


@dataclass
class SufficiencyPoint:
    seed: int
    max_data: int | None
    training_split: float
    rmsd_values: np.ndarray
    q_values: np.ndarray


def _parse_seed_and_split(seed_name: str) -> tuple[int, float | None] | None:
    if not seed_name.startswith("seed_"):
        return None
    rest = seed_name[5:]
    if rest.isdigit() or "_maxdata_" in rest and rest.split("_maxdata_", 1)[0].isdigit():
        left = rest.split("_maxdata_", 1)[0]
        return int(left), None
    if "_split_" in rest:
        left, right = rest.split("_split_", 1)
        if "_maxdata_" in right:
            right = right.split("_maxdata_", 1)[0]
        if left.isdigit():
            try:
                return int(left), float(right)
            except ValueError:
                return None
    return None


def _iter_euclideanizer_runs(seed_dir: str):
    distmap_dir = os.path.join(seed_dir, "distmap")
    if not os.path.isdir(distmap_dir):
        return
    for dm_name in sorted(os.listdir(distmap_dir), key=lambda x: (len(x), x)):
        if not dm_name.isdigit():
            continue
        eu_dir = os.path.join(distmap_dir, dm_name, "euclideanizer")
        if not os.path.isdir(eu_dir):
            continue
        for eu_name in sorted(os.listdir(eu_dir), key=lambda x: (len(x), x)):
            if not eu_name.isdigit():
                continue
            run_dir = os.path.join(eu_dir, eu_name)
            if os.path.isdir(run_dir):
                yield run_dir


def _collect_recon_npz_paths(recon_root: str, metric: str) -> list[tuple[str, str]]:
    fname = "rmsd_recon_data.npz" if metric == "rmsd" else "q_recon_data.npz"
    if not os.path.isdir(recon_root):
        return []
    out: list[tuple[str, str]] = []
    direct = os.path.join(recon_root, "data", fname)
    if os.path.isfile(direct):
        out.append(("", direct))
    for name in sorted(os.listdir(recon_root)):
        sub = os.path.join(recon_root, name)
        if not os.path.isdir(sub):
            continue
        cand = os.path.join(sub, "data", fname)
        if os.path.isfile(cand):
            out.append((name, cand))
    return out


def _load_recon_metric_arrays(seed_dir: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    for run_dir in _iter_euclideanizer_runs(seed_dir) or []:
        rmsd_recon_root = os.path.join(run_dir, "analysis", "rmsd", "recon")
        q_recon_root = os.path.join(run_dir, "analysis", "q", "recon")
        r_by_rel = {rel: p for rel, p in _collect_recon_npz_paths(rmsd_recon_root, "rmsd")}
        q_by_rel = {rel: p for rel, p in _collect_recon_npz_paths(q_recon_root, "q")}
        for rel in sorted(set(r_by_rel) & set(q_by_rel)):
            try:
                lr = np.load(r_by_rel[rel], allow_pickle=False)
                lq = np.load(q_by_rel[rel], allow_pickle=False)
                tr = lr["test_recon_rmsd"]
                tq = lq["test_recon_q"]
                lr.close()
                lq.close()
                return np.asarray(tr, dtype=np.float32), np.asarray(tq, dtype=np.float32)
            except Exception:
                continue
    return None, None


def _kde_curve(values: np.ndarray, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, 400, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64).ravel()
    if len(v) < 2:
        return x, np.zeros_like(x)
    std = np.std(v)
    bw = (std if std > 1e-12 else 1e-6) * (len(v) ** (-1.0 / 5.0))
    bw = max(bw, 1e-6)
    diffs = (x[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * diffs * diffs).sum(axis=1) / (len(v) * bw * np.sqrt(2.0 * np.pi))
    return x, dens


def _max_data_from_seed_dir(seed_dir: str, fallback_max_data: int | None) -> int | None:
    cfg_path = os.path.join(seed_dir, "pipeline_config.yaml")
    if not os.path.isfile(cfg_path):
        return fallback_max_data
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        md = (cfg.get("data") or {}).get("max_data")
        if isinstance(md, list):
            return int(md[0]) if md else fallback_max_data
        if md is None:
            return None
        return int(md)
    except Exception:
        return fallback_max_data


def collect_sufficiency_points(base_output_dir: str, max_data_values: list[int | None]) -> dict[int, list[SufficiencyPoint]]:
    points_by_seed: dict[int, list[SufficiencyPoint]] = {}
    fallback_md = max_data_values[0] if max_data_values else None
    for seed_name in sorted(os.listdir(base_output_dir) if os.path.isdir(base_output_dir) else []):
        parsed = _parse_seed_and_split(seed_name)
        if parsed is None:
            continue
        seed, split_opt = parsed
        seed_dir = os.path.join(base_output_dir, seed_name)
        if not os.path.isdir(seed_dir):
            continue
        split = float(split_opt) if split_opt is not None else 0.8
        rmsd_vals, q_vals = _load_recon_metric_arrays(seed_dir)
        if rmsd_vals is None or q_vals is None:
            continue
        md = _max_data_from_seed_dir(seed_dir, fallback_md)
        points_by_seed.setdefault(seed, []).append(
            SufficiencyPoint(
                seed=seed,
                max_data=md,
                training_split=split,
                rmsd_values=rmsd_vals,
                q_values=q_vals,
            )
        )
    return points_by_seed


def _save_pdf_if_enabled(fig, png_path: str, enabled: bool) -> None:
    if not enabled:
        return
    base, _ = os.path.splitext(png_path)
    fig.savefig(base + ".pdf")


def _save_pdf_next_to_png(fig, png_path: str) -> None:
    base, _ = os.path.splitext(png_path)
    fig.savefig(base + ".pdf")


def _horizontal_cbar_axes_in_span(
    fig,
    x_left: float,
    x_right: float,
    y_bottom: float,
    height_frac: float,
    *,
    width_frac: float = CBAR_META_WIDTH_FRAC,
) -> object:
    """Horizontal colorbar: *width_frac* of [x_left, x_right], centered in that span (figure coords)."""
    span = max(x_right - x_left, 1e-6)
    w = span * width_frac
    x0 = x_left + 0.5 * (span - w)
    return fig.add_axes([x0, y_bottom, w, height_frac])


def _y_below_axes(ax, gap: float, height_frac: float) -> float:
    """Figure y (bottom of strip) sitting *gap* below axes bbox *ax* (call ``fig.canvas.draw()`` first)."""
    pos = ax.get_position()
    return pos.y0 - gap - height_frac


def _style_horizontal_meta_colorbar(cbar, label: str, *, labelpad: int = 3) -> None:
    """Label above the color strip; tick labels below (matplotlib default for horizontal is label below)."""
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label(label, fontsize=FONT_SIZE_TICK, family=FONT_FAMILY, labelpad=labelpad)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)


def _training_split_colorbar_below_panels(
    fig,
    ax_left,
    ax_right,
    *,
    cmap,
    height_frac: float = CBAR_META_HEIGHT_FRAC,
    gap_frac: float | None = None,
) -> None:
    """75%% of panel span, centered; label above the bar."""
    fig.canvas.draw()
    p0, p1 = ax_left.get_position(), ax_right.get_position()
    row_bottom = min(p0.y0, p1.y0)
    gap = META_DIST_CBAR_GAP_BELOW_AXIS if gap_frac is None else gap_frac
    y = max(0.02, row_bottom - gap - height_frac)
    cax = _horizontal_cbar_axes_in_span(fig, p0.x0, p1.x1, y, height_frac)
    sm = cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["0%", "100%"])
    _style_horizontal_meta_colorbar(cbar, "Training Split", labelpad=3)


def _heatmap_colorbar_below_panels(
    fig,
    ax_left,
    ax_right,
    *,
    cmap,
    vmin: float,
    vmax: float,
    label: str,
    gap_frac: float | None = None,
) -> None:
    fig.canvas.draw()
    p0, p1 = ax_left.get_position(), ax_right.get_position()
    row_bottom = min(p0.y0, p1.y0)
    gap = HEATMAP_CBAR_BELOW_AXES_FRAC if gap_frac is None else gap_frac
    y = max(0.02, row_bottom - gap - CBAR_META_HEIGHT_FRAC)
    cax = _horizontal_cbar_axes_in_span(fig, p0.x0, p1.x1, y, CBAR_META_HEIGHT_FRAC)
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    _style_horizontal_meta_colorbar(cbar, label, labelpad=4)


def _sufficiency_heatmap_small_grid_steps(n_r: int, n_c: int) -> int:
    return max(0, 4 - min(n_r, n_c))


def _sufficiency_dist_small_split_steps(n_split_rows: int) -> int:
    return max(0, 4 - n_split_rows)


def _norm01_per_grid(grid: np.ndarray) -> np.ma.MaskedArray:
    """Normalize finite entries to [0, 1] per grid for a shared 0–1 colorbar."""
    g = np.asarray(grid, dtype=np.float64)
    mask = ~np.isfinite(g)
    if mask.all():
        return np.ma.masked_invalid(g)
    lo = np.nanmin(g)
    hi = np.nanmax(g)
    if hi <= lo:
        out = np.zeros_like(g)
        return np.ma.array(out, mask=mask)
    out = (g - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return np.ma.array(out, mask=mask)


def _max_data_colorbar_tick_label(md: int | None) -> str:
    """Short label for colorbar ticks (Title Case friendly)."""
    return "Full" if md is None else f"{int(md):,}"


def _curves_colorbar_max_structures_below(
    fig,
    ax_left,
    ax_right,
    *,
    cmap,
    n_curves: int,
    tick_labels: list[str],
    gap_frac: float = 0.11,
) -> None:
    """Horizontal viridis strip under both axes; tick marks = one per curve / max_data."""
    fig.canvas.draw()
    p0, p1 = ax_left.get_position(), ax_right.get_position()
    row_bottom = min(p0.y0, p1.y0)
    y = max(0.02, row_bottom - gap_frac - CBAR_META_HEIGHT_FRAC)
    cax = _horizontal_cbar_axes_in_span(fig, p0.x0, p1.x1, y, CBAR_META_HEIGHT_FRAC)
    vmax = float(max(n_curves - 1, 1))
    sm = cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    if n_curves <= 1:
        cbar.set_ticks([0.5 * vmax])
        cbar.set_ticklabels(tick_labels[:1])
    else:
        ticks = np.arange(n_curves, dtype=float)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels[:n_curves])
    cbar.ax.tick_params(labelsize=max(6, FONT_SIZE_SMALL - 1))
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Max Structures", fontsize=FONT_SIZE_TICK, family=FONT_FAMILY, labelpad=4)


def _save_sufficiency_split_curve_figures(
    seed_out: str,
    points: list[SufficiencyPoint],
    save_pdf_copy: bool,
) -> bool:
    """Median test recon RMSD / Q vs training split; one curve per ``max_data`` (sandbox exploration)."""
    split_set = sorted({p.training_split for p in points})
    if len(split_set) < 2:
        return False
    max_data_set = sorted({p.max_data for p in points}, key=lambda x: (-1 if x is None else x))
    n_md = len(max_data_set)
    if n_md < 1:
        return False

    curves_dir = os.path.join(seed_out, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    cmap = _get_cmap("viridis")
    norm_idx = Normalize(vmin=0.0, vmax=float(max(n_md - 1, 1)))

    fig, (ax_r, ax_q) = plt.subplots(
        1, 2, figsize=(SUFFICIENCY_DIST_FIG_WIDTH, SUFFICIENCY_CURVES_FIG_HEIGHT)
    )
    for j, md in enumerate(max_data_set):
        subset = [p for p in points if p.max_data == md]
        if not subset:
            continue
        by_split = {p.training_split: p for p in subset}
        xs = sorted(by_split.keys())
        if len(xs) < 1:
            continue
        med_r = [float(np.median(by_split[s].rmsd_values)) for s in xs]
        med_q = [float(np.median(by_split[s].q_values)) for s in xs]
        color = cmap(norm_idx(j))
        ax_r.plot(xs, med_r, "o-", color=color, lw=2.0, ms=5, alpha=0.9)
        ax_q.plot(xs, med_q, "o-", color=color, lw=2.0, ms=5, alpha=0.9)

    tick_lbls = [_max_data_colorbar_tick_label(m) for m in max_data_set]
    for ax, ylab in (
        (ax_r, "Median Test Recon RMSD"),
        (ax_q, "Median Test Recon Q"),
    ):
        ax.set_xlabel("Training Split", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        ax.set_ylabel(ylab, fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        ax.set_xlim(min(split_set) - 0.02, max(split_set) + 0.02)
        _apply_plain_number_axes(ax)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.20, wspace=0.28)
    _curves_colorbar_max_structures_below(
        fig,
        ax_r,
        ax_q,
        cmap=cmap,
        n_curves=n_md,
        tick_labels=tick_lbls,
    )
    png_path = os.path.join(curves_dir, "sufficiency_median_recon_vs_split_by_max_data.png")
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.18)
    _save_pdf_if_enabled(fig, png_path, save_pdf_copy)
    plt.close(fig)
    return True


def _save_generative_capacity_median_convergence(
    gc_dir: Path,
    by_r: dict[int, np.ndarray],
    by_q: dict[int, np.ndarray],
    *,
    save_pdf: bool,
) -> None:
    """Median min RMSD / median max Q vs N (linear axes); synthetic gen-cap exploration."""
    n_sorted = sorted(by_r.keys())
    if len(n_sorted) < 2:
        return
    med_r = [float(np.median(by_r[n])) for n in n_sorted]
    med_q = [float(np.median(by_q[n])) for n in n_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=GEN_CAP_CONVERGENCE_FIGSIZE)
    ax1.plot(
        n_sorted,
        med_r,
        "o-",
        color=COLOR_GEN,
        lw=GEN_CAP_CONVERGENCE_LINEWIDTH,
        ms=GEN_CAP_CONVERGENCE_MARKER_SIZE,
        markerfacecolor=GEN_CAP_CONVERGENCE_MARKER_FACE_COLOR,
        markeredgewidth=GEN_CAP_CONVERGENCE_MARKER_EDGELINEWIDTH,
    )
    ax1.set_xlabel("Number Of Gen Structures", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax1.set_ylabel("Median Min RMSD To Nearest Gen", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    _apply_plain_number_axes(ax1)

    ax2.plot(
        n_sorted,
        med_q,
        "o-",
        color=COLOR_GEN,
        lw=GEN_CAP_CONVERGENCE_LINEWIDTH,
        ms=GEN_CAP_CONVERGENCE_MARKER_SIZE,
        markerfacecolor=GEN_CAP_CONVERGENCE_MARKER_FACE_COLOR,
        markeredgewidth=GEN_CAP_CONVERGENCE_MARKER_EDGELINEWIDTH,
    )
    ax2.set_xlabel("Number Of Gen Structures", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax2.set_ylabel("Median Max Q To Nearest Gen", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    _apply_plain_number_axes(ax2)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.16, wspace=0.3)
    p = gc_dir / "convergence_median_vs_n_rmsd_q.png"
    fig.savefig(str(p), dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.15)
    if save_pdf:
        _save_pdf_next_to_png(fig, str(p))
    plt.close(fig)


def run_sufficiency_plots_sandbox(
    *,
    base_output_dir: str,
    max_data_values: list[int | None],
    save_pdf_copy: bool,
    log: Callable[[str], None] | None = None,
) -> bool:
    """Sufficiency distributions + heatmaps (layout aligned with ``Pipeline/src/meta_analysis.py``)."""
    points_by_seed = collect_sufficiency_points(base_output_dir, max_data_values)
    if not points_by_seed:
        if log:
            log("Sufficiency sandbox: no recon analysis NPZ data found. Skipping.")
        return False

    out_root = os.path.join(base_output_dir, "meta_analysis", "sufficiency")
    os.makedirs(out_root, exist_ok=True)
    made_any = False
    cmap = _get_cmap("viridis")
    split_norm = Normalize(vmin=0.0, vmax=1.0)

    for seed, points in sorted(points_by_seed.items()):
        seed_out = os.path.join(out_root, f"seed_{seed}")
        dist_root = os.path.join(seed_out, "distributions")
        heat_root = os.path.join(seed_out, "heatmap")
        os.makedirs(dist_root, exist_ok=True)
        os.makedirs(heat_root, exist_ok=True)

        max_data_set = sorted({p.max_data for p in points}, key=lambda x: (-1 if x is None else x))
        split_set = sorted({p.training_split for p in points})

        for md in max_data_set:
            subset = [p for p in points if p.max_data == md]
            if not subset:
                continue
            # Stacked borderless histograms: one row per training split, two columns.
            # Highest split at top (darkest viridis); shared x axis per column.
            splits_ordered = list(reversed(sorted({p.training_split for p in subset})))
            n_splits_local = len(splits_ordered)
            small_d = _sufficiency_dist_small_split_steps(n_splits_local)
            row_h = SUFFICIENCY_DIST_ROW_HEIGHT
            fig_h = max(4.0 + small_d * SUFFICIENCY_DIST_FIGH_EXTRA, row_h * n_splits_local)
            dist_bottom = sufficiency_dist_bottom_frac(fig_h, small_d, n_splits_local)
            dist_cb_gap = sufficiency_dist_cbar_gap_frac(fig_h, small_d, n_splits_local)

            all_r = np.concatenate([p.rmsd_values for p in subset], axis=0)
            all_q = np.concatenate([p.q_values for p in subset], axis=0)
            xr_min = float(np.percentile(all_r, 1))
            xr_max = float(np.percentile(all_r, 99))
            xq_min = float(np.percentile(all_q, 1))
            xq_max = float(np.percentile(all_q, 99))
            xq_min, xq_max = max(0.0, xq_min), min(1.0, xq_max)
            bins_r = np.linspace(xr_min, xr_max, HIST_BINS_DEFAULT + 1)
            bins_q = np.linspace(xq_min, xq_max, HIST_BINS_DEFAULT + 1)

            fig, axes = plt.subplots(
                n_splits_local,
                2,
                figsize=(SUFFICIENCY_DIST_FIG_WIDTH, fig_h),
                sharex="col",
                gridspec_kw=dict(wspace=0.3),
            )
            if n_splits_local == 1:
                axes = axes[np.newaxis, :]

            for row_i, split in enumerate(splits_ordered):
                p_match = next((p for p in subset if p.training_split == split), None)
                color = cmap(split_norm(split))
                label = f"{int(round(split * 100))}%"
                for ax, vals, bins in (
                    (axes[row_i, 0], p_match.rmsd_values if p_match else np.zeros(1), bins_r),
                    (axes[row_i, 1], p_match.q_values if p_match else np.zeros(1), bins_q),
                ):
                    if p_match and vals.size > 0:
                        ax.hist(vals, bins=bins, density=True, color=color, alpha=0.75, edgecolor="none")
                    ax.set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
                    _apply_plain_number_axes(ax)
                for ax_col in (axes[row_i, 0], axes[row_i, 1]):
                    ax_col.text(
                        0.02,
                        0.97,
                        label,
                        transform=ax_col.transAxes,
                        ha="left",
                        va="top",
                        fontsize=FONT_SIZE_TICK,
                        family=FONT_FAMILY,
                        color="0.15",
                    )

            # Equalise y limits per column so rows are visually comparable
            for col_i in range(2):
                col_axes = [axes[r, col_i] for r in range(n_splits_local)]
                ymax = max(ax.get_ylim()[1] for ax in col_axes)
                for ax in col_axes:
                    ax.set_ylim(0, ymax * 1.05)

            axes[0, 0].set_title(f"Test Recon RMSD | Max Data={md}", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
            axes[0, 1].set_title(f"Test Recon Q | Max Data={md}", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
            axes[-1, 0].set_xlabel("Test Recon RMSD", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
            axes[-1, 1].set_xlabel("Test Recon Q", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)

            fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=dist_bottom)
            _cbar_hfrac = CBAR_META_HEIGHT_FRAC * 6.2 / fig_h
            _training_split_colorbar_below_panels(
                fig,
                axes[-1, 0],
                axes[-1, 1],
                cmap=cmap,
                height_frac=_cbar_hfrac,
                gap_frac=dist_cb_gap,
            )

            md_tag = "all" if md is None else str(md)
            out_dir = os.path.join(dist_root, f"max_data_{md_tag}")
            os.makedirs(out_dir, exist_ok=True)
            png_path = os.path.join(out_dir, "distributions_rmsd_q.png")
            fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.15)
            _save_pdf_if_enabled(fig, png_path, save_pdf_copy)
            plt.close(fig)
            made_any = True

        if _save_sufficiency_split_curve_figures(seed_out, points, save_pdf_copy):
            made_any = True

        md_vals = [m for m in max_data_set if m is not None]
        if not md_vals:
            continue
        md_to_j = {m: j for j, m in enumerate(md_vals)}
        split_to_i = {s: i for i, s in enumerate(split_set)}
        rmsd_grid = np.full((len(split_set), len(md_vals)), np.nan, dtype=np.float32)
        q_grid = np.full((len(split_set), len(md_vals)), np.nan, dtype=np.float32)
        for p in points:
            if p.max_data is None:
                continue
            i = split_to_i[p.training_split]
            j = md_to_j[p.max_data]
            rmsd_grid[i, j] = float(np.median(p.rmsd_values))
            q_grid[i, j] = float(np.median(p.q_values))

        r_plot = _norm01_per_grid(rmsd_grid)
        q_plot = _norm01_per_grid(q_grid)

        n_r = len(split_set)
        n_c = len(md_vals)
        small = _sufficiency_heatmap_small_grid_steps(n_r, n_c)
        cell_in = SUFFICIENCY_HEATMAP_CELL_IN
        w_per = max(3.2, cell_in * n_c + 0.55)
        fig_h = max(4.0 + small * SUFFICIENCY_HEATMAP_FIGH_EXTRA, cell_in * n_r + 1.35)
        bottom_frac = sufficiency_heatmap_bottom_frac(fig_h, small, n_r)
        cb_gap = sufficiency_heatmap_cbar_gap_frac(fig_h, small, n_r)
        ytick_fs = sufficiency_heatmap_ytick_fontsize(n_r)
        h_wspace = sufficiency_heatmap_wspace(n_r)
        gap = 0.15
        fig_w = 2 * w_per + gap + 0.6
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw=dict(wspace=h_wspace))
        ax_r, ax_q = axes
        ax_r.imshow(r_plot, aspect="equal", cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        ax_q.imshow(q_plot, aspect="equal", cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        split_labels = [f"{int(round(s * 100))}%" for s in split_set]
        md_labels = [str(v) for v in md_vals]
        for ax, title in ((ax_r, "Median Test Recon RMSD"), (ax_q, "Median Test Recon Q")):
            ax.set_title(title, fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
            ax.set_xlabel("Max Structures", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY, labelpad=12)
            ax.set_xticks(np.arange(len(md_vals)))
            ax.set_xticklabels(md_labels, rotation=35, ha="right", fontsize=FONT_SIZE_TICK)
            ax.set_yticks(np.arange(len(split_set)))
        ax_r.set_yticklabels(split_labels, fontsize=ytick_fs)
        ax_q.set_yticklabels(split_labels, fontsize=ytick_fs)
        ax_r.set_ylabel("Training Split", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        ax_q.set_ylabel("")

        fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=bottom_frac)
        _heatmap_colorbar_below_panels(
            fig,
            ax_r,
            ax_q,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            label="Normalized Median",
            gap_frac=cb_gap,
        )
        heat_png = os.path.join(heat_root, "sufficiency_heatmap_rmsd_q.png")
        fig.savefig(heat_png, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.2)
        _save_pdf_if_enabled(fig, heat_png, save_pdf_copy)
        plt.close(fig)
        made_any = True

    return made_any


# =============================================================================
# Generative capacity (no torch)
# =============================================================================


def _apply_plain_number_axes(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def _add_n_colorbar_below_ax(fig, main_ax, *, norm, cmap, n_min: int, n_max: int) -> None:
    """Thin colorbar: 75%% of *main_ax* width, centered under that axes (log10 N scale)."""
    fig.canvas.draw()
    pos = main_ax.get_position()
    y = max(0.02, _y_below_axes(main_ax, GEN_CBAR_GAP_BELOW_AXES, GEN_CBAR_HEIGHT_FRAC))
    cbar_ax = _horizontal_cbar_axes_in_span(fig, pos.x0, pos.x1, y, GEN_CBAR_HEIGHT_FRAC)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label(
        "Number Of Gen Structures",
        fontsize=FONT_SIZE_TICK,
        family=FONT_FAMILY,
        labelpad=0,
    )
    lo = float(np.log10(n_min))
    hi = float(np.log10(n_max))
    cbar.set_ticks([lo, hi])
    cbar.set_ticklabels([str(n_min), str(n_max)])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)


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
    """Vertical colorbar on the right; top=N_max, bottom=N_min to match plot order (max at top)."""
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
        "Number Of Gen Structures",
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


def _distribution_panel_overlay_step(
    ax,
    by_n: dict[int, np.ndarray],
    x_label: str,
    *,
    n_bins: int,
    cbar_ax=None,
) -> None:
    """Original overlay: step histograms (used when cbar_ax is gridspec top)."""
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
                _add_horizontal_n_colorbar_top(fig, cbar_ax, norm=norm, cmap=cmap, n_min=n_min, n_max=n_max)
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
            _add_horizontal_n_colorbar_top(fig, cbar_ax, norm=norm, cmap=cmap, n_min=n_min, n_max=n_max)
        else:
            cbar_ax.set_visible(False)


def _add_horizontal_n_colorbar_top(fig, cbar_ax, *, norm, cmap, n_min: int, n_max: int) -> None:
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label(
        "Number Of Gen Structures",
        fontsize=FONT_SIZE_AXIS,
        labelpad=4,
        family=FONT_FAMILY,
    )
    lo = float(np.log10(n_min))
    hi = float(np.log10(n_max))
    cbar.set_ticks([lo, hi])
    cbar.set_ticklabels([str(n_min), str(n_max)])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)


def _figure_generative_capacity_top_cbar(has_multiple_n: bool) -> tuple:
    w = GEN_CAP_FIGSIZE[0]
    h_full = GEN_CAP_FIGSIZE[1]
    if not has_multiple_n:
        r_cbar, r_main = GEN_CAP_GRIDSPEC_HEIGHT_RATIOS
        h_main = h_full * (r_main / (r_cbar + r_main))
        fig, ax = plt.subplots(1, 1, figsize=(w, h_main))
        fig.subplots_adjust(**GEN_CAP_SUBPLOT_MARGINS)
        return fig, ax, None
    fig = plt.figure(figsize=(w, h_full))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=GEN_CAP_GRIDSPEC_HEIGHT_RATIOS,
        hspace=GEN_CAP_GRIDSPEC_HSPACE,
        **GEN_CAP_SUBPLOT_MARGINS,
    )
    ax_cbar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    return fig, ax, ax_cbar


def _figure_generative_capacity_overlay_bottom_cbar(has_multiple_n: bool) -> tuple:
    """Single main axes; colorbar added later under the panel."""
    w = GEN_CAP_FIGSIZE[0]
    h = GEN_CAP_FIGSIZE[1]
    if not has_multiple_n:
        fig, ax = plt.subplots(1, 1, figsize=(w, h * 0.72))
        fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.22)
        return fig, ax
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.22)
    return fig, ax


def build_nested_subsample_indices(n_max: int, n_values: list[int], seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_max))
    return {int(n): perm[: int(n)] for n in sorted(n_values)}


def synthetic_generative_capacity_by_n(
    rng: np.random.Generator,
    n_values: list[int],
    *,
    kind: str,
) -> dict[int, np.ndarray]:
    n_max = max(n_values)
    nested = build_nested_subsample_indices(n_max, n_values, seed=int(rng.integers(0, 2**31 - 1)))
    out: dict[int, np.ndarray] = {}
    if kind == "rmsd":
        base = rng.lognormal(mean=0.0, sigma=0.5, size=(n_max, n_max)).astype(np.float32)
        np.fill_diagonal(base, np.inf)
        for n in n_values:
            idx = nested[n]
            sub = base[np.ix_(idx, idx)]
            out[n] = np.min(sub, axis=1).astype(np.float32)
    else:
        base = rng.uniform(0.2, 0.95, size=(n_max, n_max)).astype(np.float32)
        np.fill_diagonal(base, -np.inf)
        for n in n_values:
            idx = nested[n]
            sub = base[np.ix_(idx, idx)]
            out[n] = np.max(sub, axis=1).astype(np.float32)
    return out


def _global_xlim_bins(by_n: dict[int, np.ndarray], n_bins: int) -> tuple[float, float, np.ndarray]:
    pieces = [by_n[n].ravel() for n in sorted(by_n.keys()) if by_n[n].size > 0]
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


def _panel_kde(ax, values: np.ndarray, x_min: float, x_max: float, *, color: str, lw: float = 2.0) -> None:
    x, d = _kde_curve(values, x_min, x_max)
    ax.plot(x, d, color=color, lw=lw, zorder=5)


def run_generative_capacity_plots(
    out_root: Path,
    rng: np.random.Generator,
    *,
    save_pdf: bool,
    n_values: list[int] | None = None,
) -> None:
    gc_dir = out_root / "generative_capacity_synthetic"
    variants = gc_dir / "variants"
    gc_dir.mkdir(parents=True, exist_ok=True)
    variants.mkdir(parents=True, exist_ok=True)
    n_values = n_values or [32, 48, 64, 96, 128, 192, 256]

    by_r = synthetic_generative_capacity_by_n(rng, n_values, kind="rmsd")
    by_q = synthetic_generative_capacity_by_n(rng, n_values, kind="q")
    _save_generative_capacity_median_convergence(gc_dir, by_r, by_q, save_pdf=save_pdf)

    def save_overlay_bottom_cbar(by_n: dict, x_label: str, stem: str) -> None:
        fig, ax = _figure_generative_capacity_overlay_bottom_cbar(len(n_values) >= 2)
        _distribution_panel_overlay_step(ax, by_n, x_label, n_bins=HIST_BINS_DEFAULT, cbar_ax=None)
        fig.canvas.draw()
        if len(n_values) >= 2:
            cmap = _get_cmap("viridis")
            n_min, n_max = min(n_values), max(n_values)
            norm = Normalize(vmin=np.log10(n_min), vmax=np.log10(n_max))
            _add_n_colorbar_below_ax(fig, ax, norm=norm, cmap=cmap, n_min=n_min, n_max=n_max)
        p = gc_dir / f"{stem}.png"
        fig.savefig(str(p), dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
        if save_pdf:
            _save_pdf_next_to_png(fig, str(p))
        plt.close(fig)

    def save_overlay_top_cbar_reference(by_n: dict, x_label: str, stem: str) -> None:
        fig, ax, ax_cbar = _figure_generative_capacity_top_cbar(len(n_values) >= 2)
        _distribution_panel_overlay_step(ax, by_n, x_label, n_bins=HIST_BINS_DEFAULT, cbar_ax=ax_cbar)
        p = variants / f"{stem}_overlay_top_cbar.png"
        fig.savefig(str(p), dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
        if save_pdf:
            _save_pdf_next_to_png(fig, str(p))
        plt.close(fig)

    save_overlay_bottom_cbar(
        by_r,
        "Min RMSD To Nearest Gen Structure",
        "generative_capacity_rmsd",
    )
    save_overlay_bottom_cbar(
        by_q,
        "Max Q To Nearest Gen Structure",
        "generative_capacity_q",
    )
    save_overlay_top_cbar_reference(
        by_r,
        "Min RMSD To Nearest Gen Structure",
        "rmsd",
    )
    save_overlay_top_cbar_reference(
        by_q,
        "Max Q To Nearest Gen Structure",
        "q",
    )

    # Stacked panels (one row per N), RMSD/Q analysis style sharex
    def stacked_figure(by_n: dict, metric: str, mode: str) -> None:
        n_vals = sorted(by_n.keys())
        lo, hi, bins = _global_xlim_bins(by_n, HIST_BINS_DEFAULT)
        cmap = _get_cmap("viridis")
        n_min, n_max = min(n_vals), max(n_vals)
        norm = Normalize(vmin=np.log10(n_min), vmax=np.log10(n_max))

        fig, axes = plt.subplots(
            len(n_vals),
            1,
            figsize=(GEN_CAP_STACKED_FIGWIDTH, GEN_CAP_STACKED_ROW_HEIGHT * len(n_vals)),
            sharex=True,
        )
        if len(n_vals) == 1:
            axes = [axes]
        for ax, n in zip(axes, reversed(n_vals)):
            vals = by_n[n].astype(np.float64).ravel()
            color = cmap(norm(np.log10(n)))
            if mode == "kde":
                _panel_kde(ax, vals, lo, hi, color=color)
            elif mode == "filled":
                ax.hist(vals, bins=bins, density=True, alpha=0.75, color=color, edgecolor=HIST_FILLED_EDGE_COLOR)
            elif mode == "step":
                ax.hist(
                    vals,
                    bins=bins,
                    density=True,
                    color=color,
                    histtype="step",
                    lw=LINEWIDTH_HIST_STEP,
                )
            elif mode == "kde_filled":
                ax.hist(vals, bins=bins, density=True, alpha=0.35, color=color, edgecolor=HIST_FILLED_EDGE_COLOR, zorder=1)
                _panel_kde(ax, vals, lo, hi, color=color, lw=2.2)
            else:
                raise ValueError(mode)
            ax.set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
            ax.set_xlim(lo, hi)
            _apply_plain_number_axes(ax)

        x_lab = (
            "Min RMSD To Nearest Gen Structure"
            if metric == "rmsd"
            else "Max Q To Nearest Gen Structure"
        )
        axes[-1].set_xlabel(x_lab, fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
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
        fig.subplots_adjust(left=0.12, right=0.88, top=0.98, bottom=0.12)
        fig.canvas.draw()
        _add_n_colorbar_right_of_stacked(fig, axes, norm=norm, cmap=cmap, n_min=n_min, n_max=n_max)
        p = variants / f"stacked_{mode}_{metric}.png"
        fig.savefig(str(p), dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.15)
        if save_pdf:
            _save_pdf_next_to_png(fig, str(p))
        plt.close(fig)

    for mode in ("kde", "filled", "step", "kde_filled"):
        stacked_figure(by_r, "rmsd", mode)
        stacked_figure(by_q, "q", mode)


def _write_npz_recon_metric(seed_dir: Path, metric: str, values: np.ndarray) -> None:
    sub = "rmsd" if metric == "rmsd" else "q"
    filename = "rmsd_recon_data.npz" if metric == "rmsd" else "q_recon_data.npz"
    key = "test_recon_rmsd" if metric == "rmsd" else "test_recon_q"
    data_dir = (
        seed_dir
        / "distmap"
        / "0"
        / "euclideanizer"
        / "0"
        / "analysis"
        / sub
        / "recon"
        / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / filename, **{key: np.asarray(values, dtype=np.float32)})


def _write_pipeline_config(seed_dir: Path, max_data: int) -> None:
    cfg = {"data": {"max_data": int(max_data)}}
    with open(seed_dir / "pipeline_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _synthetic_gen_to_test_rmsd(rng: np.random.Generator, n: int, *, scale: float, loc_shift: float) -> np.ndarray:
    return rng.lognormal(mean=np.log(scale), sigma=0.35, size=n).astype(np.float32) + float(loc_shift)


def _synthetic_gen_to_test_q(rng: np.random.Generator, n: int, *, lo: float, hi: float) -> np.ndarray:
    return rng.uniform(float(lo), float(hi), size=n).astype(np.float32)


_FAKE_SUFFICIENCY_FINGERPRINT = ".synthetic_sufficiency_fingerprint.yaml"


def _fake_grid_fingerprint_dict(
    seeds: tuple[int, ...],
    splits: tuple[float, ...],
    max_data: tuple[int, ...],
    n_struct: int,
    large_grid: bool,
) -> dict:
    return {
        "seeds": list(seeds),
        "training_splits": [float(s) for s in splits],
        "max_data_values": [int(m) for m in max_data],
        "n_structures_per_run": int(n_struct),
        "large_grid": bool(large_grid),
    }


def _normalize_grid_fp(d: dict) -> tuple:
    return (
        tuple(int(x) for x in d["seeds"]),
        tuple(round(float(x), 6) for x in d["training_splits"]),
        tuple(int(x) for x in d["max_data_values"]),
        int(d["n_structures_per_run"]),
        bool(d["large_grid"]),
    )


def _load_fake_grid_fingerprint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    return raw if isinstance(raw, dict) else None


def _fingerprints_match(path: Path, want: dict) -> bool:
    disk = _load_fake_grid_fingerprint(path)
    if not disk:
        return False
    try:
        return _normalize_grid_fp(disk) == _normalize_grid_fp(want)
    except (KeyError, TypeError, ValueError):
        return False


def _fake_base_has_seed_dirs(fake_base: Path) -> bool:
    if not fake_base.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("seed_") for p in fake_base.iterdir())


def _write_fake_grid_fingerprint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_fake_base_output(
    fake_base: Path,
    rng: np.random.Generator,
    *,
    seeds: tuple[int, ...] | None = None,
    training_splits: tuple[float, ...] | None = None,
    max_data_values: tuple[int, ...] | None = None,
    n_structures: int | None = None,
) -> None:
    fake_base.mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds is not None else SYNTHETIC_SEEDS
    splits = training_splits if training_splits is not None else SYNTHETIC_TRAINING_SPLITS
    max_data = max_data_values if max_data_values is not None else SYNTHETIC_MAX_DATA
    n_struct = int(n_structures if n_structures is not None else SYNTHETIC_N_STRUCTURES_PER_RUN)
    for seed, split, md in itertools.product(seeds, splits, max_data):
        split_tag = f"{split:g}"
        name = f"seed_{seed}_split_{split_tag}_maxdata_{md}"
        sdir = fake_base / name
        sdir.mkdir(parents=True, exist_ok=True)
        _write_pipeline_config(sdir, md)
        r_scale = 2.6 - 0.00022 * float(md)
        r_shift = 0.06 * (0.88 - split)
        q_lo = 0.36 + 0.00005 * float(md) + 0.05 * (split - 0.5)
        q_hi = 0.58 + 0.00007 * float(md) + 0.12 * (1.0 - split)
        q_lo = float(np.clip(q_lo, 0.25, 0.72))
        q_hi = float(np.clip(q_hi, q_lo + 0.12, 0.96))
        r = _synthetic_gen_to_test_rmsd(rng, n_struct, scale=max(0.35, r_scale), loc_shift=r_shift)
        q = _synthetic_gen_to_test_q(rng, n_struct, lo=q_lo, hi=q_hi)
        _write_npz_recon_metric(sdir, "rmsd", r)
        _write_npz_recon_metric(sdir, "q", q)


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic sufficiency meta-analysis + generative capacity plots (sandbox).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output root (default: synthetic_plot_sandbox/outputs)",
    )
    ap.add_argument("--pdf", action="store_true", help="Write PDF copies alongside PNGs where applicable.")
    ap.add_argument("--no-clean", action="store_true", help="Do not remove out-dir before generating.")
    ap.add_argument("--rng-seed", type=int, default=0, help="NumPy random seed for reproducible synthetic data.")
    ap.add_argument(
        "--large-grid",
        action="store_true",
        help=(
            "Many training splits × many max_data (stress-test layouts). "
            f"Uses {LARGE_GRID_N_SPLITS} splits, {LARGE_GRID_N_MAX_DATA} max_data values, "
            f"{len(LARGE_GRID_SEEDS)} seeds; larger gen-cap N ladder."
        ),
    )
    ap.add_argument(
        "--reuse-fake-base",
        action="store_true",
        help=(
            "Skip regenerating fake_base NPZ trees when "
            f"{_FAKE_SUFFICIENCY_FINGERPRINT} matches the requested grid and seed dirs exist. "
            "Use with --no-clean so the output directory (and fake_base) are not deleted first."
        ),
    )
    args = ap.parse_args()
    out_dir: Path = args.out_dir.resolve()
    if not args.no_clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.large_grid:
        grid_seeds = LARGE_GRID_SEEDS
        grid_splits = _large_grid_training_splits()
        grid_max_data = _large_grid_max_data()
        n_struct = LARGE_GRID_N_STRUCTURES
        gc_n = list(LARGE_GRID_GEN_CAP_N)
    else:
        grid_seeds = SYNTHETIC_SEEDS
        grid_splits = SYNTHETIC_TRAINING_SPLITS
        grid_max_data = SYNTHETIC_MAX_DATA
        n_struct = SYNTHETIC_N_STRUCTURES_PER_RUN
        gc_n = None

    rng = np.random.default_rng(args.rng_seed)
    fake_base = out_dir / "fake_base"
    fp_path = fake_base / _FAKE_SUFFICIENCY_FINGERPRINT
    want_fp = _fake_grid_fingerprint_dict(
        tuple(grid_seeds),
        tuple(grid_splits),
        tuple(grid_max_data),
        int(n_struct),
        bool(args.large_grid),
    )
    reuse_ok = (
        bool(args.reuse_fake_base)
        and _fingerprints_match(fp_path, want_fp)
        and _fake_base_has_seed_dirs(fake_base)
    )
    if reuse_ok:
        print("[sandbox] Reusing existing fake_base (fingerprint matches).")
    else:
        if args.reuse_fake_base and fake_base.exists():
            print("[sandbox] Regenerating fake_base (missing or mismatched fingerprint).")
            shutil.rmtree(fake_base)
        fake_base.mkdir(parents=True, exist_ok=True)
        build_fake_base_output(
            fake_base,
            rng,
            seeds=grid_seeds,
            training_splits=grid_splits,
            max_data_values=grid_max_data,
            n_structures=n_struct,
        )
        _write_fake_grid_fingerprint(fp_path, want_fp)

    ok = run_sufficiency_plots_sandbox(
        base_output_dir=str(fake_base),
        max_data_values=list(grid_max_data),
        save_pdf_copy=bool(args.pdf),
        log=lambda m: print(f"[sufficiency] {m}"),
    )
    print(f"Sufficiency sandbox produced outputs: {ok}")

    run_generative_capacity_plots(out_dir, rng, save_pdf=bool(args.pdf), n_values=gc_n)
    print(f"Done. Outputs under: {out_dir}")


if __name__ == "__main__":
    main()