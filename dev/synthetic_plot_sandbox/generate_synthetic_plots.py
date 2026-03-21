#!/usr/bin/env python3
"""
Synthetic sufficiency meta-analysis + generative capacity figures.

All plotting logic lives here (sandbox only); reads the same on-disk layout the
real pipeline uses for sufficiency NPZ discovery.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
# Plot style (aligned with Pipeline/src/plot_config.py)
# ---------------------------------------------------------------------------
PLOT_DPI = 150
HIST_BINS_DEFAULT = 50
LINEWIDTH_HIST_STEP = 2.4
HIST_FILLED_EDGE_COLOR = "none"
GEN_CAP_FIGSIZE = (10.5, 4.6)
GEN_CAP_GRIDSPEC_HEIGHT_RATIOS = (1, 6)
GEN_CAP_GRIDSPEC_HSPACE = 0.22
GEN_CAP_SUBPLOT_MARGINS = dict(left=0.11, right=0.96, top=0.86, bottom=0.14)
FONT_FAMILY = "sans-serif"
FONT_SIZE_AXIS = 10
FONT_SIZE_TITLE = 11
FONT_SIZE_TICK = 9

# Horizontal colorbars (figure fraction coordinates, 0–1)
# Meta-analysis: ~3/4 width of the panel row / single axes, centered in that span.
CBAR_META_WIDTH_FRAC = 0.75
CBAR_META_HEIGHT_FRAC = 0.028
# Generative capacity: strip under overlay panel
GEN_CBAR_HEIGHT_FRAC = 0.014
GEN_CBAR_GAP_BELOW_AXES = 0.03
# Gap below sufficiency *distribution* panels before colorbar (smaller → colorbar sits higher).
META_DIST_CBAR_GAP_BELOW_AXIS = 0.075
# Distance from bottom of heatmap axes to top of colorbar (larger → colorbar sits lower).
HEATMAP_CBAR_BELOW_AXES_FRAC = 0.21

# Synthetic sufficiency layout: many splits × many max_data → asymmetric heatmaps;
# multiple seeds → separate meta-analysis trees. Tune for "realistic" density.
SYNTHETIC_SEEDS = (1, 2)
# 5 rows × 6 columns → non-square heatmap (asymmetric grid).
SYNTHETIC_TRAINING_SPLITS = (0.5, 0.58, 0.66, 0.74, 0.82)
SYNTHETIC_MAX_DATA = (400, 800, 1600, 3200, 5000, 10000)
SYNTHETIC_N_STRUCTURES_PER_RUN = 480


# =============================================================================
# Sufficiency: on-disk scan + KDE (copied from Pipeline/src/meta_analysis.py)
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


def _run_name_has_var_one(run_name: str) -> bool:
    if "_var" in run_name:
        suffix = run_name.split("_var", 1)[-1].strip()
        try:
            return abs(float(suffix) - 1.0) < 1e-9
        except ValueError:
            return False
    if run_name.startswith("var"):
        try:
            return abs(float(run_name[3:].strip()) - 1.0) < 1e-9
        except ValueError:
            return False
    return False


def _load_gen_metric_array(seed_dir: str, metric: str) -> np.ndarray | None:
    filename = "rmsd_data.npz" if metric == "rmsd" else "q_data.npz"
    subdir = "rmsd" if metric == "rmsd" else "q"
    for run_dir in _iter_euclideanizer_runs(seed_dir) or []:
        gen_dir = os.path.join(run_dir, "analysis", subdir, "gen")
        if not os.path.isdir(gen_dir):
            continue
        for run_name in sorted(os.listdir(gen_dir)):
            data_path = os.path.join(gen_dir, run_name, "data", filename)
            if not os.path.isfile(data_path):
                continue
            if not _run_name_has_var_one(run_name):
                continue
            try:
                loaded = np.load(data_path, allow_pickle=False)
                vals = np.asarray(loaded["gen_to_test"], dtype=np.float32)
                loaded.close()
            except Exception:
                continue
            return vals
    return None


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
        rmsd_vals = _load_gen_metric_array(seed_dir, "rmsd")
        q_vals = _load_gen_metric_array(seed_dir, "q")
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


def _training_split_colorbar_below_panels(fig, ax_left, ax_right, *, cmap, height_frac: float = CBAR_META_HEIGHT_FRAC) -> None:
    """75%% of panel span, centered; label above the bar."""
    fig.canvas.draw()
    p0, p1 = ax_left.get_position(), ax_right.get_position()
    row_bottom = min(p0.y0, p1.y0)
    y = max(0.02, row_bottom - META_DIST_CBAR_GAP_BELOW_AXIS - height_frac)
    cax = _horizontal_cbar_axes_in_span(fig, p0.x0, p1.x1, y, height_frac)
    sm = cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["0%", "100%"])
    _style_horizontal_meta_colorbar(cbar, "Training Split", labelpad=3)


def _heatmap_colorbar_below_panels(fig, ax_left, ax_right, *, cmap, vmin: float, vmax: float, label: str) -> None:
    fig.canvas.draw()
    p0, p1 = ax_left.get_position(), ax_right.get_position()
    row_bottom = min(p0.y0, p1.y0)
    y = max(0.02, row_bottom - HEATMAP_CBAR_BELOW_AXES_FRAC - CBAR_META_HEIGHT_FRAC)
    cax = _horizontal_cbar_axes_in_span(fig, p0.x0, p1.x1, y, CBAR_META_HEIGHT_FRAC)
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    _style_horizontal_meta_colorbar(cbar, label, labelpad=4)


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


def run_sufficiency_plots_sandbox(
    *,
    base_output_dir: str,
    max_data_values: list[int | None],
    save_pdf_copy: bool,
    log: Callable[[str], None] | None = None,
) -> bool:
    """Sufficiency distributions + heatmaps with sandbox layout (not Pipeline meta_analysis)."""
    points_by_seed = collect_sufficiency_points(base_output_dir, max_data_values)
    if not points_by_seed:
        if log:
            log("Sufficiency sandbox: no analysis NPZ data found. Skipping.")
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

            all_r = np.concatenate([p.rmsd_values for p in subset], axis=0)
            all_q = np.concatenate([p.q_values for p in subset], axis=0)
            xr_min = float(np.percentile(all_r, 1))
            xr_max = float(np.percentile(all_r, 99))
            xq_min = float(np.percentile(all_q, 1))
            xq_max = float(np.percentile(all_q, 99))
            xq_min, xq_max = max(0.0, xq_min), min(1.0, xq_max)
            bins_r = np.linspace(xr_min, xr_max, HIST_BINS_DEFAULT + 1)
            bins_q = np.linspace(xq_min, xq_max, HIST_BINS_DEFAULT + 1)

            row_h = 3.05
            fig_h = max(4.0, row_h * n_splits_local)
            fig, axes = plt.subplots(
                n_splits_local, 2,
                figsize=(14, fig_h),
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

            axes[0, 0].set_title(f"Min RMSD | Max Data={md}", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
            axes[0, 1].set_title(f"Max Q | Max Data={md}", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
            axes[-1, 0].set_xlabel("Min RMSD (Å)", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
            axes[-1, 1].set_xlabel("Max Q", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)

            fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.22)
            _cbar_hfrac = CBAR_META_HEIGHT_FRAC * 6.2 / fig_h
            _training_split_colorbar_below_panels(fig, axes[-1, 0], axes[-1, 1], cmap=cmap, height_frac=_cbar_hfrac)

            md_tag = "all" if md is None else str(md)
            out_dir = os.path.join(dist_root, f"max_data_{md_tag}")
            os.makedirs(out_dir, exist_ok=True)
            png_path = os.path.join(out_dir, "distributions_rmsd_q.png")
            fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.15)
            _save_pdf_if_enabled(fig, png_path, save_pdf_copy)
            plt.close(fig)
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
        cell_in = 0.82
        w_per = max(3.2, cell_in * n_c + 0.55)
        fig_h = max(4.0, cell_in * n_r + 1.35)
        gap = 0.15
        fig_w = 2 * w_per + gap + 0.6
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw=dict(wspace=0.08))
        ax_r, ax_q = axes
        ax_r.imshow(r_plot, aspect="equal", cmap="viridis", vmin=0.0, vmax=1.0)
        ax_q.imshow(q_plot, aspect="equal", cmap="viridis", vmin=0.0, vmax=1.0)
        for ax, title in ((ax_r, "Median Min RMSD"), (ax_q, "Median Max Q")):
            ax.set_title(title, fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
            ax.set_xlabel("Max Structures", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY, labelpad=10)
            ax.set_xticks(np.arange(len(md_vals)))
            ax.set_xticklabels([str(v) for v in md_vals], rotation=35, ha="right", fontsize=FONT_SIZE_TICK)
            ax.set_yticks(np.arange(len(split_set)))
            ax.set_yticklabels([f"{int(round(s * 100))}%" for s in split_set], fontsize=FONT_SIZE_TICK)
            ax.set_ylabel("Training Split", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)

        fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.24)
        _heatmap_colorbar_below_panels(
            fig,
            ax_r,
            ax_q,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            label="Normalized Median",
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
        "Number Of Generated Structures",
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


def run_generative_capacity_plots(out_root: Path, rng: np.random.Generator, *, save_pdf: bool) -> None:
    gc_dir = out_root / "generative_capacity_synthetic"
    variants = gc_dir / "variants"
    gc_dir.mkdir(parents=True, exist_ok=True)
    variants.mkdir(parents=True, exist_ok=True)
    n_values = [32, 48, 64, 96, 128, 192, 256]

    by_r = synthetic_generative_capacity_by_n(rng, n_values, kind="rmsd")
    by_q = synthetic_generative_capacity_by_n(rng, n_values, kind="q")

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
        "Min RMSD To Nearest Generated Structure (Å)",
        "generative_capacity_rmsd",
    )
    save_overlay_bottom_cbar(
        by_q,
        "Max Q To Nearest Generated Structure",
        "generative_capacity_q",
    )
    save_overlay_top_cbar_reference(
        by_r,
        "Min RMSD To Nearest Generated Structure (Å)",
        "rmsd",
    )
    save_overlay_top_cbar_reference(
        by_q,
        "Max Q To Nearest Generated Structure",
        "q",
    )

    # Stacked panels (one row per N), RMSD/Q analysis style sharex
    def stacked_figure(by_n: dict, metric: str, mode: str) -> None:
        n_vals = sorted(by_n.keys())
        lo, hi, bins = _global_xlim_bins(by_n, HIST_BINS_DEFAULT)
        cmap = _get_cmap("viridis")
        n_min, n_max = min(n_vals), max(n_vals)
        norm = Normalize(vmin=np.log10(n_min), vmax=np.log10(n_max))

        fig, axes = plt.subplots(len(n_vals), 1, figsize=(11, 3.05 * len(n_vals)), sharex=True)
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
            "Min RMSD To Nearest Generated Structure (Å)"
            if metric == "rmsd"
            else "Max Q To Nearest Generated Structure"
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


def _write_npz_gen_metric(seed_dir: Path, metric: str, gen_to_test: np.ndarray) -> None:
    sub = "rmsd" if metric == "rmsd" else "q"
    filename = "rmsd_data.npz" if metric == "rmsd" else "q_data.npz"
    data_dir = (
        seed_dir
        / "distmap"
        / "0"
        / "euclideanizer"
        / "0"
        / "analysis"
        / sub
        / "gen"
        / "synthetic_var1"
        / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / filename, gen_to_test=np.asarray(gen_to_test, dtype=np.float32))


def _write_pipeline_config(seed_dir: Path, max_data: int) -> None:
    cfg = {"data": {"max_data": int(max_data)}}
    with open(seed_dir / "pipeline_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _synthetic_gen_to_test_rmsd(rng: np.random.Generator, n: int, *, scale: float, loc_shift: float) -> np.ndarray:
    return rng.lognormal(mean=np.log(scale), sigma=0.35, size=n).astype(np.float32) + float(loc_shift)


def _synthetic_gen_to_test_q(rng: np.random.Generator, n: int, *, lo: float, hi: float) -> np.ndarray:
    return rng.uniform(float(lo), float(hi), size=n).astype(np.float32)


def build_fake_base_output(fake_base: Path, rng: np.random.Generator) -> None:
    fake_base.mkdir(parents=True, exist_ok=True)
    n_struct = SYNTHETIC_N_STRUCTURES_PER_RUN
    for seed, split, md in itertools.product(SYNTHETIC_SEEDS, SYNTHETIC_TRAINING_SPLITS, SYNTHETIC_MAX_DATA):
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
        _write_npz_gen_metric(sdir, "rmsd", r)
        _write_npz_gen_metric(sdir, "q", q)


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
    args = ap.parse_args()
    out_dir: Path = args.out_dir.resolve()
    if not args.no_clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.rng_seed)
    fake_base = out_dir / "fake_base"
    build_fake_base_output(fake_base, rng)

    ok = run_sufficiency_plots_sandbox(
        base_output_dir=str(fake_base),
        max_data_values=list(SYNTHETIC_MAX_DATA),
        save_pdf_copy=bool(args.pdf),
        log=lambda m: print(f"[sufficiency] {m}"),
    )
    print(f"Sufficiency sandbox produced outputs: {ok}")

    run_generative_capacity_plots(out_dir, rng, save_pdf=bool(args.pdf))
    print(f"Done. Outputs under: {out_dir}")


if __name__ == "__main__":
    main()