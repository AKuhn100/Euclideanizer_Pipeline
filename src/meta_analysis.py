from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from .plot_config import (
    FONT_FAMILY,
    FONT_SIZE_AXIS,
    FONT_SIZE_SMALL,
    FONT_SIZE_TITLE,
    FONT_SIZE_TICK,
    HIST_BINS_DEFAULT,
    META_CBAR_HEIGHT_FRAC,
    META_CBAR_WIDTH_FRAC,
    META_DIST_CB_GAP,
    META_HEATMAP_CB_GAP,
    PLOT_DPI,
    SUFFICIENCY_DIST_FIG_WIDTH,
    SUFFICIENCY_CURVES_FIG_HEIGHT,
    SUFFICIENCY_DIST_FIGH_EXTRA,
    SUFFICIENCY_DIST_ROW_HEIGHT,
    sufficiency_dist_bottom_frac,
    sufficiency_dist_cbar_gap_frac,
    SUFFICIENCY_HEATMAP_CELL_IN,
    SUFFICIENCY_HEATMAP_FIGH_EXTRA,
    sufficiency_heatmap_bottom_frac,
    sufficiency_heatmap_cbar_gap_frac,
    sufficiency_heatmap_wspace,
    sufficiency_heatmap_ytick_fontsize,
)


@dataclass
class SufficiencyPoint:
    seed: int
    max_data: int | None
    training_split: float
    rmsd_values: np.ndarray
    q_values: np.ndarray


def _get_cmap(name: str = "viridis"):
    reg = getattr(matplotlib, "colormaps", None)
    if reg is not None:
        return reg[name]
    return cm.get_cmap(name)


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
    """List (relative_subdir, npz_path) under analysis/{rmsd|q}/recon/.

    ``relative_subdir`` is ``""`` for ``recon/data/<file>.npz``, or a subdirectory
    name such as ``train10_test10`` when outputs live under ``recon/<subdir>/data/``.
    """
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
    """Load test-set reconstruction RMSD and Q (per test structure) from recon analysis NPZ.

    Uses the first Euclideanizer run where RMSD and Q recon share the same recon
    subpath (default ``recon/data/`` or matching ``recon/train*_test*/data/``).
    """
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


def _max_data_from_seed_dir(seed_dir: str, fallback_max_data: int | None) -> int | None:
    cfg_path = os.path.join(seed_dir, "pipeline_config.yaml")
    if not os.path.isfile(cfg_path):
        return fallback_max_data
    try:
        import yaml

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


def _save_pdf_if_enabled(fig, png_path: str, enabled: bool) -> None:
    if not enabled:
        return
    base, _ = os.path.splitext(png_path)
    fig.savefig(base + ".pdf")


def _apply_plain_number_axes(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def _horizontal_cbar_axes_in_span(
    fig,
    x_left: float,
    x_right: float,
    y_bottom: float,
    height_frac: float,
    *,
    width_frac: float = META_CBAR_WIDTH_FRAC,
):
    span = max(x_right - x_left, 1e-6)
    w = span * width_frac
    x0 = x_left + 0.5 * (span - w)
    return fig.add_axes([x0, y_bottom, w, height_frac])


def _style_horizontal_meta_colorbar(cbar, label: str, *, labelpad: int = 3) -> None:
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
    height_frac: float,
    gap_frac: float | None = None,
) -> None:
    fig.canvas.draw()
    p0, p1 = ax_left.get_position(), ax_right.get_position()
    row_bottom = min(p0.y0, p1.y0)
    gap = META_DIST_CB_GAP if gap_frac is None else gap_frac
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
    gap = META_HEATMAP_CB_GAP if gap_frac is None else gap_frac
    y = max(0.02, row_bottom - gap - META_CBAR_HEIGHT_FRAC)
    cax = _horizontal_cbar_axes_in_span(fig, p0.x0, p1.x1, y, META_CBAR_HEIGHT_FRAC)
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    _style_horizontal_meta_colorbar(cbar, label, labelpad=4)


def _sufficiency_heatmap_small_grid_steps(n_r: int, n_c: int) -> int:
    """Extra layout slack when the grid has few rows or columns (aspect='equal' leaves little margin)."""
    return max(0, 4 - min(n_r, n_c))


def _sufficiency_dist_small_split_steps(n_split_rows: int) -> int:
    """Extra layout slack when few training-split rows are stacked (shared colorbar below)."""
    return max(0, 4 - n_split_rows)


def _max_data_colorbar_tick_label(md: int | None) -> str:
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
    y = max(0.02, row_bottom - gap_frac - META_CBAR_HEIGHT_FRAC)
    cax = _horizontal_cbar_axes_in_span(fig, p0.x0, p1.x1, y, META_CBAR_HEIGHT_FRAC)
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
    """Median test recon RMSD / Q vs training split; one curve per ``max_data``."""
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


def _norm01_per_grid(grid: np.ndarray) -> np.ma.MaskedArray:
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


def run_sufficiency_meta_analysis(
    *,
    base_output_dir: str,
    max_data_values: list[int | None],
    save_pdf_copy: bool,
    log: Callable[[str], None] | None = None,
) -> bool:
    """Build sufficiency distribution and heatmap figures under meta_analysis/sufficiency.

    Returns True when at least one figure is created.
    """
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
            if log:
                log(f"Sufficiency meta-analysis: missing RMSD/Q recon NPZ for {seed_name}; skipping.")
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

    if not points_by_seed:
        if log:
            log("Sufficiency meta-analysis: no recon analysis NPZ data found. Skipping.")
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
            cbar_hfrac = META_CBAR_HEIGHT_FRAC * 6.2 / fig_h
            _training_split_colorbar_below_panels(
                fig,
                axes[-1, 0],
                axes[-1, 1],
                cmap=cmap,
                height_frac=cbar_hfrac,
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
