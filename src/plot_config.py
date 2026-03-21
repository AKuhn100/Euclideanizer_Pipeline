"""
Shared plot configuration for the pipeline: colors, fonts, and font sizes.
Use these constants so all plots have consistent styling.

Conceptual groups (six): Experimental, Train, Test, Gen, Train Recon, Test Recon.
For test-to-train comparison and experimental reference we use COLOR_EXP so Train and Test
are shown as "experimental structures" (one color) in RMSD/Q panels and clustering.
"""
# ---------------------------------------------------------------------------
# Colors (canonical palette; clustering SOURCE_COLORS align with these)
# ---------------------------------------------------------------------------
COLOR_TRAIN = "#00A8CC"      # blue (used where train is distinct from "experimental" ref)
COLOR_GEN = "#D42248"        # red (generated structures)
COLOR_TEST = "#00B84E"       # orange (used where test is distinct from "experimental" ref)
COLOR_TRAIN_RECON = "#80DDED"   # darker green
COLOR_TEST_RECON = "#80DDA8"    # lighter orange
# Experimental: train/test reference and test-to-train comparison (RMSD, Q, clustering)
COLOR_EXP = "#0C3E96"

# Neutral grays (dendrograms, expected-value bars, fallback source, text)
COLOR_GRAY_LIGHT = "#aaaaaa"
COLOR_GRAY_MID = "#888888"
COLOR_GRAY_DARK = "#555555"
COLOR_GRAY_TEXT = "#333333"

# Ordered list for three-panel figures: panel 0, 1, 2
# Gen: Test→Train (experimental), Gen→Train (gen), Gen→Test (gen — same as Gen→Train)
# Recon: Test→Train (experimental), Train Recon, Test Recon
# Train/Test reference and test-to-train comparison use COLOR_EXP (experimental structures).
GEN_PANEL_COLORS = (COLOR_EXP, COLOR_GEN, COLOR_GEN)
RECON_PANEL_COLORS = (COLOR_EXP, COLOR_TRAIN_RECON, COLOR_TEST_RECON)

# Colormaps for distance maps (training video uses its own)
CMAP_DM = "viridis"
CMAP_DM_R = "viridis_r"  # reversed (e.g. avg maps: high = warm)

# DPI for saved figure files (PNG, etc.); single definition for all pipeline plots
PLOT_DPI = 150

# Filled bar histograms: no outlines (matches recon_statistics; RMSD/Q/analysis use this)
HIST_FILLED_EDGE_COLOR = "none"

# Default bin count for 1D density histograms (recon_statistics, gen analysis, lag grids, RMSD/Q analysis, generative_capacity, …)
HIST_BINS_DEFAULT = 50

# Step-outline histograms (train/test/gen overlays; exp outline on recon_statistics; generative capacity overlays)
LINEWIDTH_HIST_STEP = 2.4
# Log-log scaling P(s) and comparable curve overlays (train/test/gen; recon vs exp)
LINEWIDTH_SCALING_LOGLOG = 2.4

# Generative capacity (stacked filled): one row per n_structures; vertical colorbar on the right.
GEN_CAP_STACKED_FIGWIDTH = 11.0
GEN_CAP_STACKED_ROW_HEIGHT = 3.05
# Two-panel linear plot: median vs N (RMSD and Q); both curves use COLOR_GEN.
GEN_CAP_CONVERGENCE_FIGSIZE = (12.0, 4.8)
GEN_CAP_CONVERGENCE_LINEWIDTH = 2.2
GEN_CAP_CONVERGENCE_MARKER_SIZE = 6
GEN_CAP_CONVERGENCE_MARKER_EDGELINEWIDTH = 1.2
GEN_CAP_CONVERGENCE_MARKER_FACE_COLOR = "white"

# Legacy overlay layout (tests / reference only; main pipeline uses stacked filled).
GEN_CAP_FIGSIZE = (8.0, 3.7)
GEN_CAP_GRIDSPEC_HEIGHT_RATIOS = (1, 6)
GEN_CAP_GRIDSPEC_HSPACE = 0.22
GEN_CAP_SUBPLOT_MARGINS = dict(left=0.11, right=0.96, top=0.86, bottom=0.14)

# Sufficiency meta-analysis: horizontal colorbars (width = fraction of combined axes span).
META_CBAR_WIDTH_FRAC = 0.75
META_CBAR_HEIGHT_FRAC = 0.028
META_DIST_CB_GAP = 0.075
META_HEATMAP_CB_GAP = 0.21
SUFFICIENCY_DIST_FIG_WIDTH = 14.0
SUFFICIENCY_DIST_ROW_HEIGHT = 3.05
# Median recon vs training_split curves (two panels); width matches distribution stack.
SUFFICIENCY_CURVES_FIG_HEIGHT = 5.2
# Distribution stack: few rows (training splits) → extra slack via small-step terms below.
# Tall stacks use inch-based bottom + cbar gap (``sufficiency_dist_*_frac``) so fixed
# figure fractions do not reserve huge bands (e.g. 22% of a 40" figure).
SUFFICIENCY_DIST_BOTTOM_BASE = 0.22  # legacy reference
SUFFICIENCY_DIST_BOTTOM_EXTRA = 0.042  # per max(0, 4 - n_split_rows)
SUFFICIENCY_DIST_BOTTOM_CAP = 0.40
SUFFICIENCY_DIST_CB_GAP_EXTRA = 0.030  # added to META_DIST_CB_GAP with same rule
SUFFICIENCY_DIST_FIGH_EXTRA = 0.22  # inches per step below 4 split rows
SUFFICIENCY_DIST_BOTTOM_RESERVE_BASE_IN = 0.92
SUFFICIENCY_DIST_BOTTOM_RESERVE_PER_SMALL_IN = 0.24
SUFFICIENCY_DIST_BOTTOM_FRAC_FLOOR = 0.045
SUFFICIENCY_DIST_CB_GAP_BASE_IN = 0.38
SUFFICIENCY_DIST_CB_GAP_PER_SMALL_IN = 0.055
SUFFICIENCY_DIST_CB_GAP_FRAC_FLOOR = 0.028
SUFFICIENCY_HEATMAP_CELL_IN = 0.82
# Heatmap: few rows/cols → less "letterboxing" from aspect="equal", so reserve more
# figure fraction below axes for rotated x-tick labels + horizontal colorbar.
# Large (tall) figures: fixed figure-fraction bottom/gap wastes space; use inch targets
# via sufficiency_heatmap_*_frac() helpers (still extra slack when min(n_r,n_c) is small).
SUFFICIENCY_HEATMAP_BOTTOM_BASE = 0.24  # legacy reference; heatmaps use inch-based helper
SUFFICIENCY_HEATMAP_BOTTOM_EXTRA = 0.048  # per (4 - min(n_r, n_c)), clamped at 0
SUFFICIENCY_HEATMAP_BOTTOM_CAP = 0.42
SUFFICIENCY_HEATMAP_CB_GAP_EXTRA = 0.036  # added to META_HEATMAP_CB_GAP with same small-grid rule
SUFFICIENCY_HEATMAP_FIGH_EXTRA = 0.28  # inches added per small-grid step (room for margins)
# Inch targets for heatmap bottom margin (rotated x-labels + horizontal colorbar).
SUFFICIENCY_HEATMAP_BOTTOM_RESERVE_BASE_IN = 1.2
SUFFICIENCY_HEATMAP_BOTTOM_RESERVE_PER_SMALL_IN = 0.26
SUFFICIENCY_HEATMAP_BOTTOM_FRAC_FLOOR = 0.10
# Inch targets for vertical gap (figure coords) between heatmap axes and colorbar.
SUFFICIENCY_HEATMAP_CB_GAP_BASE_IN = 0.72
SUFFICIENCY_HEATMAP_CB_GAP_PER_SMALL_IN = 0.065
SUFFICIENCY_HEATMAP_CB_GAP_FRAC_FLOOR = 0.058
# When to lean on inch-based margins vs legacy figure fractions: use the **max** of a
# height-based and a row-count-based weight so moderate stacks (e.g. 5× tall rows) keep
# legacy proportions, while huge inch height **or** many rows switches to inch targets.
SUFFICIENCY_LAYOUT_INCH_BLEND_H_START_IN = 16.0
SUFFICIENCY_LAYOUT_INCH_BLEND_H_SPAN_IN = 26.0
SUFFICIENCY_LAYOUT_INCH_BLEND_ROWS_START = 8
SUFFICIENCY_LAYOUT_INCH_BLEND_ROWS_SPAN = 8.0

# ---------------------------------------------------------------------------
# Font and font sizes (use in every plotting call for consistency)
# ---------------------------------------------------------------------------
FONT_FAMILY = "sans-serif"

# Figure-level title (suptitle)
FONT_SIZE_SUPTITLE = 14
# Panel/subplot title (set_title)
FONT_SIZE_TITLE = 11
# Axis labels (set_xlabel, set_ylabel)
FONT_SIZE_AXIS = 10
# Tick labels
FONT_SIZE_TICK = 9
# Legend
FONT_SIZE_LEGEND = 9
# Secondary text, dense-panel legends
FONT_SIZE_SMALL = 8
# Very dense grids (e.g. bond_length_by_genomic_distance legend)
FONT_SIZE_TINY = 6


def _sufficiency_layout_inch_weight(fig_height_in: float, n_stack_rows: int) -> float:
    """Blend weight toward inch-based vertical margins (0 = legacy fractions, 1 = inch targets)."""
    h = float(fig_height_in)
    n = max(int(n_stack_rows), 1)
    lo_h = SUFFICIENCY_LAYOUT_INCH_BLEND_H_START_IN
    span_h = SUFFICIENCY_LAYOUT_INCH_BLEND_H_SPAN_IN
    w_h = 0.0 if span_h <= 0 else (h - lo_h) / span_h
    lo_r = float(SUFFICIENCY_LAYOUT_INCH_BLEND_ROWS_START)
    span_r = SUFFICIENCY_LAYOUT_INCH_BLEND_ROWS_SPAN
    w_r = 0.0 if span_r <= 0 else (float(n) - lo_r) / span_r
    return float(min(1.0, max(0.0, max(w_h, w_r))))


def sufficiency_dist_bottom_frac(
    fig_height_in: float, small_split_steps: int, n_split_rows: int
) -> float:
    """Figure fraction below stacked distribution axes (x-labels + training-split colorbar).

    Blends the legacy fraction rule (good for short figures) with inch-based reserves
    (good for tall stacks). ``small_split_steps`` is ``max(0, 4 - n_split_rows)``.
    """
    h = max(float(fig_height_in), 1e-6)
    s = int(small_split_steps)
    legacy = min(
        SUFFICIENCY_DIST_BOTTOM_CAP,
        SUFFICIENCY_DIST_BOTTOM_BASE + s * SUFFICIENCY_DIST_BOTTOM_EXTRA,
    )
    reserve_in = SUFFICIENCY_DIST_BOTTOM_RESERVE_BASE_IN + s * SUFFICIENCY_DIST_BOTTOM_RESERVE_PER_SMALL_IN
    inch_frac = max(SUFFICIENCY_DIST_BOTTOM_FRAC_FLOOR, reserve_in / h)
    w = _sufficiency_layout_inch_weight(h, n_split_rows)
    blended = (1.0 - w) * legacy + w * inch_frac
    return min(SUFFICIENCY_DIST_BOTTOM_CAP, max(SUFFICIENCY_DIST_BOTTOM_FRAC_FLOOR, blended))


def sufficiency_dist_cbar_gap_frac(
    fig_height_in: float, small_split_steps: int, n_split_rows: int
) -> float:
    """Vertical gap (figure fraction) between bottom row of distribution axes and colorbar."""
    h = max(float(fig_height_in), 1e-6)
    s = int(small_split_steps)
    legacy = META_DIST_CB_GAP + s * SUFFICIENCY_DIST_CB_GAP_EXTRA
    gap_in = SUFFICIENCY_DIST_CB_GAP_BASE_IN + s * SUFFICIENCY_DIST_CB_GAP_PER_SMALL_IN
    inch_frac = max(SUFFICIENCY_DIST_CB_GAP_FRAC_FLOOR, gap_in / h)
    w = _sufficiency_layout_inch_weight(h, n_split_rows)
    blended = (1.0 - w) * legacy + w * inch_frac
    return min(0.38, max(SUFFICIENCY_DIST_CB_GAP_FRAC_FLOOR, blended))


def sufficiency_heatmap_bottom_frac(
    fig_height_in: float, small_grid_steps: int, n_heatmap_rows: int
) -> float:
    """Figure fraction reserved below heatmap axes (x-labels + shared colorbar).

    Blends legacy fractions (short heatmaps) with inch targets (tall heatmaps).
    ``small_grid_steps`` is ``max(0, 4 - min(n_r, n_c))``.
    """
    h = max(float(fig_height_in), 1e-6)
    s = int(small_grid_steps)
    legacy = min(
        SUFFICIENCY_HEATMAP_BOTTOM_CAP,
        SUFFICIENCY_HEATMAP_BOTTOM_BASE + s * SUFFICIENCY_HEATMAP_BOTTOM_EXTRA,
    )
    reserve_in = (
        SUFFICIENCY_HEATMAP_BOTTOM_RESERVE_BASE_IN + s * SUFFICIENCY_HEATMAP_BOTTOM_RESERVE_PER_SMALL_IN
    )
    inch_frac = max(SUFFICIENCY_HEATMAP_BOTTOM_FRAC_FLOOR, reserve_in / h)
    w = _sufficiency_layout_inch_weight(h, n_heatmap_rows)
    blended = (1.0 - w) * legacy + w * inch_frac
    return min(SUFFICIENCY_HEATMAP_BOTTOM_CAP, max(SUFFICIENCY_HEATMAP_BOTTOM_FRAC_FLOOR, blended))


def sufficiency_heatmap_cbar_gap_frac(
    fig_height_in: float, small_grid_steps: int, n_heatmap_rows: int
) -> float:
    """Vertical gap (figure fraction) between heatmap row and horizontal colorbar."""
    h = max(float(fig_height_in), 1e-6)
    s = int(small_grid_steps)
    legacy = META_HEATMAP_CB_GAP + s * SUFFICIENCY_HEATMAP_CB_GAP_EXTRA
    gap_in = SUFFICIENCY_HEATMAP_CB_GAP_BASE_IN + s * SUFFICIENCY_HEATMAP_CB_GAP_PER_SMALL_IN
    inch_frac = max(SUFFICIENCY_HEATMAP_CB_GAP_FRAC_FLOOR, gap_in / h)
    w = _sufficiency_layout_inch_weight(h, n_heatmap_rows)
    blended = (1.0 - w) * legacy + w * inch_frac
    return min(0.36, max(SUFFICIENCY_HEATMAP_CB_GAP_FRAC_FLOOR, blended))


def sufficiency_heatmap_ytick_fontsize(n_rows: int) -> float:
    """Y-tick (training-split %) fontsize for both heatmap panels; avoids overlap at many rows."""
    n = max(int(n_rows), 1)
    return float(max(FONT_SIZE_TINY, min(float(FONT_SIZE_TICK), 96.0 / n)))


def sufficiency_heatmap_wspace(n_rows: int) -> float:
    """Horizontal space between the two heatmaps when both show y tick labels."""
    n = max(int(n_rows), 1)
    return float(max(0.09, min(0.26, 0.062 + 0.0142 * n)))
