"""
Per-run scoring: reads pipeline NPZ outputs and config, computes 30-component
scores per SCORING_SPEC.md (z-score normalization, MAE/Wasserstein, ratio formulas,
geometric mean). Does not run training or analysis.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    wasserstein_distance = None

# Fixed tau = 1 for exponential kernel score = exp(-d)
TAU = 1.0

# Generation data used for scoring must come from sample_variance == 1 (prior); other variances give incomparable scores.
SCORING_VARIANCE = 1.0


def _variance_equals_scoring(v: Any) -> bool:
    """True if v is numerically 1 (or 1.0). Used to accept only variance=1 gen data for scoring."""
    try:
        return abs(float(v) - SCORING_VARIANCE) < 1e-9
    except (TypeError, ValueError):
        return False


def _run_name_has_scoring_variance(run_name: str) -> bool:
    """True if run_name indicates sample_variance == SCORING_VARIANCE (e.g. default_var1.0, 1000_var1.0, var1.0)."""
    if "_var" in run_name:
        suffix = run_name.split("_var", 1)[-1].strip()
        try:
            return _variance_equals_scoring(float(suffix))
        except ValueError:
            return False
    if run_name.startswith("var"):
        try:
            return _variance_equals_scoring(float(run_name[3:].strip()))
        except ValueError:
            return False
    return False


def _gen_variance_stem_has_scoring_variance(stem: str) -> bool:
    """True if stem is gen_variance_1.0 or gen_variance_1 (variance used for that plot data)."""
    if not stem.startswith("gen_variance_"):
        return False
    suffix = stem[len("gen_variance_"):].strip()
    try:
        return _variance_equals_scoring(float(suffix))
    except ValueError:
        return False


# Canonical order for component scores (recon 8, gen 4, gen_rmsd 2, recon_rmsd 2, latent 2, gen_q 2, recon_q 2; clustering 8 in fixed order)
COMPONENT_ORDER = [
    "recon_scaling_train", "recon_scaling_test", "recon_rg_train", "recon_rg_test",
    "recon_pairwise_train", "recon_pairwise_test", "recon_avgmap_train", "recon_avgmap_test",
    "gen_rg", "gen_scaling", "gen_pairwise", "gen_avgmap",
    "gen_rmsd_train_vs_tt", "gen_rmsd_test_vs_tt",
    "recon_rmsd_train", "recon_rmsd_test",
    "latent_means", "latent_stds",
    "gen_q_train_vs_tt", "gen_q_test_vs_tt",
    "recon_q_train", "recon_q_test",
]

# All 30 component IDs in fixed order for spider and for "all present" check. Clustering order matches _mix_key_to_component.
EXPECTED_COMPONENTS = COMPONENT_ORDER + [
    "clustering_coord_gen_train", "clustering_coord_gen_test",
    "clustering_coord_recon_train", "clustering_coord_recon_test",
    "clustering_distmap_gen_train", "clustering_distmap_gen_test",
    "clustering_distmap_recon_train", "clustering_distmap_recon_test",
]

# Dedicated scoring output directory under each Euclideanizer run (like plots/, analysis/)
SCORING_DIR = "scoring"
SCORES_SPIDER_FILENAME = "scores_spider.png"

# Clustering mixing key to component id (coord vs distmap, gen vs recon)
CLUSTERING_KEY_TO_COMPONENT = {
    "Train+Gen": ("coord_gen_train", "distmap_gen_train"),
    "Test+Gen": ("coord_gen_test", "distmap_gen_test"),
    "Train+Train Recon": ("coord_recon_train", "distmap_recon_train"),
    "Test+Test Recon": ("coord_recon_test", "distmap_recon_test"),
}


def zscore_combined(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalize using combined pool: mu, sigma from concatenated (a, b)."""
    pool = np.concatenate([np.asarray(a).ravel(), np.asarray(b).ravel()])
    mu = np.nanmean(pool)
    sigma = np.nanstd(pool)
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0
    a_norm = (np.asarray(a) - mu) / sigma
    b_norm = (np.asarray(b) - mu) / sigma
    return a_norm, b_norm


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two arrays (after flattening)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if len(a) != len(b):
        return float("nan")
    return float(np.nanmean(np.abs(a - b)))


def wasserstein_on_zscored(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 (W1) on z-scored combined pool. Returns discrepancy d."""
    if wasserstein_distance is None:
        return float("nan")
    a_norm, b_norm = zscore_combined(a, b)
    return float(wasserstein_distance(a_norm.ravel(), b_norm.ravel()))


def _pairwise_wasserstein_mean_from_lags(
    k_values: np.ndarray,
    exp_d_list: np.ndarray,
    other_d_list: np.ndarray,
) -> float:
    """Mean over lags of Wasserstein(z-scored exp_d, z-scored other_d). Used for recon and gen pairwise.
    k_values, exp_d_list, other_d_list are from NPZ (object arrays of 1d arrays per lag)."""
    if wasserstein_distance is None or k_values is None:
        return float("nan")
    k_flat = np.atleast_1d(k_values).ravel()
    n = len(k_flat)
    if n == 0:
        return float("nan")
    exp_arr = np.atleast_1d(exp_d_list)
    other_arr = np.atleast_1d(other_d_list)
    if len(exp_arr) < n or len(other_arr) < n:
        return float("nan")
    w1_vals = []
    for i in range(n):
        a = np.asarray(exp_arr.flat[i]).ravel()
        b = np.asarray(other_arr.flat[i]).ravel()
        if len(a) == 0 or len(b) == 0:
            continue
        d = wasserstein_on_zscored(a, b)
        if np.isfinite(d):
            w1_vals.append(d)
    if not w1_vals:
        return float("nan")
    return float(np.mean(w1_vals))


def exp_score(d: float, tau: float = TAU) -> float:
    """score = exp(-d/tau); tau=1. Negative d yields score > 1 (finite)."""
    if not np.isfinite(d):
        return float("nan")
    return float(np.exp(-d / tau))


def geometric_mean(scores: list[float]) -> float:
    """Geometric mean of positive finite scores. Returns 0 if any is 0, nan if empty."""
    valid = [s for s in scores if np.isfinite(s) and s > 0]
    if not valid:
        return float("nan")
    if any(s <= 0 for s in scores):
        return 0.0
    return float(np.exp(np.mean(np.log(valid))))


def recon_rmsd_d(median_recon: float, median_test_to_train: float) -> float:
    """d = median(recon RMSD) / median(test->train RMSD)."""
    if not np.isfinite(median_test_to_train) or median_test_to_train <= 0:
        return float("nan")
    return median_recon / median_test_to_train


def recon_q_d(median_recon_q: float, median_tt_q: float) -> float:
    """d = (1 - median(recon Q)) / (1 - median(test->train Q))."""
    denom = 1.0 - median_tt_q
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return (1.0 - median_recon_q) / denom


def clustering_d(ratio: float) -> float:
    """d = max(0, 1 - ratio). ratio = observed/expected mixing."""
    if not np.isfinite(ratio):
        return float("nan")
    return max(0.0, 1.0 - ratio)


def _recon_components(
    exp_scaling_train: np.ndarray | None,
    recon_scaling_train: np.ndarray | None,
    exp_scaling_test: np.ndarray | None,
    recon_scaling_test: np.ndarray | None,
    exp_rg_train: np.ndarray | None,
    recon_rg_train: np.ndarray | None,
    exp_rg_test: np.ndarray | None,
    recon_rg_test: np.ndarray | None,
    exp_avgmap_train: np.ndarray | None,
    recon_avgmap_train: np.ndarray | None,
    exp_avgmap_test: np.ndarray | None,
    recon_avgmap_test: np.ndarray | None,
    pairwise_wasserstein_train: float | None,
    pairwise_wasserstein_test: float | None,
) -> dict[str, float]:
    """Compute Recon (8) component scores. Missing data -> component omitted from returned dict."""
    out = {}
    if exp_scaling_train is not None and recon_scaling_train is not None:
        a, b = zscore_combined(exp_scaling_train, recon_scaling_train)
        out["recon_scaling_train"] = exp_score(mae(a, b))
    if exp_scaling_test is not None and recon_scaling_test is not None:
        a, b = zscore_combined(exp_scaling_test, recon_scaling_test)
        out["recon_scaling_test"] = exp_score(mae(a, b))
    if exp_rg_train is not None and recon_rg_train is not None:
        out["recon_rg_train"] = exp_score(wasserstein_on_zscored(exp_rg_train, recon_rg_train))
    if exp_rg_test is not None and recon_rg_test is not None:
        out["recon_rg_test"] = exp_score(wasserstein_on_zscored(exp_rg_test, recon_rg_test))
    if pairwise_wasserstein_train is not None and np.isfinite(pairwise_wasserstein_train):
        out["recon_pairwise_train"] = exp_score(pairwise_wasserstein_train)
    if pairwise_wasserstein_test is not None and np.isfinite(pairwise_wasserstein_test):
        out["recon_pairwise_test"] = exp_score(pairwise_wasserstein_test)
    if exp_avgmap_train is not None and recon_avgmap_train is not None:
        a, b = zscore_combined(exp_avgmap_train, recon_avgmap_train)
        out["recon_avgmap_train"] = exp_score(mae(a, b))
    if exp_avgmap_test is not None and recon_avgmap_test is not None:
        a, b = zscore_combined(exp_avgmap_test, recon_avgmap_test)
        out["recon_avgmap_test"] = exp_score(mae(a, b))
    return out


def _gen_components(
    gen_rg: np.ndarray | None,
    exp_rg_composite: np.ndarray | None,
    gen_scaling: np.ndarray | None,
    exp_scaling_composite: np.ndarray | None,
    gen_avgmap: np.ndarray | None,
    exp_avgmap_composite: np.ndarray | None,
    pairwise_wasserstein_mean: float | None,
) -> dict[str, float]:
    """Compute Gen (4) component scores."""
    out = {}
    if gen_rg is not None and exp_rg_composite is not None:
        out["gen_rg"] = exp_score(wasserstein_on_zscored(gen_rg, exp_rg_composite))
    if gen_scaling is not None and exp_scaling_composite is not None:
        a, b = zscore_combined(gen_scaling, exp_scaling_composite)
        out["gen_scaling"] = exp_score(mae(a, b))
    if pairwise_wasserstein_mean is not None and np.isfinite(pairwise_wasserstein_mean):
        out["gen_pairwise"] = exp_score(pairwise_wasserstein_mean)
    if gen_avgmap is not None and exp_avgmap_composite is not None:
        a, b = zscore_combined(gen_avgmap, exp_avgmap_composite)
        out["gen_avgmap"] = exp_score(mae(a, b))
    return out


def _gen_rmsd_components(
    gen_train_rmsd: np.ndarray | None,
    gen_test_rmsd: np.ndarray | None,
    test_to_train_rmsd: np.ndarray | None,
) -> dict[str, float]:
    """Gen RMSD (2): Wasserstein on z-scored vs test->train."""
    out = {}
    if test_to_train_rmsd is None:
        return out
    if gen_train_rmsd is not None:
        out["gen_rmsd_train_vs_tt"] = exp_score(wasserstein_on_zscored(gen_train_rmsd, test_to_train_rmsd))
    if gen_test_rmsd is not None:
        out["gen_rmsd_test_vs_tt"] = exp_score(wasserstein_on_zscored(gen_test_rmsd, test_to_train_rmsd))
    return out


def _recon_rmsd_components(
    recon_train_rmsd: np.ndarray | None,
    recon_test_rmsd: np.ndarray | None,
    test_to_train_rmsd: np.ndarray | None,
) -> dict[str, float]:
    """Recon RMSD (2): ratio median(recon)/median(tt)."""
    out = {}
    if test_to_train_rmsd is None or not len(test_to_train_rmsd):
        return out
    med_tt = float(np.median(test_to_train_rmsd))
    if recon_train_rmsd is not None and len(recon_train_rmsd):
        d = recon_rmsd_d(float(np.median(recon_train_rmsd)), med_tt)
        out["recon_rmsd_train"] = exp_score(d)
    if recon_test_rmsd is not None and len(recon_test_rmsd):
        d = recon_rmsd_d(float(np.median(recon_test_rmsd)), med_tt)
        out["recon_rmsd_test"] = exp_score(d)
    return out


def _latent_components(
    mean_train: np.ndarray | None,
    mean_test: np.ndarray | None,
    std_train: np.ndarray | None,
    std_test: np.ndarray | None,
) -> dict[str, float]:
    """Latent (2): MAE on z-scored means and stds (across dimensions)."""
    out = {}
    if mean_train is not None and mean_test is not None:
        a, b = zscore_combined(mean_train, mean_test)
        out["latent_means"] = exp_score(mae(a, b))
    if std_train is not None and std_test is not None:
        a, b = zscore_combined(std_train, std_test)
        out["latent_stds"] = exp_score(mae(a, b))
    return out


def _gen_q_components(
    gen_train_q: np.ndarray | None,
    gen_test_q: np.ndarray | None,
    test_to_train_q: np.ndarray | None,
) -> dict[str, float]:
    """Gen Q (2): Wasserstein on z-scored vs test->train."""
    out = {}
    if test_to_train_q is None:
        return out
    if gen_train_q is not None:
        out["gen_q_train_vs_tt"] = exp_score(wasserstein_on_zscored(gen_train_q, test_to_train_q))
    if gen_test_q is not None:
        out["gen_q_test_vs_tt"] = exp_score(wasserstein_on_zscored(gen_test_q, test_to_train_q))
    return out


def _recon_q_components(
    recon_train_q: np.ndarray | None,
    recon_test_q: np.ndarray | None,
    test_to_train_q: np.ndarray | None,
) -> dict[str, float]:
    """Recon Q (2): ratio (1-median(recon))/(1-median(tt))."""
    out = {}
    if test_to_train_q is None or not len(test_to_train_q):
        return out
    med_tt = float(np.median(test_to_train_q))
    if recon_train_q is not None and len(recon_train_q):
        d = recon_q_d(float(np.median(recon_train_q)), med_tt)
        out["recon_q_train"] = exp_score(d)
    if recon_test_q is not None and len(recon_test_q):
        d = recon_q_d(float(np.median(recon_test_q)), med_tt)
        out["recon_q_test"] = exp_score(d)
    return out


def compute_scores_from_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Compute all component scores from a data dict (arrays and scalars).
    data keys are the same as the keyword args expected by _recon_components, _gen_components, etc.
    Returns dict with: overall_score, component_scores, present, missing.
    """
    component_scores: dict[str, float] = {}
    # Recon (8)
    component_scores.update(
        _recon_components(
            data.get("exp_scaling_train"),
            data.get("recon_scaling_train"),
            data.get("exp_scaling_test"),
            data.get("recon_scaling_test"),
            data.get("exp_rg_train"),
            data.get("recon_rg_train"),
            data.get("exp_rg_test"),
            data.get("recon_rg_test"),
            data.get("exp_avgmap_train"),
            data.get("recon_avgmap_train"),
            data.get("exp_avgmap_test"),
            data.get("recon_avgmap_test"),
            data.get("pairwise_wasserstein_train"),
            data.get("pairwise_wasserstein_test"),
        )
    )
    # Gen (4)
    component_scores.update(
        _gen_components(
            data.get("gen_rg"),
            data.get("exp_rg_composite"),
            data.get("gen_scaling"),
            data.get("exp_scaling_composite"),
            data.get("gen_avgmap"),
            data.get("exp_avgmap_composite"),
            data.get("gen_pairwise_wasserstein_mean"),
        )
    )
    # Gen RMSD (2)
    component_scores.update(
        _gen_rmsd_components(
            data.get("gen_train_rmsd"),
            data.get("gen_test_rmsd"),
            data.get("test_to_train_rmsd"),
        )
    )
    # Recon RMSD (2)
    component_scores.update(
        _recon_rmsd_components(
            data.get("recon_train_rmsd"),
            data.get("recon_test_rmsd"),
            data.get("test_to_train_rmsd"),
        )
    )
    # Latent (2)
    component_scores.update(
        _latent_components(
            data.get("latent_mean_train"),
            data.get("latent_mean_test"),
            data.get("latent_std_train"),
            data.get("latent_std_test"),
        )
    )
    # Gen Q (2)
    component_scores.update(
        _gen_q_components(
            data.get("gen_train_q"),
            data.get("gen_test_q"),
            data.get("test_to_train_q"),
        )
    )
    # Recon Q (2)
    component_scores.update(
        _recon_q_components(
            data.get("recon_train_q"),
            data.get("recon_test_q"),
            data.get("test_to_train_q"),
        )
    )
    # Clustering (8): from mixing_ratio dict keyed by e.g. "coord_Train+Gen", "distmap_Test+Test Recon"
    _mix_key_to_component = {
        "coord_Train+Gen": "clustering_coord_gen_train",
        "coord_Test+Gen": "clustering_coord_gen_test",
        "coord_Train+Train Recon": "clustering_coord_recon_train",
        "coord_Test+Test Recon": "clustering_coord_recon_test",
        "distmap_Train+Gen": "clustering_distmap_gen_train",
        "distmap_Test+Gen": "clustering_distmap_gen_test",
        "distmap_Train+Train Recon": "clustering_distmap_recon_train",
        "distmap_Test+Test Recon": "clustering_distmap_recon_test",
    }
    mixing = data.get("clustering_mixing")
    if isinstance(mixing, dict):
        for key, ratio in mixing.items():
            d = clustering_d(ratio)
            comp_name = _mix_key_to_component.get(key, "clustering_" + key.replace("+", "_").replace(" ", "_"))
            component_scores[comp_name] = exp_score(d)

    # Fill all 30 expected keys so spider and JSON always have a full set; missing = nan
    for c in EXPECTED_COMPONENTS:
        component_scores.setdefault(c, float("nan"))

    present = [c for c in EXPECTED_COMPONENTS if np.isfinite(component_scores.get(c)) and 0 <= component_scores[c] <= 1]
    missing = [c for c in EXPECTED_COMPONENTS if c not in present]
    scores_list = [component_scores[c] for c in present]
    overall = (
        geometric_mean(scores_list)
        if len(present) == len(EXPECTED_COMPONENTS)
        else float("nan")
    )

    return {
        "overall_score": overall,
        "component_scores": component_scores,
        "present": present,
        "missing": missing,
    }


def _component_order_for_spider(component_scores: dict[str, float]) -> list[str]:
    """Return all 30 component ids in canonical order for the spider (always full set)."""
    return list(EXPECTED_COMPONENTS)


def render_scores_spider(
    scores_data: dict[str, Any],
    output_dir: str,
    *,
    save_pdf: bool = False,
) -> str | None:
    """
    Draw a clean radar/spider chart of all component scores.
    Saves PNG (and optional PDF) under output_dir. Returns path to PNG or None on failure.
    All points use COLOR_GEN. Labels are orange for missing components, COLOR_GRAY_TEXT otherwise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plot_config import (
        COLOR_GEN,
        COLOR_GRAY_LIGHT,
        COLOR_GRAY_TEXT,
        PLOT_DPI,
        FONT_SIZE_TICK,
        FONT_SIZE_SMALL,
    )

    component_scores = scores_data.get("component_scores") or {}
    missing_set = set(scores_data.get("missing") or [])
    components = _component_order_for_spider(component_scores)
    # Always 30 axes; missing components plot as 0
    values = [0.0 if c in missing_set else float(component_scores.get(c, 0.0)) for c in components]
    n = len(components)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    values_closed = values + values[:1]

    # ── figure setup ──────────────────────────────────────────────────────────
    # Extra figure height so bottom labels don't get clipped
    fig = plt.figure(figsize=(9, 9), facecolor="white")
    # Shrink axes vertically to leave room at bottom for crowded labels
    ax = fig.add_axes([0.1, 0.12, 0.8, 0.8], projection="polar", facecolor="white")

    # Fully suppress matplotlib's built-in polar grid before drawing anything
    ax.grid(False)
    ax.set_axisbelow(True)

    # ── manual grid rings ─────────────────────────────────────────────────────
    ring_levels = [0.25, 0.5, 0.75, 1.0]
    for level in ring_levels:
        ax.plot(
            angles_closed, [level] * (n + 1),
            color=COLOR_GRAY_LIGHT, linewidth=0.6, linestyle="-", zorder=1,
        )

    # ── spoke lines ───────────────────────────────────────────────────────────
    for angle in angles:
        ax.plot(
            [angle, angle], [0, 1],
            color=COLOR_GRAY_LIGHT, linewidth=0.5, zorder=1,
        )

    # ── filled area + border line ─────────────────────────────────────────────
    ax.fill(angles_closed, values_closed, alpha=0.12, color=COLOR_GEN, zorder=2)
    ax.plot(angles_closed, values_closed, color=COLOR_GEN, linewidth=1.5, zorder=3)

    # ── data points (all COLOR_GEN) ───────────────────────────────────────────
    ax.scatter(angles, values, s=35, color=COLOR_GEN, zorder=4, clip_on=False)

    # ── axis config ───────────────────────────────────────────────────────────
    ax.set_ylim(0, 1)
    ax.set_yticks(ring_levels)
    ax.set_yticklabels(
        [f"{v:.2f}" for v in ring_levels],
        fontsize=FONT_SIZE_TICK, color=COLOR_GRAY_TEXT,
    )
    ax.yaxis.set_tick_params(pad=2)
    ax.set_rlabel_position(90)
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

# ── spoke labels ──────────────────────────────────────────────────────────
    label_r = 1.2  # bump out a touch more since we're no longer side-hugging
    for angle, comp in zip(angles, components):
        is_missing = comp in missing_set
        label_color = "#E07B00" if is_missing else COLOR_GRAY_TEXT
        label_weight = "semibold" if is_missing else "normal"

        # Display label: drop "_vs_tt" so we don't show "Vs Tt" on the plot
        comp_for_label = comp.replace("_vs_tt", "").replace("_", " ").title()
        raw = comp_for_label
        if len(raw) > 16:
            mid = raw.rfind(" ", 0, 17)
            if mid == -1:
                mid = 16
            raw = raw[:mid] + "\n" + raw[mid + 1:]

        deg = np.degrees(angle) % 360
        # All labels center-aligned; va flips at the bottom half so text
        # grows away from the ring rather than back through it
        va = "bottom" if deg <= 180 else "top"

        ax.text(
            angle, label_r, raw,
            ha="center", va=va,
            fontsize=FONT_SIZE_SMALL,
            color=label_color,
            fontweight=label_weight,
            linespacing=1.3,
        )

    # # ── overall score — inside axes at centre, below the rings ───────────────
    # if not missing_set:
    #     pos_scores = [v for v in values if v > 0]
    #     if pos_scores:
    #         overall = float(np.exp(np.mean(np.log(pos_scores))))
    #         ax.text(
    #             0, 0,
    #             f"{overall:.3f}",
    #             ha="center", va="center",
    #             fontsize=FONT_SIZE_TICK + 1,
    #             color=COLOR_GRAY_TEXT,
    #             transform=ax.transData,
    #             zorder=5,
    #         )

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, SCORES_SPIDER_FILENAME)
    try:
        fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
        if save_pdf:
            pdf_path = os.path.splitext(png_path)[0] + ".pdf"
            fig.savefig(pdf_path, bbox_inches="tight", format="pdf", facecolor="white")
    finally:
        plt.close(fig)
    return png_path


def _load_npz_safe(path: str) -> dict[str, np.ndarray] | None:
    """Load NPZ and return dict of arrays; None if missing/invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def _variance_lists_from_config(cfg: dict[str, Any]) -> dict[str, list[float]]:
    """Extract sample_variance lists for each gen block; normalize to list of floats. Used to enforce scoring uses only variance=1 data."""
    def _norm(key_path: list[str]) -> list[float]:
        c = cfg
        for k in key_path:
            c = (c or {}).get(k) or {}
        if c is None:
            return []
        if not isinstance(c, list):
            c = [c]
        out = []
        for v in c:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
        return out

    return {
        "plotting": _norm(["plotting", "sample_variance"]),
        "rmsd_gen": _norm(["analysis", "rmsd_gen", "sample_variance"]),
        "q_gen": _norm(["analysis", "q_gen", "sample_variance"]),
        "coord_clustering_gen": _norm(["analysis", "coord_clustering_gen", "sample_variance"]),
        "distmap_clustering_gen": _norm(["analysis", "distmap_clustering_gen", "sample_variance"]),
    }


def validate_hpo_pipeline_config(cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that the pipeline config is suitable for HPO: scoring enabled and all
    score-relevant blocks enabled with sample_variance containing 1 (so scoring can compute full scores).
    Returns (True, []) if valid; (False, [error messages]) otherwise.
    """
    errors = []
    if not (cfg.get("scoring") and cfg["scoring"].get("enabled")):
        errors.append("scoring.enabled must be true for HPO (scoring is the objective).")
    plot_cfg = (cfg.get("plotting") or {})
    if not plot_cfg.get("enabled"):
        errors.append("plotting.enabled must be true for HPO (required for scoring inputs).")
    variance_lists = _variance_lists_from_config(cfg)
    if not any(_variance_equals_scoring(v) for v in variance_lists["plotting"]):
        errors.append("plotting.sample_variance must include 1 (scoring uses gen variance 1 only).")
    for key in ("rmsd_gen", "q_gen", "coord_clustering_gen", "distmap_clustering_gen"):
        a = cfg.get("analysis") or {}
        block = a.get(key) or {}
        if not block.get("enabled"):
            errors.append(f"analysis.{key}.enabled must be true for HPO (required for scoring).")
        if not any(_variance_equals_scoring(v) for v in variance_lists[key]):
            errors.append(f"analysis.{key}.sample_variance must include 1 (scoring uses gen variance 1 only).")
    return (len(errors) == 0, errors)


def compute_and_save(
    run_dir: str,
    seed_dir: str,
    cfg: dict[str, Any],
    *,
    scores_filename: str = "scores.json",
) -> str | None:
    """
    Load NPZ from run_dir and seed_dir, compute scores, write scores_filename under run_dir.
    Generation-related scores use only data from sample_variance=1; if variance 1 is not in config for a block, those components are missing.
    Returns path to written file or None if scoring failed (e.g. no data).
    """
    data: dict[str, Any] = {}
    variance_lists = _variance_lists_from_config(cfg)
    use_gen_variance_from_plot = any(_variance_equals_scoring(v) for v in variance_lists["plotting"])
    use_rmsd_gen = any(_variance_equals_scoring(v) for v in variance_lists["rmsd_gen"])
    use_q_gen = any(_variance_equals_scoring(v) for v in variance_lists["q_gen"])
    use_coord_clustering_gen = any(_variance_equals_scoring(v) for v in variance_lists["coord_clustering_gen"])
    use_distmap_clustering_gen = any(_variance_equals_scoring(v) for v in variance_lists["distmap_clustering_gen"])

    # Seed caches
    exp_cache = os.path.join(seed_dir, "experimental_statistics")
    train_npz = os.path.join(exp_cache, "exp_stats_train.npz")
    test_npz = os.path.join(exp_cache, "exp_stats_test.npz")
    train_stats = _load_npz_safe(train_npz)
    test_stats = _load_npz_safe(test_npz)
    if train_stats:
        data["exp_rg_train"] = train_stats.get("exp_rg")
        data["exp_scaling_train"] = train_stats.get("exp_scaling")
        data["exp_avgmap_train"] = train_stats.get("avg_exp_map")
    if test_stats:
        data["exp_rg_test"] = test_stats.get("exp_rg")
        data["exp_scaling_test"] = test_stats.get("exp_scaling")
        data["exp_avgmap_test"] = test_stats.get("avg_exp_map")

    # Composite exp (train + test) for Gen
    if train_stats is not None and test_stats is not None:
        data["exp_rg_composite"] = np.concatenate([
            np.asarray(train_stats.get("exp_rg", [])).ravel(),
            np.asarray(test_stats.get("exp_rg", [])).ravel(),
        ])
        data["exp_scaling_composite"] = (
            np.asarray(train_stats.get("exp_scaling", [])) + np.asarray(test_stats.get("exp_scaling", []))
        ) / 2.0
        data["exp_avgmap_composite"] = (
            np.asarray(train_stats.get("avg_exp_map", [])) + np.asarray(test_stats.get("avg_exp_map", []))
        ) / 2.0

    # Run dir: plot data and analysis data
    plots = os.path.join(run_dir, "plots")
    for subset, key_suffix in [("train", "train"), ("test", "test")]:
        recon_dir = os.path.join(plots, "recon_statistics", "data")
        recon_npz = os.path.join(recon_dir, f"recon_statistics_{subset}_data.npz")
        r = _load_npz_safe(recon_npz)
        if r:
            data[f"recon_rg_{key_suffix}"] = r.get("recon_rg")
            data[f"recon_scaling_{key_suffix}"] = r.get("recon_scaling")
            data[f"recon_avgmap_{key_suffix}"] = r.get("recon_avg_map")
            if data[f"recon_avgmap_{key_suffix}"] is None:
                data[f"recon_avgmap_{key_suffix}"] = r.get("recon_avgmap")
            # Pairwise: raw per-lag d(i,i+s) arrays -> compute mean Wasserstein here
            kv, exp_d, recon_d = r.get("pairwise_k_values"), r.get("pairwise_exp_d"), r.get("pairwise_recon_d")
            if kv is not None and exp_d is not None and recon_d is not None:
                data[f"pairwise_wasserstein_{key_suffix}"] = _pairwise_wasserstein_mean_from_lags(kv, exp_d, recon_d)

    # Gen variance: only use data from sample_variance=1; stem = gen_variance_{var}
    gen_data_dir = os.path.join(plots, "gen_variance", "data")
    if os.path.isdir(gen_data_dir) and use_gen_variance_from_plot:
        for f in sorted(os.listdir(gen_data_dir)):
            if not (f.endswith("_data.npz") and "gen_variance" in f):
                continue
            stem = f[:-len("_data.npz")]
            if not _gen_variance_stem_has_scoring_variance(stem):
                continue
            g = _load_npz_safe(os.path.join(gen_data_dir, f))
            if g and data.get("gen_rg") is None:
                data["gen_rg"] = g.get("gen_rg")
                data["gen_scaling"] = g.get("gen_scaling")
                data["gen_avgmap"] = g.get("avg_gen_map")
                kv = g.get("pairwise_k_values")
                gen_d = g.get("pairwise_gen_d")
                exp_comp_d = g.get("pairwise_exp_composite_d")
                if kv is not None and gen_d is not None and exp_comp_d is not None:
                    data["gen_pairwise_wasserstein_mean"] = _pairwise_wasserstein_mean_from_lags(kv, exp_comp_d, gen_d)
            break

    # Analysis: RMSD, Q, clustering, latent
    analysis_dir = os.path.join(run_dir, "analysis")
    rmsd_dir = os.path.join(analysis_dir, "rmsd")
    if os.path.isdir(rmsd_dir):
        for root, _dirs, files in os.walk(rmsd_dir):
            for f in files:
                if f == "rmsd_data.npz":
                    # Gen RMSD: only from run_name with sample_variance=1 (path is .../rmsd/gen/<run_name>/data/)
                    if use_rmsd_gen:
                        run_name = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "data" else ""
                        if run_name and _run_name_has_scoring_variance(run_name):
                            rmsd = _load_npz_safe(os.path.join(root, f))
                            if rmsd:
                                data.setdefault("gen_train_rmsd", rmsd.get("gen_to_train"))
                                data.setdefault("gen_test_rmsd", rmsd.get("gen_to_test"))
                if f == "rmsd_recon_data.npz":
                    rec = _load_npz_safe(os.path.join(root, f))
                    if rec:
                        # Analysis (rmsd.py) saves train_recon_rmsd / test_recon_rmsd
                        data.setdefault("recon_train_rmsd", rec.get("recon_train_rmsd") or rec.get("train_recon_rmsd"))
                        data.setdefault("recon_test_rmsd", rec.get("recon_test_rmsd") or rec.get("test_recon_rmsd"))

    # Seed RMSD: test_to_train_rmsd.npz or test_to_train_rmsd_{max_train}_{max_test}.npz
    tt_rmsd_npz = _load_npz_safe(os.path.join(exp_cache, "test_to_train_rmsd.npz"))
    if not tt_rmsd_npz and os.path.isdir(exp_cache):
        for f in sorted(os.listdir(exp_cache)):
            if f.startswith("test_to_train_rmsd") and f.endswith(".npz"):
                tt_rmsd_npz = _load_npz_safe(os.path.join(exp_cache, f))
                break
    if tt_rmsd_npz:
        # rmsd.py saves seed cache with key "test_to_train" (not "test_to_train_rmsd")
        data["test_to_train_rmsd"] = tt_rmsd_npz.get("test_to_train")
        if data["test_to_train_rmsd"] is None:
            for k in tt_rmsd_npz:
                if "rmsd" in k.lower():
                    data["test_to_train_rmsd"] = tt_rmsd_npz[k]
                    break

    q_dir = os.path.join(analysis_dir, "q")
    if os.path.isdir(q_dir):
        for root, _dirs, files in os.walk(q_dir):
            for f in files:
                if f == "q_data.npz":
                    # Gen Q: only from run_name with sample_variance=1
                    if use_q_gen:
                        run_name = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "data" else ""
                        if run_name and _run_name_has_scoring_variance(run_name):
                            q = _load_npz_safe(os.path.join(root, f))
                            if q:
                                data.setdefault("gen_train_q", q.get("gen_to_train"))
                                data.setdefault("gen_test_q", q.get("gen_to_test"))
                if f == "q_recon_data.npz":
                    qr = _load_npz_safe(os.path.join(root, f))
                    if qr:
                        # Analysis (q_analysis.py) saves train_recon_q / test_recon_q
                        data.setdefault("recon_train_q", qr.get("recon_train_q") or qr.get("train_recon_q"))
                        data.setdefault("recon_test_q", qr.get("recon_test_q") or qr.get("test_recon_q"))

    # Seed Q: q_test_to_train.npz when both max_train/max_test null; else q_test_to_train_{mt}_{mc}.npz
    tt_q_npz = _load_npz_safe(os.path.join(exp_cache, "q_test_to_train.npz"))
    if not tt_q_npz and os.path.isdir(exp_cache):
        for f in sorted(os.listdir(exp_cache)):
            if f.startswith("q_test_to_train") and f.endswith(".npz"):
                tt_q_npz = _load_npz_safe(os.path.join(exp_cache, f))
                break
    if tt_q_npz:
        for k in tt_q_npz:
            if "q" in k.lower():
                data["test_to_train_q"] = tt_q_npz[k]
                break

    # Latent
    latent_npz = os.path.join(analysis_dir, "latent", "data", "latent_stats.npz")
    if not os.path.isfile(latent_npz):
        latent_npz = os.path.join(run_dir, "plots", "latent", "data", "latent_stats.npz")
    lat = _load_npz_safe(latent_npz)
    if lat:
        data["latent_mean_train"] = lat.get("mean_train")
        data["latent_mean_test"] = lat.get("mean_test")
        data["latent_std_train"] = lat.get("std_train")
        data["latent_std_test"] = lat.get("std_test")

    # Clustering: coord and distmap; merge clustering_data.npz. Gen only from run_name with sample_variance=1.
    for subdir, prefix in [("coord_clustering", "coord"), ("distmap_clustering", "distmap")]:
        use_gen = use_coord_clustering_gen if prefix == "coord" else use_distmap_clustering_gen
        cdir = os.path.join(analysis_dir, subdir)
        if not os.path.isdir(cdir):
            continue
        key_ratios = data.get("clustering_mixing") or {}
        for root, _dirs, files in os.walk(cdir):
            for f in files:
                if f != "clustering_data.npz":
                    continue
                # Gen paths are .../gen/<run_name>/data/clustering_data.npz; run_name is parent of current root
                if os.path.sep + "gen" + os.path.sep in root:
                    run_name = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "data" else os.path.basename(root)
                    if not use_gen or not run_name or not _run_name_has_scoring_variance(run_name):
                        continue
                cl = _load_npz_safe(os.path.join(root, f))
                if not cl or "mixing_ratio" not in cl:
                    continue
                mixing_keys = cl.get("mixing_keys")
                mixing_ratio = cl.get("mixing_ratio")
                if mixing_keys is None or mixing_ratio is None:
                    continue
                keys_flat = np.atleast_1d(mixing_keys)
                ratios_flat = np.atleast_1d(mixing_ratio)
                for i in range(min(len(keys_flat), len(ratios_flat))):
                    key_str = str(keys_flat[i]).strip()
                    r = float(ratios_flat[i])
                    key_ratios[f"{prefix}_{key_str}"] = r
        if key_ratios:
            data["clustering_mixing"] = key_ratios

    result = compute_scores_from_data(data)

    scoring_dir = os.path.join(run_dir, SCORING_DIR)
    os.makedirs(scoring_dir, exist_ok=True)
    out_path = os.path.join(scoring_dir, scores_filename)
    with open(out_path, "w") as f:
        json.dump(
            {
                "overall_score": result["overall_score"],
                "component_scores": result["component_scores"],
                "present": result["present"],
                "missing": result["missing"],
            },
            f,
            indent=2,
        )
    save_pdf = bool(cfg["scoring"]["save_pdf_copy"])
    render_scores_spider(result, scoring_dir, save_pdf=save_pdf)
    return out_path