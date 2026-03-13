"""
Per-run scoring: reads pipeline NPZ outputs and config, computes 30-component
scores per SCORING_SPEC.md (z-score normalization, MAE/EMD, ratio formulas,
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

# Categories and component ids for output (order matches spec summary)
CATEGORY_ORDER = [
    "recon",
    "gen",
    "gen_rmsd",
    "recon_rmsd",
    "latent",
    "gen_q",
    "recon_q",
    "clustering",
]

# Clustering mixing key to component id (coord vs distmap, gen vs recon)
CLUSTERING_KEY_TO_COMPONENT = {
    "Train+Gen": ("coord_gen_train", "distmap_gen_train"),  # one per coord/distmap run
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


def emd_on_zscored(a: np.ndarray, b: np.ndarray) -> float:
    """EMD (Wasserstein-1) on z-scored combined pool. Returns discrepancy d."""
    if wasserstein_distance is None:
        return float("nan")
    a_norm, b_norm = zscore_combined(a, b)
    return float(wasserstein_distance(a_norm.ravel(), b_norm.ravel()))


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
    pairwise_emd_train: float | None,
    pairwise_emd_test: float | None,
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
        out["recon_rg_train"] = exp_score(emd_on_zscored(exp_rg_train, recon_rg_train))
    if exp_rg_test is not None and recon_rg_test is not None:
        out["recon_rg_test"] = exp_score(emd_on_zscored(exp_rg_test, recon_rg_test))
    if pairwise_emd_train is not None and np.isfinite(pairwise_emd_train):
        out["recon_pairwise_train"] = exp_score(pairwise_emd_train)
    if pairwise_emd_test is not None and np.isfinite(pairwise_emd_test):
        out["recon_pairwise_test"] = exp_score(pairwise_emd_test)
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
    pairwise_emd_mean: float | None,
) -> dict[str, float]:
    """Compute Gen (4) component scores."""
    out = {}
    if gen_rg is not None and exp_rg_composite is not None:
        out["gen_rg"] = exp_score(emd_on_zscored(gen_rg, exp_rg_composite))
    if gen_scaling is not None and exp_scaling_composite is not None:
        a, b = zscore_combined(gen_scaling, exp_scaling_composite)
        out["gen_scaling"] = exp_score(mae(a, b))
    if pairwise_emd_mean is not None and np.isfinite(pairwise_emd_mean):
        out["gen_pairwise"] = exp_score(pairwise_emd_mean)
    if gen_avgmap is not None and exp_avgmap_composite is not None:
        a, b = zscore_combined(gen_avgmap, exp_avgmap_composite)
        out["gen_avgmap"] = exp_score(mae(a, b))
    return out


def _gen_rmsd_components(
    gen_train_rmsd: np.ndarray | None,
    gen_test_rmsd: np.ndarray | None,
    test_to_train_rmsd: np.ndarray | None,
) -> dict[str, float]:
    """Gen RMSD (2): EMD on z-scored vs test->train."""
    out = {}
    if test_to_train_rmsd is None:
        return out
    if gen_train_rmsd is not None:
        out["gen_rmsd_train_vs_tt"] = exp_score(emd_on_zscored(gen_train_rmsd, test_to_train_rmsd))
    if gen_test_rmsd is not None:
        out["gen_rmsd_test_vs_tt"] = exp_score(emd_on_zscored(gen_test_rmsd, test_to_train_rmsd))
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
    """Gen Q (2): EMD on z-scored vs test->train."""
    out = {}
    if test_to_train_q is None:
        return out
    if gen_train_q is not None:
        out["gen_q_train_vs_tt"] = exp_score(emd_on_zscored(gen_train_q, test_to_train_q))
    if gen_test_q is not None:
        out["gen_q_test_vs_tt"] = exp_score(emd_on_zscored(gen_test_q, test_to_train_q))
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


def _clustering_components(
    mixing_keys: np.ndarray | None,
    mixing_ratio: np.ndarray | None,
    coord_keys: set[str],
    distmap_keys: set[str],
) -> dict[str, float]:
    """
    Clustering (8): one score per (coord/distmap) x (gen_train, gen_test, recon_train, recon_test).
    mixing_keys and mixing_ratio are arrays from one NPZ; we need two NPZ (coord and distmap) to get 8.
    This function scores one set (coord or distmap) given keys and ratio arrays; caller calls twice.
    """
    out = {}
    if mixing_keys is None or mixing_ratio is None or len(mixing_keys) != len(mixing_ratio):
        return out
    key_to_components = {
        "Train+Gen": ["coord_gen_train", "distmap_gen_train"],
        "Test+Gen": ["coord_gen_test", "distmap_gen_test"],
        "Train+Train Recon": ["coord_recon_train", "distmap_recon_train"],
        "Test+Test Recon": ["coord_recon_test", "distmap_recon_test"],
    }
    for i, key in enumerate(mixing_keys):
        key_str = str(key).strip()
        ratio = float(mixing_ratio[i]) if i < len(mixing_ratio) else 0.0
        d = clustering_d(ratio)
        s = exp_score(d)
        if key_str in key_to_components:
            for comp in key_to_components[key_str]:
                if comp in coord_keys:
                    out["clustering_coord_" + comp.replace("coord_", "").replace("distmap_", "")] = s
                elif comp in distmap_keys:
                    out["clustering_distmap_" + comp.replace("coord_", "").replace("distmap_", "")] = s
    # Simpler: map key to single component id by prefix (coord vs distmap passed as which run we're in)
    return out


def compute_scores_from_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Compute all component scores from a data dict (arrays and scalars).
    data keys are the same as the keyword args expected by _recon_components, _gen_components, etc.
    Returns dict with: overall_score, category_scores, component_scores, present, missing.
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
            data.get("pairwise_emd_train"),
            data.get("pairwise_emd_test"),
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
            data.get("gen_pairwise_emd_mean"),
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

    present = [k for k, v in component_scores.items() if np.isfinite(v) and 0 <= v <= 1]
    scores_list = [component_scores[k] for k in present]
    overall = geometric_mean(scores_list) if scores_list else float("nan")

    category_scores: dict[str, float] = {}
    _category_prefix = {
        "recon": "recon_",
        "gen": "gen_",
        "gen_rmsd": "gen_rmsd_",
        "recon_rmsd": "recon_rmsd_",
        "latent": "latent_",
        "gen_q": "gen_q_",
        "recon_q": "recon_q_",
        "clustering": "clustering_",
    }
    for cat in CATEGORY_ORDER:
        prefix = _category_prefix.get(cat, cat + "_")
        cat_components = [k for k in present if k.startswith(prefix)]
        if not cat_components:
            continue
        cat_scores = [component_scores[k] for k in cat_components]
        category_scores[cat] = geometric_mean(cat_scores)

    missing = [k for k in component_scores if k not in present]

    return {
        "overall_score": overall,
        "category_scores": category_scores,
        "component_scores": component_scores,
        "present": present,
        "missing": missing,
    }


def _load_npz_safe(path: str) -> dict[str, np.ndarray] | None:
    """Load NPZ and return dict of arrays; None if missing/invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def compute_and_save(
    run_dir: str,
    seed_dir: str,
    cfg: dict[str, Any],
    *,
    scores_filename: str = "scores.json",
) -> str | None:
    """
    Load NPZ from run_dir and seed_dir, compute scores, write scores_filename under run_dir.
    Returns path to written file or None if scoring failed (e.g. no data).
    """
    data: dict[str, Any] = {}

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
        ) / 2.0  # same length curves: average
        data["exp_avgmap_composite"] = (
            np.asarray(train_stats.get("avg_exp_map", [])) + np.asarray(test_stats.get("avg_exp_map", []))
        ) / 2.0

    # Run dir: plot data and analysis data
    plots = os.path.join(run_dir, "plots")
    # Recon statistics (train/test) - optional data subdir
    for subset, key_suffix in [("train", "train"), ("test", "test")]:
        recon_dir = os.path.join(plots, "recon_statistics", "data")
        recon_npz = os.path.join(recon_dir, f"recon_statistics_{subset}_data.npz")
        r = _load_npz_safe(recon_npz)
        if r:
            data[f"recon_rg_{key_suffix}"] = r.get("recon_rg")
            data[f"recon_scaling_{key_suffix}"] = r.get("recon_scaling")
            data[f"recon_avgmap_{key_suffix}"] = r.get("recon_avg_map")  # key may vary
            if data[f"recon_avgmap_{key_suffix}"] is None:
                data[f"recon_avgmap_{key_suffix}"] = r.get("recon_avgmap")

    # Gen variance plot data (one per variance)
    gen_plots = os.path.join(plots, "gen_variance")
    if os.path.isdir(gen_plots):
        for name in os.listdir(gen_plots):
            data_dir = os.path.join(gen_plots, name, "data")
            gen_npz = os.path.join(data_dir, "gen_variance_data.npz")
            g = _load_npz_safe(gen_npz)
            if g and data.get("gen_rg") is None:
                data["gen_rg"] = g.get("gen_rg")
                data["gen_scaling"] = g.get("gen_scaling")
                data["gen_avgmap"] = g.get("avg_gen_map")
                break

    # Analysis: RMSD, Q, clustering, latent
    analysis_dir = os.path.join(run_dir, "analysis")
    rmsd_dir = os.path.join(analysis_dir, "rmsd")
    if os.path.isdir(rmsd_dir):
        for root, _dirs, files in os.walk(rmsd_dir):
            for f in files:
                if f == "rmsd_data.npz":
                    npz_path = os.path.join(root, f)
                    rmsd = _load_npz_safe(npz_path)
                    if rmsd:
                        data.setdefault("gen_train_rmsd", rmsd.get("gen_to_train_rmsd"))
                        data.setdefault("gen_test_rmsd", rmsd.get("gen_to_test_rmsd"))
                if f == "rmsd_recon_data.npz":
                    npz_path = os.path.join(root, f)
                    rec = _load_npz_safe(npz_path)
                    if rec:
                        data.setdefault("recon_train_rmsd", rec.get("recon_train_rmsd"))
                        data.setdefault("recon_test_rmsd", rec.get("recon_test_rmsd"))

    seed_rmsd = os.path.join(exp_cache, "test_to_train_rmsd.npz")
    tt_rmsd_npz = _load_npz_safe(seed_rmsd)
    if tt_rmsd_npz:
        for k in tt_rmsd_npz:
            if "rmsd" in k.lower():
                data["test_to_train_rmsd"] = tt_rmsd_npz[k]
                break

    q_dir = os.path.join(analysis_dir, "q")
    if os.path.isdir(q_dir):
        for root, _dirs, files in os.walk(q_dir):
            for f in files:
                if f == "q_data.npz":
                    q = _load_npz_safe(os.path.join(root, f))
                    if q:
                        data.setdefault("gen_train_q", q.get("gen_to_train_q"))
                        data.setdefault("gen_test_q", q.get("gen_to_test_q"))
                if f == "q_recon_data.npz":
                    qr = _load_npz_safe(os.path.join(root, f))
                    if qr:
                        data.setdefault("recon_train_q", qr.get("recon_train_q"))
                        data.setdefault("recon_test_q", qr.get("recon_test_q"))

    seed_q = os.path.join(exp_cache, "q_test_to_train_500_200.npz")
    if not os.path.isfile(seed_q):
        for f in os.listdir(exp_cache) if os.path.isdir(exp_cache) else []:
            if f.startswith("q_test_to_train") and f.endswith(".npz"):
                seed_q = os.path.join(exp_cache, f)
                break
    tt_q_npz = _load_npz_safe(seed_q)
    if tt_q_npz:
        for k in tt_q_npz:
            if "q" in k.lower():
                data["test_to_train_q"] = tt_q_npz[k]
                break

    # Latent
    latent_npz = os.path.join(run_dir, "plots", "latent", "data", "latent_stats.npz")
    if not os.path.isfile(latent_npz):
        latent_npz = os.path.join(analysis_dir, "latent", "data", "latent_stats.npz")
    lat = _load_npz_safe(latent_npz)
    if lat:
        data["latent_mean_train"] = lat.get("mean_train")
        data["latent_mean_test"] = lat.get("mean_test")
        data["latent_std_train"] = lat.get("std_train")
        data["latent_std_test"] = lat.get("std_test")

    # Clustering: coord and distmap
    for subdir, prefix in [("coord_clustering", "coord"), ("distmap_clustering", "distmap")]:
        cdir = os.path.join(analysis_dir, subdir)
        if not os.path.isdir(cdir):
            continue
        mixing_keys = None
        mixing_ratio = None
        for root, _dirs, files in os.walk(cdir):
            for f in files:
                if f == "clustering_data.npz":
                    cl = _load_npz_safe(os.path.join(root, f))
                    if cl and "mixing_ratio" in cl:
                        mixing_keys = cl.get("mixing_keys")
                        mixing_ratio = cl.get("mixing_ratio")
                        break
            if mixing_keys is not None:
                break
        if mixing_keys is not None and mixing_ratio is not None:
            key_ratios = data.get("clustering_mixing") or {}
            keys_flat = np.atleast_1d(mixing_keys)
            ratios_flat = np.atleast_1d(mixing_ratio)
            for i in range(min(len(keys_flat), len(ratios_flat))):
                key_str = str(keys_flat[i]).strip()
                r = float(ratios_flat[i])
                key_ratios[f"{prefix}_{key_str}"] = r
            data["clustering_mixing"] = key_ratios

    result = compute_scores_from_data(data)

    out_path = os.path.join(run_dir, scores_filename)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "overall_score": result["overall_score"],
                "category_scores": result["category_scores"],
                "component_scores": result["component_scores"],
                "present": result["present"],
                "missing": result["missing"],
            },
            f,
            indent=2,
        )
    return out_path
