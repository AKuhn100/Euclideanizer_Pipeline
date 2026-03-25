"""
Unit tests for scoring module: z-score, MAE, Wasserstein, ratio formulas, geometric mean, compute_scores_from_data.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.scoring import (
    zscore_combined,
    mae,
    wasserstein_on_zscored,
    exp_score,
    geometric_mean,
    recon_rmsd_d,
    recon_q_d,
    clustering_d,
    compute_scores_from_data,
    EXPECTED_COMPONENTS,
    load_scoring_tau_dict,
    SCORING_VARIANCE,
    _variance_equals_scoring,
    _run_name_has_scoring_variance,
    _gen_variance_stem_has_scoring_variance,
    _variance_lists_from_config,
    _pairwise_wasserstein_mean_from_lags,
    compute_and_save,
)

_SCORING_TAU_SAMPLE = os.path.join(_PIPELINE_ROOT, "samples", "scoring_tau_sample.yaml")


def _unit_taus() -> dict[str, float]:
    return load_scoring_tau_dict(_SCORING_TAU_SAMPLE)


def test_zscore_combined_zero_mean_unit_var():
    """Z-scored combined pool has zero mean and unit variance."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    a_norm, b_norm = zscore_combined(a, b)
    pool = np.concatenate([a_norm.ravel(), b_norm.ravel()])
    assert np.allclose(np.mean(pool), 0.0, atol=1e-10)
    assert np.allclose(np.std(pool), 1.0, atol=1e-10)


def test_mae_identical_zero():
    """MAE between identical arrays is 0."""
    x = np.array([1.0, 2.0, 3.0])
    assert mae(x, x) == 0.0


def test_mae_known():
    """MAE([0,2], [1,1]) = 1.0."""
    a = np.array([0.0, 2.0])
    b = np.array([1.0, 1.0])
    assert mae(a, b) == 1.0


def test_wasserstein_on_zscored_identical_small():
    """Wasserstein on z-scored identical samples is 0 (or very small)."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a.copy()
    d = wasserstein_on_zscored(a, b)
    assert d >= 0 and d < 0.01


def test_pairwise_wasserstein_mean_identical_distributions_near_zero():
    """Identical distributions -> Wasserstein 0 -> mean near 0."""
    pytest.importorskip("scipy.stats")
    k_values = np.array([1, 2, 3])
    d = np.array([np.ones(10), np.ones(10), np.ones(10)], dtype=object)
    result = _pairwise_wasserstein_mean_from_lags(k_values, d, d.copy())
    assert result < 0.01


def test_pairwise_wasserstein_mean_different_distributions_positive():
    """Distinct distributions -> Wasserstein > 0."""
    pytest.importorskip("scipy.stats")
    k_values = np.array([1, 2])
    exp_d = np.array([np.ones(20), np.ones(20) * 2], dtype=object)
    gen_d = np.array([np.ones(20) * 5, np.ones(20) * 10], dtype=object)
    result = _pairwise_wasserstein_mean_from_lags(k_values, exp_d, gen_d)
    assert result > 0


def test_pairwise_wasserstein_empty_returns_nan():
    """Empty k_values -> nan."""
    result = _pairwise_wasserstein_mean_from_lags(
        np.array([]), np.array([]), np.array([])
    )
    assert np.isnan(result)


def test_exp_score_zero_d_is_one():
    """d=0 -> score = 1."""
    assert exp_score(0.0, 1.0) == 1.0


def test_exp_score_one_sigma():
    """d=1, tau=1 -> score approx 0.37."""
    s = exp_score(1.0, 1.0)
    assert 0.35 < s < 0.40


def test_exp_score_negative_d_clamped():
    """Negative d should still return finite score (exp(-d) > 1 possible; spec uses d>=0)."""
    s = exp_score(-0.5, 1.0)
    assert np.isfinite(s) and s > 0


def test_recon_rmsd_d_better_than_baseline():
    """Recon median < tt median -> d < 1 -> score > 0.37."""
    d = recon_rmsd_d(0.5, 1.0)
    assert d == 0.5
    assert exp_score(d, 1.0) > 0.37


def test_recon_rmsd_d_equal_baseline():
    """Recon median = tt median -> d = 1 -> score approx 0.37."""
    d = recon_rmsd_d(1.0, 1.0)
    assert d == 1.0
    assert 0.35 < exp_score(d, 1.0) < 0.40


def test_recon_q_d_ideal():
    """Recon Q = 1 -> d = 0 -> score = 1."""
    d = recon_q_d(1.0, 0.5)
    assert d == 0.0
    assert exp_score(d, 1.0) == 1.0


def test_recon_q_d_baseline():
    """Recon Q = tt Q -> d = 1."""
    d = recon_q_d(0.6, 0.6)
    assert d == 1.0


def test_clustering_ratio_ge_one():
    """ratio >= 1 -> d = 0 -> score = 1."""
    assert clustering_d(1.0) == 0.0
    assert clustering_d(1.5) == 0.0
    assert exp_score(clustering_d(1.0), 1.0) == 1.0


def test_clustering_ratio_less_one():
    """ratio < 1 -> d = 1 - ratio."""
    assert clustering_d(0.5) == 0.5
    assert clustering_d(0.0) == 1.0


def test_geometric_mean_single():
    """Single score -> overall = that score."""
    assert geometric_mean([0.5]) == 0.5


def test_geometric_mean_two():
    """Geometric mean of [1, 1] = 1; [0.5, 0.5] = 0.5."""
    assert geometric_mean([1.0, 1.0]) == 1.0
    assert geometric_mean([0.5, 0.5]) == 0.5


def test_geometric_mean_zero():
    """One zero -> overall = 0."""
    assert geometric_mean([1.0, 0.0, 1.0]) == 0.0


def test_geometric_mean_empty_nan():
    """Empty list -> nan."""
    result = geometric_mean([])
    assert result != result or np.isnan(result)


def test_compute_scores_from_data_minimal():
    """With only test_to_train_rmsd and recon RMSD data, recon_rmsd components present; overall nan (not all 30)."""
    data = {
        "test_to_train_rmsd": np.array([1.0, 1.2, 1.1]),
        "recon_train_rmsd": np.array([0.3, 0.4]),
        "recon_test_rmsd": np.array([0.5, 0.6]),
    }
    result = compute_scores_from_data(data, _unit_taus())
    assert "overall_score" in result
    assert "recon_rmsd_train" in result["component_scores"]
    assert "recon_rmsd_test" in result["component_scores"]
    assert "recon_rmsd_train" in result["present"]
    assert len(result["component_scores"]) == len(EXPECTED_COMPONENTS)
    assert len(result["present"]) == 2
    assert len(result["missing"]) == len(EXPECTED_COMPONENTS) - 2
    assert np.isnan(result["overall_score"])


def test_compute_scores_from_data_clustering():
    """Clustering mixing ratios -> 8 components when all keys provided; overall nan (not all 30)."""
    data = {
        "clustering_mixing": {
            "coord_Train+Gen": 1.2,
            "coord_Test+Gen": 1.0,
            "coord_Train+Train Recon": 0.9,
            "coord_Test+Test Recon": 1.1,
            "distmap_Train+Gen": 1.0,
            "distmap_Test+Gen": 0.8,
            "distmap_Train+Train Recon": 1.3,
            "distmap_Test+Test Recon": 1.0,
        },
    }
    result = compute_scores_from_data(data, _unit_taus())
    assert "clustering_coord_gen_train" in result["component_scores"]
    assert "clustering_distmap_gen_test" in result["component_scores"]
    assert "clustering_distmap_gen_test" in result["present"]
    assert result["component_scores"]["clustering_coord_gen_train"] == 1.0  # ratio 1.2 -> d=0
    assert len(result["component_scores"]) == len(EXPECTED_COMPONENTS)
    assert len(result["present"]) == 8
    assert len(result["missing"]) == len(EXPECTED_COMPONENTS) - 8
    assert np.isnan(result["overall_score"])


def test_compute_scores_from_data_clustering_ignores_non_scoring_mixing_keys():
    """NPZ-style extra keys (e.g. coord_Train+Test for plots) must not affect scoring or require tau entries."""
    data = {
        "clustering_mixing": {
            "coord_Train+Test": 0.5,
            "coord_Train+Test+Gen": 0.3,
            "coord_Train+Gen": 1.2,
            "coord_Test+Gen": 1.0,
            "coord_Train+Train Recon": 0.9,
            "coord_Test+Test Recon": 1.1,
            "distmap_Train+Gen": 1.0,
            "distmap_Test+Gen": 0.8,
            "distmap_Train+Train Recon": 1.3,
            "distmap_Test+Test Recon": 1.0,
        },
    }
    result = compute_scores_from_data(data, _unit_taus())
    assert result["component_scores"]["clustering_coord_gen_train"] == 1.0
    assert len(result["present"]) == 8
    assert np.isnan(result["overall_score"])


def test_compute_scores_from_data_latent():
    """Latent means/stds -> latent_means and latent_stds components; overall nan (not all 30)."""
    data = {
        "latent_mean_train": np.array([0.0, 1.0, 0.5]),
        "latent_mean_test": np.array([0.0, 1.0, 0.5]),
        "latent_std_train": np.array([1.0, 1.0, 1.0]),
        "latent_std_test": np.array([1.0, 1.0, 1.0]),
    }
    result = compute_scores_from_data(data, _unit_taus())
    assert "latent_means" in result["component_scores"]
    assert "latent_stds" in result["component_scores"]
    assert result["component_scores"]["latent_means"] == 1.0  # identical -> MAE 0 -> score 1
    assert result["component_scores"]["latent_stds"] == 1.0
    assert len(result["present"]) == 2
    assert np.isnan(result["overall_score"])


def test_expected_components_count():
    """Scoring requires exactly 30 components for overall to be computed."""
    assert len(EXPECTED_COMPONENTS) == 30


# ---------- Variance=1-only scoring (mission-critical: gen data must be from sample_variance=1) ----------


def test_scoring_variance_constant():
    """SCORING_VARIANCE is 1.0."""
    assert SCORING_VARIANCE == 1.0


def test_variance_equals_scoring_accepts_one():
    """_variance_equals_scoring accepts 1, 1.0, and string "1"/"1.0"."""
    assert _variance_equals_scoring(1) is True
    assert _variance_equals_scoring(1.0) is True
    assert _variance_equals_scoring("1") is True
    assert _variance_equals_scoring("1.0") is True


def test_variance_equals_scoring_rejects_others():
    """_variance_equals_scoring rejects 2, 0.5, 0, "2", None."""
    assert _variance_equals_scoring(2) is False
    assert _variance_equals_scoring(0.5) is False
    assert _variance_equals_scoring(0) is False
    assert _variance_equals_scoring("2") is False
    assert _variance_equals_scoring(None) is False


def test_run_name_has_scoring_variance_accepts():
    """_run_name_has_scoring_variance accepts default_var1.0, 1000_var1.0, var1.0, var1."""
    assert _run_name_has_scoring_variance("default_var1.0") is True
    assert _run_name_has_scoring_variance("1000_var1.0") is True
    assert _run_name_has_scoring_variance("var1.0") is True
    assert _run_name_has_scoring_variance("var1") is True


def test_run_name_has_scoring_variance_rejects():
    """_run_name_has_scoring_variance rejects default_var2, default, 1000_var2.0, var10."""
    assert _run_name_has_scoring_variance("default_var2") is False
    assert _run_name_has_scoring_variance("default_var2.0") is False
    assert _run_name_has_scoring_variance("default") is False
    assert _run_name_has_scoring_variance("1000_var2.0") is False
    assert _run_name_has_scoring_variance("var10") is False


def test_gen_variance_stem_has_scoring_variance():
    """_gen_variance_stem_has_scoring_variance: gen_variance_1.0 and gen_variance_1 true; gen_variance_2 false."""
    assert _gen_variance_stem_has_scoring_variance("gen_variance_1.0") is True
    assert _gen_variance_stem_has_scoring_variance("gen_variance_1") is True
    assert _gen_variance_stem_has_scoring_variance("gen_variance_2") is False
    assert _gen_variance_stem_has_scoring_variance("gen_variance_2.0") is False
    assert _gen_variance_stem_has_scoring_variance("other_1.0") is False


def test_variance_lists_from_config():
    """_variance_lists_from_config extracts and normalizes sample_variance from nested config."""
    cfg = {
        "plotting": {"sample_variance": [1.0]},
        "analysis": {
            "rmsd_gen": {"sample_variance": [1]},
            "q_gen": {"sample_variance": 1.0},
            "coord_clustering_gen": {"sample_variance": []},
            "distmap_clustering_gen": {"sample_variance": [0.5, 1.0, 2.0]},
        },
    }
    out = _variance_lists_from_config(cfg)
    assert out["plotting"] == [1.0]
    assert out["rmsd_gen"] == [1.0]
    assert out["q_gen"] == [1.0]
    assert out["coord_clustering_gen"] == []
    assert out["distmap_clustering_gen"] == [0.5, 1.0, 2.0]


def test_compute_and_save_uses_only_variance_one_gen_variance(tmp_path):
    """Gen variance data: only gen_variance_1.0_data.npz is loaded; gen_variance_2.0 is ignored."""
    run_dir = tmp_path / "run"
    seed_dir = tmp_path / "seed"
    exp_cache = seed_dir / "experimental_statistics"
    exp_cache.mkdir(parents=True)
    (run_dir / "plots" / "recon_statistics" / "data").mkdir(parents=True)
    (run_dir / "plots" / "gen_variance" / "data").mkdir(parents=True)
    # Seed cache and recon stats (minimal so we don't need full data)
    np.savez_compressed(exp_cache / "exp_stats_train.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    np.savez_compressed(exp_cache / "exp_stats_test.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    np.savez_compressed(exp_cache / "test_to_train_rmsd.npz", test_to_train=np.array([0.5, 0.6]), train_coords_np=np.zeros((2, 3, 3)), test_coords_np=np.zeros((2, 3, 3)))
    np.savez_compressed(exp_cache / "q_test_to_train.npz", test_to_train_max_q=np.array([0.9, 0.85]))
    for subset in ("train", "test"):
        r = run_dir / "plots" / "recon_statistics" / "data" / f"recon_statistics_{subset}_data.npz"
        np.savez_compressed(r, recon_rg=np.array([1.0]), recon_scaling=np.array([1.0]), recon_avg_map=np.eye(3), pairwise_k_values=np.array([1]), pairwise_exp_d=np.array([np.array([1.0])], dtype=object), pairwise_recon_d=np.array([np.array([1.0])], dtype=object))
    # Gen variance: only 1.0 should be used for scoring
    np.savez_compressed(run_dir / "plots" / "gen_variance" / "data" / "gen_variance_2.0_data.npz", gen_rg=np.array([99.0]), gen_scaling=np.array([99.0]), avg_gen_map=np.eye(3), pairwise_k_values=np.array([1]), pairwise_gen_d=np.array([np.array([99.0])], dtype=object), pairwise_exp_composite_d=np.array([np.array([1.0])], dtype=object))
    np.savez_compressed(run_dir / "plots" / "gen_variance" / "data" / "gen_variance_1.0_data.npz", gen_rg=np.array([1.0]), gen_scaling=np.array([1.0]), avg_gen_map=np.eye(3), pairwise_k_values=np.array([1]), pairwise_gen_d=np.array([np.array([1.0])], dtype=object), pairwise_exp_composite_d=np.array([np.array([1.0])], dtype=object))
    cfg = {
        "plotting": {"sample_variance": [0.5, 1.0, 2.0]},
        "analysis": {"rmsd_gen": {"sample_variance": [1.0]}, "q_gen": {"sample_variance": [1.0]}, "coord_clustering_gen": {"sample_variance": [1.0]}, "distmap_clustering_gen": {"sample_variance": [1.0]}, "rmsd_recon": {}, "q_recon": {}, "coord_clustering_recon": {}, "distmap_clustering_recon": {}},
        "scoring": {"save_pdf_copy": False, "tau_config": _SCORING_TAU_SAMPLE},
    }
    out = compute_and_save(str(run_dir), str(seed_dir), cfg)
    assert out is not None
    with open(os.path.join(run_dir, "scoring", "scores.json")) as f:
        scores = json.load(f)
    # Gen components should come from 1.0 data (gen_rg = 1.0, not 99)
    assert "gen_rg" in scores["present"]
    assert scores["component_scores"]["gen_rg"] != np.exp(-99)  # would be if we used 2.0 data
    assert scores["component_scores"]["gen_rg"] > 0.5  # from variance=1 data


def test_compute_and_save_gen_variance_missing_when_one_absent_from_config(tmp_path):
    """When plotting.sample_variance does not contain 1.0, gen (plot) components are missing."""
    run_dir = tmp_path / "run"
    seed_dir = tmp_path / "seed"
    exp_cache = seed_dir / "experimental_statistics"
    exp_cache.mkdir(parents=True)
    (run_dir / "plots" / "recon_statistics" / "data").mkdir(parents=True)
    (run_dir / "plots" / "gen_variance" / "data").mkdir(parents=True)
    np.savez_compressed(exp_cache / "exp_stats_train.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    np.savez_compressed(exp_cache / "exp_stats_test.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    for subset in ("train", "test"):
        r = run_dir / "plots" / "recon_statistics" / "data" / f"recon_statistics_{subset}_data.npz"
        np.savez_compressed(r, recon_rg=np.array([1.0]), recon_scaling=np.array([1.0]), recon_avg_map=np.eye(3), pairwise_k_values=np.array([1]), pairwise_exp_d=np.array([np.array([1.0])], dtype=object), pairwise_recon_d=np.array([np.array([1.0])], dtype=object))
    # Only var=2.0 data present; config has only [2.0] -> scoring must not use it
    np.savez_compressed(run_dir / "plots" / "gen_variance" / "data" / "gen_variance_2.0_data.npz", gen_rg=np.array([1.0]), gen_scaling=np.array([1.0]), avg_gen_map=np.eye(3), pairwise_k_values=np.array([1]), pairwise_gen_d=np.array([np.array([1.0])], dtype=object), pairwise_exp_composite_d=np.array([np.array([1.0])], dtype=object))
    cfg = {
        "plotting": {"sample_variance": [2.0]},
        "analysis": {"rmsd_gen": {"sample_variance": []}, "q_gen": {"sample_variance": []}, "coord_clustering_gen": {"sample_variance": []}, "distmap_clustering_gen": {"sample_variance": []}, "rmsd_recon": {}, "q_recon": {}, "coord_clustering_recon": {}, "distmap_clustering_recon": {}},
        "scoring": {"save_pdf_copy": False, "tau_config": _SCORING_TAU_SAMPLE},
    }
    out = compute_and_save(str(run_dir), str(seed_dir), cfg)
    assert out is not None
    with open(os.path.join(run_dir, "scoring", "scores.json")) as f:
        scores = json.load(f)
    assert "gen_rg" in scores["missing"]
    assert "gen_scaling" in scores["missing"]
    assert "gen_pairwise" in scores["missing"]
    assert "gen_avgmap" in scores["missing"]


def test_compute_and_save_rmsd_gen_only_from_variance_one_path(tmp_path):
    """RMSD gen data: only loaded from analysis/rmsd/gen/<run_name>/data when run_name has _var1.0."""
    run_dir = tmp_path / "run"
    seed_dir = tmp_path / "seed"
    exp_cache = seed_dir / "experimental_statistics"
    exp_cache.mkdir(parents=True)
    (run_dir / "plots" / "recon_statistics" / "data").mkdir(parents=True)
    (run_dir / "plots" / "gen_variance" / "data").mkdir(parents=True)
    np.savez_compressed(exp_cache / "exp_stats_train.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    np.savez_compressed(exp_cache / "exp_stats_test.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    np.savez_compressed(exp_cache / "test_to_train_rmsd.npz", test_to_train=np.array([0.5, 0.6]), train_coords_np=np.zeros((2, 3, 3)), test_coords_np=np.zeros((2, 3, 3)))
    for subset in ("train", "test"):
        r = run_dir / "plots" / "recon_statistics" / "data" / f"recon_statistics_{subset}_data.npz"
        np.savez_compressed(r, recon_rg=np.array([1.0]), recon_scaling=np.array([1.0]), recon_avg_map=np.eye(3), pairwise_k_values=np.array([1]), pairwise_exp_d=np.array([np.array([1.0])], dtype=object), pairwise_recon_d=np.array([np.array([1.0])], dtype=object))
    np.savez_compressed(run_dir / "plots" / "gen_variance" / "data" / "gen_variance_1.0_data.npz", gen_rg=np.array([1.0]), gen_scaling=np.array([1.0]), avg_gen_map=np.eye(3), pairwise_k_values=np.array([1]), pairwise_gen_d=np.array([np.array([1.0])], dtype=object), pairwise_exp_composite_d=np.array([np.array([1.0])], dtype=object))
    # RMSD gen: only default_var1.0 should be loaded; default_var2.0 must be ignored
    (run_dir / "analysis" / "rmsd" / "gen" / "default_var2.0" / "data").mkdir(parents=True)
    (run_dir / "analysis" / "rmsd" / "gen" / "default_var1.0" / "data").mkdir(parents=True)
    np.savez_compressed(run_dir / "analysis" / "rmsd" / "gen" / "default_var2.0" / "data" / "rmsd_data.npz", gen_to_train=np.array([999.0]), gen_to_test=np.array([999.0]))
    np.savez_compressed(run_dir / "analysis" / "rmsd" / "gen" / "default_var1.0" / "data" / "rmsd_data.npz", gen_to_train=np.array([0.5]), gen_to_test=np.array([0.5]))
    cfg = {
        "plotting": {"sample_variance": [1.0]},
        "analysis": {"rmsd_gen": {"sample_variance": [1.0, 2.0]}, "q_gen": {"sample_variance": [1.0]}, "coord_clustering_gen": {"sample_variance": [1.0]}, "distmap_clustering_gen": {"sample_variance": [1.0]}, "rmsd_recon": {}, "q_recon": {}, "coord_clustering_recon": {}, "distmap_clustering_recon": {}},
        "scoring": {"save_pdf_copy": False, "tau_config": _SCORING_TAU_SAMPLE},
    }
    out = compute_and_save(str(run_dir), str(seed_dir), cfg)
    assert out is not None
    with open(os.path.join(run_dir, "scoring", "scores.json")) as f:
        scores = json.load(f)
    assert "gen_rmsd_train_vs_tt" in scores["present"]
    assert "gen_rmsd_test_vs_tt" in scores["present"]
    # Scores should reflect variance=1 data (0.5), not variance=2 data (999)
    assert scores["component_scores"]["gen_rmsd_train_vs_tt"] > 0.01


def test_compute_and_save_all_components_present_when_all_data_saved(tmp_path):
    """When pipeline runs top-to-bottom with scoring enabled, analysis saves NPZ (effective save_data).
    This test asserts: when all analysis/plot NPZ data is present, compute_and_save produces no missing components."""
    run_dir = tmp_path / "run"
    seed_dir = tmp_path / "seed"
    exp_cache = seed_dir / "experimental_statistics"
    exp_cache.mkdir(parents=True)

    # Seed caches
    np.savez_compressed(exp_cache / "exp_stats_train.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    np.savez_compressed(exp_cache / "exp_stats_test.npz", exp_rg=np.array([1.0]), exp_scaling=np.array([1.0]), avg_exp_map=np.eye(3))
    np.savez_compressed(exp_cache / "test_to_train_rmsd.npz", test_to_train=np.array([0.5, 0.6]))
    np.savez_compressed(exp_cache / "q_test_to_train.npz", test_to_train_q=np.array([0.7, 0.8]))

    # Plots
    (run_dir / "plots" / "recon_statistics" / "data").mkdir(parents=True)
    for subset in ("train", "test"):
        r = run_dir / "plots" / "recon_statistics" / "data" / f"recon_statistics_{subset}_data.npz"
        np.savez_compressed(r, recon_rg=np.array([1.0]), recon_scaling=np.array([1.0]), recon_avg_map=np.eye(3),
                            pairwise_k_values=np.array([1]), pairwise_exp_d=np.array([np.array([1.0])], dtype=object), pairwise_recon_d=np.array([np.array([1.0])], dtype=object))
    (run_dir / "plots" / "gen_variance" / "data").mkdir(parents=True)
    np.savez_compressed(run_dir / "plots" / "gen_variance" / "data" / "gen_variance_1.0_data.npz",
                        gen_rg=np.array([1.0]), gen_scaling=np.array([1.0]), avg_gen_map=np.eye(3),
                        pairwise_k_values=np.array([1]), pairwise_gen_d=np.array([np.array([1.0])], dtype=object), pairwise_exp_composite_d=np.array([np.array([1.0])], dtype=object))

    # RMSD gen + recon
    (run_dir / "analysis" / "rmsd" / "gen" / "default_var1.0" / "data").mkdir(parents=True)
    np.savez_compressed(run_dir / "analysis" / "rmsd" / "gen" / "default_var1.0" / "data" / "rmsd_data.npz", gen_to_train=np.array([0.5]), gen_to_test=np.array([0.5]))
    (run_dir / "analysis" / "rmsd" / "recon" / "data").mkdir(parents=True)
    np.savez_compressed(run_dir / "analysis" / "rmsd" / "recon" / "data" / "rmsd_recon_data.npz", recon_train_rmsd=np.array([0.3]), recon_test_rmsd=np.array([0.4]))

    # Q gen + recon
    (run_dir / "analysis" / "q" / "gen" / "default_var1.0" / "data").mkdir(parents=True)
    np.savez_compressed(run_dir / "analysis" / "q" / "gen" / "default_var1.0" / "data" / "q_data.npz", gen_to_train=np.array([0.8]), gen_to_test=np.array([0.8]))
    (run_dir / "analysis" / "q" / "recon" / "data").mkdir(parents=True)
    np.savez_compressed(run_dir / "analysis" / "q" / "recon" / "data" / "q_recon_data.npz", recon_train_q=np.array([0.9]), recon_test_q=np.array([0.9]))

    # Latent
    (run_dir / "analysis" / "latent" / "data").mkdir(parents=True)
    np.savez_compressed(run_dir / "analysis" / "latent" / "data" / "latent_stats.npz",
                        mean_train=np.array([0.0]), mean_test=np.array([0.0]), std_train=np.array([1.0]), std_test=np.array([1.0]))

    # Clustering: gen has Train+Gen, Test+Gen; recon has Train+Train Recon, Test+Test Recon (per _mix_key_to_component)
    for subdir in ("coord_clustering", "distmap_clustering"):
        (run_dir / "analysis" / subdir / "gen" / "default_var1.0" / "data").mkdir(parents=True)
        np.savez_compressed(run_dir / "analysis" / subdir / "gen" / "default_var1.0" / "data" / "clustering_data.npz",
                            mixing_keys=np.array(["Train+Gen", "Test+Gen"], dtype=object), mixing_ratio=np.array([1.0, 1.0], dtype=np.float64))
        (run_dir / "analysis" / subdir / "recon" / "data").mkdir(parents=True)
        np.savez_compressed(run_dir / "analysis" / subdir / "recon" / "data" / "clustering_data.npz",
                            mixing_keys=np.array(["Train+Train Recon", "Test+Test Recon"], dtype=object), mixing_ratio=np.array([1.0, 1.0], dtype=np.float64))

    cfg = {
        "plotting": {"sample_variance": [1.0]},
        "analysis": {
            "rmsd_gen": {"sample_variance": [1.0]}, "rmsd_recon": {},
            "q_gen": {"sample_variance": [1.0]}, "q_recon": {},
            "coord_clustering_gen": {"sample_variance": [1.0]}, "coord_clustering_recon": {},
            "distmap_clustering_gen": {"sample_variance": [1.0]}, "distmap_clustering_recon": {},
        },
        "scoring": {"save_pdf_copy": False, "tau_config": _SCORING_TAU_SAMPLE},
    }
    out = compute_and_save(str(run_dir), str(seed_dir), cfg)
    assert out is not None
    with open(os.path.join(run_dir, "scoring", "scores.json")) as f:
        scores = json.load(f)
    assert len(scores["missing"]) == 0, f"Expected no missing components; got: {scores['missing']}"
    assert len(scores["present"]) == len(EXPECTED_COMPONENTS)
    assert np.isfinite(scores["overall_score"]), f"overall_score should be finite; got {scores['overall_score']}"

