"""
Unit tests for scoring module: z-score, MAE, EMD, ratio formulas, geometric mean, compute_scores_from_data.
"""
from __future__ import annotations

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
    emd_on_zscored,
    exp_score,
    geometric_mean,
    recon_rmsd_d,
    recon_q_d,
    clustering_d,
    compute_scores_from_data,
    TAU,
)


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


def test_emd_on_zscored_identical_small():
    """EMD on z-scored identical samples is 0 (or very small)."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a.copy()
    d = emd_on_zscored(a, b)
    assert d >= 0 and d < 0.01


def test_exp_score_zero_d_is_one():
    """d=0 -> score = 1."""
    assert exp_score(0.0) == 1.0


def test_exp_score_one_sigma():
    """d=1, tau=1 -> score approx 0.37."""
    s = exp_score(1.0, tau=TAU)
    assert 0.35 < s < 0.40


def test_exp_score_negative_d_clamped():
    """Negative d should still return finite score (exp(-d) > 1 possible; spec uses d>=0)."""
    s = exp_score(-0.5, tau=TAU)
    assert np.isfinite(s) and s > 0


def test_recon_rmsd_d_better_than_baseline():
    """Recon median < tt median -> d < 1 -> score > 0.37."""
    d = recon_rmsd_d(0.5, 1.0)
    assert d == 0.5
    assert exp_score(d) > 0.37


def test_recon_rmsd_d_equal_baseline():
    """Recon median = tt median -> d = 1 -> score approx 0.37."""
    d = recon_rmsd_d(1.0, 1.0)
    assert d == 1.0
    assert 0.35 < exp_score(d) < 0.40


def test_recon_q_d_ideal():
    """Recon Q = 1 -> d = 0 -> score = 1."""
    d = recon_q_d(1.0, 0.5)
    assert d == 0.0
    assert exp_score(d) == 1.0


def test_recon_q_d_baseline():
    """Recon Q = tt Q -> d = 1."""
    d = recon_q_d(0.6, 0.6)
    assert d == 1.0


def test_clustering_ratio_ge_one():
    """ratio >= 1 -> d = 0 -> score = 1."""
    assert clustering_d(1.0) == 0.0
    assert clustering_d(1.5) == 0.0
    assert exp_score(clustering_d(1.0)) == 1.0


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
    """With only test_to_train_rmsd and recon RMSD data, recon_rmsd components present."""
    data = {
        "test_to_train_rmsd": np.array([1.0, 1.2, 1.1]),
        "recon_train_rmsd": np.array([0.3, 0.4]),
        "recon_test_rmsd": np.array([0.5, 0.6]),
    }
    result = compute_scores_from_data(data)
    assert "overall_score" in result
    assert "recon_rmsd_train" in result["component_scores"]
    assert "recon_rmsd_test" in result["component_scores"]
    assert "recon_rmsd_train" in result["present"]
    assert result["overall_score"] > 0 and result["overall_score"] <= 1


def test_compute_scores_from_data_clustering():
    """Clustering mixing ratios -> 8 components when all keys provided."""
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
    result = compute_scores_from_data(data)
    assert "clustering_coord_gen_train" in result["component_scores"]
    assert "clustering_distmap_gen_test" in result["component_scores"]
    # ratio 0.8 -> d=0.2 -> score = exp(-0.2)
    assert "clustering_distmap_gen_test" in result["present"]
    assert result["component_scores"]["clustering_coord_gen_train"] == 1.0  # ratio 1.2 -> d=0


def test_compute_scores_from_data_latent():
    """Latent means/stds -> latent_means and latent_stds components."""
    data = {
        "latent_mean_train": np.array([0.0, 1.0, 0.5]),
        "latent_mean_test": np.array([0.0, 1.0, 0.5]),
        "latent_std_train": np.array([1.0, 1.0, 1.0]),
        "latent_std_test": np.array([1.0, 1.0, 1.0]),
    }
    result = compute_scores_from_data(data)
    assert "latent_means" in result["component_scores"]
    assert "latent_stds" in result["component_scores"]
    assert result["component_scores"]["latent_means"] == 1.0  # identical -> MAE 0 -> score 1
    assert result["component_scores"]["latent_stds"] == 1.0
