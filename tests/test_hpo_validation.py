"""
Tests for HPO config validation and trial config helpers.

validate_hpo_pipeline_config gates HPO runs; _ensure_single_value collapses
template lists with a warning when not in search space.
"""
from __future__ import annotations

import os
import sys
import warnings

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.scoring import validate_hpo_pipeline_config

_SAMPLES_PIPELINE_HPO = os.path.join(_PIPELINE_ROOT, "samples", "config_sample_hpo.yaml")


def _hpo_valid_cfg():
    """Minimal pipeline config that passes HPO validation."""
    return {
        "scoring": {"enabled": True, "tau_config": "scoring_tau_sample.yaml"},
        "plotting": {"enabled": True, "sample_variance": [1.0]},
        "analysis": {
            "rmsd_gen": {"enabled": True, "sample_variance": [1.0]},
            "q_gen": {"enabled": True, "sample_variance": [1.0]},
            "coord_clustering_gen": {"enabled": True, "sample_variance": [1.0]},
            "distmap_clustering_gen": {"enabled": True, "sample_variance": [1.0]},
        },
    }


def test_validate_hpo_pipeline_config_valid():
    ok, errors = validate_hpo_pipeline_config(_hpo_valid_cfg(), _SAMPLES_PIPELINE_HPO)
    assert ok is True and errors == []


def test_validate_hpo_pipeline_config_rejects_missing_tau_config():
    cfg = _hpo_valid_cfg()
    del cfg["scoring"]["tau_config"]
    ok, errors = validate_hpo_pipeline_config(cfg, _SAMPLES_PIPELINE_HPO)
    assert ok is False
    assert any("tau_config" in e for e in errors)


def test_validate_hpo_pipeline_config_scoring_disabled():
    cfg = _hpo_valid_cfg()
    cfg["scoring"]["enabled"] = False
    ok, errors = validate_hpo_pipeline_config(cfg, _SAMPLES_PIPELINE_HPO)
    assert ok is False
    assert any("scoring" in e for e in errors)


def test_validate_hpo_pipeline_config_missing_variance_one():
    cfg = _hpo_valid_cfg()
    cfg["plotting"]["sample_variance"] = [2.0]  # no 1
    ok, errors = validate_hpo_pipeline_config(cfg, _SAMPLES_PIPELINE_HPO)
    assert ok is False
    assert any("plotting" in e and "sample_variance" in e for e in errors)


def test_validate_hpo_pipeline_config_analysis_block_disabled():
    cfg = _hpo_valid_cfg()
    cfg["analysis"]["rmsd_gen"]["enabled"] = False
    ok, errors = validate_hpo_pipeline_config(cfg, _SAMPLES_PIPELINE_HPO)
    assert ok is False
    assert any("rmsd_gen" in e for e in errors)


# _ensure_single_value lives in run_hpo; import after path setup
def test_ensure_single_value_list_emits_warning():
    import run_hpo
    cfg = {"distmap": {"beta_kl": [0.01, 0.05]}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_hpo._ensure_single_value(cfg, "distmap", "beta_kl")
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert cfg["distmap"]["beta_kl"] == 0.01  # first element for non-epoch keys


def test_ensure_single_value_epochs_uses_max():
    import run_hpo
    cfg = {"distmap": {"epochs": [100, 300, 200]}}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        run_hpo._ensure_single_value(cfg, "distmap", "epochs")
    assert cfg["distmap"]["epochs"] == 300  # max for epochs


def test_ensure_single_value_scalar_no_warning():
    import run_hpo
    cfg = {"distmap": {"beta_kl": 0.01}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_hpo._ensure_single_value(cfg, "distmap", "beta_kl")
    assert len(w) == 0  # no warning for scalar
    assert cfg["distmap"]["beta_kl"] == 0.01
