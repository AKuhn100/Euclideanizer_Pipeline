"""
Tests for dashboard scanning and build.

_scan_runs discovers seed/distmap/euclideanizer run tree; build_dashboard
produces index.html and manifest.json.
"""
from __future__ import annotations

import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.dashboard import _scan_runs, build_dashboard
from src.config import save_run_config


def _make_minimal_run_tree(tmp_path, n_seeds=1, n_dm=1, n_eu=1):
    """Create minimal completed run tree for dashboard scanning."""
    for s in range(n_seeds):
        for di in range(n_dm):
            dm_dir = tmp_path / f"seed_{s}" / "distmap" / str(di) / "model"
            dm_dir.mkdir(parents=True)
            save_run_config(
                {"distmap": {"epochs": 10}}, str(dm_dir),
                last_epoch_trained=10, best_epoch=8, best_val=0.5,
            )
            (dm_dir / "model.pt").write_bytes(b"x")
            for ei in range(n_eu):
                eu_dir = tmp_path / f"seed_{s}" / "distmap" / str(di) / "euclideanizer" / str(ei) / "model"
                eu_dir.mkdir(parents=True)
                save_run_config(
                    {"euclideanizer": {"epochs": 10}}, str(eu_dir),
                    last_epoch_trained=10, best_epoch=8, best_val=0.5,
                )
                (eu_dir / "euclideanizer.pt").write_bytes(b"x")


def test_scan_runs_finds_expected_run_ids(tmp_path):
    _make_minimal_run_tree(tmp_path, n_seeds=2, n_dm=1, n_eu=1)
    runs = _scan_runs(str(tmp_path))
    ids = {r["id"] for r in runs}
    assert "seed_0" in ids
    assert "seed_1" in ids
    assert "seed_0_dm_0" in ids
    assert "seed_0_dm_0_eu_0" in ids


def test_scan_runs_levels_correct(tmp_path):
    _make_minimal_run_tree(tmp_path)
    runs = _scan_runs(str(tmp_path))
    by_id = {r["id"]: r for r in runs}
    assert by_id["seed_0"]["level"] == "seed"
    assert by_id["seed_0_dm_0"]["level"] == "distmap"
    assert by_id["seed_0_dm_0_eu_0"]["level"] == "euclideanizer"


def test_scan_runs_parent_child_links(tmp_path):
    _make_minimal_run_tree(tmp_path)
    runs = _scan_runs(str(tmp_path))
    by_id = {r["id"]: r for r in runs}
    assert "seed_0_dm_0" in by_id["seed_0"]["children_ids"]
    assert "seed_0_dm_0_eu_0" in by_id["seed_0_dm_0"]["children_ids"]
    assert by_id["seed_0_dm_0"]["parent_id"] == "seed_0"


def test_scan_runs_empty_dir(tmp_path):
    assert _scan_runs(str(tmp_path)) == []


def test_build_dashboard_creates_files(tmp_path):
    _make_minimal_run_tree(tmp_path)
    result = build_dashboard(str(tmp_path))
    assert result is not None
    assert os.path.isfile(os.path.join(result, "index.html"))
    assert os.path.isfile(os.path.join(result, "manifest.json"))
