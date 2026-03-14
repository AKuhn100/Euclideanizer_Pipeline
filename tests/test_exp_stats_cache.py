"""
Tests for experimental statistics cache invalidation.

Ensures _load_exp_stats_cache and _load_exp_stats_split_cache return None when
data path or dimensions change, so resume does not silently reuse wrong data.
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

from run import (
    _load_exp_stats_cache,
    _save_exp_stats_cache,
    _load_exp_stats_split_cache,
    _save_exp_stats_split_cache,
)


def test_exp_stats_cache_invalidated_on_data_path_change(tmp_path):
    stats = {"exp_bonds": np.array([1.0, 2.0])}
    _save_exp_stats_cache(str(tmp_path), "/data/foo.gro", 100, 50, stats)
    # Same path/dims: should load
    assert _load_exp_stats_cache(str(tmp_path), "/data/foo.gro", 100, 50) is not None
    # Different path: should return None
    assert _load_exp_stats_cache(str(tmp_path), "/data/bar.gro", 100, 50) is None
    # Different dims: should return None
    assert _load_exp_stats_cache(str(tmp_path), "/data/foo.gro", 99, 50) is None
    assert _load_exp_stats_cache(str(tmp_path), "/data/foo.gro", 100, 51) is None


def test_exp_stats_split_cache_invalidated_on_max_train_change(tmp_path):
    stats = {"exp_bonds": np.array([1.0])}
    _save_exp_stats_split_cache(
        str(tmp_path), "/d.gro", 100, 50, 0, 0.8,
        stats, stats, max_train=500, max_test=100,
    )
    # Matching limits: hit
    train, test = _load_exp_stats_split_cache(
        str(tmp_path), "/d.gro", 100, 50, 0, 0.8,
        max_train=500, max_test=100,
    )
    assert train is not None and test is not None
    # Changed max_test: miss
    train2, test2 = _load_exp_stats_split_cache(
        str(tmp_path), "/d.gro", 100, 50, 0, 0.8,
        max_train=500, max_test=200,
    )
    assert train2 is None and test2 is None
