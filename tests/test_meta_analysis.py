"""Sufficiency meta-analysis: NPZ discovery uses test-set reconstruction arrays."""
from __future__ import annotations

import os
import sys

import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src import meta_analysis as meta_analysis_module


def test_load_recon_metric_arrays_from_flat_recon_data(tmp_path):
    seed_dir = tmp_path / "seed_1_split_0.8_maxdata_100"
    eu = seed_dir / "distmap" / "0" / "euclideanizer" / "0"
    rdir = eu / "analysis" / "rmsd" / "recon" / "data"
    qdir = eu / "analysis" / "q" / "recon" / "data"
    rdir.mkdir(parents=True)
    qdir.mkdir(parents=True)
    tr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tq = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    np.savez_compressed(rdir / "rmsd_recon_data.npz", test_recon_rmsd=tr)
    np.savez_compressed(qdir / "q_recon_data.npz", test_recon_q=tq)

    r_out, q_out = meta_analysis_module._load_recon_metric_arrays(str(seed_dir))
    assert r_out is not None and q_out is not None
    np.testing.assert_array_equal(r_out, tr)
    np.testing.assert_array_equal(q_out, tq)


def test_load_recon_metric_arrays_matching_subdir(tmp_path):
    seed_dir = tmp_path / "seed_2_split_0.9_maxdata_200"
    eu = seed_dir / "distmap" / "0" / "euclideanizer" / "0"
    sub = "train10_test10"
    rdir = eu / "analysis" / "rmsd" / "recon" / sub / "data"
    qdir = eu / "analysis" / "q" / "recon" / sub / "data"
    rdir.mkdir(parents=True)
    qdir.mkdir(parents=True)
    tr = np.array([0.1], dtype=np.float32)
    tq = np.array([0.9], dtype=np.float32)
    np.savez_compressed(rdir / "rmsd_recon_data.npz", test_recon_rmsd=tr)
    np.savez_compressed(qdir / "q_recon_data.npz", test_recon_q=tq)

    r_out, q_out = meta_analysis_module._load_recon_metric_arrays(str(seed_dir))
    assert r_out is not None and q_out is not None
    np.testing.assert_array_equal(r_out, tr)
    np.testing.assert_array_equal(q_out, tq)
