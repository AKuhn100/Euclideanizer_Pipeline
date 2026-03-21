from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.plot_config import HIST_BINS_DEFAULT
from src.generative_capacity import (
    _distribution_panel,
    _write_pairwise_matrix_npz_and_remove_npy,
    build_nested_subsample_indices,
)


def test_build_nested_subsample_indices_is_nested():
    n_values = [50, 100, 250]
    nested = build_nested_subsample_indices(250, n_values, seed=7)
    assert len(nested[50]) == 50
    assert len(nested[100]) == 100
    assert len(nested[250]) == 250
    # Monotonic/nested subset property: smaller n are prefixes of larger n.
    assert list(nested[100][:50]) == list(nested[50])
    assert list(nested[250][:100]) == list(nested[100])


def test_write_pairwise_matrix_npz_replaces_npy(tmp_path):
    npy = tmp_path / "pairwise_matrix.npy"
    npz = tmp_path / "pairwise_matrix.npz"
    np.save(str(npy), np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    _write_pairwise_matrix_npz_and_remove_npy(
        matrix_npy_path=str(npy),
        npz_path=str(npz),
        n_max=2,
        seed=42,
        n_values=[2],
        metric="rmsd",
    )
    assert not npy.is_file()
    assert npz.is_file()
    z = np.load(str(npz), allow_pickle=False)
    assert z["pairwise"].shape == (2, 2)
    assert int(z["n_max"]) == 2
    assert int(z["seed"]) == 42
    assert z["metric"].tobytes() == b"rmsd"
    z.close()


def test_write_pairwise_matrix_npz_includes_delta_for_q(tmp_path):
    npy = tmp_path / "m.npy"
    npz = tmp_path / "m.npz"
    np.save(str(npy), np.eye(3, dtype=np.float32))
    _write_pairwise_matrix_npz_and_remove_npy(
        matrix_npy_path=str(npy),
        npz_path=str(npz),
        n_max=3,
        seed=1,
        n_values=[1, 3],
        metric="q",
        delta=0.5,
    )
    z = np.load(str(npz), allow_pickle=False)
    assert float(z["delta"]) == 0.5
    assert z["metric"].tobytes() == b"q"
    z.close()


def test_distribution_panel_histogram_degenerate_values_no_crash():
    """Histogram panel must not require KDE (constant or tiny samples)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    by_n = {
        10: np.ones(15, dtype=np.float32),
        20: np.concatenate([np.ones(10), np.ones(5) * 1.0001]).astype(np.float32),
    }
    _distribution_panel(ax, by_n, "Test quantity", n_bins=HIST_BINS_DEFAULT)
    plt.close(fig)

