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
from src.generative_capacity import _distribution_panel, build_nested_subsample_indices


def test_build_nested_subsample_indices_is_nested():
    n_values = [50, 100, 250]
    nested = build_nested_subsample_indices(250, n_values, seed=7)
    assert len(nested[50]) == 50
    assert len(nested[100]) == 100
    assert len(nested[250]) == 250
    # Monotonic/nested subset property: smaller n are prefixes of larger n.
    assert list(nested[100][:50]) == list(nested[50])
    assert list(nested[250][:100]) == list(nested[100])


def test_distribution_panel_histogram_degenerate_values_no_crash():
    """Histogram panel must not require KDE (constant or tiny samples)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    by_n = {
        10: np.ones(15, dtype=np.float32),
        20: np.concatenate([np.ones(10), np.ones(5) * 1.0001]).astype(np.float32),
    }
    _distribution_panel(ax, by_n, "Test quantity", n_bins=HIST_BINS_DEFAULT)
    plt.close(fig)

