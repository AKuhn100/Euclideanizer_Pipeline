from __future__ import annotations

import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.generative_capacity import build_nested_subsample_indices


def test_build_nested_subsample_indices_is_nested():
    n_values = [50, 100, 250]
    nested = build_nested_subsample_indices(250, n_values, seed=7)
    assert len(nested[50]) == 50
    assert len(nested[100]) == 100
    assert len(nested[250]) == 250
    # Monotonic/nested subset property: smaller n are prefixes of larger n.
    assert list(nested[100][:50]) == list(nested[50])
    assert list(nested[250][:100]) == list(nested[100])

