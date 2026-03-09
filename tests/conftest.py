# Pytest conftest: ensure pipeline root is on path when running tests
import os
import sys

import numpy as np
import pytest

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)


def assert_exact_or_numerical(actual, expected, atol, name="value"):
    """
    Require exact equality. If not exact but within atol, pass and print a note
    that the difference is likely numerical error; otherwise fail.
    """
    actual = np.asarray(actual)
    if actual.shape == ():
        actual = actual.item()
    if np.isscalar(actual):
        exact = actual == expected
        within = np.isclose(actual, expected, atol=atol)
    else:
        exact = (actual == expected).all()
        within = np.allclose(actual, expected, atol=atol)
    if exact:
        return
    if within:
        print(f"\nNote: {name} not exactly {expected} (got {actual}); within atol={atol}, likely numerical error.")
        return
    pytest.fail(f"{name} = {actual}, expected exactly {expected} (outside atol={atol})")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (e.g. smoke run); included by default, skip with -m 'not slow'")
