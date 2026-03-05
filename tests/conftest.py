# Pytest conftest: ensure pipeline root is on path when running tests
import os
import sys

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (e.g. smoke run); included by default, skip with -m 'not slow'")
