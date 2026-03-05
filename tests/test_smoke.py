"""
Smoke test: run the pipeline with a minimal config and the sphere dataset, then assert key outputs exist.

Slow (trains 1 DistMap + 1 Euclideanizer, minimal plots/analysis). Skip by default:
  pytest -m "not slow"
Run only smoke test:
  pytest tests/test_smoke.py -v
  pytest tests/test_smoke.py -v -m slow
"""
from __future__ import annotations

import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

DATA_GRO = os.path.join(_PIPELINE_ROOT, "tests", "test_data", "spheres.gro")
CONFIG_SMOKE = os.path.join(_TEST_DIR, "config_smoke.yaml")


@pytest.mark.slow
def test_pipeline_smoke_run(tmp_path):
    """Run pipeline with minimal config and sphere data; assert DistMap and Euclideanizer outputs exist."""
    if not os.path.isfile(DATA_GRO):
        pytest.skip(f"Smoke test requires {DATA_GRO!r} (run from pipeline root with tests/test_data/spheres.gro)")
    if not os.path.isfile(CONFIG_SMOKE):
        pytest.skip(f"Smoke config not found: {CONFIG_SMOKE!r}")

    output_dir = os.path.join(tmp_path, "smoke_out")
    argv_save = sys.argv
    cwd_save = os.getcwd()
    try:
        sys.argv = [
            "run.py",
            "--config", CONFIG_SMOKE,
            "--data", DATA_GRO,
            "--output-dir", output_dir,
            "--no-resume",
        ]
        os.chdir(_PIPELINE_ROOT)
        from run import main
        main()
    finally:
        sys.argv = argv_save
        os.chdir(cwd_save)

    # Key outputs: one seed, one DistMap run, one Euclideanizer run
    dm_pt = os.path.join(output_dir, "seed_0", "distmap", "0", "model", "model.pt")
    eu_pt = os.path.join(output_dir, "seed_0", "distmap", "0", "euclideanizer", "0", "model", "euclideanizer.pt")
    pipeline_log = os.path.join(output_dir, "pipeline.log")

    assert os.path.isfile(dm_pt), f"DistMap checkpoint missing: {dm_pt}"
    assert os.path.isfile(eu_pt), f"Euclideanizer checkpoint missing: {eu_pt}"
    assert os.path.isfile(pipeline_log), f"Pipeline log missing: {pipeline_log}"
