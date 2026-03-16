"""
Smoke test: run the full pipeline with a minimal config and the sphere dataset; assert that key outputs exist.

Included in default pytest runs (pytest tests/ -v). On one or zero GPUs: single-task (one seed).
On two or more GPUs: multi-task (two seeds) so both devices are used and the multi-GPU path is exercised.
Marked slow; to skip it for a quicker run: pytest -m "not slow".

  pytest tests/ -v        # run all tests including smoke (single-task on 1 GPU, multi-task on 2+ GPUs)
  pytest -m "not slow"    # run all tests except smoke
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

DATA_NPZ = os.path.join(_PIPELINE_ROOT, "tests", "test_data", "spheres.npz")
CONFIG_SMOKE = os.path.join(_TEST_DIR, "config_smoke.yaml")


def _smoke_test_num_gpus():
    """Number of CUDA devices available; 0 if CUDA not available."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


@pytest.mark.slow
def test_pipeline_smoke_run(tmp_path):
    """Run the full pipeline with minimal config and sphere data; assert DistMap and Euclideanizer outputs exist."""
    if not os.path.isfile(DATA_NPZ):
        pytest.skip(f"Smoke test requires {DATA_NPZ!r} (run from pipeline root: python tests/test_data/generate_spheres.py)")
    if not os.path.isfile(CONFIG_SMOKE):
        pytest.skip(f"Smoke config not found: {CONFIG_SMOKE!r}")

    n_gpus = _smoke_test_num_gpus()
    # One GPU: single task (one seed) for a faster run. Two+ GPUs: two seeds so multi-GPU path runs both tasks.
    # --yes-overwrite avoids any overwrite confirmation prompt (non-interactive pytest would hang on input()).
    output_dir = os.path.join(tmp_path, "smoke_out")
    if n_gpus >= 2:
        seeds_to_check = (0, 1)
        argv = ["run.py", "--config", CONFIG_SMOKE, "--data", DATA_NPZ, "--output-dir", output_dir, "--no-resume", "--yes-overwrite"]
    else:
        seeds_to_check = (0,)
        argv = ["run.py", "--config", CONFIG_SMOKE, "--data", DATA_NPZ, "--output-dir", output_dir, "--no-resume", "--yes-overwrite", "--data.split_seed", "0"]

    argv_save = sys.argv
    cwd_save = os.getcwd()
    try:
        sys.argv = argv
        os.chdir(_PIPELINE_ROOT)
        from run import main
        main()
    finally:
        sys.argv = argv_save
        os.chdir(cwd_save)

    pipeline_log = os.path.join(output_dir, "pipeline.log")
    assert os.path.isfile(pipeline_log), f"Pipeline log missing: {pipeline_log}"
    for seed in seeds_to_check:
        dm_pt = os.path.join(output_dir, f"seed_{seed}", "distmap", "0", "model", "model.pt")
        eu_pt = os.path.join(output_dir, f"seed_{seed}", "distmap", "0", "euclideanizer", "0", "model", "euclideanizer.pt")
        assert os.path.isfile(dm_pt), f"DistMap checkpoint missing: {dm_pt}"
        assert os.path.isfile(eu_pt), f"Euclideanizer checkpoint missing: {eu_pt}"

    dashboard_dir = os.path.join(output_dir, "dashboard")
    assert os.path.isdir(dashboard_dir), f"Dashboard dir missing: {dashboard_dir}"
    assert os.path.isfile(os.path.join(dashboard_dir, "index.html")), "dashboard/index.html missing"
    assert os.path.isfile(os.path.join(dashboard_dir, "manifest.json")), "dashboard/manifest.json missing"
    assert os.path.isdir(os.path.join(dashboard_dir, "assets")), "dashboard/assets/ missing"
    with open(os.path.join(dashboard_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    assert "runs" in manifest, "manifest.json must contain 'runs'"
    assert len(manifest["runs"]) >= 1, "manifest should have at least one run"
