"""
Test that the pipeline's analysis call path would not fail due to missing kwargs.

run.py calls spec.run_gen_analysis(..., **pre_kw, **extra_kw) and
spec.run_recon_analysis(..., display_root=..., recon_subdir=..., **recon_extra).
This test builds the exact kwargs the pipeline would pass (from config + spec helpers)
and asserts every required keyword-only parameter of each run_* function is present.
If a new required argument is added to an analysis function but not to the
corresponding gen_extra_kwargs/recon_extra_kwargs, the test fails—catching the
bug before a long pipeline run.
"""
from __future__ import annotations

import inspect
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.config import load_config
from src.analysis_metrics import ANALYSIS_METRICS


def _required_kwonly_params(callable_fn):
    """Set of parameter names that are keyword-only and have no default."""
    sig = inspect.signature(callable_fn)
    out = set()
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty:
            out.add(name)
    return out


def _kwargs_gen_single(analysis_cfg, spec):
    """Kwargs run.py passes to spec.run_gen_analysis (single num_samples branch)."""
    pre_kw = spec.precomputed_kwargs(None, None, None)
    extra_kw = spec.gen_extra_kwargs(analysis_cfg)
    return {
        "num_samples": 10,
        "sample_variance": 1.0,
        "output_suffix": "",
        "display_root": None,
        **pre_kw,
        **extra_kw,
    }


def _kwargs_gen_multi(analysis_cfg, spec):
    """Kwargs run.py passes to spec.run_gen_analysis_multi."""
    pre_kw = spec.precomputed_kwargs(None, None, None)
    extra_kw = spec.gen_extra_kwargs(analysis_cfg)
    return {
        "num_samples_list": [10, 20],
        "sample_variance": 1.0,
        "variance_suffix": "",
        "display_root": None,
        **pre_kw,
        **extra_kw,
    }


def _kwargs_recon(analysis_cfg, spec):
    """Kwargs run.py passes to spec.run_recon_analysis."""
    recon_extra = spec.recon_extra_kwargs(analysis_cfg)
    return {
        "display_root": None,
        "recon_subdir": "",
        **recon_extra,
    }


def test_gen_analysis_kwargs_satisfy_signatures():
    """For every analysis metric, pipeline-supplied gen kwargs include all required keyword-only args."""
    cfg = load_config(os.path.join(_TEST_DIR, "config_test.yaml"))
    analysis_cfg = cfg["analysis"]
    missing_by_spec = []
    for spec in ANALYSIS_METRICS:
        required_single = _required_kwonly_params(spec.run_gen_analysis)
        required_multi = _required_kwonly_params(spec.run_gen_analysis_multi)
        kw_single = _kwargs_gen_single(analysis_cfg, spec)
        kw_multi = _kwargs_gen_multi(analysis_cfg, spec)
        miss_single = required_single - set(kw_single)
        miss_multi = required_multi - set(kw_multi)
        if miss_single or miss_multi:
            missing_by_spec.append(
                (spec.id, {"run_gen_analysis": miss_single, "run_gen_analysis_multi": miss_multi})
            )
    assert not missing_by_spec, (
        "Pipeline would pass incomplete kwargs to gen analysis; add missing keys to gen_extra_kwargs (or precomputed): "
        + str(missing_by_spec)
    )


def test_recon_analysis_kwargs_satisfy_signatures():
    """For every analysis metric, pipeline-supplied recon kwargs include all required keyword-only args."""
    cfg = load_config(os.path.join(_TEST_DIR, "config_test.yaml"))
    analysis_cfg = cfg["analysis"]
    missing_by_spec = []
    for spec in ANALYSIS_METRICS:
        required = _required_kwonly_params(spec.run_recon_analysis)
        kw = _kwargs_recon(analysis_cfg, spec)
        miss = required - set(kw)
        if miss:
            missing_by_spec.append((spec.id, miss))
    assert not missing_by_spec, (
        "Pipeline would pass incomplete kwargs to recon analysis; add missing keys to recon_extra_kwargs: "
        + str(missing_by_spec)
    )
