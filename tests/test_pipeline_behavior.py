"""
Pipeline behavior tests: run completion, need_data / data needs, resume logic, config match, and skip logic.

These tests exercise the pipeline's control flow and resume decisions without running training,
plotting, or analysis. They use a minimal config (tests/config_test.yaml), fake checkpoint
dirs (tmp_path), and the same helpers the main loop uses (_run_completed, _pipeline_need_data,
_pipeline_data_needs, PipelineDataNeeds, _distmap_training_action, _euclideanizer_training_action, etc.).

Sections:
  - Run completion: when a run is considered "complete" (best checkpoint, last_epoch_trained,
    optional last checkpoint for multi-segment).
  - need_data: when the pipeline must load something vs can skip (all runs complete, no
    missing plots/analysis). _pipeline_need_data() is need_any() from _pipeline_data_needs().
  - Data needs (structured): need_coords (training, reconstruction, recon_statistics, rmsd),
    need_exp_stats (gen_variance), need_train_test_stats (recon_statistics, gen_variance).
  - Resume logic (DistMap / Euclideanizer): skip, from_scratch, resume_from_best (first or
    later segment), resume_from_prev_last.
  - Config: load config_test.yaml, produce training groups; config mismatch raises before
    data load.
  - Plotting / analysis all-present: skip loading models when all plot/analysis outputs exist.

Run from the pipeline root:
  pytest tests/test_pipeline_behavior.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.config import (
    load_config,
    distmap_training_groups,
    euclideanizer_training_groups,
    save_run_config,
    save_pipeline_config,
    configs_match_exactly,
    config_diff,
    load_pipeline_config,
)
from run import (
    _run_completed,
    _pipeline_need_data,
    _pipeline_data_needs,
    PipelineDataNeeds,
    _distmap_plotting_all_present,
    _euclideanizer_analysis_all_present,
    _distmap_training_action,
    _euclideanizer_training_action,
    _dm_path,
    _dm_path_last,
    _eu_path,
    _eu_path_last,
    _reference_size_config,
    _reference_size_changed,
    _delete_reference_size_caches,
    EXP_STATS_CACHE_DIR,
    EXP_STATS_SPLIT_META,
    EXP_STATS_TRAIN_NPZ,
    EXP_STATS_TEST_NPZ,
)


def _load_test_config():
    """Load tests/config_test.yaml; used by tests that need a valid config and training groups."""
    path = os.path.join(_TEST_DIR, "config_test.yaml")
    return load_config(path=path)


def _get_dm_eu_groups(cfg):
    """Return (distmap_training_groups(cfg), euclideanizer_training_groups(cfg))."""
    return distmap_training_groups(cfg), euclideanizer_training_groups(cfg)


@pytest.fixture
def cfg():
    """Loaded tests/config_test.yaml for tests that need a valid config and training groups."""
    return _load_test_config()


# ---------------------------------------------------------------------------
# Run completion
# ---------------------------------------------------------------------------


def test_run_completed_requires_best_checkpoint(tmp_path):
    """Run is not complete without the best checkpoint file (model.pt / euclideanizer.pt)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_run_config({"distmap": {"epochs": 1}}, str(model_dir), last_epoch_trained=1, best_epoch=1)
    assert _run_completed(str(tmp_path), 1, section_key="distmap", expected_section=None, multi_segment=False) is False
    (model_dir / "model.pt").write_bytes(b"x")
    assert _run_completed(str(tmp_path), 1, section_key="distmap", expected_section=None, multi_segment=False) is True


def test_run_completed_without_section_match(tmp_path):
    """Completion can be checked without requiring section match (expected_section=None)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_run_config({"distmap": {"epochs": 2}}, str(model_dir), last_epoch_trained=2, best_epoch=1, best_val=0.5)
    (model_dir / "model.pt").write_bytes(b"x")
    assert _run_completed(str(tmp_path), 2, section_key="distmap", expected_section=None, multi_segment=False) is True


def test_run_completed_multi_segment_last_optional_on_final_stretch(tmp_path):
    """Multi-segment: last checkpoint required only when save_final_models_per_stretch (when false we delete it so don't require)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_run_config({"distmap": {"epochs": 1}}, str(model_dir), last_epoch_trained=1, best_epoch=1, best_val=0.5)
    (model_dir / "model.pt").write_bytes(b"x")
    # save_final_models_per_stretch=False: we never require model_last.pt (it is deleted after next segment uses it)
    assert _run_completed(str(tmp_path), 1, section_key="distmap", expected_section=None, multi_segment=True,
        checkpoint_last_name="model_last.pt", is_last_segment=False, save_final_models_per_stretch=False) is True
    assert _run_completed(str(tmp_path), 1, section_key="distmap", expected_section=None, multi_segment=True,
        checkpoint_last_name="model_last.pt", is_last_segment=True, save_final_models_per_stretch=False) is True
    # save_final_models_per_stretch=True: we require model_last.pt for non-last segment
    assert _run_completed(str(tmp_path), 1, section_key="distmap", expected_section=None, multi_segment=True,
        checkpoint_last_name="model_last.pt", is_last_segment=False, save_final_models_per_stretch=True) is False
    (model_dir / "model_last.pt").write_bytes(b"x")
    assert _run_completed(str(tmp_path), 1, section_key="distmap", expected_section=None, multi_segment=True,
        checkpoint_last_name="model_last.pt", is_last_segment=False, save_final_models_per_stretch=True) is True


# ---------------------------------------------------------------------------
# need_data (when must the pipeline load the dataset?)
# ---------------------------------------------------------------------------


def test_pipeline_need_data_true_when_seed_dir_missing(tmp_path, cfg):
    """need_data is True when the seed output directory does not exist."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    assert _pipeline_need_data(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=True, do_rmsd=True,
        do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True,
        plot_variances=[1.0], variance_list=[1.0], num_samples_list=[10]) is True


def test_pipeline_need_data_true_when_run_incomplete(tmp_path, cfg):
    """need_data is True when any run (e.g. distmap/1) is incomplete (last_epoch_trained != target)."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    (tmp_path / "seed_0").mkdir()
    for ri, ev in [(0, 1), (1, 2)]:
        d = tmp_path / "seed_0" / "distmap" / str(ri) / "model"
        d.mkdir(parents=True)
        save_run_config({"distmap": {"epochs": ev}}, str(d), last_epoch_trained=ev if ri == 0 else 0, best_epoch=1 if ri == 0 else None, best_val=0.5 if ri == 0 else None)
        (d / "model.pt").write_bytes(b"x")
        if ev == 1:
            (d / "model_last.pt").write_bytes(b"x")
    for ri in (0, 1):
        for euri, eu_ev in [(0, 1), (1, 2)]:
            d = tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model"
            d.mkdir(parents=True)
            save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(d), last_epoch_trained=eu_ev, best_epoch=1, best_val=0.5)
            (d / "euclideanizer.pt").write_bytes(b"x")
            (d / "euclideanizer_last.pt").write_bytes(b"x")
    assert _pipeline_need_data(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=False, do_rmsd=False,
        do_recon_plot=False, do_bond_rg_scaling=False, do_avg_gen=False,
        plot_variances=[], variance_list=[], num_samples_list=[]) is True


def test_pipeline_need_data_false_when_all_complete_no_plot_rmsd(tmp_path, cfg):
    """need_data is False when all runs are complete and plotting/rmsd are disabled."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    (tmp_path / "seed_0").mkdir()
    for ri, ev in [(0, 1), (1, 2)]:
        d = tmp_path / "seed_0" / "distmap" / str(ri) / "model"
        d.mkdir(parents=True)
        save_run_config({"distmap": {"epochs": ev}}, str(d), last_epoch_trained=ev, best_epoch=1, best_val=0.5)
        (d / "model.pt").write_bytes(b"x")
        if ev == 1:
            (d / "model_last.pt").write_bytes(b"x")
    for ri in (0, 1):
        for euri, eu_ev in [(0, 1), (1, 2)]:
            d = tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model"
            d.mkdir(parents=True)
            save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(d), last_epoch_trained=eu_ev, best_epoch=1, best_val=0.5)
            (d / "euclideanizer.pt").write_bytes(b"x")
            if eu_ev == 1:
                (d / "euclideanizer_last.pt").write_bytes(b"x")
    assert _pipeline_need_data(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=False, do_rmsd=False,
        do_recon_plot=False, do_bond_rg_scaling=False, do_avg_gen=False,
        plot_variances=[], variance_list=[], num_samples_list=[]) is False


def test_pipeline_need_data_true_when_plot_missing(tmp_path, cfg):
    """need_data is True when plotting is enabled but a plot file is missing."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    (tmp_path / "seed_0").mkdir()
    for ri, ev in [(0, 1), (1, 2)]:
        d = tmp_path / "seed_0" / "distmap" / str(ri) / "model"
        d.mkdir(parents=True)
        save_run_config({"distmap": {"epochs": ev}}, str(d), last_epoch_trained=ev, best_epoch=1, best_val=0.5)
        (d / "model.pt").write_bytes(b"x")
        if ev == 1:
            (d / "model_last.pt").write_bytes(b"x")
    for ri in (0, 1):
        for euri, eu_ev in [(0, 1), (1, 2)]:
            d = tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model"
            d.mkdir(parents=True)
            save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(d), last_epoch_trained=eu_ev, best_epoch=1, best_val=0.5)
            (d / "euclideanizer.pt").write_bytes(b"x")
            if eu_ev == 1:
                (d / "euclideanizer_last.pt").write_bytes(b"x")
    assert _pipeline_need_data(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=True, do_rmsd=False,
        do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True,
        plot_variances=[1.0], variance_list=[], num_samples_list=[]) is True


def _all_runs_complete_layout(tmp_path, cfg):
    """Create seed_0 with all DistMap and Euclideanizer runs complete; return dm_groups, eu_groups."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    (tmp_path / "seed_0").mkdir()
    for ri, ev in [(0, 1), (1, 2)]:
        d = tmp_path / "seed_0" / "distmap" / str(ri) / "model"
        d.mkdir(parents=True)
        save_run_config({"distmap": {"epochs": ev}}, str(d), last_epoch_trained=ev, best_epoch=1, best_val=0.5)
        (d / "model.pt").write_bytes(b"x")
        if ev == 1:
            (d / "model_last.pt").write_bytes(b"x")
    for ri in (0, 1):
        for euri, eu_ev in [(0, 1), (1, 2)]:
            d = tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model"
            d.mkdir(parents=True)
            save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(d), last_epoch_trained=eu_ev, best_epoch=1, best_val=0.5)
            (d / "euclideanizer.pt").write_bytes(b"x")
            if eu_ev == 1:
                (d / "euclideanizer_last.pt").write_bytes(b"x")
    return dm_groups, eu_groups


def test_pipeline_data_needs_need_any_matches_need_data(tmp_path, cfg):
    """_pipeline_data_needs(...).need_any() equals _pipeline_need_data(...) for the same args."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    (tmp_path / "seed_0").mkdir()
    for ri in (0, 1):
        (tmp_path / "seed_0" / "distmap" / str(ri) / "model").mkdir(parents=True)
        save_run_config({"distmap": {"epochs": 1}}, str(tmp_path / "seed_0" / "distmap" / str(ri) / "model"), last_epoch_trained=1, best_epoch=1, best_val=0.5)
        (tmp_path / "seed_0" / "distmap" / str(ri) / "model" / "model.pt").write_bytes(b"x")
        (tmp_path / "seed_0" / "distmap" / str(ri) / "model" / "model_last.pt").write_bytes(b"x")
    for ri in (0, 1):
        for euri in (0, 1):
            (tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model").mkdir(parents=True)
            save_run_config({"euclideanizer": {"epochs": 1}}, str(tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model"), last_epoch_trained=1, best_epoch=1, best_val=0.5)
            (tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model" / "euclideanizer.pt").write_bytes(b"x")
            (tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model" / "euclideanizer_last.pt").write_bytes(b"x")
    base = str(tmp_path)
    seeds = [0]
    kwargs = dict(resume=True, do_plot=True, do_rmsd=True, do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True, plot_variances=[1.0], variance_list=[1.0], num_samples_list=[10])
    assert _pipeline_data_needs(base, seeds, dm_groups, eu_groups, **kwargs).need_any() is _pipeline_need_data(base, seeds, dm_groups, eu_groups, **kwargs)


def test_pipeline_data_needs_only_reconstruction_missing(tmp_path, cfg):
    """When only reconstruction plot is missing, need_coords is True, need_exp_stats is False."""
    dm_groups, eu_groups = _all_runs_complete_layout(tmp_path, cfg)
    run_dir_dm = tmp_path / "seed_0" / "distmap" / "0"
    (run_dir_dm / "plots" / "recon_statistics").mkdir(parents=True)
    (run_dir_dm / "plots" / "recon_statistics" / "recon_statistics_train.png").write_bytes(b"x")
    (run_dir_dm / "plots" / "recon_statistics" / "recon_statistics_test.png").write_bytes(b"x")
    (run_dir_dm / "plots" / "gen_variance").mkdir(parents=True)
    (run_dir_dm / "plots" / "gen_variance" / "gen_variance_1.0.png").write_bytes(b"x")
    for ri in (0, 1):
        for euri in (0, 1):
            eu_run = tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri)
            (eu_run / "plots" / "reconstruction").mkdir(parents=True)
            (eu_run / "plots" / "reconstruction" / "reconstruction.png").write_bytes(b"x")
            (eu_run / "plots" / "recon_statistics").mkdir(parents=True)
            (eu_run / "plots" / "recon_statistics" / "recon_statistics_train.png").write_bytes(b"x")
            (eu_run / "plots" / "recon_statistics" / "recon_statistics_test.png").write_bytes(b"x")
            (eu_run / "plots" / "gen_variance").mkdir(parents=True)
            (eu_run / "plots" / "gen_variance" / "gen_variance_1.0.png").write_bytes(b"x")
    run_dir_dm_1 = tmp_path / "seed_0" / "distmap" / "1"
    (run_dir_dm_1 / "plots" / "recon_statistics").mkdir(parents=True)
    (run_dir_dm_1 / "plots" / "recon_statistics" / "recon_statistics_train.png").write_bytes(b"x")
    (run_dir_dm_1 / "plots" / "recon_statistics" / "recon_statistics_test.png").write_bytes(b"x")
    (run_dir_dm_1 / "plots" / "gen_variance").mkdir(parents=True)
    (run_dir_dm_1 / "plots" / "gen_variance" / "gen_variance_1.0.png").write_bytes(b"x")
    needs = _pipeline_data_needs(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=True, do_rmsd=False,
        do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True,
        plot_variances=[1.0], variance_list=[], num_samples_list=[])
    assert needs.need_coords is True
    assert needs.need_exp_stats is False
    assert needs.need_train_test_stats is False


def test_pipeline_data_needs_only_gen_variance_missing(tmp_path, cfg):
    """When only gen_variance plot is missing, need_coords is False, need_exp_stats is True."""
    dm_groups, eu_groups = _all_runs_complete_layout(tmp_path, cfg)
    for ri in (0, 1):
        run_dir_dm = tmp_path / "seed_0" / "distmap" / str(ri)
        (run_dir_dm / "plots" / "reconstruction").mkdir(parents=True)
        (run_dir_dm / "plots" / "reconstruction" / "reconstruction.png").write_bytes(b"x")
        (run_dir_dm / "plots" / "recon_statistics").mkdir(parents=True)
        (run_dir_dm / "plots" / "recon_statistics" / "recon_statistics_train.png").write_bytes(b"x")
        (run_dir_dm / "plots" / "recon_statistics" / "recon_statistics_test.png").write_bytes(b"x")
        for euri in (0, 1):
            eu_run = tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri)
            (eu_run / "plots" / "reconstruction").mkdir(parents=True)
            (eu_run / "plots" / "reconstruction" / "reconstruction.png").write_bytes(b"x")
            (eu_run / "plots" / "recon_statistics").mkdir(parents=True)
            (eu_run / "plots" / "recon_statistics" / "recon_statistics_train.png").write_bytes(b"x")
            (eu_run / "plots" / "recon_statistics" / "recon_statistics_test.png").write_bytes(b"x")
    needs = _pipeline_data_needs(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=True, do_rmsd=False,
        do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True,
        plot_variances=[1.0], variance_list=[], num_samples_list=[])
    assert needs.need_coords is False
    assert needs.need_exp_stats is True
    assert needs.need_train_test_stats is True


def test_pipeline_data_needs_run_incomplete_sets_need_coords(tmp_path, cfg):
    """When any run is incomplete, need_coords is True."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    (tmp_path / "seed_0").mkdir()
    d = tmp_path / "seed_0" / "distmap" / "0" / "model"
    d.mkdir(parents=True)
    save_run_config({"distmap": {"epochs": 1}}, str(d), last_epoch_trained=0, best_epoch=0, best_val=0.5)
    (d / "model.pt").write_bytes(b"x")
    for euri in (0, 1):
        ed = tmp_path / "seed_0" / "distmap" / "0" / "euclideanizer" / str(euri) / "model"
        ed.mkdir(parents=True)
        save_run_config({"euclideanizer": {"epochs": 1}}, str(ed), last_epoch_trained=1, best_epoch=1, best_val=0.5)
        (ed / "euclideanizer.pt").write_bytes(b"x")
        (ed / "euclideanizer_last.pt").write_bytes(b"x")
    d1 = tmp_path / "seed_0" / "distmap" / "1" / "model"
    d1.mkdir(parents=True)
    save_run_config({"distmap": {"epochs": 2}}, str(d1), last_epoch_trained=2, best_epoch=1, best_val=0.5)
    (d1 / "model.pt").write_bytes(b"x")
    (d1 / "model_last.pt").write_bytes(b"x")
    for euri in (0, 1):
        ed = tmp_path / "seed_0" / "distmap" / "1" / "euclideanizer" / str(euri) / "model"
        ed.mkdir(parents=True)
        save_run_config({"euclideanizer": {"epochs": 1}}, str(ed), last_epoch_trained=1, best_epoch=1, best_val=0.5)
        (ed / "euclideanizer.pt").write_bytes(b"x")
        (ed / "euclideanizer_last.pt").write_bytes(b"x")
    needs = _pipeline_data_needs(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=False, do_rmsd=False,
        do_recon_plot=False, do_bond_rg_scaling=False, do_avg_gen=False,
        plot_variances=[], variance_list=[], num_samples_list=[])
    assert needs.need_coords is True
    assert needs.need_exp_stats is False
    assert needs.need_train_test_stats is False


# ---------------------------------------------------------------------------
# Plotting / analysis "all present" (skip loading model when outputs exist)
# ---------------------------------------------------------------------------


def test_distmap_plotting_all_present_resume_false(tmp_path):
    """When resume=False, plotting is never considered 'all present' (we always regenerate)."""
    assert _distmap_plotting_all_present(str(tmp_path), resume=False, do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True, sample_variances=[1.0]) is False


def test_euclideanizer_analysis_all_present_no_rmsd():
    """When rmsd is disabled, analysis is considered all present (nothing to generate)."""
    assert _euclideanizer_analysis_all_present("/any", resume=True, do_rmsd=False, variance_list=[1.0], num_samples_list=[10]) is True


# ---------------------------------------------------------------------------
# Config load and mismatch (fail fast before data load)
# ---------------------------------------------------------------------------


def test_pipeline_config_mismatch_raises(tmp_path, cfg):
    """Resume with existing output dir and differing config must raise RuntimeError (before loading data)."""
    output_dir = tmp_path / "seed_0"
    output_dir.mkdir()
    saved = {**cfg, "output_dir": str(output_dir), "data": {**cfg["data"], "split_seed": 0}}
    save_pipeline_config(saved, str(output_dir))
    current = {**saved, "resume": False}
    assert not configs_match_exactly(load_pipeline_config(str(output_dir)), current)
    with pytest.raises(RuntimeError, match="does not match current config"):
        sc = load_pipeline_config(str(output_dir))
        if sc is None or not configs_match_exactly(sc, current):
            raise RuntimeError("Resume is enabled but pipeline config in output_dir does not match current config.")


def test_config_test_yaml_loads_and_produces_groups(cfg):
    """config_test.yaml must load and yield at least one DistMap and one Euclideanizer group with checkpoints."""
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    assert len(dm_groups) >= 1 and len(eu_groups) >= 1
    for g in dm_groups:
        assert "checkpoints" in g and len(g["checkpoints"]) >= 1
    for g in eu_groups:
        assert "checkpoints" in g and len(g["checkpoints"]) >= 1


# ---------------------------------------------------------------------------
# Resume logic: DistMap (skip, from_scratch, resume_from_best, resume_from_prev_last)
# ---------------------------------------------------------------------------

def test_resume_distmap_skip_when_run_complete(tmp_path):
    """Completed run (best + last_epoch_trained, optional last) → action skip."""
    run_dir = tmp_path / "distmap" / "0"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    ev = 100
    save_run_config({"distmap": {"epochs": ev}}, str(model_dir), last_epoch_trained=ev, best_epoch=80, best_val=0.5)
    dm_cfg = {"epochs": ev}
    (model_dir / "model.pt").write_bytes(b"x")
    act = _distmap_training_action(
        str(run_dir), ev, dm_cfg,
        prev_dm_path=None, prev_dm_ev=None, prev_run_dir_dm=None,
        resume=True, dm_multi=False, dm_last_segment=True, dm_save_final=False,
    )
    assert act["action"] == "skip"


def test_resume_distmap_first_segment_from_scratch_no_checkpoint(tmp_path):
    """First segment, no best checkpoint → from_scratch."""
    run_dir = tmp_path / "distmap" / "0"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    ev = 100
    dm_cfg = {"epochs": ev}
    act = _distmap_training_action(
        str(run_dir), ev, dm_cfg,
        prev_dm_path=None, prev_dm_ev=None, prev_run_dir_dm=None,
        resume=True, dm_multi=False, dm_last_segment=True, dm_save_final=False,
    )
    assert act["action"] == "from_scratch"


def test_resume_distmap_first_segment_from_scratch_resume_false(tmp_path):
    """First segment, best exists but resume=False → from_scratch."""
    run_dir = tmp_path / "distmap" / "0"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    ev = 100
    dm_cfg = {"epochs": ev}
    save_run_config({"distmap": {"epochs": ev}}, str(model_dir), last_epoch_trained=50, best_epoch=50, best_val=0.5)
    (model_dir / "model.pt").write_bytes(b"x")
    act = _distmap_training_action(
        str(run_dir), ev, dm_cfg,
        prev_dm_path=None, prev_dm_ev=None, prev_run_dir_dm=None,
        resume=False, dm_multi=False, dm_last_segment=True, dm_save_final=False,
    )
    assert act["action"] == "from_scratch"


def test_resume_distmap_first_segment_resume_from_best(tmp_path):
    """First segment, interrupted (best at 50, target 100) → resume_from_best, 50 more epochs."""
    run_dir = tmp_path / "distmap" / "0"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    ev = 100
    dm_cfg = {"epochs": ev}
    save_run_config({"distmap": {"epochs": ev}}, str(model_dir), last_epoch_trained=75, best_epoch=50, best_val=0.5)
    (model_dir / "model.pt").write_bytes(b"x")
    act = _distmap_training_action(
        str(run_dir), ev, dm_cfg,
        prev_dm_path=None, prev_dm_ev=None, prev_run_dir_dm=None,
        resume=True, dm_multi=False, dm_last_segment=True, dm_save_final=False,
    )
    assert act["action"] == "resume_from_best"
    assert act["additional_epochs"] == 50
    assert act["resume_from_path"] == _dm_path(str(run_dir))
    assert act["prev_run_dir"] == str(run_dir)
    assert act["best_epoch"] == 50


def test_resume_distmap_later_segment_resume_from_prev_last(tmp_path):
    """Later segment, prev complete at 100, current run empty → resume_from_prev_last, 100 more epochs."""
    prev_run_dir = tmp_path / "distmap" / "0"
    prev_model_dir = prev_run_dir / "model"
    prev_model_dir.mkdir(parents=True)
    (prev_model_dir / "model.pt").write_bytes(b"x")
    (prev_model_dir / "model_last.pt").write_bytes(b"x")
    run_dir = tmp_path / "distmap" / "1"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    ev = 200
    dm_cfg = {"epochs": ev}
    prev_ev = 100
    prev_dm_path = _dm_path(str(prev_run_dir))
    act = _distmap_training_action(
        str(run_dir), ev, dm_cfg,
        prev_dm_path=prev_dm_path, prev_dm_ev=prev_ev, prev_run_dir_dm=str(prev_run_dir),
        resume=True, dm_multi=True, dm_last_segment=False, dm_save_final=False,
    )
    assert act["action"] == "resume_from_prev_last"
    assert act["additional_epochs"] == 100
    assert act["resume_from_path"] == _dm_path_last(str(prev_run_dir))
    assert act["prev_run_dir"] == str(prev_run_dir)


def test_resume_distmap_later_segment_resume_from_best(tmp_path):
    """Later segment, interrupted (prev ended at 100, current best at 150, target 300) → resume_from_best, 150 more."""
    prev_run_dir = tmp_path / "distmap" / "0"
    prev_model_dir = prev_run_dir / "model"
    prev_model_dir.mkdir(parents=True)
    (prev_model_dir / "model.pt").write_bytes(b"x")
    (prev_model_dir / "model_last.pt").write_bytes(b"x")
    run_dir = tmp_path / "distmap" / "1"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    ev = 300
    dm_cfg = {"epochs": ev}
    save_run_config({"distmap": {"epochs": ev}}, str(model_dir), last_epoch_trained=250, best_epoch=150, best_val=0.5)
    (model_dir / "model.pt").write_bytes(b"x")
    prev_ev = 100
    prev_dm_path = _dm_path(str(prev_run_dir))
    act = _distmap_training_action(
        str(run_dir), ev, dm_cfg,
        prev_dm_path=prev_dm_path, prev_dm_ev=prev_ev, prev_run_dir_dm=str(prev_run_dir),
        resume=True, dm_multi=True, dm_last_segment=True, dm_save_final=False,
    )
    assert act["action"] == "resume_from_best"
    assert act["additional_epochs"] == 150
    assert act["resume_from_path"] == _dm_path(str(run_dir))
    assert act["prev_run_dir"] == str(run_dir)
    assert act["best_epoch"] == 150


# ---------------------------------------------------------------------------
# Resume logic: Euclideanizer (same scenarios as DistMap)
# ---------------------------------------------------------------------------

def test_resume_euclideanizer_skip_when_run_complete(tmp_path):
    """Completed Euclideanizer run → action skip."""
    run_dir = tmp_path / "euclideanizer" / "0"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    eu_ev = 100
    eu_cfg = {"epochs": eu_ev}
    save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(model_dir), last_epoch_trained=eu_ev, best_epoch=80, best_val=0.5)
    (model_dir / "euclideanizer.pt").write_bytes(b"x")
    act = _euclideanizer_training_action(
        str(run_dir), eu_ev, eu_cfg,
        prev_eu_path=None, prev_eu_ev=None, prev_eu_run_dir=None,
        resume=True, eu_multi=False, eu_last_segment=True, eu_save_final=False,
    )
    assert act["action"] == "skip"


def test_resume_euclideanizer_first_segment_from_scratch(tmp_path):
    """First Euclideanizer segment, no checkpoint → from_scratch."""
    run_dir = tmp_path / "euclideanizer" / "0"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    eu_ev = 100
    eu_cfg = {"epochs": eu_ev}
    act = _euclideanizer_training_action(
        str(run_dir), eu_ev, eu_cfg,
        prev_eu_path=None, prev_eu_ev=None, prev_eu_run_dir=None,
        resume=True, eu_multi=False, eu_last_segment=True, eu_save_final=False,
    )
    assert act["action"] == "from_scratch"


def test_resume_euclideanizer_first_segment_resume_from_best(tmp_path):
    """First Euclideanizer segment, interrupted (best 50, target 100) → resume_from_best."""
    run_dir = tmp_path / "euclideanizer" / "0"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    eu_ev = 100
    eu_cfg = {"epochs": eu_ev}
    save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(model_dir), last_epoch_trained=75, best_epoch=50, best_val=0.5)
    (model_dir / "euclideanizer.pt").write_bytes(b"x")
    act = _euclideanizer_training_action(
        str(run_dir), eu_ev, eu_cfg,
        prev_eu_path=None, prev_eu_ev=None, prev_eu_run_dir=None,
        resume=True, eu_multi=False, eu_last_segment=True, eu_save_final=False,
    )
    assert act["action"] == "resume_from_best"
    assert act["additional_epochs"] == 50
    assert act["resume_from_path"] == _eu_path(str(run_dir))
    assert act["prev_run_dir"] == str(run_dir)
    assert act["best_epoch"] == 50


def test_resume_euclideanizer_later_segment_resume_from_prev_last(tmp_path):
    """Later Euclideanizer segment, prev complete → resume_from_prev_last."""
    prev_run_dir = tmp_path / "euclideanizer" / "0"
    prev_model_dir = prev_run_dir / "model"
    prev_model_dir.mkdir(parents=True)
    (prev_model_dir / "euclideanizer.pt").write_bytes(b"x")
    (prev_model_dir / "euclideanizer_last.pt").write_bytes(b"x")
    run_dir = tmp_path / "euclideanizer" / "1"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    eu_ev = 200
    eu_cfg = {"epochs": eu_ev}
    prev_ev = 100
    prev_eu_path = _eu_path(str(prev_run_dir))
    act = _euclideanizer_training_action(
        str(run_dir), eu_ev, eu_cfg,
        prev_eu_path=prev_eu_path, prev_eu_ev=prev_ev, prev_eu_run_dir=str(prev_run_dir),
        resume=True, eu_multi=True, eu_last_segment=False, eu_save_final=False,
    )
    assert act["action"] == "resume_from_prev_last"
    assert act["additional_epochs"] == 100
    assert act["resume_from_path"] == _eu_path_last(str(prev_run_dir))
    assert act["prev_run_dir"] == str(prev_run_dir)


def test_resume_euclideanizer_later_segment_resume_from_best(tmp_path):
    """Later Euclideanizer segment, interrupted (prev 100, current best 150, target 300) → resume_from_best."""
    prev_run_dir = tmp_path / "euclideanizer" / "0"
    prev_model_dir = prev_run_dir / "model"
    prev_model_dir.mkdir(parents=True)
    (prev_model_dir / "euclideanizer.pt").write_bytes(b"x")
    (prev_model_dir / "euclideanizer_last.pt").write_bytes(b"x")
    run_dir = tmp_path / "euclideanizer" / "1"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    eu_ev = 300
    eu_cfg = {"epochs": eu_ev}
    save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(model_dir), last_epoch_trained=250, best_epoch=150, best_val=0.5)
    (model_dir / "euclideanizer.pt").write_bytes(b"x")
    prev_ev = 100
    prev_eu_path = _eu_path(str(prev_run_dir))
    act = _euclideanizer_training_action(
        str(run_dir), eu_ev, eu_cfg,
        prev_eu_path=prev_eu_path, prev_eu_ev=prev_ev, prev_eu_run_dir=str(prev_run_dir),
        resume=True, eu_multi=True, eu_last_segment=True, eu_save_final=False,
    )
    assert act["action"] == "resume_from_best"
    assert act["additional_epochs"] == 150
    assert act["resume_from_path"] == _eu_path(str(run_dir))
    assert act["prev_run_dir"] == str(run_dir)
    assert act["best_epoch"] == 150


# ---------------------------------------------------------------------------
# Additional pipeline behavior
# ---------------------------------------------------------------------------

def test_run_completed_false_when_last_epoch_mismatch(tmp_path):
    """Run is not complete when last_epoch_trained != expected_epochs."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_run_config({"distmap": {"epochs": 2}}, str(model_dir), last_epoch_trained=1, best_epoch=1, best_val=0.5)
    (model_dir / "model.pt").write_bytes(b"x")
    assert _run_completed(str(tmp_path), 2, section_key="distmap", expected_section=None, multi_segment=False) is False


def test_euclideanizer_plotting_all_present_requires_resume(tmp_path):
    """Euclideanizer 'all present' is False when resume=False (we always regenerate)."""
    from run import _euclideanizer_plotting_all_present
    assert _euclideanizer_plotting_all_present(
        str(tmp_path), resume=False, do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True, sample_variances=[1.0]
    ) is False


def test_euclideanizer_plotting_all_present_true_when_all_exist(tmp_path):
    """Euclideanizer plotting is 'all present' when recon, recon_statistics, and gen_variance files exist."""
    from run import _euclideanizer_plotting_all_present, _plot_path
    run_dir = tmp_path / "eu"
    (run_dir / "plots" / "reconstruction").mkdir(parents=True)
    (run_dir / "plots" / "recon_statistics").mkdir(parents=True)
    (run_dir / "plots" / "gen_variance").mkdir(parents=True)
    (run_dir / "plots" / "reconstruction" / "reconstruction.png").write_bytes(b"x")
    (run_dir / "plots" / "recon_statistics" / "recon_statistics_test.png").write_bytes(b"x")
    (run_dir / "plots" / "recon_statistics" / "recon_statistics_train.png").write_bytes(b"x")
    (run_dir / "plots" / "gen_variance" / "gen_variance_1.0.png").write_bytes(b"x")
    assert _euclideanizer_plotting_all_present(
        str(run_dir), resume=True, do_recon_plot=True, do_bond_rg_scaling=True, do_avg_gen=True, sample_variances=[1.0]
    ) is True


def test_euclideanizer_analysis_all_present_true_when_no_rmsd(tmp_path):
    """When do_rmsd is False, analysis is considered all present (no analysis outputs to check)."""
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=False, variance_list=[1.0], num_samples_list=[10]
    ) is True


def test_euclideanizer_analysis_all_present_true_when_no_q(tmp_path):
    """When do_q and do_q_recon are False, Q analysis is considered all present (nothing to check)."""
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=False, variance_list=[], num_samples_list=[],
        do_q=False, do_q_recon=False,
    ) is True


def test_euclideanizer_analysis_all_present_true_when_q_gen_outputs_exist(tmp_path):
    """When do_q is True and gen outputs exist at analysis/q/gen/<run_name>/q_distributions.png, Q gen is present."""
    run_dir = tmp_path / "analysis" / "q" / "gen" / "default"
    run_dir.mkdir(parents=True)
    (run_dir / "q_distributions.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=False, variance_list=[], num_samples_list=[],
        do_q=True, do_q_recon=False,
        q_variance_list=[1.0], q_num_samples_list=[10],
    ) is True


def test_euclideanizer_analysis_all_present_false_when_q_recon_enabled_but_missing(tmp_path):
    """When do_q_recon is True and recon figure is missing, analysis is not all present."""
    (tmp_path / "analysis" / "q" / "gen" / "default").mkdir(parents=True)
    (tmp_path / "analysis" / "q" / "gen" / "default" / "q_distributions.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=False, variance_list=[], num_samples_list=[],
        do_q=True, do_q_recon=True,
        q_variance_list=[1.0], q_num_samples_list=[10],
        q_max_recon_train_list=[500], q_max_recon_test_list=[200],
    ) is False


def test_euclideanizer_analysis_all_present_true_when_q_recon_and_latent_exist(tmp_path):
    """When do_q_recon and q_visualize_latent are True, recon and latent figures must exist."""
    (tmp_path / "analysis" / "q" / "gen" / "default").mkdir(parents=True)
    (tmp_path / "analysis" / "q" / "gen" / "default" / "q_distributions.png").write_bytes(b"x")
    (tmp_path / "analysis" / "q" / "recon").mkdir(parents=True)
    (tmp_path / "analysis" / "q" / "recon" / "q_distributions.png").write_bytes(b"x")
    (tmp_path / "analysis" / "q" / "recon" / "latent_distribution.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=False, variance_list=[], num_samples_list=[],
        do_q=True, do_q_recon=True, q_visualize_latent=True,
        q_variance_list=[1.0], q_num_samples_list=[10],
        q_max_recon_train_list=[500], q_max_recon_test_list=[200],
    ) is True


def test_pipeline_data_needs_need_coords_true_when_q_enabled_and_outputs_missing(tmp_path, cfg):
    """When do_q is True and Q analysis outputs are missing, need_coords is True."""
    dm_groups, eu_groups = _all_runs_complete_layout(tmp_path, cfg)
    needs = _pipeline_data_needs(
        str(tmp_path), [0], dm_groups, eu_groups,
        resume=True, do_plot=False, do_rmsd=False,
        do_recon_plot=False, do_bond_rg_scaling=False, do_avg_gen=False,
        plot_variances=[], variance_list=[], num_samples_list=[],
        do_rmsd_recon=False, max_recon_train_list=[], max_recon_test_list=[],
        do_q=True, do_q_recon=False,
        q_variance_list=[1.0], q_num_samples_list=[10],
        q_max_recon_train_list=[], q_max_recon_test_list=[],
    )
    assert needs.need_coords is True


def test_euclideanizer_analysis_all_present_true_when_gen_outputs_exist(tmp_path):
    """When do_rmsd is True and gen outputs exist at analysis/rmsd/gen/<run_name>/, analysis is all present."""
    # Single variance and single num_samples -> run_name is "default"
    run_dir = tmp_path / "analysis" / "rmsd" / "gen" / "default"
    run_dir.mkdir(parents=True)
    (run_dir / "rmsd_distributions.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=True, variance_list=[1.0], num_samples_list=[10],
        do_rmsd_recon=False, visualize_latent=False,
    ) is True


def test_euclideanizer_analysis_all_present_false_when_recon_enabled_but_missing(tmp_path):
    """When do_rmsd_recon is True and recon figure is missing, analysis is not all present."""
    run_dir = tmp_path / "analysis" / "rmsd" / "gen" / "default"
    run_dir.mkdir(parents=True)
    (run_dir / "rmsd_distributions.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=True, variance_list=[1.0], num_samples_list=[10],
        do_rmsd_recon=True, visualize_latent=False,
        max_recon_train_list=[None], max_recon_test_list=[None],
    ) is False


def test_euclideanizer_analysis_all_present_true_when_recon_and_latent_exist(tmp_path):
    (tmp_path / "analysis" / "rmsd" / "gen" / "default").mkdir(parents=True)
    (tmp_path / "analysis" / "rmsd" / "gen" / "default" / "rmsd_distributions.png").write_bytes(b"x")
    (tmp_path / "analysis" / "rmsd" / "recon").mkdir(parents=True)
    (tmp_path / "analysis" / "rmsd" / "recon" / "rmsd_distributions.png").write_bytes(b"x")
    (tmp_path / "analysis" / "rmsd" / "recon" / "latent_distribution.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=True, variance_list=[1.0], num_samples_list=[10],
        do_rmsd_recon=True, visualize_latent=True,
        max_recon_train_list=[None], max_recon_test_list=[None],
    ) is True


def test_euclideanizer_analysis_all_present_true_when_clustering_gen_exists(tmp_path):
    """When do_clustering_gen is True and primary figure exists at analysis/clustering/gen/<run_name>/mixed_dendrograms.png, clustering gen is present."""
    run_dir = tmp_path / "analysis" / "clustering" / "gen" / "default"
    run_dir.mkdir(parents=True)
    (run_dir / "mixed_dendrograms.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=False, variance_list=[], num_samples_list=[],
        do_q=False, do_q_recon=False,
        do_clustering_gen=True, do_clustering_recon=False,
        clustering_variance_list=[1.0], clustering_num_samples_list=[10],
    ) is True


def test_euclideanizer_analysis_all_present_false_when_clustering_recon_enabled_but_missing(tmp_path):
    """When do_clustering_recon is True and recon figure is missing, analysis is not all present."""
    (tmp_path / "analysis" / "clustering" / "gen" / "default").mkdir(parents=True)
    (tmp_path / "analysis" / "clustering" / "gen" / "default" / "mixed_dendrograms.png").write_bytes(b"x")
    assert _euclideanizer_analysis_all_present(
        str(tmp_path), resume=True, do_rmsd=False, variance_list=[], num_samples_list=[],
        do_q=False, do_q_recon=False,
        do_clustering_gen=True, do_clustering_recon=True,
        clustering_variance_list=[1.0], clustering_num_samples_list=[10],
        clustering_max_recon_train_list=[None], clustering_max_recon_test_list=[None],
    ) is False


def test_pipeline_need_data_false_only_when_all_runs_and_outputs_present(tmp_path):
    """need_data is False only when every run is complete and plot/analysis flags are off (no outputs to check)."""
    cfg = _load_test_config()
    dm_groups, eu_groups = _get_dm_eu_groups(cfg)
    (tmp_path / "seed_0").mkdir()
    for ri, ev in [(0, 1), (1, 2)]:
        d = tmp_path / "seed_0" / "distmap" / str(ri) / "model"
        d.mkdir(parents=True)
        save_run_config({"distmap": {"epochs": ev}}, str(d), last_epoch_trained=ev, best_epoch=1, best_val=0.5)
        (d / "model.pt").write_bytes(b"x")
        if ev == 1:
            (d / "model_last.pt").write_bytes(b"x")
    for ri in (0, 1):
        for euri, eu_ev in [(0, 1), (1, 2)]:
            d = tmp_path / "seed_0" / "distmap" / str(ri) / "euclideanizer" / str(euri) / "model"
            d.mkdir(parents=True)
            save_run_config({"euclideanizer": {"epochs": eu_ev}}, str(d), last_epoch_trained=eu_ev, best_epoch=1, best_val=0.5)
            (d / "euclideanizer.pt").write_bytes(b"x")
            if eu_ev == 1:
                (d / "euclideanizer_last.pt").write_bytes(b"x")
    assert _pipeline_need_data(str(tmp_path), [0], dm_groups, eu_groups, resume=True, do_plot=False, do_rmsd=False,
        do_recon_plot=False, do_bond_rg_scaling=False, do_avg_gen=False,
        plot_variances=[], variance_list=[], num_samples_list=[]) is False


# ---------------------------------------------------------------------------
# Reference-size config and cache purge
# ---------------------------------------------------------------------------

def test_reference_size_config_extracts_keys():
    """_reference_size_config returns plotting and analysis reference-size keys."""
    cfg = {
        "plotting": {"max_train": 100, "max_test": 200},
        "analysis": {
            "rmsd_max_train": 500, "rmsd_max_test": 100,
            "q_max_train": 300, "q_max_test": 50,
            "clustering_max_train": 400, "clustering_max_test": 80,
        },
    }
    ref = _reference_size_config(cfg)
    assert ref["plotting"] == (100, 200)
    assert ref["rmsd"] == (500, 100)
    assert ref["q"] == (300, 50)
    assert ref["clustering"] == (400, 80)


def test_reference_size_changed_detects_differences():
    """_reference_size_changed returns components whose ref config differs."""
    saved = {"plotting": (100, 200), "rmsd": (500, 100), "q": (300, 50), "clustering": (400, 80)}
    current = {"plotting": (100, 200), "rmsd": (500, 200), "q": (300, 50), "clustering": (400, 80)}
    assert _reference_size_changed(saved, current) == {"rmsd"}
    current["plotting"] = (50, 200)
    assert _reference_size_changed(saved, current) == {"plotting", "rmsd"}
    assert _reference_size_changed(saved, saved) == set()


def test_delete_reference_size_caches_removes_plotting_split_files(tmp_path):
    """_delete_reference_size_caches(plotting) removes split meta and train/test npz in each seed."""
    base = str(tmp_path)
    for seed in (0, 1):
        cache_dir = os.path.join(base, f"seed_{seed}", EXP_STATS_CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)
        for name in (EXP_STATS_SPLIT_META, EXP_STATS_TRAIN_NPZ, EXP_STATS_TEST_NPZ):
            (tmp_path / f"seed_{seed}" / EXP_STATS_CACHE_DIR / name).write_bytes(b"x")
    _delete_reference_size_caches(base, [0, 1], {"plotting"})
    for seed in (0, 1):
        for name in (EXP_STATS_SPLIT_META, EXP_STATS_TRAIN_NPZ, EXP_STATS_TEST_NPZ):
            assert not os.path.isfile(os.path.join(base, f"seed_{seed}", EXP_STATS_CACHE_DIR, name))


def test_delete_reference_size_caches_removes_rmsd_and_q_files(tmp_path):
    """_delete_reference_size_caches(rmsd, q) removes test_to_train_rmsd*.npz and q_test_to_train_*.npz."""
    base = str(tmp_path)
    cache_dir = os.path.join(base, "seed_0", EXP_STATS_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    (tmp_path / "seed_0" / EXP_STATS_CACHE_DIR / "test_to_train_rmsd.npz").write_bytes(b"x")
    (tmp_path / "seed_0" / EXP_STATS_CACHE_DIR / "q_test_to_train_500_200.npz").write_bytes(b"x")
    _delete_reference_size_caches(base, [0], {"rmsd", "q"})
    assert not os.path.isfile(os.path.join(cache_dir, "test_to_train_rmsd.npz"))
    assert not os.path.isfile(os.path.join(cache_dir, "q_test_to_train_500_200.npz"))
