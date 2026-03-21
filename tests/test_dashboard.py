"""
Tests for dashboard scanning and build.

_scan_runs discovers seed/distmap/euclideanizer run tree; build_dashboard
produces index.html and manifest.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.dashboard import _scan_runs, _score_strip_for_plot_block, build_dashboard
from src.config import save_run_config


def _make_minimal_run_tree(tmp_path, n_seeds=1, n_dm=1, n_eu=1, seed_dirname_fn=None):
    """Create minimal completed run tree for dashboard scanning.

    seed_dirname_fn(s) returns top-level dir name for seed index s (default: seed_<s>).
    """
    if seed_dirname_fn is None:
        seed_dirname_fn = lambda s: f"seed_{s}"

    for s in range(n_seeds):
        top = seed_dirname_fn(s)
        for di in range(n_dm):
            dm_dir = tmp_path / top / "distmap" / str(di) / "model"
            dm_dir.mkdir(parents=True)
            save_run_config(
                {"distmap": {"epochs": 10}}, str(dm_dir),
                last_epoch_trained=10, best_epoch=8, best_val=0.5,
            )
            (dm_dir / "model.pt").write_bytes(b"x")
            for ei in range(n_eu):
                eu_dir = tmp_path / top / "distmap" / str(di) / "euclideanizer" / str(ei) / "model"
                eu_dir.mkdir(parents=True)
                save_run_config(
                    {"euclideanizer": {"epochs": 10}}, str(eu_dir),
                    last_epoch_trained=10, best_epoch=8, best_val=0.5,
                )
                (eu_dir / "euclideanizer.pt").write_bytes(b"x")


def test_scan_runs_finds_expected_run_ids(tmp_path):
    _make_minimal_run_tree(tmp_path, n_seeds=2, n_dm=1, n_eu=1)
    runs = _scan_runs(str(tmp_path))
    ids = {r["id"] for r in runs}
    assert "seed_0" in ids
    assert "seed_1" in ids
    assert "seed_0_dm_0" in ids
    assert "seed_0_dm_0_eu_0" in ids


def test_scan_runs_levels_correct(tmp_path):
    _make_minimal_run_tree(tmp_path)
    runs = _scan_runs(str(tmp_path))
    by_id = {r["id"]: r for r in runs}
    assert by_id["seed_0"]["level"] == "seed"
    assert by_id["seed_0_dm_0"]["level"] == "distmap"
    assert by_id["seed_0_dm_0_eu_0"]["level"] == "euclideanizer"


def test_scan_runs_parent_child_links(tmp_path):
    _make_minimal_run_tree(tmp_path)
    runs = _scan_runs(str(tmp_path))
    by_id = {r["id"]: r for r in runs}
    assert "seed_0_dm_0" in by_id["seed_0"]["children_ids"]
    assert "seed_0_dm_0_eu_0" in by_id["seed_0_dm_0"]["children_ids"]
    assert by_id["seed_0_dm_0"]["parent_id"] == "seed_0"


def test_scan_runs_empty_dir(tmp_path):
    assert _scan_runs(str(tmp_path)) == []


def test_scan_runs_training_split_dirs(tmp_path):
    """Multiple training_split layout: seed_<n>_split_<frac>/distmap/..."""
    _make_minimal_run_tree(
        tmp_path,
        n_seeds=1,
        seed_dirname_fn=lambda s: f"seed_{s}_split_0.9",
    )
    runs = _scan_runs(str(tmp_path))
    ids = {r["id"] for r in runs}
    assert "seed_0_split_0.9" in ids
    assert "seed_0_split_0.9_dm_0" in ids
    assert "seed_0_split_0.9_dm_0_eu_0" in ids
    by_id = {r["id"]: r for r in runs}
    assert "Split 0.9" in by_id["seed_0_split_0.9"]["label_short"]
    assert by_id["seed_0_split_0.9_dm_0"]["parent_id"] == "seed_0_split_0.9"


def test_scan_runs_two_splits_distinct_ids(tmp_path):
    for frac in ("0.8", "0.9"):
        _make_minimal_run_tree(
            tmp_path,
            n_seeds=1,
            seed_dirname_fn=lambda s, f=frac: f"seed_{s}_split_{f}",
        )
    runs = _scan_runs(str(tmp_path))
    ids = {r["id"] for r in runs}
    assert "seed_0_split_0.8_dm_0_eu_0" in ids
    assert "seed_0_split_0.9_dm_0_eu_0" in ids
    by_id = {r["id"]: r for r in runs}
    eu8 = by_id["seed_0_split_0.8_dm_0_eu_0"]
    eu9 = by_id["seed_0_split_0.9_dm_0_eu_0"]
    assert eu8["training_split"] == 0.8
    assert eu9["training_split"] == 0.9
    assert eu8["split_seed"] == eu9["split_seed"] == 0
    assert eu8["distmap_index"] == eu9["distmap_index"] == 0
    assert eu8["euclideanizer_index"] == 0
    dm8 = by_id["seed_0_split_0.8_dm_0"]
    assert dm8["split_seed"] == 0 and dm8["distmap_index"] == 0 and dm8["training_split"] == 0.8


def test_scan_runs_training_split_and_max_data_dirs(tmp_path):
    _make_minimal_run_tree(
        tmp_path,
        n_seeds=1,
        seed_dirname_fn=lambda s: f"seed_{s}_split_0.9_maxdata_500",
    )
    runs = _scan_runs(str(tmp_path))
    ids = {r["id"] for r in runs}
    assert "seed_0_split_0.9_maxdata_500" in ids
    assert "seed_0_split_0.9_maxdata_500_dm_0_eu_0" in ids
    by_id = {r["id"]: r for r in runs}
    assert "Max Data 500" in by_id["seed_0_split_0.9_maxdata_500"]["label_short"]
    assert "Split 0.9" in by_id["seed_0_split_0.9_maxdata_500"]["label_short"]
    assert "DistMap 0" in by_id["seed_0_split_0.9_maxdata_500_dm_0"]["label_short"]
    assert "Euclideanizer 0" in by_id["seed_0_split_0.9_maxdata_500_dm_0_eu_0"]["label_short"]


def test_scan_runs_training_split_from_pipeline_config(tmp_path):
    """Single seed_<n>/ dir: training_split from pipeline_config.yaml for Vary aspect."""
    _make_minimal_run_tree(tmp_path, n_seeds=1)
    seed_dir = tmp_path / "seed_0"
    (seed_dir / "pipeline_config.yaml").write_text(
        "data:\n  path: x.npz\n  split_seed: 0\n  training_split: 0.85\n",
        encoding="utf-8",
    )
    runs = _scan_runs(str(tmp_path))
    by_id = {r["id"]: r for r in runs}
    assert by_id["seed_0_dm_0_eu_0"]["training_split"] == 0.85


def test_build_dashboard_creates_files(tmp_path):
    _make_minimal_run_tree(tmp_path)
    result = build_dashboard(str(tmp_path))
    assert result is not None
    assert os.path.isfile(os.path.join(result, "index.html"))
    assert os.path.isfile(os.path.join(result, "manifest.json"))


def test_score_strip_mapping_rmsd_and_recon():
    comp = {
        "gen_rmsd_train_vs_tt": 0.91,
        "gen_rmsd_test_vs_tt": 0.88,
        "recon_scaling_train": 0.95,
    }
    g = _score_strip_for_plot_block("rmsd_gen", "RMSD (gen) default", comp)
    assert g and len(g["groups"]) == 1 and len(g["groups"][0]["items"]) == 2
    assert g["groups"][0]["items"][0]["label"]
    r = _score_strip_for_plot_block("reconstruction", "Reconstruction", comp)
    assert r and len(r["groups"]) >= 1
    assert r["groups"][0]["title"] == "Reconstruction (Train)"


# 1×1 RGBA PNG for dashboard asset copy
_MIN_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def test_build_dashboard_manifest_includes_score_strip(tmp_path):
    _make_minimal_run_tree(tmp_path)
    eu_root = tmp_path / "seed_0" / "distmap" / "0" / "euclideanizer" / "0"
    scoring = eu_root / "scoring"
    scoring.mkdir(parents=True, exist_ok=True)
    scores = {
        "overall_score": 0.85,
        "component_scores": {
            "recon_scaling_train": 0.92,
            "recon_rg_train": 0.88,
            "recon_pairwise_train": 0.9,
            "recon_avgmap_train": 0.87,
            "gen_rmsd_train_vs_tt": 0.91,
            "gen_rmsd_test_vs_tt": 0.89,
        },
        "present": [],
        "missing": [],
    }
    (scoring / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    rec = tmp_path / "seed_0" / "distmap" / "0" / "plots" / "reconstruction"
    rec.mkdir(parents=True)
    (rec / "reconstruction.png").write_bytes(_MIN_PNG)
    rmsd = eu_root / "analysis" / "rmsd" / "gen" / "default"
    rmsd.mkdir(parents=True)
    (rmsd / "rmsd_distributions.png").write_bytes(_MIN_PNG)

    result = build_dashboard(str(tmp_path))
    assert result
    with open(os.path.join(result, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    eu = next(r for r in man["runs"] if r.get("id") == "seed_0_dm_0_eu_0")
    rmsd_block = next(b for b in eu["blocks"] if b.get("type") == "rmsd_gen")
    assert rmsd_block.get("score_strip")
    assert len(rmsd_block["score_strip"]["groups"][0]["items"]) == 2
    dm = next(r for r in man["runs"] if r["id"] == "seed_0_dm_0")
    recon_b = next(b for b in dm["blocks"] if b.get("type") == "reconstruction")
    assert not recon_b.get("score_strip")
    for sub, fn in (
        ("bond_length_by_genomic_distance_gen", "bond_length_by_genomic_distance_gen.png"),
        ("bond_length_by_genomic_distance_train", "bond_length_by_genomic_distance_train.png"),
        ("bond_length_by_genomic_distance_test", "bond_length_by_genomic_distance_test.png"),
    ):
        d = eu_root / "plots" / sub
        d.mkdir(parents=True)
        (d / fn).write_bytes(_MIN_PNG)
    result2 = build_dashboard(str(tmp_path))
    man2 = json.loads((Path(result2) / "manifest.json").read_text(encoding="utf-8"))
    eu2 = next(r for r in man2["runs"] if r.get("id") == "seed_0_dm_0_eu_0")
    bond_b = next(b for b in eu2["blocks"] if b.get("type") == "bond_length_by_genomic_distance_gen")
    assert bond_b.get("score_strip")
    assert "Pairwise Distance" in json.dumps(bond_b["score_strip"])
    assert man2.get("score_component_catalog")


def test_build_dashboard_manifest_keeps_training_split_for_vary_aspect(tmp_path):
    for frac in ("0.8", "0.9"):
        _make_minimal_run_tree(
            tmp_path,
            n_seeds=1,
            seed_dirname_fn=lambda s, f=frac: f"seed_{s}_split_{f}",
        )
    result = build_dashboard(str(tmp_path))
    assert result is not None
    with open(os.path.join(result, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    eu_runs = [r for r in man["runs"] if r.get("level") == "euclideanizer"]
    ts = {r.get("training_split") for r in eu_runs}
    assert 0.8 in ts and 0.9 in ts
    assert all(r.get("split_seed") == 0 for r in eu_runs)


def test_scan_runs_includes_sufficiency_meta_block(tmp_path):
    _make_minimal_run_tree(tmp_path, n_seeds=1)
    heat = (
        tmp_path
        / "meta_analysis"
        / "sufficiency"
        / "seed_0"
        / "heatmap"
        / "sufficiency_heatmap_rmsd_q.png"
    )
    heat.parent.mkdir(parents=True, exist_ok=True)
    heat.write_bytes(_MIN_PNG)
    runs = _scan_runs(str(tmp_path))
    seed = next(r for r in runs if r["id"] == "seed_0")
    block_types = {b.get("type") for b in seed.get("blocks", [])}
    assert "meta_analysis_sufficiency" in block_types


def test_scan_runs_includes_generative_capacity_blocks(tmp_path):
    _make_minimal_run_tree(tmp_path, n_seeds=1)
    eu_root = tmp_path / "seed_0" / "distmap" / "0" / "euclideanizer" / "0"
    gc_r = eu_root / "analysis" / "generative_capacity" / "rmsd" / "generative_capacity_rmsd.png"
    gc_q = eu_root / "analysis" / "generative_capacity" / "q" / "generative_capacity_q.png"
    gc_r.parent.mkdir(parents=True, exist_ok=True)
    gc_q.parent.mkdir(parents=True, exist_ok=True)
    gc_r.write_bytes(_MIN_PNG)
    gc_q.write_bytes(_MIN_PNG)
    runs = _scan_runs(str(tmp_path))
    eu = next(r for r in runs if r["id"] == "seed_0_dm_0_eu_0")
    block_types = {b.get("type") for b in eu.get("blocks", [])}
    assert "generative_capacity_rmsd" in block_types
    assert "generative_capacity_q" in block_types


def test_scan_runs_includes_sufficiency_meta_sources(tmp_path):
    _make_minimal_run_tree(tmp_path, n_seeds=1)
    heat = (
        tmp_path
        / "meta_analysis"
        / "sufficiency"
        / "seed_0"
        / "heatmap"
        / "sufficiency_heatmap_rmsd_q.png"
    )
    heat.parent.mkdir(parents=True, exist_ok=True)
    heat.write_bytes(_MIN_PNG)
    d400 = tmp_path / "meta_analysis" / "sufficiency" / "seed_0" / "distributions" / "max_data_400"
    d400.mkdir(parents=True)
    (d400 / "distributions_rmsd_q.png").write_bytes(_MIN_PNG)
    runs = _scan_runs(str(tmp_path))
    seed = next(r for r in runs if r["id"] == "seed_0")
    assert "sufficiency_meta_sources" in seed
    assert seed["sufficiency_meta_sources"]["heatmap_source"]
    assert len(seed["sufficiency_meta_sources"]["distributions"]) == 1


def test_build_dashboard_manifest_includes_sufficiency_meta(tmp_path):
    _make_minimal_run_tree(tmp_path, n_seeds=1)
    heat = (
        tmp_path
        / "meta_analysis"
        / "sufficiency"
        / "seed_0"
        / "heatmap"
        / "sufficiency_heatmap_rmsd_q.png"
    )
    heat.parent.mkdir(parents=True, exist_ok=True)
    heat.write_bytes(_MIN_PNG)
    d400 = tmp_path / "meta_analysis" / "sufficiency" / "seed_0" / "distributions" / "max_data_400"
    d400.mkdir(parents=True)
    (d400 / "distributions_rmsd_q.png").write_bytes(_MIN_PNG)
    result = build_dashboard(str(tmp_path))
    assert result
    man = json.loads((Path(result) / "manifest.json").read_text(encoding="utf-8"))
    seed = next(r for r in man["runs"] if r["id"] == "seed_0")
    assert "sufficiency_meta" in seed
    assert seed["sufficiency_meta"]["heatmap"].startswith("assets/")
    assert len(seed["sufficiency_meta"]["distributions"]) == 1
    assert seed["sufficiency_meta"]["distributions"][0]["max_data"] == "400"
    assert os.path.isfile(os.path.join(result, seed["sufficiency_meta"]["heatmap"]))
