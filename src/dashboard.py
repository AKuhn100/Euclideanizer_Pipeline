"""
Build an interactive HTML dashboard in the run root.

Scans base_output_dir for pipeline run roots: seed_<n>/distmap/... (single training_split)
and seed_<n>_split_<frac>/distmap/... (multiple training_split values), then
seed_*/distmap/*/euclideanizer/*, collects run_config labels, discovers plots/videos/analysis
outputs, copies them into dashboard/assets/, and writes dashboard/manifest.json and index.html.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
from datetime import datetime
from typing import Any, Optional

import numpy as np

from .config import load_pipeline_config, load_run_config
from .scoring import EXPECTED_COMPONENTS, SCORING_DIR, SCORES_SPIDER_FILENAME

# Path patterns (aligned with run.py; no import from run to avoid circular deps)
_PLOTS_BASE = "plots"
_ANALYSIS_DIR = "analysis"
# Latent analysis output lives under analysis/latent/ (same as other analysis metrics)
_LATENT_PLOTS_DIR = os.path.join(_ANALYSIS_DIR, "latent")
_RECONSTRUCTION = os.path.join(_PLOTS_BASE, "reconstruction", "reconstruction.png")
_RECON_STAT_TRAIN = os.path.join(_PLOTS_BASE, "recon_statistics", "recon_statistics_train.png")
_RECON_STAT_TEST = os.path.join(_PLOTS_BASE, "recon_statistics", "recon_statistics_test.png")
_GEN_VARIANCE_PATTERN = os.path.join(_PLOTS_BASE, "gen_variance", "gen_variance_*.png")
_BOND_LENGTH_GEN = os.path.join(_PLOTS_BASE, "bond_length_by_genomic_distance_gen", "bond_length_by_genomic_distance_gen.png")
_BOND_LENGTH_TRAIN = os.path.join(_PLOTS_BASE, "bond_length_by_genomic_distance_train", "bond_length_by_genomic_distance_train.png")
_BOND_LENGTH_TEST = os.path.join(_PLOTS_BASE, "bond_length_by_genomic_distance_test", "bond_length_by_genomic_distance_test.png")
# Training video lives outside plots/ so plotting wipe does not remove it
_TRAINING_VIDEO = os.path.join("training_video", "training_evolution.mp4")
# RMSD analysis: rmsd.py writes analysis/rmsd/gen/<run_name>/rmsd_distributions.png and analysis/rmsd/recon/...
_RMSD_DIR = "rmsd"
_RMSD_FIG = "rmsd_distributions.png"
_ANALYSIS_Q_DIR = "q"
_Q_FIG = "q_distributions.png"
_COORD_CLUSTERING_DIR = "coord_clustering"
_DISTMAP_CLUSTERING_DIR = "distmap_clustering"
# Clustering section order: pure dendrograms, mixed dendrograms, mixing analysis, rmse (Q–Q) similarity
_CLUSTERING_FIGS_ORDERED = (
    "pure_dendrograms.png",
    "mixed_dendrograms.png",
    "mixing_analysis.png",
    "rmse_similarity.png",
)

DASHBOARD_DIR = "dashboard"
ASSETS_DIR = "assets"
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "index.html"


def _run_config_dir(run_root: str) -> str:
    return os.path.join(run_root, "model")


def _parse_seed_output_dir(dirname: str) -> tuple[int, str, Optional[str]] | None:
    """
    Parse top-level output dir name from run.py layout.

    Returns (split_seed, seed_group_id, training_split_token) or None if not a seed run root.
    seed_group_id matches the directory name (unique per seed or per seed×split).
    training_split_token is the string after _split_ when multiple splits are used; else None.
    """
    m = re.match(r"^seed_(\d+)$", dirname)
    if m:
        return int(m.group(1)), dirname, None
    m = re.match(r"^seed_(\d+)_split_(.+)$", dirname)
    if m:
        return int(m.group(1)), dirname, m.group(2)
    return None


def _training_split_value_for_seed_dir(seed_dir: str, split_token: Optional[str]) -> Optional[float]:
    """Train fraction for manifest (Vary aspect: training_split). Dir name or pipeline_config.yaml."""
    if split_token is not None:
        try:
            return float(split_token)
        except ValueError:
            pass
    cfg = load_pipeline_config(seed_dir)
    if not cfg or "data" not in cfg:
        return None
    ts = cfg["data"].get("training_split")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return float(ts)
    if isinstance(ts, list) and len(ts) == 1 and isinstance(ts[0], (int, float)) and not isinstance(ts[0], bool):
        return float(ts[0])
    return None


def _label_from_distmap_config(
    cfg: Optional[dict],
    seed: int,
    dm_index: int,
    *,
    seed_group_id: str,
    split_token: Optional[str] = None,
) -> tuple[str, str]:
    prefix = f"Seed {seed} · split {split_token} · " if split_token else f"Seed {seed} · "
    short = f"{prefix}DM {dm_index}"
    if not cfg or "distmap" not in cfg:
        return short, f"{seed_group_id}_dm_{dm_index}"
    d = cfg["distmap"]
    long_parts = [f"seed={seed}"]
    if split_token is not None:
        long_parts.append(f"training_split={split_token}")
    long_parts.append(f"distmap_index={dm_index}")
    long_parts += [f"distmap.{k}={v}" for k, v in (d or {}).items()]
    return short, " ".join(long_parts)


def _label_from_euclideanizer_config(
    cfg: Optional[dict],
    seed: int,
    dm_index: int,
    eu_index: int,
    *,
    seed_group_id: str,
    split_token: Optional[str] = None,
) -> tuple[str, str]:
    prefix = f"Seed {seed} · split {split_token} · " if split_token else f"Seed {seed} · "
    short = f"{prefix}DM {dm_index} · Eu {eu_index}"
    if not cfg or "euclideanizer" not in cfg:
        return short, f"{seed_group_id}_dm_{dm_index}_eu_{eu_index}"
    e = cfg["euclideanizer"]
    long_parts = [f"seed={seed}"]
    if split_token is not None:
        long_parts.append(f"training_split={split_token}")
    long_parts += [f"dm={dm_index}", f"eu={eu_index}"]
    long_parts += [f"euclideanizer.{k}={v}" for k, v in (e or {}).items()]
    return short, " ".join(long_parts)


def _load_component_scores(run_root: str) -> dict[str, Any]:
    """Full component_scores dict from scoring/scores.json, or {}."""
    path = os.path.join(run_root, SCORING_DIR, "scores.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return dict((data or {}).get("component_scores") or {})
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


# Title-case labels for manifest score-picker and strips (aligned with SCORING_SPEC components)
_SCORE_COMPONENT_LABELS: dict[str, str] = {
    "recon_scaling_train": "Recon Scaling (Train)",
    "recon_scaling_test": "Recon Scaling (Test)",
    "recon_rg_train": "Recon Radius Of Gyration (Train)",
    "recon_rg_test": "Recon Radius Of Gyration (Test)",
    "recon_pairwise_train": "Recon Pairwise Distance (Train)",
    "recon_pairwise_test": "Recon Pairwise Distance (Test)",
    "recon_avgmap_train": "Recon Average Contact Map (Train)",
    "recon_avgmap_test": "Recon Average Contact Map (Test)",
    "gen_rg": "Gen Radius Of Gyration",
    "gen_scaling": "Gen Scaling",
    "gen_pairwise": "Gen Pairwise Distance",
    "gen_avgmap": "Gen Average Contact Map",
    "gen_rmsd_train_vs_tt": "Gen RMSD Train Vs Train Reference",
    "gen_rmsd_test_vs_tt": "Gen RMSD Test Vs Train Reference",
    "recon_rmsd_train": "Recon RMSD (Train)",
    "recon_rmsd_test": "Recon RMSD (Test)",
    "latent_means": "Latent Means (Z-Scored)",
    "latent_stds": "Latent Stds (Z-Scored)",
    "gen_q_train_vs_tt": "Gen Q Train Vs Train Reference",
    "gen_q_test_vs_tt": "Gen Q Test Vs Train Reference",
    "recon_q_train": "Recon Q (Train)",
    "recon_q_test": "Recon Q (Test)",
    "clustering_coord_gen_train": "Coord Clustering Gen Mixing (Train)",
    "clustering_coord_gen_test": "Coord Clustering Gen Mixing (Test)",
    "clustering_coord_recon_train": "Coord Clustering Recon Mixing (Train)",
    "clustering_coord_recon_test": "Coord Clustering Recon Mixing (Test)",
    "clustering_distmap_gen_train": "Distmap Clustering Gen Mixing (Train)",
    "clustering_distmap_gen_test": "Distmap Clustering Gen Mixing (Test)",
    "clustering_distmap_recon_train": "Distmap Clustering Recon Mixing (Train)",
    "clustering_distmap_recon_test": "Distmap Clustering Recon Mixing (Test)",
}


def _manifest_score_component_catalog() -> list[dict[str, str]]:
    return [
        {"id": k, "label": _SCORE_COMPONENT_LABELS.get(k, k.replace("_", " ").title())}
        for k in EXPECTED_COMPONENTS
    ]


def _json_safe_component_scores(comp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (comp or {}).items():
        try:
            f = float(v)
            out[k] = round(f, 6) if math.isfinite(f) else None
        except (TypeError, ValueError):
            out[k] = None
    return out


def _pick_score_value(comp: dict[str, Any], key: str) -> float | None:
    v = comp.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 6)


def _any_score_for_keys(comp: dict[str, Any], keys: list[str]) -> bool:
    return any(_pick_score_value(comp, k) is not None for k in keys)


def _score_strip_for_plot_block(block_type: str, block_name: str, comp: dict[str, Any]) -> dict[str, Any] | None:
    """
    Build grouped score strip for manifest (shown under matching plots).
    Returns {"groups": [{"title": str | None, "items": [{"label", "value"}]}]} or None.
    """
    if not comp or block_type in ("scores", "training_video"):
        return None

    groups: list[dict[str, Any]] = []

    def push_group(title: str | None, pairs: list[tuple[str, str]]) -> None:
        keys = [k for _, k in pairs]
        if not _any_score_for_keys(comp, keys):
            return
        groups.append({
            "title": title,
            "items": [{"label": lab, "value": _pick_score_value(comp, k)} for lab, k in pairs],
        })

    name = block_name or ""

    if block_type == "reconstruction":
        push_group(
            "Reconstruction (Train)",
            [
                ("Scaling", "recon_scaling_train"),
                ("Radius Of Gyration", "recon_rg_train"),
                ("Pairwise Distance", "recon_pairwise_train"),
                ("Average Contact Map", "recon_avgmap_train"),
            ],
        )
        push_group(
            "Reconstruction (Test)",
            [
                ("Scaling", "recon_scaling_test"),
                ("Radius Of Gyration", "recon_rg_test"),
                ("Pairwise Distance", "recon_pairwise_test"),
                ("Average Contact Map", "recon_avgmap_test"),
            ],
        )
    elif block_type == "recon_statistics_train":
        push_group(
            None,
            [
                ("Scaling", "recon_scaling_train"),
                ("Radius Of Gyration", "recon_rg_train"),
                ("Pairwise Distance", "recon_pairwise_train"),
                ("Average Contact Map", "recon_avgmap_train"),
            ],
        )
    elif block_type == "recon_statistics_test":
        push_group(
            None,
            [
                ("Scaling", "recon_scaling_test"),
                ("Radius Of Gyration", "recon_rg_test"),
                ("Pairwise Distance", "recon_pairwise_test"),
                ("Average Contact Map", "recon_avgmap_test"),
            ],
        )
    elif block_type == "bond_length_by_genomic_distance_gen":
        push_group(
            "Pairwise Distance (Gen Plot)",
            [
                ("Pairwise Distance (Train)", "recon_pairwise_train"),
                ("Pairwise Distance (Test)", "recon_pairwise_test"),
                ("Pairwise Distance (Gen)", "gen_pairwise"),
            ],
        )
    elif block_type == "bond_length_by_genomic_distance_train":
        push_group(None, [("Pairwise Distance (Train)", "recon_pairwise_train")])
    elif block_type == "bond_length_by_genomic_distance_test":
        push_group(None, [("Pairwise Distance (Test)", "recon_pairwise_test")])
    elif block_type == "gen_variance":
        m = re.search(r"Gen\s+Variance\s+([\d.]+)", name, re.IGNORECASE)
        if not m:
            return None
        try:
            if abs(float(m.group(1)) - 1.0) > 1e-6:
                return None
        except ValueError:
            return None
        push_group(
            "Generation (Sample Variance = 1)",
            [
                ("Radius Of Gyration", "gen_rg"),
                ("Scaling", "gen_scaling"),
                ("Pairwise Distance", "gen_pairwise"),
                ("Average Contact Map", "gen_avgmap"),
            ],
        )
    elif block_type == "rmsd_gen":
        push_group(
            None,
            [
                ("Gen RMSD Train Vs Train Reference", "gen_rmsd_train_vs_tt"),
                ("Gen RMSD Test Vs Train Reference", "gen_rmsd_test_vs_tt"),
            ],
        )
    elif block_type == "rmsd_recon":
        push_group(
            None,
            [
                ("Recon RMSD (Train)", "recon_rmsd_train"),
                ("Recon RMSD (Test)", "recon_rmsd_test"),
            ],
        )
    elif block_type == "latent_distribution" or block_type == "latent_correlation":
        push_group(
            "Latent Train Vs Test",
            [
                ("Means (Z-Scored)", "latent_means"),
                ("Stds (Z-Scored)", "latent_stds"),
            ],
        )
    elif block_type == "q_gen":
        push_group(
            None,
            [
                ("Gen Q Train Vs Train Reference", "gen_q_train_vs_tt"),
                ("Gen Q Test Vs Train Reference", "gen_q_test_vs_tt"),
            ],
        )
    elif block_type == "q_recon":
        push_group(
            None,
            [
                ("Recon Q (Train)", "recon_q_train"),
                ("Recon Q (Test)", "recon_q_test"),
            ],
        )
    elif block_type == "coord_clustering_gen":
        push_group(
            None,
            [
                ("Coord Clustering Gen Mixing (Train)", "clustering_coord_gen_train"),
                ("Coord Clustering Gen Mixing (Test)", "clustering_coord_gen_test"),
            ],
        )
    elif block_type == "coord_clustering_recon":
        push_group(
            None,
            [
                ("Coord Clustering Recon Mixing (Train)", "clustering_coord_recon_train"),
                ("Coord Clustering Recon Mixing (Test)", "clustering_coord_recon_test"),
            ],
        )
    elif block_type == "distmap_clustering_gen":
        push_group(
            None,
            [
                ("Distmap Clustering Gen Mixing (Train)", "clustering_distmap_gen_train"),
                ("Distmap Clustering Gen Mixing (Test)", "clustering_distmap_gen_test"),
            ],
        )
    elif block_type == "distmap_clustering_recon":
        push_group(
            None,
            [
                ("Distmap Clustering Recon Mixing (Train)", "clustering_distmap_recon_train"),
                ("Distmap Clustering Recon Mixing (Test)", "clustering_distmap_recon_test"),
            ],
        )

    if not groups:
        return None
    return {"groups": groups}


def _dashboard_display_title(name: str) -> str:
    """Title Case for dashboard section titles (RMSD/Q run names, clustering labels, etc.)."""
    if not name or not str(name).strip():
        return name
    SPEC = {"rmsd": "RMSD", "q": "Q"}

    def one_segment(seg: str) -> str:
        seg = seg.strip()
        if not seg:
            return seg
        pieces: list[str] = []
        for part in re.split(r"(\([^)]+\))", seg):
            if not part:
                continue
            if part.startswith("(") and part.endswith(")"):
                pieces.append("(" + part[1:-1].strip().title() + ")")
            else:
                ws = []
                for raw in part.split():
                    core, trail = raw, ""
                    while len(core) > 1 and core[-1] in ".,;:!?":
                        trail = core[-1] + trail
                        core = core[:-1]
                    cl = core.lower()
                    if cl in SPEC:
                        ws.append(SPEC[cl] + trail)
                    else:
                        ws.append(core.replace("_", " ").title() + trail)
                pieces.append(" ".join(ws))
        return " ".join(pieces)

    return " — ".join(one_segment(p) for p in name.split(" — "))


def _finalize_dashboard_block_titles(blocks: list) -> None:
    for b in blocks:
        n = b.get("name")
        if isinstance(n, str) and n.strip():
            b["name"] = _dashboard_display_title(n)


def _blocks_for_distmap_run(run_root: str) -> list[dict[str, str]]:
    blocks = []
    if os.path.isfile(os.path.join(run_root, _RECONSTRUCTION)):
        blocks.append({"type": "reconstruction", "name": "Reconstruction", "source_path": _RECONSTRUCTION})
    if os.path.isfile(os.path.join(run_root, _RECON_STAT_TRAIN)):
        blocks.append({"type": "recon_statistics_train", "name": "Recon statistics (train)", "source_path": _RECON_STAT_TRAIN})
    if os.path.isfile(os.path.join(run_root, _RECON_STAT_TEST)):
        blocks.append({"type": "recon_statistics_test", "name": "Recon statistics (test)", "source_path": _RECON_STAT_TEST})
    gen_dir = os.path.join(run_root, _PLOTS_BASE, "gen_variance")
    if os.path.isdir(gen_dir):
        for p in sorted(glob.glob(os.path.join(gen_dir, "gen_variance_*.png"))):
            base = os.path.basename(p)
            var = base.replace("gen_variance_", "").replace(".png", "")
            rel = os.path.join(_PLOTS_BASE, "gen_variance", base)
            blocks.append({"type": "gen_variance", "name": f"Gen variance {var}", "source_path": rel})
    if os.path.isfile(os.path.join(run_root, _BOND_LENGTH_TRAIN)):
        blocks.append({
            "type": "bond_length_by_genomic_distance_train",
            "name": "Pairwise distance by lag (train exp vs recon)",
            "source_path": _BOND_LENGTH_TRAIN,
        })
    if os.path.isfile(os.path.join(run_root, _BOND_LENGTH_TEST)):
        blocks.append({
            "type": "bond_length_by_genomic_distance_test",
            "name": "Pairwise distance by lag (test exp vs recon)",
            "source_path": _BOND_LENGTH_TEST,
        })
    if os.path.isfile(os.path.join(run_root, _BOND_LENGTH_GEN)):
        blocks.append({
            "type": "bond_length_by_genomic_distance_gen",
            "name": "Pairwise distance by lag (train test gen)",
            "source_path": _BOND_LENGTH_GEN,
        })
    if os.path.isfile(os.path.join(run_root, _TRAINING_VIDEO)):
        blocks.append({"type": "training_video", "name": "Training Video", "source_path": _TRAINING_VIDEO})
    _finalize_dashboard_block_titles(blocks)
    return blocks


def _append_rmsd_analysis_blocks(run_root: str, blocks: list[dict[str, str]]) -> None:
    """Append RMSD dashboard blocks from analysis/rmsd/ (rmsd_distributions.png)."""
    rmsd_root = os.path.join(run_root, _ANALYSIS_DIR, _RMSD_DIR)
    if not os.path.isdir(rmsd_root):
        return
    base_rel = os.path.join(_ANALYSIS_DIR, _RMSD_DIR)
    gen_dir = os.path.join(rmsd_root, "gen")
    if os.path.isdir(gen_dir):
        for run_name in sorted(os.listdir(gen_dir)):
            fig_path = os.path.join(gen_dir, run_name, _RMSD_FIG)
            if os.path.isfile(fig_path):
                rel = os.path.join(base_rel, "gen", run_name, _RMSD_FIG)
                label = run_name if run_name != "default" else ""
                name = f"RMSD (gen) {label}".rstrip() or "RMSD (gen)"
                blocks.append({"type": "rmsd_gen", "name": name, "source_path": rel})
    recon_dir = os.path.join(rmsd_root, "recon")
    recon_fig = os.path.join(recon_dir, _RMSD_FIG)
    if os.path.isfile(recon_fig):
        rel = os.path.join(base_rel, "recon", _RMSD_FIG)
        blocks.append({"type": "rmsd_recon", "name": "RMSD (recon)", "source_path": rel})
    else:
        for subdir in (sorted(os.listdir(recon_dir)) if os.path.isdir(recon_dir) else []):
            subdir_path = os.path.join(recon_dir, subdir)
            if os.path.isdir(subdir_path):
                subdir_fig = os.path.join(subdir_path, _RMSD_FIG)
                if os.path.isfile(subdir_fig):
                    rel = os.path.join(base_rel, "recon", subdir, _RMSD_FIG)
                    blocks.append({"type": "rmsd_recon", "name": f"RMSD (recon) {subdir}", "source_path": rel})


def _blocks_for_euclideanizer_run(run_root: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = _blocks_for_distmap_run(run_root)
    # One latent block per run from analysis/latent/ (not per analysis metric)
    latent_dist = os.path.join(run_root, _LATENT_PLOTS_DIR, "latent_distribution.png")
    latent_corr = os.path.join(run_root, _LATENT_PLOTS_DIR, "latent_correlation.png")
    if os.path.isfile(latent_dist):
        blocks.append({"type": "latent_distribution", "name": "Latent distribution", "source_path": os.path.join(_LATENT_PLOTS_DIR, "latent_distribution.png")})
    if os.path.isfile(latent_corr):
        blocks.append({"type": "latent_correlation", "name": "Latent correlation", "source_path": os.path.join(_LATENT_PLOTS_DIR, "latent_correlation.png")})
    # Scores block when scoring/scores.json exists (spider is generated by scoring, not dashboard)
    scores_path = os.path.join(run_root, SCORING_DIR, "scores.json")
    spider_rel = os.path.join(SCORING_DIR, SCORES_SPIDER_FILENAME)
    spider_path = os.path.join(run_root, spider_rel)
    if os.path.isfile(scores_path):
        try:
            with open(scores_path, encoding="utf-8") as f:
                data = json.load(f)
            blocks.append({
                "type": "scores",
                "name": "Scores",
                "source_path": spider_rel if os.path.isfile(spider_path) else "",
                "scores_data": {
                    "overall_score": data.get("overall_score"),
                    "component_scores": data.get("component_scores") or {},
                    "missing": data.get("missing") or [],
                },
            })
        except (json.JSONDecodeError, OSError):
            pass
    _append_rmsd_analysis_blocks(run_root, blocks)
    q_root = os.path.join(run_root, _ANALYSIS_DIR, _ANALYSIS_Q_DIR)
    if os.path.isdir(q_root):
        gen_dir = os.path.join(q_root, "gen")
        if os.path.isdir(gen_dir):
            for run_name in sorted(os.listdir(gen_dir)):
                fig_path = os.path.join(gen_dir, run_name, _Q_FIG)
                if os.path.isfile(fig_path):
                    rel = os.path.join(_ANALYSIS_DIR, _ANALYSIS_Q_DIR, "gen", run_name, _Q_FIG)
                    label = run_name if run_name != "default" else ""
                    name = f"Q (gen) {label}".rstrip() or "Q (gen)"
                    blocks.append({"type": "q_gen", "name": name, "source_path": rel})
        recon_dir = os.path.join(q_root, "recon")
        recon_fig = os.path.join(recon_dir, _Q_FIG)
        if os.path.isfile(recon_fig):
            rel = os.path.join(_ANALYSIS_DIR, _ANALYSIS_Q_DIR, "recon", _Q_FIG)
            blocks.append({"type": "q_recon", "name": "Q (recon)", "source_path": rel})
        else:
            for subdir in (sorted(os.listdir(recon_dir)) if os.path.isdir(recon_dir) else []):
                subdir_path = os.path.join(recon_dir, subdir)
                if os.path.isdir(subdir_path):
                    subdir_fig = os.path.join(subdir_path, _Q_FIG)
                    if os.path.isfile(subdir_fig):
                        rel = os.path.join(_ANALYSIS_DIR, _ANALYSIS_Q_DIR, "recon", subdir, _Q_FIG)
                        blocks.append({"type": "q_recon", "name": f"Q (recon) {subdir}", "source_path": rel})
    _append_clustering_analysis_blocks(run_root, blocks, _COORD_CLUSTERING_DIR, "coord_clustering", "Coord clustering")
    _append_clustering_analysis_blocks(run_root, blocks, _DISTMAP_CLUSTERING_DIR, "distmap_clustering", "Distmap clustering", include_latent=False)
    _finalize_dashboard_block_titles(blocks)
    return blocks


def _append_clustering_analysis_blocks(
    run_root: str,
    blocks: list[dict[str, str]],
    subdir: str,
    type_prefix: str,
    display_name: str,
    *,
    include_latent: bool = False,
) -> None:
    """Append clustering dashboard blocks from analysis/<subdir>/ (gen, recon; optionally latent for distmap)."""
    clust_root = os.path.join(run_root, _ANALYSIS_DIR, subdir)
    if not os.path.isdir(clust_root):
        return
    base_rel = os.path.join(_ANALYSIS_DIR, subdir)
    gen_dir = os.path.join(clust_root, "gen")
    if os.path.isdir(gen_dir):
        for run_name in sorted(os.listdir(gen_dir)):
            run_dir = os.path.join(gen_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            for fig_name in _CLUSTERING_FIGS_ORDERED:
                fig_path = os.path.join(run_dir, fig_name)
                if os.path.isfile(fig_path):
                    rel = os.path.join(base_rel, "gen", run_name, fig_name)
                    label = fig_name.replace(".png", "").replace("_", " ").title()
                    if run_name == "default":
                        name = f"{display_name} (gen) — {label}"
                    else:
                        name = f"{display_name} (gen) {run_name} — {label}"
                    blocks.append({"type": f"{type_prefix}_gen", "name": name, "source_path": rel})
    recon_dir = os.path.join(clust_root, "recon")
    # Recon: same figure set and order as gen (pure, mixed, mixing_analysis, rmse_similarity)
    if os.path.isdir(recon_dir):
        main_recon_run = None
        for fig_name in _CLUSTERING_FIGS_ORDERED:
            fig_path = os.path.join(recon_dir, fig_name)
            if os.path.isfile(fig_path):
                rel = os.path.join(base_rel, "recon", fig_name)
                label = fig_name.replace(".png", "").replace("_", " ").title()
                blocks.append({"type": f"{type_prefix}_recon", "name": f"{display_name} (recon) — {label}", "source_path": rel})
        for subdir_name in (sorted(os.listdir(recon_dir)) if os.path.isdir(recon_dir) else []):
            subdir_path = os.path.join(recon_dir, subdir_name)
            if not os.path.isdir(subdir_path):
                continue
            for fig_name in _CLUSTERING_FIGS_ORDERED:
                fig_path = os.path.join(subdir_path, fig_name)
                if os.path.isfile(fig_path):
                    rel = os.path.join(base_rel, "recon", subdir_name, fig_name)
                    label = fig_name.replace(".png", "").replace("_", " ").title()
                    blocks.append({"type": f"{type_prefix}_recon", "name": f"{display_name} (recon) {subdir_name} — {label}", "source_path": rel})
    if include_latent:
        latent_fig = os.path.join(recon_dir, "latent_distribution.png")
        if os.path.isfile(latent_fig):
            rel = os.path.join(base_rel, "recon", "latent_distribution.png")
            blocks.append({"type": "latent_distribution", "name": f"Latent distribution ({display_name})", "source_path": rel})
            latent_corr = os.path.join(recon_dir, "latent_correlation.png")
            if os.path.isfile(latent_corr):
                rel = os.path.join(base_rel, "recon", "latent_correlation.png")
                blocks.append({"type": "latent_correlation", "name": f"Latent correlation ({display_name})", "source_path": rel})
        else:
            for subdir_name in (sorted(os.listdir(recon_dir)) if os.path.isdir(recon_dir) else []):
                subdir_path = os.path.join(recon_dir, subdir_name)
                latent_sub = os.path.join(subdir_path, "latent_distribution.png")
                if os.path.isfile(latent_sub):
                    rel = os.path.join(base_rel, "recon", subdir_name, "latent_distribution.png")
                    blocks.append({"type": "latent_distribution", "name": f"Latent distribution ({display_name}) {subdir_name}", "source_path": rel})
                    latent_corr = os.path.join(subdir_path, "latent_correlation.png")
                    if os.path.isfile(latent_corr):
                        rel = os.path.join(base_rel, "recon", subdir_name, "latent_correlation.png")
                        blocks.append({"type": "latent_correlation", "name": f"Latent correlation ({display_name}) {subdir_name}", "source_path": rel})


def _scan_runs(base_output_dir: str) -> list[dict[str, Any]]:
    runs = []
    if not os.path.isdir(base_output_dir):
        return runs

    for seed_name in sorted(os.listdir(base_output_dir)):
        parsed = _parse_seed_output_dir(seed_name)
        if parsed is None:
            continue
        seed_num, seed_group_id, split_token = parsed
        seed_dir = os.path.join(base_output_dir, seed_name)
        if not os.path.isdir(seed_dir):
            continue
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue

        seed_id = seed_group_id
        seed_children = []
        seed_blocks = []
        training_split_val = _training_split_value_for_seed_dir(seed_dir, split_token)

        for dm_name in sorted(os.listdir(distmap_dir), key=lambda x: (len(x), x)):
            if not dm_name.isdigit():
                continue
            dm_index = int(dm_name)
            dm_run_root = os.path.join(distmap_dir, dm_name)
            dm_model_dir = _run_config_dir(dm_run_root)
            dm_cfg = load_run_config(dm_model_dir)
            label_short, label_long = _label_from_distmap_config(
                dm_cfg, seed_num, dm_index, seed_group_id=seed_group_id, split_token=split_token
            )

            dm_id = f"{seed_group_id}_dm_{dm_index}"
            dm_blocks = _blocks_for_distmap_run(dm_run_root)
            dm_children = []

            eu_dir = os.path.join(dm_run_root, "euclideanizer")
            if os.path.isdir(eu_dir):
                for eu_name in sorted(os.listdir(eu_dir), key=lambda x: (len(x), x)):
                    if not eu_name.isdigit():
                        continue
                    eu_index = int(eu_name)
                    eu_run_root = os.path.join(eu_dir, eu_name)
                    eu_model_dir = _run_config_dir(eu_run_root)
                    eu_cfg = load_run_config(eu_model_dir)
                    eu_short, eu_long = _label_from_euclideanizer_config(
                        eu_cfg, seed_num, dm_index, eu_index,
                        seed_group_id=seed_group_id, split_token=split_token,
                    )
                    eu_id = f"{dm_id}_eu_{eu_index}"
                    eu_blocks = _blocks_for_euclideanizer_run(eu_run_root)
                    eu_comp = _load_component_scores(eu_run_root)
                    eu_entry: dict[str, Any] = {
                        "id": eu_id,
                        "level": "euclideanizer",
                        "label_short": eu_short,
                        "label_long": eu_long,
                        "parent_id": dm_id,
                        "children_ids": [],
                        "blocks": eu_blocks,
                        "run_root": eu_run_root,
                        "params": (eu_cfg["euclideanizer"] if eu_cfg else {}),
                        "parent_params": (dm_cfg["distmap"] if dm_cfg else {}),
                        "split_seed": seed_num,
                        "distmap_index": dm_index,
                        "euclideanizer_index": eu_index,
                        "component_scores": eu_comp,
                    }
                    if training_split_val is not None:
                        eu_entry["training_split"] = training_split_val
                    runs.append(eu_entry)
                    dm_children.append(eu_id)
                    for b in eu_blocks:
                        seed_blocks.append({**b, "run_id": eu_id})

            dm_entry: dict[str, Any] = {
                "id": dm_id,
                "level": "distmap",
                "label_short": label_short,
                "label_long": label_long,
                "parent_id": seed_id,
                "children_ids": dm_children,
                "blocks": dm_blocks,
                "run_root": dm_run_root,
                "params": (dm_cfg["distmap"] if dm_cfg else {}),
                "split_seed": seed_num,
                "distmap_index": dm_index,
            }
            if training_split_val is not None:
                dm_entry["training_split"] = training_split_val
            runs.append(dm_entry)
            seed_children.append(dm_id)
            for b in dm_blocks:
                seed_blocks.append({**b, "run_id": dm_id})

        seed_label_short = f"Seed {seed_num} · split {split_token}" if split_token else f"Seed {seed_num}"
        seed_label_long = f"seed {seed_num} training_split {split_token}" if split_token else f"seed {seed_num}"
        runs.append({
            "id": seed_id,
            "level": "seed",
            "label_short": seed_label_short,
            "label_long": seed_label_long,
            "parent_id": None,
            "children_ids": seed_children,
            "blocks": seed_blocks,
            "run_root": None,
            "params": {},
        })
    return runs


def _block_asset_slug(block: dict, run_id: str) -> str:
    t = block["type"]
    name = block["name"]
    source = block["source_path"]
    if t == "reconstruction":
        return "reconstruction"
    if t in ("recon_statistics", "recon_statistics_train", "recon_statistics_test"):
        return "recon_statistics_train" if "train" in name.lower() else "recon_statistics_test"
    if t == "gen_variance":
        var = source.replace("gen_variance_", "").split(".")[0].split("/")[-1]
        return f"gen_variance_{var}"
    if t == "training_video":
        return "training_video"
    # rmsd_gen / rmsd_recon: stable slug from source_path so asset names don't depend on display name
    if t == "rmsd_gen":
        run_name = source.split(os.sep)[-2] if os.sep in source else "default"
        safe = re.sub(r"[^\w.-]", "_", run_name)
        return f"rmsd__gen__{safe}"
    if t == "rmsd_recon":
        if source.endswith(_RMSD_FIG):
            parts = source.replace("\\", "/").split("/")
            if len(parts) >= 2 and parts[-2] != "recon":
                sub = re.sub(r"[^\w.-]", "_", parts[-2])
                return f"rmsd__recon__{sub}"
        return "rmsd__recon_"
    if t == "bond_length_by_genomic_distance_gen":
        return "bond_length_gen"
    if t == "bond_length_by_genomic_distance_train":
        return "bond_length_train"
    if t == "bond_length_by_genomic_distance_test":
        return "bond_length_test"
    return re.sub(r"[^\w.-]", "_", name.lower().replace(" ", "_"))


def _block_extension(source_path: str) -> str:
    return ".mp4" if source_path.strip().lower().endswith(".mp4") else ".png"


def _copy_assets_and_update_paths(
    runs: list[dict], assets_dir: str, scoring_save_pdf_copy: bool
) -> list[dict[str, Any]]:
    """Copy each block's source file to assets_dir (once per unique asset); return runs with blocks containing path (assets/...) only."""
    runs_by_id = {r["id"]: r for r in runs}
    copied = set()
    os.makedirs(assets_dir, exist_ok=True)
    out_runs = []
    for run in runs:
        blocks_out = []
        for block in run["blocks"]:
            if block.get("type") == "scores":
                sd = block.get("scores_data") or {}
                owning_id = block.get("run_id") or run["id"]
                owner = runs_by_id.get(owning_id)
                spider_name = f"{owning_id}_scores_spider.png"
                path_out = None
                src = os.path.join(owner["run_root"], block.get("source_path", "")) if owner and owner.get("run_root") and block.get("source_path") else ""
                if src and os.path.isfile(src):
                    dest = os.path.join(assets_dir, spider_name)
                    if spider_name not in copied:
                        shutil.copy2(src, dest)
                        copied.add(spider_name)
                    path_out = f"assets/{spider_name}"
                blocks_out.append({
                    "type": "scores",
                    "name": block.get("name", "Scores"),
                    "path": path_out,
                    "scores_data": sd,
                })
                continue
            owning_id = block.get("run_id") or run["id"]
            owner = runs_by_id.get(owning_id)
            if not owner or not owner.get("run_root"):
                continue
            src = os.path.join(owner["run_root"], block.get("source_path", ""))
            if not src or not os.path.isfile(src):
                continue
            slug = _block_asset_slug(block, owning_id)
            ext = _block_extension(block["source_path"])
            asset_name = f"{owning_id}_{slug}{ext}"
            if asset_name not in copied:
                shutil.copy2(src, os.path.join(assets_dir, asset_name))
                copied.add(asset_name)
            own_id = block.get("run_id") or run["id"]
            owner_run = runs_by_id.get(own_id)
            comp = (owner_run or {}).get("component_scores") or {}
            strip = None
            if owner_run and owner_run.get("level") == "euclideanizer":
                strip = _score_strip_for_plot_block(block["type"], block.get("name") or "", comp)
            bo: dict[str, Any] = {
                "type": block["type"],
                "name": block["name"],
                "path": f"assets/{asset_name}",
            }
            if strip:
                bo["score_strip"] = strip
            blocks_out.append(bo)
        entry: dict[str, Any] = {
            "id": run["id"],
            "level": run["level"],
            "label_short": run["label_short"],
            "label_long": run["label_long"],
            "parent_id": run["parent_id"],
            "children_ids": run["children_ids"],
            "blocks": blocks_out,
            "params": run.get("params") or {},
            "parent_params": run.get("parent_params") or {},
        }
        # Preserved for Vary aspect (training_split) and similar; stripped from blocks/run_root only.
        for _k in ("training_split", "split_seed", "distmap_index", "euclideanizer_index"):
            if _k in run:
                entry[_k] = run[_k]
        if run.get("level") == "euclideanizer" and run.get("component_scores"):
            entry["component_scores"] = _json_safe_component_scores(run["component_scores"])
        out_runs.append(entry)
    return out_runs


def _make_manifest(base_output_dir: str, runs: list[dict]) -> dict:
    base_path = os.path.basename(os.path.abspath(base_output_dir)) or "output"
    return {
        "generated_at": datetime.now().isoformat(),
        "base_path": base_path,
        "runs": runs,
        "score_component_catalog": _manifest_score_component_catalog(),
    }


def _write_manifest(dashboard_dir: str, manifest: dict) -> None:
    path = os.path.join(dashboard_dir, MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _html_content(manifest: dict) -> str:
    # Embed manifest so the dashboard works when opened from file:// (e.g. after download)
    manifest_js = json.dumps(manifest).replace("</", "<\\/")
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pipeline Dashboard</title>
  <style>
    :root {
      --bg-page: #1a1a1a;
      --bg-toolbar: #222;
      --bg-card: #252525;
      --border: #444;
      --text: #e0e0e0;
      --text-secondary: #b0b0b0;
      --text-muted: #888;
      --accent: #5a9;
      --accent-block: #6ab;
    }
    body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; background: var(--bg-page); color: var(--text); }
    body > header { margin: 0 1.5rem; padding-top: 1rem; }
    h1 { font-size: 1.5rem; margin: 0 0 0.25rem 0; }
    header p { color: var(--text-muted); font-size: 0.9rem; margin: 0; }
    .dashboard-toolbar { background: var(--bg-toolbar); border-bottom: 1px solid var(--border); padding: 1rem 1.25rem; margin: 0; }
    .dashboard-toolbar .controls { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin: 0; }
    .dashboard-toolbar .control-group { display: inline-flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center; }
    .dashboard-toolbar .control-group-label { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; margin-right: 0.25rem; }
    .dashboard-toolbar .control-group-sep { color: var(--border); font-weight: 300; user-select: none; }
    .dashboard-toolbar label { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.9rem; color: var(--text-secondary); }
    .dashboard-toolbar select { padding: 0.4rem 0.5rem; background: #2a2a2a; color: var(--text); border: 1px solid var(--border); border-radius: 4px; min-width: 180px; max-width: 320px; font-size: 0.9rem; }
    .content { margin: 1.5rem 1.5rem 2rem; }
    .view-compare { display: inline-flex; }
    .view-aspect, .view-scoreplot { display: none; }
    .view-aspect.visible, .view-scoreplot.visible { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.9rem; }
    .score-plot-panel { max-width: 920px; margin: 0 auto; }
    .score-plot-svg-wrap { overflow-x: auto; margin-top: 1rem; }
    .score-plot-svg { display: block; margin: 0 auto; background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border); }
    .score-plot-axis { font-size: 11px; fill: var(--text-muted); }
    .score-plot-title { font-size: 0.95rem; color: var(--text-secondary); margin-bottom: 0.5rem; text-align: center; }
    .score-plot-legend { display: flex; flex-wrap: wrap; gap: 0.6rem 1.2rem; justify-content: center; margin: 0.75rem 0 0.25rem; font-size: 0.78rem; }
    .score-plot-legend-item { display: flex; align-items: center; gap: 0.35rem; }
    .score-plot-legend-swatch { width: 14px; height: 3px; border-radius: 1px; }
    .score-plot-legend-swatch-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; border: 1px solid #333; box-sizing: border-box; }
    .score-plot-row-inline { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.75rem; }
    .score-plot-dual-check { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: var(--text-secondary); user-select: none; }
    .score-plot-dual-check input { accent-color: var(--accent); width: 1rem; height: 1rem; }
    .score-plot-dual-layout { display: flex; flex-direction: row; flex-wrap: nowrap; justify-content: center; align-items: stretch; gap: 1.25rem; width: 100%; max-width: 100%; margin-top: 0.35rem; overflow-x: auto; box-sizing: border-box; }
    .score-plot-dual-layout__plot { flex: 0 0 auto; flex-shrink: 0; display: flex; align-items: flex-start; }
    .score-plot-dual-layout__plot .score-plot-svg { display: block; margin-left: auto; margin-right: auto; flex-shrink: 0; }
    .score-plot-legend-cell { flex: 0 0 auto; flex-shrink: 0; width: 11rem; min-width: 9.5rem; max-width: 14rem; padding: 0.65rem 0.8rem; background: #1e1e1e; border: 1px solid var(--border); border-radius: 8px; border-left: 3px solid var(--accent); box-sizing: border-box; display: flex; flex-direction: column; min-height: var(--score-plot-h, 430px); height: var(--score-plot-h, 430px); max-height: var(--score-plot-h, 430px); overflow: hidden; }
    .score-plot-legend-cell-title { font-size: 0.74rem; font-weight: 700; color: var(--accent); margin-bottom: 0.15rem; letter-spacing: 0.02em; flex-shrink: 0; }
    .score-plot-legend-cell-aspect { font-size: 0.68rem; color: var(--text-muted); margin-bottom: 0.5rem; line-height: 1.25; flex-shrink: 0; }
    .score-plot-legend-cell__grid { flex: 1; min-height: 0; overflow-y: auto; display: grid; grid-template-columns: 1fr; gap: 0.28rem 0.5rem; align-content: start; }
    .score-plot-legend-cell__grid--2col { grid-template-columns: 1fr 1fr; }
    .score-plot-legend-cell-row { display: flex; align-items: center; gap: 0.4rem; font-size: 0.72rem; color: var(--text-secondary); line-height: 1.2; min-width: 0; }
    .score-plot-legend-cell-row span:last-child { word-break: break-word; }
    .score-plot-tooltip-fixed { position: fixed; z-index: 200; max-width: 380px; max-height: 70vh; overflow: auto; padding: 0.65rem 0.85rem; background: #161616; border: 1px solid var(--accent); border-radius: 8px; box-shadow: 0 10px 32px rgba(0,0,0,0.5); font-size: 0.76rem; pointer-events: none; visibility: hidden; opacity: 0; transition: opacity 0.1s; }
    .score-plot-tooltip-fixed.visible { visibility: visible; opacity: 1; }
    .score-plot-tooltip-fixed .tt-run-id { font-size: 0.68rem; color: var(--accent); margin-bottom: 0.4rem; word-break: break-all; }
    .score-plot-tooltip-fixed table { width: 100%; border-collapse: collapse; }
    .score-plot-tooltip-fixed th { text-align: left; padding: 0.2rem 0.5rem 0.2rem 0; color: var(--text-muted); font-weight: 500; vertical-align: top; }
    .score-plot-tooltip-fixed td { padding: 0.2rem 0; color: var(--text); word-break: break-word; }
    .score-plot-tooltip-fixed tr.section-row th { padding-top: 0.45rem; color: var(--accent); }
    .run-card { background: var(--bg-card); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }
    .run-card-title { font-size: 1.15rem; font-weight: 600; color: var(--text); margin: 0 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }
    .block { margin-bottom: 1.75rem; }
    .block-score-strip { margin-top: 0.65rem; padding: 0.65rem 0.75rem; background: linear-gradient(165deg, rgba(255,179,71,0.07) 0%, rgba(30,30,30,0.95) 45%); border-radius: 8px; border: 1px solid var(--border); border-left: 3px solid var(--accent); }
    .score-strip-section + .score-strip-section { margin-top: 0.55rem; padding-top: 0.55rem; border-top: 1px solid rgba(255,255,255,0.06); }
    .score-strip-group-title { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.055em; color: var(--accent); margin-bottom: 0.4rem; opacity: 0.95; }
    .score-strip-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr)); gap: 0.45rem 1rem; align-items: start; }
    .score-strip-cell { display: flex; flex-direction: column; gap: 0.12rem; min-width: 0; }
    .score-strip-label { font-size: 0.72rem; color: var(--text-muted); line-height: 1.3; }
    .score-strip-value { font-family: ui-monospace, "Cascadia Code", "SF Mono", monospace; font-size: 0.86rem; font-weight: 600; color: #eaeaea; letter-spacing: 0.02em; }
    .score-strip-value.missing { font-weight: 500; color: var(--text-muted); }
    .aspect-cell .block-score-strip { margin-top: 0.5rem; text-align: left; max-width: 100%; }
    .block:last-child { margin-bottom: 0; }
    .block-title { display: block; font-size: 1rem; color: var(--text-secondary); margin: 0 0 0.6rem 0; padding: 0.4rem 0.6rem; font-weight: 600; background: #333; border-radius: 4px; border-left: 3px solid var(--accent-block); }
    .scores-overall { margin-bottom: 0.75rem; font-size: 1rem; }
    .scores-categories table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .scores-categories th, .scores-categories td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--border); }
    .scores-categories th { color: var(--text-muted); font-weight: 500; }
    .scores-empty { color: var(--text-muted); margin: 0.5rem 0; font-size: 0.9rem; }
    .compare { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .compare .column { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; background: var(--bg-card); }
    .compare .column .run-card { margin-bottom: 0; }
    .compare .column-header { font-size: 0.9rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.75rem; padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); }
    .compare-param-wrap { margin-bottom: 1rem; }
    .content-single { max-width: 56rem; margin: 0 auto; }
    .block img, .block video { max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; }
    .block .placeholder { padding: 2rem; text-align: center; background: #2a2a2a; border-radius: 8px; color: var(--text-muted); }
    .aspect-section { margin-bottom: 2.5rem; scroll-margin-top: 1rem; }
    .aspect-section h3 { font-size: 1.15rem; font-weight: 600; color: var(--text-secondary); margin: 0 0 0.75rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); letter-spacing: 0.02em; }
    .aspect-axis-caption { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; }
    .aspect-axis-table-wrap { overflow-x: auto; }
    .aspect-axis-table { width: max-content; min-width: 100%; border-collapse: collapse; }
    .aspect-axis-table th { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); padding: 0.5rem; text-align: center; border-bottom: 1px solid var(--border); }
    .aspect-axis-table tbody tr:hover { background: rgba(255,255,255,0.02); }
    .aspect-axis-table td { padding: 0.5rem; vertical-align: top; border-bottom: 1px solid #333; }
    .aspect-axis-table td.context-label { font-size: 0.8rem; color: var(--text-secondary); min-width: 11rem; max-width: 28rem; white-space: pre-wrap; word-wrap: break-word; vertical-align: top; }
    .aspect-axis-table .aspect-cell { text-align: center; min-width: 320px; max-width: 420px; }
    .aspect-axis-table .aspect-cell img, .aspect-axis-table .aspect-cell video { max-width: 100%; height: auto; display: block; margin: 0 auto; }
    .aspect-axis-table .aspect-cell .empty { min-height: 4rem; background: var(--bg-card); border-radius: 4px; color: var(--text-muted); font-size: 0.8rem; display: flex; align-items: center; justify-content: center; }
    .aspect-section-nav { position: sticky; top: 0; z-index: 10; background: var(--bg-toolbar); border-bottom: 1px solid var(--border); padding: 0.5rem 0; margin: -0.5rem 0 1rem 0; }
    .aspect-section-nav-inner { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center; font-size: 0.85rem; }
    .aspect-section-nav-inner a { color: var(--accent); text-decoration: none; }
    .aspect-section-nav-inner a:hover { text-decoration: underline; }
    .empty-state { padding: 2.5rem 2rem; text-align: center; background: #252525; border-radius: 8px; color: var(--text-muted); font-size: 0.95rem; border: 1px dashed var(--border); }
    .empty-state .empty-state-title { color: var(--text-secondary); font-weight: 500; margin-bottom: 0.5rem; }
    .content-error { padding: 1.25rem; background: rgba(180,60,60,0.12); border-left: 4px solid #c44; border-radius: 4px; color: var(--text); }
    .seed-group { margin-bottom: 1.5rem; }
    .seed-group:last-child { margin-bottom: 0; }
    .seed-group-title { font-size: 0.9rem; font-weight: 600; color: var(--text-muted); margin-bottom: 0.5rem; padding-bottom: 0.25rem; }
    .skip-link { position: absolute; left: -9999px; z-index: 100; padding: 0.5rem 1rem; background: var(--accent); color: #111; font-weight: 600; border-radius: 4px; }
    .skip-link:focus { left: 1rem; top: 1rem; }
    .breadcrumb { font-size: 0.9rem; color: var(--text-muted); margin: 0 0 1rem 0; padding: 0.35rem 0; }
    .breadcrumb a { color: var(--accent); text-decoration: none; }
    .breadcrumb a:hover { text-decoration: underline; }
    .breadcrumb span { color: var(--text-secondary); margin: 0 0.35rem; }
    .run-list { list-style: none; margin: 0; padding: 0; }
    .run-list li { border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; background: var(--bg-card); display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.75rem; }
    .run-list li:hover { border-color: var(--accent); background: #2a2a2a; }
    .run-list .run-list-params { font-size: 0.8rem; color: var(--text-secondary); font-family: ui-monospace, monospace; flex: 1 1 100%; margin-top: 0.25rem; }
    .run-list .run-list-actions { display: flex; gap: 0.5rem; flex-shrink: 0; }
    .run-list .btn { padding: 0.35rem 0.75rem; font-size: 0.85rem; border-radius: 4px; cursor: pointer; border: 1px solid var(--border); background: #2a2a2a; color: var(--text); }
    .run-list .btn:hover { background: var(--accent); color: #111; border-color: var(--accent); }
    .run-list .btn-primary { background: var(--accent); color: #111; border-color: var(--accent); }
    .param-panel { background: #1e1e1e; border: 1px solid var(--border); border-radius: 6px; margin-bottom: 1.25rem; overflow: hidden; }
    .param-panel summary { padding: 0.6rem 1rem; cursor: pointer; font-weight: 600; font-size: 0.9rem; color: var(--text-secondary); }
    .param-panel summary:hover { background: #2a2a2a; }
    .param-panel table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .param-panel th { text-align: left; padding: 0.4rem 1rem; color: var(--text-muted); font-weight: 500; width: 12rem; }
    .param-panel td { padding: 0.4rem 1rem; color: var(--text); }
    .param-panel tr.section-row th { padding-top: 0.75rem; color: var(--accent); }
    .browse-level-title { font-size: 1.1rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.75rem; }
    .filter-bar { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center; margin-bottom: 1rem; padding: 0.5rem 0; }
    .filter-bar label { font-size: 0.85rem; color: var(--text-muted); }
    .filter-bar select { min-width: 100px; }
    .radar-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1.25rem; }
    .radar-grid-cell { position: relative; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; text-align: center; cursor: pointer; }
    .radar-grid-cell:hover { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
    .radar-grid-cell img { max-width: 100%; height: auto; display: block; margin: 0 auto 0.5rem; }
    .radar-grid-score { font-size: 1rem; font-weight: 600; color: var(--accent); }
    .radar-grid-tooltip { position: absolute; left: 50%; transform: translateX(-50%); top: 100%; margin-top: 0.5rem; width: max-content; max-width: 320px; padding: 0.75rem; background: #1e1e1e; border: 1px solid var(--border); border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); font-size: 0.8rem; text-align: left; pointer-events: none; visibility: hidden; opacity: 0; transition: visibility 0.15s, opacity 0.15s; z-index: 20; }
    .radar-grid-cell:hover .radar-grid-tooltip { visibility: visible; opacity: 1; }
    .radar-grid-tooltip table { width: 100%; border-collapse: collapse; }
    .radar-grid-tooltip th { text-align: left; padding: 0.25rem 0.5rem; color: var(--text-muted); font-weight: 500; }
    .radar-grid-tooltip td { padding: 0.25rem 0.5rem; color: var(--text); }
    .radar-grid-tooltip tr.section-row th { padding-top: 0.5rem; color: var(--accent); }
    @media (max-width: 768px) {
      .compare { grid-template-columns: 1fr; }
      .content { margin-left: 1rem; margin-right: 1rem; }
      .dashboard-toolbar .controls { gap: 0.75rem; }
      .aspect-section-nav-inner { font-size: 0.8rem; }
    }
  </style>
</head>
<body>
  <a href="#content" class="skip-link">Skip To Content</a>
  <header>
    <h1 id="title">Pipeline Dashboard</h1>
    <p id="generated"></p>
  </header>
  <div class="dashboard-toolbar" role="toolbar" aria-label="Dashboard Controls">
    <div class="controls">
      <span class="control-group control-group-primary">
        <label for="viewMode">View</label>
        <select id="viewMode" aria-label="View Mode">
          <option value="browse">Browse</option>
          <option value="detail">Detail</option>
          <option value="compare">Compare</option>
          <option value="aspect">Vary Aspect</option>
          <option value="radar_grid">Radar Grid</option>
          <option value="score_plot">Score Vs Aspect</option>
        </select>
      </span>
      <span class="control-group control-group-sep" aria-hidden="true">|</span>
      <span class="view-browse control-group" id="viewBrowse"></span>
      <span class="view-compare control-group" id="viewCompare" style="display:none;">
        <span class="control-group-label">Compare</span>
        <label for="level">Type</label>
        <select id="level"><option value="distmap">DistMaps</option><option value="euclideanizer">Euclideanizers</option></select>
        <span id="levelRunCount" class="control-group-label" aria-live="polite"></span>
        <label for="runA">Run A (Left)</label>
        <select id="runA"></select>
        <label for="runB">Run B (Right)</label>
        <select id="runB"><option value="">— None —</option></select>
      </span>
      <span class="view-aspect control-group" id="viewAspect" style="display:none;">
        <span class="control-group-label">Vary By</span>
        <label for="levelAspect">Level</label>
        <select id="levelAspect"><option value="distmap">DistMap</option><option value="euclideanizer">Euclideanizer</option></select>
        <label for="aspect">Aspect (X-Axis)</label>
        <select id="aspect"></select>
      </span>
      <span class="view-scoreplot control-group" id="viewScorePlot" style="display:none;">
        <span class="control-group-label">Score Plot</span>
        <label class="score-plot-dual-check"><input type="checkbox" id="scorePlotDualMode" /> <span>Two Scores (Color By Aspect)</span></label>
        <span id="scorePlotRowSingle" class="score-plot-row-inline">
          <label for="scorePlotAspect">Aspect (X)</label>
          <select id="scorePlotAspect"></select>
          <label for="scorePlotScore">Score (Y)</label>
          <select id="scorePlotScore"></select>
        </span>
        <span id="scorePlotRowDual" class="score-plot-row-inline" style="display:none;">
          <label for="scorePlotScoreX">Score (X)</label>
          <select id="scorePlotScoreX"></select>
          <label for="scorePlotScoreY">Score (Y)</label>
          <select id="scorePlotScoreY"></select>
          <label for="scorePlotAspectColor">Color By (Aspect)</label>
          <select id="scorePlotAspectColor"></select>
        </span>
      </span>
    </div>
  </div>
  <div id="breadcrumb" class="breadcrumb" style="display:none;"></div>
  <main class="content" id="content" role="main"></main>
  <script>window.__DASHBOARD_MANIFEST__ = """ + manifest_js + """;</script>
  <script>
    let manifest = null;
    const state = { viewMode: 'browse', browseSeedId: null, browseDmId: null, detailRunId: null, compareRunA: null, compareRunB: null, compareLevel: 'euclideanizer' };
    const levelEl = document.getElementById('level');
    const runAEl = document.getElementById('runA');
    const runBEl = document.getElementById('runB');
    const contentEl = document.getElementById('content');
    const titleEl = document.getElementById('title');
    const generatedEl = document.getElementById('generated');
    const viewModeEl = document.getElementById('viewMode');
    const viewBrowseEl = document.getElementById('viewBrowse');
    const viewCompareEl = document.getElementById('viewCompare');
    const viewAspectEl = document.getElementById('viewAspect');
    const levelAspectEl = document.getElementById('levelAspect');
    const aspectEl = document.getElementById('aspect');
    const viewScorePlotEl = document.getElementById('viewScorePlot');
    const scorePlotAspectEl = document.getElementById('scorePlotAspect');
    const scorePlotScoreEl = document.getElementById('scorePlotScore');
    const scorePlotDualModeEl = document.getElementById('scorePlotDualMode');
    const scorePlotRowSingle = document.getElementById('scorePlotRowSingle');
    const scorePlotRowDual = document.getElementById('scorePlotRowDual');
    const scorePlotScoreXEl = document.getElementById('scorePlotScoreX');
    const scorePlotScoreYEl = document.getElementById('scorePlotScoreY');
    const scorePlotAspectColorEl = document.getElementById('scorePlotAspectColor');
    let scorePlotTooltipEl = null;
    const breadcrumbEl = document.getElementById('breadcrumb');

    function getRunById(id) {
      if (!manifest || !manifest.runs || !id) return null;
      return manifest.runs.find(r => r.id === id) || null;
    }
    function getSeeds() {
      if (!manifest || !manifest.runs) return [];
      return manifest.runs.filter(r => r.level === 'seed').sort((a, b) => a.id.localeCompare(b.id));
    }
    function getDistMapsForSeed(seedId) {
      if (!manifest || !manifest.runs || !seedId) return [];
      const seed = getRunById(seedId);
      if (!seed || !seed.children_ids) return [];
      return seed.children_ids.map(dmId => getRunById(dmId)).filter(Boolean);
    }
    function getEuclideanizersForDistMap(dmId) {
      if (!manifest || !manifest.runs || !dmId) return [];
      const dm = getRunById(dmId);
      if (!dm || !dm.children_ids) return [];
      return dm.children_ids.map(euId => getRunById(euId)).filter(Boolean);
    }
    function formatParams(params) {
      if (!params || typeof params !== 'object') return '';
      return Object.keys(params).sort().map(k => k + '=' + params[k]).join(', ');
    }
    function paramPanelHtml(run) {
      if (!run) return '';
      let html = '<details class="param-panel" open><summary>Parameters</summary><table>';
      if (run.parent_params && Object.keys(run.parent_params).length) {
        html += '<tr class="section-row"><th colspan="2">Frozen DistMap</th></tr>';
        Object.keys(run.parent_params).sort().forEach(k => { html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.parent_params[k])) + '</td></tr>'; });
      }
      if (run.params && Object.keys(run.params).length) {
        html += '<tr class="section-row"><th colspan="2">' + (run.level === 'distmap' ? 'DistMap' : 'Euclideanizer') + '</th></tr>';
        Object.keys(run.params).sort().forEach(k => { html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.params[k])) + '</td></tr>'; });
      }
      html += '</table></details>';
      return html;
    }
    function renderBreadcrumb(parts) {
      if (!breadcrumbEl) return;
      if (!parts || parts.length === 0) { breadcrumbEl.style.display = 'none'; breadcrumbEl.innerHTML = ''; return; }
      breadcrumbEl.style.display = 'block';
      let html = '';
      parts.forEach((p, i) => {
        if (i) html += '<span> \u203a </span>';
        if (p.runId) html += '<a href="#" data-run-id="' + escapeHtml(p.runId) + '">' + escapeHtml(p.label) + '</a>';
        else if (p.label === 'Seeds') html += '<a href="#" data-reset-browse="1">' + escapeHtml(p.label) + '</a>';
        else html += '<span>' + escapeHtml(p.label) + '</span>';
      });
      breadcrumbEl.innerHTML = html;
      breadcrumbEl.querySelectorAll('a[data-run-id]').forEach(a => {
        a.addEventListener('click', function(e) {
          e.preventDefault();
          const runId = this.getAttribute('data-run-id');
          const run = getRunById(runId);
          if (run && run.level === 'seed') { state.browseSeedId = runId; state.browseDmId = null; state.viewMode = 'browse'; viewModeEl.value = 'browse'; renderBrowse(); }
          else if (run && run.level === 'distmap') { state.browseSeedId = run.parent_id || null; state.browseDmId = runId; state.viewMode = 'browse'; viewModeEl.value = 'browse'; renderBrowse(); }
          else { state.detailRunId = runId; state.viewMode = 'detail'; viewModeEl.value = 'detail'; updateContent(); }
        });
      });
      breadcrumbEl.querySelectorAll('a[data-reset-browse]').forEach(a => {
        a.addEventListener('click', function(e) { e.preventDefault(); state.browseSeedId = null; state.browseDmId = null; renderBrowse(); });
      });
    }

    function formatGeneratedAt(iso) {
      if (!iso) return '';
      try {
        const d = new Date(iso);
        if (isNaN(d.getTime())) return iso;
        const mon = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][d.getMonth()];
        const h = d.getHours(), m = d.getMinutes();
        const ampm = h >= 12 ? 'PM' : 'AM';
        const h12 = h % 12 || 12;
        return mon + ' ' + d.getDate() + ', ' + d.getFullYear() + ', ' + h12 + ':' + (m < 10 ? '0' : '') + m + ' ' + ampm;
      } catch (e) { return iso; }
    }

    function getRunsByLevel() {
      if (!manifest || !manifest.runs) return [];
      const level = levelEl.value;
      if (level === 'seed') return manifest.runs.filter(r => r.level === 'seed');
      return manifest.runs.filter(r => r.level === level);
    }

    function getRunsByLevelForAspect() {
      if (!manifest || !manifest.runs) return [];
      const level = levelAspectEl ? levelAspectEl.value : 'euclideanizer';
      return manifest.runs.filter(r => r.level === level);
    }

    function getAspectOptions(runs) {
      if (!runs.length) return [];
      const keys = {};
      runs.forEach(r => {
        const p = r.params || {};
        Object.keys(p).forEach(k => {
          if (!keys[k]) keys[k] = new Set();
          const v = p[k];
          if (v !== undefined && v !== null) keys[k].add(typeof v === 'number' ? v : String(v));
        });
      });
      const fromParams = Object.keys(keys).filter(k => keys[k].size >= 2).sort();
      const tsSeen = new Set();
      runs.forEach(r => {
        if (r.training_split !== undefined && r.training_split !== null && !isNaN(Number(r.training_split))) tsSeen.add(Number(r.training_split));
      });
      const hasTrainingSplitAspect = tsSeen.size >= 2;
      if (hasTrainingSplitAspect) return ['training_split'].concat(fromParams.filter(function(k) { return k !== 'training_split'; }));
      return fromParams;
    }

    function stableParamJson(params) {
      const p = params || {};
      return JSON.stringify(Object.keys(p).sort().reduce(function(o, k) { o[k] = p[k]; return o; }, {}));
    }

    function trainingSplitContextKey(r) {
      var pp = stableParamJson(r.parent_params);
      var p = stableParamJson(r.params);
      if (r.level === 'euclideanizer') {
        return 'eu|' + r.split_seed + '|' + r.distmap_index + '|' + r.euclideanizer_index + '|' + pp + '|' + p;
      }
      return 'dm|' + r.split_seed + '|' + r.distmap_index + '|' + p;
    }

    function groupRunsByContext(runs, aspect) {
      if (aspect === 'training_split') {
        const byContext = {};
        const aspectValues = new Set();
        runs.forEach(r => {
          if (r.training_split === undefined || r.training_split === null) return;
          if (r.split_seed === undefined || r.distmap_index === undefined) return;
          if (r.level === 'euclideanizer' && r.euclideanizer_index === undefined) return;
          const ck = trainingSplitContextKey(r);
          if (!byContext[ck]) byContext[ck] = [];
          byContext[ck].push(r);
          aspectValues.add(Number(r.training_split));
        });
        const sortedValues = Array.from(aspectValues).sort(function(a, b) { return a - b; });
        const result = {};
        Object.keys(byContext).forEach(function(ck) {
          result[ck] = byContext[ck].slice().sort(function(a, b) { return Number(a.training_split) - Number(b.training_split); });
        });
        return { byContext: result, aspectValues: sortedValues };
      }
      const byContext = {};
      const aspectValues = new Set();
      runs.forEach(r => {
        const p = r.params || {};
        if (p[aspect] === undefined || p[aspect] === null) return;
        const contextKey = JSON.stringify(Object.keys(p).sort().reduce((o, k) => { if (k !== aspect) o[k] = p[k]; return o; }, {}));
        if (!byContext[contextKey]) byContext[contextKey] = [];
        byContext[contextKey].push(r);
        aspectValues.add(typeof p[aspect] === 'number' ? p[aspect] : p[aspect]);
      });
      const sortedValues = Array.from(aspectValues).sort((a, b) => (Number(a) - Number(b)) || String(a).localeCompare(String(b)));
      const result = {};
      Object.keys(byContext).forEach(ck => {
        const list = byContext[ck].sort((a, b) => {
          const va = a.params[aspect];
          const vb = b.params[aspect];
          return (Number(va) - Number(vb)) || String(va).localeCompare(String(vb));
        });
        result[ck] = list;
      });
      return { byContext: result, aspectValues: sortedValues };
    }

    function getBlockByName(run, blockName) {
      if (!run || !run.blocks) return null;
      return run.blocks.find(b => b.name === blockName) || null;
    }

    function slugifySectionId(name) {
      return 'section-' + (name || '').toLowerCase().replace(/\\s+/g, '-').replace(/[^a-z0-9-]/g, '');
    }

    function renderAspectView(container, runs, aspect) {
      if (!container || !runs.length || !aspect) { container.innerHTML = '<div class="empty-state"><span class="empty-state-title">Select Level And Aspect</span><p>Choose Level And Aspect (X-Axis) Above To See Outputs Along That Parameter.</p></div>'; return; }
      const { byContext, aspectValues } = groupRunsByContext(runs, aspect);
      const contextKeys = Object.keys(byContext);
      if (!contextKeys.length || !aspectValues.length) { container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No Variation On This Aspect</span><p>No Runs Have Multiple Values For This Parameter. Try Another Aspect Or Level.</p></div>'; return; }
      const allBlockNames = [];
      const seen = {};
      runs.forEach(r => { (r.blocks || []).forEach(b => { if (b.name && !seen[b.name]) { seen[b.name] = true; allBlockNames.push(b.name); } }); });
      const blockToType = {};
      runs.forEach(r => { (r.blocks || []).forEach(b => { if (b.name && !blockToType[b.name]) blockToType[b.name] = b.type || ''; }); });
      allBlockNames.sort((a, b) => blockTypeOrder(blockToType[a] || '', a) - blockTypeOrder(blockToType[b] || '', b) || a.localeCompare(b));
      const axisCaption = aspect + ' \u2192';
      let navHtml = '<div class="aspect-section-nav"><div class="aspect-section-nav-inner"><span class="control-group-label">Sections:</span>';
      allBlockNames.forEach(blockName => {
        const id = slugifySectionId(blockName);
        navHtml += '<a href="#' + id + '">' + escapeHtml(blockName) + '</a>';
      });
      navHtml += '</div></div>';
      const aspectCaption = aspect === 'training_split'
        ? 'training_split (train fraction — share of structures in train set)'
        : aspect;
      let html = navHtml + '<p class="aspect-axis-caption">Aspect: <strong>' + escapeHtml(aspectCaption) + '</strong> (x-axis)</p>';
      allBlockNames.forEach(blockName => {
        const sectionId = slugifySectionId(blockName);
        html += '<div class="aspect-section" id="' + sectionId + '"><h3>' + escapeHtml(blockName) + '</h3>';
        html += '<div class="aspect-axis-table-wrap"><table class="aspect-axis-table"><thead><tr><th class="context-label">Context</th>';
        aspectValues.forEach(v => { html += '<th>' + escapeHtml(String(v)) + '</th>'; });
        html += '</tr></thead><tbody>';
        contextKeys.forEach(ck => {
          const runList = byContext[ck];
          const run0 = runList[0];
          let contextLabel = '';
          if (run0) {
            if (aspect === 'training_split') {
              contextLabel = 'seed=' + run0.split_seed + ', distmap=' + run0.distmap_index;
              if (run0.level === 'euclideanizer') contextLabel += ', euclideanizer=' + run0.euclideanizer_index;
              contextLabel += '\\n';
            }
            const paramsMinusAspect = Object.keys(run0.params || {}).filter(k => k !== aspect).sort().reduce((o, k) => { o[k] = run0.params[k]; return o; }, {});
            if (run0.parent_params && Object.keys(run0.parent_params).length) contextLabel += 'Frozen DistMap: ' + formatParams(run0.parent_params) + '\\n';
            contextLabel += (run0.level === 'distmap' ? 'DistMap' : 'Euclideanizer') + ': ' + formatParams(paramsMinusAspect);
          } else contextLabel = ck.slice(0, 200);
          html += '<tr><td class="context-label">' + escapeHtml(contextLabel) + '</td>';
          aspectValues.forEach(aspectVal => {
            let run = null;
            if (aspect === 'training_split') {
              run = runList.find(function(r) { return Number(r.training_split) === Number(aspectVal); });
            } else {
              run = runList.find(r => (r.params && (r.params[aspect] === aspectVal || (typeof r.params[aspect] === 'number' && Number(r.params[aspect]) === Number(aspectVal)))));
            }
            const block = run ? getBlockByName(run, blockName) : null;
            html += '<td class="aspect-cell">';
            if (block) {
              if ((block.path || '').toLowerCase().endsWith('.mp4')) html += '<video controls preload="metadata" src="' + block.path + '" style="max-width:100%"></video>';
              else html += '<img loading="lazy" src="' + block.path + '" alt="' + escapeHtml(blockName) + '">';
              if (block.score_strip) html += scoreStripHtml(block.score_strip);
            } else html += '<div class="empty">—</div>';
            html += '</td>';
          });
          html += '</tr>';
        });
        html += '</tbody></table></div></div>';
      });
      container.innerHTML = html;
    }

    function shortLabel(s, maxLen) {
      if (!s) return '';
      const str = String(s);
      return str.length <= (maxLen || 50) ? str : str.slice(0, (maxLen || 50) - 1) + '\u2026';
    }

    function fillRunSelect(sel, includeEmpty) {
      const runs = getRunsByLevel();
      const current = sel.value;
      sel.innerHTML = includeEmpty ? '<option value="">— None —</option>' : '';
      runs.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.id;
        const full = r.label_short || r.id || '';
        opt.textContent = shortLabel(full, 50);
        opt.title = full;
        sel.appendChild(opt);
      });
      if (current && runs.some(r => r.id === current)) sel.value = current;
      else if (runs.length) sel.selectedIndex = includeEmpty ? 1 : 0;
      updateLevelRunCount();
    }

    function updateLevelRunCount() {
      const el = document.getElementById('levelRunCount');
      if (!el) return;
      const runs = getRunsByLevel();
      el.textContent = runs.length ? '(' + runs.length + ' run' + (runs.length !== 1 ? 's' : '') + ')' : '';
    }

    function escapeHtml(s) {
      if (!s) return '';
      return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function scoreStripHtml(strip) {
      if (!strip || !strip.groups || !strip.groups.length) return '';
      var h = '<div class="block-score-strip" role="region" aria-label="Component scores for this figure">';
      strip.groups.forEach(function(g) {
        h += '<div class="score-strip-section">';
        if (g.title) h += '<div class="score-strip-group-title">' + escapeHtml(g.title) + '</div>';
        h += '<div class="score-strip-grid">';
        (g.items || []).forEach(function(it) {
          var v = it.value;
          var vs = (typeof v === 'number' && isFinite(v)) ? Number(v).toFixed(4) : '\u2014';
          var miss = (vs === '\u2014') ? ' missing' : '';
          h += '<div class="score-strip-cell"><span class="score-strip-label">' + escapeHtml(it.label || '') + '</span><span class="score-strip-value' + miss + '">' + escapeHtml(vs) + '</span></div>';
        });
        h += '</div></div>';
      });
      h += '</div>';
      return h;
    }

    const CLUSTERING_SUB_ORDER = ['Pure Dendrograms', 'Mixed Dendrograms', 'Mixing Analysis', 'Rmse Similarity'];
    function clusteringSubOrder(name) {
      if (!name) return 999;
      const suffix = name.indexOf(' — ') >= 0 ? name.split(' — ').pop() : name;
      const i = CLUSTERING_SUB_ORDER.indexOf(suffix);
      return i >= 0 ? i : 999;
    }
    function isClusteringType(type) {
      return type === 'coord_clustering_gen' || type === 'coord_clustering_recon' || type === 'distmap_clustering_gen' || type === 'distmap_clustering_recon';
    }
    function blockTypeOrder(type, name) {
      if (type === 'scores') return -1;
      if (type === 'latent_distribution' || type === 'latent_correlation') return 100;
      const order = [
        'reconstruction', 'recon_statistics_train', 'recon_statistics_test', 'gen_variance',
        'bond_length_by_genomic_distance_train', 'bond_length_by_genomic_distance_test', 'bond_length_by_genomic_distance_gen',
        'training_video',
        'rmsd_gen', 'rmsd_recon',
        null,
        'q_gen', 'q_recon',
        null,
        'coord_clustering_gen', 'coord_clustering_recon',
        null,
        'distmap_clustering_gen', 'distmap_clustering_recon',
        null
      ];
      const i = order.indexOf(type);
      return i >= 0 ? i : 999;
    }

    function renderBlocks(run, container, label) {
      if (!run || !run.blocks || !run.blocks.length) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No outputs</span><p>This run has no plots or videos.</p></div>';
        return;
      }
      const sorted = [...run.blocks].sort((a, b) => {
        const typeDiff = blockTypeOrder(a.type, a.name) - blockTypeOrder(b.type, b.name);
        if (typeDiff !== 0) return typeDiff;
        if (isClusteringType(a.type) && isClusteringType(b.type)) {
          const subDiff = clusteringSubOrder(a.name) - clusteringSubOrder(b.name);
          if (subDiff !== 0) return subDiff;
        }
        return (a.name || '').localeCompare(b.name || '');
      });
      let html = '<div class="run-card"><h2 class="run-card-title">' + escapeHtml(run.label_short || run.id || 'Run') + '</h2>';
      sorted.forEach(b => {
        html += '<div class="block"><div class="block-title">' + escapeHtml(b.name) + '</div>';
        if (b.type === 'scores') {
          const sd = b.scores_data || {};
          const overall = sd.overall_score;
          const comps = sd.component_scores || {};
          const missing = sd.missing || [];
          let scoreHtml = '';
          if (typeof overall === 'number' && !Number.isNaN(overall)) {
            scoreHtml += '<div class="scores-overall">Overall: <strong>' + escapeHtml(String(Number(overall).toFixed(4))) + '</strong></div>';
          } else {
            const missingTip = missing.length ? ('Missing: ' + missing.join(', ')) : 'Not all 30 components present';
            scoreHtml += '<div class="scores-overall" title="' + escapeHtml(missingTip) + '">Overall: <strong>Missing data</strong></div>';
          }
          if (b.path) {
            scoreHtml += '<img loading="lazy" src="' + b.path + '" alt="Scores spider">';
          }
          if (!b.path && Object.keys(comps).length) {
            scoreHtml += '<div class="scores-components"><table><thead><tr><th>Component</th><th>Score</th></tr></thead><tbody>';
            for (const [k, v] of Object.entries(comps)) {
              if (typeof v === 'number' && !Number.isNaN(v)) {
                scoreHtml += '<tr><td>' + escapeHtml(k) + '</td><td>' + escapeHtml(Number(v).toFixed(4)) + '</td></tr>';
              }
            }
            scoreHtml += '</tbody></table></div>';
          }
          if (!b.path && !Object.keys(comps).length) scoreHtml += '<p class="scores-empty">No score data.</p>';
          html += scoreHtml;
        } else {
          const isVideo = (b.path || '').toLowerCase().endsWith('.mp4');
          if (isVideo) {
            html += '<video controls preload="metadata" src="' + (b.path || '') + '" style="max-width:100%"></video>';
          } else {
            html += '<img loading="lazy" src="' + (b.path || '') + '" alt="' + escapeHtml(b.name) + '">';
          }
          if (b.score_strip) html += scoreStripHtml(b.score_strip);
        }
        html += '</div>';
      });
      html += '</div>';
      container.innerHTML = html;
    }

    function getRunsToShow(run, runs) {
      if (!run || !runs) return [];
      if (run.level !== 'seed' || !run.children_ids || !run.children_ids.length) return [run];
      const list = [];
      run.children_ids.forEach(dmId => {
        const dmRun = runs.find(r => r.id === dmId);
        if (dmRun) {
          list.push(dmRun);
          (dmRun.children_ids || []).forEach(euId => {
            const euRun = runs.find(r => r.id === euId);
            if (euRun) list.push(euRun);
          });
        }
      });
      return list;
    }

    function getRunsToShowGrouped(run, runs) {
      if (!run || run.level !== 'seed' || !run.children_ids || !run.children_ids.length) return null;
      const groups = [];
      run.children_ids.forEach(dmId => {
        const dmRun = runs.find(r => r.id === dmId);
        if (!dmRun) return;
        const groupRuns = [dmRun];
        (dmRun.children_ids || []).forEach(euId => {
          const euRun = runs.find(r => r.id === euId);
          if (euRun) groupRuns.push(euRun);
        });
        const m = dmId.match(/dm_(\\d+)$/);
        groups.push({ title: 'DistMap ' + (m ? m[1] : ''), runs: groupRuns });
      });
      return groups;
    }

    function renderRunOrSeedRuns(run, container, runs) {
      if (!container) return;
      container.innerHTML = '';
      if (!run) { container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No run selected</span><p>Choose a run from the dropdown above.</p></div>'; return; }
      const groups = getRunsToShowGrouped(run, runs);
      if (groups && groups.length) {
        groups.forEach(grp => {
          const groupDiv = document.createElement('div');
          groupDiv.className = 'seed-group';
          groupDiv.innerHTML = '<div class="seed-group-title">' + escapeHtml(grp.title) + '</div>';
          const cardsWrap = document.createElement('div');
          groupDiv.appendChild(cardsWrap);
          container.appendChild(groupDiv);
          grp.runs.forEach(r => {
            const wrap = document.createElement('div');
            wrap.className = 'run-card-wrap';
            cardsWrap.appendChild(wrap);
            renderBlocks(r, wrap);
          });
        });
      } else {
        const toShow = getRunsToShow(run, runs);
        if (!toShow.length) { container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No outputs</span><p>This run has no plots or videos.</p></div>'; return; }
        toShow.forEach(r => {
          const wrap = document.createElement('div');
          wrap.className = 'run-card-wrap';
          container.appendChild(wrap);
          renderBlocks(r, wrap);
        });
      }
    }

    function applyFilters(runs, filters) {
      if (!runs.length || !filters) return runs;
      return runs.filter(r => {
        const p = r.params || {};
        for (const k in filters) {
          if (filters[k] === '' || filters[k] == null) continue;
          const v = p[k];
          if (v === undefined || v === null) return false;
          if (String(v) !== String(filters[k])) return false;
        }
        return true;
      });
    }

    function renderBrowse() {
      const runs = manifest.runs || [];
      const seedId = state.browseSeedId;
      const dmId = state.browseDmId;
      const seeds = getSeeds();
      if (!seeds.length) { contentEl.innerHTML = '<div class="empty-state"><span class="empty-state-title">No runs</span><p>No seed runs found in manifest.</p></div>'; return; }
      if (!seedId) {
        renderBreadcrumb([{ label: 'Seeds' }]);
        let html = '<div class="browse-level-title">Seeds</div><ul class="run-list">';
        seeds.forEach(s => {
          const dms = getDistMapsForSeed(s.id);
          const euCount = dms.reduce((n, dm) => n + (dm.children_ids ? dm.children_ids.length : 0), 0);
          html += '<li><span>' + escapeHtml(s.label_short || s.id) + '</span><span class="run-list-params">' + dms.length + ' DistMap(s), ' + euCount + ' Euclideanizer(s)</span>';
          html += '<div class="run-list-actions"><button type="button" class="btn btn-primary" data-browse-seed="' + escapeHtml(s.id) + '">Open</button></div></li>';
        });
        html += '</ul>';
        contentEl.innerHTML = html;
        contentEl.querySelectorAll('[data-browse-seed]').forEach(btn => {
          btn.addEventListener('click', function() { state.browseSeedId = this.getAttribute('data-browse-seed'); state.browseDmId = null; renderBrowse(); });
        });
        return;
      }
      const dms = getDistMapsForSeed(seedId);
      const seedRun = getRunById(seedId);
      if (!dmId) {
        renderBreadcrumb([{ label: 'Seeds', runId: null }, { label: seedRun ? (seedRun.label_short || seedId) : seedId, runId: seedId }]);
        let html = '<div class="browse-level-title">DistMap Runs</div><ul class="run-list">';
        dms.forEach(dm => {
          const paramStr = formatParams(dm.params);
          const euCount = (dm.children_ids || []).length;
          html += '<li><span>' + escapeHtml(dm.label_short || dm.id) + '</span><span class="run-list-params">' + escapeHtml(paramStr) + (paramStr ? ' \u2022 ' : '') + euCount + ' Euclideanizer(s)</span>';
          html += '<div class="run-list-actions"><button type="button" class="btn" data-view-dm="' + escapeHtml(dm.id) + '">View DistMap</button><button type="button" class="btn" data-set-compare-a="' + escapeHtml(dm.id) + '">Set As A</button><button type="button" class="btn" data-set-compare-b="' + escapeHtml(dm.id) + '">Set As B</button><button type="button" class="btn btn-primary" data-browse-dm="' + escapeHtml(dm.id) + '">Euclideanizers</button></div></li>';
        });
        html += '</ul>';
        contentEl.innerHTML = html;
        contentEl.querySelectorAll('[data-browse-dm]').forEach(btn => {
          btn.addEventListener('click', function() { state.browseDmId = this.getAttribute('data-browse-dm'); renderBrowse(); });
        });
        contentEl.querySelectorAll('[data-view-dm]').forEach(btn => {
          btn.addEventListener('click', function() { state.detailRunId = this.getAttribute('data-view-dm'); state.viewMode = 'detail'; viewModeEl.value = 'detail'; updateContent(); });
        });
        contentEl.querySelectorAll('[data-set-compare-a]').forEach(btn => {
          btn.addEventListener('click', function() {
            const id = this.getAttribute('data-set-compare-a');
            const run = getRunById(id);
            if (!run || run.level === 'seed') return;
            state.compareLevel = run.level; state.compareRunA = id; state.compareRunB = null;
            levelEl.value = run.level; fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); runAEl.value = id; runBEl.value = '';
          });
        });
        contentEl.querySelectorAll('[data-set-compare-b]').forEach(btn => {
          btn.addEventListener('click', function() {
            const id = this.getAttribute('data-set-compare-b');
            const run = getRunById(id);
            if (!run || run.level === 'seed') return;
            if (run.level !== state.compareLevel) state.compareRunA = null;
            state.compareLevel = run.level; state.compareRunB = id;
            levelEl.value = run.level; fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); if (state.compareRunA) runAEl.value = state.compareRunA; runBEl.value = id;
          });
        });
        return;
      }
      const eus = getEuclideanizersForDistMap(dmId);
      const dmRun = getRunById(dmId);
      renderBreadcrumb([
        { label: 'Seeds', runId: null },
        { label: seedRun ? (seedRun.label_short || seedId) : seedId, runId: seedId },
        { label: dmRun ? (dmRun.label_short || dmId) : dmId, runId: dmId }
      ]);
      let html = '<div class="browse-level-title">Euclideanizer Runs</div><p style="margin-bottom:0.75rem;"><button type="button" class="btn" id="btnBackToDms">\u2190 Back To DistMap Runs</button></p><ul class="run-list">';
      eus.forEach(eu => {
        const paramStr = formatParams(eu.params);
        html += '<li><span>' + escapeHtml(eu.label_short || eu.id) + '</span><span class="run-list-params">' + escapeHtml(paramStr) + '</span>';
        html += '<div class="run-list-actions"><button type="button" class="btn btn-primary" data-view-run="' + escapeHtml(eu.id) + '">View</button><button type="button" class="btn" data-set-compare-a="' + escapeHtml(eu.id) + '">Set As A</button><button type="button" class="btn" data-set-compare-b="' + escapeHtml(eu.id) + '">Set As B</button></div></li>';
      });
      html += '</ul>';
      contentEl.innerHTML = html;
      const backBtn = contentEl.querySelector('#btnBackToDms');
      if (backBtn) backBtn.addEventListener('click', function() { state.browseDmId = null; renderBrowse(); });
      contentEl.querySelectorAll('[data-view-run]').forEach(btn => {
        btn.addEventListener('click', function() { state.detailRunId = this.getAttribute('data-view-run'); state.viewMode = 'detail'; viewModeEl.value = 'detail'; updateContent(); });
      });
      contentEl.querySelectorAll('[data-set-compare-a]').forEach(btn => {
        btn.addEventListener('click', function() {
          const id = this.getAttribute('data-set-compare-a');
          const run = getRunById(id);
          if (!run || run.level === 'seed') return;
          state.compareLevel = run.level; state.compareRunA = id; state.compareRunB = null;
          levelEl.value = run.level; fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); runAEl.value = id; runBEl.value = '';
        });
      });
      contentEl.querySelectorAll('[data-set-compare-b]').forEach(btn => {
        btn.addEventListener('click', function() {
          const id = this.getAttribute('data-set-compare-b');
          const run = getRunById(id);
          if (!run || run.level === 'seed') return;
          if (run.level !== state.compareLevel) state.compareRunA = null;
          state.compareLevel = run.level; state.compareRunB = id;
          levelEl.value = run.level; fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); if (state.compareRunA) runAEl.value = state.compareRunA; runBEl.value = id;
        });
      });
    }

    function renderDetail() {
      const run = getRunById(state.detailRunId);
      if (!run) { contentEl.innerHTML = '<div class="empty-state"><span class="empty-state-title">Run not found</span></div>'; return; }
      const parts = [];
      if (run.level === 'euclideanizer' && run.parent_id) {
        const dm = getRunById(run.parent_id);
        const seedId = dm && dm.parent_id ? dm.parent_id : null;
        const seed = seedId ? getRunById(seedId) : null;
        if (seed) parts.push({ label: seed.label_short || seedId, runId: seedId });
        if (dm) parts.push({ label: dm.label_short || run.parent_id, runId: dm.id });
      } else if (run.level === 'distmap' && run.parent_id) {
        const seed = getRunById(run.parent_id);
        if (seed) parts.push({ label: seed.label_short || run.parent_id, runId: run.parent_id });
      }
      parts.push({ label: run.label_short || run.id, runId: null });
      renderBreadcrumb([{ label: 'Seeds', runId: null }].concat(parts.map(p => ({ label: p.label, runId: p.runId }))));
      let html = '<div class="run-card"><h2 class="run-card-title">' + escapeHtml(run.label_short || run.id) + '</h2>';
      html += '<details class="param-panel" open><summary>Parameters</summary><table>';
      if (run.parent_params && Object.keys(run.parent_params).length) {
        html += '<tr class="section-row"><th colspan="2">Frozen DistMap</th></tr>';
        Object.keys(run.parent_params).sort().forEach(k => { html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.parent_params[k])) + '</td></tr>'; });
      }
      if (run.params && Object.keys(run.params).length) {
        html += '<tr class="section-row"><th colspan="2">' + (run.level === 'distmap' ? 'DistMap' : 'Euclideanizer') + '</th></tr>';
        Object.keys(run.params).sort().forEach(k => { html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.params[k])) + '</td></tr>'; });
      }
      html += '</table></details>';
      html += '<div class="run-list-actions" style="margin-bottom:1rem;"><button type="button" class="btn" id="btnSetCompareA">Set As A (Left)</button><button type="button" class="btn" id="btnSetCompareB">Set As B (Right)</button></div>';
      html += '<div id="detailBlocks"></div></div>';
      contentEl.innerHTML = html;
      document.getElementById('btnSetCompareA').addEventListener('click', function() {
        state.compareLevel = run.level; state.compareRunA = run.id; state.compareRunB = null;
        levelEl.value = run.level; fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); runAEl.value = run.id; runBEl.value = '';
      });
      document.getElementById('btnSetCompareB').addEventListener('click', function() {
        if (run.level !== state.compareLevel) state.compareRunA = null;
        state.compareLevel = run.level; state.compareRunB = run.id;
        levelEl.value = run.level; fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); if (state.compareRunA) runAEl.value = state.compareRunA; runBEl.value = run.id;
      });
      renderBlocks(run, document.getElementById('detailBlocks'));
    }

    function getEuclideanizerRunsWithScores() {
      if (!manifest || !manifest.runs) return [];
      return manifest.runs.filter(function(r) {
        return r.level === 'euclideanizer' && r.component_scores && typeof r.component_scores === 'object';
      });
    }

    function fillAspectSelectInto(el, aspects, curVal) {
      if (!el) return;
      el.innerHTML = '';
      if (!aspects.length) {
        var ox = document.createElement('option');
        ox.value = '';
        ox.textContent = '— No Varying Aspect —';
        el.appendChild(ox);
        return;
      }
      aspects.forEach(function(k) {
        var opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k === 'training_split' ? 'Training Split (Train Fraction)' : k;
        el.appendChild(opt);
      });
      if (curVal && aspects.indexOf(curVal) >= 0) el.value = curVal;
      else el.value = aspects[0];
    }

    function fillScorePlotDropdowns() {
      if (!scorePlotAspectEl || !scorePlotScoreEl) return;
      var eu = getEuclideanizerRunsWithScores();
      var aspects = getAspectOptions(eu);
      var curA = scorePlotAspectEl.value;
      var curAC = scorePlotAspectColorEl ? scorePlotAspectColorEl.value : '';
      var curS = scorePlotScoreEl.value;
      var curSX = scorePlotScoreXEl ? scorePlotScoreXEl.value : '';
      var curSY = scorePlotScoreYEl ? scorePlotScoreYEl.value : '';
      fillAspectSelectInto(scorePlotAspectEl, aspects, curA);
      fillAspectSelectInto(scorePlotAspectColorEl, aspects, curAC || curA);
      var catalog = manifest.score_component_catalog || [];
      function fillScoreSel(el, cur) {
        if (!el) return;
        el.innerHTML = '';
        catalog.forEach(function(c) {
          var o = document.createElement('option');
          o.value = c.id;
          o.textContent = c.label;
          el.appendChild(o);
        });
        if (cur && catalog.some(function(c) { return c.id === cur; })) el.value = cur;
        else if (catalog.length) el.value = catalog[0].id;
      }
      fillScoreSel(scorePlotScoreEl, curS);
      fillScoreSel(scorePlotScoreXEl, curSX);
      fillScoreSel(scorePlotScoreYEl, curSY);
      if (scorePlotScoreXEl && scorePlotScoreYEl && catalog.length > 1 && scorePlotScoreXEl.value === scorePlotScoreYEl.value) {
        scorePlotScoreYEl.value = catalog[1].id;
      }
      if (scorePlotRowSingle && scorePlotRowDual && scorePlotDualModeEl) {
        var d = scorePlotDualModeEl.checked;
        scorePlotRowSingle.style.display = d ? 'none' : 'inline-flex';
        scorePlotRowDual.style.display = d ? 'inline-flex' : 'none';
      }
    }

    function runTooltipTableHtml(run) {
      var html = '<div class="tt-run-id">' + escapeHtml(run.id || '') + '</div><table>';
      if (run.parent_params && Object.keys(run.parent_params).length) {
        html += '<tr class="section-row"><th colspan="2">Frozen DistMap</th></tr>';
        Object.keys(run.parent_params).sort().forEach(function(k) {
          html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.parent_params[k])) + '</td></tr>';
        });
      }
      if (run.params && Object.keys(run.params).length) {
        html += '<tr class="section-row"><th colspan="2">Euclideanizer</th></tr>';
        Object.keys(run.params).sort().forEach(function(k) {
          html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.params[k])) + '</td></tr>';
        });
      }
      html += '</table>';
      return html;
    }

    function legendLabelForContext(ck, run0) {
      if (!run0) return ck.slice(0, 56);
      var pp = formatParams(run0.parent_params || {});
      var p = formatParams(run0.params || {});
      var s = (pp ? pp + ' \u00b7 ' : '') + p;
      return s.length > 58 ? s.slice(0, 56) + '\u2026' : (s || 'Series');
    }

    function renderScorePlotDual(container) {
      var aspectKey = scorePlotAspectColorEl && scorePlotAspectColorEl.value;
      var scoreXKey = scorePlotScoreXEl && scorePlotScoreXEl.value;
      var scoreYKey = scorePlotScoreYEl && scorePlotScoreYEl.value;
      if (!aspectKey) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No Color Aspect</span><p>Need At Least Two Values For The Chosen Dimension Across Runs (Same As Score Vs Aspect).</p></div>';
        return;
      }
      var eu = getEuclideanizerRunsWithScores();
      var catalog = manifest.score_component_catalog || [];
      var labelX = (catalog.find(function(c) { return c.id === scoreXKey; }) || {}).label || scoreXKey;
      var labelY = (catalog.find(function(c) { return c.id === scoreYKey; }) || {}).label || scoreYKey;
      var aspectLabel = aspectKey === 'training_split' ? 'Training Split' : aspectKey;
      var pts = [];
      eu.forEach(function(r) {
        var av = aspectKey === 'training_split' ? r.training_split : (r.params || {})[aspectKey];
        if (av === undefined || av === null) return;
        var sx = r.component_scores[scoreXKey];
        var sy = r.component_scores[scoreYKey];
        if (sx == null || sy == null) return;
        sx = Number(sx); sy = Number(sy);
        if (!isFinite(sx) || !isFinite(sy)) return;
        pts.push({ run: r, x: sx, y: sy, aspect: av });
      });
      if (!pts.length) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No Points</span><p>No Runs Have Both Scores And The Color Aspect Set.</p></div>';
        return;
      }
      var aspectUnique = [];
      pts.forEach(function(p) {
        if (aspectUnique.map(String).indexOf(String(p.aspect)) < 0) aspectUnique.push(p.aspect);
      });
      var numA = aspectUnique.every(function(v) { return isFinite(Number(v)); });
      if (numA) aspectUnique.sort(function(a, b) { return Number(a) - Number(b); });
      else aspectUnique.sort(function(a, b) { return String(a).localeCompare(String(b)); });
      function colorForAspect(av) {
        var i = aspectUnique.map(String).indexOf(String(av));
        if (i < 0) return '#888888';
        if (numA && aspectUnique.length > 1) {
          var lo = Number(aspectUnique[0]), hi = Number(aspectUnique[aspectUnique.length - 1]);
          var t = (Number(av) - lo) / (hi - lo || 1);
          return 'hsl(' + String(Math.round(218 - t * 178)) + ',70%,' + String(Math.round(40 + t * 20)) + '%)';
        }
        return 'hsl(' + String((38 + i * 47) % 360) + ',72%,58%)';
      }
      var buck = {};
      pts.forEach(function(p) {
        var k = p.x.toFixed(5) + '|' + p.y.toFixed(5);
        if (!buck[k]) buck[k] = [];
        buck[k].push(p);
      });
      Object.keys(buck).forEach(function(k) {
        var arr = buck[k];
        if (arr.length > 1) {
          var step = 0.014;
          arr.forEach(function(p, i) {
            p._x = Math.max(0, Math.min(1, p.x + (i - (arr.length - 1) / 2) * step));
            p._y = Math.max(0, Math.min(1, p.y + ((i % 2) * 2 - 1) * step * 0.8));
          });
        } else { arr[0]._x = arr[0].x; arr[0]._y = arr[0].y; }
      });
      var xs = pts.map(function(p) { return p._x; });
      var ys = pts.map(function(p) { return p._y; });
      var xmin = Math.max(0, Math.min.apply(null, xs) - 0.04);
      var xmax = Math.min(1, Math.max.apply(null, xs) + 0.04);
      var ymin = Math.max(0, Math.min.apply(null, ys) - 0.04);
      var ymax = Math.min(1, Math.max.apply(null, ys) + 0.04);
      if (xmax - xmin < 0.07) { xmin = Math.max(0, xmin - 0.05); xmax = Math.min(1, xmax + 0.05); }
      if (ymax - ymin < 0.07) { ymin = Math.max(0, ymin - 0.05); ymax = Math.min(1, ymax + 0.05); }
      var W = 680, H = 430, ml = 54, mr = 54, mt = 30, mb = 54;
      var pw = W - ml - mr, ph = H - mt - mb;
      var xScale = function(x) { return ml + (x - xmin) / ((xmax - xmin) || 1) * pw; };
      var yScale = function(y) { return mt + ph - (y - ymin) / ((ymax - ymin) || 1) * ph; };
      var pointElems = [];
      var svg = '<svg class="score-plot-svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg">';
      svg += '<text class="score-plot-axis" x="' + (ml + pw / 2) + '" y="' + (H - 12) + '" text-anchor="middle">' + escapeHtml(labelX) + '</text>';
      svg += '<text class="score-plot-axis" transform="rotate(-90 16 ' + (mt + ph / 2) + ')" x="16" y="' + (mt + ph / 2) + '" text-anchor="middle">' + escapeHtml(labelY) + '</text>';
      var gxi, gyi;
      for (gyi = 0; gyi <= 4; gyi++) {
        var yy = ymin + (ymax - ymin) * gyi / 4;
        var py = yScale(yy);
        svg += '<line x1="' + ml + '" y1="' + py + '" x2="' + (ml + pw) + '" y2="' + py + '" stroke="#383838" stroke-width="1"/>';
        svg += '<text class="score-plot-axis" x="' + (ml - 6) + '" y="' + (py + 4) + '" text-anchor="end">' + yy.toFixed(3) + '</text>';
      }
      for (gxi = 0; gxi <= 4; gxi++) {
        var xx = xmin + (xmax - xmin) * gxi / 4;
        var px = xScale(xx);
        svg += '<line x1="' + px + '" y1="' + mt + '" x2="' + px + '" y2="' + (mt + ph) + '" stroke="#2a2a2a" stroke-width="1"/>';
        svg += '<text class="score-plot-axis" x="' + px + '" y="' + (H - mb + 26) + '" text-anchor="middle">' + xx.toFixed(3) + '</text>';
      }
      pts.forEach(function(p) {
        var col = colorForAspect(p.aspect);
        var cx = xScale(p._x), cy = yScale(p._y);
        pointElems.push({ run: p.run, sx: p.x, sy: p.y, aspect: p.aspect, aspectLabel: aspectLabel });
        var idx = pointElems.length - 1;
        svg += '<circle class="score-plot-pt" data-idx="' + idx + '" cx="' + cx + '" cy="' + cy + '" r="6" fill="' + col + '" stroke="#0a0a0a" stroke-width="1.5" style="cursor:crosshair"/>';
      });
      svg += '</svg>';
      var legGrid = '';
      aspectUnique.forEach(function(av) {
        legGrid += '<div class="score-plot-legend-cell-row"><span class="score-plot-legend-swatch-dot" style="background:' + colorForAspect(av) + '"></span><span>' + escapeHtml(String(av)) + '</span></div>';
      });
      var legGridClass = aspectUnique.length > 8 ? ' score-plot-legend-cell__grid--2col' : '';
      var aside = '<aside class="score-plot-legend-cell" aria-label="Color key"><div class="score-plot-legend-cell-title">Color Key</div><div class="score-plot-legend-cell-aspect">' + escapeHtml(aspectLabel) + '</div><div class="score-plot-legend-cell__grid' + legGridClass + '">' + legGrid + '</div></aside>';
      var title = '<p class="score-plot-title"><span style="color:var(--accent)">Two Scores</span> <span style="color:var(--text-muted)">·</span> Color By <strong>' + escapeHtml(aspectLabel) + '</strong></p>';
      container.innerHTML = title + '<div class="score-plot-dual-layout" style="--score-plot-h:' + H + 'px"><div class="score-plot-dual-layout__plot">' + svg + '</div>' + aside + '</div>';
      function hideTip() { if (scorePlotTooltipEl) scorePlotTooltipEl.classList.remove('visible'); }
      function showTipDual(e, idx) {
        var pt = pointElems[idx];
        if (!pt || !scorePlotTooltipEl) return;
        var head = '<div style="margin-bottom:0.35rem;font-weight:600;color:var(--accent);">Score (X): ' + escapeHtml(pt.sx.toFixed(4)) + ' &nbsp;|&nbsp; Score (Y): ' + escapeHtml(pt.sy.toFixed(4)) + '</div>';
        head += '<div style="margin-bottom:0.45rem;font-size:0.8rem;color:var(--text-secondary);">' + escapeHtml(pt.aspectLabel) + ': <strong>' + escapeHtml(String(pt.aspect)) + '</strong></div>';
        scorePlotTooltipEl.innerHTML = head + runTooltipTableHtml(pt.run);
        scorePlotTooltipEl.classList.add('visible');
        var tw = 400, th = 300;
        var lx = e.clientX + 16, ly = e.clientY + 16;
        if (lx + tw > window.innerWidth) lx = e.clientX - tw - 8;
        if (ly + th > window.innerHeight) ly = e.clientY - th - 8;
        scorePlotTooltipEl.style.left = Math.max(8, lx) + 'px';
        scorePlotTooltipEl.style.top = Math.max(8, ly) + 'px';
      }
      container.querySelectorAll('.score-plot-pt').forEach(function(circ) {
        circ.addEventListener('mouseenter', function(e) { showTipDual(e, parseInt(circ.getAttribute('data-idx'), 10)); });
        circ.addEventListener('mousemove', function(e) { showTipDual(e, parseInt(circ.getAttribute('data-idx'), 10)); });
        circ.addEventListener('mouseleave', hideTip);
      });
    }

    function renderScorePlotView(container) {
      if (!container) return;
      if (scorePlotTooltipEl && scorePlotTooltipEl.parentNode) scorePlotTooltipEl.parentNode.removeChild(scorePlotTooltipEl);
      scorePlotTooltipEl = document.createElement('div');
      scorePlotTooltipEl.className = 'score-plot-tooltip-fixed';
      document.body.appendChild(scorePlotTooltipEl);
      var eu = getEuclideanizerRunsWithScores();
      if (!eu.length) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No Scored Euclideanizer Runs</span><p>Enable Scoring And Rebuild The Dashboard So Runs Include <code>component_scores</code>.</p></div>';
        return;
      }
      fillScorePlotDropdowns();
      if (scorePlotDualModeEl && scorePlotDualModeEl.checked) {
        renderScorePlotDual(container);
        return;
      }
      var aspectKey = scorePlotAspectEl.value;
      var scoreKey = scorePlotScoreEl.value;
      if (!aspectKey) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No Aspect To Vary</span><p>Need At Least Two Euclideanizer Runs Differing On One Parameter (Or Training Split).</p></div>';
        return;
      }
      var gdat = groupRunsByContext(eu, aspectKey);
      var byContext = gdat.byContext;
      var aspectVals = gdat.aspectValues;
      if (!aspectVals.length) {
        container.innerHTML = '<div class="empty-state">No data for this aspect.</div>';
        return;
      }
      var catalog = manifest.score_component_catalog || [];
      var scoreLabel = (catalog.find(function(c) { return c.id === scoreKey; }) || {}).label || scoreKey;
      var aspectLabel = aspectKey === 'training_split' ? 'Training Split' : aspectKey;
      var series = [];
      Object.keys(byContext).forEach(function(ck) {
        var runList = byContext[ck];
        var pts = [];
        runList.forEach(function(r) {
          var xv = aspectKey === 'training_split' ? r.training_split : (r.params || {})[aspectKey];
          if (xv === undefined || xv === null) return;
          var yv = r.component_scores && r.component_scores[scoreKey];
          if (yv === null || yv === undefined) return;
          var yn = Number(yv);
          if (!isFinite(yn)) return;
          var xn = Number(xv);
          pts.push({ x: xv, xn: xn, y: yn, run: r });
        });
        pts.sort(function(a, b) {
          if (isFinite(a.xn) && isFinite(b.xn) && a.xn !== b.xn) return a.xn - b.xn;
          return String(a.x).localeCompare(String(b.x));
        });
        if (pts.length) series.push({ ck: ck, pts: pts, run0: runList[0] });
      });
      if (!series.length) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No Points</span><p>No Runs Have The Selected Score At These Aspect Values.</p></div>';
        return;
      }
      var W = 760, H = 400, ml = 58, mr = 28, mt = 28, mb = 56;
      var pw = W - ml - mr, ph = H - mt - mb;
      var allY = [];
      series.forEach(function(s) { s.pts.forEach(function(p) { allY.push(p.y); }); });
      var ymin = Math.max(0, Math.min.apply(null, allY) - 0.04);
      var ymax = Math.min(1, Math.max.apply(null, allY) + 0.04);
      if (ymax - ymin < 0.08) { ymin = Math.max(0, ymin - 0.06); ymax = Math.min(1, ymax + 0.06); }
      var xNum = aspectVals.map(function(v) { return Number(v); });
      var numericX = aspectVals.every(function(v, i) { return isFinite(xNum[i]); });
      var xmin = Math.min.apply(null, xNum), xmax = Math.max.apply(null, xNum);
      var xScale;
      var nCat = aspectVals.length;
      if (!numericX || (xmax - xmin < 1e-12 && nCat > 1)) {
        xScale = function(raw) {
          var i = aspectVals.indexOf(raw);
          if (i < 0) i = aspectVals.map(String).indexOf(String(raw));
          return ml + (i >= 0 ? (i + 0.5) / nCat * pw : ml + pw / 2);
        };
      } else {
        var xr = xmax - xmin || 1;
        xScale = function(raw) {
          var n = Number(raw);
          return ml + (isFinite(n) ? (n - xmin) / xr * pw : ml);
        };
      }
      var yScale = function(y) { return mt + ph - (y - ymin) / ((ymax - ymin) || 1) * ph; };
      var colors = function(si) { return 'hsl(' + String(38 + si * 47) + ',72%,58%)'; };
      var pointElems = [];
      var svg = '<svg class="score-plot-svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg">';
      svg += '<text class="score-plot-axis" x="' + (W / 2) + '" y="' + (H - 10) + '" text-anchor="middle">' + escapeHtml(aspectLabel) + '</text>';
      svg += '<text class="score-plot-axis" transform="rotate(-90 18 ' + (mt + ph / 2) + ')" x="18" y="' + (mt + ph / 2) + '" text-anchor="middle">' + escapeHtml(scoreLabel) + '</text>';
      for (var gi = 0; gi <= 4; gi++) {
        var yy = ymin + (ymax - ymin) * gi / 4;
        var py = yScale(yy);
        svg += '<line x1="' + ml + '" y1="' + py + '" x2="' + (ml + pw) + '" y2="' + py + '" stroke="#383838" stroke-width="1"/>';
        svg += '<text class="score-plot-axis" x="' + (ml - 6) + '" y="' + (py + 4) + '" text-anchor="end">' + yy.toFixed(2) + '</text>';
      }
      if (!numericX || (xmax - xmin < 1e-12 && nCat > 1)) {
        aspectVals.forEach(function(v, i) {
          var px = ml + (i + 0.5) / nCat * pw;
          svg += '<text class="score-plot-axis" x="' + px + '" y="' + (H - mb + 24) + '" text-anchor="middle">' + escapeHtml(String(v)) + '</text>';
        });
      } else {
        for (var gx = 0; gx <= 4; gx++) {
          var xx = xmin + (xmax - xmin) * gx / 4;
          var px = xScale(xx);
          var lab = (Math.abs(xx - Math.round(xx)) < 1e-6) ? String(Math.round(xx)) : Number(xx.toFixed(4));
          svg += '<text class="score-plot-axis" x="' + px + '" y="' + (H - mb + 24) + '" text-anchor="middle">' + escapeHtml(lab) + '</text>';
        }
      }
      series.forEach(function(ser, si) {
        var col = colors(si);
        if (ser.pts.length > 1) {
          var d = 'M ' + ser.pts.map(function(p) { return xScale(p.x) + ' ' + yScale(p.y); }).join(' L ');
          svg += '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="2" opacity="0.88"/>';
        }
        ser.pts.forEach(function(p) {
          var cx = xScale(p.x), cy = yScale(p.y);
          pointElems.push({ run: p.run, y: p.y });
          var idx = pointElems.length - 1;
          svg += '<circle class="score-plot-pt" data-idx="' + idx + '" cx="' + cx + '" cy="' + cy + '" r="5.5" fill="' + col + '" stroke="#0a0a0a" stroke-width="1.5" style="cursor:crosshair"/>';
        });
      });
      svg += '</svg>';
      var leg = '<div class="score-plot-legend">';
      series.forEach(function(ser, si) {
        leg += '<div class="score-plot-legend-item"><span class="score-plot-legend-swatch" style="background:' + colors(si) + '"></span><span>' + escapeHtml(legendLabelForContext(ser.ck, ser.run0)) + '</span></div>';
      });
      leg += '</div>';
      container.innerHTML = '<p class="score-plot-title">' + escapeHtml(scoreLabel) + ' <span style="color:var(--text-muted)">vs</span> ' + escapeHtml(aspectLabel) + '</p>' + leg + '<div class="score-plot-svg-wrap">' + svg + '</div>';
      function hideTip() { if (scorePlotTooltipEl) scorePlotTooltipEl.classList.remove('visible'); }
      function showTip(e, idx) {
        var pt = pointElems[idx];
        if (!pt || !scorePlotTooltipEl) return;
        scorePlotTooltipEl.innerHTML = '<div style="margin-bottom:0.4rem;font-weight:600;color:var(--accent);">Score (Y): ' + escapeHtml(pt.y.toFixed(4)) + '</div>' + runTooltipTableHtml(pt.run);
        scorePlotTooltipEl.classList.add('visible');
        var tw = 380, th = 280;
        var lx = e.clientX + 16, ly = e.clientY + 16;
        if (lx + tw > window.innerWidth) lx = e.clientX - tw - 8;
        if (ly + th > window.innerHeight) ly = e.clientY - th - 8;
        scorePlotTooltipEl.style.left = Math.max(8, lx) + 'px';
        scorePlotTooltipEl.style.top = Math.max(8, ly) + 'px';
      }
      container.querySelectorAll('.score-plot-pt').forEach(function(circ) {
        circ.addEventListener('mouseenter', function(e) { showTip(e, parseInt(circ.getAttribute('data-idx'), 10)); });
        circ.addEventListener('mousemove', function(e) { showTip(e, parseInt(circ.getAttribute('data-idx'), 10)); });
        circ.addEventListener('mouseleave', hideTip);
      });
    }

    function getRunsWithScores() {
      if (!manifest || !manifest.runs) return [];
      const list = [];
      manifest.runs.forEach(r => {
        if (r.level !== 'euclideanizer') return;
        const scoresBlock = (r.blocks || []).find(b => b.type === 'scores');
        if (!scoresBlock || !scoresBlock.path) return;
        const overall = scoresBlock.scores_data && typeof scoresBlock.scores_data.overall_score === 'number' && !Number.isNaN(scoresBlock.scores_data.overall_score) ? scoresBlock.scores_data.overall_score : null;
        list.push({ run: r, overallScore: overall, scoresBlock: scoresBlock });
      });
      list.sort((a, b) => {
        const sa = a.overallScore;
        const sb = b.overallScore;
        if (sa != null && sb != null) return sb - sa;
        if (sa != null) return -1;
        if (sb != null) return 1;
        return 0;
      });
      return list;
    }

    function paramTooltipHtml(run) {
      if (!run) return '';
      let html = '<table>';
      if (run.parent_params && Object.keys(run.parent_params).length) {
        html += '<tr class="section-row"><th colspan="2">Frozen DistMap</th></tr>';
        Object.keys(run.parent_params).sort().forEach(k => { html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.parent_params[k])) + '</td></tr>'; });
      }
      if (run.params && Object.keys(run.params).length) {
        html += '<tr class="section-row"><th colspan="2">Euclideanizer</th></tr>';
        Object.keys(run.params).sort().forEach(k => { html += '<tr><th>' + escapeHtml(k) + '</th><td>' + escapeHtml(String(run.params[k])) + '</td></tr>'; });
      }
      html += '</table>';
      return html;
    }

    function renderRadarGrid(container) {
      const list = getRunsWithScores();
      if (!list.length) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No Scored Runs</span><p>No Euclideanizer Runs With Scores (Radar Plots) Found. Run Scoring To See The Radar Grid.</p></div>';
        return;
      }
      let html = '<div class="radar-grid">';
      list.forEach(({ run, overallScore, scoresBlock }) => {
        const scoreLabel = overallScore != null ? Number(overallScore).toFixed(4) : 'Missing data';
        let tooltipHtml = paramTooltipHtml(run);
        const missing = (scoresBlock.scores_data && scoresBlock.scores_data.missing) || [];
        if (missing.length) {
          tooltipHtml += '<tr class="section-row"><th colspan="2">Missing components</th></tr><tr><td colspan="2" style="font-size:0.85em">' + escapeHtml(missing.join(', ')) + '</td></tr>';
        }
        html += '<div class="radar-grid-cell" data-run-id="' + escapeHtml(run.id) + '" role="button" tabindex="0" aria-label="View run details">';
        html += '<img loading="lazy" src="' + escapeHtml(scoresBlock.path) + '" alt="Scores">';
        html += '<div class="radar-grid-score">' + escapeHtml(scoreLabel) + '</div>';
        if (tooltipHtml) html += '<div class="radar-grid-tooltip" role="tooltip">' + tooltipHtml + '</div>';
        html += '</div>';
      });
      html += '</div>';
      container.innerHTML = html;
      container.querySelectorAll('.radar-grid-cell').forEach(cell => {
        function goToDetail() {
          const runId = cell.getAttribute('data-run-id');
          if (!runId) return;
          state.detailRunId = runId;
          state.viewMode = 'detail';
          viewModeEl.value = 'detail';
          updateContent();
        }
        cell.addEventListener('click', goToDetail);
        cell.addEventListener('keydown', function(e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goToDetail(); } });
      });
    }

    function updateContent() {
      if (!manifest) return;
      state.viewMode = viewModeEl ? viewModeEl.value : state.viewMode;
      const runs = manifest.runs || [];
      viewCompareEl.style.display = state.viewMode === 'compare' ? 'inline-flex' : 'none';
      viewAspectEl.style.display = state.viewMode === 'aspect' ? 'inline-flex' : 'none';
      if (viewScorePlotEl) viewScorePlotEl.style.display = state.viewMode === 'score_plot' ? 'inline-flex' : 'none';
      viewBrowseEl.style.display = state.viewMode === 'browse' ? 'inline-flex' : 'none';
      if (state.viewMode === 'radar_grid') {
        breadcrumbEl.style.display = 'none';
        contentEl.innerHTML = '<div id="radarGridContainer"></div>';
        renderRadarGrid(document.getElementById('radarGridContainer'));
        return;
      }
      if (state.viewMode === 'browse') {
        breadcrumbEl.style.display = 'block';
        renderBrowse();
        return;
      }
      if (state.viewMode === 'detail') {
        if (state.detailRunId) { renderDetail(); return; }
        breadcrumbEl.style.display = 'none';
        contentEl.innerHTML = '<div class="empty-state"><span class="empty-state-title">No run selected</span><p>Click &quot;View&quot; on a run in Browse to see details.</p></div>';
        return;
      }
      if (state.viewMode === 'aspect') {
        breadcrumbEl.style.display = 'none';
        const aspectRuns = getRunsByLevelForAspect();
        const aspect = aspectEl && aspectEl.value;
        contentEl.innerHTML = '<div class="content-single"><div id="colAspect"></div></div>';
        renderAspectView(document.getElementById('colAspect'), aspectRuns, aspect);
        return;
      }
      if (state.viewMode === 'score_plot') {
        breadcrumbEl.style.display = 'none';
        contentEl.innerHTML = '<div class="content-single score-plot-panel"><div id="colScorePlot"></div></div>';
        renderScorePlotView(document.getElementById('colScorePlot'));
        return;
      }
      if (state.viewMode === 'compare') {
        breadcrumbEl.style.display = 'none';
        state.compareLevel = levelEl.value;
        const runAId = runAEl.value || state.compareRunA;
        const runBId = runBEl.value || state.compareRunB;
        const compareLevel = levelEl.value;
        const runA = runAId ? runs.find(r => r.id === runAId && r.level === compareLevel) : null;
        const runB = runBId ? runs.find(r => r.id === runBId && r.level === compareLevel) : null;
        if (runA && runB) {
          const labelA = shortLabel(runA.label_short || runA.id, 45);
          const labelB = shortLabel(runB.label_short || runB.id, 45);
          contentEl.innerHTML = '<div class="compare"><div class="column"><div class="column-header">Run A (Left): ' + escapeHtml(labelA) + '</div><div class="compare-param-wrap">' + paramPanelHtml(runA) + '</div><div id="colA"></div></div><div class="column"><div class="column-header">Run B (Right): ' + escapeHtml(labelB) + '</div><div class="compare-param-wrap">' + paramPanelHtml(runB) + '</div><div id="colB"></div></div></div>';
          renderRunOrSeedRuns(runA, document.getElementById('colA'), runs);
          renderRunOrSeedRuns(runB, document.getElementById('colB'), runs);
        } else {
          contentEl.innerHTML = '<div class="content-single"><p class="empty-state">Select Run A (Left) And Run B (Right) Above. Use &quot;Set As A&quot; Or &quot;Set As B&quot; From Browse Or Detail To Choose Each Side.</p></div>';
        }
      }
    }

    function fillAspectDropdown() {
      if (!aspectEl) return;
      const runs = getRunsByLevelForAspect();
      const options = getAspectOptions(runs);
      const cur = aspectEl.value;
      aspectEl.innerHTML = '<option value="">— Select Aspect —</option>';
      options.forEach(k => {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k === 'training_split' ? 'training_split (train / test)' : k;
        if (k === 'training_split') opt.title = 'Fraction of structures in train set; columns compare different train/test splits at fixed seed and model config.';
        aspectEl.appendChild(opt);
      });
      if (cur && options.indexOf(cur) >= 0) aspectEl.value = cur;
      else if (options.length) aspectEl.value = options[0];
    }

    function init() {
      function useManifest(data) {
        manifest = data;
        titleEl.textContent = (data.base_path || 'Pipeline') + ' — Dashboard';
        generatedEl.textContent = 'Generated: ' + formatGeneratedAt(data.generated_at);
        viewModeEl.value = 'browse';
        state.viewMode = 'browse';
        levelEl.addEventListener('change', () => { fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); updateContent(); });
        runAEl.addEventListener('change', updateContent);
        runBEl.addEventListener('change', updateContent);
        viewModeEl.addEventListener('change', () => {
          state.viewMode = viewModeEl.value;
          if (viewModeEl.value === 'aspect') fillAspectDropdown();
          if (viewModeEl.value === 'score_plot') { fillScorePlotDropdowns(); }
          if (viewModeEl.value === 'compare') { levelEl.value = state.compareLevel || 'euclideanizer'; fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); if (state.compareRunA) runAEl.value = state.compareRunA; if (state.compareRunB) runBEl.value = state.compareRunB; }
          updateContent();
        });
        levelAspectEl.addEventListener('change', () => { fillAspectDropdown(); updateContent(); });
        aspectEl.addEventListener('change', updateContent);
        if (scorePlotAspectEl) scorePlotAspectEl.addEventListener('change', updateContent);
        if (scorePlotScoreEl) scorePlotScoreEl.addEventListener('change', updateContent);
        if (scorePlotDualModeEl) scorePlotDualModeEl.addEventListener('change', function() { fillScorePlotDropdowns(); updateContent(); });
        if (scorePlotScoreXEl) scorePlotScoreXEl.addEventListener('change', updateContent);
        if (scorePlotScoreYEl) scorePlotScoreYEl.addEventListener('change', updateContent);
        if (scorePlotAspectColorEl) scorePlotAspectColorEl.addEventListener('change', updateContent);
        fillRunSelect(runAEl, false);
        fillRunSelect(runBEl, true);
        fillAspectDropdown();
        updateContent();
      }
      if (window.__DASHBOARD_MANIFEST__) {
        useManifest(window.__DASHBOARD_MANIFEST__);
      } else {
        fetch('manifest.json').then(r => r.json()).then(useManifest)
          .catch(e => { contentEl.innerHTML = '<div class="content-error"><strong>Failed to load manifest.</strong> ' + escapeHtml(e.message) + ' Make sure the dashboard folder is complete (index.html and manifest.json).</div>'; });
      }
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  </script>
</body>
</html>
"""


def _write_index_html(dashboard_dir: str, manifest: dict) -> None:
    path = os.path.join(dashboard_dir, INDEX_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_html_content(manifest))


def build_dashboard(base_output_dir: str) -> Optional[str]:
    """
    Scan run root, copy plot/video/analysis assets into dashboard/assets/,
    write dashboard/manifest.json and dashboard/index.html.
    Returns dashboard_dir path if built, None if no runs found.
    """
    runs = _scan_runs(base_output_dir)
    if not runs:
        return None
    dashboard_dir = os.path.join(base_output_dir, DASHBOARD_DIR)
    assets_dir = os.path.join(dashboard_dir, ASSETS_DIR)
    os.makedirs(assets_dir, exist_ok=True)
    cfg = load_pipeline_config(base_output_dir)
    if cfg and "scoring" in cfg:
        scoring_save_pdf_copy = cfg["scoring"]["save_pdf_copy"]
    else:
        scoring_save_pdf_copy = False
    runs_for_manifest = _copy_assets_and_update_paths(runs, assets_dir, scoring_save_pdf_copy)
    manifest = _make_manifest(base_output_dir, runs_for_manifest)
    _write_manifest(dashboard_dir, manifest)
    _write_index_html(dashboard_dir, manifest)
    return dashboard_dir


def build_manifest_with_source_paths(base_output_dir: str) -> list[dict[str, Any]]:
    """Scan base_output_dir and return list of run dicts with blocks containing source_path (no copy)."""
    return _scan_runs(base_output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.dashboard <base_output_dir>", file=sys.stderr)
        print("  Builds dashboard/ (manifest.json, index.html, assets/) inside the given run root.", file=sys.stderr)
        sys.exit(1)
    base = os.path.abspath(sys.argv[1])
    if not os.path.isdir(base):
        print(f"Error: not a directory: {base}", file=sys.stderr)
        sys.exit(1)
    result = build_dashboard(base)
    if result:
        print(f"Dashboard saved to {result}")
        sys.exit(0)
    print("No runs with outputs found; dashboard not built.", file=sys.stderr)
    sys.exit(1)
