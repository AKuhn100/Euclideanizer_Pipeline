#!/usr/bin/env python3
"""
Global orchestration for DistMap + Euclideanizer pipeline.

  python run.py --config path/to/config.yaml [--data /path/to/data.npz] [--no-plots]
  python run.py --config path/to/config.yaml --distmap.beta_kl 0.01 0.05 --no-plots

Config file is required (--config). Training requires a dataset path: set --data or data.path in config.
Any key in the config can be a single value or list (lists => grid).
Outputs: distmap/<i>/ for each DistMap (run_config.yaml + model.pt); distmap/<i>/euclideanizer/<j>/ for each Euclideanizer (run_config.yaml + euclideanizer.pt). Plotting and analysis outputs in the corresponding directories.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
import zlib
import time
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Ensure pipeline root is on path so "src" package is found
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch

from src import utils
from src.config import (
    load_config,
    expand_distmap_grid,
    expand_euclideanizer_grid,
    distmap_training_groups,
    euclideanizer_training_groups,
    get_sample_variances,
    get_data_path,
    get_output_dir,
    get_seeds,
    get_training_splits,
    get_max_data_values,
    save_run_config,
    load_run_config,
    load_pipeline_config,
    save_pipeline_config,
    configs_match_exactly,
    configs_match_sections,
    config_diff,
    run_config_section_matches,
    run_config_section_matches_allow_calibrated,
    pipeline_config_path,
    TRAINING_CRITICAL_KEYS,
    finalize_scoring_tau_config,
    peek_output_dir,
)
from src.metrics import (
    compute_exp_statistics,
    distmap_bond_lengths,
    distmap_rg,
    distmap_scaling,
)
from src.train_distmap import train_distmap
from src.train_euclideanizer import train_euclideanizer
from src.plotting import (
    plot_distmap_reconstruction,
    plot_euclideanizer_reconstruction,
    plot_recon_statistics,
    plot_gen_analysis,
    plot_bond_length_by_genomic_distance,
    plot_pairwise_distance_by_lag_exp_vs_recon,
)
from src.distmap.model import ChromVAE_Conv
from src.distmap.sample import generate_samples as dm_generate_samples
from src.euclideanizer.model import Euclideanizer, load_frozen_vae
from src.analysis_metrics import ANALYSIS_METRICS
from src.latent_analysis import plot_latent_distribution, plot_latent_correlation, save_latent_stats_npz
from src.gro_io import write_structures_gro
from src import scoring as scoring_module
from src import meta_analysis as meta_analysis_module
from src import generative_capacity as generative_capacity_module

# Log file in output root; also mirrored to stdout (set in main).
_LOG_FILE = None
# Lock for serializing log writes in the main process; unused in spawned workers (each opens its own log handle).
_LOG_LOCK = None
# When set (e.g. in run_one_hpo_trial), _log and styled stdout/stderr prefix each line with this (e.g. "[trial 3] ").
_LOG_TRIAL_PREFIX = ""
# Stdout/stderr before wrapping; restored on exit.
_pipeline_real_stdout = None
_pipeline_real_stderr = None
PIPELINE_LOG_FILENAME = "pipeline.log"


def _init_log_file(output_root: str) -> None:
    """Open pipeline.log in output_root for appending; each run gets a short header."""
    global _LOG_FILE
    if _LOG_FILE is not None:
        return
    os.makedirs(output_root, exist_ok=True)
    log_path = os.path.join(output_root, PIPELINE_LOG_FILENAME)
    _LOG_FILE = open(log_path, "a", encoding="utf-8")
    if _LOG_FILE.tell() > 0:
        _LOG_FILE.write("\n")
    _LOG_FILE.write("=" * 60 + "\n")
    _LOG_FILE.write(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FILE.write("=" * 60 + "\n")
    _LOG_FILE.flush()


OVERWRITE_CONFIRM_PHRASE = "yes delete"

# ANSI styling (only when stdout is a TTY)
def _style_code_for_line(line: str) -> str:
    """Return ANSI prefix for a line so stdout and stderr styling stay in sync. Caller appends line and reset."""
    low = line.lower()
    if "error" in low or "failed" in low:
        return "\033[1;31m"
    if "warning" in low:
        return "\033[33m"
    if "saved:" in low or "video saved" in low or "finished" in low or "complete" in low:
        return "\033[32m"
    if "assembling" in low or "epoch" in low or "loaded" in low or "generated" in low or "rmsd" in low:
        return "\033[36m"
    return "\033[2m"


def _red(s: str) -> str:
    if sys.stdout.isatty():
        return f"\033[1;31m{s}\033[0m"
    return s


def _style(s: str, style: str | None) -> str:
    """Apply log style (skip, success, error, info) for terminal; plain text when not a TTY or style is None."""
    if style is None or not sys.stdout.isatty():
        return s
    codes = {
        "skip": "\033[2;36m",    # dim cyan
        "success": "\033[32m",   # green
        "error": "\033[1;31m",   # bold red
        "info": "\033[36m",      # cyan
    }
    if style not in codes:
        return s
    return f"{codes[style]}{s}\033[0m"


# When HPO trial prefix is set, use this for the tag so it doesn't take the line's content color.
_TRIAL_TAG_STYLE = "\033[2m"  # dim

class _StyledStdout:
    """Wrap stdout so submodule print() output gets a default color (only when TTY). Pass-through if already contains ANSI."""

    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s: str) -> None:
        if _LOG_TRIAL_PREFIX:
            # Only ever add trial tag at the start of a line; never after a newline (no tag at end of line).
            if s == "\n":
                s = "\n"
            elif s.endswith("\n"):
                s = _LOG_TRIAL_PREFIX + s[:-1].replace("\n", "\n" + _LOG_TRIAL_PREFIX) + "\n"
            else:
                s = _LOG_TRIAL_PREFIX + s.replace("\n", "\n" + _LOG_TRIAL_PREFIX)
        if not getattr(self._real, "isatty", lambda: False)():
            self._real.write(s)
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line + "\n"
            if "\033[" in line:
                self._real.write(line)
            else:
                # Style only the content so color matches the message; keep trial tag dim.
                if _LOG_TRIAL_PREFIX and line.startswith(_LOG_TRIAL_PREFIX):
                    tag, content = _LOG_TRIAL_PREFIX, line[len(_LOG_TRIAL_PREFIX):]
                    code = _style_code_for_line(content)
                    self._real.write(f"{_TRIAL_TAG_STYLE}{tag}\033[0m{code}{content}\033[0m")
                else:
                    prefix = _style_code_for_line(line)
                    self._real.write(f"{prefix}{line}\033[0m")

    def flush(self) -> None:
        if self._buf:
            self._real.write(self._buf)
            self._buf = ""
        self._real.flush()

    def isatty(self) -> bool:
        return getattr(self._real, "isatty", lambda: False)()


class _StyledStderr:
    """Wrap stderr so submodule warnings/errors get color when TTY. Pass-through if already contains ANSI."""

    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s: str) -> None:
        if _LOG_TRIAL_PREFIX:
            if s == "\n":
                s = "\n"
            elif s.endswith("\n"):
                s = _LOG_TRIAL_PREFIX + s[:-1].replace("\n", "\n" + _LOG_TRIAL_PREFIX) + "\n"
            else:
                s = _LOG_TRIAL_PREFIX + s.replace("\n", "\n" + _LOG_TRIAL_PREFIX)
        if not getattr(self._real, "isatty", lambda: False)():
            self._real.write(s)
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line + "\n"
            if "\033[" in line:
                self._real.write(line)
            else:
                if _LOG_TRIAL_PREFIX and line.startswith(_LOG_TRIAL_PREFIX):
                    tag, content = _LOG_TRIAL_PREFIX, line[len(_LOG_TRIAL_PREFIX):]
                    code = _style_code_for_line(content)
                    self._real.write(f"{_TRIAL_TAG_STYLE}{tag}\033[0m{code}{content}\033[0m")
                else:
                    prefix = _style_code_for_line(line)
                    self._real.write(f"{prefix}{line}\033[0m")

    def flush(self) -> None:
        if self._buf:
            self._real.write(self._buf)
            self._buf = ""
        self._real.flush()

    def isatty(self) -> bool:
        return getattr(self._real, "isatty", lambda: False)()


def _confirm_overwrite(base_output_dir: str) -> None:
    """When resume=False and output dir exists: prompt user to type OVERWRITE_CONFIRM_PHRASE to confirm delete. Else abort."""
    width = 70
    line = "=" * width
    print()
    print(_red(line))
    print(_red("  OVERWRITE CONFIRMATION  "))
    print(_red(line))
    print(_red(f"  Resume is OFF. Output directory already exists:"))
    print(_red(f"    {base_output_dir}"))
    print(_red("  All contents will be DELETED and the run will start from scratch."))
    print(_red(""))
    print(_red(f"  To CONFIRM: type exactly  {OVERWRITE_CONFIRM_PHRASE!r}  and press Enter."))
    print(_red("  To ABORT:    type anything else, or press Ctrl+C."))
    print(_red(line))
    print()
    try:
        reply = input(_red("> ")).strip()
    except (EOFError, KeyboardInterrupt):
        print(_red("Aborted."))
        sys.exit(1)
    if reply.lower() != OVERWRITE_CONFIRM_PHRASE.lower():
        print(_red("Aborted."))
        sys.exit(1)


def _confirm_replot_one_chunk(base_output_dir: str, chunk_label: str) -> None:
    """When resume and one plotting/analysis chunk's config changed: prompt to confirm removal of that chunk's outputs."""
    width = 70
    line = "=" * width
    print()
    print(_red(line))
    print(_red(f"  CONFIG CHANGED: {chunk_label}  "))
    print(_red(line))
    print(_red(f"  Output directory: {base_output_dir}"))
    print(_red(f"  Training config matches; {chunk_label} config differs from saved."))
    print(_red(f"  Existing {chunk_label} outputs will be REMOVED, then re-run."))
    print(_red(""))
    print(_red(f"  To CONFIRM: type exactly  {OVERWRITE_CONFIRM_PHRASE!r}  and press Enter."))
    print(_red("  To ABORT:    type anything else, or press Ctrl+C."))
    print(_red(line))
    print()
    try:
        reply = input(_red("> ")).strip()
    except (EOFError, KeyboardInterrupt):
        print(_red("Aborted."))
        sys.exit(1)
    if reply.lower() != OVERWRITE_CONFIRM_PHRASE.lower():
        print(_red("Aborted."))
        sys.exit(1)


def _delete_plotting_and_analysis_outputs(base_output_dir: str, run_entries: list, training_splits: list, max_data_values: list[int | None] | None = None) -> None:
    """Remove all plots/, analysis/, and dashboard under base_output_dir for the given run entries."""
    import re
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
        if not os.path.isdir(seed_dir):
            continue
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue
        for dm_name in os.listdir(distmap_dir):
            if not dm_name.isdigit():
                continue
            dm_path = os.path.join(distmap_dir, dm_name)
            plots_dm = os.path.join(dm_path, "plots")
            if os.path.isdir(plots_dm):
                shutil.rmtree(plots_dm, ignore_errors=True)
            eu_dir = os.path.join(dm_path, "euclideanizer")
            if os.path.isdir(eu_dir):
                for eu_name in os.listdir(eu_dir):
                    if not eu_name.isdigit():
                        continue
                    eu_path = os.path.join(eu_dir, eu_name)
                    plots_eu = os.path.join(eu_path, "plots")
                    if os.path.isdir(plots_eu):
                        shutil.rmtree(plots_eu, ignore_errors=True)
                    analysis_dir = os.path.join(eu_path, "analysis")
                    if os.path.isdir(analysis_dir):
                        shutil.rmtree(analysis_dir, ignore_errors=True)
    dashboard_dir = os.path.join(base_output_dir, "dashboard")
    if os.path.isdir(dashboard_dir):
        shutil.rmtree(dashboard_dir, ignore_errors=True)


def _has_any_plotting_output(base_output_dir: str, run_entries: list, training_splits: list, max_data_values: list[int | None] | None = None) -> bool:
    """True if any plots/ or dashboard exists under base_output_dir for the given run entries."""
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
        if not os.path.isdir(seed_dir):
            continue
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue
        for dm_name in os.listdir(distmap_dir):
            if not dm_name.isdigit():
                continue
            dm_path = os.path.join(distmap_dir, dm_name)
            if os.path.isdir(os.path.join(dm_path, "plots")):
                return True
            eu_dir = os.path.join(dm_path, "euclideanizer")
            if os.path.isdir(eu_dir):
                for eu_name in os.listdir(eu_dir):
                    if not eu_name.isdigit():
                        continue
                    if os.path.isdir(os.path.join(eu_dir, eu_name, "plots")):
                        return True
    if os.path.isdir(os.path.join(base_output_dir, "dashboard")):
        return True
    return False


def _has_any_analysis_output(base_output_dir: str, run_entries: list, training_splits: list, component: str, max_data_values: list[int | None] | None = None) -> bool:
    """True if any analysis output for the given component exists. component: 'rmsd_gen', 'rmsd_recon', 'q_gen', or 'q_recon'."""
    if component in ("rmsd_gen", "rmsd_recon"):
        subdir = "gen" if component == "rmsd_gen" else "recon"
        target = os.path.join("analysis", "rmsd", subdir)
    elif component in ("q_gen", "q_recon"):
        subdir = "gen" if component == "q_gen" else "recon"
        target = os.path.join("analysis", "q", subdir)
    elif component in ("coord_clustering_gen", "coord_clustering_recon"):
        subdir = "gen" if component == "coord_clustering_gen" else "recon"
        target = os.path.join("analysis", "coord_clustering", subdir)
    elif component in ("distmap_clustering_gen", "distmap_clustering_recon"):
        subdir = "gen" if component == "distmap_clustering_gen" else "recon"
        target = os.path.join("analysis", "distmap_clustering", subdir)
    elif component == "generative_capacity_rmsd":
        target = os.path.join("analysis", "generative_capacity", "rmsd")
    elif component == "generative_capacity_q":
        target = os.path.join("analysis", "generative_capacity", "q")
    elif component == "latent":
        for target in (os.path.join("analysis", "latent"), os.path.join("plots", "latent")):
            for entry in run_entries:
                seed, training_split, max_data = _entry_seed_split_max(entry)
                seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
                if not os.path.isdir(seed_dir):
                    continue
                distmap_dir = os.path.join(seed_dir, "distmap")
                if not os.path.isdir(distmap_dir):
                    continue
                for dm_name in os.listdir(distmap_dir):
                    if not dm_name.isdigit():
                        continue
                    eu_dir = os.path.join(distmap_dir, dm_name, "euclideanizer")
                    if not os.path.isdir(eu_dir):
                        continue
                    for eu_name in os.listdir(eu_dir):
                        if not eu_name.isdigit():
                            continue
                        if os.path.isdir(os.path.join(eu_dir, eu_name, target)):
                            return True
        return False
    else:
        return False
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
        if not os.path.isdir(seed_dir):
            continue
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue
        for dm_name in os.listdir(distmap_dir):
            if not dm_name.isdigit():
                continue
            eu_dir = os.path.join(distmap_dir, dm_name, "euclideanizer")
            if not os.path.isdir(eu_dir):
                continue
            for eu_name in os.listdir(eu_dir):
                if not eu_name.isdigit():
                    continue
                if os.path.isdir(os.path.join(eu_dir, eu_name, target)):
                    return True
    return False


def _has_any_scoring_output(base_output_dir: str, run_entries: list) -> bool:
    """True if any Euclideanizer run has scoring/scores.json."""
    for run_dir_eu, _ in _iter_euclideanizer_runs(base_output_dir):
        if os.path.isfile(os.path.join(run_dir_eu, scoring_module.SCORING_DIR, "scores.json")):
            return True
    return False


def _delete_scoring_outputs(base_output_dir: str) -> None:
    """Remove scoring/ directory under each Euclideanizer run."""
    for run_dir_eu, _ in _iter_euclideanizer_runs(base_output_dir):
        scoring_dir = os.path.join(run_dir_eu, scoring_module.SCORING_DIR)
        if os.path.isdir(scoring_dir):
            shutil.rmtree(scoring_dir, ignore_errors=True)


def _has_any_sufficiency_meta_output(base_output_dir: str) -> bool:
    """True if sufficiency meta-analysis heatmap output exists."""
    root = os.path.join(base_output_dir, "meta_analysis", "sufficiency")
    if not os.path.isdir(root):
        return False
    for name in os.listdir(root):
        if not name.startswith("seed_"):
            continue
        p = os.path.join(root, name, "heatmap", "sufficiency_heatmap_rmsd_q.png")
        if os.path.isfile(p):
            return True
    return False


def _delete_sufficiency_meta_outputs(base_output_dir: str) -> None:
    """Remove sufficiency meta-analysis outputs."""
    root = os.path.join(base_output_dir, "meta_analysis", "sufficiency")
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)


def _iter_euclideanizer_runs(base_output_dir: str):
    """Yield (run_dir_eu, seed_dir) for each Euclideanizer run under base_output_dir.
    Accepts seed_<n> and seed_<n>_split_<frac> directory names."""
    if not os.path.isdir(base_output_dir):
        return
    for seed_name in sorted(os.listdir(base_output_dir)):
        if not seed_name.startswith("seed_"):
            continue
        rest = seed_name[5:]
        if not (rest.isdigit() or "_split_" in rest or "_maxdata_" in rest):
            continue
        seed_dir = os.path.join(base_output_dir, seed_name)
        if not os.path.isdir(seed_dir):
            continue
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue
        for dm_name in sorted(os.listdir(distmap_dir), key=lambda x: (len(x), x)):
            if not dm_name.isdigit():
                continue
            dm_run_root = os.path.join(distmap_dir, dm_name)
            eu_dir = os.path.join(dm_run_root, "euclideanizer")
            if not os.path.isdir(eu_dir):
                continue
            for eu_name in sorted(os.listdir(eu_dir), key=lambda x: (len(x), x)):
                if not eu_name.isdigit():
                    continue
                run_dir_eu = os.path.join(eu_dir, eu_name)
                if os.path.isdir(run_dir_eu):
                    yield run_dir_eu, seed_dir


def _delete_dashboard(base_output_dir: str) -> None:
    """Remove the dashboard directory so a fresh one can be built after re-running plotting or analysis."""
    dashboard_dir = os.path.join(base_output_dir, "dashboard")
    if os.path.isdir(dashboard_dir):
        shutil.rmtree(dashboard_dir, ignore_errors=True)


def _reference_size_config(cfg: dict) -> dict:
    """Extract reference-size keys for comparison. Used to detect when max_train/max_test (or equivalent) change."""
    plot = cfg["plotting"]
    ana = cfg["analysis"]
    return {
        "plotting": (plot["max_train"], plot["max_test"]),
        "rmsd": (ana["rmsd_max_train"], ana["rmsd_max_test"]),
        "q": (ana["q_max_train"], ana["q_max_test"]),
        "coord_clustering": (ana["coord_clustering_max_train"], ana["coord_clustering_max_test"]),
        "distmap_clustering": (ana["distmap_clustering_max_train"], ana["distmap_clustering_max_test"]),
        "latent": (ana["latent_max_train"], ana["latent_max_test"]),
    }


def _reference_size_changed(saved_ref: dict, current_ref: dict) -> set:
    """Return set of component names ('plotting', 'rmsd', 'q', 'coord_clustering', 'distmap_clustering', 'latent') whose reference-size config differs."""
    out = set()
    for key in ("plotting", "rmsd", "q", "coord_clustering", "distmap_clustering", "latent"):
        if saved_ref.get(key) != current_ref.get(key):
            out.add(key)
    return out


def _delete_reference_size_caches(
    base_output_dir: str,
    run_entries: list,
    training_splits: list,
    max_data_values: list[int | None] | set | None,
    components: set | None = None,
) -> None:
    """Remove cached data that depends on reference sizes so it can be recomputed. components: 'plotting', 'rmsd', 'q', 'coord_clustering', 'distmap_clustering', 'latent'. Latent has no seed-level cache (outputs are per-run)."""
    if components is None and isinstance(max_data_values, set):
        components = max_data_values
        max_data_values = None
    components = components or set()
    import glob as _glob
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
        cache_dir = os.path.join(seed_dir, EXP_STATS_CACHE_DIR)
        if not os.path.isdir(cache_dir):
            continue
        if "plotting" in components:
            for name in (EXP_STATS_SPLIT_META, EXP_STATS_TRAIN_NPZ, EXP_STATS_TEST_NPZ):
                path = os.path.join(cache_dir, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        if "rmsd" in components:
            for path in _glob.glob(os.path.join(cache_dir, "test_to_train_rmsd*.npz")):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if "q" in components:
            q_all = os.path.join(cache_dir, "q_test_to_train.npz")
            if os.path.isfile(q_all):
                try:
                    os.remove(q_all)
                except OSError:
                    pass
            for path in _glob.glob(os.path.join(cache_dir, "q_test_to_train_*.npz")):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if "coord_clustering" in components:
            for pattern in ("coord_clustering_train_test_feats_*.npz", "coord_clustering_v2_train_test_coords_*.npz"):
                for path in _glob.glob(os.path.join(cache_dir, pattern)):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        if "distmap_clustering" in components:
            for path in _glob.glob(os.path.join(cache_dir, "distmap_clustering_train_test_feats_*.npz")):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _confirm_reference_size_cache_purge(components: set) -> None:
    """Prompt user to confirm removal of cached data when reference-size config changed. Else abort."""
    labels = sorted(components)
    names = {"plotting": "Plotting (train/test stats)", "rmsd": "RMSD (test→train)", "q": "Q (test→train)", "coord_clustering": "Coord clustering (train/test feats)", "distmap_clustering": "Distmap clustering (train/test feats)", "latent": "Latent (analysis/latent/; no seed cache)"}
    line = "=" * 70
    print()
    print(_red(line))
    print(_red("  REFERENCE SIZE CONFIG CHANGED  "))
    print(_red(line))
    print(_red("  max_train / max_test (or equivalent) differ from the saved config for:"))
    for c in labels:
        print(_red(f"    - {names.get(c, c)}"))
    print(_red("  Cached data for these will be REMOVED so it can be recomputed with the new values."))
    print(_red(""))
    print(_red(f"  To CONFIRM: type exactly  {OVERWRITE_CONFIRM_PHRASE!r}  and press Enter."))
    print(_red("  To ABORT:   type anything else, or press Ctrl+C."))
    print(_red(line))
    print()
    try:
        reply = input(_red("> ")).strip()
    except (EOFError, KeyboardInterrupt):
        print(_red("Aborted."))
        sys.exit(1)
    if reply.lower() != OVERWRITE_CONFIRM_PHRASE.lower():
        print(_red("Aborted."))
        sys.exit(1)


def _delete_plotting_outputs_only(base_output_dir: str, run_entries: list, training_splits: list, max_data_values: list[int | None] | None = None) -> None:
    """Remove all plots/ under base_output_dir for the given run entries (no analysis dirs, no dashboard)."""
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
        if not os.path.isdir(seed_dir):
            continue
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue
        for dm_name in os.listdir(distmap_dir):
            if not dm_name.isdigit():
                continue
            dm_path = os.path.join(distmap_dir, dm_name)
            plots_dm = os.path.join(dm_path, "plots")
            if os.path.isdir(plots_dm):
                shutil.rmtree(plots_dm, ignore_errors=True)
            eu_dir = os.path.join(dm_path, "euclideanizer")
            if os.path.isdir(eu_dir):
                for eu_name in os.listdir(eu_dir):
                    if not eu_name.isdigit():
                        continue
                    plots_eu = os.path.join(eu_dir, eu_name, "plots")
                    if os.path.isdir(plots_eu):
                        shutil.rmtree(plots_eu, ignore_errors=True)


def _delete_analysis_outputs_for_component(base_output_dir: str, run_entries: list, training_splits: list, component: str, max_data_values: list[int | None] | None = None) -> None:
    """Remove only the analysis subdir for this component (e.g. analysis/q/gen). component: 'rmsd_gen', 'rmsd_recon', 'q_gen', or 'q_recon'."""
    if component in ("rmsd_gen", "rmsd_recon"):
        subdir = "gen" if component == "rmsd_gen" else "recon"
        target = os.path.join("analysis", "rmsd", subdir)
    elif component in ("q_gen", "q_recon"):
        subdir = "gen" if component == "q_gen" else "recon"
        target = os.path.join("analysis", "q", subdir)
    elif component in ("coord_clustering_gen", "coord_clustering_recon"):
        subdir = "gen" if component == "coord_clustering_gen" else "recon"
        target = os.path.join("analysis", "coord_clustering", subdir)
    elif component in ("distmap_clustering_gen", "distmap_clustering_recon"):
        subdir = "gen" if component == "distmap_clustering_gen" else "recon"
        target = os.path.join("analysis", "distmap_clustering", subdir)
    elif component == "generative_capacity_rmsd":
        target = os.path.join("analysis", "generative_capacity", "rmsd")
    elif component == "generative_capacity_q":
        target = os.path.join("analysis", "generative_capacity", "q")
    elif component == "latent":
        targets = [os.path.join("analysis", "latent"), os.path.join("plots", "latent")]
        for entry in run_entries:
            seed, training_split, max_data = _entry_seed_split_max(entry)
            seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
            if not os.path.isdir(seed_dir):
                continue
            distmap_dir = os.path.join(seed_dir, "distmap")
            if not os.path.isdir(distmap_dir):
                continue
            for dm_name in os.listdir(distmap_dir):
                if not dm_name.isdigit():
                    continue
                eu_dir = os.path.join(distmap_dir, dm_name, "euclideanizer")
                if not os.path.isdir(eu_dir):
                    continue
                for eu_name in os.listdir(eu_dir):
                    if not eu_name.isdigit():
                        continue
                    for target in targets:
                        path = os.path.join(eu_dir, eu_name, target)
                        if os.path.isdir(path):
                            shutil.rmtree(path, ignore_errors=True)
        return
    else:
        return
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, max_data_values))
        if not os.path.isdir(seed_dir):
            continue
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue
        for dm_name in os.listdir(distmap_dir):
            if not dm_name.isdigit():
                continue
            eu_dir = os.path.join(distmap_dir, dm_name, "euclideanizer")
            if not os.path.isdir(eu_dir):
                continue
            for eu_name in os.listdir(eu_dir):
                if not eu_name.isdigit():
                    continue
                path = os.path.join(eu_dir, eu_name, target)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                if component in ("generative_capacity_rmsd", "generative_capacity_q"):
                    gc_dir = os.path.join(eu_dir, eu_name, "analysis", "generative_capacity")
                    for stem in ("convergence_median_vs_n_rmsd_q.png", "convergence_median_vs_n_rmsd_q.pdf"):
                        fp = os.path.join(gc_dir, stem)
                        if os.path.isfile(fp):
                            try:
                                os.remove(fp)
                            except OSError:
                                pass


def _confirm_overwrite_outputs(labels: list) -> None:
    """Prompt user to type OVERWRITE_CONFIRM_PHRASE to confirm overwrite of the listed output types. Else abort."""
    width = 70
    line = "=" * width
    print()
    print(_red(line))
    print(_red("  OVERWRITE EXISTING OUTPUTS  "))
    print(_red(line))
    print(_red("  You requested overwrite_existing for: " + ", ".join(labels)))
    print(_red("  Existing outputs for these will be REMOVED before re-running."))
    print(_red(""))
    print(_red(f"  To CONFIRM: type exactly  {OVERWRITE_CONFIRM_PHRASE!r}  and press Enter."))
    print(_red("  To ABORT:    type anything else, or press Ctrl+C."))
    print(_red(line))
    print()
    try:
        reply = input(_red("> ")).strip()
    except (EOFError, KeyboardInterrupt):
        print(_red("Aborted."))
        sys.exit(1)
    if reply.lower() != OVERWRITE_CONFIRM_PHRASE.lower():
        print(_red("Aborted."))
        sys.exit(1)


def _log(msg: str, since_start: float | None = None, since_phase: float | None = None, style: str | None = None) -> None:
    """Write a concise log line to stdout (styled when TTY) and to pipeline.log (plain). Flushes file after each write.
    Trial prefix is added only by the stdout wrapper (once); we add it to the log file here so the file is self-contained."""
    if since_start is not None:
        prefix = f"[+{since_start / 60:5.1f}m]"
        suffix = f"  (phase {since_phase / 60:.1f}m)" if since_phase is not None else ""
        line = f"{prefix} {msg}{suffix}"
    else:
        line = "        " + msg
    # Stdout: do not add trial prefix here; _StyledStdout will add it once per line.
    print(_style(line, style))
    if _LOG_FILE is not None:
        if _LOG_LOCK is not None:
            _LOG_LOCK.acquire()
        try:
            file_line = (_LOG_TRIAL_PREFIX + line) if _LOG_TRIAL_PREFIX else line
            _LOG_FILE.write(file_line + "\n")
            _LOG_FILE.flush()
        finally:
            if _LOG_LOCK is not None:
                _LOG_LOCK.release()


def _plot_exp_stats_precompute_prefix(run_label: str, idx: int | None, n: int | None) -> str:
    """Log prefix for per-run plot train/test experimental statistics (single- and multi-GPU precompute)."""
    if idx is not None and n is not None:
        return f"Precompute plot exp stats [{idx}/{n}] {run_label}"
    return f"Precompute plot exp stats {run_label}"


def _warn_calibration_reserve_if_low(cfg: dict, pipeline_start: float) -> None:
    """If CUDA is available, warn when calibration reserve (safety_margin_gb) < 15 GB on any GPU."""
    if not torch.cuda.is_available():
        return
    safety_gb = cfg.get("calibration_safety_margin_gb")
    if safety_gb is None:
        return
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1e9
            if safety_gb < 15:
                _log(
                    f"Warning: GPU {i} ({total_gb:.1f} GB VRAM) has calibration_safety_margin_gb={safety_gb} GB reserve. Recommended ≥15 GB for stability.",
                    since_start=time.time() - pipeline_start,
                    style="warning",
                )
    except Exception:
        pass


def _log_raw(line: str, style: str | None = None) -> None:
    """Write a raw line (e.g. separator) to stdout (styled when TTY) and log file (plain).
    Trial prefix is added only by the stdout wrapper; we add it to the log file here."""
    # Stdout: do not add trial prefix here; _StyledStdout will add it once per line.
    print(_style(line, style))
    if _LOG_FILE is not None:
        if _LOG_LOCK is not None:
            _LOG_LOCK.acquire()
        try:
            file_line = (_LOG_TRIAL_PREFIX + line) if _LOG_TRIAL_PREFIX else line
            _LOG_FILE.write(file_line + "\n")
            _LOG_FILE.flush()
        finally:
            if _LOG_LOCK is not None:
                _LOG_LOCK.release()


def _append_preinit_fatal_traceback(traceback_text: str) -> None:
    """If startup failed before :func:`_init_log_file`, append the traceback to ``output_dir/pipeline.log`` when discoverable."""
    try:
        args = _parse_args()
        if getattr(args, "worker_from_pickle", None):
            return
        od = peek_output_dir(args.config, _args_to_overrides(args))
        if not od:
            return
        os.makedirs(od, exist_ok=True)
        log_path = os.path.join(od, PIPELINE_LOG_FILENAME)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n")
            lf.write("=" * 60 + "\n")
            lf.write(
                f"PIPELINE ERROR (before pipeline.log init) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            lf.write("=" * 60 + "\n")
            lf.write(traceback_text)
            if not traceback_text.endswith("\n"):
                lf.write("\n")
            lf.flush()
    except Exception:
        pass


# Output layout: run_root/model/ (model.pt or euclideanizer.pt), run_root/plots/<type>/ (PNG and optional data/*.npz)
_ACTIVE_MAX_DATA: int | None = None
_MAX_DATA_VALUES_CONTEXT: list[int | None] = [None]


def _entry_seed_split_max(entry) -> tuple[int, float, int | None]:
    if isinstance(entry, (tuple, list)) and len(entry) >= 3:
        return int(entry[0]), float(entry[1]), entry[2]
    return int(entry[0]), float(entry[1]), _ACTIVE_MAX_DATA


def _seed_split_dir_name(
    seed: int,
    training_split: float,
    max_data: int | None,
    training_splits_list: list,
    max_data_values_list: list[int | None] | None,
) -> str:
    """Directory name for a (seed, training_split, max_data) run."""
    parts = [f"seed_{seed}"]
    if len(training_splits_list) > 1:
        parts.append(f"split_{training_split}")
    md_vals = max_data_values_list or _MAX_DATA_VALUES_CONTEXT or [max_data]
    if len(md_vals) > 1:
        parts.append(f"maxdata_{'all' if max_data is None else max_data}")
    return "_".join(parts)


def _ensure_per_seed_pipeline_config(
    *,
    need_train: bool,
    output_dir: str,
    cfg: dict,
    seed: int,
    training_split: float,
    max_data: int | None,
) -> None:
    """Write per-seed ``pipeline_config.yaml`` before any code creates subdirs under ``output_dir``.

    Seed-level caches (train/test exp stats, test→train RMSD/Q, etc.) call ``os.makedirs`` under
    ``output_dir``. If that happens before this file exists, resume validation sees a directory
    without a config and aborts. Call this at the start of any path that may materialize
    ``output_dir`` (single-GPU seed runs, multi-GPU main-process precompute, worker setup).
    """
    if not need_train:
        return
    if os.path.isfile(pipeline_config_path(output_dir)):
        return
    effective_cfg = {
        **cfg,
        "output_dir": output_dir,
        "data": {**cfg["data"], "split_seed": seed, "training_split": training_split, "max_data": max_data},
    }
    save_pipeline_config(effective_cfg, output_dir)


EXP_STATS_CACHE_DIR = "experimental_statistics"
EXP_STATS_META = "meta.json"
EXP_STATS_NPZ = "exp_stats.npz"
EXP_STATS_TRAIN_NPZ = "exp_stats_train.npz"
EXP_STATS_TEST_NPZ = "exp_stats_test.npz"
EXP_STATS_SPLIT_META = "split_meta.json"


def _exp_stats_cache_dir(output_dir: str) -> str:
    return os.path.join(output_dir, EXP_STATS_CACHE_DIR)


def _distmaps_to_upper(distmaps: np.ndarray) -> np.ndarray:
    """(B,N,N) -> (B,tri) upper triangles for compact cache storage."""
    n = distmaps.shape[-1]
    ii, jj = np.triu_indices(n, k=1)
    return distmaps[:, ii, jj].astype(np.float32, copy=False)


def _upper_to_distmaps(upper: np.ndarray, num_atoms: int) -> np.ndarray:
    """(B,tri) -> (B,N,N) symmetric distance maps."""
    b = upper.shape[0]
    out = np.zeros((b, num_atoms, num_atoms), dtype=np.float32)
    ii, jj = np.triu_indices(num_atoms, k=1)
    out[:, ii, jj] = upper
    out[:, jj, ii] = upper
    return out


def _materialize_exp_stats_distmaps(stats: dict) -> dict:
    """Ensure loaded cache dict has 'exp_distmaps' in full symmetric form."""
    if "exp_distmaps" in stats:
        return stats
    if "exp_distmaps_upper" in stats:
        num_atoms = int(stats["num_atoms_in_stats"]) if "num_atoms_in_stats" in stats else None
        if num_atoms is None:
            raise ValueError("Cached stats missing num_atoms_in_stats for exp_distmaps_upper.")
        out = dict(stats)
        out["exp_distmaps"] = _upper_to_distmaps(np.asarray(stats["exp_distmaps_upper"]), num_atoms)
        return out
    return stats


def _compress_exp_stats_for_cache(stats: dict) -> dict:
    """Convert exp_distmaps -> exp_distmaps_upper for on-disk cache."""
    out = dict(stats)
    if "exp_distmaps" in out:
        dm = np.asarray(out["exp_distmaps"], dtype=np.float32)
        out["exp_distmaps_upper"] = _distmaps_to_upper(dm)
        out["num_atoms_in_stats"] = np.int32(dm.shape[-1])
        del out["exp_distmaps"]
    return out


def _output_dir_has_pipeline_content(output_dir: str) -> bool:
    """True if output_dir already contains distmap or euclideanizer runs (resume scenario)."""
    return os.path.isdir(os.path.join(output_dir, "distmap"))


def _base_has_any_seed_pipeline_content(base_output_dir: str, run_entries: list) -> bool:
    """True if any run dir (seed_<n> or seed_<n>_split_<frac>) under base_output_dir has distmap."""
    if not os.path.isdir(base_output_dir):
        return False
    for name in os.listdir(base_output_dir):
        if not name.startswith("seed_"):
            continue
        rest = name[5:]
        if not (rest.isdigit() or "_split_" in rest or "_maxdata_" in rest):
            continue
        if os.path.isdir(os.path.join(base_output_dir, name, "distmap")):
            return True
    return False


def _load_exp_stats_cache_meta(base_output_dir: str, data_path: str) -> dict | None:
    """Load base exp_stats cache meta; validate against data_path. Returns meta dict (num_structures, num_atoms, data_path) or None."""
    cache_dir = _exp_stats_cache_dir(base_output_dir)
    meta_path = os.path.join(cache_dir, EXP_STATS_META)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("data_path") != os.path.abspath(data_path):
            return None
        if "num_structures" not in meta or "num_atoms" not in meta:
            return None
        return meta
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _load_exp_stats_cache(output_dir: str, data_path: str, num_structures: int, num_atoms: int) -> dict | None:
    """Load experimental statistics from cache if present and valid for this dataset."""
    cache_dir = _exp_stats_cache_dir(output_dir)
    meta_path = os.path.join(cache_dir, EXP_STATS_META)
    npz_path = os.path.join(cache_dir, EXP_STATS_NPZ)
    if not os.path.isfile(meta_path) or not os.path.isfile(npz_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("data_path") != os.path.abspath(data_path):
            return None
        if meta.get("num_structures") != num_structures or meta.get("num_atoms") != num_atoms:
            return None
        with np.load(npz_path, allow_pickle=False) as data:
            return _materialize_exp_stats_distmaps({k: data[k] for k in data.files})
    except (json.JSONDecodeError, OSError, KeyError, zlib.error, zipfile.BadZipFile):
        return None


def _try_load_stats_only(
    base_output_dir: str,
    data_path: str,
    run_entries: list,
    training_splits: list,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[dict | None, int | None, int | None]:
    """
    Load exp_stats and train/test caches without loading coords.
    Requires base cache meta to match data_path and all run entries to have valid train/test cache (matching max_train/max_test).
    Returns (exp_stats, num_atoms, num_structures) or (None, None, None) if stats-only load is not possible.
    """
    meta = _load_exp_stats_cache_meta(base_output_dir, data_path)
    if meta is None:
        return None, None, None
    num_structures = int(meta["num_structures"])
    num_atoms = int(meta["num_atoms"])
    cache_dir = _exp_stats_cache_dir(base_output_dir)
    npz_path = os.path.join(cache_dir, EXP_STATS_NPZ)
    if not os.path.isfile(npz_path):
        return None, None, None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            exp_stats = _materialize_exp_stats_distmaps({k: data[k] for k in data.files})
    except (OSError, zlib.error, zipfile.BadZipFile):
        return None, None, None
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
        expected_n = num_structures if max_data is None else min(num_structures, int(max_data))
        if not _exp_stats_split_cache_meta_files_ok(
            output_dir, data_path, expected_n, num_atoms, seed, training_split,
            max_train=max_train, max_test=max_test,
        ):
            return None, None, None
    return exp_stats, num_atoms, num_structures


def _save_exp_stats_cache(output_dir: str, data_path: str, num_structures: int, num_atoms: int, exp_stats: dict) -> None:
    """Write experimental statistics to cache for reuse on resume."""
    cache_dir = _exp_stats_cache_dir(output_dir)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, EXP_STATS_META)
    meta = {
        "data_path": os.path.abspath(data_path),
        "num_structures": num_structures,
        "num_atoms": num_atoms,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    npz_path = os.path.join(cache_dir, EXP_STATS_NPZ)
    np.savez_compressed(npz_path, **_compress_exp_stats_for_cache(exp_stats))


def _exp_stats_split_cache_meta_files_ok(
    output_dir: str,
    data_path: str,
    num_structures: int,
    num_atoms: int,
    split_seed: int,
    training_split: float,
    max_train: int | None = None,
    max_test: int | None = None,
) -> bool:
    """True if split cache files exist and split_meta.json matches (no NPZ load). Used for fast startup checks."""
    cache_dir = _exp_stats_cache_dir(output_dir)
    meta_path = os.path.join(cache_dir, EXP_STATS_SPLIT_META)
    train_path = os.path.join(cache_dir, EXP_STATS_TRAIN_NPZ)
    test_path = os.path.join(cache_dir, EXP_STATS_TEST_NPZ)
    if not os.path.isfile(meta_path) or not os.path.isfile(train_path) or not os.path.isfile(test_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("data_path") != os.path.abspath(data_path):
            return False
        if meta.get("num_structures") != num_structures or meta.get("num_atoms") != num_atoms:
            return False
        if meta.get("split_seed") != split_seed or meta.get("training_split") != training_split:
            return False
        cached_mt = meta.get("max_train")
        cached_mc = meta.get("max_test")
        if cached_mt != max_train or cached_mc != max_test:
            return False
    except (json.JSONDecodeError, OSError, UnicodeError):
        return False
    return True


def _exp_stats_split_arrays_match_counts(
    train_stats: dict,
    test_stats: dict,
    expected_n_train: int,
    expected_n_test: int,
) -> bool:
    """True if exp_distmaps first dimension matches expected train/test structure counts."""
    try:
        tr = train_stats.get("exp_distmaps")
        te = test_stats.get("exp_distmaps")
        if tr is None or te is None:
            return False
        return (
            int(np.asarray(tr).shape[0]) == int(expected_n_train)
            and int(np.asarray(te).shape[0]) == int(expected_n_test)
        )
    except Exception:
        return False


def _load_exp_stats_split_cache(
    output_dir: str,
    data_path: str,
    num_structures: int,
    num_atoms: int,
    split_seed: int,
    training_split: float,
    max_train: int | None = None,
    max_test: int | None = None,
    expected_n_train: int | None = None,
    expected_n_test: int | None = None,
) -> tuple[dict | None, dict | None]:
    """Load train and test exp stats from seed output_dir if cache valid. Returns (train_stats, test_stats) or (None, None).
    max_train/max_test must match cached meta (None = use all).
    If expected_n_train and expected_n_test are set, loaded exp_distmaps leading dimensions must match or cache is rejected."""
    cache_dir = _exp_stats_cache_dir(output_dir)
    meta_path = os.path.join(cache_dir, EXP_STATS_SPLIT_META)
    train_path = os.path.join(cache_dir, EXP_STATS_TRAIN_NPZ)
    test_path = os.path.join(cache_dir, EXP_STATS_TEST_NPZ)
    if not os.path.isfile(meta_path) or not os.path.isfile(train_path) or not os.path.isfile(test_path):
        return None, None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("data_path") != os.path.abspath(data_path):
            return None, None
        if meta.get("num_structures") != num_structures or meta.get("num_atoms") != num_atoms:
            return None, None
        if meta.get("split_seed") != split_seed or meta.get("training_split") != training_split:
            return None, None
        cached_mt = meta.get("max_train")
        cached_mc = meta.get("max_test")
        if cached_mt != max_train or cached_mc != max_test:
            return None, None
        with np.load(train_path, allow_pickle=False) as data:
            train_stats = _materialize_exp_stats_distmaps({k: data[k] for k in data.files})
        with np.load(test_path, allow_pickle=False) as data:
            test_stats = _materialize_exp_stats_distmaps({k: data[k] for k in data.files})
        if expected_n_train is not None and expected_n_test is not None:
            if not _exp_stats_split_arrays_match_counts(
                train_stats, test_stats, expected_n_train, expected_n_test,
            ):
                return None, None
        return train_stats, test_stats
    except (json.JSONDecodeError, OSError, KeyError, zlib.error, zipfile.BadZipFile, ValueError):
        return None, None


def _save_exp_stats_split_cache(
    output_dir: str,
    data_path: str,
    num_structures: int,
    num_atoms: int,
    split_seed: int,
    training_split: float,
    train_stats: dict,
    test_stats: dict,
    max_train: int | None = None,
    max_test: int | None = None,
) -> None:
    """Write per-seed train/test experimental statistics to cache. max_train/max_test stored in meta (None = use all)."""
    cache_dir = _exp_stats_cache_dir(output_dir)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, EXP_STATS_SPLIT_META)
    meta = {
        "data_path": os.path.abspath(data_path),
        "num_structures": num_structures,
        "num_atoms": num_atoms,
        "split_seed": split_seed,
        "training_split": training_split,
        "max_train": max_train,
        "max_test": max_test,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(os.path.join(cache_dir, EXP_STATS_TRAIN_NPZ), **_compress_exp_stats_for_cache(train_stats))
    np.savez_compressed(os.path.join(cache_dir, EXP_STATS_TEST_NPZ), **_compress_exp_stats_for_cache(test_stats))


def _model_dir(run_root: str) -> str:
    return os.path.join(run_root, "model")


def _dm_path(run_root: str) -> str:
    return os.path.join(_model_dir(run_root), "model.pt")


def _eu_path(run_root: str) -> str:
    return os.path.join(_model_dir(run_root), "euclideanizer.pt")


def _dm_path_last(run_root: str) -> str:
    return os.path.join(_model_dir(run_root), "model_last.pt")


def _eu_path_last(run_root: str) -> str:
    return os.path.join(_model_dir(run_root), "euclideanizer_last.pt")


def _run_completed(
    run_dir: str,
    expected_epochs: int,
    *,
    model_subdir: str = "model",
    section_key: str | None = None,
    expected_section: dict | None = None,
    multi_segment: bool = False,
    checkpoint_last_name: str | None = None,
    is_last_segment: bool = False,
    save_final_models_per_stretch: bool = False,
    _log_fail_reason: str | None = None,
) -> bool:
    """True if run_dir has a completed run: last_epoch_trained == expected_epochs (or early_stopped and best exists), section matches if given, best checkpoint exists. When multi_segment, last checkpoint required only if save_final_models_per_stretch (we don't require it when false since it is deleted after the next segment uses it)."""
    model_dir = os.path.join(run_dir, model_subdir)
    run_cfg = load_run_config(model_dir)
    if run_cfg is None:
        if _log_fail_reason:
            _log(f"{_log_fail_reason}: run_config not found or invalid at {model_dir}", since_start=None, style="skip")
        return False
    last_trained = run_cfg.get("last_epoch_trained")
    early_stopped = bool(run_cfg.get("early_stopped", False))
    if early_stopped:
        if last_trained is None:
            if _log_fail_reason:
                _log(f"{_log_fail_reason}: early_stopped but last_epoch_trained missing", since_start=None, style="skip")
            return False
        # Consider complete when we have a best checkpoint and stopped early (no need to match expected_epochs).
    elif last_trained != expected_epochs:
        if _log_fail_reason:
            _log(f"{_log_fail_reason}: last_epoch_trained={last_trained!r} (type {type(last_trained).__name__}) != expected_epochs={expected_epochs!r} (type {type(expected_epochs).__name__})", since_start=None, style="skip")
        return False
    if section_key is not None and expected_section is not None:
        if not run_config_section_matches_allow_calibrated(run_cfg, section_key, expected_section):
            if _log_fail_reason:
                diffs = config_diff(run_cfg.get(section_key) or {}, expected_section, section_key)
                _log(f"{_log_fail_reason}: section match failed: {diffs}", since_start=None, style="skip")
            return False
    best_name = "model.pt" if section_key == "distmap" else "euclideanizer.pt"
    best_path = os.path.join(model_dir, best_name)
    if not os.path.isfile(best_path):
        if _log_fail_reason:
            _log(f"{_log_fail_reason}: best checkpoint missing: {best_path}", since_start=None, style="skip")
        return False
    # Only require last-epoch checkpoint when we keep it (save_final_models_per_stretch); when false we delete it after the next segment uses it, so it won't exist on re-run.
    require_last = multi_segment and checkpoint_last_name and save_final_models_per_stretch
    if require_last and not os.path.isfile(os.path.join(model_dir, checkpoint_last_name)):
        if _log_fail_reason:
            _log(f"{_log_fail_reason}: last checkpoint required but missing", since_start=None, style="skip")
        return False
    return True


def _distmap_training_action(
    run_dir_dm: str,
    ev: int,
    dm_cfg: dict,
    prev_dm_path: str | None,
    prev_dm_ev: int | None,
    prev_run_dir_dm: str | None,
    resume: bool,
    dm_multi: bool,
    dm_last_segment: bool,
    dm_save_final: bool,
) -> dict:
    """Determine how to run this DistMap segment: skip, from_scratch, resume_from_best, or resume_from_prev_last. Returns a dict with 'action' and, when relevant, resume_from_path, prev_run_dir, additional_epochs, best_epoch. When prev segment early_stopped, returns skip so no further segments run."""
    dm_path = _dm_path(run_dir_dm)
    run_label = os.path.basename(run_dir_dm)
    if prev_run_dir_dm is not None and dm_multi:
        prev_cfg = load_run_config(os.path.join(prev_run_dir_dm, "model"))
        if prev_cfg and prev_cfg.get("early_stopped"):
            return {"action": "skip"}
    if resume and os.path.isfile(dm_path) and _run_completed(
        run_dir_dm, ev, section_key="distmap", expected_section=dm_cfg,
        multi_segment=dm_multi, checkpoint_last_name="model_last.pt" if dm_multi else None,
        is_last_segment=dm_last_segment, save_final_models_per_stretch=dm_save_final,
        _log_fail_reason=f"DistMap run {run_label} (ev={ev}) complete check",
    ):
        return {"action": "skip"}
    dm_model_dir = os.path.join(run_dir_dm, "model")
    dm_run_cfg = load_run_config(dm_model_dir)
    dm_best_epoch = dm_run_cfg.get("best_epoch") if dm_run_cfg else None
    if prev_dm_path is None:
        if resume and os.path.isfile(dm_path) and dm_best_epoch is not None and dm_best_epoch < ev:
            return {
                "action": "resume_from_best",
                "resume_from_path": dm_path,
                "prev_run_dir": run_dir_dm,
                "additional_epochs": ev - dm_best_epoch,
                "best_epoch": dm_best_epoch,
            }
        return {"action": "from_scratch"}
    if resume and os.path.isfile(dm_path) and dm_best_epoch is not None and dm_best_epoch > prev_dm_ev:
        return {
            "action": "resume_from_best",
            "resume_from_path": dm_path,
            "prev_run_dir": run_dir_dm,
            "additional_epochs": ev - dm_best_epoch,
            "best_epoch": dm_best_epoch,
        }
    return {
        "action": "resume_from_prev_last",
        "resume_from_path": _dm_path_last(prev_run_dir_dm),
        "prev_run_dir": prev_run_dir_dm,
        "additional_epochs": ev - prev_dm_ev,
    }


def _euclideanizer_training_action(
    eu_run_dir: str,
    eu_ev: int,
    eu_cfg_seg: dict,
    prev_eu_path: str | None,
    prev_eu_ev: int | None,
    prev_eu_run_dir: str | None,
    resume: bool,
    eu_multi: bool,
    eu_last_segment: bool,
    eu_save_final: bool,
) -> dict:
    """Determine how to run this Euclideanizer segment: skip, from_scratch, resume_from_best, or resume_from_prev_last. Returns a dict with 'action' and, when relevant, resume_from_path, prev_run_dir, additional_epochs, best_epoch. When prev segment early_stopped, returns skip so no further segments run."""
    eu_path = _eu_path(eu_run_dir)
    if prev_eu_run_dir is not None and eu_multi:
        prev_cfg = load_run_config(os.path.join(prev_eu_run_dir, "model"))
        if prev_cfg and prev_cfg.get("early_stopped"):
            return {"action": "skip"}
    if resume and os.path.isfile(eu_path) and _run_completed(
        eu_run_dir, eu_ev, section_key="euclideanizer", expected_section=eu_cfg_seg,
        multi_segment=eu_multi, checkpoint_last_name="euclideanizer_last.pt" if eu_multi else None,
        is_last_segment=eu_last_segment, save_final_models_per_stretch=eu_save_final,
    ):
        return {"action": "skip"}
    eu_model_dir = os.path.join(eu_run_dir, "model")
    eu_run_cfg = load_run_config(eu_model_dir)
    eu_best_epoch = eu_run_cfg.get("best_epoch") if eu_run_cfg else None
    if prev_eu_path is None:
        if resume and os.path.isfile(eu_path) and eu_best_epoch is not None and eu_best_epoch < eu_ev:
            return {
                "action": "resume_from_best",
                "resume_from_path": eu_path,
                "prev_run_dir": eu_run_dir,
                "additional_epochs": eu_ev - eu_best_epoch,
                "best_epoch": eu_best_epoch,
            }
        return {"action": "from_scratch"}
    if resume and os.path.isfile(eu_path) and eu_best_epoch is not None and eu_best_epoch > prev_eu_ev:
        return {
            "action": "resume_from_best",
            "resume_from_path": eu_path,
            "prev_run_dir": eu_run_dir,
            "additional_epochs": eu_ev - eu_best_epoch,
            "best_epoch": eu_best_epoch,
        }
    return {
        "action": "resume_from_prev_last",
        "resume_from_path": _eu_path_last(prev_eu_run_dir),
        "prev_run_dir": prev_eu_run_dir,
        "additional_epochs": eu_ev - prev_eu_ev,
    }


# Plot type -> (subdir, filename pattern). Use _plot_path(run_root, type) or _plot_path(run_root, type, subset=..., var=...).
PLOT_TYPES = {
    "reconstruction": ("reconstruction", "reconstruction.png"),
    "recon_statistics": ("recon_statistics", "recon_statistics_{subset}.png"),
    "gen_variance": ("gen_variance", "gen_variance_{var}.png"),
    "bond_length_by_genomic_distance_gen": ("bond_length_by_genomic_distance_gen", "bond_length_by_genomic_distance_gen.png"),
    "bond_length_by_genomic_distance_train": ("bond_length_by_genomic_distance_train", "bond_length_by_genomic_distance_train.png"),
    "bond_length_by_genomic_distance_test": ("bond_length_by_genomic_distance_test", "bond_length_by_genomic_distance_test.png"),
}


def _plot_path(run_root: str, plot_type: str, **format_kw: str) -> str:
    subdir, pattern = PLOT_TYPES[plot_type]
    filename = pattern.format(**format_kw) if format_kw else pattern
    return os.path.join(run_root, "plots", subdir, filename)


def _plotting_phase_needed(
    do_plot: bool,
    exp_stats,
    coords,
    train_stats,
    test_stats,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
) -> bool:
    """True if any DistMap/Euclideanizer diagnostic plot work can run.
    gen_variance needs exp_stats; bond_length and recon_statistics need train/test stats + coords."""
    if not do_plot:
        return False
    if coords is not None:
        if do_recon_plot:
            return True
        if train_stats is not None and test_stats is not None:
            if do_bond_rg_scaling or do_bond_length_by_genomic_distance:
                return True
            if do_avg_gen and exp_stats is not None:
                return True
        return False
    return exp_stats is not None


def _analysis_path(run_root: str, analysis_type: str, filename: str) -> str:
    return os.path.join(run_root, "analysis", analysis_type, filename)


def _distmap_plotting_all_present(
    run_dir_dm: str,
    resume: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
    sample_variances: list,
) -> bool:
    """True if resume and all DistMap plot files we would generate already exist (so we can skip loading the model)."""
    if not resume:
        return False
    if do_recon_plot and not os.path.isfile(_plot_path(run_dir_dm, "reconstruction")):
        return False
    if do_bond_rg_scaling:
        for name in ("test", "train"):
            if not os.path.isfile(_plot_path(run_dir_dm, "recon_statistics", subset=name)):
                return False
    if do_avg_gen:
        for var in sample_variances:
            if not os.path.isfile(_plot_path(run_dir_dm, "gen_variance", var=str(var))):
                return False
    if do_bond_length_by_genomic_distance:
        for bt in (
            "bond_length_by_genomic_distance_gen",
            "bond_length_by_genomic_distance_train",
            "bond_length_by_genomic_distance_test",
        ):
            if not os.path.isfile(_plot_path(run_dir_dm, bt)):
                return False
    return True


def _ensure_list(val):
    """Normalize a config value to a list (single value -> [value], None -> [])."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def _analysis_cfg_from_need_data_kwargs(kw: dict) -> dict:
    """Build a full analysis_cfg (all metric blocks present) from flattened need-data kwargs for presence checks.
    Callers can use direct access; disabled metrics get enabled=False and empty/default recon lists."""
    def _gen_block(enabled: bool, sample_variance: list, num_samples: list) -> dict:
        return {"enabled": enabled, "sample_variance": sample_variance or [], "num_samples": num_samples or []}

    def _recon_block(enabled: bool, max_recon_train, max_recon_test) -> dict:
        return {
            "enabled": enabled,
            "max_recon_train": max_recon_train,
            "max_recon_test": max_recon_test,
        }

    return {
        "rmsd_gen": _gen_block(
            kw.get("do_rmsd", False),
            kw.get("variance_list") or [],
            kw.get("num_samples_list") or [],
        ),
        "rmsd_recon": _recon_block(
            kw.get("do_rmsd_recon", False),
            kw.get("max_recon_train_list"),
            kw.get("max_recon_test_list"),
        ),
        "q_gen": _gen_block(
            kw.get("do_q", False),
            kw.get("q_variance_list") or [],
            kw.get("q_num_samples_list") or [],
        ),
        "q_recon": _recon_block(
            kw.get("do_q_recon", False),
            kw.get("q_max_recon_train_list"),
            kw.get("q_max_recon_test_list"),
        ),
        "coord_clustering_gen": _gen_block(
            kw.get("do_coord_clustering_gen", False),
            kw.get("coord_clustering_variance_list") or [],
            kw.get("coord_clustering_num_samples_list") or [],
        ),
        "coord_clustering_recon": _recon_block(
            kw.get("do_coord_clustering_recon", False),
            kw.get("coord_clustering_max_recon_train_list"),
            kw.get("coord_clustering_max_recon_test_list"),
        ),
        "distmap_clustering_gen": _gen_block(
            kw.get("do_distmap_clustering_gen", False),
            kw.get("distmap_clustering_variance_list") or [],
            kw.get("distmap_clustering_num_samples_list") or [],
        ),
        "distmap_clustering_recon": _recon_block(
            kw.get("do_distmap_clustering_recon", False),
            kw.get("distmap_clustering_max_recon_train_list"),
            kw.get("distmap_clustering_max_recon_test_list"),
        ),
        "generative_capacity_rmsd": {
            "enabled": kw.get("do_generative_capacity_rmsd", False),
            "n_structures": kw.get("gc_rmsd_n_structures") or [],
        },
        "generative_capacity_q": {
            "enabled": kw.get("do_generative_capacity_q", False),
            "n_structures": kw.get("gc_q_n_structures") or [],
        },
        "latent": {"enabled": kw.get("do_latent", False)},
    }


def _euclideanizer_plotting_all_present(
    run_dir_eu: str,
    resume: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
    sample_variances: list,
) -> bool:
    """True if resume and all Euclideanizer plot files we would generate already exist."""
    if not resume:
        return False
    if do_recon_plot and not os.path.isfile(_plot_path(run_dir_eu, "reconstruction")):
        return False
    if do_bond_rg_scaling:
        for name in ("test", "train"):
            if not os.path.isfile(_plot_path(run_dir_eu, "recon_statistics", subset=name)):
                return False
    if do_avg_gen:
        for var in sample_variances:
            if not os.path.isfile(_plot_path(run_dir_eu, "gen_variance", var=str(var))):
                return False
    if do_bond_length_by_genomic_distance:
        for bt in (
            "bond_length_by_genomic_distance_gen",
            "bond_length_by_genomic_distance_train",
            "bond_length_by_genomic_distance_test",
        ):
            if not os.path.isfile(_plot_path(run_dir_eu, bt)):
                return False
    return True


def _euclideanizer_analysis_all_present(
    run_dir_eu: str,
    resume: bool,
    analysis_cfg: dict,
    metrics: list | None = None,
) -> bool:
    """True if resume and all enabled metric (rmsd, q, clustering, latent, generative capacity) outputs already exist.

    Callers must pass the same-shaped ``analysis_cfg`` as pipeline config (or use
    ``_analysis_cfg_from_need_data_kwargs`` for tests / need-data scans); there is no separate
    flattened-kw API so latent/GC toggles cannot be accidentally omitted.
    """
    if not resume:
        return True
    specs = metrics if metrics is not None else ANALYSIS_METRICS
    for spec in specs:
        gen_cfg = analysis_cfg[spec.gen_key]
        recon_cfg = analysis_cfg[spec.recon_key]
        do_gen = gen_cfg["enabled"]
        do_recon = recon_cfg["enabled"]
        if not do_gen and not do_recon:
            continue
        variance_list_s = _ensure_list(gen_cfg["sample_variance"])
        num_samples_list_s = _ensure_list(gen_cfg["num_samples"])
        if do_gen:
            for var in variance_list_s:
                # Always include variance in run_name so scoring can require sample_variance=1 only
                variance_suffix = f"_var{var}"
                for n in num_samples_list_s:
                    run_name = (str(n) if len(num_samples_list_s) > 1 else "default") + variance_suffix
                    fig_path = _analysis_path(run_dir_eu, spec.subdir, f"gen/{run_name}/{spec.figure_filename}")
                    if not os.path.isfile(fig_path):
                        return False
        max_recon_train_list_s = _ensure_list(recon_cfg["max_recon_train"])
        max_recon_test_list_s = _ensure_list(recon_cfg["max_recon_test"])
        if do_recon:
            if not max_recon_train_list_s:
                max_recon_train_list_s = [None]
            if not max_recon_test_list_s:
                max_recon_test_list_s = [None]
            n_recon = len(max_recon_train_list_s) * len(max_recon_test_list_s)
            if n_recon == 1:
                recon_fig = _analysis_path(run_dir_eu, spec.subdir, "recon/" + spec.figure_filename)
                if not os.path.isfile(recon_fig):
                    return False
            else:
                for max_train in max_recon_train_list_s:
                    for max_test in max_recon_test_list_s:
                        subdir = f"train{max_train}_test{max_test}"
                        recon_fig = _analysis_path(run_dir_eu, spec.subdir, f"recon/{subdir}/{spec.figure_filename}")
                        if not os.path.isfile(recon_fig):
                            return False
    latent_cfg = analysis_cfg["latent"]
    if latent_cfg["enabled"]:
        latent_dir = os.path.join(run_dir_eu, "analysis", "latent")
        if not os.path.isfile(os.path.join(latent_dir, "latent_distribution.png")) or not os.path.isfile(os.path.join(latent_dir, "latent_correlation.png")):
            return False
    gc_r = analysis_cfg["generative_capacity_rmsd"]
    if gc_r["enabled"]:
        if not os.path.isfile(os.path.join(run_dir_eu, "analysis", "generative_capacity", "rmsd", "generative_capacity_rmsd.png")):
            return False
    gc_q = analysis_cfg["generative_capacity_q"]
    if gc_q["enabled"]:
        if not os.path.isfile(os.path.join(run_dir_eu, "analysis", "generative_capacity", "q", "generative_capacity_q.png")):
            return False
    if gc_r["enabled"] and gc_q["enabled"]:
        comb = os.path.join(run_dir_eu, "analysis", "generative_capacity", "convergence_median_vs_n_rmsd_q.png")
        if not os.path.isfile(comb):
            return False
    return True


def _pipeline_need_data(
    base_output_dir: str,
    run_entries: list,
    training_splits: list,
    dm_groups: list,
    eu_groups: list,
    resume: bool,
    do_plot: bool,
    do_rmsd: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
    plot_variances: list,
    variance_list: list,
    num_samples_list: list,
    do_rmsd_recon: bool = False,
    max_recon_train_list: list | None = None,
    max_recon_test_list: list | None = None,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_variance_list: list | None = None,
    q_num_samples_list: list | None = None,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    do_generative_capacity_rmsd: bool = False,
    do_generative_capacity_q: bool = False,
) -> bool:
    """True if any run is incomplete or any plot/analysis output is missing (so we must load something)."""
    return _pipeline_data_needs(
        base_output_dir, run_entries, training_splits, dm_groups, eu_groups,
        resume, do_plot, do_rmsd, do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance,
        plot_variances, variance_list, num_samples_list,
        do_rmsd_recon=do_rmsd_recon,
        max_recon_train_list=max_recon_train_list, max_recon_test_list=max_recon_test_list,
        do_q=do_q, do_q_recon=do_q_recon,
        q_variance_list=q_variance_list or [], q_num_samples_list=q_num_samples_list or [],
        q_max_recon_train_list=q_max_recon_train_list or [], q_max_recon_test_list=q_max_recon_test_list or [],
        do_generative_capacity_rmsd=do_generative_capacity_rmsd,
        do_generative_capacity_q=do_generative_capacity_q,
    ).need_any()


@dataclass(frozen=True)
class PipelineDataNeeds:
    """What the pipeline must load for resume: only coords, only stats from cache, or both."""

    need_coords: bool  # training, reconstruction, recon_statistics, rmsd, or video frames
    need_exp_stats: bool  # gen_variance (full exp_stats)
    need_train_test_stats: bool  # recon_statistics or gen_variance (train/test split stats)

    def need_any(self) -> bool:
        return self.need_coords or self.need_exp_stats or self.need_train_test_stats


def _pipeline_data_needs(
    base_output_dir: str,
    run_entries: list,
    training_splits: list,
    dm_groups: list,
    eu_groups: list,
    resume: bool,
    do_plot: bool,
    do_rmsd: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
    plot_variances: list,
    variance_list: list,
    num_samples_list: list,
    do_rmsd_recon: bool = False,
    max_recon_train_list: list | None = None,
    max_recon_test_list: list | None = None,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_variance_list: list | None = None,
    q_num_samples_list: list | None = None,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    do_coord_clustering_gen: bool = False,
    do_coord_clustering_recon: bool = False,
    coord_clustering_variance_list: list | None = None,
    coord_clustering_num_samples_list: list | None = None,
    coord_clustering_max_recon_train_list: list | None = None,
    coord_clustering_max_recon_test_list: list | None = None,
    do_distmap_clustering_gen: bool = False,
    do_distmap_clustering_recon: bool = False,
    distmap_clustering_variance_list: list | None = None,
    distmap_clustering_num_samples_list: list | None = None,
    distmap_clustering_max_recon_train_list: list | None = None,
    distmap_clustering_max_recon_test_list: list | None = None,
    do_latent: bool = False,
    do_generative_capacity_rmsd: bool = False,
    do_generative_capacity_q: bool = False,
) -> PipelineDataNeeds:
    """
    Scan pipeline outputs and return which data is required.
    - need_coords: any run incomplete, or any reconstruction / recon_statistics / rmsd / q / generative capacity / … analysis missing.
    - need_exp_stats: any gen_variance plot missing.
    - need_train_test_stats: any recon_statistics or gen_variance missing.
    """
    need_coords = False
    need_exp_stats = False
    need_train_test_stats = False
    _analysis_cfg = _analysis_cfg_from_need_data_kwargs({
        "do_rmsd": do_rmsd,
        "variance_list": variance_list,
        "num_samples_list": num_samples_list,
        "do_rmsd_recon": do_rmsd_recon,
        "max_recon_train_list": max_recon_train_list,
        "max_recon_test_list": max_recon_test_list,
        "do_q": do_q,
        "do_q_recon": do_q_recon,
        "q_variance_list": q_variance_list or [],
        "q_num_samples_list": q_num_samples_list or [],
        "q_max_recon_train_list": q_max_recon_train_list,
        "q_max_recon_test_list": q_max_recon_test_list,
        "do_coord_clustering_gen": do_coord_clustering_gen,
        "do_coord_clustering_recon": do_coord_clustering_recon,
        "coord_clustering_variance_list": coord_clustering_variance_list or [],
        "coord_clustering_num_samples_list": coord_clustering_num_samples_list or [],
        "coord_clustering_max_recon_train_list": coord_clustering_max_recon_train_list,
        "coord_clustering_max_recon_test_list": coord_clustering_max_recon_test_list,
        "do_distmap_clustering_gen": do_distmap_clustering_gen,
        "do_distmap_clustering_recon": do_distmap_clustering_recon,
        "distmap_clustering_variance_list": distmap_clustering_variance_list or [],
        "distmap_clustering_num_samples_list": distmap_clustering_num_samples_list or [],
        "distmap_clustering_max_recon_train_list": distmap_clustering_max_recon_train_list,
        "distmap_clustering_max_recon_test_list": distmap_clustering_max_recon_test_list,
        "do_latent": do_latent,
        "do_generative_capacity_rmsd": do_generative_capacity_rmsd,
        "do_generative_capacity_q": do_generative_capacity_q,
    })
    for entry in run_entries:
        seed, training_split, max_data = _entry_seed_split_max(entry)
        output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
        if not os.path.isdir(output_dir):
            need_coords = True
            need_exp_stats = need_exp_stats or do_plot and do_avg_gen
            need_train_test_stats = need_train_test_stats or (do_plot and (do_bond_rg_scaling or do_avg_gen or do_bond_length_by_genomic_distance))
            continue
        for group in dm_groups:
            base_config, checkpoints = group["base_config"], group["checkpoints"]
            checkpoint_dirs = [os.path.join(output_dir, "distmap", str(ri)) for ri, _ in checkpoints]
            dm_multi = len(checkpoints) > 1
            dm_save_final = base_config["save_final_models_per_stretch"]
            for seg_idx, (ri, ev) in enumerate(checkpoints):
                run_dir_dm = checkpoint_dirs[seg_idx]
                dm_last = seg_idx == len(checkpoints) - 1
                if not _run_completed(
                    run_dir_dm, ev,
                    section_key="distmap",
                    expected_section=None,
                    multi_segment=dm_multi,
                    checkpoint_last_name="model_last.pt" if dm_multi else None,
                    is_last_segment=dm_last,
                    save_final_models_per_stretch=dm_save_final,
                ):
                    need_coords = True
                if do_plot:
                    if do_recon_plot and (not resume or not os.path.isfile(_plot_path(run_dir_dm, "reconstruction"))):
                        need_coords = True
                    if do_bond_rg_scaling:
                        for name in ("test", "train"):
                            if not resume or not os.path.isfile(_plot_path(run_dir_dm, "recon_statistics", subset=name)):
                                need_coords = True
                                need_train_test_stats = True
                    if do_avg_gen:
                        for var in plot_variances:
                            if not resume or not os.path.isfile(_plot_path(run_dir_dm, "gen_variance", var=str(var))):
                                need_exp_stats = True
                                need_train_test_stats = True
                    if do_bond_length_by_genomic_distance:
                        for _bt in (
                            "bond_length_by_genomic_distance_gen",
                            "bond_length_by_genomic_distance_train",
                            "bond_length_by_genomic_distance_test",
                        ):
                            if not resume or not os.path.isfile(_plot_path(run_dir_dm, _bt)):
                                need_coords = True
                                need_train_test_stats = True
            for egidx, eu_group in enumerate(eu_groups):
                eu_base = eu_group["base_config"]
                eu_checkpoints = eu_group["checkpoints"]
                eu_save_final = eu_base["save_final_models_per_stretch"]
                for seg_idx, (ri, ev) in enumerate(checkpoints):
                    eu_checkpoint_dirs = [os.path.join(output_dir, "distmap", str(ri), "euclideanizer", str(euri)) for euri, _ in eu_checkpoints]
                    eu_multi = len(eu_checkpoints) > 1
                    for eu_seg_idx, (euri, eu_ev) in enumerate(eu_checkpoints):
                        eu_run_dir = eu_checkpoint_dirs[eu_seg_idx]
                        eu_last = eu_seg_idx == len(eu_checkpoints) - 1
                        if not _run_completed(
                            eu_run_dir, eu_ev,
                            section_key="euclideanizer",
                            expected_section=None,
                            multi_segment=eu_multi,
                            checkpoint_last_name="euclideanizer_last.pt" if eu_multi else None,
                            is_last_segment=eu_last,
                            save_final_models_per_stretch=eu_save_final,
                        ):
                            need_coords = True
                        if do_plot:
                            if do_recon_plot and (not resume or not os.path.isfile(_plot_path(eu_run_dir, "reconstruction"))):
                                need_coords = True
                            if do_bond_rg_scaling:
                                for name in ("test", "train"):
                                    if not resume or not os.path.isfile(_plot_path(eu_run_dir, "recon_statistics", subset=name)):
                                        need_coords = True
                                        need_train_test_stats = True
                            if do_avg_gen:
                                for var in plot_variances:
                                    if not resume or not os.path.isfile(_plot_path(eu_run_dir, "gen_variance", var=str(var))):
                                        need_exp_stats = True
                                        need_train_test_stats = True
                            if do_bond_length_by_genomic_distance:
                                for _bt in (
                                    "bond_length_by_genomic_distance_gen",
                                    "bond_length_by_genomic_distance_train",
                                    "bond_length_by_genomic_distance_test",
                                ):
                                    if not resume or not os.path.isfile(_plot_path(eu_run_dir, _bt)):
                                        need_coords = True
                                        need_train_test_stats = True
                        has_any_analysis = any(
                            _analysis_cfg[spec.gen_key]["enabled"]
                            or _analysis_cfg[spec.recon_key]["enabled"]
                            for spec in ANALYSIS_METRICS
                        ) or _analysis_cfg["generative_capacity_rmsd"]["enabled"] or _analysis_cfg["generative_capacity_q"]["enabled"]
                        if has_any_analysis and not _euclideanizer_analysis_all_present(
                            eu_run_dir, resume, analysis_cfg=_analysis_cfg
                        ):
                            need_coords = True
    return PipelineDataNeeds(need_coords=need_coords, need_exp_stats=need_exp_stats, need_train_test_stats=need_train_test_stats)


def _video_frames_dir(run_root: str) -> str:
    return os.path.join(run_root, "training_video", "frames")


def _video_mp4_path(run_root: str) -> str:
    return os.path.join(run_root, "training_video", "training_evolution.mp4")


def _parse_args():
    """Parse command-line arguments; --config is required unless --worker-from-pickle is used."""
    p = argparse.ArgumentParser(description="Euclideanizer pipeline: DistMap + Euclideanizer training and plotting")
    p.add_argument("--worker-from-pickle", type=str, default=None, dest="worker_from_pickle", help=argparse.SUPPRESS)
    p.add_argument("--data", type=str, default=None, help="Path to dataset (required for training)")
    p.add_argument("--config", type=str, required=("--worker-from-pickle" not in sys.argv), help="Path to YAML config (e.g. samples/config_sample.yaml)")
    p.add_argument("--no-plots", action="store_true", help="Disable all plotting")
    p.add_argument("--no-dashboard", action="store_true", dest="no_dashboard", help="Do not build interactive dashboard in run root")
    p.add_argument("--no-resume", action="store_true", help="Do not resume; overwrite existing run outputs")
    p.add_argument("--yes-overwrite", action="store_true", help="With --no-resume: skip confirmation prompt (use for SLURM/scripted runs)")
    p.add_argument("--no-multi-gpu", action="store_true", dest="no_multi_gpu", help="Disable multi-GPU parallelization even when 2+ CUDA devices are available")
    p.add_argument("--gpus", type=int, default=None, metavar="N", help="Use at most N CUDA devices for multi-GPU (default: use all available)")
    p.add_argument("--output-dir", type=str, default=None, dest="output_dir", help="Output directory")
    # DistMap
    p.add_argument("--distmap.beta_kl", type=float, nargs="*", default=None, dest="distmap_beta_kl")
    p.add_argument("--distmap.latent_dim", type=int, default=None, dest="distmap_latent_dim")
    p.add_argument("--distmap.epochs", type=int, default=None, dest="distmap_epochs")
    p.add_argument("--distmap.batch_size", type=int, default=None, dest="distmap_batch_size")
    p.add_argument("--distmap.learning_rate", type=float, default=None, dest="distmap_lr")
    p.add_argument("--data.training_split", type=float, default=None, dest="data_training_split")
    p.add_argument("--data.split_seed", type=int, default=None, dest="data_split_seed")
    # Euclideanizer
    p.add_argument("--euclideanizer.frozen_vae_beta_kl", type=float, default=None, dest="eu_frozen_vae_beta_kl")
    p.add_argument("--euclideanizer.frozen_vae_path", type=str, default=None, dest="eu_frozen_vae_path")
    p.add_argument("--euclideanizer.epochs", type=int, nargs="*", default=None, dest="eu_epochs")
    p.add_argument("--euclideanizer.batch_size", type=int, default=None, dest="eu_batch_size")
    p.add_argument("--euclideanizer.learning_rate", type=float, nargs="*", default=None, dest="eu_lr")
    p.add_argument("--euclideanizer.lambda_mse", type=float, nargs="*", default=None, dest="eu_lambda_mse")
    p.add_argument("--euclideanizer.lambda_w_recon", type=float, nargs="*", default=None, dest="eu_lambda_w_recon")
    p.add_argument("--euclideanizer.lambda_w_gen", type=float, nargs="*", default=None, dest="eu_lambda_w_gen")
    p.add_argument("--euclideanizer.lambda_w_diag_recon", type=float, nargs="*", default=None, dest="eu_lambda_w_diag_recon")
    p.add_argument("--euclideanizer.lambda_w_diag_gen", type=float, nargs="*", default=None, dest="eu_lambda_w_diag_gen")
    # Generation / plotting
    p.add_argument("--generation.num_samples", type=int, default=None, dest="gen_num_samples")
    p.add_argument("--generation.sample_variance", type=float, nargs="*", default=None, dest="gen_sample_variance")
    p.add_argument("--plotting.num_reconstruction_samples", type=int, default=None, dest="plot_num_recon")
    return p.parse_args()


def _args_to_overrides(args) -> dict:
    """Convert parsed CLI args into a config overlay dict (nested keys for data, distmap, euclideanizer, plotting)."""
    o = {}
    if args.data is not None:
        o.setdefault("data", {})["path"] = args.data
    if args.output_dir is not None:
        o["output_dir"] = args.output_dir
    if args.no_plots:
        o.setdefault("plotting", {})["enabled"] = False
    if getattr(args, "no_dashboard", False):
        o.setdefault("dashboard", {})["enabled"] = False
    if args.no_resume:
        o["resume"] = False
    if args.distmap_beta_kl is not None and len(args.distmap_beta_kl) > 0:
        o.setdefault("distmap", {})["beta_kl"] = args.distmap_beta_kl
    if args.data_split_seed is not None:
        o.setdefault("data", {})["split_seed"] = args.data_split_seed
    if args.data_training_split is not None:
        o.setdefault("data", {})["training_split"] = args.data_training_split
    for arg_name, section, key in [
        ("distmap_latent_dim", "distmap", "latent_dim"),
        ("distmap_epochs", "distmap", "epochs"),
        ("distmap_batch_size", "distmap", "batch_size"),
        ("distmap_lr", "distmap", "learning_rate"),
    ]:
        if getattr(args, arg_name, None) is not None:
            o.setdefault(section, {})[key] = getattr(args, arg_name)
    for arg_name, section, key in [
        ("eu_frozen_vae_beta_kl", "euclideanizer", "frozen_vae_beta_kl"),
        ("eu_frozen_vae_path", "euclideanizer", "frozen_vae_path"),
        ("eu_epochs", "euclideanizer", "epochs"),
        ("eu_batch_size", "euclideanizer", "batch_size"),
        ("eu_lr", "euclideanizer", "learning_rate"),
        ("eu_lambda_mse", "euclideanizer", "lambda_mse"),
        ("eu_lambda_w_recon", "euclideanizer", "lambda_w_recon"),
        ("eu_lambda_w_gen", "euclideanizer", "lambda_w_gen"),
        ("eu_lambda_w_diag_recon", "euclideanizer", "lambda_w_diag_recon"),
        ("eu_lambda_w_diag_gen", "euclideanizer", "lambda_w_diag_gen"),
    ]:
        val = getattr(args, arg_name, None)
        if val is not None:
            if isinstance(val, list) and len(val) == 0:
                continue
            o.setdefault(section, {})[key] = val
    if args.gen_num_samples is not None:
        o.setdefault("plotting", {})["num_samples"] = args.gen_num_samples
    if args.gen_sample_variance is not None and len(args.gen_sample_variance) > 0:
        o.setdefault("plotting", {})["sample_variance"] = args.gen_sample_variance
    if args.plot_num_recon is not None:
        o.setdefault("plotting", {})["num_reconstruction_samples"] = args.plot_num_recon
    return o


def _capped_train_test_subset(subset_ds, max_structures: int | None):
    """First max_structures samples of the train/test split (same order as exp_stats); None = full split."""
    if max_structures is None or len(subset_ds) <= max_structures:
        return subset_ds
    idx = subset_ds.indices[:max_structures]
    return torch.utils.data.Subset(subset_ds.dataset, idx)


def _get_recon_dm_distmap(
    model,
    device,
    coords,
    dm_cfg,
    training_split,
    split_seed,
    utils_mod,
    use_train: bool = False,
    max_structures: int | None = None,
):
    """Compute reconstruction distance maps for the DistMap on train or test split; returns (n_samples, N, N) numpy array.
    max_structures: cap count to match plotting.max_train / max_test (experimental stats); None = all in split."""
    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    subset_ds = train_ds if use_train else test_ds
    subset_ds = _capped_train_test_subset(subset_ds, max_structures)
    dl = torch.utils.data.DataLoader(subset_ds, batch_size=dm_cfg["batch_size"], shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for batch in dl:
            batch_dm = utils_mod.get_distmaps(batch)
            mu, logvar, z, recon_tri = model(batch_dm)
            recon_full = utils_mod.upper_tri_to_symmetric(torch.expm1(recon_tri), coords.size(1))
            out.append(recon_full.cpu().numpy())
    return np.concatenate(out, axis=0)


def _get_recon_dm_euclideanizer(
    embed,
    frozen_vae,
    device,
    coords,
    eu_cfg,
    training_split,
    split_seed,
    utils_mod,
    use_train: bool = False,
    max_structures: int | None = None,
):
    """Compute reconstruction distance maps for the Euclideanizer on train or test split; returns (n_samples, N, N) numpy array.
    max_structures: cap to match plotting.max_train / max_test; None = all in split."""
    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    subset_ds = train_ds if use_train else test_ds
    subset_ds = _capped_train_test_subset(subset_ds, max_structures)
    dl = torch.utils.data.DataLoader(subset_ds, batch_size=128, shuffle=False)
    embed.eval()
    out = []
    with torch.no_grad():
        for batch in dl:
            batch_dm = utils_mod.get_distmaps(batch)
            gt_log = torch.log1p(batch_dm)
            mu = frozen_vae.encode(gt_log)
            D_noneuclid = frozen_vae._decode_to_matrix(mu)
            coords_out = embed(D_noneuclid)
            out.append(utils_mod.get_distmaps(coords_out).cpu().numpy())
    return np.concatenate(out, axis=0)


def _run_bond_length_plot_suite_distmap(
    run_dir: str,
    model,
    device,
    coords,
    dm_cfg: dict,
    training_split: float,
    split_seed: int,
    train_stats: dict,
    test_stats: dict,
    gen_dm_bond,
    plot_max_train,
    plot_max_test,
    gen_num_samples: int,
    sample_variances: list,
    gen_decode_batch_size,
    plot_dpi: int,
    save_pdf: bool,
    save_plot_data: bool,
    base_output_dir: str,
) -> None:
    if gen_dm_bond is None:
        gen_dm_bond = _get_gen_dm_distmap(
            model, device, gen_num_samples, dm_cfg["latent_dim"],
            sample_variances[0] if sample_variances else 1.0, gen_decode_batch_size,
        )
    plot_bond_length_by_genomic_distance(
        train_stats["exp_distmaps"],
        test_stats["exp_distmaps"],
        gen_dm_bond,
        _plot_path(run_dir, "bond_length_by_genomic_distance_gen"),
        label_gen="Gen", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
        display_root=base_output_dir,
    )
    if coords is None:
        return
    rtr = _get_recon_dm_distmap(
        model, device, coords, dm_cfg, training_split, split_seed, utils,
        use_train=True, max_structures=plot_max_train,
    )
    rte = _get_recon_dm_distmap(
        model, device, coords, dm_cfg, training_split, split_seed, utils,
        use_train=False, max_structures=plot_max_test,
    )
    plot_pairwise_distance_by_lag_exp_vs_recon(
        train_stats["exp_distmaps"], rtr,
        _plot_path(run_dir, "bond_length_by_genomic_distance_train"),
        exp_label="Train", recon_label="Recon", suptitle_suffix="Train — Exp Vs Recon", subset="train",
        dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
        display_root=base_output_dir, data_key_prefix="train",
    )
    plot_pairwise_distance_by_lag_exp_vs_recon(
        test_stats["exp_distmaps"], rte,
        _plot_path(run_dir, "bond_length_by_genomic_distance_test"),
        exp_label="Test", recon_label="Recon", suptitle_suffix="Test — Exp Vs Recon", subset="test",
        dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
        display_root=base_output_dir, data_key_prefix="test",
    )


def _run_bond_length_plot_suite_euclideanizer(
    run_dir: str,
    embed,
    frozen_vae,
    device,
    coords,
    eu_cfg: dict,
    dm_latent_dim: int,
    training_split: float,
    split_seed: int,
    train_stats: dict,
    test_stats: dict,
    gen_dm_bond,
    plot_max_train,
    plot_max_test,
    gen_num_samples: int,
    sample_variances: list,
    gen_decode_batch_size,
    plot_dpi: int,
    save_pdf: bool,
    save_plot_data: bool,
    base_output_dir: str,
) -> None:
    if gen_dm_bond is None:
        gen_dm_bond = _get_gen_dm_euclideanizer(
            embed, frozen_vae, device, gen_num_samples, dm_latent_dim,
            sample_variances[0] if sample_variances else 1.0, utils, gen_decode_batch_size,
        )
    plot_bond_length_by_genomic_distance(
        train_stats["exp_distmaps"],
        test_stats["exp_distmaps"],
        gen_dm_bond,
        _plot_path(run_dir, "bond_length_by_genomic_distance_gen"),
        label_gen="Gen", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
        display_root=base_output_dir,
    )
    if coords is None:
        return
    rtr = _get_recon_dm_euclideanizer(
        embed, frozen_vae, device, coords, eu_cfg, training_split, split_seed, utils,
        use_train=True, max_structures=plot_max_train,
    )
    rte = _get_recon_dm_euclideanizer(
        embed, frozen_vae, device, coords, eu_cfg, training_split, split_seed, utils,
        use_train=False, max_structures=plot_max_test,
    )
    plot_pairwise_distance_by_lag_exp_vs_recon(
        train_stats["exp_distmaps"], rtr,
        _plot_path(run_dir, "bond_length_by_genomic_distance_train"),
        exp_label="Train", recon_label="Recon", suptitle_suffix="Train — Exp Vs Recon", subset="train",
        dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
        display_root=base_output_dir, data_key_prefix="train",
    )
    plot_pairwise_distance_by_lag_exp_vs_recon(
        test_stats["exp_distmaps"], rte,
        _plot_path(run_dir, "bond_length_by_genomic_distance_test"),
        exp_label="Test", recon_label="Recon", suptitle_suffix="Test — Exp Vs Recon", subset="test",
        dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
        display_root=base_output_dir, data_key_prefix="test",
    )


def _get_recon_coords_euclideanizer(embed, frozen_vae, device, coords, training_split, split_seed, utils_mod, use_train: bool = False, max_n: int | None = None):
    """Compute reconstruction 3D coords for the Euclideanizer on train or test split; returns (n, N, 3) numpy array. If max_n is set, use at most that many structures."""
    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    subset_ds = train_ds if use_train else test_ds
    dl = torch.utils.data.DataLoader(subset_ds, batch_size=128, shuffle=False)
    embed.eval()
    out = []
    with torch.no_grad():
        for batch in dl:
            batch_dm = utils_mod.get_distmaps(batch)
            gt_log = torch.log1p(batch_dm)
            mu = frozen_vae.encode(gt_log)
            D_noneuclid = frozen_vae._decode_to_matrix(mu)
            coords_out = embed(D_noneuclid)
            out.append(coords_out.cpu().numpy())
    arr = np.concatenate(out, axis=0)
    if max_n is not None and arr.shape[0] > max_n:
        arr = arr[:max_n]
    return arr.astype(np.float32)


def _get_latent_vectors_euclideanizer(frozen_vae, device, coords, training_split, split_seed, utils_mod, max_train: int | None = None, max_test: int | None = None):
    """Return (train_mu_np, test_mu_np) each (n, latent_dim). Optional max_train/max_test cap the number of structures."""
    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    out_train = []
    out_test = []
    frozen_vae.eval()
    with torch.no_grad():
        for subset_ds, out_list, max_n in [(train_ds, out_train, max_train), (test_ds, out_test, max_test)]:
            dl = torch.utils.data.DataLoader(subset_ds, batch_size=128, shuffle=False)
            for batch in dl:
                batch_dm = utils_mod.get_distmaps(batch)
                gt_log = torch.log1p(batch_dm)
                mu = frozen_vae.encode(gt_log)
                out_list.append(mu.cpu().numpy())
    train_mu = np.concatenate(out_train, axis=0).astype(np.float32)
    test_mu = np.concatenate(out_test, axis=0).astype(np.float32)
    if max_train is not None and train_mu.shape[0] > max_train:
        train_mu = train_mu[:max_train]
    if max_test is not None and test_mu.shape[0] > max_test:
        test_mu = test_mu[:max_test]
    return train_mu, test_mu


def _run_scoring_for_run(
    run_dir_eu: str,
    seed_dir: str,
    cfg: dict,
    base_output_dir: str,
    pipeline_start: float,
) -> None:
    """Run scoring for one Euclideanizer run (after a plotting/analysis block). Always overwrites scores with current data."""
    if not cfg["scoring"]["enabled"]:
        return
    try:
        _log("Scoring: loading and computing (may take several minutes)...", since_start=time.time() - pipeline_start, style="info")
        out_path = scoring_module.compute_and_save(run_dir_eu, seed_dir, cfg, scores_filename="scores.json")
        if out_path:
            _log(
                f"Scoring: {utils.display_path(out_path, base_output_dir)}",
                since_start=time.time() - pipeline_start,
                style="success",
            )
    except Exception as e:
        _log(f"Scoring failed for {run_dir_eu}: {e}", since_start=time.time() - pipeline_start, style="error")


def _post_scoring_npz_cleanup(run_dir_eu: str, cfg: dict, *, defer_sufficiency_inputs: bool = False) -> None:
    """Remove NPZ data dirs for blocks where save_data is false (scoring has already read them). One run_dir_eu (one Euclideanizer run).

    When sufficiency meta-analysis is enabled, defers deletion of ``analysis/rmsd/recon`` and
    ``analysis/q/recon`` ``data/`` trees until after meta-analysis (see ``_finalize_deferred_npz_cleanup``).
    """
    if not cfg.get("scoring", {}).get("enabled"):
        return
    plot_cfg = cfg["plotting"]
    ana = cfg["analysis"]
    if not plot_cfg["save_data"]:
        for sub in (
            "recon_statistics",
            "gen_variance",
            "bond_length_by_genomic_distance_gen",
            "bond_length_by_genomic_distance_train",
            "bond_length_by_genomic_distance_test",
        ):
            data_dir = os.path.join(run_dir_eu, "plots", sub, "data")
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir, ignore_errors=True)
    if ana.get("latent") and not ana["latent"].get("save_data"):
        latent_data = os.path.join(run_dir_eu, "analysis", "latent", "data")
        if os.path.isdir(latent_data):
            shutil.rmtree(latent_data, ignore_errors=True)
    for spec in ANALYSIS_METRICS:
        gen_save = ana[spec.gen_key]["save_data"]
        recon_save = ana[spec.recon_key]["save_data"]
        for sub, save_ok in [("gen", gen_save), ("recon", recon_save)]:
            if save_ok:
                continue
            if defer_sufficiency_inputs and sub == "recon" and spec.id in ("rmsd", "q"):
                continue
            branch = os.path.join(run_dir_eu, "analysis", spec.subdir, sub)
            if not os.path.isdir(branch):
                continue
            for root, dirs, _ in os.walk(branch, topdown=True):
                if "data" in dirs:
                    shutil.rmtree(os.path.join(root, "data"), ignore_errors=True)


def _finalize_deferred_npz_cleanup(base_output_dir: str, cfg: dict) -> None:
    """Run post-scoring cleanup without sufficiency deferral for all Euclideanizer runs."""
    for run_dir_eu, _seed_dir in _iter_euclideanizer_runs(base_output_dir):
        _post_scoring_npz_cleanup(run_dir_eu, cfg, defer_sufficiency_inputs=False)


def _run_sufficiency_meta_analysis(
    base_output_dir: str,
    cfg: dict,
    pipeline_start: float,
) -> bool:
    """Run sufficiency meta-analysis and return True if outputs were created."""
    suff = cfg["meta_analysis"]["sufficiency"]
    if not suff["enabled"]:
        return False
    _log("Sufficiency meta-analysis: building figures...", since_start=time.time() - pipeline_start, style="info")
    made = meta_analysis_module.run_sufficiency_meta_analysis(
        base_output_dir=base_output_dir,
        max_data_values=get_max_data_values(cfg),
        save_pdf_copy=bool(suff["save_pdf_copy"]),
        log=lambda m: _log(m, since_start=time.time() - pipeline_start, style="info"),
    )
    if made:
        _log("Sufficiency meta-analysis: complete.", since_start=time.time() - pipeline_start, style="success")
    else:
        _log("Sufficiency meta-analysis: no outputs created.", since_start=time.time() - pipeline_start, style="skip")
    return made


def _run_generative_capacity_blocks_for_run(
    *,
    run_dir_eu: str,
    analysis_cfg: dict,
    seed: int,
    latent_dim: int,
    device,
    frozen_vae,
    embed,
    resume: bool,
    pipeline_start: float,
    display_root: str | None,
) -> None:
    combined_path = os.path.join(run_dir_eu, "analysis", "generative_capacity", "convergence_median_vs_n_rmsd_q.png")
    need_combined = (not resume) or (not os.path.isfile(combined_path))
    by_n_rmsd: dict[int, np.ndarray] | None = None
    by_n_q: dict[int, np.ndarray] | None = None
    ran_r = False
    ran_q = False

    gc_r = analysis_cfg["generative_capacity_rmsd"]
    if gc_r["enabled"]:
        fig_r = os.path.join(run_dir_eu, "analysis", "generative_capacity", "rmsd", "generative_capacity_rmsd.png")
        if (not resume) or (not os.path.isfile(fig_r)):
            ran_r = True
            _log("Running generative capacity RMSD analysis...", since_start=time.time() - pipeline_start, style="info")
            _, by_n_rmsd = generative_capacity_module.run_generative_capacity_rmsd(
                run_dir=run_dir_eu,
                seed=seed,
                latent_dim=latent_dim,
                device=device,
                frozen_vae=frozen_vae,
                embed=embed,
                cfg_block=gc_r,
                display_root=display_root,
            )
        elif bool(gc_r["save_data"]) and need_combined:
            by_n_rmsd = generative_capacity_module.try_load_gc_by_n_from_npz(
                run_dir_eu, metric="rmsd", n_structures=gc_r["n_structures"]
            )
    gc_q = analysis_cfg["generative_capacity_q"]
    if gc_q["enabled"]:
        fig_q = os.path.join(run_dir_eu, "analysis", "generative_capacity", "q", "generative_capacity_q.png")
        if (not resume) or (not os.path.isfile(fig_q)):
            ran_q = True
            _log("Running generative capacity Q analysis...", since_start=time.time() - pipeline_start, style="info")
            _, by_n_q = generative_capacity_module.run_generative_capacity_q(
                run_dir=run_dir_eu,
                seed=seed,
                latent_dim=latent_dim,
                device=device,
                frozen_vae=frozen_vae,
                embed=embed,
                cfg_block=gc_q,
                display_root=display_root,
            )
        elif bool(gc_q["save_data"]) and need_combined:
            by_n_q = generative_capacity_module.try_load_gc_by_n_from_npz(
                run_dir_eu, metric="q", n_structures=gc_q["n_structures"]
            )

    did_backfill = False
    if (
        gc_r["enabled"]
        and gc_q["enabled"]
        and need_combined
        and (by_n_rmsd is None or by_n_q is None)
    ):
        did_backfill = True
        _log(
            "Generative capacity: recomputing RMSD/Q for median-vs-N convergence figure...",
            since_start=time.time() - pipeline_start,
            style="info",
        )
        if by_n_rmsd is None:
            _, by_n_rmsd = generative_capacity_module.run_generative_capacity_rmsd(
                run_dir=run_dir_eu,
                seed=seed,
                latent_dim=latent_dim,
                device=device,
                frozen_vae=frozen_vae,
                embed=embed,
                cfg_block=gc_r,
                display_root=display_root,
            )
        if by_n_q is None:
            _, by_n_q = generative_capacity_module.run_generative_capacity_q(
                run_dir=run_dir_eu,
                seed=seed,
                latent_dim=latent_dim,
                device=device,
                frozen_vae=frozen_vae,
                embed=embed,
                cfg_block=gc_q,
                display_root=display_root,
            )

    if gc_r["enabled"] and gc_q["enabled"] and by_n_rmsd is not None and by_n_q is not None:
        if (not os.path.isfile(combined_path)) or ran_r or ran_q or did_backfill:
            generative_capacity_module.save_generative_capacity_convergence_combined(
                run_dir=run_dir_eu,
                by_n_rmsd=by_n_rmsd,
                by_n_q=by_n_q,
                save_pdf_copy=bool(gc_r["save_pdf_copy"] or gc_q["save_pdf_copy"]),
                display_root=display_root,
            )


def run_one_hpo_trial(
    cfg: dict,
    trial_dir: str,
    optuna_trial,
    device,
    coords,
    coords_np,
    num_atoms: int,
    num_structures: int,
    exp_stats,
    train_stats,
    test_stats,
    data_path: str,
    dm_epochs_max: int,
) -> float:
    """Run one HPO trial in-process: train one DistMap, one Euclideanizer, plot, analyze, score.
    Reports validation loss each epoch to optuna_trial and raises optuna.TrialPruned if the pruner says so.
    dm_epochs_max is the maximum DistMap epochs across all trials (caller computes it); Euclideanizer reports use step = dm_epochs_max + epoch so steps do not overlap with DistMap. Returns overall_score from scoring. Caller must load data and compute exp_stats/train_stats/test_stats.
    """
    import optuna

    global _LOG_FILE, _LOG_TRIAL_PREFIX
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.close()
        except Exception:
            pass
        _LOG_FILE = None
    _init_log_file(trial_dir)
    trial_num = getattr(optuna_trial, "number", -1)
    _LOG_TRIAL_PREFIX = f"[trial {trial_num}] "
    pipeline_start = time.time()
    base_output_dir = trial_dir
    seed = int(cfg["data"]["split_seed"])
    training_split = float(cfg["data"]["training_split"])
    seed_dir = os.path.join(trial_dir, f"seed_{seed}")
    run_dir_dm = os.path.join(seed_dir, "distmap", "0")
    run_dir_eu = os.path.join(run_dir_dm, "euclideanizer", "0")
    os.makedirs(run_dir_dm, exist_ok=True)
    os.makedirs(run_dir_eu, exist_ok=True)
    save_pipeline_config(cfg, trial_dir)

    if train_stats is not None and test_stats is not None and data_path:
        plot_cfg = cfg["plotting"]
        plot_mt = plot_cfg["max_train"]
        plot_mc = plot_cfg["max_test"]
        _save_exp_stats_split_cache(
            seed_dir, data_path, num_structures, num_atoms,
            seed, training_split, train_stats, test_stats,
            max_train=plot_mt, max_test=plot_mc,
        )

    dm_cfg = cfg["distmap"]
    eu_cfg = cfg["euclideanizer"]
    plot_cfg = cfg["plotting"]
    analysis_cfg = cfg["analysis"]
    do_plot = plot_cfg["enabled"]
    plot_dpi = int(plot_cfg["plot_dpi"])
    save_pdf = plot_cfg["save_pdf_copy"]
    scoring_enabled = bool(cfg["scoring"]["enabled"])
    save_plot_data = plot_cfg["save_data"] or scoring_enabled
    num_recon_samples = plot_cfg["num_reconstruction_samples"]
    do_recon_plot = plot_cfg["reconstruction"]
    do_bond_rg_scaling = plot_cfg["bond_rg_scaling"]
    do_avg_gen = plot_cfg["avg_gen_vs_exp"]
    do_bond_length_by_genomic_distance = plot_cfg["bond_length_by_genomic_distance"]
    sample_variances = get_sample_variances(cfg) if do_plot else []
    gen_num_samples = plot_cfg["num_samples"]
    gen_decode_batch_size = plot_cfg["gen_decode_batch_size"]

    # Match normal pipeline startup logging (run.py main)
    do_rmsd = analysis_cfg["rmsd_gen"]["enabled"]
    do_rmsd_recon_cfg = analysis_cfg["rmsd_recon"]["enabled"]
    do_q = analysis_cfg["q_gen"]["enabled"]
    do_q_recon_cfg = analysis_cfg["q_recon"]["enabled"]
    do_gc_rmsd = analysis_cfg["generative_capacity_rmsd"]["enabled"]
    do_gc_q = analysis_cfg["generative_capacity_q"]["enabled"]
    do_coord_clustering_gen = analysis_cfg["coord_clustering_gen"]["enabled"]
    do_coord_clustering_recon_cfg = analysis_cfg["coord_clustering_recon"]["enabled"]
    do_distmap_clustering_gen = analysis_cfg["distmap_clustering_gen"]["enabled"]
    do_distmap_clustering_recon_cfg = analysis_cfg["distmap_clustering_recon"]["enabled"]
    _log("Pipeline started.", since_start=time.time() - pipeline_start, style="info")
    _warn_calibration_reserve_if_low(cfg, pipeline_start)
    _log(f"config: (HPO trial)  output: {base_output_dir}  seeds: [{seed}]", since_start=time.time() - pipeline_start, style="info")
    _log(f"DistMap runs: 1  Euclideanizer: 1  resume=False  plot={do_plot}  rmsd_gen={do_rmsd}  rmsd_recon={do_rmsd_recon_cfg}  q_gen={do_q}  q_recon={do_q_recon_cfg}  gc_rmsd={do_gc_rmsd}  gc_q={do_gc_q}  coord_clustering_gen={do_coord_clustering_gen}  coord_clustering_recon={do_coord_clustering_recon_cfg}  distmap_clustering_gen={do_distmap_clustering_gen}  distmap_clustering_recon={do_distmap_clustering_recon_cfg}", since_start=time.time() - pipeline_start, style="info")

    dm_epochs = int(dm_cfg["epochs"])
    # DistMap reports step=epoch (1..dm_epochs). Euclideanizer reports step=dm_epochs_max+epoch so its range never overlaps and the pruner can compare like-with-like across trials (same step range for all Euclideanizer runs). dm_epochs_max is passed by the HPO caller.

    def _report_and_prune_dm(epoch, model, train_hist, val_hist, run_dirs=None):
        val = float(val_hist[-1]) if val_hist else float("inf")
        optuna_trial.report(val, step=epoch)
        if optuna_trial.should_prune():
            raise optuna.TrialPruned()

    def _report_and_prune_eu(epoch, embed, train_hist, val_hist, run_dirs=None):
        val = float(val_hist[-1]) if val_hist else float("inf")
        optuna_trial.report(val, step=dm_epochs_max + epoch)
        if optuna_trial.should_prune():
            raise optuna.TrialPruned()

    vis_cfg = cfg.get("training_visualization") or {}
    vis_enabled = bool(vis_cfg.get("enabled"))
    dm_epoch_cb = _report_and_prune_dm
    eu_epoch_cb = _report_and_prune_eu
    if vis_enabled:
        from src.training_visualization import (
            make_distmap_epoch_hook,
            make_euclideanizer_epoch_hook,
            assemble_video,
        )
        dm_video_hook, _ = make_distmap_epoch_hook(
            coords, dm_cfg, run_dir_dm, device, utils, vis_cfg,
            split_seed=seed, training_split=training_split, total_epochs_display=int(dm_cfg["epochs"]),
        )
        def _dm_epoch_cb(epoch, model, train_hist, val_hist, run_dirs=None):
            dm_video_hook(epoch, model, train_hist, val_hist, run_dirs)
            _report_and_prune_dm(epoch, model, train_hist, val_hist, run_dirs)
        dm_epoch_cb = _dm_epoch_cb

    # --- DistMap: train ---
    _log(f"DistMap run 0 (seed {seed}): training from scratch to {dm_epochs} epochs...", since_start=time.time() - pipeline_start, style="info")
    phase_start_dm = time.time()
    try:
        dm_path, _ = train_distmap(
            dm_cfg, device, coords, run_dir_dm,
            split_seed=seed, training_split=training_split,
            epoch_callback=dm_epoch_cb,
            plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
            memory_efficient=dm_cfg["memory_efficient"],
            is_last_segment=True,
            display_root=base_output_dir,
            calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
            calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
            calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
            on_batch_size_resolved=lambda bs: _log(f"DistMap run 0: auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
        )
        _log(f"DistMap 0: training done in {(time.time() - phase_start_dm) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")
    finally:
        # Assemble video from whatever frames exist (full run, early-stopped, or pruned).
        if vis_enabled:
            fd_dm = _video_frames_dir(run_dir_dm)
            if os.path.isdir(fd_dm):
                ok, fail_reason = assemble_video(fd_dm, _video_mp4_path(run_dir_dm), vis_cfg["fps"])
                if ok and vis_cfg.get("delete_frames_after_video"):
                    shutil.rmtree(fd_dm)
                if not ok:
                    _log(f"DistMap 0: video assembly failed — {fail_reason}. Frames kept in {fd_dm}.", since_start=time.time() - pipeline_start, style="error")

    # --- DistMap: plotting ---
    if _plotting_phase_needed(
        do_plot, exp_stats, coords, train_stats, test_stats,
        do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance,
    ):
        # Resolve auto-calibrated batch_size for plotting (train_distmap writes to run_config; caller's dm_cfg is unchanged)
        dm_model_dir = os.path.join(run_dir_dm, "model")
        _dm_run_cfg = load_run_config(dm_model_dir)
        if _dm_run_cfg and isinstance((_dm_run_cfg.get("distmap") or {}).get("batch_size"), int):
            dm_cfg = {**dm_cfg, "batch_size": _dm_run_cfg["distmap"]["batch_size"]}
        elif dm_cfg.get("batch_size") is None:
            dm_cfg = {**dm_cfg, "batch_size": 256}
        _force_gpu_cleanup(device)
        model = ChromVAE_Conv(num_atoms=num_atoms, latent_space_dim=dm_cfg["latent_dim"]).to(device)
        model.load_state_dict(torch.load(dm_path, map_location=device))
        if do_recon_plot and coords is not None:
            p = _plot_path(run_dir_dm, "reconstruction")
            plot_distmap_reconstruction(
                model, device, coords, utils, p,
                training_split=training_split, split_seed=seed,
                batch_size=dm_cfg["batch_size"], num_to_plot=num_recon_samples, dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                display_root=base_output_dir,
            )
        _plot_mt, _plot_mc = plot_cfg["max_train"], plot_cfg["max_test"]
        if do_bond_rg_scaling and train_stats is not None and test_stats is not None and coords is not None:
            for subset_name, use_train, stats, cap in [
                ("test", False, test_stats, _plot_mc),
                ("train", True, train_stats, _plot_mt),
            ]:
                p = _plot_path(run_dir_dm, "recon_statistics", subset=subset_name)
                recon_dm = _get_recon_dm_distmap(
                    model, device, coords, dm_cfg, training_split, seed, utils,
                    use_train=use_train, max_structures=cap,
                )
                plot_recon_statistics(
                    recon_dm, stats, p,
                    label_recon="Recon", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    subset_label=subset_name,
                    display_root=base_output_dir,
                )
        gen_dm_bond = None
        if do_avg_gen and train_stats is not None and test_stats is not None:
            for var in sample_variances:
                p = _plot_path(run_dir_dm, "gen_variance", var=str(var))
                gen_dm = _get_gen_dm_distmap(model, device, gen_num_samples, dm_cfg["latent_dim"], var, gen_decode_batch_size)
                if gen_dm_bond is None:
                    gen_dm_bond = gen_dm
                plot_gen_analysis(
                    exp_stats, train_stats, test_stats, gen_dm, p,
                    sample_variance=var, label_gen="Gen", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    display_root=base_output_dir,
                )
        if do_bond_length_by_genomic_distance and train_stats is not None and test_stats is not None:
            if gen_dm_bond is None:
                gen_dm_bond = _get_gen_dm_distmap(model, device, gen_num_samples, dm_cfg["latent_dim"], sample_variances[0] if sample_variances else 1.0, gen_decode_batch_size)
            _run_bond_length_plot_suite_distmap(
                run_dir_dm, model, device, coords, dm_cfg, training_split, seed,
                train_stats, test_stats, gen_dm_bond,
                plot_cfg["max_train"], plot_cfg["max_test"],
                gen_num_samples, sample_variances, gen_decode_batch_size,
                plot_dpi, save_pdf, save_plot_data, base_output_dir,
            )
        del model
        torch.cuda.empty_cache()

    # --- Euclideanizer: train ---
    if vis_enabled:
        eu_video_hook, _ = make_euclideanizer_epoch_hook(
            coords, eu_cfg, dm_path, dm_cfg["latent_dim"], run_dir_eu, device, utils, vis_cfg,
            split_seed=seed, training_split=training_split, total_epochs_display=int(eu_cfg["epochs"]),
        )
        def _eu_epoch_cb(epoch, embed, train_hist, val_hist, run_dirs=None):
            eu_video_hook(epoch, embed, train_hist, val_hist, run_dirs)
            _report_and_prune_eu(epoch, embed, train_hist, val_hist, run_dirs)
        eu_epoch_cb = _eu_epoch_cb
    else:
        eu_epoch_cb = _report_and_prune_eu

    eu_epochs = int(eu_cfg["epochs"])
    _log(f"Euclideanizer run 0 (DistMap 0): training from scratch to {eu_epochs} epochs...", since_start=time.time() - pipeline_start, style="info")
    phase_start_eu = time.time()
    try:
        eu_path, _ = train_euclideanizer(
            eu_cfg, device, coords, dm_path, run_dir_eu,
            split_seed=seed, training_split=training_split,
            frozen_latent_dim=int(dm_cfg["latent_dim"]),
            epoch_callback=eu_epoch_cb,
            plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
            memory_efficient=eu_cfg["memory_efficient"],
            is_last_segment=True,
            display_root=base_output_dir,
            calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
            calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
            calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
            on_batch_size_resolved=lambda bs: _log(f"Euclideanizer run 0 (DistMap 0): auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
        )
        _log(f"Euclideanizer 0 (DistMap 0): training done in {(time.time() - phase_start_eu) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")
    finally:
        # Assemble video from whatever frames exist (full run, early-stopped, or pruned).
        if vis_enabled:
            fd_eu = _video_frames_dir(run_dir_eu)
            if os.path.isdir(fd_eu):
                ok, fail_reason = assemble_video(fd_eu, _video_mp4_path(run_dir_eu), vis_cfg["fps"])
                if ok and vis_cfg.get("delete_frames_after_video"):
                    shutil.rmtree(fd_eu)
                if not ok:
                    _log(f"Euclideanizer 0: video assembly failed — {fail_reason}. Frames kept in {fd_eu}.", since_start=time.time() - pipeline_start, style="error")

    # --- Euclideanizer: plotting and analysis ---
    if _plotting_phase_needed(
        do_plot, exp_stats, coords, train_stats, test_stats,
        do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance,
    ):
        # Resolve auto-calibrated batch_size for plotting (train_euclideanizer writes to run_config; caller's eu_cfg is unchanged)
        eu_model_dir = os.path.join(run_dir_eu, "model")
        _eu_run_cfg = load_run_config(eu_model_dir)
        if _eu_run_cfg and isinstance((_eu_run_cfg.get("euclideanizer") or {}).get("batch_size"), int):
            eu_cfg = {**eu_cfg, "batch_size": _eu_run_cfg["euclideanizer"]["batch_size"]}
        elif eu_cfg.get("batch_size") is None:
            eu_cfg = {**eu_cfg, "batch_size": 256}
        _force_gpu_cleanup(device)
        frozen_vae = load_frozen_vae(dm_path, num_atoms, dm_cfg["latent_dim"], device)
        embed = Euclideanizer(num_atoms=num_atoms).to(device)
        embed.load_state_dict(torch.load(eu_path, map_location=device))
        if do_recon_plot and coords is not None:
            p = _plot_path(run_dir_eu, "reconstruction")
            plot_euclideanizer_reconstruction(
                embed, frozen_vae, device, coords, utils, p,
                training_split=training_split, split_seed=seed,
                batch_size=eu_cfg["batch_size"], num_to_plot=num_recon_samples, dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                display_root=base_output_dir,
            )
        _plot_mt_eu, _plot_mc_eu = plot_cfg["max_train"], plot_cfg["max_test"]
        if do_bond_rg_scaling and train_stats is not None and test_stats is not None and coords is not None:
            for subset_name, use_train, stats, cap in [
                ("test", False, test_stats, _plot_mc_eu),
                ("train", True, train_stats, _plot_mt_eu),
            ]:
                p = _plot_path(run_dir_eu, "recon_statistics", subset=subset_name)
                recon_dm = _get_recon_dm_euclideanizer(
                    embed, frozen_vae, device, coords, eu_cfg, training_split, seed, utils,
                    use_train=use_train, max_structures=cap,
                )
                plot_recon_statistics(
                    recon_dm, stats, p,
                    label_recon="Recon", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    subset_label=subset_name,
                    display_root=base_output_dir,
                )
        gen_dm_bond = None
        if do_avg_gen and train_stats is not None and test_stats is not None:
            for var in sample_variances:
                p = _plot_path(run_dir_eu, "gen_variance", var=str(var))
                gen_dm = _get_gen_dm_euclideanizer(
                    embed, frozen_vae, device, gen_num_samples, dm_cfg["latent_dim"], var, utils, gen_decode_batch_size
                )
                if gen_dm_bond is None:
                    gen_dm_bond = gen_dm
                plot_gen_analysis(
                    exp_stats, train_stats, test_stats, gen_dm, p,
                    sample_variance=var, label_gen="Gen", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    display_root=base_output_dir,
                )
        if do_bond_length_by_genomic_distance and train_stats is not None and test_stats is not None:
            if gen_dm_bond is None:
                gen_dm_bond = _get_gen_dm_euclideanizer(
                    embed, frozen_vae, device, gen_num_samples, dm_cfg["latent_dim"], sample_variances[0] if sample_variances else 1.0, utils, gen_decode_batch_size
                )
            _run_bond_length_plot_suite_euclideanizer(
                run_dir_eu, embed, frozen_vae, device, coords, eu_cfg, dm_cfg["latent_dim"],
                training_split, seed, train_stats, test_stats, gen_dm_bond,
                plot_cfg["max_train"], plot_cfg["max_test"],
                gen_num_samples, sample_variances, gen_decode_batch_size,
                plot_dpi, save_pdf, save_plot_data, base_output_dir,
            )
        # Analysis: latent + registered metrics (same structure as _run_one_distmap_group)
        any_analysis = any(
            analysis_cfg[spec.gen_key]["enabled"] or analysis_cfg[spec.recon_key]["enabled"]
            for spec in ANALYSIS_METRICS
        ) or analysis_cfg["generative_capacity_rmsd"]["enabled"] or analysis_cfg["generative_capacity_q"]["enabled"]
        seed_test_to_train_cache = {}
        if any_analysis and coords is not None:
            _force_gpu_cleanup(device)
            latent_cfg = analysis_cfg["latent"]
            if latent_cfg["enabled"] and coords is not None:
                latent_dir = os.path.join(run_dir_eu, "analysis", "latent")
                latent_fig = os.path.join(latent_dir, "latent_distribution.png")
                latent_corr_fig = os.path.join(latent_dir, "latent_correlation.png")
                latent_data_dir = os.path.join(latent_dir, "data")
                latent_stats_npz = os.path.join(latent_data_dir, "latent_stats.npz")
                train_mu_lat, test_mu_lat = _get_latent_vectors_euclideanizer(
                    frozen_vae, device, coords, training_split, seed, utils,
                    max_train=analysis_cfg["latent_max_train"],
                    max_test=analysis_cfg["latent_max_test"],
                )
                os.makedirs(latent_dir, exist_ok=True)
                plot_latent_distribution(
                    train_mu_lat, test_mu_lat, latent_fig,
                    plot_dpi=plot_dpi, display_root=base_output_dir,
                    save_pdf_copy=latent_cfg["save_pdf_copy"],
                )
                plot_latent_correlation(
                    train_mu_lat, test_mu_lat, latent_corr_fig,
                    plot_dpi=plot_dpi, display_root=base_output_dir,
                    save_pdf_copy=latent_cfg["save_pdf_copy"],
                )
                if latent_cfg["save_data"] or scoring_enabled:
                    save_latent_stats_npz(
                        train_mu_lat, test_mu_lat, latent_stats_npz,
                        display_root=base_output_dir,
                    )
            _force_gpu_cleanup(device)
            for spec in ANALYSIS_METRICS:
                _force_gpu_cleanup(device)
                do_gen = analysis_cfg[spec.gen_key]["enabled"]
                do_recon = analysis_cfg[spec.recon_key]["enabled"]
                if not (do_gen or do_recon):
                    continue
                gen_cfg = analysis_cfg[spec.gen_key]
                recon_cfg = analysis_cfg[spec.recon_key]
                _variance_list = gen_cfg["sample_variance"]
                if _variance_list is None:
                    _variance_list = []
                if not isinstance(_variance_list, list):
                    _variance_list = [_variance_list]
                _num_samples_list = gen_cfg["num_samples"]
                if _num_samples_list is None:
                    _num_samples_list = []
                if not isinstance(_num_samples_list, list):
                    _num_samples_list = [_num_samples_list]
                _max_recon_train_list = recon_cfg["max_recon_train"]
                if _max_recon_train_list is None:
                    _max_recon_train_list = [None]
                if not isinstance(_max_recon_train_list, list):
                    _max_recon_train_list = [_max_recon_train_list]
                _max_recon_test_list = recon_cfg["max_recon_test"]
                if _max_recon_test_list is None:
                    _max_recon_test_list = [None]
                if not isinstance(_max_recon_test_list, list):
                    _max_recon_test_list = [_max_recon_test_list]
                _ref_mt = analysis_cfg[f"{spec.id}_max_train"]
                _ref_mc = analysis_cfg[f"{spec.id}_max_test"]

                def _get_or_compute_cached(mt, mc):
                    cache_key = spec.id
                    if seed_test_to_train_cache.get(cache_key) is None:
                        seed_test_to_train_cache[cache_key] = {}
                    key = (mt, mc)
                    if key not in seed_test_to_train_cache[cache_key]:
                        _cache_path = os.path.join(seed_dir, EXP_STATS_CACHE_DIR, spec.cache_filename(analysis_cfg, mt, mc))
                        _run_label = os.path.basename(seed_dir)
                        _had_disk = os.path.isfile(_cache_path)
                        if not _had_disk:
                            _log(
                                f"Analysis seed cache [{spec.id}] {_run_label}: computing {os.path.basename(_cache_path)} (test→train / feats).",
                                since_start=time.time() - pipeline_start,
                                style="info",
                            )
                        seed_test_to_train_cache[cache_key][key] = spec.get_or_compute_test_to_train(
                            _cache_path, coords_np, coords, training_split, seed, base_output_dir,
                            **spec.kwargs_for_cache(analysis_cfg, mt, mc),
                        )
                        if not _had_disk:
                            _log(
                                f"Analysis seed cache [{spec.id}] {_run_label}: saved {os.path.basename(_cache_path)}.",
                                since_start=time.time() - pipeline_start,
                                style="success",
                            )
                    return seed_test_to_train_cache[cache_key][key]

                if do_gen:
                    _mt_gen = _ref_mt if spec.requires_reference_bounds else None
                    _mc_gen = _ref_mc if spec.requires_reference_bounds else None
                    if spec.requires_reference_bounds and (_mt_gen is None or _mc_gen is None):
                        continue
                    _tt, _train_c, _test_c = _get_or_compute_cached(_ref_mt, _ref_mc)
                plot_cfg_gen = spec.build_gen_plot_cfg(analysis_cfg)
                plot_cfg_gen = {
                    **plot_cfg_gen,
                    "plot_dpi": plot_dpi,
                    "save_data": plot_cfg_gen.get("save_data") or cfg.get("scoring", {}).get("enabled", False),
                }
                pre_kw = spec.precomputed_kwargs(_tt, _train_c, _test_c)
                extra_kw = spec.gen_extra_kwargs(analysis_cfg)
                for var in _variance_list:
                    variance_suffix = f"_var{var}"
                    any_missing = False
                    for n in _num_samples_list:
                        run_name = (str(n) if len(_num_samples_list) > 1 else "default") + variance_suffix
                        fig_path = _analysis_path(run_dir_eu, spec.subdir, f"gen/{run_name}/{spec.figure_filename}")
                        if not os.path.isfile(fig_path):
                            any_missing = True
                            break
                    if any_missing:
                        if len(_num_samples_list) > 1:
                            spec.run_gen_analysis_multi(
                                coords_np, coords, training_split, seed,
                                frozen_vae, embed, dm_cfg["latent_dim"], device, run_dir_eu,
                                plot_cfg_gen,
                                num_samples_list=_num_samples_list,
                                sample_variance=var,
                                variance_suffix=variance_suffix,
                                display_root=base_output_dir,
                                **pre_kw,
                                **extra_kw,
                            )
                        else:
                            n = _num_samples_list[0]
                            run_name_single = (str(n) if len(_num_samples_list) > 1 else "default") + variance_suffix
                            output_suffix = "_" + run_name_single
                            spec.run_gen_analysis(
                                coords_np, coords, training_split, seed,
                                frozen_vae, embed, dm_cfg["latent_dim"], device, run_dir_eu,
                                plot_cfg_gen,
                                num_samples=n, sample_variance=var, output_suffix=output_suffix,
                                display_root=base_output_dir,
                                **pre_kw,
                                **extra_kw,
                            )

                if do_recon and _max_recon_train_list and _max_recon_test_list:
                    n_recon = len(_max_recon_train_list) * len(_max_recon_test_list)
                    plot_cfg_recon = spec.build_recon_plot_cfg(analysis_cfg)
                    plot_cfg_recon = {
                        **plot_cfg_recon,
                        "plot_dpi": plot_dpi,
                        "save_data": plot_cfg_recon.get("save_data") or cfg.get("scoring", {}).get("enabled", False),
                    }
                    recon_extra = spec.recon_extra_kwargs(analysis_cfg)
                    for max_recon_train in _max_recon_train_list:
                        for max_recon_test in _max_recon_test_list:
                            _tt, _train_c, _test_c = _get_or_compute_cached(_ref_mt, _ref_mc)
                            if n_recon == 1:
                                recon_subdir = ""
                                recon_fig = _analysis_path(run_dir_eu, spec.subdir, f"recon/{spec.figure_filename}")
                            else:
                                recon_subdir = f"train{max_recon_train}_test{max_recon_test}"
                                recon_fig = _analysis_path(run_dir_eu, spec.subdir, f"recon/{recon_subdir}/{spec.figure_filename}")
                            train_recon_coords = _get_recon_coords_euclideanizer(
                                embed, frozen_vae, device, coords, training_split, seed, utils,
                                use_train=True, max_n=max_recon_train,
                            )
                            test_recon_coords = _get_recon_coords_euclideanizer(
                                embed, frozen_vae, device, coords, training_split, seed, utils,
                                use_train=False, max_n=max_recon_test,
                            )
                            spec.run_recon_analysis(
                                _tt, _train_c, _test_c, train_recon_coords, test_recon_coords,
                                run_dir_eu, plot_cfg_recon,
                                display_root=base_output_dir, recon_subdir=recon_subdir,
                                **recon_extra,
                            )
            _run_generative_capacity_blocks_for_run(
                run_dir_eu=run_dir_eu,
                analysis_cfg=analysis_cfg,
                seed=seed,
                latent_dim=dm_cfg["latent_dim"],
                device=device,
                frozen_vae=frozen_vae,
                embed=embed,
                resume=False,
                pipeline_start=pipeline_start,
                display_root=base_output_dir,
            )
        del embed, frozen_vae
        torch.cuda.empty_cache()
        if cfg["scoring"]["enabled"]:
            _run_scoring_for_run(run_dir_eu, seed_dir, cfg, base_output_dir, pipeline_start)
            _post_scoring_npz_cleanup(
                run_dir_eu,
                cfg,
                defer_sufficiency_inputs=bool(cfg["meta_analysis"]["sufficiency"]["enabled"]),
            )
    else:
        if cfg["scoring"]["enabled"]:
            _run_scoring_for_run(run_dir_eu, seed_dir, cfg, base_output_dir, pipeline_start)
            _post_scoring_npz_cleanup(
                run_dir_eu,
                cfg,
                defer_sufficiency_inputs=bool(cfg["meta_analysis"]["sufficiency"]["enabled"]),
            )

    _log("Pipeline complete.", since_start=time.time() - pipeline_start, style="success")
    scores_path = os.path.join(run_dir_eu, "scoring", "scores.json")
    if not os.path.isfile(scores_path):
        raise RuntimeError("HPO trial: scoring did not produce scores.json")
    with open(scores_path) as f:
        scores_data = json.load(f)
    overall = scores_data.get("overall_score")
    if overall is None:
        raise RuntimeError("HPO trial: overall_score missing in scores.json")
    try:
        return float(overall)
    except (TypeError, ValueError):
        raise RuntimeError("HPO trial: overall_score is not a valid float")


def _force_gpu_cleanup(device: torch.device) -> None:
    """Release unused GPU memory so the allocator and system see accurate free memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _apply_max_data_subset(coords_np: np.ndarray, max_data: int | None, seed: int) -> np.ndarray:
    if max_data is None or max_data >= coords_np.shape[0]:
        return coords_np
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(coords_np.shape[0], size=max_data, replace=False)
    return coords_np[idx]


def _max_data_indices(total_n: int, max_data: int | None, seed: int) -> np.ndarray:
    if max_data is None or max_data >= total_n:
        return np.arange(total_n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.choice(total_n, size=max_data, replace=False).astype(np.int64)


def _derive_stats_from_global_exp(
    global_exp_stats: dict,
    indices: np.ndarray,
    *,
    max_sep: int,
    avg_map_sample: int,
) -> dict:
    """Derive exp stats for a subset by slicing global exp_distmaps."""
    sub_dm = np.asarray(global_exp_stats["exp_distmaps"])[np.asarray(indices)]
    s, sc = distmap_scaling(sub_dm, max_sep)
    n_sample = min(avg_map_sample, len(sub_dm))
    return {
        "exp_distmaps": sub_dm,
        "exp_bonds": distmap_bond_lengths(sub_dm),
        "exp_rg": distmap_rg(sub_dm),
        "genomic_distances": s,
        "exp_scaling": sc,
        "avg_exp_map": np.mean(sub_dm[:n_sample], axis=0),
    }


def _run_one_distmap_group(
    seed: int,
    max_data: int | None,
    gidx: int,
    device,
    cfg: dict,
    base_output_dir: str,
    dm_groups: list,
    eu_groups: list,
    dm_configs: list,
    eu_configs: list,
    coords,
    coords_np,
    num_atoms: int | None,
    num_structures: int | None,
    exp_stats,
    data_path: str | None,
    need_train: bool,
    pipeline_start: float,
    training_split: float,
    training_splits: list,
    do_plot: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
    do_rmsd: bool,
    resume: bool,
    sample_variances: list,
    gen_num_samples: int,
    gen_decode_batch_size_holder: list,
    need_plot_or_rmsd: bool,
    save_structures_gro_plot: bool,
    analysis_save_data: bool,
    analysis_save_structures_gro: bool,
    plot_dpi: int,
    save_pdf: bool,
    save_plot_data: bool,
    num_recon_samples: int,
    analysis_cfg: dict,
    variance_list: list,
    num_samples_list: list,
    max_recon_train_list: list,
    max_recon_test_list: list,
    vis_enabled: bool,
    vis_cfg: dict,
    plot_cfg: dict,
    train_stats,
    test_stats,
    seed_test_to_train_holder: list,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_max_train: int | None = None,
    q_max_test: int | None = None,
    q_num_samples_list: list | None = None,
    q_variance_list: list | None = None,
    q_delta: float = 0.7071067811865475,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    q_recon_delta: float = 0.7071067811865475,
    make_distmap_epoch_hook=None,
    make_euclideanizer_epoch_hook=None,
    assemble_video_fn=None,
) -> None:
    """Run one (seed, DistMap group): that group's segments, plotting, and all Euclideanizer runs for that DistMap."""
    gen_decode_batch_size = gen_decode_batch_size_holder[0] if gen_decode_batch_size_holder else None
    output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
    split_seed = seed
    group = dm_groups[gidx]
    base_config = group["base_config"]
    checkpoints = group["checkpoints"]
    checkpoint_dirs = [os.path.join(output_dir, "distmap", str(ri)) for ri, _ in checkpoints]
    dm_max_epoch = max(ev for _, ev in checkpoints)
    prev_dm_path = None
    prev_dm_ev = None
    dm_stopped_early = False

    for seg_idx, (ri, ev) in enumerate(checkpoints):
        run_dir_dm = checkpoint_dirs[seg_idx]
        dm_path = _dm_path(run_dir_dm)
        dm_cfg = {**base_config, "epochs": ev}
    
        phase_start = time.time()
        dm_multi = len(checkpoints) > 1
        dm_save_final = base_config["save_final_models_per_stretch"]
        dm_last_segment = seg_idx == len(checkpoints) - 1
        dm_act = _distmap_training_action(
            run_dir_dm, ev, dm_cfg,
            prev_dm_path, prev_dm_ev,
            checkpoint_dirs[seg_idx - 1] if seg_idx > 0 else None,
            resume, dm_multi, dm_last_segment, dm_save_final,
        )
        if dm_act["action"] == "skip":
            _log(f"DistMap run {ri} (seed {seed}, epochs={ev}): resumed (skip training).", since_start=time.time() - pipeline_start, style="skip")
            dm_stopped_early = False
            if os.path.isfile(dm_path):
                prev_dm_path = dm_path
                prev_dm_ev = ev
        else:
            if vis_enabled:
                fd_dm = _video_frames_dir(run_dir_dm)
                if os.path.isdir(fd_dm):
                    shutil.rmtree(fd_dm)
            if dm_act["action"] == "from_scratch":
                _log(f"DistMap run {ri} (seed {seed}): training from scratch to {ev} epochs...", since_start=time.time() - pipeline_start, style="info")
                epoch_cb = None
                if vis_enabled and make_distmap_epoch_hook is not None:
                    epoch_cb, _ = make_distmap_epoch_hook(
                        coords, dm_cfg, run_dir_dm, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, total_epochs_display=dm_max_epoch
                    )
                else:
                    epoch_cb = None
                dm_path, dm_stopped_early = train_distmap(
                    dm_cfg, device, coords, run_dir_dm,
                    split_seed=split_seed, training_split=training_split,
                    epoch_callback=epoch_cb,
                    plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    memory_efficient=dm_cfg["memory_efficient"],
                    is_last_segment=dm_last_segment,
                    display_root=base_output_dir,
                    calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
                    calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
                    calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
                    on_batch_size_resolved=lambda bs, _ri=ri: _log(f"DistMap run {_ri}: auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
                )
            elif dm_act["action"] == "resume_from_best":
                _log(f"DistMap run {ri} (seed {seed}): resuming from best (epoch {dm_act['best_epoch']}), training {dm_act['additional_epochs']} more → {ev} total...", since_start=time.time() - pipeline_start, style="info")
                epoch_cb = None
                if vis_enabled and make_distmap_epoch_hook is not None:
                    epoch_cb, _ = make_distmap_epoch_hook(
                        coords, dm_cfg, run_dir_dm, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=dm_act["best_epoch"], total_epochs_display=dm_max_epoch
                    )
                dm_path, dm_stopped_early = train_distmap(
                    dm_cfg, device, coords, run_dir_dm,
                    split_seed=split_seed, training_split=training_split,
                    epoch_callback=epoch_cb,
                    plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    resume_from_path=dm_act["resume_from_path"], additional_epochs=dm_act["additional_epochs"],
                    prev_run_dir=dm_act["prev_run_dir"],
                    save_final_models_per_stretch=dm_save_final,
                    is_last_segment=dm_last_segment,
                    memory_efficient=dm_cfg["memory_efficient"],
                    display_root=base_output_dir,
                    calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
                    calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
                    calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
                    on_batch_size_resolved=lambda bs, _ri=ri: _log(f"DistMap run {_ri}: auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
                )
            else:
                _log(f"DistMap run {ri} (seed {seed}): resuming from run (epochs={prev_dm_ev}), training {dm_act['additional_epochs']} more → {ev} total...", since_start=time.time() - pipeline_start, style="info")
                epoch_cb = None
                if vis_enabled and make_distmap_epoch_hook is not None:
                    epoch_cb, _ = make_distmap_epoch_hook(
                        coords, dm_cfg, run_dir_dm, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=prev_dm_ev, total_epochs_display=dm_max_epoch
                    )
                dm_path, dm_stopped_early = train_distmap(
                    dm_cfg, device, coords, run_dir_dm,
                    split_seed=split_seed, training_split=training_split,
                    epoch_callback=epoch_cb,
                    plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    resume_from_path=dm_act["resume_from_path"], additional_epochs=dm_act["additional_epochs"],
                    prev_run_dir=dm_act["prev_run_dir"],
                    save_final_models_per_stretch=dm_save_final,
                    is_last_segment=dm_last_segment,
                    memory_efficient=dm_cfg["memory_efficient"],
                    display_root=base_output_dir,
                    calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
                    calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
                    calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
                    on_batch_size_resolved=lambda bs, _ri=ri: _log(f"DistMap run {_ri}: auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
                )
            prev_dm_path = dm_path
            prev_dm_ev = ev
            _log(f"DistMap {ri}: training done in {(time.time() - phase_start) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")

        if vis_enabled:
            fd_dm = _video_frames_dir(run_dir_dm)
            if os.path.isdir(fd_dm):
                _log(f"Assembling video for DistMap {ri}...", since_start=time.time() - pipeline_start, style="info")
                ok, fail_reason = assemble_video_fn(fd_dm, _video_mp4_path(run_dir_dm), vis_cfg["fps"])
                if ok:
                    if vis_cfg["delete_frames_after_video"]:
                        shutil.rmtree(fd_dm)
                    _log(f"DistMap {ri}: video saved.", since_start=time.time() - pipeline_start, style="success")
                else:
                    _log(f"DistMap {ri}: video assembly failed — {fail_reason}. Frames kept in {fd_dm}.", since_start=time.time() - pipeline_start, style="error")
            else:
                _log(f"DistMap {ri}: no frames dir (video skipped).", since_start=time.time() - pipeline_start, style="skip")
    
        if _plotting_phase_needed(
            do_plot, exp_stats, coords, train_stats, test_stats,
            do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance,
        ):
            # Resolve auto-calibrated batch_size for plotting (train_distmap writes to run_config; caller's dm_cfg is unchanged)
            dm_model_dir = os.path.join(run_dir_dm, "model")
            _dm_run_cfg = load_run_config(dm_model_dir)
            if _dm_run_cfg and isinstance((_dm_run_cfg.get("distmap") or {}).get("batch_size"), int):
                dm_cfg = {**dm_cfg, "batch_size": _dm_run_cfg["distmap"]["batch_size"]}
            elif dm_cfg.get("batch_size") is None:
                dm_cfg = {**dm_cfg, "batch_size": 256}
            if _distmap_plotting_all_present(
                run_dir_dm, resume, do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance, sample_variances
            ):
                _log(f"DistMap {ri}: [skip] plotting (all present)", since_start=time.time() - pipeline_start, style="skip")
            else:
                _force_gpu_cleanup(device)
                phase_start = time.time()
                _log(f"DistMap {ri}: plotting (recon, stats, gen)...", since_start=time.time() - pipeline_start, style="info")
                model = ChromVAE_Conv(num_atoms=num_atoms, latent_space_dim=dm_cfg["latent_dim"]).to(device)
                model.load_state_dict(torch.load(dm_path, map_location=device))
                if do_recon_plot and coords is not None:
                    p = _plot_path(run_dir_dm, "reconstruction")
                    if not (resume and os.path.isfile(p)):
                        plot_distmap_reconstruction(
                            model, device, coords, utils, p,
                            training_split=training_split, split_seed=split_seed,
                            batch_size=dm_cfg["batch_size"], num_to_plot=num_recon_samples, dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                            display_root=base_output_dir,
                        )
                    elif resume:
                        _log("  [skip] reconstruction", since_start=time.time() - pipeline_start, style="skip")
                _plot_mt_dm, _plot_mc_dm = plot_cfg["max_train"], plot_cfg["max_test"]
                if do_bond_rg_scaling and train_stats is not None and test_stats is not None and coords is not None:
                    for subset_name, use_train, stats, cap in [
                        ("test", False, test_stats, _plot_mc_dm),
                        ("train", True, train_stats, _plot_mt_dm),
                    ]:
                        p = _plot_path(run_dir_dm, "recon_statistics", subset=subset_name)
                        if not (resume and os.path.isfile(p)):
                            recon_dm = _get_recon_dm_distmap(
                                model, device, coords, dm_cfg, training_split, split_seed, utils,
                                use_train=use_train, max_structures=cap,
                            )
                            plot_recon_statistics(
                                recon_dm, stats, p,
                                label_recon="Recon", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                subset_label=subset_name,
                                display_root=base_output_dir,
                            )
                        elif resume:
                            _log(f"  [skip] recon_statistics_{subset_name}", since_start=time.time() - pipeline_start, style="skip")
                gen_dm_bond = None
                if do_avg_gen and train_stats is not None and test_stats is not None:
                    for var in sample_variances:
                        p = _plot_path(run_dir_dm, "gen_variance", var=str(var))
                        if not (resume and os.path.isfile(p)):
                            gen_dm = _get_gen_dm_distmap(model, device, gen_num_samples, dm_cfg["latent_dim"], var, gen_decode_batch_size)
                            if gen_dm_bond is None:
                                gen_dm_bond = gen_dm
                            plot_gen_analysis(
                                exp_stats, train_stats, test_stats, gen_dm, p,
                                sample_variance=var, label_gen="Gen", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                display_root=base_output_dir,
                            )
                        elif resume:
                            _log(f"  [skip] gen_variance_{var}", since_start=time.time() - pipeline_start, style="skip")
                if do_bond_length_by_genomic_distance and train_stats is not None and test_stats is not None:
                    _bond_all_dm = all(
                        os.path.isfile(_plot_path(run_dir_dm, bt))
                        for bt in (
                            "bond_length_by_genomic_distance_gen",
                            "bond_length_by_genomic_distance_train",
                            "bond_length_by_genomic_distance_test",
                        )
                    )
                    if gen_dm_bond is None:
                        gen_dm_bond = _get_gen_dm_distmap(model, device, gen_num_samples, dm_cfg["latent_dim"], sample_variances[0], gen_decode_batch_size)
                    if not (resume and _bond_all_dm):
                        _run_bond_length_plot_suite_distmap(
                            run_dir_dm, model, device, coords, dm_cfg, training_split, split_seed,
                            train_stats, test_stats, gen_dm_bond,
                            plot_cfg["max_train"], plot_cfg["max_test"],
                            gen_num_samples, sample_variances, gen_decode_batch_size,
                            plot_dpi, save_pdf, save_plot_data, base_output_dir,
                        )
                    elif resume:
                        _log("  [skip] bond_length_by_genomic_distance (gen/train/test)", since_start=time.time() - pipeline_start, style="skip")
                del model
                torch.cuda.empty_cache()
                _log(f"DistMap {ri}: plotting done in {(time.time() - phase_start) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")
    
        _log(f"DistMap {ri}: starting Euclideanizer runs", since_start=time.time() - pipeline_start, style="info")
        for egidx, eu_group in enumerate(eu_groups):
            eu_base = eu_group["base_config"]
            eu_checkpoints = eu_group["checkpoints"]
            eu_checkpoint_dirs = [os.path.join(output_dir, "distmap", str(ri), "euclideanizer", str(euri)) for euri, _ in eu_checkpoints]
            eu_max_epoch = max(ev for _, ev in eu_checkpoints)
            prev_eu_path = None
            prev_eu_ev = None
            eu_stopped_early = False
            for eu_seg_idx, (euri, eu_ev) in enumerate(eu_checkpoints):
                eu_run_dir = eu_checkpoint_dirs[eu_seg_idx]
                eu_path_seg = _eu_path(eu_run_dir)
                eu_cfg_seg = {**eu_base, "epochs": eu_ev}
                eu_multi = len(eu_checkpoints) > 1
                eu_save_final = eu_base["save_final_models_per_stretch"]
                eu_last_segment = eu_seg_idx == len(eu_checkpoints) - 1
                eu_act = _euclideanizer_training_action(
                    eu_run_dir, eu_ev, eu_cfg_seg,
                    prev_eu_path, prev_eu_ev,
                    eu_checkpoint_dirs[eu_seg_idx - 1] if eu_seg_idx > 0 else None,
                    resume, eu_multi, eu_last_segment, eu_save_final,
                )
                if eu_act["action"] == "skip":
                    _log(f"Euclideanizer run {euri} (DistMap {ri}, epochs={eu_ev}): resumed (skip training).", since_start=time.time() - pipeline_start, style="skip")
                    eu_stopped_early = False
                    if os.path.isfile(eu_path_seg):
                        prev_eu_path = eu_path_seg
                        prev_eu_ev = eu_ev
                else:
                    if vis_enabled:
                        fd_eu_pre = _video_frames_dir(eu_run_dir)
                        if os.path.isdir(fd_eu_pre):
                            shutil.rmtree(fd_eu_pre)
                    if eu_act["action"] == "from_scratch":
                        _log(f"Euclideanizer run {euri} (DistMap {ri}): training from scratch to {eu_ev} epochs...", since_start=time.time() - pipeline_start, style="info")
                        epoch_cb = None
                        if vis_enabled:
                            epoch_cb, _ = make_euclideanizer_epoch_hook(
                                coords, eu_cfg_seg, dm_path, dm_cfg["latent_dim"], eu_run_dir, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, total_epochs_display=eu_max_epoch
                            )
                        _, eu_stopped_early = train_euclideanizer(
                            eu_cfg_seg, device, coords, dm_path, eu_run_dir,
                            split_seed=split_seed, training_split=training_split,
                            frozen_latent_dim=dm_cfg["latent_dim"],
                            epoch_callback=epoch_cb,
                            plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                            memory_efficient=eu_cfg_seg["memory_efficient"],
                            is_last_segment=eu_last_segment,
                            display_root=base_output_dir,
                            calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
                            calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
                            calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
                            on_batch_size_resolved=lambda bs, _euri=euri, _ri=ri: _log(f"Euclideanizer run {_euri} (DistMap {_ri}): auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
                        )
                    elif eu_act["action"] == "resume_from_best":
                        _log(f"Euclideanizer run {euri} (DistMap {ri}): resuming from best (epoch {eu_act['best_epoch']}), training {eu_act['additional_epochs']} more → {eu_ev} total...", since_start=time.time() - pipeline_start, style="info")
                        epoch_cb = None
                        if vis_enabled:
                            epoch_cb, _ = make_euclideanizer_epoch_hook(
                                coords, eu_cfg_seg, dm_path, dm_cfg["latent_dim"], eu_run_dir, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=eu_act["best_epoch"], total_epochs_display=eu_max_epoch
                            )
                        _, eu_stopped_early = train_euclideanizer(
                            eu_cfg_seg, device, coords, dm_path, eu_run_dir,
                            split_seed=split_seed, training_split=training_split,
                            frozen_latent_dim=dm_cfg["latent_dim"],
                            epoch_callback=epoch_cb,
                            plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                            resume_from_path=eu_act["resume_from_path"], additional_epochs=eu_act["additional_epochs"],
                            prev_run_dir=eu_act["prev_run_dir"],
                            save_final_models_per_stretch=eu_save_final,
                            is_last_segment=eu_last_segment,
                            memory_efficient=eu_cfg_seg["memory_efficient"],
                            display_root=base_output_dir,
                            calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
                            calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
                            calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
                            on_batch_size_resolved=lambda bs, _euri=euri, _ri=ri: _log(f"Euclideanizer run {_euri} (DistMap {_ri}): auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
                        )
                    else:
                        _log(f"Euclideanizer run {euri} (DistMap {ri}): resuming from {prev_eu_ev} epochs, training {eu_act['additional_epochs']} more → {eu_ev} total...", since_start=time.time() - pipeline_start, style="info")
                        epoch_cb = None
                        if vis_enabled:
                            epoch_cb, _ = make_euclideanizer_epoch_hook(
                                coords, eu_cfg_seg, dm_path, dm_cfg["latent_dim"], eu_run_dir, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=prev_eu_ev, total_epochs_display=eu_max_epoch
                            )
                        _, eu_stopped_early = train_euclideanizer(
                            eu_cfg_seg, device, coords, dm_path, eu_run_dir,
                            split_seed=split_seed, training_split=training_split,
                            frozen_latent_dim=dm_cfg["latent_dim"],
                            epoch_callback=epoch_cb,
                            plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                            resume_from_path=eu_act["resume_from_path"], additional_epochs=eu_act["additional_epochs"],
                            prev_run_dir=eu_act["prev_run_dir"],
                            save_final_models_per_stretch=eu_save_final,
                            is_last_segment=eu_last_segment,
                            memory_efficient=eu_cfg_seg["memory_efficient"],
                            display_root=base_output_dir,
                            calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
                            calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
                            calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
                            on_batch_size_resolved=lambda bs, _euri=euri, _ri=ri: _log(f"Euclideanizer run {_euri} (DistMap {_ri}): auto-calibrated batch_size={bs}", since_start=time.time() - pipeline_start, style="info"),
                        )
                    prev_eu_path = eu_path_seg
                    prev_eu_ev = eu_ev

                if vis_enabled:
                    fd_eu = _video_frames_dir(eu_run_dir)
                    if os.path.isdir(fd_eu):
                        _log(f"Assembling video for Euclideanizer {euri} (DistMap {ri}, epochs={eu_ev})...", since_start=time.time() - pipeline_start, style="info")
                        ok, fail_reason = assemble_video_fn(fd_eu, _video_mp4_path(eu_run_dir), vis_cfg["fps"])
                        if ok:
                            if vis_cfg["delete_frames_after_video"]:
                                shutil.rmtree(fd_eu)
                            _log(f"Euclideanizer {euri} (DistMap {ri}): video saved.", since_start=time.time() - pipeline_start, style="success")
                        else:
                            _log(f"Euclideanizer {euri} (DistMap {ri}): video assembly failed — {fail_reason}.", since_start=time.time() - pipeline_start, style="error")
    
                if need_plot_or_rmsd:
                    run_dir_eu = eu_run_dir
                    eu_cfg = eu_configs[euri]
                    eu_path = eu_path_seg
                    all_plots = _euclideanizer_plotting_all_present(
                        run_dir_eu, resume, do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance, sample_variances
                    )
                    all_analysis = _euclideanizer_analysis_all_present(
                        run_dir_eu, resume, analysis_cfg=analysis_cfg
                    )
                    if resume and all_plots and all_analysis:
                        _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): [skip] plotting and analysis (all present)", since_start=time.time() - pipeline_start, style="skip")
                    else:
                        _force_gpu_cleanup(device)
                        phase_start_eu = time.time()
                        frozen_vae = load_frozen_vae(dm_path, num_atoms, dm_cfg["latent_dim"], device)
                        embed = Euclideanizer(num_atoms=num_atoms).to(device)
                        embed.load_state_dict(torch.load(eu_path, map_location=device))
                        eu_model_dir = os.path.join(run_dir_eu, "model")
                        # Resolve auto-calibrated batch_size for plotting (train_euclideanizer writes to run_config; eu_cfg from config list is unchanged)
                        _eu_run_cfg = load_run_config(eu_model_dir)
                        if _eu_run_cfg and isinstance((_eu_run_cfg.get("euclideanizer") or {}).get("batch_size"), int):
                            eu_cfg = {**eu_cfg, "batch_size": _eu_run_cfg["euclideanizer"]["batch_size"]}
                        elif eu_cfg.get("batch_size") is None:
                            eu_cfg = {**eu_cfg, "batch_size": 256}
                        if _plotting_phase_needed(
                            do_plot, exp_stats, coords, train_stats, test_stats,
                            do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance,
                        ):
                            plot_phase_start = time.time()
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): plotting (diagnostics)...", since_start=time.time() - pipeline_start, style="info")
                            if do_recon_plot and coords is not None:
                                p = _plot_path(run_dir_eu, "reconstruction")
                                if not (resume and os.path.isfile(p)):
                                    plot_euclideanizer_reconstruction(
                                        embed, frozen_vae, device, coords, utils, p,
                                        training_split=training_split, split_seed=split_seed,
                                        batch_size=eu_cfg["batch_size"], num_to_plot=num_recon_samples, dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                        display_root=base_output_dir,
                                    )
                                elif resume:
                                    _log("  [skip] reconstruction", since_start=time.time() - pipeline_start, style="skip")
                            _plot_mt_eu2, _plot_mc_eu2 = plot_cfg["max_train"], plot_cfg["max_test"]
                            if do_bond_rg_scaling and train_stats is not None and test_stats is not None and coords is not None:
                                for subset_name, use_train, stats, cap in [
                                    ("test", False, test_stats, _plot_mc_eu2),
                                    ("train", True, train_stats, _plot_mt_eu2),
                                ]:
                                    p = _plot_path(run_dir_eu, "recon_statistics", subset=subset_name)
                                    if not (resume and os.path.isfile(p)):
                                        recon_dm = _get_recon_dm_euclideanizer(
                                            embed, frozen_vae, device, coords, eu_cfg, training_split, split_seed, utils,
                                            use_train=use_train, max_structures=cap,
                                        )
                                        plot_recon_statistics(
                                            recon_dm, stats, p,
                                            label_recon="Recon", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                            subset_label=subset_name,
                                            display_root=base_output_dir,
                                        )
                                    elif resume:
                                        _log(f"  [skip] recon_statistics_{subset_name}", since_start=time.time() - pipeline_start, style="skip")
                            gen_dm_bond = None
                            if do_avg_gen and train_stats is not None and test_stats is not None:
                                for var in sample_variances:
                                    p = _plot_path(run_dir_eu, "gen_variance", var=str(var))
                                    if not (resume and os.path.isfile(p)):
                                        if save_structures_gro_plot:
                                            gen_dm, gen_coords_np = _get_gen_dm_euclideanizer(
                                                embed, frozen_vae, device, gen_num_samples, dm_cfg["latent_dim"], var, utils, gen_decode_batch_size,
                                                return_coords=True,
                                            )
                                            structures_dir = os.path.join(run_dir_eu, "plots", "gen_variance", "structures", str(var))
                                            write_structures_gro(gen_coords_np, structures_dir, display_root=base_output_dir)
                                        else:
                                            gen_dm = _get_gen_dm_euclideanizer(
                                                embed, frozen_vae, device, gen_num_samples, dm_cfg["latent_dim"], var, utils, gen_decode_batch_size
                                            )
                                        if gen_dm_bond is None:
                                            gen_dm_bond = gen_dm
                                        plot_gen_analysis(
                                            exp_stats, train_stats, test_stats, gen_dm, p,
                                            sample_variance=var, label_gen="Gen", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                            display_root=base_output_dir,
                                        )
                                    elif resume:
                                        _log(f"  [skip] gen_variance_{var}", since_start=time.time() - pipeline_start, style="skip")
                            if do_bond_length_by_genomic_distance and train_stats is not None and test_stats is not None:
                                _bond_all_eu = all(
                                    os.path.isfile(_plot_path(run_dir_eu, bt))
                                    for bt in (
                                        "bond_length_by_genomic_distance_gen",
                                        "bond_length_by_genomic_distance_train",
                                        "bond_length_by_genomic_distance_test",
                                    )
                                )
                                if gen_dm_bond is None:
                                    gen_dm_bond = _get_gen_dm_euclideanizer(
                                        embed, frozen_vae, device, gen_num_samples, dm_cfg["latent_dim"], sample_variances[0], utils, gen_decode_batch_size
                                    )
                                if not (resume and _bond_all_eu):
                                    _run_bond_length_plot_suite_euclideanizer(
                                        run_dir_eu, embed, frozen_vae, device, coords, eu_cfg, dm_cfg["latent_dim"],
                                        training_split, split_seed, train_stats, test_stats, gen_dm_bond,
                                        plot_cfg["max_train"], plot_cfg["max_test"],
                                        gen_num_samples, sample_variances, gen_decode_batch_size,
                                        plot_dpi, save_pdf, save_plot_data, base_output_dir,
                                    )
                                elif resume:
                                    _log("  [skip] bond_length_by_genomic_distance (gen/train/test)", since_start=time.time() - pipeline_start, style="skip")
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): plotting done in {(time.time() - plot_phase_start) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")

                        # Analysis: latent first (as one analysis block), then registered metrics (rmsd, q, coord_clustering, distmap_clustering). Score after each block.
                        any_analysis = any(
                            analysis_cfg[spec.gen_key]["enabled"] or analysis_cfg[spec.recon_key]["enabled"]
                            for spec in ANALYSIS_METRICS
                        ) or analysis_cfg["generative_capacity_rmsd"]["enabled"] or analysis_cfg["generative_capacity_q"]["enabled"]
                        if any_analysis and coords is not None:
                            if not isinstance(seed_test_to_train_holder[0], dict):
                                seed_test_to_train_holder[0] = {}
                            _cache = seed_test_to_train_holder[0]
                            analysis_phase_start = time.time()
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): analysis (latent + metrics + generative capacity)...", since_start=time.time() - pipeline_start, style="info")
                            # Before latent
                            _force_gpu_cleanup(device)
                            latent_cfg = analysis_cfg["latent"]
                            do_latent = latent_cfg["enabled"] and coords is not None
                            if do_latent:
                                latent_dir = os.path.join(run_dir_eu, "analysis", "latent")
                                latent_fig = os.path.join(latent_dir, "latent_distribution.png")
                                latent_corr_fig = os.path.join(latent_dir, "latent_correlation.png")
                                latent_data_dir = os.path.join(latent_dir, "data")
                                latent_stats_npz = os.path.join(latent_data_dir, "latent_stats.npz")
                                if not (resume and os.path.isfile(latent_fig) and os.path.isfile(latent_corr_fig)):
                                    train_mu_lat, test_mu_lat = _get_latent_vectors_euclideanizer(
                                        frozen_vae, device, coords, training_split, split_seed, utils,
                                        max_train=analysis_cfg["latent_max_train"],
                                        max_test=analysis_cfg["latent_max_test"],
                                    )
                                    os.makedirs(latent_dir, exist_ok=True)
                                    plot_latent_distribution(
                                        train_mu_lat, test_mu_lat, latent_fig,
                                        plot_dpi=plot_dpi, display_root=base_output_dir,
                                        save_pdf_copy=latent_cfg["save_pdf_copy"],
                                    )
                                    plot_latent_correlation(
                                        train_mu_lat, test_mu_lat, latent_corr_fig,
                                        plot_dpi=plot_dpi, display_root=base_output_dir,
                                        save_pdf_copy=latent_cfg["save_pdf_copy"],
                                    )
                                    effective_latent_save = latent_cfg["save_data"] or cfg["scoring"]["enabled"]
                                    if effective_latent_save:
                                        save_latent_stats_npz(
                                            train_mu_lat, test_mu_lat, latent_stats_npz,
                                            display_root=base_output_dir,
                                        )
                                elif resume:
                                    _log("  [skip] latent (all present)", since_start=time.time() - pipeline_start, style="skip")
                            _force_gpu_cleanup(device)
                            for spec in ANALYSIS_METRICS:
                                _force_gpu_cleanup(device)
                                do_gen = analysis_cfg[spec.gen_key]["enabled"]
                                do_recon = analysis_cfg[spec.recon_key]["enabled"]
                                if not (do_gen or do_recon):
                                    continue
                                gen_cfg = analysis_cfg[spec.gen_key]
                                recon_cfg = analysis_cfg[spec.recon_key]
                                _variance_list = gen_cfg["sample_variance"]
                                if _variance_list is None:
                                    _variance_list = []
                                if not isinstance(_variance_list, list):
                                    _variance_list = [_variance_list]
                                _num_samples_list = gen_cfg["num_samples"]
                                if _num_samples_list is None:
                                    _num_samples_list = []
                                if not isinstance(_num_samples_list, list):
                                    _num_samples_list = [_num_samples_list]
                                _max_recon_train_list = recon_cfg["max_recon_train"]
                                if _max_recon_train_list is None:
                                    _max_recon_train_list = [None]  # one run with no cap
                                if not isinstance(_max_recon_train_list, list):
                                    _max_recon_train_list = [_max_recon_train_list]
                                _max_recon_test_list = recon_cfg["max_recon_test"]
                                if _max_recon_test_list is None:
                                    _max_recon_test_list = [None]  # one run with no cap
                                if not isinstance(_max_recon_test_list, list):
                                    _max_recon_test_list = [_max_recon_test_list]
                                _ref_mt = analysis_cfg[f"{spec.id}_max_train"]
                                _ref_mc = analysis_cfg[f"{spec.id}_max_test"]

                                def _get_or_compute_cached(mt, mc):
                                    cache_key = spec.id
                                    if _cache.get(cache_key) is None:
                                        _cache[cache_key] = {}
                                    key = (mt, mc)
                                    if key not in _cache[cache_key]:
                                        _cache_path = os.path.join(output_dir, EXP_STATS_CACHE_DIR, spec.cache_filename(analysis_cfg, mt, mc))
                                        _run_label = os.path.basename(output_dir)
                                        _had_disk = os.path.isfile(_cache_path)
                                        if not _had_disk:
                                            _log(
                                                f"Analysis seed cache [{spec.id}] {_run_label}: computing {os.path.basename(_cache_path)} (test→train / feats).",
                                                since_start=time.time() - pipeline_start,
                                                style="info",
                                            )
                                        _cache[cache_key][key] = spec.get_or_compute_test_to_train(
                                            _cache_path, coords_np, coords, training_split, split_seed, base_output_dir,
                                            **spec.kwargs_for_cache(analysis_cfg, mt, mc),
                                        )
                                        if not _had_disk:
                                            _log(
                                                f"Analysis seed cache [{spec.id}] {_run_label}: saved {os.path.basename(_cache_path)}.",
                                                since_start=time.time() - pipeline_start,
                                                style="success",
                                            )
                                    return _cache[cache_key][key]

                                if do_gen:
                                    _mt_gen = _ref_mt if spec.requires_reference_bounds else None
                                    _mc_gen = _ref_mc if spec.requires_reference_bounds else None
                                    if spec.requires_reference_bounds and (_mt_gen is None or _mc_gen is None):
                                        continue
                                    _tt, _train_c, _test_c = _get_or_compute_cached(_ref_mt, _ref_mc)
                                    plot_cfg_gen = {
                                        **spec.build_gen_plot_cfg(analysis_cfg),
                                        "plot_dpi": plot_dpi,
                                        "save_data": analysis_save_data,
                                    }
                                    pre_kw = spec.precomputed_kwargs(_tt, _train_c, _test_c)
                                    extra_kw = spec.gen_extra_kwargs(analysis_cfg)
                                    for var in _variance_list:
                                        # Always include variance in run_name so scoring uses only variance=1 data
                                        variance_suffix = f"_var{var}"
                                        any_missing = False
                                        for n in _num_samples_list:
                                            run_name = (str(n) if len(_num_samples_list) > 1 else "default") + variance_suffix
                                            fig_path = _analysis_path(run_dir_eu, spec.subdir, f"gen/{run_name}/{spec.figure_filename}")
                                            if not (resume and os.path.isfile(fig_path)):
                                                any_missing = True
                                                break
                                        if any_missing:
                                            if len(_num_samples_list) > 1:
                                                spec.run_gen_analysis_multi(
                                                    coords_np, coords, training_split, split_seed,
                                                    frozen_vae, embed, dm_cfg["latent_dim"], device, run_dir_eu,
                                                    plot_cfg_gen,
                                                    num_samples_list=_num_samples_list,
                                                    sample_variance=var,
                                                    variance_suffix=variance_suffix,
                                                    display_root=base_output_dir,
                                                    **pre_kw,
                                                    **extra_kw,
                                                )
                                            else:
                                                n = _num_samples_list[0]
                                                run_name_single = (str(n) if len(_num_samples_list) > 1 else "default") + variance_suffix
                                                output_suffix = "_" + run_name_single
                                                spec.run_gen_analysis(
                                                    coords_np, coords, training_split, split_seed,
                                                    frozen_vae, embed, dm_cfg["latent_dim"], device, run_dir_eu,
                                                    plot_cfg_gen,
                                                    num_samples=n, sample_variance=var, output_suffix=output_suffix,
                                                    display_root=base_output_dir,
                                                    **pre_kw,
                                                    **extra_kw,
                                                )
                                        else:
                                            _log(f"  [skip] {spec.id} variance={var}", since_start=time.time() - pipeline_start, style="skip")

                                if do_recon and _max_recon_train_list and _max_recon_test_list:
                                    n_recon = len(_max_recon_train_list) * len(_max_recon_test_list)
                                    plot_cfg_recon = {
                                        **spec.build_recon_plot_cfg(analysis_cfg),
                                        "plot_dpi": plot_dpi,
                                        "save_data": analysis_save_data,
                                    }
                                    recon_extra = spec.recon_extra_kwargs(analysis_cfg)
                                    for max_recon_train in _max_recon_train_list:
                                        for max_recon_test in _max_recon_test_list:
                                            _tt, _train_c, _test_c = _get_or_compute_cached(_ref_mt, _ref_mc)
                                            if n_recon == 1:
                                                recon_subdir = ""
                                                recon_fig = _analysis_path(run_dir_eu, spec.subdir, f"recon/{spec.figure_filename}")
                                                latent_fig = _analysis_path(run_dir_eu, spec.subdir, "recon/latent_distribution.png")
                                            else:
                                                recon_subdir = f"train{max_recon_train}_test{max_recon_test}"
                                                recon_fig = _analysis_path(run_dir_eu, spec.subdir, f"recon/{recon_subdir}/{spec.figure_filename}")
                                                latent_fig = _analysis_path(run_dir_eu, spec.subdir, f"recon/{recon_subdir}/latent_distribution.png")
                                            if not (resume and os.path.isfile(recon_fig)):
                                                train_recon_coords = _get_recon_coords_euclideanizer(
                                                    embed, frozen_vae, device, coords, training_split, split_seed, utils,
                                                    use_train=True, max_n=max_recon_train,
                                                )
                                                test_recon_coords = _get_recon_coords_euclideanizer(
                                                    embed, frozen_vae, device, coords, training_split, split_seed, utils,
                                                    use_train=False, max_n=max_recon_test,
                                                )
                                                spec.run_recon_analysis(
                                                    _tt, _train_c, _test_c, train_recon_coords, test_recon_coords,
                                                    run_dir_eu, plot_cfg_recon,
                                                    display_root=base_output_dir, recon_subdir=recon_subdir,
                                                    **recon_extra,
                                                )
                                            elif resume and n_recon == 1:
                                                _log(f"  [skip] {spec.id} recon", since_start=time.time() - pipeline_start, style="skip")
                            _run_generative_capacity_blocks_for_run(
                                run_dir_eu=run_dir_eu,
                                analysis_cfg=analysis_cfg,
                                seed=split_seed,
                                latent_dim=dm_cfg["latent_dim"],
                                device=device,
                                frozen_vae=frozen_vae,
                                embed=embed,
                                resume=resume,
                                pipeline_start=pipeline_start,
                                display_root=base_output_dir,
                            )
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): analysis done in {(time.time() - analysis_phase_start) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")

                        del embed, frozen_vae
                        torch.cuda.empty_cache()
                        if cfg["scoring"]["enabled"]:
                            _run_scoring_for_run(run_dir_eu, output_dir, cfg, base_output_dir, pipeline_start)
                            _post_scoring_npz_cleanup(
                                run_dir_eu,
                                cfg,
                                defer_sufficiency_inputs=bool(cfg["meta_analysis"]["sufficiency"]["enabled"]),
                            )
                        _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): done in {(time.time() - phase_start_eu) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")
                if eu_stopped_early:
                    break
        if dm_stopped_early:
            break

def _run_one_seed(
    seed: int,
    max_data: int | None,
    cfg: dict,
    base_output_dir: str,
    dm_groups: list,
    eu_groups: list,
    dm_configs: list,
    eu_configs: list,
    coords,
    coords_np,
    device,
    num_atoms: int | None,
    num_structures: int | None,
    exp_stats,
    data_path: str | None,
    need_train: bool,
    pipeline_start: float,
    training_split: float,
    training_splits: list,
    do_plot: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
    do_rmsd: bool,
    resume: bool,
    sample_variances: list,
    gen_num_samples: int,
    gen_decode_batch_size: int,
    need_plot_or_rmsd: bool,
    save_structures_gro_plot: bool,
    analysis_save_data: bool,
    analysis_save_structures_gro: bool,
    plot_dpi: int,
    save_pdf: bool,
    save_plot_data: bool,
    num_recon_samples: int,
    analysis_cfg: dict,
    variance_list: list,
    num_samples_list: list,
    max_recon_train_list: list,
    max_recon_test_list: list,
    vis_enabled: bool,
    vis_cfg: dict,
    plot_cfg: dict,
    plot_max_train: int | None = None,
    plot_max_test: int | None = None,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_max_train: int | None = None,
    q_max_test: int | None = None,
    q_num_samples_list: list | None = None,
    q_variance_list: list | None = None,
    q_delta: float = 0.7071067811865475,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    q_recon_delta: float = 0.7071067811865475,
    make_distmap_epoch_hook=None,
    make_euclideanizer_epoch_hook=None,
    assemble_video_fn=None,
    run_entry_idx: int | None = None,
    run_entry_n: int | None = None,
) -> None:
    """Run the full pipeline for a single (seed, training_split): DistMap segments, Euclideanizer segments, plotting, analysis."""
    output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
    subset_indices_global = None
    if exp_stats is not None and "exp_distmaps" in exp_stats:
        subset_indices_global = _max_data_indices(np.asarray(exp_stats["exp_distmaps"]).shape[0], max_data, seed)
    if coords_np is not None and coords is not None:
        coords_np = _apply_max_data_subset(coords_np, max_data, seed)
        coords = torch.tensor(coords_np, dtype=torch.float32).to(device)
        num_structures = len(coords_np)
        num_atoms = coords.size(1)
    split_seed = seed
    effective_cfg = {**cfg, "output_dir": output_dir, "data": {**cfg["data"], "split_seed": seed, "training_split": training_split, "max_data": max_data}}

    _ensure_per_seed_pipeline_config(
        need_train=need_train,
        output_dir=output_dir,
        cfg=cfg,
        seed=seed,
        training_split=training_split,
        max_data=max_data,
    )

    if data_path:
        torch.manual_seed(split_seed)

    _log(f"Seed {seed}  output_dir={output_dir}", since_start=time.time() - pipeline_start, style="info")

    train_stats = test_stats = None
    _pe_run_label = os.path.basename(output_dir)
    _pe_prefix = _plot_exp_stats_precompute_prefix(_pe_run_label, run_entry_idx, run_entry_n)
    if data_path and (do_plot or do_rmsd or do_q or do_q_recon) and (coords is not None or (num_structures is not None and num_atoms is not None)):
        _exp_nt = _exp_ne = None
        if coords is not None:
            _exp_nt, _exp_ne = utils.capped_train_test_index_counts(
                coords, training_split, split_seed, plot_max_train, plot_max_test,
            )
        train_stats, test_stats = _load_exp_stats_split_cache(
            output_dir, data_path, num_structures, num_atoms, split_seed, training_split,
            max_train=plot_max_train, max_test=plot_max_test,
            expected_n_train=_exp_nt, expected_n_test=_exp_ne,
        )
        if train_stats is None or test_stats is None:
            if coords is not None:
                train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
                train_indices = np.array(train_ds.indices)
                test_indices = np.array(test_ds.indices)
                # Cap to available structures (slice uses at most len(indices))
                if plot_max_train is not None:
                    train_indices = train_indices[: plot_max_train]
                if plot_max_test is not None:
                    test_indices = test_indices[: plot_max_test]
                data_cfg = cfg["data"]
                if exp_stats is not None and subset_indices_global is not None:
                    _log(
                        f"{_pe_prefix}: computing (slice global experimental distance maps (derive train/test stats)).",
                        since_start=time.time() - pipeline_start,
                        style="info",
                    )
                    train_global_idx = subset_indices_global[train_indices]
                    test_global_idx = subset_indices_global[test_indices]
                    train_stats = _derive_stats_from_global_exp(
                        exp_stats, train_global_idx,
                        max_sep=min(num_atoms - 1, 999),
                        avg_map_sample=data_cfg["exp_stats_avg_map_sample"],
                    )
                    test_stats = _derive_stats_from_global_exp(
                        exp_stats, test_global_idx,
                        max_sep=min(num_atoms - 1, 999),
                        avg_map_sample=data_cfg["exp_stats_avg_map_sample"],
                    )
                else:
                    _log(
                        f"{_pe_prefix}: computing (full compute_exp_statistics on train/test indices (global exp stats not loaded or missing exp_distmaps)).",
                        since_start=time.time() - pipeline_start,
                        style="info",
                    )
                    train_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, min(num_atoms - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=train_indices)
                    test_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, min(num_atoms - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=test_indices)
                _save_exp_stats_split_cache(
                    output_dir, data_path, num_structures, num_atoms, split_seed, training_split,
                    train_stats, test_stats,
                    max_train=plot_max_train, max_test=plot_max_test,
                )
                _log(
                    f"{_pe_prefix}: saved split cache under {EXP_STATS_CACHE_DIR}/.",
                    since_start=time.time() - pipeline_start,
                    style="success",
                )
            else:
                _log("Train/test statistics not in cache (stats-only run); skipping recon_statistics/gen_variance for this seed.", since_start=time.time() - pipeline_start, style="skip")
        else:
            _log(
                f"{_pe_prefix}: using existing split cache.",
                since_start=time.time() - pipeline_start,
                style="skip",
            )

    seed_test_to_train_holder = [None]
    gen_decode_batch_size_holder = [gen_decode_batch_size]
    for gidx in range(len(dm_groups)):
        _run_one_distmap_group(
            seed, max_data, gidx, device,
            cfg, base_output_dir, dm_groups, eu_groups, dm_configs, eu_configs,
            coords, coords_np, num_atoms, num_structures, exp_stats, data_path, need_train, pipeline_start,
            training_split, training_splits, do_plot, do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance, do_rmsd, resume,
            sample_variances, gen_num_samples, gen_decode_batch_size_holder, need_plot_or_rmsd,
            save_structures_gro_plot, analysis_save_data, analysis_save_structures_gro,
            plot_dpi, save_pdf, save_plot_data, num_recon_samples, analysis_cfg, variance_list, num_samples_list,
            max_recon_train_list, max_recon_test_list,
            vis_enabled, vis_cfg, plot_cfg,
            train_stats, test_stats, seed_test_to_train_holder,
            do_q=do_q, do_q_recon=do_q_recon,
            q_max_train=q_max_train, q_max_test=q_max_test,
            q_num_samples_list=q_num_samples_list or [], q_variance_list=q_variance_list or [],
            q_delta=q_delta, q_max_recon_train_list=q_max_recon_train_list or [], q_max_recon_test_list=q_max_recon_test_list or [],
            q_recon_delta=q_recon_delta,
            make_distmap_epoch_hook=make_distmap_epoch_hook,
            make_euclideanizer_epoch_hook=make_euclideanizer_epoch_hook,
            assemble_video_fn=assemble_video_fn,
        )

    # Seed-level test_to_train RMSD cache is kept for reuse (e.g. when overwrite_existing re-runs analysis).


def _get_gen_dm_distmap(model, device, num_samples, latent_dim, sample_variance, gen_batch_size: int):
    """Generate distance maps from the DistMap decoder; returns (num_samples, N, N) numpy array."""
    z = dm_generate_samples(num_samples, latent_dim, device, variance=sample_variance)
    out = []
    for start in range(0, num_samples, gen_batch_size):
        end = min(start + gen_batch_size, num_samples)
        out.append(model.decode(z[start:end]))
    return np.concatenate(out, axis=0)


def _get_gen_dm_euclideanizer(embed, frozen_vae, device, num_samples, latent_dim, sample_variance, utils_mod, gen_batch_size: int, *, return_coords: bool = False):
    """Generate distance maps via frozen VAE decode + Euclideanizer. Returns (gen_dm,) or (gen_dm, gen_coords_np) if return_coords."""
    z = dm_generate_samples(num_samples, latent_dim, device, variance=sample_variance)
    out_dm = []
    out_coords = [] if return_coords else None
    with torch.no_grad():
        for start in range(0, num_samples, gen_batch_size):
            end = min(start + gen_batch_size, num_samples)
            D_noneuclid = frozen_vae._decode_to_matrix(z[start:end])
            coords = embed(D_noneuclid)
            out_dm.append(utils_mod.get_distmaps(coords).cpu().numpy())
            if return_coords:
                out_coords.append(coords.cpu().numpy().astype(np.float32))
    gen_dm = np.concatenate(out_dm, axis=0)
    if return_coords:
        return gen_dm, np.concatenate(out_coords, axis=0)
    return gen_dm


def _worker(
    device_id: int,
    task_list: list,
    log_path: str,
    shared_args: dict,
) -> None:
    """Worker entry: set device, open log, load data, run each (seed, DistMap group) task.
    When invoked from multi-GPU, the launcher sets CUDA_VISIBLE_DEVICES so this process sees a single device as cuda:0."""
    global _LOG_FILE, _LOG_LOCK, _MAX_DATA_VALUES_CONTEXT
    device = torch.device(f"cuda:{device_id}")
    _LOG_LOCK = None  # Workers use their own log handle; no cross-process lock.
    _LOG_FILE = open(log_path, "a", encoding="utf-8")
    worker_start = time.time()
    try:
        data_path = shared_args["data_path"]
        base_output_dir = shared_args["base_output_dir"]
        if not task_list or not data_path:
            return
        _MAX_DATA_VALUES_CONTEXT = list(shared_args.get("max_data_values") or [None])
        if shared_args.get("vis_enabled") and shared_args.get("make_distmap_epoch_hook") is None:
            from src.training_visualization import (
                make_distmap_epoch_hook,
                make_euclideanizer_epoch_hook,
                assemble_video,
            )
            shared_args["make_distmap_epoch_hook"] = make_distmap_epoch_hook
            shared_args["make_euclideanizer_epoch_hook"] = make_euclideanizer_epoch_hook
            shared_args["assemble_video_fn"] = assemble_video
        coords_np = utils.load_data(data_path)
        coords = torch.tensor(coords_np, dtype=torch.float32).to(device)
        num_atoms = coords.size(1)
        num_structures = len(coords_np)
        utils.validate_dataset_for_pipeline(num_structures, shared_args["training_splits"][0])
        exp_stats = _load_exp_stats_cache(
            base_output_dir, data_path, num_structures, num_atoms
        )
        if exp_stats is None and (shared_args["do_plot"] or shared_args["do_rmsd"] or shared_args.get("do_q") or shared_args.get("do_q_recon")):
            data_cfg = shared_args["cfg"]["data"]
            exp_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, min(num_atoms - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"])
        for seed, training_split, max_data, gidx in task_list:
            output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, shared_args["training_splits"], None))
            coords_np_run = _apply_max_data_subset(coords_np, max_data, seed)
            coords_run = torch.tensor(coords_np_run, dtype=torch.float32).to(device)
            num_structures_run = len(coords_np_run)
            num_atoms_run = coords_run.size(1)
            train_stats = test_stats = None
            if data_path and (shared_args["do_plot"] or shared_args["do_rmsd"] or shared_args.get("do_q") or shared_args.get("do_q_recon")):
                plot_mt = shared_args.get("plot_max_train")
                plot_mc = shared_args.get("plot_max_test")
                _w_nt, _w_ne = utils.capped_train_test_index_counts(
                    coords_run, training_split, seed, plot_mt, plot_mc,
                )
                train_stats, test_stats = _load_exp_stats_split_cache(
                    output_dir, data_path, num_structures_run, num_atoms_run,
                    seed, training_split,
                    max_train=plot_mt, max_test=plot_mc,
                    expected_n_train=_w_nt, expected_n_test=_w_ne,
                )
                if train_stats is None or test_stats is None:
                    train_ds, test_ds = utils.get_train_test_split(
                        coords_run.cpu(), training_split, seed
                    )
                    train_indices = np.array(train_ds.indices)
                    test_indices = np.array(test_ds.indices)
                    if plot_mt is not None:
                        train_indices = train_indices[: plot_mt]
                    if plot_mc is not None:
                        test_indices = test_indices[: plot_mc]
                    data_cfg = shared_args["cfg"]["data"]
                    if exp_stats is not None and "exp_distmaps" in exp_stats:
                        subset_idx = _max_data_indices(np.asarray(exp_stats["exp_distmaps"]).shape[0], max_data, seed)
                        train_stats = _derive_stats_from_global_exp(
                            exp_stats, subset_idx[train_indices],
                            max_sep=min(num_atoms_run - 1, 999),
                            avg_map_sample=data_cfg["exp_stats_avg_map_sample"],
                        )
                        test_stats = _derive_stats_from_global_exp(
                            exp_stats, subset_idx[test_indices],
                            max_sep=min(num_atoms_run - 1, 999),
                            avg_map_sample=data_cfg["exp_stats_avg_map_sample"],
                        )
                    else:
                        train_stats = compute_exp_statistics(
                            coords_np_run, device, utils.get_distmaps, min(num_atoms_run - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=train_indices
                        )
                        test_stats = compute_exp_statistics(
                            coords_np_run, device, utils.get_distmaps, min(num_atoms_run - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=test_indices
                        )
                    _save_exp_stats_split_cache(
                        output_dir, data_path, num_structures_run, num_atoms_run,
                        seed, training_split,
                        train_stats, test_stats,
                        max_train=plot_mt, max_test=plot_mc,
                    )
            seed_test_to_train_holder = [None]
            _run_one_distmap_group(
                seed, max_data, gidx, device,
                shared_args["cfg"], base_output_dir, shared_args["dm_groups"],
                shared_args["eu_groups"], shared_args["dm_configs"], shared_args["eu_configs"],
                coords_run, coords_np_run, num_atoms_run, num_structures_run, exp_stats, data_path,
                shared_args["need_train"], worker_start, training_split, shared_args["training_splits"],
                shared_args["do_plot"], shared_args["do_recon_plot"],
                shared_args["do_bond_rg_scaling"], shared_args["do_avg_gen"], shared_args["do_bond_length_by_genomic_distance"],
                shared_args["do_rmsd"], shared_args["resume"],
                shared_args["sample_variances"], shared_args["gen_num_samples"],
                shared_args["gen_decode_batch_size_holder"], shared_args["need_plot_or_rmsd"],
                shared_args["save_structures_gro_plot"], shared_args["analysis_save_data"],
                shared_args["analysis_save_structures_gro"],
                shared_args["plot_dpi"], shared_args["save_pdf"], shared_args["save_plot_data"],
                shared_args["num_recon_samples"], shared_args["analysis_cfg"],
                shared_args["variance_list"], shared_args["num_samples_list"],
                shared_args["max_recon_train_list"], shared_args["max_recon_test_list"],
                shared_args["vis_enabled"], shared_args["vis_cfg"], shared_args["plot_cfg"],
                train_stats, test_stats, seed_test_to_train_holder,
                do_q=shared_args.get("do_q", False),
                do_q_recon=shared_args.get("do_q_recon", False),
                q_max_train=shared_args.get("q_max_train"),
                q_max_test=shared_args.get("q_max_test"),
                q_num_samples_list=shared_args.get("q_num_samples_list", []),
                q_variance_list=shared_args.get("q_variance_list", []),
                q_delta=shared_args.get("q_delta", 0.7071067811865475),
                q_max_recon_train_list=shared_args.get("q_max_recon_train_list", []),
                q_max_recon_test_list=shared_args.get("q_max_recon_test_list", []),
                q_recon_delta=shared_args.get("q_recon_delta", 0.7071067811865475),
                make_distmap_epoch_hook=shared_args.get("make_distmap_epoch_hook"),
                make_euclideanizer_epoch_hook=shared_args.get("make_euclideanizer_epoch_hook"),
                assemble_video_fn=shared_args.get("assemble_video_fn"),
            )
    except Exception:
        if _LOG_LOCK is not None:
            _LOG_LOCK.acquire()
        try:
            _LOG_FILE.write("\n")
            _LOG_FILE.write("WORKER ERROR\n")
            _LOG_FILE.write(traceback.format_exc())
            _LOG_FILE.flush()
        finally:
            if _LOG_LOCK is not None:
                _LOG_LOCK.release()
        raise
    finally:
        if _LOG_FILE is not None:
            _LOG_FILE.close()
            _LOG_FILE = None
        _LOG_LOCK = None


def _run_multi_gpu_tasks(
    tasks: list,
    n_gpus: int,
    cfg: dict,
    base_output_dir: str,
    dm_groups: list,
    eu_groups: list,
    dm_configs: list,
    eu_configs: list,
    coords,
    coords_np,
    device,
    num_atoms,
    num_structures,
    exp_stats,
    data_path,
    need_train: bool,
    pipeline_start: float,
    training_splits: list,
    do_plot: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_bond_length_by_genomic_distance: bool,
    do_rmsd: bool,
    resume: bool,
    sample_variances: list,
    gen_num_samples: int,
    gen_decode_batch_size: int,
    need_plot_or_rmsd: bool,
    save_structures_gro_plot: bool,
    analysis_save_data: bool,
    analysis_save_structures_gro: bool,
    plot_dpi: int,
    save_pdf: bool,
    save_plot_data: bool,
    num_recon_samples: int,
    analysis_cfg: dict,
    variance_list: list,
    num_samples_list: list,
    max_recon_train_list: list,
    max_recon_test_list: list,
    vis_enabled: bool,
    vis_cfg: dict,
    plot_cfg: dict,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_max_train: int | None = None,
    q_max_test: int | None = None,
    q_num_samples_list: list | None = None,
    q_variance_list: list | None = None,
    q_delta: float = 0.7071067811865475,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    q_recon_delta: float = 0.7071067811865475,
    make_distmap_epoch_hook=None,
    make_euclideanizer_epoch_hook=None,
    assemble_video_fn=None,
) -> None:
    """Run tasks in parallel on multiple GPUs (one process per device). Tasks are (seed, training_split, max_data, group_idx)."""
    run_entries_from_tasks = list(dict.fromkeys([(t[0], t[1], t[2]) for t in tasks]))
    # Per-run setup: output dirs and pipeline config. Train/test caches are precomputed in main when multi-GPU (to free data before spawn).
    for seed, training_split, max_data in run_entries_from_tasks:
        output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
        coords_np_run = _apply_max_data_subset(coords_np, max_data, seed) if coords_np is not None else None
        coords_run = torch.tensor(coords_np_run, dtype=torch.float32).to(device) if coords_np_run is not None else None
        num_structures_run = len(coords_np_run) if coords_np_run is not None else num_structures
        num_atoms_run = coords_run.size(1) if coords_run is not None else num_atoms
        _ensure_per_seed_pipeline_config(
            need_train=need_train,
            output_dir=output_dir,
            cfg=cfg,
            seed=seed,
            training_split=training_split,
            max_data=max_data,
        )
        if data_path and coords is not None and (do_plot or do_rmsd or do_q or do_q_recon):
            plot_mt = plot_cfg["max_train"]
            plot_mc = plot_cfg["max_test"]
            if _exp_stats_split_cache_meta_files_ok(
                output_dir, data_path, num_structures_run, num_atoms_run, seed, training_split,
                max_train=plot_mt, max_test=plot_mc,
            ):
                continue
            train_ds, test_ds = utils.get_train_test_split(coords_run.cpu(), training_split, seed)
            train_indices = np.array(train_ds.indices)
            test_indices = np.array(test_ds.indices)
            if plot_mt is not None:
                train_indices = train_indices[: plot_mt]
            if plot_mc is not None:
                test_indices = test_indices[: plot_mc]
            data_cfg = cfg["data"]
            train_stats = compute_exp_statistics(coords_np_run, device, utils.get_distmaps, min(num_atoms_run - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=train_indices)
            test_stats = compute_exp_statistics(coords_np_run, device, utils.get_distmaps, min(num_atoms_run - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=test_indices)
            _save_exp_stats_split_cache(
                output_dir, data_path, num_structures_run, num_atoms_run,
                seed, training_split, train_stats, test_stats,
                max_train=plot_mt, max_test=plot_mc,
            )
    # Round-robin task assignment
    tasks_by_device = [list() for _ in range(n_gpus)]
    for i, t in enumerate(tasks):
        tasks_by_device[i % n_gpus].append(t)
    log_path = os.path.join(base_output_dir, PIPELINE_LOG_FILENAME)
    shared_args = {
        "cfg": cfg,
        "base_output_dir": base_output_dir,
        "dm_groups": dm_groups,
        "eu_groups": eu_groups,
        "dm_configs": dm_configs,
        "eu_configs": eu_configs,
        "data_path": data_path,
        "need_train": need_train,
        "pipeline_start": pipeline_start,
        "training_splits": training_splits,
        "max_data_values": _MAX_DATA_VALUES_CONTEXT,
        "do_plot": do_plot,
        "do_recon_plot": do_recon_plot,
        "do_bond_rg_scaling": do_bond_rg_scaling,
        "do_avg_gen": do_avg_gen,
        "do_bond_length_by_genomic_distance": do_bond_length_by_genomic_distance,
        "do_rmsd": do_rmsd,
        "resume": resume,
        "sample_variances": sample_variances,
        "gen_num_samples": gen_num_samples,
        "gen_decode_batch_size_holder": [gen_decode_batch_size],
        "need_plot_or_rmsd": need_plot_or_rmsd,
        "save_structures_gro_plot": save_structures_gro_plot,
        "analysis_save_data": analysis_save_data,
        "analysis_save_structures_gro": analysis_save_structures_gro,
        "plot_dpi": plot_dpi,
        "save_pdf": save_pdf,
        "save_plot_data": save_plot_data,
        "num_recon_samples": num_recon_samples,
        "analysis_cfg": analysis_cfg,
        "variance_list": variance_list,
        "num_samples_list": num_samples_list,
        "max_recon_train_list": max_recon_train_list,
        "max_recon_test_list": max_recon_test_list,
        "vis_enabled": vis_enabled,
        "vis_cfg": vis_cfg,
        "plot_cfg": plot_cfg,
        "plot_max_train": plot_cfg["max_train"],
        "plot_max_test": plot_cfg["max_test"],
        "do_q": do_q,
        "do_q_recon": do_q_recon,
        "q_max_train": q_max_train,
        "q_max_test": q_max_test,
        "q_num_samples_list": q_num_samples_list or [],
        "q_variance_list": q_variance_list or [],
        "q_delta": q_delta,
        "q_max_recon_train_list": q_max_recon_train_list or [],
        "q_max_recon_test_list": q_max_recon_test_list or [],
        "q_recon_delta": q_recon_delta,
        "make_distmap_epoch_hook": make_distmap_epoch_hook,
        "make_euclideanizer_epoch_hook": make_euclideanizer_epoch_hook,
        "assemble_video_fn": assemble_video_fn,
    }
    n_workers = min(n_gpus, len(tasks)) if tasks else 0
    shared_args = {
        **shared_args,
        "make_distmap_epoch_hook": None,
        "make_euclideanizer_epoch_hook": None,
        "assemble_video_fn": None,
    }
    # Launch workers as subprocesses with CUDA_VISIBLE_DEVICES set before process start,
    # so each worker gets a dedicated GPU (fixes SLURM/clusters where child processes
    # started via multiprocessing.Process do not get GPU access).
    run_py_path = os.path.abspath(os.path.join(_SCRIPT_DIR, "run.py"))
    procs = []
    for device_id in range(n_workers):
        task_list = tasks_by_device[device_id]
        fd, pickle_path = tempfile.mkstemp(suffix=".pkl", prefix="euclideanizer_worker_")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump((task_list, log_path, shared_args), f)
        except Exception:
            os.close(fd)
            try:
                os.remove(pickle_path)
            except OSError:
                pass
            raise
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device_id)}
        p = subprocess.Popen(
            [sys.executable, run_py_path, "--worker-from-pickle", pickle_path],
            env=env,
            cwd=os.getcwd(),
        )
        procs.append((p, pickle_path))
    for p, pickle_path in procs:
        p.wait()
        if p.returncode != 0 and os.path.isfile(pickle_path):
            try:
                os.remove(pickle_path)
            except OSError:
                pass
    failed = [p for p, _ in procs if p.returncode != 0]
    if failed:
        exit_codes = [p.returncode for p in failed]
        msg = f"Multi-GPU worker(s) exited with non-zero status: {exit_codes}"
        if any(c in (-9, -11) for c in exit_codes):
            msg += (
                " (exit -9 often indicates the process was killed, e.g. out-of-memory; -11 indicates a crash). "
                "Try reducing memory use: --gpus 2 or --no-multi-gpu."
            )
        raise RuntimeError(msg)


def main():
    global _LOG_FILE, _pipeline_real_stdout, _pipeline_real_stderr, _MAX_DATA_VALUES_CONTEXT
    pipeline_start = time.time()
    args = _parse_args()
    # Worker subprocess entry: load args from pickle and run _worker (CUDA_VISIBLE_DEVICES already set by parent).
    if getattr(args, "worker_from_pickle", None):
        with open(args.worker_from_pickle, "rb") as f:
            task_list, log_path, shared_args = pickle.load(f)
        _pipeline_real_stdout = sys.stdout
        _pipeline_real_stderr = sys.stderr
        if _pipeline_real_stdout.isatty():
            sys.stdout = _StyledStdout(_pipeline_real_stdout)
        if _pipeline_real_stderr.isatty():
            sys.stderr = _StyledStderr(_pipeline_real_stderr)
        try:
            _worker(0, task_list, log_path, shared_args)
        finally:
            try:
                os.remove(args.worker_from_pickle)
            except OSError:
                pass
        return
    overrides = _args_to_overrides(args)
    config_path = args.config
    cfg = load_config(path=config_path, overrides=overrides, validate_scoring_tau=False)
    data_path = get_data_path(cfg)
    base_output_dir = get_output_dir(cfg)
    resume = cfg["resume"]
    if not resume and os.path.isdir(base_output_dir):
        if not getattr(args, "yes_overwrite", False):
            _confirm_overwrite(base_output_dir)
        shutil.rmtree(base_output_dir)
    _init_log_file(base_output_dir)
    _pipeline_real_stdout = sys.stdout
    _pipeline_real_stderr = sys.stderr
    if _pipeline_real_stdout.isatty():
        sys.stdout = _StyledStdout(_pipeline_real_stdout)
    if _pipeline_real_stderr.isatty():
        sys.stderr = _StyledStderr(_pipeline_real_stderr)
    finalize_scoring_tau_config(cfg, config_path)
    # Save full run config in output root for reproducibility (can re-run with: run.py --config <base_output_dir>/pipeline_config.yaml)
    root_run_cfg = {**cfg, "output_dir": base_output_dir}
    save_pipeline_config(root_run_cfg, base_output_dir)
    seeds = get_seeds(cfg)
    training_splits = get_training_splits(cfg)
    max_data_values = get_max_data_values(cfg)
    _MAX_DATA_VALUES_CONTEXT = list(max_data_values)
    run_entries = [(s, t, md) for s in seeds for t in training_splits for md in max_data_values]
    plot_cfg = cfg["plotting"]
    do_plot = plot_cfg["enabled"]
    plot_dpi = int(plot_cfg["plot_dpi"])
    save_pdf = plot_cfg["save_pdf_copy"]
    scoring_enabled = cfg["scoring"]["enabled"]
    meta_suff_cfg = cfg["meta_analysis"]["sufficiency"]
    meta_suff_enabled = meta_suff_cfg["enabled"]
    save_plot_data = plot_cfg["save_data"] or scoring_enabled  # effective: save when scoring needs NPZ
    num_recon_samples = plot_cfg["num_reconstruction_samples"]
    do_recon_plot = plot_cfg["reconstruction"]
    do_bond_rg_scaling = plot_cfg["bond_rg_scaling"]
    do_avg_gen = plot_cfg["avg_gen_vs_exp"]
    do_bond_length_by_genomic_distance = plot_cfg["bond_length_by_genomic_distance"]
    analysis_cfg = cfg["analysis"]
    do_rmsd = analysis_cfg["rmsd_gen"]["enabled"]
    do_rmsd_recon_cfg = analysis_cfg["rmsd_recon"]["enabled"]
    do_q = analysis_cfg["q_gen"]["enabled"]
    do_q_recon_cfg = analysis_cfg["q_recon"]["enabled"]
    do_gc_rmsd = analysis_cfg["generative_capacity_rmsd"]["enabled"]
    do_gc_q = analysis_cfg["generative_capacity_q"]["enabled"]
    do_coord_clustering_gen = analysis_cfg["coord_clustering_gen"]["enabled"]
    do_coord_clustering_recon_cfg = analysis_cfg["coord_clustering_recon"]["enabled"]
    do_distmap_clustering_gen = analysis_cfg["distmap_clustering_gen"]["enabled"]
    do_distmap_clustering_recon_cfg = analysis_cfg["distmap_clustering_recon"]["enabled"]
    do_dashboard = cfg["dashboard"]["enabled"]
    if getattr(args, "no_dashboard", False):
        do_dashboard = False

    # Training requires dataset
    dm_configs = expand_distmap_grid(cfg)
    eu_configs = expand_euclideanizer_grid(cfg)
    need_train = len(dm_configs) > 0 or len(eu_configs) > 0
    if need_train and not data_path:
        _log_raw("ERROR: Training requested but no dataset path. Set --data or data.path in config.", style="error")
        sys.exit(1)

    _log("Pipeline started.", since_start=time.time() - pipeline_start, style="info")
    _warn_calibration_reserve_if_low(cfg, pipeline_start)
    _log(f"config: {config_path}  output: {base_output_dir}  seeds: {seeds}", since_start=time.time() - pipeline_start, style="info")
    _log(f"DistMap runs: {len(dm_configs)}  Euclideanizer: {len(eu_configs)}  resume={resume}  plot={do_plot}  rmsd_gen={do_rmsd}  rmsd_recon={do_rmsd_recon_cfg}  q_gen={do_q}  q_recon={do_q_recon_cfg}  gc_rmsd={do_gc_rmsd}  gc_q={do_gc_q}  coord_clustering_gen={do_coord_clustering_gen}  coord_clustering_recon={do_coord_clustering_recon_cfg}  distmap_clustering_gen={do_distmap_clustering_gen}  distmap_clustering_recon={do_distmap_clustering_recon_cfg}  sufficiency_meta={meta_suff_enabled}", since_start=time.time() - pipeline_start, style="info")

    num_samples_list = analysis_cfg["rmsd_gen"]["num_samples"] if do_rmsd else []
    if not isinstance(num_samples_list, list):
        num_samples_list = [num_samples_list]
    variance_list = analysis_cfg["rmsd_gen"]["sample_variance"] if do_rmsd else []
    if not isinstance(variance_list, list):
        variance_list = [variance_list]
    max_recon_train_list = analysis_cfg["rmsd_recon"]["max_recon_train"] if do_rmsd_recon_cfg else []
    if not isinstance(max_recon_train_list, list):
        max_recon_train_list = [max_recon_train_list]
    max_recon_test_list = analysis_cfg["rmsd_recon"]["max_recon_test"] if do_rmsd_recon_cfg else []
    if not isinstance(max_recon_test_list, list):
        max_recon_test_list = [max_recon_test_list]

    q_max_train = analysis_cfg["q_max_train"] if do_q else None
    q_max_test = analysis_cfg["q_max_test"] if do_q else None
    q_num_samples_list = analysis_cfg["q_gen"]["num_samples"] if do_q else []
    if not isinstance(q_num_samples_list, list):
        q_num_samples_list = [q_num_samples_list]
    q_variance_list = analysis_cfg["q_gen"]["sample_variance"] if do_q else []
    if not isinstance(q_variance_list, list):
        q_variance_list = [q_variance_list]
    q_delta = analysis_cfg["q_gen"]["delta"] if do_q else (1.0 / (2.0 ** 0.5))
    q_max_recon_train_list = analysis_cfg["q_recon"]["max_recon_train"] if do_q_recon_cfg else []
    if not isinstance(q_max_recon_train_list, list):
        q_max_recon_train_list = [q_max_recon_train_list]
    q_max_recon_test_list = analysis_cfg["q_recon"]["max_recon_test"] if do_q_recon_cfg else []
    if not isinstance(q_max_recon_test_list, list):
        q_max_recon_test_list = [q_max_recon_test_list]
    q_recon_delta = analysis_cfg["q_recon"]["delta"]
    coord_clustering_num_samples_list = analysis_cfg["coord_clustering_gen"]["num_samples"] if do_coord_clustering_gen else []
    if not isinstance(coord_clustering_num_samples_list, list):
        coord_clustering_num_samples_list = [coord_clustering_num_samples_list]
    coord_clustering_variance_list = analysis_cfg["coord_clustering_gen"]["sample_variance"] if do_coord_clustering_gen else []
    if not isinstance(coord_clustering_variance_list, list):
        coord_clustering_variance_list = [coord_clustering_variance_list]
    coord_clustering_max_recon_train_list = analysis_cfg["coord_clustering_recon"]["max_recon_train"] if do_coord_clustering_recon_cfg else []
    if not isinstance(coord_clustering_max_recon_train_list, list):
        coord_clustering_max_recon_train_list = [coord_clustering_max_recon_train_list]
    coord_clustering_max_recon_test_list = analysis_cfg["coord_clustering_recon"]["max_recon_test"] if do_coord_clustering_recon_cfg else []
    if not isinstance(coord_clustering_max_recon_test_list, list):
        coord_clustering_max_recon_test_list = [coord_clustering_max_recon_test_list]
    distmap_clustering_num_samples_list = analysis_cfg["distmap_clustering_gen"]["num_samples"] if do_distmap_clustering_gen else []
    if not isinstance(distmap_clustering_num_samples_list, list):
        distmap_clustering_num_samples_list = [distmap_clustering_num_samples_list]
    distmap_clustering_variance_list = analysis_cfg["distmap_clustering_gen"]["sample_variance"] if do_distmap_clustering_gen else []
    if not isinstance(distmap_clustering_variance_list, list):
        distmap_clustering_variance_list = [distmap_clustering_variance_list]
    distmap_clustering_max_recon_train_list = analysis_cfg["distmap_clustering_recon"]["max_recon_train"] if do_distmap_clustering_recon_cfg else []
    if not isinstance(distmap_clustering_max_recon_train_list, list):
        distmap_clustering_max_recon_train_list = [distmap_clustering_max_recon_train_list]
    distmap_clustering_max_recon_test_list = analysis_cfg["distmap_clustering_recon"]["max_recon_test"] if do_distmap_clustering_recon_cfg else []
    if not isinstance(distmap_clustering_max_recon_test_list, list):
        distmap_clustering_max_recon_test_list = [distmap_clustering_max_recon_test_list]
    dm_groups = distmap_training_groups(cfg)
    eu_groups = euclideanizer_training_groups(cfg)
    plot_variances_for_scan = get_sample_variances(cfg) if do_plot else []

    # overwrite_existing: prompt and delete existing plotting/analysis outputs up front (requires user approval)
    _overwrite_descriptors = [
        ("plotting", do_plot, plot_cfg["overwrite_existing"], lambda: _has_any_plotting_output(base_output_dir, run_entries, training_splits)),
        ("rmsd_gen", do_rmsd, analysis_cfg["rmsd_gen"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "rmsd_gen")),
        ("rmsd_recon", do_rmsd_recon_cfg, analysis_cfg["rmsd_recon"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "rmsd_recon")),
        ("q_gen", do_q, analysis_cfg["q_gen"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "q_gen")),
        ("q_recon", do_q_recon_cfg, analysis_cfg["q_recon"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "q_recon")),
        ("generative_capacity_rmsd", do_gc_rmsd, analysis_cfg["generative_capacity_rmsd"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "generative_capacity_rmsd")),
        ("generative_capacity_q", do_gc_q, analysis_cfg["generative_capacity_q"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "generative_capacity_q")),
        ("coord_clustering_gen", do_coord_clustering_gen, analysis_cfg["coord_clustering_gen"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "coord_clustering_gen")),
        ("coord_clustering_recon", do_coord_clustering_recon_cfg, analysis_cfg["coord_clustering_recon"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "coord_clustering_recon")),
        ("distmap_clustering_gen", do_distmap_clustering_gen, analysis_cfg["distmap_clustering_gen"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "distmap_clustering_gen")),
        ("distmap_clustering_recon", do_distmap_clustering_recon_cfg, analysis_cfg["distmap_clustering_recon"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "distmap_clustering_recon")),
        ("latent", analysis_cfg["latent"]["enabled"], analysis_cfg["latent"]["overwrite_existing"], lambda: _has_any_analysis_output(base_output_dir, run_entries, training_splits, "latent")),
        ("scoring", scoring_enabled, cfg["scoring"]["overwrite_existing"], lambda: _has_any_scoring_output(base_output_dir, run_entries)),
        ("meta_analysis_sufficiency", meta_suff_enabled, meta_suff_cfg["overwrite_existing"], lambda: _has_any_sufficiency_meta_output(base_output_dir)),
    ]
    to_overwrite = [label for (label, en, ov, has_out) in _overwrite_descriptors if en and ov and has_out()]
    if to_overwrite:
        if not getattr(args, "yes_overwrite", False):
            _confirm_overwrite_outputs(to_overwrite)
        _delete_dashboard(base_output_dir)
        _log("Removing existing outputs (overwrite_existing requested): " + ", ".join(to_overwrite), since_start=time.time() - pipeline_start, style="info")
        for label in to_overwrite:
            if label == "plotting":
                _delete_plotting_outputs_only(base_output_dir, run_entries, training_splits)
            elif label == "scoring":
                _delete_scoring_outputs(base_output_dir)
            elif label == "meta_analysis_sufficiency":
                _delete_sufficiency_meta_outputs(base_output_dir)
            else:
                _delete_analysis_outputs_for_component(base_output_dir, run_entries, training_splits, label)
        _log("Done removing; will re-run these components.", since_start=time.time() - pipeline_start, style="success")

    # Pipeline config: strict match for training; if only plotting/analysis differ, prompt then delete and update saved config
    def _cfg_for_compare(c):
        d = dict(c)
        d.pop("resume", None)
        return d

    if need_train and resume and data_path:
        chunks_to_update = set()
        saved_compare_for_ref = None
        for entry in run_entries:
            seed, training_split, max_data = _entry_seed_split_max(entry)
            output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
            if os.path.isdir(output_dir):
                if not os.path.isfile(pipeline_config_path(output_dir)):
                    raise RuntimeError(
                        f"Resume is enabled but no pipeline config found in output_dir ({output_dir!r}). "
                        f"Refusing to resume without an exact config copy. Use a different output_dir or run with --no-resume to overwrite existing files in that directory."
                    )
                effective_cfg = {**cfg, "output_dir": output_dir, "data": {**cfg["data"], "split_seed": seed, "training_split": training_split, "max_data": max_data}}
                saved_cfg = load_pipeline_config(output_dir)
                saved_compare = _cfg_for_compare(saved_cfg) if saved_cfg else None
                effective_compare = _cfg_for_compare(effective_cfg)
                if saved_compare_for_ref is None:
                    saved_compare_for_ref = saved_compare
                if saved_compare is None:
                    raise RuntimeError(
                        f"Resume is enabled but pipeline config in {output_dir!r} could not be loaded. "
                        f"Use a different output_dir or run with --no-resume to overwrite."
                    )
                if not configs_match_sections(saved_compare, effective_compare, TRAINING_CRITICAL_KEYS):
                    diff_lines = config_diff(
                        {k: saved_compare.get(k) for k in TRAINING_CRITICAL_KEYS if k in saved_compare},
                        {k: effective_compare.get(k) for k in TRAINING_CRITICAL_KEYS if k in effective_compare},
                    )
                    diff_msg = "\n  ".join(diff_lines[:20])
                    if len(diff_lines) > 20:
                        diff_msg += f"\n  ... and {len(diff_lines) - 20} more."
                    raise RuntimeError(
                        f"Resume is enabled but training config (data, distmap, euclideanizer, training_visualization) does not match.\n"
                        f"Output dir: {output_dir!r}\n"
                        f"Differences (saved vs current):\n  {diff_msg}\n"
                        f"Use a different output_dir or run with --no-resume to overwrite."
                    )
                # Collect which chunks (plotting, rmsd_gen, rmsd_recon) differ from saved
                if not configs_match_sections(saved_compare, effective_compare, ["plotting"]):
                    chunks_to_update.add("plotting")
                s_analysis = saved_compare["analysis"]
                e_analysis = effective_compare["analysis"]
                if s_analysis["rmsd_gen"] != e_analysis["rmsd_gen"]:
                    chunks_to_update.add("rmsd_gen")
                if s_analysis["rmsd_recon"] != e_analysis["rmsd_recon"]:
                    chunks_to_update.add("rmsd_recon")
                if s_analysis["q_gen"] != e_analysis["q_gen"]:
                    chunks_to_update.add("q_gen")
                if s_analysis["q_recon"] != e_analysis["q_recon"]:
                    chunks_to_update.add("q_recon")
                if s_analysis["generative_capacity_rmsd"] != e_analysis["generative_capacity_rmsd"]:
                    chunks_to_update.add("generative_capacity_rmsd")
                if s_analysis["generative_capacity_q"] != e_analysis["generative_capacity_q"]:
                    chunks_to_update.add("generative_capacity_q")
                if s_analysis["coord_clustering_gen"] != e_analysis["coord_clustering_gen"]:
                    chunks_to_update.add("coord_clustering_gen")
                if s_analysis["coord_clustering_recon"] != e_analysis["coord_clustering_recon"]:
                    chunks_to_update.add("coord_clustering_recon")
                if s_analysis["distmap_clustering_gen"] != e_analysis["distmap_clustering_gen"]:
                    chunks_to_update.add("distmap_clustering_gen")
                if s_analysis["distmap_clustering_recon"] != e_analysis["distmap_clustering_recon"]:
                    chunks_to_update.add("distmap_clustering_recon")
                if s_analysis["latent"] != e_analysis["latent"]:
                    chunks_to_update.add("latent")
                if saved_compare.get("meta_analysis", {}).get("sufficiency") != effective_compare.get("meta_analysis", {}).get("sufficiency"):
                    chunks_to_update.add("meta_analysis_sufficiency")
        chunks_to_update = sorted(chunks_to_update)  # stable order: rmsd_gen, rmsd_recon, q_gen, q_recon, plotting
        if "plotting" in chunks_to_update:
            chunks_to_update = ["plotting"] + [c for c in chunks_to_update if c != "plotting"]
        # Reference-size change: if max_train/max_test (or equivalent) changed and we'll run that component, purge caches (with approval) so data is recomputed
        if saved_compare_for_ref is not None:
            saved_ref = _reference_size_config(saved_compare_for_ref)
            current_ref = _reference_size_config(_cfg_for_compare(cfg))
            ref_changed = _reference_size_changed(saved_ref, current_ref)
            purge_ref = set()
            if "plotting" in ref_changed and do_plot:
                purge_ref.add("plotting")
            if "rmsd" in ref_changed and do_rmsd:
                purge_ref.add("rmsd")
            if "q" in ref_changed and do_q:
                purge_ref.add("q")
            if "coord_clustering" in ref_changed and (do_coord_clustering_gen or do_coord_clustering_recon_cfg):
                purge_ref.add("coord_clustering")
            if "distmap_clustering" in ref_changed and (do_distmap_clustering_gen or do_distmap_clustering_recon_cfg):
                purge_ref.add("distmap_clustering")
            if purge_ref:
                if not getattr(args, "yes_overwrite", False):
                    _confirm_reference_size_cache_purge(purge_ref)
                _log(
                    "Removing cached data (reference size changed): " + ", ".join(sorted(purge_ref)),
                    since_start=time.time() - pipeline_start,
                    style="info",
                )
                _delete_reference_size_caches(base_output_dir, run_entries, training_splits, purge_ref)
                _delete_dashboard(base_output_dir)
        # Chunks already deleted by overwrite_existing block above: skip second prompt and delete
        chunks_still_to_delete = [c for c in chunks_to_update if c not in to_overwrite]
        # Only prompt and delete for chunks that actually have existing outputs
        chunk_labels = {"plotting": "Plotting", "rmsd_gen": "RMSD (gen)", "rmsd_recon": "RMSD (recon)", "q_gen": "Q (gen)", "q_recon": "Q (recon)", "generative_capacity_rmsd": "Generative Capacity (RMSD)", "generative_capacity_q": "Generative Capacity (Q)", "generative_capacity_convergence": "Generative Capacity (Median Vs N)", "coord_clustering_gen": "Coord clustering (gen)", "coord_clustering_recon": "Coord clustering (recon)", "distmap_clustering_gen": "Distmap clustering (gen)", "distmap_clustering_recon": "Distmap clustering (recon)", "latent": "Latent", "meta_analysis_sufficiency": "Sufficiency Meta-Analysis"}
        chunks_with_outputs = [
            c for c in chunks_still_to_delete
            if (c == "plotting" and _has_any_plotting_output(base_output_dir, run_entries, training_splits))
            or (c == "meta_analysis_sufficiency" and _has_any_sufficiency_meta_output(base_output_dir))
            or (c != "plotting" and _has_any_analysis_output(base_output_dir, run_entries, training_splits, c))
        ]
        if chunks_to_update:
            _delete_dashboard(base_output_dir)  # Remove dashboard whenever we will re-run any plotting/analysis (even if no outputs under new paths yet)
            for chunk in chunks_with_outputs:
                if not getattr(args, "yes_overwrite", False):
                    _confirm_replot_one_chunk(base_output_dir, chunk_labels[chunk])
                _log(f"Removing existing {chunk_labels[chunk]} outputs (config changed).", since_start=time.time() - pipeline_start, style="info")
                if chunk == "plotting":
                    _delete_plotting_outputs_only(base_output_dir, run_entries, training_splits)
                elif chunk == "meta_analysis_sufficiency":
                    _delete_sufficiency_meta_outputs(base_output_dir)
                else:
                    _delete_analysis_outputs_for_component(base_output_dir, run_entries, training_splits, chunk)
            save_pipeline_config({**cfg, "output_dir": base_output_dir}, base_output_dir)
            for entry in run_entries:
                seed, training_split, max_data = _entry_seed_split_max(entry)
                output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
                if os.path.isdir(output_dir):
                    save_pipeline_config(
                        {**cfg, "output_dir": output_dir, "data": {**cfg["data"], "split_seed": seed, "training_split": training_split, "max_data": max_data}},
                        output_dir,
                    )
            _log("Saved updated pipeline config; will skip training and re-run affected plotting/analysis.", since_start=time.time() - pipeline_start, style="success")
            need_train = False  # Skip training when only plotting/analysis config changed

    # Reference-size purge when resume but we did not load saved config above (e.g. plotting/analysis-only run)
    if (
        resume
        and data_path
        and seeds
        and not need_train
        and (do_plot or do_rmsd or do_q or do_coord_clustering_gen or do_coord_clustering_recon_cfg or do_distmap_clustering_gen or do_distmap_clustering_recon_cfg)
    ):
        first_seed_dir = os.path.join(base_output_dir, _seed_split_dir_name(run_entries[0][0], run_entries[0][1], _entry_seed_split_max(run_entries[0])[2], training_splits, None))
        if os.path.isdir(first_seed_dir) and os.path.isfile(pipeline_config_path(first_seed_dir)):
            saved_cfg_ref = load_pipeline_config(first_seed_dir)
            if saved_cfg_ref is not None:
                saved_ref = _reference_size_config(_cfg_for_compare(saved_cfg_ref))
                current_ref = _reference_size_config(_cfg_for_compare(cfg))
                ref_changed = _reference_size_changed(saved_ref, current_ref)
                purge_ref = set()
                if "plotting" in ref_changed and do_plot:
                    purge_ref.add("plotting")
                if "rmsd" in ref_changed and do_rmsd:
                    purge_ref.add("rmsd")
                if "q" in ref_changed and do_q:
                    purge_ref.add("q")
                if "coord_clustering" in ref_changed and (do_coord_clustering_gen or do_coord_clustering_recon_cfg):
                    purge_ref.add("coord_clustering")
                if "distmap_clustering" in ref_changed and (do_distmap_clustering_gen or do_distmap_clustering_recon_cfg):
                    purge_ref.add("distmap_clustering")
                if purge_ref:
                    if not getattr(args, "yes_overwrite", False):
                        _confirm_reference_size_cache_purge(purge_ref)
                    _log(
                        "Removing cached data (reference size changed): " + ", ".join(sorted(purge_ref)),
                        since_start=time.time() - pipeline_start,
                        style="info",
                    )
                    _delete_reference_size_caches(base_output_dir, run_entries, training_splits, purge_ref)
                    _delete_dashboard(base_output_dir)

    # Decide what to load from pipeline segments (resume) or from flags (no resume)
    if not resume or not data_path:
        do_rmsd_recon = analysis_cfg["rmsd_recon"]["enabled"]
        do_q_recon = analysis_cfg["q_recon"]["enabled"]
        needs = PipelineDataNeeds(
            need_coords=(
                need_train
                or do_plot
                or do_rmsd
                or do_rmsd_recon
                or do_q
                or do_q_recon
                or do_coord_clustering_gen
                or do_coord_clustering_recon_cfg
                or do_distmap_clustering_gen
                or do_distmap_clustering_recon_cfg
                or do_gc_rmsd
                or do_gc_q
            ),
            need_exp_stats=do_plot,
            need_train_test_stats=do_plot,
        ) if data_path else PipelineDataNeeds(need_coords=False, need_exp_stats=False, need_train_test_stats=False)
    else:
        needs = _pipeline_data_needs(
            base_output_dir, run_entries, training_splits, dm_groups, eu_groups,
            resume, do_plot, do_rmsd, do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_bond_length_by_genomic_distance,
            plot_variances_for_scan, variance_list, num_samples_list,
            do_rmsd_recon=analysis_cfg["rmsd_recon"]["enabled"],
            max_recon_train_list=max_recon_train_list, max_recon_test_list=max_recon_test_list,
            do_q=do_q, do_q_recon=do_q_recon_cfg,
            q_variance_list=q_variance_list, q_num_samples_list=q_num_samples_list,
            q_max_recon_train_list=q_max_recon_train_list, q_max_recon_test_list=q_max_recon_test_list,
            do_coord_clustering_gen=do_coord_clustering_gen, do_coord_clustering_recon=do_coord_clustering_recon_cfg,
            coord_clustering_variance_list=coord_clustering_variance_list, coord_clustering_num_samples_list=coord_clustering_num_samples_list,
            coord_clustering_max_recon_train_list=coord_clustering_max_recon_train_list, coord_clustering_max_recon_test_list=coord_clustering_max_recon_test_list,
            do_distmap_clustering_gen=do_distmap_clustering_gen, do_distmap_clustering_recon=do_distmap_clustering_recon_cfg,
            distmap_clustering_variance_list=distmap_clustering_variance_list, distmap_clustering_num_samples_list=distmap_clustering_num_samples_list,
            distmap_clustering_max_recon_train_list=distmap_clustering_max_recon_train_list, distmap_clustering_max_recon_test_list=distmap_clustering_max_recon_test_list,
            do_latent=analysis_cfg["latent"]["enabled"],
            do_generative_capacity_rmsd=do_gc_rmsd,
            do_generative_capacity_q=do_gc_q,
        )
    # Run pipeline when any output is missing (same as plotting/analysis: overwrite is handled by upfront delete, then need = output missing)
    scoring_needs_run = scoring_enabled and not _has_any_scoring_output(base_output_dir, run_entries)
    gc_needs_run = (
        (do_gc_rmsd and not _has_any_analysis_output(base_output_dir, run_entries, training_splits, "generative_capacity_rmsd"))
        or (do_gc_q and not _has_any_analysis_output(base_output_dir, run_entries, training_splits, "generative_capacity_q"))
    )
    meta_needs_run = meta_suff_enabled and not _has_any_sufficiency_meta_output(base_output_dir)
    need_any = ((needs.need_any() or scoring_needs_run or gc_needs_run) and data_path) or meta_needs_run

    # Remove dashboard when we will run any block (plotting, analysis, or scoring) so it gets rebuilt with current outputs
    if need_any and (do_plot or do_rmsd or do_rmsd_recon_cfg or do_q or do_q_recon_cfg or do_gc_rmsd or do_gc_q or do_coord_clustering_gen or do_coord_clustering_recon_cfg or do_distmap_clustering_gen or do_distmap_clustering_recon_cfg or scoring_needs_run or meta_needs_run):
        _delete_dashboard(base_output_dir)

    # Scoring-only: no coords/stats needed; run scoring for every run without loading data.
    # Not "scoring only" if generative capacity must run (needs coords + models on the full seed loop).
    scoring_only = (
        need_any
        and scoring_needs_run
        and not needs.need_any()
        and not gc_needs_run
    )
    # Not "meta only" if generative capacity (or scoring) must run — those require the normal seed loop / data load.
    meta_only = (
        need_any
        and meta_needs_run
        and not needs.need_any()
        and not scoring_needs_run
        and not gc_needs_run
    )

    stats_only_ok = False
    if need_any and not scoring_only and not needs.need_coords and (needs.need_exp_stats or needs.need_train_test_stats):
        exp_st, num_at, num_stru = _try_load_stats_only(
            base_output_dir, data_path, run_entries, training_splits,
            max_train=plot_cfg["max_train"], max_test=plot_cfg["max_test"],
        )
        if exp_st is not None:
            stats_only_ok = True
            phase_start = time.time()
            _log("Loading experimental statistics from cache (no coordinates).", since_start=time.time() - pipeline_start, style="info")
            coords_np = None
            coords = None
            device = utils.get_device()
            num_atoms = num_at
            num_structures = num_stru
            exp_stats = exp_st
            _log(f"Reused stats for {num_structures} structures, {num_atoms} atoms.", since_start=time.time() - pipeline_start, style="success")
            _log("Data ready (stats only).", since_start=time.time() - pipeline_start, since_phase=time.time() - phase_start, style="success")

    if need_any and not stats_only_ok and not scoring_only and not meta_only:
        phase_start = time.time()
        _log("Loading data...", since_start=time.time() - pipeline_start, style="info")
        coords_np = utils.load_data(data_path, max_data=None, seed=0)
        coords = torch.tensor(coords_np, dtype=torch.float32)
        device = utils.get_device()
        coords = coords.to(device)
        num_atoms = coords.size(1)
        num_structures = len(coords_np)
        utils.validate_dataset_for_pipeline(num_structures, training_splits[0])
        _log(f"Loaded {num_structures} structures, {num_atoms} atoms.", since_start=time.time() - pipeline_start, style="success")
        exp_stats = None
        if needs.need_exp_stats:
            cache_meta_path = os.path.join(_exp_stats_cache_dir(base_output_dir), EXP_STATS_META)
            exp_stats = _load_exp_stats_cache(base_output_dir, data_path, num_structures, num_atoms)
            if exp_stats is not None:
                _log(f"Reused experimental statistics from cache ({EXP_STATS_CACHE_DIR}/).", since_start=time.time() - pipeline_start, style="skip")
            elif os.path.isfile(cache_meta_path):
                with open(cache_meta_path) as f:
                    cached = json.load(f)
                if (
                    cached.get("data_path") == os.path.abspath(data_path)
                    and cached.get("num_structures") == num_structures
                    and cached.get("num_atoms") == num_atoms
                ):
                    _log("Experimental statistics cache invalid or corrupted, recomputing.", since_start=time.time() - pipeline_start, style="info")
                    data_cfg = cfg["data"]
                    exp_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, min(num_atoms - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"])
                    _save_exp_stats_cache(base_output_dir, data_path, num_structures, num_atoms, exp_stats)
                    _log(f"Cached experimental statistics to {EXP_STATS_CACHE_DIR}/.", since_start=time.time() - pipeline_start, style="success")
                else:
                    raise RuntimeError(
                        f"Experimental statistics cache exists for a different dataset. "
                        f"Cached: data_path={cached.get('data_path')!r} num_structures={cached.get('num_structures')} num_atoms={cached.get('num_atoms')}. "
                        f"Current: data_path={os.path.abspath(data_path)!r} num_structures={num_structures} num_atoms={num_atoms}. "
                        f"Use a different output_dir for this dataset, or remove {_exp_stats_cache_dir(base_output_dir)!r} to start fresh."
                    )
            elif _base_has_any_seed_pipeline_content(base_output_dir, run_entries):
                raise RuntimeError(
                    f"Experimental statistics cache is missing but base_output_dir already has pipeline content ({base_output_dir!r}). "
                    f"This would allow the dataset to be replaced midway. Use a different output_dir for this dataset, "
                    f"or restore the cache (e.g. from backup) before resuming."
                )
            else:
                _log("Computing experimental statistics (no valid cache for this dataset).", since_start=time.time() - pipeline_start, style="info")
                data_cfg = cfg["data"]
                exp_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, min(num_atoms - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"])
                _save_exp_stats_cache(base_output_dir, data_path, num_structures, num_atoms, exp_stats)
                _log(f"Cached experimental statistics to {EXP_STATS_CACHE_DIR}/.", since_start=time.time() - pipeline_start, style="success")
        _log("Data ready.", since_start=time.time() - pipeline_start, since_phase=time.time() - phase_start, style="success")

    if not need_any:
        coords_np = coords = device = num_atoms = num_structures = exp_stats = None
        if data_path and (do_plot or do_rmsd):
            _log("Skipping data load (all runs complete and all plot/analysis outputs present).", since_start=time.time() - pipeline_start, style="skip")
    elif scoring_only:
        coords_np = coords = device = num_atoms = num_structures = exp_stats = None
        _log("Scoring only: skipping data load (reading NPZ from existing run outputs).", since_start=time.time() - pipeline_start, style="skip")
    elif meta_only:
        coords_np = coords = device = num_atoms = num_structures = exp_stats = None
        _log("Meta-analysis only: skipping data load (reading NPZ from existing run outputs).", since_start=time.time() - pipeline_start, style="skip")

    vis_cfg = cfg["training_visualization"]
    vis_enabled = vis_cfg["enabled"]
    make_dm_hook = make_eu_hook = assemble_video_fn = None
    if vis_enabled:
        from src.training_visualization import (
            make_distmap_epoch_hook,
            make_euclideanizer_epoch_hook,
            assemble_video,
        )
        make_dm_hook = make_distmap_epoch_hook
        make_eu_hook = make_euclideanizer_epoch_hook
        assemble_video_fn = assemble_video

    stats_only = coords is None and exp_stats is not None
    sample_variances = get_sample_variances(cfg) if do_plot and (coords is not None or stats_only) else []
    gen_num_samples = cfg["plotting"]["num_samples"] if do_plot and (coords is not None or stats_only) else 0
    gen_decode_batch_size = plot_cfg["gen_decode_batch_size"]
    need_plot_or_rmsd = (
        (do_plot and (
            (coords is not None and (
                exp_stats is not None
                or (do_recon_plot and not do_bond_rg_scaling and not do_avg_gen and not do_bond_length_by_genomic_distance)
                or do_bond_rg_scaling
                or do_bond_length_by_genomic_distance
            ))
            or (coords is None and exp_stats is not None)
        ))
        or (do_rmsd and coords is not None)
        or (do_q and coords is not None)
        or (do_q_recon_cfg and coords is not None)
        or (do_gc_rmsd and coords is not None)
        or (do_gc_q and coords is not None)
    )
    save_structures_gro_plot = plot_cfg["save_structures_gro"]
    analysis_save_data = analysis_cfg["rmsd_gen"]["save_data"] or scoring_enabled  # effective: save when scoring needs NPZ
    analysis_save_structures_gro = analysis_cfg["rmsd_gen"]["save_structures_gro"]

    tasks = [(s, t, md, g) for (s, t, md) in run_entries for g in range(len(dm_groups))]
    n_gpus = utils.get_available_cuda_count()
    if args.gpus is not None:
        n_gpus = min(n_gpus, args.gpus)
    use_multi_gpu = (n_gpus >= 2) and not getattr(args, "no_multi_gpu", False) and (coords is not None)
    if use_multi_gpu:
        n_workers = min(n_gpus, len(tasks))
        _log(f"Detected {n_gpus} GPU(s). Using multi-GPU with {n_workers} worker(s).", since_start=time.time() - pipeline_start, style="info")
    else:
        _log(f"Detected {n_gpus} GPU(s). Using single process.", since_start=time.time() - pipeline_start, style="info")

    # Multi-GPU: ensure per-seed train/test stats and clustering feats caches exist, then free main-process data so only workers hold copies (reduces memory use).
    _need_precompute_caches = (
        do_plot or do_rmsd or do_q or do_q_recon_cfg
        or do_coord_clustering_gen or do_coord_clustering_recon_cfg
        or do_distmap_clustering_gen or do_distmap_clustering_recon_cfg
    )
    if use_multi_gpu and data_path and coords is not None and _need_precompute_caches:
        _pre_t0 = time.time() - pipeline_start
        n_pre_entries = len(run_entries)
        _log(
            f"Multi-GPU precompute: {n_pre_entries} run entries — plot train/test experimental statistics, then seed-level analysis caches.",
            since_start=_pre_t0,
            style="info",
        )
        plot_mt = plot_cfg["max_train"]
        plot_mc = plot_cfg["max_test"]
        for plot_i, entry in enumerate(run_entries, start=1):
            seed, training_split, max_data = _entry_seed_split_max(entry)
            output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
            run_label = os.path.basename(output_dir)
            _ensure_per_seed_pipeline_config(
                need_train=need_train,
                output_dir=output_dir,
                cfg=cfg,
                seed=seed,
                training_split=training_split,
                max_data=max_data,
            )
            coords_np_run = _apply_max_data_subset(coords_np, max_data, seed)
            num_structures_run = len(coords_np_run)
            num_atoms_run = int(coords_np_run.shape[1])
            if _exp_stats_split_cache_meta_files_ok(
                output_dir, data_path, num_structures_run, num_atoms_run, seed, training_split,
                max_train=plot_mt, max_test=plot_mc,
            ):
                _log(
                    f"{_plot_exp_stats_precompute_prefix(run_label, plot_i, n_pre_entries)}: using existing split cache.",
                    since_start=time.time() - pipeline_start,
                    style="skip",
                )
                continue
            coords_run = torch.tensor(coords_np_run, dtype=torch.float32).to(device)
            train_ds, test_ds = utils.get_train_test_split(coords_run, training_split, seed)
            train_indices = np.array(train_ds.indices)
            test_indices = np.array(test_ds.indices)
            if plot_mt is not None:
                train_indices = train_indices[: plot_mt]
            if plot_mc is not None:
                test_indices = test_indices[: plot_mc]
            data_cfg = cfg["data"]
            if exp_stats is not None and "exp_distmaps" in exp_stats:
                how = "slice global experimental distance maps (derive train/test stats)"
                _log(
                    f"{_plot_exp_stats_precompute_prefix(run_label, plot_i, n_pre_entries)}: computing ({how}).",
                    since_start=time.time() - pipeline_start,
                    style="info",
                )
                subset_idx = _max_data_indices(np.asarray(exp_stats["exp_distmaps"]).shape[0], max_data, seed)
                train_stats = _derive_stats_from_global_exp(
                    exp_stats, subset_idx[train_indices],
                    max_sep=min(num_atoms_run - 1, 999),
                    avg_map_sample=data_cfg["exp_stats_avg_map_sample"],
                )
                test_stats = _derive_stats_from_global_exp(
                    exp_stats, subset_idx[test_indices],
                    max_sep=min(num_atoms_run - 1, 999),
                    avg_map_sample=data_cfg["exp_stats_avg_map_sample"],
                )
            else:
                how = (
                    "full compute_exp_statistics on train/test indices (global exp stats not loaded or missing exp_distmaps)"
                )
                _log(
                    f"{_plot_exp_stats_precompute_prefix(run_label, plot_i, n_pre_entries)}: computing ({how}).",
                    since_start=time.time() - pipeline_start,
                    style="info",
                )
                train_stats = compute_exp_statistics(coords_np_run, device, utils.get_distmaps, min(num_atoms_run - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=train_indices)
                test_stats = compute_exp_statistics(coords_np_run, device, utils.get_distmaps, min(num_atoms_run - 1, 999), data_cfg["exp_stats_chunk_size"], data_cfg["exp_stats_avg_map_sample"], indices=test_indices)
            _save_exp_stats_split_cache(
                output_dir, data_path, num_structures_run, num_atoms_run, seed, training_split,
                train_stats, test_stats,
                max_train=plot_mt, max_test=plot_mc,
            )
            _log(
                f"{_plot_exp_stats_precompute_prefix(run_label, plot_i, n_pre_entries)}: saved split cache under {EXP_STATS_CACHE_DIR}/.",
                since_start=time.time() - pipeline_start,
                style="success",
            )
        # Precompute all seed-level analysis caches (RMSD, Q, coord_clustering, distmap_clustering) so workers only read (avoid concurrent write corruption).
        for spec in ANALYSIS_METRICS:
            do_gen = analysis_cfg[spec.gen_key]["enabled"]
            do_recon = analysis_cfg[spec.recon_key]["enabled"]
            if not do_gen and not do_recon:
                continue
            _ref_mt = analysis_cfg[f"{spec.id}_max_train"]
            _ref_mc = analysis_cfg[f"{spec.id}_max_test"]
            if spec.requires_reference_bounds and (_ref_mt is None or _ref_mc is None):
                continue
            n_analysis_skip = 0
            n_analysis_comp = 0
            for entry_i, entry in enumerate(run_entries, start=1):
                seed, training_split, max_data = _entry_seed_split_max(entry)
                output_dir = os.path.join(base_output_dir, _seed_split_dir_name(seed, training_split, max_data, training_splits, None))
                run_label = os.path.basename(output_dir)
                _ensure_per_seed_pipeline_config(
                    need_train=need_train,
                    output_dir=output_dir,
                    cfg=cfg,
                    seed=seed,
                    training_split=training_split,
                    max_data=max_data,
                )
                coords_np_run = _apply_max_data_subset(coords_np, max_data, seed)
                coords_run = torch.tensor(coords_np_run, dtype=torch.float32).to(device)
                cache_path = os.path.join(output_dir, EXP_STATS_CACHE_DIR, spec.cache_filename(analysis_cfg, _ref_mt, _ref_mc))
                if os.path.isfile(cache_path):
                    n_analysis_skip += 1
                    continue
                cache_name = os.path.basename(cache_path)
                _log(
                    f"Precompute analysis [{spec.id}] [{entry_i}/{n_pre_entries}] {run_label}: computing {cache_name} (test→train / feats).",
                    since_start=time.time() - pipeline_start,
                    style="info",
                )
                spec.get_or_compute_test_to_train(
                    cache_path, coords_np_run, coords_run, training_split, seed, base_output_dir,
                    **spec.kwargs_for_cache(analysis_cfg, _ref_mt, _ref_mc),
                )
                _log(
                    f"Precompute analysis [{spec.id}] [{entry_i}/{n_pre_entries}] {run_label}: saved {cache_name}.",
                    since_start=time.time() - pipeline_start,
                    style="success",
                )
                n_analysis_comp += 1
            _log(
                f"Precompute analysis [{spec.id}]: {n_analysis_comp} computed, {n_analysis_skip} skipped (cache already present).",
                since_start=time.time() - pipeline_start,
                style="info",
            )
        _log("Freed main-process data before spawning workers (multi-GPU).", since_start=time.time() - pipeline_start, style="info")
        coords_np = coords = exp_stats = None
        gc.collect()

    if scoring_only:
        _log("Running scoring for all runs (no data load).", since_start=time.time() - pipeline_start, style="info")
        for run_dir_eu, seed_dir in _iter_euclideanizer_runs(base_output_dir):
            _run_scoring_for_run(run_dir_eu, seed_dir, cfg, base_output_dir, pipeline_start)
            _post_scoring_npz_cleanup(
                run_dir_eu,
                cfg,
                defer_sufficiency_inputs=bool(meta_suff_enabled),
            )
    elif not use_multi_gpu:
        n_run_entries = len(run_entries)
        if data_path and coords is not None and _need_precompute_caches:
            _log(
                f"Single-GPU: {n_run_entries} run entr{'y' if n_run_entries == 1 else 'ies'} — plot train/test experimental statistics per entry; seed-level analysis caches when analysis runs.",
                since_start=time.time() - pipeline_start,
                style="info",
            )
        for run_i, entry in enumerate(run_entries, start=1):
            seed, training_split, max_data = _entry_seed_split_max(entry)
            _run_one_seed(
            seed=seed,
            max_data=max_data,
            cfg=cfg,
            base_output_dir=base_output_dir,
            dm_groups=dm_groups,
            eu_groups=eu_groups,
            dm_configs=dm_configs,
            eu_configs=eu_configs,
            coords=coords,
            coords_np=coords_np,
            device=device,
            num_atoms=num_atoms,
            num_structures=num_structures,
            exp_stats=exp_stats,
            data_path=data_path,
            need_train=need_train,
            pipeline_start=pipeline_start,
            training_split=training_split,
            training_splits=training_splits,
            do_plot=do_plot,
            do_recon_plot=do_recon_plot,
            do_bond_rg_scaling=do_bond_rg_scaling,
            do_avg_gen=do_avg_gen,
            do_bond_length_by_genomic_distance=do_bond_length_by_genomic_distance,
            do_rmsd=do_rmsd,
            resume=resume,
            sample_variances=sample_variances,
            gen_num_samples=gen_num_samples,
            gen_decode_batch_size=gen_decode_batch_size,
            need_plot_or_rmsd=need_plot_or_rmsd,
            save_structures_gro_plot=save_structures_gro_plot,
            analysis_save_data=analysis_save_data,
            analysis_save_structures_gro=analysis_save_structures_gro,
            plot_dpi=plot_dpi,
            save_pdf=save_pdf,
            save_plot_data=save_plot_data,
            num_recon_samples=num_recon_samples,
            analysis_cfg=analysis_cfg,
            variance_list=variance_list,
            num_samples_list=num_samples_list,
            max_recon_train_list=max_recon_train_list,
            max_recon_test_list=max_recon_test_list,
            vis_enabled=vis_enabled,
            vis_cfg=vis_cfg,
            plot_cfg=plot_cfg,
            plot_max_train=plot_cfg["max_train"],
            plot_max_test=plot_cfg["max_test"],
            do_q=do_q,
            do_q_recon=do_q_recon_cfg,
            q_max_train=q_max_train,
            q_max_test=q_max_test,
            q_num_samples_list=q_num_samples_list,
            q_variance_list=q_variance_list,
            q_delta=q_delta,
            q_max_recon_train_list=q_max_recon_train_list,
            q_max_recon_test_list=q_max_recon_test_list,
            q_recon_delta=q_recon_delta,
            make_distmap_epoch_hook=make_dm_hook,
            make_euclideanizer_epoch_hook=make_eu_hook,
            assemble_video_fn=assemble_video_fn,
            run_entry_idx=run_i,
            run_entry_n=n_run_entries,
        )
    elif use_multi_gpu:
        _run_multi_gpu_tasks(
            tasks=tasks,
            n_gpus=n_gpus,
            cfg=cfg,
            base_output_dir=base_output_dir,
            dm_groups=dm_groups,
            eu_groups=eu_groups,
            dm_configs=dm_configs,
            eu_configs=eu_configs,
            coords=coords,
            coords_np=coords_np,
            device=device,
            num_atoms=num_atoms,
            num_structures=num_structures,
            exp_stats=exp_stats,
            data_path=data_path,
            need_train=need_train,
            pipeline_start=pipeline_start,
            training_splits=training_splits,
            do_plot=do_plot,
            do_recon_plot=do_recon_plot,
            do_bond_rg_scaling=do_bond_rg_scaling,
            do_avg_gen=do_avg_gen,
            do_bond_length_by_genomic_distance=do_bond_length_by_genomic_distance,
            do_rmsd=do_rmsd,
            resume=resume,
            sample_variances=sample_variances,
            gen_num_samples=gen_num_samples,
            gen_decode_batch_size=gen_decode_batch_size,
            need_plot_or_rmsd=need_plot_or_rmsd,
            save_structures_gro_plot=save_structures_gro_plot,
            analysis_save_data=analysis_save_data,
            analysis_save_structures_gro=analysis_save_structures_gro,
            plot_dpi=plot_dpi,
            save_pdf=save_pdf,
            save_plot_data=save_plot_data,
            num_recon_samples=num_recon_samples,
            analysis_cfg=analysis_cfg,
            variance_list=variance_list,
            num_samples_list=num_samples_list,
            max_recon_train_list=max_recon_train_list,
            max_recon_test_list=max_recon_test_list,
            vis_enabled=vis_enabled,
            vis_cfg=vis_cfg,
            plot_cfg=plot_cfg,
            do_q=do_q,
            do_q_recon=do_q_recon_cfg,
            q_max_train=q_max_train,
            q_max_test=q_max_test,
            q_num_samples_list=q_num_samples_list,
            q_variance_list=q_variance_list,
            q_delta=q_delta,
            q_max_recon_train_list=q_max_recon_train_list,
            q_max_recon_test_list=q_max_recon_test_list,
            q_recon_delta=q_recon_delta,
            make_distmap_epoch_hook=make_dm_hook,
            make_euclideanizer_epoch_hook=make_eu_hook,
            assemble_video_fn=assemble_video_fn,
        )

    should_run_meta = meta_suff_enabled and (meta_needs_run or (need_any and not meta_only))
    if should_run_meta:
        _run_sufficiency_meta_analysis(base_output_dir, cfg, pipeline_start)
        if scoring_enabled:
            _finalize_deferred_npz_cleanup(base_output_dir, cfg)

    if do_dashboard:
        from src.dashboard import build_dashboard
        _log("Building dashboard...", since_start=time.time() - pipeline_start, style="info")
        dashboard_dir = build_dashboard(base_output_dir)
        if dashboard_dir:
            _log(f"Dashboard saved to {dashboard_dir}", since_start=time.time() - pipeline_start, style="success")
        else:
            _log("Dashboard skipped (no runs with outputs found).", since_start=time.time() - pipeline_start, style="skip")

    total_min = (time.time() - pipeline_start) / 60
    _log("Pipeline complete.", since_start=time.time() - pipeline_start, style="success")
    _log_raw("=" * 60, style="success")
    out_msg = f"Total time: {total_min:.1f}m  |  Output: {base_output_dir}"
    if len(seeds) > 1:
        out_msg += f"  (seeds: {', '.join(map(str, seeds))})"
    _log_raw(out_msg, style="success")
    log_path = os.path.join(base_output_dir, PIPELINE_LOG_FILENAME) if base_output_dir else PIPELINE_LOG_FILENAME
    _log_raw(f"Log file: {log_path}", style="success")
    _log_raw("=" * 60, style="success")
    if _pipeline_real_stdout is not None:
        sys.stdout = _pipeline_real_stdout
        _pipeline_real_stdout = None
    if _pipeline_real_stderr is not None:
        sys.stderr = _pipeline_real_stderr
        _pipeline_real_stderr = None
    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        if _pipeline_real_stdout is not None:
            sys.stdout = _pipeline_real_stdout
            _pipeline_real_stdout = None
        if _pipeline_real_stderr is not None:
            sys.stderr = _pipeline_real_stderr
            _pipeline_real_stderr = None
        if _LOG_FILE is not None:
            _LOG_FILE.write("\n")
            _LOG_FILE.write("=" * 60 + "\n")
            _LOG_FILE.write("PIPELINE ERROR\n")
            _LOG_FILE.write("=" * 60 + "\n")
            _LOG_FILE.write(tb)
            if not tb.endswith("\n"):
                _LOG_FILE.write("\n")
            _LOG_FILE.flush()
        else:
            _append_preinit_fatal_traceback(tb)
        raise
