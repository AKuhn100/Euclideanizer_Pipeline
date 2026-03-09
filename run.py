#!/usr/bin/env python3
"""
Global orchestration for DistMap + Euclideanizer pipeline.

  python run.py --config path/to/config.yaml [--data /path/to/data.gro] [--no-plots]
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
    save_run_config,
    load_run_config,
    load_pipeline_config,
    save_pipeline_config,
    configs_match_exactly,
    configs_match_sections,
    config_diff,
    run_config_section_matches,
    pipeline_config_path,
    TRAINING_CRITICAL_KEYS,
)
from src.metrics import compute_exp_statistics
from src.train_distmap import train_distmap
from src.train_euclideanizer import train_euclideanizer
from src.plotting import (
    plot_distmap_reconstruction,
    plot_euclideanizer_reconstruction,
    plot_recon_statistics,
    plot_gen_analysis,
)
from src.distmap.model import ChromVAE_Conv
from src.distmap.sample import generate_samples as dm_generate_samples
from src.euclideanizer.model import Euclideanizer, load_frozen_vae
from src.analysis_metrics import ANALYSIS_METRICS
from src.min_rmsd import plot_latent_distribution
from src.gro_io import write_structures_gro

# Log file in output root; also mirrored to stdout (set in main).
_LOG_FILE = None
# Lock for serializing log writes in the main process; unused in spawned workers (each opens its own log handle).
_LOG_LOCK = None
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
    if "assembling" in low or "epoch" in low or "loaded" in low or "generated" in low or "min-rmsd" in low or "min rmsd" in low:
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


class _StyledStdout:
    """Wrap stdout so submodule print() output gets a default color (only when TTY). Pass-through if already contains ANSI."""

    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s: str) -> None:
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


def _delete_plotting_and_analysis_outputs(base_output_dir: str, seeds: list) -> None:
    """Remove all plots/, analysis/, and dashboard under base_output_dir for the given seeds."""
    import re
    for seed in seeds:
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
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


def _has_any_plotting_output(base_output_dir: str, seeds: list) -> bool:
    """True if any plots/ or dashboard exists under base_output_dir for the given seeds."""
    for seed in seeds:
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
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


def _has_any_analysis_output(base_output_dir: str, seeds: list, component: str) -> bool:
    """True if any analysis output for the given component exists. component: 'min_rmsd_gen', 'min_rmsd_recon', 'q_gen', or 'q_recon'."""
    if component in ("min_rmsd_gen", "min_rmsd_recon"):
        subdir = "gen" if component == "min_rmsd_gen" else "recon"
        target = os.path.join("analysis", "min_rmsd", subdir)
    elif component in ("q_gen", "q_recon"):
        subdir = "gen" if component == "q_gen" else "recon"
        target = os.path.join("analysis", "q", subdir)
    else:
        return False
    for seed in seeds:
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
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


def _delete_plotting_outputs_only(base_output_dir: str, seeds: list) -> None:
    """Remove all plots/ and dashboard under base_output_dir for the given seeds (no analysis dirs)."""
    for seed in seeds:
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
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
    dashboard_dir = os.path.join(base_output_dir, "dashboard")
    if os.path.isdir(dashboard_dir):
        shutil.rmtree(dashboard_dir, ignore_errors=True)


def _delete_analysis_outputs_for_component(base_output_dir: str, seeds: list, component: str) -> None:
    """Remove only the analysis subdir for this component (e.g. analysis/q/gen). component: 'min_rmsd_gen', 'min_rmsd_recon', 'q_gen', or 'q_recon'."""
    if component in ("min_rmsd_gen", "min_rmsd_recon"):
        subdir = "gen" if component == "min_rmsd_gen" else "recon"
        target = os.path.join("analysis", "min_rmsd", subdir)
    elif component in ("q_gen", "q_recon"):
        subdir = "gen" if component == "q_gen" else "recon"
        target = os.path.join("analysis", "q", subdir)
    else:
        return
    for seed in seeds:
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
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
    """Write a concise log line to stdout (styled when TTY) and to pipeline.log (plain). Flushes file after each write."""
    if since_start is not None:
        prefix = f"[+{since_start / 60:5.1f}m]"
        suffix = f"  (phase {since_phase / 60:.1f}m)" if since_phase is not None else ""
        line = f"{prefix} {msg}{suffix}"
    else:
        line = "        " + msg
    print(_style(line, style))
    if _LOG_FILE is not None:
        if _LOG_LOCK is not None:
            _LOG_LOCK.acquire()
        try:
            _LOG_FILE.write(line + "\n")
            _LOG_FILE.flush()
        finally:
            if _LOG_LOCK is not None:
                _LOG_LOCK.release()


def _log_raw(line: str, style: str | None = None) -> None:
    """Write a raw line (e.g. separator) to stdout (styled when TTY) and log file (plain)."""
    print(_style(line, style))
    if _LOG_FILE is not None:
        if _LOG_LOCK is not None:
            _LOG_LOCK.acquire()
        try:
            _LOG_FILE.write(line + "\n")
            _LOG_FILE.flush()
        finally:
            if _LOG_LOCK is not None:
                _LOG_LOCK.release()


# Output layout: run_root/model/ (model.pt or euclideanizer.pt), run_root/plots/<type>/ (PNG and optional data/*.npz)
EXP_STATS_CACHE_DIR = "experimental_statistics"
EXP_STATS_META = "meta.json"
EXP_STATS_NPZ = "exp_stats.npz"
EXP_STATS_TRAIN_NPZ = "exp_stats_train.npz"
EXP_STATS_TEST_NPZ = "exp_stats_test.npz"
EXP_STATS_SPLIT_META = "split_meta.json"


def _exp_stats_cache_dir(output_dir: str) -> str:
    return os.path.join(output_dir, EXP_STATS_CACHE_DIR)


def _output_dir_has_pipeline_content(output_dir: str) -> bool:
    """True if output_dir already contains distmap or euclideanizer runs (resume scenario)."""
    return os.path.isdir(os.path.join(output_dir, "distmap"))


def _base_has_any_seed_pipeline_content(base_output_dir: str, seeds: list) -> bool:
    """True if base_output_dir/seed_<s>/distmap exists for any seed (multi-seed resume scenario)."""
    for s in seeds:
        if os.path.isdir(os.path.join(base_output_dir, f"seed_{s}", "distmap")):
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
            return {k: data[k] for k in data.files}
    except (json.JSONDecodeError, OSError, KeyError, zlib.error, zipfile.BadZipFile):
        return None


def _try_load_stats_only(
    base_output_dir: str,
    data_path: str,
    seeds: list,
    training_split: float,
) -> tuple[dict | None, int | None, int | None]:
    """
    Load exp_stats and train/test caches without loading coords.
    Requires base cache meta to match data_path and all seeds to have valid train/test cache.
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
            exp_stats = {k: data[k] for k in data.files}
    except (OSError, zlib.error, zipfile.BadZipFile):
        return None, None, None
    for seed in seeds:
        output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        train_s, test_s = _load_exp_stats_split_cache(
            output_dir, data_path, num_structures, num_atoms, seed, training_split
        )
        if train_s is None or test_s is None:
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
    np.savez_compressed(npz_path, **exp_stats)


def _load_exp_stats_split_cache(
    output_dir: str,
    data_path: str,
    num_structures: int,
    num_atoms: int,
    split_seed: int,
    training_split: float,
) -> tuple[dict | None, dict | None]:
    """Load train and test exp stats from seed output_dir if cache valid. Returns (train_stats, test_stats) or (None, None)."""
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
        with np.load(train_path, allow_pickle=False) as data:
            train_stats = {k: data[k] for k in data.files}
        with np.load(test_path, allow_pickle=False) as data:
            test_stats = {k: data[k] for k in data.files}
        return train_stats, test_stats
    except (json.JSONDecodeError, OSError, KeyError, zlib.error, zipfile.BadZipFile):
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
) -> None:
    """Write per-seed train/test experimental statistics to cache."""
    cache_dir = _exp_stats_cache_dir(output_dir)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, EXP_STATS_SPLIT_META)
    meta = {
        "data_path": os.path.abspath(data_path),
        "num_structures": num_structures,
        "num_atoms": num_atoms,
        "split_seed": split_seed,
        "training_split": training_split,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(os.path.join(cache_dir, EXP_STATS_TRAIN_NPZ), **train_stats)
    np.savez_compressed(os.path.join(cache_dir, EXP_STATS_TEST_NPZ), **test_stats)


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
    """True if run_dir has a completed run: last_epoch_trained == expected_epochs, section matches if given, best checkpoint exists. When multi_segment, last checkpoint required only if save_final_models_per_stretch (we don't require it when false since it is deleted after the next segment uses it)."""
    model_dir = os.path.join(run_dir, model_subdir)
    run_cfg = load_run_config(model_dir)
    if run_cfg is None:
        if _log_fail_reason:
            _log(f"{_log_fail_reason}: run_config not found or invalid at {model_dir}", since_start=None, style="skip")
        return False
    last_trained = run_cfg.get("last_epoch_trained")
    if last_trained != expected_epochs:
        if _log_fail_reason:
            _log(f"{_log_fail_reason}: last_epoch_trained={last_trained!r} (type {type(last_trained).__name__}) != expected_epochs={expected_epochs!r} (type {type(expected_epochs).__name__})", since_start=None, style="skip")
        return False
    if section_key is not None and expected_section is not None:
        if not run_config_section_matches(run_cfg, section_key, expected_section):
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
    """Determine how to run this DistMap segment: skip, from_scratch, resume_from_best, or resume_from_prev_last. Returns a dict with 'action' and, when relevant, resume_from_path, prev_run_dir, additional_epochs, best_epoch."""
    dm_path = _dm_path(run_dir_dm)
    run_label = os.path.basename(run_dir_dm)
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
    """Determine how to run this Euclideanizer segment: skip, from_scratch, resume_from_best, or resume_from_prev_last. Returns a dict with 'action' and, when relevant, resume_from_path, prev_run_dir, additional_epochs, best_epoch."""
    eu_path = _eu_path(eu_run_dir)
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
}


def _plot_path(run_root: str, plot_type: str, **format_kw: str) -> str:
    subdir, pattern = PLOT_TYPES[plot_type]
    filename = pattern.format(**format_kw) if format_kw else pattern
    return os.path.join(run_root, "plots", subdir, filename)


def _analysis_path(run_root: str, analysis_type: str, filename: str) -> str:
    return os.path.join(run_root, "analysis", analysis_type, filename)


def _distmap_plotting_all_present(
    run_dir_dm: str,
    resume: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
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
    return True


def _euclideanizer_plotting_all_present(
    run_dir_eu: str,
    resume: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
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
    return True


def _euclideanizer_analysis_all_present(
    run_dir_eu: str,
    resume: bool,
    do_min_rmsd: bool,
    variance_list: list,
    num_samples_list: list,
    do_min_rmsd_recon: bool = False,
    visualize_latent: bool = False,
    max_recon_train_list: list | None = None,
    max_recon_test_list: list | None = None,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_variance_list: list | None = None,
    q_num_samples_list: list | None = None,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    q_visualize_latent: bool = False,
) -> bool:
    """True if resume and all min_rmsd and Q (gen + recon + latent) analysis outputs we would generate already exist."""
    if not resume:
        return True
    if do_min_rmsd:
        for var in variance_list:
            variance_suffix = f"_var{var}" if len(variance_list) > 1 else ""
            for n in num_samples_list:
                run_name = (str(n) + variance_suffix) if variance_suffix else (str(n) if len(num_samples_list) > 1 else "default")
                fig_path = _analysis_path(run_dir_eu, "min_rmsd", f"gen/{run_name}/min_rmsd_distributions.png")
                if not os.path.isfile(fig_path):
                    return False
    if do_min_rmsd_recon and max_recon_train_list is not None and max_recon_test_list is not None:
        n_recon = len(max_recon_train_list) * len(max_recon_test_list)
        if n_recon == 1:
            recon_fig = _analysis_path(run_dir_eu, "min_rmsd", "recon/min_rmsd_distributions.png")
            if not os.path.isfile(recon_fig):
                return False
            if visualize_latent:
                latent_fig = _analysis_path(run_dir_eu, "min_rmsd", "recon/latent_distribution.png")
                if not os.path.isfile(latent_fig):
                    return False
        else:
            for max_train in max_recon_train_list:
                for max_test in max_recon_test_list:
                    subdir = f"train{max_train}_test{max_test}"
                    recon_fig = _analysis_path(run_dir_eu, "min_rmsd", f"recon/{subdir}/min_rmsd_distributions.png")
                    if not os.path.isfile(recon_fig):
                        return False
                    if visualize_latent:
                        latent_fig = _analysis_path(run_dir_eu, "min_rmsd", f"recon/{subdir}/latent_distribution.png")
                        if not os.path.isfile(latent_fig):
                            return False
    if do_q and q_variance_list is not None and q_num_samples_list is not None:
        for var in q_variance_list:
            variance_suffix = f"_var{var}" if len(q_variance_list) > 1 else ""
            for n in q_num_samples_list:
                run_name = (str(n) + variance_suffix) if variance_suffix else (str(n) if len(q_num_samples_list) > 1 else "default")
                fig_path = _analysis_path(run_dir_eu, "q", f"gen/{run_name}/q_distributions.png")
                if not os.path.isfile(fig_path):
                    return False
    if do_q_recon and q_max_recon_train_list is not None and q_max_recon_test_list is not None:
        n_recon = len(q_max_recon_train_list) * len(q_max_recon_test_list)
        if n_recon == 1:
            recon_fig = _analysis_path(run_dir_eu, "q", "recon/q_distributions.png")
            if not os.path.isfile(recon_fig):
                return False
            if q_visualize_latent:
                latent_fig = _analysis_path(run_dir_eu, "q", "recon/latent_distribution.png")
                if not os.path.isfile(latent_fig):
                    return False
        else:
            for max_train in q_max_recon_train_list:
                for max_test in q_max_recon_test_list:
                    subdir = f"train{max_train}_test{max_test}"
                    recon_fig = _analysis_path(run_dir_eu, "q", f"recon/{subdir}/q_distributions.png")
                    if not os.path.isfile(recon_fig):
                        return False
                    if q_visualize_latent:
                        latent_fig = _analysis_path(run_dir_eu, "q", f"recon/{subdir}/latent_distribution.png")
                        if not os.path.isfile(latent_fig):
                            return False
    return True


def _pipeline_need_data(
    base_output_dir: str,
    seeds: list,
    dm_groups: list,
    eu_groups: list,
    resume: bool,
    do_plot: bool,
    do_min_rmsd: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    plot_variances: list,
    variance_list: list,
    num_samples_list: list,
    do_min_rmsd_recon: bool = False,
    visualize_latent: bool = False,
    max_recon_train_list: list | None = None,
    max_recon_test_list: list | None = None,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_variance_list: list | None = None,
    q_num_samples_list: list | None = None,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    q_visualize_latent: bool = False,
) -> bool:
    """True if any run is incomplete or any plot/analysis output is missing (so we must load something)."""
    return _pipeline_data_needs(
        base_output_dir, seeds, dm_groups, eu_groups,
        resume, do_plot, do_min_rmsd, do_recon_plot, do_bond_rg_scaling, do_avg_gen,
        plot_variances, variance_list, num_samples_list,
        do_min_rmsd_recon=do_min_rmsd_recon, visualize_latent=visualize_latent,
        max_recon_train_list=max_recon_train_list, max_recon_test_list=max_recon_test_list,
        do_q=do_q, do_q_recon=do_q_recon,
        q_variance_list=q_variance_list or [], q_num_samples_list=q_num_samples_list or [],
        q_max_recon_train_list=q_max_recon_train_list or [], q_max_recon_test_list=q_max_recon_test_list or [],
        q_visualize_latent=q_visualize_latent,
    ).need_any()


@dataclass(frozen=True)
class PipelineDataNeeds:
    """What the pipeline must load for resume: only coords, only stats from cache, or both."""

    need_coords: bool  # training, reconstruction, recon_statistics, min_rmsd, or video frames
    need_exp_stats: bool  # gen_variance (full exp_stats)
    need_train_test_stats: bool  # recon_statistics or gen_variance (train/test split stats)

    def need_any(self) -> bool:
        return self.need_coords or self.need_exp_stats or self.need_train_test_stats


def _pipeline_data_needs(
    base_output_dir: str,
    seeds: list,
    dm_groups: list,
    eu_groups: list,
    resume: bool,
    do_plot: bool,
    do_min_rmsd: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    plot_variances: list,
    variance_list: list,
    num_samples_list: list,
    do_min_rmsd_recon: bool = False,
    visualize_latent: bool = False,
    max_recon_train_list: list | None = None,
    max_recon_test_list: list | None = None,
    do_q: bool = False,
    do_q_recon: bool = False,
    q_variance_list: list | None = None,
    q_num_samples_list: list | None = None,
    q_max_recon_train_list: list | None = None,
    q_max_recon_test_list: list | None = None,
    q_visualize_latent: bool = False,
) -> PipelineDataNeeds:
    """
    Scan pipeline outputs and return which data is required.
    - need_coords: any run incomplete, or any reconstruction / recon_statistics / min_rmsd / q analysis missing.
    - need_exp_stats: any gen_variance plot missing.
    - need_train_test_stats: any recon_statistics or gen_variance missing.
    """
    need_coords = False
    need_exp_stats = False
    need_train_test_stats = False
    for seed in seeds:
        output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        if not os.path.isdir(output_dir):
            need_coords = True
            need_exp_stats = need_exp_stats or do_plot and do_avg_gen
            need_train_test_stats = need_train_test_stats or (do_plot and (do_bond_rg_scaling or do_avg_gen))
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
                        if (do_min_rmsd or do_min_rmsd_recon) and not _euclideanizer_analysis_all_present(
                            eu_run_dir, resume, do_min_rmsd, variance_list, num_samples_list,
                            do_min_rmsd_recon=do_min_rmsd_recon, visualize_latent=visualize_latent,
                            max_recon_train_list=max_recon_train_list, max_recon_test_list=max_recon_test_list,
                        ):
                            need_coords = True
                        if (do_q or do_q_recon) and not _euclideanizer_analysis_all_present(
                            eu_run_dir, resume, do_min_rmsd=False, variance_list=[], num_samples_list=[],
                            do_min_rmsd_recon=False, do_q=do_q, do_q_recon=do_q_recon,
                            q_variance_list=q_variance_list, q_num_samples_list=q_num_samples_list,
                            q_max_recon_train_list=q_max_recon_train_list, q_max_recon_test_list=q_max_recon_test_list,
                            q_visualize_latent=q_visualize_latent,
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
    p.add_argument("--euclideanizer.batch_size", type=int, nargs="*", default=None, dest="eu_batch_size")
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
    p.add_argument("--plotting.plot_dpi", type=int, default=None, dest="plot_dpi")
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
    if args.plot_dpi is not None:
        o.setdefault("plotting", {})["plot_dpi"] = args.plot_dpi
    return o


def _get_recon_dm_distmap(model, device, coords, dm_cfg, training_split, split_seed, utils_mod, use_train: bool = False):
    """Compute reconstruction distance maps for the DistMap on train or test split; returns (n_samples, N, N) numpy array."""
    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    subset_ds = train_ds if use_train else test_ds
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


def _get_recon_dm_euclideanizer(embed, frozen_vae, device, coords, eu_cfg, training_split, split_seed, utils_mod, use_train: bool = False):
    """Compute reconstruction distance maps for the Euclideanizer on train or test split; returns (n_samples, N, N) numpy array."""
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
            out.append(utils_mod.get_distmaps(coords_out).cpu().numpy())
    return np.concatenate(out, axis=0)


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


def _force_gpu_cleanup(device: torch.device) -> None:
    """Release unused GPU memory so the allocator and system see accurate free memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_one_distmap_group(
    seed: int,
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
    do_plot: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_min_rmsd: bool,
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
    q_visualize_latent: bool = False,
    make_distmap_epoch_hook=None,
    make_euclideanizer_epoch_hook=None,
    assemble_video_fn=None,
) -> None:
    """Run one (seed, DistMap group): that group's segments, plotting, and all Euclideanizer runs for that DistMap."""
    output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    split_seed = seed
    group = dm_groups[gidx]
    base_config = group["base_config"]
    checkpoints = group["checkpoints"]
    checkpoint_dirs = [os.path.join(output_dir, "distmap", str(ri)) for ri, _ in checkpoints]
    dm_max_epoch = max(ev for _, ev in checkpoints)
    prev_dm_path = None
    prev_dm_ev = None
    
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
                train_distmap(
                    dm_cfg, device, coords, run_dir_dm,
                    split_seed=split_seed, training_split=training_split,
                    epoch_callback=epoch_cb,
                    plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                    memory_efficient=dm_cfg["memory_efficient"],
                    is_last_segment=dm_last_segment,
                    display_root=base_output_dir,
                )
            elif dm_act["action"] == "resume_from_best":
                _log(f"DistMap run {ri} (seed {seed}): resuming from best (epoch {dm_act['best_epoch']}), training {dm_act['additional_epochs']} more → {ev} total...", since_start=time.time() - pipeline_start, style="info")
                epoch_cb = None
                if vis_enabled and make_distmap_epoch_hook is not None:
                    epoch_cb, _ = make_distmap_epoch_hook(
                        coords, dm_cfg, run_dir_dm, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=dm_act["best_epoch"], total_epochs_display=dm_max_epoch
                    )
                train_distmap(
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
                )
            else:
                _log(f"DistMap run {ri} (seed {seed}): resuming from run (epochs={prev_dm_ev}), training {dm_act['additional_epochs']} more → {ev} total...", since_start=time.time() - pipeline_start, style="info")
                epoch_cb = None
                if vis_enabled and make_distmap_epoch_hook is not None:
                    epoch_cb, _ = make_distmap_epoch_hook(
                        coords, dm_cfg, run_dir_dm, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=prev_dm_ev, total_epochs_display=dm_max_epoch
                    )
                train_distmap(
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
    
        if do_plot and exp_stats is not None and (coords is not None or (train_stats is not None and test_stats is not None)):
            if _distmap_plotting_all_present(
                run_dir_dm, resume, do_recon_plot, do_bond_rg_scaling, do_avg_gen, sample_variances
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
                if do_bond_rg_scaling and train_stats is not None and test_stats is not None and coords is not None:
                    for subset_name, use_train, stats in [("test", False, test_stats), ("train", True, train_stats)]:
                        p = _plot_path(run_dir_dm, "recon_statistics", subset=subset_name)
                        if not (resume and os.path.isfile(p)):
                            recon_dm = _get_recon_dm_distmap(model, device, coords, dm_cfg, training_split, split_seed, utils, use_train=use_train)
                            plot_recon_statistics(
                                recon_dm, stats, p,
                                label_recon="VAE Recon", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                subset_label=subset_name,
                                display_root=base_output_dir,
                            )
                        elif resume:
                            _log(f"  [skip] recon_statistics_{subset_name}", since_start=time.time() - pipeline_start, style="skip")
                if do_avg_gen and train_stats is not None and test_stats is not None:
                    for var in sample_variances:
                        p = _plot_path(run_dir_dm, "gen_variance", var=str(var))
                        if not (resume and os.path.isfile(p)):
                            gen_dm = _get_gen_dm_distmap(model, device, gen_num_samples, dm_cfg["latent_dim"], var, gen_decode_batch_size)
                            plot_gen_analysis(
                                exp_stats, train_stats, test_stats, gen_dm, p,
                                sample_variance=var, label_gen="VAE Gen", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                display_root=base_output_dir,
                            )
                        elif resume:
                            _log(f"  [skip] gen_variance_{var}", since_start=time.time() - pipeline_start, style="skip")
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
                        train_euclideanizer(
                            eu_cfg_seg, device, coords, dm_path, eu_run_dir,
                            split_seed=split_seed, training_split=training_split,
                            frozen_latent_dim=dm_cfg["latent_dim"],
                            epoch_callback=epoch_cb,
                            plot_loss=do_plot, plot_dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                            memory_efficient=eu_cfg_seg["memory_efficient"],
                            is_last_segment=eu_last_segment,
                            display_root=base_output_dir,
                        )
                    elif eu_act["action"] == "resume_from_best":
                        _log(f"Euclideanizer run {euri} (DistMap {ri}): resuming from best (epoch {eu_act['best_epoch']}), training {eu_act['additional_epochs']} more → {eu_ev} total...", since_start=time.time() - pipeline_start, style="info")
                        epoch_cb = None
                        if vis_enabled:
                            epoch_cb, _ = make_euclideanizer_epoch_hook(
                                coords, eu_cfg_seg, dm_path, dm_cfg["latent_dim"], eu_run_dir, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=eu_act["best_epoch"], total_epochs_display=eu_max_epoch
                            )
                        train_euclideanizer(
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
                        )
                    else:
                        _log(f"Euclideanizer run {euri} (DistMap {ri}): resuming from {prev_eu_ev} epochs, training {eu_act['additional_epochs']} more → {eu_ev} total...", since_start=time.time() - pipeline_start, style="info")
                        epoch_cb = None
                        if vis_enabled:
                            epoch_cb, _ = make_euclideanizer_epoch_hook(
                                coords, eu_cfg_seg, dm_path, dm_cfg["latent_dim"], eu_run_dir, device, utils, vis_cfg, split_seed=split_seed, training_split=training_split, epoch_start=prev_eu_ev, total_epochs_display=eu_max_epoch
                            )
                        train_euclideanizer(
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
                        run_dir_eu, resume, do_recon_plot, do_bond_rg_scaling, do_avg_gen, sample_variances
                    )
                    all_analysis = _euclideanizer_analysis_all_present(
                        run_dir_eu, resume, do_min_rmsd, variance_list, num_samples_list,
                        do_min_rmsd_recon=analysis_cfg["min_rmsd_recon"]["enabled"], visualize_latent=analysis_cfg["min_rmsd_recon"]["visualize_latent"],
                        max_recon_train_list=max_recon_train_list, max_recon_test_list=max_recon_test_list,
                        do_q=do_q, do_q_recon=do_q_recon,
                        q_variance_list=q_variance_list or [], q_num_samples_list=q_num_samples_list or [],
                        q_max_recon_train_list=q_max_recon_train_list or [], q_max_recon_test_list=q_max_recon_test_list or [],
                        q_visualize_latent=q_visualize_latent,
                    )
                    if resume and all_plots and all_analysis:
                        _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): [skip] plotting and analysis (all present)", since_start=time.time() - pipeline_start, style="skip")
                    else:
                        _force_gpu_cleanup(device)
                        phase_start_eu = time.time()
                        frozen_vae = load_frozen_vae(dm_path, num_atoms, dm_cfg["latent_dim"], device)
                        embed = Euclideanizer(num_atoms=num_atoms).to(device)
                        embed.load_state_dict(torch.load(eu_path, map_location=device))
    
                        if do_plot and exp_stats is not None and (coords is not None or (train_stats is not None and test_stats is not None)):
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
                            if do_bond_rg_scaling and train_stats is not None and test_stats is not None and coords is not None:
                                for subset_name, use_train, stats in [("test", False, test_stats), ("train", True, train_stats)]:
                                    p = _plot_path(run_dir_eu, "recon_statistics", subset=subset_name)
                                    if not (resume and os.path.isfile(p)):
                                        recon_dm = _get_recon_dm_euclideanizer(embed, frozen_vae, device, coords, eu_cfg, training_split, split_seed, utils, use_train=use_train)
                                        plot_recon_statistics(
                                            recon_dm, stats, p,
                                            label_recon="Eucl. Recon", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                            subset_label=subset_name,
                                            display_root=base_output_dir,
                                        )
                                    elif resume:
                                        _log(f"  [skip] recon_statistics_{subset_name}", since_start=time.time() - pipeline_start, style="skip")
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
                                        plot_gen_analysis(
                                            exp_stats, train_stats, test_stats, gen_dm, p,
                                            sample_variance=var, label_gen="Euclideanizer", dpi=plot_dpi, save_pdf=save_pdf, save_plot_data=save_plot_data,
                                            display_root=base_output_dir,
                                        )
                                    elif resume:
                                        _log(f"  [skip] gen_variance_{var}", since_start=time.time() - pipeline_start, style="skip")
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): plotting done in {(time.time() - plot_phase_start) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")
    
                        # Single analysis loop over registered metrics (order: min_rmsd, then q).
                        any_analysis = any(
                            analysis_cfg.get(spec.gen_key, {}).get("enabled", False) or analysis_cfg.get(spec.recon_key, {}).get("enabled", False)
                            for spec in ANALYSIS_METRICS
                        )
                        if any_analysis and coords is not None:
                            if not isinstance(seed_test_to_train_holder[0], dict):
                                seed_test_to_train_holder[0] = {}
                            _cache = seed_test_to_train_holder[0]
                            analysis_phase_start = time.time()
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): analysis (min-RMSD + Q)...", since_start=time.time() - pipeline_start, style="info")
                            for spec in ANALYSIS_METRICS:
                                do_gen = analysis_cfg.get(spec.gen_key, {}).get("enabled", False)
                                do_recon = analysis_cfg.get(spec.recon_key, {}).get("enabled", False)
                                if not (do_gen or do_recon):
                                    continue
                                gen_cfg = analysis_cfg.get(spec.gen_key) or {}
                                recon_cfg = analysis_cfg.get(spec.recon_key) or {}
                                _variance_list = gen_cfg.get("sample_variance")
                                if _variance_list is None:
                                    _variance_list = []
                                if not isinstance(_variance_list, list):
                                    _variance_list = [_variance_list]
                                _num_samples_list = gen_cfg.get("num_samples")
                                if _num_samples_list is None:
                                    _num_samples_list = []
                                if not isinstance(_num_samples_list, list):
                                    _num_samples_list = [_num_samples_list]
                                _max_recon_train_list = recon_cfg.get("max_recon_train")
                                if _max_recon_train_list is None:
                                    _max_recon_train_list = []
                                if not isinstance(_max_recon_train_list, list):
                                    _max_recon_train_list = [_max_recon_train_list]
                                _max_recon_test_list = recon_cfg.get("max_recon_test")
                                if _max_recon_test_list is None:
                                    _max_recon_test_list = []
                                if not isinstance(_max_recon_test_list, list):
                                    _max_recon_test_list = [_max_recon_test_list]
                                _visualize_latent = recon_cfg.get("visualize_latent", False)
                                _gen_max_train = gen_cfg.get("max_train")
                                _gen_max_test = gen_cfg.get("max_test")

                                def _get_or_compute_cached(mt, mc):
                                    if spec.id == "min_rmsd":
                                        if _cache.get("min_rmsd") is None:
                                            _cache_path = os.path.join(output_dir, EXP_STATS_CACHE_DIR, spec.cache_filename(analysis_cfg, None, None))
                                            _cache["min_rmsd"] = spec.get_or_compute_test_to_train(
                                                _cache_path, coords_np, coords, training_split, split_seed, base_output_dir,
                                                **spec.kwargs_for_cache(analysis_cfg, None, None),
                                            )
                                        return _cache["min_rmsd"]
                                    assert spec.id == "q"
                                    if _cache.get("q") is None:
                                        _cache["q"] = {}
                                    key = (mt, mc)
                                    if key not in _cache["q"]:
                                        _cache_path = os.path.join(output_dir, EXP_STATS_CACHE_DIR, spec.cache_filename(analysis_cfg, mt, mc))
                                        _cache["q"][key] = spec.get_or_compute_test_to_train(
                                            _cache_path, coords_np, coords, training_split, split_seed, base_output_dir,
                                            **spec.kwargs_for_cache(analysis_cfg, mt, mc),
                                        )
                                    return _cache["q"][key]

                                if do_gen:
                                    _mt_gen = _gen_max_train if spec.id == "q" else None
                                    _mc_gen = _gen_max_test if spec.id == "q" else None
                                    if spec.id == "q" and (_mt_gen is None or _mc_gen is None):
                                        continue
                                    _tt, _train_c, _test_c = _get_or_compute_cached(_mt_gen, _mc_gen)
                                    plot_cfg_gen = spec.build_gen_plot_cfg(analysis_cfg, plot_dpi)
                                    pre_kw = spec.precomputed_kwargs(_tt, _train_c, _test_c)
                                    extra_kw = spec.gen_extra_kwargs(analysis_cfg)
                                    for var in _variance_list:
                                        variance_suffix = f"_var{var}" if len(_variance_list) > 1 else ""
                                        any_missing = False
                                        for n in _num_samples_list:
                                            run_name = (str(n) + variance_suffix) if variance_suffix else (str(n) if len(_num_samples_list) > 1 else "default")
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
                                                run_name_single = (str(n) + variance_suffix) if (variance_suffix or len(_num_samples_list) > 1) else "default"
                                                output_suffix = ("_" + run_name_single) if run_name_single != "default" else ""
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
                                    plot_cfg_recon = spec.build_recon_plot_cfg(analysis_cfg, plot_dpi)
                                    recon_extra = spec.recon_extra_kwargs(analysis_cfg)
                                    for max_recon_train in _max_recon_train_list:
                                        for max_recon_test in _max_recon_test_list:
                                            _tt, _train_c, _test_c = _get_or_compute_cached(max_recon_train, max_recon_test)
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
                                            if _visualize_latent and not (resume and os.path.isfile(latent_fig)):
                                                train_mu, test_mu = _get_latent_vectors_euclideanizer(
                                                    frozen_vae, device, coords, training_split, split_seed, utils,
                                                    max_train=max_recon_train, max_test=max_recon_test,
                                                )
                                                plot_latent_distribution(
                                                    train_mu, test_mu, latent_fig,
                                                    plot_dpi=plot_dpi, display_root=base_output_dir,
                                                    save_pdf_copy=recon_cfg.get("save_pdf_copy", False),
                                                )
                                            elif resume and _visualize_latent and n_recon == 1 and os.path.isfile(latent_fig):
                                                _log(f"  [skip] latent distribution", since_start=time.time() - pipeline_start, style="skip")
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): analysis done in {(time.time() - analysis_phase_start) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")
    
                        del embed, frozen_vae
                        torch.cuda.empty_cache()
                        _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): done in {(time.time() - phase_start_eu) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")


def _run_one_seed(
    seed: int,
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
    do_plot: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_min_rmsd: bool,
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
    q_visualize_latent: bool = False,
    make_distmap_epoch_hook=None,
    make_euclideanizer_epoch_hook=None,
    assemble_video_fn=None,
) -> None:
    """Run the full pipeline for a single seed: DistMap segments, Euclideanizer segments, plotting, analysis."""
    output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    split_seed = seed
    effective_cfg = {**cfg, "output_dir": output_dir, "data": {**cfg["data"], "split_seed": seed}}

    if need_train and (not os.path.isdir(output_dir) or not os.path.isfile(pipeline_config_path(output_dir))):
        save_pipeline_config(effective_cfg, output_dir)

    if data_path:
        torch.manual_seed(split_seed)

    _log(f"Seed {seed}  output_dir={output_dir}", since_start=time.time() - pipeline_start, style="info")

    train_stats = test_stats = None
    if data_path and (do_plot or do_min_rmsd or do_q or do_q_recon) and (coords is not None or (num_structures is not None and num_atoms is not None)):
        train_stats, test_stats = _load_exp_stats_split_cache(
            output_dir, data_path, num_structures, num_atoms, split_seed, training_split
        )
        if train_stats is None or test_stats is None:
            if coords is not None:
                train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
                train_indices = np.array(train_ds.indices)
                test_indices = np.array(test_ds.indices)
                _log("Computing train/test experimental statistics...", since_start=time.time() - pipeline_start, style="info")
                train_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, indices=train_indices)
                test_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, indices=test_indices)
                _save_exp_stats_split_cache(
                    output_dir, data_path, num_structures, num_atoms, split_seed, training_split,
                    train_stats, test_stats,
                )
            else:
                _log("Train/test statistics not in cache (stats-only run); skipping recon_statistics/gen_variance for this seed.", since_start=time.time() - pipeline_start, style="skip")
        else:
            _log("Reused train/test experimental statistics from cache.", since_start=time.time() - pipeline_start, style="skip")

    seed_test_to_train_holder = [None]
    for gidx in range(len(dm_groups)):
        _run_one_distmap_group(
            seed, gidx, device,
            cfg, base_output_dir, dm_groups, eu_groups, dm_configs, eu_configs,
            coords, coords_np, num_atoms, num_structures, exp_stats, data_path, need_train, pipeline_start,
            training_split, do_plot, do_recon_plot, do_bond_rg_scaling, do_avg_gen, do_min_rmsd, resume,
            sample_variances, gen_num_samples, gen_decode_batch_size, need_plot_or_rmsd,
            save_structures_gro_plot, analysis_save_data, analysis_save_structures_gro,
            plot_dpi, save_pdf, save_plot_data, num_recon_samples, analysis_cfg, variance_list, num_samples_list,
            max_recon_train_list, max_recon_test_list,
            vis_enabled, vis_cfg, plot_cfg,
            train_stats, test_stats, seed_test_to_train_holder,
            do_q=do_q, do_q_recon=do_q_recon,
            q_max_train=q_max_train, q_max_test=q_max_test,
            q_num_samples_list=q_num_samples_list or [], q_variance_list=q_variance_list or [],
            q_delta=q_delta, q_max_recon_train_list=q_max_recon_train_list or [], q_max_recon_test_list=q_max_recon_test_list or [],
            q_recon_delta=q_recon_delta, q_visualize_latent=q_visualize_latent,
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
    global _LOG_FILE, _LOG_LOCK
    device = torch.device(f"cuda:{device_id}")
    _LOG_LOCK = None  # Workers use their own log handle; no cross-process lock.
    _LOG_FILE = open(log_path, "a", encoding="utf-8")
    worker_start = time.time()
    try:
        data_path = shared_args["data_path"]
        base_output_dir = shared_args["base_output_dir"]
        if not task_list or not data_path:
            return
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
        utils.validate_dataset_for_pipeline(num_structures, shared_args["training_split"])
        exp_stats = _load_exp_stats_cache(
            base_output_dir, data_path, num_structures, num_atoms
        )
        if exp_stats is None and (shared_args["do_plot"] or shared_args["do_min_rmsd"] or shared_args.get("do_q") or shared_args.get("do_q_recon")):
            exp_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps)
        for seed, gidx in task_list:
            output_dir = os.path.join(base_output_dir, f"seed_{seed}")
            train_stats = test_stats = None
            if data_path and (shared_args["do_plot"] or shared_args["do_min_rmsd"] or shared_args.get("do_q") or shared_args.get("do_q_recon")):
                train_stats, test_stats = _load_exp_stats_split_cache(
                    output_dir, data_path, num_structures, num_atoms,
                    seed, shared_args["training_split"],
                )
                if train_stats is None or test_stats is None:
                    train_ds, test_ds = utils.get_train_test_split(
                        coords.cpu(), shared_args["training_split"], seed
                    )
                    train_indices = np.array(train_ds.indices)
                    test_indices = np.array(test_ds.indices)
                    train_stats = compute_exp_statistics(
                        coords_np, device, utils.get_distmaps, indices=train_indices
                    )
                    test_stats = compute_exp_statistics(
                        coords_np, device, utils.get_distmaps, indices=test_indices
                    )
                    _save_exp_stats_split_cache(
                        output_dir, data_path, num_structures, num_atoms,
                        seed, shared_args["training_split"],
                        train_stats, test_stats,
                    )
            seed_test_to_train_holder = [None]
            _run_one_distmap_group(
                seed, gidx, device,
                shared_args["cfg"], base_output_dir, shared_args["dm_groups"],
                shared_args["eu_groups"], shared_args["dm_configs"], shared_args["eu_configs"],
                coords, coords_np, num_atoms, num_structures, exp_stats, data_path,
                shared_args["need_train"], worker_start, shared_args["training_split"],
                shared_args["do_plot"], shared_args["do_recon_plot"],
                shared_args["do_bond_rg_scaling"], shared_args["do_avg_gen"],
                shared_args["do_min_rmsd"], shared_args["resume"],
                shared_args["sample_variances"], shared_args["gen_num_samples"],
                shared_args["gen_decode_batch_size"], shared_args["need_plot_or_rmsd"],
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
                q_visualize_latent=shared_args.get("q_visualize_latent", False),
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
    training_split: float,
    do_plot: bool,
    do_recon_plot: bool,
    do_bond_rg_scaling: bool,
    do_avg_gen: bool,
    do_min_rmsd: bool,
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
    q_visualize_latent: bool = False,
    make_distmap_epoch_hook=None,
    make_euclideanizer_epoch_hook=None,
    assemble_video_fn=None,
) -> None:
    """Run tasks in parallel on multiple GPUs (one process per device)."""
    seeds = sorted({s for s, _ in tasks})
    # Per-seed setup: seed dirs and pipeline config. Train/test caches are precomputed in main when multi-GPU (to free data before spawn).
    for seed in seeds:
        output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        effective_cfg = {**cfg, "output_dir": output_dir, "data": {**cfg["data"], "split_seed": seed}}
        if need_train and (not os.path.isdir(output_dir) or not os.path.isfile(pipeline_config_path(output_dir))):
            save_pipeline_config(effective_cfg, output_dir)
        if data_path and coords is not None and (do_plot or do_min_rmsd or do_q or do_q_recon):
            train_stats, test_stats = _load_exp_stats_split_cache(
                output_dir, data_path, num_structures, num_atoms, seed, training_split
            )
            if train_stats is None or test_stats is None:
                train_ds, test_ds = utils.get_train_test_split(coords.cpu(), training_split, seed)
                train_indices = np.array(train_ds.indices)
                test_indices = np.array(test_ds.indices)
                train_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, indices=train_indices)
                test_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, indices=test_indices)
                _save_exp_stats_split_cache(
                    output_dir, data_path, num_structures, num_atoms,
                    seed, training_split, train_stats, test_stats,
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
        "training_split": training_split,
        "do_plot": do_plot,
        "do_recon_plot": do_recon_plot,
        "do_bond_rg_scaling": do_bond_rg_scaling,
        "do_avg_gen": do_avg_gen,
        "do_min_rmsd": do_min_rmsd,
        "resume": resume,
        "sample_variances": sample_variances,
        "gen_num_samples": gen_num_samples,
        "gen_decode_batch_size": gen_decode_batch_size,
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
        "q_visualize_latent": q_visualize_latent,
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
    global _LOG_FILE, _pipeline_real_stdout, _pipeline_real_stderr
    pipeline_start = time.time()
    args = _parse_args()
    # Worker subprocess entry: load args from pickle and run _worker (CUDA_VISIBLE_DEVICES already set by parent).
    if getattr(args, "worker_from_pickle", None):
        with open(args.worker_from_pickle, "rb") as f:
            task_list, log_path, shared_args = pickle.load(f)
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
    cfg = load_config(path=config_path, overrides=overrides)
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
    # Save full run config in output root for reproducibility (can re-run with: run.py --config <base_output_dir>/pipeline_config.yaml)
    root_run_cfg = {**cfg, "output_dir": base_output_dir}
    save_pipeline_config(root_run_cfg, base_output_dir)
    seeds = get_seeds(cfg)
    plot_cfg = cfg["plotting"]
    do_plot = plot_cfg["enabled"]
    plot_dpi = plot_cfg["plot_dpi"]
    save_pdf = plot_cfg["save_pdf_copy"]
    save_plot_data = plot_cfg["save_data"]  # config key is save_data (same as analysis); variable name for clarity
    num_recon_samples = plot_cfg["num_reconstruction_samples"]
    do_recon_plot = plot_cfg["reconstruction"]
    do_bond_rg_scaling = plot_cfg["bond_rg_scaling"]
    do_avg_gen = plot_cfg["avg_gen_vs_exp"]
    analysis_cfg = cfg["analysis"]
    do_min_rmsd = analysis_cfg["min_rmsd_gen"]["enabled"]
    do_min_rmsd_recon_cfg = analysis_cfg["min_rmsd_recon"]["enabled"]
    do_q = analysis_cfg["q_gen"]["enabled"]
    do_q_recon_cfg = analysis_cfg["q_recon"]["enabled"]
    training_split = cfg["data"]["training_split"]
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
    _log(f"config: {config_path}  output: {base_output_dir}  seeds: {seeds}", since_start=time.time() - pipeline_start, style="info")
    _log(f"DistMap runs: {len(dm_configs)}  Euclideanizer: {len(eu_configs)}  resume={resume}  plot={do_plot}  min_rmsd_gen={do_min_rmsd}  min_rmsd_recon={do_min_rmsd_recon_cfg}  q_gen={do_q}  q_recon={do_q_recon_cfg}", since_start=time.time() - pipeline_start, style="info")

    num_samples_list = analysis_cfg["min_rmsd_gen"]["num_samples"] if do_min_rmsd else []
    if not isinstance(num_samples_list, list):
        num_samples_list = [num_samples_list]
    variance_list = analysis_cfg["min_rmsd_gen"]["sample_variance"] if do_min_rmsd else []
    if not isinstance(variance_list, list):
        variance_list = [variance_list]
    max_recon_train_list = analysis_cfg["min_rmsd_recon"]["max_recon_train"] if do_min_rmsd_recon_cfg else []
    if not isinstance(max_recon_train_list, list):
        max_recon_train_list = [max_recon_train_list]
    max_recon_test_list = analysis_cfg["min_rmsd_recon"]["max_recon_test"] if do_min_rmsd_recon_cfg else []
    if not isinstance(max_recon_test_list, list):
        max_recon_test_list = [max_recon_test_list]

    q_max_train = analysis_cfg["q_gen"]["max_train"] if do_q else None
    q_max_test = analysis_cfg["q_gen"]["max_test"] if do_q else None
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
    q_recon_delta = analysis_cfg["q_recon"].get("delta", q_delta)
    dm_groups = distmap_training_groups(cfg)
    eu_groups = euclideanizer_training_groups(cfg)
    plot_variances_for_scan = get_sample_variances(cfg) if do_plot else []

    # overwrite_existing: prompt and delete existing plotting/analysis outputs up front (requires user approval)
    to_overwrite = []
    if do_plot and plot_cfg.get("overwrite_existing", False) and _has_any_plotting_output(base_output_dir, seeds):
        to_overwrite.append("plotting")
    if do_min_rmsd and analysis_cfg["min_rmsd_gen"].get("overwrite_existing", False) and _has_any_analysis_output(base_output_dir, seeds, "min_rmsd_gen"):
        to_overwrite.append("min_rmsd_gen")
    if do_min_rmsd_recon_cfg and analysis_cfg["min_rmsd_recon"].get("overwrite_existing", False) and _has_any_analysis_output(base_output_dir, seeds, "min_rmsd_recon"):
        to_overwrite.append("min_rmsd_recon")
    if do_q and analysis_cfg["q_gen"].get("overwrite_existing", False) and _has_any_analysis_output(base_output_dir, seeds, "q_gen"):
        to_overwrite.append("q_gen")
    if do_q_recon_cfg and analysis_cfg["q_recon"].get("overwrite_existing", False) and _has_any_analysis_output(base_output_dir, seeds, "q_recon"):
        to_overwrite.append("q_recon")
    if to_overwrite:
        if not getattr(args, "yes_overwrite", False):
            _confirm_overwrite_outputs(to_overwrite)
        _log("Removing existing outputs (overwrite_existing requested): " + ", ".join(to_overwrite), since_start=time.time() - pipeline_start, style="info")
        for label in to_overwrite:
            if label == "plotting":
                _delete_plotting_outputs_only(base_output_dir, seeds)
            elif label == "min_rmsd_gen":
                _delete_analysis_outputs_for_component(base_output_dir, seeds, "min_rmsd_gen")
            elif label == "min_rmsd_recon":
                _delete_analysis_outputs_for_component(base_output_dir, seeds, "min_rmsd_recon")
            elif label == "q_gen":
                _delete_analysis_outputs_for_component(base_output_dir, seeds, "q_gen")
            elif label == "q_recon":
                _delete_analysis_outputs_for_component(base_output_dir, seeds, "q_recon")
        _log("Done removing; will re-run these components.", since_start=time.time() - pipeline_start, style="success")

    # Pipeline config: strict match for training; if only plotting/analysis differ, prompt then delete and update saved config
    def _cfg_for_compare(c):
        d = dict(c)
        d.pop("resume", None)
        return d

    if need_train and resume and data_path:
        chunks_to_update = set()
        for seed in seeds:
            output_dir = os.path.join(base_output_dir, f"seed_{seed}")
            if os.path.isdir(output_dir):
                if not os.path.isfile(pipeline_config_path(output_dir)):
                    raise RuntimeError(
                        f"Resume is enabled but no pipeline config found in output_dir ({output_dir!r}). "
                        f"Refusing to resume without an exact config copy. Use a different output_dir or run with --no-resume to overwrite existing files in that directory."
                    )
                effective_cfg = {**cfg, "output_dir": output_dir, "data": {**cfg["data"], "split_seed": seed}}
                saved_cfg = load_pipeline_config(output_dir)
                saved_compare = _cfg_for_compare(saved_cfg) if saved_cfg else None
                effective_compare = _cfg_for_compare(effective_cfg)
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
                # Collect which chunks (plotting, min_rmsd_gen, min_rmsd_recon) differ from saved
                if not configs_match_sections(saved_compare, effective_compare, ["plotting"]):
                    chunks_to_update.add("plotting")
                s_analysis = (saved_compare.get("analysis") or {})
                e_analysis = (effective_compare.get("analysis") or {})
                if s_analysis.get("min_rmsd_gen") != e_analysis.get("min_rmsd_gen"):
                    chunks_to_update.add("min_rmsd_gen")
                if s_analysis.get("min_rmsd_recon") != e_analysis.get("min_rmsd_recon"):
                    chunks_to_update.add("min_rmsd_recon")
                if s_analysis.get("q_gen") != e_analysis.get("q_gen"):
                    chunks_to_update.add("q_gen")
                if s_analysis.get("q_recon") != e_analysis.get("q_recon"):
                    chunks_to_update.add("q_recon")
        chunks_to_update = sorted(chunks_to_update)  # stable order: min_rmsd_gen, min_rmsd_recon, q_gen, q_recon, plotting
        if "plotting" in chunks_to_update:
            chunks_to_update = ["plotting"] + [c for c in chunks_to_update if c != "plotting"]
        # Chunks already deleted by overwrite_existing block above: skip second prompt and delete
        chunks_still_to_delete = [c for c in chunks_to_update if c not in to_overwrite]
        # Only prompt and delete for chunks that actually have existing outputs
        chunk_labels = {"plotting": "Plotting", "min_rmsd_gen": "Min-RMSD (gen)", "min_rmsd_recon": "Min-RMSD (recon)", "q_gen": "Q (gen)", "q_recon": "Q (recon)"}
        chunks_with_outputs = [
            c for c in chunks_still_to_delete
            if (c == "plotting" and _has_any_plotting_output(base_output_dir, seeds))
            or (c != "plotting" and _has_any_analysis_output(base_output_dir, seeds, c))
        ]
        if chunks_to_update:
            for chunk in chunks_with_outputs:
                if not getattr(args, "yes_overwrite", False):
                    _confirm_replot_one_chunk(base_output_dir, chunk_labels[chunk])
                _log(f"Removing existing {chunk_labels[chunk]} outputs (config changed).", since_start=time.time() - pipeline_start, style="info")
                if chunk == "plotting":
                    _delete_plotting_outputs_only(base_output_dir, seeds)
                else:
                    _delete_analysis_outputs_for_component(base_output_dir, seeds, chunk)
            save_pipeline_config({**cfg, "output_dir": base_output_dir}, base_output_dir)
            for seed in seeds:
                output_dir = os.path.join(base_output_dir, f"seed_{seed}")
                if os.path.isdir(output_dir):
                    save_pipeline_config(
                        {**cfg, "output_dir": output_dir, "data": {**cfg["data"], "split_seed": seed}},
                        output_dir,
                    )
            _log("Saved updated pipeline config; will skip training and re-run affected plotting/analysis.", since_start=time.time() - pipeline_start, style="success")
            need_train = False  # Skip training when only plotting/analysis config changed

    # Decide what to load from pipeline segments (resume) or from flags (no resume)
    if not resume or not data_path:
        do_min_rmsd_recon = analysis_cfg["min_rmsd_recon"]["enabled"]
        do_visualize_latent = analysis_cfg["min_rmsd_recon"]["visualize_latent"]
        do_q_recon = analysis_cfg["q_recon"]["enabled"]
        needs = PipelineDataNeeds(
            need_coords=(need_train or do_plot or do_min_rmsd or do_min_rmsd_recon or do_q or do_q_recon),
            need_exp_stats=do_plot,
            need_train_test_stats=do_plot,
        ) if data_path else PipelineDataNeeds(need_coords=False, need_exp_stats=False, need_train_test_stats=False)
    else:
        needs = _pipeline_data_needs(
            base_output_dir, seeds, dm_groups, eu_groups,
            resume, do_plot, do_min_rmsd, do_recon_plot, do_bond_rg_scaling, do_avg_gen,
            plot_variances_for_scan, variance_list, num_samples_list,
            do_min_rmsd_recon=analysis_cfg["min_rmsd_recon"]["enabled"], visualize_latent=analysis_cfg["min_rmsd_recon"]["visualize_latent"],
            max_recon_train_list=max_recon_train_list, max_recon_test_list=max_recon_test_list,
            do_q=do_q, do_q_recon=do_q_recon_cfg,
            q_variance_list=q_variance_list, q_num_samples_list=q_num_samples_list,
            q_max_recon_train_list=q_max_recon_train_list, q_max_recon_test_list=q_max_recon_test_list,
            q_visualize_latent=analysis_cfg["q_recon"].get("visualize_latent", False),
        )
    need_any = needs.need_any() and data_path

    stats_only_ok = False
    if need_any and not needs.need_coords and (needs.need_exp_stats or needs.need_train_test_stats):
        exp_st, num_at, num_stru = _try_load_stats_only(base_output_dir, data_path, seeds, training_split)
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

    if need_any and not stats_only_ok:
        phase_start = time.time()
        _log("Loading data...", since_start=time.time() - pipeline_start, style="info")
        coords_np = utils.load_data(data_path)
        coords = torch.tensor(coords_np, dtype=torch.float32)
        device = utils.get_device()
        coords = coords.to(device)
        num_atoms = coords.size(1)
        num_structures = len(coords_np)
        utils.validate_dataset_for_pipeline(num_structures, training_split)
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
                    exp_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps)
                    _save_exp_stats_cache(base_output_dir, data_path, num_structures, num_atoms, exp_stats)
                    _log(f"Cached experimental statistics to {EXP_STATS_CACHE_DIR}/.", since_start=time.time() - pipeline_start, style="success")
                else:
                    raise RuntimeError(
                        f"Experimental statistics cache exists for a different dataset. "
                        f"Cached: data_path={cached.get('data_path')!r} num_structures={cached.get('num_structures')} num_atoms={cached.get('num_atoms')}. "
                        f"Current: data_path={os.path.abspath(data_path)!r} num_structures={num_structures} num_atoms={num_atoms}. "
                        f"Use a different output_dir for this dataset, or remove {_exp_stats_cache_dir(base_output_dir)!r} to start fresh."
                    )
            elif _base_has_any_seed_pipeline_content(base_output_dir, seeds):
                raise RuntimeError(
                    f"Experimental statistics cache is missing but base_output_dir already has pipeline content ({base_output_dir!r}). "
                    f"This would allow the dataset to be replaced midway. Use a different output_dir for this dataset, "
                    f"or restore the cache (e.g. from backup) before resuming."
                )
            else:
                _log("Computing experimental statistics (no valid cache for this dataset).", since_start=time.time() - pipeline_start, style="info")
                exp_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps)
                _save_exp_stats_cache(base_output_dir, data_path, num_structures, num_atoms, exp_stats)
                _log(f"Cached experimental statistics to {EXP_STATS_CACHE_DIR}/.", since_start=time.time() - pipeline_start, style="success")
        _log("Data ready.", since_start=time.time() - pipeline_start, since_phase=time.time() - phase_start, style="success")

    if not need_any:
        coords_np = coords = device = num_atoms = num_structures = exp_stats = None
        if data_path and (do_plot or do_min_rmsd):
            _log("Skipping data load (all runs complete and all plot/analysis outputs present).", since_start=time.time() - pipeline_start, style="skip")

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
            (coords is not None and (exp_stats is not None or (do_recon_plot and not do_bond_rg_scaling and not do_avg_gen)))
            or (coords is None and exp_stats is not None)
        ))
        or (do_min_rmsd and coords is not None)
        or (do_q and coords is not None)
        or (do_q_recon_cfg and coords is not None)
    )
    save_structures_gro_plot = plot_cfg["save_structures_gro"]
    analysis_save_data = analysis_cfg["min_rmsd_gen"]["save_data"]
    analysis_save_structures_gro = analysis_cfg["min_rmsd_gen"]["save_structures_gro"]

    tasks = [(s, g) for s in seeds for g in range(len(dm_groups))]
    n_gpus = utils.get_available_cuda_count()
    if args.gpus is not None:
        n_gpus = min(n_gpus, args.gpus)
    use_multi_gpu = (n_gpus >= 2) and not getattr(args, "no_multi_gpu", False) and (coords is not None)
    if use_multi_gpu:
        n_workers = min(n_gpus, len(tasks))
        _log(f"Detected {n_gpus} GPU(s). Using multi-GPU with {n_workers} worker(s).", since_start=time.time() - pipeline_start, style="info")
    else:
        _log(f"Detected {n_gpus} GPU(s). Using single process.", since_start=time.time() - pipeline_start, style="info")

    # Multi-GPU: ensure per-seed train/test stats caches exist, then free main-process data so only workers hold copies (reduces memory use).
    if use_multi_gpu and data_path and coords is not None and (do_plot or do_min_rmsd or do_q or do_q_recon_cfg):
        for seed in seeds:
            output_dir = os.path.join(base_output_dir, f"seed_{seed}")
            train_stats, test_stats = _load_exp_stats_split_cache(
                output_dir, data_path, num_structures, num_atoms, seed, training_split
            )
            if train_stats is None or test_stats is None:
                train_ds, test_ds = utils.get_train_test_split(coords, training_split, seed)
                train_indices = np.array(train_ds.indices)
                test_indices = np.array(test_ds.indices)
                train_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, indices=train_indices)
                test_stats = compute_exp_statistics(coords_np, device, utils.get_distmaps, indices=test_indices)
                _save_exp_stats_split_cache(
                    output_dir, data_path, num_structures, num_atoms, seed, training_split,
                    train_stats, test_stats,
                )
        _log("Freed main-process data before spawning workers (multi-GPU).", since_start=time.time() - pipeline_start, style="info")
        coords_np = coords = exp_stats = None
        gc.collect()

    if not use_multi_gpu:
        for seed in seeds:
            _run_one_seed(
            seed=seed,
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
            do_plot=do_plot,
            do_recon_plot=do_recon_plot,
            do_bond_rg_scaling=do_bond_rg_scaling,
            do_avg_gen=do_avg_gen,
            do_min_rmsd=do_min_rmsd,
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
            q_visualize_latent=analysis_cfg["q_recon"].get("visualize_latent", False),
            make_distmap_epoch_hook=make_dm_hook,
            make_euclideanizer_epoch_hook=make_eu_hook,
            assemble_video_fn=assemble_video_fn,
        )
    else:
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
            training_split=training_split,
            do_plot=do_plot,
            do_recon_plot=do_recon_plot,
            do_bond_rg_scaling=do_bond_rg_scaling,
            do_avg_gen=do_avg_gen,
            do_min_rmsd=do_min_rmsd,
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
            q_visualize_latent=analysis_cfg["q_recon"].get("visualize_latent", False),
            make_distmap_epoch_hook=make_dm_hook,
            make_euclideanizer_epoch_hook=make_eu_hook,
            assemble_video_fn=assemble_video_fn,
        )

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
            _LOG_FILE.write(traceback.format_exc())
            _LOG_FILE.flush()
        raise
