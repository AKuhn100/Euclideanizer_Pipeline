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
import os
import shutil
import sys
import traceback
import zlib
import time
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
    config_diff,
    run_config_section_matches,
    pipeline_config_path,
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
from src.min_rmsd import run_min_rmsd_analysis, run_min_rmsd_analysis_multi, get_or_compute_test_to_train_rmsd
from src.gro_io import write_structures_gro

# Log file in output root (set in main(); writes to stdout + file in real time)
_LOG_FILE = None
# Real stdout/stderr before wrapping (for submodule coloring); restored on exit
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
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


def _log_raw(line: str, style: str | None = None) -> None:
    """Write a raw line (e.g. separator) to stdout (styled when TTY) and log file (plain)."""
    print(_style(line, style))
    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


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
    except (json.JSONDecodeError, OSError, KeyError, zlib.error):
        return None


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
    except (json.JSONDecodeError, OSError, KeyError, zlib.error):
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
) -> bool:
    """True if run_dir has a completed run: last_epoch_trained == expected_epochs, section matches if given, best checkpoint exists. When multi_segment, last checkpoint required only if (not is_last_segment or save_final_models_per_stretch)."""
    model_dir = os.path.join(run_dir, model_subdir)
    run_cfg = load_run_config(model_dir)
    if run_cfg is None:
        return False
    if run_cfg.get("last_epoch_trained") != expected_epochs:
        return False
    if section_key is not None and expected_section is not None:
        if not run_config_section_matches(run_cfg, section_key, expected_section):
            return False
    best_name = "model.pt" if section_key == "distmap" else "euclideanizer.pt"
    if not os.path.isfile(os.path.join(model_dir, best_name)):
        return False
    require_last = multi_segment and checkpoint_last_name and (not is_last_segment or save_final_models_per_stretch)
    if require_last and not os.path.isfile(os.path.join(model_dir, checkpoint_last_name)):
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
    if resume and os.path.isfile(dm_path) and _run_completed(
        run_dir_dm, ev, section_key="distmap", expected_section=dm_cfg,
        multi_segment=dm_multi, checkpoint_last_name="model_last.pt" if dm_multi else None,
        is_last_segment=dm_last_segment, save_final_models_per_stretch=dm_save_final,
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


# Single source of truth: plot type -> (subdir, filename pattern). Use _plot_path(run_root, type) or _plot_path(run_root, type, subset=..., var=...).
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
) -> bool:
    """True if resume and do_min_rmsd and all min_rmsd analysis outputs we would generate already exist."""
    if not resume or not do_min_rmsd:
        return True  # no analysis or not resuming -> "all present" for analysis
    for var in variance_list:
        variance_suffix = f"_var{var}" if len(variance_list) > 1 else ""
        for n in num_samples_list:
            run_name = (str(n) + variance_suffix) if variance_suffix else (str(n) if len(num_samples_list) > 1 else "default")
            fig_path = _analysis_path(run_dir_eu, "min_rmsd", f"{run_name}/min_rmsd_distributions.png")
            if not os.path.isfile(fig_path):
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
) -> bool:
    """True if any run is incomplete or any plot/analysis output is missing (so we must load data). Uses completion check without section match so older run_configs (e.g. missing optional keys) still count as complete."""
    for seed in seeds:
        output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        if not os.path.isdir(output_dir):
            return True
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
                    return True
                if do_plot and not _distmap_plotting_all_present(
                    run_dir_dm, resume, do_recon_plot, do_bond_rg_scaling, do_avg_gen, plot_variances
                ):
                    return True
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
                            return True
                        if do_plot and not _euclideanizer_plotting_all_present(
                            eu_run_dir, resume, do_recon_plot, do_bond_rg_scaling, do_avg_gen, plot_variances
                        ):
                            return True
                        if not _euclideanizer_analysis_all_present(eu_run_dir, resume, do_min_rmsd, variance_list, num_samples_list):
                            return True
    return False


def _video_frames_dir(run_root: str) -> str:
    return os.path.join(run_root, "plots", "training_video", "frames")


def _video_mp4_path(run_root: str) -> str:
    return os.path.join(run_root, "plots", "training_video", "training_evolution.mp4")


def _parse_args():
    """Parse command-line arguments; --config is required (no default)."""
    p = argparse.ArgumentParser(description="Euclideanizer pipeline: DistMap + Euclideanizer training and plotting")
    p.add_argument("--data", type=str, default=None, help="Path to dataset (required for training)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g. Euclideanizer_Pipeline/config_sample.yaml)")
    p.add_argument("--no-plots", action="store_true", help="Disable all plotting")
    p.add_argument("--no-resume", action="store_true", help="Do not resume; overwrite existing run outputs")
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


def _force_gpu_cleanup(device: torch.device) -> None:
    """Release unused GPU memory so the allocator and system see accurate free memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    vis_enabled: bool,
    vis_cfg: dict,
    plot_cfg: dict,
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
    if data_path and coords is not None and (do_plot or do_min_rmsd):
        train_stats, test_stats = _load_exp_stats_split_cache(
            output_dir, data_path, num_structures, num_atoms, split_seed, training_split
        )
        if train_stats is None or test_stats is None:
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
            _log("Reused train/test experimental statistics from cache.", since_start=time.time() - pipeline_start, style="skip")

    seed_test_to_train_cache = None
    distmap_runs = [None] * len(dm_configs)

    for gidx, group in enumerate(dm_groups):
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
                _log(f"DistMap run {ri} (epochs={ev}): resumed (skip training).", since_start=time.time() - pipeline_start, style="skip")
                prev_dm_path = dm_path
                prev_dm_ev = ev
            else:
                if vis_enabled:
                    fd_dm = _video_frames_dir(run_dir_dm)
                    if os.path.isdir(fd_dm):
                        shutil.rmtree(fd_dm)
                if dm_act["action"] == "from_scratch":
                    _log(f"DistMap run {ri}: training from scratch to {ev} epochs...", since_start=time.time() - pipeline_start, style="info")
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
                    _log(f"DistMap run {ri}: resuming from best (epoch {dm_act['best_epoch']}), training {dm_act['additional_epochs']} more → {ev} total...", since_start=time.time() - pipeline_start, style="info")
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
                    _log(f"DistMap run {ri}: resuming from run (epochs={prev_dm_ev}), training {dm_act['additional_epochs']} more → {ev} total...", since_start=time.time() - pipeline_start, style="info")
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

            distmap_runs[ri] = (ri, dm_path, dm_cfg)

            if do_plot and coords is not None and exp_stats is not None:
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
                    if do_recon_plot:
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
                    if do_bond_rg_scaling and train_stats is not None and test_stats is not None:
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
                            run_dir_eu, resume, do_min_rmsd, variance_list, num_samples_list
                        )
                        if resume and all_plots and all_analysis:
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): [skip] plotting and analysis (all present)", since_start=time.time() - pipeline_start, style="skip")
                        else:
                            _force_gpu_cleanup(device)
                            phase_start_eu = time.time()
                            frozen_vae = load_frozen_vae(dm_path, num_atoms, dm_cfg["latent_dim"], device)
                            embed = Euclideanizer(num_atoms=num_atoms).to(device)
                            embed.load_state_dict(torch.load(eu_path, map_location=device))

                            if do_plot and coords is not None and exp_stats is not None:
                                plot_phase_start = time.time()
                                _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): plotting (diagnostics)...", since_start=time.time() - pipeline_start, style="info")
                                if do_recon_plot:
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
                                if do_bond_rg_scaling and train_stats is not None and test_stats is not None:
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

                            if do_min_rmsd and coords is not None:
                                if seed_test_to_train_cache is None:
                                    _cache_path = os.path.join(output_dir, EXP_STATS_CACHE_DIR, "test_to_train_rmsd.npz")
                                    seed_test_to_train_cache = get_or_compute_test_to_train_rmsd(
                                        coords_np, coords, training_split, split_seed,
                                        _cache_path,
                                        query_batch_size=analysis_cfg["min_rmsd_query_batch_size"],
                                        display_root=base_output_dir,
                                    )
                                _tt, _train_c, _test_c = seed_test_to_train_cache
                                analysis_phase_start = time.time()
                                _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): analysis (min-RMSD)...", since_start=time.time() - pipeline_start, style="info")
                                plot_cfg_analysis = {
                                    **analysis_cfg,
                                    "plot_dpi": plot_dpi,
                                    "save_plot_data": save_plot_data,
                                    "save_data": analysis_save_data,
                                    "save_structures_gro": analysis_save_structures_gro,
                                }
                                for var in variance_list:
                                    variance_suffix = f"_var{var}" if len(variance_list) > 1 else ""
                                    any_missing = False
                                    for n in num_samples_list:
                                        run_name = (str(n) + variance_suffix) if variance_suffix else (str(n) if len(num_samples_list) > 1 else "default")
                                        fig_path = _analysis_path(run_dir_eu, "min_rmsd", f"{run_name}/min_rmsd_distributions.png")
                                        if not (resume and os.path.isfile(fig_path)):
                                            any_missing = True
                                            break
                                    if any_missing:
                                        if len(num_samples_list) > 1:
                                            run_min_rmsd_analysis_multi(
                                                coords_np, coords, training_split, split_seed,
                                                frozen_vae, embed, dm_cfg["latent_dim"], device, run_dir_eu,
                                                plot_cfg_analysis,
                                                num_samples_list=num_samples_list,
                                                sample_variance=var,
                                                variance_suffix=variance_suffix,
                                                display_root=base_output_dir,
                                                precomputed_test_to_train=_tt,
                                                train_coords_np=_train_c,
                                                test_coords_np=_test_c,
                                            )
                                        else:
                                            n = num_samples_list[0]
                                            run_name_single = (str(n) + variance_suffix) if (variance_suffix or len(num_samples_list) > 1) else "default"
                                            output_suffix = ("_" + run_name_single) if run_name_single != "default" else ""
                                            run_min_rmsd_analysis(
                                                coords_np, coords, training_split, split_seed,
                                                frozen_vae, embed, dm_cfg["latent_dim"], device, run_dir_eu,
                                                plot_cfg_analysis,
                                                num_samples=n, sample_variance=var, output_suffix=output_suffix,
                                                display_root=base_output_dir,
                                                precomputed_test_to_train=_tt,
                                                train_coords_np=_train_c,
                                                test_coords_np=_test_c,
                                            )
                                    else:
                                        _log(f"  [skip] min_rmsd variance={var}", since_start=time.time() - pipeline_start, style="skip")
                                _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): analysis done in {(time.time() - analysis_phase_start) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")

                            del embed, frozen_vae
                            torch.cuda.empty_cache()
                            _log(f"Euclideanizer {euri + 1}/{len(eu_configs)} (DistMap {ri}, epochs={eu_ev}): done in {(time.time() - phase_start_eu) / 60:.1f}m.", since_start=time.time() - pipeline_start, style="success")

    if do_min_rmsd and not analysis_save_data and seed_test_to_train_cache is not None:
        _cache_path = os.path.join(output_dir, EXP_STATS_CACHE_DIR, "test_to_train_rmsd.npz")
        if os.path.isfile(_cache_path):
            try:
                os.remove(_cache_path)
            except OSError:
                pass


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


def main():
    global _LOG_FILE, _pipeline_real_stdout, _pipeline_real_stderr
    pipeline_start = time.time()
    args = _parse_args()
    overrides = _args_to_overrides(args)
    config_path = args.config
    cfg = load_config(path=config_path, overrides=overrides)
    data_path = get_data_path(cfg)
    base_output_dir = get_output_dir(cfg)
    resume = cfg["resume"]
    if not resume and os.path.isdir(base_output_dir):
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
    save_plot_data = plot_cfg["save_plot_data"]
    num_recon_samples = plot_cfg["num_reconstruction_samples"]
    do_recon_plot = plot_cfg["reconstruction"]
    do_bond_rg_scaling = plot_cfg["bond_rg_scaling"]
    do_avg_gen = plot_cfg["avg_gen_vs_exp"]
    analysis_cfg = cfg["analysis"]
    do_min_rmsd = analysis_cfg["min_rmsd"]
    training_split = cfg["data"]["training_split"]

    # Training requires dataset
    dm_configs = expand_distmap_grid(cfg)
    eu_configs = expand_euclideanizer_grid(cfg)
    need_train = len(dm_configs) > 0 or len(eu_configs) > 0
    if need_train and not data_path:
        _log_raw("ERROR: Training requested but no dataset path. Set --data or data.path in config.", style="error")
        sys.exit(1)

    _log("Pipeline started.", since_start=time.time() - pipeline_start, style="info")
    _log(f"config: {config_path}  output: {base_output_dir}  seeds: {seeds}", since_start=time.time() - pipeline_start, style="info")
    _log(f"DistMap runs: {len(dm_configs)}  Euclideanizer: {len(eu_configs)}  resume={resume}  plot={do_plot}  min_rmsd={do_min_rmsd}", since_start=time.time() - pipeline_start, style="info")

    num_samples_list = analysis_cfg["min_rmsd_num_samples"] if do_min_rmsd else []
    if not isinstance(num_samples_list, list):
        num_samples_list = [num_samples_list]
    variance_list = analysis_cfg["min_rmsd_sample_variance"] if do_min_rmsd else []
    if not isinstance(variance_list, list):
        variance_list = [variance_list]
    dm_groups = distmap_training_groups(cfg)
    eu_groups = euclideanizer_training_groups(cfg)
    plot_variances_for_scan = get_sample_variances(cfg) if do_plot else []

    # Pipeline config match first (fail fast before loading data)
    if need_train and resume and data_path:
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
                # Compare ignoring 'resume' so toggling resume true/false does not trigger a config mismatch
                def _cfg_for_compare(c):
                    d = dict(c)
                    d.pop("resume", None)
                    return d
                saved_compare = _cfg_for_compare(saved_cfg) if saved_cfg else None
                effective_compare = _cfg_for_compare(effective_cfg)
                if saved_compare is None or not configs_match_exactly(saved_compare, effective_compare):
                    diff_lines = config_diff(saved_compare or {}, effective_compare) if saved_compare else ["saved config missing or invalid"]
                    diff_msg = "\n  ".join(diff_lines[:20])
                    if len(diff_lines) > 20:
                        diff_msg += f"\n  ... and {len(diff_lines) - 20} more."
                    raise RuntimeError(
                        f"Resume is enabled but pipeline config in output_dir does not match current config (comparison ignores 'resume').\n"
                        f"Output dir: {output_dir!r}\n"
                        f"Differences (saved vs current):\n  {diff_msg}\n"
                        f"Use a different output_dir to keep existing runs, or run with --no-resume to overwrite (this will overwrite existing checkpoints and outputs in that directory)."
                    )

    # Load data only when we will use it: not resuming, or something is incomplete/missing (training or plot/analysis)
    need_data = (need_train or do_plot or do_min_rmsd) and data_path
    if need_data and resume:
        need_data = _pipeline_need_data(
            base_output_dir, seeds, dm_groups, eu_groups,
            resume, do_plot, do_min_rmsd, do_recon_plot, do_bond_rg_scaling, do_avg_gen,
            plot_variances_for_scan, variance_list, num_samples_list,
        )
    if need_data and data_path:
        phase_start = time.time()
        _log("Loading data and experimental statistics...", since_start=time.time() - pipeline_start, style="info")
        coords_np = utils.load_data(data_path)
        coords = torch.tensor(coords_np, dtype=torch.float32)
        device = utils.get_device()
        coords = coords.to(device)
        num_atoms = coords.size(1)
        num_structures = len(coords_np)
        utils.validate_dataset_for_pipeline(num_structures, training_split)
        _log(f"Loaded {num_structures} structures, {num_atoms} atoms.", since_start=time.time() - pipeline_start, style="success")
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
    else:
        coords_np = coords = device = num_atoms = num_structures = exp_stats = None
        if data_path and not need_data and (do_plot or do_min_rmsd):
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

    sample_variances = get_sample_variances(cfg) if do_plot and coords is not None else []
    gen_num_samples = cfg["plotting"]["num_samples"] if do_plot and coords is not None else 0
    gen_decode_batch_size = plot_cfg["gen_decode_batch_size"]
    need_plot_or_rmsd = (do_plot and coords is not None and exp_stats is not None) or (do_min_rmsd and coords is not None)
    save_structures_gro_plot = plot_cfg["save_structures_gro"]
    analysis_save_data = analysis_cfg["save_data"]
    analysis_save_structures_gro = analysis_cfg["save_structures_gro"]

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
            vis_enabled=vis_enabled,
            vis_cfg=vis_cfg,
            plot_cfg=plot_cfg,
            make_distmap_epoch_hook=make_dm_hook,
            make_euclideanizer_epoch_hook=make_eu_hook,
            assemble_video_fn=assemble_video_fn,
        )

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
