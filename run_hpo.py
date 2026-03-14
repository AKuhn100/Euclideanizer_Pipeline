#!/usr/bin/env python3
"""
Optuna-based HPO for DistMap + Euclideanizer. Joint optimization; objective = overall score.

  python run_hpo.py --config samples/hpo_config.yaml [--data /path/to/data.gro]
  python run_hpo.py --config samples/hpo_config.yaml --resume --n-trials-add 50

Multi-GPU: By default n_jobs = min(n_trials, n_gpus), so e.g. 8 H200s run 8 trials in
parallel. Trial N uses GPU (N % n_gpus). Pruning works with n_jobs > 1 when using
file-based SQLite storage (default). Set n_jobs: 1 in HPO config to force one trial at a time.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    import optuna
except ImportError:
    optuna = None

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# After path setup, optional torch for GPU count
try:
    import torch
except ImportError:
    torch = None

# In-process trial runner (enables pruning)
import run as run_module

SCORING_DIR = "scoring"
SCORES_FILENAME = "scores.json"
FAILED_TRIALS_LOG = "hpo_failed_trials.log"
PIPELINE_CONFIG_FILENAME = "pipeline_config.yaml"


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required. pip install pyyaml")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a dict: {path}")
    return data


def _get_n_gpus(hpo_cfg: dict) -> int:
    n = hpo_cfg.get("n_gpus")
    if n is not None:
        return int(n)
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def _build_sampler(optuna_cfg: dict, seed: int):
    """Build Optuna sampler from config. Injects top-level seed into sampler_kwargs if not already set."""
    sampler_name = optuna_cfg.get("sampler", "TPESampler")
    kwargs = dict(optuna_cfg.get("sampler_kwargs") or {})
    if "seed" not in kwargs:
        kwargs["seed"] = seed
    sampler_cls = getattr(optuna.samplers, sampler_name, None)
    if sampler_cls is None:
        raise ValueError(f"Unknown sampler: {sampler_name}. Use e.g. TPESampler, RandomSampler, CmaEsSampler (see optuna.samplers).")
    return sampler_cls(**kwargs)


def _resolve_data_path(hpo_cfg: dict, cli_data: str | None) -> str:
    path = cli_data or (hpo_cfg.get("data_path") if hpo_cfg else None)
    if not path:
        raise ValueError("Data path required: set data_path in HPO config or pass --data")
    return str(Path(path).resolve())


def _ensure_single_value(cfg: dict, section: str, key: str) -> None:
    """If section.key is a list, replace with single value (max for epochs, first otherwise)."""
    if section not in cfg or key not in cfg[section]:
        return
    v = cfg[section][key]
    if isinstance(v, list):
        cfg[section][key] = max(v) if key == "epochs" else v[0]


def _build_trial_config(
    base_cfg: dict,
    trial_params: dict,
    trial_dir: str,
    data_path: str,
    seed: int,
    epoch_cap: int | None = None,
) -> dict:
    """Build full pipeline config for one trial. base_cfg is the template (from pipeline_config);
    trial_params overlay Optuna-suggested values for search_space keys; output_dir, data.path, data.split_seed,
    and optionally epoch_cap are set from HPO. Everything else (lr, batch_size, plotting, analysis, etc.) comes from the template."""
    cfg = copy.deepcopy(base_cfg)
    cfg["output_dir"] = str(Path(trial_dir).resolve())
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["path"] = data_path
    cfg["data"]["split_seed"] = seed
    for section in ("distmap", "euclideanizer"):
        if section not in cfg:
            cfg[section] = {}
        for key, val in (trial_params.get(section) or {}).items():
            cfg[section][key] = val
    if epoch_cap is not None:
        cfg["distmap"]["epochs"] = int(epoch_cap)
        cfg["euclideanizer"]["epochs"] = int(epoch_cap)
    else:
        _ensure_single_value(cfg, "distmap", "epochs")
        _ensure_single_value(cfg, "euclideanizer", "epochs")
    if cfg.get("scoring"):
        cfg["scoring"]["enabled"] = True
    return cfg


def _suggest_params(trial: "optuna.Trial", search_space: dict) -> dict:
    out = {"distmap": {}, "euclideanizer": {}}
    for section in ("distmap", "euclideanizer"):
        for key, spec in (search_space.get(section) or {}).items():
            if not isinstance(spec, dict):
                continue
            typ = spec.get("type")
            name = f"{section}.{key}"
            if typ == "categorical":
                choices = spec.get("choices")
                if choices is None:
                    continue
                out[section][key] = trial.suggest_categorical(name, list(choices))
            elif typ == "int":
                low = spec.get("low")
                high = spec.get("high")
                if low is None or high is None:
                    continue
                out[section][key] = trial.suggest_int(name, int(low), int(high))
            elif typ == "float":
                low = spec.get("low")
                high = spec.get("high")
                if low is None or high is None:
                    continue
                out[section][key] = trial.suggest_float(name, float(low), float(high))
            elif typ == "log_float":
                low = spec.get("low")
                high = spec.get("high")
                if low is None or high is None:
                    continue
                out[section][key] = trial.suggest_float(
                    name, float(low), float(high), log=True
                )
    return out


# Per-process cache for (data_path, seed) -> (coords, coords_np, num_atoms, num_structures, exp_stats, train_stats, test_stats)
_hpo_data_cache = {}


def _load_trial_data(
    data_path: str,
    seed: int,
    base_cfg: dict,
    device,
) -> tuple:
    """Load coords, compute exp_stats and train/test stats. Uses _hpo_data_cache so each process loads once per (data_path, seed)."""
    cache_key = (os.path.abspath(data_path), seed)
    if cache_key in _hpo_data_cache:
        return _hpo_data_cache[cache_key]
    from src import utils
    from src.metrics import compute_exp_statistics

    coords_np = utils.load_data(data_path)
    coords = torch.tensor(coords_np, dtype=torch.float32).to(device)
    num_atoms = coords.size(1)
    num_structures = len(coords_np)
    data_cfg = base_cfg.get("data", {})
    plot_cfg = base_cfg.get("plotting", {})
    training_split = float(data_cfg.get("training_split", 0.8))
    chunk_size = data_cfg.get("exp_stats_chunk_size", 500)
    avg_map = data_cfg.get("exp_stats_avg_map_sample", 5000)
    n_atoms_cap = min(num_atoms - 1, 999)

    exp_stats = compute_exp_statistics(
        coords_np, device, utils.get_distmaps, n_atoms_cap, chunk_size, avg_map
    )
    train_ds, test_ds = utils.get_train_test_split(coords.cpu(), training_split, seed)
    train_indices = np.array(train_ds.indices)
    test_indices = np.array(test_ds.indices)
    plot_mt = plot_cfg.get("max_train")
    plot_mc = plot_cfg.get("max_test")
    if plot_mt is not None:
        train_indices = train_indices[:plot_mt]
    if plot_mc is not None:
        test_indices = test_indices[:plot_mc]
    train_stats = compute_exp_statistics(
        coords_np, device, utils.get_distmaps, n_atoms_cap, chunk_size, avg_map, indices=train_indices
    )
    test_stats = compute_exp_statistics(
        coords_np, device, utils.get_distmaps, n_atoms_cap, chunk_size, avg_map, indices=test_indices
    )
    result = (
        coords,
        coords_np,
        num_atoms,
        num_structures,
        exp_stats,
        train_stats,
        test_stats,
    )
    _hpo_data_cache[cache_key] = result
    return result


def _read_overall_score(trial_dir: str, seed: int) -> float | None:
    # Pipeline layout: trial_dir/seed_X/distmap/0/euclideanizer/0/scoring/scores.json
    run_dir = Path(trial_dir) / f"seed_{seed}" / "distmap" / "0" / "euclideanizer" / "0"
    scores_path = run_dir / SCORING_DIR / SCORES_FILENAME
    if not scores_path.is_file():
        return None
    with open(scores_path) as f:
        data = json.load(f)
    overall = data.get("overall_score")
    if overall is None:
        return None
    try:
        return float(overall)
    except (TypeError, ValueError):
        return None


def _append_failure_log(log_path: str, trial_number: int, params: dict, reason: str) -> None:
    with open(log_path, "a") as f:
        f.write(
            f"[{datetime.now().isoformat()}] trial={trial_number} params={json.dumps(params)} reason={reason}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HPO: Optuna joint optimization of DistMap + Euclideanizer (objective = overall score)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to HPO config YAML (e.g. samples/hpo_config.yaml)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset (overrides data_path in HPO config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume study from storage and run additional trials",
    )
    parser.add_argument(
        "--n-trials-add",
        type=int,
        default=None,
        metavar="N",
        help="With --resume: number of additional trials to run",
    )
    args = parser.parse_args()

    if optuna is None:
        print("optuna is required. pip install optuna", file=sys.stderr)
        return 1

    hpo_cfg = _load_yaml(args.config)
    output_root = str(Path(hpo_cfg["output_dir"]).resolve())
    data_path = _resolve_data_path(hpo_cfg, args.data)
    seed = int(hpo_cfg.get("seed", 10))
    pipeline_config_path = hpo_cfg.get("pipeline_config")
    if not pipeline_config_path:
        print("hpo_config must set pipeline_config (path to base pipeline YAML)", file=sys.stderr)
        return 1
    base_config_path = (Path(args.config).resolve().parent / pipeline_config_path).resolve()
    if not base_config_path.is_file():
        base_config_path = (_SCRIPT_DIR / pipeline_config_path).resolve()
    if not base_config_path.is_file():
        print(f"Base pipeline config not found: {pipeline_config_path}", file=sys.stderr)
        return 1
    base_cfg = _load_yaml(str(base_config_path))
    search_space = hpo_cfg.get("search_space") or {}
    optuna_cfg = hpo_cfg.get("optuna") or {}
    epoch_cap = hpo_cfg.get("epoch_cap")
    if epoch_cap is not None:
        epoch_cap = int(epoch_cap)
    n_trials = int(optuna_cfg.get("n_trials", 100))
    pruner_name = optuna_cfg.get("pruner", "MedianPruner")
    pruner_kwargs = optuna_cfg.get("pruner_kwargs") or {}
    n_gpus = _get_n_gpus(hpo_cfg)

    # Study storage: default is output_dir/hpo_study.db so the path identifies the run. No study_name config.
    storage_cfg = optuna_cfg.get("storage")
    if storage_cfg:
        storage = storage_cfg
        if storage.startswith("sqlite:///") and not os.path.isabs(storage.replace("sqlite:///", "")):
            storage = f"sqlite:///{Path(output_root) / storage.replace('sqlite:///', '')}"
    else:
        storage = f"sqlite:///{Path(output_root) / 'hpo_study.db'}"
    study_name = "hpo"

    sampler = _build_sampler(optuna_cfg, seed)

    # Resume: load existing study from the same output_dir (same DB). Run --n-trials-add more trials; trial numbers continue.
    if args.resume:
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
            )
        except KeyError:
            print("Resume requested but no study found at storage. Run without --resume first.", file=sys.stderr)
            return 1
        n_trials = args.n_trials_add or 20
    else:
        # Pruner: validation loss is reported each epoch; with file-based SQLite storage, pruning works across workers when n_jobs > 1.
        pruner_cls = getattr(optuna.pruners, pruner_name, None)
        if pruner_cls is None:
            pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
        else:
            pruner = pruner_cls(**pruner_kwargs)
        # Optional: wrap with PatientPruner so we don't prune until validation hasn't improved for pruner_patience steps.
        if optuna_cfg.get("pruner_patient"):
            pruner_patience = int(optuna_cfg.get("pruner_patience", 5))
            pruner = optuna.pruners.PatientPruner(pruner, patience=pruner_patience)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        if args.n_trials_add is not None and not args.resume:
            n_trials = args.n_trials_add

    failed_log_path = Path(output_root) / FAILED_TRIALS_LOG
    Path(output_root).mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        # Round-robin GPU assignment: trial N uses GPU (N % n_gpus). CUDA_VISIBLE_DEVICES set so only that device is visible.
        gpu_id = trial.number % n_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        trial_params = _suggest_params(trial, search_space)
        trial_dir = str((Path(output_root) / f"trial_{trial.number}").resolve())
        cfg = _build_trial_config(base_cfg, trial_params, trial_dir, data_path, seed, epoch_cap=epoch_cap)
        Path(trial_dir).mkdir(parents=True, exist_ok=True)
        config_out = Path(trial_dir) / PIPELINE_CONFIG_FILENAME
        with open(config_out, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        coords, coords_np, num_atoms, num_structures, exp_stats, train_stats, test_stats = _load_trial_data(
            data_path, seed, base_cfg, device
        )
        try:
            score = run_module.run_one_hpo_trial(
                cfg, trial_dir, trial, device,
                coords, coords_np, num_atoms, num_structures,
                exp_stats, train_stats, test_stats, data_path,
            )
        except optuna.TrialPruned:
            raise
        except optuna.TrialFailedException:
            raise
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            _append_failure_log(str(failed_log_path), trial.number, trial_params, reason)
            raise optuna.TrialFailedException(reason) from e

        if score != score or score < 0 or score > 1:
            reason = "overall_score out of [0, 1] or NaN"
            _append_failure_log(str(failed_log_path), trial.number, trial_params, reason)
            raise optuna.TrialFailedException(reason)
        return float(score)

    # Use up to n_gpus workers so each parallel trial gets its own GPU. With file-based SQLite storage,
    # Optuna shares intermediate values (trial.report) across processes so pruning still works. Set n_jobs: 1 in HPO config to force single-GPU (e.g. if pruning behaves oddly).
    n_jobs_cfg = hpo_cfg.get("n_jobs")
    n_jobs = int(n_jobs_cfg) if n_jobs_cfg is not None else min(n_trials, n_gpus)
    n_jobs = max(1, min(n_jobs, n_trials))
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
    except Exception as e:
        traceback.print_exc()
        return 1

    # HPO dashboard: list trials and best
    best = study.best_trial
    print(f"\nBest trial: {best.number}  value={best.value:.6f}")
    print(f"  Params: {best.params}")

    # Write a simple manifest for the HPO dashboard
    dashboard_dir = Path(output_root) / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "trials": [
            {
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "params": t.params,
                "state": str(t.state.name),
            }
            for t in study.trials
        ],
        "best_trial_number": best.number if best else None,
        "n_gpus": n_gpus,
    }
    with open(dashboard_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    index_html = dashboard_dir / "index.html"
    with open(index_html, "w") as f:
        f.write("<!DOCTYPE html><html><head><title>HPO</title></head><body>\n")
        f.write("<h1>HPO results</h1>\n")
        f.write(f"<p>Best trial: {best.number} (value={best.value:.6f})</p>\n" if best else "<p>No completed trials.</p>\n")
        f.write("<table border='1'><tr><th>Trial</th><th>Value</th><th>State</th><th>Link</th></tr>\n")
        for t in study.trials:
            link = f"../trial_{t.number}/seed_{seed}/distmap/0/euclideanizer/0/dashboard/index.html" if t.value is not None else "#"
            f.write(f"<tr><td>{t.number}</td><td>{t.value}</td><td>{t.state.name}</td><td><a href='{link}'>dashboard</a></td></tr>\n")
        f.write("</table></body></html>\n")
    print(f"HPO dashboard: {index_html}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
