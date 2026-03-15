"""
Config loading and grid expansion. Config file is required; all hyperparameters come from it.
Single values or lists in config; lists are expanded into a product of combinations.
"""
from __future__ import annotations

import os
import itertools
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

# Required keys per section (no code-side defaults). Key order is standardized:
# enabled/overwrite_existing first, then behavior params, then save_data, save_pdf_copy, save_structures_gro.
REQUIRED_KEYS = {
    "data": ["path", "split_seed", "training_split", "exp_stats_chunk_size", "exp_stats_avg_map_sample"],
    "distmap": [
        "latent_dim", "beta_kl",
        "epochs", "batch_size", "learning_rate",
        "lambda_mse", "lambda_w_recon", "lambda_w_gen",
        "memory_efficient", "save_final_models_per_stretch",
    ],
    "euclideanizer": [
        "epochs", "batch_size", "learning_rate",
        "lambda_mse", "lambda_w_recon", "lambda_w_gen",
        "lambda_w_diag_recon", "lambda_w_diag_gen",
        "num_diags",
        "lambda_kabsch_mse",
        "memory_efficient", "save_final_models_per_stretch",
    ],
    "plotting": [
        "enabled", "overwrite_existing",
        "reconstruction", "bond_rg_scaling", "avg_gen_vs_exp", "bond_length_by_genomic_distance",
        "num_samples", "gen_decode_batch_size", "sample_variance",
        "num_reconstruction_samples",
        "max_train", "max_test",
        "save_data", "save_pdf_copy", "save_structures_gro",
    ],
    "analysis": [
        "rmsd_max_train", "rmsd_max_test",
        "rmsd_gen", "rmsd_recon",
        "q_max_train", "q_max_test",
        "q_gen", "q_recon",
        "coord_clustering_max_train", "coord_clustering_max_test",
        "coord_clustering_gen", "coord_clustering_recon",
        "distmap_clustering_max_train", "distmap_clustering_max_test",
        "distmap_clustering_gen", "distmap_clustering_recon",
    ],
}
# Required keys inside each analysis sub-block (validated when top-level key exists).
REQUIRED_ANALYSIS_SUBKEYS = {
    "rmsd_gen": [
        "enabled", "overwrite_existing",
        "num_samples", "sample_variance", "query_batch_size", "gen_decode_batch_size",
        "save_data", "save_pdf_copy", "save_structures_gro",
    ],
    "rmsd_recon": [
        "enabled", "overwrite_existing",
        "max_recon_train", "max_recon_test",
        "save_data", "save_pdf_copy",
    ],
    "q_gen": [
        "enabled", "overwrite_existing",
        "num_samples", "sample_variance",
        "delta", "query_batch_size", "gen_decode_batch_size",
        "save_data", "save_pdf_copy", "save_structures_gro",
    ],
    "q_recon": [
        "enabled", "overwrite_existing",
        "max_recon_train", "max_recon_test", "delta",
        "save_data", "save_pdf_copy",
    ],
    "coord_clustering_gen": [
        "enabled", "overwrite_existing",
        "num_samples", "sample_variance", "n_subsample",
        "k_mixing", "linkage_method",
        "gen_decode_batch_size",
        "save_data", "save_pdf_copy",
    ],
    "coord_clustering_recon": [
        "enabled", "overwrite_existing",
        "max_recon_train", "max_recon_test", "n_subsample",
        "k_mixing", "linkage_method",
        "save_data", "save_pdf_copy",
    ],
    "distmap_clustering_gen": [
        "enabled", "overwrite_existing",
        "num_samples", "sample_variance", "n_subsample",
        "k_mixing", "linkage_method",
        "gen_decode_batch_size", "feats_batch_size",
        "save_data", "save_pdf_copy",
    ],
    "distmap_clustering_recon": [
        "enabled", "overwrite_existing",
        "max_recon_train", "max_recon_test", "n_subsample",
        "k_mixing", "linkage_method",
        "feats_batch_size",
        "save_data", "save_pdf_copy",
    ],
}
REQUIRED_KEYS = {
    **{k: v for k, v in REQUIRED_KEYS.items() if k not in ("analysis", "data")},
    "data": ["path", "split_seed", "training_split", "exp_stats_chunk_size", "exp_stats_avg_map_sample"],
    "analysis": [
        "rmsd_max_train", "rmsd_max_test",
        "rmsd_gen", "rmsd_recon",
        "q_max_train", "q_max_test",
        "q_gen", "q_recon",
        "coord_clustering_max_train", "coord_clustering_max_test",
        "coord_clustering_gen", "coord_clustering_recon",
        "distmap_clustering_max_train", "distmap_clustering_max_test",
        "distmap_clustering_gen", "distmap_clustering_recon",
        "latent_max_train", "latent_max_test",
        "latent",
    ],
    "training_visualization": [
        "enabled",
        "n_probe", "n_quick", "gen_sample_variance",
        "fps", "frame_width", "frame_height", "frame_dpi",
        "delete_frames_after_video",
    ],
    "dashboard": ["enabled"],
    "scoring": ["enabled", "overwrite_existing", "save_pdf_copy"],
}
# Required keys inside analysis.latent (latent is its own block: one plot per run; uses analysis.latent_max_train / latent_max_test).
REQUIRED_ANALYSIS_SUBKEYS["latent"] = [
    "enabled", "overwrite_existing",
    "save_data", "save_pdf_copy",
]
# Order: training_visualization before plotting so training-related config is grouped.
# calibration_memory_fraction: required; null only when both distmap and euclideanizer batch_size are positive integers; else float in (0, 1] for auto-calibration.
# calibration_training_batch_cap / calibration_decode_batch_cap: required; positive int; upper bound for binary search when calibrating training vs inference batch sizes.
REQUIRED_TOP_LEVEL = ["resume", "data", "output_dir", "calibration_memory_fraction", "calibration_training_batch_cap", "calibration_decode_batch_cap", "distmap", "euclideanizer", "training_visualization", "plotting", "analysis", "dashboard", "scoring"]

# Sections that must match exactly when resuming (training and training visualization).
TRAINING_CRITICAL_KEYS = ["data", "distmap", "euclideanizer", "training_visualization", "calibration_memory_fraction", "calibration_training_batch_cap", "calibration_decode_batch_cap"]
# Sections that may differ on resume; if they do, user is prompted and plotting/analysis outputs are removed and re-run.
PLOTTING_ANALYSIS_KEYS = ["plotting", "analysis", "scoring"]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate_config(cfg: Dict[str, Any]) -> None:
    """Raise KeyError with a clear message if any required key is missing."""
    missing = []
    for top in REQUIRED_TOP_LEVEL:
        if top not in cfg:
            missing.append(top)
            continue
        if top in REQUIRED_KEYS:
            for key in REQUIRED_KEYS[top]:
                if not isinstance(cfg[top], dict) or key not in cfg[top]:
                    missing.append(f"{top}.{key}")
            # Nested validation for analysis sub-blocks
            if top == "analysis" and isinstance(cfg.get("analysis"), dict):
                for block_name, sub_keys in REQUIRED_ANALYSIS_SUBKEYS.items():
                    block = cfg["analysis"].get(block_name)
                    if not isinstance(block, dict):
                        missing.append(f"{top}.{block_name} (must be a dict)")
                    else:
                        for sk in sub_keys:
                            if sk not in block:
                                missing.append(f"{top}.{block_name}.{sk}")
    if missing:
        raise KeyError(
            "Config is missing required keys (set them in your config file): " + ", ".join(missing)
        )
    # batch_size: no list; if not null, must be a positive integer. null = auto-calibrate at run time.
    for section in ("distmap", "euclideanizer"):
        if section in cfg and isinstance(cfg[section], dict):
            bs = cfg[section].get("batch_size")
            if isinstance(bs, list):
                raise ValueError(
                    f"{section}.batch_size must be a single integer or null, not a list. "
                    "Use one value, e.g. batch_size: 32, or null for auto-calibration."
                )
            if bs is not None and (not isinstance(bs, int) or bs < 1):
                raise ValueError(
                    f"{section}.batch_size must be null (auto-calibrate) or a positive integer, got {bs!r}."
                )
    # plotting.gen_decode_batch_size and analysis *.*.query_batch_size / gen_decode_batch_size: no list; if not null, positive int. null = in-run calibrate.
    def _validate_batch_key(val: Any, path: str) -> None:
        if isinstance(val, list):
            raise ValueError(f"{path} must be a single integer or null, not a list.")
        if val is not None and (not isinstance(val, int) or val < 1):
            raise ValueError(f"{path} must be null (auto-calibrate) or a positive integer, got {val!r}.")

    plot_cfg = cfg.get("plotting") or {}
    if isinstance(plot_cfg, dict) and "gen_decode_batch_size" in plot_cfg:
        _validate_batch_key(plot_cfg["gen_decode_batch_size"], "plotting.gen_decode_batch_size")
    for block_name, sub_keys in REQUIRED_ANALYSIS_SUBKEYS.items():
        block = (cfg.get("analysis") or {}).get(block_name)
        if not isinstance(block, dict):
            continue
        for key in ("query_batch_size", "gen_decode_batch_size"):
            if key in block:
                _validate_batch_key(block[key], f"analysis.{block_name}.{key}")

    # calibration_memory_fraction: required when any calibratable batch size is null or 0.
    frac = cfg.get("calibration_memory_fraction")
    dm_bs = (cfg.get("distmap") or {}).get("batch_size")
    eu_bs = (cfg.get("euclideanizer") or {}).get("batch_size")
    needs_calibration = (dm_bs is None or dm_bs == 0 or eu_bs is None or eu_bs == 0)
    if not needs_calibration and isinstance(plot_cfg, dict) and plot_cfg.get("gen_decode_batch_size") is None:
        needs_calibration = True
    if not needs_calibration:
        for block_name, sub_keys in REQUIRED_ANALYSIS_SUBKEYS.items():
            block = (cfg.get("analysis") or {}).get(block_name)
            if not isinstance(block, dict):
                continue
            if "query_batch_size" in sub_keys and block.get("query_batch_size") is None:
                needs_calibration = True
                break
            if "gen_decode_batch_size" in sub_keys and block.get("gen_decode_batch_size") is None:
                needs_calibration = True
                break
    if needs_calibration:
        if frac is None:
            raise KeyError(
                "calibration_memory_fraction is required when any of distmap.batch_size, euclideanizer.batch_size, "
                "plotting.gen_decode_batch_size, or analysis *.*.query_batch_size/gen_decode_batch_size is null or 0. "
                "Set a float in (0, 1], e.g. 0.85 for 85%% GPU memory target."
            )
        if not isinstance(frac, (int, float)) or frac <= 0 or frac > 1:
            raise ValueError(
                f"calibration_memory_fraction must be a float in (0, 1] when using auto-calibration, got {frac!r}."
            )
    else:
        if frac is not None and not isinstance(frac, (int, float)):
            raise ValueError(
                f"calibration_memory_fraction must be null or a float in (0, 1], got {type(frac).__name__}."
            )
    for key in ("calibration_training_batch_cap", "calibration_decode_batch_cap"):
        val = cfg.get(key)
        if val is None or not isinstance(val, int) or val < 1:
            raise ValueError(
                f"{key} must be a positive integer (upper bound for calibration binary search), got {val!r}."
            )


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate that cfg has all required pipeline keys. Raises KeyError or ValueError if not. Use for in-memory configs (e.g. HPO trial config)."""
    _validate_config(cfg)


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return [None]
    if isinstance(x, list):
        return x
    return [x]


def load_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load config from YAML. Path is required; file must exist. Overrides (e.g. from CLI) are merged. Validates required keys."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "Config file is required. Use --config path/to/config.yaml (e.g. samples/config_sample.yaml)."
        )
    if yaml is None:
        raise RuntimeError("PyYAML is required. pip install pyyaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a YAML object (key-value).")
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    _validate_config(cfg)
    return cfg


def _expand_grid(section: dict, keys: List[str]) -> List[dict]:
    """Expand section into list of configs (product over all keys; each key normalized to list)."""
    lists = {k: _ensure_list(section[k]) for k in keys}
    if not lists:
        return [dict(section)]
    key_order = list(lists.keys())
    values = [lists[k] for k in key_order]
    out = []
    for combo in itertools.product(*values):
        c = dict(section)
        for i, k in enumerate(key_order):
            c[k] = combo[i]
        out.append(c)
    return out


def expand_distmap_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of DistMap configs (one per combination). batch_size is single-value only (excluded from grid)."""
    section = cfg["distmap"]
    keys = [k for k in section.keys() if k != "batch_size"]
    return _expand_grid(section, keys)


def expand_euclideanizer_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of Euclideanizer configs (one per combination). batch_size is single-value only (excluded from grid)."""
    section = cfg["euclideanizer"]
    keys = [k for k in section.keys() if k != "batch_size"]
    return _expand_grid(section, keys)


def _training_groups(
    expanded_configs: List[Dict[str, Any]],
    epoch_key: str,
) -> List[Dict[str, Any]]:
    """
    Group configs by all keys except epoch_key. For each group, train once to max(epochs)
    and checkpoint at each epoch value. Returns list of:
      { "base_config": config with epochs=max, "checkpoints": [(run_idx, epoch_val), ...] }
    with checkpoints sorted by epoch_val so run indices align with increasing epochs.
    """
    if not expanded_configs:
        return []
    keys = list(expanded_configs[0].keys())
    if epoch_key not in keys:
        return [{"base_config": c, "checkpoints": [(i, c.get(epoch_key, 1))]} for i, c in enumerate(expanded_configs)]

    def group_key(c: Dict[str, Any]) -> tuple:
        return tuple((k, c[k]) for k in sorted(keys) if k != epoch_key)

    grouped: Dict[tuple, List[Tuple[int, Dict[str, Any]]]] = {}
    for run_idx, c in enumerate(expanded_configs):
        k = group_key(c)
        grouped.setdefault(k, []).append((run_idx, c))

    out = []
    for gkey, group in grouped.items():
        group_sorted = sorted(group, key=lambda x: x[1][epoch_key])
        run_indices = [x[0] for x in group_sorted]
        epoch_values = [x[1][epoch_key] for x in group_sorted]
        base = dict(group_sorted[0][1])
        base[epoch_key] = max(epoch_values)
        out.append({
            "base_config": base,
            "checkpoints": list(zip(run_indices, epoch_values)),
        })
    return out


def distmap_training_groups(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group DistMap configs by (all except epochs). Each group trains once to max(epochs) with checkpoints at each value."""
    configs = expand_distmap_grid(cfg)
    return _training_groups(configs, "epochs")


def euclideanizer_training_groups(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group Euclideanizer configs by (all except epochs). Each group trains once to max(epochs) with checkpoints at each value."""
    configs = expand_euclideanizer_grid(cfg)
    return _training_groups(configs, "epochs")


def get_sample_variances(cfg: Dict[str, Any]) -> List[float]:
    """Return list of sample variance values (for generation plots)."""
    v = cfg["plotting"]["sample_variance"]
    return _ensure_list(v)


def get_data_path(cfg: Dict[str, Any]) -> Optional[str]:
    """Return dataset path from config (may be None)."""
    p = cfg["data"]["path"]
    if isinstance(p, list):
        p = p[0] if p else None
    if p is not None:
        return os.path.abspath(str(p))
    return None


def get_output_dir(cfg: Dict[str, Any]) -> str:
    return os.path.abspath(cfg["output_dir"])


def get_seeds(cfg: Dict[str, Any]) -> List[int]:
    """Return list of seeds (1 per full pipeline run). data.split_seed can be a single int or a list of ints."""
    data = cfg["data"]
    split_seed = data["split_seed"]
    out = _ensure_list(split_seed)
    return [int(s) for s in out]


# Filename for the full pipeline config saved in output_dir (required for resume).
PIPELINE_CONFIG_FILENAME = "pipeline_config.yaml"


def _to_serializable(obj: Any) -> Any:
    """Convert to YAML-serializable types (no numpy/torch)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _config_deep_equal(a: Any, b: Any) -> bool:
    """Deep equality for configs (normalizes types for YAML round-trip)."""
    a = _to_serializable(a)
    b = _to_serializable(b)
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_config_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_config_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def configs_match_exactly(cfg1: Dict[str, Any], cfg2: Dict[str, Any]) -> bool:
    """True if the two pipeline configs are identical (for resume safety)."""
    return _config_deep_equal(cfg1, cfg2)


def configs_match_sections(cfg1: Dict[str, Any], cfg2: Dict[str, Any], top_level_keys: List[str]) -> bool:
    """True if the two configs match on the given top-level keys (deep equality). Keys missing in either config count as mismatch."""
    for k in top_level_keys:
        if k not in cfg1 or k not in cfg2:
            return False
        if not _config_deep_equal(cfg1[k], cfg2[k]):
            return False
    return True


def config_diff(cfg1: Dict[str, Any], cfg2: Dict[str, Any], prefix: str = "") -> List[str]:
    """Return a list of paths where the two configs differ (for error messages)."""
    a = _to_serializable(cfg1)
    b = _to_serializable(cfg2)
    diffs: List[str] = []
    if type(a) != type(b):
        diffs.append(f"{prefix}: type mismatch (saved {type(a).__name__} vs current {type(b).__name__})")
        return diffs
    if isinstance(a, dict):
        all_keys = set(a.keys()) | set(b.keys())
        for k in sorted(all_keys):
            path = f"{prefix}.{k}" if prefix else k
            if k not in a:
                diffs.append(f"{path}: missing in saved config (current has {repr(b[k])[:60]})")
            elif k not in b:
                diffs.append(f"{path}: present in saved config (current omits it)")
            else:
                diffs.extend(config_diff(a[k], b[k], path))
        return diffs
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{prefix}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(config_diff(x, y, f"{prefix}[{i}]"))
        if len(a) != len(b):
            return diffs
        return diffs
    if a != b:
        sa, sb = str(a)[:80], str(b)[:80]
        diffs.append(f"{prefix}: saved {sa!r} != current {sb!r}")
    return diffs


def run_config_section_matches(run_cfg: Optional[Dict[str, Any]], section_key: str, expected_section: Dict[str, Any]) -> bool:
    """True if run_cfg has section_key and it deep-equals expected_section (catches swapped run dirs)."""
    if run_cfg is None or section_key not in run_cfg:
        return False
    return _config_deep_equal(run_cfg[section_key], expected_section)


def load_run_config(directory: str, filename: str = "run_config.yaml") -> Optional[Dict[str, Any]]:
    """Load run_config from directory/filename. Returns None if file missing or invalid."""
    if yaml is None:
        raise RuntimeError("PyYAML is required. pip install pyyaml")
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            out = yaml.safe_load(f)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def pipeline_config_path(output_dir: str) -> str:
    return os.path.join(output_dir, PIPELINE_CONFIG_FILENAME)


def load_pipeline_config(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load pipeline config from output_dir. Returns None if missing or invalid."""
    return load_run_config(output_dir, PIPELINE_CONFIG_FILENAME)


def save_pipeline_config(cfg: Dict[str, Any], output_dir: str) -> str:
    """Write full pipeline config to output_dir (exact copy for resume matching)."""
    if yaml is None:
        raise RuntimeError("PyYAML is required. pip install pyyaml")
    os.makedirs(output_dir, exist_ok=True)
    path = pipeline_config_path(output_dir)
    with open(path, "w") as f:
        yaml.dump(_to_serializable(cfg), f, default_flow_style=False, sort_keys=False)
    return path


def save_run_config(
    run_cfg: Dict[str, Any],
    directory: str,
    filename: str = "run_config.yaml",
    *,
    last_epoch_trained: Optional[int] = None,
    best_epoch: Optional[int] = None,
    best_val: Optional[float] = None,
    early_stopped: bool = False,
) -> str:
    """Write the config for this run to directory/filename. Always writes last_epoch_trained, best_epoch (1-indexed), best_val (use None when not applicable). early_stopped=True when training was stopped by validation-loss patience."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to save run config. pip install pyyaml")
    os.makedirs(directory, exist_ok=True)
    out = dict(run_cfg)
    # Always write these keys so run config has a consistent schema (no backwards-compat branches when loading).
    out["last_epoch_trained"] = last_epoch_trained
    out["best_epoch"] = best_epoch
    out["best_val"] = best_val
    out["early_stopped"] = early_stopped
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        yaml.dump(_to_serializable(out), f, default_flow_style=False, sort_keys=False)
    return path
