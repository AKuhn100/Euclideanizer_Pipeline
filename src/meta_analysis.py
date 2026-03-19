from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


@dataclass
class SufficiencyPoint:
    seed: int
    max_data: int | None
    training_split: float
    rmsd_values: np.ndarray
    q_values: np.ndarray


def _parse_seed_and_split(seed_name: str) -> tuple[int, float | None] | None:
    if not seed_name.startswith("seed_"):
        return None
    rest = seed_name[5:]
    if rest.isdigit() or "_maxdata_" in rest and rest.split("_maxdata_", 1)[0].isdigit():
        left = rest.split("_maxdata_", 1)[0]
        return int(left), None
    if "_split_" in rest:
        left, right = rest.split("_split_", 1)
        if "_maxdata_" in right:
            right = right.split("_maxdata_", 1)[0]
        if left.isdigit():
            try:
                return int(left), float(right)
            except ValueError:
                return None
    return None


def _iter_euclideanizer_runs(seed_dir: str):
    distmap_dir = os.path.join(seed_dir, "distmap")
    if not os.path.isdir(distmap_dir):
        return
    for dm_name in sorted(os.listdir(distmap_dir), key=lambda x: (len(x), x)):
        if not dm_name.isdigit():
            continue
        eu_dir = os.path.join(distmap_dir, dm_name, "euclideanizer")
        if not os.path.isdir(eu_dir):
            continue
        for eu_name in sorted(os.listdir(eu_dir), key=lambda x: (len(x), x)):
            if not eu_name.isdigit():
                continue
            run_dir = os.path.join(eu_dir, eu_name)
            if os.path.isdir(run_dir):
                yield run_dir


def _run_name_has_var_one(run_name: str) -> bool:
    if "_var" in run_name:
        suffix = run_name.split("_var", 1)[-1].strip()
        try:
            return abs(float(suffix) - 1.0) < 1e-9
        except ValueError:
            return False
    if run_name.startswith("var"):
        try:
            return abs(float(run_name[3:].strip()) - 1.0) < 1e-9
        except ValueError:
            return False
    return False


def _load_gen_metric_array(seed_dir: str, metric: str) -> np.ndarray | None:
    filename = "rmsd_data.npz" if metric == "rmsd" else "q_data.npz"
    subdir = "rmsd" if metric == "rmsd" else "q"
    for run_dir in _iter_euclideanizer_runs(seed_dir) or []:
        gen_dir = os.path.join(run_dir, "analysis", subdir, "gen")
        if not os.path.isdir(gen_dir):
            continue
        for run_name in sorted(os.listdir(gen_dir)):
            data_path = os.path.join(gen_dir, run_name, "data", filename)
            if not os.path.isfile(data_path):
                continue
            if not _run_name_has_var_one(run_name):
                continue
            try:
                loaded = np.load(data_path, allow_pickle=False)
                vals = np.asarray(loaded["gen_to_test"], dtype=np.float32)
                loaded.close()
            except Exception:
                continue
            return vals
    return None


def _kde_curve(values: np.ndarray, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    # Lightweight KDE fallback that avoids scipy hard dependency in this module.
    # Uses Gaussian kernel with Scott-like bandwidth.
    x = np.linspace(x_min, x_max, 400, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64).ravel()
    if len(v) < 2:
        return x, np.zeros_like(x)
    std = np.std(v)
    bw = (std if std > 1e-12 else 1e-6) * (len(v) ** (-1.0 / 5.0))
    bw = max(bw, 1e-6)
    diffs = (x[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * diffs * diffs).sum(axis=1) / (len(v) * bw * np.sqrt(2.0 * np.pi))
    return x, dens


def _max_data_from_seed_dir(seed_dir: str, fallback_max_data: int | None) -> int | None:
    cfg_path = os.path.join(seed_dir, "pipeline_config.yaml")
    if not os.path.isfile(cfg_path):
        return fallback_max_data
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        md = (cfg.get("data") or {}).get("max_data")
        if isinstance(md, list):
            return int(md[0]) if md else fallback_max_data
        if md is None:
            return None
        return int(md)
    except Exception:
        return fallback_max_data


def _save_pdf_if_enabled(fig, png_path: str, enabled: bool) -> None:
    if not enabled:
        return
    base, _ = os.path.splitext(png_path)
    fig.savefig(base + ".pdf")


def run_sufficiency_meta_analysis(
    *,
    base_output_dir: str,
    max_data_values: list[int | None],
    save_pdf_copy: bool,
    log: Callable[[str], None] | None = None,
) -> bool:
    """Build sufficiency distribution and heatmap figures under meta_analysis/sufficiency.

    Returns True when at least one figure is created.
    """
    points_by_seed: dict[int, list[SufficiencyPoint]] = {}
    fallback_md = max_data_values[0] if max_data_values else None
    for seed_name in sorted(os.listdir(base_output_dir) if os.path.isdir(base_output_dir) else []):
        parsed = _parse_seed_and_split(seed_name)
        if parsed is None:
            continue
        seed, split_opt = parsed
        seed_dir = os.path.join(base_output_dir, seed_name)
        if not os.path.isdir(seed_dir):
            continue
        split = float(split_opt) if split_opt is not None else 0.8
        rmsd_vals = _load_gen_metric_array(seed_dir, "rmsd")
        q_vals = _load_gen_metric_array(seed_dir, "q")
        if rmsd_vals is None or q_vals is None:
            if log:
                log(f"Sufficiency meta-analysis: missing RMSD/Q NPZ for {seed_name}; skipping.")
            continue
        md = _max_data_from_seed_dir(seed_dir, fallback_md)
        points_by_seed.setdefault(seed, []).append(
            SufficiencyPoint(
                seed=seed,
                max_data=md,
                training_split=split,
                rmsd_values=rmsd_vals,
                q_values=q_vals,
            )
        )

    if not points_by_seed:
        if log:
            log("Sufficiency meta-analysis: no analysis NPZ data found. Skipping.")
        return False

    out_root = os.path.join(base_output_dir, "meta_analysis", "sufficiency")
    os.makedirs(out_root, exist_ok=True)
    made_any = False
    for seed, points in sorted(points_by_seed.items()):
        seed_out = os.path.join(out_root, f"seed_{seed}")
        dist_root = os.path.join(seed_out, "distributions")
        heat_root = os.path.join(seed_out, "heatmap")
        os.makedirs(dist_root, exist_ok=True)
        os.makedirs(heat_root, exist_ok=True)

        max_data_set = sorted({p.max_data for p in points}, key=lambda x: (-1 if x is None else x))
        split_set = sorted({p.training_split for p in points})
        split_norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = cm.get_cmap("viridis")

        for md in max_data_set:
            subset = [p for p in points if p.max_data == md]
            if not subset:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            ax_r, ax_q = axes
            all_r = np.concatenate([p.rmsd_values for p in subset], axis=0)
            all_q = np.concatenate([p.q_values for p in subset], axis=0)
            xr_min, xr_max = np.percentile(all_r, [1, 99])
            xq_min, xq_max = np.percentile(all_q, [1, 99])
            xq_min, xq_max = max(0.0, xq_min), min(1.0, xq_max)

            for p in sorted(subset, key=lambda x: x.training_split):
                color = cmap(split_norm(p.training_split))
                xr, dr = _kde_curve(p.rmsd_values, float(xr_min), float(xr_max))
                xq, dq = _kde_curve(p.q_values, float(xq_min), float(xq_max))
                label = f"{int(round(p.training_split * 100))}%"
                ax_r.plot(xr, dr, color=color, label=label)
                ax_r.fill_between(xr, 0.0, dr, color=color, alpha=0.12)
                ax_q.plot(xq, dq, color=color, label=label)
                ax_q.fill_between(xq, 0.0, dq, color=color, alpha=0.12)

            ax_r.set_title(f"Min RMSD | max_data={md}")
            ax_r.set_xlabel("Min RMSD (A)")
            ax_r.set_ylabel("Density")
            ax_q.set_title(f"Max Q | max_data={md}")
            ax_q.set_xlabel("Max Q")
            ax_q.set_ylabel("Density")
            sm = cm.ScalarMappable(norm=split_norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.08, pad=0.15)
            cbar.set_label("Training Split")
            cbar.set_ticks(split_set)
            cbar.set_ticklabels([f"{int(round(s * 100))}%" for s in split_set])
            fig.tight_layout()
            md_tag = "all" if md is None else str(md)
            out_dir = os.path.join(dist_root, f"max_data_{md_tag}")
            os.makedirs(out_dir, exist_ok=True)
            png_path = os.path.join(out_dir, "distributions_rmsd_q.png")
            fig.savefig(png_path)
            _save_pdf_if_enabled(fig, png_path, save_pdf_copy)
            plt.close(fig)
            made_any = True

        # Seed heatmap
        md_vals = [m for m in max_data_set if m is not None]
        if not md_vals:
            continue
        md_to_j = {m: j for j, m in enumerate(md_vals)}
        split_to_i = {s: i for i, s in enumerate(split_set)}
        rmsd_grid = np.full((len(split_set), len(md_vals)), np.nan, dtype=np.float32)
        q_grid = np.full((len(split_set), len(md_vals)), np.nan, dtype=np.float32)
        for p in points:
            if p.max_data is None:
                continue
            i = split_to_i[p.training_split]
            j = md_to_j[p.max_data]
            rmsd_grid[i, j] = float(np.median(p.rmsd_values))
            q_grid[i, j] = float(np.median(p.q_values))
        rmsd_plot = np.clip(rmsd_grid, 0.0, 1.0)
        q_plot = np.clip(q_grid, 0.0, 1.0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax_r, ax_q = axes
        ax_r.imshow(rmsd_plot, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax_q.imshow(q_plot, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        for ax, title in ((ax_r, "Median Min RMSD"), (ax_q, "Median Max Q")):
            ax.set_title(title)
            ax.set_xlabel("max_data")
            ax.set_xticks(np.arange(len(md_vals)))
            ax.set_xticklabels([str(v) for v in md_vals], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(split_set)))
            ax.set_yticklabels([f"{int(round(s * 100))}%" for s in split_set])
            ax.set_ylabel("Training Split")
        sm = cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.08, pad=0.15)
        cbar.set_label("Value (0-1)")
        fig.tight_layout()
        heat_png = os.path.join(heat_root, "sufficiency_heatmap_rmsd_q.png")
        fig.savefig(heat_png)
        _save_pdf_if_enabled(fig, heat_png, save_pdf_copy)
        plt.close(fig)
        made_any = True

    return made_any

