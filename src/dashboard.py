"""
Build an interactive HTML dashboard in the run root.

Scans base_output_dir for seed_*/distmap/* and seed_*/distmap/*/euclideanizer/*,
collects run_config labels, discovers plots/videos/analysis outputs,
copies them into dashboard/assets/, and writes dashboard/manifest.json and dashboard/index.html.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Optional

from .config import load_run_config

# Path patterns (aligned with run.py; no import from run to avoid circular deps)
_PLOTS_BASE = "plots"
_RECONSTRUCTION = os.path.join(_PLOTS_BASE, "reconstruction", "reconstruction.png")
_RECON_STAT_TRAIN = os.path.join(_PLOTS_BASE, "recon_statistics", "recon_statistics_train.png")
_RECON_STAT_TEST = os.path.join(_PLOTS_BASE, "recon_statistics", "recon_statistics_test.png")
_GEN_VARIANCE_PATTERN = os.path.join(_PLOTS_BASE, "gen_variance", "gen_variance_*.png")
# Training video lives outside plots/ so plotting wipe does not remove it
_TRAINING_VIDEO = os.path.join("training_video", "training_evolution.mp4")
_ANALYSIS_MIN_RMSD_DIR = "analysis"
_MIN_RMSD_FIG = "min_rmsd_distributions.png"

DASHBOARD_DIR = "dashboard"
ASSETS_DIR = "assets"
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "index.html"


def _run_config_dir(run_root: str) -> str:
    return os.path.join(run_root, "model")


def _label_from_distmap_config(cfg: Optional[dict], seed: int, dm_index: int) -> tuple[str, str]:
    if not cfg or "distmap" not in cfg:
        return f"Seed {seed} · DistMap {dm_index}", f"seed_{seed} distmap {dm_index}"
    d = cfg["distmap"]
    parts = [f"β_kl={d.get('beta_kl')}", f"epochs={d.get('epochs')}", f"latent_dim={d.get('latent_dim')}"]
    short = f"Seed {seed} · DM {dm_index} " + ", ".join(str(p) for p in parts if p.split("=")[-1] not in (None, ""))
    long_parts = [f"seed={seed}", f"distmap_index={dm_index}"] + [f"distmap.{k}={v}" for k, v in (d or {}).items()]
    return short, " ".join(long_parts)


def _label_from_euclideanizer_config(cfg: Optional[dict], seed: int, dm_index: int, eu_index: int) -> tuple[str, str]:
    if not cfg or "euclideanizer" not in cfg:
        return f"Seed {seed} · DM {dm_index} · Eu {eu_index}", f"seed_{seed}_dm_{dm_index}_eu_{eu_index}"
    e = cfg["euclideanizer"]
    parts = [f"epochs={e.get('epochs')}", f"num_diags={e.get('num_diags')}"]
    short = f"Seed {seed} · DM {dm_index} · Eu {eu_index} " + ", ".join(str(p) for p in parts if p.split("=")[-1] not in (None, ""))
    long_parts = [f"seed={seed}", f"dm={dm_index}", f"eu={eu_index}"] + [f"euclideanizer.{k}={v}" for k, v in (e or {}).items()]
    return short, " ".join(long_parts)


def _blocks_for_distmap_run(run_root: str) -> list[dict[str, str]]:
    blocks = []
    if os.path.isfile(os.path.join(run_root, _RECONSTRUCTION)):
        blocks.append({"type": "reconstruction", "name": "Reconstruction", "source_path": _RECONSTRUCTION})
    if os.path.isfile(os.path.join(run_root, _RECON_STAT_TRAIN)):
        blocks.append({"type": "recon_statistics", "name": "Recon statistics (train)", "source_path": _RECON_STAT_TRAIN})
    if os.path.isfile(os.path.join(run_root, _RECON_STAT_TEST)):
        blocks.append({"type": "recon_statistics", "name": "Recon statistics (test)", "source_path": _RECON_STAT_TEST})
    gen_dir = os.path.join(run_root, _PLOTS_BASE, "gen_variance")
    if os.path.isdir(gen_dir):
        for p in sorted(glob.glob(os.path.join(gen_dir, "gen_variance_*.png"))):
            base = os.path.basename(p)
            var = base.replace("gen_variance_", "").replace(".png", "")
            rel = os.path.join(_PLOTS_BASE, "gen_variance", base)
            blocks.append({"type": "gen_variance", "name": f"Gen variance {var}", "source_path": rel})
    if os.path.isfile(os.path.join(run_root, _TRAINING_VIDEO)):
        blocks.append({"type": "training_video", "name": "Training video", "source_path": _TRAINING_VIDEO})
    return blocks


def _blocks_for_euclideanizer_run(run_root: str) -> list[dict[str, str]]:
    blocks = _blocks_for_distmap_run(run_root)
    min_rmsd_root = os.path.join(run_root, _ANALYSIS_MIN_RMSD_DIR, "min_rmsd")
    if os.path.isdir(min_rmsd_root):
        gen_dir = os.path.join(min_rmsd_root, "gen")
        if os.path.isdir(gen_dir):
            for run_name in sorted(os.listdir(gen_dir)):
                fig_path = os.path.join(gen_dir, run_name, _MIN_RMSD_FIG)
                if os.path.isfile(fig_path):
                    rel = os.path.join(_ANALYSIS_MIN_RMSD_DIR, "min_rmsd", "gen", run_name, _MIN_RMSD_FIG)
                    blocks.append({"type": "min_rmsd_gen", "name": f"Min-RMSD (gen) {run_name}", "source_path": rel})
        recon_dir = os.path.join(min_rmsd_root, "recon")
        recon_fig = os.path.join(recon_dir, _MIN_RMSD_FIG)
        if os.path.isfile(recon_fig):
            rel = os.path.join(_ANALYSIS_MIN_RMSD_DIR, "min_rmsd", "recon", _MIN_RMSD_FIG)
            blocks.append({"type": "min_rmsd_recon", "name": "Min-RMSD (recon)", "source_path": rel})
        else:
            for subdir in (sorted(os.listdir(recon_dir)) if os.path.isdir(recon_dir) else []):
                subdir_path = os.path.join(recon_dir, subdir)
                if os.path.isdir(subdir_path):
                    subdir_fig = os.path.join(subdir_path, _MIN_RMSD_FIG)
                    if os.path.isfile(subdir_fig):
                        rel = os.path.join(_ANALYSIS_MIN_RMSD_DIR, "min_rmsd", "recon", subdir, _MIN_RMSD_FIG)
                        blocks.append({"type": "min_rmsd_recon", "name": f"Min-RMSD (recon) {subdir}", "source_path": rel})
        latent_fig = os.path.join(recon_dir, "latent_distribution.png")
        if os.path.isfile(latent_fig):
            rel = os.path.join(_ANALYSIS_MIN_RMSD_DIR, "min_rmsd", "recon", "latent_distribution.png")
            blocks.append({"type": "latent_distribution", "name": "Latent distribution", "source_path": rel})
        else:
            for subdir in (sorted(os.listdir(recon_dir)) if os.path.isdir(recon_dir) else []):
                subdir_path = os.path.join(recon_dir, subdir)
                latent_sub = os.path.join(subdir_path, "latent_distribution.png")
                if os.path.isfile(latent_sub):
                    rel = os.path.join(_ANALYSIS_MIN_RMSD_DIR, "min_rmsd", "recon", subdir, "latent_distribution.png")
                    blocks.append({"type": "latent_distribution", "name": f"Latent distribution {subdir}", "source_path": rel})
    return blocks


def _scan_runs(base_output_dir: str) -> list[dict[str, Any]]:
    runs = []
    if not os.path.isdir(base_output_dir):
        return runs

    for seed_name in sorted(os.listdir(base_output_dir)):
        if not re.match(r"^seed_\d+$", seed_name):
            continue
        seed_dir = os.path.join(base_output_dir, seed_name)
        if not os.path.isdir(seed_dir):
            continue
        try:
            seed_num = int(seed_name.split("_")[1])
        except (IndexError, ValueError):
            seed_num = 0
        distmap_dir = os.path.join(seed_dir, "distmap")
        if not os.path.isdir(distmap_dir):
            continue

        seed_id = f"seed_{seed_num}"
        seed_children = []
        seed_blocks = []

        for dm_name in sorted(os.listdir(distmap_dir), key=lambda x: (len(x), x)):
            if not dm_name.isdigit():
                continue
            dm_index = int(dm_name)
            dm_run_root = os.path.join(distmap_dir, dm_name)
            dm_model_dir = _run_config_dir(dm_run_root)
            dm_cfg = load_run_config(dm_model_dir)
            label_short, label_long = _label_from_distmap_config(dm_cfg, seed_num, dm_index)

            dm_id = f"seed_{seed_num}_dm_{dm_index}"
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
                    eu_short, eu_long = _label_from_euclideanizer_config(eu_cfg, seed_num, dm_index, eu_index)
                    eu_id = f"{dm_id}_eu_{eu_index}"
                    eu_blocks = _blocks_for_euclideanizer_run(eu_run_root)
                    runs.append({
                        "id": eu_id,
                        "level": "euclideanizer",
                        "label_short": eu_short,
                        "label_long": eu_long,
                        "parent_id": dm_id,
                        "children_ids": [],
                        "blocks": eu_blocks,
                        "run_root": eu_run_root,
                        "params": eu_cfg.get("euclideanizer", {}) if eu_cfg else {},
                    })
                    dm_children.append(eu_id)
                    for b in eu_blocks:
                        seed_blocks.append({**b, "run_id": eu_id})

            runs.append({
                "id": dm_id,
                "level": "distmap",
                "label_short": label_short,
                "label_long": label_long,
                "parent_id": seed_id,
                "children_ids": dm_children,
                "blocks": dm_blocks,
                "run_root": dm_run_root,
                "params": dm_cfg.get("distmap", {}) if dm_cfg else {},
            })
            seed_children.append(dm_id)
            for b in dm_blocks:
                seed_blocks.append({**b, "run_id": dm_id})

        runs.append({
            "id": seed_id,
            "level": "seed",
            "label_short": f"Seed {seed_num}",
            "label_long": f"seed {seed_num}",
            "parent_id": None,
            "children_ids": seed_children,
            "blocks": seed_blocks,
            "run_root": None,
            "params": {},
        })
    return runs


def _block_asset_slug(block: dict, run_id: str) -> str:
    t = block.get("type", "unknown")
    name = block.get("name", "")
    source = block.get("source_path", "")
    if t == "reconstruction":
        return "reconstruction"
    if t == "recon_statistics":
        return "recon_statistics_train" if "train" in name.lower() else "recon_statistics_test"
    if t == "gen_variance":
        var = source.replace("gen_variance_", "").split(".")[0].split("/")[-1]
        return f"gen_variance_{var}"
    if t == "training_video":
        return "training_video"
    if t == "min_rmsd":
        run_name = source.split(os.sep)[-2] if os.sep in source else "default"
        safe = re.sub(r"[^\w.-]", "_", run_name)
        return f"min_rmsd_{safe}"
    return re.sub(r"[^\w.-]", "_", name.lower().replace(" ", "_"))


def _block_extension(source_path: str) -> str:
    return ".mp4" if source_path.strip().lower().endswith(".mp4") else ".png"


def _copy_assets_and_update_paths(runs: list[dict], assets_dir: str) -> list[dict[str, Any]]:
    """Copy each block's source file to assets_dir (once per unique asset); return runs with blocks containing path (assets/...) only."""
    runs_by_id = {r["id"]: r for r in runs}
    copied = set()
    os.makedirs(assets_dir, exist_ok=True)
    out_runs = []
    for run in runs:
        blocks_out = []
        for block in run["blocks"]:
            owning_id = block.get("run_id") or run["id"]
            owner = runs_by_id.get(owning_id)
            if not owner or not owner.get("run_root"):
                continue
            src = os.path.join(owner["run_root"], block["source_path"])
            if not os.path.isfile(src):
                continue
            slug = _block_asset_slug(block, owning_id)
            ext = _block_extension(block["source_path"])
            asset_name = f"{owning_id}_{slug}{ext}"
            if asset_name not in copied:
                shutil.copy2(src, os.path.join(assets_dir, asset_name))
                copied.add(asset_name)
            blocks_out.append({
                "type": block["type"],
                "name": block["name"],
                "path": f"assets/{asset_name}",
            })
        out_runs.append({
            "id": run["id"],
            "level": run["level"],
            "label_short": run["label_short"],
            "label_long": run["label_long"],
            "parent_id": run["parent_id"],
            "children_ids": run["children_ids"],
            "blocks": blocks_out,
            "params": run.get("params") or {},
        })
    return out_runs


def _make_manifest(base_output_dir: str, runs: list[dict]) -> dict:
    base_path = os.path.basename(os.path.abspath(base_output_dir)) or "output"
    return {
        "generated_at": datetime.now().isoformat(),
        "base_path": base_path,
        "runs": runs,
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
    .view-aspect { display: none; }
    .view-aspect.visible { display: inline-flex; }
    .run-card { background: var(--bg-card); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }
    .run-card-title { font-size: 1.15rem; font-weight: 600; color: var(--text); margin: 0 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }
    .block { margin-bottom: 1.75rem; }
    .block:last-child { margin-bottom: 0; }
    .block-title { display: block; font-size: 1rem; color: var(--text-secondary); margin: 0 0 0.6rem 0; padding: 0.4rem 0.6rem; font-weight: 600; background: #333; border-radius: 4px; border-left: 3px solid var(--accent-block); }
    .compare { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .compare .column { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; background: var(--bg-card); }
    .compare .column .run-card { margin-bottom: 0; }
    .compare .column-header { font-size: 0.9rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.75rem; padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); }
    .content-single { max-width: 56rem; margin: 0 auto; }
    .block img, .block video { max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; }
    .block .placeholder { padding: 2rem; text-align: center; background: #2a2a2a; border-radius: 8px; color: var(--text-muted); }
    .aspect-section { margin-bottom: 2.5rem; scroll-margin-top: 1rem; }
    .aspect-section h3 { font-size: 1.15rem; font-weight: 600; color: var(--text-secondary); margin: 0 0 0.75rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); letter-spacing: 0.02em; }
    .aspect-axis-caption { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; }
    .aspect-axis-table-wrap { overflow-x: auto; }
    .aspect-axis-table { width: 100%; border-collapse: collapse; min-width: 400px; }
    .aspect-axis-table th { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); padding: 0.5rem; text-align: center; border-bottom: 1px solid var(--border); }
    .aspect-axis-table tbody tr:hover { background: rgba(255,255,255,0.02); }
    .aspect-axis-table td { padding: 0.5rem; vertical-align: top; border-bottom: 1px solid #333; }
    .aspect-axis-table td.context-label { font-size: 0.85rem; color: var(--text-secondary); min-width: 11rem; max-width: 14rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .aspect-axis-table .aspect-cell { text-align: center; }
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
    @media (max-width: 768px) {
      .compare { grid-template-columns: 1fr; }
      .content { margin-left: 1rem; margin-right: 1rem; }
      .dashboard-toolbar .controls { gap: 0.75rem; }
      .aspect-section-nav-inner { font-size: 0.8rem; }
    }
  </style>
</head>
<body>
  <a href="#content" class="skip-link">Skip to content</a>
  <header>
    <h1 id="title">Pipeline Dashboard</h1>
    <p id="generated"></p>
  </header>
  <div class="dashboard-toolbar" role="toolbar" aria-label="Dashboard controls">
    <div class="controls">
      <span class="control-group control-group-primary">
        <label for="viewMode">View</label>
        <select id="viewMode" aria-label="View mode: compare runs or vary by parameter">
          <option value="compare">Compare runs</option>
          <option value="aspect">Vary aspect</option>
        </select>
      </span>
      <span class="control-group control-group-sep" aria-hidden="true">|</span>
      <span class="view-compare control-group" id="viewCompare">
        <span class="control-group-label">Compare</span>
        <label for="level">Level</label>
        <select id="level" aria-label="Level: seed, distmap, or euclideanizer"><option value="seed">Seed</option><option value="distmap">DistMap</option><option value="euclideanizer">Euclideanizer</option></select>
        <span id="levelRunCount" class="control-group-label" aria-live="polite"></span>
        <label for="runA">Run A</label>
        <select id="runA" aria-label="Run A"></select>
        <label for="runB">Run B</label>
        <select id="runB" aria-label="Run B (optional)"><option value="">— none —</option></select>
      </span>
      <span class="view-aspect control-group" id="viewAspect">
        <span class="control-group-label">Vary by</span>
        <label for="levelAspect">Level</label>
        <select id="levelAspect" aria-label="Level for vary aspect"><option value="distmap">DistMap</option><option value="euclideanizer">Euclideanizer</option></select>
        <label for="aspect">Aspect (x-axis)</label>
        <select id="aspect" aria-label="Parameter to vary on x-axis"></select>
      </span>
    </div>
  </div>
  <main class="content" id="content" role="main"></main>
  <script>window.__DASHBOARD_MANIFEST__ = """ + manifest_js + """;</script>
  <script>
    let manifest = null;
    const levelEl = document.getElementById('level');
    const runAEl = document.getElementById('runA');
    const runBEl = document.getElementById('runB');
    const contentEl = document.getElementById('content');
    const titleEl = document.getElementById('title');
    const generatedEl = document.getElementById('generated');
    const viewModeEl = document.getElementById('viewMode');
    const viewCompareEl = document.getElementById('viewCompare');
    const viewAspectEl = document.getElementById('viewAspect');
    const levelAspectEl = document.getElementById('levelAspect');
    const aspectEl = document.getElementById('aspect');

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
      return Object.keys(keys).filter(k => keys[k].size >= 2).sort();
    }

    function groupRunsByContext(runs, aspect) {
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
      if (!container || !runs.length || !aspect) { container.innerHTML = '<div class="empty-state"><span class="empty-state-title">Select level and aspect</span><p>Choose Level and Aspect (x-axis) above to see outputs along that parameter.</p></div>'; return; }
      const { byContext, aspectValues } = groupRunsByContext(runs, aspect);
      const contextKeys = Object.keys(byContext);
      if (!contextKeys.length || !aspectValues.length) { container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No variation on this aspect</span><p>No runs have multiple values for this parameter. Try another aspect or level.</p></div>'; return; }
      const allBlockNames = [];
      const seen = {};
      runs.forEach(r => { (r.blocks || []).forEach(b => { if (b.name && !seen[b.name]) { seen[b.name] = true; allBlockNames.push(b.name); } }); });
      const blockToType = {};
      runs.forEach(r => { (r.blocks || []).forEach(b => { if (b.name && !blockToType[b.name]) blockToType[b.name] = b.type || ''; }); });
      allBlockNames.sort((a, b) => blockTypeOrder(blockToType[a] || '') - blockTypeOrder(blockToType[b] || '') || a.localeCompare(b));
      const axisCaption = aspect + ' \u2192';
      let navHtml = '<div class="aspect-section-nav"><div class="aspect-section-nav-inner"><span class="control-group-label">Sections:</span>';
      allBlockNames.forEach(blockName => {
        const id = slugifySectionId(blockName);
        navHtml += '<a href="#' + id + '">' + escapeHtml(blockName) + '</a>';
      });
      navHtml += '</div></div>';
      let html = navHtml + '<p class="aspect-axis-caption">Aspect: <strong>' + escapeHtml(aspect) + '</strong> (x-axis)</p>';
      allBlockNames.forEach(blockName => {
        const sectionId = slugifySectionId(blockName);
        html += '<div class="aspect-section" id="' + sectionId + '"><h3>' + escapeHtml(blockName) + '</h3>';
        html += '<div class="aspect-axis-table-wrap"><table class="aspect-axis-table"><thead><tr><th class="context-label">Context</th>';
        aspectValues.forEach(v => { html += '<th>' + escapeHtml(String(v)) + '</th>'; });
        html += '</tr></thead><tbody>';
        contextKeys.forEach(ck => {
          const runList = byContext[ck];
          const fullLabel = runList[0] ? (runList[0].label_short || runList[0].id) : ck.slice(0, 80);
          const contextLabel = shortLabel(fullLabel, 42);
          html += '<tr><td class="context-label" title="' + escapeHtml(fullLabel) + '">' + escapeHtml(contextLabel) + '</td>';
          aspectValues.forEach(aspectVal => {
            const run = runList.find(r => (r.params && (r.params[aspect] === aspectVal || (typeof r.params[aspect] === 'number' && Number(r.params[aspect]) === Number(aspectVal)))));
            const block = run ? getBlockByName(run, blockName) : null;
            html += '<td class="aspect-cell">';
            if (block) {
              if ((block.path || '').toLowerCase().endsWith('.mp4')) html += '<video controls preload="metadata" src="' + block.path + '" style="max-width:100%"></video>';
              else html += '<img loading="lazy" src="' + block.path + '" alt="' + escapeHtml(blockName) + '">';
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
      sel.innerHTML = includeEmpty ? '<option value="">— none —</option>' : '';
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

    function blockTypeOrder(type) {
      const order = ['reconstruction', 'recon_statistics', 'gen_variance', 'training_video', 'min_rmsd'];
      const i = order.indexOf(type);
      return i >= 0 ? i : order.length;
    }

    function renderBlocks(run, container, label) {
      if (!run || !run.blocks || !run.blocks.length) {
        container.innerHTML = '<div class="empty-state"><span class="empty-state-title">No outputs</span><p>This run has no plots or videos.</p></div>';
        return;
      }
      const sorted = [...run.blocks].sort((a, b) => blockTypeOrder(a.type) - blockTypeOrder(b.type) || (a.name || '').localeCompare(b.name || ''));
      let html = '<div class="run-card"><h2 class="run-card-title">' + escapeHtml(run.label_short || run.id || 'Run') + '</h2>';
      sorted.forEach(b => {
        html += '<div class="block"><div class="block-title">' + escapeHtml(b.name) + '</div>';
        const isVideo = (b.path || '').toLowerCase().endsWith('.mp4');
        if (isVideo) {
          html += '<video controls preload="metadata" src="' + b.path + '" style="max-width:100%"></video>';
        } else {
          html += '<img loading="lazy" src="' + b.path + '" alt="' + escapeHtml(b.name) + '">';
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

    function updateContent() {
      if (!manifest) return;
      const runs = manifest.runs || [];
      if (viewModeEl && viewModeEl.value === 'aspect') {
        viewCompareEl.style.display = 'none';
        viewAspectEl.classList.add('visible');
        const aspectRuns = getRunsByLevelForAspect();
        const aspect = aspectEl && aspectEl.value;
        contentEl.innerHTML = '<div class="content-single"><div id="colAspect"></div></div>';
        renderAspectView(document.getElementById('colAspect'), aspectRuns, aspect);
      } else {
        viewCompareEl.style.display = 'inline-flex';
        viewAspectEl.classList.remove('visible');
        const runAId = runAEl.value;
        const runBId = runBEl.value;
        const runA = runAId ? runs.find(r => r.id === runAId) : null;
        const runB = runBId ? runs.find(r => r.id === runBId) : null;
        if (runB) {
          const labelA = runA ? shortLabel(runA.label_short || runA.id, 45) : '—';
          const labelB = runB ? shortLabel(runB.label_short || runB.id, 45) : '—';
          contentEl.innerHTML = '<div class="compare"><div class="column"><div class="column-header" id="colHeaderA">Run A: ' + escapeHtml(labelA) + '</div><div id="colA"></div></div><div class="column"><div class="column-header" id="colHeaderB">Run B: ' + escapeHtml(labelB) + '</div><div id="colB"></div></div></div>';
          renderRunOrSeedRuns(runA, document.getElementById('colA'), runs);
          renderRunOrSeedRuns(runB, document.getElementById('colB'), runs);
        } else {
          const singleLabel = runA ? (runA.label_short || runA.id) : '—';
          contentEl.innerHTML = '<div class="content-single"><div class="column-header single-run-header">' + escapeHtml(singleLabel) + '</div><div id="colSingle"></div></div>';
          renderRunOrSeedRuns(runA, document.getElementById('colSingle'), runs);
        }
      }
    }

    function fillAspectDropdown() {
      if (!aspectEl) return;
      const runs = getRunsByLevelForAspect();
      const options = getAspectOptions(runs);
      const cur = aspectEl.value;
      aspectEl.innerHTML = '<option value="">— select aspect —</option>';
      options.forEach(k => {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k;
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
        levelEl.addEventListener('change', () => { fillRunSelect(runAEl, false); fillRunSelect(runBEl, true); updateContent(); });
        runAEl.addEventListener('change', updateContent);
        runBEl.addEventListener('change', updateContent);
        viewModeEl.addEventListener('change', () => {
          if (viewModeEl.value === 'aspect') { fillAspectDropdown(); }
          updateContent();
        });
        levelAspectEl.addEventListener('change', () => { fillAspectDropdown(); updateContent(); });
        aspectEl.addEventListener('change', updateContent);
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
    runs_for_manifest = _copy_assets_and_update_paths(runs, assets_dir)
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
