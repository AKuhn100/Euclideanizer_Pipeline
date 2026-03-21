# Synthetic plot sandbox

Standalone harness for iterating on **sufficiency meta-analysis** and **generative capacity** figures without running the full pipeline.

## What it does

1. **Sufficiency (sandbox plotting only)** — Writes a minimal fake `seed_*` tree, scans NPZ the same way the pipeline does, then builds figures with **sandbox-specific layout** (not `Pipeline/src/meta_analysis.py`):
   - Thin **full-width** training-split colorbar **under** both panels; ticks **0%** and **100%** only (colormap still encodes the true split fraction).
   - Extra bottom margin so rotated `max_data` tick labels are not clipped.
   - Heatmaps use **per-panel min–max normalization** to 0–1 so RMSD and Q both show variation under one shared viridis scale; the colorbar label states that explicitly (the production pipeline currently clips raw medians into `[0,1]`, which washes out RMSD).

2. **Generative capacity** — Synthetic `by_n` data; **no torch**.
   - Default outputs: `generative_capacity_rmsd.png` / `generative_capacity_q.png` — overlay step histograms with a **thin N colorbar under** the panel (pipeline currently uses a top row).
   - `variants/`:
     - `*_overlay_top_cbar.png` — previous “colorbar on top” layout for comparison.
     - `stacked_{kde,filled,step,kde_filled}_{rmsd,q}.png` — one row per `N`, shared x-axis (RMSD/Q distribution style): KDE only, filled histogram, step outline, and KDE + semi-transparent filled.

## Usage

```bash
python synthetic_plot_sandbox/generate_synthetic_plots.py
```

Default output root: `synthetic_plot_sandbox/outputs/`.

```text
--out-dir PATH    Override output root
--pdf             Also write PDFs next to PNGs
--no-clean        Keep existing out-dir (append/overwrite files in place)
--rng-seed N      Reproducible synthetic data
```

## Synthetic scale (tweak in code)

In `generate_synthetic_plots.py`, module constants control how “busy” the sufficiency meta-analysis looks:

- `SYNTHETIC_SEEDS`, `SYNTHETIC_TRAINING_SPLITS`, `SYNTHETIC_MAX_DATA` — full Cartesian product of fake runs (heatmap is **5 × 6** splits × `max_data`, i.e. asymmetric).
- `SYNTHETIC_N_STRUCTURES_PER_RUN` — length of each `gen_to_test` vector (KDE / distributions).
- Generative-capacity `n_values` — list of structure counts in the stacked / overlay figures.

## Dependencies

`numpy`, `matplotlib`, `pyyaml` (same as pipeline plotting).
