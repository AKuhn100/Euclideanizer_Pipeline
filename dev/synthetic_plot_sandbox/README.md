# Synthetic plot sandbox

Standalone harness for **sufficiency meta-analysis** and **generative capacity** figures without running the full pipeline.

## What it does

1. **Sufficiency** — Writes a minimal fake `seed_*_split_*_maxdata_*` tree with **recon** NPZ under `analysis/{rmsd,q}/recon/data/` (`test_recon_rmsd`, `test_recon_q`), matching what `Pipeline/src/meta_analysis.py` reads. Figures use the **same layout and `plot_config` constants** as the pipeline when `Pipeline/src/plot_config.py` is importable (run from the repo with `Pipeline` on `sys.path`); otherwise embedded fallbacks match those defaults.
   - Stacked distributions: inch-based bottom margin + training-split colorbar gap (with extra slack when fewer than four split rows) so tall stacks are not dominated by whitespace.
   - Heatmaps: `origin="lower"` (training split **increases upward**), **Training Split** y-label on the left panel only, **%** ticks on both panels, adaptive margins, **Normalized Median** colorbar.

2. **Generative capacity** — Synthetic `by_n` data; **no torch**. Stacked filled rows use **`GEN_CAP_STACKED_FIGWIDTH`** / **`GEN_CAP_STACKED_ROW_HEIGHT`** from `plot_config` when imported. `variants/` keeps overlay and legacy top-colorbar comparisons.

## Usage

From the **`Pipeline`** directory (so `src.plot_config` resolves):

```bash
cd /path/to/Euclideanizer/Pipeline
python dev/synthetic_plot_sandbox/generate_synthetic_plots.py
```

**Stress many conditions** (larger heatmaps / more distribution rows):

```bash
python dev/synthetic_plot_sandbox/generate_synthetic_plots.py --large-grid
```

Default output root: `dev/synthetic_plot_sandbox/outputs/` (under `Pipeline`).

| Flag | Meaning |
|------|---------|
| `--out-dir PATH` | Override output root |
| `--pdf` | Also write PDFs next to PNGs |
| `--no-clean` | Keep existing `out-dir` |
| `--rng-seed N` | Reproducible synthetic data |
| `--large-grid` | ~14 splits × ~12 `max_data` × 4 seeds; longer gen-cap N ladder |
| `--reuse-fake-base` | Skip rewriting `fake_base/` NPZ trees when `.synthetic_sufficiency_fingerprint.yaml` matches the requested grid (pair with **`--no-clean`**) |

**Refresh plots only** (keep synthetic NPZ on disk):

```bash
python dev/synthetic_plot_sandbox/generate_synthetic_plots.py --no-clean --reuse-fake-base
```

Use the same flags (`--large-grid`, `--out-dir`, etc.) as when the tree was generated so the fingerprint matches.

## Tweaking the default grid

In `generate_synthetic_plots.py`: **`SYNTHETIC_SEEDS`**, **`SYNTHETIC_TRAINING_SPLITS`**, **`SYNTHETIC_MAX_DATA`**, **`SYNTHETIC_N_STRUCTURES_PER_RUN`**. Large-grid sizes: **`LARGE_GRID_*`** constants and **`_large_grid_*()`** helpers.

## Dependencies

`numpy`, `matplotlib`, `pyyaml` (same as pipeline plotting).
