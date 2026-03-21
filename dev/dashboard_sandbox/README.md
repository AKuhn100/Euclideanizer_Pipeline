# Dashboard sandbox

Iterate on **dashboard layout** without running the full pipeline. The first target is the **Meta-Analysis (sufficiency)** page: static HTML + copied PNGs so you can open `index.html` from disk or serve the folder locally.

## Meta-Analysis page preview

1. Generate synthetic figures (or use a real pipeline output directory that contains `meta_analysis/sufficiency/`).

   ```bash
   python synthetic_plot_sandbox/generate_synthetic_plots.py
   ```

2. Build the preview (default input: `synthetic_plot_sandbox/outputs/fake_base`).

   ```bash
   python dashboard_sandbox/build_meta_analysis_page.py
   ```

3. Open `dashboard_sandbox/outputs/meta_analysis/index.html` in a browser.

The preview uses **full-width** blocks (same idea as pipeline **Detail** view). Re-run **`generate_synthetic_plots.py`** after sufficiency layout changes, then **`build_meta_analysis_page.py`**, so PNGs and `index.html` stay in sync.

Options:

- `--input PATH` — root that contains `meta_analysis/sufficiency/seed_*/` (heatmap + distributions).
- `--out PATH` — where to write `index.html` and `assets/` (defaults under `dashboard_sandbox/outputs/meta_analysis/`).

The script copies referenced PNGs into `assets/` so paths resolve under `file://`. To refresh after changing synthetic plots, run the build script again (it overwrites copies).

## Pipeline dashboard (production)

Detail-view **block order** for Euclideanizer runs follows the **analysis** section of the config: clusterings → **generative capacity** → **latent** → seed-level **sufficiency** meta block last in the ordered list. Implemented in `Pipeline/src/dashboard.py` (`blockTypeOrder` in the embedded JS and block append order in `_blocks_for_euclideanizer_run`).

## Dependencies

None beyond the Python standard library (uses `pathlib`, `shutil`).
