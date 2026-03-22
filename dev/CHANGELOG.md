# Changelog

## 2026-03-21

- **Multi-GPU precompute logging:** Main-process precompute (before worker spawn) now logs an overview line, per–run-entry plot split-cache progress (hit vs slice vs full recompute, then saved), per-metric analysis cache compute lines with filenames, and per-metric computed/skipped summaries. (`run.py`, `README.md` § Multi-GPU, `STYLE_GUIDE.md` checklist)

- **Single-GPU parity:** Same plot split-cache log lines (shared `_plot_exp_stats_precompute_prefix`), a single-GPU overview line when precompute-style work applies, and **Analysis seed cache** lines when test→train/feats caches are built during analysis (including HPO trial path). (`run.py`, `README.md`, `STYLE_GUIDE.md`)

- **Split-cache startup vs load validation:** Stats-only / precompute paths check **existence + `split_meta.json`** alignment without loading train/test NPZs. When split caches are **loaded** for training/plotting/analysis, `exp_distmaps` leading dimensions are checked against `capped_train_test_index_counts` for that run; mismatch → recompute. Seed-level **RMSD/Q** caches after `np.load` validate per-test row counts and train/test coord rows vs the same capped counts (`cached_test_to_train_rows_match_capped_split` in `utils.py`). Multi-GPU main-process precompute skips full NPZ load when meta matches. (`run.py`, `src/utils.py`, `src/rmsd.py`, `src/q_analysis.py`, `tests/test_exp_stats_cache.py`, `STYLE_GUIDE.md`)

## 2026-03-23

- **HPO dashboard:** Remove **Vary Aspect** from the View menu when the manifest includes `hpo_trial` runs (hyperparameter trials are not a single swept axis). `dashboard.js` embedded in `dashboard.py`.

- **Dashboard `<title>` / `<h1>` placeholder fix:** `_html_content` applied `.replace("__PAGE_TITLE__", …)` only to the last string segment (`a + b + c.replace` precedence). Wrapped the full HTML concatenation in parentheses so the tab title and header substitute correctly. `dashboard.py`.

- **HPO dashboard `assets/`:** `run_hpo.py` now creates `output_dir/dashboard/assets/` and writes `assets/style.css`; `index.html` links it (same `dashboard/` + `assets/` convention as the pipeline dashboard). Updated `HPO_SPEC.md`, `STYLE_GUIDE.md`, README HPO blurb.

- **Style alignment (GC + sufficiency curves):** `plot_config` adds **`GEN_CAP_CONVERGENCE_*`**, **`SUFFICIENCY_CURVES_FIG_HEIGHT`**; `generative_capacity.py` / `meta_analysis.py` use them; GC figure text uses short **Gen** (not “Generated”) per §4.10. `run.py` types `by_n_*` as **`dict[int, np.ndarray] | None`**. Sandbox + **`GENERATIVE_CAPACITY_ANALYSIS.md`** updated. **`STYLE_GUIDE`** §3.3 / checklist §4.10 cross-refs fixed.

- **Documentation:** README expanded (meta-analysis, GC convergence, output trees, resume/overwrite for shared GC convergence + sufficiency, dashboard views); production README avoids `dev/` paths—spec pointers use generic “bundled with sources” wording. `DATA_SUFFICIENCY_META_ANALYSIS.md` §8 (dashboard + manifest curves). `STYLE_GUIDE` spec paths → `dev/specs/`. Sample config comments; synthetic and dashboard sandbox READMEs; sandbox `build_meta_analysis_page.py` includes sufficiency **curves** preview.

- **Sufficiency meta-analysis curves:** `meta_analysis/sufficiency/seed_*/curves/sufficiency_median_recon_vs_split_by_max_data.png` — median test recon RMSD and Q vs training split (≥2 splits), viridis **Max Structures** colorbar under both panels. (`meta_analysis.py`, `DATA_SUFFICIENCY_META_ANALYSIS.md`, `dashboard.py` manifest + Meta-Analysis view, `STYLE_GUIDE.md`)
- **Generative capacity convergence:** `analysis/generative_capacity/convergence_median_vs_n_rmsd_q.png` when both RMSD and Q GC blocks are enabled; both lines use **`COLOR_GEN`**; resume/backfill from per-`n` NPZ or full re-run when needed; overwrite deletes shared PNG/PDF; `all_present` requires this file when both blocks on. (`generative_capacity.py`, `run.py`, `GENERATIVE_CAPACITY_ANALYSIS.md`, `dashboard.py`, `STYLE_GUIDE.md`)
- **Plot labels:** Removed ångström assumptions from sufficiency distribution x-labels and titles (Title Case **Test Recon …**); stacked GC RMSD x-label unchanged string but spec/docs note user units. Synthetic sandbox aligned (`generate_synthetic_plots.py`).

## 2026-03-22

- **Plot titles removed:** Reconstruction statistics and gen variance no longer use figure suptitles; pairwise-distance-by-lag figures drop suptitles and per-panel `k = …` titles; clustering figures drop all suptitles and panel titles (dendrogram y-labels, legends, and mixing line text remain). Removed unused helpers/imports. (`plotting.py`, `clustering.py`, `README.md`, `STYLE_GUIDE.md`)

- **Synthetic plot sandbox:** Sufficiency **curve** figure (`curves/sufficiency_median_recon_vs_split_by_max_data.png`) — median test recon RMSD/Q vs training split, one line per `max_data`. Gen-cap **convergence** figure (`convergence_median_vs_n_rmsd_q.png`) — median min RMSD / max Q vs N (**linear** axes; **`COLOR_GEN`** when using `plot_config`). (`dev/synthetic_plot_sandbox/generate_synthetic_plots.py`, `README.md`)

- **Sufficiency vertical layout:** `sufficiency_*_*_frac` blend legacy figure-fraction margins with inch-based targets; blend weight is `max(height-based, row-count-based)` (`SUFFICIENCY_LAYOUT_INCH_BLEND_*`) so default grids keep legacy proportions while very tall figures **or** many stacked rows favor inch targets (avoids huge whitespace on large grids, tight bars on small ones). Helpers take split/heatmap row count; heatmap `labelpad=12` on “Max Structures”. (`plot_config.py`, `meta_analysis.py`, sandbox, spec, STYLE_GUIDE)

- **Synthetic sandbox:** **`--reuse-fake-base`** + fingerprint file skips regenerating `fake_base/` when it matches the requested grid (use with **`--no-clean`**). (`generate_synthetic_plots.py`, `README.md`)

- **Synthetic plot sandbox:** Sufficiency plotting aligned with **`meta_analysis`** via **`plot_config`** import from `Pipeline`; stacked gen-cap width/height from **`GEN_CAP_STACKED_*`**. Added **`--large-grid`** for many splits × many `max_data`. README updated (recon NPZ, no outdated “sandbox-only layout” claims). (`dev/synthetic_plot_sandbox/generate_synthetic_plots.py`, `README.md`)

- **Dashboard labels:** Manifest `label_short` uses Title Case phrases (**Seed**, **Max Data**, **Split**, **DistMap**, **Euclideanizer**); `label_long` and param table keys stay lowercase. Vary-aspect UI: **Training Split** caption and dropdown text; **(X-Axis)** on aspect caption. (`dashboard.py`, `test_dashboard.py`)

- **Sufficiency heatmap layout:** Small grids (`min(rows, cols) < 4`) still get extra slack via `SUFFICIENCY_HEATMAP_FIGH_EXTRA` and the `sufficiency_heatmap_*_frac` small-step term. **Tall** heatmaps use **inch-based** bottom + colorbar-gap targets (`sufficiency_heatmap_bottom_frac`, `sufficiency_heatmap_cbar_gap_frac`) so the colorbar is not pushed down by a huge fixed figure fraction. **Training Split** `ylabel` only on the **left** panel; **both** panels show **%** y-tick labels, with `sufficiency_heatmap_ytick_fontsize` and `sufficiency_heatmap_wspace` to limit overlap. (`plot_config.py`, `meta_analysis.py`, `DATA_SUFFICIENCY_META_ANALYSIS.md`, sandbox `generate_synthetic_plots.py`, STYLE_GUIDE)

- **Sufficiency distribution layout:** Same adaptive idea for stacked split rows (`n_rows < 4`): **`SUFFICIENCY_DIST_*`** bottom margin, extra gap before the training-split colorbar, and figure height; all distribution labels unchanged. (`meta_analysis.py`, `plot_config.py`, spec, sandbox, STYLE_GUIDE)

- **Sufficiency heatmap y-order:** `imshow(..., origin="lower")` so training-split **% increases upward** (matches spec: low split at bottom, high at top). (`meta_analysis.py`, sandbox `generate_synthetic_plots.py`, spec note)

- **Sufficiency meta-analysis:** Uses **test-set reconstruction** metrics (`test_recon_rmsd`, `test_recon_q` from `analysis/rmsd/recon` and `analysis/q/recon` NPZ), not generative `gen_to_test` arrays. Post-scoring NPZ deferral for sufficiency now applies to **recon** RMSD/Q `data/` trees (replacing gen deferral). (`meta_analysis.py`, `run.py`, `DATA_SUFFICIENCY_META_ANALYSIS.md`, `samples/config_sample.yaml`, `synthetic_plot_sandbox/generate_synthetic_plots.py`, `STYLE_GUIDE.md`)

- **Generative capacity `save_data`:** When true, persists per-`n` histogram NPZ files **and** **`data/pairwise_matrix.npz`** (full `pairwise` array + `n_max`, `seed`, `n_structures`, `metric`; **`delta`** for Q). Temporary memmap **`pairwise_matrix.npy`** is removed after the NPZ is written. (`generative_capacity.py`, `GENERATIVE_CAPACITY_ANALYSIS.md`, `README.md`, sample/test/dev YAML comments, `tests/test_generative_capacity.py`)

## 2026-03-21

- **Sufficiency heatmap colorbar:** Label is **“Normalized Median”** only (removed long parenthetical). (`meta_analysis.py`, `synthetic_plot_sandbox/generate_synthetic_plots.py`, `DATA_SUFFICIENCY_META_ANALYSIS.md`)

- **Dashboard Meta-Analysis view:** New **View → Meta-Analysis** page: full-width heatmap + stacked distribution figures per seed; manifest field **`sufficiency_meta`** with copied asset paths (`dashboard.py` scan + `_copy_assets_and_update_paths`; tests in `test_dashboard.py`).

- **Sufficiency distribution figures:** Training split **%** label drawn **top-left on both** RMSD and Q columns (`meta_analysis.py`; `synthetic_plot_sandbox/generate_synthetic_plots.py`). **Meta-analysis dashboard sandbox:** full-width layout and **Title Case** UI labels (`dashboard_sandbox/build_meta_analysis_page.py`).

- **Dashboard:** Euclideanizer **detail** view block order matches **analysis** config: coord/distmap clusterings → **generative capacity** → **latent** → **meta_analysis_sufficiency** (JS `blockTypeOrder`; `_blocks_for_euclideanizer_run` appends latent after GC). **`dashboard_sandbox/`** — `build_meta_analysis_page.py` builds a static **Meta-Analysis** preview (`outputs/meta_analysis/index.html` + copied assets); **`dashboard_sandbox/README.md`**. (`dashboard.py`, CHANGELOG, STYLE_GUIDE)

## 2026-03-20

- **Sufficiency meta-analysis figures:** Distributions are **stacked filled** histograms (one row per `training_split`, two columns RMSD|Q), **horizontal** training-split colorbar **below** panels (~75% width, label above strip). Heatmaps: **`aspect="equal"`**, x-label **“Max Structures”**, **per-panel** min–max normalization to shared viridis **0–1** with explanatory colorbar label. (`meta_analysis.py`, `plot_config.py` additions `META_*`, `SUFFICIENCY_*`; `DATA_SUFFICIENCY_META_ANALYSIS.md`, STYLE_GUIDE)
- **Generative capacity figures:** Main pipeline uses **stacked filled** histograms (largest `n` top), **vertical** `log10(n)` colorbar right of stack; no figure suptitle. **`GEN_CAP_STACKED_*`** in `plot_config.py`; legacy **`GEN_CAP_FIGSIZE`** / gridspec kept for reference. **`_distribution_panel`** retained for **`test_generative_capacity`**. (`generative_capacity.py`, `GENERATIVE_CAPACITY_ANALYSIS.md`, `README.md`, STYLE_GUIDE)

## 2026-03-19

- **Resume / data needs (generative capacity + meta-analysis):** `_pipeline_data_needs` now passes **`do_generative_capacity_rmsd` / `do_generative_capacity_q`** into `_analysis_cfg_from_need_data_kwargs`, so missing GC figures set **`need_coords`** like other analysis. Fixes overwrite/regen GC together with sufficiency meta-analysis incorrectly taking the **meta-only** branch (no data load, GC never re-run). **`meta_only` / `scoring_only`** also require **`not gc_needs_run`**. Non-resume `PipelineDataNeeds` includes **`do_gc_rmsd` / `do_gc_q`** in `need_coords`. **`_euclideanizer_analysis_all_present`** no longer accepts flattened kwargs when `analysis_cfg` is omitted (tests use **`_analysis_cfg_for_presence`**). (`run.py`, `tests/test_pipeline_behavior.py`, STYLE_GUIDE)

- **Generative capacity figures:** **`GEN_CAP_FIGSIZE`** (8"×~3.7") and gridspec margins match RMSD/Q **width** and ~**one panel** height; **horizontal** full-width colorbar **above** the histogram with Title Case label on top of the strip; plain axis formatting (no `1e7` offset). Overlapping **step** histograms (`LINEWIDTH_HIST_STEP`); **`HIST_BINS_DEFAULT`**. (`plot_config.py`, `generative_capacity.py`; removed `COLORBAR_VERTICAL_*`.)

- **Implementation (Sufficiency meta-analysis):** Added pipeline support for `meta_analysis.sufficiency` and `max_data` config keys. New module `src/meta_analysis.py` builds per-seed sufficiency outputs under `meta_analysis/sufficiency/seed_<n>/` (`distributions/max_data_*/distributions_rmsd_q.png` and `heatmap/sufficiency_heatmap_rmsd_q.png`, plus optional PDF copies). `run.py` now supports deferred post-scoring NPZ cleanup for required RMSD/Q gen inputs when sufficiency meta-analysis is enabled, runs sufficiency meta-analysis near pipeline end, and finalizes deferred cleanup afterward.
- **Config schema cleanup:** Moved `max_data` from top-level config into `data.max_data` (no backward-compat path), updated schema validation and helper access in `src/config.py`, and updated all sample/test/dev config YAMLs plus the sufficiency spec YAML examples to use `data.max_data`.
- **Dashboard integration:** `src/dashboard.py` now adds a seed-level `meta_analysis_sufficiency` block when the sufficiency heatmap exists, and includes this block type in plot ordering so each seed can show a single sufficiency heatmap panel.
- **Config + samples/tests:** `src/config.py` now validates required `max_data` and `meta_analysis.sufficiency` schema; `src/utils.py` supports optional `max_data` subsampling in `load_data()`. Updated sample/test/dev pipeline config YAMLs to include `max_data` and `meta_analysis.sufficiency` keys. Added dashboard coverage for sufficiency block discovery in `tests/test_dashboard.py`.

- **Spec (Sufficiency meta-analysis):** Updated `dev/specs/DATA_SUFFICIENCY_META_ANALYSIS.md` config schema to use `meta_analysis.sufficiency.{enabled,overwrite_existing,save_pdf_copy}` only (removed meta-analysis `save_data`). Aligned YAML examples to pipeline schema (`data.split_seed`, `data.training_split`), renamed `save_pdf` references to `save_pdf_copy`, removed meta-analysis saved-data section and `data/` output dir, retained required-file purge deferral for analysis NPZ only, and clarified per-seed outputs as `distributions/` (combined RMSD+Q panels per `max_data`) and `heatmap/` (combined RMSD+Q median heatmap per seed) plus dashboard one-panel-per-seed behavior.
- **Spec follow-up (Sufficiency heatmap/color semantics):** In `dev/specs/DATA_SUFFICIENCY_META_ANALYSIS.md`, clarified that heatmap right panel is **median max-Q** per grid cell (not maximum). Updated shared heatmap color-scale behavior to fixed `0..1` for both panels, with **no normalization** of median min-RMSD values; min-RMSD is kept in original units and color-clipped to `[0,1]` (values >1 saturate). Confirmed distribution-figure colorbar encodes `training_split` in `[0,1]`.
- **Spec revision (Generative capacity):** Reworked `dev/specs/GENERATIVE_CAPACITY_ANALYSIS.md` to align with pipeline conventions and implementation constraints: flattened config shape (`analysis.generative_capacity_rmsd` / `analysis.generative_capacity_q`), required `overwrite_existing`, `query_batch_size`, and `save_pdf_copy`, corrected output layout to seed-first run directories, dashboard integration via standard block types (no custom tab model), and explicit note that scoring does not consume these outputs yet. Updated execution design to always use an on-disk intermediate full pairwise matrix and retain/delete it based on `save_data`; moved monotonicity verification from runtime warnings to test-suite responsibility.
- **Sample config note:** Added commented generative-capacity block templates and an explicit `O(n_max^2)` runtime-cost note in `samples/config_sample.yaml` near the analysis section.
- **Implementation (Generative capacity blocks):** Added `src/generative_capacity.py` and integrated `analysis.generative_capacity_rmsd` / `analysis.generative_capacity_q` in `run.py` (overwrite/config-diff chunk handling, presence checks, run orchestration, and dashboard rebuild triggers). The implementation always writes a full on-disk pairwise matrix (`pairwise_matrix.npy`) via memmap, then keeps or deletes it according to each block’s `save_data`.
- **Schema + docs + dashboard/tests:** Updated `src/config.py` required analysis keys/validation for generative capacity blocks; updated `src/dashboard.py` to scan and render `generative_capacity_rmsd` / `generative_capacity_q` blocks; added `tests/test_generative_capacity.py` (nested subsampling monotonicity) and dashboard coverage for new block discovery. Updated `README.md`, `dev/STYLE_GUIDE.md`, and config YAMLs in `samples/`, `tests/`, and `dev/configs/` to include the new blocks.
- **Config ordering cleanup:** Reordered analysis blocks in all pipeline YAML configs so `generative_capacity_rmsd` and `generative_capacity_q` appear directly before `latent` for consistent readability (`samples/`, `tests/`, and `dev/configs/` files).
- **Max-data sweep support:** Removed the single-value guard for `data.max_data` and enabled full `seed × training_split × max_data` run matrix behavior in `run.py`. Seed run directories now include a max-data suffix when sweeping (e.g. `seed_15_split_0.9_maxdata_500`), with matching dashboard scan support and a dashboard test for split+max_data directory parsing.
- **Experimental statistics cache optimization:** Updated split-cache generation to derive train/test stats from global `exp_stats` on cache miss by slicing global `exp_distmaps` indices, then recomputing derived arrays (`exp_bonds`, `exp_rg`, `exp_scaling`, `avg_exp_map`) for that slice. Added compact cache storage for distance maps as upper triangles (`exp_distmaps_upper` + `num_atoms_in_stats`) in both global and split caches, with automatic materialization back to full `exp_distmaps` on load so downstream plotting/scoring invocations remain compatible.
- **Resume / multi-GPU:** Per-seed directories could exist with only `experimental_statistics/` (cache precompute) and no `pipeline_config.yaml`, so resume failed before data load. Added `_ensure_per_seed_pipeline_config()` so each seed’s `pipeline_config.yaml` is written before any cache subdirs are created. (`run.py`, `tests/test_pipeline_behavior.py`, `dev/STYLE_GUIDE.md`, README)

## 2026-03-18

- **Plots:** **Gen variance** (bond + Rg) and **bond_length_by_genomic_distance_gen** (Train/Test/Gen lag grid): **Gen** distributions are **filled** bars (behind); **Train** and **Test** remain **step** outlines. Legend order Train, Test, Gen via `_legend_train_test_gen`. (`plotting.py`, STYLE_GUIDE, README)

- **Config:** **`plotting.plot_dpi`** is required (positive int): DPI for main pipeline PNGs and for RMSD/Q/clustering analysis figures. Wired in `run.py` into analysis `plot_cfg`; validated in `src/config.py`. Samples, test YAMLs, and `config_sample_hpo` updated; dev configs already had `plot_dpi`. README condensed reference fixed (removed stale `calibration_decode_batch_cap`). HPO sample + `dev/configs/hpo_config.yaml`: commented **`optuna.storage`** option. (`run.py`, `config.py`, samples/tests, STYLE_GUIDE)

- **Plots:** **Genomic-distance / bond / Rg / scaling:** `LINEWIDTH_HIST_STEP` (2.4) for **step** histograms (train/test in gen_variance and bond-lag gen grid; experimental outlines in recon_statistics); **Gen** in those Train/Test/Gen overlays is **filled**. `LINEWIDTH_SCALING_LOGLOG` (2.4) for P(s) curves. (`plot_config.py`, `plotting.py`, `training_visualization.py`, STYLE_GUIDE)

- **Plots:** RMSD and Q analysis histograms (gen + recon panels) use **borderless** filled bars (`HIST_FILLED_EDGE_COLOR` in `plot_config.py`), matching recon_statistics. Training-video Rg uses step/filled per video design. (`rmsd.py`, `q_analysis.py`, `plotting.py`, `training_visualization.py`, STYLE_GUIDE)

- **Resume:** Fixed multi-GPU resume crash (`shutil.SameFileError`) in `train_euclideanizer.py` when `resume_from_best` tries to copy `euclideanizer.pt` onto itself (src==dst). Now skips the copy in that case. (`train_euclideanizer.py`, pipeline.log)

- **Resume:** Fixed missing **bond_length_by_genomic_distance** (and DistMap bond plots) when `gen_variance` was already on disk: the pipeline skipped loading full-dataset `exp_stats` and gated the whole plotting phase on `exp_stats is not None`. Added `_plotting_phase_needed()` and widened `need_plot_or_rmsd` so bond-length / recon_statistics runs without `exp_stats`. (`run.py`, `test_pipeline_behavior.py`, README, STYLE_GUIDE)

- **Pairwise distance by lag:** Three figures when `plotting.bond_length_by_genomic_distance` is on: **train** (exp vs recon train), **test** (exp vs recon test), **gen** (train/test/gen overlay). Dirs `plots/bond_length_by_genomic_distance_{train,test,gen}/`. Recon statistics histograms: **experimental** train/test as **step outline** (foreground), **recon** filled behind. Dashboard: Title Case on block titles (RMSD/Q/gen run names, pairwise plots, etc.); three bond block types + score strips. (`plotting.py`, `run.py`, `dashboard.py`, tests, README, STYLE_GUIDE)

- **Plotting / scoring alignment:** Reconstruction distmaps for **recon_statistics** (DistMap + Euclideanizer) are capped by **`plotting.max_train`** / **`max_test`** like experimental stats (`null` = full split), so recon pairwise scoring compares equal sample sizes. (`run.py` `_capped_train_test_subset`, `test_pipeline_behavior.py`, STYLE_GUIDE, README)

- **Dashboard:** `_scan_runs` now includes `seed_<n>_split_<frac>/` (multi–`training_split` layout) with unique run IDs (`seed_<n>_split_<frac>_dm_*` / `_eu_*`) and labels showing seed + split. (`Pipeline/src/dashboard.py`, `Pipeline/tests/test_dashboard.py`, `Pipeline/dev/STYLE_GUIDE.md`, README directory/dashboard notes)
- **Dashboard:** Vary-aspect mode adds **training_split (train / test)** when at least two runs differ in train fraction (from `seed_*_split_*` dirs or `pipeline_config.yaml`). Rows group by seed + distmap/Eu index + model params; columns are train fractions. (`Pipeline/src/dashboard.py`, tests, README, STYLE_GUIDE)
- **Dashboard fix:** `_copy_assets_and_update_paths` now copies `training_split`, `split_seed`, `distmap_index`, `euclideanizer_index` into `manifest.json` / embedded manifest (they were dropped before, so Vary aspect never saw training split).
- **Dashboard:** Under each scored plot (recon, gen @ var 1, RMSD, Q, latent, clustering), show grouped **component scores** (named labels + values) from `scoring/scores.json`; DistMap runs inherit scores from their Euclideanizer. Vary-aspect cells include the same strips. (`Pipeline/src/dashboard.py`, tests, STYLE_GUIDE, README)
- **Dashboard (tweak):** Score strips **only under Euclideanizer** outputs (not DistMap-only detail); labels **Title Case**; **Bond length by genomic distance** also shows recon **pairwise distance** scores; new **Score vs aspect** view (SVG plot, Y = chosen component, X = aspect or training split; hover = Frozen DistMap + Euclideanizer table). Manifest adds `score_component_catalog` and per-Eu `component_scores`.
- **Dashboard:** **Score Vs Aspect** — optional **Two Scores (Color By Aspect)** (X/Y = two components, color = aspect); toolbar/browse strings **Title Case**. (`dashboard.py` embedded HTML/JS)
- **Dashboard:** Dual score plot: plot + color-key card **same row** (`flex-wrap: nowrap`), horizontal scroll if narrow; legend card **height matches** plot SVG via `--score-plot-h` (legend grid scrolls inside if many aspects).

## 2026-03-16

- **Pipeline:** List support for `data.training_split`. Config can specify a single float or a list of floats in (0, 1); each value runs one full pipeline (like `split_seed`). Output dirs: when only one split value, keep `seed_<n>/`; when multiple, use `seed_<n>_split_<frac>/`. Validation in `src/config.py`; helper `get_training_splits(cfg)`; `run.py` outer loop over (seed, training_split), pipeline config and resume key off per-run split; cache and `_iter_euclideanizer_runs` recognize both dir patterns. (`Pipeline/src/config.py`, `Pipeline/run.py`, `Pipeline/samples/config_sample.yaml`)

### GRO / converter coordinate mis-parse and where safety lives

**The bug (e.g. chromosome21_converter.py and similar GRO parsers)**  
Converters that parse text formats (e.g. GRO) using **fixed character column positions** for x, y, z (e.g. cols 20–28, 28–36, 36–44) can mis-parse when the file uses **variable spacing**. In practice, `line[20:28]` then often captured part of the line that was not the x coordinate: e.g. `'      -0'` → -0.0, `'       1'` → 1.0, `'       2'` → 2.0, so the **first column of the stored coords became 0, 1, 2, …, N−1 (atom index)** instead of spatial x. The second and third columns picked up misaligned slices (parts of real x, y, z or adjacent fields). The NPZ still had shape (n_structures, n_atoms, 3) and finite values, so it passed naive validation.

**Why distance maps and reconstruction plots still looked realistic**  
Distance maps are computed from pairwise Euclidean distances between the stored 3D points. With “coords” whose first column was 0, 1, 2, …, the geometry was still a **chain-like 3D point cloud** (points spread along one axis with some spread in the other two). So “near in index” still meant “near in space,” and the distance matrix had the expected banded structure (small distances near the diagonal, increasing with |i−j|). DistMap training and reconstruction plots could look plausible; the bug did **not** break distance-based learning or plots.

**Where the bug was visible**  
Only when plotting **coordinates directly** (e.g. training visualization “Exp. Structure” row, which plots column 0 vs column 1 as x vs y): that plot then showed **atom index on the x-axis** (0, 1, 2, …) vs the second stored value — a horizontal, index-like pattern — instead of a 2D projection of the real structure.

**Fix in the converter**  
Parsers must not rely on fixed columns when the format has variable spacing. For GRO (and similar), the fix is to **split each line on whitespace and use the last three numeric fields as x, y, z**. That yields correct coordinates regardless of column alignment. In `setup_wizard_scripts/chromosome21_converter.py`, parsing now prefers this; fixed-width is used only when there are fewer than six fields.

**Where safety lives: setup wizard only**  
- **No workarounds in the training visualization.** The training viz correctly plots `coords[:, 0]` vs `coords[:, 1]` as x vs y. It does **not** implement “index-in-first-column” detection or alternate column choices. Correctness is ensured by valid input (converters that write real x, y, z).
- **Setup wizard** is the safety layer: (1) **Validation** in `validate_converter` rejects NPZ whose first column looks like atom index (0, 1, 2, …) and tells the user to parse x,y,z from whitespace, not fixed columns. (2) **Prompts** in `wizard_prompts.CONVERTER_SYSTEM_PROMPT` require that text-based formats parse coordinates from whitespace-split fields (e.g. last three numeric fields) and state that validation checks the first column is spatial. (3) **Coordinate preview:** after a successful conversion, the wizard writes a **coordinate preview image** next to the NPZ (same pipeline plotting as the “Exp. Structure” row) so the user can visually confirm that structures look like their input. The completion message tells the user to open this preview and to fix the converter if they see a horizontal/index-like plot.

**Disclaimer**  
Before running the wizard, the user must type "Accept" after a short disclaimer stating that LLM-generated code can be faulty, that validation and the coordinate preview do not guarantee correctness, and that the user agrees to review the converter and preview and to take responsibility for parsing accuracy. (`src/wizard.py`: DISCLAIMER, confirm_disclaimer, called after check_api_key.)

**Files**  
- Converter fix: `setup_wizard_scripts/chromosome21_converter.py` (prefer whitespace for x,y,z).  
- Wizard: `src/wizard.py` (`_coords_first_column_looks_like_atom_index`, `validate_converter`, `_write_coordinate_preview`, `print_getting_started` with preview path and step “0. CHECK THE COORDINATE PREVIEW”).  
- Prompts: `src/wizard_prompts.py` (CONVERTER_SYSTEM_PROMPT: coordinate parsing rules and validation criterion).  
- Training viz: `src/training_visualization.py` (`_plot_chain_2d` docstring only; no index-detection or workarounds).

- **Calibration memory limit:** `src/calibrate.py` `_compute_memory_limit` now takes `(device, safety_margin_gb)` and uses `torch.cuda.mem_get_info(device)` (free, total); limit = free - safety_bytes with floor 10% of total VRAM. All callers (calibrate_distmap_batch_size, calibrate_euclideanizer_batch_size) updated to pass `device` only. STYLE_GUIDE §3.2 updated.
- **Analysis GPU cleanup:** In `run.py`, `_force_gpu_cleanup(device)` added before the latent analysis block and before the `for spec in ANALYSIS_METRICS` loop in both `run_one_hpo_trial` and `_run_one_distmap_group` to reduce VRAM pressure between analysis phases.
- **gen_decode_batch_size: no calibration; required in config.** Removed all calibration for gen_decode_batch_size. `plotting.gen_decode_batch_size` and each analysis block's `gen_decode_batch_size` must be a positive integer (no `null`). Removed `calibrate_gen_decode_batch_size` and `calibrate_gen_decode_batch_size_distmap_only` from `src/calibrate.py`; removed `_any_inference_batch_null`, `_resolve_inference_batch_sizes`, and `_apply_resolved_inference_batch_sizes` from `run.py`. Removed `calibration_decode_batch_cap` from required config keys and from all sample/dev/test configs. All configs in `samples/` and `dev/configs/` set `gen_decode_batch_size: 256`. Config validation and STYLE_GUIDE updated. Test: `test_validate_config_rejects_null_gen_decode_batch_size`; removed gen_decode calibration from `tests/test_calibrate.py`.
- **HPO: deterministic output_dir and SQLite path.** `output_dir` is resolved relative to the HPO config file's directory when relative, so the same config yields the same output root regardless of process cwd (avoids Slurm vs interactive sharing or crossing paths). The SQLite study DB path is not a config option: it is always `output_dir/hpo_study.db`. `optuna.storage` is only honored for non-SQLite backends (e.g. PostgreSQL). `run_hpo.py`.

## 2026-03-15

- **Pipeline:** Consolidated scoring to run once per Euclideanizer run at end of run, with inline NPZ cleanup. Removed interleaved `_run_scoring_for_run` calls from `_run_one_distmap_group` and `run_one_hpo_trial`; added a single scoring + `_post_scoring_npz_cleanup` block at end of each EU run. Removed post-loop `_post_scoring_npz_cleanup` from `main()`. (`Pipeline/run.py`)
- **Pipeline:** Fixed `shutil.SameFileError` when resuming an interrupted DistMap run from its own best checkpoint. Skip copying `model.pt` when source and destination are the same (same-run resume). (`Pipeline/src/train_distmap.py`)


- **Setup wizard (SETUP_WIZARD.md).** New entry point `run_setup_wizard.py` guides from raw data (file or directory) to a converted NPZ file. Uses Claude API (model claude-sonnet-4-6) to generate a standalone Python converter script from sampled file content; validates output NPZ (coords key, shape, finiteness) with one retry on failure. Requires `ANTHROPIC_API_KEY`. Added `src/wizard.py` (main, check_api_key, collect_samples, call_claude, validate_converter, save_converter, resolve_output_path, print_getting_started), `src/wizard_prompts.py` (CONVERTER_SYSTEM_PROMPT, build_user_prompt, build_retry_prompt), `setup_wizard_scripts/` for generated scripts. CLI: `--data` (required), `--output`, `--max-files`, `--sample-lines`, `--confirm-large`. Dependencies: `anthropic>=0.21.0`. README (Data format / Setup wizard, Project layout), STYLE_GUIDE (entrypoints, package layout), and requirements.txt updated.

- **NPZ internal format (INTERNAL_DATA_NPZ.md).** Pipeline input is now a single NPZ file with key `coords` (shape `(n_structures, n_atoms, 3)`). `src/utils.load_data` reads NPZ only; validates presence of `coords`, shape, and finiteness; returns float32. All GRO parsing removed. `src/gro_io.write_structures_gro`: parameter `title_prefix` renamed to `title` (default `None` → per-frame "generated frame {i}"); `residue_name` default `"STRUC"` → `"MOL"`; canonical GRO output (plain title, fixed-width atom lines, box line three space-separated floats, no velocities; residue/atom numbers wrap at 99999). Test data: `tests/test_data/generate_spheres.py` now writes `spheres.npz` (no GRO); sample and test configs/smoke use `spheres.npz`. Tests: `test_load_data_valid_npz` and NPZ error cases replace GRO tests; `test_gro_roundtrip` / `test_gro_single_structure` assert on `write_structures_gro` output only; `test_kabsch_rmsd.py` uses `npz_path` and `spheres.npz`. README, STYLE_GUIDE, HPO_SPEC, benchmark help, run.py docstring, and .gitignore updated for NPZ/spheres.npz.

- **Pipeline:** Fixed `shutil.SameFileError` when resuming an interrupted DistMap run from its own best checkpoint. Skip copying `model.pt` when source and destination are the same (same-run resume). (`Pipeline/src/train_distmap.py`)

## (Decouple gen_decode_batch_size from query_batch_size: VRAM vs CPU RAM)

- **Behavior:** Only **gen_decode_batch_size** is auto-calibrated (decode path; VRAM). **query_batch_size** (analysis blocks) must be set in config and is **not** calibrated; it limits CPU RAM for analysis (e.g. RMSD nearest-neighbour, Q matrix batch). This avoids OOM when a VRAM-optimized decode batch size is reused for CPU-side analysis (e.g. Q with large structures).
- **Config:** `analysis.*.query_batch_size` is now required to be a positive integer (no `null`). `gen_decode_batch_size` (plotting and analysis) remains nullable for in-run VRAM calibration. Validation: `src/config.py` (_validate_query_batch_size_key vs _validate_gen_decode_batch_key).
- **run.py:** `_resolve_inference_batch_sizes` returns a single int (gen_decode); no longer reads or writes `query_batch_size` in run_config. `_apply_resolved_inference_batch_sizes(gen_decode, plot_cfg, analysis_cfg)` only fills null `gen_decode_batch_size`; query_batch_size always from config. DistMap-only calibration path saves only `gen_decode_batch_size` to run_config. `_any_inference_batch_null` checks only gen_decode_batch_size.
- **calibrate.py:** Module and function docstrings updated: calibration is for gen_decode_batch_size (VRAM) only; query_batch_size is set in config (CPU RAM).
- **Samples:** `samples/config_sample.yaml` comments updated. `samples/config_sample_hpo.yaml` and `dev/configs/config.yaml`: replaced null query_batch_size with explicit values (128 for rmsd_gen, 64 for q_gen).
- **Docs:** README and STYLE_GUIDE §3.2: gen_decode_batch_size = VRAM (null = calibrate); query_batch_size = CPU RAM (must set; not calibrated).
- **Test:** `test_pipeline_behavior.test_validate_config_rejects_null_query_batch_size` asserts that validate_config raises when analysis.rmsd_gen.query_batch_size is None.

## (README: Benchmark and calibration section; benchmark single-config + warning)

- **README:** New section **Benchmark and calibration** covering: (1) Calibration—in-run, model-specific (DistMap vs Euclideanizer vs inference), training vs inference independent; (2) Batch-size benchmark—rough optimal for given config and dataset, somewhat crude (fixed epochs, no full pipeline); (3) Config for benchmark—single combination only (first value for any list), with firm warning when config has lists; user should use single-value config to optimize for a different combination. Replaced long batch-size benchmark paragraph with a pointer to this section.
- **benchmark_batch_size.py:** Resolve to one training config: for every distmap/euclideanizer key that is a list, take the first value (all keys, not a fixed set). When any key was a list, print a firm warning to stderr that the benchmark uses the first combination only and that to optimize for different hyperparameters the user should use a config with single values. Print the fixed training parameters used (DistMap and Euclideanizer) before the sweep. Docstring updated to state single-combination behavior.
- **STYLE_GUIDE:** Batch-size benchmark bullet now mentions single config (first of each list) and warning; references README § Benchmark and calibration.

## (Fix: scoring missing components when analysis save_data was false)

- **Bug:** With `scoring.enabled: true` and analysis blocks having `save_data: false`, the pipeline computed effective `analysis_save_data = save_data or scoring_enabled` but never passed it into the analysis plot_cfg. Analysis therefore did not write NPZ files (rmsd_data.npz, q_data.npz, clustering_data.npz, etc.), so `compute_and_save` had no data for RMSD, Q, and clustering components and reported them as missing.
- **Fix:** Override `save_data` in the plot_cfg passed to each analysis spec: when building `plot_cfg_gen` and `plot_cfg_recon`, set `save_data` to the effective value (block `save_data` or `scoring.enabled`). Applied in the single-seed single-run path and in `_run_one_distmap_group`. `run.py`.
- **Test:** `test_scoring.test_compute_and_save_all_components_present_when_all_data_saved` asserts that when all analysis/plot NPZ data is present, `compute_and_save` produces no missing components and a finite overall_score.

## (Smoke test: avoid hang on overwrite prompt)

- **test_smoke.py:** Smoke test now passes `--yes-overwrite` so any overwrite confirmation in the pipeline never blocks on `input()` when pytest runs non-interactively (e.g. CI or no TTY).

## (Benchmark: learning rate sweep; grid batch_size × learning_rate)

- **benchmark_batch_size.py:** Added `--learning-rates` (e.g. `1e-4 5e-4 1e-3`). For each specified batch size the script sweeps the specified learning rates. Default: one value from config per model when the flag is omitted. `BenchmarkResult` now includes `learning_rate`. Tables and JSON have one record per (model, batch_size, learning_rate). Suggested (batch_size, learning_rate) when multiple LRs are used. README and STYLE_GUIDE updated.

## (Doc: batch-size calibration—training vs inference independent; benchmark for optimal training)

- **README:** Batch-size calibration paragraph now states that training and inference batch sizes are independent (fixed training sizes can be used while leaving gen_decode/query as null for inference calibration). Added tip that auto-calibration maximizes throughput but ideal training performance can be batch-dependent, and that the batch-size benchmark script can be used to find the optimal training batch size. Batch-size benchmark section now explicitly ties the script to finding optimal *training* batch size.
- **STYLE_GUIDE:** Same points added to the in-run batch-size calibration bullet (independence of training vs inference; benchmark for optimal training batch size).

## (Benchmark: --mode dm | eu | both; eu trains DM 50 epochs in temp, purged after)

- **benchmark_batch_size.py:** Added `--mode dm | eu | both` (default: both). **dm** = DistMap batch-size benchmark only. **eu** = train a DistMap for 50 epochs in a temporary directory, run the Euclideanizer batch-size benchmark using that checkpoint, then purge the temp dir. **both** = DistMap sweep then Euclideanizer sweep (when no `--dm-checkpoint`, a 5-epoch feeder DistMap is trained in a temp dir and purged after). Temp dir cleanup uses a `try/finally` so the dir is removed even if the EU sweep fails. `--model` kept for backward compatibility but deprecated in help text. README and STYLE_GUIDE updated.

## (Configs: align user configs with current calibration schema)

- **configs/config_test.yaml, configs/config_21.yaml, configs/config_2.yaml:** Replaced deprecated `calibration_memory_fraction` with `calibration_safety_margin_gb` and `calibration_binary_search_steps`. Test config uses 2.0 GB / 5 steps; config_21 and config_2 use 15.0 GB / 5 steps to match production config. `configs/config.yaml` and Pipeline test/sample configs were already up to date.

## (Fix: resolve gen_decode_batch_size before DistMap plotting when null)

- **Bug:** When `plotting.gen_decode_batch_size` was `null`, the pipeline calibrated inference batch size only in the Euclideanizer plotting block. DistMap plotting (gen_variance, bond_length_by_genomic_distance) uses the same batch size and was called with `None`, causing `TypeError: 'NoneType' object cannot be interpreted as an integer` in `_get_gen_dm_distmap`.
- **Fix:** Resolve `gen_decode_batch_size` before the first use (DistMap plotting). Added `calibrate_gen_decode_batch_size_distmap_only()` in `src/calibrate.py` (DistMap VAE decode-only probe). In `_run_one_distmap_group`, before DistMap gen/bond-length plotting: if `gen_decode_batch_size` is None, load from run_config in the DistMap model dir or run DistMap-only calibration, save to that run_config, and update a mutable holder so the value is reused for the rest of the seed. `_run_one_seed` now passes `gen_decode_batch_size_holder` (single-element list) into `_run_one_distmap_group`; EU plotting and `_resolve_inference_batch_sizes` use the holder and accept optional `fallback_run_config_dir` (DistMap model dir) to reuse the value without re-calibrating. Same resolution logic added to the HPO single-trial path (DistMap plotting block and EU fallback). Multi-GPU worker passes the holder. `run.py`, `src/calibrate.py`.

## (Doc: batch-size benchmark script; README and STYLE_GUIDE aligned with pipeline)

- **Benchmark script:** Documented `tests/benchmark_batch_size.py` in README (§ Batch-size benchmark) and STYLE_GUIDE (§1 Running and testing, §4.2 Config and tests). Script sweeps batch sizes and reports time per epoch, samples/sec, validation loss, peak VRAM for DistMap and/or Euclideanizer.
- **Benchmark script fix:** `benchmark_batch_size.py` now calls `train_distmap` with current calibration API (`calibration_safety_margin_gb`, `calibration_training_batch_cap`, `calibration_binary_search_steps` from config) instead of deprecated `calibration_memory_fraction`.
- **README:** Added calibration keys to Config reference (condensed). Corrected scoring spec path to `specs/SCORING_SPEC.md`. Listed `benchmark_batch_size.py` in Project layout.

## (Calibration: memory metric, fixed GB margin, true binary search, config keys)

- **Memory metric:** All three calibration probes (DistMap training, Euclideanizer training, decode-only) now use `torch.cuda.max_memory_reserved()` instead of `max_memory_allocated()` for the peak memory check.
- **Config:** Replaced **`calibration_memory_fraction`** with **`calibration_safety_margin_gb`** (positive float, fixed GB reserved) and **`calibration_binary_search_steps`** (non-negative int, max halving iterations after OOM). Removed the "required when batch sizes are null" logic for the old key; the new keys are always required. `TRAINING_CRITICAL_KEYS` and `REQUIRED_TOP_LEVEL` updated in `src/config.py`. Sample configs, `configs/config.yaml`, and tests updated.
- **Limit computation:** `src/calibrate.py` adds `_compute_memory_limit(total_mem, safety_margin_gb)` (limit = total_mem - safety_bytes; hard floor 50% VRAM usable). All three calibration functions take `safety_margin_gb` and (where applicable) `binary_search_steps` instead of `threshold`.
- **Three-phase calibration:** Doubling phase finds bracket [last_good, first_bad]; refinement phase binary-searches between them for up to `binary_search_steps` halvings; final verification re-probes the result and backs off 25% if it fails. Trainers and `_resolve_inference_batch_sizes` pass new params from config. `_warn_calibration_reserve_if_low` warns when `calibration_safety_margin_gb` &lt; 15 GB. README and STYLE_GUIDE updated.

## (batch_size: null not propagated to plotting — HPO and normal runs)

- **Bug:** When `distmap.batch_size` or `euclideanizer.batch_size` is `null`, trainers calibrate and write the resolved value to `run_config.yaml` but only update a local copy; the caller's config still had `batch_size: None`. Passing `batch_size=None` to the DataLoader yields unbatched samples `(N, 3)` instead of `(B, N, 3)`, which led to the "expected 1 channels, got 651" conv error in DistMap reconstruction plotting.
- **Fix:** After `train_distmap` and after `train_euclideanizer`, the caller now reads the resolved `batch_size` from the run's `run_config.yaml` and updates the local config before plotting. Applied in `run_one_hpo_trial` (DistMap and Euclideanizer) and in `_run_one_distmap_group` (DistMap plotting block and Euclideanizer plotting block). If run_config has no resolved batch_size and config still has `None`, fallback to 256. `run.py` only.

## (Doc: calibration applies to inference batch sizes too)

- **Config comments:** User configs under `configs/` and sample analysis batch-size comments now state that in-run calibration applies when any of distmap/euclideanizer batch_size, plotting.gen_decode_batch_size, or analysis query_batch_size/gen_decode_batch_size is null (not only model batch sizes). Added "null = auto-calibrate" to analysis block batch-size comments in `samples/config_sample.yaml`.

## (Q reference-size: remove hardcoded 500/200; config-centric null = use all)

- **Q cache and config:** Removed hardcoded default reference sizes (500 train, 200 test) for Q analysis. When `q_max_train` and `q_max_test` are `null`, the pipeline now uses all train/test structures (same as RMSD). Seed-level cache filename when both null is `q_test_to_train.npz`; when set, `q_test_to_train_{max_train}_{max_test}.npz`. `src/analysis_metrics.py`: `_q_cache_filename` returns `q_test_to_train.npz` for both null; `_q_kwargs_for_cache` passes through `None`; Q spec `requires_reference_bounds=False`. `src/q_analysis.py`: `get_or_compute_test_to_train_q` accepts `max_train`/`max_test` as `int | None` and uses full slice when `None`. `src/scoring.py`: try `q_test_to_train.npz` first, then glob. `run.py`: purge Q caches also removes `q_test_to_train.npz`. Config/docs: added note that large reference-size values can be slow (O(n²) pairwise work) in `samples/config_sample.yaml` and README. Tests and README cache layout updated to use `q_test_to_train.npz`.

## 2026-03-14 (Calibration caps in config; align with style guide)

- **No code-side defaults for calibration search bounds.** Per STYLE_GUIDE §3.2, calibration upper bounds are now in config: **`calibration_training_batch_cap`** (default 512 in samples) and **`calibration_decode_batch_cap`** (default 4096 in samples). Both are required, positive integers. Removed `DEFAULT_HIGH` and `DEFAULT_HIGH_DECODE` from `src/calibrate.py`; callers pass caps from config. Trainers accept `calibration_training_batch_cap`; `_resolve_inference_batch_sizes` uses `cfg["calibration_decode_batch_cap"]`. Added to `TRAINING_CRITICAL_KEYS` and all sample/test/user configs. STYLE_GUIDE updated.

## 2026-03-14 (In-run calibration for inference batch sizes; remove root calibrate.py)

- **Inference batch sizes:** `plotting.gen_decode_batch_size` and analysis blocks' `query_batch_size` / `gen_decode_batch_size` may be `null`. When null, the pipeline calibrates after Euclideanizer training (decode-only path in `src/calibrate.py`: `calibrate_gen_decode_batch_size`); one value is used for both and written to `run_config.yaml`. Config validation allows null for these keys and requires `calibration_memory_fraction` when any is null.
- **run.py:** `_resolve_inference_batch_sizes` and `_apply_resolved_inference_batch_sizes` resolve nulls using run_config or calibration; used in HPO single-seed path and in `_run_one_seed` before plotting/analysis.
- **Removed:** `Pipeline/calibrate.py` (standalone script). All batch-size calibration is now in-run via `src/calibrate.py`. README and STYLE_GUIDE updated.

## 2026-03-14 (In-run batch-size auto-calibration)

- **Config:** `distmap.batch_size` and `euclideanizer.batch_size` may be `null` for auto-calibration; validation still disallows lists and requires a positive integer when not null. New required top-level **`calibration_memory_fraction`** (float in (0, 1] when any batch size is null; may be null when both are fixed). Added to `TRAINING_CRITICAL_KEYS` for resume matching.
- **`src/calibrate.py`:** New module with `calibrate_distmap_batch_size()` and `calibrate_euclideanizer_batch_size()`. Binary search over batch size with full training step per probe; cap at train split size; CUDA fallback to conservative default with warning; batch_size=1 over threshold warns and returns 1; cleanup after each probe and at end.
- **Trainers:** `train_distmap.py` and `train_euclideanizer.py` resolve `batch_size` before building optimizer/DataLoaders: if null, load from `run_config.yaml` (resume) else run calibration; pass `calibration_memory_fraction` from config; optional `on_batch_size_resolved` callback for logging. Resolved batch size written to `run_config.yaml`.
- **run.py:** Early warning when `(1 - calibration_memory_fraction) * VRAM_GB < 15` on any GPU; pass `calibration_memory_fraction` and `on_batch_size_resolved` to all `train_distmap`/`train_euclideanizer` calls; log auto-calibrated batch size when used.
- **Samples/tests:** `samples/config_sample.yaml` and `tests/config_test.yaml` add `calibration_memory_fraction`; sample sets both `batch_size` to `null` with comments. `test_utils_and_config.py`: `batch_size` null passes, list raises, 0 raises. New `tests/test_calibrate.py`: CPU fallback, cleanup, (with CUDA) binary search and OOM handling. STYLE_GUIDE and CHANGELOG updated.

## 2026-03-14 (HPO: Euclideanizer Optuna step = dm_epochs_max + epoch)

- **Optuna step ranges:** DistMap still reports `step=epoch` (1..dm_epochs). Euclideanizer now reports `step=dm_epochs_max + epoch` so its steps never overlap with DistMap and all trials use the same Euclideanizer step range. This removes the "reported value is ignored because this step is already reported" warnings and lets the pruner compare Euclideanizer epochs across trials. `run_hpo.py` computes `dm_epochs_max` and passes it as an argument to `run_one_hpo_trial`; it is HPO runtime state, not pipeline config, so it is not stored in the config dict. Updated `specs/HPO_SPEC.md` pruning section.
- **dm_epochs_max robustness:** `_get_dm_epochs_max` now raises `KeyError` with a clear message when `distmap.epochs` cannot be determined (no epoch_cap, no search_space int spec, no base config value) instead of falling back to 500.

## 2026-03-14 (calibrate.py: batch-size calibration)

- **calibrate.py:** New script in Pipeline root. Finds optimal batch sizes for the user's dataset and GPU: DistMap training, Euclideanizer training, gen_decode_batch_size (inference), and query_batch_size (analysis). Search strategy: double from 1 until OOM or over memory threshold, then binary search in the viable range. Each candidate runs a full training step (forward, loss, backward, optimizer step) with one warmup step before measuring; uses `torch.cuda.reset_peak_memory_stats()` and `max_memory_allocated()`. CLI: `--data` (required), `--threshold` (default 0.85), `--output` (default calibrate_config.yaml). Output: skeleton YAML with calibrated batch sizes and all other keys as placeholders (paths `""`, list keys `[]`). STYLE_GUIDE updated.

## 2026-03-14 (Test suite gaps: cache, early-stop, groups, HPO, scoring, dashboard, GRO, metrics)

- **New/expanded tests:** Added tests for previously untested behavior: (1) `tests/test_exp_stats_cache.py` — exp stats and split cache invalidation on data path/dims/max_train/max_test change; (2) `test_pipeline_behavior.py` — early-stopped runs treated as complete, distmap_training_groups single/segment order/cartesian, _has_any_plotting_output and _has_any_analysis_output; (3) `tests/test_hpo_validation.py` — validate_hpo_pipeline_config (valid, scoring disabled, missing variance 1, analysis block disabled) and _ensure_single_value (list warning, epochs max, scalar no warning); (4) `test_scoring.py` — _pairwise_wasserstein_mean_from_lags (identical≈0, different>0, empty→nan); (5) `tests/test_dashboard.py` — _scan_runs (run ids, levels, parent/child, empty dir) and build_dashboard (creates index.html, manifest.json); (6) `test_utils_and_config.py` — GRO roundtrip (write_structures_gro + load_data with Chromosome title), compute_exp_statistics output keys. Run from pipeline root: `pytest tests/ -v`.

## 2026-03-14 (HPO workers: ffmpeg path propagation)

- **Multi-GPU HPO and ffmpeg:** When spawning worker subprocesses, the launcher now resolves `ffmpeg` via `shutil.which("ffmpeg")` and sets `EUCLIDEANIZER_FFMPEG` in each worker's environment so video assembly works even when the worker's PATH doesn't include the module-loaded ffmpeg. `training_visualization.assemble_video()` uses `EUCLIDEANIZER_FFMPEG` if set, otherwise `ffmpeg`. Load the ffmpeg module in the same shell before starting `run_hpo.py` so the launcher can find it. README HPO section updated.

## 2026-03-14 (_ensure_single_value truncation warning)

- **HPO template list truncation:** In `run_hpo.py`, `_ensure_single_value` now emits a `UserWarning` when a pipeline template key (e.g. `distmap.epochs`, `euclideanizer.latent_dim`) is a list but not in the search space, and the value is collapsed to a single value (first element, or max for epochs). This avoids silent truncation; use single values in the template for non-search keys to avoid the warning.

## 2026-03-13 (Specs consolidated under specs/)

- **Single HPO spec:** Merged `Pipeline/HPO_SPEC.md` into `Pipeline/specs/HPO_SPEC.md` (root version had extra pruning detail and wording). Removed root `HPO_SPEC.md`. All specs now live under `Pipeline/specs/`. Updated README and STYLE_GUIDE to reference `specs/HPO_SPEC.md`; added convention in STYLE_GUIDE that specs live under `specs/`.

## 2026-03-14 (HPO full runs + score-centric validation)

- **HPO trials are full pipeline runs:** Each trial now runs with the same outputs as a normal run: training video (frames + mp4 when `training_visualization.enabled`), all plotting, analysis, and scoring. Training video hook is chained with the prune callback so frames are written each epoch; video is assembled after DistMap and after Euclideanizer training (pruned trials keep frames up to the prune epoch).
- **Score-centric config validation:** Before running HPO (or spawning workers), `run_hpo.py` validates the pipeline config: `scoring.enabled` and `plotting.enabled` must be true; `plotting.sample_variance` and each of `analysis.rmsd_gen`, `analysis.q_gen`, `analysis.coord_clustering_gen`, `analysis.distmap_clustering_gen` must have `enabled: true` and `sample_variance` including 1 (scoring uses gen variance 1 only). If validation fails, the script prints the errors and exits. Added `validate_hpo_pipeline_config()` in `src/scoring.py`; `config_sample_hpo.yaml` comment documents the requirement.

## 2026-03-14 (Single HPO entry point; auto multi-GPU)

- **One entry point:** `run_hpo.py` is the only HPO entry point. When more than one GPU is detected (or `n_gpus` in config is > 1), it automatically spawns one worker per GPU (shared SQLite study DB) and waits for them. Otherwise it runs in-process. Removed `run_hpo_launch.py`; use `run_hpo.py` only.
- **n_gpus:** Omit or set to null in config to use all available GPUs; set e.g. `n_gpus: 2` to limit. Updated README and HPO_SPEC.

## 2026-03-14 (Multi-GPU HPO launcher; n_jobs removed)

- **Multi-GPU HPO:** Added `run_hpo_launch.py`: reads `n_gpus` from HPO config (or all visible GPUs), spawns one `run_hpo.py --worker` per GPU with `CUDA_VISIBLE_DEVICES=i`. All workers share the same SQLite study DB; Optuna coordinates trials; workers stop when total trials reach `n_trials` (MaxTrialsCallback). Killing the launcher stops all workers; study is resumable.
- **run_hpo.py --worker:** Uses single GPU (cuda:0); adds MaxTrialsCallback(n_trials, states=None); optimize(n_trials=n_trials+10000) so callback stops first. Objective does not set CUDA_VISIBLE_DEVICES in worker mode (launcher sets it).
- **n_jobs removed:** HPO always runs one trial at a time per process (n_jobs=1). Removed `n_jobs` from config, config-match keys, and sample `hpo_config.yaml`. Parallelism only via launcher. Updated README, HPO_SPEC.md, specs/HPO_SPEC.md.

## 2026-03-14 (HPO n_jobs and GPU OOM)

- **Optuna uses threads for n_jobs>1**, not separate processes, so all parallel trials run in one process and share the same GPU(s). With n_jobs=8 that puts 8× (coords, model, batch) on one GPU → OOM even with 651 atoms and latent_dim=512. Default when GPUs are present is now **n_jobs=1**; added a warning when n_jobs>1 with GPU. README and sample hpo_config updated. Use n_jobs: 1 for GPU runs; to use multiple GPUs run multiple separate processes sharing the same output_dir.

## 2026-03-14 (HPO TrialFailedException compatibility)

- **Optuna 4.7 compatibility.** Optuna 4.7 does not provide `TrialFailedException`; any exception from the objective marks the trial as FAIL. Replaced `optuna.TrialFailedException` with `RuntimeError` when signalling trial failure so HPO runs on Optuna 4.7+.

## 2026-03-14 (HPO config saved and enforced on resume)

- **Saved HPO config.** When the first trials are run for a study, the HPO config is saved to `output_dir/hpo_config.yaml` (with data_path and pipeline_config resolved) so the run is documented and reproducible.
- **Strict match when adding trials.** When loading an existing study (resume or same output_dir), the current config must match the saved config except for `optuna.n_trials` and `optuna.show_progress_bar`. Any other difference (search_space, pipeline_config, seed, epoch_cap, data_path, sampler, pruner, etc.) causes an error so runs cannot accidentally add trials with different parameters. Use a new output_dir for a different study.

## 2026-03-14 (HPO logging aligned with normal run)

- **HPO trial pipeline.log** now matches normal run.py logging: "Pipeline started." plus config/output/seeds and DistMap/Euclideanizer summary lines immediately after the banner; DistMap/Euclideanizer training messages use the same wording as main ("DistMap run 0 (seed X): training from scratch to Y epochs...", "DistMap 0: training done in Zm."); "Pipeline complete." at end of trial.

## 2026-03-13 (HPO code audit vs style guide)

- **Config validation:** Added `src.config.validate_config(cfg)` for in-memory pipeline configs. `run_hpo.py` validates each trial config after `_build_trial_config()` so pipeline code can use direct config access.
- **run_hpo.py:** Required HPO keys `output_dir`, `pipeline_config`, `optuna` are enforced at startup (no .get() for these); direct access used after checks. Added return type `-> Any` for `_build_sampler()`.
- **run.py run_one_hpo_trial:** Uses direct access for required pipeline keys (`cfg["data"]`, `cfg["distmap"]`, `cfg["plotting"]`, `cfg["analysis"]`, `cfg["scoring"]`, and required sub-keys) per §3.2; no code-side defaults for config-sourced parameters.

## 2026-03-13 (Style guide and docs alignment)

- **Style guide:** Documented HPO entrypoint (`run_hpo.py`), HPO vs pipeline config, output layout (`output_dir/trial_N/`, `hpo.log`, `hpo_study.db`), HPO logging and sample configs (`samples/hpo_config.yaml`, `samples/config_sample_hpo.yaml`); added checklist item for HPO changes; noted `run_hpo.py` uses pathlib for HPO paths.
- **README:** HPO subsection updated for in-process trials, epoch_cap, sampler/pruner options, show_progress_bar, n_jobs, and logging (trial_N/pipeline.log, hpo.log, styled stdout when n_jobs=1 + TTY).

## 2026-03-13 (HPO pipeline template)

- **HPO pipeline config template.** Added `Pipeline/samples/config_sample_hpo.yaml` for use as `pipeline_config` in HPO runs. Optuna-filled keys (distmap/euclideanizer search_space) are null; output_dir, data.path, data.split_seed set by run_hpo; everything else fixed: learning_rate 1e-4, variance 1, patience 50, early_stopping true, memory_efficient and save_final_models_per_stretch false, overwrite_existing false, save_pdf false, save_data true; plotting/analysis max_train/max_test 5000 (rmsd), 500 (q), null (clustering, latent); num_samples 5000 (rmsd/gen), 500 (q), 5000 (clustering gen); exp_stats_chunk_size 1000; resume false. Default `pipeline_config` in `samples/hpo_config.yaml` set to `config_sample_hpo.yaml`.

## 2026-03-13 (HPO in-process + pruning)

- **HPO trials run in-process; pruning enabled.** Trials no longer run in a subprocess: `run_hpo.py` loads data and experimental statistics once per process (cached by data_path and seed), then calls `run.run_one_hpo_trial()` for each trial. Validation loss is reported every epoch via `trial.report(val_loss, step=epoch)`; the Optuna pruner can stop poor trials early (`trial.should_prune()` → `TrialPruned`). `Pipeline/run.py`: added `run_one_hpo_trial()` (train one DistMap and one Euclideanizer with epoch callbacks, plotting, analysis, scoring; returns overall_score). `Pipeline/run_hpo.py`: removed subprocess; added `_load_trial_data()` and per-process data cache; objective uses `n_jobs=1` so pruning works. `Pipeline/samples/hpo_config.yaml`: pruner comment updated. `Pipeline/HPO_SPEC.md`: §2.10 updated for in-process trials and n_jobs=1.

## 2026-03-13 (HPO)

- **Optuna HPO framework.** New entry point `run_hpo.py` for joint optimization of DistMap and Euclideanizer hyperparameters; objective = pipeline overall score. HPO config (`samples/hpo_config.yaml`) defines pipeline base, search space (categorical/int/float/log_float), Optuna settings (n_trials, storage, TPE sampler, pruner), and n_gpus. Parallelization: round-robin GPU assignment (trial N → GPU N % n_gpus); `CUDA_VISIBLE_DEVICES` set per trial. Each trial runs the full pipeline in a subprocess; scores read from `scoring/scores.json`. Resume via `--resume --n-trials-add N`; study stored in SQLite. Failed trials logged to `hpo_failed_trials.log`. HPO dashboard (manifest.json + index.html) at `{output_dir}/dashboard/`. Spec: `HPO_SPEC.md` updated with §2.10 Parallelization. `requirements.txt`: added optuna. README: HPO subsection.

## 2026-03-13 (early stopping)

- **Validation-loss early stopping.** Optional patience-based early stopping for DistMap and Euclideanizer training. Config: `distmap.early_stopping` / `euclideanizer.early_stopping` (bool) and `distmap.patience` / `euclideanizer.patience` (int). When validation loss does not improve for `patience` epochs, training stops and the best model is kept; `run_config.yaml` is written with `early_stopped: true` and `last_epoch_trained` set to the actual epoch. Multi-segment behaviour: if a segment stops early (e.g. DistMap epochs [100, 200] stops at 80), remaining segments for that config are skipped (no 200-epoch run); for DistMap the pipeline still runs video and plotting for the stopped run then proceeds to Euclideanizers; for Euclideanizer it runs video, plotting, analysis, and scoring then moves to the next Euclideanizer config or DistMap. Resume: runs with `early_stopped` are treated as complete; subsequent segments in the same group are skipped. `Pipeline/src/train_distmap.py`, `Pipeline/src/train_euclideanizer.py` return `(model_path, stopped_early)`; `Pipeline/src/config.py` `save_run_config` accepts `early_stopped`; `Pipeline/run.py` implements segment skip and break logic. Sample and test configs updated with `early_stopping: false` and `patience: 20`.

## 2026-03-13 (continued)

- **Six missing metrics and per-lag scoring.** (1) Scoring now loads recon RMSD and recon Q from the keys analysis actually writes: `train_recon_rmsd`/`test_recon_rmsd` and `train_recon_q`/`test_recon_q` (with fallback to `recon_train_*`/`recon_test_*`). Fixes the four recon components when `rmsd_recon_data.npz` and `q_recon_data.npz` exist. (2) Pairwise distance scoring uses **all genomic lags** (1..N−1) for the mean Wasserstein; recon_statistics and gen_variance plot data now save per-lag arrays for all lags. The bond-length-by-genomic-distance plot still shows up to 20 lags for display. SCORING_SPEC: pairwise “mean over all genomic lags”; recon NPZ key note in §7.
- **Gen RMSD and spider labels.** (1) Seed-level RMSD cache is saved by rmsd.py with key `test_to_train`; scoring now loads that key explicitly (was only looking for keys containing "rmsd", so gen_rmsd_train_vs_tt and gen_rmsd_test_vs_tt stayed missing). (2) Spider plot labels no longer show "Vs Tt": component ids ending in `_vs_tt` are displayed without that suffix (e.g. "Gen Rmsd Train", "Gen Q Test").
- **Clustering gen scoring path fix.** Clustering writes `clustering_data.npz` under `gen/<run_name>/data/` (not directly under `gen/<run_name>/`). Scoring now takes run_name from the parent of `data` so variance=1 check applies correctly; all four clustering gen components load when present.
- **Scoring uses only variance=1 generation data.** Generation-related scores are computed only from data produced with sample_variance=1 (mission-critical: different prior variance gives incomparable scores). Config’s sample_variance lists (plotting, rmsd_gen, q_gen, coord_clustering_gen, distmap_clustering_gen) are read; if a block does not include 1 (or 1.0), its gen components are marked missing. Gen paths always include variance in the name (e.g. gen_variance_1.0_data.npz, analysis/rmsd/gen/default_var1.0/). Scoring loads only from paths that indicate variance=1. Run.py analysis loop always writes gen outputs with variance in run_name (default_var1.0, 1000_var1.0, etc.). SCORING_SPEC §7 documents the rule; test_scoring and test_pipeline_behavior extended with rigorous variance=1 tests.
- **Scoring plots and missing-data fixes.** (1) Radar labels use Title Case. (2) Gen variance data path fixed: scoring now loads from `plots/gen_variance/data/gen_variance_*_data.npz` (single data dir) instead of per-subdir. (3) RMSD/Q scoring keys fixed: load `gen_to_train` / `gen_to_test` from rmsd_data.npz and q_data.npz to match analysis save keys. (4) Seed cache lookup: scoring tries default then globs `test_to_train_rmsd*.npz` and `q_test_to_train_*.npz` so config-specific cache filenames (e.g. q_max_train/max_test) are found. (5) Terminology: EMD → Wasserstein throughout (scoring.py, SCORING_SPEC.md, tests, README); `emd_on_zscored` → `wasserstein_on_zscored`, pairwise_emd_* → pairwise_wasserstein_*. (6) STYLE_GUIDE §4.10.1: plot etiquette (avoid grid lines except radar; title case; centralize colors/fonts in plot_config). (7) Dashboard run labels shortened to "Seed # · DM # · Eu #" (no param list in short label).
- **Scoring: all 30 components required; radar always 30 axes.** Scoring now defines a fixed set of 30 component IDs (`EXPECTED_COMPONENTS`). `overall_score` is computed only when all 30 are present; otherwise NaN. Radar/spider plot always shows 30 axes; missing components plot as 0 with label in red. `Pipeline/src/scoring.py`: added `EXPECTED_COMPONENTS`, `_pairwise_wasserstein_mean_from_lags`, fill-all-30 logic, clustering merge from all `clustering_data.npz` under each modality.
- **Pairwise Wasserstein: raw data saved, scoring computes scalars.** Recon and gen plot data now save raw per-lag d(i,i+s) arrays. Scoring loads these and computes the three pairwise Wasserstein scalars. No aggregate statistics in saved data (STYLE_GUIDE §4.4 data/scoring modularity).
- **Dashboard: "Missing data" when overall is NaN.** Detail and radar grid show "Missing data" instead of NaN for the overall score; hover shows the list of missing components. `Pipeline/src/dashboard.py`: scores_data includes `missing`; JS renders "Missing data" and tooltip.
- **Tests and docs.** Updated `test_scoring.py` and `test_pipeline_behavior.py` for all-30 semantics. SCORING_SPEC §7 and STYLE_GUIDE §4.4 updated.

## 2026-03-13

- **Dashboard: Radar grid view.** Added a "Radar grid" page to the Pipeline dashboard (`Pipeline/src/dashboard.py`). The view shows a grid of radar (spider) plots for all scored Euclideanizer runs, each labeled with its overall score, ordered best-to-worst (left to right, top to bottom). Hovering a cell shows a tooltip with that run's parameters (frozen DistMap + Euclideanizer). Updated `Pipeline/README.md` and `Pipeline/STYLE_GUIDE.md` to document the new view.
