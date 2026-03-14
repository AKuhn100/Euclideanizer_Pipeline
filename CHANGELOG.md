# Changelog

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
