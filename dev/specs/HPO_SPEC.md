# Hyperparameter Optimization (HPO) Specification

*Optuna-based joint optimization of DistMap and Euclideanizer parameters, with validation-loss pruning and early stopping. Objective: maximize overall score from the scoring module (Euclideanizer outputs only).*

---

## 1. Overview

- **Objective:** Maximize the pipeline's **overall score** (geometric mean of 30 component scores when all present) produced by the existing scoring module. Scoring evaluates Euclideanizer outputs only; DistMap is trained only to produce the latent used by the Euclideanizer.
- **Optimization:** **Joint** optimization over DistMap and Euclideanizer hyperparameters in a single Optuna study. One trial = one (DistMap config, Euclideanizer config) pair: train that DistMap, then train that Euclideanizer on it, run full analysis and scoring; report overall score (or handle failure as below).
- **Entry point:** Dedicated HPO run script and HPO-specific config. All analysis is mandatory; generation variance for scoring is fixed to 1; learning rate, batch size, and epochs are fixed/capped as specified below. No reuse of existing pipeline runs for trials: each trial runs the pipeline with the suggested config. Resuming means continuing the same Optuna study for additional trials (e.g. 100 then +50).

---

## 2. How it works (walkthrough)

1. **Invocation.** Run `python run_hpo.py --config samples/hpo_config.yaml [--data /path/to/data.npz]`. If multiple GPUs are available (or `n_gpus` in config is > 1), the script automatically spawns one worker per GPU (shared study DB). Otherwise it runs a single process. The script loads the HPO config (output_dir, data_path, seed, **epoch_cap**, pipeline_config, search_space, optuna, n_gpus) and the **base pipeline config** (see below).

2. **Pipeline config = template.** The `pipeline_config` YAML is the **template** for every trial. It must contain the full pipeline structure (distmap, euclideanizer, plotting, analysis, scoring, data, etc.). For each trial we: (a) deep-copy this template; (b) set `output_dir`, `data.path`, `data.split_seed` from the HPO config; (c) if `epoch_cap` is set in the HPO config, set both `distmap.epochs` and `euclideanizer.epochs` to it (otherwise use the template's epochs, collapsed to a single value); (d) overlay Optuna-suggested values **only** for the keys listed in `search_space`. Everything not in search_space (learning_rate, batch_size, plotting, analysis, memory_efficient, etc.) comes from the template unchanged.

3. **Study.** An Optuna study is created or loaded from SQLite at `output_dir/hpo_study.db`. It uses a TPE sampler (seed from HPO config) and a pruner (e.g. MedianPruner). Trials run one at a time per process; parallelism is only via the multi-GPU launcher (one process per GPU).

4. **Per-trial objective.** For each trial:
   - **GPU:** In single-process mode, `CUDA_VISIBLE_DEVICES` is set to `trial.number % n_gpus` so trials round-robin. In worker mode (launcher), the launcher sets `CUDA_VISIBLE_DEVICES` per process so each worker uses one GPU (`cuda:0`).
   - **Suggest:** Optuna suggests one value per key in `search_space` (distmap.* and euclideanizer.*). Those are merged into the trial config as above; epochs come from HPO `epoch_cap` if set, else from the template.
   - **Data:** Coords and experimental statistics (full exp_stats, train_stats, test_stats) are loaded or computed once per process and **cached** by `(data_path, seed)` so later trials reuse them.
   - **Run trial:** `run.run_one_hpo_trial(cfg, trial_dir, trial, device, coords, coords_np, ...)` is called with the full per-trial pipeline config. That function:
     - Saves the pipeline config under `trial_dir`, creates `trial_dir/seed_{S}/distmap/0/` and `.../euclideanizer/0/`, and writes train/test stats into `seed_dir/experimental_statistics/` for scoring.
     - Trains **one** DistMap to the epoch cap. At the end of each epoch, an **epoch callback** runs: it reports the current validation loss with `trial.report(val_loss, step=epoch)` and, if `trial.should_prune()` is true, raises `optuna.TrialPruned()`. If pruned, the trial stops and no score is returned; Optuna records the trial as pruned.
     - If not pruned: runs DistMap plotting (recon, recon_statistics, gen_variance, bond_length_by_genomic_distance).
     - Trains **one** Euclideanizer on that DistMap, with the same report/prune callback each epoch.
     - If not pruned: runs Euclideanizer plotting, then analysis (latent, rmsd, q, clustering, etc.), then scoring. Reads `overall_score` from `.../euclideanizer/0/scoring/scores.json` and returns it.
   - **Result:** The objective returns that score (or re-raises `TrialPruned` / `TrialFailedException`). Failures are appended to `hpo_failed_trials.log`.

5. **Pruning.** The pruner (e.g. MedianPruner) compares each trial's reported validation loss at the same **step** to other trials. If the trial is clearly worse, `should_prune()` becomes true and the callback raises `TrialPruned`, so training and the rest of the pipeline for that trial are skipped. **DistMap reports step = epoch (1..dm_epochs). Euclideanizer reports step = dm_epochs_max + epoch**, where `dm_epochs_max` is the same for all trials (from `epoch_cap`, search space high, or base config). So DistMap steps are 1..dm_epochs_max and Euclideanizer steps are dm_epochs_max+1, dm_epochs_max+2, ... with no overlap; the pruner compares DistMap epoch 50 to DistMap epoch 50 across trials, and Euclideanizer epoch 50 to Euclideanizer epoch 50 across trials (at step dm_epochs_max+50). Pruning can occur in either phase. Pruner options (see HPO config comment): MedianPruner, SuccessiveHalvingPruner, HyperbandPruner, NopPruner, PercentilePruner, ThresholdPruner, WilcoxonPruner, PatientPruner.

6. **After optimize.** The script prints the best trial and writes the HPO dashboard under `output_dir/dashboard/`: `manifest.json`, `index.html`, and `assets/` (e.g. `assets/style.css` for page styling; same top-level layout as the per-run pipeline dashboard). Links point to each trial's pipeline dashboard.

7. **Resume.** With `--resume --n-trials-add N`, the same study is loaded from the same DB and N **new** trials are run; trial numbers continue from the previous run. Trials that were in progress when the run was stopped (or that failed) are **not** re-run; they stay in the study as incomplete/failed. The run adds new trials until the total trial count (all states) reaches the previous count plus N.

---

## 3. Design Decisions

### 3.1 Parameters

- **Fixed (not tuned):** `learning_rate`, `batch_size`; and `epochs` unless overridden. The **epoch cap** can be set in the HPO config as `epoch_cap` (one value for both DistMap and Euclideanizer); if set, it overrides the template's epochs for every trial. Otherwise epochs come from the pipeline template (single value or max of list). Early stopping (see §3.2) still applies within that cap.
- **Tuned (DistMap):** `latent_dim`, `beta_kl`, `lambda_mse`, `lambda_w_recon`, `lambda_w_gen`. Search space is defined in the HPO config / Optuna suggest calls. `memory_efficient` and `save_final_models_per_stretch` are not tuned; they come from the base pipeline config.
- **Tuned (Euclideanizer):** `lambda_mse`, `lambda_w_recon`, `lambda_w_gen`, `lambda_w_diag_recon`, `lambda_w_diag_gen`, `num_diags`, `lambda_kabsch_mse`. Same: search space in HPO config. `memory_efficient` and `save_final_models_per_stretch` are not tuned; they come from the base config.

### 3.2 Epochs, early stopping, and pruning

- **Epoch cap:** Set `epoch_cap` in the HPO config (e.g. 500 or 1000) to fix max epochs for both DistMap and Euclideanizer in every trial; if unset, the pipeline template's epochs are used (single value or max of list). Training never exceeds this cap.
- **Early stopping:** Validation-loss patience-based early stopping will be implemented in the pipeline (separate from HPO). Training may stop before the epoch cap when validation loss does not improve for a configured number of epochs. This spec will be updated when that feature is implemented; HPO will use the same early-stopping behavior.
- **Pruning:** Optuna pruning is used. The pruning metric is **validation loss** (same as used for early stopping). Trials that are clearly worse than others can be stopped early to save compute. Pruning is implemented via Optuna's `Trial.report()` and a pruning callback (e.g. `MedianPruner` or `SuccessiveHalvingPruner`). Training code must report validation loss at the end of each epoch (or at a fixed reporting interval) so the pruner can decide whether to stop the trial.

### 3.3 Seed and reproducibility

- **Single seed:** HPO uses exactly one seed (top-level `seed` in HPO config). The same value is used for (a) the pipeline train/test split and (b) the Optuna TPE sampler. No separate sampler seed.
- **Reproducibility:** That seed propagates to all random components in the models (as in the current pipeline) and to the sampler so that the same HPO config and study state yield the same sequence of suggestions and training.

### 3.4 Failed or invalid scores

- If scoring fails (e.g. loss divergence, missing data, exception during scoring), the trial does not report a numeric objective to Optuna (e.g. raise `TrialPruned` or use a sentinel that Optuna treats as failure).
- **Failure log:** Every such failure is appended to a dedicated log file (e.g. `hpo_failed_trials.log` or equivalent under the HPO output root) with: trial number, suggested hyperparameters, timestamp, and the reason for failure (exception message or short description). No automatic retry; the log is for manual inspection and follow-up.

### 3.5 Number of trials and resuming

- **Fixed number of trials:** The HPO config specifies a maximum number of trials (e.g. `n_trials: 100`). The run stops when that number of trials is completed (excluding pruned trials from the count or including them—define consistently; typically completed trials are counted and pruned trials are extra).
- **Resuming for additional trials:** Run the HPO script with the same config and `--resume --n-trials-add N`. The script loads the existing study from the same storage (see below) and runs N **new** trials; trial numbers continue (e.g. after 100 trials, adding 50 gives trials 100–149). **In-progress or failed trials are not re-run:** any trial that was running when the process stopped (or that failed) remains in the study and is counted toward the total; only new trials are created until the target total is reached. No separate study name: the path (output_dir) identifies the run.

### 3.6 Sampler and pruner

- **Sampler:** The optimization algorithm is configurable in the HPO config (`optuna.sampler`, `optuna.sampler_kwargs`). Default is **TPESampler** (Tree-structured Parzen Estimator), which works well for mixed continuous/categorical search spaces. Other options include **RandomSampler**, **CmaEsSampler** (continuous only; no categoricals), **NSGAIISampler**, **QMCSampler**, **GridSampler**, **BruteForceSampler**. The top-level `seed` is injected into `sampler_kwargs` for reproducibility unless overridden there.
- **Pruner:** A validation-loss-based pruner is used (e.g. `optuna.pruners.MedianPruner` or `SuccessiveHalvingPruner`). Exact choice (and hyperparameters) are set in the HPO config (`optuna.pruner`, `optuna.pruner_kwargs`). Optionally set `pruner_patient: true` and `pruner_patience: N` to wrap the pruner in **PatientPruner**, so a trial is only pruned after validation loss has not improved for N consecutive steps (reduces aggressive pruning on jittery objectives).

### 3.7 HPO config and entry point

- **Separate entry point:** HPO is invoked by a dedicated script (e.g. `run_hpo.py` in the Pipeline directory), not by flags on the main `run.py`. This script loads an HPO-specific config, builds the Optuna search space from it, and runs the study.
- **HPO config file:** A separate config schema (e.g. `hpo_config.yaml` or a section in a shared config) defines:
  - Fixed pipeline settings: data path, output root, single seed, `learning_rate`, `batch_size`, `epochs` (cap), and any other fixed pipeline options.
  - Analysis: all analysis blocks are enabled and mandatory; generation variance for scoring is 1 (for plotting and all gen-based analysis).
  - Search space: for each tuned DistMap and Euclideanizer parameter, the range or choices (e.g. `latent_dim`: [64, 128, 256, 512], `beta_kl`: log-uniform from 1e-3 to 0.1).
  - Optuna settings: `n_trials`, pruner type and arguments. The same `seed` (top-level) is used for the Optuna TPE sampler. Study storage defaults to `output_dir/hpo_study.db` so the path identifies the run; optional `storage` override in config.
- **No grid expansion:** The HPO entry point does not use the existing grid expansion (lists in distmap/euclideanizer). It generates a single (DistMap, Euclideanizer) config per trial from Optuna suggestions and runs the pipeline once per trial (in-process or via subprocess, as chosen in implementation).

### 3.8 Output layout and dashboards

- **Per-trial output:** Each trial writes into its own directory under the HPO output root, using the **existing pipeline structure**: e.g. `{hpo_output_root}/trial_{N}/seed_{s}/distmap/0/euclideanizer/0/` with the same contents as a single run today (model checkpoints, plots/, analysis/, scoring/, dashboard/). This allows existing plotting, analysis, scoring, and dashboard code to run unchanged on each trial's directory.
- **Full artifacts per trial:** For every trial (including pruned ones that run at least one epoch), generate all outputs: score, plots, analysis data, structures (if enabled), and the per-trial dashboard. No optional skipping of plots for HPO; analysis is mandatory and plots are produced.
- **HPO dashboard:** In addition to the per-trial dashboards, an **HPO-level dashboard** is produced at the HPO output root. It allows comparison across trials: table or list of trials with trial id, hyperparameters, overall score, validation loss (if available), and links to each trial's dashboard. Optionally: parallel coordinates or scatter plots of key hyperparameters vs score. Implementation follows the same top-level layout as the pipeline run dashboard (`dashboard/manifest.json`, `index.html`, `assets/`).

### 3.9 Reuse of existing runs

- **No reuse of pipeline outputs across HPO runs:** We do not reuse existing run directories (e.g. from a previous HPO or a manual run) to avoid subtle mismatches (config, code version). Each trial runs the pipeline from scratch for the suggested config.
- **Resume = continue study:** "Resuming" means loading the Optuna study and running more trials; it does not mean reusing past trial outputs as the objective value (those are already stored in the study).

### 3.10 Multi-GPU parallelization

- **Single entry point:** Run `run_hpo.py --config ... [--data ...]`. The script detects available GPUs (or uses `n_gpus` from config) and, when more than one, **automatically** spawns one process per GPU, each with `CUDA_VISIBLE_DEVICES=i` running `run_hpo.py --worker`. All workers share the same SQLite study at `output_dir/hpo_study.db`; Optuna coordinates trial creation and pruning. Workers stop when the **total** number of trials reaches `n_trials` (MaxTrialsCallback). Set `n_gpus` in config to limit (e.g. 2); omit or null to use all. Killing the parent stops all workers; the study is resumable.
- **Single GPU:** When only one GPU is available (or `n_gpus: 1`), `run_hpo.py` runs in-process; GPU assignment is trivial.
- **No n_jobs:** Parallelism is only via multiple processes (one per GPU). Optuna's `n_jobs` is fixed at 1.
- **Implementation:** In worker mode the parent sets `CUDA_VISIBLE_DEVICES`; the worker uses `cuda:0`. Data and experimental statistics are cached **per process** (keyed by data_path and seed).

---

## 4. Search Space (Summary)

- **DistMap (tuned):** `latent_dim` (discrete), `beta_kl` (continuous, log scale), `lambda_mse`, `lambda_w_recon`, `lambda_w_gen`. Not tuned: `memory_efficient`, `save_final_models_per_stretch` (from base config).
- **Euclideanizer (tuned):** `lambda_mse`, `lambda_w_recon`, `lambda_w_gen`, `lambda_w_diag_recon`, `lambda_w_diag_gen`, `num_diags` (discrete), `lambda_kabsch_mse`. Not tuned: `memory_efficient`, `save_final_models_per_stretch` (from base config).
- **Fixed in HPO config:** `epochs` (single value, cap), `learning_rate`, `batch_size`, single seed, data path, output root, analysis all enabled, generation variance 1 for scoring.

---

## 5. Implementation Plan

1. **Validation-loss early stopping (pipeline)**
  Implement validation-loss patience-based early stopping in the pipeline for DistMap and Euclideanizer training. Expose a callback or hook that reports validation loss (and optionally training loss) each epoch so that (a) training can stop when validation loss does not improve for N epochs, and (b) HPO can use the same validation loss for pruning. Update this spec once early stopping is implemented (exact config keys, behavior, and interaction with epoch cap).
2. **Optuna integration layer**
  Add an HPO entry script (e.g. `run_hpo.py`) that: loads HPO config; creates an Optuna study with TPE sampler (seeded) and chosen pruner; defines the objective function that, given a trial, builds a single (DistMap, Euclideanizer) config from `trial.suggest_`* calls, runs one pipeline run (train DistMap → train Euclideanizer → plotting → analysis → scoring) for the single seed, and returns the overall score (or handles failure and logs to the failure log). Support study storage (e.g. SQLite) so studies can be resumed.
3. **Pipeline invocation from HPO**
  Decide whether each trial runs the pipeline in-process (call training/plot/analysis/scoring functions directly) or via subprocess (e.g. invoke `run.py` with a generated config). In-process is simpler and allows reporting validation loss to Optuna each epoch for pruning; subprocess requires either intermediate reporting (e.g. write validation loss to a file each epoch and have the main process read it) or no pruning until the end. Prefer in-process so that pruning and early stopping are straightforward.
4. **Reporting validation loss for pruning**
  From the training step of each trial, report validation loss to Optuna at the end of each epoch (or at a fixed interval) via `trial.report(val_loss, step=epoch)`. If the pruner suggests pruning, raise `TrialPruned`. Do this for both DistMap and Euclideanizer phases; define whether pruning is per-phase (e.g. prune DistMap early, then run Euclideanizer only for non-pruned trials) or only for the full trial (e.g. report a combined or final validation signal). Spec recommendation: report during Euclideanizer training (since the objective is Euclideanizer score); DistMap can use early stopping only, or report DistMap validation loss for pruning as well (design choice in implementation).
5. **HPO config schema**
  Define and document the HPO config format: fixed pipeline section (data, seed, epochs, lr, batch_size, analysis all on, gen variance 1), and search space section (ranges/choices for each tuned parameter). Validate required keys on load.
6. **Failure logging**
  Implement the failure log: when a trial fails to produce a valid score (exception or NaN overall score), append one line or block to the HPO failure log with trial id, params, timestamp, and reason. Do not overwrite; allow multiple runs (and resume) to append to the same log or a timestamped log file.
7. **Output directory layout**
  Ensure each trial writes to `{hpo_output_root}/trial_{N}/seed_{s}/distmap/0/euclideanizer/0/` (or equivalent) so that existing dashboard/plotting/analysis code can run per trial without change. Use a single seed; N is the Optuna trial number.
8. **HPO dashboard**
  After each trial (or at the end of the run), build the HPO-level dashboard: scan HPO output root for `trial_`*, collect trial id, hyperparameters, overall score from `scores.json`, and link to each trial's dashboard. Generate `dashboard/manifest.json`, `dashboard/index.html`, and `dashboard/assets/` at the HPO root. Optionally add plots (e.g. param vs score) under `assets/`. Reuse patterns from `src/dashboard.py` where possible.
9. **Resume support**
  Implement resume: same HPO config (same output_dir) and `--resume --n-trials-add N`. Load study from output_dir/hpo_study.db and run N more trials; trial numbers continue; new trial dirs go under the same HPO output root.
10. **Reproducibility**
  Use a single seed (HPO config) for both the pipeline train/test split and the Optuna TPE sampler so that the same config + study storage yields reproducible trial order and model training.
11. **Documentation and changelog**
  Add a short HPO section to the Pipeline README (entry point, config, resume). Update CHANGELOG when HPO and early stopping are implemented. Keep this spec updated when early stopping is added and when any design change is made.

---

## 6. Spec Updates Pending

- **Early stopping:** Once validation-loss patience-based early stopping is implemented in the pipeline, update this spec with: config keys for patience and min delta, behavior when validation loss is NaN or missing, and how the epoch cap and early stopping interact. Then reference the early-stopping spec or section from this document.

---

## 7. File and Config Summary


| Item             | Location / name                                                                   |
| ---------------- | --------------------------------------------------------------------------------- |
| HPO entry script | Pipeline script (e.g. `run_hpo.py`)                                               |
| HPO config       | Dedicated file (e.g. `samples/hpo_config.yaml`) or section in existing config     |
| Study storage    | Default: `output_dir/hpo_study.db` (path identifies run; no study_name). Optional `optuna.storage` override. |
| Parallelization  | `run_hpo.py` auto-spawns one worker per GPU when n_gpus > 1 (shared DB; MaxTrialsCallback). Single GPU: one process. |
| Failure log      | Under HPO output root (e.g. `hpo_failed_trials.log`)                              |
| Per-trial output | `{hpo_output_root}/trial_{N}/seed_{s}/distmap/0/euclideanizer/0/`                 |
| HPO dashboard    | `{hpo_output_root}/dashboard/` (`manifest.json`, `index.html`, `assets/` for cross-trial comparison) |

