# Euclideanizer Pipeline

A self-contained pipeline for training **DistMap** (a distance-map VAE) and **Euclideanizer** (a model that maps non-Euclidean distance maps to 3D coordinates). One entrypoint runs training, evaluation, and default plotting. Suitable for any use case where you have structural ensembles (e.g. molecular dynamics, structural biology) and want a latent representation plus Euclideanization for downstream analysis or visualization.

---

## Overview

- **DistMap**: Variational autoencoder over pairwise distance matrices; encodes/decodes in a latent space.
- **Euclideanizer**: Takes the (non-Euclidean) decoded distance maps from the frozen DistMap and produces 3D coordinates so that their pairwise distances match the decoded map.

The pipeline trains one or more DistMap configurations, then for each trained DistMap trains one or more Euclideanizer configurations. All hyperparameters are driven by a single YAML config (with optional CLI overrides). Outputs include checkpoints, plots (reconstruction, statistics, generation), optional training videos, and optional analysis (e.g. min-RMSD, Q / max Q).

---

## Requirements

- **Python** 3.9+
- **PyTorch** 2.0+ (CPU, CUDA, or MPS)
- **PyYAML**, **NumPy**, **Matplotlib**, **tqdm** (see `requirements.txt`)
- **ffmpeg** (optional): required to generate training videos
- **Multi-GPU**: Supported only with CUDA when 2+ devices are available; single-GPU and CPU runs are unchanged.

---

## Installation

From the pipeline directory:

```bash
pip install -r requirements.txt
```

No package install step for the pipeline itself; run the script from a working directory where the pipeline folder is available (see **Quick start**).

---

## Data format

The pipeline expects coordinate data as a **GRO-style text file**: one or more frames, each frame with a title line, an atom count, and then one line per atom (columns must include x, y, z coordinates). The default loader in `src/utils.py` looks for frame title lines starting with `"Chromosome"`; if your data uses a different convention, you will need to point to a compatible file or adapt the loader.

- Input: path to a single file (e.g. `data.gro`).
- Interpreted as: `(n_structures, n_atoms, 3)` array of coordinates.
- The same train/test split (by `data.split_seed` and `data.training_split`) is used for training, validation, plotting, and analysis.

**Bundled dataset:** The project does not include large chromosome GRO files. All data-dependent use (smoke test, sample config, demos) relies on the **sphere dataset**: run `python tests/test_data/generate_spheres.py` to create `tests/test_data/spheres.gro`, then use it with the sample config or `--data tests/test_data/spheres.gro`. Regenerate or customize with optional args: `--num-structures`, `--beads`, `--output`.

---

## Quick start

Run the pipeline from a directory that contains (or can see) the `Euclideanizer_Pipeline` folder. You must pass `**--config`** with the path to your YAML config; there is no default config file.

```bash
# From project root (replace with your paths)
python Euclideanizer_Pipeline/run.py --config Euclideanizer_Pipeline/samples/config_sample.yaml --data /path/to/coordinates.gro

# Or from inside the pipeline directory
python run.py --config samples/config_sample.yaml --data /path/to/coordinates.gro
```

Training requires a dataset path: set it with `--data` or in the config under `data.path`. All other options (output dir, hyperparameters, plotting, etc.) come from the config and can be overridden with CLI flags.

**Common options:**


| Goal                                                | Example                                                      |
| --------------------------------------------------- | ------------------------------------------------------------ |
| Training only (no plots)                            | `--no-plots`                                                 |
| Overwrite existing runs (wipe output dir, then run) | `--no-resume`                                                |
| Skip overwrite confirmation (for SLURM/scripts)     | `--yes-overwrite` (use with `--no-resume`)                   |
| Custom output directory                             | `--output-dir /path/to/output`                               |
| Override hyperparameters                            | `--distmap.beta_kl 0.01 0.05 --euclideanizer.epochs 150 300` |
| Disable multi-GPU                                   | `--no-multi-gpu` (run on one device even with 2+ GPUs)       |
| Limit GPUs used                                     | `--gpus N` (use at most N CUDA devices)                      |


### Running on a cluster (SLURM)

An example SLURM job script is provided as `samples/run.sh`. It is a template with no user-specific paths: it runs from the pipeline root and uses `samples/config_sample.yaml`; job logs go to `slurm_logs/`. Edit the script to activate your Python environment and load any required modules (e.g. ffmpeg for training videos). For overwriting an existing run, add `--no-resume --yes-overwrite` to the `python run.py` line.

---

## Testing

Behavior tests live in `tests/test_pipeline_behavior.py`. They cover pipeline logic **without** running training, plotting, or analysis: they use a minimal config (`tests/config_test.yaml`), temporary directories, and the same helpers the main loop uses.

**What is tested**


| Area                         | Description                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Run completion**           | A run is complete only when the best checkpoint exists, `last_epoch_trained` matches the target, and (for multi-segment) the last-epoch checkpoint is present when required. Final segment with `save_final_models_per_stretch: false` does not require the last checkpoint.                                                                                                             |
| **need_data**                | The pipeline loads only what is required. It must load *something* when the seed dir is missing, any run is incomplete, or (when enabled) a plot or analysis output is missing. It skips loading when all runs are complete and plot/analysis are disabled or already present. See **Resume and data loading** for coords-only vs stats-only vs full load.                               |
| **Resume logic**             | For both DistMap and Euclideanizer: **skip** when the run is complete; **from_scratch** when no previous run or `resume=False`; **resume_from_best** when the current run was interrupted (best epoch < target for first segment, or best > previous segment end for later segments); **resume_from_prev_last** when starting a new segment from the previous segment’s last checkpoint. |
| **Config**                   | Loading `config_test.yaml` yields valid training groups; if resume is on and the saved pipeline config in the output dir does not match the current config, the pipeline raises before loading data.                                                                                                                                                                                     |
| **Plotting / analysis skip** | When `resume=True` and all expected plot (or analysis) files exist, the pipeline skips loading the model for that run.                                                                                                                                                                                                                                                                   |


**How to run**

From the pipeline directory:

```bash
pytest tests/test_pipeline_behavior.py -v
```

No dataset or GPU is required; tests use `tmp_path` and dummy checkpoints.

**Smoke test (full pipeline run)**

A single end-to-end smoke test runs the pipeline with a minimal config using `tests/test_data/spheres.gro` and a temporary output dir, then asserts that key outputs exist (DistMap and Euclideanizer checkpoints, `pipeline.log`). It is **included in default pytest runs** (`pytest tests/ -v`). **On one or zero GPUs** the test uses single-task (one seed); **on two or more GPUs** it uses multi-task (two seeds) so both devices are used and the multi-GPU path is exercised. The test is marked slow; to skip it for a quicker run, use `pytest -m "not slow"`.

- Run **all** tests including smoke (default): `pytest tests/ -v`
- Run all tests **except** smoke (quicker): `pytest tests/ -v -m "not slow"`
- Run **only** the smoke test: `pytest tests/test_smoke.py -v`

The smoke test requires `tests/test_data/spheres.gro` (e.g. from `python tests/test_data/generate_spheres.py`).

---

## Configuration

### Config file

- **Path**: Required. Pass with `--config path/to/config.yaml` (no default; you must specify the config file and know where it is).
- **Content**: Every config key is required (no code-side defaults). All top-level sections and their keys must be present; see `samples/config_sample.yaml` and the **Config reference** below. Missing keys cause a clear error at load time.
- **Overrides**: CLI flags are merged over the config (e.g. `--distmap.epochs 100` replaces the config value).

### Key options (summary)


| Section                    | Purpose                                                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **data**                   | `path`, `split_seed` (int or list for multiple seeds), `training_split`                                                                                                               |
| **output_dir**             | Base directory for all outputs (each seed: `output_dir/seed_<n>/`)                                                                                                                    |
| **distmap**                | VAE: `latent_dim`, `beta_kl`, `epochs`, `batch_size`, `learning_rate`, lambda weights, `memory_efficient`, `save_final_models_per_stretch`                                            |
| **euclideanizer**          | Same idea; no `latent_dim` (inherited from the frozen DistMap). Includes diagonal Wasserstein weights and `num_diags`.                                                                |
| **plotting**               | `enabled`, `overwrite_existing`, reconstruction / bond_rg_scaling / avg_gen_vs_exp, numeric params, `plot_dpi`, then `save_data`, `save_pdf_copy`, `save_structures_gro`.             |
| **training_visualization** | `enabled`, `n_probe`, `n_quick`, `fps`, frame size/dpi, `delete_frames_after_video`                                                                                                   |
| **analysis**               | Nested blocks: same key order as plotting (enabled, overwrite_existing, params, save_data, save_pdf_copy, save_structures_gro or visualize_latent). `rmsd_gen`, `rmsd_recon`, `q_gen`, `q_recon`. |


- **Lists in config**: Any distmap or euclideanizer key can be a list; the pipeline runs one job per element of the Cartesian product (e.g. `beta_kl: [0.01, 0.05]` and `epochs: [100, 300]` → 4 DistMap runs).
- **Epochs as list (segments)**: If `epochs` is a list (e.g. `[100, 300]`), the pipeline trains in segments: first to 100, then resume from the **last** epoch of that run and train to 300. Each segment gets its own run directory (e.g. `distmap/0/`, `distmap/1/`). The **best** checkpoint (by validation loss) is carried across segments. **Resume behavior**: (1) If a segment is interrupted (e.g. first segment stops at epoch 75 with best at 50), rerunning resumes from the **best** (50) and trains the remaining 50 epochs. (2) If a later segment is interrupted (e.g. 300-epoch run has best at 150 and stops at 250), rerunning resumes from the **best** (150) and trains the remaining 150 epochs. So the pipeline prefers resuming from the most recent of “previous segment’s last” or “current run’s best” when the best is more recent. The previous segment’s last checkpoint is deleted only **after** the current segment’s last is written (so a corrupted best save still has a fallback). When `**save_final_models_per_stretch`** is `false`, the **last** segment does not save a final-epoch checkpoint (no next segment needs it). Set `save_final_models_per_stretch` to `true` to keep each segment’s last checkpoint for inspection.
- **Euclideanizer ↔ DistMap**: For each trained DistMap, the pipeline trains one Euclideanizer run per Euclideanizer config combination; the correct frozen DistMap is chosen automatically.

---

## Pipeline behavior

### High-level flow

```mermaid
flowchart TB
    subgraph init["Initialization"]
        A["run.py --config <yaml>"]
        A --> B["Load config + CLI overrides"]
        B --> C["Validate required keys"]
        C --> D{"Training requested?"}
    end

    D -->|No| E["Exit"]
    D -->|Yes| F{"Data path set?"}
    F -->|No| G["Error: set --data or data.path"]
    F -->|Yes| H{"resume=false and output exists?"}
    H -->|Yes| I{"--yes-overwrite?"}
    I -->|No| I2["Confirm overwrite or abort"]
    I -->|Yes| J["Initialize log, save pipeline_config"]
    I2 --> J
    H -->|No| J

    J --> K["Expand DistMap and Euclideanizer grids"]
    K --> CUDA{"2+ CUDA devices?"}
    CUDA -->|No| L{"Need to load data?"}
    L -->|Yes| M["Load data, compute/cache exp stats"]
    L -->|No| N["Skip load"]
    M --> O["For each seed"]
    N --> O
    CUDA -->|Yes| L2["Load data, caches, seed dirs"]
    L2 --> L3["Assign tasks to GPUs (round-robin)"]
    L3 --> L4["Spawn one worker per GPU"]
    L4 --> L5["Workers run tasks: DistMap + EU per task"]
    L5 --> AA

    subgraph per_seed["Per seed (single-GPU path)"]
        O --> P["Cache train/test exp stats"]
        P --> Q["For each DistMap group"]
        Q --> R["Train DistMap segment"]
        R --> S["Assemble training video"]
        S --> T["DistMap plotting"]
        T --> U["For each Euclideanizer group"]
        U --> V["Train Euclideanizer segment"]
        V --> W["Assemble EU video"]
        W --> X["EU plotting + analysis"]
        X --> UE{"More EU?"}
        UE -->|Yes| U
        UE -->|No| YD{"More DistMap?"}
    end

    YD -->|Yes| Q
    YD -->|No| Z{"More seeds?"}
    Z -->|Yes| O
    Z -->|No| AA["Pipeline complete"]
```



### Multi-GPU execution

When **2+ CUDA devices** are available, the pipeline splits work into independent **(seed, DistMap group)** tasks and runs them in parallel: one worker process per GPU, each running its assigned tasks sequentially on that device. No change to config or output layout; resume and overwrite rules are unchanged.

**Memory:** Each worker loads its own copy of the dataset and experimental statistics. The main process precomputes and caches per-seed train/test statistics, then frees its copy before spawning workers so that only the workers hold data (avoiding 1 + N copies and OOM). If you are still killed by the system (e.g. OOM killer) on large datasets, run with `**--no-multi-gpu`** or `**--gpus 1`** so only one copy is in memory.

**CPU RAM (host memory):** Multi-GPU runs use **CPU RAM** as well as GPU memory. Each worker keeps the full dataset in host memory and, during Euclideanizer plotting (e.g. recon_statistics, gen_variance), builds large NumPy arrays (e.g. full train/test reconstruction distance maps). With two workers doing heavy plotting or training at the same time, total host usage can exceed 64 GB on large runs; one worker may then block waiting for memory or the job may be killed. If one worker appears to stop making progress during Euclideanizer plotting (especially when resuming with only some plots present), **increase job CPU RAM** (e.g. 128–256 GB for 2 workers and thousands of structures). Alternatively run with `--no-multi-gpu` so only one process runs and peak CPU RAM is lower.

- **When it runs**: Automatically when `torch.cuda.is_available()` and `torch.cuda.device_count() >= 2`. Single-GPU and CPU (or MPS) runs use the same single-process loop as before.
- **Restrict devices**: Set `CUDA_VISIBLE_DEVICES` (e.g. `CUDA_VISIBLE_DEVICES=0,1`) to limit which GPUs are seen. You can also use `--no-multi-gpu` to force the single-process path, or `--gpus N` to use at most N devices.

### Config and CLI reference (flow)


| Source                                                         | Effect                                                                                                                                                                                                                             |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **data.split_seed**                                            | Single int → one run under `output_dir`. List → one full pipeline per seed under `base_output_dir/seed_<n>/`.                                                                                                                      |
| **data.path**                                                  | Required for training. Used for train/test split and all plotting/analysis.                                                                                                                                                        |
| **data.training_split**                                        | Fraction for train (e.g. 0.8); same for DistMap, Euclideanizer, plotting, analysis.                                                                                                                                                |
| **distmap** (any key single or list)                           | Cartesian product → one DistMap run per combination. List for `epochs` → multi-segment training (e.g. 50, 100).                                                                                                                    |
| **euclideanizer** (any key single or list)                     | One Euclideanizer run per (DistMap run × euclideanizer combination). Same epoch-segment logic.                                                                                                                                     |
| **plotting.enabled**                                           | If false (or `--no-plots`), no plotting.                                                                                                                                                                                           |
| **plotting.overwrite_existing**                                | If true and plots exist: prompt then remove plots/dashboard up front and re-run plotting (use `--yes-overwrite` to skip prompt).                                                                                                   |
| **plotting.reconstruction / bond_rg_scaling / avg_gen_vs_exp** | Toggle reconstruction, Rg/scaling stats, and gen-vs-exp plots.                                                                                                                                                                     |
| **plotting.sample_variance**                                   | List → one gen_variance plot set per value.                                                                                                                                                                                        |
| **training_visualization.enabled**                             | One MP4 per DistMap and per Euclideanizer run (requires ffmpeg).                                                                                                                                                                   |
| **analysis.rmsd_gen**                                          | Nested block: `enabled`, `overwrite_existing`, `num_samples`, `sample_variance`, `query_batch_size`, `save_data`, `save_pdf_copy`, `save_structures_gro`. If enabled, RMSD (gen) outputs under `analysis/rmsd/gen/<run>/`. |
| **analysis.rmsd_recon**                                        | Nested block: `enabled`, `overwrite_existing`, `max_recon_train`, `max_recon_test`, `save_data`, `save_pdf_copy`, `visualize_latent`. If enabled, recon figure and optional latent figure under `analysis/rmsd/recon/`.        |
| **analysis.q_gen**                                             | Nested block: `enabled`, `overwrite_existing`, `max_train`, `max_test`, `num_samples`, `sample_variance`, `delta`, `query_batch_size`, `save_data`, `save_pdf_copy`, `save_structures_gro`. If enabled, max Q (gen) outputs under `analysis/q/gen/<run_name>/`. Default `delta`: 1/√2. |
| **analysis.q_recon**                                           | Nested block: `enabled`, `overwrite_existing`, `max_recon_train`, `max_recon_test`, `delta`, `save_data`, `save_pdf_copy`, `visualize_latent`. If enabled, max Q (recon) figure under `analysis/q/recon/`. Default `delta`: 1/√2. |
| **resume**                                                     | If true: skip complete runs and existing plot/analysis outputs. If false: confirm then delete output_dir and run from scratch (unless **--yes-overwrite**, which skips the prompt for non-interactive use).                        |
| **CUDA devices**                                               | If 2+ available: tasks (seed × DistMap group) run in parallel, one process per GPU. Use `--no-multi-gpu` to disable or `--gpus N` to cap device count.                                                                             |
| **--no-multi-gpu**                                             | Disable multi-GPU even when 2+ CUDA devices are available (single-process loop).                                                                                                                                                   |
| **--gpus N**                                                   | Use at most N CUDA devices for multi-GPU (e.g. `--gpus 2` on a 4-GPU node).                                                                                                                                                        |


### Order of operations

1. **Setup**: Load config, resolve paths, load dataset, compute or load cached **experimental statistics** (full-dataset and, per seed, train/test).
2. **Per seed** (if `data.split_seed` is a list): `output_dir = base_output_dir/seed_<n>`; train/test split uses that seed.
3. **Per DistMap segment** (each segment = one epoch target, e.g. 100 then 300):
  - Train DistMap (from scratch or resume previous segment) → save to `distmap/<i>/`.
  - If enabled: assemble training video from frames (or generate frames then assemble); optionally delete frames.
  - DistMap plotting: reconstruction, recon statistics (train + test), generation-variance plots.
  - **Per Euclideanizer** (for this DistMap): for each epoch segment (e.g. 50 then 100):
    - Train Euclideanizer segment → save to `distmap/<i>/euclideanizer/<j>/`.
    - Assemble training video for this segment (if enabled).
    - Plotting (reconstruction, recon statistics, gen-variance) and analysis (e.g. min-RMSD) for this segment.
4. Repeat from step 3 for the next DistMap segment.

So for DistMap `epochs: [300, 500]` and Euclideanizer `epochs: [50, 100]`: **DistMap 300** → video → plots → **EU** segment 50 (train → video → plots → analysis) → **EU** segment 100 (train → video → plots → analysis) → **DistMap 500** → same pattern.

### Training action (per segment)

Each DistMap and Euclideanizer segment is assigned one of four actions. Same logic for both; Euclideanizer uses `euclideanizer.pt` / `euclideanizer_last.pt`.

```mermaid
flowchart TB
    subgraph action["Training action"]
        A["Run dir, target epochs, config"]
        A --> B{"resume and run complete?"}
        B -->|Yes| C["Skip"]
        B -->|No| D{"Previous segment exists?"}
        D -->|No| E{"resume and best < target?"}
        E -->|Yes| F["Resume from best"]
        E -->|No| G["From scratch"]
        D -->|Yes| H{"resume and best > prev segment?"}
        H -->|Yes| F
        H -->|No| I["Resume from prev last"]
    end
```



### Plotting and analysis conditions

```mermaid
flowchart LR
    subgraph plot["Plotting"]
        P1{"plotting.enabled?"}
        P1 -->|No| P2["No plots"]
        P1 -->|Yes| P3{"resume and all present?"}
        P3 -->|Yes| P4["Skip"]
        P3 -->|No| P5["Reconstruction, recon stats, gen_variance"]
    end

    subgraph analysis["Analysis (Eucl. only)"]
        A1{"analysis.rmsd_gen.enabled or rmsd_recon.enabled?"}
        A1 -->|No| A2["No analysis"]
        A1 -->|Yes| A3{"resume and all present?"}
        A3 -->|Yes| A4["Skip"]
        A3 -->|No| A5["RMSD gen + optional recon + latent"]
    end
```



### Resume behavior

- **Default** (`resume: true`): Skip training a run if the checkpoint exists and the run is “complete” (see below). Also skip regenerating plot or analysis files that already exist.
- **Overwrite** (`resume: false` or `--no-resume`): If the output directory already exists, the pipeline prompts you to type `yes delete` and press Enter to confirm; anything else (or Ctrl+C) aborts without deleting. Once confirmed, the output directory is removed and the run starts from scratch. For **non-interactive runs** (e.g. SLURM), add `**--yes-overwrite`** to skip the prompt and avoid the job blocking on input.

**When is a run skipped?**

A run is skipped only if (1) the best checkpoint file exists, (2) the saved run config’s `last_epoch_trained` equals the expected max epochs (and the relevant config section matches), and (3) for multi-segment runs, the last-epoch checkpoint is required only when there is a **next** segment that needs it—i.e. on the **last** segment with `save_final_models_per_stretch: false`, the last-epoch file is not required (and is not written). If a run is incomplete (e.g. interrupted), the pipeline resumes from the run’s **best** checkpoint when that is available (within-segment or mid-segment resume), or from the previous segment’s **last** checkpoint when starting a new segment.

**Resume and config mismatch:** If resume is on and the output directory already exists, **training-related** config (data, distmap, euclideanizer, training_visualization) must match the saved config exactly; otherwise the run fails with a diff. If **plotting** or **analysis** config differs, the pipeline handles each **chunk** independently: **Plotting**, **RMSD (gen)**, **RMSD (recon)**, **Q (gen)**, and **Q (recon)**. For each chunk whose config differs from saved, the pipeline prompts once to confirm, then removes only that chunk’s outputs, then re-runs that chunk (training is skipped). So if only the rmsd_recon block changed, you get one prompt and only recon analysis outputs are removed; plotting and rmsd_gen outputs are left intact. After any such updates, the saved pipeline config is overwritten with the current config.

**Overwriting only plotting or analysis:** You can re-run plotting or analysis over existing outputs without changing config by setting `**overwrite_existing: true`** in `plotting` or in an analysis sub-block (`rmsd_gen`, `rmsd_recon`, `q_gen`, `q_recon`). When that option is true and the corresponding outputs already exist, the pipeline prompts you to type `yes delete` to confirm; then it **deletes only those outputs up front** (plots/dashboard for plotting; that component’s analysis subdir only (e.g. analysis/q/gen); other metrics stay intact). You are only prompted when output exists for that component, then re-runs them. This avoids mixing old and new results. Use `**--yes-overwrite`** to skip the prompt (e.g. in scripts).

**Resume and data loading:** For replot-only runs (e.g. after config diff or overwrite_existing), the pipeline assumes the same inputs as for a full run: the **root dataset file** (`data.path` / `--data`, e.g. the .gro) when any step needs coordinates, and the **experimental_statistics caches** when it can do a stats-only load (e.g. only gen_variance missing). If the .gro is moved or the caches are missing/invalid, the run can fail when it tries to load. What gets loaded is tied to which outputs are missing:

- **Coords only:** When only training, reconstruction plots, recon_statistics, or RMSD analysis are missing, the pipeline loads the coordinate dataset and (if needed) computes or reuses train/test statistics from cache. It does *not* compute or load full experimental statistics (exp_stats) when only those outputs are needed.
- **Stats only (no coords):** When only gen_variance plots are missing and the base experimental-statistics cache plus every seed’s train/test split cache are present and valid, the pipeline loads only those caches (no coordinate file). It then regenerates gen_variance from the saved models. If any cache is missing or invalid, it falls back to a full load.
- **Full load:** When both coords-dependent and stats-dependent outputs are missing, or when stats-only is not possible, the pipeline loads the dataset and (if plotting/analysis need them) experimental statistics and train/test stats.
- **No load:** When all runs are complete and all plot/analysis outputs are present (e.g. you only run to assemble training videos from existing frames), nothing is loaded.

**Q metric (max Q):** The Q analysis blocks use a pairwise-distance similarity score **Q(α, β)** = (1/N) × Σ_{i<j} exp(−(r_ij(α) − r_ij(β))²/(2δ²)), where r_ij are pairwise distances (upper triangle, no diagonal), N = n_beads×(n_beads−1)/2, and **δ** is configurable (default **1/√2** so 2δ² = 1). Q is in [0, 1]; higher means more similar. For each query structure the pipeline reports **max Q** (best match over reference structures). All Q plot labels and outputs use “max Q”.

Experimental statistics are cached under `output_dir/experimental_statistics/` (full) and `output_dir/seed_<n>/experimental_statistics/` (train/test). They are reused when the dataset path and dimensions match. Test→train RMSD is cached at seed level (`seed_<n>/experimental_statistics/test_to_train_rmsd.npz`): it is **saved whenever it is computed** (i.e. whenever RMSD analysis runs for that seed), **independent of the analysis block’s `save_data`**. Similarly, test→train **max Q** is cached at seed level as `seed_<n>/experimental_statistics/q_test_to_train_{max_train}_{max_test}.npz` (saved whenever Q analysis runs for that seed with those train/test sizes; reused by both Q_gen and Q_recon when the same sizes are requested). That cache is reused for all RMSD (or Q) analyses in that seed (including when `overwrite_existing` re-runs analysis) and is not duplicated in the per-run analysis data dirs. The **per-run** analysis outputs (e.g. RMSD gen/recon `.npz` under `analysis/rmsd/.../data/`, `q_data.npz`, `q_recon_data.npz`) are still controlled by `save_data`.

---

## Output

### Directory structure

All outputs live under `output_dir` (from config or `--output-dir`). With multiple seeds, each seed uses `output_dir/seed_<n>/`.

- **Log**: `output_dir/pipeline.log` — concise, real-time log (elapsed time per line). Use `tail -f output_dir/pipeline.log` to monitor.
- **Experimental statistics cache**: `output_dir/experimental_statistics/` (full dataset) and per-seed under `output_dir/seed_<n>/experimental_statistics/` (train/test). Reused when path and dataset size match. Per-seed dir also gets `test_to_train_rmsd.npz` when min-RMSD analysis runs and `q_test_to_train_{max_train}_{max_test}.npz` when Q analysis runs (saved whenever used, independent of analysis `save_data`).
- **DistMap run**: `output_dir/seed_<n>/distmap/<i>/`
  - `model/model.pt` (best), `model/model_last.pt` (last epoch; present only when there is a next segment or `save_final_models_per_stretch: true`), `model/run_config.yaml`
  - `plots/reconstruction/`, `plots/recon_statistics/`, `plots/gen_variance/`, `plots/loss_curves/`
  - `training_video/` (frames and `training_evolution.mp4`) — separate from `plots/` so a plotting wipe does not remove it
- **Euclideanizer run**: `output_dir/seed_<n>/distmap/<i>/euclideanizer/<j>/`
  - Same idea: `model/euclideanizer.pt` (best), `model/euclideanizer_last.pt` (when not the last segment or `save_final_models_per_stretch: true`), `model/run_config.yaml`, plus the same plot types under `plots/`, and `training_video/` when enabled.
  - When RMSD (gen) is enabled: `analysis/rmsd/gen/<run_name>/` per (num_samples, variance), with `rmsd_distributions.png`, optional `data/`, optional `structures/`. When RMSD recon is enabled: `analysis/rmsd/recon/` with `rmsd_distributions.png` and optional `latent_distribution.png` (if `visualize_latent`). When Q (gen) is enabled: `analysis/q/gen/<run_name>/` with `q_distributions.png` and optional `data/`, `structures/`. When Q (recon) is enabled: `analysis/q/recon/` (or `.../recon/<subdir>/` for multiple sizes) with `q_distributions.png` and optional `latent_distribution.png` (if `visualize_latent`).
  - When `plotting.save_structures_gro` is true, generated structures used for gen_variance plots are saved as one multi-frame GRO file per set under `plots/gen_variance/structures/<variance>/structures.gro` (Euclideanizer only; each structure is a frame/timestep).

Index `i` is the run index in the expanded DistMap grid; `j` is the Euclideanizer config index. When `plotting.save_data` is true, many plots also write a `data/` subdir with `.npz` files (see **Saved plot data**).

### Example tree (2 DistMap runs, 2 Euclideanizer configs)

```
output_dir/
  pipeline_config.yaml
  pipeline.log
  experimental_statistics/
  seed_0/
    pipeline_config.yaml
    experimental_statistics/
    distmap/
      0/  model/, plots/, training_video/, euclideanizer/
            0/  model/, plots/ (reconstruction, recon_statistics, gen_variance, loss_curves), training_video/, analysis/rmsd/gen/<run_name>/ (rmsd_distributions.png, data/, structures/), analysis/rmsd/recon/ (rmsd_distributions.png, optional latent_distribution.png), analysis/q/gen/<run_name>/ (q_distributions.png, optional data/, structures/), analysis/q/recon/ (q_distributions.png, optional latent_distribution.png)
            1/  ...
      1/  model/, plots/, euclideanizer/
            0/  ...
            1/  ...
```

### Detailed output layout

```
base_output_dir/
├── pipeline_config.yaml
├── pipeline.log
├── experimental_statistics/
│   ├── meta.json
│   └── exp_stats.npz
└── seed_<s>/
    ├── pipeline_config.yaml
    ├── experimental_statistics/
    │   ├── split_meta.json
    │   ├── exp_stats_train.npz
    │   ├── exp_stats_test.npz
    │   ├── test_to_train_rmsd.npz   # when RMSD analysis runs; always saved when used (independent of save_data)
    │   └── q_test_to_train_{max_train}_{max_test}.npz   # when Q analysis runs; always saved when used (independent of save_data)
    └── distmap/<i>/
        ├── model/
        │   ├── run_config.yaml
        │   ├── model.pt
        │   └── model_last.pt        # if multi-segment
        ├── plots/
        │   ├── reconstruction/
        │   ├── recon_statistics/
        │   ├── gen_variance/
        │   │   └── structures/      # if save_structures_gro
        │   └── loss_curves/
        ├── training_video/          # separate from plots/ so plotting wipe does not remove it
        │   ├── frames/
        │   └── training_evolution.mp4
        └── euclideanizer/<j>/
            ├── model/
            │   ├── run_config.yaml
            │   ├── euclideanizer.pt
            │   └── euclideanizer_last.pt
            ├── plots/
            ├── training_video/     # when enabled
            └── analysis/
                ├── rmsd/
                │   ├── gen/<run_name>/
                │   │   ├── rmsd_distributions.png
                │   │   ├── data/        # if save_data
                │   │   └── structures/  # if save_structures_gro
                │   └── recon/
                │       ├── rmsd_distributions.png
                │       ├── data/        # if save_data
                │       └── latent_distribution.png   # if visualize_latent
                └── q/
                    ├── gen/<run_name>/
                    │   ├── q_distributions.png
                    │   ├── data/        # if save_data (q_data.npz)
                    │   └── structures/  # if save_structures_gro
                    └── recon/   # or recon/<subdir>/ for multiple (max_recon_train, max_recon_test)
                        ├── q_distributions.png
                        ├── data/        # if save_data (q_recon_data.npz)
                        └── latent_distribution.png   # if visualize_latent
```

---

## Project layout

```
Euclideanizer_Pipeline/
  run.py                 # Single entrypoint: training, plotting, analysis
  requirements.txt
  README.md
  LICENSE
  samples/
    run.sh               # Example SLURM job script (template; edit venv and modules for your cluster)
    config_sample.yaml   # Example config (all required keys)
  tests/
    test_pipeline_behavior.py  # Behavior tests (run completion, need_data, resume, config)
    test_utils_and_config.py  # Config, utils, metrics, rmsd, plot paths
    test_smoke.py             # Full pipeline smoke run (slow; requires tests/test_data/spheres.gro)
    conftest.py               # Pytest markers (e.g. slow)
    config_test.yaml          # Minimal config for behavior tests (no dataset required)
    config_smoke.yaml         # Minimal config for smoke test (2 seeds in config; test uses 1 seed on 1 GPU, 2 on 2+ GPUs)
    test_data/                # Bundled sphere dataset: generate_spheres.py, spheres.gro (after generation)
  src/
    _worker_main.py      # Multi-GPU worker launcher (used by run.py when 2+ GPUs)
    config.py            # Config load, validation, grid expansion
    utils.py             # Data loading (GRO-style), device, distance maps, tri/symmetric helpers
    metrics.py           # Experimental statistics (bonds, Rg, scaling)
    plotting.py           # Reconstruction, recon stats, gen analysis, loss curves
    train_distmap.py     # One DistMap training run
    train_euclideanizer.py
    rmsd.py              # RMSD analysis (optional, via analysis.rmsd_gen / rmsd_recon)
    q_analysis.py        # Q / max Q analysis (optional, via analysis.q_gen / q_recon)
    gro_io.py            # Write 3D structures to GROMACS GRO format
    training_visualization.py  # Training videos (optional, requires ffmpeg)
    distmap/             # DistMap VAE (model, loss, sampling)
    euclideanizer/       # Euclideanizer model and frozen VAE loader
```

---

## Plots and analysis

### Analysis metrics

The pipeline supports **pluggable analysis metrics**: RMSD and Q (max Q). Output layout: `analysis/rmsd/...` and `analysis/q/...`, with nested blocks (`rmsd_gen`, `rmsd_recon`, `q_gen`, `q_recon`). The implementation uses a single metric-agnostic loop in `run.py` over a small **analysis metric registry** in `src/analysis_metrics.py`. Each metric is described by an **AnalysisMetricSpec** (config keys, subdir, figure filename, and callables for cache, gen, and recon). Adding a new metric requires implementing that interface and appending a spec to `ANALYSIS_METRICS` in `src/analysis_metrics.py`; no change to the driver or dashboard loop is needed beyond registering the new metric.

### Plot types (when plotting enabled)


| Plot                               | Description                                                                                                                                                                                                                                                                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reconstruction**                 | Test-set samples: original vs reconstructed (DistMap) or original / VAE decode / Euclideanizer (Euclideanizer).                                                                                                                                                                                                                           |
| **Recon statistics**               | Bond lengths, radius of gyration, genomic scaling: experimental vs reconstruction. Separate figures for **test** and **train** subsets.                                                                                                                                                                                                   |
| **Generation (gen variance)**      | For each `plotting.sample_variance`: distributions (bonds, Rg, scaling) for full/train/test/generated; row of average distance maps (train, test, gen); row of difference maps (test−train, train−gen, test−gen).                                                                                                                         |
| **Loss curves**                    | Train and validation loss per epoch (saved under `plots/loss_curves/`).                                                                                                                                                                                                                                                                   |
| **RMSD (gen)** (analysis)          | When `analysis.rmsd_gen.enabled: true`: histograms of min-RMSD (test→train, gen→train, gen→test) per (DistMap, Euclideanizer) pair. Outputs under `analysis/rmsd/gen/<run_name>/` (`rmsd_distributions.png`, optional `data/`, optional `structures/`). Use `num_samples`, `sample_variance`, `save_data`, `save_structures_gro` in the same block. |
| **RMSD (recon)** (analysis)        | When `analysis.rmsd_recon.enabled: true`: one figure with test→train (reused), train recon RMSD, test recon RMSD under `analysis/rmsd/recon/` (`rmsd_distributions.png`). Use `max_recon_train` / `max_recon_test`, `save_data` in the same block.                                                                                                           |
| **Q (gen)** (analysis)             | When `analysis.q_gen.enabled: true`: histograms of max Q (test→train, gen→train, gen→test) per (DistMap, Euclideanizer) pair. Outputs under `analysis/q/gen/<run_name>/` (figure, optional `data/`, optional `structures/`). Use `max_train`, `max_test`, `num_samples`, `sample_variance`, `delta`, `query_batch_size`, `save_data`, `save_structures_gro`. |
| **Q (recon)** (analysis)           | When `analysis.q_recon.enabled: true`: one figure with test→train (max Q), train recon Q (one-to-one), test recon Q (one-to-one) under `analysis/q/recon/`. Use `max_recon_train`, `max_recon_test`, `delta`, `save_data`, `visualize_latent`.                                                                                               |
| **Latent distribution** (analysis) | When `analysis.rmsd_recon.visualize_latent: true` or `analysis.q_recon.visualize_latent: true`: box plots (train/test) and mean/std per dimension under the corresponding `analysis/.../recon/latent_distribution.png`.                                                                                                                                                   |


### Saved plot data (.npz)

With `plotting.save_data: true`, many plots write a `data/` subdir with `*_data.npz`. Load in Python with `np.load("path.npz")`. Representative keys:


| Plot                               | Keys (examples)                                                                                                                                                                                                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reconstruction** (DistMap)       | `original_dms`, `recon_dms`                                                                                                                                                                                                                                                                                   |
| **Reconstruction** (Euclideanizer) | `original`, `vae`, `euclideanizer`                                                                                                                                                                                                                                                                            |
| **Recon statistics**               | `exp_bonds`, `exp_rg`, `genomic_distances`, `exp_scaling`, `recon_bonds`, `recon_rg`, `recon_scaling`                                                                                                                                                                                                         |
| **Gen variance**                   | `sample_variance`, `full_bonds`, `train_bonds`, `test_bonds`, `gen_bonds`, `avg_train_map`, `avg_test_map`, `avg_gen_map`, `diff_test_train`, `diff_train_gen`, `diff_test_gen`, etc.                                                                                                                         |
| **Loss curves**                    | `epoch`, `train_loss`, `val_loss`                                                                                                                                                                                                                                                                             |
| **RMSD (gen)** (analysis)          | When `analysis.rmsd_gen.save_data: true`: data under `analysis/rmsd/gen/<run_name>/data/` with keys such as `gen_to_train`, `gen_to_test`, `bins`. Test→train RMSD is only at seed level (`experimental_statistics/test_to_train_rmsd.npz`), saved whenever analysis runs (independent of `save_data`). |
| **RMSD (recon)** (analysis)        | When `analysis.rmsd_recon.save_data: true`: data under `analysis/rmsd/recon/data/` (or `.../recon/<subdir>/data/`) with keys such as `train_recon_rmsd`, `test_recon_rmsd`, `bins`. Test→train RMSD is at seed level only (saved when used).                                                                                      |
| **Q (gen)** (analysis)             | When `analysis.q_gen.save_data: true`: `analysis/q/gen/<run_name>/data/q_data.npz` with keys `gen_to_train`, `gen_to_test`, `bins`. Test→train max Q is at seed level (`experimental_statistics/q_test_to_train_{max_train}_{max_test}.npz`), saved whenever Q analysis runs (independent of `save_data`).   |
| **Q (recon)** (analysis)           | When `analysis.q_recon.save_data: true`: `analysis/q/recon/data/q_recon_data.npz` (or `.../recon/<subdir>/data/...`) with keys `train_recon_q`, `test_recon_q`, `bins`. Test→train max Q is at seed level only (saved when used).                                                                          |


---

## Config reference (condensed)

All keys below are **required** (no defaults in code). Omit any and the pipeline raises at load time.

- **resume**: `true` (skip complete runs) or `false` (overwrite after confirmation).
- **data**: `path` (dataset file), `split_seed` (int or list of ints), `training_split` (e.g. 0.8).
- **distmap**: `latent_dim`, `beta_kl`, `epochs`, `batch_size`, `learning_rate`, `lambda_mse`, `lambda_w_recon`, `lambda_w_gen`, `memory_efficient`, `save_final_models_per_stretch`.
- **euclideanizer**: `epochs`, `batch_size`, `learning_rate`, same lambdas plus `lambda_w_diag_recon`, `lambda_w_diag_gen`, `num_diags` (diagonals for diagonal Wasserstein), `memory_efficient`, `save_final_models_per_stretch`.
- **plotting**: `enabled`, `overwrite_existing`, `reconstruction`, `bond_rg_scaling`, `avg_gen_vs_exp`, `num_samples`, `gen_decode_batch_size`, `sample_variance`, `num_reconstruction_samples`, `plot_dpi`, `save_data`, `save_pdf_copy`, `save_structures_gro`. (Key order standardized: behavior then save options.)
- **training_visualization**: `enabled`, `n_probe`, `n_quick`, `fps`, `frame_width`, `frame_height`, `frame_dpi`, `delete_frames_after_video`.
- **analysis**: Nested blocks; same key order (enabled, overwrite_existing, params, then save_data, save_pdf_copy, save_structures_gro or visualize_latent). **rmsd_gen**: `enabled`, `overwrite_existing`, `num_samples`, `sample_variance`, `query_batch_size`, `save_data`, `save_pdf_copy`, `save_structures_gro`. **rmsd_recon**: `enabled`, `overwrite_existing`, `max_recon_train`, `max_recon_test`, `save_data`, `save_pdf_copy`, `visualize_latent`. **q_gen**: `enabled`, `overwrite_existing`, `max_train`, `max_test`, `num_samples`, `sample_variance`, `delta`, `query_batch_size`, `save_data`, `save_pdf_copy`, `save_structures_gro`. **q_recon**: `enabled`, `overwrite_existing`, `max_recon_train`, `max_recon_test`, `delta`, `save_data`, `save_pdf_copy`, `visualize_latent`.

For full structure and comments, use `samples/config_sample.yaml` as the template.