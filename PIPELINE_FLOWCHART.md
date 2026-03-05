# Euclideanizer Pipeline — Flow and Options

This document describes the pipeline flow and all config/CLI options and outcomes. It is the single source of truth for pipeline structure when adding features (e.g. analysis) or parallelization.

---

## 1. High-level pipeline flow

```mermaid
flowchart TB
    subgraph init["Initialization"]
        A["run.py --config &lt;yaml&gt;"]
        A --> B["Load config + CLI overrides"]
        B --> C["Validate required keys"]
        C --> D{"Training requested?"}
    end

    D -->|No| E["Exit"]
    D -->|Yes| F{"Data path set?"}
    F -->|No| G["Error: set --data or data.path"]
    F -->|Yes| H{"resume=false and output exists?"}
    H -->|Yes| I["Confirm overwrite or abort"]
    H -->|No| J["Initialize log, save pipeline_config"]
    I --> J

    J --> K["Expand DistMap and Euclideanizer grids"]
    K --> L{"Need to load data?"}
    L -->|Yes| M["Load data, compute/cache exp stats"]
    L -->|No| N["Skip load"]
    M --> O["For each seed"]
    N --> O

    subgraph per_seed["Per seed"]
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
        UE -->|No| Y
    end

    Y --> YD{"More DistMap?"}
    YD -->|Yes| Q
    YD -->|No| Z{"More seeds?"}
    Z -->|Yes| O
    Z -->|No| AA["Pipeline complete"]
```

---

## 2. Config and CLI reference

| Source | Effect |
|--------|--------|
| **data.split_seed** | Single int → one run under `output_dir`. List → one full pipeline per seed under `base_output_dir/seed_<n>/`. |
| **data.path** | Required for training. Used for train/test split and all plotting/analysis. |
| **data.training_split** | Fraction for train (e.g. 0.8); same for DistMap, Euclideanizer, plotting, analysis. |
| **distmap** (any key single or list) | Cartesian product → one DistMap run per combination. List for `epochs` → multi-segment training (e.g. 50, 100). |
| **euclideanizer** (any key single or list) | One Euclideanizer run per (DistMap run × euclideanizer combination). Same epoch-segment logic. |
| **plotting.enabled** | If false (or `--no-plots`), no plotting. |
| **plotting.reconstruction / bond_rg_scaling / avg_gen_vs_exp** | Toggle reconstruction, Rg/scaling stats, and gen-vs-exp plots. |
| **plotting.sample_variance** | List → one gen_variance plot set per value. |
| **training_visualization.enabled** | One MP4 per DistMap and per Euclideanizer run (requires ffmpeg). |
| **analysis.min_rmsd** | If true, min-RMSD analysis after each Euclideanizer (per variance × num_samples). |
| **analysis.min_rmsd_num_samples** | Int or list → one min_rmsd figure set per value. |
| **analysis.min_rmsd_sample_variance** | Float or list → one min_rmsd figure set per value. |
| **resume** | If true: skip complete runs and existing plot/analysis outputs. If false: confirm then delete output_dir and run from scratch. |

---

## 3. Training action (per segment)

Each DistMap and Euclideanizer segment is assigned one of four actions. Same logic for both; Euclideanizer uses `euclideanizer.pt` / `euclideanizer_last.pt`.

```mermaid
flowchart TB
    subgraph action["Training action"]
        A["Run dir, target epochs, config"]
        A --> B{"resume and run complete?"}
        B -->|Yes| C["Skip"]
        B -->|No| D{"Previous segment exists?"}
        D -->|No| E{"resume and best &lt; target?"}
        E -->|Yes| F["Resume from best"]
        E -->|No| G["From scratch"]
        D -->|Yes| H{"resume and best &gt; prev segment?"}
        H -->|Yes| F
        H -->|No| I["Resume from prev last"]
    end
```

---

## 4. Plotting and analysis conditions

```mermaid
flowchart LR
    subgraph plot["Plotting"]
        P1{"plotting.enabled?"}
        P1 -->|No| P2["No plots"]
        P1 -->|Yes| P3{"resume and all present?"}
        P3 -->|Yes| P4["Skip"]
        P3 -->|No| P5["Reconstruction, recon stats, gen_variance"]
    end

    subgraph analysis["Analysis (Euclideanizer only)"]
        A1{"analysis.min_rmsd?"}
        A1 -->|No| A2["No analysis"]
        A1 -->|Yes| A3{"resume and all present?"}
        A3 -->|Yes| A4["Skip"]
        A3 -->|No| A5["Min-RMSD per variance × num_samples"]
    end
```

---

## 5. Output layout

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
    │   └── test_to_train_rmsd.npz   # if analysis.min_rmsd & save_data
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
        │   └── training_video/
        │       └── training_evolution.mp4
        └── euclideanizer/<j>/
            ├── model/
            │   ├── run_config.yaml
            │   ├── euclideanizer.pt
            │   └── euclideanizer_last.pt
            ├── plots/
            └── analysis/min_rmsd/<run_name>/
                ├── min_rmsd_distributions.png
                ├── data/            # if save_data
                └── structures/      # if save_structures_gro
```

---

## Viewing and exporting

- **Preview**: Mermaid blocks render in GitHub, GitLab, and in VS Code/Cursor with a Mermaid extension. For an editor, paste a block into [Mermaid Live](https://mermaid.live/).
- **Export**: In Mermaid Live use Export → PNG/SVG, or from the repo: `npx -y @mermaid-js/mermaid-cli -i PIPELINE_FLOWCHART.md -o pipeline_flowchart.png` (requires Node).

When you add analysis features or seed-level parallelization, update this document so the flow and options stay accurate.
