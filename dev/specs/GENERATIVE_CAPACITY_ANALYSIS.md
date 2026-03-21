# Spec: Generative Capacity Analysis Block

## Overview

The generative capacity analysis answers: **how many generated structures are needed
before additional samples become mostly redundant?**

For each trained Euclideanizer run, this analysis:

1. Generates structures once at `n_max = max(n_structures)`.
2. Computes and stores a full `n_max x n_max` pairwise metric matrix on disk.
3. Evaluates smaller `n` values using nested subsamples of that same matrix
   (no re-generation, no repeated pairwise compute).

There are two independent analysis blocks:

- `analysis.generative_capacity_rmsd`
- `analysis.generative_capacity_q`

Both follow the same structure and output pattern.

---

## 1. Config Schema

```yaml
analysis:
  generative_capacity_rmsd:
    enabled: true
    overwrite_existing: false
    n_structures: [100, 250, 500, 1000, 2500, 5000]
    gen_decode_batch_size: 256
    query_batch_size: 128
    save_data: true
    save_pdf_copy: true

  generative_capacity_q:
    enabled: true
    overwrite_existing: false
    n_structures: [100, 250, 500, 1000, 2500, 5000]
    gen_decode_batch_size: 256
    query_batch_size: 128
    delta: 0.7071
    save_data: true
    save_pdf_copy: true
```

Notes:

- `n_structures` can be a single integer or a list.
- A list must be strictly ascending.
- `n_max = max(n_structures)` is used for one-time generation and one-time full
  pairwise matrix computation.
- `query_batch_size` is required for pairwise-compute chunking and I/O chunking.
- No separate `save_distance_matrix` field: the intermediate full pairwise matrix is
  always created, and its retention is controlled by `save_data`.

---

## 2. Execution Logic

Steps below describe RMSD. Q is identical except:

- pairwise value is Q instead of RMSD
- nearest-neighbour summary is `max` instead of `min`
- `delta` is used for Q calculation

### Step 1 - Generate n_max structures

```python
n_max = max(n_structures)
```

Generate `n_max` structures using the existing generation path in batches of
`gen_decode_batch_size`.

Output:

- `coords_generated` shape `(n_max, n_atoms, 3)`

### Step 2 - Compute full pairwise matrix (always on disk)

The full pairwise matrix is always written to an intermediate file under the analysis
block's data directory. The pipeline should not switch between in-memory vs memmap
heuristics here; use the on-disk matrix path consistently.

Example:

```python
matrix_path = "<run_dir>/analysis/generative_capacity/rmsd/data/pairwise_matrix.npy"
mat = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n_max, n_max))
```

Computation should be chunked by `query_batch_size` so pairwise work is not held in one
large temporary block.

Diagonal policy:

- RMSD matrix diagonal: `np.inf` (exclude self-match from minima)
- Q matrix diagonal: `-np.inf` (exclude self-match from maxima)

### Step 3 - Nested subsampling and per-n distributions

Use one random permutation from global run seed:

```python
rng = np.random.default_rng(seed)
all_indices = rng.permutation(n_max)
```

For each `n` in ascending `n_structures`:

```python
idx = all_indices[:n]
submat = full_matrix[np.ix_(idx, idx)]
nearest = submat.min(axis=1)   # RMSD
# nearest = submat.max(axis=1) # Q
```

The nested index rule ensures each smaller `n` is a subset of larger `n`.

### Step 4 - Save outputs and finalize intermediate file

- Save per-`n` distribution NPZ files only when `save_data: true`.
- Save figure PNG always; save PDF copy when `save_pdf_copy: true`.
- Intermediate full pairwise matrix file:
  - keep when `save_data: true`
  - delete when `save_data: false`

This block is **not used by scoring** in the current design, so matrix retention/deletion
is handled inside this block and is independent of post-scoring cleanup.

---

## 3. Output Directory Structure

Per Euclideanizer run:

```
{seed_dir}/distmap/{dm_run}/euclideanizer/{eu_run}/analysis/
  generative_capacity/
    rmsd/
      data/
        pairwise_matrix.npy           # kept only if save_data: true
        n100_min_rmsd.npz             # only if save_data: true
        n250_min_rmsd.npz             # only if save_data: true
        ...
      generative_capacity_rmsd.png
      generative_capacity_rmsd.pdf    # only if save_pdf_copy: true
    q/
      data/
        pairwise_matrix.npy           # kept only if save_data: true
        n100_max_q.npz                # only if save_data: true
        ...
      generative_capacity_q.png
      generative_capacity_q.pdf       # only if save_pdf_copy: true
```

`{seed_dir}` follows existing pipeline naming (`seed_<n>` or `seed_<n>_split_<frac>`).

---

## 4. Saved Data Format

Each `n{N}_min_rmsd.npz` (or `n{N}_max_q.npz`) contains:

```text
values: float32 array, shape (n,)
n: int
median: float
p25: float
p75: float
seed: int
```

The intermediate matrix file stores the full pairwise matrix for that metric.

---

## 5. Figure Design

One figure per metric block:

- `generative_capacity_rmsd.png`
- `generative_capacity_q.png`
- optional PDF copy when `save_pdf_copy: true`

Layout:

- **Stacked filled histograms:** one **row per** `n` in `n_structures`, **largest `n` at
  the top**; **shared** x-axis and **shared** bin edges from pooled data (1st–99th
  percentile over all `n`). Each row: **borderless filled** bars
  (`plot_config.HIST_FILLED_EDGE_COLOR`), **`HIST_BINS_DEFAULT`** bins, density
  normalization, color from viridis on **`log10(n)`**. **`N = {n}`** in the top-left
  of each row (axes coordinates). Y limits matched across rows. **Vertical** colorbar
  to the **right** of the stack when **two or more** distinct `n`; ticks at min and max
  `n`; label **“Number Of Generated Structures”** (Title Case), `rotation=270`,
  `labelpad=-6`. Single-`n` runs omit the colorbar and use a wider right margin.
- **Figure width / row height:** **`plot_config.GEN_CAP_STACKED_FIGWIDTH`**,
  **`plot_config.GEN_CAP_STACKED_ROW_HEIGHT`**.
- Main axes: plain numeric ticks (**`ScalarFormatter(useOffset=False)`**, no scientific
  `1e7` styling on value axes).

Axis labels:

- RMSD: `"Min RMSD To Nearest Generated Structure (Å)"`
- Q: `"Max Q To Nearest Generated Structure"`

No curve fitting, asymptote overlays, or extra annotations. (Legacy **step** overlay +
horizontal top colorbar remains in **`_distribution_panel`** for unit tests only.)

---

## 6. Validation and Testing

Config validation:

- `n_structures`: positive integers, strictly ascending
- `gen_decode_batch_size`: positive integer
- `query_batch_size`: positive integer
- `delta` (Q block): positive float

Runtime handling:

- If generation count is short of `n_max`, log an error and skip the block.
- If intermediate matrix deletion fails when `save_data: false`, log a warning and continue.

Monotonicity check policy:

- Nested subsampling implies monotonic medians in expectation.
- Monotonicity verification is a **test-suite responsibility**, not a runtime pipeline check.
- Add/maintain tests that assert nested-index behavior and expected monotonic trend on
  controlled synthetic data.

---

## 7. Dashboard Integration

Use the existing dashboard block model (no custom tab system).

Add per-run analysis blocks:

- `generative_capacity_rmsd` -> `analysis/generative_capacity/rmsd/generative_capacity_rmsd.png`
- `generative_capacity_q` -> `analysis/generative_capacity/q/generative_capacity_q.png`

If outputs are missing for a run, the dashboard shows the standard "no outputs" behavior.

---

## 8. Notes for Sample Config

Sample config should include a comment near `n_structures` stating:

- large `n_max` values can be computationally expensive due to `O(n_max^2)` pairwise work,
- choose `n_structures` accordingly for available runtime/storage budgets.