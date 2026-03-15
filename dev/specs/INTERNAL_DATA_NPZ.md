# SPEC: NPZ Internal Format Migration

## Overview

This spec covers the replacement of the GRO-based input format with NPZ as the canonical internal data format for the pipeline, the removal of all legacy GRO parsing code, and the revision of `gro_io.py` to produce maximally compatible GRO output for downstream visualization tools.

---

## Motivation

The GRO format was designed for GROMACS and carries significant legacy constraints: fixed-width column formatting, a bespoke title line convention (`Chromosome...`) that is not part of the GRO standard, implicit unit assumptions, and a requirement for a custom parser rather than any standard library function. As a general-purpose ML pipeline, the internal data format should be as simple and unambiguous as possible. NPZ satisfies this: it is loaded with a single standard library call, compresses well, is completely unambiguous in schema, and requires no column-width parsing. The `load_data` function in `src/utils.py` already immediately converts GRO to a numpy array — NPZ makes this conversion trivial and eliminates the fragile parsing layer entirely.

GRO output for generated structures is retained because users need to visualize results in tools like VMD and PyMOL. However, the current output does not conform to the most canonical GRO spec and is revised here to maximize compatibility with all downstream tools.

---

## NPZ Input Format Specification

### Schema

The pipeline accepts a single `.npz` file as its dataset. The file must contain exactly one required key:

**`coords`** — a numeric array of shape `(n_structures, n_atoms, 3)`. Dtype may be float32 or float64; the pipeline will cast to float32 internally. The three values along the final axis represent x, y, z spatial coordinates. Units are not specified by the format; the user is responsible for consistency within and across their dataset. The pipeline does not assume or enforce any unit system.

No other keys are required. Additional keys present in the file are ignored silently.

### Validation

`load_data` must raise `ValueError` with a clear message for each of the following:

- The file does not exist or cannot be read
- The file does not contain a `coords` key
- `coords` has ndim != 3
- `coords.shape[2] != 3`
- `coords.shape[0] < 1` (zero structures)
- `coords.shape[1] < 2` (fewer than 2 atoms; cannot form a distance map)
- `coords` contains any non-finite values (NaN or inf)

### Creating a conforming NPZ file

Any user converting their own data needs only:

```python
import numpy as np
coords = ...  # your (n_structures, n_atoms, 3) array
np.savez_compressed("my_data.npz", coords=coords)
```

This is the only instruction users who convert manually need to follow.

---

## Code Changes

### `src/utils.py`

Replace the existing `load_data` function entirely. The new implementation signature is:

```python
def load_data(npz_path: str) -> np.ndarray:
```

Behavior:

- Opens the file with `np.load(npz_path, allow_pickle=False)`
- Extracts `data["coords"]`
- Validates all conditions listed above, raising `ValueError` with a descriptive message for each
- Casts to `np.float32` if not already
- Returns the array

The old GRO parsing logic (title line detection, fixed-width column parsing, multi-frame loop) is deleted entirely. No compatibility shim is provided. The function signature is unchanged (`path: str -> np.ndarray`) so all call sites remain valid.

Remove the `Chromosome` title line convention from all documentation and comments throughout the file.

### `src/gro_io.py`

The function signature, purpose, and output location logic are unchanged. What changes is the format of the written GRO file.

#### Canonical GRO format requirements

The title line must be a plain string with no special prefix convention. The default title used when none is specified is `"generated frame {i}"` where `i` is the zero-indexed frame number.

Atom lines must follow the standard fixed-width GRO column layout exactly:

```
%5d%-5s%5s%5d%8.3f%8.3f%8.3f
```

Columns: residue number (1-indexed, wraps at 99999), residue name, atom name, atom number (1-indexed, wraps at 99999), x, y, z. No velocity columns are written. Velocities are the most common source of incompatibility between GRO readers; omitting them produces a file that is valid for all tools that accept GRO.

The box vector line must be written as three space-separated float values on a single line:

```
   0.00000   0.00000   0.00000
```

Three zeros indicates an undefined or non-periodic box, which is the correct representation for structural ensembles that are not simulation frames with a defined periodic cell. All major GRO readers (MDAnalysis, VMD via molfile plugin, PyMOL) accept this form.

Residue and atom numbering wraps at 99999 using modulo arithmetic, not truncation. This is the GROMACS convention for large structures.

The function must not write a `Chromosome` or any other non-standard prefix in the title line. The old `title_prefix` parameter is renamed to `title` and its default value changes from `"generated"` to `None`, in which case the per-frame default above is used.

#### Function signature after revision

```python
def write_structures_gro(
    coords: np.ndarray,
    directory: str,
    *,
    title: str | None = None,
    residue_name: str = "MOL",
    atom_name: str = "CA",
    display_root: str | None = None,
    filename: str = DEFAULT_STRUCTURES_FILENAME,
) -> list[str]:
```

The `residue_name` default changes from `"STRUC"` to `"MOL"`, which is more universally recognized as a generic residue name across VMD, PyMOL, and MDAnalysis.

---

## Test Data

### `tests/test_data/generate_spheres.py`

The script is revised to output `spheres.npz` instead of `spheres.gro`. The `write_gro` function is removed. The final output is:

```python
np.savez_compressed(out, coords=coords.astype(np.float32))
```

The default output path changes from `tests/test_data/spheres.gro` to `tests/test_data/spheres.npz`.

The `--output` CLI argument, `--num-structures`, `--beads`, `--radius-min`, `--radius-max`, and `--seed` arguments are all retained with identical semantics.

The `.gitignore` entry for `tests/test_data/spheres.gro` is updated to `tests/test_data/spheres.npz`.

---

## Config Changes

### `data.path`

The config key `data.path` continues to hold the path to the dataset file. No key name change. The expected extension is now `.npz`. The validation in `src/config.py` does not enforce the extension (the user might have a file named without extension) but the error message from `load_data` will be clear if a non-NPZ file is passed.

All sample configs (`samples/config_sample.yaml`, `samples/config_sample_hpo.yaml`, `tests/config_test.yaml`, `tests/config_smoke.yaml`) have their `data.path` values updated to reference `spheres.npz` where applicable.

---

## Files to Update

The following files require changes beyond what is described above:

- `README.md` — all references to GRO input format, the `Chromosome` title line convention, the `load_data` description, and the bundled dataset description
- `samples/config_sample.yaml` — `data.path` value
- `samples/config_sample_hpo.yaml` — `data.path` value and any comments referencing GRO
- `tests/config_test.yaml` — `data.path` value
- `tests/config_smoke.yaml` — `data.path` value
- `tests/test_utils_and_config.py` — the `test_load_data_valid_gro` test is replaced with `test_load_data_valid_npz`; the `test_gro_roundtrip` and `test_gro_single_structure` tests are updated to use `write_structures_gro` with the revised signature; all GRO-specific error message tests are updated to match new NPZ error messages
- `tests/test_kabsch_rmsd.py` — `gro_path` references updated to `npz_path`; `load_data` call is unchanged
- Any inline comments throughout `src/` that reference the GRO format or `Chromosome` title convention

---

## What Is Not Changed

- All downstream processing after `load_data` returns the numpy array. The rest of the pipeline sees `(n_structures, n_atoms, 3)` float32 exactly as before.
- `gro_io.py` is retained and its output is still GRO. Only the format of that output changes.
- The `write_structures_gro` function continues to be called from `run.py` for saving generated structures when `save_structures_gro: true`.
- The `DEFAULT_STRUCTURES_FILENAME` constant in `gro_io.py` remains `"structures.gro"`.
- No changes to any model, loss, training, analysis, scoring, or dashboard code.