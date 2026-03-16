"""
Setup wizard prompt construction. System prompt and user/retry prompt builders.
See specs/SETUP_WIZARD.md.
"""
from __future__ import annotations

CONVERTER_SYSTEM_PROMPT = """You are helping to generate a Python script that converts raw coordinate data into a single NPZ file for use with a machine learning pipeline.

## NPZ output schema

The converter must produce a single .npz file containing exactly one key: `coords`. The value must be a numeric array of shape (n_structures, n_atoms, 3), with dtype float32 or float64. The file must be written with:

    np.savez_compressed(output_path, coords=array)

No other keys are required. The key name must be exactly `coords`.

## Validation criteria the output must satisfy

- ndim == 3
- shape[2] == 3 (last dimension is x, y, z spatial coordinates)
- shape[0] >= 1
- shape[1] >= 2
- All values finite (no NaN, no inf)
- Dtype must be castable to float32
- The first column must be spatial x (or equivalent); it must NOT be atom index (0, 1, 2, ...). Validation rejects output where the first column is strongly correlated with 0..n_atoms-1.

## Coordinate parsing (critical)

- The three columns of each structure must be **spatial x, y, z** (in any consistent units). The first column must NOT be atom index, residue number, or any non-spatial field.
- For text-based formats (GRO, XYZ, PDB, CSV, etc.): **do not parse x,y,z using fixed character column positions**. Many files use variable spacing, so fixed columns mis-parse (e.g. reading atom number or part of a number as x). Instead, split each line on whitespace and take the **last three numeric fields** as x, y, z (or the three fields that are clearly x, y, z). This ensures correct coordinates regardless of column alignment.
- If the format has a fixed-width spec (e.g. GRO cols 20-27 for x), only use it when you have verified the sample lines actually align; otherwise prefer whitespace splitting for the coordinate fields.

## Converter script requirements

- Must be a standalone Python script.
- Must accept exactly two positional command-line arguments via sys.argv[1] and sys.argv[2]: input path and output path, in that order. The first argument is the path given by the user (either a single file or a directory of files). The second argument is the path where the script must write the output NPZ file.
- Must import only from the Python standard library and numpy; no other dependencies may be assumed present.
- Must handle both cases: (1) input is a single file, and (2) input is a directory of files. In the directory case, all structures found across all files must be combined into a single coords array along axis 0 (so the result has shape (total_structures, n_atoms, 3)).
- Must print a summary on successful completion: number of structures found, number of atoms, output path.
- Must not hardcode any paths; all paths come from the two command-line arguments.
- Must not require modification to run; the user will run it as: python script.py <input_path> <output_path>.

## Failure behavior

If the format cannot be determined from the sample, or the data does not match what the script expects, the script must raise a ValueError with a descriptive message explaining what it expected and what it found. It must never silently produce an array of wrong shape or wrong values. It must never write an NPZ file that fails the validation criteria above."""


def build_user_prompt(samples: list[tuple[str, str]]) -> str:
    """Build the user prompt from the list of (filename, sample_text) tuples."""
    lines = [
        "Generate a Python converter script for the data shown below. The script must read the input (single file or directory of files), parse the coordinate data, and write a single NPZ file with key `coords` and shape (n_structures, n_atoms, 3), using np.savez_compressed(output_path, coords=array).",
        "",
    ]
    n = len(samples)
    for i, (filename, sample_text) in enumerate(samples):
        if n == 1:
            lines.append(f"File: {filename}")
        else:
            lines.append(f"File {i + 1} of {n}: {filename}")
        lines.append("---")
        lines.append(sample_text)
        lines.append("---")
        lines.append("")
    lines.append(
        "Output requirements: a single .npz file with key `coords`, array shape (n_structures, n_atoms, 3), written with np.savez_compressed(output_path, coords=array). All values must be finite; dtype float32 or float64."
    )
    return "\n".join(lines)


def build_retry_prompt(original_user_prompt: str, error_message: str) -> str:
    """Build the retry prompt including the original prompt and the validation error."""
    return f"""{original_user_prompt}

The converter script you generated failed validation with the following error:

{error_message}

Please provide a corrected version of the script that addresses this error.
The output requirements remain the same."""
