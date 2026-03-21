"""
Setup wizard: guides from raw data to a converted NPZ file for pipeline input.
Uses Claude API to generate a custom converter script. Requires ANTHROPIC_API_KEY.
See dev/specs/SETUP_WIZARD.md (under the Pipeline tree).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .wizard_prompts import (
    CONVERTER_SYSTEM_PROMPT,
    build_retry_prompt,
    build_user_prompt,
)

API_KEY_ERROR_MESSAGE = """ERROR: ANTHROPIC_API_KEY not found in environment.

The setup wizard uses the Claude API to generate a custom data converter
for your dataset. This requires an Anthropic API key.

To set your key:
    export ANTHROPIC_API_KEY=your-key-here

The API key is never written to disk or logged by the wizard.

If you prefer not to use the API, the pipeline's native input format is
NPZ. To convert your data manually:

    import numpy as np
    coords = ...  # your (n_structures, n_atoms, 3) array, any numeric dtype
    np.savez_compressed("my_data.npz", coords=coords)

See README.md for full format details.
"""

CONVERTER_SUBPROCESS_TIMEOUT = 120

DISCLAIMER = """
The setup wizard uses an LLM to generate a converter script from a sample of your data.
LLM-generated code can contain errors. Validation checks the output NPZ for shape and
basic sanity (e.g. that coordinates are not plainly wrong), and a coordinate preview
is written for visual inspection — but correctness of parsing is not guaranteed.

By proceeding, you agree to review the generated converter and the coordinate preview,
and to take responsibility for verifying that the conversion accurately represents your
input data before using the NPZ in the pipeline.
"""


def _red(s: str) -> str:
    """Bold red for TTY (matches run.py styling); plain text when not a TTY."""
    if sys.stdout.isatty():
        return f"\033[1;31m{s}\033[0m"
    return s


def confirm_disclaimer() -> None:
    """Print the disclaimer and require the user to type Accept to continue. Exit otherwise."""
    for line in DISCLAIMER.strip().split("\n"):
        print(_red(line))
    print()
    reply = input(_red('Type "Accept" to proceed, or anything else to abort: ')).strip()
    if reply.lower() != "accept":
        print(_red("Aborted."))
        sys.exit(0)


def main() -> None:
    """Top-level orchestrator. Parses args, runs wizard steps, handles errors."""
    parser = argparse.ArgumentParser(
        description="Setup wizard: convert raw data to NPZ for pipeline input (requires ANTHROPIC_API_KEY)."
    )
    parser.add_argument("--data", required=True, help="Path to a single data file or directory of data files")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the converted .npz file (must end with .npz if set; default: same dir as input, stem from input name)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum number of files to process when --data is a directory (default: 50); use --confirm-large if raising above 50",
    )
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=100,
        help="Number of lines to sample from each input file for the API (default: 100); binary files send first 2048 bytes as hex",
    )
    parser.add_argument(
        "--confirm-large",
        action="store_true",
        help="Suppress interactive confirmation when --data is a directory (required when --max-files > 50)",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        parser.error(f"--data path does not exist: {data_path}")

    output_path = resolve_output_path(data_path, args.output)
    api_key = check_api_key()
    confirm_disclaimer()
    samples = collect_samples(data_path, args.sample_lines, args.max_files, args.confirm_large)
    user_prompt = build_user_prompt(samples)

    try:
        script_text = call_claude(api_key, CONVERTER_SYSTEM_PROMPT, user_prompt)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"API call failed: {e}", file=sys.stderr)
        sys.exit(1)

    input_stem = _input_stem(data_path)
    success, error_message, temp_npz_path = validate_converter(script_text, data_path)
    if not success:
        script_text_retry = None
        try:
            retry_prompt = build_retry_prompt(user_prompt, error_message)
            script_text_retry = call_claude(api_key, CONVERTER_SYSTEM_PROMPT, retry_prompt)
            success, error_message, temp_npz_path = validate_converter(script_text_retry, data_path)
        except (ValueError, Exception):
            pass
        if not success:
            saved_failed = save_converter(script_text_retry if script_text_retry is not None else script_text, input_stem, failed=True)
            print(f"Converter validation failed (retry exhausted). Failing script saved to: {saved_failed}", file=sys.stderr)
            print(error_message, file=sys.stderr)
            sys.exit(1)
        script_text = script_text_retry

    converter_path = save_converter(script_text, input_stem, failed=False)
    if os.path.exists(output_path):
        print(f"Warning: output path already exists; overwriting: {output_path}")
    shutil.copy2(temp_npz_path, output_path)
    os.remove(temp_npz_path)

    with np.load(output_path, allow_pickle=False) as data:
        coords = data["coords"]
        n_structures, n_atoms, _ = coords.shape

    preview_path = _write_coordinate_preview(output_path)
    if preview_path:
        print(f"Coordinate preview saved: {preview_path}")
        print("  (Inspect that structures look like your input; axes are x vs y from coords.)")

    print_getting_started(data_path, output_path, n_structures, n_atoms, converter_path, preview_path)


def check_api_key() -> str:
    """Read ANTHROPIC_API_KEY from environment. Raise SystemExit with error message if absent."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not key.strip():
        print(API_KEY_ERROR_MESSAGE, file=sys.stderr)
        sys.exit(1)
    return key.strip()


def collect_samples(
    data_path: str,
    sample_lines: int,
    max_files: int,
    confirm_large: bool,
) -> list[tuple[str, str]]:
    """Collect sample text from input file(s). Handle directory confirmation and binary files."""
    if os.path.isfile(data_path):
        name = os.path.basename(data_path)
        samples = [(name, _sample_one_file(data_path, name, sample_lines))]
        return samples

    names = sorted(f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)))
    n_files = len(names)
    if n_files > max_files:
        print(
            f"ERROR: Directory contains {n_files} files, which exceeds --max-files={max_files}.",
            file=sys.stderr,
        )
        print(
            "Raise the limit with --max-files and pass --confirm-large, or narrow the input directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {n_files} files in {data_path}:")
    for name in names:
        print(f"  {name}")
    print()
    if not confirm_large:
        print(
            f"The wizard will send up to {sample_lines} lines from each file to the Claude API\n"
            f"to generate a converter. This will consume approximately {n_files * sample_lines} tokens.\n"
            "API usage is billed to the account associated with your ANTHROPIC_API_KEY.\n"
        )
        reply = input('Type "yes" to continue or anything else to abort: ').strip()
        if reply.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    samples = []
    for name in names:
        full = os.path.join(data_path, name)
        samples.append((name, _sample_one_file(full, name, sample_lines)))
    return samples


def _sample_one_file(full_path: str, name: str, sample_lines: int) -> str:
    """Read first sample_lines lines, or first 2048 bytes as hex if binary."""
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = []
            for _ in range(sample_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n\r"))
            return "\n".join(lines)
    except UnicodeDecodeError:
        with open(full_path, "rb") as f:
            raw = f.read(2048)
        return raw.hex()


def call_claude(api_key: str, system_prompt: str, user_prompt: str) -> str:
    """Call Claude API; extract and return the first Python code block. Raise ValueError if none found."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ValueError(
            "The setup wizard requires the anthropic package. Install with: pip install anthropic"
        ) from None

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if not match:
        print("No Python code block in API response. Full response:", file=sys.stderr)
        print(text, file=sys.stderr)
        raise ValueError("No Python code block found in API response.")
    return match.group(1).strip()


def _coords_first_column_looks_like_atom_index(coords: np.ndarray) -> bool:
    """True if the first column is strongly index-like (0,1,...,N-1), indicating mis-parse."""
    n_atoms = coords.shape[1]
    if n_atoms < 3:
        return False
    idx = np.arange(n_atoms, dtype=np.float64)
    # Check first few structures; if any pass, we flag it
    n_check = min(3, coords.shape[0])
    for s in range(n_check):
        c0 = np.asarray(coords[s, :, 0], dtype=np.float64)
        if np.allclose(c0, idx, atol=0.6):
            return True
        r = np.corrcoef(c0, idx)[0, 1]
        if np.isfinite(r) and r > 0.995 and np.all(np.diff(c0) > 1e-6):
            return True
    return False


def validate_converter(script_text: str, input_path: str) -> tuple[bool, str, str | None]:
    """
    Write script to temp file, run with input_path and temp output path, validate NPZ.
    Returns (success, error_message, output_npz_path). output_npz_path is set only when success is True.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_text)
        temp_script = f.name
    fd, temp_npz = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, temp_script, input_path, temp_npz],
            capture_output=True,
            text=True,
            timeout=CONVERTER_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        os.unlink(temp_script)
        os.unlink(temp_npz)
        return (False, "Converter script timed out (120s).", None)
    finally:
        os.unlink(temp_script)

    if result.returncode != 0:
        err = result.stderr or result.stdout or "Unknown error"
        os.unlink(temp_npz)
        return (False, err, None)

    try:
        data = np.load(temp_npz, allow_pickle=False)
    except Exception as e:
        os.unlink(temp_npz)
        return (False, f"Output file could not be loaded as NPZ: {e}", None)
    try:
        coords = data["coords"]
    except KeyError:
        data.close()
        os.unlink(temp_npz)
        return (False, "Output NPZ does not contain required key 'coords'.", None)
    if coords.ndim != 3:
        data.close()
        os.unlink(temp_npz)
        return (False, f"coords must have 3 dimensions, got ndim={coords.ndim}.", None)
    if coords.shape[2] != 3:
        data.close()
        os.unlink(temp_npz)
        return (False, f"coords shape[-1] must be 3, got {coords.shape[2]}.", None)
    if coords.shape[0] < 1:
        data.close()
        os.unlink(temp_npz)
        return (False, f"coords must have at least one structure, got shape[0]={coords.shape[0]}.", None)
    if coords.shape[1] < 2:
        data.close()
        os.unlink(temp_npz)
        return (False, f"coords must have at least 2 atoms, got shape[1]={coords.shape[1]}.", None)
    if not np.isfinite(coords).all():
        data.close()
        os.unlink(temp_npz)
        return (False, "coords contains non-finite values (NaN or inf).", None)
    try:
        np.asarray(coords, dtype=np.float32)
    except (ValueError, TypeError):
        data.close()
        os.unlink(temp_npz)
        return (False, "coords dtype is not castable to float32.", None)

    # Reject if first column looks like atom index (0,1,2,...) — indicates fixed-width
    # parsing on variable-spaced lines (e.g. GRO), which puts atom number in column 0.
    if _coords_first_column_looks_like_atom_index(coords):
        data.close()
        os.unlink(temp_npz)
        return (
            False,
            "The first column of coords appears to be atom index (0, 1, 2, ...) instead of "
            "spatial x. This usually means the converter used fixed character positions for x,y,z "
            "on a file with variable spacing (e.g. GRO). Parse x,y,z from the last three "
            "whitespace-separated numeric fields per line, not fixed columns.",
            None,
        )

    data.close()
    return (True, "", temp_npz)


def save_converter(script_text: str, input_stem: str, failed: bool) -> str:
    """Save converter script to setup_wizard_scripts/. On success use numeric suffix if collision. On failed overwrite _FAILED. Returns final path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(root, "setup_wizard_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    if failed:
        name = f"{input_stem}_converter_FAILED.py"
        path = os.path.join(scripts_dir, name)
        with open(path, "w") as f:
            f.write(script_text)
        return path
    base = f"{input_stem}_converter.py"
    path = os.path.join(scripts_dir, base)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(script_text)
        return path
    n = 2
    while True:
        path = os.path.join(scripts_dir, f"{input_stem}_converter_{n}.py")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(script_text)
            return path
        n += 1


def _write_coordinate_preview(npz_path: str) -> str | None:
    """
    Write a coordinate preview image next to the NPZ using the same pipeline
    plotting as the training visualization (Exp. Structure row). Lets the user
    confirm that converted structures look like their input (x,y,z not index).
    Returns the path of the written image, or None if plotting failed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .training_visualization import _plot_chain_2d, _RC
    except Exception:
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            coords = np.asarray(data["coords"], dtype=np.float32)
    except Exception:
        return None
    n_structures, n_atoms, _ = coords.shape
    if n_structures < 1 or n_atoms < 2:
        return None
    n_probe = min(6, n_structures)
    if n_structures <= n_probe:
        indices = list(range(n_structures))
    else:
        indices = [int(round(i * (n_structures - 1) / (n_probe - 1))) for i in range(n_probe)]
    probe = coords[indices].copy()
    probe -= probe.mean(axis=1, keepdims=True)
    axis_lim = float(np.abs(probe).max()) * 1.15
    dirname = os.path.dirname(npz_path)
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(dirname, f"{stem}_coordinate_preview.png")
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1, n_probe, figsize=(2.0 * n_probe, 2.0), facecolor=_RC["figure.facecolor"]
        )
        if n_probe == 1:
            axes = [axes]
        for j, ax in enumerate(axes):
            _plot_chain_2d(ax, probe[j], lim=axis_lim)
        fig.suptitle(
            "Coordinate preview (x vs y from converted NPZ — check against your input)",
            fontsize=9, color=_RC["axes.labelcolor"],
        )
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, facecolor=_RC["figure.facecolor"])
    plt.close(fig)
    return out_path


def resolve_output_path(input_path: str, output_arg: str | None) -> str:
    """Determine final NPZ output path from input path and optional --output. Raise SystemExit if --output given but does not end with .npz."""
    if output_arg and str(output_arg).strip():
        out = str(output_arg).strip()
        if not out.endswith(".npz"):
            print("ERROR: --output must end with .npz", file=sys.stderr)
            sys.exit(1)
        return os.path.abspath(out)
    input_path = os.path.abspath(input_path)
    parent = os.path.dirname(input_path)
    if os.path.isfile(input_path):
        stem = os.path.splitext(os.path.basename(input_path))[0]
    else:
        stem = os.path.basename(input_path.rstrip(os.sep))
    return os.path.join(parent, f"{stem}.npz")


def _input_stem(data_path: str) -> str:
    """Return stem for naming the converter script (filename stem or directory name)."""
    data_path = os.path.abspath(data_path)
    if os.path.isfile(data_path):
        return os.path.splitext(os.path.basename(data_path))[0]
    return os.path.basename(data_path.rstrip(os.sep))


def print_getting_started(
    input_path: str,
    output_npz: str,
    n_structures: int,
    n_atoms: int,
    converter_path: str,
    coordinate_preview_path: str | None = None,
) -> None:
    """Print completion summary and getting-started workflow."""
    converter_rel = os.path.basename(converter_path)
    print()
    print("=" * 60)
    print("  DATA CONVERSION COMPLETE")
    print("=" * 60)
    print()
    print(f"Input:        {input_path}")
    print(f"Output NPZ:   {output_npz}")
    print(f"Structures:   {n_structures}")
    print(f"Atoms:        {n_atoms}")
    print(f"Converter:    setup_wizard_scripts/{converter_rel}")
    if coordinate_preview_path:
        print(f"Preview:      {coordinate_preview_path}")
    print()
    print("To re-run the converter on new data:")
    print(f"    python setup_wizard_scripts/{converter_rel} /path/to/new/data output.npz")
    print()
    print("=" * 60)
    print("  GETTING STARTED WITH THE PIPELINE")
    print("=" * 60)
    print()
    print("Your data is now ready. Here is the recommended workflow:")
    print()
    if coordinate_preview_path:
        print("0. CHECK THE COORDINATE PREVIEW")
        print("   Open the coordinate_preview image next to your NPZ. Confirm that the")
        print("   plotted structures look like your input (real 3D shapes). If you see")
        print("   a horizontal line or 'atom index vs value', the converter parsed")
        print("   coordinates incorrectly; fix the converter and re-run.")
        print()
    print("1. CONFIGURE")
    print("   Copy samples/config_sample.yaml to your working directory and edit it.")
    print("   Set data.path to your output NPZ file.")
    print("   Set output_dir to a directory where run outputs should be written.")
    print()
    print("2. EXPLORE THE PIPELINE WITH A SMALL RUN")
    print("   Before committing to a full training run or HPO, run the pipeline with")
    print("   a small, simple config: one DistMap and one Euclideanizer configuration,")
    print("   a modest number of epochs (enough to see the loss curve stabilize or not),")
    print("   with plotting enabled. Inspect the reconstruction plots, gen_variance plots,")
    print("   and loss curves to get a feel for how the pipeline interacts with your data.")
    print("   See README.md for a description of each plot type.")
    print()
    print("   Example command:")
    print("       python run.py --config your_config.yaml")
    print()
    print("3. CALIBRATE BATCH SIZE AND LEARNING RATE")
    print("   Once you have a stable config that produces sensible outputs, use the")
    print("   batch size benchmark to find the optimal batch size and learning rate")
    print("   for your hardware and dataset:")
    print()
    print(f"       python tests/benchmark_batch_size.py --config your_config.yaml \\")
    print(f"           --data {output_npz} --mode both \\")
    print("           --batch-sizes 32 64 128 256 --learning-rates 1e-4 5e-4 1e-3")
    print()
    print("   See README.md (Benchmark and calibration section) for how to interpret results.")
    print()
    print("4. RUN HPO")
    print("   With a calibrated batch size and learning rate, set up an HPO config")
    print("   based on samples/hpo_config.yaml and run hyperparameter optimization:")
    print()
    print(f"       python run_hpo.py --config your_hpo_config.yaml --data {output_npz}")
    print()
    print("   See README.md (Quick start / Hyperparameter optimization) for HPO config details.")
    print()
    print("For full documentation see README.md.")
    print("=" * 60)
