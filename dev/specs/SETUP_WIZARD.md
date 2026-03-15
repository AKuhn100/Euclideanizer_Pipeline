# SPEC: Setup Wizard

## Overview

The setup wizard is a standalone entry point that guides a new user from raw data in an unknown format to a converted NPZ file ready for pipeline input, with a clear description of the recommended workflow for getting started with training. It is designed to be the first thing a new user runs. It requires an Anthropic API key in the environment to function; this requirement is non-negotiable and the wizard errors clearly if the key is absent.

---

## File Layout

```
run_setup_wizard.py               # entry point, root directory alongside run.py
src/
  wizard.py                       # all wizard logic
  wizard_prompts.py               # system prompt and user prompt construction
setup_wizard_scripts/             # generated converter scripts, created on first run
  .gitkeep
```

`setup_wizard_scripts/` is created at the project root on first wizard run if it does not exist. It is not gitignored — generated scripts should be committed for auditability and reproducibility. Any script that fails validation is also saved here with a `_FAILED` suffix so the user can inspect what went wrong.

---

## Entry Point: `run_setup_wizard.py`

Minimal. Adds pipeline root to `sys.path`, imports and calls `src.wizard.main()`. No logic lives here.

---

## CLI Interface

```
python run_setup_wizard.py --data PATH [--output PATH] [--max-files N] [--sample-lines N] [--confirm-large]
```

### `--data PATH` (required)

Path to either a single data file or a directory of data files. If a directory is given the wizard samples from all files within it, subject to `--max-files`. Subdirectories are not recursed into.

### `--output PATH` (optional)

Output path for the converted `.npz` file. Must end with `.npz` if specified; the wizard raises an argument error before any other processing if this condition is not met. If not specified, the output is written to:

- The same directory as the input file (single file mode), with the stem derived from the input filename
- The same directory as the input directory (directory mode), with the stem derived from the directory name

Examples: `--data /data/my_coords.h5` produces `/data/my_coords.npz` by default. `--data /data/run_files/` produces `/data/run_files.npz` by default.

### `--max-files N` (optional, default: 50)

Maximum number of files to process when `--data` is a directory. If the directory contains more than `N` files, the wizard errors before any API call is made and instructs the user to either raise the limit with `--max-files` and pass `--confirm-large`, or narrow the input directory. Files beyond `N` are never silently ignored; the error is explicit.

### `--sample-lines N` (optional, default: 100)

Number of lines to sample from each input file and send to Claude. Lines are taken from the beginning of the file. For binary files (detected by attempting UTF-8 decode of the first 2048 bytes and catching `UnicodeDecodeError`) a hex dump of the first 2048 bytes is sent instead.

### `--confirm-large` (optional flag)

Required when `--max-files` is set above the default of 50. Suppresses the large-batch interactive confirmation prompt. Without this flag, the wizard prompts interactively even if `--max-files` is explicitly set above 50.

---

## API Key Handling

Before any other action, `src/wizard.py` checks for `ANTHROPIC_API_KEY` in the environment via `os.environ.get`. If absent, the wizard prints the following message and exits with a non-zero code:

```
ERROR: ANTHROPIC_API_KEY not found in environment.

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
```

The key is read from the environment once at startup and passed to the API call as a parameter. It is never written to any file, logged, printed, or included in generated converter scripts.

---

## Directory Input Handling

When `--data` is a directory, the following always occurs regardless of file count:

1. The wizard scans the directory for files (non-recursive), collects their names, and prints the count and list.
2. The wizard prints a cost notice and prompts for interactive confirmation unless `--confirm-large` is passed.
3. If the file count exceeds `--max-files`, the wizard errors before any API call is made.
4. If confirmed, the wizard samples from each file and concatenates the samples into a single prompt, clearly delimited by filename headers.

The interactive confirmation prompt for any directory input is:

```
Found N files in PATH:
  file1.ext
  file2.ext
  ...

The wizard will send up to SAMPLE_LINES lines from each file to the Claude API
to generate a converter. This will consume approximately N * SAMPLE_LINES tokens.
API usage is billed to the account associated with your ANTHROPIC_API_KEY.

Type "yes" to continue or anything else to abort:
```

If the user types anything other than `"yes"`, the wizard prints `Aborted.` and exits with code zero.

---

## Converter Generation

### Prompt construction (`src/wizard_prompts.py`)

#### System prompt

The system prompt is a module-level string constant `CONVERTER_SYSTEM_PROMPT`. It does not vary by input. It must contain the following in full:

**1. The NPZ output schema:**

The converter must produce a single `.npz` file containing one key, `coords`, holding a numeric array of shape `(n_structures, n_atoms, 3)` where dtype is float32 or float64. The file must be written with `np.savez_compressed(output_path, coords=array)`. No other keys are required. The `coords` key is required exactly and by that name.

**2. The validation criteria the output must satisfy:**

- `ndim == 3`
- `shape[2] == 3`
- `shape[0] >= 1`
- `shape[1] >= 2`
- All values finite (no NaN, no inf)
- Dtype castable to float32

**3. Converter script requirements:**

- Must be a standalone Python script
- Must accept exactly two positional command-line arguments via `sys.argv[1]` and `sys.argv[2]`: input path and output path, in that order
- Must import only from the Python standard library and numpy; no other dependencies may be assumed present
- Must handle both cases: input is a single file, and input is a directory of files. In the directory case, all structures found across all files must be combined into a single `coords` array along axis 0
- Must print a summary on successful completion: number of structures found, number of atoms, output path
- Must not hardcode any paths
- Must not require modification to run

**4. Failure behavior:**

If the format cannot be determined from the sample, the script must raise a `ValueError` with a descriptive message explaining what it expected and what it found. It must never silently produce an array of wrong shape or wrong values.

#### User prompt

The user prompt is constructed by `build_user_prompt(samples)` at call time. It contains:

- A statement of the task: generate a Python converter script for the data shown
- The sample data, formatted as follows

For a single file:

```
File: filename.ext
---
[first N lines of the file, or hex dump for binary]
---
```

For a directory:

```
File 1 of M: filename1.ext
---
[first N lines]
---

File 2 of M: filename2.ext
---
[first N lines]
---
```

- A closing restatement of the output requirements (shape, key name, `np.savez_compressed`)

#### Retry prompt

`build_retry_prompt(original_user_prompt, error_message)` returns a prompt that includes the full original user prompt followed by:

```
The converter script you generated failed validation with the following error:

{error_message}

Please provide a corrected version of the script that addresses this error.
The output requirements remain the same.
```

### API call

The API call uses the standard Anthropic messages endpoint with the following fixed parameters:

- Model: `claude-sonnet-4-6`
- `max_tokens`: 4096
- System: `CONVERTER_SYSTEM_PROMPT`
- Messages: the user prompt as a single user turn

The response is expected to contain a single Python code block delimited by ````python` and `````. The wizard extracts everything between the first occurrence of these markers. If no code block is found in the response, the wizard exits with a non-zero code and prints the full response text to help the user debug.

### Validation loop

After extracting the converter script text, the wizard:

1. Writes the script to a temporary file in the system temp directory
2. Runs it via `subprocess.run([sys.executable, temp_script, str(input_path), str(temp_output_path)], capture_output=True, text=True, timeout=120)`
3. If the subprocess exits non-zero, stderr is captured and treated as the failure message
4. If the subprocess exits zero, attempts `np.load(temp_output_path, allow_pickle=False)` and validates the `coords` key against all criteria in the NPZ spec above
5. If validation passes, proceeds to saving
6. If validation fails on the first attempt, makes exactly one retry: sends `build_retry_prompt(original_user_prompt, error_message)` to the same model with the same parameters
7. If the retry also fails, saves the failing script with a `_FAILED` suffix (see below), prints the final error, and exits with a non-zero code

The timeout of 120 seconds is intentionally generous to accommodate formats that require reading large files.

### Saving the converter script

On success, the converter script is written to:

```
setup_wizard_scripts/{input_stem}_converter.py
```

Where `input_stem` is the stem of the input file (single file mode) or the name of the input directory (directory mode).

If a file with that name already exists, it is not overwritten. A numeric suffix is added instead: `{input_stem}_converter_2.py`, `{input_stem}_converter_3.py`, and so on.

On failure (both attempts exhausted), the failing script is saved to:

```
setup_wizard_scripts/{input_stem}_converter_FAILED.py
```

If a `_FAILED` file already exists for that stem it is overwritten, since it represents the most recent failure attempt.

---

## Output

On success, the wizard prints:

```
============================================================
  DATA CONVERSION COMPLETE
============================================================

Input:        /path/to/input
Output NPZ:   /path/to/output.npz
Structures:   N
Atoms:        M
Converter:    setup_wizard_scripts/mydata_converter.py

To re-run the converter on new data:
    python setup_wizard_scripts/mydata_converter.py /path/to/new/data output.npz

============================================================
  GETTING STARTED WITH THE PIPELINE
============================================================

Your data is now ready. Here is the recommended workflow:

1. CONFIGURE
   Copy samples/config_sample.yaml to your working directory and edit it.
   Set data.path to your output NPZ file.
   Set output_dir to a directory where run outputs should be written.

2. EXPLORE THE PIPELINE WITH A SMALL RUN
   Before committing to a full training run or HPO, run the pipeline with
   a small, simple config: one DistMap and one Euclideanizer configuration,
   a modest number of epochs (enough to see the loss curve stabilize or not),
   with plotting enabled. Inspect the reconstruction plots, gen_variance plots,
   and loss curves to get a feel for how the pipeline interacts with your data.
   See README.md for a description of each plot type.

   Example command:
       python run.py --config your_config.yaml

3. CALIBRATE BATCH SIZE AND LEARNING RATE
   Once you have a stable config that produces sensible outputs, use the
   batch size benchmark to find the optimal batch size and learning rate
   for your hardware and dataset:

       python tests/benchmark_batch_size.py --config your_config.yaml \
           --data /path/to/output.npz --mode both \
           --batch-sizes 32 64 128 256 --learning-rates 1e-4 5e-4 1e-3

   See README.md (Benchmark and calibration section) for how to interpret results.

4. RUN HPO
   With a calibrated batch size and learning rate, set up an HPO config
   based on samples/hpo_config.yaml and run hyperparameter optimization:

       python run_hpo.py --config your_hpo_config.yaml --data /path/to/output.npz

   See README.md (Quick start / Hyperparameter optimization) for HPO config details.

For full documentation see README.md.
============================================================
```

The wizard does not suggest specific hyperparameter values, config grids, learning rate ranges, or any other numerical guidance at any point. The workflow description is procedural only.

---

## `src/wizard.py` Structure

The module contains the following functions. No function has code-side defaults for behavior-affecting parameters; all defaults live in the CLI argument parser and are documented in this spec.

### `main()`

Top-level orchestrator called by `run_setup_wizard.py`. Parses args, calls remaining functions in order, handles all top-level error cases.

### `check_api_key()`

Reads `ANTHROPIC_API_KEY` from `os.environ`. Returns the key string if present. Raises `SystemExit` with the full error message specified above if absent.

### `collect_samples(data_path, sample_lines, max_files, confirm_large)`

Returns a list of `(filename, sample_text)` tuples. Handles single file and directory cases. Performs all interactive confirmation for directory input. For binary files, sends a hex dump. Raises `SystemExit` on file count exceeded or user abort.

### `call_claude(api_key, system_prompt, user_prompt)`

Makes the API call, extracts the Python code block from the response. Raises `ValueError` with the full response text if no code block is found.

### `validate_converter(script_text, input_path)`

Writes script to temp file, runs subprocess with 120-second timeout, validates output NPZ against all criteria. Returns `(success: bool, error_message: str, output_npz_path: str | None)`.

### `save_converter(script_text, input_stem, failed)`

Saves to `setup_wizard_scripts/`. When `failed=False`, handles name collision with numeric suffix. When `failed=True`, uses `_FAILED` suffix and overwrites. Returns the final saved path.

### `resolve_output_path(input_path, output_arg)`

Determines the final NPZ output path from the input path and optional CLI argument. Raises argument error if `output_arg` is specified but does not end with `.npz`.

### `print_getting_started(output_npz, n_structures, n_atoms, converter_path)`

Prints the full completion summary and workflow guidance as shown above.

---

## `src/wizard_prompts.py` Structure

### `CONVERTER_SYSTEM_PROMPT`

Module-level string constant containing the full system prompt as described above. A constant, not a function, because it does not vary by input.

### `build_user_prompt(samples)`

Takes the list of `(filename, sample_text)` tuples from `collect_samples` and returns the formatted user prompt string.

### `build_retry_prompt(original_user_prompt, error_message)`

Returns a retry prompt that includes the original user prompt and the failure output, formatted as described above.

---

## Error Handling Summary


| Condition                                                  | Behavior                                                                                     |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| No `ANTHROPIC_API_KEY`                                     | Print error with manual conversion instructions, exit non-zero                               |
| `--data` path does not exist                               | `argparse` error before any other processing                                                 |
| `--output` specified but does not end with `.npz`          | Argument error before any other processing                                                   |
| `--data` is a directory and file count > `--max-files`     | Error before API call; instruct user to raise limit with `--max-files` and `--confirm-large` |
| User aborts interactive directory confirmation             | Print `Aborted.`, exit zero                                                                  |
| API call fails (network, auth, rate limit)                 | Print error with status code and message, exit non-zero                                      |
| No code block in API response                              | Print full response for debugging, exit non-zero                                             |
| Converter validation fails on first attempt                | Retry once automatically with error context                                                  |
| Converter validation fails on retry                        | Save failing script with `_FAILED` suffix, print final error, exit non-zero                  |
| Output NPZ path already exists                             | Overwrite with a printed warning                                                             |
| Converter script name collision in `setup_wizard_scripts/` | Add numeric suffix; never overwrite existing scripts                                         |
| `--max-files` raised above 50 without `--confirm-large`    | Interactive confirmation prompt regardless of `--max-files` value                            |


---

## Style Guide Compliance

- No code-side defaults anywhere; all defaults are in the CLI argument parser and documented in this spec
- All required behavior-affecting parameters are explicit
- Error messages are specific and actionable
- The wizard does not modify any existing pipeline files, configs, or source files
- The wizard does not write to any directory other than `setup_wizard_scripts/` and the output NPZ location
- The API key never touches disk and is never logged
- The wizard does not suggest hyperparameters, config values, or numerical guidance of any kind

