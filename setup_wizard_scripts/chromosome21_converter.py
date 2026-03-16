#!/usr/bin/env python3
"""
Converter script for GROMACS .gro files to NPZ format.

Usage:
    python script.py <input_path> <output_path>

Where input_path is either a single .gro file or a directory containing .gro files.
Output is a single .npz file with key 'coords' and shape (n_structures, n_atoms, 3).
"""

import sys
import os
import numpy as np


def parse_gro_file(filepath):
    """
    Parse a GROMACS .gro file and return a list of coordinate arrays.

    GRO format per frame:
        Line 1: title
        Line 2: number of atoms (integer)
        Lines 3 to 3+n_atoms-1: atom records
        Last line: box vectors

    Atom record format (fixed-width, but we use whitespace splitting for safety):
        residue number + residue name (cols 1-10)
        atom name (cols 10-15)
        atom number (cols 15-20)
        x (cols 20-28), y (cols 28-36), z (cols 36-44)  [in nm]

    We split on whitespace and take the last 3 fields as x, y, z to avoid
    mis-parsing due to variable-width residue/atom numbering.
    """
    structures = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    total_lines = len(lines)

    while i < total_lines:
        # Skip blank lines between frames
        if lines[i].strip() == '':
            i += 1
            continue

        # Line 1: title (skip)
        title_line = lines[i].strip()
        i += 1

        if i >= total_lines:
            break

        # Line 2: number of atoms
        atom_count_line = lines[i].strip()
        i += 1

        try:
            n_atoms = int(atom_count_line)
        except ValueError:
            raise ValueError(
                f"Expected integer atom count, got: '{atom_count_line}' "
                f"at line {i} in file '{filepath}'"
            )

        if n_atoms < 2:
            raise ValueError(
                f"Expected at least 2 atoms, but found n_atoms={n_atoms} "
                f"in frame starting near line {i} of '{filepath}'"
            )

        if i + n_atoms >= total_lines:
            raise ValueError(
                f"File '{filepath}' appears truncated: expected {n_atoms} atom lines "
                f"starting at line {i}, but only {total_lines - i} lines remain."
            )

        # Read n_atoms atom lines
        coords = np.zeros((n_atoms, 3), dtype=np.float64)

        for atom_idx in range(n_atoms):
            line = lines[i]
            i += 1

            # Split on whitespace; the last 3 tokens should be x, y, z
            # GRO format: resNUM+resNAME atomNAME atomNUM x y z [vx vy vz]
            # We robustly take the last 3 (or last 6 if velocities present)
            # fields as x, y, z by looking at all numeric fields after position.
            parts = line.split()

            if len(parts) < 6:
                raise ValueError(
                    f"Atom line has fewer than 6 fields at line {i} in '{filepath}':\n"
                    f"  '{line.rstrip()}'\n"
                    f"  Expected format: resNAME atomNAME atomNUM x y z [vx vy vz]"
                )

            # In GRO format, the first field may be resnum+resname concatenated (e.g. '1LYS'),
            # so fields are: [resname_merged, atomname, atomnum, x, y, z, (vx, vy, vz)]
            # x, y, z are at positions 3, 4, 5 (0-indexed) regardless of velocities.
            # However, with concatenated resnum+resname the split gives:
            #   parts[0] = '1LYS', parts[1] = 'CA', parts[2] = '1',
            #   parts[3] = x, parts[4] = y, parts[5] = z
            # We take parts[3], parts[4], parts[5] when len(parts) in {6, 9}.
            # As a fallback, we take the last 3 numeric values that look like floats.

            try:
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except (ValueError, IndexError):
                # Fallback: take last 3 fields that are floats
                float_parts = []
                for p in reversed(parts):
                    try:
                        float_parts.append(float(p))
                        if len(float_parts) == 3:
                            break
                    except ValueError:
                        pass
                if len(float_parts) < 3:
                    raise ValueError(
                        f"Cannot parse x, y, z coordinates from line {i} in '{filepath}':\n"
                        f"  '{line.rstrip()}'"
                    )
                # float_parts was collected in reverse order
                z, y, x = float_parts[0], float_parts[1], float_parts[2]

            # Validate finiteness
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                raise ValueError(
                    f"Non-finite coordinate(s) at atom {atom_idx+1} in frame "
                    f"near line {i} of '{filepath}': x={x}, y={y}, z={z}"
                )

            coords[atom_idx, 0] = x
            coords[atom_idx, 1] = y
            coords[atom_idx, 2] = z

        # Line after atoms: box vectors (skip)
        if i < total_lines:
            box_line = lines[i].strip()
            i += 1
            # box_line is the box vector line; we skip it

        # Validate that x column is not atom indices 0..n_atoms-1
        x_col = coords[:, 0]
        atom_indices = np.arange(n_atoms, dtype=np.float64)
        # Check correlation with 0-based indices
        if n_atoms > 1:
            corr = np.corrcoef(x_col, atom_indices)[0, 1]
            if abs(corr) > 0.99:
                raise ValueError(
                    f"The first coordinate column appears to be atom indices "
                    f"(correlation={corr:.4f} with 0..{n_atoms-1}). "
                    f"This suggests a parsing error in '{filepath}'."
                )

        structures.append(coords)

    if len(structures) == 0:
        raise ValueError(f"No structures found in file '{filepath}'.")

    return structures


def collect_gro_files(input_path):
    """
    Collect all .gro files from the given path.
    If input_path is a file, return [input_path].
    If input_path is a directory, return all .gro files within it (sorted).
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        gro_files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith('.gro')
        ])
        if len(gro_files) == 0:
            raise ValueError(
                f"No .gro files found in directory '{input_path}'. "
                f"Expected one or more files with '.gro' extension."
            )
        return gro_files
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Collect files to process
    gro_files = collect_gro_files(input_path)

    all_structures = []
    n_atoms_expected = None

    for filepath in gro_files:
        structures = parse_gro_file(filepath)

        for frame_idx, coords in enumerate(structures):
            n_atoms = coords.shape[0]

            if n_atoms_expected is None:
                n_atoms_expected = n_atoms
            elif n_atoms != n_atoms_expected:
                raise ValueError(
                    f"Inconsistent atom count: expected {n_atoms_expected} atoms "
                    f"(from previous frames), but frame {frame_idx} in '{filepath}' "
                    f"has {n_atoms} atoms. All frames must have the same number of atoms."
                )

            all_structures.append(coords)

    if len(all_structures) == 0:
        raise ValueError("No structures were parsed from the input.")

    # Stack into (n_structures, n_atoms, 3)
    coords_array = np.array(all_structures, dtype=np.float64)

    # Final validation
    if coords_array.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got shape {coords_array.shape}"
        )
    if coords_array.shape[2] != 3:
        raise ValueError(
            f"Expected last dimension to be 3 (x, y, z), got {coords_array.shape[2]}"
        )
    if coords_array.shape[0] < 1:
        raise ValueError(
            f"Expected at least 1 structure, got {coords_array.shape[0]}"
        )
    if coords_array.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 atoms, got {coords_array.shape[1]}"
        )
    if not np.all(np.isfinite(coords_array)):
        raise ValueError(
            "Coordinate array contains non-finite values (NaN or inf)."
        )

    # Write output
    np.savez_compressed(output_path, coords=coords_array)

    n_structures = coords_array.shape[0]
    n_atoms = coords_array.shape[1]
    print(f"Successfully wrote {n_structures} structure(s) with {n_atoms} atom(s) each.")
    print(f"Output shape: {coords_array.shape}, dtype: {coords_array.dtype}")
    print(f"Output file: {output_path}")


if __name__ == '__main__':
    main()