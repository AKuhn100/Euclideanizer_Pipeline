#!/usr/bin/env python3
"""
Converter script for GROMACS .gro files to NPZ format.

Usage:
    python script.py <input_path> <output_path>

where <input_path> is either a single .gro file or a directory containing .gro files,
and <output_path> is the path where the output .npz file will be written.

The output NPZ file will contain a single key 'coords' with shape
(n_structures, n_atoms, 3) and dtype float32.
"""

import sys
import os
import numpy as np


def parse_gro_file(filepath):
    """
    Parse a GROMACS .gro file and return a list of coordinate arrays.

    Each frame in the .gro file yields one array of shape (n_atoms, 3).

    GRO format per atom line (fixed-width spec):
        cols  0- 4: residue number
        cols  5- 9: residue name
        cols 10-14: atom name
        cols 15-19: atom number
        cols 20-27: x (nm)
        cols 28-35: y (nm)
        cols 36-43: z (nm)
        (optional velocity columns follow)

    However, to be robust to variable spacing, we split on whitespace and
    take the last three tokens of numeric fields as x, y, z.
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

        # Line 1: title / comment line
        title_line = lines[i].strip()
        i += 1

        if i >= total_lines:
            break

        # Line 2: number of atoms
        natoms_line = lines[i].strip()
        i += 1

        try:
            n_atoms = int(natoms_line)
        except ValueError:
            raise ValueError(
                f"Expected integer atom count on line {i} of '{filepath}', "
                f"but got: '{natoms_line}'"
            )

        if n_atoms < 2:
            raise ValueError(
                f"Expected at least 2 atoms per structure, but got {n_atoms} "
                f"in file '{filepath}' (frame starting near line {i})."
            )

        if i + n_atoms > total_lines:
            raise ValueError(
                f"File '{filepath}' is truncated: expected {n_atoms} atom lines "
                f"but only {total_lines - i} lines remain."
            )

        coords = np.zeros((n_atoms, 3), dtype=np.float64)

        for atom_idx in range(n_atoms):
            line = lines[i]
            i += 1

            # Split the line on whitespace and take the last three numeric tokens
            # as x, y, z. This handles variable spacing robustly.
            #
            # GRO atom line format (cols are 0-based):
            #   resnum(5) resname(5) atomname(5) atomnum(5) x(8) y(8) z(8) [vx vy vz]
            #
            # We use whitespace splitting but need to be careful: the first token
            # might combine resnum+resname (no space between them when resnum >= 10).
            # The safe approach: tokens from whitespace split, and spatial coords
            # are the 3 tokens starting at index 3 (0-indexed: resnum, resname,
            # atomname, atomnum, x, y, z, ...) -- but resnum+resname may merge.
            #
            # Most robust: take the LAST 3 tokens (ignoring optional velocities
            # would be wrong if velocities present). Actually take tokens at
            # positions [3], [4], [5] counting from the split -- but that breaks
            # when resnum and resname merge.
            #
            # Best robust approach: use fixed-width parsing for the coord columns
            # (cols 20-27, 28-35, 36-43) as per GRO spec, with fallback to
            # whitespace splitting of the last 3 or specific tokens.

            # Try fixed-width first (GRO spec: x at cols 20-28, y 28-36, z 36-44)
            parsed_ok = False
            if len(line) >= 44:
                try:
                    x = float(line[20:28])
                    y = float(line[28:36])
                    z = float(line[36:44])
                    parsed_ok = True
                except ValueError:
                    parsed_ok = False

            if not parsed_ok:
                # Fallback: whitespace split, take last 3 tokens
                # (works when no velocity columns; if velocities present,
                # we need tokens [-6:-3] but let's detect)
                tokens = line.split()
                # Try to find 3 consecutive floats starting from index 3
                # (after resnum/resname/atomname/atomnum which is 4 tokens
                # when no merging, or 3 when merging)
                found = False
                for start in range(len(tokens) - 2):
                    try:
                        x = float(tokens[start])
                        y = float(tokens[start + 1])
                        z = float(tokens[start + 2])
                        # Check that prior token is an integer (atom number)
                        # to confirm we have the right position
                        if start > 0:
                            try:
                                int(tokens[start - 1])
                                found = True
                                break
                            except ValueError:
                                pass
                    except ValueError:
                        continue

                if not found:
                    # Last resort: just take last 3 numeric tokens
                    # (works if no velocities)
                    try:
                        x = float(tokens[-3])
                        y = float(tokens[-2])
                        z = float(tokens[-1])
                    except (ValueError, IndexError):
                        raise ValueError(
                            f"Cannot parse coordinates from line {i} "
                            f"in file '{filepath}': '{line.rstrip()}'"
                        )

            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                raise ValueError(
                    f"Non-finite coordinate value at line {i} in '{filepath}': "
                    f"x={x}, y={y}, z={z}"
                )

            coords[atom_idx, 0] = x
            coords[atom_idx, 1] = y
            coords[atom_idx, 2] = z

        structures.append(coords)

        # After atom lines, there may be a box vector line (skip it)
        if i < total_lines and lines[i].strip() != '':
            # Peek: if this line has 3 or 9 floats, it's a box line
            tokens = lines[i].split()
            is_box = False
            if len(tokens) in (3, 9):
                try:
                    [float(t) for t in tokens]
                    is_box = True
                except ValueError:
                    pass
            if is_box:
                i += 1  # skip box line
            # else: it might be the start of the next frame (title line)

    return structures


def collect_gro_files(input_path):
    """
    Return a sorted list of .gro file paths from input_path
    (which may be a single file or a directory).
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = []
        for fname in sorted(os.listdir(input_path)):
            if fname.lower().endswith('.gro'):
                files.append(os.path.join(input_path, fname))
        if not files:
            raise ValueError(
                f"No .gro files found in directory '{input_path}'."
            )
        return files
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )


def validate_coords(coords_array):
    """Validate the final coords array against required criteria."""
    if coords_array.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got ndim={coords_array.ndim}."
        )
    if coords_array.shape[2] != 3:
        raise ValueError(
            f"Expected last dimension == 3 (x,y,z), got {coords_array.shape[2]}."
        )
    if coords_array.shape[0] < 1:
        raise ValueError(
            f"Expected at least 1 structure, got {coords_array.shape[0]}."
        )
    if coords_array.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 atoms, got {coords_array.shape[1]}."
        )
    if not np.all(np.isfinite(coords_array)):
        raise ValueError(
            "Coordinate array contains non-finite values (NaN or inf)."
        )
    # Check first column is not atom indices (0, 1, 2, ..., n_atoms-1)
    n_atoms = coords_array.shape[1]
    atom_indices = np.arange(n_atoms, dtype=np.float64)
    for struct_idx in range(min(coords_array.shape[0], 5)):
        first_col = coords_array[struct_idx, :, 0].astype(np.float64)
        # Correlation with atom indices
        if np.std(first_col) > 1e-10 and np.std(atom_indices) > 1e-10:
            corr = np.corrcoef(first_col, atom_indices)[0, 1]
            if abs(corr) > 0.99:
                raise ValueError(
                    f"First column of structure {struct_idx} appears to be atom "
                    f"indices (correlation={corr:.4f} with 0..n_atoms-1). "
                    f"Coordinate parsing may be wrong."
                )


def main():
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} <input_path> <output_path>",
            file=sys.stderr
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Collect files
    gro_files = collect_gro_files(input_path)

    # Parse all structures
    all_structures = []
    for fpath in gro_files:
        structures = parse_gro_file(fpath)
        if not structures:
            raise ValueError(
                f"No structures found in file '{fpath}'."
            )
        all_structures.extend(structures)

    if not all_structures:
        raise ValueError(
            f"No structures found in input path '{input_path}'."
        )

    # Check all structures have the same number of atoms
    n_atoms_list = [s.shape[0] for s in all_structures]
    if len(set(n_atoms_list)) > 1:
        raise ValueError(
            f"Inconsistent atom counts across structures: "
            f"{sorted(set(n_atoms_list))}. All structures must have "
            f"the same number of atoms."
        )

    # Stack into single array
    coords_array = np.array(all_structures, dtype=np.float32)

    # Validate
    validate_coords(coords_array)

    # Write output
    np.savez_compressed(output_path, coords=coords_array)

    n_structures = coords_array.shape[0]
    n_atoms = coords_array.shape[1]
    print(
        f"Successfully wrote {n_structures} structure(s) with {n_atoms} atom(s) "
        f"each to '{output_path}'."
    )
    print(f"Array shape: {coords_array.shape}, dtype: {coords_array.dtype}")


if __name__ == '__main__':
    main()