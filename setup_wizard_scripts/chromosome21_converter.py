#!/usr/bin/env python3
"""
Converter script for GROMACS .gro format coordinate files.

Reads one or more .gro files and writes a single NPZ file with key 'coords'
and shape (n_structures, n_atoms, 3).

Usage:
    python script.py <input_path> <output_path>

Where input_path is either a single .gro file or a directory containing .gro files.
"""

import sys
import os
import numpy as np


def parse_gro_file(filepath):
    """
    Parse a GROMACS .gro file and return a list of coordinate arrays.

    Each frame in the file is parsed and returned as an ndarray of shape (n_atoms, 3).

    GRO format per frame:
        Line 0:   Title/comment line
        Line 1:   Number of atoms (integer)
        Lines 2..(n_atoms+1): Atom records
        Last line: Box vectors (3 or 9 floats)

    Atom record format (fixed-width fields):
        cols  0- 4  : residue number   (5 chars)
        cols  5- 9  : residue name     (5 chars)
        cols 10-14  : atom name        (5 chars)
        cols 15-19  : atom number      (5 chars)
        cols 20-27  : x coordinate     (8.3f, nm)
        cols 28-35  : y coordinate     (8.3f, nm)
        cols 36-43  : z coordinate     (8.3f, nm)
        (optional velocities follow)

    Returns:
        List of numpy arrays, each of shape (n_atoms, 3), dtype float64.
    """
    structures = []

    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    if len(lines) == 0:
        raise ValueError(
            f"File '{filepath}' is empty. Expected a GROMACS .gro file."
        )

    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        # Skip blank lines between frames
        if lines[idx].strip() == '':
            idx += 1
            continue

        # Line 0 of frame: title (skip)
        title_line = lines[idx].rstrip('\n')
        idx += 1

        if idx >= total_lines:
            # Trailing title line with no frame body — stop
            break

        # Line 1 of frame: number of atoms
        natoms_line = lines[idx].strip()
        idx += 1

        try:
            n_atoms = int(natoms_line)
        except ValueError:
            raise ValueError(
                f"Expected an integer (atom count) on line {idx} of '{filepath}', "
                f"but found: '{natoms_line}'. "
                "This does not look like a valid GROMACS .gro file."
            )

        if n_atoms < 2:
            raise ValueError(
                f"File '{filepath}': frame starting near line {idx - 2} reports "
                f"{n_atoms} atom(s), but at least 2 atoms are required."
            )

        if idx + n_atoms >= total_lines:
            raise ValueError(
                f"File '{filepath}': expected {n_atoms} atom lines plus a box line "
                f"after line {idx - 1}, but the file ends prematurely (only "
                f"{total_lines - idx} lines remain)."
            )

        coords = np.empty((n_atoms, 3), dtype=np.float64)

        for atom_i in range(n_atoms):
            line = lines[idx]
            idx += 1

            # GRO fixed-width format:
            #   [0:5]  resid
            #   [5:10] resname
            #   [10:15] atomname
            #   [15:20] atomnr
            #   [20:28] x
            #   [28:36] y
            #   [36:44] z
            # Some writers use slightly different widths; we try fixed-width first,
            # then fall back to whitespace splitting.
            try:
                if len(line.rstrip('\n')) >= 44:
                    x = float(line[20:28])
                    y = float(line[28:36])
                    z = float(line[36:44])
                else:
                    # Shorter line: fall back to splitting on whitespace
                    # Fields: resid+resname (merged), atomname, atomnr, x, y, z
                    parts = line.split()
                    if len(parts) < 6:
                        raise ValueError(f"Too few fields: {parts}")
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"File '{filepath}', frame atom line {idx} (0-based): "
                    f"could not parse coordinates.\n"
                    f"  Line content : '{line.rstrip()}'\n"
                    f"  Expected fixed-width GRO format with x at cols 20-28, "
                    f"y at 28-36, z at 36-44 (or whitespace-separated fallback).\n"
                    f"  Original error: {exc}"
                )

            coords[atom_i, 0] = x
            coords[atom_i, 1] = y
            coords[atom_i, 2] = z

        # Box vectors line (skip, but consume)
        if idx < total_lines and lines[idx].strip() != '':
            idx += 1  # consume box line
        elif idx < total_lines:
            idx += 1  # blank box line edge-case

        # Validate finiteness for this frame
        if not np.all(np.isfinite(coords)):
            bad = np.argwhere(~np.isfinite(coords))
            raise ValueError(
                f"File '{filepath}': non-finite values (NaN or inf) found in frame "
                f"{len(structures) + 1} at atom indices: {bad[:, 0].tolist()}"
            )

        structures.append(coords)

    if len(structures) == 0:
        raise ValueError(
            f"No valid frames were parsed from '{filepath}'. "
            "The file may be empty, malformed, or not in GROMACS .gro format."
        )

    return structures


def collect_files(input_path):
    """
    Given a path (file or directory), return a sorted list of .gro file paths.
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        entries = sorted(os.listdir(input_path))
        gro_files = [
            os.path.join(input_path, e)
            for e in entries
            if e.lower().endswith('.gro') and os.path.isfile(os.path.join(input_path, e))
        ]
        if len(gro_files) == 0:
            raise ValueError(
                f"Directory '{input_path}' contains no .gro files. "
                "Please provide a directory with one or more GROMACS .gro files, "
                "or pass a single .gro file path directly."
            )
        return gro_files
    else:
        raise ValueError(
            f"Input path '{input_path}' does not exist or is neither a file nor a "
            "directory. Please provide a valid path to a .gro file or a directory "
            "containing .gro files."
        )


def validate_array(coords_array, output_path):
    """
    Validate the final coords array against the required schema.
    Raises ValueError with a descriptive message if any check fails.
    """
    if coords_array.ndim != 3:
        raise ValueError(
            f"Output array has {coords_array.ndim} dimension(s), but exactly 3 are "
            f"required (n_structures, n_atoms, 3). Shape was: {coords_array.shape}"
        )
    if coords_array.shape[2] != 3:
        raise ValueError(
            f"Output array last dimension is {coords_array.shape[2]}, but must be 3 "
            f"(x, y, z). Shape was: {coords_array.shape}"
        )
    if coords_array.shape[0] < 1:
        raise ValueError(
            f"Output array has 0 structures (shape[0] == 0). "
            "At least 1 structure is required."
        )
    if coords_array.shape[1] < 2:
        raise ValueError(
            f"Output array has {coords_array.shape[1]} atom(s) (shape[1]). "
            "At least 2 atoms are required."
        )
    if not np.all(np.isfinite(coords_array)):
        n_bad = np.sum(~np.isfinite(coords_array))
        raise ValueError(
            f"Output array contains {n_bad} non-finite value(s) (NaN or inf). "
            "All coordinate values must be finite."
        )
    # Check castability to float32
    try:
        coords_array.astype(np.float32)
    except Exception as exc:
        raise ValueError(
            f"Output array dtype '{coords_array.dtype}' cannot be cast to float32: {exc}"
        )


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python script.py <input_path> <output_path>\n"
            "  input_path  : path to a single .gro file, or a directory of .gro files\n"
            "  output_path : path where the output .npz file will be written"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Collect file(s) to process
    gro_files = collect_files(input_path)

    all_structures = []
    n_atoms_reference = None

    for filepath in gro_files:
        print(f"  Parsing: {filepath}")
        frames = parse_gro_file(filepath)

        for frame_idx, frame_coords in enumerate(frames):
            n_atoms_this = frame_coords.shape[0]
            if n_atoms_reference is None:
                n_atoms_reference = n_atoms_this
            elif n_atoms_this != n_atoms_reference:
                raise ValueError(
                    f"Inconsistent atom count across frames/files.\n"
                    f"  First frame had {n_atoms_reference} atoms.\n"
                    f"  Frame {frame_idx + 1} in '{filepath}' has {n_atoms_this} atoms.\n"
                    "All frames must contain the same number of atoms."
                )
            all_structures.append(frame_coords)

    if len(all_structures) == 0:
        raise ValueError(
            "No structures were collected from the provided input. "
            f"Input path was: '{input_path}'"
        )

    # Stack into (n_structures, n_atoms, 3)
    coords_array = np.stack(all_structures, axis=0).astype(np.float64)

    # Final validation
    validate_array(coords_array, output_path)

    # Write output
    np.savez_compressed(output_path, coords=coords_array)

    print(
        f"\nConversion complete.\n"
        f"  Structures : {coords_array.shape[0]}\n"
        f"  Atoms      : {coords_array.shape[1]}\n"
        f"  Output     : {output_path}"
    )


if __name__ == '__main__':
    main()