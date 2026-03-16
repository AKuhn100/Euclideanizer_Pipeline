#!/usr/bin/env python3
"""
Converter for GROMACS .gro trajectory/coordinate files to NPZ format.

Usage:
    python script.py <input_path> <output_path>

Where input_path is either a single .gro file or a directory containing .gro files.
Output is a compressed NPZ file with key 'coords' and shape (n_structures, n_atoms, 3).
"""

import sys
import os
import numpy as np


def parse_gro_file(filepath):
    """
    Parse a GROMACS .gro file and return a list of coordinate arrays.

    A .gro file can contain multiple frames (structures). Each frame has:
      - Line 0: title/comment line
      - Line 1: number of atoms (integer)
      - Lines 2 to 2+n_atoms-1: atom records
      - Last line of frame: box vectors (3 or 9 floats)

    Atom record format (fixed-width):
      Columns 0-4:   residue number (5 chars)
      Columns 5-9:   residue name (5 chars)
      Columns 10-14: atom name (5 chars)
      Columns 15-19: atom number (5 chars)
      Columns 20-28: x coordinate (8.3f)
      Columns 29-37: y coordinate (8.3f)
      Columns 38-46: z coordinate (8.3f)
      (optional velocity columns follow)

    Returns:
        List of numpy arrays, each of shape (n_atoms, 3), dtype float64.
    """
    structures = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise ValueError(
            f"File '{filepath}' is empty. Expected a GROMACS .gro file with "
            "at least a title line, atom count line, atom records, and box line."
        )

    idx = 0
    frame_index = 0

    while idx < len(lines):
        # Skip blank lines between frames
        if lines[idx].strip() == '':
            idx += 1
            continue

        # Line 0 of frame: title (any string)
        title_line = lines[idx].rstrip('\n')
        idx += 1

        if idx >= len(lines):
            raise ValueError(
                f"File '{filepath}', frame {frame_index}: Expected atom count line after "
                f"title '{title_line}', but reached end of file."
            )

        # Line 1 of frame: number of atoms
        atom_count_line = lines[idx].strip()
        idx += 1

        try:
            n_atoms = int(atom_count_line)
        except ValueError:
            raise ValueError(
                f"File '{filepath}', frame {frame_index}: Expected an integer atom count "
                f"on the line after the title, but found: '{atom_count_line}'"
            )

        if n_atoms < 2:
            raise ValueError(
                f"File '{filepath}', frame {frame_index}: Atom count must be >= 2, "
                f"but found {n_atoms}."
            )

        # Read n_atoms atom lines
        coords = np.zeros((n_atoms, 3), dtype=np.float64)

        for atom_idx in range(n_atoms):
            if idx >= len(lines):
                raise ValueError(
                    f"File '{filepath}', frame {frame_index}: Expected {n_atoms} atom "
                    f"records but file ended after {atom_idx} atoms."
                )

            line = lines[idx]
            idx += 1

            # GRO format is fixed-width. The coordinate columns start at position 20.
            # Each coordinate field is 8 characters wide.
            # Format: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
            # residue number: cols 0-4 (5 chars)
            # residue name:   cols 5-9 (5 chars)
            # atom name:      cols 10-14 (5 chars)
            # atom number:    cols 15-19 (5 chars)
            # x:              cols 20-27 (8 chars)
            # y:              cols 28-35 (8 chars)
            # z:              cols 36-43 (8 chars)

            if len(line.rstrip('\n')) < 44:
                # Try whitespace splitting as fallback
                parts = line.split()
                if len(parts) < 6:
                    raise ValueError(
                        f"File '{filepath}', frame {frame_index}, atom {atom_idx+1}: "
                        f"Line is too short for GRO format and cannot be parsed by "
                        f"whitespace splitting. Line: '{line.rstrip()}'"
                    )
                try:
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                except ValueError:
                    raise ValueError(
                        f"File '{filepath}', frame {frame_index}, atom {atom_idx+1}: "
                        f"Could not parse coordinates from parts {parts[3:6]}. "
                        f"Line: '{line.rstrip()}'"
                    )
            else:
                # Parse using fixed-width columns
                try:
                    x = float(line[20:28])
                    y = float(line[28:36])
                    z = float(line[36:44])
                except ValueError:
                    # Fallback: try whitespace splitting
                    parts = line.split()
                    if len(parts) < 6:
                        raise ValueError(
                            f"File '{filepath}', frame {frame_index}, atom {atom_idx+1}: "
                            f"Fixed-width parsing failed and line has fewer than 6 "
                            f"whitespace-delimited fields. Line: '{line.rstrip()}'"
                        )
                    try:
                        x = float(parts[3])
                        y = float(parts[4])
                        z = float(parts[5])
                    except ValueError:
                        raise ValueError(
                            f"File '{filepath}', frame {frame_index}, atom {atom_idx+1}: "
                            f"Could not parse coordinates from '{line.rstrip()}'. "
                            f"Expected floats at positions [20:28], [28:36], [36:44] "
                            f"or as fields 3, 4, 5 (0-indexed) in whitespace-split."
                        )

            coords[atom_idx, 0] = x
            coords[atom_idx, 1] = y
            coords[atom_idx, 2] = z

        # Read box vectors line (mandatory in .gro format)
        if idx >= len(lines):
            raise ValueError(
                f"File '{filepath}', frame {frame_index}: Expected box vector line "
                f"after {n_atoms} atom records, but reached end of file."
            )

        box_line = lines[idx].strip()
        idx += 1

        # Validate box line has at least 3 floats
        box_parts = box_line.split()
        if len(box_parts) < 3:
            raise ValueError(
                f"File '{filepath}', frame {frame_index}: Expected box vector line with "
                f"at least 3 floats after atom records, but found: '{box_line}'"
            )
        try:
            [float(b) for b in box_parts[:3]]
        except ValueError:
            raise ValueError(
                f"File '{filepath}', frame {frame_index}: Box vector line does not "
                f"contain valid floats: '{box_line}'"
            )

        # Validate all coordinates are finite
        if not np.all(np.isfinite(coords)):
            bad_count = np.sum(~np.isfinite(coords))
            raise ValueError(
                f"File '{filepath}', frame {frame_index}: Found {bad_count} non-finite "
                f"coordinate value(s) (NaN or inf). All coordinates must be finite."
            )

        structures.append(coords)
        frame_index += 1

    if len(structures) == 0:
        raise ValueError(
            f"File '{filepath}': No valid structures (frames) were parsed. "
            "The file may be empty or malformed."
        )

    return structures


def collect_gro_files(input_path):
    """
    Collect all .gro files from input_path (file or directory).

    Returns:
        Sorted list of file paths.
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = []
        for entry in os.listdir(input_path):
            if entry.lower().endswith('.gro'):
                files.append(os.path.join(input_path, entry))
        if len(files) == 0:
            raise ValueError(
                f"Directory '{input_path}' contains no .gro files. "
                "Please provide a directory with at least one GROMACS .gro file."
            )
        return sorted(files)
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory. "
            "Please provide a valid path to a .gro file or a directory of .gro files."
        )


def validate_coords(array, output_path):
    """
    Validate the final coords array against the required criteria.
    Raises ValueError with a descriptive message if any check fails.
    """
    if array.ndim != 3:
        raise ValueError(
            f"Coords array has {array.ndim} dimensions, expected 3 (n_structures, n_atoms, 3)."
        )
    if array.shape[2] != 3:
        raise ValueError(
            f"Coords array shape is {array.shape}; the last dimension must be 3 (x, y, z), "
            f"but found {array.shape[2]}."
        )
    if array.shape[0] < 1:
        raise ValueError(
            f"Coords array has {array.shape[0]} structures; at least 1 is required."
        )
    if array.shape[1] < 2:
        raise ValueError(
            f"Coords array has {array.shape[1]} atoms per structure; at least 2 are required."
        )
    if not np.all(np.isfinite(array)):
        bad = np.sum(~np.isfinite(array))
        raise ValueError(
            f"Coords array contains {bad} non-finite value(s) (NaN or inf). "
            "All coordinate values must be finite."
        )
    # Check castable to float32
    try:
        array.astype(np.float32)
    except Exception as e:
        raise ValueError(
            f"Coords array dtype '{array.dtype}' cannot be cast to float32: {e}"
        )


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python script.py <input_path> <output_path>\n"
            "  input_path:  path to a single .gro file or a directory of .gro files\n"
            "  output_path: path where the output .npz file will be written"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Collect files
    gro_files = collect_gro_files(input_path)

    all_structures = []
    n_atoms_expected = None

    for filepath in gro_files:
        structures = parse_gro_file(filepath)

        for frame_idx, frame_coords in enumerate(structures):
            n_atoms_this = frame_coords.shape[0]

            if n_atoms_expected is None:
                n_atoms_expected = n_atoms_this
            elif n_atoms_this != n_atoms_expected:
                raise ValueError(
                    f"Atom count mismatch: the first structure has {n_atoms_expected} atoms, "
                    f"but frame {frame_idx} in '{filepath}' has {n_atoms_this} atoms. "
                    "All structures must have the same number of atoms to form a "
                    "(n_structures, n_atoms, 3) array."
                )

            all_structures.append(frame_coords)

    if len(all_structures) == 0:
        raise ValueError(
            "No structures were parsed from the input. "
            f"Input path: '{input_path}'"
        )

    # Stack into a single array of shape (n_structures, n_atoms, 3)
    coords_array = np.stack(all_structures, axis=0).astype(np.float64)

    # Final validation
    validate_coords(coords_array, output_path)

    # Write output
    np.savez_compressed(output_path, coords=coords_array)

    n_structures = coords_array.shape[0]
    n_atoms = coords_array.shape[1]
    print(
        f"Successfully wrote NPZ file.\n"
        f"  Structures : {n_structures}\n"
        f"  Atoms      : {n_atoms}\n"
        f"  Array shape: {coords_array.shape}\n"
        f"  Dtype      : {coords_array.dtype}\n"
        f"  Output     : {output_path}"
    )


if __name__ == '__main__':
    main()