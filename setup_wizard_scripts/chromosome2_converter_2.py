#!/usr/bin/env python3
"""
Converter script for GROMACS .gro files to NPZ format.
Reads coordinate data and writes a single NPZ file with key 'coords'
and shape (n_structures, n_atoms, 3).

Usage:
    python solution.py <input_path> <output_path>

Where input_path is either a single .gro file or a directory containing .gro files.
"""

import sys
import os
import numpy as np


def parse_gro_file(filepath):
    """
    Parse a GROMACS .gro file and return a list of coordinate arrays.
    
    Each frame in the file yields an array of shape (n_atoms, 3) containing
    the x, y, z coordinates (in nm).
    
    GRO format per atom line (whitespace-split approach):
      Fields: residue_number+resname, atom_name, atom_number, x, y, z [vx, vy, vz]
    We take the last 3 numeric fields (or fields at positions -3, -2, -1 of the split),
    but specifically we want x, y, z which are columns 3, 4, 5 (0-indexed) in the split.
    
    We use whitespace splitting and take fields at indices 3, 4, 5 as x, y, z.
    """
    structures = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    total_lines = len(lines)
    
    while i < total_lines:
        # Skip blank lines
        if lines[i].strip() == '':
            i += 1
            continue
        
        # Line 1: title/comment line
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
                f"Expected integer number of atoms, got: '{natoms_line}' "
                f"in file '{filepath}' after title line '{title_line}'"
            )
        
        if n_atoms < 2:
            raise ValueError(
                f"Number of atoms must be >= 2, got {n_atoms} "
                f"in file '{filepath}'"
            )
        
        # Read n_atoms coordinate lines
        if i + n_atoms > total_lines:
            raise ValueError(
                f"Expected {n_atoms} atom lines but only {total_lines - i} lines "
                f"remaining in file '{filepath}'"
            )
        
        coords = np.zeros((n_atoms, 3), dtype=np.float64)
        
        for atom_idx in range(n_atoms):
            line = lines[i]
            i += 1
            
            # GRO atom line format (variable width):
            # "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" [optional velocities]
            # Fields when split by whitespace:
            #   0: residue+resname (e.g., "1LYS")
            #   1: atom name (e.g., "CA")
            #   2: atom number (e.g., "1")
            #   3: x coordinate
            #   4: y coordinate
            #   5: z coordinate
            #   [6, 7, 8: optional vx, vy, vz]
            #
            # Note: residue number and residue name are concatenated without space,
            # so splitting by whitespace keeps them as one token.
            
            parts = line.split()
            
            if len(parts) < 6:
                raise ValueError(
                    f"Atom line {atom_idx + 1} in frame starting at title '{title_line}' "
                    f"in file '{filepath}' has fewer than 6 whitespace-separated fields.\n"
                    f"Expected format: resnum+resname atomname atomnum x y z [vx vy vz]\n"
                    f"Got: '{line.rstrip()}'"
                )
            
            try:
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Could not parse x, y, z coordinates from fields [3], [4], [5] "
                    f"of atom line {atom_idx + 1} in frame '{title_line}' "
                    f"in file '{filepath}'.\n"
                    f"Line: '{line.rstrip()}'\n"
                    f"Parts: {parts}\n"
                    f"Error: {e}"
                )
            
            # Validate coordinates are finite
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                raise ValueError(
                    f"Non-finite coordinate found at atom {atom_idx + 1} "
                    f"in frame '{title_line}' in file '{filepath}': "
                    f"x={x}, y={y}, z={z}"
                )
            
            coords[atom_idx, 0] = x
            coords[atom_idx, 1] = y
            coords[atom_idx, 2] = z
        
        # Validate that first column does not look like atom indices (0,1,2,...)
        # Atom indices would be strongly correlated with arange(n_atoms)
        expected_indices = np.arange(n_atoms, dtype=np.float64)
        x_col = coords[:, 0]
        
        # Check correlation with atom indices
        if n_atoms > 1:
            # Compute Pearson correlation
            x_mean = np.mean(x_col)
            idx_mean = np.mean(expected_indices)
            x_std = np.std(x_col)
            idx_std = np.std(expected_indices)
            
            if x_std > 1e-10 and idx_std > 1e-10:
                corr = np.mean((x_col - x_mean) * (expected_indices - idx_mean)) / (x_std * idx_std)
                if corr > 0.99:
                    raise ValueError(
                        f"First coordinate column appears to be atom indices "
                        f"(correlation with 0..{n_atoms-1} is {corr:.4f} > 0.99). "
                        f"Check parsing logic in file '{filepath}'."
                    )
        
        # Skip the box vector line (last line of the frame)
        if i < total_lines and lines[i].strip() != '':
            # This should be the box vectors line
            box_line = lines[i].strip()
            # Verify it looks like numbers (box vectors), not a new frame title
            box_parts = box_line.split()
            all_numeric = all(
                _is_float(p) for p in box_parts
            )
            if all_numeric and len(box_parts) >= 3:
                i += 1  # consume box vectors line
            # else: might be the start of next frame or end of file; don't consume
        
        structures.append(coords)
    
    return structures


def _is_float(s):
    """Check if a string can be converted to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def collect_gro_files(input_path):
    """
    Given input_path (file or directory), return a sorted list of .gro file paths.
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = []
        for entry in sorted(os.listdir(input_path)):
            if entry.lower().endswith('.gro'):
                files.append(os.path.join(input_path, entry))
        if not files:
            raise ValueError(
                f"No .gro files found in directory '{input_path}'. "
                f"Expected files with .gro extension."
            )
        return files
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )


def validate_coords(coords_array, output_path):
    """
    Validate the final coords array against the required criteria.
    Raises ValueError if any criterion is not met.
    """
    if coords_array.ndim != 3:
        raise ValueError(
            f"Output array must have ndim=3, got ndim={coords_array.ndim}. "
            f"Shape: {coords_array.shape}"
        )
    
    if coords_array.shape[2] != 3:
        raise ValueError(
            f"Last dimension must be 3 (x, y, z), got {coords_array.shape[2]}. "
            f"Shape: {coords_array.shape}"
        )
    
    if coords_array.shape[0] < 1:
        raise ValueError(
            f"Must have at least 1 structure, got {coords_array.shape[0]}."
        )
    
    if coords_array.shape[1] < 2:
        raise ValueError(
            f"Must have at least 2 atoms per structure, got {coords_array.shape[1]}."
        )
    
    if not np.all(np.isfinite(coords_array)):
        n_bad = np.sum(~np.isfinite(coords_array))
        raise ValueError(
            f"Output array contains {n_bad} non-finite values (NaN or inf). "
            f"All coordinate values must be finite."
        )
    
    # Check dtype castability to float32
    try:
        coords_array.astype(np.float32)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Output array dtype {coords_array.dtype} cannot be cast to float32: {e}"
        )
    
    # Check first column is not atom indices
    n_structures, n_atoms, _ = coords_array.shape
    if n_atoms > 1:
        expected_indices = np.arange(n_atoms, dtype=np.float64)
        idx_mean = np.mean(expected_indices)
        idx_std = np.std(expected_indices)
        
        for struct_idx in range(min(n_structures, 5)):  # Check first 5 structures
            x_col = coords_array[struct_idx, :, 0].astype(np.float64)
            x_mean = np.mean(x_col)
            x_std = np.std(x_col)
            
            if x_std > 1e-10 and idx_std > 1e-10:
                corr = np.mean((x_col - x_mean) * (expected_indices - idx_mean)) / (x_std * idx_std)
                if corr > 0.99:
                    raise ValueError(
                        f"First coordinate column of structure {struct_idx} appears to be "
                        f"atom indices (correlation={corr:.4f} > 0.99). "
                        f"The x column must contain spatial coordinates, not indices."
                    )


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python solution.py <input_path> <output_path>\n"
            "  input_path  : path to a single .gro file or directory of .gro files\n"
            "  output_path : path for the output .npz file"
        )
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Collect input files
    gro_files = collect_gro_files(input_path)
    
    print(f"Found {len(gro_files)} .gro file(s) to process.")
    
    # Parse all files and collect structures
    all_structures = []
    n_atoms_ref = None
    
    for filepath in gro_files:
        print(f"  Parsing: {filepath}")
        structures = parse_gro_file(filepath)
        
        if not structures:
            print(f"    Warning: No structures found in '{filepath}', skipping.")
            continue
        
        for struct_idx, coords in enumerate(structures):
            if n_atoms_ref is None:
                n_atoms_ref = coords.shape[0]
            elif coords.shape[0] != n_atoms_ref:
                raise ValueError(
                    f"Inconsistent number of atoms: structure {struct_idx} in '{filepath}' "
                    f"has {coords.shape[0]} atoms, but previous structures had {n_atoms_ref} atoms. "
                    f"All structures must have the same number of atoms."
                )
        
        all_structures.extend(structures)
        print(f"    Found {len(structures)} structure(s) with {structures[0].shape[0]} atoms each.")
    
    if not all_structures:
        raise ValueError(
            f"No structures were parsed from input '{input_path}'. "
            f"Check that the .gro files are properly formatted."
        )
    
    # Stack into single array: (n_structures, n_atoms, 3)
    coords_array = np.array(all_structures, dtype=np.float64)
    
    # Validate
    validate_coords(coords_array, output_path)
    
    # Write output
    np.savez_compressed(output_path, coords=coords_array)
    
    n_structures, n_atoms, _ = coords_array.shape
    print(f"\nSuccessfully wrote NPZ file:")
    print(f"  Output path : {output_path}")
    print(f"  Structures  : {n_structures}")
    print(f"  Atoms       : {n_atoms}")
    print(f"  Array shape : {coords_array.shape}")
    print(f"  Dtype       : {coords_array.dtype}")
    print(f"  x range     : [{coords_array[:,:,0].min():.4f}, {coords_array[:,:,0].max():.4f}]")
    print(f"  y range     : [{coords_array[:,:,1].min():.4f}, {coords_array[:,:,1].max():.4f}]")
    print(f"  z range     : [{coords_array[:,:,2].min():.4f}, {coords_array[:,:,2].max():.4f}]")


if __name__ == '__main__':
    main()