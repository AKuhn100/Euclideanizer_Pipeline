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
    Parse a GROMACS .gro file and extract all structures (frames).
    
    GRO format:
        Line 1: title
        Line 2: number of atoms (integer)
        Lines 3 to 3+n_atoms-1: atom data
            residue number, residue name, atom name, atom number, x, y, z
            (optionally followed by vx, vy, vz velocities)
        Last line: box vectors
    
    We parse coordinates by splitting on whitespace and taking the last 3
    numeric fields from each atom line (or fields at known positions after split).
    
    Returns:
        list of numpy arrays, each of shape (n_atoms, 3)
    """
    structures = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError(f"File {filepath} is empty.")
    
    i = 0
    total_lines = len(lines)
    
    while i < total_lines:
        # Skip blank lines between frames
        if lines[i].strip() == '':
            i += 1
            continue
        
        # Line 1: title (can be anything)
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
                f"Expected integer number of atoms, got '{natoms_line}' "
                f"at line {i} of file {filepath}"
            )
        
        if n_atoms < 2:
            raise ValueError(
                f"Number of atoms must be >= 2, got {n_atoms} "
                f"at line {i} of file {filepath}"
            )
        
        if i + n_atoms > total_lines:
            raise ValueError(
                f"File {filepath} is truncated: expected {n_atoms} atom lines "
                f"starting at line {i+1}, but file only has {total_lines} lines."
            )
        
        coords = np.zeros((n_atoms, 3), dtype=np.float64)
        
        for atom_idx in range(n_atoms):
            line = lines[i]
            i += 1
            
            # GRO atom line format (fixed width spec):
            #   cols  0-4:  residue number (5 chars)
            #   cols  5-9:  residue name (5 chars)
            #   cols 10-14: atom name (5 chars)
            #   cols 15-19: atom number (5 chars)
            #   cols 20-27: x (8 chars)
            #   cols 28-35: y (8 chars)
            #   cols 36-43: z (8 chars)
            #   (optional velocity columns follow)
            #
            # However, fixed-width parsing can fail with unusual formatting.
            # We use whitespace splitting and take the last 3 (or last 3 of
            # numeric fields) as x, y, z for robustness.
            
            parts = line.split()
            
            # GRO atom lines have at least 6 fields:
            # resnum+resname (merged or separate), atomname, atomnum, x, y, z
            # The merged field is like "1LYS" or "  1LYS"
            # After split we typically get:
            #   ['1LYS', 'CA', '1', '-0.374', '-0.409', '0.080']  (6 fields, no vel)
            # or with velocities:
            #   ['1LYS', 'CA', '1', '-0.374', '-0.409', '0.080', 'vx', 'vy', 'vz']
            
            if len(parts) < 6:
                raise ValueError(
                    f"Expected at least 6 fields on atom line {i} of {filepath}, "
                    f"got {len(parts)}: '{line.rstrip()}'"
                )
            
            # Identify coordinate fields: after resname+resnum (field 0),
            # atomname (field 1), atomnum (field 2), we have x, y, z at [3], [4], [5]
            # But to be safe, we take the last 3 fields if there are no velocities,
            # or fields [3:6] if there are velocities (6 or 9 total fields).
            # 
            # Strategy: take fields at positions 3, 4, 5 (0-indexed).
            # These should always be x, y, z in standard GRO format.
            # If there are velocities, they appear at positions 6, 7, 8.
            
            try:
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except (ValueError, IndexError) as e:
                # Fallback: try last 3 fields
                try:
                    x = float(parts[-3])
                    y = float(parts[-2])
                    z = float(parts[-1])
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Cannot parse x,y,z from atom line {i} of {filepath}: "
                        f"'{line.rstrip()}'. Error: {e}"
                    )
            
            coords[atom_idx, 0] = x
            coords[atom_idx, 1] = y
            coords[atom_idx, 2] = z
        
        # Skip the box vectors line
        if i < total_lines and lines[i].strip() != '':
            i += 1  # box line
        elif i < total_lines:
            i += 1  # blank or box line
        
        # Validate no NaN or inf
        if not np.all(np.isfinite(coords)):
            bad_count = np.sum(~np.isfinite(coords))
            raise ValueError(
                f"Found {bad_count} non-finite values in structure "
                f"{len(structures)+1} of file {filepath}."
            )
        
        structures.append(coords)
    
    return structures


def validate_coords_array(coords_array, filepath_hint=""):
    """
    Validate the final coords array against required criteria.
    """
    if coords_array.ndim != 3:
        raise ValueError(
            f"Expected 3D array (n_structures, n_atoms, 3), "
            f"got shape {coords_array.shape}"
        )
    if coords_array.shape[2] != 3:
        raise ValueError(
            f"Last dimension must be 3 (x, y, z), got {coords_array.shape[2]}"
        )
    if coords_array.shape[0] < 1:
        raise ValueError(
            f"Must have at least 1 structure, got {coords_array.shape[0]}"
        )
    if coords_array.shape[1] < 2:
        raise ValueError(
            f"Must have at least 2 atoms, got {coords_array.shape[1]}"
        )
    if not np.all(np.isfinite(coords_array)):
        bad = np.sum(~np.isfinite(coords_array))
        raise ValueError(
            f"Found {bad} non-finite values in coords array."
        )
    
    # Check that first column is not atom indices (0, 1, 2, ..., n_atoms-1)
    # For each structure, compute correlation of x with arange(n_atoms)
    n_structures, n_atoms, _ = coords_array.shape
    atom_indices = np.arange(n_atoms, dtype=np.float64)
    
    suspicious_count = 0
    for s in range(min(n_structures, 10)):  # check up to 10 structures
        x_col = coords_array[s, :, 0].astype(np.float64)
        # Compute Pearson correlation
        x_std = np.std(x_col)
        idx_std = np.std(atom_indices)
        if x_std > 0 and idx_std > 0:
            corr = np.corrcoef(x_col, atom_indices)[0, 1]
            if abs(corr) > 0.99:
                suspicious_count += 1
    
    if suspicious_count > min(n_structures, 10) * 0.5:
        raise ValueError(
            f"First column appears to be atom indices, not spatial x coordinates. "
            f"{suspicious_count} out of {min(n_structures, 10)} structures have "
            f"x column strongly correlated (r>0.99) with atom index sequence."
        )
    
    # Check dtype can be cast to float32
    try:
        coords_array.astype(np.float32)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Array dtype {coords_array.dtype} cannot be cast to float32: {e}")


def collect_gro_files(input_path):
    """
    Given a path (file or directory), return a sorted list of .gro file paths.
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
                f"No .gro files found in directory: {input_path}"
            )
        return files
    else:
        raise ValueError(
            f"Input path does not exist or is not a file/directory: {input_path}"
        )


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python solution.py <input_path> <output_path>\n"
            "  input_path:  a single .gro file or a directory containing .gro files\n"
            "  output_path: path for the output .npz file"
        )
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Collect files to process
    gro_files = collect_gro_files(input_path)
    
    all_structures = []
    n_atoms_expected = None
    
    for filepath in gro_files:
        structures = parse_gro_file(filepath)
        
        if not structures:
            print(f"Warning: No structures found in {filepath}, skipping.")
            continue
        
        for s_idx, struct in enumerate(structures):
            n_atoms = struct.shape[0]
            
            if n_atoms_expected is None:
                n_atoms_expected = n_atoms
            elif n_atoms != n_atoms_expected:
                raise ValueError(
                    f"Inconsistent atom count: expected {n_atoms_expected} atoms "
                    f"(from first structure), but structure {s_idx+1} in {filepath} "
                    f"has {n_atoms} atoms. All structures must have the same number of atoms."
                )
            
            all_structures.append(struct)
    
    if not all_structures:
        raise ValueError(
            f"No valid structures found in input: {input_path}"
        )
    
    # Stack all structures into a single array
    coords_array = np.stack(all_structures, axis=0)  # shape: (n_structures, n_atoms, 3)
    
    # Validate
    validate_coords_array(coords_array, filepath_hint=input_path)
    
    # Write output
    np.savez_compressed(output_path, coords=coords_array)
    
    n_structures, n_atoms, _ = coords_array.shape
    print(
        f"Successfully wrote {n_structures} structure(s) with {n_atoms} atom(s) each "
        f"to '{output_path}' (shape={coords_array.shape}, dtype={coords_array.dtype})."
    )


if __name__ == '__main__':
    main()