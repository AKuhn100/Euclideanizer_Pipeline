"""
Write 3D structures to GROMACS GRO format (one multi-frame file per set).
Coordinates are written as-is; use nm for standard GRO compatibility.
Each structure is written as a frame (timestep) in a single GRO file, like pipeline input data.
"""
from __future__ import annotations

import os
import numpy as np

from .utils import display_path

DEFAULT_STRUCTURES_FILENAME = "structures.gro"


def write_structures_gro(
    coords: np.ndarray,
    directory: str,
    *,
    title_prefix: str = "generated",
    residue_name: str = "STRUC",
    atom_name: str = "CA",
    display_root: str | None = None,
    filename: str = DEFAULT_STRUCTURES_FILENAME,
) -> list[str]:
    """
    Write (n_structures, n_atoms, 3) coords to directory as one multi-frame GRO file.
    Each structure is one frame (title line, atom count, atom lines, box line), so the
    result can be loaded like pipeline input data. One file per generated set.
    Coordinates are written in nm (scale before calling if your data is in Angstrom).
    Returns list containing the single written file path.
    """
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]
    n_structures, n_atoms, _ = coords.shape
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        for i in range(n_structures):
            f.write(f"{title_prefix} frame {i}\n")
            f.write(f"{n_atoms}\n")
            for a in range(n_atoms):
                x, y, z = coords[i, a, 0], coords[i, a, 1], coords[i, a, 2]
                # GRO: %5d resnum, %-5s resname, %5s atomname, %5d atomnum, %8.3f %8.3f %8.3f x y z
                f.write(f"{a + 1:5d}{residue_name:<5s}{atom_name:>5s}{a + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            f.write(f"{0:10.5f}{0:10.5f}{0:10.5f}\n")
    if display_root is not None:
        print(f"  Saved: {display_path(path, display_root)} ({n_structures} frames)")
    return [path]
