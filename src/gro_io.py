"""
Write 3D structures to GROMACS GRO format (one multi-frame file per set).
Coordinates are written as-is; use nm for standard GRO compatibility.
Each structure is written as a frame (timestep) in a single GRO file.
Canonical GRO: plain title line, fixed-width atom lines, box line with three space-separated floats.
Residue and atom numbers wrap at 99999 (GROMACS convention).
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
    title: str | None = None,
    residue_name: str = "MOL",
    atom_name: str = "CA",
    display_root: str | None = None,
    filename: str = DEFAULT_STRUCTURES_FILENAME,
) -> list[str]:
    """
    Write (n_structures, n_atoms, 3) coords to directory as one multi-frame GRO file.
    Each structure is one frame (title line, atom count, atom lines, box line).
    Title line is plain text; when title is None, uses "generated frame {i}" per frame.
    Atom lines follow GRO fixed-width layout; no velocity columns. Box line: three space-separated floats (zeros = undefined).
    Returns list containing the single written file path.
    """
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]
    n_structures, n_atoms, _ = coords.shape
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        for i in range(n_structures):
            title_line = title if title is not None else f"generated frame {i}"
            f.write(f"{title_line}\n")
            f.write(f"{n_atoms}\n")
            for a in range(n_atoms):
                resnum = (a + 1) % 100000
                atomnum = (a + 1) % 100000
                x, y, z = coords[i, a, 0], coords[i, a, 1], coords[i, a, 2]
                # GRO: %5d resnum, %-5s resname, %5s atomname, %5d atomnum, %8.3f %8.3f %8.3f x y z (no velocities)
                f.write(f"{resnum:5d}{residue_name:<5s}{atom_name:>5s}{atomnum:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            f.write("   0.00000   0.00000   0.00000\n")
    if display_root is not None:
        print(f"  Saved: {display_path(path, display_root)} ({n_structures} frames)")
    return [path]
