#!/usr/bin/env python3
"""
Generate the pipeline's bundled dataset: a multi-frame GRO file of sphere structures.

This is the only data shipped with the project. All data-dependent usage (smoke test,
sample config, demos) uses this sphere data. Large chromosome GRO files are not
included; use your own data path for production runs.

Usage:
  python tests/test_data/generate_spheres.py [--output PATH] [--num-structures N] [--beads B] ...
  Default: writes tests/test_data/spheres.gro with 10 structures, 100 beads each.

The pipeline loader (utils.load_data) expects frame title lines to start with
"Chromosome", so this script uses that prefix for compatibility.
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def points_on_sphere(n: int, seed: int = 42) -> np.ndarray:
    """Return n points evenly distributed on the unit sphere (Fibonacci-style). Shape (n, 3)."""
    # Golden spiral for roughly even spacing
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    indices = np.arange(n, dtype=float) + 0.5
    y = 1.0 - (indices / (n - 1)) * 2.0  # y from 1 to -1
    y = np.clip(y, -1.0, 1.0)  # avoid y outside [-1,1] from float rounding (would give NaN in radius)
    radius = np.sqrt(1.0 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1)


def write_gro(path: str, coords: np.ndarray, title_prefix: str = "Chromosome") -> None:
    """Write (n_frames, n_atoms, 3) coords to a multi-frame GRO file."""
    n_frames, n_atoms, _ = coords.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for i in range(n_frames):
            f.write(f"{title_prefix} test_data frame {i + 1} (sphere)\n")
            f.write(f"{n_atoms}\n")
            for a in range(n_atoms):
                x, y, z = coords[i, a, 0], coords[i, a, 1], coords[i, a, 2]
                # GRO: resnum, resname, atomname, atomnum, x, y, z
                f.write(f"{a + 1:5d}{'STRUC':<5s}{'CA':>5s}{a + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            f.write(f"{0:10.5f}{0:10.5f}{0:10.5f}\n")
    print(f"Wrote {n_frames} structures, {n_atoms} beads each -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate sphere GRO dataset for pipeline testing and demos.")
    p.add_argument("--output", "-o", default=None, help="Output GRO path (default: tests/test_data/spheres.gro)")
    p.add_argument("--num-structures", "-n", type=int, default=10, help="Number of structures (default: 10)")
    p.add_argument("--beads", "-b", type=int, default=100, help="Beads per structure (default: 100)")
    p.add_argument("--radius-min", type=float, default=1.0, help="Min sphere radius (nm) (default: 1.0)")
    p.add_argument("--radius-max", type=float, default=2.0, help="Max sphere radius (nm) (default: 2.0)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sphere points (default: 42)")
    args = p.parse_args()

    out = args.output
    if out is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(script_dir, "spheres.gro")

    n_structures = args.num_structures
    n_beads = args.beads
    unit = points_on_sphere(n_beads, seed=args.seed)  # (n_beads, 3)

    # Radii for each structure (linear spacing)
    radii = np.linspace(args.radius_min, args.radius_max, n_structures)
    coords = np.zeros((n_structures, n_beads, 3))
    for i in range(n_structures):
        coords[i] = unit * radii[i]

    write_gro(out, coords)
    return None


if __name__ == "__main__":
    main()
