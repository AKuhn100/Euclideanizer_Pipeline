#!/usr/bin/env python3
"""
Test script: load one structure, rotate it, then align with our Kabsch and with scipy.
Reports RMSD(ref, our_aligned), RMSD(ref, scipy_aligned), and RMSD(our_aligned, scipy_aligned).

Run from Pipeline root:
  python tests/test_kabsch_rmsd.py [path/to/coords.npz]
  pytest tests/test_kabsch_rmsd.py -v
If no path is given, uses tests/test_data/spheres.npz (generate with
  python tests/test_data/generate_spheres.py).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_TEST_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.euclideanizer.loss import kabsch_align
from src.utils import load_data


def rmsd_centered(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two (N, 3) arrays (already centered or we center here)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return np.sqrt(np.mean((a - b) ** 2))


def main(npz_path: Optional[str] = None):
    """Run Kabsch vs scipy comparison. Returns (rmsd_ours, rmsd_scipy, rmsd_ours_vs_scipy)."""
    if npz_path is None:
        if len(sys.argv) > 1:
            npz_path = os.path.abspath(sys.argv[1])
        else:
            npz_path = os.path.join(_PIPELINE_ROOT, "tests", "test_data", "spheres.npz")
    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        print("Generate spheres with: python tests/test_data/generate_spheres.py")
        print("Or pass an NPZ path: python tests/test_kabsch_rmsd.py path/to/coords.npz")
        sys.exit(1)

    coords = load_data(npz_path)
    ref = coords[0].astype(np.float64)
    n_atoms = ref.shape[0]

    # Center reference
    ref_centered = ref - ref.mean(axis=0)

    # Random rotation (same as rotating the structure)
    rng = np.random.default_rng(42)
    rot = Rotation.random(random_state=rng)
    rotated = rot.apply(ref_centered)

    # ---- Our Kabsch ----
    ref_t = torch.from_numpy(ref_centered).float().unsqueeze(0)
    rotated_t = torch.from_numpy(rotated).float().unsqueeze(0)
    aligned_ours, gt_c = kabsch_align(rotated_t, ref_t)
    aligned_ours_np = aligned_ours.squeeze(0).numpy()
    ref_c_np = gt_c.squeeze(0).numpy()
    rmsd_ours = rmsd_centered(aligned_ours_np, ref_c_np)

    # ---- Scipy Kabsch ----
    rot_scipy, _ = Rotation.align_vectors(ref_centered, rotated)
    aligned_scipy = rot_scipy.apply(rotated)
    rmsd_scipy = rmsd_centered(aligned_scipy, ref_centered)

    # RMSD between our aligned and scipy aligned (should be ~0)
    rmsd_ours_vs_scipy = rmsd_centered(aligned_ours_np, aligned_scipy)

    print(f"Loaded: {npz_path} (1 frame, {n_atoms} atoms)")
    print()
    print("RMSD(ref_centered, our_Kabsch_aligned(rotated))  = {:.6f}".format(rmsd_ours))
    print("RMSD(ref_centered, scipy_Kabsch_aligned(rotated)) = {:.6f}".format(rmsd_scipy))
    print("RMSD(our_aligned, scipy_aligned)                  = {:.6f}".format(rmsd_ours_vs_scipy))
    print()
    if rmsd_ours < 1e-5 and rmsd_scipy < 1e-5 and rmsd_ours_vs_scipy < 1e-5:
        print("PASS: Both alignments recover reference; our Kabsch matches scipy.")
    else:
        print("Check: expected all three RMSDs near zero.")

    return rmsd_ours, rmsd_scipy, rmsd_ours_vs_scipy


def test_kabsch_matches_scipy_on_rotated_structure():
    """Our Kabsch alignment matches scipy and recovers reference (run with pytest tests/ -v)."""
    import pytest

    npz_path = os.path.join(_PIPELINE_ROOT, "tests", "test_data", "spheres.npz")
    if not os.path.isfile(npz_path):
        pytest.skip(
            "spheres.npz not found; run python tests/test_data/generate_spheres.py"
        )

    # Pass npz_path explicitly so main() does not use pytest's sys.argv
    rmsd_ours, rmsd_scipy, rmsd_ours_vs_scipy = main(npz_path=npz_path)
    assert rmsd_ours < 1e-5, f"our Kabsch RMSD to ref should be ~0, got {rmsd_ours}"
    assert rmsd_scipy < 1e-5, f"scipy Kabsch RMSD to ref should be ~0, got {rmsd_scipy}"
    assert rmsd_ours_vs_scipy < 1e-5, f"our vs scipy aligned should be ~0, got {rmsd_ours_vs_scipy}"


if __name__ == "__main__":
    main()
