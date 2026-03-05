"""
Shared utilities for the Euclideanizer pipeline (DistMap + Euclideanizer).
load_data, get_device, get_distmaps, get_upper_tri, upper_tri_to_symmetric, display_path, get_train_test_split.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset


def load_data(gro_file: str) -> np.ndarray:
    """Load coordinate frames from a GRO-style file; returns (n_frames, n_atoms, 3) array."""
    coords_list = []
    with open(gro_file, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        title = lines[i].strip()
        if not title.startswith("Chromosome"):
            i += 1
            continue
        if not title:
            raise ValueError(
                f"GRO file {gro_file!r} line {i + 1}: title line is empty after strip. Expected non-empty title."
            )
        i += 1
        if i >= len(lines):
            break
        try:
            n_atoms = int(lines[i].strip())
        except ValueError as e:
            raise ValueError(
                f"GRO file {gro_file!r} line {i + 1}: expected integer atom count, got {lines[i].strip()!r}."
            ) from e
        if n_atoms <= 0:
            raise ValueError(
                f"GRO file {gro_file!r} line {i + 1}: atom count must be positive, got {n_atoms}."
            )
        i += 1
        frame_coords = np.zeros((n_atoms, 3))
        for j in range(n_atoms):
            line_idx = i + j
            if line_idx >= len(lines):
                raise ValueError(
                    f"GRO file {gro_file!r} line {line_idx + 1}: expected {n_atoms} atom lines, got {j}."
                )
            parts = lines[line_idx].split()
            if len(parts) < 6:
                raise ValueError(
                    f"GRO file {gro_file!r} line {line_idx + 1}: atom line must have at least 6 columns "
                    f"(resnum, resname, atomname, atomnum, x, y, z), got {len(parts)}."
                )
            try:
                frame_coords[j] = [float(parts[3]), float(parts[4]), float(parts[5])]
            except ValueError as e:
                raise ValueError(
                    f"GRO file {gro_file!r} line {line_idx + 1}: columns 4–6 must be numeric x,y,z, got "
                    f"{parts[3]!r}, {parts[4]!r}, {parts[5]!r}."
                ) from e
        coords_list.append(frame_coords)
        i += n_atoms + 1
    out = np.array(coords_list)
    if not np.isfinite(out).all():
        n_bad = np.logical_not(np.isfinite(out)).sum()
        raise ValueError(
            f"GRO file {gro_file!r} contains non-finite coordinates ({n_bad} NaN/inf values). "
            "Coordinates must be finite."
        )
    return out


def get_device() -> torch.device:
    """Return the best available device: MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_distmaps(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance matrices from (B, N, 3) coordinates; returns (B, N, N)."""
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    return torch.norm(diff, dim=-1)


def get_upper_tri(distmaps: torch.Tensor) -> torch.Tensor:
    """Extract upper-triangular (offset=1) elements from (B, N, N) distance maps; returns (B, tri_dim)."""
    num_atoms = distmaps.size(-1)
    idx = torch.triu_indices(num_atoms, num_atoms, offset=1)
    return distmaps[:, idx[0], idx[1]]


def upper_tri_to_symmetric(flat_tri: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Build symmetric (B, N, N) matrices from upper-triangle vectors (B, tri_dim)."""
    batch_size = flat_tri.size(0)
    mat = torch.zeros(batch_size, num_atoms, num_atoms, device=flat_tri.device, dtype=flat_tri.dtype)
    idx = torch.triu_indices(num_atoms, num_atoms, offset=1)
    mat[:, idx[0], idx[1]] = flat_tri
    mat[:, idx[1], idx[0]] = flat_tri
    return mat


def display_path(path: str, root: str | None) -> str:
    """Return path relative to root for display, or path unchanged if root is None."""
    if not root:
        return path
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def validate_dataset_for_pipeline(num_structures: int, training_split: float) -> None:
    """
    Raise ValueError if the dataset is not valid for the pipeline: need at least 2 structures
    and after applying training_split both train and test sizes must be >= 1.
    """
    if num_structures < 2:
        raise ValueError(
            f"Dataset has {num_structures} structure(s). At least 2 structures are required for train/test split."
        )
    train_size = int(training_split * num_structures)
    test_size = num_structures - train_size
    if train_size < 1 or test_size < 1:
        raise ValueError(
            f"Dataset has {num_structures} structure(s); train/test split (training_split={training_split}) gives "
            f"train_size={train_size}, test_size={test_size}. At least one structure in each of train and test "
            "is required."
        )


def get_train_test_split(
    coords: torch.Tensor,
    training_split: float,
    split_seed: int,
) -> Tuple[Subset, Subset]:
    """Split coords into train and test subsets using the given fraction and seed. Returns (train_ds, test_ds)."""
    train_size = int(training_split * len(coords))
    test_size = len(coords) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_ds, test_ds = random_split(coords, [train_size, test_size], generator=generator)
    return train_ds, test_ds
