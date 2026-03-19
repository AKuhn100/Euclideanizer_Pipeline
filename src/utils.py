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


def load_data(npz_path: str, max_data: int | None = None, seed: int = 0) -> np.ndarray:
    """Load coordinate array from NPZ (key 'coords') and optionally subsample up to max_data."""
    if not os.path.isfile(npz_path):
        raise ValueError(f"Dataset file does not exist or cannot be read: {npz_path!r}.")
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Cannot load NPZ file {npz_path!r}: {e}.") from e
    try:
        coords = data["coords"]
    except KeyError:
        raise ValueError(
            f"NPZ file {npz_path!r} does not contain required key 'coords'. "
            "Expected one array with key 'coords' of shape (n_structures, n_atoms, 3)."
        )
    if coords.ndim != 3:
        raise ValueError(
            f"NPZ file {npz_path!r}: 'coords' must have 3 dimensions (n_structures, n_atoms, 3), got ndim={coords.ndim}."
        )
    if coords.shape[2] != 3:
        raise ValueError(
            f"NPZ file {npz_path!r}: 'coords' last dimension must be 3 (x,y,z), got shape[-1]={coords.shape[2]}."
        )
    if coords.shape[0] < 1:
        raise ValueError(
            f"NPZ file {npz_path!r}: 'coords' must have at least one structure, got shape[0]={coords.shape[0]}."
        )
    if coords.shape[1] < 2:
        raise ValueError(
            f"NPZ file {npz_path!r}: 'coords' must have at least 2 atoms (distance map requires 2+), got shape[1]={coords.shape[1]}."
        )
    if not np.isfinite(coords).all():
        n_bad = np.logical_not(np.isfinite(coords)).sum()
        raise ValueError(
            f"NPZ file {npz_path!r}: 'coords' contains non-finite values ({n_bad} NaN/inf). Coordinates must be finite."
        )
    out = np.asarray(coords, dtype=np.float32)
    if max_data is not None and max_data < out.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(out.shape[0], size=max_data, replace=False)
        out = out[idx]
    return out


def get_available_cuda_count() -> int:
    """Return the number of available CUDA devices (0 if CUDA not available)."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_device(device_index: int | None = None) -> torch.device:
    """Return the best available device: MPS, CUDA, or CPU.
    If device_index is not None and CUDA is available, return cuda:device_index (must be < device_count).
    """
    if device_index is not None:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        n = torch.cuda.device_count()
        if device_index < 0 or device_index >= n:
            raise ValueError(f"device_index must be in [0, {n}), got {device_index}")
        return torch.device(f"cuda:{device_index}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
