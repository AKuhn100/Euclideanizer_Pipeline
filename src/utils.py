"""
Shared utilities for the Euclideanizer pipeline (DistMap + Euclideanizer).
Load_Data, Get_Device, Get_Distmaps, get_upper_tri, upper_tri_to_symmetric.
"""
import numpy as np
import torch


def Load_Data(gro_file):
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
        i += 1
        if i >= len(lines):
            break
        n_atoms = int(lines[i].strip())
        i += 1
        frame_coords = np.zeros((n_atoms, 3))
        for j in range(n_atoms):
            if i + j >= len(lines):
                break
            parts = lines[i + j].split()
            frame_coords[j] = [float(parts[3]), float(parts[4]), float(parts[5])]
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


def Get_Device():
    """Return the best available device: MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def Get_Distmaps(coords):
    """Compute pairwise distance matrices from (B, N, 3) coordinates; returns (B, N, N)."""
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    return torch.norm(diff, dim=-1)


def get_upper_tri(distmaps):
    """Extract upper-triangular (offset=1) elements from (B, N, N) distance maps; returns (B, tri_dim)."""
    num_atoms = distmaps.size(-1)
    idx = torch.triu_indices(num_atoms, num_atoms, offset=1)
    return distmaps[:, idx[0], idx[1]]


def upper_tri_to_symmetric(flat_tri, num_atoms):
    """Build symmetric (B, N, N) matrices from upper-triangle vectors (B, tri_dim)."""
    batch_size = flat_tri.size(0)
    mat = torch.zeros(batch_size, num_atoms, num_atoms, device=flat_tri.device, dtype=flat_tri.dtype)
    idx = torch.triu_indices(num_atoms, num_atoms, offset=1)
    mat[:, idx[0], idx[1]] = flat_tri
    mat[:, idx[1], idx[0]] = flat_tri
    return mat
