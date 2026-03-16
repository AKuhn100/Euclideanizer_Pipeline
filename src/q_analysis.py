"""
Q (pairwise-distance similarity) distributions: integrate into pipeline for each Euclideanizer run.
Computes test→train, gen→train, gen→test max Q (best match) and recon Q. Uses seed-level cache for test→train max Q.
"""
from __future__ import annotations

import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import display_path, get_train_test_split
from .plotting import _save_pdf_copy
from .plot_config import (
    GEN_PANEL_COLORS,
    RECON_PANEL_COLORS,
    FONT_FAMILY,
    FONT_SIZE_TITLE,
    FONT_SIZE_AXIS,
    FONT_SIZE_LEGEND,
)

def _distmap_from_coords(coords: np.ndarray) -> np.ndarray:
    """Single structure (N, 3) -> (N, N) pairwise distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _upper_tri_from_distmap(D: np.ndarray) -> np.ndarray:
    """(..., N, N) -> (..., tri) upper triangle (offset=1, no diagonal)."""
    N = D.shape[-1]
    i_idx, j_idx = np.triu_indices(N, k=1)
    return D[..., i_idx, j_idx]


def q_single(coords_alpha: np.ndarray, coords_beta: np.ndarray, delta: float) -> float:
    """
    Q(α, β) = (1/N) * sum_{i<j} exp( - (r_ij(α) - r_ij(β))^2 / (2*delta^2) ).
    Two (N, 3) arrays -> one float in [0, 1]. Higher = more similar.
    """
    D_alpha = _distmap_from_coords(coords_alpha)
    D_beta = _distmap_from_coords(coords_beta)
    tri_alpha = _upper_tri_from_distmap(D_alpha[np.newaxis, ...]).squeeze(0)
    tri_beta = _upper_tri_from_distmap(D_beta[np.newaxis, ...]).squeeze(0)
    diff_sq = (tri_alpha - tri_beta) ** 2
    return float(np.mean(np.exp(-diff_sq / (2.0 * delta ** 2))))


def _q_matrix_batch(queries: np.ndarray, ref_coords: np.ndarray, delta: float) -> np.ndarray:
    """
    (B, N, 3) queries vs (M, N, 3) refs -> (B, M) Q matrix.
    Each entry is Q(query_b, ref_m) = mean over upper-triangle distances of exp(-(d_ij^q - d_ij^r)^2/(2*delta^2)).
    """
    B, N, _ = queries.shape
    M = ref_coords.shape[0]
    # (B, N, N) and (M, N, N) distance matrices
    diff_q = queries[:, np.newaxis, :, :] - queries[:, :, np.newaxis, :]
    D_q = np.sqrt(np.sum(diff_q ** 2, axis=-1))
    diff_r = ref_coords[:, np.newaxis, :, :] - ref_coords[:, :, np.newaxis, :]
    D_r = np.sqrt(np.sum(diff_r ** 2, axis=-1))
    i_idx, j_idx = np.triu_indices(N, k=1)
    tri_q = D_q[:, i_idx, j_idx]
    tri_r = D_r[:, i_idx, j_idx]
    # (B, tri) and (M, tri) -> (B, M, tri)
    diff_sq = (tri_q[:, np.newaxis, :] - tri_r[np.newaxis, :, :]) ** 2
    Q_mat = np.mean(np.exp(-diff_sq / (2.0 * delta ** 2)), axis=-1)
    return Q_mat.astype(np.float32)


def max_q_batch(
    queries: np.ndarray,
    ref_coords: np.ndarray,
    delta: float,
    query_batch_size: int,
    desc: str | None = None,
) -> np.ndarray:
    """
    For each of `queries` (Q, N, 3), return max Q (best match) to any of `ref_coords` (M, N, 3).
    Returns (Q,) array. Docstrings and progress text say "max Q".
    """
    Q = queries.shape[0]
    out = np.empty(Q, dtype=np.float32)
    batch_starts = range(0, Q, query_batch_size)
    for start in tqdm(batch_starts, desc=desc or "max Q", unit="batch", leave=True):
        end = min(start + query_batch_size, Q)
        batch = queries[start:end]
        q_mat = _q_matrix_batch(batch, ref_coords, delta)
        out[start:end] = q_mat.max(axis=1)
    return out


def get_or_compute_test_to_train_q(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    cache_path: str,
    delta: float,
    query_batch_size: int,
    max_train: int | None = None,
    max_test: int | None = None,
    display_root: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test→train max Q (and train/test coords) from seed-level cache, or compute and save.
    Returns (test_to_train_max_q, train_coords_np, test_coords_np). max_train/max_test cap reference
    set sizes (None = use all). Cache is keyed by (max_train, max_test) via cache_path.
    Always written when computed (independent of analysis save_data).
    """
    if os.path.isfile(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            out = (
                np.asarray(loaded["test_to_train_max_q"], dtype=np.float32),
                np.asarray(loaded["train_coords_np"], dtype=np.float32),
                np.asarray(loaded["test_coords_np"], dtype=np.float32),
            )
            loaded.close()
            if display_root is not None:
                print(f"  Loaded seed-level test→train max Q cache: {display_path(cache_path, display_root)}")
            return out
        except Exception:
            pass
    if display_root is not None:
        print("  Computing test→train max Q (seed-level cache)...", flush=True)
    coords = coords_tensor
    train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
    tr_idx = train_ds.indices
    te_idx = test_ds.indices
    if hasattr(tr_idx, "tolist"):
        tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
    train_coords_np = coords_np[tr_idx]
    test_coords_np = coords_np[te_idx]
    if max_train is not None:
        train_coords_np = train_coords_np[:max_train]
    if max_test is not None:
        test_coords_np = test_coords_np[:max_test]
    test_to_train_max_q = max_q_batch(
        test_coords_np, train_coords_np, delta,
        query_batch_size=query_batch_size,
        desc="Test → Train (max Q)",
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        test_to_train_max_q=test_to_train_max_q,
        train_coords_np=train_coords_np,
        test_coords_np=test_coords_np,
    )
    if display_root is not None:
        print(f"  Saved seed-level test→train max Q cache: {display_path(cache_path, display_root)}")
    return test_to_train_max_q, train_coords_np, test_coords_np


def _run_one_q(
    run_dir_this: str,
    test_to_train_max_q: np.ndarray,
    gen_to_train_max_q: np.ndarray,
    gen_to_test_max_q: np.ndarray,
    gen_coords_np: np.ndarray | None,
    plot_cfg: dict,
    display_root: str | None,
    save_data: bool,
    save_structures_gro: bool,
) -> str:
    """Write Q figure (max Q histograms), optional data/, optional structures/. Returns path to figure."""
    dpi = plot_cfg["plot_dpi"]
    all_vals = np.concatenate([test_to_train_max_q, gen_to_train_max_q, gen_to_test_max_q])
    x_min = max(0.0, np.percentile(all_vals, 0.5) - 0.02)
    x_max = min(1.0, np.percentile(all_vals, 99.5) + 0.02)
    bins = np.linspace(x_min, x_max, 50)

    c0, c1, c2 = GEN_PANEL_COLORS
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].hist(test_to_train_max_q, bins=bins, density=True, alpha=0.7, color=c0, edgecolor="k", linewidth=0.3)
    axes[0].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[0].set_title("Test → Train (Max Q to Train Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[0].set_xlim(x_min, x_max)
    axes[1].hist(gen_to_train_max_q, bins=bins, density=True, alpha=0.7, color=c1, edgecolor="k", linewidth=0.3)
    axes[1].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[1].set_title("Gen → Train (Max Q to Train Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[1].set_xlim(x_min, x_max)
    axes[2].hist(gen_to_test_max_q, bins=bins, density=True, alpha=0.7, color=c2, edgecolor="k", linewidth=0.3)
    axes[2].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_xlabel("Max Q", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_title("Gen → Test (Max Q to Test Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[2].set_xlim(x_min, x_max)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)
    plt.tight_layout()
    os.makedirs(run_dir_this, exist_ok=True)
    out_path = os.path.join(run_dir_this, "q_distributions.png")
    plt.savefig(out_path, dpi=dpi)
    if plot_cfg["save_pdf_copy"]:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close()

    if save_data:
        data_dir = os.path.join(run_dir_this, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "q_data.npz")
        save_kw: dict = dict(
            gen_to_train=gen_to_train_max_q,
            gen_to_test=gen_to_test_max_q,
            bins=bins,
        )
        if gen_coords_np is not None and plot_cfg["save_gen_coords_in_npz"]:
            save_kw["gen_coords"] = gen_coords_np
        np.savez_compressed(data_path, **save_kw)
        print(f"  Saved: {display_path(data_path, display_root)}")

    if save_structures_gro and gen_coords_np is not None:
        from .gro_io import write_structures_gro
        structures_dir = os.path.join(run_dir_this, "structures")
        write_structures_gro(gen_coords_np, structures_dir, display_root=display_root)

    print(f"  Saved: {display_path(out_path, display_root)}")
    return out_path


def run_q_analysis(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    frozen_vae,
    embed,
    latent_dim: int,
    device: torch.device,
    run_dir: str,
    plot_cfg: dict,
    *,
    num_samples: int | None = None,
    sample_variance: float | None = None,
    query_batch_size: int,
    delta: float,
    output_suffix: str = "",
    display_root: str | None = None,
    precomputed_test_to_train_max_q: np.ndarray | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
) -> str:
    """
    Compute max Q distributions (test→train, gen→train, gen→test). Saves to run_dir/analysis/q/gen/<run_name>/
    with: q_distributions.png, data/ (if save_data), structures/ (if save_structures_gro).
    When precomputed_test_to_train_max_q, train_coords_np, test_coords_np are provided (e.g. from seed-level cache), they are reused.
    """
    run_name = output_suffix.lstrip("_") if output_suffix else "default"
    run_dir_this = os.path.join(run_dir, "analysis", "q", "gen", run_name)
    save_data = plot_cfg["save_data"]
    save_structures_gro = plot_cfg["save_structures_gro"]

    _n = plot_cfg["q_num_samples"]
    n_gen = num_samples if num_samples is not None else (_n[0] if isinstance(_n, list) else _n)
    _v = plot_cfg["q_sample_variance"]
    variance = sample_variance if sample_variance is not None else (_v[0] if isinstance(_v, list) else _v)
    batch_size = query_batch_size
    print(f"  Q (max Q): n_gen={n_gen}, variance={variance} (test→train, gen→train, gen→test)...")

    if precomputed_test_to_train_max_q is not None and train_coords_np is not None and test_coords_np is not None:
        test_to_train_max_q = precomputed_test_to_train_max_q
    else:
        coords = coords_tensor
        train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
        tr_idx = train_ds.indices
        te_idx = test_ds.indices
        if hasattr(tr_idx, "tolist"):
            tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
        train_coords_np = coords_np[tr_idx]
        test_coords_np = coords_np[te_idx]
        test_to_train_max_q = max_q_batch(
            test_coords_np, train_coords_np, delta,
            query_batch_size=batch_size, desc="Test → Train (max Q)",
        )

    from .distmap.sample import generate_samples
    gen_decode_batch_size = plot_cfg["gen_decode_batch_size"]
    embed.eval()
    with torch.no_grad():
        z = generate_samples(n_gen, latent_dim, device, variance=variance)
        out_coords = []
        for start in range(0, n_gen, gen_decode_batch_size):
            end = min(start + gen_decode_batch_size, n_gen)
            D_noneuclid = frozen_vae._decode_to_matrix(z[start:end])
            coords_chunk = embed(D_noneuclid)
            out_coords.append(coords_chunk.cpu().numpy().astype(np.float32))
        gen_coords_np = np.concatenate(out_coords, axis=0)

    gen_to_train_max_q = max_q_batch(
        gen_coords_np, train_coords_np, delta, query_batch_size=batch_size, desc="Gen → Train (max Q)"
    )
    gen_to_test_max_q = max_q_batch(
        gen_coords_np, test_coords_np, delta, query_batch_size=batch_size, desc="Gen → Test (max Q)"
    )

    return _run_one_q(
        run_dir_this, test_to_train_max_q, gen_to_train_max_q, gen_to_test_max_q,
        gen_coords_np if save_structures_gro else None,
        plot_cfg, display_root, save_data, save_structures_gro,
    )


def run_q_analysis_multi(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    frozen_vae,
    embed,
    latent_dim: int,
    device: torch.device,
    run_dir: str,
    plot_cfg: dict,
    *,
    num_samples_list: list[int],
    sample_variance: float,
    delta: float,
    query_batch_size: int,
    variance_suffix: str = "",
    display_root: str | None = None,
    precomputed_test_to_train_max_q: np.ndarray | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
) -> list[str]:
    """
    Run max Q analysis for multiple num_samples with the same variance, using incremental generation and merging.
    When precomputed_test_to_train_max_q, train_coords_np, test_coords_np are provided (e.g. from seed-level cache), they are reused.
    """
    from .distmap.sample import generate_samples
    import shutil

    batch_size = query_batch_size
    gen_decode_batch_size = plot_cfg["gen_decode_batch_size"]
    save_data = plot_cfg["save_data"]
    save_structures_gro = plot_cfg["save_structures_gro"]
    sorted_n = sorted(set(num_samples_list))
    if not sorted_n:
        return []

    data_dirs_to_remove: list[str] = []
    write_data = True

    if precomputed_test_to_train_max_q is not None and train_coords_np is not None and test_coords_np is not None:
        test_to_train_max_q = precomputed_test_to_train_max_q
    else:
        coords = coords_tensor
        train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
        tr_idx = train_ds.indices
        te_idx = test_ds.indices
        if hasattr(tr_idx, "tolist"):
            tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
        train_coords_np = coords_np[tr_idx]
        test_coords_np = coords_np[te_idx]
        test_to_train_max_q = max_q_batch(
            test_coords_np, train_coords_np, delta,
            query_batch_size=batch_size, desc="Test → Train (max Q)",
        )

    embed.eval()
    acc_gen_to_train: list[np.ndarray] = []
    acc_gen_to_test: list[np.ndarray] = []
    acc_gen_coords: list[np.ndarray] = []
    out_paths: list[str] = []

    plot_cfg_resume = dict(plot_cfg)
    if save_structures_gro:
        plot_cfg_resume["save_gen_coords_in_npz"] = True
    loaded_n = 0
    for n in reversed(sorted_n):
        run_name = (str(n) + variance_suffix) if variance_suffix else str(n)
        run_dir_n = os.path.join(run_dir, "analysis", "q", "gen", run_name)
        fig_p = os.path.join(run_dir_n, "q_distributions.png")
        data_p = os.path.join(run_dir_n, "data", "q_data.npz")
        if os.path.isfile(fig_p) and os.path.isfile(data_p):
            try:
                loaded = np.load(data_p, allow_pickle=False)
                gtt = loaded["gen_to_train"]
                gte = loaded["gen_to_test"]
                loaded_n = int(gtt.shape[0])
                acc_gen_to_train = [np.asarray(gtt, dtype=np.float32)]
                acc_gen_to_test = [np.asarray(gte, dtype=np.float32)]
                if "gen_coords" in loaded:
                    acc_gen_coords = [np.asarray(loaded["gen_coords"], dtype=np.float32)]
                else:
                    acc_gen_coords = []
                loaded.close()
                print(f"  Q (max Q): resuming from n={n} (loaded {loaded_n} samples), generating remaining...")
            except Exception:
                pass
            break

    for n in sorted_n:
        need = n - sum(a.shape[0] for a in acc_gen_to_train)
        if need <= 0:
            gen_to_train_max_q = np.concatenate(acc_gen_to_train, axis=0) if acc_gen_to_train else np.array([], dtype=np.float32)
            gen_to_test_max_q = np.concatenate(acc_gen_to_test, axis=0) if acc_gen_to_test else np.array([], dtype=np.float32)
            gen_coords_np = np.concatenate(acc_gen_coords, axis=0) if acc_gen_coords else None
        else:
            with torch.no_grad():
                z = generate_samples(need, latent_dim, device, variance=sample_variance)
                out_coords = []
                for start in range(0, need, gen_decode_batch_size):
                    end = min(start + gen_decode_batch_size, need)
                    D_noneuclid = frozen_vae._decode_to_matrix(z[start:end])
                    coords_chunk = embed(D_noneuclid)
                    out_coords.append(coords_chunk.cpu().numpy().astype(np.float32))
                gen_chunk_np = np.concatenate(out_coords, axis=0)
            acc_gen_coords.append(gen_chunk_np)
            gen_to_train_chunk = max_q_batch(
                gen_chunk_np, train_coords_np, delta, query_batch_size=batch_size,
                desc=f"Gen → Train (max Q) +{need}",
            )
            gen_to_test_chunk = max_q_batch(
                gen_chunk_np, test_coords_np, delta, query_batch_size=batch_size,
                desc=f"Gen → Test (max Q) +{need}",
            )
            acc_gen_to_train.append(gen_to_train_chunk)
            acc_gen_to_test.append(gen_to_test_chunk)
            gen_to_train_max_q = np.concatenate(acc_gen_to_train, axis=0)
            gen_to_test_max_q = np.concatenate(acc_gen_to_test, axis=0)
            gen_coords_np = np.concatenate(acc_gen_coords, axis=0)

        run_name = (str(n) + variance_suffix) if variance_suffix else str(n)
        run_dir_this = os.path.join(run_dir, "analysis", "q", "gen", run_name)
        print(f"  Q (max Q): n={n}, variance={sample_variance} (merged from {len(acc_gen_to_train)} chunk(s))...")
        path = _run_one_q(
            run_dir_this, test_to_train_max_q, gen_to_train_max_q, gen_to_test_max_q,
            gen_coords_np if save_structures_gro else None,
            plot_cfg_resume, display_root, write_data, save_structures_gro,
        )
        out_paths.append(path)
        if not save_data:
            data_dirs_to_remove.append(os.path.join(run_dir_this, "data"))

    if not save_data and data_dirs_to_remove:
        for d in data_dirs_to_remove:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    return out_paths


def _recon_q_one_to_one(original_coords: np.ndarray, recon_coords: np.ndarray, delta: float) -> np.ndarray:
    """Q between each original and its reconstruction (same index). Returns (n,) array."""
    if original_coords.shape[0] != recon_coords.shape[0]:
        raise ValueError("original_coords and recon_coords must have same number of structures")
    n = original_coords.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = q_single(original_coords[i], recon_coords[i], delta)
    return out


def _run_one_q_recon(
    run_dir_recon: str,
    test_to_train_max_q: np.ndarray,
    train_recon_q: np.ndarray,
    test_recon_q: np.ndarray,
    plot_cfg: dict,
    display_root: str | None,
    save_data: bool,
) -> str:
    """Write recon Q figure (test→train max Q, train recon Q, test recon Q) and optional data. Returns path to figure."""
    dpi = plot_cfg["plot_dpi"]
    all_vals = np.concatenate([test_to_train_max_q, train_recon_q, test_recon_q])
    x_min = max(0.0, np.percentile(all_vals, 0.5) - 0.02)
    x_max = min(1.0, np.percentile(all_vals, 99.5) + 0.02)
    bins = np.linspace(x_min, x_max, 50)

    c0, c1, c2 = RECON_PANEL_COLORS
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].hist(test_to_train_max_q, bins=bins, density=True, alpha=0.7, color=c0, edgecolor="k", linewidth=0.3)
    axes[0].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[0].set_title("Test → Train (Max Q to Train Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[0].set_xlim(x_min, x_max)
    axes[1].hist(train_recon_q, bins=bins, density=True, alpha=0.7, color=c1, edgecolor="k", linewidth=0.3)
    axes[1].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[1].set_title("Train Recon (Q to Original)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[1].set_xlim(x_min, x_max)
    axes[2].hist(test_recon_q, bins=bins, density=True, alpha=0.7, color=c2, edgecolor="k", linewidth=0.3)
    axes[2].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_xlabel("Max Q", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_title("Test Recon (Q to Original)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[2].set_xlim(x_min, x_max)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)
    plt.tight_layout()
    os.makedirs(run_dir_recon, exist_ok=True)
    out_path = os.path.join(run_dir_recon, "q_distributions.png")
    plt.savefig(out_path, dpi=dpi)
    if plot_cfg["save_pdf_copy"]:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close()

    if save_data:
        data_dir = os.path.join(run_dir_recon, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "q_recon_data.npz")
        np.savez_compressed(
            data_path,
            train_recon_q=train_recon_q,
            test_recon_q=test_recon_q,
            bins=bins,
        )
        print(f"  Saved: {display_path(data_path, display_root)}")

    print(f"  Saved: {display_path(out_path, display_root)}")
    return out_path


def run_q_recon_analysis(
    test_to_train_max_q: np.ndarray,
    train_coords_np: np.ndarray,
    test_coords_np: np.ndarray,
    train_recon_coords_np: np.ndarray,
    test_recon_coords_np: np.ndarray,
    run_dir: str,
    plot_cfg: dict,
    delta: float,
    display_root: str | None = None,
    recon_subdir: str = "",
) -> str:
    """
    Compute recon Q figure: test→train (max Q, reused), train recon Q (one-to-one), test recon Q (one-to-one).
    Saves to run_dir/analysis/q/recon[/recon_subdir]/q_distributions.png and optional data/.
    """
    run_dir_recon = os.path.join(run_dir, "analysis", "q", "recon", recon_subdir) if recon_subdir else os.path.join(run_dir, "analysis", "q", "recon")
    save_data = plot_cfg["save_data"]
    n_train = min(train_coords_np.shape[0], train_recon_coords_np.shape[0])
    n_test = min(test_coords_np.shape[0], test_recon_coords_np.shape[0])
    train_coords_np = train_coords_np[:n_train]
    train_recon_coords_np = train_recon_coords_np[:n_train]
    test_coords_np = test_coords_np[:n_test]
    test_recon_coords_np = test_recon_coords_np[:n_test]
    print(f"  Q recon: train {n_train}, test {n_test} (test→train max Q + recon Q)...")
    train_recon_q = _recon_q_one_to_one(train_coords_np, train_recon_coords_np, delta)
    test_recon_q = _recon_q_one_to_one(test_coords_np, test_recon_coords_np, delta)
    return _run_one_q_recon(
        run_dir_recon, test_to_train_max_q, train_recon_q, test_recon_q,
        plot_cfg, display_root, save_data,
    )
