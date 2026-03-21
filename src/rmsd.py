"""
RMSD distributions: integrate into pipeline for each Euclideanizer run.
Uses Kabsch alignment; computes test→train, gen→train, gen→test min-RMSD and saves a figure.
Output under analysis/rmsd/; figure axis labels describe the metric (e.g. min RMSD to train set).
"""
from __future__ import annotations

import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
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
    HIST_BINS_DEFAULT,
    HIST_FILLED_EDGE_COLOR,
)


def _rmsd_matrix_batch(queries: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
    """
    (B, N, 3) queries vs (M, N, 3) refs -> (B, M) RMSDs after Kabsch alignment.
    Uses scipy.spatial.transform.Rotation.align_vectors (Kabsch) for correctness.
    """
    B, N, _ = queries.shape
    M = ref_coords.shape[0]
    q_c = queries - queries.mean(axis=1, keepdims=True)
    r_c = ref_coords - ref_coords.mean(axis=1, keepdims=True)
    ref_means = ref_coords.mean(axis=1)

    rmsd_out = np.empty((B, M), dtype=queries.dtype)
    for b in range(B):
        for m in range(M):
            rot, _ = Rotation.align_vectors(r_c[m], q_c[b])  # R @ query_centered = ref_centered
            aligned = rot.apply(q_c[b]) + ref_means[m]
            rmsd_out[b, m] = np.sqrt(np.mean((aligned - ref_coords[m]) ** 2))
    return rmsd_out


def min_rmsd_batch(
    queries: np.ndarray,
    ref_coords: np.ndarray,
    query_batch_size: int = 128,
    desc: str | None = None,
) -> np.ndarray:
    """
    For each of `queries` (Q, N, 3), return min RMSD to any of `ref_coords` (M, N, 3).
    Returns (Q,) array.
    """
    Q = queries.shape[0]
    out = np.empty(Q, dtype=queries.dtype)
    batch_starts = range(0, Q, query_batch_size)
    for start in tqdm(batch_starts, desc=desc or "min-RMSD", unit="batch", leave=True):
        end = min(start + query_batch_size, Q)
        batch = queries[start:end]
        rmsd_mat = _rmsd_matrix_batch(batch, ref_coords)
        out[start:end] = rmsd_mat.min(axis=1)
    return out


def get_or_compute_test_to_train_rmsd(
    coords_np: np.ndarray,
    coords_tensor: torch.Tensor,
    training_split: float,
    split_seed: int,
    cache_path: str,
    query_batch_size: int = 128,
    display_root: str | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test→train min-RMSD (and train/test coords) from seed-level cache, or compute and save.
    Same for all analyses in the seed (same split). Returns (test_to_train, train_coords_np, test_coords_np).
    max_train/max_test cap the reference set sizes (None = use all).

    The seed-level cache is always written when computed (independent of the analysis block's save_data).
    Per-run outputs (rmsd_data.npz, rmsd_recon_data.npz) are still gated by save_data.
    """
    if os.path.isfile(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            out = (
                np.asarray(loaded["test_to_train"], dtype=np.float32),
                np.asarray(loaded["train_coords_np"], dtype=np.float32),
                np.asarray(loaded["test_coords_np"], dtype=np.float32),
            )
            loaded.close()
            if display_root is not None:
                print(f"  Loaded seed-level test→train RMSD cache: {display_path(cache_path, display_root)}")
            return out
        except Exception:
            pass
    coords = coords_tensor
    train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
    tr_idx = train_ds.indices
    te_idx = test_ds.indices
    if hasattr(tr_idx, "tolist"):
        tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
    train_coords_np = coords_np[tr_idx]
    test_coords_np = coords_np[te_idx]
    # Truncate to reference set size; slice uses at most available (no error if max_* > len)
    if max_train is not None:
        train_coords_np = train_coords_np[:max_train]
    if max_test is not None:
        test_coords_np = test_coords_np[:max_test]
    test_to_train = min_rmsd_batch(
        test_coords_np, train_coords_np, query_batch_size=query_batch_size, desc="Test → Train (min RMSD)"
    )
    # Always save when computed (independent of analysis save_data); reused for all runs in this seed.
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        test_to_train=test_to_train,
        train_coords_np=train_coords_np,
        test_coords_np=test_coords_np,
    )
    if display_root is not None:
        print(f"  Saved seed-level test→train RMSD cache: {display_path(cache_path, display_root)}")
    return test_to_train, train_coords_np, test_coords_np


def _run_one_min_rmsd(
    run_dir_this: str,
    test_to_train: np.ndarray,
    gen_to_train: np.ndarray,
    gen_to_test: np.ndarray,
    gen_coords_np: np.ndarray | None,
    plot_cfg: dict,
    display_root: str | None,
    save_data: bool,
    save_structures_gro: bool,
) -> str:
    """Write figure, optional data/, optional structures/ for one run. Returns path to figure."""
    dpi = plot_cfg["plot_dpi"]
    all_vals = np.concatenate([test_to_train, gen_to_train, gen_to_test])
    x_min = max(0.0, np.percentile(all_vals, 0.5) - 0.5)
    x_max = np.percentile(all_vals, 99.5) + 0.5
    bins = np.linspace(x_min, x_max, HIST_BINS_DEFAULT + 1)

    c0, c1, c2 = GEN_PANEL_COLORS
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].hist(test_to_train, bins=bins, density=True, alpha=0.7, color=c0, edgecolor=HIST_FILLED_EDGE_COLOR)
    axes[0].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[0].set_title("Test → Train (Min RMSD to Train Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[0].set_xlim(x_min, x_max)
    axes[1].hist(gen_to_train, bins=bins, density=True, alpha=0.7, color=c1, edgecolor=HIST_FILLED_EDGE_COLOR)
    axes[1].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[1].set_title("Gen → Train (Min RMSD to Train Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[1].set_xlim(x_min, x_max)
    axes[2].hist(gen_to_test, bins=bins, density=True, alpha=0.7, color=c2, edgecolor=HIST_FILLED_EDGE_COLOR)
    axes[2].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_xlabel("Min RMSD (Aligned Coords)", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_title("Gen → Test (Min RMSD to Test Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[2].set_xlim(x_min, x_max)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)
    plt.tight_layout()
    os.makedirs(run_dir_this, exist_ok=True)
    out_path = os.path.join(run_dir_this, "rmsd_distributions.png")
    plt.savefig(out_path, dpi=dpi)
    if plot_cfg["save_pdf_copy"]:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close()

    if save_data:
        data_dir = os.path.join(run_dir_this, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "rmsd_data.npz")
        # test_to_train is stored at seed level (experimental_statistics/test_to_train_rmsd.npz), not duplicated here
        save_kw: dict = dict(
            gen_to_train=gen_to_train,
            gen_to_test=gen_to_test,
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


def run_min_rmsd_analysis(
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
    query_batch_size: int | None = None,
    output_suffix: str = "",
    display_root: str | None = None,
    precomputed_test_to_train: np.ndarray | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
) -> str:
    """
    Compute min-RMSD distributions (test→train, gen→train, gen→test). Saves to run_dir/analysis/rmsd/<run_name>/
    with: rmsd_distributions.png, data/ (if save_data), structures/ (if save_structures_gro).
    When precomputed_test_to_train, train_coords_np, test_coords_np are provided (e.g. from seed-level cache), they are reused.
    """
    run_name = output_suffix.lstrip("_") if output_suffix else "default"
    run_dir_this = os.path.join(run_dir, "analysis", "rmsd", "gen", run_name)
    save_data = plot_cfg["save_data"]
    save_structures_gro = plot_cfg["save_structures_gro"]

    _n = plot_cfg["rmsd_num_samples"]
    n_gen = num_samples if num_samples is not None else (_n[0] if isinstance(_n, list) else _n)
    _v = plot_cfg["rmsd_sample_variance"]
    variance = sample_variance if sample_variance is not None else (_v[0] if isinstance(_v, list) else _v)
    batch_size = query_batch_size if query_batch_size is not None else plot_cfg["rmsd_query_batch_size"]
    print(f"  RMSD: n_gen={n_gen}, variance={variance} (test→train, gen→train, gen→test)...")

    if precomputed_test_to_train is not None and train_coords_np is not None and test_coords_np is not None:
        test_to_train = precomputed_test_to_train
    else:
        coords = coords_tensor
        train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
        tr_idx = train_ds.indices
        te_idx = test_ds.indices
        if hasattr(tr_idx, "tolist"):
            tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
        train_coords_np = coords_np[tr_idx]
        test_coords_np = coords_np[te_idx]
        test_to_train = min_rmsd_batch(
            test_coords_np, train_coords_np, query_batch_size=batch_size, desc="Test → Train (min RMSD)"
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

    gen_to_train = min_rmsd_batch(
        gen_coords_np, train_coords_np, query_batch_size=batch_size, desc="Gen → Train (min RMSD)"
    )
    gen_to_test = min_rmsd_batch(
        gen_coords_np, test_coords_np, query_batch_size=batch_size, desc="Gen → Test (min RMSD)"
    )

    return _run_one_min_rmsd(
        run_dir_this, test_to_train, gen_to_train, gen_to_test,
        gen_coords_np if save_structures_gro else None,
        plot_cfg, display_root, save_data, save_structures_gro,
    )


def run_min_rmsd_analysis_multi(
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
    query_batch_size: int | None = None,
    variance_suffix: str = "",
    display_root: str | None = None,
    precomputed_test_to_train: np.ndarray | None = None,
    train_coords_np: np.ndarray | None = None,
    test_coords_np: np.ndarray | None = None,
) -> list[str]:
    """
    Run min-RMSD for multiple num_samples with the same variance, using incremental generation
    and merging. When precomputed_test_to_train, train_coords_np, test_coords_np are provided
    (e.g. from seed-level cache), they are reused.
    """
    from .distmap.sample import generate_samples
    import shutil

    batch_size = query_batch_size or plot_cfg["rmsd_query_batch_size"]
    gen_decode_batch_size = plot_cfg["gen_decode_batch_size"]
    save_data = plot_cfg["save_data"]
    save_structures_gro = plot_cfg["save_structures_gro"]
    sorted_n = sorted(set(num_samples_list))
    if not sorted_n:
        return []

    data_dirs_to_remove: list[str] = []
    write_data = True

    if precomputed_test_to_train is not None and train_coords_np is not None and test_coords_np is not None:
        test_to_train = precomputed_test_to_train
    else:
        coords = coords_tensor
        train_ds, test_ds = get_train_test_split(coords, training_split, split_seed)
        tr_idx = train_ds.indices
        te_idx = test_ds.indices
        if hasattr(tr_idx, "tolist"):
            tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
        train_coords_np = coords_np[tr_idx]
        test_coords_np = coords_np[te_idx]
        test_to_train = min_rmsd_batch(
            test_coords_np, train_coords_np, query_batch_size=batch_size, desc="Test → Train (min RMSD)"
        )

    embed.eval()
    acc_gen_to_train: list[np.ndarray] = []
    acc_gen_to_test: list[np.ndarray] = []
    acc_gen_coords: list[np.ndarray] = []
    out_paths: list[str] = []

    # Resume: find largest n that has both figure and data, load and restore acc_*
    plot_cfg_resume = dict(plot_cfg)
    if save_structures_gro:
        plot_cfg_resume["save_gen_coords_in_npz"] = True
    loaded_n = 0
    for n in reversed(sorted_n):
        run_name = (str(n) + variance_suffix) if variance_suffix else str(n)
        run_dir_n = os.path.join(run_dir, "analysis", "rmsd", "gen", run_name)
        fig_p = os.path.join(run_dir_n, "rmsd_distributions.png")
        data_p = os.path.join(run_dir_n, "data", "rmsd_data.npz")
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
                print(f"  RMSD: resuming from n={n} (loaded {loaded_n} samples), generating remaining...")
            except Exception:
                pass
            break

    for n in sorted_n:
        need = n - sum(a.shape[0] for a in acc_gen_to_train)
        if need <= 0:
            gen_to_train = np.concatenate(acc_gen_to_train, axis=0) if acc_gen_to_train else np.array([], dtype=np.float32)
            gen_to_test = np.concatenate(acc_gen_to_test, axis=0) if acc_gen_to_test else np.array([], dtype=np.float32)
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
            gen_to_train_chunk = min_rmsd_batch(
                gen_chunk_np, train_coords_np, query_batch_size=batch_size,
                desc=f"Gen → Train (min RMSD) +{need}",
            )
            gen_to_test_chunk = min_rmsd_batch(
                gen_chunk_np, test_coords_np, query_batch_size=batch_size,
                desc=f"Gen → Test (min RMSD) +{need}",
            )
            acc_gen_to_train.append(gen_to_train_chunk)
            acc_gen_to_test.append(gen_to_test_chunk)
            gen_to_train = np.concatenate(acc_gen_to_train, axis=0)
            gen_to_test = np.concatenate(acc_gen_to_test, axis=0)
            gen_coords_np = np.concatenate(acc_gen_coords, axis=0)

        run_name = (str(n) + variance_suffix) if variance_suffix else str(n)
        run_dir_this = os.path.join(run_dir, "analysis", "rmsd", "gen", run_name)
        print(f"  RMSD: n={n}, variance={sample_variance} (merged from {len(acc_gen_to_train)} chunk(s))...")
        path = _run_one_min_rmsd(
            run_dir_this, test_to_train, gen_to_train, gen_to_test,
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


def _recon_rmsd_one_to_one(original_coords: np.ndarray, recon_coords: np.ndarray) -> np.ndarray:
    """RMSD between each original and its reconstruction (same index). Returns (n,) array. Uses Kabsch via _rmsd_matrix_batch diagonal."""
    if original_coords.shape[0] != recon_coords.shape[0]:
        raise ValueError("original_coords and recon_coords must have same number of structures")
    rmsd_mat = _rmsd_matrix_batch(recon_coords, original_coords)
    return np.diag(rmsd_mat).astype(np.float32)


def _run_one_min_rmsd_recon(
    run_dir_recon: str,
    test_to_train: np.ndarray,
    train_recon_rmsd: np.ndarray,
    test_recon_rmsd: np.ndarray,
    plot_cfg: dict,
    display_root: str | None,
    save_data: bool,
) -> str:
    """Write recon min-RMSD figure (test→train, train recon, test recon) and optional data. Returns path to figure."""
    dpi = plot_cfg["plot_dpi"]
    all_vals = np.concatenate([test_to_train, train_recon_rmsd, test_recon_rmsd])
    x_min = max(0.0, np.percentile(all_vals, 0.5) - 0.5)
    x_max = np.percentile(all_vals, 99.5) + 0.5
    bins = np.linspace(x_min, x_max, HIST_BINS_DEFAULT + 1)

    c0, c1, c2 = RECON_PANEL_COLORS
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].hist(test_to_train, bins=bins, density=True, alpha=0.7, color=c0, edgecolor=HIST_FILLED_EDGE_COLOR)
    axes[0].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[0].set_title("Test → Train (Min RMSD to Train Set)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[0].set_xlim(x_min, x_max)
    axes[1].hist(train_recon_rmsd, bins=bins, density=True, alpha=0.7, color=c1, edgecolor=HIST_FILLED_EDGE_COLOR)
    axes[1].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[1].set_title("Train Recon (Aligned RMSD to Original)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[1].set_xlim(x_min, x_max)
    axes[2].hist(test_recon_rmsd, bins=bins, density=True, alpha=0.7, color=c2, edgecolor=HIST_FILLED_EDGE_COLOR)
    axes[2].set_ylabel("Density", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_xlabel("Min RMSD (Aligned Coords)", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    axes[2].set_title("Test Recon (Aligned RMSD to Original)", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    axes[2].set_xlim(x_min, x_max)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)
    plt.tight_layout()
    os.makedirs(run_dir_recon, exist_ok=True)
    out_path = os.path.join(run_dir_recon, "rmsd_distributions.png")
    plt.savefig(out_path, dpi=dpi)
    if plot_cfg["save_pdf_copy"]:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close()

    if save_data:
        data_dir = os.path.join(run_dir_recon, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "rmsd_recon_data.npz")
        # test_to_train is at seed level (experimental_statistics/test_to_train_rmsd.npz), not duplicated here
        np.savez_compressed(
            data_path,
            train_recon_rmsd=train_recon_rmsd,
            test_recon_rmsd=test_recon_rmsd,
            bins=bins,
        )
        print(f"  Saved: {display_path(data_path, display_root)}")

    print(f"  Saved: {display_path(out_path, display_root)}")
    return out_path


def run_min_rmsd_recon_analysis(
    test_to_train: np.ndarray,
    train_coords_np: np.ndarray,
    test_coords_np: np.ndarray,
    train_recon_coords_np: np.ndarray,
    test_recon_coords_np: np.ndarray,
    run_dir: str,
    plot_cfg: dict,
    display_root: str | None = None,
    recon_subdir: str = "",
) -> str:
    """
    Compute recon min-RMSD figure: test→train (reused), train recon RMSD, test recon RMSD.
    Saves to run_dir/analysis/rmsd/recon[/recon_subdir]/rmsd_distributions.png and optional data/.
    When recon_subdir is non-empty (e.g. "train100_test50"), outputs go under recon/recon_subdir/.
    """
    run_dir_recon = os.path.join(run_dir, "analysis", "rmsd", "recon", recon_subdir) if recon_subdir else os.path.join(run_dir, "analysis", "rmsd", "recon")
    save_data = plot_cfg["save_data"]
    # Ensure we have the same count (cap may have been applied to recon)
    n_train = min(train_coords_np.shape[0], train_recon_coords_np.shape[0])
    n_test = min(test_coords_np.shape[0], test_recon_coords_np.shape[0])
    train_coords_np = train_coords_np[:n_train]
    train_recon_coords_np = train_recon_coords_np[:n_train]
    test_coords_np = test_coords_np[:n_test]
    test_recon_coords_np = test_recon_coords_np[:n_test]
    print(f"  RMSD recon: train {n_train}, test {n_test} (test→train + recon RMSD)...")
    train_recon_rmsd = _recon_rmsd_one_to_one(train_coords_np, train_recon_coords_np)
    test_recon_rmsd = _recon_rmsd_one_to_one(test_coords_np, test_recon_coords_np)
    return _run_one_min_rmsd_recon(
        run_dir_recon, test_to_train, train_recon_rmsd, test_recon_rmsd,
        plot_cfg, display_root, save_data,
    )
