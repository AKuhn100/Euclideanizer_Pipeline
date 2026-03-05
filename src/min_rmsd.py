"""
Min-RMSD distributions: integrate into pipeline for each Euclideanizer run.
Uses Kabsch alignment; computes test→train, gen→train, gen→test min-RMSD and saves a figure.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def _rmsd_matrix_batch(queries: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
    """
    Vectorized: (B, N, 3) queries vs (M, N, 3) refs -> (B, M) RMSDs after Kabsch.
    """
    B, N, _ = queries.shape
    M = ref_coords.shape[0]
    q_c = queries - queries.mean(axis=1, keepdims=True)
    r_c = ref_coords - ref_coords.mean(axis=1, keepdims=True)
    ref_means = ref_coords.mean(axis=1)

    H = np.einsum("bni,mnj->bmij", q_c, r_c)
    H_flat = H.reshape(B * M, 3, 3)

    U, _, Vt = np.linalg.svd(H_flat)
    d = np.linalg.det(U @ Vt)
    S_diag = np.ones((B * M, 3), dtype=H_flat.dtype)
    S_diag[:, 2] = np.sign(d)
    R = np.einsum("mki,mk,mjl->mij", Vt, S_diag, U)

    rmsd_out = np.empty((B, M), dtype=queries.dtype)
    for b in range(B):
        R_b = R[b * M : (b + 1) * M]
        aligned_b = np.einsum("nj,mji->mni", q_c[b], R_b) + ref_means[:, np.newaxis, :]
        rmsd_out[b] = np.sqrt(np.mean((aligned_b - ref_coords) ** 2, axis=(1, 2)))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test→train min-RMSD (and train/test coords) from seed-level cache, or compute and save.
    Same for all analyses in the seed (same split). Returns (test_to_train, train_coords_np, test_coords_np).
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
            if display_root:
                try:
                    print(f"  Loaded seed-level test→train RMSD cache: {os.path.relpath(cache_path, display_root)}")
                except ValueError:
                    print(f"  Loaded seed-level test→train RMSD cache: {cache_path}")
            return out
        except Exception:
            pass
    coords = coords_tensor
    train_size = int(training_split * len(coords))
    test_size = len(coords) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_ds, test_ds = random_split(coords, [train_size, test_size], generator=generator)
    tr_idx = train_ds.indices
    te_idx = test_ds.indices
    if hasattr(tr_idx, "tolist"):
        tr_idx, te_idx = tr_idx.tolist(), te_idx.tolist()
    train_coords_np = coords_np[tr_idx]
    test_coords_np = coords_np[te_idx]
    test_to_train = min_rmsd_batch(
        test_coords_np, train_coords_np, query_batch_size=query_batch_size, desc="Test → Train (min RMSD)"
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        test_to_train=test_to_train,
        train_coords_np=train_coords_np,
        test_coords_np=test_coords_np,
    )
    if display_root:
        try:
            print(f"  Saved seed-level test→train RMSD cache: {os.path.relpath(cache_path, display_root)}")
        except ValueError:
            print(f"  Saved seed-level test→train RMSD cache: {cache_path}")
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
    dpi = plot_cfg.get("plot_dpi", 150)
    all_vals = np.concatenate([test_to_train, gen_to_train, gen_to_test])
    x_min = max(0.0, np.percentile(all_vals, 0.5) - 0.5)
    x_max = np.percentile(all_vals, 99.5) + 0.5
    bins = np.linspace(x_min, x_max, 50)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].hist(test_to_train, bins=bins, density=True, alpha=0.7, color="C0", edgecolor="k", linewidth=0.3)
    axes[0].set_ylabel("Density")
    axes[0].set_title("Test → Train (min RMSD to training set)")
    axes[0].set_xlim(x_min, x_max)
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(gen_to_train, bins=bins, density=True, alpha=0.7, color="C1", edgecolor="k", linewidth=0.3)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Generated → Train (min RMSD to training set)")
    axes[1].set_xlim(x_min, x_max)
    axes[1].grid(True, alpha=0.3)
    axes[2].hist(gen_to_test, bins=bins, density=True, alpha=0.7, color="C2", edgecolor="k", linewidth=0.3)
    axes[2].set_ylabel("Density")
    axes[2].set_xlabel("Min RMSD (aligned coords)")
    axes[2].set_title("Generated → Test (min RMSD to test set)")
    axes[2].set_xlim(x_min, x_max)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(run_dir_this, exist_ok=True)
    out_path = os.path.join(run_dir_this, "min_rmsd_distributions.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    if save_data:
        data_dir = os.path.join(run_dir_this, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "min_rmsd_data.npz")
        save_kw: dict = dict(
            test_to_train=test_to_train,
            gen_to_train=gen_to_train,
            gen_to_test=gen_to_test,
            bins=bins,
        )
        if gen_coords_np is not None and plot_cfg.get("save_gen_coords_in_npz", False):
            save_kw["gen_coords"] = gen_coords_np
        np.savez_compressed(data_path, **save_kw)
        try:
            print(f"  Saved: {os.path.relpath(data_path, display_root) if display_root else data_path}")
        except ValueError:
            print(f"  Saved: {data_path}")

    if save_structures_gro and gen_coords_np is not None:
        from .gro_io import write_structures_gro
        structures_dir = os.path.join(run_dir_this, "structures")
        write_structures_gro(gen_coords_np, structures_dir, display_root=display_root)

    try:
        print(f"  Saved: {os.path.relpath(out_path, display_root) if display_root else out_path}")
    except ValueError:
        print(f"  Saved: {out_path}")
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
    Compute min-RMSD distributions (test→train, gen→train, gen→test). Saves to run_dir/analysis/min_rmsd/<run_name>/
    with: min_rmsd_distributions.png, data/ (if save_data), structures/ (if save_structures_gro).
    When precomputed_test_to_train, train_coords_np, test_coords_np are provided (e.g. from seed-level cache), they are reused.
    """
    run_name = output_suffix.lstrip("_") if output_suffix else "default"
    run_dir_this = os.path.join(run_dir, "analysis", "min_rmsd", run_name)
    save_data = plot_cfg.get("save_data", plot_cfg.get("save_plot_data", False))
    save_structures_gro = plot_cfg.get("save_structures_gro", False)

    n_gen = num_samples if num_samples is not None else plot_cfg.get("min_rmsd_num_samples", 500)
    variance = sample_variance if sample_variance is not None else plot_cfg.get("min_rmsd_sample_variance", 1.0)
    batch_size = query_batch_size if query_batch_size is not None else plot_cfg.get("min_rmsd_query_batch_size", 128)
    print(f"  Min-RMSD: n_gen={n_gen}, variance={variance} (test→train, gen→train, gen→test)...")

    if precomputed_test_to_train is not None and train_coords_np is not None and test_coords_np is not None:
        test_to_train = precomputed_test_to_train
    else:
        coords = coords_tensor
        train_size = int(training_split * len(coords))
        test_size = len(coords) - train_size
        generator = torch.Generator().manual_seed(split_seed)
        train_ds, test_ds = random_split(coords, [train_size, test_size], generator=generator)
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
    embed.eval()
    with torch.no_grad():
        z = generate_samples(n_gen, latent_dim, device, variance=variance)
        D_noneuclid = frozen_vae._decode_to_matrix(z)
        gen_coords = embed(D_noneuclid)
    gen_coords_np = gen_coords.cpu().numpy().astype(np.float32)

    gen_to_train = min_rmsd_batch(
        gen_coords_np, train_coords_np, query_batch_size=batch_size, desc="Generated → Train (min RMSD)"
    )
    gen_to_test = min_rmsd_batch(
        gen_coords_np, test_coords_np, query_batch_size=batch_size, desc="Generated → Test (min RMSD)"
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

    batch_size = query_batch_size or plot_cfg.get("min_rmsd_query_batch_size", 128)
    save_data = plot_cfg.get("save_data", plot_cfg.get("save_plot_data", False))
    save_structures_gro = plot_cfg.get("save_structures_gro", False)
    sorted_n = sorted(set(num_samples_list))
    if not sorted_n:
        return []

    data_dirs_to_remove: list[str] = []
    write_data = True

    if precomputed_test_to_train is not None and train_coords_np is not None and test_coords_np is not None:
        test_to_train = precomputed_test_to_train
    else:
        coords = coords_tensor
        train_size = int(training_split * len(coords))
        test_size = len(coords) - train_size
        generator = torch.Generator().manual_seed(split_seed)
        train_ds, test_ds = random_split(coords, [train_size, test_size], generator=generator)
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
        run_dir_n = os.path.join(run_dir, "analysis", "min_rmsd", run_name)
        fig_p = os.path.join(run_dir_n, "min_rmsd_distributions.png")
        data_p = os.path.join(run_dir_n, "data", "min_rmsd_data.npz")
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
                print(f"  Min-RMSD: resuming from n={n} (loaded {loaded_n} samples), generating remaining...")
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
                D_noneuclid = frozen_vae._decode_to_matrix(z)
                gen_coords = embed(D_noneuclid)
            gen_chunk_np = gen_coords.cpu().numpy().astype(np.float32)
            acc_gen_coords.append(gen_chunk_np)
            gen_to_train_chunk = min_rmsd_batch(
                gen_chunk_np, train_coords_np, query_batch_size=batch_size,
                desc=f"Generated → Train (min RMSD) +{need}",
            )
            gen_to_test_chunk = min_rmsd_batch(
                gen_chunk_np, test_coords_np, query_batch_size=batch_size,
                desc=f"Generated → Test (min RMSD) +{need}",
            )
            acc_gen_to_train.append(gen_to_train_chunk)
            acc_gen_to_test.append(gen_to_test_chunk)
            gen_to_train = np.concatenate(acc_gen_to_train, axis=0)
            gen_to_test = np.concatenate(acc_gen_to_test, axis=0)
            gen_coords_np = np.concatenate(acc_gen_coords, axis=0)

        run_name = (str(n) + variance_suffix) if variance_suffix else str(n)
        run_dir_this = os.path.join(run_dir, "analysis", "min_rmsd", run_name)
        print(f"  Min-RMSD: n={n}, variance={sample_variance} (merged from {len(acc_gen_to_train)} chunk(s))...")
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
