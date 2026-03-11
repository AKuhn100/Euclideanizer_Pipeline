"""
Plotting: reconstruction (test set), bond/Rg/scaling, avg gen vs exp + difference map.
Used for both DistMap and Euclideanizer outputs.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .metrics import distmap_bond_lengths, distmap_distances_at_lag, distmap_rg, distmap_scaling
from .utils import display_path, get_train_test_split


def _save_pdf_copy(fig, png_path: str, save_pdf: bool = True, display_root: str | None = None) -> None:
    """If save_pdf is True, save a PDF version of the figure in a pdf/ subdir next to the PNG."""
    if not save_pdf:
        return
    pdf_dir = os.path.join(os.path.dirname(png_path), "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, os.path.splitext(os.path.basename(png_path))[0] + ".pdf")
    fig.savefig(pdf_path)
    print(f"  Saved: {display_path(pdf_path, display_root)}")


def _save_plot_data_npz(output_path: str, display_root: str | None = None, **arrays: np.ndarray) -> None:
    """Save plot data in a data/ subdir of the plot directory: .../plots/<type>/data/<stem>_data.npz."""
    plot_dir = os.path.dirname(output_path)
    stem = os.path.splitext(os.path.basename(output_path))[0]
    data_dir = os.path.join(plot_dir, "data")
    data_path = os.path.join(data_dir, stem + "_data.npz")
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(data_path, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"  Saved: {display_path(data_path, display_root)}")


# -------- DistMap: reconstruction (test set) --------
def plot_distmap_reconstruction(
    model,
    device: torch.device,
    coords: torch.Tensor,
    utils_mod,
    output_path: str,
    *,
    training_split: float = 0.8,
    split_seed: int = 42,
    batch_size: int = 256,
    num_to_plot: int = 5,
    dpi: int = 150,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    display_root: str | None = None,
) -> None:
    """Original vs reconstructed distance maps (test set)."""
    _, test_dataset = get_train_test_split(coords, training_split, split_seed)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_atoms = coords.size(1)
    model.eval()
    original_dms, recon_dms = [], []
    with torch.no_grad():
        count = 0
        for batch in test_dl:
            if count >= num_to_plot:
                break
            batch_dm = utils_mod.get_distmaps(batch)
            mu, logvar, z, recon_tri = model(batch_dm)
            recon_tri = torch.expm1(recon_tri)
            recon_full = utils_mod.upper_tri_to_symmetric(recon_tri, num_atoms)
            for i in range(min(batch_dm.size(0), num_to_plot - count)):
                original_dms.append(batch_dm[i].cpu().numpy())
                recon_dms.append(recon_full[i].cpu().numpy())
                count += 1
    n_actual = len(original_dms)
    if n_actual == 0:
        return
    n_plot = n_actual
    vmin = min(dm.min() for dm in original_dms + recon_dms)
    vmax = max(dm.max() for dm in original_dms + recon_dms)
    fig, axes = plt.subplots(n_plot, 2, figsize=(10, 5 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_plot):
        axes[i, 0].imshow(original_dms[i], cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax, interpolation="none")
        axes[i, 0].set_title(f"Original Test {i+1}")
        axes[i, 1].imshow(recon_dms[i], cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax, interpolation="none")
        axes[i, 1].set_title(f"Reconstructed {i+1}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    _save_pdf_copy(fig, output_path, save_pdf, display_root=display_root)
    if save_plot_data:
        _save_plot_data_npz(output_path, display_root=display_root, original_dms=np.array(original_dms), recon_dms=np.array(recon_dms))
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


# -------- Euclideanizer: reconstruction (test set) --------
def plot_euclideanizer_reconstruction(
    embed,
    frozen_vae,
    device: torch.device,
    coords: torch.Tensor,
    utils_mod,
    output_path: str,
    *,
    training_split: float = 0.8,
    split_seed: int = 42,
    batch_size: int = 32,
    num_to_plot: int = 5,
    dpi: int = 150,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    display_root: str | None = None,
) -> None:
    """Original, VAE (non-Eucl.), Euclideanizer (test set)."""
    _, test_dataset = get_train_test_split(coords, training_split, split_seed)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    embed.eval()
    orig_list, vae_list, euclid_list = [], [], []
    with torch.no_grad():
        count = 0
        for batch in test_dl:
            if count >= num_to_plot:
                break
            batch_dm = utils_mod.get_distmaps(batch)
            gt_log = torch.log1p(batch_dm)
            mu = frozen_vae.encode(gt_log)
            D_noneuclid = frozen_vae._decode_to_matrix(mu)
            D_euclid = embed.forward_to_distmap(D_noneuclid)
            D_vae_lin = torch.expm1(D_noneuclid)
            D_euc_lin = torch.expm1(D_euclid)
            for i in range(min(batch_dm.size(0), num_to_plot - count)):
                orig_list.append(batch_dm[i].cpu().numpy())
                vae_list.append(D_vae_lin[i].cpu().numpy())
                euclid_list.append(D_euc_lin[i].cpu().numpy())
                count += 1
    n_actual = len(orig_list)
    if n_actual == 0:
        return
    n_plot = n_actual
    vmin = min(d.min() for d in orig_list + vae_list + euclid_list)
    vmax = max(d.max() for d in orig_list + vae_list + euclid_list)
    fig, axes = plt.subplots(n_plot, 3, figsize=(12, 5 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_plot):
        for j, (data, title) in enumerate([
            (orig_list[i], f"Original {i+1}"),
            (vae_list[i], f"VAE (non-Eucl.) {i+1}"),
            (euclid_list[i], f"Euclideanizer {i+1}"),
        ]):
            axes[i, j].imshow(data, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax, interpolation="none")
            axes[i, j].set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    _save_pdf_copy(fig, output_path, save_pdf, display_root=display_root)
    if save_plot_data:
        _save_plot_data_npz(output_path, display_root=display_root, original=np.array(orig_list), vae=np.array(vae_list), euclideanizer=np.array(euclid_list))
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


# -------- Bond length, Rg, scaling (reconstruction statistics) --------
def plot_recon_statistics(
    recon_dm: np.ndarray,
    exp_stats: dict,
    output_path: str,
    *,
    label_recon: str = "Recon",
    subset_label: str | None = None,
    dpi: int = 150,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    display_root: str | None = None,
) -> None:
    """Bond lengths, Rg, scaling: exp (same subset) vs recon. subset_label e.g. 'test' or 'train' for title."""
    true_bonds = exp_stats["exp_bonds"]
    true_rg = exp_stats["exp_rg"]
    s = exp_stats["genomic_distances"]
    true_sc = exp_stats["exp_scaling"]
    recon_bonds = distmap_bond_lengths(recon_dm)
    recon_rg = distmap_rg(recon_dm)
    _, recon_sc = distmap_scaling(recon_dm)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    bmax = max(np.percentile(true_bonds, 99), np.percentile(recon_bonds, 99))
    axes[0].hist(true_bonds, bins=60, alpha=0.5, label="Exp", density=True, range=(0, bmax))
    axes[0].hist(recon_bonds, bins=60, alpha=0.5, label=label_recon, density=True, range=(0, bmax))
    axes[0].set_title("Bond Lengths")
    axes[0].legend()
    rmax = max(np.percentile(true_rg, 99), np.percentile(recon_rg, 99)) * 1.1
    axes[1].hist(true_rg, bins=40, alpha=0.5, label="Exp", density=True, range=(0, rmax))
    axes[1].hist(recon_rg, bins=40, alpha=0.5, label=label_recon, density=True, range=(0, rmax))
    axes[1].set_title("Radius of Gyration")
    axes[1].legend()
    axes[2].loglog(s, true_sc, label="Exp", lw=2)
    axes[2].loglog(s, recon_sc, label=label_recon, lw=2, ls="--")
    axes[2].set_title("Spatial Scaling P(s)")
    axes[2].legend()
    title = "Reconstruction Statistics" + (f" ({subset_label} set)" if subset_label else " (test set)")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    _save_pdf_copy(fig, output_path, save_pdf, display_root=display_root)
    if save_plot_data:
        _save_plot_data_npz(
            output_path,
            display_root=display_root,
            exp_bonds=true_bonds, exp_rg=true_rg, genomic_distances=s, exp_scaling=true_sc,
            recon_bonds=recon_bonds, recon_rg=recon_rg, recon_scaling=recon_sc,
        )
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


# -------- Generation: bond, Rg, scaling; avg maps (train, test, gen); difference row --------
def plot_gen_analysis(
    full_stats: dict,
    train_stats: dict,
    test_stats: dict,
    gen_distmaps: np.ndarray,
    output_path: str,
    *,
    sample_variance: float = 1.0,
    label_gen: str = "Gen",
    dpi: int = 150,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    display_root: str | None = None,
) -> None:
    """Bond, Rg, scaling (full/train/test/gen); row2: avg maps train, test, gen; row3: diff test-train, train-gen, test-gen."""
    s = full_stats["genomic_distances"]
    full_b, full_rg, full_sc = full_stats["exp_bonds"], full_stats["exp_rg"], full_stats["exp_scaling"]
    train_b, train_rg, train_sc = train_stats["exp_bonds"], train_stats["exp_rg"], train_stats["exp_scaling"]
    test_b, test_rg, test_sc = test_stats["exp_bonds"], test_stats["exp_rg"], test_stats["exp_scaling"]
    avg_train = train_stats["avg_exp_map"]
    avg_test = test_stats["avg_exp_map"]
    gen_b = distmap_bond_lengths(gen_distmaps)
    gen_rg = distmap_rg(gen_distmaps)
    _, gen_sc = distmap_scaling(gen_distmaps)
    avg_gen = np.mean(gen_distmaps[: min(100, len(gen_distmaps))], axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # Row 0: Bonds, Rg, Scaling — Exp (full), Train, Test, Gen
    bmax = max(
        np.percentile(full_b, 99), np.percentile(train_b, 99),
        np.percentile(test_b, 99), np.percentile(gen_b, 99),
    )
    axes[0, 0].hist(full_b, bins=50, alpha=0.4, label="Exp (full)", density=True, range=(0, bmax), histtype="step", lw=1.5)
    axes[0, 0].hist(train_b, bins=50, alpha=0.4, label="Train", density=True, range=(0, bmax), histtype="step", lw=1.5)
    axes[0, 0].hist(test_b, bins=50, alpha=0.4, label="Test", density=True, range=(0, bmax), histtype="step", lw=1.5)
    axes[0, 0].hist(gen_b, bins=50, alpha=0.6, label=label_gen, density=True, range=(0, bmax))
    axes[0, 0].set_title("Bond Lengths")
    axes[0, 0].legend(fontsize=7)

    rmax = max(
        np.percentile(full_rg, 99), np.percentile(train_rg, 99),
        np.percentile(test_rg, 99), np.percentile(gen_rg, 99),
    ) * 1.1
    axes[0, 1].hist(full_rg, bins=30, alpha=0.4, label="Exp (full)", density=True, range=(0, rmax), histtype="step", lw=1.5)
    axes[0, 1].hist(train_rg, bins=30, alpha=0.4, label="Train", density=True, range=(0, rmax), histtype="step", lw=1.5)
    axes[0, 1].hist(test_rg, bins=30, alpha=0.4, label="Test", density=True, range=(0, rmax), histtype="step", lw=1.5)
    axes[0, 1].hist(gen_rg, bins=30, alpha=0.6, label=label_gen, density=True, range=(0, rmax))
    axes[0, 1].set_title("Radius of Gyration")
    axes[0, 1].legend(fontsize=7)

    y_lo = min(full_sc.min(), train_sc.min(), test_sc.min(), gen_sc.min())
    y_hi = max(full_sc.max(), train_sc.max(), test_sc.max(), gen_sc.max())
    axes[0, 2].loglog(s, full_sc, label="Exp (full)", lw=1.5)
    axes[0, 2].loglog(s, train_sc, label="Train", lw=1.2, ls="--")
    axes[0, 2].loglog(s, test_sc, label="Test", lw=1.2, ls="-.")
    axes[0, 2].loglog(s, gen_sc, label=label_gen, lw=1.5, ls=":")
    axes[0, 2].set_ylim(y_lo * 0.9, y_hi * 1.1)
    axes[0, 2].set_title("Spatial Scaling P(s)")
    axes[0, 2].legend(fontsize=7)

    # Row 1: Avg maps — train, test, gen
    vmin = min(avg_train.min(), avg_test.min(), avg_gen.min())
    vmax_m = max(avg_train.max(), avg_test.max(), avg_gen.max())
    axes[1, 0].imshow(avg_train, cmap="viridis_r", aspect="equal", vmin=vmin, vmax=vmax_m, interpolation="none")
    axes[1, 0].set_title("Train: Avg Map")
    im1 = axes[1, 1].imshow(avg_test, cmap="viridis_r", aspect="equal", vmin=vmin, vmax=vmax_m, interpolation="none")
    axes[1, 1].set_title("Test: Avg Map")
    axes[1, 2].imshow(avg_gen, cmap="viridis_r", aspect="equal", vmin=vmin, vmax=vmax_m, interpolation="none")
    axes[1, 2].set_title(f"{label_gen}: Avg Map")
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax)

    # Row 2: Difference maps — test−train, train−gen, test−gen
    diff_tt = avg_test - avg_train
    diff_tg = avg_train - avg_gen
    diff_test_gen = avg_test - avg_gen
    v_diff = max(np.abs(diff_tt).max(), np.abs(diff_tg).max(), np.abs(diff_test_gen).max()) or 1.0
    axes[2, 0].imshow(diff_tt, cmap="RdBu_r", aspect="equal", vmin=-v_diff, vmax=v_diff, interpolation="none")
    axes[2, 0].set_title("Test − Train")
    imd1 = axes[2, 1].imshow(diff_tg, cmap="RdBu_r", aspect="equal", vmin=-v_diff, vmax=v_diff, interpolation="none")
    axes[2, 1].set_title(f"Train − {label_gen}")
    axes[2, 2].imshow(diff_test_gen, cmap="RdBu_r", aspect="equal", vmin=-v_diff, vmax=v_diff, interpolation="none")
    axes[2, 2].set_title(f"Test − {label_gen}")
    divider2 = make_axes_locatable(axes[2, 1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(imd1, cax=cax2)

    plt.suptitle(f"Sample Variance = {sample_variance}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    _save_pdf_copy(fig, output_path, save_pdf, display_root=display_root)
    if save_plot_data:
        _save_plot_data_npz(
            output_path,
            display_root=display_root,
            sample_variance=np.array(sample_variance),
            full_bonds=full_b, full_rg=full_rg, full_scaling=full_sc,
            train_bonds=train_b, train_rg=train_rg, train_scaling=train_sc,
            test_bonds=test_b, test_rg=test_rg, test_scaling=test_sc,
            gen_bonds=gen_b, gen_rg=gen_rg, gen_scaling=gen_sc,
            avg_train_map=avg_train, avg_test_map=avg_test, avg_gen_map=avg_gen,
            diff_test_train=diff_tt, diff_train_gen=diff_tg, diff_test_gen=diff_test_gen,
        )
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


# -------- Bond length by genomic distance (train / test / gen) --------
NUM_K_DEFAULT = 20
GRID_SHAPE = (5, 4)  # 5x4 subplots


def _k_values_evenly_spaced(N: int, num_k: int = NUM_K_DEFAULT) -> np.ndarray:
    """Return up to num_k integer lags k in [1, N-1], evenly spaced along the structure."""
    max_sep = max(1, N - 1)
    n = min(num_k, max_sep)
    if n <= 0:
        return np.array([], dtype=np.int64)
    # Evenly spaced indices into [1, 2, ..., max_sep]
    k_vals = np.round(np.linspace(1, max_sep, n)).astype(np.int64)
    return np.unique(np.clip(k_vals, 1, max_sep))


def plot_bond_length_by_genomic_distance(
    train_dm: np.ndarray,
    test_dm: np.ndarray,
    gen_dm: np.ndarray,
    output_path: str,
    *,
    num_k: int = NUM_K_DEFAULT,
    label_gen: str = "Gen",
    dpi: int = 150,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    display_root: str | None = None,
) -> None:
    """
    Plot distribution of pairwise distance d(i, i+k) for up to num_k lags k (evenly spaced).
    5x4 grid; each subplot overlays train, test, and generated histograms (same style as gen_analysis).
    """
    N = train_dm.shape[1]
    k_values = _k_values_evenly_spaced(N, num_k)
    n_plots = len(k_values)
    if n_plots == 0:
        return
    nrows, ncols = GRID_SHAPE
    # Flatten grid; we may have fewer than nrows*ncols subplots
    fig, axes_flat = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.asarray(axes_flat).flatten()
    for idx, k in enumerate(k_values):
        ax = axes_flat[idx]
        train_vals = distmap_distances_at_lag(train_dm, int(k))
        test_vals = distmap_distances_at_lag(test_dm, int(k))
        gen_vals = distmap_distances_at_lag(gen_dm, int(k))
        all_vals = np.concatenate([train_vals, test_vals, gen_vals])
        if len(all_vals) == 0:
            ax.set_title(f"k = {k}")
            continue
        x_max = float(np.percentile(all_vals, 99)) * 1.05 if len(all_vals) > 0 else 1.0
        x_max = max(x_max, 1e-6)
        bins = 50
        ax.hist(train_vals, bins=bins, alpha=0.4, label="Train", density=True, range=(0, x_max), histtype="step", lw=1.5)
        ax.hist(test_vals, bins=bins, alpha=0.4, label="Test", density=True, range=(0, x_max), histtype="step", lw=1.5)
        ax.hist(gen_vals, bins=bins, alpha=0.6, label=label_gen, density=True, range=(0, x_max))
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Distance")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.suptitle("Pairwise distance d(i, i+k) by genomic lag k", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    _save_pdf_copy(fig, output_path, save_pdf, display_root=display_root)
    if save_plot_data and n_plots > 0:
        k_arr = np.array(k_values)
        data = {"k_values": k_arr}
        for i, k in enumerate(k_values):
            data[f"train_k{k}"] = distmap_distances_at_lag(train_dm, int(k))
            data[f"test_k{k}"] = distmap_distances_at_lag(test_dm, int(k))
            data[f"gen_k{k}"] = distmap_distances_at_lag(gen_dm, int(k))
        _save_plot_data_npz(output_path, display_root=display_root, **data)
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")


def plot_loss_curves(
    train_loss: list,
    val_loss: list,
    output_path: str,
    title: str = "Training Loss",
    *,
    dpi: int = 150,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    display_root: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Train", lw=2, alpha=0.8)
    ax.plot(epochs, val_loss, label="Val", lw=2, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if max(train_loss) / (min(train_loss) + 1e-10) > 100:
        ax.set_yscale("log")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    _save_pdf_copy(fig, output_path, save_pdf, display_root=display_root)
    if save_plot_data:
        _save_plot_data_npz(
            output_path,
            display_root=display_root,
            epoch=np.arange(1, len(train_loss) + 1, dtype=np.int64),
            train_loss=np.array(train_loss),
            val_loss=np.array(val_loss),
        )
    plt.close()
    print(f"  Saved: {display_path(output_path, display_root)}")
