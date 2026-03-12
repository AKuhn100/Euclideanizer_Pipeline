"""
Latent space visualization: distribution (box plots + mean/std per dimension) and
correlation (train vs test mean and std). Used by recon analysis blocks (RMSD, Q,
coord clustering, distmap clustering) when visualize_latent is enabled.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import display_path
from .plotting import _save_pdf_copy
from .plot_config import (
    COLOR_TRAIN,
    COLOR_TEST,
    FONT_FAMILY,
    FONT_SIZE_TITLE,
    FONT_SIZE_AXIS,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
)


def _latent_dim_xticks(n_dim: int, max_labels: int = 5):
    """
    Tick positions and string labels for latent dimension axis.
    With large n_dim, label every dim is unreadable; use a few evenly spaced ticks only.
    endpoint=False so we never force n_dim-1 as a tick—the axis still spans full range via xlim.
    """
    if n_dim <= 0:
        return [], []
    if n_dim <= max_labels:
        positions = list(range(n_dim))
        return positions, [str(i) for i in positions]
    # Evenly spaced in [0, n_dim); do not label last dim explicitly
    raw = np.linspace(0, n_dim - 1, max_labels, endpoint=False)
    positions_int = np.clip(np.round(raw).astype(int), 0, n_dim - 1)
    positions = []
    for p in positions_int:
        if not positions or positions[-1] != p:
            positions.append(int(p))
    return positions, [str(p) for p in positions]


def plot_latent_distribution(
    train_mu_np: np.ndarray,
    test_mu_np: np.ndarray,
    out_path: str,
    plot_dpi: int = 150,
    display_root: str | None = None,
    save_pdf_copy: bool = False,
) -> None:
    """
    Plot latent distribution: top row two box plots (train left, test right), shared y;
    middle row one plot full width = mean per dimension (train vs test lines);
    bottom row one plot full width = std per dimension (train vs test lines).
    """
    n_dim = train_mu_np.shape[1]
    dims = np.arange(n_dim)
    fig = plt.figure(figsize=(12, 10))
    # Top row: two box plots side by side (train left, test right)
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.boxplot(
        [train_mu_np[:, d] for d in range(n_dim)],
        positions=np.arange(n_dim),
        widths=0.6,
        patch_artist=True,
    )
    ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax1.set_ylabel("Latent Value", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax1.set_title("Train Latent Distribution", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    ax1.set_xlabel("Dimension", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax1.set_xlim(-0.5, n_dim - 0.5)
    _pos1, _lab1 = _latent_dim_xticks(n_dim)
    if _pos1:
        ax1.set_xticks(_pos1)
        ax1.set_xticklabels(_lab1)

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.boxplot(
        [test_mu_np[:, d] for d in range(n_dim)],
        positions=np.arange(n_dim),
        widths=0.6,
        patch_artist=True,
    )
    ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax2.set_ylabel("Latent Value", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax2.set_title("Test Latent Distribution", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
    ax2.set_xlabel("Dimension", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax2.set_xlim(-0.5, n_dim - 0.5)
    if _pos1:
        ax2.set_xticks(_pos1)
        ax2.set_xticklabels(_lab1)
    y_min = min(train_mu_np.min(), test_mu_np.min())
    y_max = max(train_mu_np.max(), test_mu_np.max())
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Middle row: mean per dimension (full width)
    ax_mean = fig.add_subplot(3, 2, (3, 4))
    mean_train = np.mean(train_mu_np, axis=0)
    mean_test = np.mean(test_mu_np, axis=0)
    ax_mean.plot(dims, mean_train, label="Train", color=COLOR_TRAIN)
    ax_mean.plot(dims, mean_test, label="Test", color=COLOR_TEST)
    ax_mean.set_xlabel("Dimension", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax_mean.set_ylabel("Mean", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax_mean.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    ax_mean.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    if n_dim > 5:
        ax_mean.set_xticks(_pos1)
        ax_mean.set_xticklabels(_lab1)

    # Bottom row: std per dimension (full width)
    ax_std = fig.add_subplot(3, 2, (5, 6))
    std_train = np.std(train_mu_np, axis=0)
    std_test = np.std(test_mu_np, axis=0)
    ax_std.plot(dims, std_train, label="Train", color=COLOR_TRAIN)
    ax_std.plot(dims, std_test, label="Test", color=COLOR_TEST)
    ax_std.set_xlabel("Dimension", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax_std.set_ylabel("Std", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
    ax_std.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    ax_std.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    if n_dim > 5:
        ax_std.set_xticks(_pos1)
        ax_std.set_xticklabels(_lab1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=plot_dpi)
    if save_pdf_copy:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close()
    if display_root is not None:
        print(f"  Saved: {display_path(out_path, display_root)}")


def plot_latent_correlation(
    train_mu_np: np.ndarray,
    test_mu_np: np.ndarray,
    out_path: str,
    plot_dpi: int = 150,
    display_root: str | None = None,
    save_pdf_copy: bool = False,
) -> None:
    """
    Plot Pearson correlation between train and test latent statistics: two square panels.
    Left: mean per dimension (train mean vs test mean); right: std per dimension (train std vs test std).
    Each panel has x = training value, y = test value, one point per dimension, dashed y=x, with Pearson r and R².
    """
    mean_train = np.mean(train_mu_np, axis=0)
    mean_test = np.mean(test_mu_np, axis=0)
    std_train = np.std(train_mu_np, axis=0)
    std_test = np.std(test_mu_np, axis=0)

    def _pearson_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        r = np.corrcoef(x, y)[0, 1]
        r = float(r) if not np.isnan(r) else 0.0
        return r, r * r

    r_mean, r2_mean = _pearson_r2(mean_train, mean_test)
    r_std, r2_std = _pearson_r2(std_train, std_test)

    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, x_vals, y_vals, r_val, r2_val, title in [
        (ax_mean, mean_train, mean_test, r_mean, r2_mean, "Mean (Train vs Test)"),
        (ax_std, std_train, std_test, r_std, r2_std, "Std (Train vs Test)"),
    ]:
        ax.scatter(x_vals, y_vals, alpha=0.8, s=20)
        ax.set_xlabel("Train", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        ax.set_ylabel("Test", fontsize=FONT_SIZE_AXIS, family=FONT_FAMILY)
        ax.set_title(f"{title}\nPearson r = {r_val:.4f}, R² = {r2_val:.4f}", fontsize=FONT_SIZE_TITLE, family=FONT_FAMILY)
        lo = min(x_vals.min(), y_vals.min())
        hi = max(x_vals.max(), y_vals.max())
        margin = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal")
        diag_lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        diag_hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", alpha=0.6, label="y = x")
        ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=plot_dpi)
    if save_pdf_copy:
        _save_pdf_copy(fig, out_path, save_pdf=True, display_root=display_root)
    plt.close()
    if display_root is not None:
        print(f"  Saved: {display_path(out_path, display_root)}")
