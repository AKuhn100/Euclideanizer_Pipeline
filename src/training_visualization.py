"""
Training video: one MP4 per trained model (DistMap or Euclideanizer), showing epoch-by-epoch evolution.
Controlled by config section training_visualization. Requires ffmpeg for assembly.
"""
from __future__ import annotations

import os
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation

# Visual theme for training video frames (dark background, accent colors).
# Note: ACC_EXP/ACC_GEN here differ from plot_config (COLOR_TRAIN/COLOR_GEN) for contrast on dark BG.
BG = "#0d1117"
FG = "#e6edf3"
ACC_EXP = "#58a6ff"
ACC_GEN = "#ff7b72"
ACC_PROBE = "#7ee787"
DM_CMAP = "plasma"

_RC = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "grid.color": "#21262d",
    "grid.linewidth": 0.5,
    "font.family": "monospace",
}

_ROW_LABELS = [
    "Exp.\nStructure", "Exp.\nDist Map", "Output\nDist Map",
    "Gen\nStructure", "Gen\nDist Map",
]
_ROW_COLORS = [ACC_PROBE, ACC_PROBE, ACC_GEN, ACC_GEN, ACC_GEN]


def _rg_coords(c):
    return np.sqrt(((c - c.mean(1, keepdims=True)) ** 2).sum(-1).mean(-1))


def _rg_dm(dms):
    N = dms.shape[-1]
    return np.sqrt((dms ** 2).sum((-1, -2)) / (2.0 * N * N))


def _scaling_dm(dms, n_k=200):
    N = dms.shape[-1]
    n_k = min(n_k, N - 1)
    mean_dm = dms.mean(0)
    ks = np.arange(1, n_k + 1)
    vals = np.array([np.diag(mean_dm, k).mean() for k in ks], dtype=np.float32)
    return ks, vals


def _scaling_coords(coords, n_k=200, batch=64):
    N_at = coords.shape[1]
    n_k = min(n_k, N_at - 1)
    dm_sum = np.zeros((N_at, N_at), dtype=np.float64)
    for i in range(0, len(coords), batch):
        bc = coords[i : i + batch]
        sq = (bc ** 2).sum(-1)
        D2 = sq[:, :, None] + sq[:, None, :] - 2.0 * (bc @ bc.transpose(0, 2, 1))
        dm_sum += np.sqrt(np.maximum(D2, 0.0)).sum(0)
    dm_mean = dm_sum / len(coords)
    ks = np.arange(1, n_k + 1)
    vals = np.array([np.diag(dm_mean, k).mean() for k in ks], dtype=np.float32)
    return ks, vals


def _kabsch(q, r):
    """Align q to r via scipy Kabsch. Both (N, 3). Returns centered aligned q."""
    q_c = q - q.mean(0)
    r_c = r - r.mean(0)
    rot, _ = Rotation.align_vectors(r_c, q_c)  # R @ query_centered = ref_centered
    return rot.apply(q_c)


def _coord_to_dm(c):
    sq = (c ** 2).sum(-1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (c @ c.T)
    return np.sqrt(np.maximum(D2, 0.0)).astype(np.float32)


def _select_probes(train_ds, n=6, seed=0):
    rng = np.random.default_rng(seed)
    nsub = min(500, len(train_ds))
    sidx = rng.choice(len(train_ds), nsub, replace=False).tolist()
    crds = torch.stack([train_ds[i] for i in sidx]).cpu().numpy()
    rg = _rg_coords(crds)
    quantiles = np.linspace(5, 95, n)
    return [sidx[int(np.argmin(np.abs(rg - np.percentile(rg, q))))] for q in quantiles]


def _compute_exp_ref(train_ds, test_ds, device, num_atoms, batch=64):
    def _load(ds):
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False)
        parts = []
        with torch.no_grad():
            for b in dl:
                c = b.cpu().numpy().astype(np.float32)
                c -= c.mean(1, keepdims=True)
                parts.append(c)
        return np.concatenate(parts)

    all_c = np.concatenate([_load(train_ds), _load(test_ds)])
    rg = _rg_coords(all_c)
    ks, sc = _scaling_coords(all_c[: min(2000, len(all_c))], n_k=400)
    return rg, ks, sc


def _build_figure(n_probe, frame_w, frame_h):
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(frame_w, frame_h), facecolor=BG)
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.0, 5],
        hspace=0.20, left=0.05, right=0.99, top=0.93, bottom=0.02,
    )
    plot_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.40)
    data_gs = gridspec.GridSpecFromSubplotSpec(
        5, n_probe, subplot_spec=outer[1], hspace=0.12, wspace=0.03,
    )
    with plt.rc_context(_RC):
        ax_sc = fig.add_subplot(plot_gs[0])
        ax_rg = fig.add_subplot(plot_gs[1])
        ax_loss = fig.add_subplot(plot_gs[2])
        data_axes = [[fig.add_subplot(data_gs[r, c]) for c in range(n_probe)] for r in range(5)]
    return fig, ax_sc, ax_rg, ax_loss, data_axes


def _header(fig, phase_label, epoch, total_epochs, loss, val_loss):
    pct = epoch / max(total_epochs, 1)
    bar_len = 48
    filled = int(round(bar_len * pct))
    bar = "█" * filled + "░" * (bar_len - filled)
    title = f"{phase_label}   Epoch {epoch}/{total_epochs}   [{bar}]   train={loss:.4f}  val={val_loss:.4f}"
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=9, color=FG, fontfamily="monospace", transform=fig.transFigure)


def _plot_dm(ax, dm, vmax):
    ax.imshow(dm, cmap=DM_CMAP, vmin=0, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_chain_2d(ax, coords, ref_coords=None, lim=None):
    """Plot 2D chain: coords (N, 3) with columns x, y, z; plots x vs y. No workarounds."""
    if ref_coords is not None:
        coords = _kabsch(coords, ref_coords)
    x, y = coords[:, 0], coords[:, 1]
    N = len(x)
    cols = cm.rainbow(np.linspace(0, 1, N))
    for i in range(N - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=cols[i], lw=0.7, alpha=0.85, solid_capstyle="round")
    ax.scatter(x, y, c=np.linspace(0, 1, N), cmap="rainbow", s=3, zorder=5, linewidths=0)
    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def _placeholder(ax, msg="Phase 2\nonly"):
    """Draw a placeholder panel (e.g. for DistMap-only phase where EU output is not yet available)."""
    ax.set_facecolor("#0a0e14")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=7, color="#30363d", transform=ax.transAxes, linespacing=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    for sp in ax.spines.values():
        sp.set_color("#1c2128")


def _plot_right_panel(ax_sc, ax_rg, ax_loss, exp_rg, exp_ks, exp_sc, gen_rg, gen_ks, gen_sc, loss_history, val_history, epoch, rg_xlim, rg_ylim=None, epoch_for_loss_line=None):
    ax_sc.loglog(exp_ks, exp_sc, lw=1.5, color=ACC_EXP, label="Exp", zorder=3)
    if gen_sc is not None:
        ax_sc.loglog(gen_ks, gen_sc, lw=1.5, color=ACC_GEN, ls="--", label="Gen", zorder=3)
    ax_sc.set_xlabel("Genomic Sep", fontsize=7, color=FG)
    ax_sc.set_ylabel("Mean Dist", fontsize=7, color=FG)
    ax_sc.set_title("Genomic Scaling", fontsize=8, color=FG, pad=3)
    ax_sc.legend(fontsize=6.5, framealpha=0.2)
    ax_sc.spines[:].set_color("#30363d")

    bins = np.linspace(rg_xlim[0], rg_xlim[1], 35)
    _exp_ok = exp_rg[np.isfinite(exp_rg)]
    if len(_exp_ok) > 1:
        ax_rg.hist(_exp_ok, bins=bins, density=True, alpha=0.55, color=ACC_EXP, label="Exp")
    if gen_rg is not None and len(gen_rg) > 1:
        _gen_ok = gen_rg[np.isfinite(gen_rg)]
        if len(_gen_ok) > 1:
            ax_rg.hist(_gen_ok, bins=bins, density=True, alpha=0.55, color=ACC_GEN, label="Gen")
    ax_rg.set_xlim(rg_xlim)
    if rg_ylim is not None:
        ax_rg.set_ylim(rg_ylim)
    ax_rg.set_xlabel("Rg", fontsize=7, color=FG)
    ax_rg.set_ylabel("Density", fontsize=7, color=FG)
    ax_rg.set_title("Radius of Gyration", fontsize=8, color=FG, pad=3)
    ax_rg.legend(fontsize=6.5, framealpha=0.2)
    ax_rg.spines[:].set_color("#30363d")

    ep = np.arange(1, len(loss_history) + 1)
    ax_loss.plot(ep, loss_history, lw=1.2, color=ACC_GEN, label="Train")
    if val_history:
        ax_loss.plot(np.arange(1, len(val_history) + 1), val_history, lw=1.2, color=ACC_EXP, ls="--", label="Val")
    line_epoch = epoch_for_loss_line if epoch_for_loss_line is not None else epoch
    ax_loss.axvline(line_epoch, color=FG, lw=0.8, alpha=0.6)
    ax_loss.set_xlabel("Epoch", fontsize=7, color=FG)
    ax_loss.set_ylabel("Loss", fontsize=7, color=FG)
    ax_loss.set_title("Training Loss", fontsize=8, color=FG, pad=3)
    ax_loss.legend(fontsize=6.5, framealpha=0.2)
    ax_loss.spines[:].set_color("#30363d")


def _add_row_labels(fig, data_axes, override_last_two=None):
    LPAD = 0.05
    fig.canvas.draw()
    labels = list(_ROW_LABELS)
    if override_last_two is not None:
        labels[3], labels[4] = override_last_two[0], override_last_two[1]
    for r, (lbl, col) in enumerate(zip(labels, _ROW_COLORS)):
        bbox = data_axes[r][0].get_position()
        y_mid = bbox.y0 + bbox.height * 0.5
        x_pos = LPAD * 0.42
        fig.text(x_pos, y_mid, lbl, ha="center", va="center", fontsize=6.5, color=col, transform=fig.transFigure, linespacing=1.4)


def _save_frame(fig, frame_idx, frames_dir, frame_dpi):
    os.makedirs(frames_dir, exist_ok=True)
    path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
    with plt.rc_context(_RC):
        fig.savefig(path, dpi=frame_dpi, facecolor=BG)
    plt.close(fig)
    return path


def render_dm_frame(epoch, total_epochs, probe_ref_coords, probe_input_dms, probe_recon_dms, gen_rg, gen_ks, gen_sc, exp_rg, exp_ks, exp_sc, rg_xlim, rg_ylim, loss_history, val_history, dm_vmax, axis_lim, frame_idx, frames_dir, vis_cfg, epoch_for_loss_line=None):
    n_probe = vis_cfg["n_probe"]
    fw, fh, fdpi = vis_cfg["frame_width"], vis_cfg["frame_height"], vis_cfg["frame_dpi"]
    fig, ax_sc, ax_rg, ax_loss, da = _build_figure(n_probe, fw, fh)
    for j in range(n_probe):
        _plot_chain_2d(da[0][j], probe_ref_coords[j], lim=axis_lim)
        _plot_dm(da[1][j], probe_input_dms[j], dm_vmax)
        _plot_dm(da[2][j], probe_recon_dms[j], dm_vmax)
        _placeholder(da[3][j], "Gen\nStructure\n(Phase 2)")
        _placeholder(da[4][j], "Gen\nDist Map\n(Phase 2)")
    _plot_right_panel(ax_sc, ax_rg, ax_loss, exp_rg, exp_ks, exp_sc, gen_rg, gen_ks, gen_sc, loss_history, val_history, epoch, rg_xlim, rg_ylim, epoch_for_loss_line=epoch_for_loss_line)
    loss_val = loss_history[-1] if loss_history else float("nan")
    val_val = val_history[-1] if val_history else float("nan")
    _header(fig, "DISTMAP TRAINING", epoch, total_epochs, loss_val, val_val)
    _add_row_labels(fig, da)
    return _save_frame(fig, frame_idx, frames_dir, fdpi)


def render_eu_frame(epoch, total_epochs, probe_ref_coords, probe_input_dms, probe_non_euclidean_dms, probe_eu_coords_aligned, probe_euclidean_dms, gen_rg, gen_ks, gen_sc, exp_rg, exp_ks, exp_sc, rg_xlim, rg_ylim, loss_history, val_history, dm_vmax, axis_lim, frame_idx, last_row_is_coords, frames_dir, vis_cfg, epoch_for_loss_line=None):
    n_probe = vis_cfg["n_probe"]
    fw, fh, fdpi = vis_cfg["frame_width"], vis_cfg["frame_height"], vis_cfg["frame_dpi"]
    fig, ax_sc, ax_rg, ax_loss, da = _build_figure(n_probe, fw, fh)
    for j in range(n_probe):
        _plot_chain_2d(da[0][j], probe_ref_coords[j], lim=axis_lim)
        _plot_dm(da[1][j], probe_input_dms[j], dm_vmax)
        _plot_dm(da[2][j], probe_non_euclidean_dms[j], dm_vmax)
        if last_row_is_coords:
            _plot_dm(da[3][j], probe_euclidean_dms[j], dm_vmax)
            _plot_chain_2d(da[4][j], probe_eu_coords_aligned[j], lim=axis_lim)
        else:
            _plot_chain_2d(da[3][j], probe_eu_coords_aligned[j], lim=axis_lim)
            _plot_dm(da[4][j], probe_euclidean_dms[j], dm_vmax)
    _plot_right_panel(ax_sc, ax_rg, ax_loss, exp_rg, exp_ks, exp_sc, gen_rg, gen_ks, gen_sc, loss_history, val_history, epoch, rg_xlim, rg_ylim, epoch_for_loss_line=epoch_for_loss_line)
    loss_val = loss_history[-1] if loss_history else float("nan")
    val_val = val_history[-1] if val_history else float("nan")
    _header(fig, "EUCLIDEANIZER TRAINING", epoch, total_epochs, loss_val, val_val)
    override = ("Gen\nDist Map", "Gen\nStructure") if last_row_is_coords else None
    _add_row_labels(fig, da, override_last_two=override)
    return _save_frame(fig, frame_idx, frames_dir, fdpi)


def assemble_video(frames_dir, output_path, fps):
    """Assemble frame_%05d.png in frames_dir into output_path.
    Returns (True, None) if successful; (False, reason) on failure (reason for pipeline log).
    Uses EUCLIDEANIZER_FFMPEG if set (e.g. by multi-GPU HPO launcher so workers see ffmpeg)."""
    pattern = os.path.join(frames_dir, "frame_%05d.png")
    ffmpeg_exe = os.environ.get("EUCLIDEANIZER_FFMPEG", "ffmpeg")
    cmd = [
        ffmpeg_exe, "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path,
    ]
    print(f"  Assembling video: {output_path}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("  WARNING: ffmpeg not found. Skipping video assembly. Frames kept in:", frames_dir)
        return False, "ffmpeg not found (not in PATH)"
    except OSError as e:
        print(f"  WARNING: Failed to run ffmpeg ({e}). Skipping video assembly. Frames kept in:", frames_dir)
        return False, f"ffmpeg failed to run: {e}"
    if result.returncode != 0:
        print("  WARNING: ffmpeg failed (non-zero exit). Frames kept in:", frames_dir)
        if result.stderr:
            print("  ffmpeg stderr:", result.stderr[-2000:])
        return False, f"ffmpeg exited with code {result.returncode}"
    print(f"  Video saved → {output_path}")
    return True, None


def make_distmap_epoch_hook(coords, dm_cfg, run_dir, device, utils_mod, vis_cfg, *, split_seed: int, training_split: float, epoch_start: int = 0, total_epochs_display: int | None = None):
    """Return a callback(epoch, model, train_losses, val_losses) that renders one frame per epoch.
    When training in segments (resume), pass epoch_start = end epoch of previous segment so the progress bar shows N..M.
    Pass total_epochs_display (e.g. max over all segment targets) so the first segment's video shows 1/M .. N/M."""
    n_probe = vis_cfg["n_probe"]
    n_quick = vis_cfg["n_quick"]
    latent_dim = dm_cfg["latent_dim"]
    frames_dir = os.path.join(run_dir, "training_video", "frames")
    total_epochs = int(total_epochs_display) if total_epochs_display is not None else dm_cfg["epochs"]

    from . import utils
    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    probe_idx = _select_probes(train_ds, n=n_probe, seed=42)
    probe_ref_coords = torch.stack([train_ds[i] for i in probe_idx]).cpu().numpy().astype(np.float32)
    probe_ref_coords -= probe_ref_coords.mean(1, keepdims=True)
    sample_batch = torch.stack([train_ds[i] for i in probe_idx]).to(device)
    probe_input_dms = utils_mod.get_distmaps(sample_batch).cpu().numpy().astype(np.float32)
    dm_vmax = float(np.percentile(probe_input_dms, 97))
    axis_lim = float(np.abs(probe_ref_coords).max()) * 1.15

    exp_rg, exp_ks, exp_sc = _compute_exp_ref(train_ds, test_ds, device, coords.size(1))
    _exp_ok = exp_rg[np.isfinite(exp_rg)]
    if _exp_ok.size == 0:
        n_total = len(exp_rg)
        n_nan = np.isnan(exp_rg).sum()
        n_inf = np.isinf(exp_rg).sum()
        raise RuntimeError(
            "Training visualization requires finite experimental Rg values. "
            f"Got exp_rg with length {n_total}: {n_nan} NaN, {n_inf} inf. "
            "This should not happen with valid coordinates (non-empty train+test and no NaN/inf in coords). "
            "Check that the dataset has enough structures for the train/test split and that coordinates are finite."
        )
    _q05, _q95 = np.percentile(_exp_ok, [2, 98])
    _margin = max((_q95 - _q05) * 0.3, 0.05)
    rg_xlim = (max(0.0, _q05 - _margin), _q95 + _margin)
    _rg_bins = np.linspace(rg_xlim[0], rg_xlim[1], 35)
    _exp_hist, _ = np.histogram(_exp_ok, bins=_rg_bins, density=True)
    rg_ylim = (0.0, float(np.max(_exp_hist) * 1.15) if _exp_hist.size else 1.0)

    from .distmap.sample import generate_samples
    torch.manual_seed(12345)
    z_rg_fixed = generate_samples(n_quick, latent_dim, device, variance=vis_cfg["gen_sample_variance"])

    def on_epoch(epoch, model, train_losses, val_losses, run_dirs=None):
        model.eval()
        with torch.no_grad():
            dm_p = utils_mod.get_distmaps(sample_batch)
            _, _, _, recon_p = model(dm_p)
            recon_sym = utils_mod.upper_tri_to_symmetric(recon_p, coords.size(1))
            probe_recon_dms = torch.expm1(recon_sym).cpu().numpy().astype(np.float32)
            gen_log = model._decode_to_matrix(z_rg_fixed)
            gen_dms = torch.expm1(gen_log).cpu().numpy().astype(np.float32)
        gen_rg = _rg_dm(gen_dms)
        gen_ks, gen_sc = _scaling_dm(gen_dms, n_k=300)
        display_epoch = epoch_start + epoch
        epoch_for_loss_line = epoch if epoch_start else None
        if run_dirs:
            for d in run_dirs:
                fd = os.path.join(d, "training_video", "frames")
                render_dm_frame(
                    display_epoch, total_epochs,
                    probe_ref_coords, probe_input_dms, probe_recon_dms,
                    gen_rg, gen_ks, gen_sc,
                    exp_rg, exp_ks, exp_sc, rg_xlim, rg_ylim,
                    train_losses, val_losses,
                    dm_vmax, axis_lim, epoch, fd, vis_cfg,
                    epoch_for_loss_line=epoch_for_loss_line,
                )
        else:
            render_dm_frame(
                display_epoch, total_epochs,
                probe_ref_coords, probe_input_dms, probe_recon_dms,
                gen_rg, gen_ks, gen_sc,
                exp_rg, exp_ks, exp_sc, rg_xlim, rg_ylim,
                train_losses, val_losses,
                dm_vmax, axis_lim, epoch, frames_dir, vis_cfg,
                epoch_for_loss_line=epoch_for_loss_line,
            )

    return on_epoch, frames_dir


def make_euclideanizer_epoch_hook(coords, eu_cfg, frozen_vae_path, frozen_latent_dim, run_dir, device, utils_mod, vis_cfg, *, split_seed: int, training_split: float, epoch_start: int = 0, total_epochs_display: int | None = None):
    """Return a callback(epoch, embed, train_losses, val_losses) that renders one frame per epoch. Loads frozen_vae once inside.
    When training in segments (resume), pass epoch_start = end epoch of previous segment so the progress bar shows N..M.
    Pass total_epochs_display (e.g. max over all segment targets) so the first segment's video shows 1/M .. N/M."""
    from .euclideanizer.model import load_frozen_vae

    num_atoms = coords.size(1)
    frozen_vae = load_frozen_vae(frozen_vae_path, num_atoms, frozen_latent_dim, device)

    n_probe = vis_cfg["n_probe"]
    n_quick = vis_cfg["n_quick"]
    latent_dim = frozen_latent_dim
    total_epochs = int(total_epochs_display) if total_epochs_display is not None else eu_cfg["epochs"]
    frames_dir = os.path.join(run_dir, "training_video", "frames")

    from . import utils
    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    probe_idx = _select_probes(train_ds, n=n_probe, seed=42)
    probe_ref_coords = torch.stack([train_ds[i] for i in probe_idx]).cpu().numpy().astype(np.float32)
    probe_ref_coords -= probe_ref_coords.mean(1, keepdims=True)
    sample_batch = torch.stack([train_ds[i] for i in probe_idx]).to(device)
    probe_input_dms = utils_mod.get_distmaps(sample_batch).cpu().numpy().astype(np.float32)
    dm_vmax = float(np.percentile(probe_input_dms, 97))
    axis_lim = float(np.abs(probe_ref_coords).max()) * 1.15

    with torch.no_grad():
        dm_p = utils_mod.get_distmaps(sample_batch)
        gt_log_p = torch.log1p(dm_p)
        mu_p = frozen_vae.encode(gt_log_p)
        D_ne_p = frozen_vae._decode_to_matrix(mu_p)
    probe_non_euclidean_dms = torch.expm1(D_ne_p).cpu().numpy().astype(np.float32)

    exp_rg, exp_ks, exp_sc = _compute_exp_ref(train_ds, test_ds, device, coords.size(1))
    _exp_ok = exp_rg[np.isfinite(exp_rg)]
    if _exp_ok.size == 0:
        n_total = len(exp_rg)
        n_nan = np.isnan(exp_rg).sum()
        n_inf = np.isinf(exp_rg).sum()
        raise RuntimeError(
            "Training visualization requires finite experimental Rg values. "
            f"Got exp_rg with length {n_total}: {n_nan} NaN, {n_inf} inf. "
            "This should not happen with valid coordinates (non-empty train+test and no NaN/inf in coords). "
            "Check that the dataset has enough structures for the train/test split and that coordinates are finite."
        )
    _q05, _q95 = np.percentile(_exp_ok, [2, 98])
    _margin = max((_q95 - _q05) * 0.3, 0.05)
    rg_xlim = (max(0.0, _q05 - _margin), _q95 + _margin)
    _rg_bins = np.linspace(rg_xlim[0], rg_xlim[1], 35)
    _exp_hist, _ = np.histogram(_exp_ok, bins=_rg_bins, density=True)
    rg_ylim = (0.0, float(np.max(_exp_hist) * 1.15) if _exp_hist.size else 1.0)

    from .distmap.sample import generate_samples
    torch.manual_seed(54321)
    z_rg_fixed = generate_samples(n_quick, latent_dim, device, variance=vis_cfg["gen_sample_variance"])

    def on_epoch(epoch, embed, train_losses, val_losses, run_dirs=None):
        embed.eval()
        with torch.no_grad():
            probe_eu_coords = embed(D_ne_p).cpu().numpy().astype(np.float32)
        probe_eu_coords -= probe_eu_coords.mean(1, keepdims=True)
        probe_euclidean_dms = np.array([_coord_to_dm(probe_eu_coords[j]) for j in range(n_probe)])
        probe_eu_coords_aligned = np.array([_kabsch(probe_eu_coords[j], probe_ref_coords[j]) for j in range(n_probe)])

        with torch.no_grad():
            D_ne_q = frozen_vae._decode_to_matrix(z_rg_fixed)
            gen_c = embed(D_ne_q).cpu().numpy().astype(np.float32)
        gen_c -= gen_c.mean(1, keepdims=True)
        gen_rg = _rg_coords(gen_c)
        gen_ks, gen_sc = _scaling_coords(gen_c, n_k=300)

        display_epoch = epoch_start + epoch
        epoch_for_loss_line = epoch if epoch_start else None
        if run_dirs:
            for d in run_dirs:
                fd = os.path.join(d, "training_video", "frames")
                render_eu_frame(
                    display_epoch, total_epochs,
                    probe_ref_coords, probe_input_dms, probe_non_euclidean_dms,
                    probe_eu_coords_aligned, probe_euclidean_dms,
                    gen_rg, gen_ks, gen_sc,
                    exp_rg, exp_ks, exp_sc, rg_xlim, rg_ylim,
                    train_losses, val_losses,
                    dm_vmax, axis_lim, epoch, False, fd, vis_cfg,
                    epoch_for_loss_line=epoch_for_loss_line,
                )
        else:
            render_eu_frame(
                display_epoch, total_epochs,
                probe_ref_coords, probe_input_dms, probe_non_euclidean_dms,
                probe_eu_coords_aligned, probe_euclidean_dms,
                gen_rg, gen_ks, gen_sc,
                exp_rg, exp_ks, exp_sc, rg_xlim, rg_ylim,
                train_losses, val_losses,
                dm_vmax, axis_lim, epoch, False, frames_dir, vis_cfg,
                epoch_for_loss_line=epoch_for_loss_line,
            )

    return on_epoch, frames_dir
