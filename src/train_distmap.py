"""
Train a single DistMap (ChromVAE_Conv) with given config.
"""
from __future__ import annotations

import os
from typing import Callable
import shutil
import time
import torch
from . import utils
from .config import load_run_config, save_run_config
from .distmap.model import ChromVAE_Conv
from .distmap.loss import distmap_vae_loss
from .plot_config import PLOT_DPI
from .plotting import plot_loss_curves


def train_distmap(
    dm_cfg: dict,
    device: torch.device,
    coords: torch.Tensor,
    output_dir: str,
    *,
    split_seed: int,
    training_split: float,
    epoch_callback=None,
    plot_loss: bool = True,
    plot_dpi: int = PLOT_DPI,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    resume_from_path: str | None = None,
    additional_epochs: int | None = None,
    prev_run_dir: str | None = None,
    save_final_models_per_stretch: bool = False,
    is_last_segment: bool = False,
    memory_efficient: bool = False,
    display_root: str | None = None,
    calibration_memory_fraction: float | None = None,
    on_batch_size_resolved: Callable[[int], None] | None = None,
) -> tuple[str, bool]:
    """
    Train one DistMap model. When resuming: load from prev run's model_last.pt, carry over best;
    save model.pt (best) and model_last.pt (last). Returns (path to model.pt, stopped_early).
    When early_stopping is True and validation loss does not improve for patience epochs, training
    stops and stopped_early is True; best model and run_config with early_stopped=True are saved.
    """
    early_stopping = bool(dm_cfg.get("early_stopping", False))
    patience = int(dm_cfg.get("patience", 20))
    num_atoms = coords.size(1)
    beta_kl = dm_cfg["beta_kl"]
    latent_dim = dm_cfg["latent_dim"]
    lambda_mse = dm_cfg["lambda_mse"]
    lambda_w_gen = dm_cfg["lambda_w_gen"]
    lambda_w_recon = dm_cfg["lambda_w_recon"]
    batch_size = dm_cfg["batch_size"]
    lr = dm_cfg["learning_rate"]
    is_resume = resume_from_path is not None and additional_epochs is not None

    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    if is_resume:
        epochs = additional_epochs
        model = ChromVAE_Conv(num_atoms=num_atoms, latent_space_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(resume_from_path, map_location=device))
    else:
        epochs = dm_cfg["epochs"]
        model = ChromVAE_Conv(num_atoms=num_atoms, latent_space_dim=latent_dim).to(device)

    if batch_size is None:
        run_cfg = load_run_config(model_dir)
        if run_cfg and isinstance(run_cfg.get("distmap", {}).get("batch_size"), int):
            batch_size = run_cfg["distmap"]["batch_size"]
        else:
            from .calibrate import calibrate_distmap_batch_size
            threshold = calibration_memory_fraction if calibration_memory_fraction is not None else 0.85
            batch_size = calibrate_distmap_batch_size(
                model, dm_cfg, coords, device, threshold=threshold,
                training_split=training_split, split_seed=split_seed,
            )
            dm_cfg = {**dm_cfg, "batch_size": batch_size}
            print(f"  Auto-calibrated batch_size: {batch_size}")
            if on_batch_size_resolved is not None:
                on_batch_size_resolved(batch_size)
    else:
        batch_size = dm_cfg["batch_size"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_loss_hist, val_loss_hist = [], []
    best_val = float("inf")
    best_state = None
    best_epoch = None
    start_epoch_offset = 0

    if is_resume and prev_run_dir is not None:
        prev_model_dir = os.path.join(prev_run_dir, "model")
        prev_cfg = load_run_config(prev_model_dir)
        prev_best = os.path.join(prev_model_dir, "model.pt")
        # Resuming from same run's best (within-segment or interrupted later segment): offset = best_epoch
        resume_from_best = os.path.normpath(resume_from_path) == os.path.normpath(prev_best)
        if prev_cfg is not None:
            best_val = prev_cfg["best_val"] if prev_cfg["best_val"] is not None else float("inf")
            best_epoch = prev_cfg["best_epoch"]
            start_epoch_offset = (prev_cfg["best_epoch"] or 0) if resume_from_best else (prev_cfg["last_epoch_trained"] or 0)
        if os.path.isfile(prev_best):
            shutil.copy2(prev_best, os.path.join(model_dir, "model.pt"))
            if not memory_efficient:
                best_state = {k: v.clone() for k, v in torch.load(prev_best, map_location="cpu").items()}
        save_run_config({"distmap": dm_cfg}, model_dir, last_epoch_trained=0, best_epoch=best_epoch, best_val=best_val)
    else:
        save_run_config({"distmap": dm_cfg}, model_dir, last_epoch_trained=0, best_epoch=None)

    epochs_without_improvement = 0
    train_start = time.time()
    stopped_early = False
    for epoch in range(epochs):
        model.train()
        ep_loss, ep_kl, ep_recon, n_b = 0.0, 0.0, 0.0, 0
        for batch in train_dl:
            batch_dm = utils.get_distmaps(batch)
            mu, logvar, z, recon_tri = model(batch_dm)
            gt_full = torch.log1p(batch_dm)
            recon_full = utils.upper_tri_to_symmetric(recon_tri, num_atoms)
            loss, kl, recon, _, _, _ = distmap_vae_loss(
                mu, logvar, gt_full, recon_full,
                model, latent_dim, device, beta_kl, lambda_mse, lambda_w_gen, lambda_w_recon,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            ep_loss += loss.item()
            ep_kl += kl.item()
            ep_recon += recon.item()
            n_b += 1
        avg_train = ep_loss / n_b
        scheduler.step()

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in test_dl:
                batch_dm = utils.get_distmaps(batch)
                mu, logvar, z, recon_tri = model(batch_dm)
                gt_full = torch.log1p(batch_dm)
                recon_full = utils.upper_tri_to_symmetric(recon_tri, num_atoms)
                loss, *_ = distmap_vae_loss(
                    mu, logvar, gt_full, recon_full,
                    model, latent_dim, device, beta_kl, lambda_mse, lambda_w_gen, lambda_w_recon,
                )
                val_sum += loss.item()
                val_n += 1
        avg_val = val_sum / val_n
        train_loss_hist.append(avg_train)
        val_loss_hist.append(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = start_epoch_offset + epoch + 1  # 1-indexed global
            epochs_without_improvement = 0
            if memory_efficient:
                torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
                save_run_config({"distmap": dm_cfg}, model_dir, last_epoch_trained=0, best_epoch=best_epoch, best_val=best_val)
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epoch_callback is not None:
            epoch_callback(epoch + 1, model, train_loss_hist, val_loss_hist, run_dirs=[output_dir])
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = (time.time() - train_start) / 60
            print(f"  Epoch {epoch+1}/{epochs} Train: {avg_train:.6f} Val: {avg_val:.6f} (elapsed {elapsed:.1f}m)" + (" *" if avg_val <= best_val else ""))
        if early_stopping and epochs_without_improvement >= patience:
            stopped_early = True
            actual_epochs = start_epoch_offset + epoch + 1
            print(f"  DistMap early stopping at epoch {actual_epochs} (no validation improvement for {patience} epochs).")
            break

    elapsed = (time.time() - train_start) / 60
    print(f"  DistMap training finished in {elapsed:.1f}m.")
    save_last = (not is_last_segment) or save_final_models_per_stretch
    if save_last:
        last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(last_state, os.path.join(model_dir, "model_last.pt"))
    if not memory_efficient:
        # Guard: if validation loss never improved (e.g. NaN/Inf from the start),
        # best_state is None. Fall back to the current model state rather than
        # crashing with a TypeError. This mirrors the same guard in train_euclideanizer.py.
        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
    model_path = os.path.join(model_dir, "model.pt")
    if not memory_efficient:
        torch.save(best_state, model_path)
    last_epochs = (start_epoch_offset + len(train_loss_hist)) if stopped_early else dm_cfg["epochs"]
    save_run_config(
        {"distmap": dm_cfg}, model_dir,
        last_epoch_trained=last_epochs, best_epoch=best_epoch, best_val=best_val,
        early_stopped=stopped_early,
    )
    print(f"  Saved: {utils.display_path(model_path, display_root)}")
    # Delete previous segment's last only after writing current last (fallback if best save was interrupted)
    if save_last and is_resume and prev_run_dir is not None and not save_final_models_per_stretch:
        prev_last = os.path.join(prev_run_dir, "model", "model_last.pt")
        if os.path.isfile(prev_last):
            try:
                os.remove(prev_last)
            except OSError:
                pass
    if plot_loss:
        loss_dir = os.path.join(output_dir, "plots", "loss_curves")
        os.makedirs(loss_dir, exist_ok=True)
        plot_loss_curves(
            train_loss_hist, val_loss_hist,
            os.path.join(loss_dir, "loss_curves.png"),
            title=f"DistMap Loss (beta_kl={beta_kl})",
            dpi=plot_dpi,
            save_pdf=save_pdf,
            save_plot_data=save_plot_data,
            display_root=display_root,
        )
    return model_path, stopped_early