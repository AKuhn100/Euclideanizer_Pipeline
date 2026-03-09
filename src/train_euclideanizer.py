"""
Train a single Euclideanizer with given config (frozen DistMap).
"""
from __future__ import annotations

import os
import time
import torch
import shutil

from . import utils
from .config import load_run_config, save_run_config
from .euclideanizer.model import Euclideanizer, load_frozen_vae
from .euclideanizer.loss import euclideanizer_loss
from .plotting import plot_loss_curves


def train_euclideanizer(
    eu_cfg: dict,
    device: torch.device,
    coords: torch.Tensor,
    frozen_vae_path: str,
    output_dir: str,
    *,
    split_seed: int,
    training_split: float,
    frozen_latent_dim: int,
    epoch_callback=None,
    plot_loss: bool = True,
    plot_dpi: int = 150,
    save_pdf: bool = True,
    save_plot_data: bool = False,
    resume_from_path: str | None = None,
    additional_epochs: int | None = None,
    prev_run_dir: str | None = None,
    save_final_models_per_stretch: bool = False,
    is_last_segment: bool = False,
    memory_efficient: bool = False,
    display_root: str | None = None,
) -> str:
    """
    Train one Euclideanizer. When resuming (resume_from_path + additional_epochs): load from
    prev run's euclideanizer_last.pt (last epoch of previous segment), carry over best from prev
    segment; save best (euclideanizer.pt) and last (euclideanizer_last.pt). Returns path to euclideanizer.pt.
    """
    num_atoms = coords.size(1)
    latent_dim = frozen_latent_dim
    batch_size = eu_cfg["batch_size"]
    lr = eu_cfg["learning_rate"]
    lambda_mse = eu_cfg["lambda_mse"]
    lambda_w_recon = eu_cfg["lambda_w_recon"]
    lambda_w_gen = eu_cfg["lambda_w_gen"]
    lambda_w_diag_recon = eu_cfg["lambda_w_diag_recon"]
    lambda_w_diag_gen = eu_cfg["lambda_w_diag_gen"]
    num_diags = eu_cfg["num_diags"]
    is_resume = resume_from_path is not None and additional_epochs is not None

    if is_resume:
        epochs = additional_epochs
    else:
        epochs = eu_cfg["epochs"]

    frozen_vae = load_frozen_vae(frozen_vae_path, num_atoms, latent_dim, device)
    embed = Euclideanizer(num_atoms=num_atoms).to(device)
    if is_resume:
        embed.load_state_dict(torch.load(resume_from_path, map_location=device))
    optimizer = torch.optim.Adam(embed.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_ds, test_ds = utils.get_train_test_split(coords, training_split, split_seed)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    loss_kwargs = dict(
        lambda_mse=lambda_mse,
        lambda_w_recon=lambda_w_recon,
        lambda_w_gen=lambda_w_gen,
        lambda_w_diag_recon=lambda_w_diag_recon,
        lambda_w_diag_gen=lambda_w_diag_gen,
        num_diags=num_diags,
    )
    train_loss_hist, val_loss_hist = [], []
    best_val = float("inf")
    best_state = None
    best_epoch = None

    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    start_epoch_offset = 0  # global epoch offset when resuming (best_epoch if resuming from best, else last_epoch_trained)
    if is_resume and prev_run_dir is not None:
        prev_model_dir = os.path.join(prev_run_dir, "model")
        prev_cfg = load_run_config(prev_model_dir)
        prev_best = os.path.join(prev_model_dir, "euclideanizer.pt")
        resume_from_best = os.path.normpath(resume_from_path) == os.path.normpath(prev_best)
        if prev_cfg is not None:
            best_val = prev_cfg.get("best_val", float("inf"))
            best_epoch = prev_cfg.get("best_epoch")
            start_epoch_offset = prev_cfg.get("best_epoch", 0) if resume_from_best else prev_cfg.get("last_epoch_trained", 0)
        if os.path.isfile(prev_best):
            shutil.copy2(prev_best, os.path.join(model_dir, "euclideanizer.pt"))
            if not memory_efficient:
                best_state = {k: v.clone() for k, v in torch.load(prev_best, map_location="cpu").items()}
        save_run_config({"euclideanizer": eu_cfg}, model_dir, last_epoch_trained=0, best_epoch=best_epoch, best_val=best_val)
    else:
        save_run_config({"euclideanizer": eu_cfg}, model_dir, last_epoch_trained=0, best_epoch=None)

    train_start = time.time()
    for epoch in range(epochs):
        embed.train()
        ep_loss, n_b = 0.0, 0
        for batch in train_dl:
            B = batch.shape[0]
            batch_dm = utils.get_distmaps(batch)
            gt_log = torch.log1p(batch_dm)
            with torch.no_grad():
                mu = frozen_vae.encode(gt_log)
                D_noneuclid = frozen_vae._decode_to_matrix(mu)
            D_euclid_recon = embed.forward_to_distmap(D_noneuclid)
            z_gen = torch.randn(B, latent_dim, device=device)
            with torch.no_grad():
                D_noneuclid_gen = frozen_vae._decode_to_matrix(z_gen)
            D_euclid_gen = embed.forward_to_distmap(D_noneuclid_gen)
            loss, *_ = euclideanizer_loss(gt_log, D_euclid_recon, D_euclid_gen, **loss_kwargs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embed.parameters(), max_norm=10.0)
            optimizer.step()
            ep_loss += loss.item()
            n_b += 1
        avg_train = ep_loss / n_b
        scheduler.step()

        embed.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in test_dl:
                B = batch.shape[0]
                batch_dm = utils.get_distmaps(batch)
                gt_log = torch.log1p(batch_dm)
                mu = frozen_vae.encode(gt_log)
                D_noneuclid = frozen_vae._decode_to_matrix(mu)
                D_euclid_recon = embed.forward_to_distmap(D_noneuclid)
                z_gen = torch.randn(B, latent_dim, device=device)
                D_noneuclid_gen = frozen_vae._decode_to_matrix(z_gen)
                D_euclid_gen = embed.forward_to_distmap(D_noneuclid_gen)
                loss, *_ = euclideanizer_loss(gt_log, D_euclid_recon, D_euclid_gen, **loss_kwargs)
                val_sum += loss.item()
                val_n += 1
        avg_val = val_sum / val_n
        train_loss_hist.append(avg_train)
        val_loss_hist.append(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = start_epoch_offset + epoch + 1  # 1-indexed global
            if memory_efficient:
                torch.save(embed.state_dict(), os.path.join(model_dir, "euclideanizer.pt"))
                save_run_config({"euclideanizer": eu_cfg}, model_dir, last_epoch_trained=0, best_epoch=best_epoch, best_val=best_val)
            else:
                best_state = {k: v.cpu().clone() for k, v in embed.state_dict().items()}
        if epoch_callback is not None:
            epoch_callback(epoch + 1, embed, train_loss_hist, val_loss_hist, run_dirs=[output_dir])
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = (time.time() - train_start) / 60
            print(f"  Epoch {epoch+1}/{epochs} Train: {avg_train:.6f} Val: {avg_val:.6f} (elapsed {elapsed:.1f}m)" + (" *" if avg_val <= best_val else ""))

    elapsed = (time.time() - train_start) / 60
    print(f"  Euclideanizer training finished in {elapsed:.1f}m.")
    save_last = (not is_last_segment) or save_final_models_per_stretch
    if save_last:
        last_state = {k: v.cpu().clone() for k, v in embed.state_dict().items()}
        torch.save(last_state, os.path.join(model_dir, "euclideanizer_last.pt"))
    if not memory_efficient:
        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in embed.state_dict().items()}
        embed.load_state_dict(best_state)
        torch.save(best_state, os.path.join(model_dir, "euclideanizer.pt"))
    model_path = os.path.join(model_dir, "euclideanizer.pt")
    save_run_config(
        {"euclideanizer": eu_cfg}, model_dir,
        last_epoch_trained=eu_cfg["epochs"], best_epoch=best_epoch, best_val=best_val,
    )
    print(f"  Saved: {utils.display_path(model_path, display_root)}")
    if save_last and is_resume and prev_run_dir is not None and not save_final_models_per_stretch:
        prev_last = os.path.join(prev_run_dir, "model", "euclideanizer_last.pt")
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
            title="Euclideanizer Training Loss",
            dpi=plot_dpi,
            save_pdf=save_pdf,
            save_plot_data=save_plot_data,
            display_root=display_root,
        )
    return model_path
