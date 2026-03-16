#!/usr/bin/env python3
"""
Batch size and learning rate sweep for DistMap + Euclideanizer pipeline.

For each specified batch size, sweeps the specified learning rates (and vice versa).
Trains for a fixed number of epochs per (batch_size, learning_rate) and reports:
  - Wall-clock time per epoch
  - Samples per second
  - Final validation loss
  - Peak reserved VRAM

Modes (--mode):
  - dm:  Benchmark DistMap only (batch-size × learning-rate grid).
  - eu:  Train a DistMap for 50 epochs in a temporary directory, then run the
         Euclideanizer benchmark over batch sizes × learning rates; temp dir
         is purged at the end.
  - both: DistMap grid, then Euclideanizer grid (if no --dm-checkpoint, a
          quick 5-epoch DistMap is trained in a temp dir and purged).

Config: The benchmark does NOT run the full combinatorial grid of distmap/euclideanizer
options. For any key that is a list in the config, the first value is used. A warning
is printed when lists are present. To optimize for a different combination, use a
config with single values for the options you care about.

Usage:
    python benchmark_batch_size.py --config samples/config_sample.yaml --data /path/to/data.gro --mode dm
    python benchmark_batch_size.py --config samples/config_sample.yaml --data /path/to/data.gro --mode eu
    python benchmark_batch_size.py --config samples/config_sample.yaml --data /path/to/data.gro \
        --batch-sizes 32 64 128 --learning-rates 1e-4 5e-4 1e-3 --epochs 20 --mode both
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src import utils
from src.config import load_config, get_data_path
from src.distmap.model import ChromVAE_Conv
from src.distmap.loss import distmap_vae_loss
from src.euclideanizer.model import Euclideanizer, load_frozen_vae
from src.euclideanizer.loss import euclideanizer_loss


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model: str           # "distmap" or "euclideanizer"
    batch_size: int
    learning_rate: float
    epochs_run: int
    avg_epoch_wall_sec: float
    samples_per_sec: float
    final_train_loss: float
    final_val_loss: float
    peak_reserved_gb: float
    peak_allocated_gb: float
    oom: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_distmap_epoch(model, optimizer, train_dl, device, dm_cfg):
    """One training epoch for DistMap. Returns average loss and number of samples processed."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_atoms = None
    latent_dim = dm_cfg["latent_dim"]

    for batch in train_dl:
        batch = batch.to(device)
        if num_atoms is None:
            num_atoms = batch.size(1)
        batch_dm = utils.get_distmaps(batch)
        mu, logvar, z, recon_tri = model(batch_dm)
        gt_full = torch.log1p(batch_dm)
        recon_full = utils.upper_tri_to_symmetric(recon_tri, num_atoms)
        loss, *_ = distmap_vae_loss(
            mu, logvar, gt_full, recon_full,
            model, latent_dim, device,
            dm_cfg["beta_kl"], dm_cfg["lambda_mse"],
            dm_cfg["lambda_w_gen"], dm_cfg["lambda_w_recon"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
        total_samples += batch.size(0)

    return total_loss / max(total_samples, 1), total_samples


def _run_distmap_val(model, val_dl, device, dm_cfg, num_atoms):
    """One validation pass for DistMap. Returns average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    latent_dim = dm_cfg["latent_dim"]

    with torch.no_grad():
        for batch in val_dl:
            batch = batch.to(device)
            batch_dm = utils.get_distmaps(batch)
            mu, logvar, z, recon_tri = model(batch_dm)
            gt_full = torch.log1p(batch_dm)
            recon_full = utils.upper_tri_to_symmetric(recon_tri, num_atoms)
            loss, *_ = distmap_vae_loss(
                mu, logvar, gt_full, recon_full,
                model, latent_dim, device,
                dm_cfg["beta_kl"], dm_cfg["lambda_mse"],
                dm_cfg["lambda_w_gen"], dm_cfg["lambda_w_recon"],
            )
            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

    return total_loss / max(total_samples, 1)


def _run_euclideanizer_epoch(embed, frozen_vae, optimizer, train_dl, device, eu_cfg, latent_dim):
    """One training epoch for Euclideanizer. Returns average loss and samples processed."""
    embed.train()
    total_loss = 0.0
    total_samples = 0

    loss_kwargs = dict(
        lambda_mse=eu_cfg["lambda_mse"],
        lambda_w_recon=eu_cfg["lambda_w_recon"],
        lambda_w_gen=eu_cfg["lambda_w_gen"],
        lambda_w_diag_recon=eu_cfg["lambda_w_diag_recon"],
        lambda_w_diag_gen=eu_cfg["lambda_w_diag_gen"],
        num_diags=eu_cfg["num_diags"],
        lambda_kabsch_mse=eu_cfg["lambda_kabsch_mse"],
    )

    for batch in train_dl:
        batch = batch.to(device)
        B = batch.size(0)
        batch_dm = utils.get_distmaps(batch)
        gt_log = torch.log1p(batch_dm)
        with torch.no_grad():
            mu = frozen_vae.encode(gt_log)
            D_noneuclid = frozen_vae._decode_to_matrix(mu)
        recon_coords = embed(D_noneuclid)
        D_euclid_recon = torch.log1p(utils.get_distmaps(recon_coords))
        z_gen = torch.randn(B, latent_dim, device=device)
        with torch.no_grad():
            D_noneuclid_gen = frozen_vae._decode_to_matrix(z_gen)
        D_euclid_gen = embed.forward_to_distmap(D_noneuclid_gen)
        loss, *_ = euclideanizer_loss(
            gt_log, D_euclid_recon, D_euclid_gen,
            gt_coords=batch, recon_coords=recon_coords,
            **loss_kwargs,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embed.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item() * B
        total_samples += B

    return total_loss / max(total_samples, 1), total_samples


def _run_euclideanizer_val(embed, frozen_vae, val_dl, device, eu_cfg, latent_dim):
    """One validation pass for Euclideanizer. Returns average loss."""
    embed.eval()
    total_loss = 0.0
    total_samples = 0

    loss_kwargs = dict(
        lambda_mse=eu_cfg["lambda_mse"],
        lambda_w_recon=eu_cfg["lambda_w_recon"],
        lambda_w_gen=eu_cfg["lambda_w_gen"],
        lambda_w_diag_recon=eu_cfg["lambda_w_diag_recon"],
        lambda_w_diag_gen=eu_cfg["lambda_w_diag_gen"],
        num_diags=eu_cfg["num_diags"],
        lambda_kabsch_mse=eu_cfg["lambda_kabsch_mse"],
    )

    with torch.no_grad():
        for batch in val_dl:
            batch = batch.to(device)
            B = batch.size(0)
            batch_dm = utils.get_distmaps(batch)
            gt_log = torch.log1p(batch_dm)
            mu = frozen_vae.encode(gt_log)
            D_noneuclid = frozen_vae._decode_to_matrix(mu)
            recon_coords = embed(D_noneuclid)
            D_euclid_recon = torch.log1p(utils.get_distmaps(recon_coords))
            z_gen = torch.randn(B, latent_dim, device=device)
            D_noneuclid_gen = frozen_vae._decode_to_matrix(z_gen)
            D_euclid_gen = embed.forward_to_distmap(D_noneuclid_gen)
            loss, *_ = euclideanizer_loss(
                gt_log, D_euclid_recon, D_euclid_gen,
                gt_coords=batch, recon_coords=recon_coords,
                **loss_kwargs,
            )
            total_loss += loss.item() * B
            total_samples += B

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def benchmark_distmap(
    batch_size: int,
    n_epochs: int,
    coords: torch.Tensor,
    coords_np: np.ndarray,
    dm_cfg: dict,
    device: torch.device,
    training_split: float,
    split_seed: int,
) -> BenchmarkResult:
    num_atoms = coords.size(1)
    train_ds, val_ds = utils.get_train_test_split(coords, training_split, split_seed)

    if len(train_ds) < batch_size:
        return BenchmarkResult(
            model="distmap", batch_size=batch_size, learning_rate=dm_cfg["learning_rate"],
            epochs_run=0, avg_epoch_wall_sec=0, samples_per_sec=0,
            final_train_loss=float("nan"), final_val_loss=float("nan"),
            peak_reserved_gb=0, peak_allocated_gb=0, oom=False,
            notes=f"Skipped: train set ({len(train_ds)}) smaller than batch_size ({batch_size})",
        )

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ChromVAE_Conv(num_atoms=num_atoms, latent_space_dim=dm_cfg["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=dm_cfg["learning_rate"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    train_losses = []
    val_losses = []

    try:
        # Warmup: one forward pass to trigger cuDNN algorithm selection
        with torch.no_grad():
            dummy = torch.rand(min(batch_size, 4), num_atoms, 3, device=device)
            dummy_dm = utils.get_distmaps(dummy)
            model(dummy_dm)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(n_epochs):
            t0 = time.time()
            train_loss, n_samples = _run_distmap_epoch(model, optimizer, train_dl, device, dm_cfg)
            val_loss = _run_distmap_val(model, val_dl, device, dm_cfg, num_atoms)
            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"    [DistMap B={batch_size}] epoch {epoch+1}/{n_epochs} "
                f"train={train_loss:.5f} val={val_loss:.5f} "
                f"time={elapsed:.1f}s"
            )

        avg_epoch = float(np.mean(epoch_times))
        # samples/sec: train samples processed per second, averaged over epochs
        train_size = len(train_ds) - (len(train_ds) % batch_size)  # drop_last
        samples_per_sec = train_size / avg_epoch

        peak_reserved_gb = 0.0
        peak_allocated_gb = 0.0
        if device.type == "cuda":
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
            peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3

        return BenchmarkResult(
            model="distmap",
            batch_size=batch_size,
            learning_rate=dm_cfg["learning_rate"],
            epochs_run=n_epochs,
            avg_epoch_wall_sec=avg_epoch,
            samples_per_sec=samples_per_sec,
            final_train_loss=train_losses[-1],
            final_val_loss=val_losses[-1],
            peak_reserved_gb=peak_reserved_gb,
            peak_allocated_gb=peak_allocated_gb,
            oom=False,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchmarkResult(
            model="distmap", batch_size=batch_size, learning_rate=dm_cfg["learning_rate"],
            epochs_run=len(epoch_times),
            avg_epoch_wall_sec=float(np.mean(epoch_times)) if epoch_times else 0,
            samples_per_sec=0, final_train_loss=float("nan"), final_val_loss=float("nan"),
            peak_reserved_gb=0, peak_allocated_gb=0, oom=True,
            notes="OOM during training",
        )
    finally:
        del model, optimizer
        torch.cuda.empty_cache() if device.type == "cuda" else None


def benchmark_euclideanizer(
    batch_size: int,
    n_epochs: int,
    coords: torch.Tensor,
    coords_np: np.ndarray,
    dm_checkpoint: str,
    dm_cfg: dict,
    eu_cfg: dict,
    device: torch.device,
    training_split: float,
    split_seed: int,
) -> BenchmarkResult:
    num_atoms = coords.size(1)
    latent_dim = dm_cfg["latent_dim"]
    train_ds, val_ds = utils.get_train_test_split(coords, training_split, split_seed)

    if len(train_ds) < batch_size:
        return BenchmarkResult(
            model="euclideanizer", batch_size=batch_size, learning_rate=eu_cfg["learning_rate"],
            epochs_run=0, avg_epoch_wall_sec=0, samples_per_sec=0,
            final_train_loss=float("nan"), final_val_loss=float("nan"),
            peak_reserved_gb=0, peak_allocated_gb=0, oom=False,
            notes=f"Skipped: train set ({len(train_ds)}) smaller than batch_size ({batch_size})",
        )

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    frozen_vae = load_frozen_vae(dm_checkpoint, num_atoms, latent_dim, device)
    embed = Euclideanizer(num_atoms=num_atoms).to(device)
    optimizer = torch.optim.Adam(embed.parameters(), lr=eu_cfg["learning_rate"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    train_losses = []
    val_losses = []

    try:
        # Warmup
        with torch.no_grad():
            dummy = torch.rand(min(batch_size, 4), num_atoms, 3, device=device)
            dummy_dm = utils.get_distmaps(dummy)
            gt_log = torch.log1p(dummy_dm)
            mu = frozen_vae.encode(gt_log)
            D_ne = frozen_vae._decode_to_matrix(mu)
            embed(D_ne)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(n_epochs):
            t0 = time.time()
            train_loss, n_samples = _run_euclideanizer_epoch(
                embed, frozen_vae, optimizer, train_dl, device, eu_cfg, latent_dim
            )
            val_loss = _run_euclideanizer_val(embed, frozen_vae, val_dl, device, eu_cfg, latent_dim)
            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"    [Euclideanizer B={batch_size}] epoch {epoch+1}/{n_epochs} "
                f"train={train_loss:.5f} val={val_loss:.5f} "
                f"time={elapsed:.1f}s"
            )

        avg_epoch = float(np.mean(epoch_times))
        train_size = len(train_ds) - (len(train_ds) % batch_size)
        samples_per_sec = train_size / avg_epoch

        peak_reserved_gb = 0.0
        peak_allocated_gb = 0.0
        if device.type == "cuda":
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
            peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3

        return BenchmarkResult(
            model="euclideanizer",
            batch_size=batch_size,
            learning_rate=eu_cfg["learning_rate"],
            epochs_run=n_epochs,
            avg_epoch_wall_sec=avg_epoch,
            samples_per_sec=samples_per_sec,
            final_train_loss=train_losses[-1],
            final_val_loss=val_losses[-1],
            peak_reserved_gb=peak_reserved_gb,
            peak_allocated_gb=peak_allocated_gb,
            oom=False,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchmarkResult(
            model="euclideanizer", batch_size=batch_size, learning_rate=eu_cfg["learning_rate"],
            epochs_run=len(epoch_times),
            avg_epoch_wall_sec=float(np.mean(epoch_times)) if epoch_times else 0,
            samples_per_sec=0, final_train_loss=float("nan"), final_val_loss=float("nan"),
            peak_reserved_gb=0, peak_allocated_gb=0, oom=True,
            notes="OOM during training",
        )
    finally:
        del embed, optimizer, frozen_vae
        torch.cuda.empty_cache() if device.type == "cuda" else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: list[BenchmarkResult]) -> None:
    if not results:
        return

    model_name = results[0].model
    lrs = sorted(set(r.learning_rate for r in results))
    multi_lr = len(lrs) > 1
    print(f"\n{'='*100}")
    print(f"  {model_name.upper()} BENCHMARK RESULTS")
    print(f"{'='*100}")
    header = (
        f"  {'B':>6}  {'LR':>10}  {'Epoch(s)':>9}  {'Samples/s':>10}  "
        f"{'Train Loss':>11}  {'Val Loss':>10}  "
        f"{'Reserved(GB)':>13}  {'Alloc(GB)':>10}  {'Status':>8}"
    )
    if not multi_lr:
        header = header.replace("  {'LR':>10}  ", "  ")
    print(header)
    sep = "  " + "-"*6 + "  " + ("-"*10 + "  " if multi_lr else "") + "-"*9 + "  " + "-"*10 + "  " + "-"*11 + "  " + "-"*10 + "  " + "-"*13 + "  " + "-"*10 + "  " + "-"*8
    print(sep)

    for r in results:
        status = "OOM" if r.oom else ("SKIP" if r.epochs_run == 0 else "OK")
        lr_str = f"{r.learning_rate:>10.2e}  " if multi_lr else ""
        row = (
            f"  {r.batch_size:>6}  {lr_str}{r.avg_epoch_wall_sec:>8.1f}s  {r.samples_per_sec:>10.1f}  "
            f"{r.final_train_loss:>11.5f}  {r.final_val_loss:>10.5f}  "
            f"{r.peak_reserved_gb:>13.2f}  {r.peak_allocated_gb:>10.2f}  {status:>8}"
        )
        print(row)
        if r.notes:
            print(f"  {'':>6}  {'':>10}  {r.notes}" if multi_lr else f"  {'':>6}  {r.notes}")

    # Find efficiency sweet spot: best samples/sec before val loss degrades >10% from minimum
    valid = [r for r in results if not r.oom and r.epochs_run > 0 and r.samples_per_sec > 0]
    if len(valid) >= 2:
        min_val_loss = min(r.final_val_loss for r in valid)
        threshold = min_val_loss * 1.10  # 10% degradation tolerance
        candidates = [r for r in valid if r.final_val_loss <= threshold]
        if candidates:
            best = max(candidates, key=lambda r: r.samples_per_sec)
            print(f"\n  Suggested: batch_size={best.batch_size}" + (f" learning_rate={best.learning_rate:.2e}" if multi_lr else ""))
            print(f"  (Highest throughput within 10% of best validation loss)")
            best_val = min(valid, key=lambda r: r.final_val_loss)
            print(f"  Best val loss: {min_val_loss:.5f} at B={best_val.batch_size}" + (f" lr={best_val.learning_rate:.2e}" if multi_lr else ""))
            print(f"  Suggested throughput: {best.samples_per_sec:.1f} samples/s")
            print(f"  Suggested VRAM reserved: {best.peak_reserved_gb:.2f} GB")

    print(f"{'='*100}\n")


def _save_results(results: list[BenchmarkResult], output_path: str) -> None:
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch size efficiency benchmark for DistMap + Euclideanizer."
    )
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--data", default=None, help="Path to NPZ dataset (overrides config data.path)")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512],
        help="Batch sizes to benchmark (default: 8 16 32 64 128 256 512)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Epochs to run at each batch size (default: 20)",
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=None,
        help="Learning rates to sweep (default: from config, one value per model). E.g. 1e-4 5e-4 1e-3. For each batch size we run each learning rate.",
    )
    parser.add_argument(
        "--mode", choices=["dm", "eu", "both"], default="both",
        help="dm = DistMap benchmark only; eu = train DistMap 50 epochs in temp dir, then Euclideanizer benchmark only (temp purged); both = DistMap sweep then Euclideanizer sweep (default: both)",
    )
    parser.add_argument(
        "--model", choices=["distmap", "euclideanizer", "both"], default="both",
        help="Deprecated: use --mode. Which model to benchmark (default: both). Ignored if --mode is set.",
    )
    parser.add_argument(
        "--dm-checkpoint", type=str, default=None,
        help="Path to trained DistMap checkpoint for Euclideanizer benchmark. "
             "If not provided and mode=eu or both, a DistMap is trained first (eu: 50 epochs in temp; both: 5 epochs in temp). Temp dirs are purged after use.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0,
        help="Train/test split seed (default: 0)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Path to save JSON results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override, e.g. cuda:0 (default: auto-detect)",
    )
    args = parser.parse_args()

    # Load config and data
    overrides = {}
    if args.data:
        overrides["data"] = {"path": args.data}
    cfg = load_config(args.config, overrides)
    data_path = get_data_path(cfg)
    if not data_path:
        print("ERROR: No data path. Set --data or data.path in config.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device) if args.device else utils.get_device()
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}  ({props.total_memory / 1024**3:.1f} GB VRAM)")

    print(f"Loading data from: {data_path}")
    coords_np = utils.load_data(data_path)
    coords = torch.tensor(coords_np, dtype=torch.float32).to(device)
    num_atoms = coords.size(1)
    num_structures = len(coords_np)
    print(f"Loaded: {num_structures} structures, {num_atoms} atoms")

    dm_cfg_raw = cfg["distmap"]
    eu_cfg_raw = cfg["euclideanizer"]
    training_split = cfg["data"]["training_split"]
    split_seed = args.split_seed

    # Single combination only: take first value for every key that is a list (benchmark does not run combinatorial grid)
    dm_cfg = {}
    dm_list_keys = []
    for k, v in dm_cfg_raw.items():
        if isinstance(v, list):
            dm_cfg[k] = v[0]
            dm_list_keys.append(k)
        else:
            dm_cfg[k] = v
    eu_cfg = {}
    eu_list_keys = []
    for k, v in eu_cfg_raw.items():
        if isinstance(v, list):
            eu_cfg[k] = v[0]
            eu_list_keys.append(k)
        else:
            eu_cfg[k] = v

    if dm_list_keys or eu_list_keys:
        print("\n" + "!" * 72, file=sys.stderr)
        print("  WARNING: Config has multiple options (lists) for some keys.", file=sys.stderr)
        print("  The benchmark uses the FIRST value only; it does not run a combinatorial grid.", file=sys.stderr)
        if dm_list_keys:
            print(f"  DistMap keys reduced to first: {', '.join(dm_list_keys)}", file=sys.stderr)
        if eu_list_keys:
            print(f"  Euclideanizer keys reduced to first: {', '.join(eu_list_keys)}", file=sys.stderr)
        print("  Results apply only to this single training configuration.", file=sys.stderr)
        print("  To optimize for a different combination, use a config with single values for those keys.", file=sys.stderr)
        print("!" * 72 + "\n", file=sys.stderr)

    # Echo training parameters used (fixed; batch_size and learning_rate are swept by the benchmark)
    print("Training parameters used (fixed; batch_size and learning_rate are swept):")
    print(f"  DistMap:      latent_dim={dm_cfg.get('latent_dim')}, beta_kl={dm_cfg.get('beta_kl')}, epochs={dm_cfg.get('epochs')}, learning_rate={dm_cfg.get('learning_rate')}, lambda_mse={dm_cfg.get('lambda_mse')}, ...")
    print(f"  Euclideanizer: epochs={eu_cfg.get('epochs')}, learning_rate={eu_cfg.get('learning_rate')}, lambda_mse={eu_cfg.get('lambda_mse')}, ...")
    print()

    batch_sizes = sorted(set(args.batch_sizes))
    n_epochs = args.epochs
    all_results: list[BenchmarkResult] = []

    # Resolve mode: --mode takes precedence over deprecated --model
    mode = args.mode
    run_dm_bench = mode in ("dm", "both")
    run_eu_bench = mode in ("eu", "both")
    # For EU benchmark, when we train a feeder DistMap: eu mode = 50 epochs, both = 5 epochs
    dm_epochs_for_eu = 50 if mode == "eu" else 5

    # ---- DistMap benchmark ----
    if run_dm_bench:
        learning_rates_dm = args.learning_rates if args.learning_rates is not None else [dm_cfg["learning_rate"]]
        learning_rates_dm = sorted(set(learning_rates_dm))
        print(f"\nBenchmarking DistMap: batch_sizes={batch_sizes}, learning_rates={learning_rates_dm}")
        print(f"Epochs per run: {n_epochs}\n")
        dm_results = []
        for bs in batch_sizes:
            for lr in learning_rates_dm:
                dm_cfg_lr = {**dm_cfg, "learning_rate": lr}
                print(f"  → DistMap batch_size={bs} lr={lr:.2e}")
                result = benchmark_distmap(
                    batch_size=bs,
                    n_epochs=n_epochs,
                    coords=coords,
                    coords_np=coords_np,
                    dm_cfg=dm_cfg_lr,
                    device=device,
                    training_split=training_split,
                    split_seed=split_seed,
                )
                dm_results.append(result)
                all_results.append(result)
                if result.oom:
                    print(f"  OOM at batch_size={bs} lr={lr:.2e}")

        _print_table(dm_results)

    # ---- Euclideanizer benchmark ----
    if run_eu_bench:
        dm_checkpoint = args.dm_checkpoint
        tmp_dir = None

        if dm_checkpoint is None:
            from src.train_distmap import train_distmap
            tmp_dir = tempfile.mkdtemp(prefix="benchmark_dm_")
            print(f"\nNo --dm-checkpoint provided. Training a DistMap for {dm_epochs_for_eu} epochs in temp dir (will be purged)...")
            quick_cfg = {**dm_cfg, "epochs": dm_epochs_for_eu, "batch_size": min(64, len(coords_np) // 2)}
            dm_checkpoint, _ = train_distmap(
                quick_cfg, device, coords, tmp_dir,
                split_seed=split_seed, training_split=training_split,
                plot_loss=False, save_plot_data=False,
                memory_efficient=False, is_last_segment=True,
                calibration_safety_margin_gb=cfg["calibration_safety_margin_gb"],
                calibration_training_batch_cap=cfg["calibration_training_batch_cap"],
                calibration_binary_search_steps=cfg["calibration_binary_search_steps"],
            )
            print(f"  DistMap checkpoint: {dm_checkpoint}\n")

        try:
            learning_rates_eu = args.learning_rates if args.learning_rates is not None else [eu_cfg["learning_rate"]]
            learning_rates_eu = sorted(set(learning_rates_eu))
            print(f"\nBenchmarking Euclideanizer: batch_sizes={batch_sizes}, learning_rates={learning_rates_eu}")
            print(f"Epochs per run: {n_epochs}\n")
            eu_results = []
            for bs in batch_sizes:
                for lr in learning_rates_eu:
                    eu_cfg_lr = {**eu_cfg, "learning_rate": lr}
                    print(f"  → Euclideanizer batch_size={bs} lr={lr:.2e}")
                    result = benchmark_euclideanizer(
                        batch_size=bs,
                        n_epochs=n_epochs,
                        coords=coords,
                        coords_np=coords_np,
                        dm_checkpoint=dm_checkpoint,
                        dm_cfg=dm_cfg,
                        eu_cfg=eu_cfg_lr,
                        device=device,
                        training_split=training_split,
                        split_seed=split_seed,
                    )
                    eu_results.append(result)
                    all_results.append(result)
                    if result.oom:
                        print(f"  OOM at batch_size={bs} lr={lr:.2e}")

            _print_table(eu_results)
        finally:
            if tmp_dir is not None and os.path.isdir(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                    print(f"  Purged temp DistMap dir: {tmp_dir}")
                except OSError as e:
                    print(f"  WARNING: Could not remove temp dir {tmp_dir}: {e}", file=sys.stderr)

    # Save results
    _save_results(all_results, args.output)
    print("Done.")


if __name__ == "__main__":
    main()