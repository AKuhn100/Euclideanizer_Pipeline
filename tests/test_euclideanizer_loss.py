"""
Unit tests for Euclideanizer loss (including Kabsch-aligned coordinate MSE).

Run from pipeline root: pytest tests/test_euclideanizer_loss.py -v
"""
from __future__ import annotations

import torch

from src.euclideanizer.loss import (
    calc_MSE_kabsch,
    euclideanizer_loss,
    kabsch_align,
)


def test_kabsch_align_shape():
    """kabsch_align returns (aligned, gt_centered) with same shapes as input."""
    B, N = 4, 10
    recon = torch.randn(B, N, 3)
    gt = torch.randn(B, N, 3)
    aligned, gt_c = kabsch_align(recon, gt)
    assert aligned.shape == (B, N, 3)
    assert gt_c.shape == (B, N, 3)


def test_calc_MSE_kabsch_scalar():
    """calc_MSE_kabsch returns a scalar."""
    B, N = 4, 10
    gt_coords = torch.randn(B, N, 3)
    recon_coords = torch.randn(B, N, 3)
    mse = calc_MSE_kabsch(gt_coords, recon_coords)
    assert mse.ndim == 0
    assert mse.item() >= 0


def test_euclideanizer_loss_with_kabsch_backward():
    """Full loss with lambda_kabsch_mse and coords has grad and backward() runs."""
    B, N = 4, 10
    device = torch.device("cpu")
    gt_coords = torch.randn(B, N, 3, device=device)
    recon_coords = torch.randn(B, N, 3, device=device, requires_grad=True)
    gts = torch.rand(B, N, N, device=device) * 0.5 + 1.0
    D_recon = gts.clone() + 0.1 * torch.randn(B, N, N, device=device, requires_grad=True)
    D_gen = gts.clone() + 0.2 * torch.randn(B, N, N, device=device)

    loss, mse, rw, gw, drw, dgw, kabsch_mse = euclideanizer_loss(
        gts,
        D_recon,
        D_gen,
        lambda_mse=1.0,
        lambda_w_recon=0.1,
        lambda_w_gen=0.1,
        lambda_w_diag_recon=0.0,
        lambda_w_diag_gen=0.0,
        num_diags=0,
        gt_coords=gt_coords,
        recon_coords=recon_coords,
        lambda_kabsch_mse=1.0,
    )
    assert loss.requires_grad
    loss.backward()
    # recon_coords is leaf; D_recon is non-leaf (gts.clone() + ...) so .grad is not populated
    assert recon_coords.grad is not None


def test_euclideanizer_loss_without_kabsch():
    """Loss without coords / lambda_kabsch_mse=0 still returns 7 values."""
    B, N = 4, 10
    gts = torch.rand(B, N, N) * 0.5 + 1.0
    D_recon = gts.clone() + 0.1 * torch.randn(B, N, N)
    D_gen = gts.clone() + 0.2 * torch.randn(B, N, N)
    out = euclideanizer_loss(
        gts, D_recon, D_gen,
        1.0, 0.1, 0.1, 0.0, 0.0, 0,
        lambda_kabsch_mse=0.0,
    )
    assert len(out) == 7
    assert out[6].item() == 0.0  # kabsch_mse
