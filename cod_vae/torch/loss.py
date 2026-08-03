"""Public COD-VAE loss functions (PyTorch)."""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = ["occupancy_loss"]


def occupancy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_vol: int,
    vol_coeff: float = 1.0,
    near_coeff: float = 0.1,
) -> torch.Tensor:
    """
    COD-VAE's occupancy reconstruction loss: binary cross entropy of decoded occupancy
    logits (B, N) against ground-truth labels (B, N), where the first ``num_vol``
    queries of each sample are the uniform-volume ones (weight ``vol_coeff``) and the
    remainder are near-surface (weight ``near_coeff``). Returns the per-sample loss
    (B,); the training loss is its mean.
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return vol_coeff * bce[:, :num_vol].mean(dim=-1) + near_coeff * bce[
        :, num_vol:
    ].mean(dim=-1)
