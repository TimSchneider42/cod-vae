"""Public COD-VAE loss functions (PyTorch)."""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = ["occupancy_loss", "sdf_loss", "sdf_target"]


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


def sdf_target(sdf: torch.Tensor, truncation: float) -> torch.Tensor:
    """
    The decoder's regression target for signed distances (negative inside): the
    truncated distance in units of the truncation, with the sign flipped so the output
    stays positive inside -- the same convention as an occupancy logit, so the zero
    level set, ``logits > 0`` metrics and meshing apply unchanged.
    """
    return (-sdf / truncation).clamp(-1.0, 1.0)


def sdf_loss(
    logits: torch.Tensor,
    sdf: torch.Tensor,
    num_vol: int,
    truncation: float,
    vol_coeff: float = 1.0,
    near_coeff: float = 0.1,
) -> torch.Tensor:
    """
    The truncated signed distance counterpart of :func:`occupancy_loss`: L1 between the
    decoder outputs (B, N) and :func:`sdf_target` of the ground-truth distances (B, N),
    with the same volume / near-surface split and weights. Returns the per-sample loss
    (B,).
    """
    error = (logits - sdf_target(sdf, truncation)).abs()
    return vol_coeff * error[:, :num_vol].mean(dim=-1) + near_coeff * error[
        :, num_vol:
    ].mean(dim=-1)
