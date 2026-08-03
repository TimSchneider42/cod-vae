"""Public COD-VAE loss functions (JAX)."""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["bce_with_logits", "occupancy_loss"]


def bce_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable elementwise binary cross entropy with logits."""
    return (
        jnp.maximum(logits, 0.0)
        - logits * labels
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def occupancy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_vol: int,
    vol_coeff: float = 1.0,
    near_coeff: float = 0.1,
) -> jnp.ndarray:
    """
    COD-VAE's occupancy reconstruction loss: binary cross entropy of decoded occupancy
    logits (B, N) against ground-truth labels (B, N), where the first ``num_vol``
    queries of each sample are the uniform-volume ones (weight ``vol_coeff``) and the
    remainder are near-surface (weight ``near_coeff``). Returns the per-sample loss
    (B,); the training loss is its mean.
    """
    losses = bce_with_logits(logits, labels)
    return vol_coeff * losses[:, :num_vol].mean(axis=-1) + near_coeff * losses[
        :, num_vol:
    ].mean(axis=-1)
