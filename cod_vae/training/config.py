from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TrainingConfig"]


@dataclass(frozen=True)
class TrainingConfig:
    """
    Hyperparameters of COD-VAE training, mirroring the reference implementation's
    two-stage recipe:

    Stage 1 trains the autoencoder (encoder, triplane decoder, occupancy head) with the
    occupancy reconstruction loss on both the refined and the initial prediction plus a
    supervision loss for the uncertainty head. Stage 2 freezes the autoencoder and
    trains the latent VAE modules (latent_proj_in/out, latent_decoder) with a feature
    matching loss, the occupancy reconstruction loss through the frozen decoder, and a
    KL term.

    The learning rate is scaled by effective_batch_size / base_batch_size, where the
    effective batch size is batch_size (per device) times the number of devices times
    accumulate_grad_batches.
    """

    stage: int = 1
    epochs: int = 100
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    lr: float = 1e-4
    base_batch_size: int = 256
    weight_decay: float = 0.01
    grad_clip: float = 0.5
    seed: int = 0

    # Stage 1 loss coefficients
    vol_coeff: float = 1.0
    near_coeff: float = 0.1
    init_coeff: float = 1.0
    uncertainty_coeff: float = 0.01
    uncertainty_range: tuple[float, float] = (0.01, 1.0)

    # Stage 2 loss coefficients. Note: the reference configuration nominally sets the
    # KL coefficient to 1e-3, but applies it twice, so the effective value is 1e-6.
    feat_coeff: float = 1.0
    recon_coeff: float = 1.0
    kl_coeff: float = 1e-6
    # Stage 2 learning rate schedule: multiply by lr_decay at these epochs.
    lr_milestones: tuple[int, ...] = (60, 70, 80, 90)
    lr_decay: float = 0.5

    log_every: int = 50

    def scaled_lr(self, num_devices: int) -> float:
        effective = self.batch_size * num_devices * self.accumulate_grad_batches
        return self.lr * effective / self.base_batch_size
