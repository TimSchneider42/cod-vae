"""
JAX training of COD-VAE (stage 1: autoencoder, stage 2: latent VAE), following the
reference implementation's recipe (losses, coefficients, and schedules; see
:class:`cod_vae.training.TrainingConfig`). Requires optax (``pip install cod-vae[train]``).

Multi-GPU training is data-parallel over all (or the given) local devices: the batch is
sharded across devices and parameters are replicated, so a single process drives all
GPUs; batch_size is per device. In stage 2 the frozen autoencoder runs
deterministically (see note in :mod:`cod_vae.torch.training`).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

try:
    import optax
except ImportError as e:
    raise ImportError(
        "Training requires optax; install it via pip install cod-vae[train]"
    ) from e

from ..checkpoint import Params, save_npz
from ..config import CODVAEConfig
from ..init import LATENT_PREFIXES, init_params
from ..training.config import TrainingConfig
from ..training.data import MeshOccupancyDataset, iterate_batches
from .loss import bce_with_logits, occupancy_loss
from .model import (
    DropPath,
    decode_embed,
    decode_latents,
    decode_logits,
    decode_uncertainty,
    encode_embed,
    encode_moments,
)

__all__ = ["train"]


def _occupancy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_vol: int,
    vol_coeff: float,
    near_coeff: float,
) -> jnp.ndarray:
    return occupancy_loss(logits, labels, num_vol, vol_coeff, near_coeff).mean()


def _stage1_loss(
    trainable: Params,
    frozen: Params,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    *,
    config: CODVAEConfig,
    train_config: TrainingConfig,
    num_vol: int,
):
    cfg = train_config
    params = {**trainable, **frozen}
    dp = DropPath(config.droppath_rate, rng)
    z = encode_embed(params, batch["surface"], config=config, dp=dp)
    planes, init_planes, uncertainty_planes = decode_embed(
        params, z, config=config, dp=dp
    )
    logits = decode_logits(params, planes, batch["queries"], config=config)
    init_logits = decode_logits(params, init_planes, batch["queries"], config=config)

    recon_loss = _occupancy_loss(
        logits, batch["labels"], num_vol, cfg.vol_coeff, cfg.near_coeff
    )
    init_loss = _occupancy_loss(
        init_logits, batch["labels"], num_vol, cfg.vol_coeff, cfg.near_coeff
    )

    # The uncertainty head learns to predict the (normalized) error of the initial
    # occupancy prediction at each query point.
    uncertainty = decode_uncertainty(uncertainty_planes, batch["queries"])
    start, end = cfg.uncertainty_range
    query_loss = bce_with_logits(init_logits, batch["labels"])
    target = jax.lax.stop_gradient(
        jnp.clip(query_loss - start, 0.0, end - start) / (end - start)
    )
    uncertainty_loss = jnp.mean((uncertainty - target) ** 2)

    loss = (
        recon_loss
        + cfg.init_coeff * init_loss
        + cfg.uncertainty_coeff * uncertainty_loss
    )
    return loss, {
        "recon_loss": recon_loss,
        "init_loss": init_loss,
        "uncertainty_loss": uncertainty_loss,
    }


def _stage2_loss(
    trainable: Params,
    frozen: Params,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    *,
    config: CODVAEConfig,
    train_config: TrainingConfig,
    num_vol: int,
):
    cfg = train_config
    params = {**trainable, **frozen}
    droppath_rng, noise_rng = jax.random.split(rng)
    z_enc = jax.lax.stop_gradient(encode_embed(params, batch["surface"], config=config))

    moments = encode_moments(params, z_enc)
    mean, logvar = jnp.split(moments, 2, axis=-1)
    logvar = jnp.clip(logvar, -30.0, 20.0)
    std = jnp.exp(0.5 * logvar)
    z = mean + std * jax.random.normal(noise_rng, std.shape, dtype=std.dtype)

    dp = DropPath(config.droppath_rate, droppath_rng)
    z_recon = decode_latents(params, z, config=config, dp=dp)
    target = jax.lax.stop_gradient(_layer_norm_no_affine(z_enc))
    feat_loss = jnp.mean((z_recon - target) ** 2)

    planes = decode_embed(params, z_recon, config=config, aux=False)[0]
    logits = decode_logits(params, planes, batch["queries"], config=config)
    recon_loss = _occupancy_loss(
        logits, batch["labels"], num_vol, cfg.vol_coeff, cfg.near_coeff
    )

    var = jnp.exp(logvar)
    kl_loss = 0.5 * jnp.mean(mean**2 + var - 1.0 - logvar)

    loss = (
        cfg.feat_coeff * feat_loss
        + cfg.recon_coeff * recon_loss
        + cfg.kl_coeff * kl_loss
    )
    return loss, {"feat_loss": feat_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}


def _layer_norm_no_affine(x: jnp.ndarray) -> jnp.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)


def _split_params(params: Params, stage: int) -> tuple[dict, dict]:
    is_trainable = (
        (lambda key: key.startswith(LATENT_PREFIXES))
        if stage == 2
        else (lambda key: not key.startswith(LATENT_PREFIXES))
    )
    trainable = {k: v for k, v in params.items() if is_trainable(k)}
    frozen = {k: v for k, v in params.items() if not is_trainable(k)}
    # The point embedding basis is a constant buffer, never trained.
    if "autoencoder.point_embed.basis" in trainable:
        frozen["autoencoder.point_embed.basis"] = trainable.pop(
            "autoencoder.point_embed.basis"
        )
    return trainable, frozen


def train(
    config: CODVAEConfig,
    train_config: TrainingConfig,
    dataset: MeshOccupancyDataset,
    params: Params | None = None,
    out_dir: Path | str | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> Params:
    """
    Train COD-VAE and return the resulting parameters as a flat numpy dict.

    For stage 2, ``params`` must contain the trained stage-1 autoencoder weights (all
    parameters are loaded; the autoencoder is frozen). If ``out_dir`` is given, a
    checkpoint ("checkpoint_epoch_*.npz" and "checkpoint_last.npz") is written after
    every epoch. Training is data-parallel across all (or the given) local devices.
    """
    if devices is None:
        devices = jax.devices()
    mesh = Mesh(np.array(devices), ("data",))
    batch_sharding = NamedSharding(mesh, PartitionSpec("data"))
    replicated = NamedSharding(mesh, PartitionSpec())

    if params is None:
        params = init_params(config, seed=train_config.seed)
    trainable, frozen = _split_params(params, train_config.stage)
    trainable = jax.device_put(
        {k: jnp.asarray(v, jnp.float32) for k, v in trainable.items()}, replicated
    )
    frozen = jax.device_put(
        {k: jnp.asarray(v, jnp.float32) for k, v in frozen.items()}, replicated
    )

    global_batch = train_config.batch_size * len(devices)
    steps_per_epoch = len(dataset) // global_batch
    if steps_per_epoch == 0:
        raise ValueError(
            f"Dataset ({len(dataset)} items) is smaller than the global batch size "
            f"({global_batch})."
        )

    lr = train_config.scaled_lr(len(devices))
    if train_config.stage == 2 and train_config.lr_milestones:
        schedule = optax.piecewise_constant_schedule(
            lr,
            {
                epoch * steps_per_epoch: train_config.lr_decay
                for epoch in train_config.lr_milestones
            },
        )
    else:
        schedule = lr
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip),
        optax.adamw(schedule, weight_decay=train_config.weight_decay),
    )
    if train_config.accumulate_grad_batches > 1:
        optimizer = optax.MultiSteps(optimizer, train_config.accumulate_grad_batches)
    opt_state = jax.device_put(optimizer.init(trainable), replicated)

    loss_fn = _stage1_loss if train_config.stage == 1 else _stage2_loss
    loss_fn = partial(
        loss_fn,
        config=config,
        train_config=train_config,
        num_vol=dataset.num_vol_queries,
    )

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(trainable, opt_state, frozen, batch, rng):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            trainable, frozen, batch, rng
        )
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, opt_state, loss, metrics

    base_rng = jax.random.key(train_config.seed)
    global_step = 0
    for epoch in range(train_config.epochs):
        for step, batch in enumerate(
            iterate_batches(dataset, global_batch, epoch, seed=train_config.seed)
        ):
            batch = jax.device_put(
                {k: jnp.asarray(v) for k, v in batch.items()}, batch_sharding
            )
            step_rng = jax.random.fold_in(base_rng, global_step)
            trainable, opt_state, loss, metrics = train_step(
                trainable, opt_state, frozen, batch, step_rng
            )
            if step % train_config.log_every == 0:
                parts = ", ".join(
                    f"{key}={float(value):.4f}" for key, value in metrics.items()
                )
                print(f"epoch {epoch} step {step}: loss={float(loss):.4f} ({parts})")
            global_step += 1
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            result = _merge_params(trainable, frozen)
            save_npz(out_dir / f"checkpoint_epoch_{epoch:04d}.npz", config, result)
            save_npz(out_dir / "checkpoint_last.npz", config, result)

    return _merge_params(trainable, frozen)


def _merge_params(trainable: dict, frozen: dict) -> Params:
    merged = {**trainable, **frozen}
    return {key: np.asarray(value, dtype=np.float32) for key, value in merged.items()}
