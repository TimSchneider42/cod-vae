"""
PyTorch training of COD-VAE (stage 1: autoencoder, stage 2: latent VAE), following the
reference implementation's recipe (losses, coefficients, and schedules; see
:class:`cod_vae.training.TrainingConfig`).

Multi-GPU training uses standard DistributedDataParallel: launch the training script
with ``torchrun --nproc_per_node=<num_gpus>`` and this module picks up the distributed
environment automatically; batch_size is per process.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from ..checkpoint import Params, load_npz, save_npz
from ..config import CODVAEConfig
from ..init import LATENT_PREFIXES, init_params
from ..training.config import TrainingConfig
from ..training.data import MeshOccupancyDataset
from .loss import occupancy_loss
from .model import CODVAEModule

__all__ = ["train"]


def _occupancy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_vol: int,
    vol_coeff: float,
    near_coeff: float,
) -> torch.Tensor:
    return occupancy_loss(logits, labels, num_vol, vol_coeff, near_coeff).mean()


class _LossModule(nn.Module):
    """Wraps the model so its forward computes the training loss (required for DDP)."""

    def __init__(
        self, module: CODVAEModule, train_config: TrainingConfig, num_vol_queries: int
    ):
        super().__init__()
        self.module = module
        self.train_config = train_config
        self.num_vol_queries = num_vol_queries

    def forward(
        self,
        surface: torch.Tensor,
        queries: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.train_config.stage == 1:
            outputs = self._stage1(surface, queries, labels)
        else:
            outputs = self._stage2(surface, queries, labels)
        if self.train_config.distill_coeff and teacher_logits is not None:
            cfg = self.train_config
            # Soft-target distillation: the same occupancy BCE, but against the
            # teacher's probabilities (optionally softened by the temperature) at the
            # very same query points. BCE against soft targets is still a valid cross
            # entropy, so the existing loss applies unchanged.
            soft = torch.sigmoid(teacher_logits / cfg.distill_temperature)
            distill_loss = _occupancy_loss(
                outputs["logits"], soft, self.num_vol_queries,
                cfg.vol_coeff, cfg.near_coeff,
            )
            outputs["loss"] = outputs["loss"] + cfg.distill_coeff * distill_loss
            outputs["distill_loss"] = distill_loss.detach()
        outputs.pop("logits", None)
        return outputs

    def _stage1(self, surface, queries, labels):
        cfg = self.train_config
        z = self.module.encode_embed(surface)
        planes, init_planes, uncertainty_planes = self.module.decode_embed(z)
        logits = self.module.decode_logits(planes, queries)
        init_logits = self.module.decode_logits(init_planes, queries)

        recon_loss = _occupancy_loss(
            logits, labels, self.num_vol_queries, cfg.vol_coeff, cfg.near_coeff
        )
        init_loss = _occupancy_loss(
            init_logits, labels, self.num_vol_queries, cfg.vol_coeff, cfg.near_coeff
        )

        # The uncertainty head learns to predict the (normalized) error of the initial
        # occupancy prediction at each query point.
        uncertainty = self.module.decode_uncertainty(uncertainty_planes, queries)
        start, end = cfg.uncertainty_range
        query_loss = F.binary_cross_entropy_with_logits(
            init_logits, labels, reduction="none"
        )
        target = ((query_loss - start).clamp(0, end - start) / (end - start)).detach()
        uncertainty_loss = F.mse_loss(uncertainty, target)

        loss = (
            recon_loss
            + cfg.init_coeff * init_loss
            + cfg.uncertainty_coeff * uncertainty_loss
        )
        return {
            "loss": loss,
            "logits": logits,
            "recon_loss": recon_loss.detach(),
            "init_loss": init_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
        }

    def _stage2(self, surface, queries, labels):
        cfg = self.train_config
        with torch.no_grad():
            z_enc = self.module.encode_embed(surface)

        moments = self.module.encode_moments(z_enc)
        mean, logvar = torch.chunk(moments, 2, dim=-1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        z_recon = self.module.decode_latents(z)
        target = F.layer_norm(z_enc, z_enc.shape[-1:]).detach()
        feat_loss = F.mse_loss(z_recon, target)

        planes = self.module.decode_embed(z_recon, aux=False)[0]
        logits = self.module.decode_logits(planes, queries)
        recon_loss = _occupancy_loss(
            logits, labels, self.num_vol_queries, cfg.vol_coeff, cfg.near_coeff
        )

        var = torch.exp(logvar)
        kl_loss = 0.5 * torch.mean(mean**2 + var - 1.0 - logvar)

        loss = (
            cfg.feat_coeff * feat_loss
            + cfg.recon_coeff * recon_loss
            + cfg.kl_coeff * kl_loss
        )
        return {
            "loss": loss,
            "logits": logits,
            "feat_loss": feat_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }


def _set_trainable(module: CODVAEModule, stage: int) -> None:
    for name, parameter in module.named_parameters():
        is_latent = name.startswith(LATENT_PREFIXES)
        parameter.requires_grad_(is_latent if stage == 2 else not is_latent)


def train(
    config: CODVAEConfig,
    train_config: TrainingConfig,
    dataset: MeshOccupancyDataset,
    params: Params | None = None,
    out_dir: Path | str | None = None,
    device: str | None = None,
    num_workers: int = 0,
    resume: bool = False,
) -> Params:
    """
    Train COD-VAE and return the resulting parameters as a flat numpy dict.

    For stage 2, ``params`` must contain the trained stage-1 autoencoder weights (all
    parameters are loaded; the autoencoder is frozen). If ``out_dir`` is given, a
    checkpoint ("checkpoint_epoch_*.npz" and "checkpoint_last.npz") is written after
    every epoch, together with the optimizer state ("train_state_last.pt"); with
    ``resume``, training continues from that state instead of starting over, which is
    what an interrupted run (job time limit, node failure) needs. Runs under torchrun
    for multi-GPU data parallelism.
    """
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if device is None:
        if torch.cuda.is_available():
            device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        else:
            device = "cpu"
    device = torch.device(device)
    if distributed and not torch.distributed.is_initialized():
        # The backend must match the device actually used for training, not CUDA
        # availability: CPU training on a CUDA-capable machine requires gloo.
        torch.distributed.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo"
        )
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(train_config.seed + rank)
    module = CODVAEModule(config).to(device)
    if params is None:
        params = init_params(config, seed=train_config.seed)
    module.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in params.items()})
    _set_trainable(module, train_config.stage)
    module.train()
    if train_config.stage == 2:
        # Run the frozen autoencoder deterministically (no stochastic depth). Note that
        # the reference implementation keeps the whole model in train mode here, so its
        # frozen autoencoder is still subject to DropPath noise during stage 2.
        module.autoencoder.eval()

    loss_module = _LossModule(module, train_config, dataset.num_vol_queries)
    model: nn.Module = loss_module
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            loss_module, device_ids=[device] if device.type == "cuda" else None
        )

    trainable = [p for p in module.parameters() if p.requires_grad]
    lr = train_config.scaled_lr(world_size)
    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=train_config.weight_decay
    )
    scheduler = None
    if train_config.stage == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(train_config.lr_milestones),
            gamma=train_config.lr_decay,
        )

    start_epoch = 0
    if resume and out_dir is not None:
        state_path = Path(out_dir) / "train_state_last.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            # The state refers to the per-epoch checkpoint written just before it, so a
            # crash while writing "checkpoint_last.npz" cannot make the two disagree.
            checkpoint = Path(out_dir) / f"checkpoint_epoch_{state['epoch']:04d}.npz"
            module.load_state_dict(
                {
                    key: torch.from_numpy(value.copy())
                    for key, value in load_npz(checkpoint)[1].items()
                }
            )
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state["scheduler"] is not None:
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            if rank == 0:
                print(f"resuming from {checkpoint} at epoch {start_epoch}")

    sampler = (
        DistributedSampler(dataset, seed=train_config.seed) if distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
        num_workers=num_workers,
    )

    for epoch in range(start_epoch, train_config.epochs):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            updating = (step + 1) % train_config.accumulate_grad_batches == 0
            # Gradients of the micro-batches in between are only accumulated locally;
            # all-reducing them too would move the same 750 MB twice per update for a
            # mathematically identical result.
            sync = (
                contextlib.nullcontext()
                if updating or not distributed
                else model.no_sync()
            )
            with sync:
                outputs = model(
                    batch["surface"],
                    batch["queries"],
                    batch["labels"],
                    teacher_logits=batch.get("teacher_logits"),
                )
                loss = outputs["loss"] / train_config.accumulate_grad_batches
                loss.backward()
            if updating:
                torch.nn.utils.clip_grad_norm_(trainable, train_config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if rank == 0 and step % train_config.log_every == 0:
                parts = ", ".join(
                    f"{key}={float(value):.4f}"
                    for key, value in outputs.items()
                    if key != "loss"
                )
                print(
                    f"epoch {epoch} step {step}: "
                    f"loss={float(outputs['loss'].detach()):.4f} ({parts})"
                )
        if scheduler is not None:
            scheduler.step()
        if out_dir is not None and rank == 0:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            result = _module_params(module)
            save_npz(out_dir / f"checkpoint_epoch_{epoch:04d}.npz", config, result)
            save_npz(out_dir / "checkpoint_last.npz", config, result)
            state = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            # Written last and renamed atomically: a "train_state_last.pt" that exists
            # always belongs to a complete checkpoint.
            state_tmp = out_dir / "train_state_last.pt.tmp"
            torch.save(state, state_tmp)
            state_tmp.rename(out_dir / "train_state_last.pt")
        if distributed:
            torch.distributed.barrier()

    return _module_params(module)


def _module_params(module: CODVAEModule) -> Params:
    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in module.state_dict().items()
    }
