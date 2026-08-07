#!/usr/bin/env python3
"""
Train COD-VAE on the preprocessed ShapeNet dataset of 3DShape2VecSet, following the
original training recipe (see TRAINING.md in the repository root for the full guide).

Multi-GPU:
  torch backend: torchrun --nproc_per_node=<n> examples/train_shapenet.py --stage 1 ...
  jax backend:   python examples/train_shapenet.py --backend jax --stage 1 ...
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import replace
from pathlib import Path

from cod_vae import CODVAEConfig, load_npz
from cod_vae.init import adapt_params
from cod_vae.training import ShapeNetVecSetDataset, TrainingConfig


def _parse_arch(entries: list[str]) -> dict:
    """Parse --arch FIELD=VALUE overrides, typed against CODVAEConfig's fields."""
    fields = {f.name: f.type for f in dataclasses.fields(CODVAEConfig)}
    overrides = {}
    for entry in entries:
        name, _, value = entry.partition("=")
        if name not in fields:
            raise SystemExit(f"--arch: {name} is not a CODVAEConfig field")
        overrides[name] = float(value) if "float" in str(fields[name]) else int(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        help="dataset root containing ShapeNetV2_point/ and ShapeNetV2_surface/",
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--backend", choices=("torch", "jax"), default="torch")
    parser.add_argument(
        "--init-from",
        type=Path,
        help="stage-1 npz checkpoint (required for stage 2; optional resume for stage 1)",
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        default=None,
        help="number of latent tokens (default 32: vae_m32; 64 for vae_m64). The "
        "autoencoder is trained through this many tokens, so it cannot be changed "
        "between the stages of one model",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="width of each latent vector (default 32). Only the latent VAE's "
        "projections depend on it, and stage 1 never trains those, so several stage-2 "
        "models with different widths can share one stage-1 checkpoint",
    )
    parser.add_argument(
        "--arch",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a CODVAEConfig field (e.g. --arch embed_dim=256 --arch "
        "num_heads=4). The autoencoder's architecture is baked into checkpoints, so "
        "with --init-from only fields that shape the latent VAE modules (latent_dim, "
        "num_latent_layers, latent_mlp_ratio) may differ from the checkpoint's",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="per-device batch size (default: 32 for stage 1, 128 for stage 2)",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=None,
        help="gradient accumulation steps (default: 2 for stage 1, 1 for stage 2)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=16,
        help="how often the dataset is repeated per epoch (the reference uses 16)",
    )
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from the last completed epoch in out_dir "
        "(torch backend)",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="allow TF32 matmuls/convolutions on Ampere+ GPUs: faster than float32 "
        "and still more precise than the reference's 16-mixed precision (torch backend)",
    )
    args = parser.parse_args()
    if (args.resume or args.tf32) and args.backend != "torch":
        parser.error("--resume and --tf32 are only supported by the torch backend")

    arch = _parse_arch(args.arch)
    if args.latent_dim is not None:
        arch["latent_dim"] = args.latent_dim
    params = None
    if args.init_from is not None:
        config, params = load_npz(args.init_from)
        if args.num_latents is not None and args.num_latents != config.num_latents:
            parser.error(
                f"{args.init_from} was trained with num_latents={config.num_latents}, "
                f"not {args.num_latents}: the autoencoder attends from that many tokens, "
                f"so a different count needs its own stage-1 run"
            )
        mismatched = {
            name: value
            for name, value in arch.items()
            if getattr(config, name) != value
        }
        # Fields that only shape the latent VAE modules may differ from the checkpoint:
        # stage 1 never trains those, so they are drawn fresh (adapt_params) and the
        # autoencoder carries over untouched. Everything else is baked in by stage 1.
        latent_only = {"latent_dim", "num_latent_layers", "latent_mlp_ratio"}
        fixed = {k: v for k, v in mismatched.items() if k not in latent_only}
        if fixed:
            parser.error(
                f"{args.init_from} was trained with a different architecture than "
                f"--arch requests ({fixed}); these fields are fixed by the stage-1 run"
            )
        if mismatched:
            config = replace(config, **mismatched)
            params, reinitialized = adapt_params(params, config, seed=args.seed)
            print(
                f"adapting checkpoint to {mismatched}: re-initialized "
                f"{len(reinitialized)} parameters ({', '.join(reinitialized)})"
            )
    else:
        if args.stage == 2:
            parser.error("--init-from is required for stage 2")
        config = CODVAEConfig(
            num_latents=args.num_latents if args.num_latents is not None else 32,
            latent_dim=args.latent_dim if args.latent_dim is not None else 32,
            **arch,
        )

    train_config = TrainingConfig(
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size or (32 if args.stage == 1 else 128),
        accumulate_grad_batches=args.accumulate or (2 if args.stage == 1 else 1),
        seed=args.seed,
    )
    dataset = ShapeNetVecSetDataset(
        args.root_dir, split="train", repeat=args.repeat, seed=args.seed
    )
    print(
        f"Stage {args.stage} on {len(dataset)} samples/epoch "
        f"({len(dataset.items)} shapes x repeat {args.repeat}), "
        f"{config.num_latents} x {config.latent_dim} latents"
    )

    if args.backend == "torch":
        import torch

        from cod_vae.torch.training import train

        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        train(
            config,
            train_config,
            dataset,
            params=params,
            out_dir=args.out_dir,
            num_workers=args.num_workers,
            resume=args.resume,
        )
    else:
        from cod_vae.jax.training import train

        train(config, train_config, dataset, params=params, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
