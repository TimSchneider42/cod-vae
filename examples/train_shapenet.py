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
from pathlib import Path

from cod_vae import CODVAEConfig, load_npz
from cod_vae.training import ShapeNetVecSetDataset, TrainingConfig


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
        "--num-latents", type=int, default=32, help="32 for vae_m32, 64 for vae_m64"
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
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    params = None
    if args.init_from is not None:
        config, params = load_npz(args.init_from)
    else:
        if args.stage == 2:
            parser.error("--init-from is required for stage 2")
        config = CODVAEConfig(num_latents=args.num_latents)

    train_config = TrainingConfig(
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size or (32 if args.stage == 1 else 128),
        accumulate_grad_batches=args.accumulate or (2 if args.stage == 1 else 1),
        seed=args.seed,
    )
    dataset = ShapeNetVecSetDataset(
        args.root_dir, split="train", repeat=16, seed=args.seed
    )
    print(
        f"Stage {args.stage} on {len(dataset)} samples/epoch "
        f"({len(dataset.items)} shapes x repeat 16)"
    )

    if args.backend == "torch":
        from cod_vae.torch.training import train

        train(
            config,
            train_config,
            dataset,
            params=params,
            out_dir=args.out_dir,
            num_workers=args.num_workers,
        )
    else:
        from cod_vae.jax.training import train

        train(config, train_config, dataset, params=params, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
