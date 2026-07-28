"""
Command line entry points.

cod-vae-convert: convert an official COD-VAE release directory into the self-contained
npz format (and optionally push it to the Hugging Face Hub).

cod-vae-train: train COD-VAE on a directory of watertight meshes. For multi-GPU
training with the torch backend, launch via ``torchrun --nproc_per_node=<n> -m
cod_vae.cli train ...``; the jax backend uses all visible GPUs from a single process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MESH_SUFFIXES = {".obj", ".off", ".ply", ".stl", ".glb", ".gltf"}


def convert_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert an official COD-VAE release directory (config.yaml + *.pt) "
        "into the self-contained npz format."
    )
    parser.add_argument("weights_dir", type=Path, help="release directory")
    parser.add_argument("output", type=Path, help="output npz path")
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="additionally upload the result to this Hugging Face Hub repository",
    )
    args = parser.parse_args(argv)

    from .checkpoint import load_torch_release, save_npz

    config, params = load_torch_release(args.weights_dir)
    save_npz(args.output, config, params)
    print(f"Wrote {args.output} ({len(params)} arrays)")
    if args.push_to_hub:
        from .hub import push_to_hub

        url = push_to_hub(args.push_to_hub, config, params)
        print(f"Pushed to {url}")


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train COD-VAE on a directory of watertight meshes."
    )
    parser.add_argument("mesh_dir", type=Path, help="directory containing mesh files")
    parser.add_argument("out_dir", type=Path, help="output directory for checkpoints")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--backend", choices=("torch", "jax"), default="torch",
        help="training backend (default: torch)",
    )
    parser.add_argument(
        "--init-from", type=Path,
        help="npz checkpoint to initialize from (required for stage 2: the trained "
        "stage-1 autoencoder)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=16, help="dataset repeat factor per epoch")
    parser.add_argument("--num-latents", type=int, default=32)
    parser.add_argument(
        "--cache-dir", type=Path, help="cache directory for precomputed occupancy pools"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="dataloader workers (torch backend)"
    )
    args = parser.parse_args(argv)

    from .checkpoint import load_npz
    from .config import CODVAEConfig
    from .training import MeshOccupancyDataset, TrainingConfig

    mesh_files = sorted(
        path for path in args.mesh_dir.rglob("*") if path.suffix.lower() in MESH_SUFFIXES
    )
    if not mesh_files:
        print(f"No mesh files found in {args.mesh_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Training on {len(mesh_files)} meshes from {args.mesh_dir}")

    params = None
    if args.init_from is not None:
        config, params = load_npz(args.init_from)
    else:
        if args.stage == 2:
            print("--init-from is required for stage 2", file=sys.stderr)
            sys.exit(1)
        config = CODVAEConfig(num_latents=args.num_latents)

    train_config = TrainingConfig(
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    dataset = MeshOccupancyDataset(
        mesh_files, repeat=args.repeat, cache_dir=args.cache_dir, seed=args.seed
    )

    if args.backend == "torch":
        from .torch.training import train

        train(
            config, train_config, dataset, params=params, out_dir=args.out_dir,
            num_workers=args.num_workers,
        )
    else:
        from .jax.training import train

        train(config, train_config, dataset, params=params, out_dir=args.out_dir)


def main(argv: list[str] | None = None) -> None:
    """Dispatcher so that ``python -m cod_vae.cli {convert,train} ...`` works."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("convert", "train"):
        print("usage: python -m cod_vae.cli {convert,train} ...", file=sys.stderr)
        sys.exit(2)
    if argv[0] == "convert":
        convert_main(argv[1:])
    else:
        train_main(argv[1:])


if __name__ == "__main__":
    main()
