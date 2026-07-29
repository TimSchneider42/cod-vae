"""
Command line entry points.

cod-vae-convert: convert an official COD-VAE release directory into the self-contained
npz format (and optionally push it to the Hugging Face Hub).

cod-vae-train: train COD-VAE, either directly on a directory of watertight meshes or
on a dataset built with cod-vae-dataset. For multi-GPU training with the torch
backend, launch via ``torchrun --nproc_per_node=<n> -m cod_vae.cli train ...``; the
jax backend uses all visible GPUs from a single process.

cod-vae-dataset: build a training dataset in the 3DShape2VecSet layout by merging any
number of sources — directories of (arbitrary, not necessarily watertight) mesh files,
Hugging Face mesh datasets in the Tactile MNIST format, and preprocessed
3DShape2VecSet roots — preprocessing all meshes with the original authors' sdf_gen
recipe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
        description="Train COD-VAE on a directory of watertight meshes or on a "
        "dataset built with cod-vae-dataset (a root containing ShapeNetV2_point/)."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="directory of watertight mesh files, or a dataset root built with "
        "cod-vae-dataset",
    )
    parser.add_argument("out_dir", type=Path, help="output directory for checkpoints")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--backend",
        choices=("torch", "jax"),
        default="torch",
        help="training backend (default: torch)",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        help="npz checkpoint to initialize from (required for stage 2: the trained "
        "stage-1 autoencoder)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size per device"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeat", type=int, default=16, help="dataset repeat factor per epoch"
    )
    parser.add_argument("--num-latents", type=int, default=32)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="cache directory for precomputed occupancy pools (mesh directories only)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="dataloader workers (torch backend)"
    )
    args = parser.parse_args(argv)

    from .checkpoint import load_npz
    from .config import CODVAEConfig
    from .training import MeshOccupancyDataset, ShapeNetVecSetDataset, TrainingConfig
    from .training.preprocess import MESH_SUFFIXES, POINT_DIR

    if (args.data_dir / POINT_DIR).is_dir():
        dataset = ShapeNetVecSetDataset(
            args.data_dir, split="train", repeat=args.repeat, seed=args.seed
        )
        print(
            f"Training on {len(dataset.items)} preprocessed shapes from "
            f"{args.data_dir}"
        )
    else:
        mesh_files = sorted(
            path
            for path in args.data_dir.rglob("*")
            if path.suffix.lower() in MESH_SUFFIXES
        )
        if not mesh_files:
            print(
                f"{args.data_dir} contains neither {POINT_DIR}/ nor mesh files",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Training on {len(mesh_files)} meshes from {args.data_dir}")
        dataset = MeshOccupancyDataset(
            mesh_files, repeat=args.repeat, cache_dir=args.cache_dir, seed=args.seed
        )

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

    if args.backend == "torch":
        from .torch.training import train

        train(
            config,
            train_config,
            dataset,
            params=params,
            out_dir=args.out_dir,
            num_workers=args.num_workers,
        )
    else:
        from .jax.training import train

        train(config, train_config, dataset, params=params, out_dir=args.out_dir)


def dataset_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge data sources into a training dataset in the 3DShape2VecSet "
        "directory layout, ready for cod-vae-train. Sources are directories of mesh "
        "files (meshes need not be watertight), Hugging Face mesh datasets in the "
        "Tactile MNIST format, and preprocessed 3DShape2VecSet roots (linked or "
        "copied as-is). All meshes are preprocessed with the original authors' "
        "sdf_gen recipe (https://github.com/1zb/sdf_gen), which includes "
        "watertighting. Requires cod-vae[preprocess]."
    )
    parser.add_argument("out_dir", type=Path, help="output dataset root")
    parser.add_argument(
        "--meshes",
        action="append",
        default=[],
        metavar="[NAME=]DIR",
        help="directory of mesh files (searched recursively; the meshes need not be "
        "watertight); becomes one category directory, named NAME (default: the "
        "directory name). Meshes in train/val/test subdirectories are assigned to "
        "the corresponding splits, otherwise everything becomes training data. May "
        "be given multiple times",
    )
    parser.add_argument(
        "--vecset",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="preprocessed 3DShape2VecSet root (ShapeNetV2_point + "
        "ShapeNetV2_surface) to merge; may be given multiple times",
    )
    parser.add_argument(
        "--hf",
        action="append",
        default=[],
        metavar="[NAME=]DATASET",
        help="Hugging Face mesh dataset in the Tactile MNIST format (Hub repository "
        "id or local path); becomes one category directory, named NAME (default: the "
        "last path component). May be given multiple times",
    )
    parser.add_argument(
        "--hf-split",
        action="append",
        metavar="SRC[=DST]",
        help="source split to include from every --hf dataset, mapped to the vecset "
        "split DST (train/val/test, default: SRC); may be given multiple times. "
        "Default: train/val(idation)/test, as far as present",
    )
    parser.add_argument(
        "--link",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="how to merge --vecset sources (default: symlink)",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=0,
        help="preprocess meshes in this many parallel processes (default: 0, "
        "in-process)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute existing outputs instead of resuming",
    )
    parser.add_argument(
        "--num-vol", type=int, default=250_000, help="volume query pool size"
    )
    parser.add_argument(
        "--num-surface",
        type=int,
        default=125_000,
        help="surface point pool size (the near-surface pool has "
        "num_surface * len(near_stddevs) points)",
    )
    parser.add_argument(
        "--near-stddev",
        action="append",
        type=float,
        metavar="STD",
        help="near-surface noise standard deviation; may be given multiple times "
        "(default: 0.005 and 0.05)",
    )
    parser.add_argument("--object-scale", type=float, default=0.9)
    parser.add_argument("--watertight-resolution", type=int, default=50_000)
    args = parser.parse_args(argv)

    from .training.preprocess import SdfGenSettings, build_vecset_dataset

    if not args.meshes and not args.vecset and not args.hf:
        parser.error("at least one source (--meshes, --vecset, or --hf) is required")

    def parse_source(spec: str) -> tuple[str, str]:
        if "=" in spec:
            name, source = spec.split("=", 1)
        else:
            source = spec
            name = Path(spec.rstrip("/")).name
        return name, source

    mesh_sources = [parse_source(spec) for spec in args.meshes]
    hf_sources = [parse_source(spec) for spec in args.hf]
    split_map = None
    if args.hf_split:
        split_map = {}
        for spec in args.hf_split:
            src, _, dst = spec.partition("=")
            split_map[src] = dst or src
    settings = SdfGenSettings(
        num_vol=args.num_vol,
        num_surface=args.num_surface,
        near_stddevs=tuple(args.near_stddev or (0.005, 0.05)),
        object_scale=args.object_scale,
        watertight_resolution=args.watertight_resolution,
    )
    build_vecset_dataset(
        args.out_dir,
        vecset_sources=args.vecset,
        mesh_sources=mesh_sources,
        hf_sources=hf_sources,
        link=args.link,
        split_map=split_map,
        settings=settings,
        seed=args.seed,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(f"Dataset written to {args.out_dir}")


def main(argv: list[str] | None = None) -> None:
    """Dispatcher so that ``python -m cod_vae.cli {convert,train,dataset} ...`` works."""
    argv = sys.argv[1:] if argv is None else argv
    commands = {"convert": convert_main, "train": train_main, "dataset": dataset_main}
    if not argv or argv[0] not in commands:
        print(
            "usage: python -m cod_vae.cli {convert,train,dataset} ...", file=sys.stderr
        )
        sys.exit(2)
    commands[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
