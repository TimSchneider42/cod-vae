#!/usr/bin/env python3
"""
End-to-end training demo: train a small COD-VAE from scratch on a few geometric
primitives (stage 1: autoencoder, stage 2: latent VAE) and reconstruct one of them.

This demonstrates the training API on toy data; real training uses the full-size
default config, a large mesh dataset, and many epochs (see the README). Select the
backend with --backend; the torch backend supports multi-GPU via
``torchrun --nproc_per_node=<n> examples/train_primitives.py --backend torch``, the jax
backend uses all visible devices automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from cod_vae import CODVAE, CODVAEConfig
from cod_vae.training import MeshOccupancyDataset, SdfGenSettings, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "jax"), default="torch")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    config = CODVAEConfig(
        latent_dim=16,
        num_latents=16,
        embed_dim=128,
        query_dim=16,
        num_latent_layers=4,
        encoder_num_patches=128,
        encoder_num_blocks=2,
        encoder_num_layers_per_block=2,
        decoder_output_resolution=64,
        decoder_output_patch_size=8,
        decoder_num_layers=4,
        decoder_num_init_layers=1,
        decoder_num_merged_tokens=8,
    )
    meshes = [
        trimesh.creation.box(extents=[1.0, 0.6, 0.4]),
        trimesh.creation.icosphere(subdivisions=3, radius=0.5),
        trimesh.creation.cylinder(radius=0.3, height=1.0),
        trimesh.creation.capsule(radius=0.25, height=0.7),
    ]
    dataset = MeshOccupancyDataset(
        meshes,
        pc_size=1024,
        num_vol_queries=1024,
        num_near_queries=1024,
        repeat=16,
        settings=SdfGenSettings(
            num_vol=50_000, num_surface=25_000, watertight_resolution=10_000
        ),
    )
    print("Precomputing occupancy pools...")
    dataset.precompute(verbose=True)

    if args.backend == "torch":
        from cod_vae.torch.training import train
    else:
        from cod_vae.jax.training import train

    print("=== Stage 1: autoencoder ===")
    stage1 = TrainingConfig(
        stage=1,
        epochs=args.epochs,
        batch_size=8,
        lr=1e-3,
        base_batch_size=8,
        log_every=20,
    )
    params = train(config, stage1, dataset, out_dir=args.out_dir / "stage1")

    print("=== Stage 2: latent VAE ===")
    stage2 = TrainingConfig(
        stage=2,
        epochs=args.epochs,
        batch_size=8,
        lr=1e-3,
        base_batch_size=8,
        log_every=20,
    )
    params = train(
        config, stage2, dataset, params=params, out_dir=args.out_dir / "stage2"
    )

    vae = CODVAE(config, params, backend=args.backend)
    latent, transform = vae.encode_mesh(
        meshes[0], num_points=1024, return_transform=True
    )
    reconstruction = vae.decode_mesh(latent, resolution=64, transform=transform)
    output = args.out_dir / "reconstruction.obj"
    reconstruction.export(output)
    print(f"Wrote {output} ({len(reconstruction.vertices)} vertices)")


if __name__ == "__main__":
    main()
