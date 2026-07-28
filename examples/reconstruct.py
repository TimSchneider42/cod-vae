#!/usr/bin/env python3
"""
Encode a mesh into COD-VAE latents and decode it back into a mesh.

The model can be loaded from a Hugging Face Hub repository id, a local npz file, or a
directory containing an official COD-VAE release (config.yaml + *.pt), e.g.:

    python reconstruct.py bunny.obj reconstruction.obj --model user/cod-vae
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from cod_vae import CODVAE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mesh", type=Path)
    parser.add_argument("output_mesh", type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face repo id, npz file, or official release directory",
    )
    parser.add_argument("--backend", choices=("auto", "torch", "jax"), default="auto")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vae = CODVAE.from_pretrained(args.model, backend=args.backend)
    print(f"Loaded {vae!r} using the {vae.backend} backend")

    mesh = trimesh.load(args.input_mesh, force="mesh")
    latent, transform = vae.encode_mesh(
        mesh, num_points=args.num_points, seed=args.seed, return_transform=True
    )
    print(f"Encoded {args.input_mesh.name} into a {latent.shape} latent")

    reconstruction = vae.decode_mesh(
        latent, resolution=args.resolution, transform=transform
    )
    reconstruction.export(args.output_mesh)
    print(
        f"Wrote {args.output_mesh} ({len(reconstruction.vertices)} vertices, "
        f"{len(reconstruction.faces)} faces)"
    )


if __name__ == "__main__":
    main()
