"""
Backend-independent high-level interface of COD-VAE.

:class:`CODVAEBase` implements the public numpy/trimesh API on top of a small set of
abstract array operations that each backend (:mod:`cod_vae.torch`, :mod:`cod_vae.jax`)
provides. All public methods accept and return numpy arrays and trimesh meshes; inputs
may be batched or unbatched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import trimesh

from .checkpoint import Params
from .config import CODVAEConfig
from .mesh import (
    CubeTransform,
    grid_queries,
    normalize_to_cube,
    occupancy_grid_to_mesh,
    sample_surface_points,
)

__all__ = ["CODVAEBase"]


class CODVAEBase(ABC):
    """A COD-VAE model bound to a specific compute backend."""

    #: Name of the backend ("torch" or "jax"), set by subclasses.
    backend: str

    def __init__(self, config: CODVAEConfig, params: Params):
        self.config = config
        self._load_params(params)

    ## -- abstract backend operations (batched numpy in / numpy or handle out) --------

    @abstractmethod
    def _load_params(self, params: Params) -> None:
        """Transfer the flat parameter dict to the backend/device."""

    @abstractmethod
    def get_params(self) -> Params:
        """Return the current parameters as a flat numpy dict."""

    @abstractmethod
    def _encode(self, points: np.ndarray) -> np.ndarray:
        """Encode point clouds (B, N, 3) into posterior-mean latents (B, L, D)."""

    @abstractmethod
    def _decode_planes(self, latents: np.ndarray) -> Any:
        """Decode latents (B, L, D) into a backend-native triplane handle."""

    @abstractmethod
    def _decode_logits(self, planes: Any, queries: np.ndarray) -> np.ndarray:
        """Evaluate occupancy logits (B, N) at query points (B, N, 3)."""

    ## -- public numpy interface ------------------------------------------------------

    def encode(self, points: np.ndarray) -> np.ndarray:
        """
        Encode surface point clouds in the [-1, 1] cube into latents. Accepts (N, 3) or
        (B, N, 3) and returns (num_latents, latent_dim) or (B, num_latents, latent_dim)
        accordingly. The returned latent is the deterministic posterior mean.
        """
        points = np.asarray(points, dtype=np.float32)
        batched = points.ndim == 3
        latents = self._encode(points if batched else points[None])
        return latents if batched else latents[0]

    def decode(
        self, latents: np.ndarray, queries: np.ndarray, chunk_size: int = 65536
    ) -> np.ndarray:
        """
        Evaluate occupancy logits (positive inside the object) at query points in the
        [-1, 1] cube. latents: (L, D) or (B, L, D); queries: (N, 3) or (B, N, 3).
        """
        latents = np.asarray(latents, dtype=np.float32)
        queries = np.asarray(queries, dtype=np.float32)
        batched = latents.ndim == 3
        if not batched:
            latents = latents[None]
        if queries.ndim == 2:
            queries = np.broadcast_to(queries[None], (latents.shape[0], *queries.shape))
        planes = self._decode_planes(latents)
        logits = self._decode_logits_chunked(planes, queries, chunk_size)
        return logits if batched else logits[0]

    def decode_planes(self, latents: np.ndarray) -> Any:
        """
        Decode latents (B, num_latents, latent_dim) into a backend-native triplane
        handle for :meth:`decode_logits`. Splitting :meth:`decode` into these two steps
        avoids recomputing the (expensive) triplane decoding when the same latents are
        evaluated at multiple query sets.
        """
        latents = np.asarray(latents, dtype=np.float32)
        if latents.ndim != 3:
            raise ValueError(
                f"Expected batched latents of shape (B, num_latents, latent_dim), got "
                f"shape {latents.shape}."
            )
        return self._decode_planes(latents)

    def decode_logits(
        self, planes: Any, queries: np.ndarray, chunk_size: int = 65536
    ) -> np.ndarray:
        """
        Evaluate occupancy logits (B, N) (positive inside the object) at query points
        (B, N, 3) in the [-1, 1] cube given a triplane handle from
        :meth:`decode_planes`.
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 3:
            raise ValueError(
                f"Expected batched queries of shape (B, N, 3), got shape "
                f"{queries.shape}."
            )
        return self._decode_logits_chunked(planes, queries, chunk_size)

    def decode_volume(
        self,
        latents: np.ndarray,
        resolution: int | None = None,
        chunk_size: int = 65536,
    ) -> np.ndarray:
        """
        Decode latents into a dense occupancy logit grid of shape (resolution,) * 3
        (batched: (B, resolution, resolution, resolution)), indexed as [x, y, z] over a
        uniform grid covering [-1, 1]^3.
        """
        if resolution is None:
            resolution = self.config.decoder_output_resolution
        latents = np.asarray(latents, dtype=np.float32)
        batched = latents.ndim == 3
        logits = self.decode(latents, grid_queries(resolution), chunk_size=chunk_size)
        shape = (resolution,) * 3
        return logits.reshape((-1, *shape) if batched else shape)

    def _decode_logits_chunked(
        self, planes: Any, queries: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        num_queries = queries.shape[1]
        # Pad to a multiple of the chunk size so backends see a fixed shape (avoids
        # recompilation for jitted backends).
        padded = max(1, -(-num_queries // chunk_size)) * chunk_size
        queries = np.pad(queries, ((0, 0), (0, padded - num_queries), (0, 0)))
        logits = np.concatenate(
            [
                self._decode_logits(planes, queries[:, i : i + chunk_size])
                for i in range(0, padded, chunk_size)
            ],
            axis=1,
        )
        return logits[:, :num_queries]

    ## -- trimesh interface -----------------------------------------------------------

    def encode_mesh(
        self,
        mesh: trimesh.Trimesh | Sequence[trimesh.Trimesh],
        num_points: int = 2048,
        object_scale: float = 0.9,
        seed: int | None = None,
        return_transform: bool = False,
    ):
        """
        Encode meshes into latents by normalizing them into the model's [-1, 1] cube and
        sampling a surface point cloud. Accepts a single mesh or a sequence; returns
        (num_latents, latent_dim) or (B, num_latents, latent_dim). With
        return_transform=True additionally returns the cube transform(s), whose inverse
        maps decoded geometry back into the original mesh frame (see
        :meth:`decode_mesh`).
        """
        batched = not isinstance(mesh, trimesh.Trimesh)
        meshes = list(mesh) if batched else [mesh]
        points, transforms = [], []
        for m in meshes:
            normalized, transform = normalize_to_cube(m, object_scale)
            points.append(sample_surface_points(normalized, num_points, seed=seed))
            transforms.append(transform)
        latents = self._encode(np.stack(points))
        if not batched:
            latents, transforms = latents[0], transforms[0]
        return (latents, transforms) if return_transform else latents

    def decode_mesh(
        self,
        latents: np.ndarray,
        resolution: int | None = None,
        transform: CubeTransform | Sequence[CubeTransform] | None = None,
        chunk_size: int = 65536,
    ) -> trimesh.Trimesh | list[trimesh.Trimesh]:
        """
        Decode latents into meshes by extracting the zero level set of the dense
        occupancy grid. latents: (L, D) for a single mesh or (B, L, D) for a list. If
        the cube transform(s) from :meth:`encode_mesh` are given, the meshes are mapped
        back into the original frame.
        """
        latents = np.asarray(latents, dtype=np.float32)
        batched = latents.ndim == 3
        grids = self.decode_volume(latents, resolution, chunk_size=chunk_size)
        if not batched:
            return occupancy_grid_to_mesh(grids, transform)
        if transform is None:
            transform = [None] * len(grids)
        return [occupancy_grid_to_mesh(g, t) for g, t in zip(grids, transform)]

    ## -- persistence -----------------------------------------------------------------

    def save(self, path) -> None:
        """Save config and parameters to a self-contained npz file."""
        from .checkpoint import save_npz

        save_npz(path, self.config, self.get_params())

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        """Upload config and parameters to a Hugging Face Hub model repository."""
        from .hub import push_to_hub

        return push_to_hub(repo_id, self.config, self.get_params(), **kwargs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_latents={self.config.num_latents}, "
            f"latent_dim={self.config.latent_dim})"
        )
