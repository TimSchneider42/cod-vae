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
    pack_cube_transform,
    sample_surface_points,
    unpack_cube_transform,
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

    @abstractmethod
    def _decode_planes_full(self, full_latents: np.ndarray) -> Any:
        """Split full latents (B, F) and decode triplanes into a backend handle."""

    @abstractmethod
    def _decode_logits_full(
        self, handle: Any, queries: np.ndarray, object_scale: float
    ) -> np.ndarray:
        """
        Evaluate occupancy logits (B, N) at query points (B, N, 3) in the normalized
        world frame given a handle from :meth:`_decode_planes_full`.
        """

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
        logits = self._decode_logits_chunked(
            lambda chunk: self._decode_logits(planes, chunk), queries, chunk_size
        )
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
        return self._decode_logits_chunked(
            lambda chunk: self._decode_logits(planes, chunk), queries, chunk_size
        )

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
        self, decode_chunk, queries: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        num_queries = queries.shape[1]
        # Pad to a multiple of the chunk size so backends see a fixed shape (avoids
        # recompilation for jitted backends).
        padded = max(1, -(-num_queries // chunk_size)) * chunk_size
        queries = np.pad(queries, ((0, 0), (0, padded - num_queries), (0, 0)))
        logits = np.concatenate(
            [
                decode_chunk(queries[:, i : i + chunk_size])
                for i in range(0, padded, chunk_size)
            ],
            axis=1,
        )
        return logits[:, :num_queries]

    def occupancy_loss(
        self,
        latents: np.ndarray,
        queries: np.ndarray,
        labels: np.ndarray,
        num_vol: int,
        vol_coeff: float = 1.0,
        near_coeff: float = 0.1,
        chunk_size: int = 65536,
    ) -> np.ndarray:
        """
        Numpy wrapper of COD-VAE's occupancy reconstruction loss (see
        :func:`cod_vae.torch.occupancy_loss` / :func:`cod_vae.jax.occupancy_loss`):
        decodes occupancy logits of latents (B, L, D) at query points (B, N, 3) on the
        model's backend/device and computes the per-sample (B,) binary cross entropy
        against labels (B, N), where the first ``num_vol`` queries of each sample are
        the uniform-volume ones (weight ``vol_coeff``) and the remainder are
        near-surface (weight ``near_coeff``).
        """
        latents = np.asarray(latents, dtype=np.float32)
        queries = np.asarray(queries, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if latents.ndim != 3 or queries.ndim != 3 or labels.ndim != 2:
            raise ValueError(
                f"Expected batched latents (B, L, D), queries (B, N, 3), and labels "
                f"(B, N), got shapes {latents.shape}, {queries.shape}, and "
                f"{labels.shape}."
            )
        logits = self.decode(latents, queries, chunk_size=chunk_size)
        bce = (
            np.maximum(logits, 0.0)
            - logits * labels
            + np.log1p(np.exp(-np.abs(logits)))
        )
        return vol_coeff * bce[:, :num_vol].mean(axis=-1) + near_coeff * bce[
            :, num_vol:
        ].mean(axis=-1)

    ## -- full latent interface -------------------------------------------------------

    @property
    def full_latent_size(self) -> int:
        """Size of the flat full-latent vector: num_latents * latent_dim + 4."""
        return self.config.num_latents * self.config.latent_dim + 4

    def pack_full_latent(
        self,
        latents: np.ndarray,
        transform: CubeTransform | Sequence[CubeTransform],
        frame_half_size: float = 1.0,
        object_scale: float = 0.9,
    ) -> np.ndarray:
        """
        Pack latents ((L, D) or (B, L, D)) and the corresponding cube transform(s) into
        flat full-latent vectors [flattened latent, center (3), size (1)] of size
        :attr:`full_latent_size`. center is the bounding box center and size the
        maximum half-extent of the encoded geometry, both divided by
        ``frame_half_size``, i.e. expressed in a world frame normalized to [-1, 1] by
        ``frame_half_size``. A full latent contains everything needed to reconstruct
        geometry (see :meth:`decode_full` and :meth:`decode_mesh_full`); the transform
        parameters are stored in float32 like the latent.
        """
        latents = np.asarray(latents, dtype=np.float32)
        batched = latents.ndim == 3
        if not batched:
            latents, transform = latents[None], [transform]
        expected = (self.config.num_latents, self.config.latent_dim)
        if latents.ndim != 3 or latents.shape[1:] != expected:
            raise ValueError(
                f"Expected latents of shape (B,) + {expected}, got {latents.shape}."
            )
        if len(transform) != len(latents):
            raise ValueError(
                f"Got {len(latents)} latents but {len(transform)} transforms."
            )
        transforms = np.stack(
            [
                pack_cube_transform(
                    t, frame_half_size=frame_half_size, object_scale=object_scale
                )
                for t in transform
            ]
        ).astype(np.float32)
        full = np.concatenate([latents.reshape(len(latents), -1), transforms], axis=1)
        return full if batched else full[0]

    def unpack_full_latent(
        self,
        full_latents: np.ndarray,
        frame_half_size: float = 1.0,
        object_scale: float = 0.9,
    ) -> tuple[np.ndarray, CubeTransform | list[CubeTransform]]:
        """
        Inverse of :meth:`pack_full_latent`: split full latents ((full_latent_size,) or
        (B, full_latent_size)) into latents ((L, D) or (B, L, D)) and the cube
        transform(s). Sizes are clamped to 1e-3 (in normalized frame units) to guard
        against (near-)zero size values, which would yield an infinite cube scale.
        """
        full_latents = np.asarray(full_latents, dtype=np.float32)
        original_shape = full_latents.shape
        batched = full_latents.ndim == 2
        if not batched:
            full_latents = full_latents[None]
        if full_latents.ndim != 2 or full_latents.shape[1] != self.full_latent_size:
            raise ValueError(
                f"Expected full latents of shape (B, {self.full_latent_size}) or "
                f"({self.full_latent_size},), got shape {original_shape}."
            )
        dims = self.config.num_latents * self.config.latent_dim
        latents = full_latents[:, :dims].reshape(
            -1, self.config.num_latents, self.config.latent_dim
        )
        transforms = [
            unpack_cube_transform(
                row, frame_half_size=frame_half_size, object_scale=object_scale
            )
            for row in full_latents[:, dims:]
        ]
        return (latents, transforms) if batched else (latents[0], transforms[0])

    def encode_mesh_full(
        self,
        mesh: trimesh.Trimesh | Sequence[trimesh.Trimesh],
        num_points: int = 2048,
        object_scale: float = 0.9,
        seed: int | None = None,
        frame_half_size: float = 1.0,
    ) -> np.ndarray:
        """
        Encode meshes into full latents: the flattened COD-VAE latent followed by the
        bounding box center and size of the mesh, both normalized by
        ``frame_half_size`` (see :meth:`pack_full_latent`). Unlike the plain latent
        returned by :meth:`encode_mesh`, a full latent contains everything needed to
        reconstruct the mesh in its original frame; use :meth:`decode_mesh_full` or
        :meth:`decode_full` to do so. Returns (full_latent_size,) for a single mesh
        or (B, full_latent_size) for a sequence.
        """
        latents, transforms = self.encode_mesh(
            mesh,
            num_points=num_points,
            object_scale=object_scale,
            seed=seed,
            return_transform=True,
        )
        return self.pack_full_latent(
            latents,
            transforms,
            frame_half_size=frame_half_size,
            object_scale=object_scale,
        )

    def decode_full(
        self,
        full_latents: np.ndarray,
        queries: np.ndarray,
        chunk_size: int = 65536,
        object_scale: float = 0.9,
    ) -> np.ndarray:
        """
        Evaluate occupancy logits (positive inside the object) of full latents at query
        points given in the [-1, 1] normalized world frame (i.e. original coordinates
        divided by the ``frame_half_size`` the full latent was created with). The
        mapping into the model's cube happens in the backend, so it dispatches like
        :meth:`decode`; the differentiable backend-native variants are
        ``cod_vae.torch.CODVAEModule.decode_full`` and ``cod_vae.jax.decode_full``.
        full_latents: (full_latent_size,) or (B, full_latent_size); queries: (N, 3) or
        (B, N, 3).
        """
        full_latents = np.asarray(full_latents, dtype=np.float32)
        queries = np.asarray(queries, dtype=np.float32)
        batched = full_latents.ndim == 2
        if not batched:
            full_latents = full_latents[None]
        if full_latents.ndim != 2 or full_latents.shape[1] != self.full_latent_size:
            raise ValueError(
                f"Expected full latents of shape (B, {self.full_latent_size}) or "
                f"({self.full_latent_size},), got shape "
                f"{full_latents.shape if batched else full_latents[0].shape}."
            )
        if queries.ndim == 2:
            queries = np.broadcast_to(
                queries[None], (len(full_latents), *queries.shape)
            )
        handle = self._decode_planes_full(full_latents)
        logits = self._decode_logits_chunked(
            lambda chunk: self._decode_logits_full(handle, chunk, object_scale),
            queries,
            chunk_size,
        )
        return logits if batched else logits[0]

    def decode_mesh_full(
        self,
        full_latents: np.ndarray,
        resolution: int | None = None,
        chunk_size: int = 65536,
        frame_half_size: float = 1.0,
        object_scale: float = 0.9,
    ) -> trimesh.Trimesh | list[trimesh.Trimesh]:
        """
        Decode full latents into meshes in their original frames (the counterpart of
        :meth:`encode_mesh_full`; pass the same ``frame_half_size``). full_latents:
        (full_latent_size,) for a single mesh or (B, full_latent_size) for a list.
        """
        latents, transforms = self.unpack_full_latent(
            full_latents, frame_half_size=frame_half_size, object_scale=object_scale
        )
        return self.decode_mesh(
            latents, resolution, transform=transforms, chunk_size=chunk_size
        )

    def occupancy_loss_full(
        self,
        full_latents: np.ndarray,
        queries: np.ndarray,
        labels: np.ndarray,
        num_vol: int,
        vol_coeff: float = 1.0,
        near_coeff: float = 0.1,
        object_scale: float = 0.9,
        chunk_size: int = 65536,
    ) -> np.ndarray:
        """
        Like :meth:`occupancy_loss`, but for full latents (B, full_latent_size) with
        query points (B, N, 3) given in the [-1, 1] normalized world frame (see
        :meth:`decode_full`).
        """
        full_latents = np.asarray(full_latents, dtype=np.float32)
        queries = np.asarray(queries, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if full_latents.ndim != 2 or queries.ndim != 3 or labels.ndim != 2:
            raise ValueError(
                f"Expected batched full latents (B, {self.full_latent_size}), queries "
                f"(B, N, 3), and labels (B, N), got shapes {full_latents.shape}, "
                f"{queries.shape}, and {labels.shape}."
            )
        logits = self.decode_full(
            full_latents, queries, chunk_size=chunk_size, object_scale=object_scale
        )
        bce = (
            np.maximum(logits, 0.0)
            - logits * labels
            + np.log1p(np.exp(-np.abs(logits)))
        )
        return vol_coeff * bce[:, :num_vol].mean(axis=-1) + near_coeff * bce[
            :, num_vol:
        ].mean(axis=-1)

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
