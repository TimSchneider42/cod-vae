"""
Backend-agnostic training data pipeline.

Training items are random subsamples of per-shape occupancy pools: surface points plus
volume query points (uniform in the [-1, 1] cube) and near-surface query points with
ground-truth occupancy labels. :class:`MeshOccupancyDataset` computes these pools
lazily (and optionally disk-cached) from arbitrary meshes — watertight or not — using
the original authors' sdf_gen preprocessing
(:func:`cod_vae.training.preprocess.preprocess_mesh`), and serves random subsamples
per training step (with the reference's AxisScaling augmentation) as plain numpy
dictionaries usable from both backends. The exact same preprocessing run ahead of time
is available via ``cod-vae-dataset``, whose output is served by
:class:`cod_vae.training.ShapeNetVecSetDataset` with the same item interface.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import trimesh

from .preprocess import SdfGenSettings, preprocess_mesh

__all__ = [
    "axis_scaling",
    "MeshOccupancyDataset",
    "iterate_batches",
]


def axis_scaling(
    surface: np.ndarray,
    queries: np.ndarray,
    rng: np.random.Generator,
    jitter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The reference implementation's AxisScaling augmentation: random anisotropic scaling
    in [0.75, 1.25] per axis, renormalization into the [-1, 1] cube, and point jitter on
    the surface point cloud. Occupancy labels are invariant under this transform.
    """
    scaling = (rng.random(3) * 0.5 + 0.75).astype(np.float32)
    surface = surface * scaling
    queries = queries * scaling
    scale = np.float32(0.999999 / max(np.abs(surface).max(), 0.1))
    surface = surface * scale
    queries = queries * scale
    if jitter:
        surface = surface + 0.005 * rng.standard_normal(surface.shape).astype(
            np.float32
        )
        surface = np.clip(surface, -1.0, 1.0)
    return surface, queries


class MeshOccupancyDataset:
    """
    Training dataset over a set of meshes, which do not need to be watertight (the
    preprocessing repairs them automatically; requires the ``cod-vae[preprocess]``
    extra).

    Every item is a random subsample of the (lazily computed, optionally disk-cached)
    occupancy pools of one mesh, generated with the sdf_gen preprocessing of
    :func:`cod_vae.training.preprocess.preprocess_mesh` (parameterized via
    ``settings``):

    - "surface": (pc_size, 3) surface point cloud,
    - "queries": (num_vol_queries + num_near_queries, 3) query points (volume first),
    - "labels": matching occupancy labels (1 inside, 0 outside).

    Sampling is deterministic given (seed, epoch, index); call :meth:`set_epoch` at the
    start of every epoch to draw fresh subsamples. ``repeat`` virtually enlarges the
    dataset (the reference uses repeat=16 with small datasets so that one "epoch" sees
    every shape multiple times with different subsamples).
    """

    def __init__(
        self,
        meshes: (
            Sequence[trimesh.Trimesh]
            | Sequence[Path | str]
            | Callable[[int], trimesh.Trimesh]
        ),
        num_meshes: int | None = None,
        pc_size: int = 2048,
        num_vol_queries: int = 4096,
        num_near_queries: int = 4096,
        augment: bool = True,
        repeat: int = 1,
        cache_dir: Path | str | None = None,
        seed: int = 0,
        settings: SdfGenSettings | None = None,
    ):
        if callable(meshes):
            if num_meshes is None:
                raise ValueError("num_meshes is required when meshes is a callable")
            self._get_mesh = meshes
            self.num_meshes = num_meshes
        else:
            meshes = list(meshes)
            self._get_mesh = lambda i: (
                meshes[i]
                if isinstance(meshes[i], trimesh.Trimesh)
                else trimesh.load(meshes[i], force="mesh")
            )
            self.num_meshes = len(meshes)
        self.pc_size = pc_size
        self.num_vol_queries = num_vol_queries
        self.num_near_queries = num_near_queries
        self.augment = augment
        self.repeat = repeat
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.seed = seed
        self.epoch = 0
        self.settings = settings if settings is not None else SdfGenSettings()
        self._pools: dict[int, dict[str, np.ndarray]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_meshes * self.repeat

    def pool(self, mesh_index: int) -> dict[str, np.ndarray]:
        """
        Return the full occupancy pool dict of one mesh (as produced by
        :func:`cod_vae.training.preprocess.preprocess_mesh`), computing and disk-caching
        it on first access.
        """
        if mesh_index in self._pools:
            return self._pools[mesh_index]
        cache_path = None
        if self.cache_dir is not None:
            # The version suffix invalidates caches from before the pool format gained
            # the "shifts"/"scale" normalization transform entries.
            tag = hashlib.sha1(f"{self.settings!r}:v2".encode()).hexdigest()[:8]
            cache_path = self.cache_dir / f"occupancy_{mesh_index:06d}_{tag}.npz"
            if cache_path.exists():
                data = dict(np.load(cache_path))
                self._pools[mesh_index] = data
                return data
        mesh = self._get_mesh(mesh_index)
        data = preprocess_mesh(
            mesh.vertices, mesh.faces, self.settings, seed=self.seed + mesh_index
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **data)
        self._pools[mesh_index] = data
        return data

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        mesh_index = index % self.num_meshes
        pool = self.pool(mesh_index)
        rng = np.random.default_rng((self.seed, self.epoch, index))

        surface = pool["surface"][
            rng.choice(pool["surface"].shape[0], self.pc_size, replace=False)
        ]
        vol_idx = rng.choice(
            pool["vol_points"].shape[0], self.num_vol_queries, replace=False
        )
        near_idx = rng.choice(
            pool["near_points"].shape[0], self.num_near_queries, replace=False
        )
        queries = np.concatenate(
            [pool["vol_points"][vol_idx], pool["near_points"][near_idx]]
        )
        labels = np.concatenate(
            [pool["vol_label"][vol_idx], pool["near_label"][near_idx]]
        )
        if self.augment:
            surface, queries = axis_scaling(surface, queries, rng)
        return {
            "surface": surface.astype(np.float32),
            "queries": queries.astype(np.float32),
            "labels": labels.astype(np.float32),
        }

    def precompute(self, verbose: bool = False) -> None:
        """Compute (and cache) the occupancy pools of all meshes up front."""
        for index in range(self.num_meshes):
            if verbose:
                print(f"Computing occupancy pools: {index + 1}/{self.num_meshes}")
            self.pool(index)


def iterate_batches(
    dataset: MeshOccupancyDataset,
    batch_size: int,
    epoch: int,
    seed: int = 0,
    drop_last: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
):
    """
    Iterate over shuffled batches of a dataset as dicts of stacked numpy arrays. With
    num_shards > 1, yields only every num_shards-th batch (for multi-process data
    parallelism); all shards see the same shuffling and thus disjoint batches.
    """
    dataset.set_epoch(epoch)
    order = np.random.default_rng((seed, epoch)).permutation(len(dataset))
    if drop_last:
        order = order[: len(order) - len(order) % (batch_size * num_shards)]
    for start in range(batch_size * shard_index, len(order), batch_size * num_shards):
        indices = order[start : start + batch_size]
        if len(indices) < batch_size:
            break
        items = [dataset[int(i)] for i in indices]
        yield {key: np.stack([item[key] for item in items]) for key in items[0]}
