"""
Backend-agnostic training data pipeline.

The reference implementation trains on the preprocessed ShapeNet occupancy data of
3DShape2VecSet: per shape, a pool of surface points plus volume query points (uniform in
the [-1, 1] cube) and near-surface query points with ground-truth occupancy labels.
:func:`compute_occupancy_data` reproduces this preprocessing for arbitrary watertight
meshes, and :class:`MeshOccupancyDataset` serves random subsamples per training step
(with the reference's AxisScaling augmentation), as plain numpy dictionaries usable from
both backends.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import trimesh

from ..mesh import normalize_to_cube, sample_surface_points

__all__ = [
    "compute_occupancy_data",
    "axis_scaling",
    "MeshOccupancyDataset",
    "iterate_batches",
]


def compute_occupancy_data(
    mesh: trimesh.Trimesh,
    num_surface: int = 100_000,
    num_vol: int = 100_000,
    num_near: int = 100_000,
    near_stddevs: Sequence[float] = (0.01, 0.02),
    object_scale: float = 0.9,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Normalize a watertight mesh into the [-1, 1] cube and build pools of surface points,
    uniform volume query points, and near-surface query points (surface samples plus
    Gaussian noise with the given standard deviations) with occupancy labels.
    """
    normalized, _ = normalize_to_cube(mesh, object_scale)
    rng = np.random.default_rng(seed)
    surface = sample_surface_points(
        normalized, num_surface, seed=int(rng.integers(2**31))
    )
    vol_points = rng.uniform(-1.0, 1.0, (num_vol, 3)).astype(np.float32)
    near_base = sample_surface_points(
        normalized, num_near, seed=int(rng.integers(2**31))
    )
    stddevs = np.asarray(near_stddevs, dtype=np.float32)
    per_point_std = stddevs[np.arange(num_near) % len(stddevs), None]
    near_points = (
        near_base
        + rng.standard_normal((num_near, 3)).astype(np.float32) * per_point_std
    )
    near_points = np.clip(near_points, -1.0, 1.0)
    if not normalized.is_watertight:
        raise ValueError(
            "Occupancy labels require a watertight mesh; got a non-watertight mesh. "
            "For arbitrary meshes, build a dataset with cod-vae-dataset instead, "
            "which makes meshes watertight automatically."
        )
    vol_label = normalized.contains(vol_points).astype(np.float32)
    near_label = normalized.contains(near_points).astype(np.float32)
    return {
        "surface": surface,
        "vol_points": vol_points,
        "vol_label": vol_label,
        "near_points": near_points,
        "near_label": near_label,
    }


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
    Training dataset over a set of watertight meshes.

    Every item is a random subsample of the (lazily computed, optionally disk-cached)
    occupancy pools of one mesh:

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
        **pool_kwargs,
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
        self.pool_kwargs = pool_kwargs
        self._pools: dict[int, dict[str, np.ndarray]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_meshes * self.repeat

    def _pool(self, mesh_index: int) -> dict[str, np.ndarray]:
        if mesh_index in self._pools:
            return self._pools[mesh_index]
        cache_path = None
        if self.cache_dir is not None:
            settings = repr(sorted(self.pool_kwargs.items())).encode()
            tag = hashlib.sha1(settings).hexdigest()[:8]
            cache_path = self.cache_dir / f"occupancy_{mesh_index:06d}_{tag}.npz"
            if cache_path.exists():
                data = dict(np.load(cache_path))
                self._pools[mesh_index] = data
                return data
        data = compute_occupancy_data(
            self._get_mesh(mesh_index), seed=self.seed + mesh_index, **self.pool_kwargs
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **data)
        self._pools[mesh_index] = data
        return data

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        mesh_index = index % self.num_meshes
        pool = self._pool(mesh_index)
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
            self._pool(index)


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
