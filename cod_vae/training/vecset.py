"""
Dataset over the preprocessed ShapeNet distribution of 3DShape2VecSet
(https://github.com/1zb/3DShape2VecSet), which is what the original COD-VAE was trained
on. Expects the directory layout used by the reference implementation:

    root_dir/
        ShapeNetV2_point/
            {synset_id}/
                {train,val,test}.lst      # object ids, one per line
                {object_id}.npz           # vol_points, vol_label, near_points, near_label
                {object_id}.npy           # scalar normalization factor for the surface
        ShapeNetV2_surface/
            {synset_id}/4_pointcloud/{object_id}.npz   # points

Items follow the same interface as :class:`cod_vae.training.MeshOccupancyDataset`
(numpy dicts with "surface", "queries" (volume first), and "labels"), so the dataset
plugs directly into both backends' training loops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .data import axis_scaling

__all__ = ["ShapeNetVecSetDataset"]


class ShapeNetVecSetDataset:
    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        categories: Sequence[str] | None = None,
        pc_size: int = 2048,
        num_vol_queries: int = 4096,
        num_near_queries: int = 4096,
        augment: bool = True,
        repeat: int = 1,
        seed: int = 0,
    ):
        self.root_dir = Path(root_dir)
        self.point_dir = self.root_dir / "ShapeNetV2_point"
        self.surface_dir = self.root_dir / "ShapeNetV2_surface"
        self.pc_size = pc_size
        self.num_vol_queries = num_vol_queries
        self.num_near_queries = num_near_queries
        self.augment = augment
        self.repeat = repeat
        self.seed = seed
        self.epoch = 0

        if categories is None:
            categories = sorted(
                path.name for path in self.point_dir.iterdir() if path.is_dir()
            )
        self.items: list[tuple[str, str]] = []
        for category in sorted(categories):
            split_file = self.point_dir / category / f"{split}.lst"
            with split_file.open() as f:
                object_ids = [
                    line.replace(".npz", "").strip() for line in f if line.strip()
                ]
            self.items.extend((category, object_id) for object_id in object_ids)
        if not self.items:
            raise ValueError(f"No objects found for split {split!r} in {self.root_dir}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items) * self.repeat

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        category, object_id = self.items[index % len(self.items)]
        rng = np.random.default_rng((self.seed, self.epoch, index))

        query_data = np.load(self.point_dir / category / f"{object_id}.npz")
        scale = float(np.load(self.point_dir / category / f"{object_id}.npy"))
        surface_data = np.load(
            self.surface_dir / category / "4_pointcloud" / f"{object_id}.npz"
        )

        surface = surface_data["points"]
        surface = surface[rng.choice(surface.shape[0], self.pc_size, replace=False)]
        surface = surface.astype(np.float32) * scale

        vol_points, vol_label = query_data["vol_points"], query_data["vol_label"]
        near_points, near_label = query_data["near_points"], query_data["near_label"]
        vol_idx = rng.choice(vol_points.shape[0], self.num_vol_queries, replace=False)
        near_idx = rng.choice(
            near_points.shape[0], self.num_near_queries, replace=False
        )
        queries = np.concatenate([vol_points[vol_idx], near_points[near_idx]]).astype(
            np.float32
        )
        labels = np.concatenate([vol_label[vol_idx], near_label[near_idx]]).astype(
            np.float32
        )

        if self.augment:
            surface, queries = axis_scaling(surface, queries, rng)
        return {
            "surface": surface.astype(np.float32),
            "queries": queries,
            "labels": labels,
        }
