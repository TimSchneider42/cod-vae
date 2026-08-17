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

Every item reads a few thousand of the ~500k points a pool file holds, so the pools are
read in row blocks instead of whole (see :class:`_PoolFile`), which is what keeps
training IO-bound only on pathologically slow storage.
"""

from __future__ import annotations

import ast
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import axis_scaling

__all__ = ["ShapeNetVecSetDataset"]


class _PoolFile:
    """
    Row-block reads from an .npz, without materializing whole arrays.

    np.savez stores every array as a contiguous, uncompressed .npy member inside a zip
    container, so a block of rows is one seek plus one read -- 50 KB instead of the 7 MB
    a full pool file costs, which is the difference between an IO-bound and a GPU-bound
    training step. Compressed archives (np.savez_compressed) have no such layout; for
    those, :meth:`rows` falls back to reading the array and slicing it.
    """

    # Layouts are parsed once per file and reused: with repeat > 1 the same pools are
    # revisited many times per epoch, and on network storage every avoided open or seek
    # is a round trip.
    _layouts: dict[Path, dict[str, tuple[int, np.dtype, tuple[int, ...]]] | None] = {}
    _layout_limit = 200_000

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._handle = None
        if self.path not in self._layouts:
            if len(self._layouts) >= self._layout_limit:
                self._layouts.clear()
            self._layouts[self.path] = self._read_layout()
        self.members = self._layouts[self.path]

    def __enter__(self) -> "_PoolFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self.path.open("rb")
        return self._handle

    def _read_layout(self):
        """Offset, dtype and shape of every member, or None if they are not readable."""
        handle = self.handle
        with zipfile.ZipFile(handle) as archive:
            infos = archive.infolist()
        if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
            return None
        layout = {}
        for info in infos:
            handle.seek(info.header_offset)
            # The local header repeats the name and may carry a different extra field
            # than the central directory entry, so it decides where the bytes start.
            name_len, extra_len = struct.unpack("<HH", handle.read(30)[26:30])
            handle.seek(info.header_offset + 30 + name_len + extra_len)
            if handle.read(6) != b"\x93NUMPY":
                return None
            major = handle.read(2)[0]
            header_len = int.from_bytes(handle.read(2 if major == 1 else 4), "little")
            header = ast.literal_eval(handle.read(header_len).decode("latin1"))
            if header["fortran_order"]:
                return None
            layout[info.filename.removesuffix(".npy")] = (
                handle.tell(),
                np.dtype(header["descr"]),
                header["shape"],
            )
        return layout

    def length(self, key: str) -> int:
        if self.members is None:
            with np.load(self.path) as data:
                return len(data[key])
        return self.members[key][2][0]

    def rows(self, key: str, start: int, count: int) -> np.ndarray:
        if self.members is None:
            with np.load(self.path) as data:
                return data[key][start : start + count]
        offset, dtype, shape = self.members[key]
        stride = dtype.itemsize * int(np.prod(shape[1:], dtype=np.int64))
        self.handle.seek(offset + start * stride)
        raw = self.handle.read(count * stride)
        return np.frombuffer(raw, dtype=dtype).reshape((count, *shape[1:]))


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
        near_groups: int = 2,
        io_retry_seconds: float = 240.0,
        teacher_logit_dir: Path | str | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.point_dir = self.root_dir / "ShapeNetV2_point"
        self.surface_dir = self.root_dir / "ShapeNetV2_surface"
        # A directory tree mirroring ShapeNetV2_point ({category}/{object_id}.npz with
        # "vol_logit" and "near_logit" rows aligned with the pool files) makes every
        # item also carry "teacher_logits" — a teacher model's logits at the very query
        # points served, for distillation. Occupancy at a query point is invariant
        # under the AxisScaling augmentation, and so are the teacher's precomputed
        # logits, exactly like the hard labels.
        self.teacher_logit_dir = (
            Path(teacher_logit_dir) if teacher_logit_dir is not None else None
        )
        self.pc_size = pc_size
        self.num_vol_queries = num_vol_queries
        self.num_near_queries = num_near_queries
        self.augment = augment
        self.repeat = repeat
        self.seed = seed
        # The near-surface pool is the surface samples perturbed once per noise standard
        # deviation and concatenated (two in the reference data), so a sample takes a
        # block out of each group to keep the mix of noise levels.
        self.near_groups = near_groups
        self.io_retry_seconds = io_retry_seconds
        self.epoch = 0
        self._scales: dict[tuple[str, str], float] = {}

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

    @staticmethod
    def _start(
        rng: np.random.Generator, total: int, count: int, offset: int = 0
    ) -> int:
        """Start of a random block of ``count`` rows within ``total`` rows."""
        if count > total:
            raise ValueError(f"Cannot draw {count} points from a pool of {total}")
        return offset + int(rng.integers(0, total - count + 1))

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """
        Read one item, riding out a storage layer that transiently loses a file.

        Network filesystems can refuse an individual file for minutes and then serve it
        again (seen here as ``OSError`` 521, the NFS ``EBADHANDLE``, on one compute node
        while every other node read the same file without complaint). Multi-day training
        must not die on a blip, so a failed read is retried until ``io_retry_seconds``
        has passed -- and a retry is free of consequences, because it eventually returns
        the very sample that was asked for.

        A file still unreadable after that is not a blip, and this raises. Quietly
        training on a different shape instead would buy a run that finishes over a run
        that is correct: the substitution is invisible in the weights, so a model trained
        through a bad node would be silently incomparable to its neighbours in the grid.
        Failing here is loud, and the job's own retry loop resumes from the last epoch
        checkpoint, which costs at most one epoch.

        The retry budget stays well under the distributed watchdog timeout: a stalled
        worker holds up its rank's collective, and a rank that is late enough looks like
        a dead one.
        """
        deadline = time.monotonic() + self.io_retry_seconds
        delay = 0.5
        while True:
            try:
                return self._load(index)
            except OSError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OSError(
                        f"item {index} is still unreadable after "
                        f"{self.io_retry_seconds:.0f}s of retries under {self.root_dir}: "
                        f"{exc}"
                    ) from exc
                print(
                    f"[vecset] read failed for item {index}, retrying for another "
                    f"{remaining:.0f}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 30.0)

    def _load(self, index: int) -> dict[str, np.ndarray]:
        category, object_id = self.items[index % len(self.items)]
        rng = np.random.default_rng((self.seed, self.epoch, index))

        scale = self._scales.get((category, object_id))
        if scale is None:
            scale = float(np.load(self.point_dir / category / f"{object_id}.npy"))
            self._scales[(category, object_id)] = scale

        with (
            _PoolFile(self.point_dir / category / f"{object_id}.npz") as queries_file,
            _PoolFile(
                self.surface_dir / category / "4_pointcloud" / f"{object_id}.npz"
            ) as surface_file,
        ):
            # The pools are sampled independently and identically per shape, so a
            # contiguous block is distributed exactly like a random subset -- and costs
            # one read instead of one per point. This is also what the reference
            # implementation's chunked HDF5 sampling approximates.
            start = self._start(rng, surface_file.length("points"), self.pc_size)
            surface = surface_file.rows("points", start, self.pc_size)
            surface = surface.astype(np.float32) * scale

            vol_start = self._start(
                rng, queries_file.length("vol_points"), self.num_vol_queries
            )
            blocks = [
                (
                    queries_file.rows("vol_points", vol_start, self.num_vol_queries),
                    queries_file.rows("vol_label", vol_start, self.num_vol_queries),
                )
            ]
            near_total = queries_file.length("near_points")
            groups = self.near_groups if near_total % self.near_groups == 0 else 1
            group_size = near_total // groups
            near_blocks = []
            for group in range(groups):
                count = self.num_near_queries // groups
                if group == groups - 1:
                    count = self.num_near_queries - count * (groups - 1)
                near_start = self._start(
                    rng, group_size, count, offset=group * group_size
                )
                near_blocks.append((near_start, count))
                blocks.append(
                    (
                        queries_file.rows("near_points", near_start, count),
                        queries_file.rows("near_label", near_start, count),
                    )
                )
        queries = np.concatenate([points for points, _ in blocks]).astype(np.float32)
        labels = np.concatenate([label for _, label in blocks]).astype(np.float32)

        item = {}
        if self.teacher_logit_dir is not None:
            with _PoolFile(
                self.teacher_logit_dir / category / f"{object_id}.npz"
            ) as logit_file:
                logit_blocks = [
                    logit_file.rows("vol_logit", vol_start, self.num_vol_queries)
                ] + [
                    logit_file.rows("near_logit", start, count)
                    for start, count in near_blocks
                ]
            item["teacher_logits"] = np.concatenate(logit_blocks).astype(np.float32)

        if self.augment:
            surface, queries = axis_scaling(surface, queries, rng)
        return {
            "surface": surface.astype(np.float32),
            "queries": queries,
            "labels": labels,
            **item,
        }
