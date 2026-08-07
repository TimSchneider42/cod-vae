"""
Build training datasets in the 3DShape2VecSet directory layout (the format
:class:`cod_vae.training.ShapeNetVecSetDataset` reads) by merging any number of
sources:

- existing preprocessed roots as distributed by the 3DShape2VecSet authors, whose
  category directories are linked or copied into the output unchanged,
- directories of mesh files (not necessarily watertight), and
- Hugging Face mesh datasets in the Tactile MNIST format (rows with "mesh.vertices"
  and "mesh.faces" columns and optionally "id" and "label"), given as Hub repository
  ids or local paths.

Meshes are preprocessed with the recipe the original authors
used to build their ShapeNet data (https://github.com/1zb/sdf_gen, preprocess/box.py):
the mesh is made watertight with point_cloud_utils' Manifold wrapper, centered and
scaled so its largest extent spans 0.9 of the [-1, 1] cube, and pools of surface
points, uniform volume queries, and near-surface queries (surface points plus Gaussian
noise with standard deviations 0.005 and 0.05) are sampled. sdf_gen stores signed
distances; the training data derives occupancy from their sign, so this module directly
stores occupancy labels (1 inside, 0 outside), which is what the vecset layout and the
training pipeline consume.

Requires the optional dependencies point-cloud-utils and, for Hugging Face sources,
datasets (``pip install cod-vae[preprocess]``).
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import trimesh

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars are optional (cod-vae[preprocess])
    tqdm = None

__all__ = [
    "MESH_SUFFIXES",
    "SdfGenSettings",
    "is_closed_mesh",
    "watertight_mesh",
    "sample_occupancy_pools",
    "preprocess_mesh",
    "write_vecset_object",
    "merge_vecset_root",
    "add_mesh_dir_source",
    "add_hf_source",
    "build_vecset_dataset",
]

MESH_SUFFIXES = {".obj", ".off", ".ply", ".stl", ".glb", ".gltf"}
SPLITS = ("train", "val", "test")
POINT_DIR = "ShapeNetV2_point"
SURFACE_DIR = "ShapeNetV2_surface"

# Hugging Face split names that map onto a vecset split without explicit configuration.
DEFAULT_SPLIT_MAP = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}


def _require_point_cloud_utils():
    try:
        import point_cloud_utils
    except ImportError as exc:
        raise ImportError(
            "Mesh preprocessing requires point-cloud-utils; install it via "
            "pip install cod-vae[preprocess]"
        ) from exc
    return point_cloud_utils


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "Hugging Face sources require the datasets package; install it via "
            "pip install cod-vae[preprocess]"
        ) from exc
    return datasets


@dataclass(frozen=True)
class SdfGenSettings:
    """
    Parameters of the sdf_gen preprocessing; defaults match preprocess/box.py. Setting
    watertight_resolution to None skips the watertighting step, assuming the input
    meshes are already watertight.
    """

    num_vol: int = 250_000
    num_surface: int = 125_000
    near_stddevs: tuple[float, ...] = (0.005, 0.05)
    object_scale: float = 0.9
    watertight_resolution: int | None = 50_000
    # Watertighting is a repair, and a mesh that already bounds a volume needs none: its
    # occupancy is exact as it is, while the step would resample the surface onto an
    # octree (turning 3.5k vertices into 77k, at seconds per mesh, and rounding sharp
    # features to the octree resolution). It is therefore applied only where it is
    # needed. Set this to run it unconditionally, as the reference script does.
    watertight_closed_meshes: bool = False


def is_closed_mesh(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """
    Whether the mesh bounds a volume: every edge shared by exactly two faces, with
    consistent winding. Vertices that coincide are merged first, since a surface split
    along a seam is still closed geometrically, which is all the occupancy query needs.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return bool(mesh.is_watertight and mesh.is_winding_consistent)


def watertight_mesh(
    vertices: np.ndarray, faces: np.ndarray, resolution: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a mesh watertight with point_cloud_utils' ManifoldPlus wrapper, as in the
    sdf_gen reference script. Returns the watertight (vertices, faces).
    """
    pcu = _require_point_cloud_utils()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    return pcu.make_mesh_watertight(vertices, faces, resolution, seed=seed)


def sample_occupancy_pools(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_surface: int,
    near_stddevs: Sequence[float],
    object_scale: float,
    rng: np.random.Generator | int,
    num_vol: int | None = None,
    vol_points: np.ndarray | None = None,
    near_dtype: np.dtype | type | None = None,
) -> dict[str, np.ndarray]:
    """
    The sampling stage of the sdf_gen preprocessing, applied to an already watertight
    mesh: normalize it into the [-1, 1] cube (centered at the AABB center, largest
    extent spanning ``object_scale`` of the cube), sample ``num_surface`` surface
    points, label volume query points, and generate labeled near-surface points (the
    surface samples perturbed once per standard deviation, as in the reference
    script, so ``num_surface * len(near_stddevs)`` points).

    The volume query points are either ``num_vol`` fresh uniform samples of the cube
    or the externally supplied cube-frame ``vol_points`` (exactly one must be given).
    If ``near_dtype`` is set, the near-surface points are quantized to that dtype
    *before* labeling, so the returned points and labels stay exactly consistent
    even within the tightest band.

    Returns "surface", "vol_points", "vol_label", "near_points", and "near_label"
    (occupancy labels as bool, True inside), all in the normalized frame, plus the
    normalization transform mapping original mesh coordinates into that frame as
    "shifts" and "scale" (cube = (original - shifts) * scale).
    """
    pcu = _require_point_cloud_utils()
    if (num_vol is None) == (vol_points is None):
        raise ValueError("Exactly one of num_vol and vol_points must be given.")
    rng = np.random.default_rng(rng)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    shifts = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vw = vertices - shifts
    scale = (1.0 / np.abs(vw).max()) * object_scale
    vw = vw * scale

    fid, bc = pcu.sample_mesh_random(
        vw, faces, num_surface, random_seed=int(rng.integers(1, 2**31))
    )
    surface = pcu.interpolate_barycentric_coords(faces, fid, bc, vw)

    if vol_points is None:
        vol_points = rng.random((num_vol, 3)) * 2 - 1
    vol_sdf, _, _ = pcu.signed_distance_to_mesh(
        np.ascontiguousarray(vol_points, dtype=np.float64), vw, faces
    )

    near_points = np.concatenate(
        [
            surface + rng.normal(scale=stddev, size=surface.shape)
            for stddev in near_stddevs
        ]
    )
    if near_dtype is not None:
        near_points = near_points.astype(near_dtype)
    near_sdf, _, _ = pcu.signed_distance_to_mesh(
        near_points.astype(np.float64), vw, faces
    )

    return {
        "surface": surface,
        "vol_points": vol_points,
        "vol_label": vol_sdf < 0,
        "near_points": near_points,
        "near_label": near_sdf < 0,
        "shifts": shifts,
        "scale": scale,
    }


def preprocess_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    settings: SdfGenSettings | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Apply the sdf_gen preprocessing to a single mesh: watertighting
    (:func:`watertight_mesh`, skipped where the mesh does not need it, and entirely if
    ``settings.watertight_resolution`` is None, in which case the input mesh must
    already be watertight) followed by normalization and pool sampling
    (:func:`sample_occupancy_pools`).
    Returns "surface", "vol_points", "vol_label", "near_points", and "near_label"
    (occupancy: 1 inside, 0 outside), all float32 and in the same normalized frame,
    plus the normalization transform mapping original mesh coordinates into that frame
    as "shifts" and "scale" (cube = (original - shifts) * scale).
    """
    if settings is None:
        settings = SdfGenSettings()
    rng = np.random.default_rng(seed)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    skip_watertighting = settings.watertight_resolution is None or (
        not settings.watertight_closed_meshes and is_closed_mesh(vertices, faces)
    )
    if skip_watertighting:
        vw, fw = vertices, faces
    else:
        vw, fw = watertight_mesh(
            vertices,
            faces,
            settings.watertight_resolution,
            seed=int(rng.integers(2**31)),
        )
    pools = sample_occupancy_pools(
        vw,
        fw,
        num_surface=settings.num_surface,
        near_stddevs=settings.near_stddevs,
        object_scale=settings.object_scale,
        rng=rng,
        num_vol=settings.num_vol,
    )
    return {
        "surface": pools["surface"].astype(np.float32),
        "vol_points": pools["vol_points"].astype(np.float32),
        "vol_label": pools["vol_label"].astype(np.float32),
        "near_points": pools["near_points"].astype(np.float32),
        "near_label": pools["near_label"].astype(np.float32),
        "shifts": pools["shifts"].astype(np.float32),
        "scale": np.float32(pools["scale"]),
    }


def write_vecset_object(
    root: Path | str,
    category: str,
    object_id: str,
    data: Mapping[str, np.ndarray],
    extras: Mapping[str, Any] | None = None,
) -> None:
    """
    Write one preprocessed object (as returned by :func:`preprocess_mesh`) into the
    vecset layout under ``root``: the query npz + surface normalization npy in
    ShapeNetV2_point and the surface point cloud in ShapeNetV2_surface. The surface is
    already in the query frame, so the normalization factor is 1. ``extras`` (e.g. the
    source row's id and label) are stored as additional keys in the query npz.
    """
    root = Path(root)
    point_dir = root / POINT_DIR / category
    surface_dir = root / SURFACE_DIR / category / "4_pointcloud"
    point_dir.mkdir(parents=True, exist_ok=True)
    surface_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        point_dir / f"{object_id}.npz",
        vol_points=data["vol_points"],
        vol_label=data["vol_label"],
        near_points=data["near_points"],
        near_label=data["near_label"],
        **{key: np.asarray(value) for key, value in (extras or {}).items()},
    )
    np.save(point_dir / f"{object_id}.npy", np.float64(1.0))
    np.savez(surface_dir / f"{object_id}.npz", points=data["surface"])


def _subsample(items: Sequence, fraction: float, seed: int, key: str) -> list:
    """
    Deterministically keep round(fraction * len) items (at least one), preserving
    order. The selection depends only on (seed, key), not on the processing order.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"Subsampling fraction must be in (0, 1], got {fraction}")
    if fraction == 1 or not items:
        return list(items)
    count = max(1, round(len(items) * fraction))
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    keep = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[int(i)] for i in keep]


def _shard(items: Sequence, shard: int, num_shards: int) -> list:
    """Keep every ``num_shards``-th item, starting at ``shard``."""
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"Invalid shard {shard} of {num_shards}")
    return list(items[shard::num_shards])


def _object_seed(seed: int, split: str, index: int) -> int:
    """Stable per-object preprocessing seed, independent of subsampling."""
    entropy = [seed, zlib.crc32(split.encode()), index]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _link_file(src: Path, dst: Path, link: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing")
    if link == "symlink":
        dst.symlink_to(src.resolve())
    elif link == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def merge_vecset_root(
    src_root: Path | str,
    out_root: Path | str,
    link: str = "symlink",
    fraction: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> list[str]:
    """
    Merge an existing preprocessed root (ShapeNetV2_point + ShapeNetV2_surface) into
    the output root by symlinking, hardlinking, or copying every category directory.
    With ``fraction < 1``, a deterministic random subsample of each category's splits
    is kept instead: the .lst files are rewritten and only the sampled objects' files
    are linked. Returns the merged category names; raises FileExistsError on name
    collisions.
    """
    if link not in ("symlink", "hardlink", "copy"):
        raise ValueError(f"Unknown link mode {link!r}")
    src_root, out_root = Path(src_root), Path(out_root)
    for sub in (POINT_DIR, SURFACE_DIR):
        if not (src_root / sub).is_dir():
            raise FileNotFoundError(
                f"{src_root} is not a preprocessed vecset root: "
                f"{src_root / sub} is missing"
            )
    missing_surface = sorted(
        p.name
        for p in (src_root / POINT_DIR).iterdir()
        if p.is_dir() and not (src_root / SURFACE_DIR / p.name).is_dir()
    )
    if missing_surface:
        raise FileNotFoundError(
            f"{src_root} is missing surface data for categories {missing_surface}"
        )

    if fraction >= 1.0:
        categories: list[str] = []
        for sub in (POINT_DIR, SURFACE_DIR):
            for src_cat in sorted(p for p in (src_root / sub).iterdir() if p.is_dir()):
                dst = out_root / sub / src_cat.name
                if dst.exists() or dst.is_symlink():
                    raise FileExistsError(
                        f"{dst} already exists; category names must be unique across "
                        f"sources"
                    )
                dst.parent.mkdir(parents=True, exist_ok=True)
                if link == "symlink":
                    dst.symlink_to(src_cat.resolve(), target_is_directory=True)
                elif link == "hardlink":
                    shutil.copytree(src_cat, dst, copy_function=os.link)
                else:
                    shutil.copytree(src_cat, dst)
                if sub == POINT_DIR:
                    categories.append(src_cat.name)
        return categories

    categories = []
    for src_cat in sorted(p for p in (src_root / POINT_DIR).iterdir() if p.is_dir()):
        category = src_cat.name
        dst_point = out_root / POINT_DIR / category
        dst_surface_cat = out_root / SURFACE_DIR / category
        for dst in (dst_point, dst_surface_cat):
            if dst.exists() or dst.is_symlink():
                raise FileExistsError(
                    f"{dst} already exists; category names must be unique across "
                    f"sources"
                )
        src_surface = src_root / SURFACE_DIR / category / "4_pointcloud"
        dst_surface = dst_surface_cat / "4_pointcloud"
        dst_point.mkdir(parents=True)
        dst_surface.mkdir(parents=True)
        kept_ids: set[str] = set()
        for split in SPLITS:
            split_file = src_cat / f"{split}.lst"
            if not split_file.exists():
                continue
            with split_file.open() as f:
                ids = [line.replace(".npz", "").strip() for line in f if line.strip()]
            kept = _subsample(ids, fraction, seed, f"{category}/{split}")
            (dst_point / f"{split}.lst").write_text(
                "".join(f"{object_id}\n" for object_id in kept)
            )
            kept_ids.update(kept)
        bar = _progress_bar(len(kept_ids), f"{category} ({link})", "object", verbose)
        try:
            for object_id in sorted(kept_ids):
                _link_file(
                    src_cat / f"{object_id}.npz", dst_point / f"{object_id}.npz", link
                )
                _link_file(
                    src_cat / f"{object_id}.npy", dst_point / f"{object_id}.npy", link
                )
                _link_file(
                    src_surface / f"{object_id}.npz",
                    dst_surface / f"{object_id}.npz",
                    link,
                )
                if bar is not None:
                    bar.update(1)
        finally:
            if bar is not None:
                bar.close()
        categories.append(category)
    return categories


def _resolve_split_map(
    available: Sequence[str], split_map: Mapping[str, str] | None
) -> dict[str, str]:
    """Map source split names to vecset splits, defaulting to DEFAULT_SPLIT_MAP."""
    if split_map is None:
        resolved = {
            name: DEFAULT_SPLIT_MAP[name]
            for name in available
            if name in DEFAULT_SPLIT_MAP
        }
        if not resolved:
            raise ValueError(
                f"None of the splits {sorted(available)} maps onto train/val/test; "
                f"pass an explicit split map (--hf-split)"
            )
    else:
        missing = sorted(set(split_map) - set(available))
        if missing:
            raise ValueError(
                f"Requested splits {missing} not found; available: {sorted(available)}"
            )
        resolved = dict(split_map)
    for target in resolved.values():
        if target not in SPLITS:
            raise ValueError(f"Split targets must be one of {SPLITS}, got {target!r}")
    return resolved


def _load_hf_splits(dataset: str, split_map: Mapping[str, str] | None) -> dict:
    """
    Load the relevant splits of a Hugging Face mesh dataset (local save_to_disk /
    load_dataset path or Hub repository id) as {source_split: datasets.Dataset}.
    """
    datasets = _require_datasets()

    path = Path(dataset)
    if path.exists():
        try:
            loaded = datasets.load_from_disk(str(path))
        except FileNotFoundError:
            loaded = datasets.load_dataset(str(path))
        if isinstance(loaded, datasets.Dataset):
            loaded = {"train": loaded}
        resolved = _resolve_split_map(list(loaded), split_map)
        return {name: loaded[name] for name in resolved}
    available = datasets.get_dataset_split_names(dataset)
    resolved = _resolve_split_map(available, split_map)
    return {name: datasets.load_dataset(dataset, split=name) for name in resolved}


def _process_object(task: tuple) -> None:
    """Preprocess and write one mesh (module-level so worker processes can pickle it)."""
    root, category, object_id, mesh, settings, seed, extras = task
    if isinstance(mesh, str):
        loaded = trimesh.load(mesh, skip_materials=True, process=True, force="mesh")
        vertices, faces = loaded.vertices, loaded.faces
    else:
        vertices, faces = mesh
    data = preprocess_mesh(vertices, faces, settings, seed=seed)
    write_vecset_object(root, category, object_id, data, extras)


def _check_category_free(out_root: Path, category: str) -> None:
    if (out_root / POINT_DIR / category).is_symlink() or (
        out_root / SURFACE_DIR / category
    ).is_symlink():
        raise FileExistsError(
            f"Category {category!r} collides with a linked source in {out_root}"
        )


def _outputs_exist(out_root: Path, category: str, object_id: str) -> bool:
    point_dir = out_root / POINT_DIR / category
    return all(
        path.exists()
        for path in (
            point_dir / f"{object_id}.npz",
            point_dir / f"{object_id}.npy",
            out_root / SURFACE_DIR / category / "4_pointcloud" / f"{object_id}.npz",
        )
    )


def _pin_worker(counter, workers: int) -> None:
    """
    Give each worker process its own slice of the available cores.

    The watertighting and winding number code parallelizes internally over
    ``hardware_concurrency`` threads, which ignores both the worker count and any cpuset
    the process runs under. Without a pin, every worker spawns as many threads as the
    machine has cores -- on a 80-core allocation that is thousands of threads fighting
    over the same cores, and preprocessing runs several times slower than it should.
    """
    if not hasattr(os, "sched_setaffinity") or workers < 2:
        return
    cpus = sorted(os.sched_getaffinity(0))
    per_worker = max(1, len(cpus) // workers)
    if per_worker >= len(cpus):
        return
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    start = (index * per_worker) % len(cpus)
    os.sched_setaffinity(0, cpus[start : start + per_worker])


def _new_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers,
        initializer=_pin_worker,
        initargs=(multiprocessing.Value("i", 0), workers),
    )


def _progress_bar(total: int, desc: str, unit: str, enabled: bool):
    """A tqdm bar, or None if tqdm is unavailable or there is nothing to show."""
    if not enabled or total == 0 or tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _run_category(
    out_root: Path,
    category: str,
    tasks: Iterable[tuple],
    total: int,
    lst: Mapping[str, list[str]],
    workers: int,
    skip_failed: bool,
    verbose: bool,
    write_lists: bool = True,
    timeout: float | None = None,
) -> dict[str, list[str]]:
    """
    Run ``total`` preprocessing tasks of one category (in ``workers`` parallel
    processes if requested) with a progress bar and write the category's
    train/val/test .lst files. A failing mesh aborts the run unless ``skip_failed``
    is set, in which case it is reported and dropped (missing dependencies always
    abort). ``tasks`` may be a lazy iterable — it is consumed incrementally, so
    sources can stream mesh data instead of materializing it up front. Returns the
    surviving object ids per split.

    ``write_lists=False`` leaves the .lst files untouched, which is what one shard of a
    distributed build has to do: it only knows about its own share of the objects.
    ``timeout`` treats a mesh that takes longer than that many seconds as failed, which
    ``skip_failed`` alone cannot do: the watertighting of a pathological mesh does not
    raise, it simply never returns.
    """
    failed: set[str] = set()
    skipped = sum(len(ids) for ids in lst.values()) - total
    if verbose and skipped > 0:
        print(f"[{category}] {skipped} objects already preprocessed; resuming")
    bar = _progress_bar(total, category, "mesh", verbose)

    def handle_failure(object_id: str, exc: BaseException) -> None:
        if isinstance(exc, ImportError):
            # A missing dependency is not a per-mesh problem; always abort.
            raise exc
        if not skip_failed:
            raise RuntimeError(
                f"[{category}] preprocessing of {object_id} failed: {exc} "
                f"(pass --skip-failed to drop failing meshes instead of aborting)"
            ) from exc
        failed.add(object_id)
        message = f"[{category}] {object_id}: preprocessing failed ({exc}); dropping"
        if bar is not None:
            bar.write(message, file=sys.stderr)
        else:
            print(message, file=sys.stderr)

    def advance(object_id: str, done_count: int) -> None:
        if bar is not None:
            bar.update(1)
        elif verbose:
            print(f"[{category}] {done_count}/{total} {object_id}")

    try:
        if workers > 0:
            pool = _new_pool(workers)
            try:
                # future -> (task, submission time), so that the tasks of a pool that
                # has to be torn down can be resubmitted to its replacement.
                pending: dict = {}
                queue = iter(tasks)
                done_count = 0
                while pending or done_count < total:
                    while len(pending) < 2 * workers:
                        task = next(queue, None)
                        if task is None:
                            break
                        pending[pool.submit(_process_object, task)] = (
                            task,
                            time.monotonic(),
                        )
                    if not pending:
                        break
                    finished, _ = wait(
                        pending, timeout=timeout, return_when=FIRST_COMPLETED
                    )
                    for future in finished:
                        task, _ = pending.pop(future)
                        done_count += 1
                        if future.exception() is not None:
                            handle_failure(task[2], future.exception())
                        advance(task[2], done_count)
                    if timeout is None:
                        continue
                    now = time.monotonic()
                    overdue = [f for f, (_, t) in pending.items() if now - t > timeout]
                    if not overdue:
                        continue
                    # A worker stuck inside the watertighting cannot be interrupted, so
                    # the whole pool goes down with it and a fresh one picks up the
                    # tasks that were still in flight.
                    for future in overdue:
                        task, _ = pending.pop(future)
                        done_count += 1
                        handle_failure(task[2], TimeoutError(f"exceeded {timeout:g}s"))
                        advance(task[2], done_count)
                    requeued = [task for task, _ in pending.values()]
                    pending.clear()
                    # The stuck worker sits in a C call and ignores shutdown, so it has
                    # to be killed outright -- otherwise it keeps a core busy and the
                    # interpreter joins it at exit.
                    stuck = list(getattr(pool, "_processes", {}).values())
                    pool.shutdown(wait=False, cancel_futures=True)
                    for process in stuck:
                        process.kill()
                    pool = _new_pool(workers)
                    queue = itertools.chain(requeued, queue)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown()
        else:
            for done_count, task in enumerate(tasks, start=1):
                try:
                    _process_object(task)
                except Exception as exc:
                    handle_failure(task[2], exc)
                advance(task[2], done_count)
    finally:
        if bar is not None:
            bar.close()

    if failed:
        print(
            f"[{category}] dropped {len(failed)} of {total} objects due to "
            f"preprocessing failures",
            file=sys.stderr,
        )
    lst = {
        split: [object_id for object_id in ids if object_id not in failed]
        for split, ids in lst.items()
    }
    point_dir = out_root / POINT_DIR / category
    point_dir.mkdir(parents=True, exist_ok=True)
    if write_lists:
        for split, ids in lst.items():
            (point_dir / f"{split}.lst").write_text(
                "".join(f"{object_id}\n" for object_id in ids)
            )
    return lst


def add_mesh_dir_source(
    out_root: Path | str,
    category: str,
    mesh_dir: Path | str,
    settings: SdfGenSettings | None = None,
    seed: int = 0,
    workers: int = 0,
    overwrite: bool = False,
    fraction: float = 1.0,
    skip_failed: bool = False,
    verbose: bool = True,
    shard: int = 0,
    num_shards: int = 1,
    timeout: float | None = None,
) -> dict[str, list[str]]:
    """
    Preprocess a directory of mesh files (recursively, all trimesh-loadable formats in
    :data:`MESH_SUFFIXES`) into a new category directory of the vecset layout. The
    meshes do not need to be watertight. If the directory contains train/val/test
    subdirectories, meshes are assigned to the corresponding splits; otherwise
    everything becomes training data. With ``fraction < 1``, only a deterministic
    random subsample of each split (seeded by ``seed``) is preprocessed; object ids
    keep their position in the full file list, so a mesh's outputs are identical
    across fractions. Existing outputs are reused (skipped) unless ``overwrite`` is
    set, so an interrupted run can be resumed as long as the source directory is
    unchanged. A failing mesh aborts the run unless ``skip_failed`` is set, in which
    case it is reported and dropped. Returns the object ids per split, which are also
    written to the category's .lst files.

    With ``num_shards > 1`` only every ``num_shards``-th object is processed and the
    .lst files are left alone, so the work can be spread over several machines; a final
    run without sharding then writes the lists (and picks up whatever a shard dropped).
    """
    out_root, mesh_dir = Path(out_root), Path(mesh_dir)
    _require_point_cloud_utils()
    if settings is None:
        settings = SdfGenSettings()
    _check_category_free(out_root, category)

    def mesh_files(directory: Path) -> list[Path]:
        return sorted(
            path
            for path in directory.rglob("*")
            if path.suffix.lower() in MESH_SUFFIXES
        )

    split_dirs = {
        split: mesh_dir / split for split in SPLITS if (mesh_dir / split).is_dir()
    }
    if split_dirs:
        files = {split: mesh_files(d) for split, d in split_dirs.items()}
        stray = [
            path
            for path in mesh_files(mesh_dir)
            if not any(d in path.parents for d in split_dirs.values())
        ]
        if stray:
            raise ValueError(
                f"{mesh_dir} mixes train/val/test subdirectories with mesh files "
                f"outside of them (e.g. {stray[0]})"
            )
    else:
        files = {"train": mesh_files(mesh_dir)}
    if not any(files.values()):
        raise ValueError(f"No mesh files found in {mesh_dir}")

    tasks: list[tuple] = []
    lst: dict[str, list[str]] = {split: [] for split in SPLITS}
    for split, paths in files.items():
        indexed = _shard(
            _subsample(list(enumerate(paths)), fraction, seed, f"{category}/{split}"),
            shard,
            num_shards,
        )
        for index, path in indexed:
            object_id = f"{split}_{index:06d}"
            lst[split].append(object_id)
            if _outputs_exist(out_root, category, object_id) and not overwrite:
                continue
            extras = {"source_file": str(path.relative_to(mesh_dir))}
            tasks.append(
                (
                    str(out_root),
                    category,
                    object_id,
                    str(path),
                    settings,
                    _object_seed(seed, split, index),
                    extras,
                )
            )
    return _run_category(
        out_root,
        category,
        tasks,
        len(tasks),
        lst,
        workers,
        skip_failed,
        verbose,
        write_lists=num_shards == 1,
        timeout=timeout,
    )


def add_hf_source(
    out_root: Path | str,
    category: str,
    dataset: str,
    split_map: Mapping[str, str] | None = None,
    settings: SdfGenSettings | None = None,
    seed: int = 0,
    workers: int = 0,
    overwrite: bool = False,
    fraction: float = 1.0,
    skip_failed: bool = False,
    verbose: bool = True,
    shard: int = 0,
    num_shards: int = 1,
    timeout: float | None = None,
) -> dict[str, list[str]]:
    """
    Preprocess a Hugging Face mesh dataset into a new category directory of the vecset
    layout. With ``fraction < 1``, only a deterministic random subsample of each
    source split (seeded by ``seed``) is preprocessed; object ids keep their row index
    in the full split, so a row's outputs are identical across fractions. Existing
    outputs are reused (skipped) unless ``overwrite`` is set, so an interrupted run
    can be resumed. A mesh that fails preprocessing (e.g. watertighting) aborts the
    run unless ``skip_failed`` is set, in which case it is reported and dropped (as in
    the reference script). With ``workers > 0``, meshes are processed in that many parallel processes. Returns the object ids per split, which
    are also written to the category's train/val/test .lst files.

    With ``num_shards > 1`` only every ``num_shards``-th object is processed and the
    .lst files are left alone, so the work can be spread over several machines; a final
    run without sharding then writes the lists (and picks up whatever a shard dropped).
    """
    out_root = Path(out_root)
    _require_point_cloud_utils()
    if settings is None:
        settings = SdfGenSettings()
    if verbose:
        print(f"[{category}] loading {dataset}")
    split_datasets = {
        src_split: ds.with_format("numpy")
        for src_split, ds in _load_hf_splits(dataset, split_map).items()
    }
    resolved_map = _resolve_split_map(list(split_datasets), split_map)
    _check_category_free(out_root, category)

    lst: dict[str, list[str]] = {split: [] for split in SPLITS}
    pending: list[tuple[str, int, str]] = []
    for src_split, ds in split_datasets.items():
        for required in ("mesh.vertices", "mesh.faces"):
            if required not in ds.column_names:
                raise ValueError(
                    f"{dataset}[{src_split}] has no {required!r} column; expected a "
                    f"mesh dataset in the Tactile MNIST format"
                )
        indices = _shard(
            _subsample(list(range(len(ds))), fraction, seed, f"{category}/{src_split}"),
            shard,
            num_shards,
        )
        for index in indices:
            object_id = f"{src_split}_{index:06d}"
            lst[resolved_map[src_split]].append(object_id)
            if _outputs_exist(out_root, category, object_id) and not overwrite:
                continue
            pending.append((src_split, index, object_id))

    def tasks() -> Iterable[tuple]:
        # Rows are loaded lazily, one mesh at a time, so preprocessing (and the
        # progress bar) starts immediately and memory use stays flat.
        for src_split, index, object_id in pending:
            ds = split_datasets[src_split]
            row = ds[index]
            columns = set(ds.column_names)
            extras = {key: row[key] for key in ("id", "label") if key in columns}
            yield (
                str(out_root),
                category,
                object_id,
                (np.asarray(row["mesh.vertices"]), np.asarray(row["mesh.faces"])),
                settings,
                _object_seed(seed, src_split, index),
                extras,
            )

    return _run_category(
        out_root,
        category,
        tasks(),
        len(pending),
        lst,
        workers,
        skip_failed,
        verbose,
        write_lists=num_shards == 1,
        timeout=timeout,
    )


def build_vecset_dataset(
    out_dir: Path | str,
    vecset_sources: Sequence[Path | str | tuple[Path | str, float]] = (),
    mesh_sources: Sequence[tuple] = (),
    hf_sources: Sequence[tuple] = (),
    link: str = "symlink",
    split_map: Mapping[str, str] | None = None,
    settings: SdfGenSettings | None = None,
    seed: int = 0,
    workers: int = 0,
    overwrite: bool = False,
    skip_failed: bool = False,
    verbose: bool = True,
    shard: int = 0,
    num_shards: int = 1,
    timeout: float | None = None,
) -> None:
    """
    Merge preprocessed vecset roots, directories of mesh files, and Hugging Face mesh
    datasets (the latter two given as ``(category_name, source)`` pairs) into a dataset
    at ``out_dir`` in the vecset layout, directly loadable with
    :class:`cod_vae.training.ShapeNetVecSetDataset`. Every source optionally takes a
    subsampling fraction — ``(root, fraction)`` for vecset sources,
    ``(name, source, fraction)`` for the others — to deterministically keep only that
    share of each split (seeded by ``seed``).

    With ``num_shards > 1`` only every ``num_shards``-th mesh is preprocessed, no .lst
    files are written, and vecset sources (which are only linked) are skipped — that is
    one job of a build spread over several machines. Run the same call once without
    sharding afterwards to link the vecset sources and assemble the lists.
    """
    out_dir = Path(out_dir)
    vecset_sources = [
        entry if isinstance(entry, tuple) else (entry, 1.0) for entry in vecset_sources
    ]
    mesh_sources = [(e[0], e[1], e[2] if len(e) > 2 else 1.0) for e in mesh_sources]
    hf_sources = [(e[0], e[1], e[2] if len(e) > 2 else 1.0) for e in hf_sources]
    names = [name for name, _, _ in (*mesh_sources, *hf_sources)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate source names: {duplicates}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def describe(source, fraction):
        return source if fraction >= 1 else f"{source} (fraction {fraction})"

    for src_root, fraction in vecset_sources if num_shards == 1 else []:
        categories = merge_vecset_root(
            src_root, out_dir, link=link, fraction=fraction, seed=seed, verbose=verbose
        )
        if verbose:
            print(
                f"Merged {len(categories)} categories from "
                f"{describe(src_root, fraction)} ({link})"
            )

    def report(source, fraction, name, lst):
        if verbose:
            counts = ", ".join(f"{split}: {len(ids)}" for split, ids in lst.items())
            print(f"Added {describe(source, fraction)} as category {name!r} ({counts})")

    for name, mesh_dir, fraction in mesh_sources:
        lst = add_mesh_dir_source(
            out_dir,
            name,
            mesh_dir,
            settings=settings,
            seed=seed,
            workers=workers,
            overwrite=overwrite,
            fraction=fraction,
            skip_failed=skip_failed,
            verbose=verbose,
            shard=shard,
            num_shards=num_shards,
            timeout=timeout,
        )
        report(mesh_dir, fraction, name, lst)
    for name, dataset, fraction in hf_sources:
        lst = add_hf_source(
            out_dir,
            name,
            dataset,
            split_map=split_map,
            settings=settings,
            seed=seed,
            workers=workers,
            overwrite=overwrite,
            fraction=fraction,
            skip_failed=skip_failed,
            verbose=verbose,
            shard=shard,
            num_shards=num_shards,
            timeout=timeout,
        )
        report(dataset, fraction, name, lst)
