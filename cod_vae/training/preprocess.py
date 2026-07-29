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

import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "MESH_SUFFIXES",
    "SdfGenSettings",
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


@dataclass(frozen=True)
class SdfGenSettings:
    """Parameters of the sdf_gen preprocessing; defaults match preprocess/box.py."""

    num_vol: int = 250_000
    num_surface: int = 125_000
    near_stddevs: tuple[float, ...] = (0.005, 0.05)
    object_scale: float = 0.9
    watertight_resolution: int = 50_000


def preprocess_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    settings: SdfGenSettings | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Apply the sdf_gen preprocessing to a single mesh: watertighting, normalization into
    the [-1, 1] cube, and sampling of the surface / volume / near-surface pools. The
    near-surface pool has ``num_surface * len(near_stddevs)`` points (the surface
    samples perturbed once per standard deviation, as in the reference script).
    Returns "surface", "vol_points", "vol_label", "near_points", and "near_label"
    (occupancy: 1 inside, 0 outside), all float32 and in the same normalized frame.
    """
    try:
        import point_cloud_utils as pcu
    except ImportError as exc:
        raise ImportError(
            "Mesh preprocessing requires point-cloud-utils; install it via "
            "pip install cod-vae[preprocess]"
        ) from exc

    if settings is None:
        settings = SdfGenSettings()
    rng = np.random.default_rng(seed)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    vw, fw = pcu.make_mesh_watertight(
        vertices,
        faces,
        settings.watertight_resolution,
        seed=int(rng.integers(2**31)),
    )

    shifts = (vw.max(axis=0) + vw.min(axis=0)) / 2
    vw = vw - shifts
    scale = (1.0 / np.abs(vw).max()) * settings.object_scale
    vw = vw * scale

    fid, bc = pcu.sample_mesh_random(
        vw, fw, settings.num_surface, random_seed=int(rng.integers(1, 2**31))
    )
    surface = pcu.interpolate_barycentric_coords(fw, fid, bc, vw)

    vol_points = rng.random((settings.num_vol, 3)) * 2 - 1
    vol_sdf, _, _ = pcu.signed_distance_to_mesh(vol_points, vw, fw)

    near_points = np.concatenate(
        [
            surface + rng.normal(scale=stddev, size=surface.shape)
            for stddev in settings.near_stddevs
        ]
    )
    near_sdf, _, _ = pcu.signed_distance_to_mesh(near_points, vw, fw)

    return {
        "surface": surface.astype(np.float32),
        "vol_points": vol_points.astype(np.float32),
        "vol_label": (vol_sdf < 0).astype(np.float32),
        "near_points": near_points.astype(np.float32),
        "near_label": (near_sdf < 0).astype(np.float32),
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


def merge_vecset_root(
    src_root: Path | str, out_root: Path | str, link: str = "symlink"
) -> list[str]:
    """
    Merge an existing preprocessed root (ShapeNetV2_point + ShapeNetV2_surface) into
    the output root by symlinking, hardlinking, or copying every category directory.
    Returns the merged category names; raises FileExistsError on name collisions.
    """
    if link not in ("symlink", "hardlink", "copy"):
        raise ValueError(f"Unknown link mode {link!r}")
    src_root, out_root = Path(src_root), Path(out_root)
    categories: list[str] = []
    for sub in (POINT_DIR, SURFACE_DIR):
        src_sub = src_root / sub
        if not src_sub.is_dir():
            raise FileNotFoundError(
                f"{src_root} is not a preprocessed vecset root: {src_sub} is missing"
            )
        for src_cat in sorted(p for p in src_sub.iterdir() if p.is_dir()):
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
    import datasets

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
        import trimesh

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


def _run_category(
    out_root: Path,
    category: str,
    tasks: list[tuple],
    lst: Mapping[str, list[str]],
    workers: int,
    verbose: bool,
) -> dict[str, list[str]]:
    """
    Run the preprocessing tasks of one category (in ``workers`` parallel processes if
    requested), drop objects whose preprocessing failed, and write the category's
    train/val/test .lst files. Returns the surviving object ids per split.
    """
    failed: set[str] = set()

    def handle_failure(object_id: str, exc: BaseException) -> None:
        failed.add(object_id)
        print(
            f"[{category}] {object_id}: preprocessing failed ({exc}); dropping",
            file=sys.stderr,
        )

    total = len(tasks)
    if workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {}
            queue = iter(tasks)
            done_count = 0
            while pending or done_count < total:
                while len(pending) < 2 * workers:
                    task = next(queue, None)
                    if task is None:
                        break
                    pending[pool.submit(_process_object, task)] = task[2]
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    object_id = pending.pop(future)
                    done_count += 1
                    if future.exception() is not None:
                        handle_failure(object_id, future.exception())
                    elif verbose:
                        print(f"[{category}] {done_count}/{total} {object_id}")
    else:
        for done_count, task in enumerate(tasks, start=1):
            try:
                _process_object(task)
                if verbose:
                    print(f"[{category}] {done_count}/{total} {task[2]}")
            except Exception as exc:
                handle_failure(task[2], exc)

    lst = {
        split: [object_id for object_id in ids if object_id not in failed]
        for split, ids in lst.items()
    }
    point_dir = out_root / POINT_DIR / category
    point_dir.mkdir(parents=True, exist_ok=True)
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
    verbose: bool = True,
) -> dict[str, list[str]]:
    """
    Preprocess a directory of mesh files (recursively, all trimesh-loadable formats in
    :data:`MESH_SUFFIXES`) into a new category directory of the vecset layout. The
    meshes do not need to be watertight. If the directory contains train/val/test
    subdirectories, meshes are assigned to the corresponding splits; otherwise
    everything becomes training data. Existing outputs are reused (skipped) unless
    ``overwrite`` is set, so an interrupted run can be resumed as long as the source
    directory is unchanged. Failed meshes are reported and dropped. Returns the object
    ids per split, which are also written to the category's .lst files.
    """
    out_root, mesh_dir = Path(out_root), Path(mesh_dir)
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
    counter = 0
    for split, paths in files.items():
        for index, path in enumerate(paths):
            object_id = f"{split}_{index:06d}"
            lst[split].append(object_id)
            object_seed = seed + counter
            counter += 1
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
                    object_seed,
                    extras,
                )
            )
    return _run_category(out_root, category, tasks, lst, workers, verbose)


def add_hf_source(
    out_root: Path | str,
    category: str,
    dataset: str,
    split_map: Mapping[str, str] | None = None,
    settings: SdfGenSettings | None = None,
    seed: int = 0,
    workers: int = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """
    Preprocess a Hugging Face mesh dataset into a new category directory of the vecset
    layout. Existing outputs are reused (skipped) unless ``overwrite`` is set, so an
    interrupted run can be resumed. Meshes that fail preprocessing (e.g. watertighting)
    are reported and dropped, as in the reference script. With ``workers > 0``, meshes
    are processed in that many parallel processes. Returns the object ids per split,
    which are also written to the category's train/val/test .lst files.
    """
    out_root = Path(out_root)
    if settings is None:
        settings = SdfGenSettings()
    split_datasets = _load_hf_splits(dataset, split_map)
    resolved_map = _resolve_split_map(list(split_datasets), split_map)
    _check_category_free(out_root, category)

    tasks: list[tuple] = []
    lst: dict[str, list[str]] = {split: [] for split in SPLITS}
    counter = 0
    for src_split, ds in split_datasets.items():
        ds = ds.with_format("numpy")
        columns = set(ds.column_names)
        for required in ("mesh.vertices", "mesh.faces"):
            if required not in columns:
                raise ValueError(
                    f"{dataset}[{src_split}] has no {required!r} column; expected a "
                    f"mesh dataset in the Tactile MNIST format"
                )
        for index in range(len(ds)):
            object_id = f"{src_split}_{index:06d}"
            lst[resolved_map[src_split]].append(object_id)
            object_seed = seed + counter
            counter += 1
            if _outputs_exist(out_root, category, object_id) and not overwrite:
                continue
            row = ds[index]
            extras = {key: row[key] for key in ("id", "label") if key in columns}
            tasks.append(
                (
                    str(out_root),
                    category,
                    object_id,
                    (np.asarray(row["mesh.vertices"]), np.asarray(row["mesh.faces"])),
                    settings,
                    object_seed,
                    extras,
                )
            )
    return _run_category(out_root, category, tasks, lst, workers, verbose)


def build_vecset_dataset(
    out_dir: Path | str,
    vecset_sources: Sequence[Path | str] = (),
    mesh_sources: Sequence[tuple[str, Path | str]] = (),
    hf_sources: Sequence[tuple[str, str]] = (),
    link: str = "symlink",
    split_map: Mapping[str, str] | None = None,
    settings: SdfGenSettings | None = None,
    seed: int = 0,
    workers: int = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Merge preprocessed vecset roots, directories of mesh files, and Hugging Face mesh
    datasets (the latter two given as ``(category_name, source)`` pairs) into a dataset
    at ``out_dir`` in the vecset layout, directly loadable with
    :class:`cod_vae.training.ShapeNetVecSetDataset`.
    """
    out_dir = Path(out_dir)
    names = [name for name, _ in (*mesh_sources, *hf_sources)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate source names: {duplicates}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_root in vecset_sources:
        categories = merge_vecset_root(src_root, out_dir, link=link)
        if verbose:
            print(f"Merged {len(categories)} categories from {src_root} ({link})")

    def report(source, name, lst):
        if verbose:
            counts = ", ".join(f"{split}: {len(ids)}" for split, ids in lst.items())
            print(f"Added {source} as category {name!r} ({counts})")

    for name, mesh_dir in mesh_sources:
        lst = add_mesh_dir_source(
            out_dir,
            name,
            mesh_dir,
            settings=settings,
            seed=seed,
            workers=workers,
            overwrite=overwrite,
            verbose=verbose,
        )
        report(mesh_dir, name, lst)
    for name, dataset in hf_sources:
        lst = add_hf_source(
            out_dir,
            name,
            dataset,
            split_map=split_map,
            settings=settings,
            seed=seed,
            workers=workers,
            overwrite=overwrite,
            verbose=verbose,
        )
        report(dataset, name, lst)
