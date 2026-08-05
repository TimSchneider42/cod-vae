"""
Tests for the sdf_gen-style preprocessing and the cod-vae-dataset merge tool: the
preprocessing must produce geometrically consistent occupancy pools, and merged outputs
(linked vecset roots + preprocessed Hugging Face mesh datasets) must load directly with
ShapeNetVecSetDataset.
"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("point_cloud_utils")
datasets = pytest.importorskip("datasets")

from cod_vae.training import ShapeNetVecSetDataset
from cod_vae.training.preprocess import (
    SdfGenSettings,
    build_vecset_dataset,
    preprocess_mesh,
)

TINY = SdfGenSettings(num_vol=2000, num_surface=1000, watertight_resolution=1000)


def test_preprocess_mesh_geometry():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    data = preprocess_mesh(sphere.vertices, sphere.faces, TINY, seed=0)

    assert data["surface"].shape == (1000, 3)
    assert data["vol_points"].shape == (2000, 3)
    assert data["vol_label"].shape == (2000,)
    # Two noise scales, each applied to the full surface pool (as in sdf_gen).
    assert data["near_points"].shape == (2000, 3)
    assert data["near_label"].shape == (2000,)
    for key in data:
        assert data[key].dtype == np.float32

    # The sphere is normalized to radius ~0.9; surface points must lie on it and
    # volume queries must be labeled by containment (1 inside, 0 outside).
    radii = np.linalg.norm(data["surface"], axis=1)
    assert 0.75 < radii.mean() < 1.0
    vol_radii = np.linalg.norm(data["vol_points"], axis=1)
    np.testing.assert_array_equal(data["vol_label"][vol_radii < 0.5], 1.0)
    np.testing.assert_array_equal(data["vol_label"][vol_radii > 1.2], 0.0)

    # The normalization transform maps original mesh coordinates into the query frame:
    # the sphere is centered at the origin with radius 2, so shifts ~ 0 and
    # scale ~ 0.9 / 2.
    assert data["shifts"].shape == (3,)
    assert data["scale"].shape == ()
    np.testing.assert_allclose(data["shifts"], 0.0, atol=0.05)
    np.testing.assert_allclose(data["scale"], 0.45, atol=0.05)
    # Surface points in the query frame map back onto the original sphere (up to the
    # inflation of the low-resolution watertighting).
    original_surface = data["surface"] / data["scale"] + data["shifts"]
    np.testing.assert_allclose(np.linalg.norm(original_surface, axis=1), 2.0, atol=0.25)

    # Deterministic given the seed.
    again = preprocess_mesh(sphere.vertices, sphere.faces, TINY, seed=0)
    np.testing.assert_array_equal(data["vol_points"], again["vol_points"])
    other = preprocess_mesh(sphere.vertices, sphere.faces, TINY, seed=1)
    assert not np.array_equal(data["vol_points"], other["vol_points"])


def test_preprocess_mesh_skip_watertighting():
    import dataclasses

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    settings = dataclasses.replace(TINY, watertight_resolution=None)
    data = preprocess_mesh(sphere.vertices, sphere.faces, settings, seed=0)

    # Without watertighting, the transform is exact and surface points map exactly
    # back onto the original sphere.
    np.testing.assert_allclose(data["shifts"], 0.0, atol=1e-6)
    np.testing.assert_allclose(data["scale"], 0.45, rtol=1e-6)
    original_surface = data["surface"] / data["scale"] + data["shifts"]
    # Sampled points lie on the icosphere's flat triangles, slightly inside the
    # nominal radius.
    np.testing.assert_allclose(np.linalg.norm(original_surface, axis=1), 2.0, atol=0.01)
    vol_radii = np.linalg.norm(data["vol_points"], axis=1)
    np.testing.assert_array_equal(data["vol_label"][vol_radii < 0.85], 1.0)
    np.testing.assert_array_equal(data["vol_label"][vol_radii > 0.95], 0.0)


@pytest.fixture(scope="module")
def hf_dataset_dir(tmp_path_factory):
    """A tiny mesh dataset in the Tactile MNIST format, saved with save_to_disk."""
    meshes = [
        trimesh.creation.box(extents=[1.0, 0.6, 0.4]),
        trimesh.creation.icosphere(subdivisions=2, radius=0.5),
    ]

    def split(items):
        return datasets.Dataset.from_dict(
            {
                "id": list(range(len(items))),
                "label": [i % 2 for i in range(len(items))],
                "mesh.vertices": [np.asarray(m.vertices, np.float32) for m in items],
                "mesh.faces": [np.asarray(m.faces, np.int32) for m in items],
            }
        )

    dataset = datasets.DatasetDict(
        {
            "train": split(meshes),
            "test": split(meshes[:1]),
            "holdout": split(meshes[:1]),
        }
    )
    path = tmp_path_factory.mktemp("hf") / "toy_meshes"
    dataset.save_to_disk(str(path))
    return path


@pytest.fixture(scope="module")
def vecset_source_dir(tmp_path_factory):
    """A minimal preprocessed root in the layout of the original authors' data."""
    root = tmp_path_factory.mktemp("vecset") / "root"
    point_dir = root / "ShapeNetV2_point" / "02691156"
    surface_dir = root / "ShapeNetV2_surface" / "02691156" / "4_pointcloud"
    point_dir.mkdir(parents=True)
    surface_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.savez(
        point_dir / "obj0.npz",
        vol_points=rng.uniform(-1, 1, (200, 3)).astype(np.float32),
        vol_label=rng.integers(0, 2, 200).astype(np.float32),
        near_points=rng.uniform(-1, 1, (200, 3)).astype(np.float32),
        near_label=rng.integers(0, 2, 200).astype(np.float32),
    )
    np.save(point_dir / "obj0.npy", np.float64(1.0))
    np.savez(
        surface_dir / "obj0.npz",
        points=rng.uniform(-1, 1, (200, 3)).astype(np.float32),
    )
    (point_dir / "train.lst").write_text("obj0\n")
    (point_dir / "val.lst").write_text("")
    (point_dir / "test.lst").write_text("")
    return root


def test_build_and_load_merged_dataset(tmp_path, hf_dataset_dir, vecset_source_dir):
    out = tmp_path / "merged"
    build_vecset_dataset(
        out,
        vecset_sources=[vecset_source_dir],
        hf_sources=[("toy", str(hf_dataset_dir))],
        settings=TINY,
        verbose=False,
    )

    assert (out / "ShapeNetV2_point" / "02691156").is_symlink()
    toy_point = out / "ShapeNetV2_point" / "toy"
    # Default split mapping: train and test are picked up, "holdout" is ignored.
    assert toy_point / "train.lst" in list(toy_point.iterdir())
    assert (toy_point / "train.lst").read_text().split() == [
        "train_000000",
        "train_000001",
    ]
    assert (toy_point / "test.lst").read_text().split() == ["test_000000"]
    assert (toy_point / "val.lst").read_text() == ""
    # Extras from the source rows are preserved for provenance.
    stored = np.load(toy_point / "train_000001.npz")
    assert int(stored["id"]) == 1 and int(stored["label"]) == 1

    dataset = ShapeNetVecSetDataset(
        out, split="train", pc_size=64, num_vol_queries=64, num_near_queries=64
    )
    assert len(dataset) == 3  # obj0 + two toy meshes
    item = dataset[len(dataset) - 1]
    assert item["surface"].shape == (64, 3)
    assert item["queries"].shape == (128, 3)
    assert item["labels"].shape == (128,)
    assert set(np.unique(item["labels"])) <= {0.0, 1.0}

    # A second build over the same output must fail on the linked source collision.
    with pytest.raises(FileExistsError):
        build_vecset_dataset(
            out, vecset_sources=[vecset_source_dir], settings=TINY, verbose=False
        )


def test_mesh_dir_source(tmp_path):
    from cod_vae.cli import dataset_main
    from cod_vae.training.preprocess import add_mesh_dir_source

    # A directory with train/val subdirectories and one non-watertight mesh (an open
    # box), which the sdf_gen preprocessing must repair via watertighting.
    mesh_dir = tmp_path / "meshes"
    (mesh_dir / "train").mkdir(parents=True)
    (mesh_dir / "val").mkdir()
    box = trimesh.creation.box(extents=[1.0, 0.6, 0.4])
    open_box = trimesh.Trimesh(box.vertices, box.faces[:-2])
    assert not open_box.is_watertight
    box.export(mesh_dir / "train" / "box.stl")
    open_box.export(mesh_dir / "train" / "open_box.obj")
    trimesh.creation.icosphere(subdivisions=2).export(mesh_dir / "val" / "sphere.ply")
    # A flat directory (no split subdirectories): everything becomes training data.
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    box.export(flat_dir / "box.glb")

    out = tmp_path / "merged"
    dataset_main(
        [
            str(out),
            "--meshes",
            f"toy={mesh_dir}",
            "--meshes",
            str(flat_dir),
            "--num-vol",
            "2000",
            "--num-surface",
            "1000",
            "--watertight-resolution",
            "1000",
        ]
    )
    toy_point = out / "ShapeNetV2_point" / "toy"
    assert (toy_point / "train.lst").read_text().split() == [
        "train_000000",
        "train_000001",
    ]
    assert (toy_point / "val.lst").read_text().split() == ["val_000000"]
    assert (out / "ShapeNetV2_point" / "flat" / "train.lst").read_text().split() == [
        "train_000000"
    ]
    stored = np.load(toy_point / "train_000001.npz")
    assert str(stored["source_file"]) == "train/open_box.obj"

    dataset = ShapeNetVecSetDataset(
        out, split="train", pc_size=64, num_vol_queries=64, num_near_queries=64
    )
    assert len(dataset) == 3
    item = dataset[0]
    assert item["labels"].shape == (128,)

    # Mesh files outside the split subdirectories are rejected, not silently dropped.
    box.export(mesh_dir / "stray.stl")
    with pytest.raises(ValueError, match="mixes"):
        add_mesh_dir_source(tmp_path / "other", "toy2", mesh_dir)


def test_failures_abort_by_default(tmp_path, monkeypatch):
    from cod_vae.training import preprocess as pp

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    trimesh.creation.box(extents=[1.0, 0.6, 0.4]).export(mesh_dir / "box.stl")

    def boom(*args, **kwargs):
        raise ValueError("bad geometry")

    monkeypatch.setattr(pp, "preprocess_mesh", boom)
    with pytest.raises(RuntimeError, match="train_000000"):
        pp.add_mesh_dir_source(tmp_path / "out1", "toy", mesh_dir, verbose=False)

    # Opt-in skipping restores the reference script's drop-with-warning behavior.
    lst = pp.add_mesh_dir_source(
        tmp_path / "out2", "toy", mesh_dir, skip_failed=True, verbose=False
    )
    assert lst["train"] == []

    # A missing dependency is never skippable.
    def missing(*args, **kwargs):
        raise ImportError("point-cloud-utils is not installed")

    monkeypatch.setattr(pp, "preprocess_mesh", missing)
    with pytest.raises(ImportError):
        pp.add_mesh_dir_source(
            tmp_path / "out3", "toy", mesh_dir, skip_failed=True, verbose=False
        )


def test_vecset_root_missing_surface_fails(tmp_path):
    from cod_vae.training.preprocess import merge_vecset_root

    root = tmp_path / "root"
    (root / "ShapeNetV2_point" / "cat").mkdir(parents=True)
    (root / "ShapeNetV2_surface").mkdir()
    with pytest.raises(FileNotFoundError, match="surface"):
        merge_vecset_root(root, tmp_path / "out")


def test_subsampling(tmp_path, hf_dataset_dir, vecset_source_dir):
    from cod_vae.cli import dataset_main
    from cod_vae.training.preprocess import _subsample

    # Deterministic given (seed, key), independent of list processing order.
    items = list(range(100))
    assert _subsample(items, 0.1, seed=0, key="a") == _subsample(
        items, 0.1, seed=0, key="a"
    )
    assert len(_subsample(items, 0.1, seed=0, key="a")) == 10
    assert _subsample(items, 0.1, seed=0, key="a") != _subsample(
        items, 0.1, seed=1, key="a"
    )
    assert _subsample(items, 1.0, seed=0, key="a") == items
    assert _subsample(items, 0.001, seed=0, key="a")  # at least one item survives
    with pytest.raises(ValueError, match="fraction"):
        _subsample(items, 0.0, seed=0, key="a")

    # Vecset roots: subsampled categories are real directories with rewritten .lst
    # files and per-object links; a mesh dir with :0.5 keeps one of two train meshes,
    # with the object id of its position in the full file list.
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    trimesh.creation.box(extents=[1.0, 0.6, 0.4]).export(mesh_dir / "box.stl")
    trimesh.creation.icosphere(subdivisions=2).export(mesh_dir / "sphere.stl")
    out = tmp_path / "merged"
    dataset_main(
        [
            str(out),
            "--vecset",
            f"{vecset_source_dir}:0.5",
            "--meshes",
            f"toy={mesh_dir}:0.5",
            "--hf",
            f"toyhf={hf_dataset_dir}:0.5",
            "--num-vol",
            "2000",
            "--num-surface",
            "1000",
            "--watertight-resolution",
            "1000",
        ]
    )
    vecset_cat = out / "ShapeNetV2_point" / "02691156"
    assert vecset_cat.is_dir() and not vecset_cat.is_symlink()
    assert (vecset_cat / "train.lst").read_text().split() == ["obj0"]
    assert (vecset_cat / "obj0.npz").exists()
    toy_train = (out / "ShapeNetV2_point" / "toy" / "train.lst").read_text().split()
    assert len(toy_train) == 1 and toy_train[0] in ("train_000000", "train_000001")
    toyhf_train = (out / "ShapeNetV2_point" / "toyhf" / "train.lst").read_text().split()
    assert len(toyhf_train) == 1

    # The same command with the same seed selects the same subsets.
    out2 = tmp_path / "merged2"
    dataset_main(
        [
            str(out2),
            "--meshes",
            f"toy={mesh_dir}:0.5",
            "--num-vol",
            "2000",
            "--num-surface",
            "1000",
            "--watertight-resolution",
            "1000",
        ]
    )
    assert (
        out2 / "ShapeNetV2_point" / "toy" / "train.lst"
    ).read_text().split() == toy_train

    dataset = ShapeNetVecSetDataset(
        out, split="train", pc_size=64, num_vol_queries=64, num_near_queries=64
    )
    assert len(dataset) == 3  # one object per category after subsampling
    assert dataset[0]["labels"].shape == (128,)


def test_dataset_cli_with_split_map_and_workers(tmp_path, hf_dataset_dir):
    from cod_vae.cli import dataset_main

    out = tmp_path / "merged"
    dataset_main(
        [
            str(out),
            "--hf",
            f"toy={hf_dataset_dir}",
            "--hf-split",
            "train",
            "--hf-split",
            "holdout=val",
            "--workers",
            "2",
            "--num-vol",
            "2000",
            "--num-surface",
            "1000",
            "--watertight-resolution",
            "1000",
        ]
    )
    toy_point = out / "ShapeNetV2_point" / "toy"
    assert (toy_point / "train.lst").read_text().split() == [
        "train_000000",
        "train_000001",
    ]
    assert (toy_point / "val.lst").read_text().split() == ["holdout_000000"]
    assert (toy_point / "test.lst").read_text() == ""
    dataset = ShapeNetVecSetDataset(
        out, split="val", pc_size=64, num_vol_queries=64, num_near_queries=64
    )
    assert len(dataset) == 1
