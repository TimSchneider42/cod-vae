import numpy as np
import pytest

from cod_vae.training import ShapeNetVecSetDataset, TrainingConfig


@pytest.fixture()
def vecset_root(tmp_path):
    """Synthetic dataset in the 3DShape2VecSet ShapeNet layout (2 categories x 2 objects)."""
    rng = np.random.default_rng(0)
    for category in ("02691156", "03001627"):
        point_dir = tmp_path / "ShapeNetV2_point" / category
        surface_dir = tmp_path / "ShapeNetV2_surface" / category / "4_pointcloud"
        point_dir.mkdir(parents=True)
        surface_dir.mkdir(parents=True)
        object_ids = [f"obj{i}" for i in range(2)]
        (point_dir / "train.lst").write_text("\n".join(f"{o}.npz" for o in object_ids))
        (point_dir / "val.lst").write_text(f"{object_ids[0]}.npz\n")
        for object_id in object_ids:
            np.savez(
                point_dir / f"{object_id}.npz",
                vol_points=rng.uniform(-1, 1, (500, 3)).astype(np.float16),
                vol_label=(rng.random(500) < 0.3).astype(np.float32),
                near_points=rng.uniform(-1, 1, (500, 3)).astype(np.float16),
                near_label=(rng.random(500) < 0.5).astype(np.float32),
            )
            np.save(point_dir / f"{object_id}.npy", np.float32(0.9))
            np.savez(
                surface_dir / f"{object_id}.npz",
                points=rng.uniform(-1, 1, (1000, 3)).astype(np.float32),
            )
    return tmp_path


def test_vecset_dataset(vecset_root):
    dataset = ShapeNetVecSetDataset(
        vecset_root,
        split="train",
        pc_size=128,
        num_vol_queries=64,
        num_near_queries=64,
        repeat=3,
    )
    assert len(dataset) == 4 * 3
    item = dataset[0]
    assert item["surface"].shape == (128, 3)
    assert item["queries"].shape == (128, 3)
    assert item["labels"].shape == (128,)
    assert all(value.dtype == np.float32 for value in item.values())
    assert set(np.unique(item["labels"])) <= {0.0, 1.0}
    assert np.abs(item["surface"]).max() <= 1.0

    # Deterministic given (seed, epoch, index); fresh subsamples across epochs.
    np.testing.assert_array_equal(item["surface"], dataset[0]["surface"])
    dataset.set_epoch(1)
    assert not np.array_equal(item["surface"], dataset[0]["surface"])

    val = ShapeNetVecSetDataset(
        vecset_root, split="val", pc_size=128, num_vol_queries=64, num_near_queries=64
    )
    assert len(val) == 2


def test_vecset_teacher_logits(vecset_root):
    """Teacher-logit rows are read at the same offsets as the labels, and enabling
    them changes nothing about the rest of the item."""
    for category in ("02691156", "03001627"):
        point_dir = vecset_root / "ShapeNetV2_point" / category
        logit_dir = vecset_root / "teacher" / category
        logit_dir.mkdir(parents=True)
        for path in point_dir.glob("*.npz"):
            pool = np.load(path)
            # The teacher logit of every row encodes its hard label, so alignment is
            # checkable on the served item regardless of which rows were drawn.
            np.savez(
                logit_dir / path.name,
                vol_logit=(pool["vol_label"] * 2 - 1).astype(np.float16),
                near_logit=(pool["near_label"] * 2 - 1).astype(np.float16),
            )

    kwargs = dict(split="train", pc_size=128, num_vol_queries=64, num_near_queries=64)
    plain = ShapeNetVecSetDataset(vecset_root, **kwargs)
    distill = ShapeNetVecSetDataset(
        vecset_root, teacher_logit_dir=vecset_root / "teacher", **kwargs
    )
    for index in range(4):
        item = distill[index]
        assert item["teacher_logits"].shape == (128,)
        assert item["teacher_logits"].dtype == np.float32
        np.testing.assert_array_equal(item["teacher_logits"], item["labels"] * 2 - 1)
        reference = plain[index]
        for key in ("surface", "queries", "labels"):
            np.testing.assert_array_equal(item[key], reference[key])


def test_vecset_dataset_trains(vecset_root, tiny_config):
    pytest.importorskip("torch")
    from cod_vae.torch.training import train

    dataset = ShapeNetVecSetDataset(
        vecset_root,
        split="train",
        pc_size=128,
        num_vol_queries=64,
        num_near_queries=64,
        repeat=2,
    )
    train_config = TrainingConfig(stage=1, epochs=1, batch_size=2, log_every=1000)
    params = train(tiny_config, train_config, dataset, device="cpu")
    assert all(np.isfinite(value).all() for value in params.values())


@pytest.mark.parametrize("stage", (1, 2))
def test_vecset_dataset_trains_with_distillation(vecset_root, tiny_config, stage):
    pytest.importorskip("torch")
    from cod_vae.torch.training import train

    rng = np.random.default_rng(1)
    for category in ("02691156", "03001627"):
        logit_dir = vecset_root / "teacher" / category
        logit_dir.mkdir(parents=True)
        for path in (vecset_root / "ShapeNetV2_point" / category).glob("*.npz"):
            np.savez(
                logit_dir / path.name,
                vol_logit=rng.normal(size=500).astype(np.float16),
                near_logit=rng.normal(size=500).astype(np.float16),
            )
    dataset = ShapeNetVecSetDataset(
        vecset_root,
        split="train",
        pc_size=128,
        num_vol_queries=64,
        num_near_queries=64,
        teacher_logit_dir=vecset_root / "teacher",
    )
    train_config = TrainingConfig(
        stage=stage,
        epochs=1,
        batch_size=2,
        log_every=1000,
        distill_coeff=1.0,
        distill_temperature=2.0,
    )
    params = train(tiny_config, train_config, dataset, device="cpu")
    assert all(np.isfinite(value).all() for value in params.values())
