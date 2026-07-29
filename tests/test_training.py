"""
Training smoke tests: a few steps of both stages on both backends must run, actually
decrease the stage-1 loss on a trivial dataset, only update the intended parameters,
and work multi-device (torchrun/DDP on CPU, jax with forced host devices).
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import trimesh

from cod_vae import CODVAEConfig
from cod_vae.init import LATENT_PREFIXES
from cod_vae.training import MeshOccupancyDataset, SdfGenSettings, TrainingConfig

# Mesh-based training preprocesses with the sdf_gen recipe, which needs pcu.
pytest.importorskip("point_cloud_utils")


@pytest.fixture(scope="module")
def dataset():
    meshes = [
        trimesh.creation.box(extents=[1.0, 0.6, 0.4]),
        trimesh.creation.icosphere(subdivisions=2, radius=0.5),
    ]
    return MeshOccupancyDataset(
        meshes,
        pc_size=256,
        num_vol_queries=128,
        num_near_queries=128,
        repeat=8,
        settings=SdfGenSettings(
            num_vol=2000, num_surface=1000, watertight_resolution=1000
        ),
    )


def test_dataset_items(dataset):
    item = dataset[0]
    assert item["surface"].shape == (256, 3)
    assert item["queries"].shape == (256, 3)
    assert item["labels"].shape == (256,)
    assert set(np.unique(item["labels"])) <= {0.0, 1.0}
    # Deterministic given (seed, epoch, index); fresh subsamples across epochs.
    same = dataset[0]
    np.testing.assert_array_equal(item["surface"], same["surface"])
    dataset.set_epoch(1)
    assert not np.array_equal(item["surface"], dataset[0]["surface"])
    dataset.set_epoch(0)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("stage", [1, 2])
def test_training_step_updates_correct_params(backend, stage, tiny_config, dataset):
    pytest.importorskip(backend)
    train_config = TrainingConfig(
        stage=stage, epochs=1, batch_size=2, log_every=1000, seed=0
    )
    from cod_vae import init_params

    params = init_params(tiny_config, seed=0)
    if backend == "torch":
        from cod_vae.torch.training import train

        result = train(tiny_config, train_config, dataset, params=params, device="cpu")
    else:
        from cod_vae.jax.training import train

        result = train(tiny_config, train_config, dataset, params=params)

    assert set(result) == set(params)
    changed = {k for k in params if not np.array_equal(result[k], params[k])}
    latent_keys = {k for k in params if k.startswith(LATENT_PREFIXES)}
    frozen = latent_keys if stage == 1 else set(params) - latent_keys
    assert changed, "training did not update any parameters"
    assert not (changed & frozen), "training updated frozen parameters"
    assert all(np.isfinite(result[k]).all() for k in result)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_stage1_loss_decreases(backend, dataset):
    """A tiny model on a tiny dataset: the stage-1 loss must clearly decrease."""
    pytest.importorskip(backend)
    config = CODVAEConfig(
        latent_dim=8,
        num_latents=8,
        embed_dim=64,
        query_dim=8,
        num_heads=4,
        num_latent_layers=1,
        encoder_num_patches=32,
        encoder_num_blocks=1,
        encoder_num_layers_per_block=1,
        decoder_output_resolution=32,
        decoder_output_patch_size=8,
        decoder_num_layers=1,
        decoder_num_init_layers=1,
        decoder_num_merged_tokens=4,
        droppath_rate=0.0,
    )
    train_config = TrainingConfig(
        stage=1,
        epochs=10,
        batch_size=4,
        lr=1e-3,
        base_batch_size=4,
        log_every=1000,
        seed=0,
    )
    from cod_vae import CODVAE, init_params
    from cod_vae.training.data import iterate_batches

    params = init_params(config, seed=0)

    def stage1_loss(model_params):
        model = CODVAE(config, model_params, backend=backend, device="cpu")
        batch = next(iterate_batches(dataset, 4, epoch=0, seed=123))
        logits = model.decode(model.encode(batch["surface"]), batch["queries"])
        # Plain BCE as a backend-neutral progress metric.
        p = 1 / (1 + np.exp(-logits))
        return -np.mean(
            batch["labels"] * np.log(p + 1e-9)
            + (1 - batch["labels"]) * np.log(1 - p + 1e-9)
        )

    loss_before = stage1_loss(params)
    if backend == "torch":
        from cod_vae.torch.training import train

        result = train(config, train_config, dataset, params=params, device="cpu")
    else:
        from cod_vae.jax.training import train

        result = train(config, train_config, dataset, params=params)
    loss_after = stage1_loss(result)
    assert loss_after < loss_before * 0.85, (loss_before, loss_after)


def test_torch_ddp_cpu(tiny_config, dataset, tmp_path):
    """Two-process DDP training on CPU (gloo) via torchrun."""
    pytest.importorskip("torch")
    script = tmp_path / "ddp_train.py"
    script.write_text(
        textwrap.dedent(
            """
            import trimesh
            from cod_vae import CODVAEConfig, init_params
            from cod_vae.training import (
                MeshOccupancyDataset, SdfGenSettings, TrainingConfig,
            )
            from cod_vae.torch.training import train

            config = CODVAEConfig(
                latent_dim=8, num_latents=8, embed_dim=64, query_dim=8, num_heads=4,
                num_latent_layers=2, encoder_num_patches=32, encoder_num_blocks=1,
                encoder_num_layers_per_block=1, decoder_output_resolution=32,
                decoder_output_patch_size=8, decoder_num_layers=2,
                decoder_num_init_layers=1, decoder_num_merged_tokens=4,
            )
            meshes = [trimesh.creation.box(extents=[1.0, 0.6, 0.4])] * 4
            dataset = MeshOccupancyDataset(
                meshes, pc_size=128, num_vol_queries=64, num_near_queries=64,
                settings=SdfGenSettings(
                    num_vol=1000, num_surface=500, watertight_resolution=1000,
                ),
            )
            train_config = TrainingConfig(stage=1, epochs=1, batch_size=1, log_every=1)
            train(config, train_config, dataset, device="cpu", out_dir="OUT_DIR")
            print("DDP_TRAINING_DONE")
            """
        ).replace("OUT_DIR", str(tmp_path / "out"))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_port=29517",
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DDP_TRAINING_DONE" in result.stdout
    assert (tmp_path / "out" / "checkpoint_last.npz").exists()


def test_jax_multi_device_cpu(tmp_path):
    """Data-parallel jax training across two forced host devices."""
    pytest.importorskip("jax")
    script = tmp_path / "jax_train.py"
    script.write_text(
        textwrap.dedent(
            """
            import jax, trimesh
            assert jax.device_count() == 2, jax.devices()
            from cod_vae import CODVAEConfig
            from cod_vae.training import (
                MeshOccupancyDataset, SdfGenSettings, TrainingConfig,
            )
            from cod_vae.jax.training import train

            config = CODVAEConfig(
                latent_dim=8, num_latents=8, embed_dim=64, query_dim=8, num_heads=4,
                num_latent_layers=2, encoder_num_patches=32, encoder_num_blocks=1,
                encoder_num_layers_per_block=1, decoder_output_resolution=32,
                decoder_output_patch_size=8, decoder_num_layers=2,
                decoder_num_init_layers=1, decoder_num_merged_tokens=4,
            )
            meshes = [trimesh.creation.box(extents=[1.0, 0.6, 0.4])] * 4
            dataset = MeshOccupancyDataset(
                meshes, pc_size=128, num_vol_queries=64, num_near_queries=64,
                settings=SdfGenSettings(
                    num_vol=1000, num_surface=500, watertight_resolution=1000,
                ),
            )
            train_config = TrainingConfig(stage=1, epochs=1, batch_size=1, log_every=1)
            train(config, train_config, dataset)
            print("JAX_MULTI_DEVICE_DONE")
            """
        )
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "JAX_MULTI_DEVICE_DONE" in result.stdout
