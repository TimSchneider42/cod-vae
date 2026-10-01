"""
Rotation-equivariant models (CODVAEConfig.rotation_equivariant): the 3+1 channel
groups, the order-independent encoder, the rotation-pair data, and training with the
equivariance loss.
"""

import dataclasses

import numpy as np
import pytest
import trimesh

from cod_vae import CODVAE, CODVAEConfig, init_params, normalize_to_cube
from cod_vae.init import LATENT_PREFIXES
from cod_vae.training import (
    MeshOccupancyDataset,
    SdfGenSettings,
    TrainingConfig,
    random_rotation,
    rotation_pair,
)

torch = pytest.importorskip("torch")

from cod_vae.torch.training import (  # noqa: E402
    equivariance_loss,
    expand_logvar,
    rotate_groups,
    train,
)


@pytest.fixture(scope="module")
def eq_config(tiny_config) -> CODVAEConfig:
    return dataclasses.replace(tiny_config, rotation_equivariant=True)


@pytest.fixture(scope="module")
def pair_dataset():
    pytest.importorskip("point_cloud_utils")
    return MeshOccupancyDataset(
        [
            trimesh.creation.box(extents=[1.0, 0.6, 0.4]),
            trimesh.creation.icosphere(subdivisions=2, radius=0.5),
        ],
        pc_size=256,
        num_vol_queries=128,
        num_near_queries=128,
        repeat=4,
        settings=SdfGenSettings(
            num_vol=2000, num_surface=1000, watertight_resolution=1000
        ),
        rotation_pairs=True,
    )


def test_config_needs_groups_of_four(tiny_config):
    with pytest.raises(ValueError, match="multiples of 4"):
        dataclasses.replace(tiny_config, latent_dim=6, rotation_equivariant=True)
    config = dataclasses.replace(tiny_config, latent_dim=12, rotation_equivariant=True)
    assert config.num_logvars == 6 and config.moments_dim == 18
    assert init_params(config)["latent_proj_in.1.weight"].shape[0] == 18
    # Without the flag, the posterior keeps one log-variance per channel.
    assert tiny_config.moments_dim == 2 * tiny_config.latent_dim


def test_rotate_groups():
    rng = np.random.default_rng(0)
    rotation = torch.from_numpy(random_rotation(rng)).float()[None]
    x = torch.randn(1, 3, 8)
    y = rotate_groups(x, rotation)
    for group in range(2):
        vec, scalar = slice(4 * group, 4 * group + 3), 4 * group + 3
        torch.testing.assert_close(y[0, :, vec], x[0, :, vec] @ rotation[0].T)
        torch.testing.assert_close(y[0, :, scalar], x[0, :, scalar])
    assert float(equivariance_loss(x, y, rotation)) < 1e-10
    assert float(equivariance_loss(x, x + 1.0, rotation)) > 0.1


def test_expand_logvar(eq_config):
    logvar = torch.arange(4.0).reshape(1, 1, 4)  # two groups: (vector, scalar) pairs
    expanded = expand_logvar(eq_config, logvar)
    assert expanded.flatten().tolist() == [0, 0, 0, 1, 2, 2, 2, 3]


def test_random_rotation():
    rng = np.random.default_rng(0)
    for max_degrees in (None, 30.0):
        rotation = random_rotation(rng, max_degrees)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0)
        if max_degrees is not None:
            angle = np.degrees(np.arccos((np.trace(rotation) - 1) / 2))
            assert angle <= max_degrees + 1e-6


def test_rotation_pair():
    rng = np.random.default_rng(0)
    item = {
        "surface": rng.uniform(-1, 1, (64, 3)).astype(np.float32),
        "queries": rng.uniform(-1, 1, (32, 3)).astype(np.float32),
        "labels": rng.integers(0, 2, 32).astype(np.float32),
    }
    pair = rotation_pair(item, rng)
    assert pair["surface"].shape == (2, 64, 3)
    assert pair["queries"].shape == (2, 32, 3)
    np.testing.assert_array_equal(pair["labels"][0], pair["labels"][1])
    # view b = R_ab view a, centered on the centroid inside the unit ball
    rotation = pair["rotation"]
    np.testing.assert_allclose(
        pair["surface"][0] @ rotation.T, pair["surface"][1], atol=1e-5
    )
    np.testing.assert_allclose(
        pair["queries"][0] @ rotation.T, pair["queries"][1], atol=1e-5
    )
    np.testing.assert_allclose(pair["surface"][0].mean(0), 0.0, atol=1e-5)
    assert np.linalg.norm(pair["surface"], axis=-1).max() <= 1.0


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_encoder_is_a_function_of_the_point_set(backend, eq_config, point_batch):
    """
    With canonical FPS (farthest-from-centroid start, pick-order slots), reordering the
    input points must not change the latent; the lexsorted, first-point-seeded encoder
    of a regular model does change.
    """
    pytest.importorskip(backend)
    params = init_params(eq_config, seed=0)
    model = CODVAE(eq_config, params, backend=backend, device="cpu")
    permuted = point_batch[
        :, np.random.default_rng(1).permutation(point_batch.shape[1])
    ]
    np.testing.assert_allclose(
        model.encode(point_batch), model.encode(permuted), atol=1e-4
    )
    regular = dataclasses.replace(eq_config, rotation_equivariant=False)
    regular_model = CODVAE(
        regular, init_params(regular, seed=0), backend=backend, device="cpu"
    )
    assert not np.allclose(
        regular_model.encode(point_batch), regular_model.encode(permuted), atol=1e-4
    )


def test_equivariant_encode_parity(eq_config, point_batch):
    pytest.importorskip("jax")
    params = init_params(eq_config, seed=0)
    latents_torch = CODVAE(eq_config, params, backend="torch").encode(point_batch)
    latents_jax = CODVAE(eq_config, params, backend="jax").encode(point_batch)
    assert latents_torch.shape == (2, eq_config.num_latents, eq_config.latent_dim)
    np.testing.assert_allclose(latents_torch, latents_jax, atol=1e-4)


def test_slots_follow_the_rotation(eq_config, point_batch):
    """Rotating the input must leave the FPS slot order (and so the anchors) intact."""
    from cod_vae.torch.model import _fps_loop

    rng = np.random.default_rng(2)
    rotation = random_rotation(rng).astype(np.float32)
    points = torch.from_numpy(point_batch)
    rotated = torch.from_numpy(point_batch @ rotation.T)
    a = _fps_loop(points, 16, farthest_start=True)
    b = _fps_loop(rotated, 16, farthest_start=True)
    assert torch.equal(a, b)


def test_sphere_normalization():
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    mesh.apply_translation([3.0, -1.0, 2.0])
    normalized, transform = normalize_to_cube(mesh, 0.9, sphere=True)
    np.testing.assert_allclose(transform.center, [3.0, -1.0, 2.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(normalized.vertices, axis=-1).max(), 0.9)
    # The same transform, up to rotation, at any orientation.
    rotation = np.eye(4)
    rotation[:3, :3] = random_rotation(np.random.default_rng(3))
    turned = mesh.copy()
    turned.apply_transform(rotation)
    _, turned_transform = normalize_to_cube(turned, 0.9, sphere=True)
    assert np.isclose(turned_transform.scale, transform.scale)


def test_training_rejects_mismatched_pairs(tiny_config, eq_config, pair_dataset):
    with pytest.raises(ValueError, match="rotation"):
        train(
            tiny_config, TrainingConfig(stage=1, epochs=1, batch_size=2), pair_dataset
        )
    with pytest.raises(ValueError, match="rotation"):
        train(
            eq_config,
            TrainingConfig(stage=1, epochs=1, batch_size=2, rotation_pairs=True),
            _no_pairs(pair_dataset),
        )


def _no_pairs(dataset):
    clone = MeshOccupancyDataset.__new__(MeshOccupancyDataset)
    clone.__dict__.update(dataset.__dict__, rotation_pairs=False)
    return clone


@pytest.mark.parametrize("stage", [1, 2])
def test_equivariant_training_step(stage, eq_config, pair_dataset, capsys):
    train_config = TrainingConfig(
        stage=stage,
        epochs=1,
        batch_size=2,
        log_every=1,
        seed=0,
        rotation_pairs=True,
        eq_warmup_epochs=2,
    )
    assert train_config.eq_weight(0) == pytest.approx(0.1)
    assert train_config.eq_weight(1) == pytest.approx(0.55)
    assert train_config.eq_weight(5) == pytest.approx(1.0)
    params = init_params(eq_config, seed=0)
    result = train(eq_config, train_config, pair_dataset, params=params, device="cpu")

    assert "eq_loss=" in capsys.readouterr().out
    changed = {k for k in params if not np.array_equal(result[k], params[k])}
    latent_keys = {k for k in params if k.startswith(LATENT_PREFIXES)}
    frozen = latent_keys if stage == 1 else set(params) - latent_keys
    assert changed and not (changed & frozen)
    assert all(np.isfinite(result[k]).all() for k in result)


def test_equivariance_loss_cannot_be_escaped():
    """
    Neither shrinking the vector channels nor a constant scalar offset may lower the
    loss: each channel kind is measured relative to itself.
    """
    rng = np.random.default_rng(4)
    rotation = torch.from_numpy(random_rotation(rng)).float()[None].expand(4, 3, 3)
    a, b = torch.randn(4, 3, 8), torch.randn(4, 3, 8)
    loss = float(equivariance_loss(a, b, rotation))
    shrink = torch.ones(8)
    shrink[[0, 1, 2, 4, 5, 6]] = 0.01
    assert float(equivariance_loss(a * shrink, b * shrink, rotation)) == pytest.approx(
        loss, rel=1e-4
    )
    offset = torch.zeros(8)
    offset[[3, 7]] = 100.0
    assert float(equivariance_loss(a + offset, b + offset, rotation)) == pytest.approx(
        loss, rel=1e-4
    )
