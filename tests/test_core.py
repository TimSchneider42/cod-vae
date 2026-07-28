import dataclasses

import numpy as np
import pytest

from cod_vae import CODVAEConfig, init_params, load_npz, save_npz
from cod_vae.mesh import grid_queries, normalize_to_cube, occupancy_grid_to_mesh


def test_config_roundtrip():
    config = CODVAEConfig(num_latents=64)
    assert config.plane_resolution == 16
    assert config.num_output_patches == 768
    assert config.num_kept_tokens == 192
    assert CODVAEConfig(**dataclasses.asdict(config)) == config


def test_npz_roundtrip(tmp_path, tiny_config, tiny_params):
    path = tmp_path / "model.npz"
    save_npz(path, tiny_config, tiny_params)
    config, params = load_npz(path)
    assert config == tiny_config
    assert set(params) == set(tiny_params)
    for key in params:
        np.testing.assert_array_equal(params[key], tiny_params[key])


def test_init_params_deterministic(tiny_config):
    a = init_params(tiny_config, seed=1)
    b = init_params(tiny_config, seed=1)
    c = init_params(tiny_config, seed=2)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert any(not np.array_equal(a[key], c[key]) for key in a)


def test_normalize_to_cube(meshes):
    normalized, transform = normalize_to_cube(meshes[0], object_scale=0.9)
    assert np.abs(normalized.vertices).max() == pytest.approx(0.9, abs=1e-6)
    restored = transform.apply_inverse(np.asarray(normalized.vertices))
    np.testing.assert_allclose(restored, meshes[0].vertices, atol=1e-6)


def test_occupancy_grid_to_mesh():
    resolution = 32
    queries = grid_queries(resolution).reshape(resolution, resolution, resolution, 3)
    # Signed "logit" field of a sphere of radius 0.5 (positive inside).
    logits = 0.5 - np.linalg.norm(queries, axis=-1)
    mesh = occupancy_grid_to_mesh(logits)
    assert mesh.is_watertight
    radii = np.linalg.norm(mesh.vertices, axis=-1)
    assert np.abs(radii - 0.5).max() < 0.05


def test_occupancy_grid_to_mesh_empty():
    assert occupancy_grid_to_mesh(np.full((8, 8, 8), -1.0)).is_empty
