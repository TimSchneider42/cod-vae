import dataclasses

import numpy as np
import pytest

from cod_vae import CODVAEConfig, init_params, load_npz, save_npz
from cod_vae.mesh import (
    grid_queries,
    normalize_to_cube,
    occupancy_grid_to_mesh,
    pack_cube_transform,
    points_to_cube_transform,
    unpack_cube_transform,
)


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


def test_points_to_cube_transform(meshes):
    _, reference = normalize_to_cube(meshes[0], object_scale=0.9)
    transform = points_to_cube_transform(np.asarray(meshes[0].vertices), 0.9)
    np.testing.assert_array_equal(transform.center, reference.center)
    assert transform.scale == reference.scale
    # A subset of points with the same bounding box (e.g. convex hull vertices)
    # yields the identical transform.
    hull = points_to_cube_transform(np.asarray(meshes[0].convex_hull.vertices), 0.9)
    np.testing.assert_allclose(hull.center, reference.center, atol=1e-12)
    assert hull.scale == pytest.approx(reference.scale, rel=1e-12)


def test_pack_cube_transform_roundtrip(meshes):
    _, transform = normalize_to_cube(meshes[0], object_scale=0.9)
    row = pack_cube_transform(transform, frame_half_size=0.06, object_scale=0.9)
    assert row.shape == (4,)
    # size is the maximum half-extent of the geometry, normalized by frame_half_size.
    assert row[3] == pytest.approx(np.max(meshes[0].extents / 2) / 0.06, rel=1e-6)
    np.testing.assert_allclose(row[:3], np.asarray(transform.center) / 0.06, rtol=1e-12)
    restored = unpack_cube_transform(row, frame_half_size=0.06, object_scale=0.9)
    np.testing.assert_allclose(restored.center, transform.center, rtol=1e-12)
    assert restored.scale == pytest.approx(transform.scale, rel=1e-6)


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
