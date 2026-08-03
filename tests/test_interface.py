import numpy as np
import pytest
import trimesh

from cod_vae import CODVAE, CODVAEBase


@pytest.fixture(scope="module", params=["torch", "jax"])
def model(request, tiny_config, tiny_params):
    pytest.importorskip(request.param)
    return CODVAE(tiny_config, tiny_params, backend=request.param)


def test_factory_returns_backend(model):
    assert isinstance(model, CODVAEBase)
    assert model.backend in ("torch", "jax")


def test_encode_decode_mesh(model, meshes):
    latent, transform = model.encode_mesh(
        meshes[0], num_points=256, seed=0, return_transform=True
    )
    assert latent.shape == (model.config.num_latents, model.config.latent_dim)
    reconstruction = model.decode_mesh(latent, resolution=16, transform=transform)
    assert isinstance(reconstruction, trimesh.Trimesh)


def test_encode_decode_mesh_batched(model, meshes):
    latents, transforms = model.encode_mesh(
        meshes, num_points=256, seed=0, return_transform=True
    )
    assert latents.shape == (
        len(meshes),
        model.config.num_latents,
        model.config.latent_dim,
    )
    reconstructions = model.decode_mesh(latents, resolution=16, transform=transforms)
    assert isinstance(reconstructions, list) and len(reconstructions) == len(meshes)


def test_decode_chunking_consistent(model, point_batch):
    latents = model.encode(point_batch)
    queries = np.random.default_rng(3).uniform(-1, 1, (777, 3)).astype(np.float32)
    a = model.decode(latents, queries, chunk_size=100)
    b = model.decode(latents, queries, chunk_size=4096)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert a.shape == (len(point_batch), 777)


def test_decode_planes_logits_matches_decode(model, point_batch):
    latents = model.encode(point_batch)
    queries = (
        np.random.default_rng(4)
        .uniform(-1, 1, (len(point_batch), 333, 3))
        .astype(np.float32)
    )
    planes = model.decode_planes(latents)
    a = model.decode_logits(planes, queries)
    b = model.decode_logits(planes, queries, chunk_size=100)
    reference = model.decode(latents, queries)
    np.testing.assert_allclose(a, reference, atol=1e-5)
    np.testing.assert_allclose(b, reference, atol=1e-5)
    assert a.shape == (len(point_batch), 333)


def test_decode_planes_logits_reject_unbatched(model, point_batch):
    latents = model.encode(point_batch)
    with pytest.raises(ValueError):
        model.decode_planes(latents[0])
    planes = model.decode_planes(latents)
    with pytest.raises(ValueError):
        model.decode_logits(planes, np.zeros((5, 3), dtype=np.float32))


def test_occupancy_loss_matches_backend(model, point_batch):
    rng = np.random.default_rng(5)
    latents = model.encode(point_batch)
    num_vol = 200
    queries = rng.uniform(-1, 1, (len(point_batch), num_vol + 100, 3)).astype(
        np.float32
    )
    labels = (rng.random((len(point_batch), num_vol + 100)) > 0.5).astype(np.float32)
    reference = model.occupancy_loss(latents, queries, labels, num_vol)
    assert reference.shape == (len(point_batch),)
    assert np.all(np.isfinite(reference))

    logits = model.decode(latents, queries)
    if model.backend == "torch":
        import torch

        from cod_vae.torch import occupancy_loss

        backend_loss = occupancy_loss(
            torch.from_numpy(logits), torch.from_numpy(labels), num_vol
        ).numpy()
    else:
        from cod_vae.jax import occupancy_loss

        backend_loss = np.asarray(occupancy_loss(logits, labels, num_vol))
    np.testing.assert_allclose(reference, backend_loss, rtol=1e-5, atol=1e-6)


def test_save_load_roundtrip(model, tmp_path, point_batch):
    path = tmp_path / "model.npz"
    model.save(path)
    reloaded = CODVAE.load(path, backend=model.backend)
    np.testing.assert_allclose(
        model.encode(point_batch), reloaded.encode(point_batch), atol=1e-6
    )
