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


def test_full_latent_roundtrip(model, meshes):
    from cod_vae import CubeTransform

    full = model.encode_mesh_full(meshes, num_points=256, seed=0)
    assert full.shape == (len(meshes), model.full_latent_size)

    latents, transforms = model.unpack_full_latent(full)
    assert latents.shape == (
        len(meshes),
        model.config.num_latents,
        model.config.latent_dim,
    )
    assert all(isinstance(t, CubeTransform) for t in transforms)
    np.testing.assert_allclose(
        model.pack_full_latent(latents, transforms), full, rtol=1e-5, atol=1e-5
    )

    # Matches encode_mesh with return_transform=True (up to float32 rounding of the
    # transform parameters).
    reference_latents, reference_transforms = model.encode_mesh(
        meshes, num_points=256, seed=0, return_transform=True
    )
    np.testing.assert_allclose(latents, reference_latents)
    for transform, reference in zip(transforms, reference_transforms):
        np.testing.assert_allclose(transform.center, reference.center, atol=1e-6)
        np.testing.assert_allclose(transform.scale, reference.scale, rtol=1e-5)

    # The frame normalization round-trips: packing with a frame half size stores the
    # bounding box center and size in the normalized world frame.
    frame_half_size = 0.06
    full_scaled = model.pack_full_latent(
        latents, transforms, frame_half_size=frame_half_size
    )
    dims = model.config.num_latents * model.config.latent_dim
    np.testing.assert_allclose(
        full_scaled[:, dims:], full[:, dims:] / frame_half_size, rtol=1e-5, atol=1e-5
    )
    _, transforms_scaled = model.unpack_full_latent(
        full_scaled, frame_half_size=frame_half_size
    )
    for transform, reference in zip(transforms_scaled, reference_transforms):
        np.testing.assert_allclose(transform.center, reference.center, atol=1e-6)
        np.testing.assert_allclose(transform.scale, reference.scale, rtol=1e-4)

    # Unbatched variant. Different batch sizes compile to numerically slightly
    # different kernels, hence the tolerance.
    full_single = model.encode_mesh_full(meshes[0], num_points=256, seed=0)
    assert full_single.shape == (model.full_latent_size,)
    np.testing.assert_allclose(full_single, full[0], atol=5e-3)
    latent_single, transform_single = model.unpack_full_latent(full_single)
    assert isinstance(transform_single, CubeTransform)
    np.testing.assert_allclose(latent_single, latents[0], atol=5e-3)

    with pytest.raises(ValueError):
        model.unpack_full_latent(full[:, :-1])


def test_decode_full(model, meshes):
    full = model.encode_mesh_full(meshes, num_points=256, seed=0)
    latents, transforms = model.unpack_full_latent(full)

    # Query in the original mesh frame (= normalized world frame for the default
    # frame_half_size of 1); must match decoding manually mapped queries.
    rng = np.random.default_rng(7)
    queries = np.stack(
        [
            rng.uniform(mesh.bounds[0], mesh.bounds[1], (123, 3)).astype(np.float32)
            for mesh in meshes
        ]
    )
    logits = model.decode_full(full, queries)
    cube_queries = np.stack(
        [t.apply(q) for t, q in zip(transforms, queries)]
    ).astype(np.float32)
    # The backend applies the transform in float32 while the reference maps the
    # queries in float64, hence the tolerance.
    np.testing.assert_allclose(
        logits, model.decode(latents, cube_queries), atol=1e-3
    )

    # Unbatched variant. Different batch sizes compile to numerically slightly
    # different kernels, hence the tolerance.
    logits_single = model.decode_full(full[0], queries[0])
    np.testing.assert_allclose(logits_single, logits[0], atol=0.05)


def test_decode_full_backend_native(model, meshes):
    """The numpy decode_full must dispatch to the backend implementation, and the
    backend-native variant must be differentiable w.r.t. the full latent."""
    full = model.encode_mesh_full(meshes, num_points=256, seed=0)
    rng = np.random.default_rng(8)
    queries = rng.uniform(-1, 1, (len(meshes), 65, 3)).astype(np.float32)
    reference = model.decode_full(full, queries)

    if model.backend == "torch":
        import torch

        full_tensor = (
            torch.from_numpy(full).to(model.device).requires_grad_(True)
        )
        logits = model.module.decode_full(
            full_tensor, torch.from_numpy(queries).to(model.device)
        )
        logits.sum().backward()
        grad = full_tensor.grad.cpu().numpy()
        backend_logits = logits.detach().cpu().numpy()
    else:
        import jax

        from cod_vae.jax import decode_full

        def loss(full_latent):
            return decode_full(
                model.params, full_latent, queries, config=model.config
            ).sum()

        backend_logits = np.asarray(
            jax.jit(
                lambda f: decode_full(model.params, f, queries, config=model.config)
            )(full)
        )
        grad = np.asarray(jax.grad(loss)(full))
    np.testing.assert_allclose(backend_logits, reference, atol=1e-3)
    assert np.all(np.isfinite(grad))
    # Gradients must flow into the latent as well as the center/size entries.
    dims = model.config.num_latents * model.config.latent_dim
    assert np.any(grad[:, :dims] != 0.0)
    assert np.any(grad[:, dims:] != 0.0)


def test_decode_mesh_full(model, meshes):
    full = model.encode_mesh_full(meshes, num_points=256, seed=0)
    reconstructions = model.decode_mesh_full(full, resolution=16)
    assert isinstance(reconstructions, list) and len(reconstructions) == len(meshes)

    latents, transforms = model.unpack_full_latent(full)
    references = model.decode_mesh(latents, resolution=16, transform=transforms)
    for reconstruction, reference in zip(reconstructions, references):
        np.testing.assert_allclose(reconstruction.vertices, reference.vertices)

    single = model.decode_mesh_full(full[0], resolution=16)
    assert isinstance(single, trimesh.Trimesh)


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


def test_occupancy_loss_full(model, meshes):
    full = model.encode_mesh_full(meshes, num_points=256, seed=0)
    rng = np.random.default_rng(9)
    num_vol = 150
    queries = rng.uniform(-1, 1, (len(meshes), num_vol + 50, 3)).astype(np.float32)
    labels = (rng.random((len(meshes), num_vol + 50)) > 0.5).astype(np.float32)
    loss = model.occupancy_loss_full(full, queries, labels, num_vol)
    assert loss.shape == (len(meshes),)

    logits = model.decode_full(full, queries)
    bce = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    reference = 1.0 * bce[:, :num_vol].mean(-1) + 0.1 * bce[:, num_vol:].mean(-1)
    np.testing.assert_allclose(loss, reference, rtol=1e-5, atol=1e-6)


def test_save_load_roundtrip(model, tmp_path, point_batch):
    path = tmp_path / "model.npz"
    model.save(path)
    reloaded = CODVAE.load(path, backend=model.backend)
    np.testing.assert_allclose(
        model.encode(point_batch), reloaded.encode(point_batch), atol=1e-6
    )
