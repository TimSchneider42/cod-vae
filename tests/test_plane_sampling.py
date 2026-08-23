import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cod_vae.jax import plane_sampling
from cod_vae.jax.plane_sampling import _native_sum, sample_planes_sum


@pytest.fixture(autouse=True)
def interpret_kernel(monkeypatch):
    """
    Run the Pallas kernel in the interpreter so the tests work without a GPU, with a
    small block so several grid programs accumulate into each plane.

    The interpreter resolves duplicate indices within one atomic call as
    last-write-wins instead of accumulating (GPU hardware atomics accumulate), so
    these tests place queries in pairwise-distinct texels away from the clamping
    border; the full duplicate/boundary behavior is GPU-verified by
    cod-vae-runs/scripts/check_sampler.py.
    """
    monkeypatch.setattr(plane_sampling, "_interpret", True)
    monkeypatch.setattr(plane_sampling, "_BLOCK_N", 8)


def _distinct_texel_case(batch=2, channels=4, resolution=18, queries=16):
    # Queries on the texel diagonal (pixel i + a fraction), one per texel, so no two
    # lanes of any atomic call touch the same texel corner; resolution 18 > 16 + 1
    # keeps every corner in bounds.
    rng = np.random.default_rng(0)
    pixels = np.arange(queries)[:, None] + rng.uniform(0.1, 0.9, (queries, 3))
    q = (2.0 * pixels + 1.0) / resolution - 1.0
    queries_arr = jnp.asarray(np.broadcast_to(q, (batch, queries, 3)), jnp.float32)
    planes = jnp.asarray(
        rng.standard_normal((batch, 3, resolution, resolution, channels)), jnp.float32
    )
    cotangent = jnp.asarray(
        rng.standard_normal((batch, queries, channels)), jnp.float32
    )
    return planes, queries_arr, cotangent


def test_forward_matches_native():
    rng = np.random.default_rng(1)
    planes = jnp.asarray(rng.standard_normal((2, 3, 9, 7, 4)), jnp.float32)
    queries = jnp.asarray(rng.uniform(-1.0, 0.999, (2, 50, 3)), jnp.float32)
    np.testing.assert_array_equal(
        sample_planes_sum(planes, queries), _native_sum(planes, queries)
    )


def test_gradients_match_native():
    planes, queries, cotangent = _distinct_texel_case()

    def loss(fn):
        return lambda p, q: jnp.sum(fn(p, q) * cotangent)

    dp_custom, dq_custom = jax.grad(loss(sample_planes_sum), argnums=(0, 1))(
        planes, queries
    )
    dp_native, dq_native = jax.grad(loss(_native_sum), argnums=(0, 1))(planes, queries)
    # Summation order differs between the scatter kernel and XLA's adjoint.
    np.testing.assert_allclose(dp_custom, dp_native, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(dq_custom, dq_native, atol=1e-5, rtol=1e-5)


def test_gradients_match_native_under_jit():
    planes, queries, cotangent = _distinct_texel_case(batch=1, queries=8)

    def loss(fn):
        return lambda p, q: jnp.sum(fn(p, q) * cotangent)

    dp_custom = jax.jit(jax.grad(loss(sample_planes_sum)))(planes, queries)
    dp_native = jax.grad(loss(_native_sum))(planes, queries)
    np.testing.assert_allclose(dp_custom, dp_native, atol=1e-5, rtol=1e-5)


def test_vmap_gradients_match_per_element():
    # custom_vjp batches its fwd/bwd and pallas_call has a batching rule, so the
    # sampler is vmappable; this pins that contract (GPU-verified separately).
    planes, queries, cotangent = _distinct_texel_case(batch=1, queries=8)
    stacked_p = jnp.stack([planes, planes * 2.0])
    stacked_q = jnp.stack([queries, queries])

    def loss(p, q):
        return jnp.sum(sample_planes_sum(p, q) * cotangent)

    out = jax.vmap(jax.grad(loss))(stacked_p, stacked_q)
    ref = jnp.stack([jax.grad(loss)(stacked_p[i], stacked_q[i]) for i in range(2)])
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_border_corners_are_dropped(monkeypatch):
    # A single query (block 1: no padding lanes, so no duplicate clamped indices in
    # the interpreter) in the outermost texel: the out-of-range corners must be
    # dropped exactly like the native zero-padding path drops them.
    monkeypatch.setattr(plane_sampling, "_BLOCK_N", 1)
    planes = jnp.asarray(
        np.random.default_rng(2).standard_normal((1, 3, 4, 4, 2)), jnp.float32
    )
    queries = jnp.full((1, 1, 3), 0.999, jnp.float32)

    def loss(fn):
        return lambda p, q: jnp.sum(fn(p, q))

    dp_custom = jax.grad(loss(sample_planes_sum))(planes, queries)
    dp_native = jax.grad(loss(_native_sum))(planes, queries)
    np.testing.assert_allclose(dp_custom, dp_native, atol=1e-6, rtol=1e-6)


def test_grid_sample_matches_torch_reference():
    # Pin the channel-last layout semantics against torch.nn.functional.grid_sample
    # (zero padding, align_corners=False), which the docstring promises: a channel-
    # first torch plane sampled by torch must match our channel-last plane sampled
    # by _grid_sample_plane at the same coordinates.
    torch = pytest.importorskip("torch")
    from cod_vae.jax.model import _grid_sample_plane

    rng = np.random.default_rng(5)
    plane_cf = rng.standard_normal((1, 6, 11, 13)).astype(np.float32)  # (1, C, H, W)
    coords = rng.uniform(-1.0, 0.999, (1, 40, 2)).astype(np.float32)

    ref = torch.nn.functional.grid_sample(
        torch.from_numpy(plane_cf),
        torch.from_numpy(coords)[:, None],  # (1, 1, N, 2), x indexes width
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, :, 0].T.numpy()  # -> (N, C)

    out = _grid_sample_plane(
        jnp.asarray(np.moveaxis(plane_cf[0], 0, -1)),  # (H, W, C)
        jnp.asarray(coords[0]),
    )
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_patches_to_planes_matches_channel_first_formulation():
    # The channel-last unpatchify must be exactly the old channel-first result with
    # the channel axis moved: same patch tiling, same H/W orientation.
    from cod_vae.config import CODVAEConfig
    from cod_vae.jax.model import _patches_to_planes

    config = CODVAEConfig(
        num_latents=4,
        latent_dim=8,
        query_dim=5,
        decoder_output_resolution=12,
        decoder_output_patch_size=4,
    )
    resolution = config.plane_resolution
    rng = np.random.default_rng(6)
    patches = jnp.asarray(
        rng.standard_normal(
            (2, 3 * resolution**2, config.decoder_output_patch_size**2 * config.query_dim)
        ),
        jnp.float32,
    )

    reference = (
        patches.reshape(2, 3, resolution, resolution, 4, 4, config.query_dim)
        .transpose(0, 1, 6, 2, 4, 3, 5)
        .reshape(2, 3, config.query_dim, config.decoder_output_resolution, -1)
    )
    out = _patches_to_planes(patches, config)
    assert out.shape == (2, 3, 12, 12, 5)
    np.testing.assert_array_equal(out.transpose(0, 1, 4, 2, 3), reference)


def test_fp16_planes_gradients_match_native():
    # Half-precision model path: fp16 planes, float32 cotangent; the custom backward
    # must round the plane cotangent to fp16 exactly like the native adjoint does.
    planes, queries, cotangent = _distinct_texel_case()
    planes = planes.astype(jnp.float16)

    def loss(fn):
        return lambda p, q: jnp.sum(fn(p, q) * cotangent)

    dp_custom = jax.grad(loss(sample_planes_sum))(planes, queries)
    dp_native = jax.grad(loss(_native_sum))(planes, queries)
    assert dp_custom.dtype == jnp.float16
    np.testing.assert_allclose(
        np.asarray(dp_custom, np.float32), np.asarray(dp_native, np.float32),
        atol=2e-3, rtol=2e-2,
    )


def test_uncertainty_mult_mode_layout():
    # decode_uncertainty consumes channel-last single-channel planes via the native
    # mult-mode path; pin its value against a direct per-plane product.
    from cod_vae.jax.model import _grid_sample_plane, _sample_planes

    rng = np.random.default_rng(7)
    planes = jnp.asarray(rng.standard_normal((1, 3, 6, 6, 1)), jnp.float32)
    queries = jnp.asarray(rng.uniform(-0.9, 0.9, (1, 9, 3)), jnp.float32)

    out = _sample_planes(planes, queries, mode="mult")[0, :, 0]
    expected = np.ones(9, np.float32)
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        expected *= np.asarray(
            _grid_sample_plane(planes[0, axis], queries[0][:, other])[:, 0]
        )
    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)
