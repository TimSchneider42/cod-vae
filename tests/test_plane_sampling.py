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
        rng.standard_normal((batch, 3, channels, resolution, resolution)), jnp.float32
    )
    cotangent = jnp.asarray(
        rng.standard_normal((batch, queries, channels)), jnp.float32
    )
    return planes, queries_arr, cotangent


def test_forward_matches_native():
    rng = np.random.default_rng(1)
    planes = jnp.asarray(rng.standard_normal((2, 3, 4, 9, 7)), jnp.float32)
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


def test_border_corners_are_dropped(monkeypatch):
    # A single query (block 1: no padding lanes, so no duplicate clamped indices in
    # the interpreter) in the outermost texel: the out-of-range corners must be
    # dropped exactly like the native zero-padding path drops them.
    monkeypatch.setattr(plane_sampling, "_BLOCK_N", 1)
    planes = jnp.asarray(
        np.random.default_rng(2).standard_normal((1, 3, 2, 4, 4)), jnp.float32
    )
    queries = jnp.full((1, 1, 3), 0.999, jnp.float32)

    def loss(fn):
        return lambda p, q: jnp.sum(fn(p, q))

    dp_custom = jax.grad(loss(sample_planes_sum))(planes, queries)
    dp_native = jax.grad(loss(_native_sum))(planes, queries)
    np.testing.assert_allclose(dp_custom, dp_native, atol=1e-6, rtol=1e-6)
