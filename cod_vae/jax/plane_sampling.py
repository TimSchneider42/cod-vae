"""
Custom backward for triplane bilinear sampling (Pallas/Triton).

XLA's adjoint of the sampling gather is a scatter-add into the (C, H, W) plane
gradients: for every query it issues C strided 4-byte atomic adds per texel corner,
which dominates the decode-through-decoder training step at production batch sizes
(~30 ms of an 86 ms step at batch 1024). The kernel here scatters into a
channel-LAST (H, W, C) buffer instead, so each corner update is one contiguous
C-vector of atomics, and the result is transposed to the model's (C, H, W) layout in
a single dense pass that XLA fuses with the downstream patches-gradient permute.

:func:`sample_planes_sum` is numerically the same computation as the native
sum-mode sampling loop in :mod:`cod_vae.jax.model` (align_corners=False, zero
padding), differing only in float32 summation order. The forward pass stays XLA's
native gather; only the backward scatter is custom. The gradient with respect to the
queries is taken from the native implementation's VJP, so callers that stop the
transform gradient (the production path) pay nothing for it -- XLA removes it as
dead code.

The kernel requires a GPU; :data:`_interpret` runs it in Pallas's interpreter for
CPU-only tests. Beware that the interpreter resolves duplicate indices within one
atomic call as last-write-wins instead of accumulating (GPU hardware atomics
accumulate correctly), so interpreter-based tests must keep each atomic call's
target texels pairwise distinct.

vmap is supported: ``jax.custom_vjp`` batches the fwd/bwd functions and
``pallas_call`` has its own batching rule that lifts the mapped axis into the launch
grid. Verified on the GPU Triton lowering against per-element loops -- gradients
match to float32 noise (and to fp16 rounding for a half-precision model).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

__all__ = ["sample_planes_sum"]

_BLOCK_N = 128

# Set to True to run the kernel in Pallas's interpreter (CPU-only tests).
_interpret = False


def _native_sum(planes: jnp.ndarray, queries: jnp.ndarray) -> jnp.ndarray:
    """The native gather formulation; forward pass and query-gradient reference."""
    from . import model

    result = None
    for axis in range(3):
        other_axes = [j for j in range(3) if j != axis]
        features = jax.vmap(model._grid_sample_plane)(
            planes[:, axis], queries[..., other_axes]
        )
        result = features if result is None else result + features
    return result


def _scatter_kernel(
    coords_ref, dfeat_ref, zeros_ref, out_ref, *, num_queries, height, width
):
    # Blocked refs: coords (BLOCK_N, 2) and dfeat (BLOCK_N, C) hold this program's
    # query block, out (H, W, C) this program's whole plane-gradient slice.
    del zeros_ref  # aliased with out_ref; exists only to zero-initialize it
    offsets = pl.program_id(2) * _BLOCK_N + jnp.arange(_BLOCK_N)
    in_block = offsets < num_queries

    # Out-of-block lanes are neutralized by value (zero contribution at a clamped
    # index) rather than by masking the atomics, which Pallas's interpreter does not
    # support; the where's also keep boundary-block padding out of the index math.
    cx = jnp.where(in_block, coords_ref[:, 0], 0.0)
    cy = jnp.where(in_block, coords_ref[:, 1], 0.0)
    dfeat = jnp.where(in_block[:, None], dfeat_ref[...], 0.0)

    # Pixel mapping and corner weights exactly as in model._grid_sample_plane
    # (align_corners=False; out-of-range corners are dropped, i.e. zero padding).
    x = ((cx + 1.0) * width - 1.0) / 2.0
    y = ((cy + 1.0) * height - 1.0) / 2.0
    x0f, y0f = jnp.floor(x), jnp.floor(y)
    x0, y0 = x0f.astype(jnp.int32), y0f.astype(jnp.int32)
    channels = jnp.arange(dfeat.shape[-1])
    for xi, wx in ((x0, 1.0 - (x - x0f)), (x0 + 1, x - x0f)):
        for yi, wy in ((y0, 1.0 - (y - y0f)), (y0 + 1, y - y0f)):
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & in_block
            weight = jnp.where(valid, wx * wy, 0.0)
            plt.atomic_add(
                out_ref,
                (
                    jnp.clip(yi, 0, height - 1)[:, None],
                    jnp.clip(xi, 0, width - 1)[:, None],
                    channels[None, :],
                ),
                weight[:, None] * dfeat,
            )


def _scatter_plane_grads(
    coords: jnp.ndarray, dfeat: jnp.ndarray, height: int, width: int
) -> jnp.ndarray:
    """
    Scatter query-feature cotangents (B, N, C) through the bilinear weights at
    normalized coordinates (B, 3, N, 2) into plane gradients (B, 3, C, H, W).
    """
    batch_size, num_axes, num_queries, _ = coords.shape
    channels = dfeat.shape[-1]
    zeros = jnp.zeros((batch_size, num_axes, height, width, channels), jnp.float32)
    grads = pl.pallas_call(
        functools.partial(
            _scatter_kernel, num_queries=num_queries, height=height, width=width
        ),
        grid=(batch_size, num_axes, pl.cdiv(num_queries, _BLOCK_N)),
        in_specs=[
            pl.BlockSpec((None, None, _BLOCK_N, 2), lambda b, a, nb: (b, a, nb, 0)),
            pl.BlockSpec((None, _BLOCK_N, channels), lambda b, a, nb: (b, nb, 0)),
            pl.BlockSpec(
                (None, None, height, width, channels),
                lambda b, a, nb: (b, a, 0, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (None, None, height, width, channels), lambda b, a, nb: (b, a, 0, 0, 0)
        ),
        out_shape=jax.ShapeDtypeStruct(zeros.shape, zeros.dtype),
        input_output_aliases={2: 0},
        # Force the Triton lowering: refs stay in global memory there, whereas the
        # default Mosaic GPU backend stages blocks through shared memory, which the
        # whole-plane gradient block does not fit (and its warpgroup lowering lacks
        # some of the primitives used here).
        compiler_params=plt.CompilerParams(),
        interpret=_interpret,
    )(coords, dfeat, zeros)
    return grads.transpose(0, 1, 4, 2, 3)


@jax.custom_vjp
def sample_planes_sum(planes: jnp.ndarray, queries: jnp.ndarray) -> jnp.ndarray:
    """
    Sum-mode triplane sampling (planes (B, 3, C, H, W) in the model's compute dtype,
    queries (B, N, 3) float32 already clipped to the sampling range, float32 output)
    with the custom backward scatter.
    """
    return _native_sum(planes, queries)


def _fwd(planes, queries):
    return _native_sum(planes, queries), (planes, queries)


def _bwd(residuals, dfeat):
    planes, queries = residuals
    height, width = planes.shape[-2:]
    # queries[..., 0] indexes the width axis of each plane, [..., 1] the height axis;
    # the per-axis coordinate pairs mirror the native loop's queries[..., other_axes].
    coords = jnp.stack(
        [queries[..., [1, 2]], queries[..., [0, 2]], queries[..., [0, 1]]], axis=1
    )
    # The scatter accumulates in float32; a half-precision model receives its plane
    # cotangent rounded to the plane dtype, exactly as the former explicit
    # float32-cast-then-sample formulation did through the cast's adjoint.
    dplanes = _scatter_plane_grads(coords, dfeat, height, width).astype(planes.dtype)
    _, native_vjp = jax.vjp(lambda q: _native_sum(planes, q), queries)
    (dqueries,) = native_vjp(dfeat)
    return dplanes, dqueries


sample_planes_sum.defvjp(_fwd, _bwd)
