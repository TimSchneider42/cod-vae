"""
Pure-functional JAX implementation of COD-VAE.

Encoding: point cloud in [-1, 1]^3 -> transformer encoder -> num_latents x latent_dim
posterior moments. Decoding: latent -> transformer decoder with uncertainty-based token
pruning -> triplane features -> occupancy logits (positive inside) at query points.

All functions operate on batched inputs and take the parameters as a flat mapping from
the original torch state-dict names to arrays (see :mod:`cod_vae.checkpoint`);
:class:`CODVAEJax` wraps them behind the backend-independent numpy interface with
jit-compiled functions.

The implementation matches the reference torch implementation, including its
farthest-point-sampling shim that canonicalizes the sample order by lexicographic
sorting (the encoder is permutation-equivariant, so this only fixes the latent token
order) and torch.nn.functional.grid_sample semantics (bilinear, zero padding,
align_corners=False), which are replicated exactly in :func:`_grid_sample_plane`.

For training, functions accept an optional :class:`DropPath` carrying the stochastic
depth rate and rng; passing None (the default) yields deterministic eval-mode behavior.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

import trimesh

from ..base import CODVAEBase
from ..checkpoint import Params
from ..config import AttentionImplementation, CODVAEConfig
from ..mesh import (
    CubeTransform,
    occupancy_grid_to_mesh,
    occupancy_grid_to_mesh_warp,
)

__all__ = [
    "CODVAEJax",
    "DropPath",
    "farthest_point_sampling",
    "encode",
    "encode_embed",
    "encode_moments",
    "decode_latents",
    "decode_embed",
    "decode_planes",
    "decode_logits",
    "decode_logits_full",
    "decode_full",
    "decode_uncertainty",
    "split_full_latent",
]

logger = logging.getLogger(__name__)

_LAYER_NORM_EPS = 1e-5


class DropPath:
    """
    Stochastic depth on residual branches (train mode only). Holds the drop rate and an
    rng key; every application consumes a fresh subkey. Pass None instead of an instance
    for eval-mode (identity) behavior.
    """

    def __init__(self, rate: float, key: jax.Array):
        self.rate = rate
        self.key = key

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.rate == 0.0:
            return x
        self.key, subkey = jax.random.split(self.key)
        keep = 1.0 - self.rate
        mask = jax.random.bernoulli(subkey, keep, (x.shape[0],) + (1,) * (x.ndim - 1))
        return x * mask / keep


def _drop(dp: DropPath | None, x: jnp.ndarray) -> jnp.ndarray:
    return x if dp is None else dp(x)


def _linear(params, name: str, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params[f"{name}.weight"].T + params[f"{name}.bias"]


def _layer_norm(params, name: str, x: jnp.ndarray) -> jnp.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + _LAYER_NORM_EPS)
    return normalized * params[f"{name}.weight"] + params[f"{name}.bias"]


def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    # The reference uses torch's exact (erf-based) GELU, not the tanh approximation.
    return jax.nn.gelu(x, approximate=False)


def _geglu(x: jnp.ndarray) -> jnp.ndarray:
    x, gates = jnp.split(x, 2, axis=-1)
    return x * _gelu(gates)


def _attention(
    params,
    name: str,
    num_heads: int,
    query: jnp.ndarray,
    source: jnp.ndarray,
    attn_impl: AttentionImplementation = "default",
) -> jnp.ndarray:
    """
    torch.nn.MultiheadAttention (batch_first, packed QKV projection).

    With attn_impl="cudnn" the scores are computed by cuDNN's fused kernel, which never
    materializes the (tokens x tokens) score matrix -- the dominant term in the backward
    pass of long sequences. Numerically it differs only by floating-point reassociation.
    It requires an even sequence length and a cuDNN able to build a plan for the shape,
    and raises rather than falling back if either does not hold.
    """
    weight = params[f"{name}.in_proj_weight"]
    bias = params[f"{name}.in_proj_bias"]
    embed_dim = query.shape[-1]
    q = query @ weight[:embed_dim].T + bias[:embed_dim]
    k = source @ weight[embed_dim : 2 * embed_dim].T + bias[embed_dim : 2 * embed_dim]
    v = source @ weight[2 * embed_dim :].T + bias[2 * embed_dim :]

    # The decoder is reached with a varying number of leading batch-like axes: (B, N, C)
    # from the inference path, but (E, B, N, C) from the critic loss, where E is CrossQ's
    # stacked critic ensemble. Both branches below want exactly one batch axis -- cuDNN's
    # kernel takes rank 4, and the manual path's transposes are written for it -- so
    # collapse whatever leading axes the caller has into one here and restore them at the
    # end. Broadcasting first lets cross-attention pair a per-ensemble query with a shared
    # source.
    lead = jnp.broadcast_shapes(q.shape[:-2], k.shape[:-2])
    query_len, source_len = q.shape[-2], k.shape[-2]
    head_dim = embed_dim // num_heads

    def split_heads(x: jnp.ndarray, length: int) -> jnp.ndarray:
        # (B, N, H, D); the manual path transposes to (B, H, N, D) below, the fused one
        # takes this layout directly.
        x = jnp.broadcast_to(x, (*lead, length, x.shape[-1]))
        return x.reshape(-1, length, num_heads, head_dim)

    q = split_heads(q, query_len)
    k, v = split_heads(k, source_len), split_heads(v, source_len)
    if attn_impl == "cudnn":
        # cuDNN rejects odd sequence lengths on the backward pass, and the decoder's
        # token count is always odd: 3 * plane_resolution ** 2 patches plus one register
        # token. Pad to even and mask the padded key so it contributes nothing. The
        # padded query row still attends to real keys, so its softmax is well defined
        # (a fully masked row would produce NaNs that survive into the gradient); it is
        # sliced off below.
        pad_query, pad_source = query_len % 2, source_len % 2
        mask = None
        if pad_query or pad_source:
            pad = ((0, 0), (0, 1), (0, 0), (0, 0))
            if pad_query:
                q = jnp.pad(q, pad)
            if pad_source:
                k, v = jnp.pad(k, pad), jnp.pad(v, pad)
                mask = jnp.ones((1, 1, q.shape[1], k.shape[1]), dtype=bool)
                mask = mask.at[..., -1].set(False)
        # Default scale is 1/sqrt(head_dim), matching the manual path.
        out = jax.nn.dot_product_attention(q, k, v, mask=mask, implementation="cudnn")[
            :, :query_len
        ]
    else:
        q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
        out = (jax.nn.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3)
    out = out.reshape(*lead, query_len, embed_dim)
    return _linear(params, f"{name}.out_proj", out)


def _ffn_geglu(params, name: str, x: jnp.ndarray) -> jnp.ndarray:
    return _linear(params, f"{name}.c_proj", _geglu(_linear(params, f"{name}.c_fc", x)))


def _self_attn_block(
    params,
    name: str,
    num_heads: int,
    x: jnp.ndarray,
    dp: DropPath | None = None,
    attn_impl: AttentionImplementation = "default",
) -> jnp.ndarray:
    """Pre-LN self-attention followed by a GEGLU FFN, both residual with DropPath."""
    h = _layer_norm(params, f"{name}.ln_1", x)
    x = x + _drop(dp, _attention(params, f"{name}.attn", num_heads, h, h, attn_impl))
    h = _ffn_geglu(params, f"{name}.mlp", _layer_norm(params, f"{name}.ln_2", x))
    return x + _drop(dp, h)


def _cross_attn_block(
    params,
    name: str,
    num_heads: int,
    x: jnp.ndarray,
    source: jnp.ndarray,
    dp: DropPath | None = None,
    attn_impl: AttentionImplementation = "default",
) -> jnp.ndarray:
    """Residual cross-attention, self-attention, and GEGLU FFN with DropPath."""
    h = _attention(
        params,
        f"{name}.cross_attn",
        num_heads,
        _layer_norm(params, f"{name}.ln_cross", x),
        _layer_norm(params, f"{name}.ln_source", source),
        attn_impl,
    )
    x = x + _drop(dp, h)
    h = _layer_norm(params, f"{name}.ln_1", x)
    x = x + _drop(
        dp, _attention(params, f"{name}.self_attn", num_heads, h, h, attn_impl)
    )
    h = _ffn_geglu(params, f"{name}.mlp", _layer_norm(params, f"{name}.ln_2", x))
    return x + _drop(dp, h)


def _cross_attn(
    params,
    name: str,
    num_heads: int,
    x: jnp.ndarray,
    source: jnp.ndarray,
    attn_impl: AttentionImplementation = "default",
) -> jnp.ndarray:
    """Normalized cross-attention without residual or output head (no DropPath)."""
    return _attention(
        params,
        f"{name}.cross_attn",
        num_heads,
        _layer_norm(params, f"{name}.ln_cross", x),
        _layer_norm(params, f"{name}.ln_source", source),
        attn_impl,
    )


def _point_embed(params, points: jnp.ndarray) -> jnp.ndarray:
    """Sinusoidal point embedding: (B, N, 3) -> (B, N, embed_dim)."""
    projections = points @ params["autoencoder.point_embed.basis"]
    features = jnp.concatenate(
        [jnp.sin(projections), jnp.cos(projections), points], axis=-1
    )
    return _linear(params, "autoencoder.point_embed.mlp", features)


def farthest_point_sampling(
    points: jnp.ndarray, num_samples: int, canonicalize: bool = True
) -> jnp.ndarray:
    """
    Farthest point sampling of an unbatched point cloud (N, 3) -> (num_samples, 3),
    starting from the first point. With canonicalize=True, the (arbitrary) FPS ordering
    is replaced by lexicographic order so the result is a deterministic function of the
    point set.
    """

    points = jnp.asarray(points)

    def body(j, state):
        indices, distances = state
        delta = points - points[indices[j - 1]]
        distances = jnp.minimum(distances, jnp.sum(delta * delta, axis=-1))
        return indices.at[j].set(jnp.argmax(distances)), distances

    indices = jnp.zeros(num_samples, dtype=jnp.int32)
    distances = jnp.full(points.shape[0], jnp.inf, dtype=points.dtype)
    indices, _ = jax.lax.fori_loop(1, num_samples, body, (indices, distances))
    selected = points[indices]
    if canonicalize:
        selected = selected[jnp.lexsort(selected.T[::-1])]
    return selected


def encode_embed(
    params: Mapping[str, jnp.ndarray],
    points: jnp.ndarray,
    *,
    config: CODVAEConfig,
    dp: DropPath | None = None,
) -> jnp.ndarray:
    """Encode surface point clouds (B, N, 3) into latent embeddings (B, L, embed_dim)."""
    num_heads = config.num_heads
    z = jax.vmap(partial(farthest_point_sampling, num_samples=config.num_latents))(
        points
    )
    z = _layer_norm(params, "autoencoder.norm_latent", _point_embed(params, z))

    point_features = _layer_norm(
        params, "autoencoder.encoder.norm_point", _point_embed(params, points)
    )
    patch_pos = jax.vmap(
        partial(farthest_point_sampling, num_samples=config.encoder_num_patches)
    )(points)
    patches = _layer_norm(
        params, "autoencoder.encoder.norm_point", _point_embed(params, patch_pos)
    )

    for block in range(config.encoder_num_blocks):
        name = f"autoencoder.encoder.blocks.{block}"
        patches = patches + _cross_attn(
            params,
            f"{name}.points2patch",
            num_heads,
            patches,
            point_features,
            attn_impl=config.attention_implementation,
        )
        # The reference applies ln_ffn and the FFN's own LayerNorm back to back.
        h = _layer_norm(params, f"{name}.ln_ffn", patches)
        h = _layer_norm(params, f"{name}.patch_ffn.ln", h)
        h = _linear(params, f"{name}.patch_ffn.mlp.0", h)
        h = _linear(params, f"{name}.patch_ffn.mlp.2", _geglu(h))
        patches = patches + _drop(dp, h)
        for layer in range(config.encoder_num_layers_per_block):
            patches = _self_attn_block(
                params,
                f"{name}.processing_layers.{layer}",
                num_heads,
                patches,
                dp,
                attn_impl=config.attention_implementation,
            )
        z = _cross_attn_block(
            params,
            f"{name}.patch2latents",
            num_heads,
            z,
            patches,
            dp,
            attn_impl=config.attention_implementation,
        )
        point_features = point_features + _cross_attn(
            params,
            f"{name}.latents2points",
            num_heads,
            point_features,
            z,
            attn_impl=config.attention_implementation,
        )

    return _cross_attn_block(
        params,
        "autoencoder.encoder.last_block",
        num_heads,
        z,
        point_features,
        dp,
        attn_impl=config.attention_implementation,
    )


def encode_moments(
    params: Mapping[str, jnp.ndarray], z_embed: jnp.ndarray
) -> jnp.ndarray:
    """Posterior moments (B, L, 2 * latent_dim): mean and log-variance."""
    return _linear(
        params, "latent_proj_in.1", _layer_norm(params, "latent_proj_in.0", z_embed)
    )


def encode(
    params: Mapping[str, jnp.ndarray], points: jnp.ndarray, *, config: CODVAEConfig
) -> jnp.ndarray:
    """
    Encode surface point clouds (B, N, 3) in [-1, 1]^3 into their latent representation
    (B, num_latents, latent_dim). Returns the deterministic posterior mean (not a
    sample), so the latent is a well-defined function of the point cloud.
    """
    moments = encode_moments(params, encode_embed(params, points, config=config))
    return moments[..., : config.latent_dim]


def decode_latents(
    params, latent: jnp.ndarray, *, config: CODVAEConfig, dp: DropPath | None = None
) -> jnp.ndarray:
    """Map latents (B, num_latents, latent_dim) back to embeddings (B, num_latents, embed_dim)."""
    z = _linear(params, "latent_proj_out.0", latent)
    z = _layer_norm(params, "latent_proj_out.1", z)
    for layer in range(config.num_latent_layers):
        z = _self_attn_block(
            params,
            f"latent_decoder.transformer.{layer}",
            config.num_heads,
            z,
            dp,
            attn_impl=config.attention_implementation,
        )
    z = _layer_norm(params, "latent_decoder.linear_out.0", z)
    return _linear(params, "latent_decoder.linear_out.1", z)


def _decode_init_tokens(
    params, z: jnp.ndarray, *, config: CODVAEConfig, dp: DropPath | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Initial pass over all triplane patch tokens given latent embeddings z; returns the
    initial tokens (B, num_output_patches, embed_dim) and their per-token uncertainty
    (B, num_output_patches, 1).
    """
    batch_size, embed_dim = z.shape[0], config.embed_dim
    tokens = jnp.broadcast_to(
        params["autoencoder.decoder.mask_pos"],
        (batch_size, config.num_output_patches, embed_dim),
    )
    register = jnp.broadcast_to(
        params["autoencoder.decoder.init_transformer.register"][None],
        (batch_size, 1, embed_dim),
    )
    x = jnp.concatenate([tokens, register], axis=1)
    x = _layer_norm(params, "autoencoder.decoder.init_transformer.ln_pre", x)
    source = _layer_norm(params, "autoencoder.decoder.init_transformer.ln_source", z)
    for layer in range(config.decoder_num_init_layers):
        x = _cross_attn_block(
            params,
            f"autoencoder.decoder.init_transformer.transformer.{layer}",
            config.num_heads,
            x,
            source,
            dp,
            attn_impl=config.attention_implementation,
        )
    init_tokens = x[:, : config.num_output_patches]

    u = _layer_norm(params, "autoencoder.decoder.uncertainty_out.0", init_tokens)
    u = _geglu(_linear(params, "autoencoder.decoder.uncertainty_out.1", u))
    uncertainty = jax.nn.sigmoid(
        _linear(params, "autoencoder.decoder.uncertainty_out.3", u)
    )
    return init_tokens, uncertainty


def _patches_to_planes(patches: jnp.ndarray, config: CODVAEConfig) -> jnp.ndarray:
    batch_size = patches.shape[0]
    resolution = config.plane_resolution
    patch_size = config.decoder_output_patch_size
    patches = patches.reshape(
        batch_size, 3, resolution, resolution, patch_size, patch_size, config.query_dim
    )
    return patches.transpose(0, 1, 6, 2, 4, 3, 5).reshape(
        batch_size, 3, config.query_dim, config.decoder_output_resolution, -1
    )


def decode_embed(
    params: Mapping[str, jnp.ndarray],
    z: jnp.ndarray,
    *,
    config: CODVAEConfig,
    dp: DropPath | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Decode latent embeddings (B, L, embed_dim) into triplane features: returns (planes,
    init_planes, uncertainty_planes) of shapes (B, 3, query_dim, R, R) twice and
    (B, 3, 1, plane_resolution, plane_resolution).
    """
    num_heads = config.num_heads
    batch_size = z.shape[0]
    embed_dim = config.embed_dim

    init_tokens, uncertainty = _decode_init_tokens(params, z, config=config, dp=dp)

    ## uncertainty-based token pruning: only refine the most uncertain tokens
    order = jnp.argsort(-uncertainty[..., 0], axis=1)
    num_keep = config.num_kept_tokens
    keep_indices, prune_indices = order[:, :num_keep], order[:, num_keep:]
    kept = (
        jnp.take_along_axis(init_tokens, keep_indices[..., None], axis=1)
        + params["autoencoder.decoder.mask_token"]
    )
    pruned = jnp.take_along_axis(init_tokens, prune_indices[..., None], axis=1)
    merge_tokens = jnp.broadcast_to(
        params["autoencoder.decoder.merging_module.tokens"],
        (batch_size, config.decoder_num_merged_tokens, embed_dim),
    )
    merged = merge_tokens + _cross_attn(
        params,
        "autoencoder.decoder.merging_module.cross_attn",
        num_heads,
        merge_tokens,
        pruned,
        attn_impl=config.attention_implementation,
    )

    ## refine the kept tokens (with merged pruned tokens and latents as context)
    class_token = (
        params["autoencoder.decoder.transformer.class_token"]
        + params["autoencoder.decoder.transformer.class_pos"]
    )
    x = jnp.concatenate(
        [
            kept,
            merged,
            z,
            jnp.broadcast_to(class_token[None], (batch_size, 1, embed_dim)),
        ],
        axis=1,
    )
    x = _layer_norm(params, "autoencoder.decoder.transformer.ln_pre", x)
    for layer in range(config.decoder_num_layers):
        x = _self_attn_block(
            params,
            f"autoencoder.decoder.transformer.transformer.{layer}",
            num_heads,
            x,
            dp,
            attn_impl=config.attention_implementation,
        )
    refined = x[:, :num_keep]
    full_tokens = init_tokens.at[jnp.arange(batch_size)[:, None], keep_indices].set(
        refined
    )

    ## project tokens to triplane patches; unrefined patches use the initial prediction
    init_patches = _linear(params, "autoencoder.decoder.init_out", init_tokens)
    patches = init_patches + uncertainty * _linear(
        params, "autoencoder.decoder.decoder_out", full_tokens
    )
    resolution = config.plane_resolution
    uncertainty_planes = uncertainty.reshape(batch_size, 3, 1, resolution, resolution)
    return (
        _patches_to_planes(patches, config),
        _patches_to_planes(init_patches, config),
        uncertainty_planes,
    )


def decode_planes(
    params: Mapping[str, jnp.ndarray], latent: jnp.ndarray, *, config: CODVAEConfig
) -> jnp.ndarray:
    """
    Decode latents (B, num_latents, latent_dim) into triplane features
    (B, 3, query_dim, output_resolution, output_resolution).
    """
    z = decode_latents(params, latent, config=config)
    return decode_embed(params, z, config=config)[0]


def _grid_sample_plane(plane: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear sampling of a feature plane (C, H, W) at normalized coordinates (N, 2),
    where coords[:, 0] indexes the width and coords[:, 1] the height axis; matches
    torch.nn.functional.grid_sample with zero padding and align_corners=False.
    """
    channels, height, width = plane.shape
    x = ((coords[:, 0] + 1.0) * width - 1.0) / 2.0
    y = ((coords[:, 1] + 1.0) * height - 1.0) / 2.0
    x0, y0 = jnp.floor(x), jnp.floor(y)
    weights_x = [
        (x0.astype(jnp.int32), 1.0 - (x - x0)),
        (x0.astype(jnp.int32) + 1, x - x0),
    ]
    weights_y = [
        (y0.astype(jnp.int32), 1.0 - (y - y0)),
        (y0.astype(jnp.int32) + 1, y - y0),
    ]
    result = jnp.zeros((coords.shape[0], channels), dtype=plane.dtype)
    for xi, wx in weights_x:
        for yi, wy in weights_y:
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            values = plane[:, jnp.clip(yi, 0, height - 1), jnp.clip(xi, 0, width - 1)]
            result = result + (wx * wy * valid)[:, None] * values.T
    return result


def _sample_planes(planes: jnp.ndarray, queries: jnp.ndarray, mode: str) -> jnp.ndarray:
    """
    Sample triplanes (B, 3, C, R, R) at queries (B, N, 3); sum or multiply planes.

    The interpolation always runs in float32, even for a half-precision model: the
    gradient with respect to the query coordinates is a difference of adjacent texels,
    which cancels catastrophically at half precision (measurably degrading the
    bounding box gradients of :func:`decode_logits_full`). The features are returned in
    float32; callers feeding them to a half-precision head must cast them back.
    """
    planes = planes.astype(jnp.float32)
    queries = jnp.clip(queries.astype(jnp.float32), -1.0, 0.999)
    result = None
    for axis in range(3):
        other_axes = [j for j in range(3) if j != axis]
        features = jax.vmap(_grid_sample_plane)(
            planes[:, axis], queries[..., other_axes]
        )
        if result is None:
            result = features
        else:
            result = result + features if mode == "sum" else result * features
    return result


def decode_logits(
    params: Mapping[str, jnp.ndarray],
    planes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    config: CODVAEConfig,
) -> jnp.ndarray:
    """
    Evaluate occupancy logits (positive inside the object) at query points (B, N, 3) in
    [-1, 1]^3 given triplane features from :func:`decode_planes`; returns (B, N) in the
    parameters' dtype. The queries may be float32 for a half-precision model (see
    :func:`_sample_planes`); only the head runs in the parameters' dtype.
    """
    features = _sample_planes(planes, queries, mode="sum").astype(
        params["autoencoder.head.0.weight"].dtype
    )
    x = _linear(params, "autoencoder.head.0", features)
    x = _linear(params, "autoencoder.head.2", _gelu(x))
    return x[..., 0]


def decode_uncertainty(
    uncertainty_planes: jnp.ndarray, queries: jnp.ndarray
) -> jnp.ndarray:
    """Multiplicative uncertainty (B, N) at query points (B, N, 3)."""
    return _sample_planes(uncertainty_planes, queries, mode="mult")[..., 0]


def split_full_latent(
    full_latent: jnp.ndarray, *, config: CODVAEConfig
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Split full latents (B, num_latents * latent_dim + 4) into latents
    (B, num_latents, latent_dim), bounding box centers (B, 3), and sizes (B,)
    (see :meth:`cod_vae.base.CODVAEBase.pack_full_latent` for the layout).
    """
    dims = config.num_latents * config.latent_dim
    latent = full_latent[:, :dims].reshape(-1, config.num_latents, config.latent_dim)
    return latent, full_latent[:, dims : dims + 3], full_latent[:, dims + 3]


def decode_logits_full(
    params: Mapping[str, jnp.ndarray],
    planes: jnp.ndarray,
    center: jnp.ndarray,
    size: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    config: CODVAEConfig,
    object_scale: float = 0.9,
    stop_transform_gradient: bool = True,
) -> jnp.ndarray:
    """
    Evaluate occupancy logits (B, N) at query points (B, N, 3) given in the [-1, 1]
    normalized world frame, mapping them into the model's cube via the bounding box
    center (B, 3) and size (B,) of a full latent (same frame). Sizes are clamped to
    1e-3 to guard against (near-)zero size values, which would yield an infinite
    cube scale.

    By default (``stop_transform_gradient``), center and size are excluded from the
    gradient: their only gradient path is the triplane interpolation of the mapped
    queries, which is piecewise-constant at the texel scale, noisy under query
    subsampling, and identically zero once the queries are clamped to the cube's
    boundary — callers optimizing the transform should penalize it directly instead.
    Pass ``stop_transform_gradient=False`` to differentiate through the query mapping
    anyway.

    The query mapping runs in float32 for the same reason the interpolation does (see
    :func:`_sample_planes`), so queries given in float32 keep their full precision even
    for a half-precision model.
    """
    if stop_transform_gradient:
        center = jax.lax.stop_gradient(center)
        size = jax.lax.stop_gradient(size)
    scale = object_scale / jnp.maximum(size.astype(jnp.float32), 1e-3)
    cube_queries = (
        queries.astype(jnp.float32) - center.astype(jnp.float32)[:, None, :]
    ) * scale[:, None, None]
    return decode_logits(params, planes, cube_queries, config=config)


def decode_full(
    params: Mapping[str, jnp.ndarray],
    full_latent: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    config: CODVAEConfig,
    object_scale: float = 0.9,
    stop_transform_gradient: bool = True,
) -> jnp.ndarray:
    """
    Evaluate occupancy logits (B, N) of full latents
    (B, num_latents * latent_dim + 4) at query points (B, N, 3) given in the [-1, 1]
    normalized world frame. Differentiable with respect to the latent part; the
    bounding box center and size entries are excluded from the gradient by default
    (see :func:`decode_logits_full` for the rationale and
    ``stop_transform_gradient=False`` to differentiate through them).
    """
    latent, center, size = split_full_latent(full_latent, config=config)
    planes = decode_planes(params, latent, config=config)
    return decode_logits_full(
        params,
        planes,
        center,
        size,
        queries,
        config=config,
        object_scale=object_scale,
        stop_transform_gradient=stop_transform_gradient,
    )


def _resolve_attention(
    requested: AttentionImplementation, device, dtype
) -> AttentionImplementation:
    """
    Resolve "auto" to "cudnn" where the fused kernel can actually run, else "default".
    Three things have to hold, and the last one is only knowable by trying: a CUDA
    device, a half-precision compute dtype (cuDNN rejects float32 outright), and a cuDNN
    able to build an execution plan -- which fails on installations whose NVRTC cannot
    compile, independently of the GPU.
    """
    if requested != "auto":
        return requested
    if device.platform != "gpu":
        logger.info("Attention: using the default kernel (no CUDA device).")
        return "default"
    if dtype not in (jnp.float16, jnp.bfloat16):
        logger.info(
            "Attention: using the default kernel (cuDNN needs float16/bfloat16, this "
            "model computes in %s).",
            jnp.dtype(dtype).name,
        )
        return "default"
    probe = jax.ShapeDtypeStruct((1, 2, 1, 8), dtype)
    try:
        jax.jit(
            lambda q, k, v: jax.nn.dot_product_attention(
                q, k, v, implementation="cudnn"
            )
        ).lower(probe, probe, probe).compile()
    except Exception as e:
        logger.info(
            "Attention: using the default kernel (cuDNN cannot build a plan here: %s).",
            str(e).splitlines()[0][:120],
        )
        return "default"
    logger.info("Attention: using cuDNN's fused kernel.")
    return "cudnn"


class CODVAEJax(CODVAEBase):
    """JAX backend of COD-VAE (see :class:`cod_vae.base.CODVAEBase`)."""

    backend = "jax"

    def __init__(self, config: CODVAEConfig, params: Params, device=None, dtype=None):
        if isinstance(device, str):
            device = jax.devices(device)[0]
        self.device = device if device is not None else jax.devices()[0]
        self.dtype = jnp.float32 if dtype is None else jnp.dtype(dtype)
        config = dataclasses.replace(
            config,
            attention_implementation=_resolve_attention(
                config.attention_implementation, self.device, self.dtype
            ),
        )
        self._grid_queries_cache: dict[int, "jnp.ndarray"] = {}
        # Parameters are committed to the device below; jitted computations follow them.
        self._jit_encode = jax.jit(partial(encode, config=config))
        self._jit_decode_planes = jax.jit(partial(decode_planes, config=config))
        self._jit_decode_logits = jax.jit(partial(decode_logits, config=config))
        self._jit_decode_logits_full = jax.jit(
            lambda params, planes, center, size, object_scale, queries: decode_logits_full(
                params,
                planes,
                center,
                size,
                queries,
                config=config,
                object_scale=object_scale,
            )
        )
        super().__init__(config, params)

    def _load_params(self, params: Params) -> None:
        self.params = {
            key: jax.device_put(jnp.asarray(value, dtype=self.dtype), self.device)
            for key, value in params.items()
        }

    def get_params(self) -> Params:
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in self.params.items()
        }

    def _to_device(self, array, dtype=None) -> "jnp.ndarray":
        """Move an array to the model's device, in the model's compute dtype unless
        overridden. Inputs meant for the transformer must match that dtype, as jnp
        promotion would otherwise silently compute in float32; query points instead
        stay float32 (see :func:`_sample_planes`)."""
        return jax.device_put(
            jnp.asarray(array, dtype=self.dtype if dtype is None else dtype),
            self.device,
        )

    def occupancy_grid_to_mesh(
        self, logits: "jnp.ndarray", transform: CubeTransform | None = None
    ) -> trimesh.Trimesh:
        """
        Turn a dense occupancy logit grid that is still on the device -- as returned by
        :meth:`_decode_grid_native` -- into a mesh, without a host round trip: Warp
        adopts the jax buffer via DLPack and marching cubes runs where the data already
        is. Falls back to the host implementation off CUDA.
        """
        if self.device.platform != "gpu":
            return occupancy_grid_to_mesh(self._to_numpy(logits), transform)
        return occupancy_grid_to_mesh_warp(logits.astype(jnp.float32), transform)

    def _to_numpy(self, array) -> np.ndarray:
        # np.array, not np.asarray: converting a jax array yields a read-only buffer, and
        # consumers such as skimage's marching cubes require a writable one.
        return np.array(array, dtype=np.float32)

    def _encode_native(self, points):
        return self._jit_encode(self.params, self._to_device(points))

    def _grid_queries(self, resolution: int) -> "jnp.ndarray":
        """Dense [-1, 1]^3 grid of shape (resolution ** 3, 3), built and cached on the
        model's device. Same layout as :func:`cod_vae.mesh.grid_queries`, which builds
        the identical grid on the host."""
        cached = self._grid_queries_cache.get(resolution)
        if cached is None:
            axis = jnp.linspace(-1.0, 1.0, resolution, dtype=jnp.float32)
            grid = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
            cached = jax.device_put(grid.reshape(-1, 3), self.device)
            self._grid_queries_cache[resolution] = cached
        return cached

    def _decode_grid_native(self, latents, resolution: int, chunk_size: int):
        # The grid is generated on the device and cached, so nothing is uploaded per
        # call: it is the same array every time and is by far the largest transfer in
        # this path (3 * resolution ** 3 floats, against resolution ** 3 coming back).
        # Chunks keep the fixed shape the jitted decoder was compiled for.
        planes = self._decode_planes(latents)
        queries = self._grid_queries(resolution)
        num_queries = queries.shape[0]
        padded = max(1, -(-num_queries // chunk_size)) * chunk_size
        if padded != num_queries:
            queries = jnp.pad(queries, ((0, padded - num_queries), (0, 0)))
        batch = latents.shape[0]
        logits = jnp.concatenate(
            [
                self._jit_decode_logits(
                    self.params,
                    planes,
                    jnp.broadcast_to(
                        queries[i : i + chunk_size][None], (batch, chunk_size, 3)
                    ),
                )
                for i in range(0, padded, chunk_size)
            ],
            axis=1,
        )
        return logits[:, :num_queries]

    def _decode_planes(self, latents):
        return self._jit_decode_planes(self.params, self._to_device(latents))

    def _decode_logits_native(self, planes, queries):
        return self._jit_decode_logits(
            self.params, planes, self._to_device(queries, jnp.float32)
        )

    def _decode_planes_full(self, full_latents):
        dims = self.config.num_latents * self.config.latent_dim
        latents = full_latents[:, :dims].reshape(
            -1, self.config.num_latents, self.config.latent_dim
        )
        planes = self._jit_decode_planes(self.params, self._to_device(latents))
        center = self._to_device(full_latents[:, dims : dims + 3])
        size = self._to_device(full_latents[:, dims + 3])
        return planes, center, size

    def _decode_logits_full_native(self, handle, queries, object_scale):
        planes, center, size = handle
        return self._jit_decode_logits_full(
            self.params,
            planes,
            center,
            size,
            object_scale,
            self._to_device(queries, jnp.float32),
        )
