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

import math
from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..base import CODVAEBase
from ..checkpoint import Params
from ..config import CODVAEConfig

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
    "decode_uncertainty",
]

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
        mask = jax.random.bernoulli(
            subkey, keep, (x.shape[0],) + (1,) * (x.ndim - 1)
        )
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
    params, name: str, num_heads: int, query: jnp.ndarray, source: jnp.ndarray
) -> jnp.ndarray:
    """torch.nn.MultiheadAttention (batch_first, packed QKV projection)."""
    weight = params[f"{name}.in_proj_weight"]
    bias = params[f"{name}.in_proj_bias"]
    embed_dim = query.shape[-1]
    q = query @ weight[:embed_dim].T + bias[:embed_dim]
    k = source @ weight[embed_dim : 2 * embed_dim].T + bias[embed_dim : 2 * embed_dim]
    v = source @ weight[2 * embed_dim :].T + bias[2 * embed_dim :]

    def split_heads(x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(*x.shape[:2], num_heads, -1).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
    out = jax.nn.softmax(scores, axis=-1) @ v
    out = out.transpose(0, 2, 1, 3).reshape(*query.shape[:2], embed_dim)
    return _linear(params, f"{name}.out_proj", out)


def _ffn_geglu(params, name: str, x: jnp.ndarray) -> jnp.ndarray:
    return _linear(params, f"{name}.c_proj", _geglu(_linear(params, f"{name}.c_fc", x)))


def _self_attn_block(
    params, name: str, num_heads: int, x: jnp.ndarray, dp: DropPath | None = None
) -> jnp.ndarray:
    """Pre-LN self-attention followed by a GEGLU FFN, both residual with DropPath."""
    h = _layer_norm(params, f"{name}.ln_1", x)
    x = x + _drop(dp, _attention(params, f"{name}.attn", num_heads, h, h))
    h = _ffn_geglu(params, f"{name}.mlp", _layer_norm(params, f"{name}.ln_2", x))
    return x + _drop(dp, h)


def _cross_attn_block(
    params,
    name: str,
    num_heads: int,
    x: jnp.ndarray,
    source: jnp.ndarray,
    dp: DropPath | None = None,
) -> jnp.ndarray:
    """Residual cross-attention, self-attention, and GEGLU FFN with DropPath."""
    h = _attention(
        params,
        f"{name}.cross_attn",
        num_heads,
        _layer_norm(params, f"{name}.ln_cross", x),
        _layer_norm(params, f"{name}.ln_source", source),
    )
    x = x + _drop(dp, h)
    h = _layer_norm(params, f"{name}.ln_1", x)
    x = x + _drop(dp, _attention(params, f"{name}.self_attn", num_heads, h, h))
    h = _ffn_geglu(params, f"{name}.mlp", _layer_norm(params, f"{name}.ln_2", x))
    return x + _drop(dp, h)


def _cross_attn(
    params, name: str, num_heads: int, x: jnp.ndarray, source: jnp.ndarray
) -> jnp.ndarray:
    """Normalized cross-attention without residual or output head (no DropPath)."""
    return _attention(
        params,
        f"{name}.cross_attn",
        num_heads,
        _layer_norm(params, f"{name}.ln_cross", x),
        _layer_norm(params, f"{name}.ln_source", source),
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
            params, f"{name}.points2patch", num_heads, patches, point_features
        )
        # The reference applies ln_ffn and the FFN's own LayerNorm back to back.
        h = _layer_norm(params, f"{name}.ln_ffn", patches)
        h = _layer_norm(params, f"{name}.patch_ffn.ln", h)
        h = _linear(params, f"{name}.patch_ffn.mlp.0", h)
        h = _linear(params, f"{name}.patch_ffn.mlp.2", _geglu(h))
        patches = patches + _drop(dp, h)
        for layer in range(config.encoder_num_layers_per_block):
            patches = _self_attn_block(
                params, f"{name}.processing_layers.{layer}", num_heads, patches, dp
            )
        z = _cross_attn_block(params, f"{name}.patch2latents", num_heads, z, patches, dp)
        point_features = point_features + _cross_attn(
            params, f"{name}.latents2points", num_heads, point_features, z
        )

    return _cross_attn_block(
        params, "autoencoder.encoder.last_block", num_heads, z, point_features, dp
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
            params, f"latent_decoder.transformer.{layer}", config.num_heads, z, dp
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
    weights_x = [(x0.astype(jnp.int32), 1.0 - (x - x0)), (x0.astype(jnp.int32) + 1, x - x0)]
    weights_y = [(y0.astype(jnp.int32), 1.0 - (y - y0)), (y0.astype(jnp.int32) + 1, y - y0)]
    result = jnp.zeros((coords.shape[0], channels), dtype=plane.dtype)
    for xi, wx in weights_x:
        for yi, wy in weights_y:
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            values = plane[:, jnp.clip(yi, 0, height - 1), jnp.clip(xi, 0, width - 1)]
            result = result + (wx * wy * valid)[:, None] * values.T
    return result


def _sample_planes(planes: jnp.ndarray, queries: jnp.ndarray, mode: str) -> jnp.ndarray:
    """Sample triplanes (B, 3, C, R, R) at queries (B, N, 3); sum or multiply planes."""
    queries = jnp.clip(queries, -1.0, 0.999)
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
    [-1, 1]^3 given triplane features from :func:`decode_planes`; returns (B, N).
    """
    features = _sample_planes(planes, queries, mode="sum")
    x = _linear(params, "autoencoder.head.0", features)
    x = _linear(params, "autoencoder.head.2", _gelu(x))
    return x[..., 0]


def decode_uncertainty(
    uncertainty_planes: jnp.ndarray, queries: jnp.ndarray
) -> jnp.ndarray:
    """Multiplicative uncertainty (B, N) at query points (B, N, 3)."""
    return _sample_planes(uncertainty_planes, queries, mode="mult")[..., 0]


class CODVAEJax(CODVAEBase):
    """JAX backend of COD-VAE (see :class:`cod_vae.base.CODVAEBase`)."""

    backend = "jax"

    def __init__(
        self, config: CODVAEConfig, params: Params, device=None
    ):
        if isinstance(device, str):
            device = jax.devices(device)[0]
        self.device = device if device is not None else jax.devices()[0]
        # Parameters are committed to the device below; jitted computations follow them.
        self._jit_encode = jax.jit(partial(encode, config=config))
        self._jit_decode_planes = jax.jit(partial(decode_planes, config=config))
        self._jit_decode_logits = jax.jit(partial(decode_logits, config=config))
        super().__init__(config, params)

    def _load_params(self, params: Params) -> None:
        self.params = {
            key: jax.device_put(jnp.asarray(value, dtype=jnp.float32), self.device)
            for key, value in params.items()
        }

    def get_params(self) -> Params:
        return {key: np.asarray(value) for key, value in self.params.items()}

    def _encode(self, points):
        return np.asarray(self._jit_encode(self.params, jnp.asarray(points)))

    def _decode_planes(self, latents):
        return self._jit_decode_planes(self.params, jnp.asarray(latents))

    def _decode_logits(self, planes, queries):
        return np.asarray(
            self._jit_decode_logits(self.params, planes, jnp.asarray(queries))
        )
