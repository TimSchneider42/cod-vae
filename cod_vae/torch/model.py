"""
PyTorch implementation of COD-VAE.

:class:`CODVAEModule` is a plain ``nn.Module`` whose state-dict keys match the flat
parameter names of :mod:`cod_vae.checkpoint` (i.e. the reference implementation's state
dict without the "model." prefix), usable directly for training. :class:`CODVAETorch`
wraps it behind the backend-independent numpy/trimesh interface.

Encoding: point cloud in [-1, 1]^3 -> transformer encoder -> num_latents x latent_dim
posterior moments. Decoding: latent -> transformer decoder with uncertainty-based token
pruning -> triplane features -> occupancy logits (positive inside) at query points.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

import trimesh

from ..base import CODVAEBase
from ..checkpoint import Params
from ..config import CODVAEConfig
from ..mesh import (
    CubeTransform,
    occupancy_grid_to_mesh,
    occupancy_grid_to_mesh_warp,
)
from .modules import (
    CrossAttention,
    CrossAttnBlock,
    GEGLU,
    GEGLUFFN,
    PointEmbed,
    SelfAttnBlock,
)

__all__ = ["CODVAETorch", "CODVAEModule", "farthest_point_sampling"]

logger = logging.getLogger(__name__)


def _lexsort_indices(points: torch.Tensor) -> torch.Tensor:
    """Indices sorting points (M, 3) lexicographically (x primary, then y, then z)."""
    indices = torch.arange(points.shape[0], device=points.device)
    for dim in (2, 1, 0):
        order = torch.sort(points[indices, dim], stable=True).indices
        indices = indices[order]
    return indices


def farthest_point_sampling(
    points: torch.Tensor, num_samples: int, canonicalize: bool = True
) -> torch.Tensor:
    """
    Farthest point sampling of batched point clouds (B, N, 3) -> (B, num_samples, 3),
    starting from the first point. With canonicalize=True, the (arbitrary) FPS ordering
    is replaced by lexicographic order so the result is a deterministic function of the
    point set.
    """
    batch_size, num_points, _ = points.shape
    batch = torch.arange(batch_size, device=points.device)
    indices = points.new_zeros((batch_size, num_samples), dtype=torch.long)
    distances = points.new_full((batch_size, num_points), torch.inf)
    last = points[:, 0]
    for j in range(1, num_samples):
        delta = points - last[:, None]
        distances = torch.minimum(distances, (delta * delta).sum(-1))
        indices[:, j] = distances.argmax(1)
        last = points[batch, indices[:, j]]
    selected = points[batch[:, None], indices]
    if canonicalize:
        selected = torch.stack([s[_lexsort_indices(s)] for s in selected])
    return selected


class _Encoder(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        self.config = config
        dim, heads, ratio = config.embed_dim, config.num_heads, config.encoder_mlp_ratio
        dp = config.droppath_rate
        self.norm_point = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            _EncoderBlock(config) for _ in range(config.encoder_num_blocks)
        )
        self.last_block = CrossAttnBlock(dim, heads, ratio, droppath=dp)

    def forward(
        self, pc: torch.Tensor, z: torch.Tensor, point_embed: PointEmbed
    ) -> torch.Tensor:
        point_features = self.norm_point(point_embed(pc))
        patch_pos = farthest_point_sampling(pc, self.config.encoder_num_patches)
        patches = self.norm_point(point_embed(patch_pos))
        for block in self.blocks:
            point_features, patches, z = block(point_features, patches, z)
        return self.last_block(z, point_features)


class _EncoderBlock(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        dim, heads, ratio = config.embed_dim, config.num_heads, config.encoder_mlp_ratio
        dp = config.droppath_rate
        self.points2patch = CrossAttention(dim, heads)
        self.ln_ffn = nn.LayerNorm(dim)
        self.patch_ffn = GEGLUFFN(dim, ratio, droppath=dp)
        self.processing_layers = nn.ModuleList(
            SelfAttnBlock(dim, heads, ratio, droppath=dp)
            for _ in range(config.encoder_num_layers_per_block)
        )
        self.patch2latents = CrossAttnBlock(dim, heads, ratio, droppath=dp)
        self.latents2points = CrossAttention(dim, heads)

    def forward(self, point_features, patches, z):
        patches = patches + self.points2patch(patches, point_features)
        patches = patches + self.patch_ffn(self.ln_ffn(patches))
        for layer in self.processing_layers:
            patches = layer(patches)
        z = self.patch2latents(z, patches)
        point_features = point_features + self.latents2points(point_features, z)
        return point_features, patches, z


class _InitTransformer(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        dim = config.embed_dim
        self.register = nn.Parameter(torch.randn(1, dim) * dim**-0.5)
        self.ln_pre = nn.LayerNorm(dim)
        self.ln_source = nn.LayerNorm(dim)
        self.transformer = nn.ModuleList(
            CrossAttnBlock(
                dim,
                config.num_heads,
                config.decoder_mlp_ratio,
                droppath=config.droppath_rate,
            )
            for _ in range(config.decoder_num_init_layers)
        )

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        register = self.register.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = self.ln_pre(torch.cat([x, register], dim=1))
        source = self.ln_source(source)
        for layer in self.transformer:
            x = layer(x, source)
        return x[:, :length]


class _RefineTransformer(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        dim = config.embed_dim
        self.class_token = nn.Parameter(torch.randn(1, dim) * dim**-0.5)
        self.class_pos = nn.Parameter(torch.randn(1, dim) * dim**-0.5)
        self.ln_pre = nn.LayerNorm(dim)
        self.transformer = nn.ModuleList(
            SelfAttnBlock(
                dim,
                config.num_heads,
                config.decoder_mlp_ratio,
                droppath=config.droppath_rate,
            )
            for _ in range(config.decoder_num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = (self.class_token + self.class_pos).unsqueeze(0)
        x = torch.cat([x, cls_token.expand(x.shape[0], -1, -1)], dim=1)
        x = self.ln_pre(x)
        for layer in self.transformer:
            x = layer(x)
        return x[:, :-1]


class _MergingModule(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        dim = config.embed_dim
        self.cross_attn = CrossAttention(dim, config.num_heads)
        self.tokens = nn.Parameter(
            torch.randn(config.decoder_num_merged_tokens, dim) * dim**-0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        return tokens + self.cross_attn(tokens, x)


class _Decoder(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.init_transformer = _InitTransformer(config)
        self.transformer = _RefineTransformer(config)
        self.init_out = nn.Linear(dim, config.patch_head_dim)
        self.decoder_out = nn.Linear(dim, config.patch_head_dim)
        self.uncertainty_out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), GEGLU(), nn.Linear(dim, 1)
        )
        self.merging_module = _MergingModule(config)
        self.mask_token = nn.Parameter(torch.randn(1, dim) * dim**-0.5)
        self.mask_pos = nn.Parameter(
            torch.randn(config.num_output_patches, dim) * dim**-0.5
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent embeddings into (planes, init_planes, uncertainty_planes)."""
        config = self.config
        batch_size = z.shape[0]
        tokens = self.mask_pos.unsqueeze(0).expand(batch_size, -1, -1)
        init_tokens = self.init_transformer(tokens, z)
        uncertainty = torch.sigmoid(self.uncertainty_out(init_tokens))

        ## uncertainty-based token pruning: only refine the most uncertain tokens
        order = torch.argsort(uncertainty, dim=1, descending=True)
        keep_indices = order[:, : config.num_kept_tokens]
        prune_indices = order[:, config.num_kept_tokens :]
        kept = torch.gather(
            init_tokens, 1, keep_indices.expand(-1, -1, init_tokens.shape[-1])
        )
        kept = kept + self.mask_token.unsqueeze(0)
        pruned = torch.gather(
            init_tokens, 1, prune_indices.expand(-1, -1, init_tokens.shape[-1])
        )
        merged = self.merging_module(pruned)

        ## refine the kept tokens (with merged pruned tokens and latents as context)
        x = torch.cat([kept, merged, z], dim=1)
        refined = self.transformer(x)[:, : config.num_kept_tokens]
        full_tokens = init_tokens.scatter(
            1, keep_indices.expand(-1, -1, init_tokens.shape[-1]), refined
        )

        ## project tokens to triplane patches; unrefined patches use the initial prediction
        init_patches = self.init_out(init_tokens)
        patches = init_patches + uncertainty * self.decoder_out(full_tokens)
        resolution = config.plane_resolution
        uncertainty_planes = uncertainty.view(batch_size, 3, 1, resolution, resolution)
        return (
            self.patches_to_planes(patches),
            self.patches_to_planes(init_patches),
            uncertainty_planes,
        )

    def patches_to_planes(self, patches: torch.Tensor) -> torch.Tensor:
        config = self.config
        resolution, patch_size = (
            config.plane_resolution,
            config.decoder_output_patch_size,
        )
        patches = patches.view(
            patches.shape[0],
            3,
            resolution,
            resolution,
            patch_size,
            patch_size,
            config.query_dim,
        )
        return patches.permute(0, 1, 6, 2, 4, 3, 5).reshape(
            patches.shape[0],
            3,
            config.query_dim,
            config.decoder_output_resolution,
            config.decoder_output_resolution,
        )


def _sample_planes(
    planes: torch.Tensor, queries: torch.Tensor, mode: str
) -> torch.Tensor:
    """
    Sample triplanes (B, 3, C, R, R) at queries (B, N, 3); sum or multiply planes.

    The interpolation always runs in float32, even for a half-precision model: the
    gradient with respect to the query coordinates is a difference of adjacent texels,
    which cancels catastrophically at half precision (measurably degrading the
    bounding box gradients of :meth:`CODVAEModule.decode_logits_full`). The features are
    returned in float32; callers feeding them to a half-precision head must cast them
    back.
    """
    planes = planes.float()
    queries = queries.float().clamp(-1, 0.999)
    result = None
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        grid = queries[..., other].unsqueeze(2)
        features = (
            F.grid_sample(
                planes[:, axis],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(-1)
            .transpose(1, 2)
        )
        if result is None:
            result = features
        else:
            result = result + features if mode == "sum" else result * features
    return result


class CODVAEModule(nn.Module):
    """Full COD-VAE model (autoencoder plus latent VAE modules)."""

    def __init__(self, config: CODVAEConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.autoencoder = _Autoencoder(config)
        self.latent_proj_in = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * config.latent_dim)
        )
        self.latent_proj_out = nn.Sequential(
            nn.Linear(config.latent_dim, dim), nn.LayerNorm(dim)
        )
        self.latent_decoder = _LatentDecoder(config)

    ## -- encoding --------------------------------------------------------------------

    def encode_embed(self, pc: torch.Tensor) -> torch.Tensor:
        """Encode point clouds (B, N, 3) into latent embeddings (B, L, embed_dim)."""
        return self.autoencoder.encode_embed(pc)

    def encode_moments(self, z_embed: torch.Tensor) -> torch.Tensor:
        """Posterior moments (B, L, 2 * latent_dim): mean and log-variance."""
        return self.latent_proj_in(z_embed.to(self.latent_proj_in[1].weight.dtype))

    def _attention_ctx(self):
        """
        Select the SDPA backend nn.MultiheadAttention dispatches to. "cudnn" pins cuDNN's
        fused kernel, which never materializes the (tokens x tokens) score matrix and so
        cuts the backward pass's peak memory; it raises rather than falling back if the
        shape is unsupported. "default" leaves torch's own backend selection alone.
        """
        if self.config.attention_implementation == "cudnn":
            return sdpa_kernel([SDPBackend.CUDNN_ATTENTION])
        return contextlib.nullcontext()

    def encode(self, pc: torch.Tensor) -> torch.Tensor:
        """Encode point clouds into posterior-mean latents (B, L, latent_dim)."""
        with self._attention_ctx():
            moments = self.encode_moments(self.encode_embed(pc))
        return moments[..., : self.config.latent_dim]

    ## -- decoding --------------------------------------------------------------------

    def decode_latents(self, latent: torch.Tensor) -> torch.Tensor:
        """Map latents (B, L, latent_dim) back to embeddings (B, L, embed_dim)."""
        with self._attention_ctx():
            return self.latent_decoder(self.latent_proj_out(latent))

    def decode_embed(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent embeddings into (planes, init_planes, uncertainty_planes)."""
        with self._attention_ctx():
            return self.autoencoder.decoder(z)

    def decode_planes(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents (B, L, latent_dim) into triplanes (B, 3, C, R, R)."""
        return self.decode_embed(self.decode_latents(latent))[0]

    def decode_logits(
        self, planes: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """
        Occupancy logits (B, N) at query points (B, N, 3) in [-1, 1]^3, in the module's
        dtype. The queries may be float32 for a half-precision module (see
        :func:`_sample_planes`); only the head runs in the module's dtype.
        """
        features = _sample_planes(planes, queries, mode="sum")
        head = self.autoencoder.head
        return head(features.to(head[0].weight.dtype)).squeeze(-1)

    def decode_uncertainty(
        self, uncertainty_planes: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """Multiplicative uncertainty (B, N) at query points (B, N, 3)."""
        return _sample_planes(uncertainty_planes, queries, mode="mult").squeeze(-1)

    ## -- full latents ----------------------------------------------------------------

    def split_full_latent(
        self, full_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split full latents (B, num_latents * latent_dim + 4) into latents
        (B, num_latents, latent_dim), bounding box centers (B, 3), and sizes (B,)
        (see :meth:`cod_vae.base.CODVAEBase.pack_full_latent` for the layout).
        """
        dims = self.config.num_latents * self.config.latent_dim
        latent = full_latent[:, :dims].reshape(
            -1, self.config.num_latents, self.config.latent_dim
        )
        return latent, full_latent[:, dims : dims + 3], full_latent[:, dims + 3]

    def decode_logits_full(
        self,
        planes: torch.Tensor,
        center: torch.Tensor,
        size: torch.Tensor,
        queries: torch.Tensor,
        object_scale: float = 0.9,
        stop_transform_gradient: bool = True,
    ) -> torch.Tensor:
        """
        Occupancy logits (B, N) at query points (B, N, 3) given in the [-1, 1]
        normalized world frame, mapping them into the model's cube via the bounding
        box center (B, 3) and size (B,) of a full latent (same frame). Sizes are
        clamped to 1e-3 to guard against (near-)zero size values, which would yield
        an infinite cube scale.

        By default (``stop_transform_gradient``), center and size are detached from
        the gradient: their only gradient path is the triplane interpolation of the
        mapped queries, which is piecewise-constant at the texel scale, noisy under
        query subsampling, and identically zero once the queries are clamped to the
        cube's boundary — callers optimizing the transform should penalize it
        directly instead. Pass ``stop_transform_gradient=False`` to differentiate
        through the query mapping anyway.

        The query mapping runs in float32 for the same reason the interpolation does
        (see :func:`_sample_planes`), so queries given in float32 keep their full
        precision even for a half-precision module.
        """
        if stop_transform_gradient:
            center = center.detach()
            size = size.detach()
        scale = object_scale / torch.clamp(size.float(), min=1e-3)
        cube_queries = (queries.float() - center.float()[:, None, :]) * scale[
            :, None, None
        ]
        return self.decode_logits(planes, cube_queries)

    def decode_full(
        self,
        full_latent: torch.Tensor,
        queries: torch.Tensor,
        object_scale: float = 0.9,
        stop_transform_gradient: bool = True,
    ) -> torch.Tensor:
        """
        Occupancy logits (B, N) of full latents (B, num_latents * latent_dim + 4) at
        query points (B, N, 3) given in the [-1, 1] normalized world frame.
        Differentiable with respect to the latent part; the bounding box center and
        size entries are detached from the gradient by default (see
        :meth:`decode_logits_full` for the rationale and
        ``stop_transform_gradient=False`` to differentiate through them).
        """
        latent, center, size = self.split_full_latent(full_latent)
        planes = self.decode_planes(latent)
        return self.decode_logits_full(
            planes,
            center,
            size,
            queries,
            object_scale=object_scale,
            stop_transform_gradient=stop_transform_gradient,
        )


class _Autoencoder(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.point_embed = PointEmbed(dim, config.point_embed_hidden_dim)
        self.norm_latent = nn.LayerNorm(dim)
        self.encoder = _Encoder(config)
        self.decoder = _Decoder(config)
        self.head = nn.Sequential(
            nn.Linear(config.query_dim, config.query_dim),
            nn.GELU(),
            nn.Linear(config.query_dim, config.output_dim),
        )

    def encode_embed(self, pc: torch.Tensor) -> torch.Tensor:
        z = farthest_point_sampling(pc, self.config.num_latents)
        z = self.norm_latent(self.point_embed(z))
        return self.encoder(pc, z, self.point_embed)


class _LatentDecoder(nn.Module):
    def __init__(self, config: CODVAEConfig):
        super().__init__()
        dim = config.embed_dim
        self.transformer = nn.ModuleList(
            SelfAttnBlock(
                dim,
                config.num_heads,
                config.latent_mlp_ratio,
                droppath=config.droppath_rate,
            )
            for _ in range(config.num_latent_layers)
        )
        self.linear_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer:
            z = layer(z)
        return self.linear_out(z)


def _resolve_attention(requested, device, dtype):
    """
    Resolve "auto" to "cudnn" where the fused kernel can run, else "default". Needs a
    CUDA device, a half-precision compute dtype (cuDNN rejects float32), and a cuDNN
    build that supports the shape -- torch reports the last through
    can_use_cudnn_attention, so unlike the jax side no trial compile is required.
    """
    if requested != "auto":
        return requested
    if device.type != "cuda":
        logger.info("Attention: using the default kernel (no CUDA device).")
        return "default"
    if dtype not in (torch.float16, torch.bfloat16):
        logger.info(
            "Attention: using the default kernel (cuDNN needs float16/bfloat16, this "
            "model computes in %s).",
            str(dtype).replace("torch.", ""),
        )
        return "default"
    logger.info("Attention: using cuDNN's fused kernel.")
    return "cudnn"


class CODVAETorch(CODVAEBase):
    """PyTorch backend of COD-VAE (see :class:`cod_vae.base.CODVAEBase`)."""

    backend = "torch"

    def __init__(
        self,
        config: CODVAEConfig,
        params: Params,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = (
            torch.float32
            if dtype is None
            else (dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype))
        )
        config = dataclasses.replace(
            config,
            attention_implementation=_resolve_attention(
                config.attention_implementation, self.device, self.dtype
            ),
        )
        self.module = CODVAEModule(config).to(self.device, self.dtype).eval()
        self._grid_queries_cache: dict[int, torch.Tensor] = {}
        super().__init__(config, params)

    def _load_params(self, params: Params) -> None:
        # The checkpoint is float32; copy_ casts it to the module's dtype.
        state_dict = {
            key: torch.from_numpy(np_array.copy()) for key, np_array in params.items()
        }
        self.module.load_state_dict(state_dict)

    def get_params(self) -> Params:
        return {
            key: value.detach().float().cpu().numpy()
            for key, value in self.module.state_dict().items()
        }

    def _to_device(self, array, dtype: "torch.dtype | None" = None) -> torch.Tensor:
        """Move an array to the model's device, in the model's dtype unless overridden
        (query points stay float32, see :func:`_sample_planes`)."""
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(
            self.device, self.dtype if dtype is None else dtype
        )

    def occupancy_grid_to_mesh(
        self, logits: torch.Tensor, transform: CubeTransform | None = None
    ) -> trimesh.Trimesh:
        """
        Turn a dense occupancy logit grid that is still on the device -- as returned by
        :meth:`_decode_grid_native` -- into a mesh, without a host round trip: Warp
        adopts the torch buffer via DLPack and marching cubes runs where the data
        already is. Falls back to the host implementation off CUDA.
        """
        if self.device.type != "cuda":
            return occupancy_grid_to_mesh(self._to_numpy(logits), transform)
        return occupancy_grid_to_mesh_warp(logits.float().contiguous(), transform)

    def _to_numpy(self, array) -> np.ndarray:
        return array.float().cpu().numpy()

    def _grid_queries(self, resolution: int) -> torch.Tensor:
        """Dense [-1, 1]^3 grid of shape (resolution ** 3, 3), built and cached on the
        model's device. Same layout as :func:`cod_vae.mesh.grid_queries`, which builds
        the identical grid on the host."""
        cached = self._grid_queries_cache.get(resolution)
        if cached is None:
            axis = torch.linspace(
                -1.0, 1.0, resolution, dtype=torch.float32, device=self.device
            )
            grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
            cached = grid.reshape(-1, 3)
            self._grid_queries_cache[resolution] = cached
        return cached

    @torch.no_grad()
    def _decode_grid_native(self, latents, resolution: int, chunk_size: int):
        # The grid is generated on the device and cached, so nothing is uploaded per
        # call: it is the same array every time and is by far the largest transfer in
        # this path (3 * resolution ** 3 floats, against resolution ** 3 coming back).
        planes = self._decode_planes(latents)
        queries = self._grid_queries(resolution)
        batch = latents.shape[0]
        chunks = [
            self.module.decode_logits(
                planes, queries[i : i + chunk_size].expand(batch, -1, -1)
            )
            for i in range(0, queries.shape[0], chunk_size)
        ]
        return torch.cat(chunks, dim=1)

    @torch.no_grad()
    def _encode_native(self, points):
        return self.module.encode(self._to_device(points))

    @torch.no_grad()
    def _decode_planes(self, latents):
        return self.module.decode_planes(self._to_device(latents))

    @torch.no_grad()
    def _decode_logits_native(self, planes, queries):
        return self.module.decode_logits(
            planes, self._to_device(queries, torch.float32)
        )

    @torch.no_grad()
    def _decode_planes_full(self, full_latents):
        latent, center, size = self.module.split_full_latent(
            self._to_device(full_latents)
        )
        return self.module.decode_planes(latent), center, size

    @torch.no_grad()
    def _decode_logits_full_native(self, handle, queries, object_scale):
        planes, center, size = handle
        return self.module.decode_logits_full(
            planes,
            center,
            size,
            self._to_device(queries, torch.float32),
            object_scale,
        )
