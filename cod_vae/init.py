"""
Random initialization of COD-VAE parameters for training from scratch.

Produces the same flat parameter dictionary that :mod:`cod_vae.checkpoint` loads from
pretrained checkpoints, using numpy so both backends initialize identically. The
initialization schemes mirror the reference torch implementation: torch defaults for
Linear/LayerNorm/MultiheadAttention modules and scaled normal draws for embedding
parameters (registers, class/mask tokens, positional embeddings).
"""

from __future__ import annotations

import numpy as np

from .checkpoint import Params
from .config import CODVAEConfig

__all__ = ["init_params", "point_embed_basis", "parameter_names", "AUTOENCODER_PREFIX", "LATENT_PREFIXES"]

# Parameters belonging to the stage-1 autoencoder vs. the stage-2 latent VAE modules.
AUTOENCODER_PREFIX = "autoencoder."
LATENT_PREFIXES = ("latent_proj_in.", "latent_proj_out.", "latent_decoder.")


def point_embed_basis(hidden_dim: int) -> np.ndarray:
    """Deterministic sinusoidal basis of the point embedding, shape (3, hidden_dim / 2)."""
    num_freqs = hidden_dim // 6
    freqs = (2.0 ** np.arange(num_freqs)) * np.pi
    basis = np.zeros((3, 3 * num_freqs), dtype=np.float32)
    for axis in range(3):
        basis[axis, axis * num_freqs : (axis + 1) * num_freqs] = freqs
    return basis


class _Init:
    def __init__(self, config: CODVAEConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.params: Params = {}

    def _uniform(self, shape: tuple[int, ...], bound: float) -> np.ndarray:
        return self.rng.uniform(-bound, bound, size=shape).astype(np.float32)

    def linear(self, name: str, out_dim: int, in_dim: int) -> None:
        # torch.nn.Linear default: kaiming uniform with a=sqrt(5), i.e. U(+-1/sqrt(fan_in)).
        bound = 1.0 / np.sqrt(in_dim)
        self.params[f"{name}.weight"] = self._uniform((out_dim, in_dim), bound)
        self.params[f"{name}.bias"] = self._uniform((out_dim,), bound)

    def layer_norm(self, name: str, dim: int | None = None) -> None:
        dim = self.config.embed_dim if dim is None else dim
        self.params[f"{name}.weight"] = np.ones(dim, dtype=np.float32)
        self.params[f"{name}.bias"] = np.zeros(dim, dtype=np.float32)

    def attention(self, name: str) -> None:
        # torch.nn.MultiheadAttention default: xavier uniform packed QKV projection with
        # zero biases; the output projection keeps the Linear default weight init.
        dim = self.config.embed_dim
        bound = np.sqrt(6.0 / (3 * dim + dim))
        self.params[f"{name}.in_proj_weight"] = self._uniform((3 * dim, dim), bound)
        self.params[f"{name}.in_proj_bias"] = np.zeros(3 * dim, dtype=np.float32)
        self.params[f"{name}.out_proj.weight"] = self._uniform(
            (dim, dim), 1.0 / np.sqrt(dim)
        )
        self.params[f"{name}.out_proj.bias"] = np.zeros(dim, dtype=np.float32)

    def embedding(self, name: str, *shape: int) -> None:
        scale = self.config.embed_dim**-0.5
        self.params[name] = (
            self.rng.standard_normal(shape).astype(np.float32) * scale
        )

    def geglu_ffn(self, name: str, mlp_ratio: float) -> None:
        dim = self.config.embed_dim
        width = int(dim * mlp_ratio)
        self.linear(f"{name}.c_fc", 2 * width, dim)
        self.linear(f"{name}.c_proj", dim, width)

    def self_attn_block(self, name: str, mlp_ratio: float) -> None:
        self.layer_norm(f"{name}.ln_1")
        self.attention(f"{name}.attn")
        self.layer_norm(f"{name}.ln_2")
        self.geglu_ffn(f"{name}.mlp", mlp_ratio)

    def cross_attn_block(self, name: str, mlp_ratio: float) -> None:
        self.layer_norm(f"{name}.ln_cross")
        self.layer_norm(f"{name}.ln_source")
        self.attention(f"{name}.cross_attn")
        self.layer_norm(f"{name}.ln_1")
        self.attention(f"{name}.self_attn")
        self.layer_norm(f"{name}.ln_2")
        self.geglu_ffn(f"{name}.mlp", mlp_ratio)

    def cross_attn(self, name: str) -> None:
        self.layer_norm(f"{name}.ln_cross")
        self.layer_norm(f"{name}.ln_source")
        self.attention(f"{name}.cross_attn")


def init_params(config: CODVAEConfig, seed: int = 0) -> Params:
    """Randomly initialize all COD-VAE parameters (autoencoder and latent VAE modules)."""
    init = _Init(config, np.random.default_rng(seed))
    dim = config.embed_dim

    ## point embedding and latent normalization
    init.params["autoencoder.point_embed.basis"] = point_embed_basis(
        config.point_embed_hidden_dim
    )
    init.linear(
        "autoencoder.point_embed.mlp", dim, config.point_embed_hidden_dim + 3
    )
    init.layer_norm("autoencoder.norm_latent")

    ## encoder
    init.layer_norm("autoencoder.encoder.norm_point")
    for block in range(config.encoder_num_blocks):
        name = f"autoencoder.encoder.blocks.{block}"
        init.cross_attn(f"{name}.points2patch")
        init.layer_norm(f"{name}.ln_ffn")
        init.layer_norm(f"{name}.patch_ffn.ln")
        width = int(dim * config.encoder_mlp_ratio)
        init.linear(f"{name}.patch_ffn.mlp.0", 2 * width, dim)
        init.linear(f"{name}.patch_ffn.mlp.2", dim, width)
        for layer in range(config.encoder_num_layers_per_block):
            init.self_attn_block(
                f"{name}.processing_layers.{layer}", config.encoder_mlp_ratio
            )
        init.cross_attn_block(f"{name}.patch2latents", config.encoder_mlp_ratio)
        init.cross_attn(f"{name}.latents2points")
    init.cross_attn_block("autoencoder.encoder.last_block", config.encoder_mlp_ratio)

    ## triplane decoder
    init.embedding("autoencoder.decoder.init_transformer.register", 1, dim)
    init.layer_norm("autoencoder.decoder.init_transformer.ln_pre")
    init.layer_norm("autoencoder.decoder.init_transformer.ln_source")
    for layer in range(config.decoder_num_init_layers):
        init.cross_attn_block(
            f"autoencoder.decoder.init_transformer.transformer.{layer}",
            config.decoder_mlp_ratio,
        )
    init.embedding("autoencoder.decoder.transformer.class_token", 1, dim)
    init.embedding("autoencoder.decoder.transformer.class_pos", 1, dim)
    init.layer_norm("autoencoder.decoder.transformer.ln_pre")
    for layer in range(config.decoder_num_layers):
        init.self_attn_block(
            f"autoencoder.decoder.transformer.transformer.{layer}",
            config.decoder_mlp_ratio,
        )
    init.linear("autoencoder.decoder.init_out", config.patch_head_dim, dim)
    init.linear("autoencoder.decoder.decoder_out", config.patch_head_dim, dim)
    init.layer_norm("autoencoder.decoder.uncertainty_out.0")
    init.linear("autoencoder.decoder.uncertainty_out.1", 2 * dim, dim)
    init.linear("autoencoder.decoder.uncertainty_out.3", 1, dim)
    init.cross_attn("autoencoder.decoder.merging_module.cross_attn")
    init.embedding(
        "autoencoder.decoder.merging_module.tokens",
        config.decoder_num_merged_tokens,
        dim,
    )
    init.embedding("autoencoder.decoder.mask_token", 1, dim)
    init.embedding("autoencoder.decoder.mask_pos", config.num_output_patches, dim)

    ## occupancy head
    init.linear("autoencoder.head.0", config.query_dim, config.query_dim)
    init.linear("autoencoder.head.2", config.output_dim, config.query_dim)

    ## latent VAE modules (stage 2)
    init.layer_norm("latent_proj_in.0")
    init.linear("latent_proj_in.1", 2 * config.latent_dim, dim)
    init.linear("latent_proj_out.0", dim, config.latent_dim)
    init.layer_norm("latent_proj_out.1")
    for layer in range(config.num_latent_layers):
        init.self_attn_block(
            f"latent_decoder.transformer.{layer}", config.latent_mlp_ratio
        )
    init.layer_norm("latent_decoder.linear_out.0")
    init.linear("latent_decoder.linear_out.1", dim, dim)

    return init.params


def parameter_names(config: CODVAEConfig) -> list[str]:
    """Names of all parameters of a model with the given config."""
    return list(init_params(config, seed=0).keys())
