from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

__all__ = ["AttentionImplementation", "CODVAEConfig"]


AttentionImplementation = Literal["default", "cudnn"]


@dataclass(frozen=True)
class CODVAEConfig:
    """
    Hyperparameters of a COD-VAE model.

    The defaults correspond to the released vae_m32 variant; vae_m64 differs only in
    num_latents. Options of the original implementation that no released checkpoint uses
    (learnable latent positions, convolutional triplane refinement) are not supported.
    """

    # Shared
    latent_dim: int = 32
    num_latents: int = 32
    embed_dim: int = 512
    query_dim: int = 32
    output_dim: int = 1
    num_heads: int = 8
    point_embed_hidden_dim: int = 48

    # Latent decoder (maps compressed latents back to embeddings)
    num_latent_layers: int = 12
    latent_mlp_ratio: float = 4.0

    # Point cloud encoder
    encoder_num_patches: int = 512
    encoder_num_blocks: int = 4
    encoder_num_layers_per_block: int = 3
    encoder_mlp_ratio: float = 4.0

    # Triplane decoder
    decoder_output_resolution: int = 128
    decoder_output_patch_size: int = 8
    decoder_num_layers: int = 12
    decoder_num_init_layers: int = 1
    decoder_mlp_ratio: float = 2.0
    decoder_keep_ratio: float = 0.25
    decoder_num_merged_tokens: int = 8

    # Stochastic depth rate on residual branches; only used during training.
    droppath_rate: float = 0.1

    # Attention kernel. "default" builds the (tokens x tokens) score matrix explicitly;
    # "cudnn" uses cuDNN's fused kernel, which never materializes it and so cuts the
    # backward pass's peak memory, but requires an even sequence length and a cuDNN able
    # to build a plan for the shape -- it raises rather than falling back.
    attention_implementation: AttentionImplementation = "default"

    @property
    def plane_resolution(self) -> int:
        return self.decoder_output_resolution // self.decoder_output_patch_size

    @property
    def num_output_patches(self) -> int:
        return 3 * self.plane_resolution**2

    @property
    def num_kept_tokens(self) -> int:
        return int(self.num_output_patches * self.decoder_keep_ratio)

    @property
    def patch_head_dim(self) -> int:
        return self.query_dim * self.decoder_output_patch_size**2

    @classmethod
    def from_cod_vae_config(cls, model_config: Mapping[str, Any]) -> CODVAEConfig:
        """Build a config from the "model" section of an official release's config.yaml."""
        encoder = model_config["encoder_params"]
        decoder = model_config["decoder_params"]
        if model_config.get("use_learnable_pos", False):
            raise NotImplementedError("use_learnable_pos is not supported")
        if decoder.get("use_conv_refine", False):
            raise NotImplementedError("use_conv_refine is not supported")
        if decoder.get("num_merged_tokens", -1) <= 0:
            raise NotImplementedError("decoding without token merging is not supported")
        return cls(
            latent_dim=model_config["latent_dim"],
            num_latents=model_config["num_latents"],
            embed_dim=model_config["embed_dim"],
            query_dim=model_config["query_dim"],
            output_dim=model_config["output_dim"],
            num_heads=model_config.get("num_heads", 8),
            num_latent_layers=model_config["num_latent_layers"],
            latent_mlp_ratio=model_config.get("mlp_ratio", 4.0),
            encoder_num_patches=encoder["num_patches"],
            encoder_num_blocks=encoder["num_blocks"],
            encoder_num_layers_per_block=encoder["num_layers_per_block"],
            encoder_mlp_ratio=encoder.get("mlp_ratio", 4.0),
            decoder_output_resolution=decoder["output_resolution"],
            decoder_output_patch_size=decoder["output_patch_size"],
            decoder_num_layers=decoder["num_layers"],
            decoder_num_init_layers=decoder["num_init_layers"],
            decoder_mlp_ratio=decoder.get("mlp_ratio", 4.0),
            decoder_keep_ratio=decoder.get("keep_ratio", 0.5),
            decoder_num_merged_tokens=decoder["num_merged_tokens"],
        )
