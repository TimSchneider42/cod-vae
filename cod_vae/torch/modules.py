"""
PyTorch building blocks of COD-VAE.

The module/parameter names deliberately mirror the reference implementation's state
dict, so that the flat parameter dictionaries of :mod:`cod_vae.checkpoint` load directly
via ``load_state_dict`` and stay interchangeable with the JAX backend. "Dropout" in the
reference is stochastic depth (DropPath) on residual branches; it is only active in
train mode.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..init import point_embed_basis

__all__ = [
    "GEGLU",
    "PointEmbed",
    "CrossAttention",
    "SelfAttnBlock",
    "CrossAttnBlock",
    "GEGLUFFN",
    "drop_path",
]


def drop_path(x: torch.Tensor, rate: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly zero the whole residual branch per sample."""
    if rate == 0.0 or not training:
        return x
    keep = 1.0 - rate
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def _geglu_mlp(embed_dim: int, mlp_ratio: float) -> nn.Sequential:
    width = int(embed_dim * mlp_ratio)
    return nn.Sequential(
        OrderedDict(
            [
                ("c_fc", nn.Linear(embed_dim, 2 * width)),
                ("gelu", GEGLU()),
                ("c_proj", nn.Linear(width, embed_dim)),
            ]
        )
    )


class PointEmbed(nn.Module):
    """Sinusoidal point embedding: (B, N, 3) -> (B, N, embed_dim)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.register_buffer(
            "basis",
            torch.from_numpy(np.ascontiguousarray(point_embed_basis(hidden_dim))),
        )
        self.mlp = nn.Linear(hidden_dim + 3, embed_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        projections = points @ self.basis
        features = torch.cat([projections.sin(), projections.cos(), points], dim=-1)
        return self.mlp(features)


class CrossAttention(nn.Module):
    """Pre-normalized cross-attention without residual or output head."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln_cross = nn.LayerNorm(embed_dim)
        self.ln_source = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        x = self.ln_cross(x)
        source = self.ln_source(source)
        return self.cross_attn(x, source, source, need_weights=False)[0]


class SelfAttnBlock(nn.Module):
    """Pre-LN self-attention followed by a GEGLU FFN, both residual with DropPath."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, droppath: float = 0.0
    ):
        super().__init__()
        self.droppath = droppath
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = _geglu_mlp(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + drop_path(h, self.droppath, self.training)
        h = self.mlp(self.ln_2(x))
        return x + drop_path(h, self.droppath, self.training)


class CrossAttnBlock(nn.Module):
    """Residual cross-attention, self-attention, and GEGLU FFN with DropPath."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, droppath: float = 0.0
    ):
        super().__init__()
        self.droppath = droppath
        self.ln_cross = nn.LayerNorm(embed_dim)
        self.ln_source = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = _geglu_mlp(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        h = self.ln_cross(x)
        source = self.ln_source(source)
        h = self.cross_attn(h, source, source, need_weights=False)[0]
        x = x + drop_path(h, self.droppath, self.training)
        h = self.ln_1(x)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + drop_path(h, self.droppath, self.training)
        h = self.mlp(self.ln_2(x))
        return x + drop_path(h, self.droppath, self.training)


class GEGLUFFN(nn.Module):
    """Standalone GEGLU FFN with its own LayerNorm and DropPath (encoder patch FFN)."""

    def __init__(self, embed_dim: int, mlp_ratio: float, droppath: float = 0.0):
        super().__init__()
        self.droppath = droppath
        self.ln = nn.LayerNorm(embed_dim)
        width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * width), GEGLU(), nn.Linear(width, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(self.mlp(self.ln(x)), self.droppath, self.training)
