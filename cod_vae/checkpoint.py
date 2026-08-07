"""
Loading, saving, and converting COD-VAE weights.

Parameters are represented framework-agnostically as a flat mapping from the original
torch state-dict names (without the "model." prefix) to numpy arrays, so the mapping
between this reimplementation and the reference implementation stays transparent.

:func:`load_torch_release` reads an official release directory (config.yaml + *.pt) and
requires torch and pyyaml; :func:`save_npz`/:func:`load_npz` store the converted weights
in a self-contained npz file that can be loaded without torch.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict

import numpy as np

from .config import CODVAEConfig

__all__ = ["Params", "load_torch_release", "save_npz", "load_npz"]

Params = Dict[str, np.ndarray]

_CONFIG_KEY = "__config__"


def load_torch_release(weights_dir: Path | str) -> tuple[CODVAEConfig, Params]:
    """
    Load an official COD-VAE release directory containing config.yaml and a *.pt
    checkpoint (e.g. the released vae_m32 or vae_m64 folders). Requires torch and pyyaml.
    """
    # Deferred: this module is imported by `import cod_vae` and must stay usable
    # without the convert extra (and without paying the torch import on every use).
    try:
        import torch
        import yaml
    except ImportError as e:
        raise ImportError(
            "Loading an official release requires torch and pyyaml; install them "
            "via pip install cod-vae[convert]"
        ) from e

    weights_dir = Path(weights_dir)
    with (weights_dir / "config.yaml").open() as f:
        config = CODVAEConfig.from_cod_vae_config(yaml.safe_load(f)["model"])
    weights_path = next(weights_dir.glob("*.pt"))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    params = {
        key.removeprefix("model."): value.numpy().astype(np.float32)
        for key, value in state_dict.items()
    }
    return config, params


def save_npz(path: Path | str, config: CODVAEConfig, params: Params) -> None:
    """Save config and parameters into a single self-contained npz file."""
    np.savez_compressed(
        path, **{_CONFIG_KEY: json.dumps(dataclasses.asdict(config))}, **params
    )


def load_npz(path: Path | str) -> tuple[CODVAEConfig, Params]:
    """Load config and parameters from a file written by :func:`save_npz`."""
    data = np.load(path)
    config = CODVAEConfig(**json.loads(str(data[_CONFIG_KEY])))
    params = {key: data[key] for key in data.files if key != _CONFIG_KEY}
    return config, params
