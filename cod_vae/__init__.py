"""
cod-vae: unofficial PyTorch/JAX reimplementation of COD-VAE.

COD-VAE ("Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models", Cho
et al., ICCV 2025) is a 3D shape VAE that compresses a mesh into a small set of latent
vectors. This package provides inference (encoding and decoding) and training in both
PyTorch and JAX behind a common numpy/trimesh interface:

    from cod_vae import CODVAE

    vae = CODVAE.from_pretrained("user/repo")   # picks the best available backend
    latent, transform = vae.encode_mesh(mesh, return_transform=True)
    reconstruction = vae.decode_mesh(latent, transform=transform)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

try:
    from ._version import version as __version__
except ImportError:  # not installed / no build metadata available
    try:
        from importlib.metadata import version as _package_version

        __version__ = _package_version("cod-vae")
    except Exception:
        __version__ = "0.0.0"

from .base import CODVAEBase
from .checkpoint import Params, load_npz, load_torch_release, save_npz
from .config import CODVAEConfig
from .init import init_params
from .mesh import (
    CubeTransform,
    normalize_to_cube,
    occupancy_grid_to_mesh,
    pack_cube_transform,
    points_to_cube_transform,
    unpack_cube_transform,
)

if TYPE_CHECKING:
    from .jax import CODVAEJax
    from .torch import CODVAETorch

__all__ = [
    "CODVAE",
    "CODVAEBase",
    "CODVAEConfig",
    "CubeTransform",
    "Params",
    "init_params",
    "load_npz",
    "load_torch_release",
    "normalize_to_cube",
    "occupancy_grid_to_mesh",
    "pack_cube_transform",
    "points_to_cube_transform",
    "save_npz",
    "unpack_cube_transform",
]

Backend = Literal["auto", "torch", "jax"]
#: Name of a compute dtype ("float32", "float16", "bfloat16"); None means float32.
DType = str | None


def _mk_torch(config, params, device, dtype=None) -> "CODVAETorch":
    try:
        from .torch import CODVAETorch
    except ImportError as e:
        raise ImportError(
            "Could not import torch. Install it to use the torch backend "
            "(pip install cod-vae[torch])."
        ) from e
    return CODVAETorch(config, params, device=device, dtype=dtype)


def _mk_jax(config, params, device, dtype=None) -> "CODVAEJax":
    try:
        from .jax import CODVAEJax
    except ImportError as e:
        raise ImportError(
            "Could not import jax. Install it to use the jax backend "
            "(pip install cod-vae[jax] or cod-vae[jax-cpu])."
        ) from e
    return CODVAEJax(config, params, device=device, dtype=dtype)


def _make(config, params, backend: Backend, device, dtype: DType = None) -> CODVAEBase:
    if backend == "torch":
        return _mk_torch(config, params, device, dtype)
    if backend == "jax":
        return _mk_jax(config, params, device, dtype)
    if backend == "auto":
        try:
            return _mk_jax(config, params, device, dtype)
        except ImportError:
            pass
        try:
            return _mk_torch(config, params, device, dtype)
        except ImportError:
            raise ImportError(
                "Could load neither the JAX nor the PyTorch backend. Install either "
                "to use cod-vae (pip install cod-vae[jax] or cod-vae[torch])."
            )
    raise ValueError(f"Unknown backend {backend!r}")


class CODVAE:
    """
    Factory for COD-VAE models. Constructing an instance returns a backend-specific
    implementation of :class:`CODVAEBase` (JAX if available, PyTorch otherwise; override
    with backend="torch"/"jax").
    """

    def __new__(
        cls,
        config: CODVAEConfig,
        params: Params,
        *,
        backend: Backend = "auto",
        device: str | None = None,
        dtype: DType = None,
    ) -> CODVAEBase:
        return _make(config, params, backend, device, dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        filename: str | None = None,
        revision: str | None = None,
        backend: Backend = "auto",
        device: str | None = None,
        dtype: DType = None,
    ) -> CODVAEBase:
        """
        Load a model from a Hugging Face Hub repository id, a local npz file, or a local
        directory containing an official COD-VAE release (config.yaml + *.pt).

        ``dtype`` selects the compute dtype of the model ("float16"/"bfloat16" halve
        its memory footprint and roughly double its throughput on GPUs); the numpy
        interface stays float32 regardless, as does the triplane interpolation (see
        :attr:`cod_vae.CODVAEBase.dtype`).
        """
        path = Path(model_name_or_path)
        if path.is_file():
            config, params = load_npz(path)
        elif path.is_dir():
            config, params = load_torch_release(path)
        else:
            from .hub import DEFAULT_WEIGHTS_FILENAME, download_pretrained

            config, params = download_pretrained(
                str(model_name_or_path),
                filename=filename or DEFAULT_WEIGHTS_FILENAME,
                revision=revision,
            )
        return _make(config, params, backend, device, dtype)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: Backend = "auto",
        device: str | None = None,
        dtype: DType = None,
    ) -> CODVAEBase:
        """Load a model from a local npz file written by :meth:`CODVAEBase.save`."""
        config, params = load_npz(path)
        return _make(config, params, backend, device, dtype)

    @classmethod
    def from_torch_release(
        cls,
        weights_dir: str | Path,
        *,
        backend: Backend = "auto",
        device: str | None = None,
        dtype: DType = None,
    ) -> CODVAEBase:
        """Load a model from an official COD-VAE release directory (requires torch)."""
        config, params = load_torch_release(weights_dir)
        return _make(config, params, backend, device, dtype)

    @classmethod
    def from_random(
        cls,
        config: CODVAEConfig | None = None,
        seed: int = 0,
        *,
        backend: Backend = "auto",
        device: str | None = None,
        dtype: DType = None,
    ) -> CODVAEBase:
        """Create a randomly initialized model (e.g. as a starting point for training)."""
        config = config if config is not None else CODVAEConfig()
        return _make(config, init_params(config, seed=seed), backend, device, dtype)
