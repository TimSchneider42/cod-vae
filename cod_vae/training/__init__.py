"""
Backend-independent training utilities: the training configuration and the occupancy
data pipeline. The actual training loops live in :mod:`cod_vae.torch.training` and
:mod:`cod_vae.jax.training`.
"""

from .config import TrainingConfig
from .data import (
    MeshOccupancyDataset,
    axis_scaling,
    compute_occupancy_data,
    iterate_batches,
)
from .preprocess import (
    SdfGenSettings,
    build_vecset_dataset,
    preprocess_mesh,
)
from .vecset import ShapeNetVecSetDataset

__all__ = [
    "MeshOccupancyDataset",
    "SdfGenSettings",
    "ShapeNetVecSetDataset",
    "TrainingConfig",
    "axis_scaling",
    "build_vecset_dataset",
    "compute_occupancy_data",
    "iterate_batches",
    "preprocess_mesh",
]
