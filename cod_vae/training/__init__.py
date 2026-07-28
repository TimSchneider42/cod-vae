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
from .vecset import ShapeNetVecSetDataset

__all__ = [
    "MeshOccupancyDataset",
    "ShapeNetVecSetDataset",
    "TrainingConfig",
    "axis_scaling",
    "compute_occupancy_data",
    "iterate_batches",
]
