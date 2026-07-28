"""
Mesh helpers shared by both backends: normalizing meshes into the model's [-1, 1] cube,
sampling surface point clouds, and turning decoded occupancy grids back into meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

__all__ = [
    "CubeTransform",
    "normalize_to_cube",
    "sample_surface_points",
    "grid_queries",
    "occupancy_grid_to_mesh",
]


@dataclass(frozen=True)
class CubeTransform:
    """Similarity transform mapping an original mesh into the model's [-1, 1] cube."""

    center: np.ndarray
    scale: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center) * self.scale

    def apply_inverse(self, points: np.ndarray) -> np.ndarray:
        return points / self.scale + self.center

    def apply_inverse_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh = mesh.copy()
        mesh.apply_scale(1.0 / self.scale)
        mesh.apply_translation(self.center)
        return mesh


def normalize_to_cube(
    mesh: trimesh.Trimesh, object_scale: float = 0.9
) -> tuple[trimesh.Trimesh, CubeTransform]:
    """
    Scale/translate a mesh into the model's [-1, 1] cube such that its largest extent
    spans [-object_scale, object_scale]. Returns the normalized mesh and the transform
    (whose inverse maps decoded geometry back into the original frame).
    """
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    scale = object_scale / np.max(mesh.extents / 2)
    normalized = mesh.copy()
    normalized.apply_translation(-center)
    normalized.apply_scale(scale)
    return normalized, CubeTransform(center=center, scale=float(scale))


def sample_surface_points(
    mesh: trimesh.Trimesh, num_points: int, seed: int | None = None
) -> np.ndarray:
    """Uniformly sample points on the mesh surface, shape (num_points, 3), float32."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    return np.asarray(points, dtype=np.float32)


def grid_queries(resolution: int) -> np.ndarray:
    """Dense grid of query points covering [-1, 1]^3, shape (resolution^3, 3)."""
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def occupancy_grid_to_mesh(
    logits: np.ndarray, transform: CubeTransform | None = None
) -> trimesh.Trimesh:
    """
    Extract the zero level set of a dense occupancy logit grid (R, R, R) indexed as
    [x, y, z] over :func:`grid_queries` (positive logits inside the object) via marching
    cubes. If a transform is given, the mesh is mapped back into the original frame.
    Returns an empty mesh if the grid contains no sign change.
    """
    from skimage import measure

    if logits.min() >= 0 or logits.max() <= 0:
        return trimesh.Trimesh()
    spacing = 2.0 / (logits.shape[0] - 1)
    vertices, faces, _, _ = measure.marching_cubes(
        logits, level=0.0, spacing=(spacing,) * 3, gradient_direction="ascent"
    )
    mesh = trimesh.Trimesh(vertices - 1.0, faces)
    if transform is not None:
        mesh = transform.apply_inverse_mesh(mesh)
    return mesh
