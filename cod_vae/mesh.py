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
    "points_to_cube_transform",
    "pack_cube_transform",
    "unpack_cube_transform",
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


def points_to_cube_transform(
    points: np.ndarray, object_scale: float = 0.9
) -> CubeTransform:
    """
    The transform mapping points into the model's [-1, 1] cube such that the largest
    extent of their axis-aligned bounding box spans [-object_scale, object_scale]:
    center is the bounding box center, scale is object_scale over the maximum
    half-extent. This is the transform :func:`normalize_to_cube` computes; passing a
    subset of points with the same bounding box (e.g. convex hull vertices) yields the
    identical transform.
    """
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    scale = object_scale / np.max((upper - lower) / 2)
    return CubeTransform(center=center, scale=float(scale))


def pack_cube_transform(
    transform: CubeTransform,
    frame_half_size: float = 1.0,
    object_scale: float = 0.9,
) -> np.ndarray:
    """
    The normalized bounding box representation of a cube transform used in full
    latents (see :meth:`cod_vae.CODVAEBase.pack_full_latent`): a float64 vector
    [center (3), size (1)], center being the bounding box center and size the maximum
    half-extent of the geometry, both divided by ``frame_half_size``.
    """
    return np.concatenate(
        [
            np.asarray(transform.center, dtype=np.float64) / frame_half_size,
            [(object_scale / transform.scale) / frame_half_size],
        ]
    )


def unpack_cube_transform(
    row: np.ndarray,
    frame_half_size: float = 1.0,
    object_scale: float = 0.9,
) -> CubeTransform:
    """
    Inverse of :func:`pack_cube_transform`. The size is clamped to 1e-3 (in normalized
    frame units) to guard against (near-)zero size values, which would yield an
    infinite cube scale.
    """
    return CubeTransform(
        center=np.asarray(row[:3], dtype=np.float64) * frame_half_size,
        scale=object_scale / (max(float(row[3]), 1e-3) * frame_half_size),
    )


def normalize_to_cube(
    mesh: trimesh.Trimesh, object_scale: float = 0.9
) -> tuple[trimesh.Trimesh, CubeTransform]:
    """
    Scale/translate a mesh into the model's [-1, 1] cube such that its largest extent
    spans [-object_scale, object_scale]. Returns the normalized mesh and the transform
    (whose inverse maps decoded geometry back into the original frame).
    """
    # mesh.bounds (as opposed to mesh.vertices) ignores unreferenced vertices; its two
    # corners are a valid point set with the same bounding box.
    transform = points_to_cube_transform(mesh.bounds, object_scale)
    normalized = mesh.copy()
    normalized.apply_translation(-transform.center)
    normalized.apply_scale(transform.scale)
    return normalized, transform


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
