"""
Mesh helpers shared by both backends: normalizing meshes into the model's [-1, 1] cube,
sampling surface point clouds, and turning decoded occupancy grids back into meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import trimesh
import warp as wp
from skimage import measure

__all__ = [
    "CubeTransform",
    "points_to_cube_transform",
    "pack_cube_transform",
    "unpack_cube_transform",
    "normalize_to_cube",
    "sample_surface_points",
    "grid_queries",
    "occupancy_grid_to_mesh",
    "occupancy_grid_to_mesh_warp",
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
    if logits.min() >= 0 or logits.max() <= 0:
        return trimesh.Trimesh()
    if _warp_cuda_available():
        return occupancy_grid_to_mesh_warp(logits, transform)
    return _occupancy_grid_to_mesh_skimage(logits, transform)


def _occupancy_grid_to_mesh_skimage(
    logits: np.ndarray, transform: CubeTransform | None = None
) -> trimesh.Trimesh:
    """CPU marching cubes. Reference implementation and the fallback used wherever Warp
    or a CUDA device is unavailable."""
    spacing = 2.0 / (logits.shape[0] - 1)
    vertices, faces, _, _ = measure.marching_cubes(
        logits, level=0.0, spacing=(spacing,) * 3, gradient_direction="ascent"
    )
    mesh = trimesh.Trimesh(vertices - 1.0, faces)
    if transform is not None:
        mesh = transform.apply_inverse_mesh(mesh)
    return mesh


@lru_cache(maxsize=1)
def _warp_cuda_available() -> bool:
    """Whether a CUDA device Warp can use is present. Cached: it cannot change within a
    process, and ``wp.init()`` is not cheap."""
    wp.init()
    return wp.get_cuda_device_count() > 0


def _to_warp_field(logits: Any) -> Any:
    """Wrap a dense occupancy grid as a Warp CUDA array. Device arrays exposing
    ``__dlpack__`` (jax, torch) are adopted without a copy; host arrays are uploaded."""
    if isinstance(logits, wp.array):
        field = logits
    elif isinstance(logits, np.ndarray):
        field = wp.array(
            np.ascontiguousarray(logits, dtype=np.float32),
            dtype=wp.float32,
            device="cuda",
        )
    else:
        field = wp.from_dlpack(logits)
    if field.dtype != wp.float32:
        raise ValueError(
            f"Warp marching cubes requires a float32 grid, got {field.dtype}. Cast the "
            f"grid before calling (half-precision models decode in their compute dtype)."
        )
    if field.ndim != 3:
        raise ValueError(f"Expected a 3D (R, R, R) grid, got shape {field.shape}.")
    return field


def occupancy_grid_to_mesh_warp(
    logits: Any, transform: CubeTransform | None = None
) -> trimesh.Trimesh:
    """
    GPU counterpart of :func:`occupancy_grid_to_mesh`, using NVIDIA Warp's marching
    cubes. ``logits`` is a dense (R, R, R) occupancy logit grid, either a host array or
    any CUDA array implementing ``__dlpack__`` (a jax or torch device array), the latter
    being consumed in place without a host round trip. Requires a CUDA device.
    """
    field = _to_warp_field(logits)
    resolution = field.shape[0]
    # max_verts/max_tris/device are deprecated in warp 1.16 and removed in 1.19; the
    # output arrays size themselves.
    mc = wp.MarchingCubes(nx=resolution, ny=resolution, nz=resolution)
    mc.surface(field, 0.0)
    vertices = mc.verts.numpy()
    if len(vertices) == 0:
        return trimesh.Trimesh()
    # Warp winds triangles opposite to skimage's gradient_direction="ascent", which
    # would leave the surface inside-out (negated signed volume, backfaces toward the
    # camera). The vertices themselves agree exactly, so reversing each triangle is all
    # that is needed.
    faces = mc.indices.numpy().reshape(-1, 3)[:, ::-1]
    # Warp emits vertices in voxel index space; skimage applies the same scaling
    # internally via its `spacing` argument, so both land in [-1, 1].
    spacing = 2.0 / (resolution - 1)
    mesh = trimesh.Trimesh(vertices * spacing - 1.0, faces)
    if transform is not None:
        mesh = transform.apply_inverse_mesh(mesh)
    return mesh
