"""Ray casting engine for directional shielding analysis.

Casts rays from target points through habitat geometry, computing
path lengths through each material region and converting to areal density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing import Optional

from .primitives import MeshData
from .scene import GeometryLayer, Scene


@dataclass
class RayHit:
    """A ray intersection with a geometry surface."""

    t: float  # parametric distance along ray
    layer_name: str
    material_id: str
    entering: bool  # True if entering material, False if exiting
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class RaySegment:
    """A segment of a ray through a single material."""

    material_id: str
    layer_name: str
    path_length: float  # meters
    entry_point: np.ndarray
    exit_point: np.ndarray

    @property
    def path_length_cm(self) -> float:
        return self.path_length * 100.0


@dataclass
class RayResult:
    """Complete result for a single ray from a target point."""

    direction: np.ndarray  # unit direction vector
    theta: float  # polar angle from zenith (radians)
    phi: float  # azimuthal angle (radians)
    segments: list[RaySegment]

    @property
    def total_path_length(self) -> float:
        """Total path length through all materials (meters)."""
        return sum(s.path_length for s in self.segments)

    def areal_density(self, material_densities: dict[str, float]) -> float:
        """Total areal density along this ray (g/cm²)."""
        total = 0.0
        for seg in self.segments:
            density = material_densities.get(seg.material_id, 1.0)
            total += density * seg.path_length_cm
        return total

    def per_material_areal_density(
        self, material_densities: dict[str, float]
    ) -> dict[str, float]:
        """Areal density breakdown by material (g/cm²)."""
        result: dict[str, float] = {}
        for seg in self.segments:
            density = material_densities.get(seg.material_id, 1.0)
            ad = density * seg.path_length_cm
            result[seg.material_id] = result.get(seg.material_id, 0.0) + ad
        return result


@dataclass
class TargetRayResults:
    """All ray results for a single dosimetry point."""

    target_name: str
    point_name: str
    position: np.ndarray
    rays: list[RayResult]

    @property
    def num_rays(self) -> int:
        return len(self.rays)

    def mean_areal_density(self, material_densities: dict[str, float]) -> float:
        """Solid-angle-weighted mean areal density (g/cm²)."""
        if not self.rays:
            return 0.0
        total = sum(r.areal_density(material_densities) for r in self.rays)
        return total / len(self.rays)

    def min_areal_density(self, material_densities: dict[str, float]) -> float:
        if not self.rays:
            return 0.0
        return min(r.areal_density(material_densities) for r in self.rays)

    def max_areal_density(self, material_densities: dict[str, float]) -> float:
        if not self.rays:
            return 0.0
        return max(r.areal_density(material_densities) for r in self.rays)

    def directional_map(
        self, material_densities: dict[str, float]
    ) -> np.ndarray:
        """Return (N, 3) array of [theta, phi, areal_density]."""
        data = []
        for r in self.rays:
            ad = r.areal_density(material_densities)
            data.append([r.theta, r.phi, ad])
        return np.array(data) if data else np.empty((0, 3))


def generate_ray_directions(n_directions: int = 162) -> np.ndarray:
    """Generate approximately uniform ray directions on the unit sphere.

    Uses a Fibonacci spiral (golden ratio) method for near-uniform distribution.

    Args:
        n_directions: Number of ray directions (default 162 ~ icosphere level 2)

    Returns:
        (N, 3) array of unit direction vectors
    """
    directions = []
    golden_ratio = (1 + math.sqrt(5)) / 2

    for i in range(n_directions):
        theta = math.acos(1 - 2 * (i + 0.5) / n_directions)
        phi = 2 * math.pi * i / golden_ratio

        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        directions.append([x, y, z])

    return np.array(directions)


def ray_triangle_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> Optional[float]:
    """Moller-Trumbore ray-triangle intersection.

    Returns parametric t value if intersection exists, None otherwise.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = np.dot(edge1, pvec)

    if abs(det) < 1e-10:
        return None

    inv_det = 1.0 / det
    tvec = origin - v0
    u = np.dot(tvec, pvec) * inv_det

    if u < 0.0 or u > 1.0:
        return None

    qvec = np.cross(tvec, edge1)
    v = np.dot(direction, qvec) * inv_det

    if v < 0.0 or u + v > 1.0:
        return None

    t = np.dot(edge2, qvec) * inv_det

    if t > 1e-6:
        return t

    return None


def ray_mesh_intersections(
    origin: np.ndarray,
    direction: np.ndarray,
    mesh: MeshData,
) -> list[float]:
    """Find all intersection distances of a ray with a triangle mesh.

    Uses vectorized numpy operations for performance.
    """
    faces = mesh.faces
    verts = mesh.vertices

    v0 = verts[faces[:, 0]]  # (M, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    edge1 = v1 - v0  # (M, 3)
    edge2 = v2 - v0

    pvec = np.cross(direction, edge2)  # (M, 3)
    det = np.sum(edge1 * pvec, axis=1)  # (M,)

    # Filter non-degenerate triangles
    valid = np.abs(det) > 1e-10
    if not np.any(valid):
        return []

    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]

    tvec = origin - v0  # (M, 3)
    u = np.sum(tvec * pvec, axis=1) * inv_det

    mask1 = valid & (u >= 0.0) & (u <= 1.0)
    if not np.any(mask1):
        return []

    qvec = np.cross(tvec, edge1)
    v = np.sum(direction * qvec, axis=1) * inv_det

    mask2 = mask1 & (v >= 0.0) & (u + v <= 1.0)
    if not np.any(mask2):
        return []

    t = np.sum(edge2 * qvec, axis=1) * inv_det
    mask3 = mask2 & (t > 1e-6)

    return sorted(t[mask3].tolist())


class RayCaster:
    """Ray casting engine for shielding analysis.

    Casts rays from dosimetry points through scene geometry, computing
    material path lengths and areal densities.
    """

    def __init__(self, n_directions: int = 162):
        self.n_directions = n_directions
        self.directions = generate_ray_directions(n_directions)
        self._direction_angles = self._compute_angles()

    def _compute_angles(self) -> np.ndarray:
        """Compute (theta, phi) for each direction."""
        angles = np.zeros((len(self.directions), 2))
        for i, d in enumerate(self.directions):
            theta = math.acos(np.clip(d[2], -1, 1))
            phi = math.atan2(d[1], d[0])
            angles[i] = [theta, phi]
        return angles

    def cast_from_point(
        self,
        origin: np.ndarray,
        layers: list[GeometryLayer],
        target_name: str = "",
        point_name: str = "",
    ) -> TargetRayResults:
        """Cast all rays from a single dosimetry point.

        For each ray direction, finds intersections with all geometry layers
        and computes path segments through each material.
        """
        ray_results = []

        for i, direction in enumerate(self.directions):
            theta, phi = self._direction_angles[i]
            segments = []

            for layer in layers:
                t_values = ray_mesh_intersections(origin, direction, layer.mesh)

                if not t_values:
                    continue

                # Handle origin-inside-mesh case:
                # If odd number of intersections, ray starts inside the mesh.
                # The first intersection is an exit, so pair from t=0 to first hit,
                # then pair remaining as entry/exit.
                if len(t_values) % 2 == 1:
                    # Origin is inside this layer: first segment from origin to first exit
                    t_exit = t_values[0]
                    if t_exit > 1e-6:
                        exit_pt = origin + direction * t_exit
                        segments.append(
                            RaySegment(
                                material_id=layer.material_id,
                                layer_name=layer.name,
                                path_length=t_exit,
                                entry_point=origin.copy(),
                                exit_point=exit_pt,
                            )
                        )
                    # Process remaining pairs normally
                    remaining = t_values[1:]
                else:
                    remaining = t_values

                # Pair intersections: entry/exit pairs
                for j in range(0, len(remaining) - 1, 2):
                    t_enter = remaining[j]
                    t_exit = remaining[j + 1] if j + 1 < len(remaining) else t_enter

                    if t_exit <= t_enter:
                        continue

                    entry_pt = origin + direction * t_enter
                    exit_pt = origin + direction * t_exit
                    path_len = t_exit - t_enter

                    segments.append(
                        RaySegment(
                            material_id=layer.material_id,
                            layer_name=layer.name,
                            path_length=path_len,
                            entry_point=entry_pt,
                            exit_point=exit_pt,
                        )
                    )

            # Sort segments by distance from origin
            segments.sort(key=lambda s: np.linalg.norm(s.entry_point - origin))

            ray_results.append(
                RayResult(
                    direction=direction.copy(),
                    theta=theta,
                    phi=phi,
                    segments=segments,
                )
            )

        return TargetRayResults(
            target_name=target_name,
            point_name=point_name,
            position=origin.copy(),
            rays=ray_results,
        )

    def cast_all_targets(self, scene: Scene) -> list[TargetRayResults]:
        """Cast rays from all dosimetry points in the scene."""
        results = []
        layers = scene.all_layers()

        for target in scene.targets:
            for point_name, position in target.world_dosimetry_points():
                result = self.cast_from_point(
                    origin=position,
                    layers=layers,
                    target_name=target.name,
                    point_name=point_name,
                )
                results.append(result)

        return results
