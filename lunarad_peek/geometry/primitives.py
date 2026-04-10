"""Procedural geometry generation for lunar habitat primitives.

Generates triangulated mesh data for shell dome and cylindrical tunnel habitats,
with support for multi-layer walls, terrain, and overburden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MeshData:
    """Triangulated mesh representation.

    vertices: (N, 3) float array of vertex positions in meters
    faces: (M, 3) int array of triangle vertex indices
    normals: (M, 3) float array of face normals (outward)
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None

    def __post_init__(self):
        if self.normals is None:
            self.normals = self._compute_normals()

    def _compute_normals(self) -> np.ndarray:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return normals / norms

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def translate(self, offset: np.ndarray) -> MeshData:
        return MeshData(
            vertices=self.vertices + offset,
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
        )

    def scale(self, factor: float) -> MeshData:
        return MeshData(
            vertices=self.vertices * factor,
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
        )


@dataclass
class WallLayer:
    """A single layer of a multi-layer wall."""

    material_id: str  # key into material library
    thickness: float  # meters
    name: str = ""


@dataclass
class HabitatGeometry:
    """Base class for habitat geometry definitions."""

    name: str
    wall_layers: list[WallLayer]
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def total_wall_thickness(self) -> float:
        return sum(layer.thickness for layer in self.wall_layers)

    def generate_mesh(self) -> dict[str, MeshData]:
        """Generate mesh data for each wall layer. Returns {layer_name: MeshData}."""
        raise NotImplementedError


def _generate_sphere_mesh(
    center: np.ndarray, radius: float, n_lat: int = 32, n_lon: int = 64
) -> MeshData:
    """Generate a UV-sphere mesh."""
    vertices = []
    faces = []

    # Top pole
    vertices.append(center + np.array([0, 0, radius]))

    # Latitude rings
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            vertices.append(center + np.array([x, y, z]))

    # Bottom pole
    vertices.append(center + np.array([0, 0, -radius]))

    vertices = np.array(vertices)

    # Top cap triangles
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])

    # Body quads (as triangle pairs)
    for i in range(n_lat - 2):
        row_start = 1 + i * n_lon
        next_row_start = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            a = row_start + j
            b = row_start + j_next
            c = next_row_start + j_next
            d = next_row_start + j
            faces.append([a, d, b])
            faces.append([b, d, c])

    # Bottom cap triangles
    bottom_idx = len(vertices) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([bottom_idx, last_row + j_next, last_row + j])

    return MeshData(vertices=vertices, faces=np.array(faces))


def _generate_hemisphere_mesh(
    center: np.ndarray, radius: float, n_lat: int = 16, n_lon: int = 64
) -> MeshData:
    """Generate upper hemisphere mesh (z >= center_z)."""
    vertices = []
    faces = []

    # Top pole
    vertices.append(center + np.array([0, 0, radius]))

    # Latitude rings (only upper hemisphere: theta from 0 to pi/2)
    for i in range(1, n_lat + 1):
        theta = (math.pi / 2) * i / n_lat
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            vertices.append(center + np.array([x, y, z]))

    vertices = np.array(vertices)

    # Top cap
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])

    # Body
    for i in range(n_lat - 1):
        row_start = 1 + i * n_lon
        next_row_start = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            a = row_start + j
            b = row_start + j_next
            c = next_row_start + j_next
            d = next_row_start + j
            faces.append([a, d, b])
            faces.append([b, d, c])

    return MeshData(vertices=vertices, faces=np.array(faces))


def _generate_cylinder_mesh(
    center: np.ndarray,
    radius: float,
    length: float,
    n_circumference: int = 64,
    n_length: int = 32,
    caps: bool = True,
) -> MeshData:
    """Generate a cylinder mesh along the X axis."""
    vertices = []
    faces = []

    half_len = length / 2.0

    # Generate body vertices
    for i in range(n_length + 1):
        x = -half_len + length * i / n_length
        for j in range(n_circumference):
            angle = 2 * math.pi * j / n_circumference
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append(center + np.array([x, y, z]))

    # Body faces
    for i in range(n_length):
        for j in range(n_circumference):
            j_next = (j + 1) % n_circumference
            a = i * n_circumference + j
            b = i * n_circumference + j_next
            c = (i + 1) * n_circumference + j_next
            d = (i + 1) * n_circumference + j
            faces.append([a, d, b])
            faces.append([b, d, c])

    if caps:
        # Hemispherical end caps
        # Left cap (x = -half_len)
        cap_center_left = center + np.array([-half_len, 0, 0])
        left_cap_start = len(vertices)
        # Pole
        vertices.append(cap_center_left + np.array([-radius, 0, 0]))
        n_cap_lat = 8
        for i in range(1, n_cap_lat):
            theta = (math.pi / 2) * i / n_cap_lat
            for j in range(n_circumference):
                phi = 2 * math.pi * j / n_circumference
                x = -radius * math.cos(theta)
                y = radius * math.sin(theta) * math.cos(phi)
                z = radius * math.sin(theta) * math.sin(phi)
                vertices.append(cap_center_left + np.array([x, y, z]))

        # Left cap faces
        for j in range(n_circumference):
            j_next = (j + 1) % n_circumference
            faces.append([left_cap_start, left_cap_start + 1 + j_next, left_cap_start + 1 + j])

        for i in range(n_cap_lat - 2):
            row = left_cap_start + 1 + i * n_circumference
            next_row = left_cap_start + 1 + (i + 1) * n_circumference
            for j in range(n_circumference):
                j_next = (j + 1) % n_circumference
                a = row + j
                b = row + j_next
                c = next_row + j_next
                d = next_row + j
                faces.append([a, b, d])
                faces.append([b, c, d])

        # Right cap (x = +half_len) - mirror of left
        cap_center_right = center + np.array([half_len, 0, 0])
        right_cap_start = len(vertices)
        vertices.append(cap_center_right + np.array([radius, 0, 0]))
        for i in range(1, n_cap_lat):
            theta = (math.pi / 2) * i / n_cap_lat
            for j in range(n_circumference):
                phi = 2 * math.pi * j / n_circumference
                x = radius * math.cos(theta)
                y = radius * math.sin(theta) * math.cos(phi)
                z = radius * math.sin(theta) * math.sin(phi)
                vertices.append(cap_center_right + np.array([x, y, z]))

        for j in range(n_circumference):
            j_next = (j + 1) % n_circumference
            faces.append([right_cap_start, right_cap_start + 1 + j, right_cap_start + 1 + j_next])

        for i in range(n_cap_lat - 2):
            row = right_cap_start + 1 + i * n_circumference
            next_row = right_cap_start + 1 + (i + 1) * n_circumference
            for j in range(n_circumference):
                j_next = (j + 1) % n_circumference
                a = row + j
                b = row + j_next
                c = next_row + j_next
                d = next_row + j
                faces.append([a, d, b])
                faces.append([b, d, c])

    vertices = np.array(vertices)
    return MeshData(vertices=vertices, faces=np.array(faces))


def _generate_hemisphere_shell_mesh(
    center: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    n_lat: int = 16,
    n_lon: int = 64,
) -> MeshData:
    """Generate a closed hemisphere shell mesh between inner and outer radii.

    Creates a watertight shell volume by combining an outward-facing outer
    hemisphere, an inward-facing inner hemisphere, and an annular base ring
    connecting them at the equator.  The ray caster will see exactly two
    intersections (entry + exit) for any ray passing through the wall,
    giving the correct wall-thickness path length.
    """
    outer = _generate_hemisphere_mesh(center, outer_radius, n_lat, n_lon)
    inner = _generate_hemisphere_mesh(center, inner_radius, n_lat, n_lon)

    n_outer_verts = len(outer.vertices)

    # Reverse inner face winding so normals point inward
    inner_faces_rev = inner.faces[:, [0, 2, 1]] + n_outer_verts

    # Base annular ring connecting equator edges of outer and inner hemispheres.
    # Equator ring indices: 1 + (n_lat - 1) * n_lon  ..  1 + n_lat * n_lon - 1
    eq_outer = 1 + (n_lat - 1) * n_lon
    eq_inner = n_outer_verts + 1 + (n_lat - 1) * n_lon

    base_faces = []
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        oa, ob = eq_outer + j, eq_outer + j_next
        ia, ib = eq_inner + j, eq_inner + j_next
        base_faces.append([oa, ob, ia])
        base_faces.append([ob, ib, ia])

    all_verts = np.concatenate([outer.vertices, inner.vertices])
    all_faces = np.concatenate([
        outer.faces,
        inner_faces_rev,
        np.array(base_faces),
    ])

    return MeshData(vertices=all_verts, faces=all_faces)


def _generate_cylinder_shell_mesh(
    center: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    length: float,
    n_circumference: int = 64,
    n_length: int = 32,
) -> MeshData:
    """Generate a closed cylindrical shell mesh with hemispherical end caps.

    Combines an outward-facing outer cylinder+caps with an inward-facing
    inner cylinder+caps.  Both surfaces are individually closed, so together
    they form a watertight shell that the ray caster can traverse correctly.
    """
    outer = _generate_cylinder_mesh(
        center, outer_radius, length, n_circumference, n_length, caps=True,
    )
    inner = _generate_cylinder_mesh(
        center, inner_radius, length, n_circumference, n_length, caps=True,
    )

    n_outer_verts = len(outer.vertices)
    inner_faces_rev = inner.faces[:, [0, 2, 1]] + n_outer_verts

    all_verts = np.concatenate([outer.vertices, inner.vertices])
    all_faces = np.concatenate([outer.faces, inner_faces_rev])

    return MeshData(vertices=all_verts, faces=all_faces)


class ShellDomeHabitat(HabitatGeometry):
    """Hemispherical shell dome habitat.

    Parameters:
        inner_radius: Interior radius of the dome (meters)
        dome_height_ratio: Ratio of dome height to radius (1.0 = hemisphere)
        floor_elevation: Height of floor above ground plane (meters)
    """

    def __init__(
        self,
        name: str = "Shell Dome",
        inner_radius: float = 5.0,
        dome_height_ratio: float = 1.0,
        floor_elevation: float = 0.0,
        wall_layers: list[WallLayer] | None = None,
        position: np.ndarray | None = None,
    ):
        if wall_layers is None:
            wall_layers = [WallLayer("regolith_peek_composite", 0.30, "Primary Wall")]
        if position is None:
            position = np.zeros(3)
        super().__init__(name=name, wall_layers=wall_layers, position=position)
        self.inner_radius = inner_radius
        self.dome_height_ratio = dome_height_ratio
        self.floor_elevation = floor_elevation

    def generate_mesh(self) -> dict[str, MeshData]:
        meshes = {}
        current_radius = self.inner_radius

        for i, layer in enumerate(self.wall_layers):
            outer_radius = current_radius + layer.thickness
            layer_key = layer.name or f"layer_{i}"

            # Generate closed shell (inner + outer surfaces + base ring)
            mesh = _generate_hemisphere_shell_mesh(
                center=self.position + np.array([0, 0, self.floor_elevation]),
                inner_radius=current_radius,
                outer_radius=outer_radius,
            )
            meshes[layer_key] = mesh
            current_radius = outer_radius

        return meshes

    @property
    def interior_volume_approx(self) -> float:
        """Approximate interior volume (m³)."""
        return (2 / 3) * math.pi * self.inner_radius ** 3 * self.dome_height_ratio


class CylindricalTunnelHabitat(HabitatGeometry):
    """Cylindrical tunnel habitat with hemispherical end caps.

    Parameters:
        inner_radius: Interior radius (meters)
        length: Length of cylindrical section (meters), excluding end caps
        burial_depth: Depth below surface for buried scenarios (meters)
    """

    def __init__(
        self,
        name: str = "Cylindrical Tunnel",
        inner_radius: float = 3.0,
        length: float = 15.0,
        burial_depth: float = 0.0,
        wall_layers: list[WallLayer] | None = None,
        position: np.ndarray | None = None,
    ):
        if wall_layers is None:
            wall_layers = [WallLayer("regolith_peek_composite", 0.30, "Primary Wall")]
        if position is None:
            position = np.zeros(3)
        super().__init__(name=name, wall_layers=wall_layers, position=position)
        self.inner_radius = inner_radius
        self.length = length
        self.burial_depth = burial_depth

    def generate_mesh(self) -> dict[str, MeshData]:
        meshes = {}
        current_radius = self.inner_radius

        for i, layer in enumerate(self.wall_layers):
            outer_radius = current_radius + layer.thickness
            layer_key = layer.name or f"layer_{i}"

            # Generate closed shell (inner + outer surfaces)
            mesh = _generate_cylinder_shell_mesh(
                center=self.position,
                inner_radius=current_radius,
                outer_radius=outer_radius,
                length=self.length,
            )
            meshes[layer_key] = mesh
            current_radius = outer_radius

        return meshes

    @property
    def interior_volume_approx(self) -> float:
        """Approximate interior volume (m³)."""
        cylinder_vol = math.pi * self.inner_radius ** 2 * self.length
        caps_vol = (4 / 3) * math.pi * self.inner_radius ** 3  # two hemispheres = sphere
        return cylinder_vol + caps_vol


def generate_terrain_plane(
    center: np.ndarray, size: float = 50.0, resolution: int = 10
) -> MeshData:
    """Generate a flat terrain ground plane."""
    half = size / 2.0
    vertices = []
    faces = []

    for i in range(resolution + 1):
        for j in range(resolution + 1):
            x = center[0] - half + size * i / resolution
            y = center[1] - half + size * j / resolution
            z = center[2]
            vertices.append([x, y, z])

    for i in range(resolution):
        for j in range(resolution):
            a = i * (resolution + 1) + j
            b = a + 1
            c = a + (resolution + 1)
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return MeshData(
        vertices=np.array(vertices),
        faces=np.array(faces),
    )


def generate_regolith_cover(
    habitat: HabitatGeometry, cover_thickness: float = 2.0
) -> MeshData:
    """Generate a conformal regolith cover shell around a habitat."""
    total_radius = (
        habitat.inner_radius
        if hasattr(habitat, "inner_radius")
        else 5.0
    ) + habitat.total_wall_thickness + cover_thickness

    if isinstance(habitat, ShellDomeHabitat):
        return _generate_hemisphere_mesh(
            center=habitat.position + np.array([0, 0, habitat.floor_elevation]),
            radius=total_radius,
        )
    elif isinstance(habitat, CylindricalTunnelHabitat):
        return _generate_cylinder_mesh(
            center=habitat.position,
            radius=total_radius,
            length=habitat.length + 2 * cover_thickness,
            caps=True,
        )
    else:
        # Generic: generate a sphere cover
        return _generate_sphere_mesh(center=habitat.position, radius=total_radius)


def generate_overburden(
    habitat: HabitatGeometry, overburden_thickness: float = 5.0
) -> MeshData:
    """Generate lava-tube style overburden above a habitat."""
    # Model as an arched ceiling
    total_radius = (
        habitat.inner_radius
        if hasattr(habitat, "inner_radius")
        else 5.0
    ) + habitat.total_wall_thickness + overburden_thickness

    # For lava tube, use full cylinder (no end caps - the tube extends)
    if isinstance(habitat, CylindricalTunnelHabitat):
        return _generate_cylinder_mesh(
            center=habitat.position,
            radius=total_radius,
            length=habitat.length + 2 * overburden_thickness,
            caps=False,
        )
    else:
        return _generate_hemisphere_mesh(
            center=habitat.position,
            radius=total_radius,
        )
