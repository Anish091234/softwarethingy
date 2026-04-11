"""Scene graph and data model for LunaRad.

The Scene contains all geometry, material assignments, and target points
needed for a radiation shielding analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .primitives import HabitatGeometry, MeshData, WallLayer


class TargetType(Enum):
    POINT = "point"
    HUMANOID = "humanoid"


@dataclass
class DosimetryPoint:
    """A named dosimetry measurement point on a target."""

    name: str
    local_offset: np.ndarray  # offset from target position (meters)


# Standard dosimetry points for a standing humanoid (1.75m tall)
HUMANOID_DOSIMETRY_POINTS = [
    DosimetryPoint("head", np.array([0, 0, 1.70])),
    DosimetryPoint("chest", np.array([0, 0, 1.30])),
    DosimetryPoint("abdomen", np.array([0, 0, 1.05])),
    DosimetryPoint("gonads", np.array([0, 0, 0.90])),
]


@dataclass
class HumanoidModel:
    """Simplified standing humanoid geometry for visualization.

    Composed of ellipsoid head, cylinder torso, and cylinder legs.
    Total height approximately 1.75m.
    """

    height: float = 1.75

    @property
    def head_center(self) -> np.ndarray:
        return np.array([0, 0, 1.65])

    @property
    def head_radii(self) -> np.ndarray:
        return np.array([0.10, 0.09, 0.12])  # semi-axes

    @property
    def torso_center(self) -> np.ndarray:
        return np.array([0, 0, 1.15])

    @property
    def torso_radius(self) -> float:
        return 0.18

    @property
    def torso_height(self) -> float:
        return 0.55

    @property
    def leg_centers(self) -> list[np.ndarray]:
        return [
            np.array([-0.08, 0, 0.43]),
            np.array([0.08, 0, 0.43]),
        ]

    @property
    def leg_radius(self) -> float:
        return 0.08

    @property
    def leg_height(self) -> float:
        return 0.85


@dataclass
class AnalysisTarget:
    """An analysis target within the habitat."""

    name: str
    position: np.ndarray  # world position (meters)
    target_type: TargetType = TargetType.POINT
    dosimetry_points: list[DosimetryPoint] = field(default_factory=list)
    humanoid: HumanoidModel | None = None

    def __post_init__(self):
        if self.target_type == TargetType.HUMANOID and not self.dosimetry_points:
            self.dosimetry_points = [
                DosimetryPoint(dp.name, dp.local_offset.copy())
                for dp in HUMANOID_DOSIMETRY_POINTS
            ]
            self.humanoid = HumanoidModel()

    def world_dosimetry_points(self) -> list[tuple[str, np.ndarray]]:
        """Get dosimetry point positions in world coordinates."""
        if not self.dosimetry_points:
            return [(self.name, self.position.copy())]
        return [
            (dp.name, self.position + dp.local_offset)
            for dp in self.dosimetry_points
        ]


@dataclass
class GeometryLayer:
    """A geometry layer in the scene with material assignment."""

    name: str
    mesh: MeshData
    material_id: str  # key into material library
    is_habitat_wall: bool = True
    is_terrain: bool = False
    is_overburden: bool = False
    visible: bool = True
    opacity: float = 1.0


class Scene:
    """Top-level scene containing all geometry and analysis configuration.

    Coordinate system: right-handed, Z-up, meters.
    """

    def __init__(self):
        self.habitat: HabitatGeometry | None = None
        self.layers: list[GeometryLayer] = []
        self.targets: list[AnalysisTarget] = []
        self.terrain: GeometryLayer | None = None
        self.overburden: GeometryLayer | None = None
        self.metadata: dict = {
            "units": "meters",
            "coordinate_system": "Z-up, right-handed",
            "description": "",
        }

    def set_habitat(self, habitat: HabitatGeometry, material_library: dict):
        """Set the habitat geometry and generate mesh layers."""
        self.habitat = habitat
        self.layers = []

        meshes = habitat.generate_mesh()
        for i, (layer_name, mesh) in enumerate(meshes.items()):
            mat_id = (
                habitat.wall_layers[i].material_id
                if i < len(habitat.wall_layers)
                else "regolith_peek_composite"
            )
            self.layers.append(
                GeometryLayer(
                    name=layer_name,
                    mesh=mesh,
                    material_id=mat_id,
                    is_habitat_wall=True,
                )
            )

    def add_target(self, target: AnalysisTarget):
        self.targets.append(target)

    def add_terrain(self, mesh: MeshData, material_id: str = "highland_regolith"):
        self.terrain = GeometryLayer(
            name="Terrain",
            mesh=mesh,
            material_id=material_id,
            is_habitat_wall=False,
            is_terrain=True,
        )

    def add_overburden(self, mesh: MeshData, material_id: str = "lavatube_rock"):
        self.overburden = GeometryLayer(
            name="Overburden",
            mesh=mesh,
            material_id=material_id,
            is_habitat_wall=False,
            is_overburden=True,
        )

    def all_layers(self) -> list[GeometryLayer]:
        """Get all geometry layers including terrain and overburden."""
        layers = list(self.layers)
        if self.terrain:
            layers.append(self.terrain)
        if self.overburden:
            layers.append(self.overburden)
        return layers

    def all_dosimetry_points(self) -> list[tuple[str, np.ndarray]]:
        """Get all dosimetry points from all targets."""
        points = []
        for target in self.targets:
            for name, pos in target.world_dosimetry_points():
                points.append((f"{target.name}/{name}", pos))
        return points

    def to_dict(self) -> dict:
        """Serialize scene to dictionary for project save."""
        return {
            "metadata": self.metadata,
            "habitat_type": type(self.habitat).__name__ if self.habitat else None,
            "num_layers": len(self.layers),
            "num_targets": len(self.targets),
            "targets": [
                {
                    "name": t.name,
                    "position": t.position.tolist(),
                    "type": t.target_type.value,
                }
                for t in self.targets
            ],
        }

    def clear(self):
        """Reset scene to empty state."""
        self.habitat = None
        self.layers = []
        self.targets = []
        self.terrain = None
        self.overburden = None
