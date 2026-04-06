"""Application state management for LunaRad.

Centralized state that tracks all user selections, geometry, materials,
environment config, and analysis results. Emits Qt signals on changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from lunarad_peek.geometry.primitives import (
    HabitatGeometry,
    ShellDomeHabitat,
    CylindricalTunnelHabitat,
    WallLayer,
    generate_terrain_plane,
    generate_regolith_cover,
    generate_overburden,
)
from lunarad_peek.geometry.scene import (
    AnalysisTarget,
    GeometryLayer,
    Scene,
    TargetType,
)
from lunarad_peek.materials.material import Material, create_preset_materials
from lunarad_peek.radiation.environments import (
    GCREnvironment,
    RadiationEnvironmentConfig,
    SPEEnvironment,
    SPE_EVENT_LIBRARY,
    SolarWindEnvironment,
    SolarCyclePhase,
)
from lunarad_peek.analysis.engine import AnalysisEngine, ScenarioResult


class AppState(QObject):
    """Central application state with change signals."""

    scene_changed = Signal()
    materials_changed = Signal()
    environment_changed = Signal()
    analysis_started = Signal()
    analysis_progress = Signal(float, str)  # fraction, message
    analysis_completed = Signal(object)  # ScenarioResult
    scenario_added = Signal(str)  # scenario name

    def __init__(self):
        super().__init__()

        self.scene = Scene()
        self.material_library: dict[str, Material] = create_preset_materials()
        self.environment = RadiationEnvironmentConfig()
        self.scenarios: list[ScenarioResult] = []
        self._project_path: Path | None = None

    # --- Geometry ---

    def create_dome_habitat(
        self,
        inner_radius: float = 5.0,
        dome_height_ratio: float = 1.0,
        wall_layers: list[tuple[str, float]] | None = None,
    ):
        if wall_layers is None:
            wall_layers = [("regolith_peek_composite", 0.30)]

        layers = [
            WallLayer(mat_id, thickness, f"Layer {i+1}")
            for i, (mat_id, thickness) in enumerate(wall_layers)
        ]
        habitat = ShellDomeHabitat(
            inner_radius=inner_radius,
            dome_height_ratio=dome_height_ratio,
            wall_layers=layers,
        )
        self.scene.set_habitat(habitat, self.material_library)
        self.scene_changed.emit()

    def create_tunnel_habitat(
        self,
        inner_radius: float = 3.0,
        length: float = 15.0,
        burial_depth: float = 0.0,
        wall_layers: list[tuple[str, float]] | None = None,
    ):
        if wall_layers is None:
            wall_layers = [("regolith_peek_composite", 0.30)]

        layers = [
            WallLayer(mat_id, thickness, f"Layer {i+1}")
            for i, (mat_id, thickness) in enumerate(wall_layers)
        ]
        habitat = CylindricalTunnelHabitat(
            inner_radius=inner_radius,
            length=length,
            burial_depth=burial_depth,
            wall_layers=layers,
        )
        self.scene.set_habitat(habitat, self.material_library)
        self.scene_changed.emit()

    def add_terrain(self):
        if self.scene.habitat:
            mesh = generate_terrain_plane(
                center=self.scene.habitat.position,
                size=50.0,
            )
            self.scene.add_terrain(mesh)
            self.scene_changed.emit()

    def add_regolith_cover(self, thickness: float = 2.0):
        if self.scene.habitat:
            mesh = generate_regolith_cover(self.scene.habitat, thickness)
            self.scene.add_overburden(mesh, "highland_regolith")
            self.scene_changed.emit()

    def add_overburden(self, thickness: float = 5.0):
        if self.scene.habitat:
            mesh = generate_overburden(self.scene.habitat, thickness)
            self.scene.add_overburden(mesh, "lavatube_rock")
            self.scene_changed.emit()

    def add_astronaut(self, name: str, x: float, y: float, z: float):
        target = AnalysisTarget(
            name=name,
            position=np.array([x, y, z]),
            target_type=TargetType.HUMANOID,
        )
        self.scene.add_target(target)
        self.scene_changed.emit()

    # --- Materials ---

    def update_material(self, material_id: str, material: Material):
        self.material_library[material_id] = material
        self.materials_changed.emit()

    # --- Environment ---

    def set_gcr_phase(self, phase: SolarCyclePhase):
        self.environment.gcr = GCREnvironment.from_phase(phase)
        self.environment_changed.emit()

    def set_gcr_phi(self, phi_MV: float):
        self.environment.gcr = GCREnvironment(phi_MV=phi_MV)
        self.environment_changed.emit()

    def set_spe_event(self, event_key: str | None):
        if event_key and event_key in SPE_EVENT_LIBRARY:
            self.environment.spe = SPEEnvironment(event=SPE_EVENT_LIBRARY[event_key])
        else:
            self.environment.spe = None
        self.environment_changed.emit()

    def toggle_solar_wind(self, enabled: bool):
        if enabled:
            self.environment.solar_wind = SolarWindEnvironment()
        else:
            self.environment.solar_wind = None
        self.environment_changed.emit()

    # --- Analysis ---

    def run_analysis(self, scenario_name: str = "Default", n_directions: int = 162):
        self.analysis_started.emit()

        engine = AnalysisEngine(n_directions=n_directions)
        engine.set_progress_callback(
            lambda frac, msg: self.analysis_progress.emit(frac, msg)
        )

        result = engine.run_analysis(
            scene=self.scene,
            material_library=self.material_library,
            environment=self.environment,
            scenario_name=scenario_name,
        )

        self.scenarios.append(result)
        self.scenario_added.emit(scenario_name)
        self.analysis_completed.emit(result)
        return result

    # --- Project Save/Load ---

    def save_project(self, filepath: Path):
        data = {
            "scene": self.scene.to_dict(),
            "environment": self.environment.to_dict(),
            "scenarios": [s.summary() for s in self.scenarios],
        }
        filepath.write_text(json.dumps(data, indent=2, default=str))
        self._project_path = filepath

    def clear(self):
        self.scene.clear()
        self.scenarios = []
        self.environment = RadiationEnvironmentConfig()
        self.scene_changed.emit()
