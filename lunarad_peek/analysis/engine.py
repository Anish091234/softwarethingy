"""Shielding analysis engine.

Combines geometry ray casting with radiation environment models to produce
dose, dose equivalent, and flux attenuation estimates at target points.

This is a conceptual analysis tool using areal-density-based approximations.
It is NOT a Monte Carlo transport solver.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from lunarad_peek.geometry.raycaster import (
    RayCaster,
    RayResult,
    TargetRayResults,
)
from lunarad_peek.geometry.scene import Scene
from lunarad_peek.materials.material import Material
from lunarad_peek.radiation.environments import (
    GCREnvironment,
    RadiationEnvironmentConfig,
    SPEEnvironment,
)


class OutputMetric(Enum):
    DOSE = "dose"  # Gy or mGy
    DOSE_EQUIVALENT = "dose_equivalent"  # Sv or mSv
    FLUX_ATTENUATION = "flux_attenuation"  # dimensionless 0-1
    AREAL_DENSITY = "areal_density"  # g/cm²
    SHIELDING_EFFECTIVENESS = "shielding_effectiveness"  # dimensionless


class ConfidenceLevel(Enum):
    """Label for output confidence/provenance."""

    CONCEPTUAL_ESTIMATE = "Conceptual Estimate"
    LITERATURE_DERIVED = "Literature-Derived Trend"
    VALIDATED_AGAINST_TRANSPORT = "Validated Against Transport Code"


@dataclass
class DirectionalResult:
    """Result for a single direction from a target point."""

    theta: float  # polar angle (rad)
    phi: float  # azimuthal angle (rad)
    areal_density: float  # g/cm² total
    per_material_areal_density: dict[str, float]  # material_id -> g/cm²
    gcr_dose_rate: float  # mSv/year
    gcr_dose_equivalent_rate: float  # mSv/year
    spe_dose: float  # mSv (for selected event)
    spe_dose_equivalent: float  # mSv
    flux_attenuation: float  # 0-1


@dataclass
class PointResult:
    """Complete analysis result for a single dosimetry point."""

    target_name: str
    point_name: str
    position: np.ndarray
    directional_results: list[DirectionalResult]
    confidence: ConfidenceLevel = ConfidenceLevel.CONCEPTUAL_ESTIMATE

    @property
    def num_directions(self) -> int:
        return len(self.directional_results)

    # --- Aggregate metrics ---

    @property
    def mean_areal_density(self) -> float:
        """Solid-angle-averaged areal density (g/cm²)."""
        if not self.directional_results:
            return 0.0
        return np.mean([r.areal_density for r in self.directional_results])

    @property
    def min_areal_density(self) -> float:
        if not self.directional_results:
            return 0.0
        return min(r.areal_density for r in self.directional_results)

    @property
    def max_areal_density(self) -> float:
        if not self.directional_results:
            return 0.0
        return max(r.areal_density for r in self.directional_results)

    @property
    def mean_gcr_dose_rate(self) -> float:
        """Direction-averaged GCR dose rate (mSv/year)."""
        if not self.directional_results:
            return 0.0
        return np.mean([r.gcr_dose_rate for r in self.directional_results])

    @property
    def mean_gcr_dose_equivalent_rate(self) -> float:
        """Direction-averaged GCR dose equivalent rate (mSv/year)."""
        if not self.directional_results:
            return 0.0
        return np.mean([r.gcr_dose_equivalent_rate for r in self.directional_results])

    @property
    def mean_spe_dose(self) -> float:
        """Direction-averaged SPE dose (mSv)."""
        if not self.directional_results:
            return 0.0
        return np.mean([r.spe_dose for r in self.directional_results])

    @property
    def mean_spe_dose_equivalent(self) -> float:
        """Direction-averaged SPE dose equivalent (mSv)."""
        if not self.directional_results:
            return 0.0
        return np.mean([r.spe_dose_equivalent for r in self.directional_results])

    @property
    def mean_flux_attenuation(self) -> float:
        if not self.directional_results:
            return 0.0
        return np.mean([r.flux_attenuation for r in self.directional_results])

    def directional_map(self, metric: str = "areal_density") -> np.ndarray:
        """Get (theta, phi, value) array for directional visualization."""
        data = []
        for r in self.directional_results:
            val = getattr(r, metric, r.areal_density)
            data.append([r.theta, r.phi, val])
        return np.array(data) if data else np.empty((0, 3))


@dataclass
class ScenarioResult:
    """Complete analysis result for a scenario (scene + environment)."""

    scenario_name: str
    point_results: list[PointResult]
    environment_config: dict
    computation_time_s: float = 0.0
    timestamp: str = ""

    @property
    def num_points(self) -> int:
        return len(self.point_results)

    def summary(self) -> dict:
        """Summary statistics across all dosimetry points."""
        if not self.point_results:
            return {}

        return {
            "scenario_name": self.scenario_name,
            "num_points": self.num_points,
            "mean_areal_density_gcm2": np.mean(
                [p.mean_areal_density for p in self.point_results]
            ),
            "min_areal_density_gcm2": min(
                p.min_areal_density for p in self.point_results
            ),
            "max_areal_density_gcm2": max(
                p.max_areal_density for p in self.point_results
            ),
            "mean_gcr_dose_rate_mSv_yr": np.mean(
                [p.mean_gcr_dose_rate for p in self.point_results]
            ),
            "mean_gcr_dose_eq_rate_mSv_yr": np.mean(
                [p.mean_gcr_dose_equivalent_rate for p in self.point_results]
            ),
            "mean_spe_dose_mSv": np.mean(
                [p.mean_spe_dose for p in self.point_results]
            ),
            "mean_spe_dose_eq_mSv": np.mean(
                [p.mean_spe_dose_equivalent for p in self.point_results]
            ),
            "computation_time_s": self.computation_time_s,
            "confidence": ConfidenceLevel.CONCEPTUAL_ESTIMATE.value,
        }


class AnalysisEngine:
    """Main shielding analysis engine.

    Coordinates ray casting, material lookups, and radiation response
    computation to produce dose estimates at target points.
    """

    def __init__(self, n_directions: int = 162):
        self.raycaster = RayCaster(n_directions=n_directions)
        self._progress_callback: Callable[[float, str], None] | None = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set a callback for progress updates: callback(fraction, message)."""
        self._progress_callback = callback

    def _report_progress(self, fraction: float, message: str):
        if self._progress_callback:
            self._progress_callback(fraction, message)

    def run_analysis(
        self,
        scene: Scene,
        material_library: dict[str, Material],
        environment: RadiationEnvironmentConfig,
        scenario_name: str = "Default",
    ) -> ScenarioResult:
        """Run complete shielding analysis.

        Steps:
        1. Cast rays from all target points through geometry
        2. Compute areal density along each ray
        3. Apply radiation environment response functions
        4. Aggregate results
        """
        start_time = time.time()

        # Build material density lookup
        material_densities = {
            mid: mat.effective_density for mid, mat in material_library.items()
        }

        self._report_progress(0.0, "Starting ray casting...")

        # Step 1: Ray casting
        ray_results_list = self.raycaster.cast_all_targets(scene)

        total_points = len(ray_results_list)
        point_results = []

        for idx, ray_results in enumerate(ray_results_list):
            self._report_progress(
                (idx + 1) / total_points,
                f"Analyzing point {idx + 1}/{total_points}: "
                f"{ray_results.target_name}/{ray_results.point_name}",
            )

            # Step 2 & 3: Compute dose for each direction
            dir_results = []
            for ray in ray_results.rays:
                # Compute areal density
                ad = ray.areal_density(material_densities)
                per_mat_ad = ray.per_material_areal_density(material_densities)

                # Compute radiation responses
                gcr_dose = environment.gcr.dose_behind_shielding(ad)
                gcr_dose_eq = environment.gcr.dose_equivalent_behind_shielding(ad)
                flux_atten = environment.gcr.flux_attenuation(ad)

                spe_dose = 0.0
                spe_dose_eq = 0.0
                if environment.spe:
                    spe_dose = environment.spe.dose_behind_shielding(ad)
                    spe_dose_eq = environment.spe.dose_equivalent_behind_shielding(ad)

                dir_results.append(
                    DirectionalResult(
                        theta=ray.theta,
                        phi=ray.phi,
                        areal_density=ad,
                        per_material_areal_density=per_mat_ad,
                        gcr_dose_rate=gcr_dose,
                        gcr_dose_equivalent_rate=gcr_dose_eq,
                        spe_dose=spe_dose,
                        spe_dose_equivalent=spe_dose_eq,
                        flux_attenuation=flux_atten,
                    )
                )

            point_results.append(
                PointResult(
                    target_name=ray_results.target_name,
                    point_name=ray_results.point_name,
                    position=ray_results.position,
                    directional_results=dir_results,
                )
            )

        elapsed = time.time() - start_time
        self._report_progress(1.0, f"Analysis complete in {elapsed:.1f}s")

        return ScenarioResult(
            scenario_name=scenario_name,
            point_results=point_results,
            environment_config=environment.to_dict(),
            computation_time_s=elapsed,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


def compute_dose_vs_thickness(
    material: Material,
    environment: RadiationEnvironmentConfig,
    thickness_range_cm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute dose metrics as a function of material thickness.

    Returns dict with keys: thickness_cm, areal_density_gcm2,
    gcr_dose_mSv_yr, gcr_dose_eq_mSv_yr, spe_dose_mSv, spe_dose_eq_mSv
    """
    if thickness_range_cm is None:
        thickness_range_cm = np.linspace(0, 200, 200)

    areal_densities = material.effective_density * thickness_range_cm

    gcr_doses = np.array([
        environment.gcr.dose_behind_shielding(ad) for ad in areal_densities
    ])
    gcr_dose_eqs = np.array([
        environment.gcr.dose_equivalent_behind_shielding(ad) for ad in areal_densities
    ])

    spe_doses = np.zeros_like(areal_densities)
    spe_dose_eqs = np.zeros_like(areal_densities)
    if environment.spe:
        spe_doses = np.array([
            environment.spe.dose_behind_shielding(ad) for ad in areal_densities
        ])
        spe_dose_eqs = np.array([
            environment.spe.dose_equivalent_behind_shielding(ad) for ad in areal_densities
        ])

    return {
        "thickness_cm": thickness_range_cm,
        "areal_density_gcm2": areal_densities,
        "gcr_dose_mSv_yr": gcr_doses,
        "gcr_dose_eq_mSv_yr": gcr_dose_eqs,
        "spe_dose_mSv": spe_doses,
        "spe_dose_eq_mSv": spe_dose_eqs,
    }
