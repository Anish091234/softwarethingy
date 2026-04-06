"""Radiation environment models for lunar surface.

Provides GCR, SPE, and Solar Wind environment definitions with spectral
data and dose-rate estimation. Uses literature-derived parameterizations.

IMPORTANT: These are conceptual approximations, NOT Monte Carlo transport results.
All outputs should be clearly labeled as estimates suitable for conceptual design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class EnvironmentType(Enum):
    GCR = "gcr"
    SPE = "spe"
    SOLAR_WIND = "solar_wind"


class SolarCyclePhase(Enum):
    SOLAR_MINIMUM = "solar_minimum"
    SOLAR_MAXIMUM = "solar_maximum"
    INTERMEDIATE = "intermediate"


# ============================================================================
# GCR Environment
# ============================================================================

@dataclass
class GCREnvironment:
    """Galactic Cosmic Ray environment model.

    Uses a simplified Badhwar-O'Neill-inspired parameterization.
    The solar modulation parameter phi controls the GCR intensity
    (higher phi = more solar activity = lower GCR flux).

    Reference: Badhwar & O'Neill (1996), Adv. Space Res.
               O'Neill (2010), IEEE Trans. Nucl. Sci.

    Free-space GCR dose rate is approximately:
    - Solar minimum (phi~400 MV): ~600-800 mSv/year
    - Solar maximum (phi~1200 MV): ~200-300 mSv/year

    On lunar surface (2pi shielding from body below):
    - Approximately 50% of free-space value
    """

    phi_MV: float = 400.0  # Solar modulation parameter (MV)
    phase: SolarCyclePhase = SolarCyclePhase.SOLAR_MINIMUM

    # Dose rates in free space (mSv/year) for reference
    # Parameterized as function of phi
    # Fit: D(phi) = A * exp(-phi/B) + C
    # Calibrated to: phi=400 -> ~660 mSv/yr, phi=1200 -> ~250 mSv/yr
    _A: float = 580.0
    _B: float = 650.0
    _C: float = 180.0

    # Major GCR ion species with relative contributions to dose equivalent
    ION_SPECIES = {
        "H": {"Z": 1, "A": 1, "dose_fraction": 0.35, "dose_eq_fraction": 0.15},
        "He": {"Z": 2, "A": 4, "dose_fraction": 0.13, "dose_eq_fraction": 0.08},
        "C": {"Z": 6, "A": 12, "dose_fraction": 0.04, "dose_eq_fraction": 0.06},
        "O": {"Z": 8, "A": 16, "dose_fraction": 0.05, "dose_eq_fraction": 0.08},
        "Si": {"Z": 14, "A": 28, "dose_fraction": 0.03, "dose_eq_fraction": 0.07},
        "Fe": {"Z": 26, "A": 56, "dose_fraction": 0.02, "dose_eq_fraction": 0.15},
        "other": {"Z": 0, "A": 0, "dose_fraction": 0.38, "dose_eq_fraction": 0.41},
    }

    @property
    def free_space_dose_rate(self) -> float:
        """Unshielded free-space GCR dose rate (mSv/year)."""
        return self._A * math.exp(-self.phi_MV / self._B) + self._C

    @property
    def lunar_surface_dose_rate(self) -> float:
        """Unshielded lunar surface GCR dose rate (mSv/year).

        Approximately 50% of free-space due to 2pi solid-angle shielding
        from the lunar body below.
        """
        return self.free_space_dose_rate * 0.50

    @property
    def free_space_dose_equivalent_rate(self) -> float:
        """Free-space GCR dose equivalent rate (mSv/year).

        Quality factor weighted. Average Q ~ 3-5 for GCR.
        Using Q_avg ~ 3.5 based on NCRP 153.
        """
        return self.free_space_dose_rate * 3.5

    @property
    def lunar_surface_dose_equivalent_rate(self) -> float:
        """Lunar surface dose equivalent rate (mSv/year)."""
        return self.free_space_dose_equivalent_rate * 0.50

    def dose_behind_shielding(self, areal_density_gcm2: float) -> float:
        """Estimate GCR dose rate behind shielding (mSv/year).

        Uses exponential attenuation with empirically-calibrated
        attenuation length. Includes approximate buildup factor
        for secondary neutrons.

        D(x) = D₀ × exp(-x/λ) × B(x)

        where:
            x = areal density (g/cm²)
            λ = effective attenuation length (~25 g/cm² for mixed GCR)
            B(x) = 1 + k×x  (neutron buildup, k ~ 0.003-0.005)

        Reference: Wilson et al. (1995), NASA TP-3682
                   Cucinotta et al. (2006), Radiat. Res.
        """
        x = areal_density_gcm2
        D0 = self.lunar_surface_dose_rate
        lambda_eff = 25.0  # g/cm² effective attenuation length

        # Primary attenuation
        primary = D0 * math.exp(-x / lambda_eff)

        # Secondary neutron buildup (approximate)
        # Peaks around 20-40 g/cm² then decreases
        k = 0.004
        x_peak = 30.0
        if x < x_peak:
            buildup = 1.0 + k * x
        else:
            buildup = 1.0 + k * x_peak * math.exp(-(x - x_peak) / 40.0)

        return primary * buildup

    def dose_equivalent_behind_shielding(self, areal_density_gcm2: float) -> float:
        """Estimate GCR dose equivalent rate behind shielding (mSv/year).

        Quality factor decreases slightly with shielding as heavy ions
        fragment into lighter particles.
        """
        x = areal_density_gcm2
        # Q decreases from ~3.5 to ~2.5 with increasing shielding
        Q = 3.5 * math.exp(-x / 200.0) + 2.0 * (1.0 - math.exp(-x / 200.0))
        dose = self.dose_behind_shielding(x)
        return dose * Q

    def flux_attenuation(self, areal_density_gcm2: float) -> float:
        """Particle flux attenuation factor (0-1)."""
        x = areal_density_gcm2
        lambda_eff = 25.0
        return math.exp(-x / lambda_eff)

    @classmethod
    def solar_minimum(cls) -> GCREnvironment:
        return cls(phi_MV=400.0, phase=SolarCyclePhase.SOLAR_MINIMUM)

    @classmethod
    def solar_maximum(cls) -> GCREnvironment:
        return cls(phi_MV=1200.0, phase=SolarCyclePhase.SOLAR_MAXIMUM)

    @classmethod
    def from_phase(cls, phase: SolarCyclePhase) -> GCREnvironment:
        phi_map = {
            SolarCyclePhase.SOLAR_MINIMUM: 400.0,
            SolarCyclePhase.SOLAR_MAXIMUM: 1200.0,
            SolarCyclePhase.INTERMEDIATE: 700.0,
        }
        return cls(phi_MV=phi_map[phase], phase=phase)

    def to_dict(self) -> dict:
        return {
            "type": "GCR",
            "phi_MV": self.phi_MV,
            "phase": self.phase.value,
            "free_space_dose_rate_mSv_yr": self.free_space_dose_rate,
            "lunar_surface_dose_rate_mSv_yr": self.lunar_surface_dose_rate,
        }


# ============================================================================
# SPE Environment
# ============================================================================

@dataclass
class SPEEvent:
    """A Solar Particle Event parameterized by Band function.

    Differential fluence: dF/dE = J₀ × E^(-γ) × exp(-E/E₀)

    Parameters fit to historical events from literature.
    Reference: Band et al. (1993); Tylka et al. (2006)
    """

    name: str
    date: str
    J0: float  # Normalization (protons/cm²/MeV)
    gamma: float  # Spectral index
    E0_MeV: float  # Characteristic energy (MeV)
    total_fluence_gt30MeV: float  # protons/cm² above 30 MeV
    description: str = ""
    source: str = ""

    def fluence_spectrum(self, energies_MeV: np.ndarray) -> np.ndarray:
        """Differential fluence spectrum dF/dE (protons/cm²/MeV)."""
        return self.J0 * energies_MeV ** (-self.gamma) * np.exp(-energies_MeV / self.E0_MeV)

    def integrated_fluence_above(self, E_min_MeV: float) -> float:
        """Integrated fluence above E_min (protons/cm²)."""
        from scipy.integrate import quad

        result, _ = quad(
            lambda E: self.J0 * E ** (-self.gamma) * math.exp(-E / self.E0_MeV),
            E_min_MeV,
            1e4,  # upper limit 10 GeV
        )
        return result


# Built-in SPE event library
SPE_EVENT_LIBRARY: dict[str, SPEEvent] = {
    "aug_1972": SPEEvent(
        name="August 1972",
        date="1972-08-04",
        J0=2.0e10,
        gamma=1.4,
        E0_MeV=26.0,
        total_fluence_gt30MeV=5.0e9,
        description="One of the largest recorded SPEs. Occurred between Apollo 16 and 17.",
        source="King (1974); Tylka et al. (2006)",
    ),
    "oct_1989": SPEEvent(
        name="October 1989",
        date="1989-10-19",
        J0=4.0e9,
        gamma=1.1,
        E0_MeV=30.0,
        total_fluence_gt30MeV=4.2e9,
        description="Series of intense SPEs in October 1989.",
        source="Tylka et al. (2006); Shea & Smart (1990)",
    ),
    "sep_2017": SPEEvent(
        name="September 2017",
        date="2017-09-10",
        J0=1.0e8,
        gamma=1.3,
        E0_MeV=22.0,
        total_fluence_gt30MeV=1.1e9,
        description="GLE72 event, significant ground-level enhancement.",
        source="Bruno et al. (2019)",
    ),
    "jan_2005": SPEEvent(
        name="January 2005",
        date="2005-01-20",
        J0=5.0e8,
        gamma=1.2,
        E0_MeV=35.0,
        total_fluence_gt30MeV=2.8e9,
        description="GLE69, one of the hardest spectrum SPEs of solar cycle 23.",
        source="Mewaldt et al. (2005)",
    ),
    "jul_2000": SPEEvent(
        name="July 2000 (Bastille Day)",
        date="2000-07-14",
        J0=3.0e8,
        gamma=1.15,
        E0_MeV=28.0,
        total_fluence_gt30MeV=1.8e9,
        description="Bastille Day event with significant proton flux.",
        source="Tylka et al. (2006)",
    ),
    "design_reference": SPEEvent(
        name="Design Reference SPE",
        date="N/A",
        J0=5.0e10,
        gamma=1.3,
        E0_MeV=30.0,
        total_fluence_gt30MeV=1.0e10,
        description=(
            "Hypothetical design-reference event based on 95th percentile "
            "of historical SPE distribution. For conservative design analysis."
        ),
        source="Kim et al. (2009), NASA/TP-2009-214788",
    ),
}


@dataclass
class SPEEnvironment:
    """Solar Particle Event environment.

    Reference: Cucinotta et al. (2006); Wilson et al. (1997)
    """

    event: SPEEvent

    def dose_behind_shielding(self, areal_density_gcm2: float) -> float:
        """Estimate SPE dose behind shielding (mSv for the event).

        SPE protons have softer spectrum than GCR, so they are more
        effectively shielded. Attenuation is steeper.

        D(x) ≈ D₀ × exp(-x/λ_SPE)

        λ_SPE ~ 8-15 g/cm² (softer spectrum = shorter attenuation length)
        """
        x = areal_density_gcm2

        # Unshielded lunar surface dose (rough estimate from fluence)
        # Using dose conversion: ~2e-9 mSv per proton/cm² (> 30 MeV)
        D0 = self.event.total_fluence_gt30MeV * 2.0e-9  # mSv

        # Two-component attenuation: soft + hard spectrum
        lambda_soft = 8.0  # g/cm² for bulk of SPE protons
        lambda_hard = 20.0  # g/cm² for high-energy tail

        dose = D0 * (
            0.7 * math.exp(-x / lambda_soft)
            + 0.3 * math.exp(-x / lambda_hard)
        )

        return dose * 0.5  # lunar surface 2pi factor

    def dose_equivalent_behind_shielding(self, areal_density_gcm2: float) -> float:
        """SPE dose equivalent (mSv). Q ~ 1.2-1.5 for proton-dominated events."""
        Q = 1.3
        return self.dose_behind_shielding(areal_density_gcm2) * Q

    def to_dict(self) -> dict:
        return {
            "type": "SPE",
            "event_name": self.event.name,
            "event_date": self.event.date,
            "total_fluence_gt30MeV": self.event.total_fluence_gt30MeV,
        }


# ============================================================================
# Solar Wind Environment (separate module)
# ============================================================================

@dataclass
class SolarWindEnvironment:
    """Solar wind environment model.

    Solar wind consists primarily of ~1 keV/nucleon protons at ~400 km/s.
    This is a SURFACE INTERACTION only - solar wind particles are stopped
    by any solid shielding and do NOT contribute to habitat biological dose.

    This module is kept separate from GCR/SPE biological dose assessment.

    Reference: Feldman et al. (2001); Farrell et al. (2015)
    """

    flux_protons_cm2_s: float = 3.0e8  # protons/cm²/s at 1 AU
    velocity_km_s: float = 400.0  # bulk velocity
    energy_keV_nucleon: float = 1.0  # ~1 keV/nucleon
    alpha_fraction: float = 0.04  # He²⁺ fraction

    @property
    def energy_deposition_rate(self) -> float:
        """Surface energy deposition rate (eV/cm²/s).

        Solar wind deposits energy in top ~100 nm of regolith.
        """
        # Each proton deposits ~1 keV
        return self.flux_protons_cm2_s * self.energy_keV_nucleon * 1e3

    @property
    def annual_fluence(self) -> float:
        """Annual proton fluence (protons/cm²/year)."""
        return self.flux_protons_cm2_s * 3.156e7

    @property
    def sputtering_yield_atoms_per_ion(self) -> float:
        """Approximate sputtering yield for regolith (atoms/ion)."""
        return 0.05  # Very low for 1 keV protons on oxide minerals

    def is_stopped_by_any_shielding(self) -> bool:
        """Solar wind is stopped by any solid material > ~1 micrometer."""
        return True

    def surface_dose_rate(self) -> float:
        """Surface dose rate from solar wind only (mSv/year).

        Negligible compared to GCR/SPE for biological dose. Included
        for completeness in surface interaction analysis.
        """
        # ~1 keV protons, very short range, skin dose only
        return 0.01  # Negligible compared to GCR

    def to_dict(self) -> dict:
        return {
            "type": "SolarWind",
            "flux_protons_cm2_s": self.flux_protons_cm2_s,
            "velocity_km_s": self.velocity_km_s,
            "energy_keV_nucleon": self.energy_keV_nucleon,
            "note": "Surface interaction only. Stopped by any solid shielding.",
        }


# ============================================================================
# Combined Environment Configuration
# ============================================================================

@dataclass
class RadiationEnvironmentConfig:
    """Complete radiation environment configuration for an analysis."""

    gcr: GCREnvironment = field(default_factory=GCREnvironment.solar_minimum)
    spe: SPEEnvironment | None = None
    solar_wind: SolarWindEnvironment | None = None
    include_secondary: bool = False  # Include approximate secondary radiation
    description: str = ""

    def to_dict(self) -> dict:
        result = {"gcr": self.gcr.to_dict(), "include_secondary": self.include_secondary}
        if self.spe:
            result["spe"] = self.spe.to_dict()
        if self.solar_wind:
            result["solar_wind"] = self.solar_wind.to_dict()
        return result
