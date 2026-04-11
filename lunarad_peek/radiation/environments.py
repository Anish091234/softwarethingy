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
# NASA Dose Limits (NASA-STD-3001 Rev B, 2022; NCRP 98)
# ============================================================================

NASA_DOSE_LIMITS = {
    "bfo_annual_mSv_yr": 500,       # BFO annual limit (mGy-Eq/yr)
    "bfo_30day_mSv": 250,           # BFO 30-day limit (mGy-Eq/30 days)
    "bfo_spe_mSv": 250,             # BFO short-term SPE (mSv)
    "career_mSv": 600,              # Career limit (universal, 2022)
    "skin_30day_mGy_Eq": 1500,      # Skin 30-day limit
    "eye_30day_mGy_Eq": 1000,       # Eye (lens) 30-day limit
}


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
               O'Neill (2010/2014), IEEE Trans. Nucl. Sci.
               CRaTER measurements: Schwadron et al. (2012)

    Free-space GCR dose equivalent rate:
    - Solar minimum (phi~400 MV): ~480-500 mSv/year
    - Solar maximum (phi~1200 MV): ~200-300 mSv/year

    On lunar surface (~80% of free-space, 2π shielding + albedo neutrons):
    - Solar minimum: ~380-400 mSv/year (CRaTER measurements)
    """

    phi_MV: float = 400.0  # Solar modulation parameter (MV)
    phase: SolarCyclePhase = SolarCyclePhase.SOLAR_MINIMUM

    # Dose equivalent rate parameterization: D_eq(phi) = A * exp(-phi/B) + C
    # Calibrated to: phi=400 -> ~490 mSv/yr, phi=1200 -> ~250 mSv/yr
    _A: float = 635.0
    _B: float = 600.0
    _C: float = 164.0

    # Lunar surface factor (accounts for 2π geometry + albedo neutrons)
    _LUNAR_SURFACE_FACTOR: float = 0.80

    # Asymptotic deep-shielding floor as a fraction of the unshielded lunar
    # surface dose. Captures the >1 GeV/n HZE primaries that are essentially
    # unshieldable by passive mass, plus neutron-buildup plateau.
    # Calibrated so D(∞) ≈ 200-220 mSv/yr (Slaba et al. 2017; Cucinotta 2015
    # deep-shielding asymptotes).
    _GCR_FLOOR_FRACTION: float = 0.55

    # GCR composition by fluence (approximate)
    ION_SPECIES = {
        "H": {"Z": 1, "A": 1, "flux_fraction": 0.87, "dose_eq_fraction": 0.15},
        "He": {"Z": 2, "A": 4, "flux_fraction": 0.12, "dose_eq_fraction": 0.08},
        "C": {"Z": 6, "A": 12, "flux_fraction": 0.003, "dose_eq_fraction": 0.06},
        "O": {"Z": 8, "A": 16, "flux_fraction": 0.003, "dose_eq_fraction": 0.08},
        "Si": {"Z": 14, "A": 28, "flux_fraction": 0.001, "dose_eq_fraction": 0.07},
        "Fe": {"Z": 26, "A": 56, "flux_fraction": 0.001, "dose_eq_fraction": 0.15},
        "other_hze": {"Z": 0, "A": 0, "flux_fraction": 0.002, "dose_eq_fraction": 0.41},
    }

    @property
    def free_space_dose_equivalent_rate(self) -> float:
        """Unshielded free-space GCR dose equivalent rate (mSv/year)."""
        return self._A * math.exp(-self.phi_MV / self._B) + self._C

    @property
    def lunar_surface_dose_equivalent_rate(self) -> float:
        """Unshielded lunar surface GCR dose equivalent rate (mSv/year).

        ~80% of free-space (2π solid-angle shielding from lunar body +
        albedo neutron contribution from surface).
        Consistent with LRO/CRaTER measurements (~380-400 mSv/yr).
        """
        return self.free_space_dose_equivalent_rate * self._LUNAR_SURFACE_FACTOR

    @property
    def free_space_dose_rate(self) -> float:
        """Unshielded free-space GCR absorbed dose rate (mSv/year).

        Approximate: dose_equivalent / Q_avg where Q_avg ~ 2.5 for GCR.
        """
        return self.free_space_dose_equivalent_rate / 2.5

    @property
    def lunar_surface_dose_rate(self) -> float:
        """Unshielded lunar surface absorbed dose rate (mSv/year).

        ~50% of free-space (2π geometric shielding, no albedo for absorbed dose).
        """
        return self.free_space_dose_rate * 0.50

    def dose_behind_shielding(self, areal_density_gcm2: float,
                              lambda_eff: float = 25.0,
                              hydrogen_fraction: float = 0.0) -> float:
        """Estimate GCR absorbed dose rate behind shielding (mSv/year).

        Single-component asymptotic exponential with unshieldable HZE floor:

            D(x) = D_floor + (D0 - D_floor) × exp(-x / λ_h)

        where:
            D0       = unshielded lunar surface dose rate (2π body shielding
                       already baked in)
            D_floor  = D0 × _GCR_FLOOR_FRACTION (deep-shielding asymptote)
            λ_h      = lambda_eff × hydrogen-content correction

        This restores the pre-fix attenuation length (λ ≈ 25 g/cm² for Al)
        and only adds the H-content correction multiplicatively, plus an
        unshieldable HZE plateau so dose cannot fall below ~200 mSv/yr at
        any thickness — consistent with literature deep-shielding values.

        Per-direction response. Downward rays with no geometry hit return
        D0 from this function and are excluded from the target-point mean
        at the engine level (the moon body absorbs them physically).

        Args:
            areal_density_gcm2: Shielding areal density along ray (g/cm²)
            lambda_eff: Material-dependent attenuation length (g/cm²).
                       Al~25, PEEK~21, Regolith~24 (Material.gcr_effective_lambda).
            hydrogen_fraction: Hydrogen weight fraction (0-1).
        """
        x = areal_density_gcm2
        D0 = self.lunar_surface_dose_rate
        D_floor = D0 * self._GCR_FLOOR_FRACTION
        # H-rich materials shield more effectively per g/cm² (more spallation
        # of heavy primaries, fewer secondary neutrons). Compress λ.
        h_factor = max(1.0 - 2.0 * hydrogen_fraction, 0.5)
        lam_h = lambda_eff * h_factor
        return D_floor + (D0 - D_floor) * math.exp(-x / lam_h)

    def dose_equivalent_behind_shielding(self, areal_density_gcm2: float,
                                          lambda_eff: float = 25.0,
                                          hydrogen_fraction: float = 0.0) -> float:
        """Estimate GCR dose equivalent rate behind shielding (mSv/year).

        Single-component asymptotic exponential with unshieldable HZE floor:

            H(x) = H_floor + (H0 - H_floor) × exp(-x / λ_h)

        Calibrated so:
          - H(0)  ≈ 380-400 mSv/yr (CRaTER lunar surface, solar min)
          - H(85) ≈ 200-220 mSv/yr (literature mid-range shielding)
          - H(∞)  ≈ 215 mSv/yr (deep-shielding asymptote ≥ 100 mSv/yr)
        Ref: Slaba et al. (2017); Cucinotta (2015); Schwadron et al. (2012).
        """
        x = areal_density_gcm2
        D0 = self.lunar_surface_dose_equivalent_rate
        D_floor = D0 * self._GCR_FLOOR_FRACTION
        h_factor = max(1.0 - 2.0 * hydrogen_fraction, 0.5)
        lam_h = lambda_eff * h_factor
        return D_floor + (D0 - D_floor) * math.exp(-x / lam_h)

    def flux_attenuation(self, areal_density_gcm2: float,
                         lambda_eff: float = 25.0) -> float:
        """Particle flux attenuation factor (0-1)."""
        x = areal_density_gcm2
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
            "free_space_dose_eq_rate_mSv_yr": self.free_space_dose_equivalent_rate,
            "lunar_surface_dose_eq_rate_mSv_yr": self.lunar_surface_dose_equivalent_rate,
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
    total_fluence_gt10MeV: float = 0.0  # protons/cm² above 10 MeV
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
        total_fluence_gt10MeV=1.1e10,
        description=(
            "Design-driving worst-case SPE. IMP-5 satellite data. "
            "Occurred between Apollo 16 and 17. Duration ~2-3 days. "
            "Unshielded free-space skin dose ~10-15 Gy."
        ),
        source="King (1974); Parsons & Townsend (2000); Jiggens et al. (2014)",
    ),
    "oct_1989": SPEEvent(
        name="October 1989",
        date="1989-10-19",
        J0=4.0e9,
        gamma=1.1,
        E0_MeV=30.0,
        total_fluence_gt30MeV=4.2e9,
        total_fluence_gt10MeV=8.0e9,
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
        total_fluence_gt10MeV=3.0e9,
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
        total_fluence_gt10MeV=5.5e9,
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
        total_fluence_gt10MeV=4.0e9,
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
        total_fluence_gt10MeV=2.2e10,
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

    SPE proton attenuation is MATERIAL-DEPENDENT, and is dominated by two
    separable effects:

    1. HYDROGEN content (primary): SPEs are pure proton events, so proton-
       proton elastic scattering gives maximum kinematic energy transfer
       (equal masses → full energy exchange per collision). H also provides
       the highest electron density per gram (~1 e/g vs ~0.48 e/g for Al)
       driving ionisation energy loss. Hydrogen-rich materials therefore
       attenuate SPE protons ~1.3–1.8× more effectively per g/cm² than Al,
       independently of mean atomic mass.
    2. Mean atomic mass (secondary): nuclear interaction cross-section
       scales as ~A^(1/3) per unit mass, so lower-A materials give a mild
       additional boost to per-mass attenuation.

    Combined mean free path model:
        λ_SPE(mat) = λ_SPE(Al) × (A_eff / 27)^(1/3) × h_factor
        h_factor   = max(1 − 10·H_wt, 0.4)

    The hydrogen coefficient (10.0) and the 0.4 floor are calibrated to
    reproduce the Wilson et al. (1997) and Cucinotta et al. (2006) finding
    that pure polyethylene (H_wt ≈ 0.143) attenuates SPE protons roughly
    2–2.5× better per g/cm² than aluminum for the Aug 1972 spectrum
    (h_factor → 0.4, the floor). PEEK (H_wt ≈ 0.047, h_factor ≈ 0.53) and
    the regolith-PEEK composite (H_wt ≈ 0.028, h_factor ≈ 0.72) therefore
    pull below Al on a per-areal-density basis, as observed experimentally,
    while hydrogen-free materials (aluminum, bare regolith, basalt) retain
    h_factor = 1 and show the largest per-g/cm² SPE dose.

    Reference: Wilson et al. (1997) "Shielding Strategies for Human Space
    Exploration"; Cucinotta et al. (2006) NASA/TP-2006-213689;
    Singleterry (2013) Acta Astronautica 91.
    """

    event: SPEEvent

    # Fluence-to-dose conversion factor (mSv per proton/cm² for >30 MeV)
    # Calibrated so Aug 1972 free-space skin dose ≈ 12,500 mSv (~12.5 Gy)
    _DOSE_CONVERSION: float = 2.5e-6

    def dose_behind_shielding(self, areal_density_gcm2: float,
                              mean_atomic_mass: float = 27.0,
                              hydrogen_fraction: float = 0.0) -> float:
        """Estimate SPE dose behind shielding (mSv for the event).

        SPE protons have softer spectrum than GCR, so they are more
        effectively shielded. Uses material-dependent mean free path with
        both A_eff^(1/3) and explicit hydrogen corrections.

        Args:
            areal_density_gcm2: Shielding areal density (g/cm²)
            mean_atomic_mass: Effective mean atomic mass of shielding (g/mol).
                Al=27.0, PEEK≈8.2, Regolith≈21.7. Controls the weak nuclear-
                structure piece of the mean free path via (A/27)^(1/3).
            hydrogen_fraction: Hydrogen weight fraction (0-1). Drives the
                dominant proton-shielding correction. Al=0, PEEK≈0.047,
                Reg-PEEK composite≈0.028, polyethylene≈0.143.
        """
        x = areal_density_gcm2

        # Unshielded free-space dose from fluence
        D0 = self.event.total_fluence_gt30MeV * self._DOSE_CONVERSION  # mSv

        # (1) Mild nuclear-mass correction (~A^(1/3))
        a_scale = (mean_atomic_mass / 27.0) ** (1.0 / 3.0)

        # (2) Dominant hydrogen correction. Protons lose the most energy
        # per collision against H (equal-mass elastic scatter), and H has
        # the highest electron density per gram. We shrink λ proportional
        # to H weight fraction, with a floor so the formula cannot go
        # non-physical for pure-H or hydride shields. Coefficient tuned to
        # polyethylene benchmark (see class docstring).
        h_factor = max(1.0 - 10.0 * hydrogen_fraction, 0.4)

        lambda_soft = 8.0 * a_scale * h_factor   # bulk SPE protons
        lambda_hard = 20.0 * a_scale * h_factor  # high-energy tail

        dose = D0 * (
            0.7 * math.exp(-x / lambda_soft)
            + 0.3 * math.exp(-x / lambda_hard)
        )

        # Per-direction free-space response. Lunar 2π shielding is applied
        # by the engine zeroing rays that point into the lunar body.
        return dose

    def dose_equivalent_behind_shielding(self, areal_density_gcm2: float,
                                          mean_atomic_mass: float = 27.0,
                                          hydrogen_fraction: float = 0.0) -> float:
        """SPE dose equivalent (mSv). Q ~ 1.2-1.5 for proton-dominated events."""
        Q = 1.3
        return self.dose_behind_shielding(
            areal_density_gcm2,
            mean_atomic_mass=mean_atomic_mass,
            hydrogen_fraction=hydrogen_fraction,
        ) * Q

    def to_dict(self) -> dict:
        return {
            "type": "SPE",
            "event_name": self.event.name,
            "event_date": self.event.date,
            "total_fluence_gt30MeV": self.event.total_fluence_gt30MeV,
            "total_fluence_gt10MeV": self.event.total_fluence_gt10MeV,
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
