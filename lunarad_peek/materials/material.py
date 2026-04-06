"""Material definitions and composite material system for radiation shielding analysis.

Materials are characterized by their elemental composition, density, and derived
radiation interaction properties. Composite materials support weight-fraction and
volume-fraction definitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class CompositeMode(Enum):
    WEIGHT_FRACTION = "weight_fraction"
    VOLUME_FRACTION = "volume_fraction"


@dataclass
class Element:
    """Atomic element with radiation-relevant properties."""

    symbol: str
    Z: int  # Atomic number
    A: float  # Atomic mass (g/mol)
    I_eV: float  # Mean excitation energy (eV) for Bethe-Bloch


# Common elements used in lunar materials
ELEMENTS = {
    "H": Element("H", 1, 1.008, 19.2),
    "C": Element("C", 6, 12.011, 78.0),
    "N": Element("N", 7, 14.007, 82.0),
    "O": Element("O", 8, 15.999, 95.0),
    "Na": Element("Na", 11, 22.990, 149.0),
    "Mg": Element("Mg", 12, 24.305, 156.0),
    "Al": Element("Al", 13, 26.982, 166.0),
    "Si": Element("Si", 14, 28.086, 173.0),
    "K": Element("K", 19, 39.098, 190.0),
    "Ca": Element("Ca", 20, 40.078, 191.0),
    "Ti": Element("Ti", 22, 47.867, 233.0),
    "Cr": Element("Cr", 24, 51.996, 257.0),
    "Mn": Element("Mn", 25, 54.938, 272.0),
    "Fe": Element("Fe", 26, 55.845, 286.0),
}


@dataclass
class Material:
    """A material defined by its elemental composition and physical properties.

    Composition is given as a dict of element symbol -> weight fraction (0-1).
    All weight fractions must sum to 1.0 (within tolerance).
    """

    name: str
    density: float  # g/cm³
    composition: dict[str, float]  # element symbol -> weight fraction
    porosity: float = 0.0  # 0-1, fraction of void space
    description: str = ""
    source: str = ""  # literature reference

    def __post_init__(self):
        total = sum(self.composition.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Material '{self.name}' composition weight fractions sum to "
                f"{total:.4f}, expected 1.0"
            )

    @property
    def effective_density(self) -> float:
        """Density accounting for porosity (g/cm³)."""
        return self.density * (1.0 - self.porosity)

    @property
    def Z_eff(self) -> float:
        """Effective atomic number (power-law weighted)."""
        z_sum = 0.0
        for symbol, wf in self.composition.items():
            elem = ELEMENTS[symbol]
            n_i = wf / elem.A  # number fraction proportional
            z_sum += n_i * elem.Z ** 2.94
        n_total = sum(wf / ELEMENTS[s].A for s, wf in self.composition.items())
        return (z_sum / n_total) ** (1.0 / 2.94)

    @property
    def mean_A(self) -> float:
        """Mean atomic mass (g/mol)."""
        return 1.0 / sum(wf / ELEMENTS[s].A for s, wf in self.composition.items())

    @property
    def mean_excitation_energy(self) -> float:
        """Mean excitation energy I (eV) via Bragg additivity."""
        import math

        ln_I = 0.0
        for symbol, wf in self.composition.items():
            elem = ELEMENTS[symbol]
            n_frac = (wf / elem.A)
            ln_I += n_frac * elem.Z * math.log(elem.I_eV)
        z_total = sum(
            (wf / ELEMENTS[s].A) * ELEMENTS[s].Z
            for s, wf in self.composition.items()
        )
        return math.exp(ln_I / z_total)

    @property
    def radiation_length_approx(self) -> float:
        """Approximate radiation length X₀ (g/cm²) using Tsai's formula."""
        X0_inv = 0.0
        for symbol, wf in self.composition.items():
            elem = ELEMENTS[symbol]
            Z = elem.Z
            A = elem.A
            # Tsai approximation for radiation length
            a = 1.0 / 137.036
            Lrad = {
                1: 5.31, 2: 4.79, 3: 4.74, 4: 4.71
            }.get(Z, 0.0)
            if Lrad == 0.0:
                import math
                Lrad = math.log(184.15 * Z ** (-1.0 / 3.0))
            Lrad_prime = {
                1: 6.144, 2: 5.621, 3: 5.805, 4: 5.924
            }.get(Z, 0.0)
            if Lrad_prime == 0.0:
                import math
                Lrad_prime = math.log(1194.0 * Z ** (-2.0 / 3.0))
            X0_elem = 716.408 * A / (Z * Z * Lrad + Z * Lrad_prime)
            X0_inv += wf / X0_elem
        return 1.0 / X0_inv if X0_inv > 0 else 100.0

    @property
    def nuclear_interaction_length(self) -> float:
        """Approximate nuclear interaction length λ_I (g/cm²)."""
        # Empirical: λ_I ≈ 35 * A^(1/3) g/cm²
        lambda_inv = 0.0
        for symbol, wf in self.composition.items():
            elem = ELEMENTS[symbol]
            lambda_elem = 35.0 * elem.A ** (1.0 / 3.0)
            lambda_inv += wf / lambda_elem
        return 1.0 / lambda_inv if lambda_inv > 0 else 100.0

    def areal_density(self, thickness_cm: float) -> float:
        """Convert physical thickness to areal density (g/cm²)."""
        return self.effective_density * thickness_cm

    def thickness_from_areal(self, areal_density: float) -> float:
        """Convert areal density (g/cm²) to physical thickness (cm)."""
        if self.effective_density <= 0:
            return 0.0
        return areal_density / self.effective_density

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "density": self.density,
            "composition": self.composition,
            "porosity": self.porosity,
            "description": self.description,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Material:
        return cls(**data)


@dataclass
class CompositeMaterial:
    """A composite material made from constituent materials.

    Supports both weight-fraction and volume-fraction definitions.
    The resulting material properties are computed from the mixture rule.
    """

    name: str
    mode: CompositeMode
    constituents: list[tuple[Material, float]]  # (material, fraction)
    description: str = ""
    source: str = ""

    def __post_init__(self):
        total = sum(f for _, f in self.constituents)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Composite '{self.name}' fractions sum to {total:.4f}, expected 1.0"
            )

    @property
    def density(self) -> float:
        """Effective composite density (g/cm³)."""
        if self.mode == CompositeMode.VOLUME_FRACTION:
            return sum(mat.effective_density * vf for mat, vf in self.constituents)
        else:
            # Weight fraction: 1/ρ = Σ(wf_i / ρ_i)
            inv_rho = sum(wf / mat.effective_density for mat, wf in self.constituents)
            return 1.0 / inv_rho if inv_rho > 0 else 0.0

    @property
    def composition(self) -> dict[str, float]:
        """Effective elemental composition as weight fractions."""
        if self.mode == CompositeMode.WEIGHT_FRACTION:
            merged: dict[str, float] = {}
            for mat, wf in self.constituents:
                for elem, elem_wf in mat.composition.items():
                    merged[elem] = merged.get(elem, 0.0) + wf * elem_wf
            return merged
        else:
            # Convert volume fractions to weight fractions first
            total_mass = sum(
                mat.effective_density * vf for mat, vf in self.constituents
            )
            merged: dict[str, float] = {}
            for mat, vf in self.constituents:
                wf = (mat.effective_density * vf) / total_mass
                for elem, elem_wf in mat.composition.items():
                    merged[elem] = merged.get(elem, 0.0) + wf * elem_wf
            return merged

    def to_material(self) -> Material:
        """Flatten composite into a single effective Material."""
        return Material(
            name=self.name,
            density=self.density,
            composition=self.composition,
            porosity=0.0,
            description=self.description,
            source=self.source,
        )


def create_preset_materials() -> dict[str, Material]:
    """Create the built-in preset material library."""
    presets = {}

    # Lunar Highland Regolith (Apollo 16 average, simplified)
    # Major oxides: SiO2 45%, Al2O3 27%, CaO 16%, FeO 5%, MgO 6%
    # Converted to elemental weight fractions
    presets["highland_regolith"] = Material(
        name="Lunar Highland Regolith",
        density=1.5,  # bulk density with porosity
        composition={
            "O": 0.446,
            "Si": 0.210,
            "Al": 0.143,
            "Ca": 0.114,
            "Fe": 0.039,
            "Mg": 0.036,
            "Ti": 0.004,
            "Na": 0.004,
            "K": 0.001,
            "Mn": 0.001,
            "Cr": 0.002,
        },
        porosity=0.0,  # already bulk density
        description=(
            "Average lunar highland regolith based on Apollo 16 samples. "
            "Anorthositic composition, high Al and Ca content."
        ),
        source="Heiken et al., Lunar Sourcebook (1991), Ch. 7",
    )

    # Lunar Mare Regolith (Apollo 11/12 average, simplified)
    presets["mare_regolith"] = Material(
        name="Lunar Mare Regolith",
        density=1.7,  # bulk density
        composition={
            "O": 0.417,
            "Si": 0.198,
            "Fe": 0.126,
            "Ca": 0.079,
            "Al": 0.069,
            "Mg": 0.058,
            "Ti": 0.042,
            "Na": 0.003,
            "Mn": 0.002,
            "Cr": 0.004,
            "K": 0.002,
        },
        porosity=0.0,
        description=(
            "Average lunar mare regolith based on Apollo 11/12 samples. "
            "Basaltic composition, higher Fe and Ti content than highland."
        ),
        source="Heiken et al., Lunar Sourcebook (1991), Ch. 7",
    )

    # Lava-tube rock (dense basalt)
    presets["lavatube_rock"] = Material(
        name="Lava-tube Basalt",
        density=2.9,  # solid basalt
        composition={
            "O": 0.430,
            "Si": 0.215,
            "Fe": 0.100,
            "Ca": 0.080,
            "Al": 0.075,
            "Mg": 0.055,
            "Ti": 0.030,
            "Na": 0.005,
            "Mn": 0.003,
            "Cr": 0.005,
            "K": 0.002,
        },
        porosity=0.0,
        description=(
            "Dense lunar basalt as found in lava-tube walls and ceilings. "
            "Minimal porosity, higher density than regolith."
        ),
        source="Kiefer et al. (2012), Geophys. Res. Lett.",
    )

    # PEEK (Polyether ether ketone) - C19H12O3 repeating unit
    # Molecular weight of repeat unit: 288.3 g/mol
    # C: 19*12/288.3 = 0.791, H: 12*1/288.3 = 0.042, O: 3*16/288.3 = 0.167
    presets["peek"] = Material(
        name="PEEK",
        density=1.30,
        composition={
            "C": 0.791,
            "H": 0.042,  # stored as part of composition; H not in ELEMENTS yet
            "O": 0.167,
        },
        porosity=0.0,
        description=(
            "Polyether ether ketone, a semi-crystalline thermoplastic. "
            "High radiation resistance, good mechanical properties."
        ),
        source="Victrex PEEK datasheet; Sato et al. (2011)",
    )

    # Regolith-PEEK Composite (70 wt% regolith, 30 wt% PEEK)
    # Using highland regolith as default base
    highland = presets["highland_regolith"]
    peek = presets["peek"]

    # Compute blended composition
    composite_comp: dict[str, float] = {}
    regolith_wf = 0.70
    peek_wf = 0.30
    for elem, wf in highland.composition.items():
        composite_comp[elem] = composite_comp.get(elem, 0.0) + regolith_wf * wf
    for elem, wf in peek.composition.items():
        composite_comp[elem] = composite_comp.get(elem, 0.0) + peek_wf * wf

    presets["regolith_peek_composite"] = Material(
        name="Regolith-PEEK Composite (70/30 wt%)",
        density=1.85,  # estimated from mixture rule
        composition=composite_comp,
        porosity=0.0,
        description=(
            "Composite of 70 wt% lunar highland regolith and 30 wt% PEEK. "
            "Designed for dual-function structural and radiation shielding."
        ),
        source="Research composite; density estimated via rule of mixtures",
    )

    # Aluminum (reference material for validation)
    presets["aluminum"] = Material(
        name="Aluminum (reference)",
        density=2.70,
        composition={"Al": 1.0},
        porosity=0.0,
        description="Pure aluminum, standard reference for spacecraft shielding.",
        source="NIST Standard Reference Data",
    )

    return presets


# Add H to ELEMENTS since PEEK needs it
ELEMENTS["H"] = Element("H", 1, 1.008, 19.2)
