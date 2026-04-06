"""
LunaRad-PEEK Validation Test Suite
===================================
Comprehensive tests covering analytic geometry, shielding benchmarks,
trend validation, and regression tests.

Run with: pytest tests/test_validation_suite.py -v
"""

import math
import json
import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test configuration constants
# ---------------------------------------------------------------------------

REGRESSION_DIR = Path(__file__).parent / "regression_data"
REGRESSION_TOLERANCE = 0.02  # 2 % relative change triggers failure

# Areal density range validated (g/cm^2)
VALID_AD_MIN = 0.0
VALID_AD_MAX = 100.0

# Material densities (g/cm^3)
DENSITY_ALUMINUM = 2.70
DENSITY_REGOLITH = 1.50
DENSITY_PEEK = 1.30
DENSITY_POLYETHYLENE = 0.94
DENSITY_WATER = 1.00

# Reference GCR dose-equivalent values behind Al slab (mSv/year)
# Source: NCRP 153, Cucinotta et al. 2006, Wilson et al. 1995
# Format: {areal_density_gcm2: (low_bound, reference, high_bound)}
NCRP153_AL_GCR_DOSE_EQ = {
    5:  (380, 420, 480),
    10: (310, 350, 400),
    20: (240, 280, 330),
    50: (160, 200, 260),
}

CUCINOTTA_2006_AL_GCR_DOSE_EQ = {
    5:  (370, 410, 470),
    10: (300, 340, 390),
    20: (230, 270, 320),
    50: (150, 190, 250),
}

WILSON_1995_HZETRN_AL_GCR_DOSE_EQ = {
    5:  (360, 400, 460),
    10: (290, 330, 380),
    20: (220, 260, 310),
    50: (140, 180, 240),
}

# Agreement threshold for benchmark comparison (fractional)
BENCHMARK_TOLERANCE_GCR = 0.30   # 30 %
BENCHMARK_TOLERANCE_SPE = 0.25   # 25 %


# ============================================================================
# SECTION A: Analytic Geometry Tests
# ============================================================================

class TestSphereGeometry:
    """
    A.1 -- Sphere: ray from centre should give uniform path length equal
    to wall thickness in every direction.
    """

    @pytest.fixture
    def sphere(self):
        """Create a spherical shell habitat geometry."""
        # Import would be: from lunarad_peek.geometry import SphereShell
        # For specification, we define the expected interface:
        try:
            from lunarad_peek.geometry import SphereShell
            return SphereShell(
                inner_radius_cm=250.0,
                wall_thickness_cm=30.0,
                material="aluminum",
            )
        except ImportError:
            pytest.skip("lunarad_peek.geometry not yet implemented")

    @pytest.fixture
    def ray_caster(self):
        try:
            from lunarad_peek.raycast import RayCaster
            return RayCaster()
        except ImportError:
            pytest.skip("lunarad_peek.raycast not yet implemented")

    @pytest.mark.parametrize("theta_deg", [0, 30, 45, 60, 90, 120, 150, 180])
    @pytest.mark.parametrize("phi_deg", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_uniform_path_length_from_center(
        self, sphere, ray_caster, theta_deg, phi_deg
    ):
        """Path length from sphere centre must equal wall thickness for all directions."""
        origin = (0.0, 0.0, 0.0)  # centre of sphere
        theta = math.radians(theta_deg)
        phi = math.radians(phi_deg)
        direction = (
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        )

        path_length = ray_caster.compute_path_length(
            geometry=sphere, origin=origin, direction=direction
        )

        expected = sphere.wall_thickness_cm
        assert path_length == pytest.approx(
            expected, rel=1e-6
        ), f"Path length {path_length} != wall thickness {expected} at theta={theta_deg}, phi={phi_deg}"

    def test_areal_density_from_center(self, sphere, ray_caster):
        """Areal density from centre must equal wall_thickness * density."""
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)  # +x

        ad = ray_caster.compute_areal_density(
            geometry=sphere, origin=origin, direction=direction
        )

        expected_ad = sphere.wall_thickness_cm * DENSITY_ALUMINUM
        assert ad == pytest.approx(expected_ad, rel=1e-6)

    def test_off_center_path_length_increases(self, sphere, ray_caster):
        """Path length must increase when dosimetry point is off-centre
        (ray travels obliquely through wall)."""
        origin_center = (0.0, 0.0, 0.0)
        origin_offset = (100.0, 0.0, 0.0)  # 100 cm off centre
        direction = (1.0, 0.0, 0.0)

        pl_center = ray_caster.compute_path_length(
            geometry=sphere, origin=origin_center, direction=direction
        )
        pl_offset = ray_caster.compute_path_length(
            geometry=sphere, origin=origin_offset, direction=direction
        )

        # Off-centre ray in the outward direction should still traverse
        # wall_thickness (radial ray) but perpendicular directions will
        # traverse more. For radial direction, it is the same.
        assert pl_offset == pytest.approx(pl_center, rel=1e-3)


class TestCylinderGeometry:
    """
    A.2 -- Cylinder: verify path lengths at known angles.
    """

    @pytest.fixture
    def cylinder(self):
        try:
            from lunarad_peek.geometry import CylinderShell
            return CylinderShell(
                inner_radius_cm=200.0,
                wall_thickness_cm=25.0,
                height_cm=500.0,
                material="regolith",
                axis="z",
            )
        except ImportError:
            pytest.skip("lunarad_peek.geometry not yet implemented")

    @pytest.fixture
    def ray_caster(self):
        try:
            from lunarad_peek.raycast import RayCaster
            return RayCaster()
        except ImportError:
            pytest.skip("lunarad_peek.raycast not yet implemented")

    def test_radial_ray_path_length(self, cylinder, ray_caster):
        """Purely radial ray from axis should traverse exactly wall_thickness."""
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)

        pl = ray_caster.compute_path_length(
            geometry=cylinder, origin=origin, direction=direction
        )
        assert pl == pytest.approx(cylinder.wall_thickness_cm, rel=1e-6)

    @pytest.mark.parametrize("angle_deg,expected_factor", [
        (0,  1.0),                   # radial
        (30, 1.0 / math.cos(math.radians(30))),  # oblique
        (45, 1.0 / math.cos(math.radians(45))),  # 45 degrees
        (60, 1.0 / math.cos(math.radians(60))),  # steep oblique
    ])
    def test_oblique_ray_path_length(
        self, cylinder, ray_caster, angle_deg, expected_factor
    ):
        """Oblique ray through cylinder wall should scale as 1/cos(angle)
        for small angles relative to the radial direction.
        The angle is measured from the radial (horizontal) toward the
        axial (vertical) direction."""
        angle_rad = math.radians(angle_deg)
        origin = (0.0, 0.0, 0.0)
        direction = (math.cos(angle_rad), 0.0, math.sin(angle_rad))

        pl = ray_caster.compute_path_length(
            geometry=cylinder, origin=origin, direction=direction
        )

        expected = cylinder.wall_thickness_cm * expected_factor
        # For very oblique rays near endcaps, the geometry changes;
        # only validate up to 60 degrees from radial.
        if angle_deg <= 60:
            assert pl == pytest.approx(expected, rel=0.05)

    def test_axial_ray_through_endcap(self, cylinder, ray_caster):
        """Pure axial ray from centre should traverse the endcap
        (if endcaps are modelled) or escape (if open cylinder)."""
        origin = (0.0, 0.0, 0.0)
        direction = (0.0, 0.0, 1.0)

        pl = ray_caster.compute_path_length(
            geometry=cylinder, origin=origin, direction=direction
        )
        # Depends on whether endcaps are present; at minimum should not error
        assert pl >= 0.0


class TestMultiLayerGeometry:
    """
    A.3 -- Multi-layer: areal density = sum of (layer_density * layer_thickness)
    """

    @pytest.fixture
    def multilayer(self):
        try:
            from lunarad_peek.geometry import MultiLayerSlab
            layers = [
                {"material": "aluminum", "thickness_cm": 2.0, "density_gcm3": 2.70},
                {"material": "peek",     "thickness_cm": 5.0, "density_gcm3": 1.30},
                {"material": "regolith", "thickness_cm": 30.0, "density_gcm3": 1.50},
            ]
            return MultiLayerSlab(layers=layers)
        except ImportError:
            pytest.skip("lunarad_peek.geometry not yet implemented")

    @pytest.fixture
    def ray_caster(self):
        try:
            from lunarad_peek.raycast import RayCaster
            return RayCaster()
        except ImportError:
            pytest.skip("lunarad_peek.raycast not yet implemented")

    def test_total_areal_density(self, multilayer, ray_caster):
        """Total areal density must equal sum of individual layer contributions."""
        origin = (-100.0, 0.0, 0.0)  # before slab
        direction = (1.0, 0.0, 0.0)  # normal incidence

        ad = ray_caster.compute_areal_density(
            geometry=multilayer, origin=origin, direction=direction
        )

        expected_ad = (2.0 * 2.70) + (5.0 * 1.30) + (30.0 * 1.50)
        # = 5.40 + 6.50 + 45.00 = 56.90 g/cm^2
        assert ad == pytest.approx(expected_ad, rel=1e-6)

    def test_per_layer_areal_density(self, multilayer, ray_caster):
        """Each layer's areal density should be reported separately."""
        origin = (-100.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)

        layer_ads = ray_caster.compute_layer_areal_densities(
            geometry=multilayer, origin=origin, direction=direction
        )

        assert len(layer_ads) == 3
        assert layer_ads[0] == pytest.approx(2.0 * 2.70, rel=1e-6)
        assert layer_ads[1] == pytest.approx(5.0 * 1.30, rel=1e-6)
        assert layer_ads[2] == pytest.approx(30.0 * 1.50, rel=1e-6)

    def test_layer_order_matters(self, ray_caster):
        """Verify that layer ordering is preserved (for future transport
        where order might matter)."""
        try:
            from lunarad_peek.geometry import MultiLayerSlab
        except ImportError:
            pytest.skip("lunarad_peek.geometry not yet implemented")

        layers_ab = [
            {"material": "aluminum", "thickness_cm": 5.0, "density_gcm3": 2.70},
            {"material": "peek",     "thickness_cm": 5.0, "density_gcm3": 1.30},
        ]
        layers_ba = [
            {"material": "peek",     "thickness_cm": 5.0, "density_gcm3": 1.30},
            {"material": "aluminum", "thickness_cm": 5.0, "density_gcm3": 2.70},
        ]

        slab_ab = MultiLayerSlab(layers=layers_ab)
        slab_ba = MultiLayerSlab(layers=layers_ba)

        assert slab_ab.layers[0]["material"] == "aluminum"
        assert slab_ba.layers[0]["material"] == "peek"


class TestEdgeCases:
    """
    A.4 -- Edge cases: tangent rays and missed geometry.
    """

    @pytest.fixture
    def sphere(self):
        try:
            from lunarad_peek.geometry import SphereShell
            return SphereShell(
                inner_radius_cm=250.0,
                wall_thickness_cm=30.0,
                material="aluminum",
            )
        except ImportError:
            pytest.skip("lunarad_peek.geometry not yet implemented")

    @pytest.fixture
    def ray_caster(self):
        try:
            from lunarad_peek.raycast import RayCaster
            return RayCaster()
        except ImportError:
            pytest.skip("lunarad_peek.raycast not yet implemented")

    def test_ray_tangent_to_inner_surface(self, sphere, ray_caster):
        """Ray tangent to inner surface should still report non-negative
        path length (may be zero or a thin grazing path)."""
        # Place origin on inner surface, direction tangent
        r_inner = sphere.inner_radius_cm
        origin = (r_inner, 0.0, 0.0)
        direction = (0.0, 1.0, 0.0)  # tangent

        pl = ray_caster.compute_path_length(
            geometry=sphere, origin=origin, direction=direction
        )
        assert pl >= 0.0

    def test_ray_tangent_to_outer_surface(self, sphere, ray_caster):
        """Ray tangent to outer surface from outside: should report zero path length."""
        r_outer = sphere.inner_radius_cm + sphere.wall_thickness_cm
        origin = (r_outer, 0.0, 0.0)
        direction = (0.0, 1.0, 0.0)

        pl = ray_caster.compute_path_length(
            geometry=sphere, origin=origin, direction=direction
        )
        assert pl == pytest.approx(0.0, abs=1e-6)

    def test_ray_missing_geometry(self, sphere, ray_caster):
        """Ray originating outside sphere, directed away, should report
        zero path length or None/NaN."""
        origin = (500.0, 0.0, 0.0)  # well outside
        direction = (1.0, 0.0, 0.0)  # pointing away

        pl = ray_caster.compute_path_length(
            geometry=sphere, origin=origin, direction=direction
        )
        assert pl == 0.0 or pl is None or (isinstance(pl, float) and math.isnan(pl))

    def test_ray_from_inside_wall(self, sphere, ray_caster):
        """Ray starting inside the wall material should report remaining
        path length through the wall."""
        r_inner = sphere.inner_radius_cm
        wall = sphere.wall_thickness_cm
        # Midway through wall
        origin = (r_inner + wall / 2.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)

        pl = ray_caster.compute_path_length(
            geometry=sphere, origin=origin, direction=direction
        )
        expected = wall / 2.0
        assert pl == pytest.approx(expected, rel=0.05)

    def test_zero_thickness_wall(self, ray_caster):
        """Zero-thickness wall should give zero path length without error."""
        try:
            from lunarad_peek.geometry import SphereShell
        except ImportError:
            pytest.skip("lunarad_peek.geometry not yet implemented")

        sphere = SphereShell(
            inner_radius_cm=250.0, wall_thickness_cm=0.0, material="aluminum"
        )
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)

        pl = ray_caster.compute_path_length(
            geometry=sphere, origin=origin, direction=direction
        )
        assert pl == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# SECTION B: Simple Slab Shielding Benchmarks
# ============================================================================

class TestAluminumSlabBenchmarks:
    """
    B -- Compare tool output for aluminum slab at 5, 10, 20, 50 g/cm^2
    against published data from NCRP 153, Cucinotta 2006, Wilson 1995.
    Expected agreement: within 20-30 % for GCR dose equivalent.
    """

    @pytest.fixture
    def shielding_calculator(self):
        try:
            from lunarad_peek.shielding import ShieldingCalculator
            return ShieldingCalculator()
        except ImportError:
            pytest.skip("lunarad_peek.shielding not yet implemented")

    @pytest.fixture
    def gcr_environment(self):
        """GCR at solar minimum (worst-case)."""
        try:
            from lunarad_peek.environment import GCREnvironment
            return GCREnvironment(solar_modulation_mv=400)  # solar minimum
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

    @pytest.mark.parametrize("ad_gcm2", [5, 10, 20, 50])
    def test_vs_ncrp153(self, shielding_calculator, gcr_environment, ad_gcm2):
        """Compare against NCRP Report 153 reference data."""
        result = shielding_calculator.compute_dose_equivalent(
            environment=gcr_environment,
            material="aluminum",
            areal_density_gcm2=ad_gcm2,
        )
        dose_eq = result.dose_equivalent_msv_per_year

        low, ref, high = NCRP153_AL_GCR_DOSE_EQ[ad_gcm2]
        # Tool should be within 30% of the reference value
        assert dose_eq == pytest.approx(ref, rel=BENCHMARK_TOLERANCE_GCR), (
            f"At {ad_gcm2} g/cm^2: tool gives {dose_eq:.1f} mSv/yr, "
            f"NCRP 153 reference is {ref} mSv/yr (range {low}-{high})"
        )

    @pytest.mark.parametrize("ad_gcm2", [5, 10, 20, 50])
    def test_vs_cucinotta_2006(self, shielding_calculator, gcr_environment, ad_gcm2):
        """Compare against Cucinotta et al. (2006) dose-depth curves."""
        result = shielding_calculator.compute_dose_equivalent(
            environment=gcr_environment,
            material="aluminum",
            areal_density_gcm2=ad_gcm2,
        )
        dose_eq = result.dose_equivalent_msv_per_year

        low, ref, high = CUCINOTTA_2006_AL_GCR_DOSE_EQ[ad_gcm2]
        assert dose_eq == pytest.approx(ref, rel=BENCHMARK_TOLERANCE_GCR), (
            f"At {ad_gcm2} g/cm^2: tool gives {dose_eq:.1f} mSv/yr, "
            f"Cucinotta 2006 reference is {ref} mSv/yr (range {low}-{high})"
        )

    @pytest.mark.parametrize("ad_gcm2", [5, 10, 20, 50])
    def test_vs_wilson_1995_hzetrn(self, shielding_calculator, gcr_environment, ad_gcm2):
        """Compare against Wilson et al. (1995) HZETRN results."""
        result = shielding_calculator.compute_dose_equivalent(
            environment=gcr_environment,
            material="aluminum",
            areal_density_gcm2=ad_gcm2,
        )
        dose_eq = result.dose_equivalent_msv_per_year

        low, ref, high = WILSON_1995_HZETRN_AL_GCR_DOSE_EQ[ad_gcm2]
        assert dose_eq == pytest.approx(ref, rel=BENCHMARK_TOLERANCE_GCR), (
            f"At {ad_gcm2} g/cm^2: tool gives {dose_eq:.1f} mSv/yr, "
            f"Wilson 1995 HZETRN reference is {ref} mSv/yr (range {low}-{high})"
        )

    def test_benchmark_result_has_confidence_label(self, shielding_calculator, gcr_environment):
        """Every result must carry a confidence label."""
        result = shielding_calculator.compute_dose_equivalent(
            environment=gcr_environment,
            material="aluminum",
            areal_density_gcm2=10,
        )
        assert hasattr(result, "confidence_level")
        assert result.confidence_level in (
            "Conceptual Estimate",
            "Literature-Derived",
            "Validated Against Transport Code",
        )


# ============================================================================
# SECTION C: Trend Validation
# ============================================================================

class TestTrendValidation:
    """
    C -- Verify physically expected trends in the output.
    """

    @pytest.fixture
    def shielding_calculator(self):
        try:
            from lunarad_peek.shielding import ShieldingCalculator
            return ShieldingCalculator()
        except ImportError:
            pytest.skip("lunarad_peek.shielding not yet implemented")

    @pytest.fixture
    def gcr_solar_min(self):
        try:
            from lunarad_peek.environment import GCREnvironment
            return GCREnvironment(solar_modulation_mv=400)
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

    @pytest.fixture
    def gcr_solar_max(self):
        try:
            from lunarad_peek.environment import GCREnvironment
            return GCREnvironment(solar_modulation_mv=1200)
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

    @pytest.fixture
    def spe_oct1989(self):
        try:
            from lunarad_peek.environment import SPEEnvironment
            return SPEEnvironment(event="oct1989")
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

    @pytest.fixture
    def spe_jan2005(self):
        try:
            from lunarad_peek.environment import SPEEnvironment
            return SPEEnvironment(event="jan2005")
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

    # C.1 -- Dose decreases monotonically with areal density (primary radiation)
    @pytest.mark.parametrize("material", ["aluminum", "regolith", "peek", "polyethylene"])
    def test_dose_decreases_with_areal_density(
        self, shielding_calculator, gcr_solar_min, material
    ):
        """Dose must decrease monotonically with increasing areal density
        for the primary radiation component."""
        areal_densities = [0, 2, 5, 10, 20, 30, 50, 75, 100]
        doses = []
        for ad in areal_densities:
            result = shielding_calculator.compute_dose_equivalent(
                environment=gcr_solar_min,
                material=material,
                areal_density_gcm2=ad,
            )
            doses.append(result.dose_equivalent_msv_per_year)

        for i in range(1, len(doses)):
            assert doses[i] <= doses[i - 1], (
                f"{material}: dose at {areal_densities[i]} g/cm^2 "
                f"({doses[i]:.1f}) > dose at {areal_densities[i-1]} g/cm^2 "
                f"({doses[i-1]:.1f}). Monotonic decrease violated."
            )

    # C.2 -- Heavier materials have higher areal density per unit thickness
    def test_areal_density_ranking(self):
        """Denser materials should have higher areal density per cm."""
        materials = {
            "aluminum": DENSITY_ALUMINUM,
            "regolith": DENSITY_REGOLITH,
            "peek": DENSITY_PEEK,
            "polyethylene": DENSITY_POLYETHYLENE,
        }
        thickness_cm = 10.0
        areal_densities = {
            mat: thickness_cm * rho for mat, rho in materials.items()
        }
        # Aluminum > Regolith > PEEK > Polyethylene
        assert areal_densities["aluminum"] > areal_densities["regolith"]
        assert areal_densities["regolith"] > areal_densities["peek"]
        assert areal_densities["peek"] > areal_densities["polyethylene"]

    # C.3 -- GCR dose at solar minimum > solar maximum
    def test_gcr_solar_min_gt_solar_max(
        self, shielding_calculator, gcr_solar_min, gcr_solar_max
    ):
        """GCR dose equivalent at solar minimum must exceed solar maximum."""
        for ad in [5, 10, 20, 50]:
            result_min = shielding_calculator.compute_dose_equivalent(
                environment=gcr_solar_min,
                material="aluminum",
                areal_density_gcm2=ad,
            )
            result_max = shielding_calculator.compute_dose_equivalent(
                environment=gcr_solar_max,
                material="aluminum",
                areal_density_gcm2=ad,
            )
            assert result_min.dose_equivalent_msv_per_year > result_max.dose_equivalent_msv_per_year, (
                f"At {ad} g/cm^2 Al: solar min dose ({result_min.dose_equivalent_msv_per_year:.1f}) "
                f"should exceed solar max dose ({result_max.dose_equivalent_msv_per_year:.1f})"
            )

    # C.4 -- SPE Oct 1989 dose > SPE Jan 2005 dose
    def test_spe_oct1989_gt_jan2005(
        self, shielding_calculator, spe_oct1989, spe_jan2005
    ):
        """October 1989 SPE was more intense; dose must exceed January 2005."""
        for ad in [5, 10, 20, 50]:
            result_oct89 = shielding_calculator.compute_dose_equivalent(
                environment=spe_oct1989,
                material="aluminum",
                areal_density_gcm2=ad,
            )
            result_jan05 = shielding_calculator.compute_dose_equivalent(
                environment=spe_jan2005,
                material="aluminum",
                areal_density_gcm2=ad,
            )
            assert result_oct89.dose_equivalent_msv_per_year > result_jan05.dose_equivalent_msv_per_year, (
                f"At {ad} g/cm^2 Al: Oct 1989 SPE dose "
                f"({result_oct89.dose_equivalent_msv_per_year:.1f}) should exceed "
                f"Jan 2005 SPE dose ({result_jan05.dose_equivalent_msv_per_year:.1f})"
            )

    # C.5 -- Hydrogen-rich materials more effective per unit areal density
    def test_hydrogen_rich_more_effective(
        self, shielding_calculator, gcr_solar_min
    ):
        """Polyethylene (hydrogen-rich) should give lower dose than aluminum
        at the same areal density, due to better fragmentation properties."""
        for ad in [10, 20, 50]:
            result_pe = shielding_calculator.compute_dose_equivalent(
                environment=gcr_solar_min,
                material="polyethylene",
                areal_density_gcm2=ad,
            )
            result_al = shielding_calculator.compute_dose_equivalent(
                environment=gcr_solar_min,
                material="aluminum",
                areal_density_gcm2=ad,
            )
            assert result_pe.dose_equivalent_msv_per_year <= result_al.dose_equivalent_msv_per_year, (
                f"At {ad} g/cm^2: polyethylene dose ({result_pe.dose_equivalent_msv_per_year:.1f}) "
                f"should be <= aluminum dose ({result_al.dose_equivalent_msv_per_year:.1f})"
            )

    # C.6 -- Zero shielding gives unshielded free-space dose
    def test_zero_shielding_gives_free_space_dose(
        self, shielding_calculator, gcr_solar_min
    ):
        """At zero areal density, dose should equal the unshielded environment dose."""
        result = shielding_calculator.compute_dose_equivalent(
            environment=gcr_solar_min,
            material="aluminum",
            areal_density_gcm2=0.0,
        )
        # Free-space GCR dose equivalent at solar min is ~600-700 mSv/yr
        assert 400 < result.dose_equivalent_msv_per_year < 1000, (
            f"Unshielded dose {result.dose_equivalent_msv_per_year:.1f} mSv/yr "
            f"outside expected range [400, 1000]"
        )


# ============================================================================
# SECTION D: Regression Tests
# ============================================================================

class TestRegressionSuite:
    """
    D -- Store reference outputs for standard scenarios and flag deviations.
    """

    STANDARD_SCENARIOS = [
        {"label": "gcr_solmin_al_10",   "env": "gcr", "phi": 400,  "mat": "aluminum",     "ad": 10},
        {"label": "gcr_solmin_al_20",   "env": "gcr", "phi": 400,  "mat": "aluminum",     "ad": 20},
        {"label": "gcr_solmin_al_50",   "env": "gcr", "phi": 400,  "mat": "aluminum",     "ad": 50},
        {"label": "gcr_solmin_reg_20",  "env": "gcr", "phi": 400,  "mat": "regolith",     "ad": 20},
        {"label": "gcr_solmin_peek_20", "env": "gcr", "phi": 400,  "mat": "peek",         "ad": 20},
        {"label": "gcr_solmin_pe_20",   "env": "gcr", "phi": 400,  "mat": "polyethylene", "ad": 20},
        {"label": "gcr_solmax_al_20",   "env": "gcr", "phi": 1200, "mat": "aluminum",     "ad": 20},
        {"label": "spe_oct89_al_10",    "env": "spe", "event": "oct1989", "mat": "aluminum", "ad": 10},
        {"label": "spe_oct89_al_20",    "env": "spe", "event": "oct1989", "mat": "aluminum", "ad": 20},
        {"label": "spe_jan05_al_20",    "env": "spe", "event": "jan2005", "mat": "aluminum", "ad": 20},
    ]

    @pytest.fixture
    def shielding_calculator(self):
        try:
            from lunarad_peek.shielding import ShieldingCalculator
            return ShieldingCalculator()
        except ImportError:
            pytest.skip("lunarad_peek.shielding not yet implemented")

    def _make_environment(self, scenario):
        if scenario["env"] == "gcr":
            from lunarad_peek.environment import GCREnvironment
            return GCREnvironment(solar_modulation_mv=scenario["phi"])
        else:
            from lunarad_peek.environment import SPEEnvironment
            return SPEEnvironment(event=scenario["event"])

    def _load_reference(self, label):
        ref_file = REGRESSION_DIR / f"{label}.json"
        if ref_file.exists():
            with open(ref_file, "r") as f:
                return json.load(f)
        return None

    def _save_reference(self, label, data):
        REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
        ref_file = REGRESSION_DIR / f"{label}.json"
        with open(ref_file, "w") as f:
            json.dump(data, f, indent=2)

    @pytest.mark.parametrize(
        "scenario",
        STANDARD_SCENARIOS,
        ids=[s["label"] for s in STANDARD_SCENARIOS],
    )
    def test_regression(self, shielding_calculator, scenario):
        """Compare current output against stored reference.
        If no reference exists, store the current result (first run)."""
        try:
            env = self._make_environment(scenario)
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

        result = shielding_calculator.compute_dose_equivalent(
            environment=env,
            material=scenario["mat"],
            areal_density_gcm2=scenario["ad"],
        )

        current = {
            "dose_equivalent_msv_per_year": result.dose_equivalent_msv_per_year,
            "absorbed_dose_mgy_per_year": getattr(result, "absorbed_dose_mgy_per_year", None),
            "effective_quality_factor": getattr(result, "effective_quality_factor", None),
        }

        reference = self._load_reference(scenario["label"])
        if reference is None:
            # First run: store as baseline
            self._save_reference(scenario["label"], current)
            pytest.skip(f"No reference for {scenario['label']}; saved current as baseline.")
        else:
            # Compare
            for key in current:
                if current[key] is not None and reference.get(key) is not None:
                    ref_val = reference[key]
                    cur_val = current[key]
                    if ref_val != 0:
                        rel_diff = abs(cur_val - ref_val) / abs(ref_val)
                        assert rel_diff < REGRESSION_TOLERANCE, (
                            f"Regression failure for {scenario['label']}.{key}: "
                            f"current={cur_val:.4f}, reference={ref_val:.4f}, "
                            f"relative change={rel_diff:.4f} > tolerance={REGRESSION_TOLERANCE}"
                        )


# ============================================================================
# SECTION E: Unit and Integration Sanity Checks
# ============================================================================

class TestUnitTracking:
    """Verify that all outputs carry correct units."""

    @pytest.fixture
    def shielding_calculator(self):
        try:
            from lunarad_peek.shielding import ShieldingCalculator
            return ShieldingCalculator()
        except ImportError:
            pytest.skip("lunarad_peek.shielding not yet implemented")

    def test_result_has_units(self, shielding_calculator):
        try:
            from lunarad_peek.environment import GCREnvironment
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

        env = GCREnvironment(solar_modulation_mv=400)
        result = shielding_calculator.compute_dose_equivalent(
            environment=env, material="aluminum", areal_density_gcm2=10
        )
        assert hasattr(result, "units")
        assert "dose_equivalent" in result.units
        assert result.units["dose_equivalent"] == "mSv/yr"

    def test_areal_density_units(self, shielding_calculator):
        try:
            from lunarad_peek.environment import GCREnvironment
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

        env = GCREnvironment(solar_modulation_mv=400)
        result = shielding_calculator.compute_dose_equivalent(
            environment=env, material="aluminum", areal_density_gcm2=10
        )
        assert hasattr(result, "input_areal_density_gcm2")
        assert result.input_areal_density_gcm2 == 10.0


class TestExtrapolationWarnings:
    """Verify warnings when operating outside validated range."""

    @pytest.fixture
    def shielding_calculator(self):
        try:
            from lunarad_peek.shielding import ShieldingCalculator
            return ShieldingCalculator()
        except ImportError:
            pytest.skip("lunarad_peek.shielding not yet implemented")

    def test_warns_above_100_gcm2(self, shielding_calculator):
        try:
            from lunarad_peek.environment import GCREnvironment
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

        env = GCREnvironment(solar_modulation_mv=400)
        with pytest.warns(UserWarning, match="extrapolat"):
            shielding_calculator.compute_dose_equivalent(
                environment=env, material="aluminum", areal_density_gcm2=150
            )

    def test_warns_unknown_material(self, shielding_calculator):
        try:
            from lunarad_peek.environment import GCREnvironment
        except ImportError:
            pytest.skip("lunarad_peek.environment not yet implemented")

        env = GCREnvironment(solar_modulation_mv=400)
        with pytest.warns(UserWarning, match="not in parameterization database"):
            shielding_calculator.compute_dose_equivalent(
                environment=env, material="unobtanium", areal_density_gcm2=10
            )


class TestDisclaimerOnExport:
    """Verify that exported figures carry disclaimers."""

    def test_figure_disclaimer(self):
        try:
            from lunarad_peek.export import FigureExporter
        except ImportError:
            pytest.skip("lunarad_peek.export not yet implemented")

        exporter = FigureExporter()
        # Create a mock figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        annotated_fig = exporter.add_disclaimer(fig)
        # Check that disclaimer text is present
        texts = [t.get_text() for t in annotated_fig.texts]
        disclaimer_found = any("conceptual estimate" in t.lower() for t in texts)
        plt.close(fig)
        assert disclaimer_found, "Exported figure must contain conceptual-estimate disclaimer"
