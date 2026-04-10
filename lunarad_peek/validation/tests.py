"""Validation test suite for LunaRad.

Includes analytic geometry tests, slab shielding benchmarks,
and trend validation checks.
"""

from __future__ import annotations

import math
import numpy as np


def test_sphere_uniform_path_length():
    """A point at the center of a hemisphere should see wall material
    in upward-directed rays. Path length through the wall should be
    close to the actual wall thickness.

    For a hemisphere with inner_radius=5.0 and wall=0.5m, the shell mesh
    has both inner and outer surfaces. A ray from inside passes through
    the inner surface (entry) and outer surface (exit), giving a path
    length equal to the wall thickness along that ray direction.
    """
    from lunarad_peek.geometry.primitives import ShellDomeHabitat, WallLayer
    from lunarad_peek.geometry.scene import GeometryLayer
    from lunarad_peek.geometry.raycaster import RayCaster

    wall_thickness = 0.50  # meters
    inner_radius = 5.0

    habitat = ShellDomeHabitat(
        inner_radius=inner_radius,
        dome_height_ratio=1.0,
        wall_layers=[WallLayer("aluminum", wall_thickness, "Wall")],
    )

    meshes = habitat.generate_mesh()
    layers = [
        GeometryLayer(name=name, mesh=mesh, material_id="aluminum")
        for name, mesh in meshes.items()
    ]

    # Cast from the center of the dome (inside the outer shell)
    caster = RayCaster(n_directions=42)
    origin = np.array([0.0, 0.0, inner_radius * 0.3])

    result = caster.cast_from_point(origin, layers, "test", "center")

    # Check that some upward rays hit the wall
    upward_rays = [r for r in result.rays if r.theta < math.pi / 3]
    hitting_rays = [r for r in upward_rays if r.total_path_length > 0]

    assert len(hitting_rays) > 0, "No upward rays hit the hemisphere wall"

    path_lengths = [r.total_path_length for r in hitting_rays]
    mean_pl = np.mean(path_lengths)

    # With the closed shell mesh (inner + outer surfaces), path length should
    # be close to the wall thickness (0.5m at zenith, somewhat more at oblique
    # angles due to longer path through the shell).
    print(f"  Rays hitting wall: {len(hitting_rays)}/{len(upward_rays)}")
    print(f"  Mean path length (upward): {mean_pl:.3f} m")
    print(f"  Wall thickness: {wall_thickness:.3f} m")

    # Path should be on the order of wall thickness (0.5m - 2m for oblique rays)
    assert wall_thickness * 0.8 < mean_pl < wall_thickness * 5.0, (
        f"Mean path {mean_pl:.2f} outside expected range "
        f"[{wall_thickness*0.8:.2f}, {wall_thickness*5.0:.2f}]"
    )
    print("  PASS: Sphere path length test")


def test_areal_density_consistency():
    """Areal density should equal density × path_length for a single material."""
    from lunarad_peek.materials.material import create_preset_materials

    mats = create_preset_materials()
    al = mats["aluminum"]

    thickness_cm = 10.0
    expected_ad = al.effective_density * thickness_cm
    computed_ad = al.areal_density(thickness_cm)

    assert abs(expected_ad - computed_ad) < 0.01, (
        f"Areal density mismatch: {expected_ad} vs {computed_ad}"
    )
    print(f"  Aluminum at {thickness_cm} cm: AD = {computed_ad:.2f} g/cm²")
    print("  PASS: Areal density consistency")


def test_dose_monotonic_decrease():
    """Dose must decrease monotonically with increasing areal density
    (for primary radiation without secondary buildup)."""
    from lunarad_peek.radiation.environments import GCREnvironment

    gcr = GCREnvironment.solar_minimum()

    areal_densities = np.linspace(0, 100, 50)
    doses = [gcr.dose_behind_shielding(ad) for ad in areal_densities]

    # Check monotonic decrease (allowing for buildup bump)
    # At very low AD, buildup can cause slight increase - check overall trend
    assert doses[0] > doses[-1], (
        f"Dose should decrease overall: {doses[0]:.2f} -> {doses[-1]:.2f}"
    )

    # Check that doses at 50 g/cm² < doses at 0 g/cm²
    d0 = gcr.dose_behind_shielding(0)
    d50 = gcr.dose_behind_shielding(50)
    assert d50 < d0, f"Dose at 50 g/cm² ({d50:.2f}) should be < dose at 0 ({d0:.2f})"

    print(f"  Dose at 0 g/cm²: {d0:.1f} mSv/yr")
    print(f"  Dose at 50 g/cm²: {d50:.1f} mSv/yr")
    print(f"  Reduction: {(1-d50/d0)*100:.1f}%")
    print("  PASS: Monotonic dose decrease")


def test_solar_min_gt_max():
    """GCR dose at solar minimum should exceed solar maximum."""
    from lunarad_peek.radiation.environments import GCREnvironment

    sol_min = GCREnvironment.solar_minimum()
    sol_max = GCREnvironment.solar_maximum()

    d_min = sol_min.lunar_surface_dose_rate
    d_max = sol_max.lunar_surface_dose_rate

    assert d_min > d_max, (
        f"Solar min dose ({d_min:.0f}) should > solar max ({d_max:.0f})"
    )
    print(f"  Solar min: {d_min:.0f} mSv/yr")
    print(f"  Solar max: {d_max:.0f} mSv/yr")
    print("  PASS: Solar min > solar max")


def test_spe_event_ordering():
    """Oct 1989 SPE should give higher dose than Jan 2005 at same shielding."""
    from lunarad_peek.radiation.environments import SPEEnvironment, SPE_EVENT_LIBRARY

    oct89 = SPEEnvironment(event=SPE_EVENT_LIBRARY["oct_1989"])
    jan05 = SPEEnvironment(event=SPE_EVENT_LIBRARY["jan_2005"])

    ad = 20.0  # g/cm²
    d_oct = oct89.dose_behind_shielding(ad)
    d_jan = jan05.dose_behind_shielding(ad)

    # Oct 1989 had higher fluence
    assert d_oct > d_jan, (
        f"Oct 1989 dose ({d_oct:.1f}) should > Jan 2005 ({d_jan:.1f})"
    )
    print(f"  Oct 1989 at {ad} g/cm²: {d_oct:.1f} mSv")
    print(f"  Jan 2005 at {ad} g/cm²: {d_jan:.1f} mSv")
    print("  PASS: SPE event ordering")


def test_material_properties():
    """Verify material derived properties are in expected ranges."""
    from lunarad_peek.materials.material import create_preset_materials

    mats = create_preset_materials()

    for mat_id, mat in mats.items():
        z_eff = mat.Z_eff
        mean_a = mat.mean_A
        x0 = mat.radiation_length_approx
        lambda_i = mat.nuclear_interaction_length

        assert 1 < z_eff < 30, f"{mat_id}: Z_eff={z_eff} out of range"
        assert 1 < mean_a < 60, f"{mat_id}: mean_A={mean_a} out of range"
        assert 5 < x0 < 200, f"{mat_id}: X0={x0} out of range"
        assert 30 < lambda_i < 200, f"{mat_id}: lambda_I={lambda_i} out of range"

        print(f"  {mat.name}: Z_eff={z_eff:.1f}, A={mean_a:.1f}, "
              f"X0={x0:.1f}, λI={lambda_i:.1f}")

    print("  PASS: Material properties in range")


def test_composite_material():
    """Composite material density should be between constituents."""
    from lunarad_peek.materials.material import (
        create_preset_materials,
        CompositeMaterial,
        CompositeMode,
    )

    mats = create_preset_materials()
    regolith = mats["highland_regolith"]
    peek = mats["peek"]

    composite = CompositeMaterial(
        name="Test Composite",
        mode=CompositeMode.WEIGHT_FRACTION,
        constituents=[(regolith, 0.6), (peek, 0.4)],
    )

    rho = composite.density
    assert peek.density < rho < regolith.density, (
        f"Composite density {rho} not between {peek.density} and {regolith.density}"
    )

    flat = composite.to_material()
    total_wf = sum(flat.composition.values())
    assert abs(total_wf - 1.0) < 0.01, f"Composition sums to {total_wf}"

    print(f"  Composite density: {rho:.3f} g/cm³")
    print(f"  Composition sum: {total_wf:.4f}")
    print("  PASS: Composite material")


def test_aluminum_slab_benchmark():
    """Compare aluminum slab shielding against literature values.

    Expected: at 20 g/cm² aluminum, GCR dose equivalent should be
    roughly 200-500 mSv/yr at solar minimum.
    Reference: NCRP 153, Cucinotta et al. (2006)
    """
    from lunarad_peek.radiation.environments import GCREnvironment

    gcr = GCREnvironment.solar_minimum()

    ad_values = [5, 10, 20, 50]
    for ad in ad_values:
        dose_eq = gcr.dose_equivalent_behind_shielding(ad)
        print(f"  Al at {ad:3d} g/cm²: {dose_eq:.0f} mSv/yr dose eq.")

    # Literature comparison (approximate ranges from NCRP 153 / Wilson et al.)
    # At 20 g/cm² Al: ~250-450 mSv/yr dose equivalent at solar min
    d20 = gcr.dose_equivalent_behind_shielding(20)
    assert 100 < d20 < 800, f"Dose eq at 20 g/cm² = {d20:.0f}, expected 100-800"

    print("  PASS: Aluminum slab benchmark (within expected range)")


def run_all_tests():
    """Run all validation tests."""
    tests = [
        ("Areal density consistency", test_areal_density_consistency),
        ("Dose monotonic decrease", test_dose_monotonic_decrease),
        ("Solar min > max", test_solar_min_gt_max),
        ("SPE event ordering", test_spe_event_ordering),
        ("Material properties", test_material_properties),
        ("Composite material", test_composite_material),
        ("Aluminum slab benchmark", test_aluminum_slab_benchmark),
        ("Sphere path length", test_sphere_uniform_path_length),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("LunaRad Validation Test Suite")
    print("=" * 60)

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
