#!/usr/bin/env python3
"""Generate the three key paper figures for:

"A Dual-Function Regolith-Based Composite Wall for Lunar Habitats:
 Mechanical Strength and Radiation Attenuation"

Outputs:
  figures/figure1_interior_map.png
  figures/figure2_dose_vs_shielding.png
  figures/figure3_scenario_comparison.png
  figures/directional_map.png (bonus)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

import numpy as np
from lunarad_peek.app.state import AppState
from lunarad_peek.materials.material import create_preset_materials
from lunarad_peek.radiation.environments import (
    GCREnvironment,
    RadiationEnvironmentConfig,
    SPEEnvironment,
    SPE_EVENT_LIBRARY,
    SolarCyclePhase,
)
from lunarad_peek.analysis.engine import AnalysisEngine, compute_dose_vs_thickness
from lunarad_peek.visualization.plots import (
    plot_cross_section_dose_map,
    plot_dose_vs_shielding,
    plot_scenario_comparison,
    plot_directional_shielding_map,
    save_figure,
)

print("=" * 60)
print("LunaRad-PEEK Paper Figure Generator")
print("=" * 60)

# ============================================================================
# Setup
# ============================================================================
mats = create_preset_materials()
env = RadiationEnvironmentConfig(
    gcr=GCREnvironment.solar_minimum(),
    spe=SPEEnvironment(event=SPE_EVENT_LIBRARY["oct_1989"]),
)

# ============================================================================
# Scenario 1: Shell Dome with regolith-PEEK composite wall
# ============================================================================
print("\n--- Scenario 1: Shell Dome (30cm composite wall) ---")
state1 = AppState()
state1.create_dome_habitat(
    inner_radius=5.0,
    dome_height_ratio=1.0,
    wall_layers=[("regolith_peek_composite", 0.30)],
)
# Add astronauts in a ring
import math
for i in range(4):
    angle = 2 * math.pi * i / 4
    state1.add_astronaut(f"Crew-{i+1}", 2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.0)

state1.environment = env
result1 = state1.run_analysis("Dome (surface)", n_directions=162)
s1 = result1.summary()
print(f"  GCR dose eq: {s1['mean_gcr_dose_eq_rate_mSv_yr']:.1f} mSv/yr")
print(f"  SPE dose eq: {s1['mean_spe_dose_eq_mSv']:.1f} mSv")
print(f"  Mean AD: {s1['mean_areal_density_gcm2']:.1f} g/cm²")

# ============================================================================
# Scenario 2: Cylindrical Tunnel with same wall
# ============================================================================
print("\n--- Scenario 2: Cylindrical Tunnel (30cm composite wall) ---")
state2 = AppState()
state2.create_tunnel_habitat(
    inner_radius=3.0,
    length=15.0,
    wall_layers=[("regolith_peek_composite", 0.30)],
)
for i in range(4):
    state2.add_astronaut(f"Crew-{i+1}", -3.0 + 2.0*i, 0.0, 0.0)

state2.environment = env
result2 = state2.run_analysis("Tunnel (surface)", n_directions=162)
s2 = result2.summary()
print(f"  GCR dose eq: {s2['mean_gcr_dose_eq_rate_mSv_yr']:.1f} mSv/yr")
print(f"  SPE dose eq: {s2['mean_spe_dose_eq_mSv']:.1f} mSv")
print(f"  Mean AD: {s2['mean_areal_density_gcm2']:.1f} g/cm²")

# ============================================================================
# Scenario 3: Dome with regolith cover (2m burial)
# ============================================================================
print("\n--- Scenario 3: Dome + 2m Regolith Cover ---")
state3 = AppState()
state3.create_dome_habitat(
    inner_radius=5.0,
    dome_height_ratio=1.0,
    wall_layers=[("regolith_peek_composite", 0.30)],
)
state3.add_regolith_cover(2.0)
for i in range(4):
    angle = 2 * math.pi * i / 4
    state3.add_astronaut(f"Crew-{i+1}", 2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.0)

state3.environment = env
result3 = state3.run_analysis("Dome + 2m regolith", n_directions=162)
s3 = result3.summary()
print(f"  GCR dose eq: {s3['mean_gcr_dose_eq_rate_mSv_yr']:.1f} mSv/yr")
print(f"  SPE dose eq: {s3['mean_spe_dose_eq_mSv']:.1f} mSv")
print(f"  Mean AD: {s3['mean_areal_density_gcm2']:.1f} g/cm²")

# ============================================================================
# Scenario 4: Tunnel with overburden (lava tube)
# ============================================================================
print("\n--- Scenario 4: Tunnel + 5m Lava-tube Overburden ---")
state4 = AppState()
state4.create_tunnel_habitat(
    inner_radius=3.0,
    length=15.0,
    wall_layers=[("regolith_peek_composite", 0.30)],
)
state4.add_overburden(5.0)
for i in range(4):
    state4.add_astronaut(f"Crew-{i+1}", -3.0 + 2.0*i, 0.0, 0.0)

state4.environment = env
result4 = state4.run_analysis("Tunnel + overburden", n_directions=162)
s4 = result4.summary()
print(f"  GCR dose eq: {s4['mean_gcr_dose_eq_rate_mSv_yr']:.1f} mSv/yr")
print(f"  SPE dose eq: {s4['mean_spe_dose_eq_mSv']:.1f} mSv")
print(f"  Mean AD: {s4['mean_areal_density_gcm2']:.1f} g/cm²")

# ============================================================================
# FIGURE 1: Interior Radiation Map
# ============================================================================
print("\n--- Generating Figure 1: Interior Radiation Map ---")
fig1 = plot_cross_section_dose_map(
    result1.point_results,
    slice_axis="y",
    metric="gcr_dose_equivalent_rate",
)
save_figure(fig1, "figures/figure1_interior_map.png")
print("  Saved: figures/figure1_interior_map.png")

# ============================================================================
# FIGURE 2: Performance vs Shielding
# ============================================================================
print("\n--- Generating Figure 2: Dose vs Shielding ---")
materials_for_fig2 = {
    k: mats[k] for k in ["highland_regolith", "peek", "regolith_peek_composite", "aluminum"]
}
fig2 = plot_dose_vs_shielding(
    materials_for_fig2,
    env,
    metric="gcr_dose_eq_mSv_yr",
    max_thickness_cm=200,
)
save_figure(fig2, "figures/figure2_dose_vs_shielding.png")
print("  Saved: figures/figure2_dose_vs_shielding.png")

# ============================================================================
# FIGURE 3: Scenario Comparison
# ============================================================================
print("\n--- Generating Figure 3: Scenario Comparison ---")
all_scenarios = [result1, result2, result3, result4]
fig3 = plot_scenario_comparison(all_scenarios)
save_figure(fig3, "figures/figure3_scenario_comparison.png")
print("  Saved: figures/figure3_scenario_comparison.png")

# ============================================================================
# BONUS: Directional Shielding Map
# ============================================================================
print("\n--- Generating Bonus: Directional Shielding Map ---")
if result1.point_results:
    fig_dir = plot_directional_shielding_map(
        result1.point_results[0],
        metric="areal_density",
    )
    save_figure(fig_dir, "figures/directional_map.png")
    print("  Saved: figures/directional_map.png")

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "=" * 60)
print("Summary of Scenario Results")
print("=" * 60)
print(f"{'Scenario':<25} {'GCR (mSv/yr)':>12} {'SPE (mSv)':>10} {'AD (g/cm²)':>11}")
print("-" * 60)
for result in all_scenarios:
    s = result.summary()
    print(f"{result.scenario_name:<25} "
          f"{s['mean_gcr_dose_eq_rate_mSv_yr']:>12.1f} "
          f"{s['mean_spe_dose_eq_mSv']:>10.1f} "
          f"{s['mean_areal_density_gcm2']:>11.1f}")
print("-" * 60)
print("\nNote: All values are conceptual estimates. See docs/assumptions.md")
print("for detailed methodology and limitations.\n")
print("All figures generated successfully!")
