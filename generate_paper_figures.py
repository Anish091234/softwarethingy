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
print("LunaRad Paper Figure Generator")
print("=" * 60)

# ============================================================================
# Setup
# ============================================================================
mats = create_preset_materials()
env = RadiationEnvironmentConfig(
    gcr=GCREnvironment.solar_minimum(),
    spe=SPEEnvironment(event=SPE_EVENT_LIBRARY["oct_1989"]),
)

import math

# Baseline geometry: SolidWorks reference dome, R=5.0 m, t=0.5 m
BASELINE_RADIUS = 5.0
BASELINE_THICKNESS = 0.50

# Run a 50cm dome scenario for each candidate wall material
WALL_MATERIALS = [
    ("aluminum", "Aluminum"),
    ("regolith_peek_composite", "Regolith-PEEK Composite"),
    ("peek", "PEEK"),
    ("highland_regolith", "Highland Regolith"),
]


def run_dome_scenario(material_id: str, label: str):
    state = AppState()
    state.create_dome_habitat(
        inner_radius=BASELINE_RADIUS,
        dome_height_ratio=1.0,
        wall_layers=[(material_id, BASELINE_THICKNESS)],
    )
    state.add_astronaut("Crew-Center", 0.0, 0.0, 0.0)
    for i in range(4):
        angle = 2 * math.pi * i / 4
        state.add_astronaut(
            f"Crew-{i+1}",
            2.0 * math.cos(angle),
            2.0 * math.sin(angle),
            0.0,
        )
    state.environment = env
    name = f"{label} ({BASELINE_THICKNESS*100:.0f}cm)"
    result = state.run_analysis(name, n_directions=162)
    s = result.summary()
    print(f"  {label}:")
    print(f"    GCR dose eq      : {s['mean_gcr_dose_eq_rate_mSv_yr']:.1f} mSv/yr")
    print(f"    SPE dose eq      : {s['mean_spe_dose_eq_mSv']:.1f} mSv")
    print(f"    Combined annual  : {s['combined_annual_dose_mSv_yr']:.1f} mSv/yr"
          f"  [{s['nasa_limit_status']}]")
    print(f"    Mean AD          : {s['mean_areal_density_gcm2']:.1f} g/cm²")
    return result


print(
    f"\n--- Running 4 scenarios: R={BASELINE_RADIUS:.1f} m dome, "
    f"t={BASELINE_THICKNESS*100:.0f} cm walls ---"
)
all_results = []
for mat_id, label in WALL_MATERIALS:
    print(f"\n  Scenario: {label}")
    all_results.append(run_dome_scenario(mat_id, label))

# Reference handles for the figures
result1 = all_results[1]  # Composite as the canonical interior map
result2 = all_results[0]  # Aluminum
result3 = all_results[2]  # PEEK
result4 = all_results[3]  # Regolith

# ============================================================================
# FIGURE 1: Interior Radiation Map (composite baseline scenario)
# ============================================================================
print("\n--- Generating Figure 1: Interior Radiation Map ---")
habitat_info = {
    "type": "ShellDomeHabitat",
    "inner_radius": BASELINE_RADIUS,
    "total_wall_thickness": BASELINE_THICKNESS,
    "position": [0.0, 0.0, 0.0],
    "length": None,
}
fig1 = plot_cross_section_dose_map(
    result1.point_results,
    slice_axis="y",
    metric="gcr_dose_equivalent_rate",
    habitat_info=habitat_info,
    scenario_name=result1.scenario_name,
)
save_figure(fig1, "figures/figure1_interior_map.png")
print("  Saved: figures/figure1_interior_map.png")

# ============================================================================
# FIGURE 2: Combined annual dose vs thickness (NASA limit overlay)
# ============================================================================
print("\n--- Generating Figure 2: Dose vs Shielding ---")
materials_for_fig2 = {
    k: mats[k]
    for k in ["highland_regolith", "peek", "regolith_peek_composite", "aluminum"]
}
fig2 = plot_dose_vs_shielding(
    materials_for_fig2,
    env,
    metric="combined_annual_dose_mSv_yr",
    max_thickness_cm=200,
    scenario_result=result1,
)
save_figure(fig2, "figures/figure2_dose_vs_shielding.png")
print("  Saved: figures/figure2_dose_vs_shielding.png")

# ============================================================================
# FIGURE 3: Scenario Comparison (4 materials at 50cm wall)
# ============================================================================
print("\n--- Generating Figure 3: Scenario Comparison ---")
fig3 = plot_scenario_comparison(all_results)
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
# Summary Table — verbose for paper
# ============================================================================
print("\n" + "=" * 84)
print(
    f"{'Scenario':<32} {'GCR (mSv/yr)':>13} {'SPE (mSv)':>11} "
    f"{'Combined':>11} {'AD (g/cm²)':>12} {'Status':>7}"
)
print("-" * 84)
for result in all_results:
    s = result.summary()
    print(
        f"{result.scenario_name:<32} "
        f"{s['mean_gcr_dose_eq_rate_mSv_yr']:>13.1f} "
        f"{s['mean_spe_dose_eq_mSv']:>11.1f} "
        f"{s['combined_annual_dose_mSv_yr']:>11.1f} "
        f"{s['mean_areal_density_gcm2']:>12.1f} "
        f"{s['nasa_limit_status']:>7}"
    )
print("-" * 84)
print(
    f"\nReference: NASA-STD-3001 Rev B (2022) BFO annual limit = "
    f"500 mSv/yr; career = 600 mSv."
)
print("All values are conceptual estimates. See docs/assumptions.md")
print("for detailed methodology and limitations.\n")
print("All figures generated successfully!")
