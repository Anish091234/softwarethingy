"""2D plotting and figure generation for LunaRad.

Generates publication-quality matplotlib figures for paper support.
All figures include appropriate axis labels, units, and disclaimers.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; overridden when embedded in Qt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from lunarad_peek.analysis.engine import (
    PointResult,
    ScenarioResult,
    compute_dose_vs_thickness,
)
from lunarad_peek.materials.material import Material
from lunarad_peek.radiation.environments import RadiationEnvironmentConfig

# Color palette for materials (Catppuccin-inspired)
MATERIAL_COLORS = {
    "highland_regolith": "#f9e2af",
    "mare_regolith": "#fab387",
    "lavatube_rock": "#a6adc8",
    "peek": "#89b4fa",
    "regolith_peek_composite": "#a6e3a1",
    "aluminum": "#cdd6f4",
}

RESULT_COLORMAP = "viridis"
DISCLAIMER_TEXT = (
    "Conceptual estimate — not Monte Carlo transport.\n"
    "LunaRad v1.0"
)


def _style_axes(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent styling to axes."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_disclaimer(fig: Figure, x: float = 0.99, y: float = 0.01):
    """Add small disclaimer text to figure."""
    fig.text(
        x, y, DISCLAIMER_TEXT,
        fontsize=6, color="gray", alpha=0.7,
        ha="right", va="bottom",
        fontstyle="italic",
    )


# ============================================================================
# Figure 1: Interior Radiation Map (2D cross-section)
# ============================================================================

def plot_cross_section_dose_map(
    point_results: list[PointResult],
    slice_axis: str = "y",
    slice_value: float = 0.0,
    metric: str = "gcr_dose_equivalent_rate",
    habitat_info: dict | None = None,
    scenario_name: str = "",
    figsize: tuple = (10, 8),
) -> Figure:
    """Generate 2D cross-section dose map.

    Shows dose equivalent (or other metric) distribution on a slice plane
    through the habitat with habitat geometry outline and dose annotations.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    if not point_results:
        ax.text(0.5, 0.5, "No results to display", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=14)
        return fig

    # Collect point positions and metric values
    positions = []
    values = []
    names = []
    for pr in point_results:
        pos = pr.position
        val = getattr(pr, f"mean_{metric}", pr.mean_gcr_dose_equivalent_rate)
        positions.append(pos)
        values.append(val)
        names.append(pr.point_name)

    positions = np.array(positions)
    values = np.array(values)

    # Map axes based on slice
    axis_map = {"x": (1, 2, "Y (m)", "Z (m)"),
                "y": (0, 2, "X (m)", "Z (m)"),
                "z": (0, 1, "X (m)", "Y (m)")}
    ax1_idx, ax2_idx, xlabel, ylabel = axis_map.get(
        slice_axis, (0, 2, "X (m)", "Z (m)"))

    # Draw habitat outline first (behind scatter points)
    if habitat_info:
        _draw_habitat_outline(ax, habitat_info, slice_axis, ax1_idx, ax2_idx)

    metric_labels = {
        "gcr_dose_equivalent_rate": "GCR Dose Eq. Rate (mSv/yr)",
        "gcr_dose_rate": "GCR Dose Rate (mSv/yr)",
        "areal_density": "Areal Density (g/cm\u00b2)",
        "spe_dose_equivalent": "SPE Dose Equivalent (mSv)",
    }
    metric_short = {
        "gcr_dose_equivalent_rate": "mSv/yr",
        "gcr_dose_rate": "mSv/yr",
        "areal_density": "g/cm\u00b2",
        "spe_dose_equivalent": "mSv",
    }

    scatter = ax.scatter(
        positions[:, ax1_idx], positions[:, ax2_idx],
        c=values, cmap=RESULT_COLORMAP, s=200,
        edgecolors="white", linewidths=1.0, zorder=5,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(metric_labels.get(metric, metric), fontsize=9, color="#cdd6f4")
    cbar.ax.tick_params(labelsize=8, colors="#cdd6f4")

    # Annotate each point with name and dose value
    unit = metric_short.get(metric, "")
    for i, pr in enumerate(point_results):
        val = values[i]
        label = f"{names[i]}\n{val:.2f} {unit}"
        ax.annotate(
            label,
            (positions[i, ax1_idx], positions[i, ax2_idx]),
            textcoords="offset points", xytext=(10, 8),
            fontsize=8, color="#f5e0dc",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#313244",
                      edgecolor="#585b70", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="#585b70", lw=0.5),
            zorder=6,
        )

    # Title includes scenario name
    title = f"Interior Radiation Map \u2014 {slice_axis.upper()}={slice_value:.1f}m slice"
    if scenario_name:
        title = f"{scenario_name}: {title}"
    _style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal")
    ax.title.set_color("#cdd6f4")
    ax.xaxis.label.set_color("#cdd6f4")
    ax.yaxis.label.set_color("#cdd6f4")
    ax.tick_params(colors="#cdd6f4")

    _add_disclaimer(fig)
    fig.tight_layout()
    return fig


def _draw_habitat_outline(
    ax: plt.Axes,
    habitat_info: dict,
    slice_axis: str,
    h_idx: int,
    v_idx: int,
):
    """Draw habitat wall cross-section outline on the plot."""
    h_type = habitat_info.get("type", "")
    inner_r = habitat_info.get("inner_radius", 5.0)
    wall_t = habitat_info.get("total_wall_thickness", 0.3)
    outer_r = inner_r + wall_t
    pos = np.array(habitat_info.get("position", [0, 0, 0]))
    outline_color = "#6c7086"
    wall_fill = "#313244"

    if "Dome" in h_type:
        if slice_axis in ("x", "y"):
            # Vertical slice through dome: semicircle cross-section
            theta = np.linspace(0, np.pi, 200)
            hi = pos[h_idx] + inner_r * np.cos(theta)
            vi = pos[v_idx] + inner_r * np.sin(theta)
            ho = pos[h_idx] + outer_r * np.cos(theta)
            vo = pos[v_idx] + outer_r * np.sin(theta)
            ax.fill(
                np.concatenate([ho, hi[::-1]]),
                np.concatenate([vo, vi[::-1]]),
                color=wall_fill, alpha=0.35, zorder=1, label="Wall",
            )
            ax.plot(hi, vi, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
            ax.plot(ho, vo, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
            # Floor line
            ax.plot(
                [pos[h_idx] - outer_r, pos[h_idx] + outer_r],
                [pos[v_idx], pos[v_idx]],
                color=outline_color, lw=1, ls="--", alpha=0.4, zorder=2,
            )
        else:
            # Horizontal slice (Z=mid): full circle
            theta = np.linspace(0, 2 * np.pi, 200)
            hi = pos[h_idx] + inner_r * np.cos(theta)
            vi = pos[v_idx] + inner_r * np.sin(theta)
            ho = pos[h_idx] + outer_r * np.cos(theta)
            vo = pos[v_idx] + outer_r * np.sin(theta)
            ax.fill(
                np.concatenate([ho, hi[::-1]]),
                np.concatenate([vo, vi[::-1]]),
                color=wall_fill, alpha=0.35, zorder=1,
            )
            ax.plot(hi, vi, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
            ax.plot(ho, vo, color=outline_color, lw=1.5, alpha=0.8, zorder=2)

    elif "Cylinder" in h_type:
        length = habitat_info.get("length") or 15.0
        half_len = length / 2.0

        if slice_axis == "y":
            # XZ side profile: rectangle + end caps
            cx, cz = pos[0], pos[2]
            # Wall bands (top, bottom, left, right)
            ax.fill_between(
                [cx - half_len, cx + half_len],
                cz + inner_r, cz + outer_r,
                color=wall_fill, alpha=0.35, zorder=1,
            )
            ax.fill_between(
                [cx - half_len, cx + half_len],
                cz - outer_r, cz - inner_r,
                color=wall_fill, alpha=0.35, zorder=1,
            )
            # Inner outline
            rect_h = [cx - half_len, cx + half_len, cx + half_len,
                       cx - half_len, cx - half_len]
            rect_v = [cz + inner_r, cz + inner_r, cz - inner_r,
                       cz - inner_r, cz + inner_r]
            ax.plot(rect_h, rect_v, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
            # Outer outline
            rect_vo = [cz + outer_r, cz + outer_r, cz - outer_r,
                        cz - outer_r, cz + outer_r]
            ax.plot(rect_h, rect_vo, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
        elif slice_axis == "x":
            # YZ circular cross-section
            theta = np.linspace(0, 2 * np.pi, 200)
            yi = pos[1] + inner_r * np.cos(theta)
            zi = pos[2] + inner_r * np.sin(theta)
            yo = pos[1] + outer_r * np.cos(theta)
            zo = pos[2] + outer_r * np.sin(theta)
            ax.fill(
                np.concatenate([yo, yi[::-1]]),
                np.concatenate([zo, zi[::-1]]),
                color=wall_fill, alpha=0.35, zorder=1,
            )
            ax.plot(yi, zi, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
            ax.plot(yo, zo, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
        else:
            # Z-slice: rectangle top-down view
            cx, cy = pos[0], pos[1]
            rect_h = [cx - half_len, cx + half_len, cx + half_len,
                       cx - half_len, cx - half_len]
            rect_vi = [cy + inner_r, cy + inner_r, cy - inner_r,
                        cy - inner_r, cy + inner_r]
            rect_vo = [cy + outer_r, cy + outer_r, cy - outer_r,
                        cy - outer_r, cy + outer_r]
            ax.plot(rect_h, rect_vi, color=outline_color, lw=1.5, alpha=0.8, zorder=2)
            ax.plot(rect_h, rect_vo, color=outline_color, lw=1.5, alpha=0.8, zorder=2)


def plot_directional_shielding_map(
    point_result: PointResult,
    metric: str = "areal_density",
    figsize: tuple = (10, 5),
) -> Figure:
    """Mollweide projection of directional shielding from a single point."""
    fig = plt.figure(figsize=figsize, facecolor="#1e1e2e")
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor("#313244")

    data = point_result.directional_map(metric)
    if len(data) == 0:
        return fig

    # Convert to Mollweide coords: longitude = phi (-pi, pi), latitude = pi/2 - theta
    lon = data[:, 1]  # phi
    lat = np.pi / 2 - data[:, 0]  # convert polar to latitude
    values = data[:, 2]

    scatter = ax.scatter(
        lon, lat, c=values, cmap=RESULT_COLORMAP,
        s=15, alpha=0.9, edgecolors="none",
    )

    cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal",
                        shrink=0.6, pad=0.08)
    metric_labels = {
        "areal_density": "Areal Density (g/cm²)",
        "gcr_dose_rate": "GCR Dose Rate (mSv/yr)",
        "gcr_dose_equivalent_rate": "GCR Dose Eq. Rate (mSv/yr)",
    }
    cbar.set_label(metric_labels.get(metric, metric), fontsize=9, color="#cdd6f4")
    cbar.ax.tick_params(labelsize=8, colors="#cdd6f4")

    ax.set_title(
        f"Directional Shielding — {point_result.target_name}/{point_result.point_name}",
        fontsize=11, color="#cdd6f4", pad=15,
    )
    ax.tick_params(colors="#cdd6f4")
    ax.grid(True, alpha=0.2)

    _add_disclaimer(fig)
    fig.tight_layout()
    return fig


# ============================================================================
# Figure 2: Performance vs Shielding Thickness / Areal Density
# ============================================================================

def plot_dose_vs_shielding(
    materials: dict[str, Material],
    environment: RadiationEnvironmentConfig,
    metric: str = "gcr_dose_eq_mSv_yr",
    max_thickness_cm: float = 200.0,
    scenario_result: "ScenarioResult | None" = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """Dual-panel plot: dose vs physical thickness and dose vs areal density.

    If scenario_result is provided, overlays the actual analysis operating
    point on the curves so the plot reflects the current scenario.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#1e1e2e")
    ax1.set_facecolor("#1e1e2e")
    ax2.set_facecolor("#1e1e2e")

    thicknesses = np.linspace(0, max_thickness_cm, 200)

    for mat_id, material in materials.items():
        color = MATERIAL_COLORS.get(mat_id, "#cdd6f4")
        data = compute_dose_vs_thickness(material, environment, thicknesses)

        y = data[metric]

        # Left: vs thickness
        ax1.semilogy(
            data["thickness_cm"], y,
            label=material.name, color=color, linewidth=2,
        )

        # Right: vs areal density
        ax2.semilogy(
            data["areal_density_gcm2"], y,
            label=material.name, color=color, linewidth=2,
        )

    metric_ylabel = {
        "gcr_dose_mSv_yr": "GCR Dose Rate (mSv/yr)",
        "gcr_dose_eq_mSv_yr": "GCR Dose Equivalent Rate (mSv/yr)",
        "spe_dose_mSv": "SPE Event Dose (mSv)",
        "spe_dose_eq_mSv": "SPE Event Dose Equivalent (mSv)",
    }

    _style_axes(ax1, title="Dose vs Physical Thickness",
                xlabel="Thickness (cm)", ylabel=metric_ylabel.get(metric, metric))
    _style_axes(ax2, title="Dose vs Areal Density",
                xlabel="Areal Density (g/cm²)", ylabel=metric_ylabel.get(metric, metric))

    for ax in (ax1, ax2):
        ax.title.set_color("#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.legend(fontsize=8, loc="upper right", facecolor="#313244",
                  edgecolor="#45475a", labelcolor="#cdd6f4")

    # Add dose limit reference lines
    for ax in (ax1, ax2):
        ax.axhline(y=250, color="#f38ba8", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, 270, "NASA 30-day limit (250 mSv)",
                fontsize=7, color="#f38ba8", ha="right", alpha=0.7)

    # Overlay actual scenario results if available
    if scenario_result and scenario_result.point_results:
        _overlay_scenario_results(ax1, ax2, scenario_result, metric)

    _add_disclaimer(fig)
    fig.tight_layout()
    return fig


def _overlay_scenario_results(
    ax_thickness: plt.Axes,
    ax_areal: plt.Axes,
    scenario_result: "ScenarioResult",
    metric: str,
):
    """Overlay actual analysis results as markers on dose-vs-shielding plots."""
    metric_to_summary = {
        "gcr_dose_mSv_yr": "mean_gcr_dose_rate_mSv_yr",
        "gcr_dose_eq_mSv_yr": "mean_gcr_dose_eq_rate_mSv_yr",
        "spe_dose_mSv": "mean_spe_dose_mSv",
        "spe_dose_eq_mSv": "mean_spe_dose_eq_mSv",
    }
    summary = scenario_result.summary()
    dose_key = metric_to_summary.get(metric)
    mean_dose = summary.get(dose_key, 0.0) if dose_key else 0.0
    mean_ad = summary.get("mean_areal_density_gcm2", 0.0)

    if mean_dose <= 0 or mean_ad <= 0:
        return

    marker_color = "#f5c2e7"  # pink accent

    # Star marker on the areal density panel
    ax_areal.plot(
        mean_ad, mean_dose, "*",
        color=marker_color, markersize=18, zorder=10,
        markeredgecolor="white", markeredgewidth=0.8,
    )

    # Reference lines through the operating point
    for ax in (ax_thickness, ax_areal):
        ax.axhline(
            y=mean_dose, color=marker_color, linestyle=":",
            alpha=0.5, linewidth=1, zorder=5,
        )
    ax_areal.axvline(
        x=mean_ad, color=marker_color, linestyle=":",
        alpha=0.5, linewidth=1, zorder=5,
    )

    # Annotation on areal density panel
    label = scenario_result.scenario_name
    ax_areal.annotate(
        f"{label}\n{mean_ad:.1f} g/cm\u00b2 \u2192 {mean_dose:.1f} mSv/yr",
        xy=(mean_ad, mean_dose),
        xytext=(15, 15), textcoords="offset points",
        fontsize=8, color=marker_color,
        arrowprops=dict(arrowstyle="->", color=marker_color, lw=1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#313244",
                  edgecolor=marker_color, alpha=0.9),
        zorder=11,
    )

    # Per-point markers if multiple points
    if len(scenario_result.point_results) > 1:
        for pr in scenario_result.point_results:
            pr_ad = pr.mean_areal_density
            pr_metric_map = {
                "gcr_dose_mSv_yr": pr.mean_gcr_dose_rate,
                "gcr_dose_eq_mSv_yr": pr.mean_gcr_dose_equivalent_rate,
                "spe_dose_mSv": pr.mean_spe_dose,
                "spe_dose_eq_mSv": pr.mean_spe_dose_equivalent,
            }
            pr_dose = pr_metric_map.get(metric, pr.mean_gcr_dose_equivalent_rate)
            if pr_dose > 0 and pr_ad > 0:
                ax_areal.plot(
                    pr_ad, pr_dose, "o",
                    color=marker_color, markersize=7, zorder=9,
                    markeredgecolor="white", markeredgewidth=0.5, alpha=0.7,
                )


# ============================================================================
# Figure 3: Scenario Comparison
# ============================================================================

def plot_scenario_comparison(
    scenarios: list[ScenarioResult],
    metrics: list[str] | None = None,
    figsize: tuple = (12, 6),
) -> Figure:
    """Grouped bar chart comparing scenarios across multiple metrics."""
    if metrics is None:
        metrics = [
            "mean_gcr_dose_eq_rate_mSv_yr",
            "mean_spe_dose_eq_mSv",
            "mean_areal_density_gcm2",
        ]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, facecolor="#1e1e2e")
    if len(metrics) == 1:
        axes = [axes]

    # Build descriptive labels from geometry config when available
    scenario_labels = []
    for s in scenarios:
        label = s.scenario_name
        gc = s.geometry_config
        if gc:
            geo_type = gc.get("name", "")
            wall_cm = gc.get("total_wall_thickness_m", 0) * 100
            layers = gc.get("wall_layers", [])
            mat_names = [
                l["material_id"].replace("_", " ").title()
                for l in layers[:2]
            ]
            mat_str = " / ".join(mat_names) if mat_names else ""
            detail = f"{geo_type}\n{wall_cm:.0f}cm {mat_str}"
            label = f"{s.scenario_name}\n{detail}"
        scenario_labels.append(label)

    colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#94e2d5"]

    metric_labels = {
        "mean_gcr_dose_eq_rate_mSv_yr": ("GCR Dose Eq. Rate", "mSv/yr"),
        "mean_spe_dose_eq_mSv": ("SPE Dose Equivalent", "mSv"),
        "mean_areal_density_gcm2": ("Mean Areal Density", "g/cm\u00b2"),
        "mean_gcr_dose_rate_mSv_yr": ("GCR Dose Rate", "mSv/yr"),
        "min_areal_density_gcm2": ("Min Areal Density", "g/cm\u00b2"),
    }

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("#1e1e2e")

        values = []
        for scenario in scenarios:
            summary = scenario.summary()
            values.append(summary.get(metric, 0.0))

        bars = ax.bar(
            range(len(scenario_labels)), values,
            color=colors[: len(scenario_labels)],
            edgecolor="#45475a", linewidth=0.5,
        )

        label, unit = metric_labels.get(metric, (metric, ""))
        _style_axes(ax, title=label, xlabel="", ylabel=f"{label} ({unit})")
        ax.set_xticks(range(len(scenario_labels)))
        ax.set_xticklabels(scenario_labels, rotation=30, ha="right", fontsize=7)
        ax.title.set_color("#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.tick_params(colors="#cdd6f4")

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#cdd6f4",
            )

    _add_disclaimer(fig)
    fig.tight_layout()
    return fig


# ============================================================================
# Utility: Export figures
# ============================================================================

def save_figure(fig: Figure, path: str, dpi: int = 300):
    """Save figure to file (PNG, SVG, or PDF)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
