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
    figsize: tuple = (10, 8),
) -> Figure:
    """Generate 2D cross-section dose map.

    Shows dose equivalent (or other metric) distribution on a slice plane
    through the habitat, interpolated from directional results at target points.
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
    for pr in point_results:
        pos = pr.position
        val = getattr(pr, f"mean_{metric}", pr.mean_gcr_dose_equivalent_rate)
        positions.append(pos)
        values.append(val)

    positions = np.array(positions)
    values = np.array(values)

    # Map axes based on slice
    axis_map = {"x": (1, 2, "Y (m)", "Z (m)"),
                "y": (0, 2, "X (m)", "Z (m)"),
                "z": (0, 1, "X (m)", "Y (m)")}
    ax1, ax2, xlabel, ylabel = axis_map.get(slice_axis, (0, 2, "X (m)", "Z (m)"))

    scatter = ax.scatter(
        positions[:, ax1], positions[:, ax2],
        c=values, cmap=RESULT_COLORMAP, s=100,
        edgecolors="white", linewidths=0.5, zorder=5,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    metric_labels = {
        "gcr_dose_equivalent_rate": "GCR Dose Equivalent Rate (mSv/yr)",
        "gcr_dose_rate": "GCR Dose Rate (mSv/yr)",
        "areal_density": "Areal Density (g/cm²)",
        "spe_dose_equivalent": "SPE Dose Equivalent (mSv)",
    }
    cbar.set_label(metric_labels.get(metric, metric), fontsize=9, color="#cdd6f4")
    cbar.ax.tick_params(labelsize=8, colors="#cdd6f4")

    _style_axes(ax, title=f"Interior Radiation Map — {slice_axis.upper()}={slice_value:.1f}m slice",
                xlabel=xlabel, ylabel=ylabel)
    ax.title.set_color("#cdd6f4")
    ax.xaxis.label.set_color("#cdd6f4")
    ax.yaxis.label.set_color("#cdd6f4")
    ax.tick_params(colors="#cdd6f4")

    # Mark astronaut positions
    for pr in point_results:
        ax.annotate(
            pr.point_name, (pr.position[ax1], pr.position[ax2]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, color="#f5e0dc", alpha=0.8,
        )

    _add_disclaimer(fig)
    fig.tight_layout()
    return fig


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
    figsize: tuple = (12, 5),
) -> Figure:
    """Dual-panel plot: dose vs physical thickness and dose vs areal density."""
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

    _add_disclaimer(fig)
    fig.tight_layout()
    return fig


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

    scenario_names = [s.scenario_name for s in scenarios]
    colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#94e2d5"]

    metric_labels = {
        "mean_gcr_dose_eq_rate_mSv_yr": ("GCR Dose Eq. Rate", "mSv/yr"),
        "mean_spe_dose_eq_mSv": ("SPE Dose Equivalent", "mSv"),
        "mean_areal_density_gcm2": ("Mean Areal Density", "g/cm²"),
        "mean_gcr_dose_rate_mSv_yr": ("GCR Dose Rate", "mSv/yr"),
        "min_areal_density_gcm2": ("Min Areal Density", "g/cm²"),
    }

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("#1e1e2e")

        values = []
        for scenario in scenarios:
            summary = scenario.summary()
            values.append(summary.get(metric, 0.0))

        bars = ax.bar(
            range(len(scenario_names)), values,
            color=colors[: len(scenario_names)],
            edgecolor="#45475a", linewidth=0.5,
        )

        label, unit = metric_labels.get(metric, (metric, ""))
        _style_axes(ax, title=label, xlabel="", ylabel=f"{label} ({unit})")
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=8)
        ax.title.set_color("#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.tick_params(colors="#cdd6f4")

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=8, color="#cdd6f4",
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
