"""Visualization and export tab for results display and paper figure generation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QFormLayout,
    QSplitter,
    QTabWidget,
    QFileDialog,
    QDoubleSpinBox,
    QScrollArea,
    QMessageBox,
    QApplication,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("QtAgg")

from lunarad_peek.app.state import AppState
from lunarad_peek.analysis.engine import ScenarioResult


class VisualizationTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._current_result: ScenarioResult | None = None

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(350)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Figure selection
        fig_group = QGroupBox("Paper Figures")
        fig_layout = QVBoxLayout()

        self.auto_all_btn = QPushButton("Auto-Generate All Paper Figures")
        self.auto_all_btn.setObjectName("runButton")
        self.auto_all_btn.setToolTip(
            "One-click: clears current scenarios, sets up a R=5m dome with "
            "0.5m walls and 5 astronauts (1 center + ring of 4), runs analysis "
            "for Regolith-PEEK, PEEK, Aluminum, and Highland Regolith under "
            "August 1972 SPE, then saves Fig 1/2/3 to a folder."
        )
        self.auto_all_btn.clicked.connect(self._generate_all_paper_figures)
        fig_layout.addWidget(self.auto_all_btn)

        self.fig1_btn = QPushButton("Fig 1: Interior Radiation Map")
        self.fig1_btn.clicked.connect(self._generate_fig1)
        fig_layout.addWidget(self.fig1_btn)

        self.fig2_btn = QPushButton("Fig 2: Dose vs Shielding")
        self.fig2_btn.clicked.connect(self._generate_fig2)
        fig_layout.addWidget(self.fig2_btn)

        self.fig3_btn = QPushButton("Fig 3: Scenario Comparison")
        self.fig3_btn.clicked.connect(self._generate_fig3)
        fig_layout.addWidget(self.fig3_btn)

        fig_group.setLayout(fig_layout)
        left_layout.addWidget(fig_group)

        # Additional plots
        extra_group = QGroupBox("Additional Plots")
        extra_layout = QVBoxLayout()

        self.dir_map_btn = QPushButton("Directional Shielding Map")
        self.dir_map_btn.clicked.connect(self._generate_directional_map)
        extra_layout.addWidget(self.dir_map_btn)

        extra_group.setLayout(extra_layout)
        left_layout.addWidget(extra_group)

        # Plot options
        opt_group = QGroupBox("Plot Options")
        opt_form = QFormLayout()

        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "GCR Dose Equivalent Rate",
            "GCR Dose Rate",
            "SPE Dose Equivalent",
            "Areal Density",
        ])
        opt_form.addRow("Metric:", self.metric_combo)

        self.slice_combo = QComboBox()
        self.slice_combo.addItems(["Y=0 (XZ plane)", "X=0 (YZ plane)", "Z=mid (XY plane)"])
        opt_form.addRow("Slice Plane:", self.slice_combo)

        self.max_thickness_spin = QDoubleSpinBox()
        self.max_thickness_spin.setRange(10, 500)
        self.max_thickness_spin.setValue(200)
        self.max_thickness_spin.setSuffix(" cm")
        opt_form.addRow("Max Thickness:", self.max_thickness_spin)

        opt_group.setLayout(opt_form)
        left_layout.addWidget(opt_group)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        export_png_btn = QPushButton("Export as PNG (300 DPI)")
        export_png_btn.clicked.connect(lambda: self._export_figure("png"))
        export_layout.addWidget(export_png_btn)

        export_svg_btn = QPushButton("Export as SVG")
        export_svg_btn.clicked.connect(lambda: self._export_figure("svg"))
        export_layout.addWidget(export_svg_btn)

        export_pdf_btn = QPushButton("Export as PDF")
        export_pdf_btn.clicked.connect(lambda: self._export_figure("pdf"))
        export_layout.addWidget(export_pdf_btn)

        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # Right: figure canvas with interactive toolbar
        self._right_widget = QWidget()
        self._right_layout = QVBoxLayout(self._right_widget)

        self.figure = Figure(figsize=(10, 7), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self._right_widget)
        self.toolbar.setStyleSheet("background-color: #2e2e3e; color: #cdd6f4;")
        self._right_layout.addWidget(self.toolbar)
        self._right_layout.addWidget(self.canvas)

        # Placeholder message
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1e1e2e")
        ax.text(
            0.5, 0.5,
            "Run an analysis to generate visualizations",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="#6c7086",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

        splitter.addWidget(self._right_widget)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

    def update_results(self, result: ScenarioResult):
        self._current_result = result
        # Auto-generate Fig 2 with scenario overlay so the plot reflects the analysis
        self._generate_fig2()

    def _get_metric_key(self) -> str:
        mapping = {
            0: "gcr_dose_equivalent_rate",
            1: "gcr_dose_rate",
            2: "spe_dose_equivalent",
            3: "areal_density",
        }
        return mapping.get(self.metric_combo.currentIndex(), "gcr_dose_equivalent_rate")

    def _get_dose_metric_key(self) -> str:
        mapping = {
            0: "gcr_dose_eq_mSv_yr",
            1: "gcr_dose_mSv_yr",
            2: "spe_dose_eq_mSv",
            3: "gcr_dose_eq_mSv_yr",
        }
        return mapping.get(self.metric_combo.currentIndex(), "gcr_dose_eq_mSv_yr")

    def _generate_fig1(self):
        if not self._current_result:
            return

        from lunarad_peek.visualization.plots import plot_cross_section_dose_map

        slice_map = {0: "y", 1: "x", 2: "z"}
        slice_axis = slice_map.get(self.slice_combo.currentIndex(), "y")

        # Build habitat info for outline drawing
        habitat_info = None
        if self.state.scene.habitat:
            h = self.state.scene.habitat
            habitat_info = {
                "type": type(h).__name__,
                "inner_radius": getattr(h, "inner_radius", 5.0),
                "total_wall_thickness": h.total_wall_thickness,
                "position": h.position.tolist(),
                "length": getattr(h, "length", None),
            }

        self.figure.clear()
        fig = plot_cross_section_dose_map(
            self._current_result.point_results,
            slice_axis=slice_axis,
            metric=self._get_metric_key(),
            habitat_info=habitat_info,
            scenario_name=self._current_result.scenario_name,
        )

        # Copy axes to our canvas figure
        self._copy_figure(fig)

    def _generate_fig2(self):
        from lunarad_peek.visualization.plots import plot_dose_vs_shielding

        # Select materials to compare
        materials_to_plot = {}
        for key in ["highland_regolith", "peek", "regolith_peek_composite", "aluminum"]:
            if key in self.state.material_library:
                materials_to_plot[key] = self.state.material_library[key]

        self.figure.clear()
        fig = plot_dose_vs_shielding(
            materials_to_plot,
            self.state.environment,
            metric=self._get_dose_metric_key(),
            max_thickness_cm=self.max_thickness_spin.value(),
            scenario_result=self._current_result,
        )
        self._copy_figure(fig)

    def _generate_fig3(self):
        if len(self.state.scenarios) < 1:
            return

        from lunarad_peek.visualization.plots import plot_scenario_comparison

        self.figure.clear()
        fig = plot_scenario_comparison(self.state.scenarios)
        self._copy_figure(fig)

    def _generate_directional_map(self):
        if not self._current_result or not self._current_result.point_results:
            return

        from lunarad_peek.visualization.plots import plot_directional_shielding_map

        self.figure.clear()
        fig = plot_directional_shielding_map(
            self._current_result.point_results[0],
            metric=self._get_metric_key(),
        )
        self._copy_figure(fig)

    def _copy_figure(self, source_fig: Figure):
        """Replace the canvas with the source figure so it stays interactive."""
        import matplotlib.pyplot as plt

        # Remove old canvas and toolbar
        self._right_layout.removeWidget(self.canvas)
        self.canvas.setParent(None)
        self.canvas.deleteLater()
        self._right_layout.removeWidget(self.toolbar)
        self.toolbar.setParent(None)
        self.toolbar.deleteLater()

        # Close old figure
        plt.close(self.figure)

        # Use the source figure directly as the new canvas figure
        self.figure = source_fig
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self._right_widget)
        self.toolbar.setStyleSheet("background-color: #2e2e3e; color: #cdd6f4;")
        self._right_layout.addWidget(self.toolbar)
        self._right_layout.addWidget(self.canvas)
        self.canvas.draw()

    def _generate_all_paper_figures(self):
        """One-click batch: set up baseline dome + 5 astronauts, run each
        candidate wall material under August 1972 SPE, and save all three
        paper figures to a user-chosen folder."""
        import math

        from lunarad_peek.visualization.plots import (
            plot_cross_section_dose_map,
            plot_dose_vs_shielding,
            plot_scenario_comparison,
            save_figure,
        )

        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save Paper Figures"
        )
        if not folder:
            return

        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reset project state so scenario list and habitat start clean.
        # clear() also re-seats the Aug 1972 SPE default.
        self.state.clear()
        self.state.set_spe_event("aug_1972")

        # Baseline dome geometry (matches SolidWorks reference)
        BASELINE_RADIUS = 5.0
        BASELINE_HEIGHT_RATIO = 1.0
        BASELINE_WALL_THICKNESS = 0.5
        RING_RADIUS = BASELINE_RADIUS * 0.5  # 2.5 m, matches "Add Ring" button

        # Place the 5 astronauts once; set_habitat() preserves targets
        # when we rebuild the habitat per material.
        self.state.create_dome_habitat(
            inner_radius=BASELINE_RADIUS,
            dome_height_ratio=BASELINE_HEIGHT_RATIO,
            wall_layers=[("regolith_peek_composite", BASELINE_WALL_THICKNESS)],
        )
        self.state.add_astronaut("Crew-Center", 0.0, 0.0, 0.0)
        for i in range(4):
            angle = 2 * math.pi * i / 4
            self.state.add_astronaut(
                f"Crew-{i+1}",
                RING_RADIUS * math.cos(angle),
                RING_RADIUS * math.sin(angle),
                0.0,
            )

        WALL_MATERIALS = [
            ("regolith_peek_composite", "Regolith-PEEK"),
            ("peek", "PEEK"),
            ("aluminum", "Aluminum"),
            ("highland_regolith", "Highland Regolith"),
            ("mare_regolith", "Mare Regolith"),
        ]
        available = [
            (mid, label) for mid, label in WALL_MATERIALS
            if mid in self.state.material_library
        ]
        if not available:
            QMessageBox.warning(
                self, "Missing Materials",
                "None of the expected wall materials are in the library.",
            )
            return

        n_dirs = 162
        results = []
        for idx, (mat_id, label) in enumerate(available, start=1):
            # Rebuild habitat with this material; targets persist on the scene.
            self.state.create_dome_habitat(
                inner_radius=BASELINE_RADIUS,
                dome_height_ratio=BASELINE_HEIGHT_RATIO,
                wall_layers=[(mat_id, BASELINE_WALL_THICKNESS)],
            )
            name = f"S{idx}: {label} {BASELINE_WALL_THICKNESS*100:.0f}cm"
            result = self.state.run_analysis(
                scenario_name=name, n_directions=n_dirs
            )
            results.append(result)
            # Keep the UI from feeling frozen between scenarios.
            QApplication.processEvents()

        # Use the Regolith-PEEK scenario as the "hero" for Fig 1/2
        hero_result = results[0]
        habitat_info = {
            "type": "ShellDomeHabitat",
            "inner_radius": BASELINE_RADIUS,
            "total_wall_thickness": BASELINE_WALL_THICKNESS,
            "position": [0.0, 0.0, 0.0],
            "length": None,
        }

        fig1 = plot_cross_section_dose_map(
            hero_result.point_results,
            slice_axis="y",
            metric="gcr_dose_equivalent_rate",
            habitat_info=habitat_info,
            scenario_name=hero_result.scenario_name,
        )
        save_figure(fig1, str(output_dir / "figure1_interior_map.png"))

        materials_for_fig2 = {
            k: self.state.material_library[k]
            for k in ["highland_regolith", "peek", "regolith_peek_composite", "aluminum"]
            if k in self.state.material_library
        }
        fig2 = plot_dose_vs_shielding(
            materials_for_fig2,
            self.state.environment,
            metric="combined_annual_dose_mSv_yr",
            max_thickness_cm=200,
            scenario_result=hero_result,
        )
        save_figure(fig2, str(output_dir / "figure2_dose_vs_shielding.png"))

        fig3 = plot_scenario_comparison(results)
        save_figure(fig3, str(output_dir / "figure3_scenario_comparison.png"))

        # Show Fig 3 in the canvas so the user sees the full comparison.
        self._current_result = hero_result
        self._copy_figure(fig3)

        QMessageBox.information(
            self, "Paper Figures Generated",
            f"Saved 3 figures to:\n{output_dir}\n\n"
            f"  figure1_interior_map.png\n"
            f"  figure2_dose_vs_shielding.png\n"
            f"  figure3_scenario_comparison.png\n\n"
            f"{len(results)} scenarios ran under August 1972 SPE.",
        )

    def _export_figure(self, fmt: str):
        extensions = {"png": "PNG (*.png)", "svg": "SVG (*.svg)", "pdf": "PDF (*.pdf)"}
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export as {fmt.upper()}", f"lunarad_figure.{fmt}",
            extensions.get(fmt, f"{fmt.upper()} (*.{fmt})")
        )
        if filepath:
            dpi = 300 if fmt == "png" else 150
            self.figure.savefig(
                filepath, dpi=dpi, bbox_inches="tight",
                facecolor=self.figure.get_facecolor(),
            )
