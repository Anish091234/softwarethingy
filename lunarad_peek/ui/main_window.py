"""Main application window for LunaRad-PEEK."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QStatusBar,
    QToolBar,
    QMenuBar,
    QMenu,
    QMessageBox,
    QFileDialog,
    QProgressBar,
    QLabel,
    QWidget,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QKeySequence

from lunarad_peek.app.state import AppState
from lunarad_peek.ui.tabs.geometry_tab import GeometryTab
from lunarad_peek.ui.tabs.materials_tab import MaterialsTab
from lunarad_peek.ui.tabs.environment_tab import EnvironmentTab
from lunarad_peek.ui.tabs.analysis_tab import AnalysisTab
from lunarad_peek.ui.tabs.visualization_tab import VisualizationTab


class MainWindow(QMainWindow):
    """Main application window with tabbed workflow."""

    def __init__(self):
        super().__init__()
        self.state = AppState()

        self.setWindowTitle("LunaRad-PEEK — Lunar Habitat Radiation Shielding Analysis")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central()
        self._setup_statusbar()
        self._connect_signals()

    def _setup_menubar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)

        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        import_stl = QAction("Import STL...", self)
        import_stl.triggered.connect(self._import_stl)
        file_menu.addAction(import_stl)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        reset_view = QAction("Reset 3D View", self)
        reset_view.triggered.connect(self._reset_3d_view)
        view_menu.addAction(reset_view)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        run_action = QAction("&Run Analysis", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self._run_analysis)
        analysis_menu.addAction(run_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About LunaRad-PEEK", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        methods_action = QAction("&Methods && References", self)
        methods_action.triggered.connect(self._show_methods)
        help_menu.addAction(methods_action)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction("New").triggered.connect(self._new_project)
        toolbar.addAction("Save").triggered.connect(self._save_project)
        toolbar.addSeparator()
        toolbar.addAction("Import STL").triggered.connect(self._import_stl)
        toolbar.addSeparator()

        run_btn = toolbar.addAction("Run Analysis")
        run_btn.triggered.connect(self._run_analysis)

    def _setup_central(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.geometry_tab = GeometryTab(self.state)
        self.materials_tab = MaterialsTab(self.state)
        self.environment_tab = EnvironmentTab(self.state)
        self.analysis_tab = AnalysisTab(self.state)
        self.visualization_tab = VisualizationTab(self.state)

        self.tabs.addTab(self.geometry_tab, "1. Geometry")
        self.tabs.addTab(self.materials_tab, "2. Materials")
        self.tabs.addTab(self.environment_tab, "3. Environment")
        self.tabs.addTab(self.analysis_tab, "4. Analysis")
        self.tabs.addTab(self.visualization_tab, "5. Visualization")

        self.setCentralWidget(self.tabs)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label)

    def _connect_signals(self):
        self.state.analysis_started.connect(self._on_analysis_started)
        self.state.analysis_progress.connect(self._on_analysis_progress)
        self.state.analysis_completed.connect(self._on_analysis_completed)

    # --- Slots ---

    def _new_project(self):
        self.state.clear()
        self.status_label.setText("New project created")

    def _save_project(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "LunaRad Project (*.lrp);;JSON (*.json)"
        )
        if filepath:
            self.state.save_project(Path(filepath))
            self.status_label.setText(f"Saved: {filepath}")

    def _import_stl(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import STL", "", "STL Files (*.stl);;All Files (*)"
        )
        if filepath:
            try:
                from lunarad_peek.geometry.stl_io import read_stl
                from lunarad_peek.geometry.scene import GeometryLayer

                mesh = read_stl(filepath)
                layer = GeometryLayer(
                    name=Path(filepath).stem,
                    mesh=mesh,
                    material_id="regolith_peek_composite",
                )
                self.state.scene.layers.append(layer)
                self.state.scene_changed.emit()
                self.status_label.setText(
                    f"Imported: {Path(filepath).name} "
                    f"({mesh.num_vertices} verts, {mesh.num_faces} faces)"
                )
            except Exception as e:
                QMessageBox.critical(self, "Import Error", str(e))

    def _run_analysis(self):
        if not self.state.scene.layers:
            QMessageBox.warning(
                self, "No Geometry",
                "Please create or import habitat geometry before running analysis."
            )
            return
        if not self.state.scene.targets:
            QMessageBox.warning(
                self, "No Targets",
                "Please place at least one astronaut target before running analysis."
            )
            return

        name = f"Scenario {len(self.state.scenarios) + 1}"
        n_dirs = self.analysis_tab.get_n_directions()
        self.state.run_analysis(scenario_name=name, n_directions=n_dirs)

    def _reset_3d_view(self):
        self.geometry_tab.reset_3d_view()

    def _on_analysis_started(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis running...")

    def _on_analysis_progress(self, fraction: float, message: str):
        self.progress_bar.setValue(int(fraction * 100))
        self.status_label.setText(message)

    def _on_analysis_completed(self, result):
        self.progress_bar.setVisible(False)
        summary = result.summary()
        self.status_label.setText(
            f"Analysis complete: {summary.get('mean_gcr_dose_eq_rate_mSv_yr', 0):.1f} mSv/yr "
            f"mean GCR dose eq. ({result.computation_time_s:.1f}s)"
        )
        self.tabs.setCurrentWidget(self.visualization_tab)
        self.visualization_tab.update_results(result)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About LunaRad-PEEK",
            "<h3>LunaRad-PEEK v1.0.0-alpha</h3>"
            "<p>Conceptual Radiation Visualization and Shielding-Analysis Tool "
            "for Lunar Habitats</p>"
            "<p><b>IMPORTANT:</b> This is an early-stage conceptual design tool. "
            "It is NOT a Monte Carlo transport solver. Results are approximate "
            "estimates suitable for conceptual design comparison.</p>"
            "<p>Uses OLTARIS-inspired areal-density-based workflow with "
            "literature-derived response functions.</p>"
            "<hr>"
            "<p>Supports research paper:<br>"
            "<i>A Dual-Function Regolith-Based Composite Wall for Lunar Habitats: "
            "Mechanical Strength and Radiation Attenuation</i></p>",
        )

    def _show_methods(self):
        from lunarad_peek.ui.dialogs.methods_dialog import MethodsDialog
        dialog = MethodsDialog(self)
        dialog.exec()
