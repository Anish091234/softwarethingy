"""Analysis configuration and execution tab."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QComboBox,
    QPushButton,
    QCheckBox,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QTextEdit,
)
from PySide6.QtCore import Qt

from lunarad_peek.app.state import AppState


class AnalysisTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)

        # Configuration
        config_group = QGroupBox("Analysis Configuration")
        config_layout = QFormLayout()

        self.n_directions_spin = QSpinBox()
        self.n_directions_spin.setRange(42, 2562)
        self.n_directions_spin.setValue(162)
        self.n_directions_spin.setToolTip(
            "Number of ray directions. Common values:\n"
            "42 = fast preview\n"
            "162 = standard (default)\n"
            "642 = high resolution\n"
            "2562 = very high resolution (slow)"
        )
        config_layout.addRow("Ray Directions:", self.n_directions_spin)

        # Output metrics
        self.dose_check = QCheckBox("Dose (mGy)")
        self.dose_check.setChecked(True)
        config_layout.addRow("Output Metrics:", self.dose_check)

        self.dose_eq_check = QCheckBox("Dose Equivalent (mSv)")
        self.dose_eq_check.setChecked(True)
        config_layout.addRow("", self.dose_eq_check)

        self.flux_check = QCheckBox("Flux Attenuation")
        self.flux_check.setChecked(True)
        config_layout.addRow("", self.flux_check)

        self.areal_check = QCheckBox("Areal Density Map")
        self.areal_check.setChecked(True)
        config_layout.addRow("", self.areal_check)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Scenario management
        scenario_group = QGroupBox("Scenario Comparison")
        scenario_layout = QVBoxLayout()

        self.scenario_table = QTableWidget(0, 4)
        self.scenario_table.setHorizontalHeaderLabels([
            "Scenario", "GCR Dose Eq. (mSv/yr)", "SPE Dose Eq. (mSv)", "Mean AD (g/cm²)"
        ])
        self.scenario_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        scenario_layout.addWidget(self.scenario_table)

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Clear Scenarios")
        clear_btn.clicked.connect(self._clear_scenarios)
        btn_layout.addWidget(clear_btn)

        export_btn = QPushButton("Export Table (CSV)")
        export_btn.clicked.connect(self._export_csv)
        btn_layout.addWidget(export_btn)

        scenario_layout.addLayout(btn_layout)
        scenario_group.setLayout(scenario_layout)
        layout.addWidget(scenario_group)

        # Results log
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()

        # Connect signals
        self.state.analysis_completed.connect(self._on_analysis_completed)
        self.state.analysis_progress.connect(self._on_progress)

    def get_n_directions(self) -> int:
        return self.n_directions_spin.value()

    def _on_analysis_completed(self, result):
        summary = result.summary()
        row = self.scenario_table.rowCount()
        self.scenario_table.setRowCount(row + 1)

        self.scenario_table.setItem(
            row, 0, QTableWidgetItem(summary.get("scenario_name", ""))
        )
        self.scenario_table.setItem(
            row, 1, QTableWidgetItem(
                f"{summary.get('mean_gcr_dose_eq_rate_mSv_yr', 0):.1f}"
            )
        )
        self.scenario_table.setItem(
            row, 2, QTableWidgetItem(
                f"{summary.get('mean_spe_dose_eq_mSv', 0):.1f}"
            )
        )
        self.scenario_table.setItem(
            row, 3, QTableWidgetItem(
                f"{summary.get('mean_areal_density_gcm2', 0):.1f}"
            )
        )

        self.log_text.append(
            f"[{result.timestamp}] {result.scenario_name}: "
            f"GCR={summary.get('mean_gcr_dose_eq_rate_mSv_yr', 0):.1f} mSv/yr, "
            f"SPE={summary.get('mean_spe_dose_eq_mSv', 0):.1f} mSv, "
            f"AD={summary.get('mean_areal_density_gcm2', 0):.1f} g/cm², "
            f"Time={summary.get('computation_time_s', 0):.1f}s"
        )

    def _on_progress(self, fraction: float, message: str):
        pass  # Handled by status bar in main window

    def _clear_scenarios(self):
        self.scenario_table.setRowCount(0)
        self.state.scenarios.clear()
        self.log_text.append("[Cleared all scenarios]")

    def _export_csv(self):
        from PySide6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "scenarios.csv", "CSV (*.csv)"
        )
        if not filepath:
            return

        lines = []
        headers = []
        for col in range(self.scenario_table.columnCount()):
            headers.append(self.scenario_table.horizontalHeaderItem(col).text())
        lines.append(",".join(headers))

        for row in range(self.scenario_table.rowCount()):
            row_data = []
            for col in range(self.scenario_table.columnCount()):
                item = self.scenario_table.item(row, col)
                row_data.append(item.text() if item else "")
            lines.append(",".join(row_data))

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        self.log_text.append(f"[Exported to {filepath}]")
