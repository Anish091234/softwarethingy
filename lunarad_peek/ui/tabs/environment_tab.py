"""Radiation environment configuration tab."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QComboBox,
    QCheckBox,
    QFormLayout,
    QTabWidget,
    QTextEdit,
    QFrame,
)
from PySide6.QtCore import Qt

from lunarad_peek.app.state import AppState
from lunarad_peek.radiation.environments import (
    SolarCyclePhase,
    SPE_EVENT_LIBRARY,
)


class EnvironmentTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)

        # Sub-tabs for each environment type
        env_tabs = QTabWidget()

        # GCR tab
        gcr_widget = QWidget()
        gcr_layout = QVBoxLayout(gcr_widget)

        gcr_group = QGroupBox("Galactic Cosmic Rays (GCR)")
        gcr_form = QFormLayout()

        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["Solar Minimum", "Solar Maximum", "Intermediate"])
        self.phase_combo.currentIndexChanged.connect(self._on_phase_changed)
        gcr_form.addRow("Solar Cycle Phase:", self.phase_combo)

        self.phi_spin = QDoubleSpinBox()
        self.phi_spin.setRange(200, 1500)
        self.phi_spin.setValue(400)
        self.phi_spin.setSuffix(" MV")
        self.phi_spin.setSingleStep(50)
        self.phi_spin.valueChanged.connect(self._on_phi_changed)
        gcr_form.addRow("Modulation Parameter (φ):", self.phi_spin)

        gcr_group.setLayout(gcr_form)
        gcr_layout.addWidget(gcr_group)

        # GCR summary
        gcr_summary = QGroupBox("GCR Environment Summary")
        summary_layout = QFormLayout()

        self.gcr_free_dose = QLabel("—")
        summary_layout.addRow("Free-space dose rate:", self.gcr_free_dose)

        self.gcr_surface_dose = QLabel("—")
        summary_layout.addRow("Lunar surface dose rate:", self.gcr_surface_dose)

        self.gcr_surface_dose_eq = QLabel("—")
        summary_layout.addRow("Lunar surface dose eq. rate:", self.gcr_surface_dose_eq)

        gcr_summary.setLayout(summary_layout)
        gcr_layout.addWidget(gcr_summary)

        gcr_layout.addStretch()
        env_tabs.addTab(gcr_widget, "GCR")

        # SPE tab
        spe_widget = QWidget()
        spe_layout = QVBoxLayout(spe_widget)

        spe_group = QGroupBox("Solar Particle Events (SPE)")
        spe_form = QFormLayout()

        self.spe_combo = QComboBox()
        self.spe_combo.addItem("None (GCR only)", "")
        for key, event in SPE_EVENT_LIBRARY.items():
            self.spe_combo.addItem(f"{event.name} ({event.date})", key)
        # Match the default SPE event set in AppState (August 1972).
        default_spe_key = "aug_1972"
        default_idx = self.spe_combo.findData(default_spe_key)
        if default_idx >= 0:
            self.spe_combo.setCurrentIndex(default_idx)
        self.spe_combo.currentIndexChanged.connect(self._on_spe_changed)
        spe_form.addRow("Select Event:", self.spe_combo)

        spe_group.setLayout(spe_form)
        spe_layout.addWidget(spe_group)

        # SPE details
        self.spe_details = QTextEdit()
        self.spe_details.setReadOnly(True)
        self.spe_details.setMaximumHeight(200)
        self.spe_details.setPlaceholderText("Select an SPE event to see details...")
        spe_layout.addWidget(self.spe_details)

        spe_layout.addStretch()
        env_tabs.addTab(spe_widget, "SPE")

        # Solar Wind tab
        sw_widget = QWidget()
        sw_layout = QVBoxLayout(sw_widget)

        sw_group = QGroupBox("Solar Wind")
        sw_form = QVBoxLayout()

        self.sw_check = QCheckBox("Include Solar Wind Analysis")
        self.sw_check.stateChanged.connect(self._on_sw_changed)
        sw_form.addWidget(self.sw_check)

        sw_note = QLabel(
            "Note: Solar wind consists of ~1 keV/nucleon protons. "
            "It is a SURFACE INTERACTION only — stopped by any solid material. "
            "Not included in habitat biological dose calculation."
        )
        sw_note.setWordWrap(True)
        sw_note.setStyleSheet("color: #a6adc8; font-style: italic; padding: 10px;")
        sw_form.addWidget(sw_note)

        sw_params = QFormLayout()
        self.sw_flux = QDoubleSpinBox()
        self.sw_flux.setRange(1e6, 1e10)
        self.sw_flux.setValue(3e8)
        self.sw_flux.setDecimals(0)
        self.sw_flux.setSuffix(" p/cm²/s")
        self.sw_flux.setEnabled(False)
        sw_params.addRow("Proton Flux:", self.sw_flux)

        self.sw_velocity = QDoubleSpinBox()
        self.sw_velocity.setRange(200, 800)
        self.sw_velocity.setValue(400)
        self.sw_velocity.setSuffix(" km/s")
        self.sw_velocity.setEnabled(False)
        sw_params.addRow("Bulk Velocity:", self.sw_velocity)

        sw_form.addLayout(sw_params)
        sw_group.setLayout(sw_form)
        sw_layout.addWidget(sw_group)
        sw_layout.addStretch()
        env_tabs.addTab(sw_widget, "Solar Wind")

        layout.addWidget(env_tabs)

        # Secondary radiation option
        secondary_group = QGroupBox("Secondary Radiation")
        sec_layout = QVBoxLayout()

        self.secondary_check = QCheckBox("Include approximate secondary radiation estimate")
        sec_layout.addWidget(self.secondary_check)

        sec_note = QLabel(
            "Secondary radiation (neutron buildup, target fragmentation) is "
            "estimated using empirical correction factors. Results are labeled "
            "as approximate."
        )
        sec_note.setWordWrap(True)
        sec_note.setStyleSheet("color: #a6adc8; font-style: italic;")
        sec_layout.addWidget(sec_note)

        secondary_group.setLayout(sec_layout)
        layout.addWidget(secondary_group)

        # Update display
        self._update_gcr_summary()
        self._on_spe_changed(self.spe_combo.currentIndex())

    def _on_phase_changed(self, index: int):
        phases = [SolarCyclePhase.SOLAR_MINIMUM, SolarCyclePhase.SOLAR_MAXIMUM,
                  SolarCyclePhase.INTERMEDIATE]
        phase = phases[index]
        self.state.set_gcr_phase(phase)

        phi_values = {0: 400, 1: 1200, 2: 700}
        self.phi_spin.blockSignals(True)
        self.phi_spin.setValue(phi_values[index])
        self.phi_spin.blockSignals(False)

        self._update_gcr_summary()

    def _on_phi_changed(self, value: float):
        self.state.set_gcr_phi(value)
        self._update_gcr_summary()

    def _on_spe_changed(self, index: int):
        event_key = self.spe_combo.currentData()
        self.state.set_spe_event(event_key if event_key else None)

        if event_key and event_key in SPE_EVENT_LIBRARY:
            event = SPE_EVENT_LIBRARY[event_key]
            self.spe_details.setHtml(
                f"<b>{event.name}</b> ({event.date})<br><br>"
                f"{event.description}<br><br>"
                f"Total fluence (>30 MeV): {event.total_fluence_gt30MeV:.2e} p/cm²<br>"
                f"Spectral index γ: {event.gamma}<br>"
                f"Characteristic energy E₀: {event.E0_MeV} MeV<br><br>"
                f"<i>Source: {event.source}</i>"
            )
        else:
            self.spe_details.clear()

    def _on_sw_changed(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.state.toggle_solar_wind(enabled)
        self.sw_flux.setEnabled(enabled)
        self.sw_velocity.setEnabled(enabled)

    def _update_gcr_summary(self):
        gcr = self.state.environment.gcr
        self.gcr_free_dose.setText(f"{gcr.free_space_dose_rate:.0f} mSv/yr")
        self.gcr_surface_dose.setText(f"{gcr.lunar_surface_dose_rate:.0f} mSv/yr")
        self.gcr_surface_dose_eq.setText(
            f"{gcr.lunar_surface_dose_equivalent_rate:.0f} mSv/yr"
        )
