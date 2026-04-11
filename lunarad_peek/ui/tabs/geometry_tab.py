"""Geometry tab for habitat creation and configuration."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QCheckBox,
    QScrollArea,
    QSplitter,
    QFormLayout,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt

from lunarad_peek.app.state import AppState
from lunarad_peek.ui.viewer3d import Viewer3D


class GeometryTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(400)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Habitat type selection
        type_group = QGroupBox("Habitat Type")
        type_layout = QFormLayout()

        self.habitat_combo = QComboBox()
        self.habitat_combo.addItems(["Shell Dome", "Cylindrical Tunnel", "Import STL"])
        self.habitat_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addRow("Type:", self.habitat_combo)

        type_group.setLayout(type_layout)
        left_layout.addWidget(type_group)

        # Dome parameters
        self.dome_group = QGroupBox("Dome Parameters")
        dome_layout = QFormLayout()

        self.dome_radius = QDoubleSpinBox()
        self.dome_radius.setRange(1.0, 50.0)
        self.dome_radius.setValue(5.0)
        self.dome_radius.setSuffix(" m")
        self.dome_radius.setDecimals(1)
        dome_layout.addRow("Inner Radius:", self.dome_radius)

        self.dome_height_ratio = QDoubleSpinBox()
        self.dome_height_ratio.setRange(0.3, 1.5)
        self.dome_height_ratio.setValue(1.0)
        self.dome_height_ratio.setSingleStep(0.1)
        dome_layout.addRow("Height Ratio:", self.dome_height_ratio)

        baseline_label = QLabel(
            "Default matches SolidWorks baseline:\n"
            "R = 5.0 m, wall = 0.5 m"
        )
        baseline_label.setStyleSheet("color: #94e2d5; font-size: 9pt; font-style: italic;")
        baseline_label.setWordWrap(True)
        dome_layout.addRow(baseline_label)

        self.dome_group.setLayout(dome_layout)
        left_layout.addWidget(self.dome_group)

        # Tunnel parameters
        self.tunnel_group = QGroupBox("Tunnel Parameters")
        tunnel_layout = QFormLayout()

        self.tunnel_radius = QDoubleSpinBox()
        self.tunnel_radius.setRange(1.0, 20.0)
        self.tunnel_radius.setValue(3.0)
        self.tunnel_radius.setSuffix(" m")
        tunnel_layout.addRow("Inner Radius:", self.tunnel_radius)

        self.tunnel_length = QDoubleSpinBox()
        self.tunnel_length.setRange(5.0, 100.0)
        self.tunnel_length.setValue(15.0)
        self.tunnel_length.setSuffix(" m")
        tunnel_layout.addRow("Length:", self.tunnel_length)

        self.tunnel_burial = QDoubleSpinBox()
        self.tunnel_burial.setRange(0.0, 20.0)
        self.tunnel_burial.setValue(0.0)
        self.tunnel_burial.setSuffix(" m")
        tunnel_layout.addRow("Burial Depth:", self.tunnel_burial)

        self.tunnel_group.setLayout(tunnel_layout)
        self.tunnel_group.setVisible(False)
        left_layout.addWidget(self.tunnel_group)

        # Wall layers
        wall_group = QGroupBox("Wall Layers")
        wall_layout = QVBoxLayout()

        self.wall_table = QTableWidget(1, 3)
        self.wall_table.setHorizontalHeaderLabels(["Material", "Thickness (m)", ""])
        self.wall_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.wall_table.setMaximumHeight(150)
        self._populate_wall_table()
        wall_layout.addWidget(self.wall_table)

        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self._add_wall_layer)
        wall_layout.addWidget(add_layer_btn)

        wall_group.setLayout(wall_layout)
        left_layout.addWidget(wall_group)

        # Environmental geometry
        env_group = QGroupBox("Environmental Geometry")
        env_layout = QVBoxLayout()

        self.terrain_check = QCheckBox("Add Terrain Ground Plane")
        env_layout.addWidget(self.terrain_check)

        self.cover_check = QCheckBox("Add Regolith Cover")
        env_layout.addWidget(self.cover_check)

        cover_form = QFormLayout()
        self.cover_thickness = QDoubleSpinBox()
        self.cover_thickness.setRange(0.1, 20.0)
        self.cover_thickness.setValue(2.0)
        self.cover_thickness.setSuffix(" m")
        cover_form.addRow("Cover Thickness:", self.cover_thickness)
        env_layout.addLayout(cover_form)

        self.overburden_check = QCheckBox("Add Lava-tube Overburden")
        env_layout.addWidget(self.overburden_check)

        ob_form = QFormLayout()
        self.overburden_thickness = QDoubleSpinBox()
        self.overburden_thickness.setRange(1.0, 50.0)
        self.overburden_thickness.setValue(5.0)
        self.overburden_thickness.setSuffix(" m")
        ob_form.addRow("Overburden Thickness:", self.overburden_thickness)
        env_layout.addLayout(ob_form)

        env_group.setLayout(env_layout)
        left_layout.addWidget(env_group)

        # Astronaut placement
        astro_group = QGroupBox("Astronaut Targets")
        astro_layout = QVBoxLayout()

        add_center_btn = QPushButton("Add Astronaut at Center")
        add_center_btn.clicked.connect(self._add_center_astronaut)
        astro_layout.addWidget(add_center_btn)

        add_ring_btn = QPushButton("Add Ring (4 astronauts)")
        add_ring_btn.clicked.connect(self._add_ring_astronauts)
        astro_layout.addWidget(add_ring_btn)

        self.target_count_label = QLabel("Targets: 0")
        astro_layout.addWidget(self.target_count_label)

        astro_group.setLayout(astro_layout)
        left_layout.addWidget(astro_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Geometry")
        self.generate_btn.setObjectName("runButton")
        self.generate_btn.clicked.connect(self._generate_geometry)
        left_layout.addWidget(self.generate_btn)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # Right panel: 3D viewer
        self.viewer = Viewer3D(state)
        splitter.addWidget(self.viewer)

        splitter.setSizes([350, 850])
        layout.addWidget(splitter)

        # Connect state signals
        self.state.scene_changed.connect(self._on_scene_changed)

    def _on_type_changed(self, index: int):
        self.dome_group.setVisible(index == 0)
        self.tunnel_group.setVisible(index == 1)

    def _populate_wall_table(self):
        self.wall_table.setRowCount(1)
        mat_combo = QComboBox()
        mat_combo.addItems(list(self.state.material_library.keys()))
        mat_combo.setCurrentText("regolith_peek_composite")
        self.wall_table.setCellWidget(0, 0, mat_combo)

        thickness_spin = QDoubleSpinBox()
        thickness_spin.setRange(0.01, 10.0)
        thickness_spin.setValue(0.50)
        thickness_spin.setSuffix(" m")
        thickness_spin.setDecimals(3)
        self.wall_table.setCellWidget(0, 1, thickness_spin)

    def _add_wall_layer(self):
        row = self.wall_table.rowCount()
        self.wall_table.setRowCount(row + 1)

        mat_combo = QComboBox()
        mat_combo.addItems(list(self.state.material_library.keys()))
        self.wall_table.setCellWidget(row, 0, mat_combo)

        thickness_spin = QDoubleSpinBox()
        thickness_spin.setRange(0.01, 10.0)
        thickness_spin.setValue(0.10)
        thickness_spin.setSuffix(" m")
        thickness_spin.setDecimals(3)
        self.wall_table.setCellWidget(row, 1, thickness_spin)

    def _get_wall_layers(self) -> list[tuple[str, float]]:
        layers = []
        for row in range(self.wall_table.rowCount()):
            mat_widget = self.wall_table.cellWidget(row, 0)
            thick_widget = self.wall_table.cellWidget(row, 1)
            if mat_widget and thick_widget:
                layers.append((mat_widget.currentText(), thick_widget.value()))
        return layers

    def _add_center_astronaut(self):
        n = len(self.state.scene.targets)
        self.state.add_astronaut(f"Crew-{n+1}", 0.0, 0.0, 0.0)
        self.target_count_label.setText(f"Targets: {len(self.state.scene.targets)}")

    def _add_ring_astronauts(self):
        import math
        import numpy as np

        radius = 2.0
        if self.state.scene.habitat and hasattr(self.state.scene.habitat, "inner_radius"):
            radius = self.state.scene.habitat.inner_radius * 0.5

        for i in range(4):
            angle = 2 * math.pi * i / 4
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            n = len(self.state.scene.targets)
            self.state.add_astronaut(f"Crew-{n+1}", x, y, 0.0)

        self.target_count_label.setText(f"Targets: {len(self.state.scene.targets)}")

    def _generate_geometry(self):
        layers = self._get_wall_layers()
        idx = self.habitat_combo.currentIndex()

        if idx == 0:  # Dome
            self.state.create_dome_habitat(
                inner_radius=self.dome_radius.value(),
                dome_height_ratio=self.dome_height_ratio.value(),
                wall_layers=layers,
            )
        elif idx == 1:  # Tunnel
            self.state.create_tunnel_habitat(
                inner_radius=self.tunnel_radius.value(),
                length=self.tunnel_length.value(),
                burial_depth=self.tunnel_burial.value(),
                wall_layers=layers,
            )

        if self.terrain_check.isChecked():
            self.state.add_terrain()

        if self.cover_check.isChecked():
            self.state.add_regolith_cover(self.cover_thickness.value())

        if self.overburden_check.isChecked():
            self.state.add_overburden(self.overburden_thickness.value())

    def _on_scene_changed(self):
        self.viewer.update_scene()
        self.target_count_label.setText(f"Targets: {len(self.state.scene.targets)}")

    def reset_3d_view(self):
        self.viewer.reset_camera()
