"""Materials tab for material library management."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFormLayout,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
    QLineEdit,
)
from PySide6.QtCore import Qt

from lunarad_peek.app.state import AppState
from lunarad_peek.materials.material import Material, CompositeMaterial, CompositeMode


class MaterialsTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: material list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(QLabel("Material Library"))

        self.material_list = QListWidget()
        self._populate_list()
        self.material_list.currentRowChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.material_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Custom")
        add_btn.clicked.connect(self._add_custom_material)
        btn_layout.addWidget(add_btn)

        duplicate_btn = QPushButton("Duplicate")
        duplicate_btn.clicked.connect(self._duplicate_material)
        btn_layout.addWidget(duplicate_btn)
        left_layout.addLayout(btn_layout)

        splitter.addWidget(left_widget)

        # Right: material editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Properties
        props_group = QGroupBox("Material Properties")
        props_form = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setReadOnly(True)
        props_form.addRow("Name:", self.name_edit)

        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.01, 20.0)
        self.density_spin.setDecimals(3)
        self.density_spin.setSuffix(" g/cm³")
        props_form.addRow("Density:", self.density_spin)

        self.porosity_spin = QDoubleSpinBox()
        self.porosity_spin.setRange(0.0, 0.99)
        self.porosity_spin.setDecimals(2)
        props_form.addRow("Porosity:", self.porosity_spin)

        props_group.setLayout(props_form)
        right_layout.addWidget(props_group)

        # Composition
        comp_group = QGroupBox("Elemental Composition (weight fractions)")
        comp_layout = QVBoxLayout()

        self.comp_table = QTableWidget(0, 2)
        self.comp_table.setHorizontalHeaderLabels(["Element", "Weight Fraction"])
        self.comp_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        comp_layout.addWidget(self.comp_table)

        comp_group.setLayout(comp_layout)
        right_layout.addWidget(comp_group)

        # Derived properties
        derived_group = QGroupBox("Derived Radiation Properties")
        derived_form = QFormLayout()

        self.zeff_label = QLabel("—")
        derived_form.addRow("Z_eff:", self.zeff_label)

        self.mean_a_label = QLabel("—")
        derived_form.addRow("Mean A:", self.mean_a_label)

        self.x0_label = QLabel("—")
        derived_form.addRow("X₀ (g/cm²):", self.x0_label)

        self.lambda_label = QLabel("—")
        derived_form.addRow("λ_I (g/cm²):", self.lambda_label)

        self.eff_density_label = QLabel("—")
        derived_form.addRow("Eff. Density:", self.eff_density_label)

        self.source_label = QLabel("—")
        self.source_label.setWordWrap(True)
        derived_form.addRow("Source:", self.source_label)

        derived_group.setLayout(derived_form)
        right_layout.addWidget(derived_group)

        # Apply button
        apply_btn = QPushButton("Apply Changes")
        apply_btn.setObjectName("runButton")
        apply_btn.clicked.connect(self._apply_changes)
        right_layout.addWidget(apply_btn)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        splitter.setSizes([300, 700])
        layout.addWidget(splitter)

        # Select first item
        if self.material_list.count() > 0:
            self.material_list.setCurrentRow(0)

    def _populate_list(self):
        self.material_list.clear()
        for mat_id, mat in self.state.material_library.items():
            item = QListWidgetItem(f"{mat.name}")
            item.setData(Qt.ItemDataRole.UserRole, mat_id)
            self.material_list.addItem(item)

    def _on_selection_changed(self, row: int):
        if row < 0:
            return

        item = self.material_list.item(row)
        mat_id = item.data(Qt.ItemDataRole.UserRole)
        mat = self.state.material_library.get(mat_id)
        if not mat:
            return

        self.name_edit.setText(mat.name)
        self.density_spin.setValue(mat.density)
        self.porosity_spin.setValue(mat.porosity)

        # Composition table
        self.comp_table.setRowCount(len(mat.composition))
        for i, (elem, wf) in enumerate(sorted(mat.composition.items())):
            self.comp_table.setItem(i, 0, QTableWidgetItem(elem))
            self.comp_table.setItem(i, 1, QTableWidgetItem(f"{wf:.4f}"))

        # Derived properties
        try:
            self.zeff_label.setText(f"{mat.Z_eff:.2f}")
            self.mean_a_label.setText(f"{mat.mean_A:.2f} g/mol")
            self.x0_label.setText(f"{mat.radiation_length_approx:.1f}")
            self.lambda_label.setText(f"{mat.nuclear_interaction_length:.1f}")
            self.eff_density_label.setText(f"{mat.effective_density:.3f} g/cm³")
            self.source_label.setText(mat.source or "—")
        except Exception:
            self.zeff_label.setText("—")

    def _add_custom_material(self):
        mat_id = f"custom_{len(self.state.material_library)}"
        mat = Material(
            name="Custom Material",
            density=1.0,
            composition={"Si": 0.5, "O": 0.5},
            description="User-defined material",
        )
        self.state.material_library[mat_id] = mat
        self._populate_list()
        self.material_list.setCurrentRow(self.material_list.count() - 1)

    def _duplicate_material(self):
        row = self.material_list.currentRow()
        if row < 0:
            return
        item = self.material_list.item(row)
        src_id = item.data(Qt.ItemDataRole.UserRole)
        src = self.state.material_library[src_id]

        new_id = f"{src_id}_copy"
        new_mat = Material(
            name=f"{src.name} (copy)",
            density=src.density,
            composition=dict(src.composition),
            porosity=src.porosity,
            description=src.description,
            source=src.source,
        )
        self.state.material_library[new_id] = new_mat
        self._populate_list()

    def _apply_changes(self):
        row = self.material_list.currentRow()
        if row < 0:
            return
        item = self.material_list.item(row)
        mat_id = item.data(Qt.ItemDataRole.UserRole)

        # Read composition from table
        comp = {}
        for i in range(self.comp_table.rowCount()):
            elem_item = self.comp_table.item(i, 0)
            wf_item = self.comp_table.item(i, 1)
            if elem_item and wf_item:
                try:
                    comp[elem_item.text()] = float(wf_item.text())
                except ValueError:
                    pass

        mat = Material(
            name=self.name_edit.text(),
            density=self.density_spin.value(),
            composition=comp,
            porosity=self.porosity_spin.value(),
            source=self.state.material_library[mat_id].source,
        )
        self.state.update_material(mat_id, mat)
        self._on_selection_changed(row)
