"""3D viewer widget for habitat geometry visualization.

Uses raw VTK-in-Qt on macOS because the pyvistaqt wrapper can freeze the Qt
event loop on some Macs. On other platforms we still prefer pyvistaqt, with
raw VTK as a fallback if needed.
"""

from __future__ import annotations

import os
import platform

import numpy as np
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from lunarad_peek.app.state import AppState

# Material colors for 3D rendering (RGB tuples 0-1)
LAYER_COLORS = {
    "highland_regolith": (0.976, 0.886, 0.686),
    "mare_regolith": (0.980, 0.702, 0.529),
    "lavatube_rock": (0.651, 0.678, 0.784),
    "peek": (0.537, 0.706, 0.980),
    "regolith_peek_composite": (0.651, 0.890, 0.631),
    "aluminum": (0.804, 0.839, 0.957),
}
DEFAULT_COLOR = (0.7, 0.7, 0.8)
TARGET_COLOR = (0.953, 0.549, 0.659)
EDGE_COLOR = (0.82, 0.86, 0.94)
BACKGROUND_COLOR = (30 / 255, 30 / 255, 46 / 255)


class _OffscreenCanvas(QLabel):
    """Qt label that forwards mouse interactions to the parent viewer."""

    def __init__(self, viewer: "Viewer3D"):
        super().__init__(viewer)
        self._viewer = viewer
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #313244;")

    def mousePressEvent(self, event):
        self._viewer._offscreen_mouse_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._viewer._offscreen_mouse_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._viewer._offscreen_mouse_release(event)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self._viewer._offscreen_wheel(event)
        super().wheelEvent(event)


def _viewer_backend_candidates() -> list[str]:
    """Return preferred 3D backends in priority order."""
    if os.environ.get("LUNARAD_DISABLE_3D") == "1":
        return []

    forced_backend = os.environ.get("LUNARAD_3D_BACKEND", "").strip().lower()
    if forced_backend:
        return [forced_backend]

    if platform.system() == "Darwin":
        return ["pyvista-offscreen", "vtk", "pyvistaqt"]

    return ["pyvistaqt", "vtk"]


class Viewer3D(QWidget):
    """3D viewport for scene visualization."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._backend = "fallback"
        self._plotter = None
        self._vtk_widget = None
        self._vtk_renderer = None
        self._axes_widget = None
        self._offscreen_label = None
        self._ray_actors: list[object] = []
        self._scene_update_pending = False
        self._scene_has_content = False
        self._camera_initialized = False
        self._last_drag_pos: QPoint | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        last_error: Exception | None = None
        for backend in _viewer_backend_candidates():
            try:
                if backend == "pyvista-offscreen":
                    self._init_pyvista_offscreen_backend(layout)
                elif backend == "vtk":
                    self._init_vtk_backend(layout)
                elif backend == "pyvistaqt":
                    self._init_pyvistaqt_backend(layout)
                else:
                    raise RuntimeError(f"Unknown 3D backend: {backend}")
            except Exception as exc:
                last_error = exc
                continue

            self._backend = backend
            self._has_3d = True
            return

        self._has_3d = False
        self._add_fallback_label(layout, last_error)

    def _init_pyvistaqt_backend(self, layout: QVBoxLayout) -> None:
        from pyvistaqt import QtInteractor

        self._plotter = QtInteractor(self, auto_update=False, multi_samples=0)
        self._plotter.set_background("#1e1e2e")
        layout.addWidget(self._plotter.interactor)

    def _init_pyvista_offscreen_backend(self, layout: QVBoxLayout) -> None:
        import pyvista as pv

        self._plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
        self._plotter.set_background("#1e1e2e")
        self._offscreen_label = _OffscreenCanvas(self)
        self._offscreen_label.setText(
            "Generate geometry to preview in 3D.\n\nDrag to orbit, scroll to zoom."
        )
        self._offscreen_label.setStyleSheet(
            "background-color: #1e1e2e; border: 1px solid #313244; "
            "color: #6c7086; font-size: 13px; padding: 24px;"
        )
        layout.addWidget(self._offscreen_label)

    def _init_vtk_backend(self, layout: QVBoxLayout) -> None:
        # Imported for side effects required by VTK's Qt/OpenGL pipeline.
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
        from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
        from vtkmodules.vtkRenderingCore import vtkRenderer

        self._vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self._vtk_widget)

        self._vtk_renderer = vtkRenderer()
        self._vtk_renderer.SetBackground(*BACKGROUND_COLOR)

        render_window = self._vtk_widget.GetRenderWindow()
        render_window.SetMultiSamples(0)
        render_window.AddRenderer(self._vtk_renderer)

        interactor = render_window.GetInteractor()
        axes_actor = vtkAxesActor()
        self._axes_widget = vtkOrientationMarkerWidget()
        self._axes_widget.SetOrientationMarker(axes_actor)
        self._axes_widget.SetInteractor(interactor)
        self._axes_widget.SetViewport(0.0, 0.0, 0.18, 0.18)
        self._axes_widget.SetEnabled(1)
        self._axes_widget.InteractiveOff()

    def _add_fallback_label(
        self, layout: QVBoxLayout, last_error: Exception | None = None
    ) -> None:
        from PySide6.QtCore import Qt as QtCoreQt

        if os.environ.get("LUNARAD_DISABLE_3D") == "1":
            message = "Interactive 3D viewer disabled by LUNARAD_DISABLE_3D=1."
        elif last_error is not None:
            message = str(last_error).strip() or "Interactive 3D viewer unavailable."
        else:
            message = "Interactive 3D viewer unavailable."

        fallback_label = QLabel(
            f"{message}\n\nGeometry will still be generated for analysis."
        )
        fallback_label.setAlignment(QtCoreQt.AlignmentFlag.AlignCenter)
        fallback_label.setStyleSheet("color: #6c7086; font-size: 13px; padding: 40px;")
        layout.addWidget(fallback_label)

    def _mesh_to_vtk_polydata(self, mesh):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)

        points = vtkPoints()
        points.SetData(numpy_to_vtk(vertices, deep=True))

        connectivity = np.column_stack(
            [np.full(mesh.num_faces, 3, dtype=np.int64), faces]
        ).ravel()
        cells = vtkCellArray()
        cells.SetCells(mesh.num_faces, numpy_to_vtkIdTypeArray(connectivity, deep=True))

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        return poly

    def _add_vtk_mesh(
        self,
        polydata,
        *,
        color: tuple[float, float, float],
        opacity: float,
        show_edges: bool = True,
        line_width: float = 0.5,
    ):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().LightingOff()
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().SetSpecular(0.0)
        if show_edges:
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(*EDGE_COLOR)
            actor.GetProperty().SetLineWidth(max(line_width, 1.0))

        self._vtk_renderer.AddActor(actor)
        return actor

    def _add_vtk_sphere(
        self,
        *,
        center: np.ndarray,
        radius: float,
        color: tuple[float, float, float],
        opacity: float,
    ):
        from vtkmodules.vtkFiltersSources import vtkSphereSource

        source = vtkSphereSource()
        source.SetCenter(*[float(v) for v in center])
        source.SetRadius(radius)
        source.SetThetaResolution(24)
        source.SetPhiResolution(24)
        source.Update()
        return self._add_vtk_mesh(
            source.GetOutput(),
            color=color,
            opacity=opacity,
            show_edges=False,
        )

    def _add_vtk_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        color: tuple[float, float, float],
        opacity: float,
        line_width: float,
    ):
        from vtkmodules.vtkFiltersSources import vtkLineSource

        source = vtkLineSource()
        source.SetPoint1(*[float(v) for v in start])
        source.SetPoint2(*[float(v) for v in end])
        source.Update()
        return self._add_vtk_mesh(
            source.GetOutput(),
            color=color,
            opacity=opacity,
            show_edges=False,
            line_width=line_width,
        )

    def _render_vtk_scene(self) -> None:
        if self._vtk_widget is not None:
            if platform.system() == "Darwin" and self._vtk_widget.isVisible():
                self._vtk_widget.Render()
                return
            self._vtk_widget.GetRenderWindow().Render()

    def _refresh_offscreen_render(self, reset_camera: bool = False) -> None:
        if not self._plotter or not self._offscreen_label:
            return

        width = max(self._offscreen_label.width(), 640)
        height = max(self._offscreen_label.height(), 480)
        self._plotter.window_size = (width, height)

        if reset_camera or not self._camera_initialized:
            self._plotter.view_isometric()
            self._plotter.reset_camera()
            self._camera_initialized = True
        else:
            self._plotter.renderer.ResetCameraClippingRange()

        image = np.ascontiguousarray(self._plotter.screenshot(return_img=True))
        qimage = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.shape[1] * image.shape[2],
            QImage.Format.Format_RGB888,
        ).copy()
        self._offscreen_label.setPixmap(QPixmap.fromImage(qimage))
        self._offscreen_label.setText("")

    def update_scene(self):
        """Redraw the scene from current state."""
        if not self._has_3d:
            return

        if self._scene_update_pending:
            return

        self._scene_update_pending = True
        QTimer.singleShot(0, self._perform_scene_update)

    def _perform_scene_update(self) -> None:
        self._scene_update_pending = False

        self._scene_has_content = bool(
            self.state.scene.all_layers() or self.state.scene.targets
        )

        if self._backend == "pyvista-offscreen":
            self._update_scene_pyvista_offscreen()
            return

        if self._backend == "pyvistaqt":
            self._update_scene_pyvistaqt()
            return

        if self._backend == "vtk":
            self._update_scene_vtk()

    def _update_scene_pyvista_offscreen(self) -> None:
        import pyvista as pv

        self._plotter.clear()

        if not self._scene_has_content:
            if self._offscreen_label is not None:
                self._offscreen_label.setPixmap(QPixmap())
                self._offscreen_label.setText(
                    "Generate geometry to preview in 3D.\n\nDrag to orbit, scroll to zoom."
                )
            return

        for layer in self.state.scene.all_layers():
            mesh = layer.mesh
            if mesh.num_faces == 0:
                continue

            faces_pv = np.column_stack([np.full(mesh.num_faces, 3), mesh.faces]).ravel()
            poly = pv.PolyData(mesh.vertices, faces_pv)

            color = LAYER_COLORS.get(layer.material_id, DEFAULT_COLOR)
            opacity = 0.4 if not layer.is_habitat_wall else 0.7
            if layer.is_terrain:
                opacity = 0.3
                color = (0.4, 0.4, 0.35)

            self._plotter.add_mesh(
                poly,
                color=color,
                opacity=opacity,
                show_edges=True,
                edge_color=EDGE_COLOR,
                line_width=1.0,
            )

        for target in self.state.scene.targets:
            pos = target.position
            sphere = pv.Sphere(radius=0.15, center=pos + np.array([0, 0, 1.0]))
            self._plotter.add_mesh(sphere, color=TARGET_COLOR, opacity=0.95)

            head = pv.Sphere(radius=0.1, center=pos + np.array([0, 0, 1.65]))
            self._plotter.add_mesh(head, color=TARGET_COLOR, opacity=0.95)

            for _, dp_pos in target.world_dosimetry_points():
                dot = pv.Sphere(radius=0.03, center=dp_pos)
                self._plotter.add_mesh(dot, color=(1.0, 1.0, 0.3), opacity=0.95)

        self._refresh_offscreen_render(reset_camera=True)

    def _update_scene_pyvistaqt(self) -> None:
        import pyvista as pv

        self._plotter.clear()

        for layer in self.state.scene.all_layers():
            mesh = layer.mesh
            if mesh.num_faces == 0:
                continue

            faces_pv = np.column_stack([np.full(mesh.num_faces, 3), mesh.faces]).ravel()
            poly = pv.PolyData(mesh.vertices, faces_pv)

            color = LAYER_COLORS.get(layer.material_id, DEFAULT_COLOR)
            opacity = 0.4 if not layer.is_habitat_wall else 0.7
            if layer.is_terrain:
                opacity = 0.3
                color = (0.4, 0.4, 0.35)

            self._plotter.add_mesh(
                poly,
                color=color,
                opacity=opacity,
                show_edges=True,
                edge_color=EDGE_COLOR,
                line_width=0.5,
                label=layer.name,
            )

        for target in self.state.scene.targets:
            pos = target.position
            sphere = pv.Sphere(radius=0.15, center=pos + np.array([0, 0, 1.0]))
            self._plotter.add_mesh(sphere, color=TARGET_COLOR, opacity=0.9)

            head = pv.Sphere(radius=0.1, center=pos + np.array([0, 0, 1.65]))
            self._plotter.add_mesh(head, color=TARGET_COLOR, opacity=0.9)

            for _, dp_pos in target.world_dosimetry_points():
                dot = pv.Sphere(radius=0.03, center=dp_pos)
                self._plotter.add_mesh(dot, color=(1.0, 1.0, 0.3), opacity=0.9)

        self._plotter.add_axes()
        self._plotter.reset_camera()
        self._plotter.render()

    def _update_scene_vtk(self) -> None:
        self._vtk_renderer.RemoveAllViewProps()
        self._ray_actors = []

        for layer in self.state.scene.all_layers():
            mesh = layer.mesh
            if mesh.num_faces == 0:
                continue

            poly = self._mesh_to_vtk_polydata(mesh)
            color = LAYER_COLORS.get(layer.material_id, DEFAULT_COLOR)
            opacity = 0.4 if not layer.is_habitat_wall else 0.7
            if layer.is_terrain:
                opacity = 0.3
                color = (0.4, 0.4, 0.35)

            self._add_vtk_mesh(poly, color=color, opacity=opacity)

        for target in self.state.scene.targets:
            pos = target.position
            self._add_vtk_sphere(
                center=pos + np.array([0, 0, 1.0]),
                radius=0.15,
                color=TARGET_COLOR,
                opacity=0.9,
            )
            self._add_vtk_sphere(
                center=pos + np.array([0, 0, 1.65]),
                radius=0.1,
                color=TARGET_COLOR,
                opacity=0.9,
            )

            for _, dp_pos in target.world_dosimetry_points():
                self._add_vtk_sphere(
                    center=dp_pos,
                    radius=0.03,
                    color=(1.0, 1.0, 0.3),
                    opacity=0.9,
                )

        self._vtk_renderer.ResetCamera()
        self._render_vtk_scene()

    def reset_camera(self):
        if not self._has_3d:
            return

        if self._backend == "pyvista-offscreen" and self._plotter:
            self._refresh_offscreen_render(reset_camera=True)
        elif self._backend == "pyvistaqt" and self._plotter:
            self._plotter.reset_camera()
            self._plotter.render()
        elif self._backend == "vtk" and self._vtk_renderer is not None:
            self._vtk_renderer.ResetCamera()
            self._render_vtk_scene()

    def show_ray_visualization(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        values: np.ndarray | None = None,
    ):
        """Overlay ray lines from a target point (for debugging/visualization)."""
        if not self._has_3d:
            return

        if self._backend == "pyvista-offscreen":
            self._show_ray_visualization_pyvista_offscreen(origin, directions, values)
            return

        if self._backend == "pyvistaqt":
            self._show_ray_visualization_pyvistaqt(origin, directions, values)
            return

        if self._backend == "vtk":
            self._show_ray_visualization_vtk(origin, directions, values)

    def _show_ray_visualization_pyvista_offscreen(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        values: np.ndarray | None = None,
    ) -> None:
        import pyvista as pv

        max_len = 10.0
        for i in range(min(len(directions), 50)):
            d = directions[i]
            end = origin + d * max_len
            line = pv.Line(origin, end)

            color = (0.5, 0.5, 0.5)
            if values is not None and len(values) > i:
                norm_val = min(values[i] / 100.0, 1.0)
                color = (1.0 - norm_val, 0.2, norm_val)

            self._plotter.add_mesh(line, color=color, opacity=0.35, line_width=1)

        self._refresh_offscreen_render(reset_camera=False)

    def _show_ray_visualization_pyvistaqt(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        values: np.ndarray | None = None,
    ) -> None:
        import pyvista as pv

        max_len = 10.0
        for i in range(min(len(directions), 50)):
            d = directions[i]
            end = origin + d * max_len
            line = pv.Line(origin, end)

            color = (0.5, 0.5, 0.5)
            if values is not None and len(values) > i:
                norm_val = min(values[i] / 100.0, 1.0)
                color = (1.0 - norm_val, 0.2, norm_val)

            self._plotter.add_mesh(line, color=color, opacity=0.3, line_width=1)

    def _show_ray_visualization_vtk(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        values: np.ndarray | None = None,
    ) -> None:
        for actor in self._ray_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._ray_actors = []

        max_len = 10.0
        for i in range(min(len(directions), 50)):
            d = directions[i]
            end = origin + d * max_len

            color = (0.5, 0.5, 0.5)
            if values is not None and len(values) > i:
                norm_val = min(values[i] / 100.0, 1.0)
                color = (1.0 - norm_val, 0.2, norm_val)

            actor = self._add_vtk_line(
                origin,
                end,
                color=color,
                opacity=0.3,
                line_width=1.0,
            )
            self._ray_actors.append(actor)

        self._render_vtk_scene()

    def closeEvent(self, event):
        if self._backend in {"pyvista-offscreen", "pyvistaqt"} and self._plotter:
            self._plotter.close()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._backend == "pyvista-offscreen" and self._scene_has_content:
            QTimer.singleShot(0, lambda: self._refresh_offscreen_render(False))

    def _offscreen_mouse_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_drag_pos = event.position().toPoint()

    def _offscreen_mouse_move(self, event) -> None:
        if (
            self._backend != "pyvista-offscreen"
            or not self._plotter
            or not self._scene_has_content
            or self._last_drag_pos is None
        ):
            return

        current_pos = event.position().toPoint()
        dx = current_pos.x() - self._last_drag_pos.x()
        dy = current_pos.y() - self._last_drag_pos.y()
        self._last_drag_pos = current_pos

        if dx == 0 and dy == 0:
            return

        camera = self._plotter.camera
        camera.Azimuth(-dx * 0.5)
        camera.Elevation(dy * 0.5)
        camera.OrthogonalizeViewUp()
        self._refresh_offscreen_render(reset_camera=False)

    def _offscreen_mouse_release(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_drag_pos = None

    def _offscreen_wheel(self, event) -> None:
        if self._backend != "pyvista-offscreen" or not self._plotter or not self._scene_has_content:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_factor = 1.1 if delta > 0 else 0.9
        self._plotter.camera.Zoom(zoom_factor)
        self._refresh_offscreen_render(reset_camera=False)
