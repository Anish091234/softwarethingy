"""Main entry point for LunaRad-PEEK application."""

import os
import sys
from pathlib import Path


def _debug_enabled() -> bool:
    return os.environ.get("LUNARAD_DEBUG_STARTUP") == "1"


def _debug(message: str) -> None:
    if _debug_enabled():
        print(f"[LunaRad startup] {message}", flush=True)


def main():
    """Launch the LunaRad-PEEK desktop application."""
    if _debug_enabled():
        import faulthandler

        faulthandler.enable()

    _debug("Importing QApplication")
    from PySide6.QtWidgets import QApplication

    _debug("Importing MainWindow")
    from lunarad_peek.ui.main_window import MainWindow

    _debug("Creating QApplication")
    app = QApplication(sys.argv)
    app.setApplicationName("LunaRad-PEEK")
    app.setApplicationVersion("1.0.0-alpha")
    app.setOrganizationName("LunaRad Research")

    _debug("Configuring application font")
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    _debug("Loading stylesheet")
    app.setStyleSheet(_load_stylesheet())

    _debug("Constructing MainWindow")
    window = MainWindow()
    _debug("Showing MainWindow")
    window.show()

    _debug("Entering Qt event loop")
    sys.exit(app.exec())


def _load_stylesheet() -> str:
    """Load the application stylesheet."""
    style_path = Path(__file__).parent.parent / "ui" / "style.qss"
    if style_path.exists():
        return style_path.read_text()
    return _default_stylesheet()


def _default_stylesheet() -> str:
    return """
    QMainWindow {
        background-color: #1e1e2e;
    }
    QTabWidget::pane {
        border: 1px solid #45475a;
        background-color: #1e1e2e;
    }
    QTabBar::tab {
        background-color: #313244;
        color: #cdd6f4;
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #45475a;
        color: #f5e0dc;
    }
    QGroupBox {
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 4px;
        margin-top: 1em;
        padding-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QLabel {
        color: #cdd6f4;
    }
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        border: 1px solid #585b70;
        border-radius: 4px;
        padding: 6px 16px;
        min-height: 24px;
    }
    QPushButton:hover {
        background-color: #585b70;
    }
    QPushButton:pressed {
        background-color: #6c7086;
    }
    QPushButton#runButton {
        background-color: #a6e3a1;
        color: #1e1e2e;
        font-weight: bold;
    }
    QPushButton#runButton:hover {
        background-color: #94e2d5;
    }
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 3px;
        padding: 4px;
    }
    QSlider::groove:horizontal {
        height: 4px;
        background: #45475a;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #89b4fa;
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }
    QProgressBar {
        border: 1px solid #45475a;
        border-radius: 3px;
        background-color: #313244;
        text-align: center;
        color: #cdd6f4;
    }
    QProgressBar::chunk {
        background-color: #89b4fa;
        border-radius: 2px;
    }
    QTreeWidget, QTableWidget, QListWidget {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        alternate-background-color: #363849;
    }
    QHeaderView::section {
        background-color: #45475a;
        color: #cdd6f4;
        padding: 4px;
        border: 1px solid #585b70;
    }
    QMenuBar {
        background-color: #181825;
        color: #cdd6f4;
    }
    QMenuBar::item:selected {
        background-color: #45475a;
    }
    QMenu {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
    }
    QMenu::item:selected {
        background-color: #45475a;
    }
    QStatusBar {
        background-color: #181825;
        color: #a6adc8;
    }
    QToolBar {
        background-color: #181825;
        border-bottom: 1px solid #45475a;
        spacing: 4px;
    }
    QSplitter::handle {
        background-color: #45475a;
    }
    QCheckBox {
        color: #cdd6f4;
    }
    QCheckBox::indicator:checked {
        background-color: #89b4fa;
        border: 1px solid #89b4fa;
        border-radius: 2px;
    }
    """


if __name__ == "__main__":
    main()
