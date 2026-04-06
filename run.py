#!/usr/bin/env python3
"""Quick-start launcher for LunaRad.

This launcher prefers the current interpreter when it already has the GUI
dependencies installed. If it does not, it will try to re-exec using a
compatible Conda/Miniforge Python automatically before giving up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)
REQUIRED_MODULES = ("PySide6", "numpy", "matplotlib")
REEXEC_GUARD_ENV = "LUNARAD_PEEK_REEXEC"


def _missing_modules() -> list[str]:
    missing: list[str] = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def _current_runtime_supported() -> bool:
    return sys.version_info >= MIN_PYTHON and not _missing_modules()


def _candidate_interpreters() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path:
            return
        normalized = str(Path(path).expanduser())
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    add(sys.executable)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        add(os.path.join(conda_prefix, "bin", "python"))
        add(os.path.join(conda_prefix, "bin", "python3"))

    for command in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        add(shutil.which(command))

    for path in (
        "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
        "/opt/homebrew/Caskroom/miniforge/base/bin/python",
        "/usr/local/Caskroom/miniforge/base/bin/python3",
        "/usr/local/Caskroom/miniforge/base/bin/python",
        "/opt/homebrew/anaconda3/bin/python3",
        "/opt/homebrew/anaconda3/bin/python",
        "/usr/local/anaconda3/bin/python3",
        "/usr/local/anaconda3/bin/python",
        str(Path.home() / "miniforge3" / "bin" / "python3"),
        str(Path.home() / "miniforge3" / "bin" / "python"),
    ):
        add(path)

    return candidates


def _interpreter_supports(candidate: str) -> bool:
    path = Path(candidate)
    if not path.exists():
        return False

    check_script = """
import sys

mods = ("PySide6", "numpy", "matplotlib")
missing = []
for mod in mods:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)

sys.exit(0 if sys.version_info >= (3, 10) and not missing else 1)
"""

    try:
        result = subprocess.run(
            [str(path), "-c", check_script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return result.returncode == 0


def _find_supported_interpreter() -> str | None:
    current = Path(sys.executable)

    for candidate in _candidate_interpreters():
        candidate_path = Path(candidate)
        if candidate_path == current:
            continue
        if _interpreter_supports(candidate):
            return candidate

    return None


def _ensure_supported_runtime() -> None:
    if _current_runtime_supported():
        return

    if os.environ.get(REEXEC_GUARD_ENV) != "1":
        replacement = _find_supported_interpreter()
        if replacement:
            print(f"Switching to compatible Python: {replacement}")
            os.environ[REEXEC_GUARD_ENV] = "1"
            os.execv(replacement, [replacement, *sys.argv])

    missing = _missing_modules()
    missing_display = ", ".join(missing) if missing else "project dependencies"

    print(f"Error: Missing or incompatible runtime for: {missing_display}")
    print()
    print("Your current Python is:", sys.executable, f"({sys.version.split()[0]})")
    print()
    print("This app needs Python 3.10+ with PySide6, numpy, and matplotlib.")
    print("Fix by running one of:")
    print("  conda activate base && python run.py")
    print("  /opt/homebrew/Caskroom/miniforge/base/bin/python3 run.py")
    sys.exit(1)


def main() -> None:
    _ensure_supported_runtime()

    sys.path.insert(0, str(PROJECT_ROOT))
    from lunarad_peek.app.main import main as app_main

    app_main()


if __name__ == "__main__":
    main()
