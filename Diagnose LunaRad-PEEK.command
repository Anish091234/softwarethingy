#!/bin/zsh

set -u
set -o pipefail

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR" || exit 1

OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/gui-diagnose.log"

exec > >(tee "$LOG_FILE") 2>&1

echo "=== LunaRad-PEEK GUI Diagnostics ==="
echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "cwd: $SCRIPT_DIR"
echo "shell: $SHELL"
echo

echo "--- PATH ---"
echo "$PATH"
echo

echo "--- python3 discovery ---"
which -a python3 || true
echo

echo "--- python3 version ---"
/usr/bin/env python3 --version || true
echo

echo "--- launcher interpreter probe ---"
/usr/bin/env python3 -u - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import run

print("current python:", sys.executable)
print("current supported:", run._current_runtime_supported())
print("recommended interpreter:", run._find_supported_interpreter())
PY
echo

PYTHON_BIN=$(
/usr/bin/env python3 -u - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import run

if run._current_runtime_supported():
    print(sys.executable)
else:
    print(run._find_supported_interpreter() or sys.executable)
PY
)

echo "--- chosen interpreter ---"
echo "$PYTHON_BIN"
echo

echo "--- chosen interpreter version ---"
"$PYTHON_BIN" --version || true
echo

echo "--- stage-by-stage GUI construction ---"
"$PYTHON_BIN" -u -X faulthandler - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

print("stage: import QApplication")
from PySide6.QtWidgets import QApplication

print("stage: create QApplication")
app = QApplication([])

print("stage: import MainWindow")
from lunarad_peek.ui.main_window import MainWindow

print("stage: create MainWindow")
window = MainWindow()

print("stage: show MainWindow")
window.show()

print("result: show() completed successfully")
PY
echo

echo "--- short event-loop test ---"
LUNARAD_DEBUG_STARTUP=1 "$PYTHON_BIN" -u -X faulthandler - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from lunarad_peek.ui.main_window import MainWindow

print("stage: create QApplication")
app = QApplication([])

print("stage: create MainWindow")
window = MainWindow()
window.show()

print("stage: enter event loop for 1500 ms")
QTimer.singleShot(1500, app.quit)
code = app.exec()
print("event loop exited with:", code)
PY
echo

echo "--- full launcher attempt ---"
LUNARAD_DEBUG_STARTUP=1 /usr/bin/env python3 -u "$SCRIPT_DIR/run.py"
exit_code=$?
echo
echo "launcher exit code: $exit_code"
echo "diagnostic log written to: $LOG_FILE"
echo
echo "Press Return to close this window."
read -r _

exit $exit_code
