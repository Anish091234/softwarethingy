#!/bin/zsh

set -u

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR" || exit 1

OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/gui-launch.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching LunaRad from $SCRIPT_DIR"

/usr/bin/env python3 -u "$SCRIPT_DIR/run.py"
exit_code=$?

if [[ $exit_code -ne 0 ]]; then
  echo
  echo "LunaRad failed to launch (exit code: $exit_code)."
  echo "Log file: $LOG_FILE"
  echo "Press Return to close this window."
  read -r _
fi

exit $exit_code
