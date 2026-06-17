#!/usr/bin/env bash

# Usage:
#   python_log path/to/script.py [args...]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: python_log script.py [args...]"
    exit 1
fi

PYTHON_SCRIPT="$1"
shift

# Remove .py extension
RELATIVE_PATH="${PYTHON_SCRIPT%.py}"
SCRIPT_REL_PATH="$PYTHON_SCRIPT"

# Log file preserving directory structure
LOG_FILE="./logs/${RELATIVE_PATH}.log"

# Create parent directories
mkdir -p "$(dirname "$LOG_FILE")"

TEE_MODE=""
NOW="$(date '+%Y-%m-%d %H:%M:%S %Z')"

if [ $# -gt 0 ]; then
    ARGS_STRING="$*"
else
    ARGS_STRING="(none)"
fi

# Check if log already exists
if [ -f "$LOG_FILE" ]; then
    echo "Log file already exists:"
    echo "  $LOG_FILE"
    echo
    echo "[1] Overwrite (default)"
    echo "[2] Append"
    echo "[3] Cancel"
    echo

    read -p "Choose an option [1-3] (default=1): " OPTION

    case "$OPTION" in
        ""|1)
            rm -f "$LOG_FILE"
            echo "Previous log deleted."
            TEE_MODE=""
            ;;
        2)
            echo "Appending to existing log."
            TEE_MODE="-a"
            ;;
        *)
            echo "Execution cancelled."
            exit 0
            ;;
    esac
fi

# In append mode, visually separate previous execution from the new one.
if [ "$TEE_MODE" = "-a" ]; then
    {
        echo
        echo "################################################################################"
        echo "########################### NEW EXECUTION - $NOW ###############################"
        echo "################################################################################"
    } >> "$LOG_FILE"
fi

echo "==================================================" | tee $TEE_MODE "$LOG_FILE"
echo "Running: $PYTHON_SCRIPT" | tee $TEE_MODE "$LOG_FILE"
echo "Date of execution: $NOW" | tee $TEE_MODE "$LOG_FILE"
echo "Script path (relative): $SCRIPT_REL_PATH" | tee $TEE_MODE "$LOG_FILE"
echo "Arguments: $ARGS_STRING" | tee $TEE_MODE "$LOG_FILE"
echo "Started at: $NOW" | tee $TEE_MODE "$LOG_FILE"
echo "==================================================" | tee $TEE_MODE "$LOG_FILE"

# Run python with live logging
python3 -u "$PYTHON_SCRIPT" "$@" 2>&1 | tee $TEE_MODE "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "==================================================" | tee -a "$LOG_FILE"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE