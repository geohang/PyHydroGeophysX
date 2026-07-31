#!/bin/bash
# PyHydroGeophysX Desktop Workbench Launcher (Linux/macOS)
# Starts the Qt desktop application. Extra arguments pass through, for example:
#   ./start_workbench.sh --module hydro_geophysics
#
# On macOS, copy this file to start_workbench.command to make it double-clickable
# from Finder.

set -u

# Resolve paths from this file, not from wherever the shell happened to start.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "========================================"
echo "PyHydroGeophysX Desktop Workbench"
echo "========================================"
echo ""

if [ ! -f "${REPO_DIR}/PyHydroGeophysX/qt_apps/launcher.py" ]; then
    echo "[ERROR] The workbench was not found under:"
    echo "        ${REPO_DIR}"
    echo ""
    echo "Keep start_workbench.sh in the examples folder of the downloaded"
    echo "PyHydroGeophysX source package."
    exit 1
fi

# Prefer an explicit interpreter, then an active conda environment, then PATH.
for CANDIDATE in \
    "${PYHYDROGEOPHYSX_PYTHON:-}" \
    "${CONDA_PREFIX:-}/bin/python" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"
do
    [ -n "${CANDIDATE}" ] && [ -x "${CANDIDATE}" ] || continue
    if "${CANDIDATE}" -c "import PySide6, pyqtgraph" >/dev/null 2>&1; then
        PYTHON_EXE="${CANDIDATE}"
        break
    fi
done

if [ -z "${PYTHON_EXE:-}" ]; then
    echo "[ERROR] No Python with the desktop dependencies was found."
    echo ""
    echo "Install them into the environment you use, then run this again:"
    echo "    pip install -e \"${REPO_DIR}[desktop]\""
    echo ""
    echo "Set PYHYDROGEOPHYSX_PYTHON to choose an interpreter explicitly."
    exit 1
fi

echo "Using Python:"
echo "  ${PYTHON_EXE}"
echo ""
echo "Starting the workbench. Keep this terminal open while using the app:"
echo "it carries the startup log and any error message."
echo "========================================"
echo ""

# Used by automated checks; a normal run never sets this variable.
if [ "${PHGX_WORKBENCH_DRY_RUN:-}" = "1" ]; then
    exit 0
fi

exec "${PYTHON_EXE}" -m PyHydroGeophysX.qt_apps.launcher "$@"
