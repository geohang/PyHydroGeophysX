#!/usr/bin/env bash
# PyHydroGeophysX Web Application Launcher (Linux/macOS)
#
# On macOS, copy this file to start_webapp.command to make it double-clickable
# from Finder.

set -u

# Resolve paths from this file, not from wherever the shell happened to start.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_FILE="${REPO_DIR}/examples/app_geophysics_workflow.py"
VENV_DIR="${REPO_DIR}/.venv-webapp"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_EXE=""
FALLBACK_EXE=""

echo "========================================"
echo "PyHydroGeophysX Web Interface"
echo "========================================"
echo ""

if [ ! -f "${APP_FILE}" ]; then
    echo "[ERROR] The web application was not found:"
    echo "        ${APP_FILE}"
    echo ""
    echo "Keep start_webapp.sh in the PyHydroGeophysX package folder of the"
    echo "downloaded PyHydroGeophysX source package."
    exit 1
fi

try_python() {
    local candidate="$1"
    [ -n "${candidate}" ] && [ -x "${candidate}" ] || return 0
    "${candidate}" -c \
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" \
        >/dev/null 2>&1 || return 0

    if [ -z "${FALLBACK_EXE}" ]; then
        FALLBACK_EXE="${candidate}"
    fi

    if "${candidate}" -c "import streamlit" >/dev/null 2>&1; then
        PYTHON_EXE="${candidate}"
    fi
}

# Use the same discovery order as the desktop workbench: the reusable launcher
# environment, an explicitly selected interpreter, an active Conda environment,
# and finally Python commands available on PATH.
try_python "${VENV_DIR}/bin/python"

if [ -z "${PYTHON_EXE}" ] && [ -n "${PYHYDROGEOPHYSX_PYTHON:-}" ]; then
    try_python "${PYHYDROGEOPHYSX_PYTHON}"
fi

if [ -z "${PYTHON_EXE}" ] && [ -n "${CONDA_PREFIX:-}" ]; then
    try_python "${CONDA_PREFIX}/bin/python"
fi

if [ -z "${PYTHON_EXE}" ]; then
    try_python "$(command -v python3 2>/dev/null || true)"
fi
if [ -z "${PYTHON_EXE}" ]; then
    try_python "$(command -v python 2>/dev/null || true)"
fi

# GUI launchers do not always inherit Conda's PATH. Search common environment
# roots as a final discovery step, just like the Windows launcher.
if [ -z "${PYTHON_EXE}" ]; then
    for env_root in \
        "${HOME}/.conda/envs" \
        "${HOME}/miniconda3/envs" \
        "${HOME}/anaconda3/envs" \
        "/opt/miniconda3/envs" \
        "/opt/anaconda3/envs"
    do
        [ -d "${env_root}" ] || continue
        for candidate in "${env_root}"/*/bin/python; do
            [ -e "${candidate}" ] || continue
            try_python "${candidate}"
            [ -z "${PYTHON_EXE}" ] || break 2
        done
    done
fi

# If Python exists but Streamlit does not, create one isolated, reusable
# environment rather than modifying any of the user's existing environments.
if [ -z "${PYTHON_EXE}" ]; then
    if [ -z "${FALLBACK_EXE}" ]; then
        echo "[ERROR] Python 3.8 or newer was not found."
        echo ""
        echo "Install Python 3, then run this launcher again."
        echo "Set PYHYDROGEOPHYSX_PYTHON to choose an interpreter explicitly."
        exit 1
    fi

    echo "Streamlit is not installed in an available Python environment."
    echo "Creating a reusable web-app environment:"
    echo "  ${VENV_DIR}"
    echo ""
    "${FALLBACK_EXE}" -m venv "${VENV_DIR}" || {
        echo "[ERROR] Python could not create the web-app environment."
        exit 1
    }

    echo "Installing PyHydroGeophysX web-app dependencies..."
    echo "This one-time setup can take several minutes."
    echo ""
    "${VENV_DIR}/bin/python" -m pip install --disable-pip-version-check \
        -e "${REPO_DIR}[webapp]" || {
        echo "[ERROR] Dependency installation failed. Check the messages above"
        echo "        and your internet connection, then run this file again."
        exit 1
    }
    PYTHON_EXE="${VENV_DIR}/bin/python"
fi

echo "Using Python:"
"${PYTHON_EXE}" -c "import sys; print('  ' + sys.executable)"
echo ""
echo "Opening the web app at http://localhost:8501"
echo "Keep this terminal open while using the app."
echo "Press Ctrl+C here to stop the server."
echo "========================================"
echo ""

# Used by automated checks; a normal run never sets this variable.
if [ "${PHGX_WEBAPP_DRY_RUN:-}" = "1" ]; then
    exit 0
fi

exec "${PYTHON_EXE}" -m streamlit run "${APP_FILE}" \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
