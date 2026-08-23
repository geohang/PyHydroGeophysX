#!/usr/bin/env bash
# Launch the PyHydroGeophysX professional Qt desktop studio (macOS / Linux).
# Any extra arguments (e.g. --context <path> --module hydro_geophysics) pass through.
exec python -m PyHydroGeophysX.qt_apps.launcher "$@"
