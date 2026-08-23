@echo off
REM Launch the PyHydroGeophysX professional Qt desktop studio (Windows).
REM Any extra arguments (e.g. --context <path> --module hydro_geophysics) pass through.
python -m PyHydroGeophysX.qt_apps.launcher %*
