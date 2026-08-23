"""PyHydroGeophysX professional Qt desktop studio.

This subpackage is intentionally lightweight at import time: importing
``PyHydroGeophysX.qt_apps`` must NOT pull in PySide6, pyqtgraph, pygimli, or any
other heavy GUI/geophysics dependency. The Streamlit app only needs to know this
package *exists* so it can launch ``python -m PyHydroGeophysX.qt_apps.launcher``
as a separate process. All heavy imports happen inside the launcher / module
files, which only run in the dedicated desktop process.
"""

__all__ = ["launcher", "main_window", "state", "io_utils", "hydro_pipeline"]

__version__ = "0.1.0"
