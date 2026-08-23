"""Reusable Qt widgets for the studio (PySide6 + pyqtgraph).

Importing this package pulls in PySide6 and pyqtgraph, so only do it inside the
desktop process (never from the Streamlit app).
"""

from PyHydroGeophysX.qt_apps.widgets.log_panel import LogPanel
from PyHydroGeophysX.qt_apps.widgets.project_tree import ProjectTree
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.curve_viewer import CurveViewer

__all__ = ["LogPanel", "ProjectTree", "ArrayViewer", "CurveViewer"]
