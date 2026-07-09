"""Main application window for the PyHydroGeophysX professional workbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.agent.chat_panel import AquahChatPanel
from PyHydroGeophysX.qt_apps.agent.controller import WorkbenchController
from PyHydroGeophysX.qt_apps.modules import build_module
from PyHydroGeophysX.qt_apps.modules.base import BaseModule
from PyHydroGeophysX.qt_apps.state import WorkbenchState
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.log_panel import LogPanel
from PyHydroGeophysX.qt_apps.widgets.project_tree import ProjectTree

WINDOW_TITLE = "PyHydroGeophysX Professional Workbench"


class PyHydroGeophysXWorkbench(QMainWindow):
    """Top-level window: tree | stacked modules | properties, with a log dock."""

    def __init__(self, context_path: Optional[str] = None, initial_module: str = "home") -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.state = WorkbenchState.from_context(context_path)
        if initial_module and initial_module != "home":
            self.state.selected_module = initial_module
        self._pages: Dict[str, BaseModule] = {}

        # Bottom: log panel (built first so module construction can log).
        self._log_panel = LogPanel()
        self._log_dock = self._make_dock("Log", self._log_panel, Qt.BottomDockWidgetArea)

        # Center: branded header + stacked module pages.
        self._stack = QStackedWidget()
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.addWidget(self._stack)
        outer.addWidget(content, stretch=1)
        self.setCentralWidget(container)

        # Left: navigator tree.
        self._tree = ProjectTree()
        self._tree.moduleSelected.connect(self.show_module)
        self._tree_dock = self._make_dock("Project", self._tree, Qt.LeftDockWidgetArea)

        # Right: AQUAH chat assistant + properties summary, in a tabbed dock.
        self._properties = QTextEdit()
        self._properties.setReadOnly(True)
        self._controller = WorkbenchController(self)
        self._chat = AquahChatPanel(self._controller, self.log)
        right_tabs = QTabWidget()
        right_tabs.addTab(self._chat, "AQUAH Chat")
        right_tabs.addTab(self._properties, "Properties")
        right_tabs.setMinimumWidth(440)
        self._properties_dock = self._make_dock("Assistant", right_tabs, Qt.RightDockWidgetArea)

        self._build_menus()
        self._build_toolbar()
        self._geometry_restored = self._restore_window_settings()

        self._status_label = QLabel("Ready")
        self.statusBar().addWidget(self._status_label)

        self.log(f"Workbench started. Context: {self.state.context_path or '(none)'}", "info")
        if self.state.context_path and not self.state.context:
            self.log("Context file missing or unreadable; running with defaults.", "warn")
        self.show_module(self.state.selected_module or "home")

    # -- layout helpers ------------------------------------------------------
    def _restore_window_settings(self) -> bool:
        """Restore window geometry and dock layout saved on the last close.

        Returns True when a saved geometry was applied, so the launcher knows
        not to override it with the default window size.
        """
        settings = QSettings("PyHydroGeophysX", "Workbench")
        geometry = settings.value("main/geometry")
        window_state = settings.value("main/windowState")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if window_state is not None:
            self.restoreState(window_state)
        return geometry is not None

    def _save_window_settings(self) -> None:
        settings = QSettings("PyHydroGeophysX", "Workbench")
        settings.setValue("main/geometry", self.saveGeometry())
        settings.setValue("main/windowState", self.saveState())

    def _make_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(f"dock_{title.lower()}")
        self.addDockWidget(area, dock)
        return dock

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("HeaderBar")
        header.setFixedHeight(60)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        logo_path = theme._logo_path()
        if logo_path is not None:
            logo = QLabel()
            logo.setPixmap(QPixmap(str(logo_path)).scaledToHeight(42, Qt.SmoothTransformation))
            layout.addWidget(logo)

        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        title = QLabel("Professional Workbench")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("AQUAH — Autonomous Query-driven Understanding Agent for Hydrogeophysics")
        subtitle.setObjectName("HeaderSubtitle")
        text_box.addWidget(title)
        text_box.addWidget(subtitle)
        layout.addLayout(text_box)
        layout.addStretch(1)
        return header

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self._add_action(file_menu, "Open Project Context…", self._open_context)
        self._add_action(file_menu, "Save Workbench Result", self._save_result)
        self._add_action(file_menu, "Export Current Module Result…", self._export_current_result)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close)

        view_menu = menubar.addMenu("&View")
        self._add_action(view_menu, "Reset Layout", self._reset_layout)
        view_menu.addSeparator()
        view_menu.addAction(self._tree_dock.toggleViewAction())
        view_menu.addAction(self._properties_dock.toggleViewAction())
        view_menu.addAction(self._log_dock.toggleViewAction())

        tools_menu = menubar.addMenu("&Tools")
        self._add_action(tools_menu, "Geophysical Data Processing", lambda: self.show_module("seismic"))
        self._add_action(tools_menu, "Hydro → Geophysics", lambda: self.show_module("hydro_geophysics"))
        subsurface = tools_menu.addMenu("Geophy → Hydrology")
        self._add_action(subsurface, "Seismic → Structure", lambda: self.show_module("seismic3d"))
        self._add_action(subsurface, "ERT → Water Content", lambda: self.show_module("geo_hydrology"))

        help_menu = menubar.addMenu("&Help")
        self._add_action(help_menu, "About", self._about)

    def _build_toolbar(self) -> None:
        from PySide6.QtCore import QSize

        toolbar = QToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(toolbar)
        self._add_action(toolbar, "Open", self._open_context, icon_name="fa5s.folder-open")
        self._add_action(toolbar, "Save", self._save_result, icon_name="fa5s.save")
        toolbar.addSeparator()
        self._add_action(toolbar, "Select", lambda: self._set_mouse_mode(rect=False), icon_name="fa5s.mouse-pointer")
        self._add_action(toolbar, "Pan", lambda: self._set_mouse_mode(rect=False), icon_name="fa5s.arrows-alt")
        self._add_action(toolbar, "Zoom", lambda: self._set_mouse_mode(rect=True), icon_name="fa5s.search-plus")
        self._pick_action = self._add_action(toolbar, "Pick", self._toggle_pick, checkable=True, icon_name="fa5s.crosshairs")
        self._add_action(toolbar, "Delete", self._delete_last_marker, icon_name="fa5s.eraser")
        toolbar.addSeparator()
        self._add_action(toolbar, "Export", self._export_current_result, icon_name="fa5s.file-export")

    def _add_action(self, target, text: str, slot, checkable: bool = False, icon_name: str = "") -> QAction:
        action = QAction(text, self)
        if icon_name:
            action.setIcon(theme.icon(icon_name))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        target.addAction(action)
        return action

    # -- navigation ----------------------------------------------------------
    def show_module(self, key: str) -> None:
        key = key or "home"
        if key not in self._pages:
            page = build_module(key, self.state, self.log)
            page.resultsUpdated.connect(self._refresh_properties)
            page.viewMeshRequested.connect(self._view_mesh_in_3d)
            page.navigateRequested.connect(self.show_module)
            self._stack.addWidget(page)
            self._pages[key] = page
        self._stack.setCurrentWidget(self._pages[key])
        self.state.selected_module = key
        self._tree.select_module(key)
        if self._pick_action.isChecked():
            self._pick_action.setChecked(False)
        self._refresh_properties()
        title = getattr(self._pages[key], "module_title", key)
        self._status_label.setText(f"Module: {title}    ·    Ready")

    def _view_mesh_in_3d(self, path: str) -> None:
        """Open the Mesh 3D module and load ``path`` (e.g. a seismic 3D volume)."""
        if not path:
            return
        self.show_module("mesh3d")
        page = self._pages.get("mesh3d")
        if page is not None and hasattr(page, "load_view_file"):
            page.load_view_file(path)
        else:
            self.log("Mesh 3D module cannot display the file.", "warn")

    def _current_page(self) -> Optional[QWidget]:
        return self._stack.currentWidget()

    def current_module(self) -> Optional[QWidget]:
        """Public accessor for the active module page (used by the AQUAH agent)."""
        return self._stack.currentWidget()

    # -- toolbar behavior ----------------------------------------------------
    def _array_viewers(self):
        page = self._current_page()
        return page.findChildren(ArrayViewer) if page is not None else []

    def _set_mouse_mode(self, rect: bool) -> None:
        page = self._current_page()
        if page is None:
            return
        mode = pg.ViewBox.RectMode if rect else pg.ViewBox.PanMode
        boxes = page.findChildren(pg.ViewBox)
        for vb in boxes:
            vb.setMouseMode(mode)
        if not boxes:
            self.log("No plot in the current module to change mouse mode.", "debug")

    def _toggle_pick(self, checked: bool) -> None:
        viewers = self._array_viewers()
        for av in viewers:
            av.set_pick_mode(checked)
        if not viewers:
            self.log("Pick mode is not applicable in this module.", "debug")
            self._pick_action.setChecked(False)

    def _delete_last_marker(self) -> None:
        viewers = self._array_viewers()
        for av in viewers:
            av.remove_last_marker()
        if not viewers:
            self.log("Nothing to delete in this module.", "debug")

    # -- file actions --------------------------------------------------------
    def _open_context(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project context", "", "JSON (*.json)")
        if not path:
            return
        self.state = WorkbenchState.from_context(path)
        self._pages.clear()
        while self._stack.count():
            widget = self._stack.widget(0)
            self._stack.removeWidget(widget)
            widget.deleteLater()
        self.log(f"Loaded context {path}", "success")
        self.show_module(self.state.selected_module or "home")

    def _save_result(self) -> None:
        try:
            path = self.state.save_result()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to save result: {exc}", "error")
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.log(f"Saved workbench result to {path}", "success")
        QMessageBox.information(self, "Result saved", f"Workbench result written to:\n{path}")

    def _export_current_result(self) -> None:
        key = self.state.selected_module
        result = self.state.module_results.get(key)
        if not result:
            self.log(f"No result to export for module '{key}'.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export module result", f"{key}_result.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)
        self.log(f"Exported '{key}' result to {path}", "success")

    # -- view / help ---------------------------------------------------------
    def _reset_layout(self) -> None:
        for dock, area in (
            (self._tree_dock, Qt.LeftDockWidgetArea),
            (self._properties_dock, Qt.RightDockWidgetArea),
            (self._log_dock, Qt.BottomDockWidgetArea),
        ):
            dock.setFloating(False)
            dock.show()
            self.addDockWidget(area, dock)

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"<b>{WINDOW_TITLE}</b><br><br>"
            "Local desktop companion to the PyHydroGeophysX Streamlit app for "
            "professional geophysical mouse interaction: data processing, "
            "hydro-to-geophysics profile selection, picking, and forward modeling.",
        )

    # -- properties panel ----------------------------------------------------
    def _refresh_properties(self) -> None:
        key = self.state.selected_module
        payload = {
            "current_module": key,
            "context": self.state.context_summary(),
            "module_results": self.state.module_results,
        }
        self._properties.setPlainText(json.dumps(payload, indent=2, default=str))

    # -- logging -------------------------------------------------------------
    def log(self, message: str, level: str = "info") -> None:
        self._log_panel.log(message, level)

    # -- shutdown ------------------------------------------------------------
    def closeEvent(self, event) -> None:
        """Persist the layout, then cancel/join module workers before closing."""
        try:
            self._save_window_settings()
        except Exception:  # noqa: BLE001 - persistence is best effort
            pass
        for page in list(self._pages.values()):
            try:
                page.stop_workers()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass
        super().closeEvent(event)
