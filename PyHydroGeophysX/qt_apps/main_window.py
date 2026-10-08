"""Main application window for the PyHydroGeophysX professional studio."""

from __future__ import annotations

from html import escape
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import pyqtgraph as pg
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.core import mesh_serialization
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.agents import assistants as assistant_registry
from PyHydroGeophysX.qt_apps.agent.chat_panel import AssistantChatPanel
from PyHydroGeophysX.qt_apps.agent.controller import StudioController
from PyHydroGeophysX.qt_apps import stall_watch
from PyHydroGeophysX.qt_apps.layout_fit import relax_minimum_width
from PyHydroGeophysX.qt_apps.modules import build_module
from PyHydroGeophysX.qt_apps.modules.base import GENERIC_ACTIVITY, BaseModule
from PyHydroGeophysX.qt_apps.results_store import run_title
from PyHydroGeophysX.qt_apps.state import StudioState
from PyHydroGeophysX.qt_apps.workers import prepare_workflow_process
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.ai_presence import AgentGlowFrame
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.log_panel import LogPanel
from PyHydroGeophysX.qt_apps.widgets.project_dialogs import NewProjectDialog, SaveRunsDialog
from PyHydroGeophysX.qt_apps.widgets.project_tree import ProjectTree

WINDOW_TITLE = "PyHydroGeophysX Professional Studio"


#: Longest activity phrase the status bar shows; the label never wraps, and a
#: long one would push the output folder off the bar.
_ACTIVITY_CHARS = 70


def _page_activity(page) -> str:
    """What ``page`` is busy with ("" when idle), shortened for the status bar."""
    try:
        phrase = str(page.activity() or "") if hasattr(page, "activity") else ""
    except RuntimeError:                  # a page Qt has already destroyed
        return ""
    return phrase if len(phrase) <= _ACTIVITY_CHARS else phrase[:_ACTIVITY_CHARS - 1] + "…"


def _elapsed_since(page, now: float) -> str:
    """How long ``page`` has been busy, in words: "45 s", "3 min 05 s", "1 h 02 min"."""
    since = page.busy_since() if hasattr(page, "busy_since") else None
    if since is None:
        return ""
    seconds = max(0, int(now - since))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def _stop_page_workers(pages) -> None:
    """Cancel and join every page's workers; stopping is best effort."""
    for page in list(pages):
        try:
            page.stop_workers()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            pass


class PyHydroGeophysXStudio(QMainWindow):
    """Top-level window: tree | stacked modules | properties, with a log dock."""

    def __init__(self, context_path: Optional[str] = None, initial_module: str = "home") -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.state = StudioState.from_context(context_path)
        if initial_module and initial_module != "home":
            self.state.selected_module = initial_module
        self._pages: Dict[str, BaseModule] = {}

        # Bottom: log panel (built first so module construction can log).
        self._log_panel = LogPanel()
        self._log_dock = self._make_dock("Log", self._log_panel, Qt.BottomDockWidgetArea)

        # Home owns its welcome area; other pages use the full central space.
        self._stack = QStackedWidget()
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._independent_tool_note = QLabel()
        self._independent_tool_note.setWordWrap(True)
        self._independent_tool_note.setContentsMargins(12, 8, 12, 8)
        self._independent_tool_note.hide()
        outer.addWidget(self._independent_tool_note)
        # The margin round the modules is where the agent's glow is drawn while
        # an automatic run is in control, so it never covers a control.
        content = AgentGlowFrame(margin=8)
        self._agent_glow = content
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.addWidget(self._stack)
        outer.addWidget(content, stretch=1)
        self.setCentralWidget(container)

        # Left: navigator tree.
        self._tree = ProjectTree()
        self._tree.moduleSelected.connect(self.show_module)
        self._tree_dock = self._make_dock("Project", self._tree, Qt.LeftDockWidgetArea)

        # Right: the assistant's chat + properties summary, in a tabbed dock.
        # A normal PyHydroGeophysX launch always starts in its native AQUAH
        # experience. Domain launchers may explicitly switch after construction.
        self._restore_assistant()
        self._properties = QTextEdit()
        self._properties.setReadOnly(True)
        self._controller = StudioController(self)
        self._chat = AssistantChatPanel(self._controller, self.log)
        right_tabs = QTabWidget()
        right_tabs.addTab(self._chat, "Assistant")
        right_tabs.addTab(self._properties, "Properties")
        # Adds directly to the window's minimum width, so it is a floor for
        # comfort rather than for function: the chat panel itself needs 273, and
        # the dock is resizable for anyone who wants it wider.
        right_tabs.setMinimumWidth(300)
        self._properties_dock = self._make_dock("Assistant", right_tabs, Qt.RightDockWidgetArea)

        # Off unless PHGX_STALL_WATCH_MS is set; see stall_watch for the contract.
        self._stall_watch = stall_watch.install(self)
        if self._stall_watch is not None:
            self._stall_watch.stalled.connect(lambda msg: self.log(msg, "warn"))

        # Before any page draws: every plot reads the unit when it draws.
        length_units.restore()
        self._build_menus()
        self._build_toolbar()
        self._geometry_restored = self._restore_window_settings()

        self._status_label = QLabel("Ready")
        self.statusBar().addWidget(self._status_label)
        # Ticks only while a page is busy, to keep the elapsed time current.
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status)
        # Runs are not recorded in the Project until saved, so how many are
        # waiting has to be visible without opening a menu.
        self._unsaved_label = QLabel()
        self.statusBar().addPermanentWidget(self._unsaved_label)
        self.state.on_runs_changed = self._refresh_unsaved_state
        # Every module writes under state.output_dir, so which Project that is
        # belongs on screen rather than only in whichever log line mentions a
        # path: its name here and in the title bar, the full path on hover.
        self._output_label = QLabel()
        self._output_label.setTextFormat(Qt.RichText)
        self._output_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        self._output_label.linkActivated.connect(lambda _link: self._new_project())
        self.statusBar().addPermanentWidget(self._output_label)
        #: "Use Default Folder" was the answer this session; the next asks again.
        self._kept_default_project = False
        self._restore_output_dir()
        self._activate_results_store(self.state.output_dir)
        self._refresh_unsaved_state()

        self.log(f"Studio started. Context: {self.state.context_path or '(none)'}", "info")
        if self.state.context_path and not self.state.context:
            self.log("Context file missing or unreadable; running with defaults.", "warn")
        self.show_module(self.state.selected_module or "home")
        self._focus_workspace()
        # The first run's workflow process, started once the window is up (see
        # workers._Standby) so that run does not wait for one either.
        QTimer.singleShot(2000, prepare_workflow_process)

    # -- layout helpers ------------------------------------------------------
    def _restore_window_settings(self) -> bool:
        """Restore window geometry and dock layout saved on the last close.

        Returns True when a saved geometry was applied, so the launcher knows
        not to override it with the default window size.
        """
        settings = QSettings("PyHydroGeophysX", "Studio")
        geometry = settings.value("main/geometry")
        window_state = settings.value("main/windowState")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if window_state is not None:
            self.restoreState(window_state)
        return geometry is not None

    def _save_window_settings(self) -> None:
        settings = QSettings("PyHydroGeophysX", "Studio")
        settings.setValue("main/geometry", self.saveGeometry())
        settings.setValue("main/windowState", self.saveState())

    # -- output folder -------------------------------------------------------
    def _restore_output_dir(self) -> None:
        """Reapply the folder chosen last time, unless a context already set one.

        A bridge context is the launcher telling us where this run's results go,
        so it outranks a remembered preference.
        """
        override = os.environ.get("PYHYDROGEOPHYSX_OUTPUT_DIR")
        if override and not self.state.context:
            self.state.output_dir = Path(override).expanduser()
            self.state.default_project = False
        elif not self.state.context:
            saved = QSettings("PyHydroGeophysX", "Studio").value("main/outputDir")
            if saved:
                # Only a Project the user created or opened is remembered.
                self.state.output_dir = Path(str(saved))
                self.state.default_project = False
        self._refresh_output_label()

    def _project_name(self) -> str:
        """What the Project in use is called on screen."""
        return self.state.project_name

    def _refresh_output_label(self) -> None:
        """Name the Project in use in the status bar and the window title."""
        path = self.state.project_directory
        if path is None:
            self.setWindowTitle(WINDOW_TITLE)
            self._output_label.setText("Project: (none)")
            self._output_label.setToolTip("Results go to the current working directory.")
            return
        name = self._project_name()
        self.setWindowTitle(f"{name} — {WINDOW_TITLE}")
        text = str(path)
        warn = not mesh_serialization.ansi_safe(text)
        default = bool(getattr(self.state, "default_project", False))
        # Left alone, the default folder collects every survey of every
        # session, so it says it is not a named Project and offers to make one.
        self._output_label.setText(
            ("⚠ " if warn else "") + f"Project: {escape(name)}"
            + ("  ·  <a href='new'>New Project…</a>" if default else ""))
        self._output_label.setToolTip(
            text
            + ("\n\nResults go to this default folder, shared by every session, until "
               "you create a project. File › New Project… names one." if default else "")
            + ("\n\nWindows' ANSI codepage cannot represent this path. PyGIMLi writes "
               "are staged through a temporary folder to work around it, which is "
               "slower and fails outright if TEMP has the same problem. A path "
               "without such characters avoids it." if warn else ""))

    def _switch_project(self, path: Path, landing: Optional[str]) -> bool:
        """Point the studio at *path* and open *landing* (stay put for None).

        The Project folder is also the output folder — the two were separate
        commands that set the same field, which left it possible to "open" one
        Project and write results into another. Every entry point now runs the
        same write probe, so an unwritable folder is refused before a run starts
        rather than after one finishes.

        Leaving the default folder keeps every page as it is. Nobody chose that
        folder, so whatever is open is the work the new Project is for, and the
        question about it comes as data is being added - resetting the pages
        then would throw away the very survey being set up. Moving between two
        named Projects starts the pages afresh, as it always has.
        """
        carry_session = bool(getattr(self.state, "default_project", False))
        # Unsaved runs live in the Project being left behind, so the decision has
        # to happen before the store is swapped out from under them.
        if not self._resolve_unsaved_runs("They belong to the Project you are leaving."):
            return False
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".phgx_write_test"
            probe.touch()
            probe.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "Project folder",
                                f"{path} cannot be written to:\n{exc}")
            return False
        if not self._activate_results_store(path):
            return False
        self.state.default_project = False
        self._offer_to_clear_abandoned()
        if carry_session:
            # Only the run browser lists a Project; the other pages read the
            # store each time they write, so they follow it without a reset.
            viewer = self._pages.get("model_viewer")
            if viewer is not None and hasattr(viewer, "reset_project"):
                viewer.reset_project()
        else:
            self._reset_pages(clear_session=True)
        QSettings("PyHydroGeophysX", "Studio").setValue("main/outputDir", str(path))
        self._refresh_output_label()
        self._refresh_unsaved_state()
        self.log(f"Project {self._project_name()}: results for this session go to {path}",
                 "success")
        if not mesh_serialization.ansi_safe(str(path)):
            self.log(
                "This path contains characters Windows' ANSI codepage cannot represent. "
                "PyGIMLi cannot open such paths directly, so mesh and model writes are "
                "staged through a temporary folder. It works, but a plainer path is safer.",
                "warn")
        if landing:
            self.show_module(landing)
        else:
            # Staying on the page the user is working in; Home and the map name
            # the Project, so they are brought up to date where they are open.
            current = self.state.selected_module
            if current in ("home", "project_map") and hasattr(self._pages.get(current), "refresh"):
                self._pages[current].refresh()
            self._refresh_properties()
        return True

    def _make_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(f"dock_{title.lower()}")
        self.addDockWidget(area, dock)
        return dock

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # Three groups, in the order a session uses them: choose where results
        # live, take them out, leave. Every computation is already persisted the
        # moment it finishes, so nothing here is a "save your work or lose it"
        # command and none of it is on the critical path.
        file_menu = menubar.addMenu("&File")
        self._add_action(file_menu, "New Project…", self._new_project)
        self._add_action(file_menu, "Open Project…", self._open_project)
        self._add_action(file_menu, "Import Existing Results…", self._import_existing_results)
        file_menu.addSeparator()
        self._save_action = self._add_action(
            file_menu, "Save Runs to Project", self._save_runs)
        self._save_action.setShortcut("Ctrl+S")
        self._save_action.setStatusTip(
            "Add this session's finished runs to the Project's history. "
            "Nothing is recorded there until you do."
        )
        self._discard_action = self._add_action(
            file_menu, "Discard Unsaved Runs…", self._discard_runs)
        export = self._add_action(file_menu, "Export Results…", self._export_results)
        export.setShortcut("Ctrl+E")
        export.setStatusTip(
            "Write the current module's results to a folder you choose, CSV included."
        )
        file_menu.addSeparator()
        # The Streamlit bridge and the raw module JSON are for driving this app
        # from the web workflow. They are not how a person gets their results
        # out, so they no longer sit next to the command that is.
        bridge = file_menu.addMenu("Streamlit Bridge")
        self._add_action(bridge, "Open Project Context…", self._open_context)
        self._add_action(bridge, "Save Studio Result", self._save_result)
        self._add_action(bridge, "Export Module Result (JSON)…", self._export_current_result)
        self._add_action(bridge, "Rebuild Run Index", self._save_project)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close)

        view_menu = menubar.addMenu("&View")
        self._add_action(view_menu, "Reset Layout", self._reset_layout)
        view_menu.addSeparator()
        # One unit for every plot in the studio. Only the axes change: models,
        # meshes and exports stay in metres.
        units_menu = view_menu.addMenu("Length Units")
        self._unit_group = QActionGroup(self)
        self._unit_group.setExclusive(True)
        for unit, text in (("m", "Metres (m)"), ("ft", "Feet (ft)")):
            action = self._add_action(
                units_menu, text, lambda _checked=False, u=unit: self._set_length_unit(u),
                checkable=True)
            action.setData(unit)
            action.setChecked(unit == length_units.current())
            action.setStatusTip("Show distances, elevations and depths on every plot in "
                                f"{text.lower()}. The data themselves stay in metres.")
            self._unit_group.addAction(action)
        # Light, Dark, or whatever the operating system is set to; remembered.
        appearance_menu = view_menu.addMenu("Appearance")
        self._appearance_group = QActionGroup(self)
        self._appearance_group.setExclusive(True)
        for value, text in (("system", "Match System"), ("light", "Light"), ("dark", "Dark")):
            action = self._add_action(
                appearance_menu, text,
                lambda _checked=False, v=value: self._set_appearance(v), checkable=True)
            action.setData(value)
            action.setChecked(value == theme.appearance())
            self._appearance_group.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(self._tree_dock.toggleViewAction())
        view_menu.addAction(self._properties_dock.toggleViewAction())
        view_menu.addAction(self._log_dock.toggleViewAction())
        self._all_tools_action = self._add_action(view_menu, "Show all processing tools", self._toggle_all_tools, checkable=True)
        self._all_tools_action.setChecked(True)

        tools_menu = menubar.addMenu("&Tools")
        self._add_action(tools_menu, "Model Viewer", lambda: self.show_module("model_viewer"))
        tools_menu.addSeparator()
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
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        self._main_toolbar = toolbar
        toolbar.addAction(self._properties_dock.toggleViewAction())
        # The toolbar carries the two commands a session actually repeats. The
        # bridge "Save" that used to sit here wrote a JSON manifest for the
        # Streamlit app, which read as the button that saved your results.
        # New beside Open: a Project is made by naming it, and that was easy to
        # miss with only a File menu entry that opened a bare folder picker.
        self._add_action(toolbar, "New Project", self._new_project, icon_name="fa5s.folder-plus")
        self._add_action(toolbar, "Open Project", self._open_project, icon_name="fa5s.folder-open")
        # "Save" now means what a user reads it to mean. It used to write the
        # Streamlit bridge manifest, which is why it moved off the toolbar.
        self._save_button = self._add_action(
            toolbar, "Save", self._save_runs, icon_name="fa5s.save")
        self._add_action(toolbar, "Export", self._export_results, icon_name="fa5s.file-export")
        toolbar.addSeparator()
        self._add_action(toolbar, "Select", lambda: self._set_mouse_mode(rect=False), icon_name="fa5s.mouse-pointer")
        self._add_action(toolbar, "Pan", lambda: self._set_mouse_mode(rect=False), icon_name="fa5s.arrows-alt")
        self._add_action(toolbar, "Zoom", lambda: self._set_mouse_mode(rect=True), icon_name="fa5s.search-plus")
        self._pick_action = self._add_action(toolbar, "Pick", self._toggle_pick, checkable=True, icon_name="fa5s.crosshairs")
        self._add_action(toolbar, "Delete", self._delete_last_marker, icon_name="fa5s.eraser")
        # Day and night, at the far end of the toolbar.
        spacer = QWidget()
        spacer.setObjectName("toolbarSpacer")
        spacer.setStyleSheet("#toolbarSpacer { background: transparent; }")
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self._appearance_action = self._add_action(
            toolbar, "", self._toggle_appearance)
        self._sync_appearance_controls()
        theme.notifier().changed.connect(lambda _mode: self._sync_appearance_controls())

    def _set_appearance(self, value: str) -> None:
        """Switch the whole studio to Light, Dark or the system's appearance."""
        from PySide6.QtWidgets import QApplication

        applied = theme.set_appearance(QApplication.instance(), value)
        self._sync_appearance_controls()
        label = {"system": f"Match System ({applied})", "light": "Light",
                 "dark": "Dark"}.get(value, value)
        self.log(f"Appearance: {label}.", "info")

    def _toggle_appearance(self) -> None:
        """The toolbar's day/night switch: the other of Light and Dark."""
        self._set_appearance("light" if theme.is_dark() else "dark")

    def _sync_appearance_controls(self) -> None:
        """Point the menu and the toolbar switch at the appearance in force."""
        group = getattr(self, "_appearance_group", None)
        if group is not None:
            for action in group.actions():
                action.setChecked(action.data() == theme.appearance())
        action = getattr(self, "_appearance_action", None)
        if action is not None:
            dark = theme.is_dark()
            action.setIcon(theme.icon("fa5s.sun" if dark else "fa5s.moon"))
            action.setText("Light" if dark else "Dark")
            action.setToolTip("Switch to the light appearance" if dark
                              else "Switch to the dark appearance")

    def _set_length_unit(self, unit: str) -> None:
        """Redraw every plot with its axes in ``unit`` (``'m'`` or ``'ft'``)."""
        unit = length_units.set_unit(unit)
        self.log(f"Plots now show lengths in {'feet' if unit == 'ft' else 'metres'}; "
                 "the data stay in metres.", "info")

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
            # A page's own content width becomes a hard floor on the window, which
            # on a 1920 px screen the OS cannot satisfy. Relax it once, at build.
            relax_minimum_width(page)
            page.resultsUpdated.connect(self._refresh_properties)
            page.viewMeshRequested.connect(self._view_mesh_in_3d)
            page.navigateRequested.connect(self.show_module)
            if hasattr(page, "activityChanged"):
                page.activityChanged.connect(self._refresh_status)
            if hasattr(page, 'startAIRequested'):
                page.startAIRequested.connect(self._start_task_ai)
            if hasattr(page, 'viewRunRequested'):
                page.viewRunRequested.connect(self._view_workflow_run)
            if hasattr(page, 'viewArtifactRequested'):
                page.viewArtifactRequested.connect(self._view_workflow_run)
            if key == "home":
                page.newProjectRequested.connect(self._new_project)
                page.openProjectRequested.connect(self._open_project)
            self._stack.addWidget(page)
            self._pages[key] = page
        self._stack.setCurrentWidget(self._pages[key])
        self.state.selected_module = key
        self._sync_independent_tool_note()
        if key in ("home", "project_map") and hasattr(self._pages[key], "refresh"):
            self._pages[key].refresh()
        self._tree.select_module(key)
        if self._pick_action.isChecked():
            self._pick_action.setChecked(False)
        self._refresh_properties()
        self._refresh_status()

    # -- status bar ----------------------------------------------------------
    def _refresh_status(self, *_changed) -> None:
        """Say in the status bar what the studio is doing, not only where it is.

        The open page's work comes first, with how long it has been going; a
        page that is not on screen but still working is named as well, so
        moving to another page never makes a run look finished. A 420-step
        time-lapse run once spent an hour writing its results after the
        inversion under a status bar that read "Ready" - which is how a studio
        that is working gets closed.
        """
        label = getattr(self, "_status_label", None)
        if label is None:
            return
        current = self._stack.currentWidget()
        now = time.monotonic()
        parts = [f"Module: {getattr(current, 'module_title', '') or self.state.selected_module}"]
        doing = _page_activity(current)
        if doing:
            parts.append("Working…" if doing == GENERIC_ACTIVITY else f"Working: {doing}")
            parts.extend(filter(None, [_elapsed_since(current, now)]))
        others = [page for page in self._pages.values()
                  if page is not current and _page_activity(page)]
        if len(others) == 1:
            other = others[0]
            phrase = _page_activity(other)
            text = f"{getattr(other, 'module_title', 'Another page')} is still working"
            if phrase != GENERIC_ACTIVITY:
                text += f": {phrase}"
            parts.extend(filter(None, [text, _elapsed_since(other, now)]))
        elif others:
            names = [str(getattr(page, "module_title", "a page")) for page in others]
            parts.append(f"{', '.join(names[:-1])} and {names[-1]} are still working")
        busy = bool(doing or others)
        if not busy:
            parts.append("Ready")
        text = "    ·    ".join(parts)
        if text != label.text():
            label.setText(text)
            # Painted now rather than at the next turn of the event loop: a
            # page announcing work on the GUI thread (set_activity) is about to
            # hold that loop, and the phrase has to be on screen meanwhile.
            label.repaint()
        if bool(label.property("tone")) != busy:
            theme.set_tone(label, "busy" if busy else None)
        if busy and not self._status_timer.isActive():
            self._status_timer.start()
        elif not busy:
            self._status_timer.stop()

    def _restore_assistant(self) -> None:
        """Start an ordinary Studio session with its native default assistant."""
        self._apply_assistant(
            assistant_registry.get_assistant(assistant_registry.DEFAULT_KEY)
        )

    def _apply_assistant(self, agent) -> None:
        assistant_registry.set_active(agent.key)
        theme.set_ai_colors(agent.colors)
        # Home names the assistant's domain and draws its mark in its colours.
        home = self._pages.get("home")
        if home is not None:
            home.refresh()

    def set_assistant(self, key: str) -> None:
        """Work with the assistant ``key``: its chat, its Workflow page, its glow.

        Refused while a workflow is running, since the run belongs to the
        assistant that started it.
        """
        workflow = self._pages.get("one_click")
        if workflow is not None and getattr(workflow, "_worker", None) is not None:
            self.log("A workflow is running; finish or stop it before switching assistant.",
                     "warn")
            return
        agent = assistant_registry.get_assistant(key)
        ready, why = agent.availability()
        if not ready:
            self.log(why, "warn")
            return
        self._apply_assistant(agent)
        if workflow is not None and hasattr(workflow, "set_assistant"):
            workflow.set_assistant(agent)
        if hasattr(self, '_chat'):
            self._chat.sync_assistant()
        self._focus_workspace()
        self._sync_independent_tool_note()
        self.log(f"Assistant: {agent.name} ({agent.domain}).", "info")

    def _start_task_ai(self, text):
        self._properties_dock.show()
        self._chat.start_workflow(text)

    def _sync_independent_tool_note(self):
        agent = assistant_registry.active()
        independent = bool(getattr(agent, 'workflow_setup', '')) and self.state.selected_module in {'mesh3d', 'gravmag', 'joint_inversion'}
        self._independent_tool_note.setVisible(independent)
        self._independent_tool_note.setText(
            f'Independent processing tool · These settings do not change the {agent.name} workflow. '
            'Use Data & reports → Data to edit and check its run configuration.')

    def _view_workflow_run(self, run_id, kind='model'):
        self.show_module('model_viewer')
        viewer = self._pages['model_viewer']
        viewer.refresh()
        if run_id:
            viewer._agent_select(run_id)
            record = viewer._records.get(run_id)
            if record is not None:
                artifact = next((a for a in viewer._virtual_artifacts(record)
                                 if ('_data_fit.' in str(a.get('path', '')) if kind == 'fit' else
                                     a.get('kind') == 'model')), None)
                if artifact:
                    viewer._agent_show_artifact(Path(artifact.get('path', '')).name)

    def _toggle_all_tools(self, checked):
        assistant = assistant_registry.active()
        modules = ('one_click', 'model_viewer') if getattr(assistant, 'focused_workspace', False) else assistant.studio_modules
        self._tree.focus_modules(() if checked else modules)

    def _focus_workspace(self):
        focused = getattr(assistant_registry.active(), 'focused_workspace', False)
        self._all_tools_action.setChecked(not focused)
        self._toggle_all_tools(not focused)
        for action in self._main_toolbar.actions():
            if action.text() in {'Export', 'Select', 'Pan', 'Zoom', 'Pick', 'Delete'} or action.isSeparator():
                action.setVisible(not focused)
        for page in self._pages.values():
            compact = getattr(page, 'set_compact', None)
            if callable(compact):
                compact(focused)
        if focused:
            self._log_dock.hide()
            # Focus the processing workspace without dismissing the chat the
            # user just used to choose this assistant.
            self._properties_dock.show()
        else:
            self._properties_dock.show()

    def set_agent_presence(self, state: str) -> None:
        """Light the central area's edge while the assistant is doing the work.

        ``state`` is one of the :mod:`~PyHydroGeophysX.qt_apps.widgets.ai_presence`
        states: thinking and working glow in the agent's colours, waiting pulses
        amber, done and failed flash green or red and fade, idle clears it.
        """
        glow = getattr(self, "_agent_glow", None)
        if glow is not None:
            glow.set_state('idle' if getattr(assistant_registry.active(), 'focused_workspace', False) else state)

    def _view_mesh_in_3d(self, path: str) -> None:
        """Open the Mesh 3D module and load ``path`` (e.g. a seismic 3D volume)."""
        if not path:
            return
        self.show_module("mesh3d")
        page = self._pages.get("mesh3d")
        if page is not None and hasattr(page, "load_view_file"):
            # QVTK/QtInteractor is a native OpenGL widget.  Let QStackedWidget
            # finish exposing the Mesh 3D page before asking VTK to render;
            # rendering while the page is still hidden can leave stale pixels
            # from the previous module in the viewport.
            QTimer.singleShot(
                0, lambda page=page, path=path: page.load_view_file(path))
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
    def _activate_results_store(self, root: Optional[Path]) -> bool:
        if root is None:
            return False
        try:
            store = self.state.set_results_store(root)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not open Project at {root}: {exc}", "error")
            return False
        self._refresh_output_label()
        self.log(f"Project ready: {store.root}", "success")
        return True

    def _reset_pages(self, *, clear_session: bool = False) -> None:
        # The model browser's OpenGL child belongs to the window lifetime.
        # Replacing it after a Windows native file dialog can invalidate the
        # compositor for the entire top-level window. Reset its project data
        # in place, while other modules retain their normal fresh-session path.
        viewer = self._pages.get('model_viewer')
        for page in self._pages.values():
            page.stop_workers()
        self._pages.clear()
        for index in reversed(range(self._stack.count())):
            widget = self._stack.widget(index)
            if widget is viewer:
                continue
            self._stack.removeWidget(widget)
            widget.deleteLater()
        if clear_session:
            self.state.clear_project_session()
        if viewer is not None:
            self._pages['model_viewer'] = viewer
            viewer.reset_project()

    def _project_location(self) -> Path:
        """Where a new Project's folder is offered: beside the last one made."""
        saved = QSettings("PyHydroGeophysX", "Studio").value("main/projectLocation")
        if saved and Path(str(saved)).is_dir():
            return Path(str(saved))
        current = self.state.output_dir
        if current is not None and not getattr(self.state, "default_project", False):
            return Path(current).parent
        documents = Path.home() / "Documents"
        return documents if documents.is_dir() else Path.home()

    def _create_project(self, dialog: NewProjectDialog, landing: Optional[str]) -> bool:
        QSettings("PyHydroGeophysX", "Studio").setValue(
            "main/projectLocation", str(dialog.location()))
        return self._switch_project(dialog.project_path(), landing)

    def _new_project(self) -> None:
        """Make a Project from a name and a place, ``<location>/<name>``.

        An existing folder is a Project to open, not one to create, so that is
        Open Project's job and this dialog refuses one already in use.
        """
        dialog = NewProjectDialog(self, self._project_location())
        if dialog.exec() != QDialog.Accepted:
            return
        # From the default folder the open pages come along (see _switch_project),
        # so the user stays where they were; otherwise Home shows the new Project.
        landing = None if getattr(self.state, "default_project", False) else "home"
        self._create_project(dialog, landing)

    def ask_for_project_before_data(self) -> bool:
        """Offer once to name a Project before data first goes to the default folder.

        Pages call this through ``project_dialogs.confirm_project_for_data``
        first thing in their own "add data" action, before their file dialog.
        Results otherwise land in the fallback folder nobody chose, together
        with every other survey, and nobody can later tell which run was which.
        Making the Project keeps every open page as it is, so the instrument
        and anything else already set up stays. Returns False when the user
        cancelled, and the page then stops.

        "Use Default Folder" holds for the rest of the session; "Don't ask
        again" for good. Nothing is asked while a computation runs, because the
        Project cannot change under it.
        """
        if not getattr(self.state, "default_project", False):
            return True
        if getattr(self, "_kept_default_project", False):
            return True
        settings = QSettings("PyHydroGeophysX", "Studio")
        if not settings.value("main/askForProject", True, type=bool):
            return True
        if any(record.status == "running" for record in self.state.unsaved_runs()):
            return True
        dialog = NewProjectDialog(
            self, self._project_location(), title="Save This Work in a Project",
            message=("Results are going to the default folder, which every session "
                     "shares. Name a project for this survey to keep its runs "
                     "together and easy to find."),
            offer_default=True)
        answer = dialog.exec()
        if answer == NewProjectDialog.USE_DEFAULT:
            self._kept_default_project = True
            if dialog.dont_ask_again():
                settings.setValue("main/askForProject", False)
            self.log("Results keep going to the default folder. File › New Project… "
                     "names a project at any time.", "info")
            return True
        if answer != QDialog.Accepted:
            return False
        return self._create_project(dialog, None)

    def _open_project(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Open Project", str(self.state.results_store_root or Path.cwd())
        )
        if chosen:
            # Land on the run browser: opening an existing Project is almost
            # always about looking at what is already in it.
            self._switch_project(Path(chosen), "model_viewer")

    # -- saving runs ---------------------------------------------------------
    def _refresh_unsaved_state(self) -> None:
        """Show how many finished runs are still outside the Project."""
        workflow = self._pages.get('one_click')
        if workflow is not None:
            workflow._sync_result_storage()
        runs = self.state.unsaved_runs()
        pending = [record for record in runs if record.status != "running"]
        for action in (getattr(self, "_save_action", None),
                       getattr(self, "_discard_action", None),
                       getattr(self, "_save_button", None)):
            if action is not None:
                action.setEnabled(bool(pending))
        if not hasattr(self, "_unsaved_label"):
            return
        if not pending:
            self._unsaved_label.setText("")
            self._unsaved_label.setToolTip("")
            return
        plural = "" if len(pending) == 1 else "s"
        self._unsaved_label.setText(f"● {len(pending)} unsaved run{plural}")
        self._unsaved_label.setToolTip(
            "Finished runs that are not in the Project's history yet.\n"
            "Ctrl+S adds them; closing the window will ask.\n\n"
            + "\n".join(f"· {run_title(record)}" for record in pending[:8])
            + (f"\n… and {len(pending) - 8} more" if len(pending) > 8 else "")
        )

    def _save_runs(self) -> None:
        pending = [record for record in self.state.unsaved_runs()
                   if record.status != "running"]
        if not pending:
            self.log("No finished runs are waiting to be saved.", "info")
            return
        # Saving is when a run is worth naming, so the names are asked for here,
        # each filled in already: keeping them as they are is one press of Enter.
        dialog = SaveRunsDialog(self, pending)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.state.name_runs(dialog.names())
            saved = self.state.save_all_runs()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not save runs: {exc}", "error")
            QMessageBox.warning(self, "Save runs", str(exc))
            return
        if not saved:
            self.log("No finished runs are waiting to be saved.", "info")
            return
        self.log(
            f"Saved {len(saved)} run(s) to {self.state.results_store_root}.", "success")
        self._refresh_model_viewer()

    def _discard_runs(self) -> None:
        pending = [item for item in self.state.unsaved_runs() if item.status != "running"]
        if not pending:
            self.log("No unsaved runs to discard.", "info")
            return
        answer = QMessageBox.question(
            self, "Discard unsaved runs",
            f"Permanently delete {len(pending)} unsaved run folder(s)?\n\n"
            "Their inputs, outputs, and logs go with them. This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            removed = self.state.discard_all_runs()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not discard runs: {exc}", "error")
            QMessageBox.warning(self, "Discard runs", str(exc))
            return
        self.log(f"Discarded {removed} unsaved run(s).", "info")
        self._refresh_model_viewer()

    def _refresh_model_viewer(self) -> None:
        viewer = self._pages.get("model_viewer")
        if viewer is not None and hasattr(viewer, "refresh"):
            try:
                viewer.refresh()
            except Exception:  # noqa: BLE001 - refreshing a view is best effort
                pass

    def _resolve_unsaved_runs(self, reason: str,
                              stop_running: Optional[Callable[[], None]] = None) -> bool:
        """Ask what to do with unsaved runs. False means the user cancelled.

        ``stop_running``, when given, stops the computations still going. They
        count towards the question, because stopping one leaves an unsaved
        record of it, but they are stopped only once the user has answered:
        Cancel must leave them running. The answer is then applied once, over
        everything pending after the stop. The runs are listed with their
        names, which can be changed there before Save.
        """
        runs = self.state.unsaved_runs()
        pending = [item for item in runs if item.status != "running"]
        running = ([item for item in runs if item.status == "running"]
                   if stop_running is not None else [])
        if not pending and not running:
            if stop_running is not None:
                stop_running()
            return True
        lines = []
        if pending:
            plural = "" if len(pending) == 1 else "s"
            lines.append(f"{len(pending)} finished run{plural} "
                         f"{'is' if len(pending) == 1 else 'are'} not in the "
                         "Project's history yet.")
        if running:
            plural = "" if len(running) == 1 else "s"
            lines.append(f"{len(running)} computation{plural} "
                         f"{'is' if len(running) == 1 else 'are'} still running and "
                         "will be stopped; Cancel leaves "
                         f"{'it' if len(running) == 1 else 'them'} running.")
        dialog = SaveRunsDialog(
            self, pending + running, title="Unsaved runs", allow_discard=True,
            message=" ".join(lines) + f"\n\n{reason}\n\n"
            "Save keeps them, with the names below; Discard deletes their folders.")
        answer = dialog.exec()
        if answer not in (QDialog.Accepted, SaveRunsDialog.DISCARD):
            return False
        if stop_running is not None:
            stop_running()
        try:
            if answer == QDialog.Accepted:
                self.state.name_runs(dialog.names())
                saved = self.state.save_all_runs()
                self.log(f"Saved {len(saved)} run(s) before continuing.", "success")
            else:
                removed = self.state.discard_all_runs()
                self.log(f"Discarded {removed} unsaved run(s).", "info")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Unsaved runs", str(exc))
            return False
        return True

    def _offer_to_clear_abandoned(self) -> None:
        """Offer to recover or remove run folders an earlier session left unsaved.

        A crash or a forced quit leaves a marked folder with no record. Nothing
        reads it and nothing lists it, so without this it would accumulate in
        the Project unseen. A run that had finished still holds its recipe and
        result, so it can be put back among the unsaved runs instead.
        """
        store = self.state.results_store
        if store is None or store.read_only:
            return
        try:
            abandoned = store.abandoned_run_dirs()
        except OSError:
            return
        if not abandoned:
            return
        plural = "" if len(abandoned) == 1 else "s"
        finished = sum(1 for path in abandoned if store.is_recoverable(path))
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle("Unsaved runs from an earlier session")
        text = (f"This Project holds {len(abandoned)} run folder{plural} that an earlier "
                "session never saved. They are not in the run history.")
        if finished:
            text += (f"\n\n{finished} of them finished. Recover puts "
                     f"{'it' if finished == 1 else 'them'} back among the unsaved runs, "
                     "to save or discard; the others stay where they are.")
        dialog.setText(text + "\n\nDelete deletes all of them.")
        recover = dialog.addButton("Recover", QMessageBox.AcceptRole) if finished else None
        delete = dialog.addButton("Delete", QMessageBox.DestructiveRole)
        later = dialog.addButton("Not now", QMessageBox.RejectRole)
        dialog.setDefaultButton(recover if recover is not None else later)
        dialog.exec()
        chosen = dialog.clickedButton()
        if recover is not None and chosen is recover:
            recovered = store.recover_abandoned_runs()
            self.log(f"Recovered {len(recovered)} unsaved run(s) from an earlier session; "
                     "save them to keep them.", "success")
            self._refresh_unsaved_state()
            self._refresh_model_viewer()
            return
        answer = QMessageBox.Yes if chosen is delete else QMessageBox.No
        if answer != QMessageBox.Yes:
            self.log(
                f"{len(abandoned)} abandoned run folder(s) left in place under "
                f"{store.runs_dir}.", "info")
            return
        try:
            removed = store.clear_abandoned_runs()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not clear abandoned runs: {exc}", "warn")
            return
        self.log(f"Removed {removed} abandoned run folder(s).", "success")

    def _export_results(self) -> None:
        """Write the current module's results, whatever form they take.

        Each module owns its own formats, so this asks the open page what it can
        write rather than holding a table of them here. One offer runs straight
        away; several present a chooser instead of making the user hunt for the
        button that belongs to the tab they are on.
        """
        page = self._current_page()
        title = getattr(page, "module_title", "This module")
        actions = []
        if page is not None and hasattr(page, "export_actions"):
            try:
                actions = list(page.export_actions() or [])
            except Exception as exc:  # noqa: BLE001 - a broken hook must not block the menu
                self.log(f"Could not list exports for {title}: {exc}", "warn")
        if not actions:
            store = self.state.results_store_root
            QMessageBox.information(
                self, "Export results",
                f"{title} has no results to export yet. Run a computation first.\n\n"
                + (f"Every run is also saved automatically under:\n{store}"
                   if store else "No Project folder is open yet."),
            )
            return
        if len(actions) == 1:
            actions[0][1]()
            return
        labels = [str(label) for label, _ in actions]
        choice, accepted = QInputDialog.getItem(
            self, "Export results", f"{title} can export:", labels, 0, False
        )
        if accepted and choice in labels:
            actions[labels.index(choice)][1]()

    def _save_project(self) -> None:
        try:
            store = self.state.ensure_results_store()
            count = len(store.rebuild_index())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save Project", str(exc))
            return
        self.log(f"Project index saved ({count} runs).", "success")

    def _import_existing_results(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Import existing results folder", str(self.state.output_dir or Path.cwd())
        )
        if not chosen:
            return
        store = self.state.ensure_results_store()
        preview = store.preview_legacy(chosen)
        if not preview:
            QMessageBox.information(
                self, "Import existing results",
                "No explicit workflow recipe/result pairs were found. No files were changed.",
            )
            return
        answer = QMessageBox.question(
            self, "Import existing results",
            f"Found {len(preview)} recipe/result pair(s). Add run metadata without "
            "moving the scientific files?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            imported = store.import_legacy(chosen)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Import existing results", str(exc))
            return
        self.log(f"Imported {len(imported)} legacy run(s).", "success")
        self.show_module("model_viewer")
        viewer = self._pages.get("model_viewer")
        if viewer is not None and hasattr(viewer, "use_current_store"):
            viewer.use_current_store()

    def _open_context(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project context", "", "JSON (*.json)")
        if not path:
            return
        # The whole state is replaced, runs and all, so this leaves the Project
        # as Open Project does: its unsaved runs are settled first, and what is
        # still computing is stopped only once the user has agreed to it.
        if not self._resolve_unsaved_runs(
                "They belong to the Project you are leaving.",
                stop_running=lambda: _stop_page_workers(self._pages.values())):
            return
        self._reset_pages()
        self.state = StudioState.from_context(path)
        # The window's own hook, or the unsaved-run count and Save stop updating.
        self.state.on_runs_changed = self._refresh_unsaved_state
        self._activate_results_store(self.state.output_dir or Path(path).parent)
        self._refresh_unsaved_state()
        self.log(f"Loaded context {path}", "success")
        self.show_module(self.state.selected_module or "home")

    def _save_result(self) -> None:
        try:
            path = self.state.save_result()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to save result: {exc}", "error")
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.log(f"Saved studio result to {path}", "success")
        QMessageBox.information(self, "Result saved", f"Studio result written to:\n{path}")

    def _export_current_result(self) -> None:
        # Ask the page for its own key rather than reusing the navigator's.
        # The two differ for four modules ("ert" navigates to a page whose
        # module_key is "ert_processing"), and a module writes its result under
        # the page's key. Looking it up by the navigator's key found nothing on
        # exactly the modules that had something, and reported it as no result.
        page = self._current_page()
        key = getattr(page, "module_key", None) or self.state.selected_module
        result = self.state.module_results.get(key)
        if not result:
            title = getattr(page, "module_title", key)
            self.log(f"No result to export for module '{title}'.", "warn")
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
        """Settle unsaved runs, join module workers, then persist the layout.

        The question comes before anything is stopped, so Cancel keeps the
        window open with every computation still running - stopping them first
        had already killed an inversion the user then chose to keep. Stopping a
        worker cancels its run, which leaves an unsaved record of its own, so
        the running ones count in the question and the answer covers them too.
        """
        if not self._resolve_unsaved_runs(
                "They are lost if you close without saving.",
                stop_running=lambda: _stop_page_workers(self._pages.values())):
            event.ignore()
            return
        try:
            self._save_window_settings()
        except Exception:  # noqa: BLE001 - persistence is best effort
            pass
        super().closeEvent(event)
