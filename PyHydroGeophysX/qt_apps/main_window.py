"""Main application window for the PyHydroGeophysX professional studio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pyqtgraph as pg
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
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

from PyHydroGeophysX.core import mesh_serialization
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.agent.chat_panel import AquahChatPanel
from PyHydroGeophysX.qt_apps.agent.controller import StudioController
from PyHydroGeophysX.qt_apps import stall_watch
from PyHydroGeophysX.qt_apps.layout_fit import elide_label, relax_minimum_width
from PyHydroGeophysX.qt_apps.modules import build_module
from PyHydroGeophysX.qt_apps.modules.base import BaseModule
from PyHydroGeophysX.qt_apps.state import StudioState
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.log_panel import LogPanel
from PyHydroGeophysX.qt_apps.widgets.project_tree import ProjectTree

WINDOW_TITLE = "PyHydroGeophysX Professional Studio"


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
        self._controller = StudioController(self)
        self._chat = AquahChatPanel(self._controller, self.log)
        right_tabs = QTabWidget()
        right_tabs.addTab(self._chat, "AQUAH Chat")
        right_tabs.addTab(self._properties, "Properties")
        # Adds directly to the window's minimum width, so it is a floor for
        # comfort rather than for function: the chat panel itself needs 273, and
        # the dock is resizable for anyone who wants it wider.
        right_tabs.setMinimumWidth(360)
        self._properties_dock = self._make_dock("Assistant", right_tabs, Qt.RightDockWidgetArea)

        # Off unless PHGX_STALL_WATCH_MS is set; see stall_watch for the contract.
        self._stall_watch = stall_watch.install(self)
        if self._stall_watch is not None:
            self._stall_watch.stalled.connect(lambda msg: self.log(msg, "warn"))

        self._build_menus()
        self._build_toolbar()
        self._geometry_restored = self._restore_window_settings()

        self._status_label = QLabel("Ready")
        self.statusBar().addWidget(self._status_label)
        # Runs are not recorded in the Project until saved, so how many are
        # waiting has to be visible without opening a menu.
        self._unsaved_label = QLabel()
        self.statusBar().addPermanentWidget(self._unsaved_label)
        self.state.on_runs_changed = self._refresh_unsaved_state
        # Every module writes under state.output_dir, so where that points belongs
        # on screen rather than only in whichever log line mentions a path.
        self._output_label = QLabel()
        self._output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.statusBar().addPermanentWidget(self._output_label)
        self._restore_output_dir()
        self._activate_results_store(self.state.output_dir)
        self._refresh_unsaved_state()

        self.log(f"Studio started. Context: {self.state.context_path or '(none)'}", "info")
        if self.state.context_path and not self.state.context:
            self.log("Context file missing or unreadable; running with defaults.", "warn")
        self.show_module(self.state.selected_module or "home")

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
        if not self.state.context:
            saved = QSettings("PyHydroGeophysX", "Studio").value("main/outputDir")
            if saved:
                self.state.output_dir = Path(str(saved))
        self._refresh_output_label()

    def _refresh_output_label(self) -> None:
        path = self.state.output_dir
        if path is None:
            self._output_label.setText("Output: (not set)")
            self._output_label.setToolTip("Results go to the current working directory.")
            return
        text = str(path)
        shown = text if len(text) <= 48 else "…" + text[-47:]
        warn = not mesh_serialization.ansi_safe(text)
        self._output_label.setText(("⚠ " if warn else "") + f"Output: {shown}")
        self._output_label.setToolTip(
            text + ("\n\nWindows' ANSI codepage cannot represent this path. PyGIMLi writes "
                    "are staged through a temporary folder to work around it, which is "
                    "slower and fails outright if TEMP has the same problem. A path "
                    "without such characters avoids it." if warn else ""))

    def _switch_project(self, path: Path, landing: str) -> bool:
        """Point the studio at *path* and open *landing*.

        The Project folder is also the output folder — the two were separate
        commands that set the same field, which left it possible to "open" one
        Project and write results into another. Every entry point now runs the
        same write probe, so an unwritable folder is refused before a run starts
        rather than after one finishes.
        """
        # Unsaved runs live in the Project being left behind, so the decision has
        # to happen before the store is swapped out from under them.
        if not self._resolve_unsaved_runs(f"They belong to the Project you are leaving."):
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
        self._offer_to_clear_abandoned()
        self._reset_pages(clear_session=True)
        QSettings("PyHydroGeophysX", "Studio").setValue("main/outputDir", str(path))
        self._refresh_output_label()
        self._refresh_unsaved_state()
        self.log(f"Results for this session go to {path}", "success")
        if not mesh_serialization.ansi_safe(str(path)):
            self.log(
                "This path contains characters Windows' ANSI codepage cannot represent. "
                "PyGIMLi cannot open such paths directly, so mesh and model writes are "
                "staged through a temporary folder. It works, but a plainer path is safer.",
                "warn")
        self.show_module(landing)
        return True

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
        title = QLabel("Professional Studio")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("AQUAH — Autonomous Query-driven Understanding Agent for Hydrogeophysics")
        subtitle.setObjectName("HeaderSubtitle")
        # The header has a fixed height, so this line cannot wrap; without this it
        # sets a 635 px floor under every module page.
        elide_label(subtitle)
        text_box.addWidget(title)
        text_box.addWidget(subtitle)
        layout.addLayout(text_box)
        layout.addStretch(1)
        return header

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
        view_menu.addAction(self._tree_dock.toggleViewAction())
        view_menu.addAction(self._properties_dock.toggleViewAction())
        view_menu.addAction(self._log_dock.toggleViewAction())

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
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(toolbar)
        # The toolbar carries the two commands a session actually repeats. The
        # bridge "Save" that used to sit here wrote a JSON manifest for the
        # Streamlit app, which read as the button that saved your results.
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
        for page in self._pages.values():
            page.stop_workers()
        self._pages.clear()
        while self._stack.count():
            widget = self._stack.widget(0)
            self._stack.removeWidget(widget)
            widget.deleteLater()
        if clear_session:
            self.state.clear_project_session()

    def _new_project(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose or create Project folder", str(self.state.output_dir or Path.cwd())
        )
        if chosen:
            self._switch_project(Path(chosen), "home")

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
            + "\n".join(f"· {record.label}" for record in pending[:8])
            + (f"\n… and {len(pending) - 8} more" if len(pending) > 8 else "")
        )

    def _save_runs(self) -> None:
        try:
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

    def _resolve_unsaved_runs(self, reason: str) -> bool:
        """Ask what to do with unsaved runs. False means the user cancelled."""
        pending = [item for item in self.state.unsaved_runs() if item.status != "running"]
        if not pending:
            return True
        plural = "" if len(pending) == 1 else "s"
        answer = QMessageBox.question(
            self, "Unsaved runs",
            f"{len(pending)} finished run{plural} {'is' if len(pending) == 1 else 'are'} "
            f"not in the Project's history yet.\n\n{reason}\n\n"
            "Save keeps them; Discard deletes their folders.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        try:
            if answer == QMessageBox.Save:
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
        """Offer to remove run folders an earlier session left unsaved.

        A crash or a forced quit leaves a marked folder with no record. Nothing
        reads it and nothing lists it, so without this it would accumulate in
        the Project unseen.
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
        answer = QMessageBox.question(
            self, "Unsaved runs from an earlier session",
            f"This Project holds {len(abandoned)} run folder{plural} that an earlier "
            "session never saved. They are not in the run history and nothing reads "
            "them.\n\nDelete them now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
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
        self._reset_pages()
        self.state = StudioState.from_context(path)
        self._activate_results_store(self.state.output_dir or Path(path).parent)
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
        """Settle unsaved runs, persist the layout, then join module workers.

        Stopping the workers cancels any run still going, which itself produces
        an unsaved record, so the workers are stopped first and the question is
        asked once afterwards over everything that is pending.
        """
        for page in list(self._pages.values()):
            try:
                page.stop_workers()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass
        if not self._resolve_unsaved_runs("They are lost if you close without saving."):
            event.ignore()
            return
        try:
            self._save_window_settings()
        except Exception:  # noqa: BLE001 - persistence is best effort
            pass
        super().closeEvent(event)
