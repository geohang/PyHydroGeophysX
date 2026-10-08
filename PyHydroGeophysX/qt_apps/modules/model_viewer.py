"""Read-only-first browser for durable studio Result Store runs."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import gc
from html import escape
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps.artifact_renderers import select_renderer
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import ContentWidthScrollArea
from PyHydroGeophysX.qt_apps.results_store import (
    ResultsStore, RunRecord, is_placeholder_label, run_title, short_run_id)
from PyHydroGeophysX.qt_apps.run_records import run_documents
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.curve_viewer import CurveViewer
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.widgets.project_dialogs import SaveRunsDialog
from PyHydroGeophysX.qt_apps.widgets import temperature_panel
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.workers import TaskWorker


_RUN_ROLE = Qt.UserRole

#: Symbol, colour and wording per Result Store status. A run list is scanned, not
#: read, so the outcome has to survive peripheral vision.
_STATUS_DISPLAY = {
    "success": ("✓", "#34c759", "Succeeded"),
    "failed": ("✕", "#ff3b30", "Failed"),
    "cancelled": ("⊘", "#ff9500", "Cancelled"),
    "interrupted": ("⚠", "#ff9500", "Interrupted"),
    "incomplete": ("◐", "#ff9500", "Incomplete"),
    "needs_review": ("⚠", "#ff9500", "Needs review"),
    "running": ("●", "#0a84ff", "Running"),
    "unknown": ("?", "#616161", "Unknown"),
}

#: Metrics worth putting in the run summary before the raw record.
#: ``smoothing_alpha`` is the weight R2 and R3t chose themselves, in place of lambda.
_HEADLINE_METRICS = ("chi2", "rrms", "mean_chi2", "iterations", "n_data", "lambda",
                     "smoothing_alpha")

#: Heading for runs this session produced that are not in the Project yet.
_UNSAVED_GROUP = "⬤ Unsaved (this session)"
_UNSAVED_COLOUR = "#0a84ff"

#: What the Visualization tab's chooser leaves to Details in the compact view,
#: and the run records it never offers there: a long series' per-survey QC
#: logs are hundreds, listed on the Files tab instead.
_DETAIL_RENDERERS = {"json", "file", "text"}


def _in_chooser(artifact: Dict[str, Any], detailed: bool) -> bool:
    # files_only: the reciprocal error PNG, where the run kept the pairs the
    # chooser draws as the ERT page does (run_records.run_documents).
    if {"listing_only", "files_only"} & set(artifact.get("metadata") or {}):
        return False
    return detailed or select_renderer(artifact) not in _DETAIL_RENDERERS


#: Beyond this a text record is shown from its end, where a log is read first.
_TEXT_VIEW_LIMIT = 8 * 1024 * 1024


def _text_view(path: Path) -> QWidget:
    """A run's text record - settings, QC report, a log - in a monospaced,
    unwrapped, read-only view, so its aligned columns stay aligned."""
    from PySide6.QtGui import QFontDatabase

    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setLineWrapMode(QPlainTextEdit.NoWrap)
    view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    size = path.stat().st_size
    with open(path, "rb") as handle:
        if size > _TEXT_VIEW_LIMIT:
            handle.seek(size - _TEXT_VIEW_LIMIT)
        text = handle.read().decode("utf-8", errors="replace")
    if size > _TEXT_VIEW_LIMIT:
        text = (f"[The first {_human_size(size - _TEXT_VIEW_LIMIT)} of this "
                f"{_human_size(size)} file are not shown; double-click it on the "
                "Files tab to open all of it.]\n" + text.partition("\n")[2])
    view.setPlainText(text)
    return view


def _human_size(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or suffix == "TB":
            return f"{value:.1f} {suffix}" if suffix != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def _close_array_resource(resource: Any) -> None:
    try:
        mmap = getattr(resource, "_mmap", None)
        if mmap is not None:
            mmap.close()
        elif hasattr(resource, "close"):
            resource.close()
    except (OSError, ValueError):
        pass


def _module_title(module_key: str) -> str:
    """Return the navigator's wording for a module key, e.g. ``ert`` -> ``ERT``.

    A stored record carries the module's *result* key (``ert_processing``) while
    the navigator is keyed by its navigation key (``ert``). Translating through
    the registry keeps both spellings in one group; matching the raw string put
    older runs under a separate "Ert Processing" heading beside "ERT Processing".

    Imported lazily: the modules package builds pages on demand, and a top-level
    import from one of those pages back into the package is a needless cycle.
    """
    try:
        from PyHydroGeophysX.qt_apps.modules import MODULE_SPECS
        from PyHydroGeophysX.workflows.registry import navigation_key_for

        spec = MODULE_SPECS.get(navigation_key_for(module_key))
        if spec and len(spec) > 2 and spec[2]:
            return str(spec[2])
    except Exception:  # noqa: BLE001 - a label must never break the viewer
        pass
    return str(module_key).replace("_", " ").title() or "Other"


def _group_key(module_key: str) -> str:
    """Collapse a module's spellings onto one tree group."""
    try:
        from PyHydroGeophysX.workflows.registry import navigation_key_for

        return navigation_key_for(module_key)
    except Exception:  # noqa: BLE001 - grouping must never break the viewer
        return str(module_key)


def _operation_title(record_workflow: str, record_operation: str) -> str:
    """Turn ``ert.timelapse_inversion`` into ``Timelapse Inversion``."""
    raw = str(record_workflow or record_operation or "").strip()
    if not raw:
        return "Other"
    return raw.rpartition(".")[2].replace("_", " ").title() or raw


def _pretty_kind(kind: str) -> str:
    """``resistivity_model`` -> ``Resistivity model``."""
    text = str(kind or "").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "File"


def _artifact_label(artifact: Dict[str, Any], *, missing: bool = False) -> str:
    """Prefer an explicit display label, falling back to the filename.

    ``artifact_id`` values like ``output:outputs/qc.png`` are how the record
    addresses a file; a chooser should show what the file is.
    """
    path_value = str(artifact.get("path") or "")
    kind = _pretty_kind(artifact.get("kind", ""))
    if not path_value:
        return f"{artifact.get('label') or kind} (in record)"
    label = str(artifact.get('label') or f"{Path(path_value).name} — {kind}")
    return f"{label} (missing)" if missing else label


def _artifact_plot_options(artifact: Dict[str, Any], path: Path) -> tuple[str, str, bool]:
    """Return ``(title, value_label, log_scale)`` for numeric previews.

    Result artifacts already carry field metadata in several workflows.  Keep
    that meaning when the file reaches the generic viewer instead of reducing
    every grid to anonymous rows, columns, and an unlabeled linear colour bar.
    """
    metadata = artifact.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    kind = str(artifact.get("kind") or "")
    field = str(
        metadata.get("label")
        or metadata.get("field")
        or artifact.get("label")
        or (_pretty_kind(kind) if kind else "Value")
    ).strip()
    units = str(metadata.get("units") or "").strip()
    tokens = " ".join((kind, path.stem, field)).lower()
    is_resistivity = any(token in tokens for token in ("resistiv", "rhoa", "rho_a"))
    if not units and is_resistivity:
        units = "Ω·m"
    value_label = field or "Value"
    if units and units.lower() not in value_label.lower():
        value_label = f"{value_label} ({units})"
    explicit_log = metadata.get("log_scale")
    if isinstance(explicit_log, str):
        log_scale = explicit_log.strip().lower() in {"1", "true", "yes", "log", "log10"}
    elif explicit_log is None:
        log_scale = is_resistivity
    else:
        log_scale = bool(explicit_log)
    title = str(metadata.get("title") or path.stem.replace("_", " ").strip()).strip()
    return title, value_label, log_scale


def _run_title(record: "RunRecord") -> str:
    """Show the user's own name for a run, or a short stable one.

    The store's default label repeats the operation and the start time, both of
    which the tree already shows in the group path and the ``When`` column.
    """
    return run_title(record)


class _RunNameDelegate(QStyledItemDelegate):
    """Renames a run in place in the run list (F2, a double-click, or Rename…).

    The editor starts from the run's own name rather than the text shown, which
    can carry a short id to tell two equal names apart, and what is typed goes
    to the run's record through the page instead of into the tree's text.
    """

    def __init__(self, page: "ModelViewerModule") -> None:
        super().__init__(page._tree)
        self._page = page

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt override
        editor = QLineEdit(parent)
        editor.setPlaceholderText("Name this run")
        return editor

    def setEditorData(self, editor, index) -> None:  # noqa: N802 - Qt override
        record = self._page._records.get(str(index.data(_RUN_ROLE)))
        named = record is not None and not is_placeholder_label(record)
        editor.setText(str(record.label).strip() if named else "")
        editor.selectAll()

    def setModelData(self, editor, model, index) -> None:  # noqa: N802 - Qt override
        self._page._rename_run(str(index.data(_RUN_ROLE)), editor.text())


def _parse_utc(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _local_time(value: str) -> str:
    """Render a stored UTC stamp in the reader's own timezone.

    Runs are keyed and sorted in UTC so directories stay ordered, but a user
    comparing this morning's inversions should not have to do the arithmetic.
    """
    moment = _parse_utc(value)
    if moment is None:
        return str(value or "—")
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def _duration(start: str, end: str) -> str:
    """Elapsed wall time between two stored stamps, or ``—`` when unknown."""
    first, last = _parse_utc(start), _parse_utc(end)
    if first is None or last is None:
        return "—"
    seconds = (last - first).total_seconds()
    if seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {rest:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


class ModelViewerModule(BaseModule):
    module_key = "model_viewer"
    module_title = "Model Viewer"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._store: Optional[ResultsStore] = None
        self._records: Dict[str, RunRecord] = {}
        #: Runs finished this session that are not in the Project's history yet.
        self._unsaved_ids: set[str] = set()
        self._current: Optional[RunRecord] = None
        self._current_artifacts: List[Dict[str, Any]] = []
        self._series = None        # the time-lapse view currently on screen
        self._series_source = None # its models as inverted, before any correction
        self._temperature_panel = None  # the correction panel beside it
        self._size_cache: Dict[str, int] = {}
        self._visual_resources: List[Any] = []
        # Keep the OpenGL child alive while displaying ordinary Qt images.
        # Destroying the last GL child during an artifact switch can invalidate
        # the top-level compositor on Windows (the entire window turns black).
        self._vtk_view = None
        self._vtk_cache_key = None
        self._quality_ok = False
        self._compact = False

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self._path = QLabel("Project: (not available)")
        self._path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self._path, stretch=1)
        self._project_buttons = []
        for text, slot, tip in (
            ("Current Project", self.use_current_store,
             "Show the Project that new computations are written to."),
            ("Browse Other Project…", self.browse_store,
             "Open another Project folder read-only, without changing where results are written."),
            ("Open Folder", self.open_project_folder,
             "Open this Project folder in the system file manager."),
            ("Refresh", self.refresh, "Re-read the run history from disk."),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            top.addWidget(button)
            self._project_buttons.append(button)
        self._details_button = QPushButton('Details')
        self._details_button.setCheckable(True)
        self._details_button.toggled.connect(self._toggle_details)
        top.addWidget(self._details_button)
        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        filters = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search runs…")
        self._search.textChanged.connect(self._apply_filter)
        self._status = QComboBox()
        self._status.addItems([
            "All statuses", "success", "needs_review", "incomplete", "failed", "cancelled", "interrupted",
            "running", "unknown"
        ])
        self._status.currentTextChanged.connect(self._apply_filter)
        filters.addWidget(self._search, stretch=1)
        filters.addWidget(self._status)
        left_layout.addLayout(filters)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Run", "Status", "When", "Took"])
        self._tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self._tree.itemSelectionChanged.connect(self._selection_changed)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_tree_menu)
        # A run is renamed where it is listed: F2, a double-click, or Rename…
        # in its menu. Only the name column edits, and only through this page,
        # so the stock triggers (which would edit whichever cell was clicked)
        # are off and the three ways in call editItem on column 0 themselves.
        self._tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tree.setItemDelegateForColumn(0, _RunNameDelegate(self))
        self._tree.itemDoubleClicked.connect(
            lambda item, _column: self._rename_item(item))
        self._tree.installEventFilter(self)
        left_layout.addWidget(self._tree, stretch=1)
        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setVisible(False)
        left_layout.addWidget(self._hint)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self._metadata_panel = QWidget()
        metadata_layout = QVBoxLayout(self._metadata_panel)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._metadata_panel)
        edit_row = QHBoxLayout()
        # The name is kept as soon as it is typed (Enter, or moving on). A
        # separate save button for it went unnoticed, and the name with it.
        self._label = QLineEdit()
        self._label.setPlaceholderText("Name this run")
        self._label.setToolTip("The name this run is listed under. It is kept when you "
                               "press Enter or move on; F2 renames a run in the list too.")
        self._label_run_id: Optional[str] = None
        self._label.textEdited.connect(self._label_edited)
        self._label.editingFinished.connect(self._commit_label)
        self._size = QLabel("Size: —")
        self._save_notes_button = QPushButton("Save Notes")
        self._save_notes_button.setToolTip("Keep these notes with the run.")
        self._save_notes_button.clicked.connect(self._save_notes)
        self._save_notes_button.setVisible(False)
        self._save_run = QPushButton("Save to Project")
        self._save_run.setToolTip(
            "Add this run to the Project's history. Until then it exists only as "
            "a folder this session wrote, and closing the studio will ask."
        )
        self._save_run.setVisible(False)
        self._save_run.clicked.connect(self._save_current_run_named)
        self._open_run = QPushButton("Open Folder")
        self._open_run.setToolTip("Open this run's folder in the system file manager.")
        self._open_run.clicked.connect(self.open_run_folder)
        self._delete = QPushButton("Delete Run…")
        self._delete.clicked.connect(self._delete_run)
        edit_row.addWidget(self._label, stretch=1)
        edit_row.addWidget(self._size)
        edit_row.addWidget(self._save_run)
        edit_row.addWidget(self._open_run)
        edit_row.addWidget(self._delete)
        metadata_layout.addLayout(edit_row)
        self._notes = QTextEdit()
        self._notes.setPlaceholderText("Notes")
        self._notes.setMaximumHeight(75)
        self._notes_toggle = QCheckBox('Notes')
        self._notes_toggle.toggled.connect(self._notes.setVisible)
        self._notes_toggle.toggled.connect(self._save_notes_button.setVisible)
        self._notes.hide()
        notes_row = QHBoxLayout()
        notes_row.addWidget(self._notes_toggle)
        notes_row.addStretch(1)
        notes_row.addWidget(self._save_notes_button)
        metadata_layout.addLayout(notes_row)
        metadata_layout.addWidget(self._notes)

        self._tabs = QTabWidget()
        self._overview = QTextEdit(); self._overview.setReadOnly(True)
        from .one_click import _FittedReportBrowser
        self._report = _FittedReportBrowser()
        self._report.setOpenExternalLinks(True)
        self._metrics_page = QWidget()
        metrics_layout = QVBoxLayout(self._metrics_page)
        self._metrics = QTableWidget(0, 4)
        self._metrics.setHorizontalHeaderLabels(["Metric", "Run A", "Run B", "Difference"])
        self._quality = self._build_quality_view()
        metrics_layout.addWidget(self._metrics)
        metrics_layout.addWidget(self._quality, stretch=1)
        self._visual_page = QWidget()
        visual_layout = QVBoxLayout(self._visual_page)
        visual_bar = QHBoxLayout()
        self._artifact = QComboBox()
        self._artifact.currentIndexChanged.connect(self._render_selected_artifact)
        visual_bar.addWidget(QLabel("Artifact:"))
        visual_bar.addWidget(self._artifact, stretch=1)
        self._compare_models = QPushButton('Compare two models')
        self._compare_models.setToolTip('Select two runs with Ctrl-click, then compare aligned model grids.')
        self._compare_models.setEnabled(False)
        self._compare_models.clicked.connect(self._compare_selected_models)
        visual_bar.addWidget(self._compare_models)
        visual_layout.addLayout(visual_bar)
        self._visual_host = QWidget()
        self._visual_layout = QVBoxLayout(self._visual_host)
        self._visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.addWidget(self._visual_host, stretch=1)
        self._files = QTableWidget(0, 5)
        self._files.setHorizontalHeaderLabels(["Kind", "Format", "Path", "Exists", "Size"])
        self._files.setEditTriggers(QTableWidget.NoEditTriggers)
        # A listed file is one to open: the run log, a survey's QC log.
        self._files.cellDoubleClicked.connect(self._open_listed_file)
        self._tabs.addTab(self._overview, "Overview")
        self._tabs.addTab(self._metrics_page, "Metrics")
        self._tabs.addTab(self._visual_page, "Visualization")
        self._tabs.addTab(self._files, "Files")
        self._tabs.addTab(self._report, "Report")
        self._tabs.setTabVisible(self._tabs.indexOf(self._report), False)
        right_layout.addWidget(self._tabs, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)
        self.use_current_store()
        from PyHydroGeophysX.agents import assistants
        self.set_compact(getattr(assistants.active(), 'focused_workspace', False))

    def set_compact(self, compact):
        changed = self._compact != compact
        self._compact = compact
        self._details_button.setVisible(compact)
        self._toggle_details(self._details_button.isChecked())
        self._refresh_project_label()
        if changed:
            self.refresh()

    def _toggle_details(self, expanded):
        detailed = not self._compact or expanded
        selected_artifact = self._artifact.currentData()
        self._artifact.blockSignals(True)
        self._artifact.clear()
        for row, artifact in enumerate(self._current_artifacts):
            if _in_chooser(artifact, detailed):
                self._artifact.addItem(_artifact_label(artifact), row)
        restored = self._artifact.findData(selected_artifact)
        if restored >= 0:
            self._artifact.setCurrentIndex(restored)
        self._artifact.blockSignals(False)
        if selected_artifact is not None and restored < 0:
            self._render_selected_artifact(self._artifact.currentIndex())
        self._metadata_panel.setVisible(detailed)
        self._status.setVisible(detailed)
        for button in self._project_buttons:
            button.setVisible(detailed or button.text() == 'Refresh')
        for widget in (self._metrics_page, self._files):
            self._tabs.setTabVisible(self._tabs.indexOf(widget), detailed)
        for index in range(self._visual_layout.count()):
            widget = self._visual_layout.itemAt(index).widget()
            compact = getattr(widget, 'set_compact', None)
            if callable(compact):
                compact(not detailed)
        if self._compact:
            self._tree.setColumnHidden(2, True)
            self._tree.setColumnHidden(3, True)
        else:
            self._tree.setColumnHidden(2, False)
            self._tree.setColumnHidden(3, False)

    def _colormaps(self) -> Optional[Dict[str, str]]:
        """The session's colormap choices, which every page's views share."""
        return cmaps.colormap_settings(self.state)

    def _build_quality_view(self) -> QWidget:
        """Build the convergence chart, or a stand-in if it cannot be created.

        ``InversionQualityView`` embeds a matplotlib Qt canvas. Where matplotlib
        has bound a different Qt binding than the studio uses, constructing it
        raises — and without this guard the factory would replace the entire
        Model Viewer with a placeholder over one optional chart.
        """
        try:
            view = InversionQualityView()
            self._quality_ok = True
            return view
        except Exception as exc:  # noqa: BLE001 - the page must still open
            self.log(f"Convergence chart unavailable: {exc}", "warn")
            self._quality_ok = False
            fallback = QLabel(
                "The convergence chart is unavailable in this environment.\n"
                "Metric values are still listed above."
            )
            fallback.setWordWrap(True)
            fallback.setAlignment(Qt.AlignCenter)
            return fallback

    def use_current_store(self) -> None:
        try:
            self._store = self.state.ensure_results_store()
        except Exception as exc:
            self.log(f"Could not open current Project: {exc}", "error")
            return
        self.refresh()

    def reset_project(self) -> None:
        """Forget prior project state without tearing down the GL compositor."""
        self._reset_details()
        self._vtk_cache_key = None
        self._records.clear()
        self._unsaved_ids.clear()
        self._size_cache.clear()
        self._search.clear()
        self._status.setCurrentIndex(0)
        self.use_current_store()

    def browse_store(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Browse Result Store", str(self.state.results_store_root or Path.cwd())
        )
        if not chosen:
            return
        try:
            self._store = ResultsStore.open_or_create(chosen, read_only=True)
        except Exception as exc:
            QMessageBox.warning(self, "Open Project", str(exc))
            return
        self.refresh()

    def _refresh_project_label(self) -> None:
        if self._store is None:
            return
        name = (self.state.project_name if self._store is self.state.results_store
                else self._store.root.name or str(self._store.root))
        read_only = " — read-only" if self._store.read_only else ""
        self._path.setText(f"Project: {name}{read_only}")
        self._path.setToolTip(
            f"{self._store.root}\n\n" + (
            "Runs are browsed read-only; new computations still go to the active Project."
            if self._store.read_only
            else "New computations are written here, one folder per run.")
        )

    def refresh(self) -> None:
        if self._store is None:
            return
        self._refresh_project_label()
        self._store.rebuild_index()
        self._tree.clear()
        # Unsaved runs lead the list. They are the ones that disappear if the
        # session ends without a decision, so they should not be found by
        # scrolling past a year of history.
        unsaved = self._store.list_unsaved_runs() if not self._store.read_only else []
        self._unsaved_ids = {record.run_id for record in unsaved}
        self._records = {item.run_id: item for item in [*unsaved, *self._store.list_runs()]}
        editable = self._editable()
        groups: Dict[tuple[str, str, str], QTreeWidgetItem] = {}
        counts: Dict[tuple[str, str, str], int] = {}
        for record in self._records.values():
            pending = record.run_id in self._unsaved_ids
            module_name = (
                _UNSAVED_GROUP if pending else _module_title(record.module_key)
            )
            operation_name = _operation_title(record.workflow_id, record.operation_id)
            # Group by the day the user saw, not the stored UTC day.
            started = _local_time(record.created_at)
            date_name = started.split(" ")[0] if started != "—" else "Unknown date"
            # Group on the navigation key so a managed run ("ert_processing") and
            # a legacy import of the same module ("ert") land in one heading
            # rather than in two that read identically.
            group_key = "\x00unsaved" if pending else _group_key(record.module_key)
            path = [
                (group_key, "", ""),
                (group_key, operation_name, ""),
                (group_key, operation_name, date_name),
            ]
            for depth, (key, name) in enumerate(
                [] if self._compact else zip(path, (module_name, operation_name, date_name))
            ):
                if key not in groups:
                    groups[key] = self._group_item(name)
                    if depth == 0:
                        if pending:
                            # Ahead of the saved history, not appended to it.
                            self._tree.insertTopLevelItem(0, groups[key])
                            groups[key].setForeground(0, QColor(_UNSAVED_COLOUR))
                        else:
                            self._tree.addTopLevelItem(groups[key])
                    else:
                        groups[path[depth - 1]].addChild(groups[key])
                counts[key] = counts.get(key, 0) + 1

            symbol, colour, wording = _STATUS_DISPLAY.get(
                record.status, _STATUS_DISPLAY["unknown"]
            )
            item = QTreeWidgetItem([
                _run_title(record),
                f"{symbol} {wording}",
                started,
                _duration(record.created_at, record.finished_at),
            ])
            item.setData(0, _RUN_ROLE, record.run_id)
            if editable:
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setForeground(1, QColor(colour))
            tooltip = [f"Run ID: {record.run_id}", f"Folder: {record.run_dir}"]
            if editable:
                tooltip.append("F2 or a double-click renames it; the folder keeps its name.")
            if record.error:
                tooltip.append(f"Error: {record.error}")
            if record.imported:
                tooltip.append("Imported in place from an existing results folder.")
            if pending:
                item.setForeground(0, QColor(_UNSAVED_COLOUR))
                tooltip.append(
                    "Not saved to the Project. Use “Save to Project” to keep it."
                )
            item.setToolTip(0, "\n".join(tooltip))
            if self._compact:
                self._tree.addTopLevelItem(item)
            else:
                groups[path[2]].addChild(item)
                for key in path:
                    groups[key].setExpanded(True)

        for key, group in groups.items():
            total = counts.get(key, 0)
            group.setText(0, f"{group.text(0)}  ({total})")
        self._retitle_runs()
        for column in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(column)
        if not self._records:
            self._reset_details()
        self._apply_filter()

    def _reset_details(self) -> None:
        """Leave no stale run on screen when the list becomes empty."""
        self._report.clear()
        self._tabs.setTabVisible(self._tabs.indexOf(self._report), False)
        self._current = None
        self._label_run_id = None
        self._label.clear()
        self._notes.clear()
        self._size.setText("Size: —")
        self._delete.setEnabled(False)
        self._save_run.setVisible(False)
        self._open_run.setEnabled(False)
        self._overview.setHtml("")
        self._metrics.setRowCount(0)
        self._files.setRowCount(0)
        self._artifact.clear()
        self._current_artifacts = []
        self._clear_visual("Select a run on the left to see its results.")

    @staticmethod
    def _group_item(label: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label])
        item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
        return item

    def _update_hint(self) -> None:
        """Explain an empty list rather than leaving the user with blank space."""
        if not self._records:
            message = (
                "No runs in this Project yet. Run a computation from any module; it "
                "appears here under “Unsaved” until you save it to the Project."
            )
            root = self._store.root if self._store is not None else None
            if root is not None and all((root / name).is_file() for name in (
                    'mesh/mesh_core.msh', 'inversion_result/joint_density_core.npy',
                    'inversion_result/joint_susceptibility_core.npy')):
                message = (
                    'This folder contains GeoSAGE models. To view them, choose View results in Data & reports. '
                    'Use a separate Project to save your work.'
                )
            self._hint.setText(message)
            self._overview.setHtml('<h3>No saved Studio runs</h3><p>' + escape(message) + '</p>')
            self._hint.setVisible(True)
            return
        if self._current is None:
            self._overview.setHtml('<p>Select a run on the left to see its results.</p>')
        visible = any(
            not self._tree.topLevelItem(index).isHidden()
            for index in range(self._tree.topLevelItemCount())
        )
        self._hint.setText(
            "" if visible else
            f"None of the {len(self._records)} recorded runs match the current "
            "search and status filter."
        )
        self._hint.setVisible(not visible)

    # -- reaching the files on disk -----------------------------------------
    def _open_path(self, path: Path, what: str) -> None:
        """Hand a folder to the system file manager.

        The Project is an ordinary directory tree, so the fastest route to a
        result is often the file manager rather than another viewer tab.
        """
        if not Path(path).exists():
            QMessageBox.warning(self, "Open folder", f"{what} no longer exists:\n{path}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(self, "Open folder", f"Could not open {path}")

    def open_project_folder(self) -> None:
        if self._store is not None:
            self._open_path(self._store.root, "This Project folder")

    def open_run_folder(self) -> None:
        if self._current is not None:
            self._open_path(self._current.run_dir, "This run folder")

    def _open_listed_file(self, row: int, column: int) -> None:
        """Open a Files-tab row's file with the program the system uses for it."""
        cell = self._files.item(row, column)
        target = cell.data(Qt.UserRole) if cell is not None else None
        if target:
            self._open_path(Path(str(target)), "This file")

    def _copy_to_clipboard(self, text: str, what: str) -> None:
        QApplication.clipboard().setText(str(text))
        self.log(f"{what} copied to clipboard: {text}", "info")

    def _show_tree_menu(self, point) -> None:
        item = self._tree.itemAt(point)
        run_id = item.data(0, _RUN_ROLE) if item is not None else None
        record = self._records.get(str(run_id)) if run_id else None
        menu = QMenu(self)
        if record is not None:
            # Right-click acts on what was right-clicked, so move the selection
            # first; the detail pane and the Delete guard both follow from it.
            self._tree.setCurrentItem(item)
            rename = menu.addAction("Rename…\tF2", lambda: self._rename_item(item))
            rename.setEnabled(self._editable())
            menu.addSeparator()
            menu.addAction("Open Run Folder", self.open_run_folder)
            menu.addAction(
                "Copy Folder Path",
                lambda: self._copy_to_clipboard(record.run_dir, "Run folder"),
            )
            menu.addAction(
                "Copy Run ID", lambda: self._copy_to_clipboard(record.run_id, "Run ID")
            )
            menu.addSeparator()
            delete = menu.addAction("Delete Run…", self._delete_run)
            delete.setEnabled(self._delete.isEnabled())
            menu.addSeparator()
        menu.addAction("Open Project Folder", self.open_project_folder)
        menu.addAction("Refresh", self.refresh)
        menu.exec(self._tree.viewport().mapToGlobal(point))

    # -- naming runs -----------------------------------------------------------
    def _editable(self) -> bool:
        """Runs can be renamed only in the Project new results go to.

        Another Project is browsed read-only, so nothing here writes into it.
        """
        return bool(self._store is not None and self._store is self.state.results_store
                    and not self._store.read_only)

    def _retitle_runs(self) -> None:
        """Show each run under its name; equal names get the run's short id.

        Two time-lapse runs of the same surveys share their default name, and
        the list must still tell them apart at a glance.
        """
        titles = {run_id: _run_title(record) for run_id, record in self._records.items()}
        repeated = Counter(titles.values())
        for run_id, item in self._agent_items(titles).items():
            record = self._records[run_id]
            title = titles[run_id]
            if repeated[title] > 1 and not is_placeholder_label(record):
                title = f"{title}  ({short_run_id(record)})"
            item.setText(0, title)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        # F2 renames on every platform; the stock edit key is off (see __init__).
        if (watched is self._tree and event.type() == QEvent.KeyPress
                and event.key() == Qt.Key_F2):
            item = self._tree.currentItem()
            if item is not None and item.data(0, _RUN_ROLE):
                self._rename_item(item)
                return True
        return super().eventFilter(watched, event)

    def _rename_item(self, item: Optional[QTreeWidgetItem]) -> None:
        """Open the name of the run ``item`` lists for editing, in place."""
        if item is None or not item.data(0, _RUN_ROLE) or not self._editable():
            return
        self._tree.setCurrentItem(item)
        self._tree.editItem(item, 0)

    def _rename_run(self, run_id: str, name: str) -> bool:
        """Give a run a new name, saved or not. Only its record changes.

        The folder keeps its id for a name: modules hold paths into it that
        must keep resolving. A saved run's ``run.json`` and the index are
        written at once; an unsaved run carries the name until it is saved.
        """
        name = str(name or "").strip()
        record = self._records.get(str(run_id))
        if record is None or not name or name == str(record.label).strip():
            return False
        if not self._editable():
            return False
        try:
            record = self._store.update_run(record.run_id, label=name)
        except Exception as exc:  # noqa: BLE001 - reported, the old name stays
            self.log(f"Could not rename the run: {exc}", "error")
            return False
        self._records[record.run_id] = record
        self._retitle_runs()
        if self._current is not None and self._current.run_id == record.run_id:
            self._current = record
            self._label.setText(record.label)
        # The window lists unsaved runs by name in its status bar.
        if callable(getattr(self.state, "on_runs_changed", None)):
            self.state.on_runs_changed()
        self.log(f"Renamed run {short_run_id(record)} to “{name}”.", "info")
        return True

    def _label_edited(self, _text: str) -> None:
        # Remember whose name is being typed: by the time editing finishes, a
        # click in the list may already be selecting another run.
        if self._current is not None:
            self._label_run_id = self._current.run_id

    def _commit_label(self) -> None:
        run_id, self._label_run_id = self._label_run_id, None
        if run_id:
            self._rename_run(run_id, self._label.text())

    def _save_notes(self) -> None:
        if self._current is None or not self._editable():
            return
        try:
            self._current = self._store.update_run(
                self._current.run_id, notes=self._notes.toPlainText())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save notes", str(exc))
            return
        self.log("Notes kept with the run"
                 + ("; they go into the Project when the run is saved."
                    if self._current.run_id in self._unsaved_ids else "."), "info")

    def _apply_filter(self) -> None:
        needle = self._search.text().strip().lower()
        status = self._status.currentText()
        def filter_item(item: QTreeWidgetItem) -> bool:
            run_id = item.data(0, _RUN_ROLE)
            record = self._records.get(str(run_id)) if run_id else None
            if record is not None:
                show = status == "All statuses" or record.status == status
                if needle:
                    haystack = " ".join([
                        record.label, record.notes, record.module_key,
                        record.operation_id, record.workflow_id, record.run_id,
                    ]).lower()
                    show = show and needle in haystack
            else:
                child_visibility = [
                    filter_item(item.child(index))
                    for index in range(item.childCount())
                ]
                show = any(child_visibility)
            item.setHidden(not show)
            return show

        for index in range(self._tree.topLevelItemCount()):
            filter_item(self._tree.topLevelItem(index))
        self._update_hint()

    def _selected_records(self) -> List[RunRecord]:
        values = []
        for item in self._tree.selectedItems():
            run_id = item.data(0, _RUN_ROLE)
            record = self._records.get(str(run_id))
            if record is not None:
                values.append(record)
        return values

    def _selection_changed(self) -> None:
        selected = self._selected_records()
        self._compare_models.setEnabled(len(selected) == 2)
        self._compare_models.setVisible(not self._compact or len(selected) == 2)
        if not selected:
            return
        self._current = selected[0]
        record = self._current
        # A run nobody named shows an empty name field, not the store's stand-in.
        self._label_run_id = None
        self._label.setText("" if is_placeholder_label(record) else record.label)
        self._notes.setPlainText(record.notes)
        self._notes_toggle.setChecked(bool(record.notes))
        editable = self._editable()
        self._label.setReadOnly(not editable)
        self._notes.setReadOnly(not editable)
        pending = record.run_id in self._unsaved_ids
        self._save_run.setVisible(pending)
        self._save_run.setEnabled(editable and pending and record.status != "running")
        self._delete.setEnabled(editable and record.managed and not record.imported and record.status != "running")
        # Discarding an unsaved run and deleting a saved one remove the same
        # folder, but only one of them loses something the Project was keeping.
        self._delete.setText("Discard…" if pending else "Delete Run…")
        self._delete.setToolTip(
            "This run was never added to the Project. Discarding deletes its folder."
            if pending else
            "Imported runs point at files this Project does not own."
            if record.imported else
            "A running computation cannot be deleted." if record.status == "running"
            else "Permanently delete this run's folder."
        )
        self._open_run.setEnabled(record.run_dir.exists())
        cached_size = self._size_cache.get(record.run_id)
        self._size.setText(
            f"Size: {_human_size(cached_size)}" if cached_size is not None
            else "Size: calculating…"
        )
        if cached_size is None:
            worker = TaskWorker(self._store.run_size, record)
            worker.succeeded.connect(
                lambda size, run_id=record.run_id: self._show_run_size(run_id, size)
            )
            worker.failed.connect(
                lambda _message, run_id=record.run_id: self._show_run_size(run_id, None)
            )
            self.register_worker(worker).start()
        self._show_overview(record)
        self._show_metrics(selected[:2])
        self._populate_artifacts(record)
        if self._compact and self._current_artifacts:
            self._tabs.setCurrentWidget(self._visual_page)

    def _show_run_size(self, run_id: str, size: Optional[int]) -> None:
        if size is not None:
            self._size_cache[run_id] = int(size)
        if self._current is not None and self._current.run_id == run_id:
            self._size.setText(
                f"Size: {_human_size(int(size))}" if size is not None else "Size: unavailable"
            )

    def _show_overview(self, record: RunRecord) -> None:
        """Lead with what a person asks first, and keep the full record below.

        The stored record is the authority and stays verbatim at the bottom, but
        "did it work, when, how long, and what were the numbers" should not
        require reading JSON.
        """
        payload = {
            "run_id": record.run_id,
            "module": record.module_key,
            "operation": record.operation_id,
            "workflow": record.workflow_id,
            "status": record.status,
            "raw_status": record.raw_status,
            "created_at": record.created_at,
            "finished_at": record.finished_at,
            "error": record.error,
            "summary": record.summary,
            "warnings": record.warnings,
            "provenance": record.provenance,
        }
        symbol, colour, wording = _STATUS_DISPLAY.get(
            record.status, _STATUS_DISPLAY["unknown"]
        )
        rows = [
            ("Module", _module_title(record.module_key)),
            ("Operation", _operation_title(record.workflow_id, record.operation_id)),
            ("Started", _local_time(record.created_at)),
            ("Took", _duration(record.created_at, record.finished_at)),
            ("Folder", str(record.run_dir)),
            ("Run ID", record.run_id),
        ]
        if record.imported:
            rows.append(("Origin", "Imported in place from an existing results folder"))
        # When the inverted survey was measured, beside when the run started: a
        # single ERT run records it (time-lapse runs head each step by theirs).
        survey_time = (record.summary or {}).get("survey_time")
        if survey_time:
            from PyHydroGeophysX.data_processing.survey_timing import format_time

            survey_file = (record.summary or {}).get("survey_file")
            rows.insert(2, ("Survey measured", format_time(survey_time, seconds=True)
                            + (f" ({survey_file})" if survey_file else "")))
        documents = run_documents(record.run_dir)
        if documents:
            # Named here so a reopened run says where its settings and log are;
            # each opens from the Visualization chooser or the Files tab.
            shown = [doc["path"] for doc in documents
                     if not (doc.get("metadata") or {}).get("listing_only")]
            surveys = len(documents) - len(shown)
            rows.append(("Run records", ", ".join(shown)
                         + (f" and {surveys} survey QC logs in qc/" if surveys else "")))

        parts = [
            f'<p style="font-size:13pt;color:{colour};margin:0 0 8px 0;">'
            f"<b>{symbol} {wording}</b></p>",
            '<table cellspacing="0" cellpadding="3">',
        ]
        for name, value in rows:
            parts.append(
                f'<tr><td style="color:#666;">{escape(name)}</td>'
                f"<td>{escape(str(value))}</td></tr>"
            )
        parts.append("</table>")

        headline = [
            (key, record.metrics[key])
            for key in _HEADLINE_METRICS if key in record.metrics
        ]
        headline += [
            (key, value) for key, value in sorted(record.metrics.items())
            if key not in _HEADLINE_METRICS
        ][:6]
        if headline:
            parts.append("<p><b>Metrics</b></p><table cellspacing='0' cellpadding='3'>")
            for key, value in headline:
                parts.append(
                    f'<tr><td style="color:#666;">{escape(str(key))}</td>'
                    f"<td>{escape(str(value))}</td></tr>"
                )
            parts.append("</table>")

        if record.error:
            parts.append(
                f'<p style="color:{_STATUS_DISPLAY["failed"][1]};">'
                f"<b>Error</b><br>{escape(record.error)}</p>"
            )
        if record.warnings:
            warnings = "<br>".join(escape(str(item)) for item in record.warnings[:10])
            more = len(record.warnings) - 10
            if more > 0:
                warnings += f"<br>… and {more} more"
            parts.append(
                f'<p style="color:{_STATUS_DISPLAY["interrupted"][1]};">'
                f"<b>Warnings</b><br>{warnings}</p>"
            )

        parts.append("<hr><p style='color:#666;'><b>Full record</b></p>")
        parts.append(
            f"<pre>{escape(json.dumps(payload, indent=2, default=str))}</pre>"
        )
        self._overview.setHtml("".join(parts))

    def _show_metrics(self, records: List[RunRecord]) -> None:
        keys = sorted({key for record in records for key in record.metrics})
        self._metrics.setRowCount(len(keys))
        for row, key in enumerate(keys):
            a = records[0].metrics.get(key) if records else None
            b = records[1].metrics.get(key) if len(records) > 1 else None
            diff = ""
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diff = f"{float(b) - float(a):.6g}"
            for col, value in enumerate((key, a, b, diff)):
                self._metrics.setItem(row, col, QTableWidgetItem("" if value is None else str(value)))
        first = records[0] if records else None
        if not self._quality_ok:
            return
        try:
            if first is not None:
                self._quality.show_quality(first.metrics)
            else:
                self._quality.clear()
        except Exception as exc:  # noqa: BLE001 - a chart must not hide the run
            self.log(f"Could not draw the convergence chart: {exc}", "warn")

    def _virtual_artifacts(self, record: RunRecord) -> List[Dict[str, Any]]:
        # One entry per file: a workflow may register the same file under two
        # kinds (the ERT run's model VTK is both "vtk" and "fixed_vtk_path").
        artifacts: List[Dict[str, Any]] = []
        seen_paths: set = set()
        for item in record.artifacts:
            key = str(item.get("path") or "").replace("\\", "/").lower()
            if key and key in seen_paths:
                continue
            seen_paths.add(key)
            artifacts.append(dict(item))
        registered = {str(item.get("path") or "") for item in artifacts}
        # Reports are persisted in the result envelope, but older runs did not
        # register them as artifacts. Recover that explicit reference on read.
        if record.result_path and self._store is not None:
            try:
                result_file = self._store.locate_run_artifact(record, {"path": record.result_path})
                result = json.loads(result_file.read_text(encoding="utf-8"))
                reports = result.get("report_files") or {}
                if isinstance(reports, dict):
                    for key, value in reports.items():
                        if isinstance(value, str) and value not in registered:
                            artifacts.append({"artifact_id": f"report:{key}", "kind": "report",
                                              "format": Path(value).suffix.lstrip("."), "path": value})
                            registered.add(value)
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        for path_value, kind in (
            ("run.json", "run_metadata"),
            (record.recipe_path, "workflow_recipe"),
            (record.result_path, "workflow_result"),
        ):
            if path_value and path_value not in registered:
                artifacts.append({
                    "artifact_id": f"{record.run_id}:{kind}",
                    "kind": kind,
                    "format": Path(path_value).suffix.lstrip(".").lower(),
                    "path": path_value,
                    "metadata": {"record_file": True},
                })
                registered.add(path_value)
        # The run's records for people - settings, QC report, its log - which
        # the page that ran it offers too (qt_utils.ReproduceBar).
        for document in run_documents(record.run_dir):
            if document["path"] not in registered:
                artifacts.append(document)
                registered.add(document["path"])
        # The run's model first, then the fixed-λ one - only when it is another
        # model: with a single λ both keys name the same files.
        bundles: List[Dict[str, Any]] = []
        for key, label in (("model_bundle", "Resistivity model"),
                           ("fixed_model_bundle", "Fixed-λ resistivity model")):
            bundle = record.summary.get(key)
            if not isinstance(bundle, dict) or not bundle:
                continue
            if any(existing["metadata"]["bundle"] == bundle for existing in bundles):
                continue
            bundles.append({
                "artifact_id": f"{record.run_id}:{key}",
                "kind": "ert_model_bundle",
                "format": "bundle",
                "path": "",
                "label": label + ("s" if "models" in bundle else ""),
                "metadata": {"bundle": bundle},
            })
        return bundles + artifacts

    def _populate_artifacts(self, record: RunRecord) -> None:
        assert self._store is not None
        self._current_artifacts = self._virtual_artifacts(record)
        self._show_report(record)
        self._artifact.blockSignals(True)
        self._artifact.clear()
        self._files.setRowCount(len(self._current_artifacts))
        for row, artifact in enumerate(self._current_artifacts):
            path_value = str(artifact.get("path") or "")
            path = None
            if path_value:
                try:
                    path = self._store.locate_run_artifact(record, artifact)
                except ValueError:
                    pass
            missing = bool(path_value) and not (path and path.exists())
            if _in_chooser(artifact, not self._compact or self._details_button.isChecked()):
                self._artifact.addItem(_artifact_label(artifact, missing=missing), row)
            values = [
                _pretty_kind(artifact.get("kind", "")),
                str(artifact.get("format", "")),
                path_value,
                "missing" if missing else ("✓" if path_value else "in record"),
                _human_size(path.stat().st_size) if path and path.is_file() else "",
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if col == 3 and missing:
                    cell.setForeground(QColor(_STATUS_DISPLAY["failed"][1]))
                if path is not None and not missing:
                    cell.setData(Qt.UserRole, str(path))
                    cell.setToolTip(f"{path}\nDouble-click to open it.")
                self._files.setItem(row, col, cell)
        self._files.resizeColumnsToContents()
        self._artifact.blockSignals(False)
        if self._artifact.count():
            self._artifact.setCurrentIndex(0)
            self._render_selected_artifact(0)
        else:
            self._clear_visual("This run has no registered artifacts.")

    def _show_report(self, record):
        reports = [a for a in self._current_artifacts
                   if a.get('kind') == 'report' and str(a.get('format', '')).lower() == 'md']
        self._report.clear()
        self._tabs.setTabVisible(self._tabs.indexOf(self._report), bool(reports))
        if not reports:
            return
        try:
            path = self._store.locate_run_artifact(record, reports[0])
            self._report.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + '/'))
            self._report.setMarkdown(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            self._report.setPlainText(f'Could not open the saved report: {exc}')

    def _clear_visual(self, message: str = "") -> List[Any]:
        """Empty the pane, retaining its single reusable OpenGL child.

        Returns the widgets scheduled for deletion so a caller that has to see
        them gone, rather than merely queued, can flush those and only those.
        The VTK view holds an in-memory volume, not an open file mapping; its
        renderer remains parented here until this page itself is destroyed.
        """
        retired: List[Any] = []
        while self._visual_layout.count():
            item = self._visual_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Removed widgets can paint over their replacement until Qt
                # processes DeferredDelete, particularly native VTK children.
                widget.hide()
                if widget is self._vtk_view:
                    continue
                widget.deleteLater()
                retired.append(widget)
        for resource in self._visual_resources:
            _close_array_resource(resource)
        self._visual_resources.clear()
        if message:
            label = QLabel(message)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)
            self._visual_layout.addWidget(label)
        return retired

    def _render_selected_artifact(self, _index: int) -> None:
        if self._current is None or self._store is None:
            return
        data_index = self._artifact.currentData()
        if data_index is None or not (0 <= int(data_index) < len(self._current_artifacts)):
            return
        artifact = self._current_artifacts[int(data_index)]
        if select_renderer(artifact) == "mesh_bundle":
            self._render_mesh_bundle(artifact)
            return
        try:
            path = self._store.locate_run_artifact(self._current, artifact)
        except ValueError as exc:
            self._clear_visual(str(exc)); return
        if not path.is_file():
            self._clear_visual(f"Artifact is missing:\n{path}"); return
        renderer = select_renderer(artifact)
        try:
            if renderer in {"array", "array_stack", "curve"} and path.suffix.lower() in {".npy", ".npz"}:
                self._render_numpy(path, renderer, artifact)
            elif renderer == "curve":
                self._render_curve_file(path)
            elif renderer == "image":
                view = ZoomableImageView()
                if not view.set_image_file(path):
                    raise ValueError("the image decoder could not read this file")
                self._replace_visual(view)
            elif renderer == "reciprocal_errors":
                # The ERT page's Reciprocal errors view, from the pairs the run kept.
                from PyHydroGeophysX.qt_apps.widgets.reciprocal_view import ReciprocalErrorView
                view = ReciprocalErrorView(chooser=False)
                view.show_saved(path)
                self._replace_visual(view)
            elif renderer == "vtk":
                from PyHydroGeophysX.qt_apps.widgets.model3d_view import VTKVolumeView
                if self._vtk_view is None:
                    self._vtk_view = VTKVolumeView(self._visual_host, colormaps=self._colormaps())
                view = self._vtk_view
                meta = artifact.get('metadata') or {}
                stat = path.stat()
                key = (str(path), stat.st_mtime_ns, stat.st_size, json.dumps(meta, sort_keys=True, default=str))
                self._replace_visual(view)
                if key != self._vtk_cache_key:
                    self._vtk_cache_key = None
                    if view.show_file(path, scalar_cmaps=meta.get('scalar_cmaps'),
                                      field_metadata=meta.get('field_metadata'), linked_sections=meta.get('linked_sections', False)):
                        self._vtk_cache_key = key
            elif renderer == "mesh":
                self._render_mesh_file(path)
            elif renderer == "table":
                self._render_table(path)
            elif renderer == "json":
                text = QTextEdit(); text.setReadOnly(True)
                text.setPlainText(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
                self._replace_visual(text)
            elif renderer == "text":
                self._replace_visual(_text_view(path))
            else:
                suffix = path.suffix.lstrip(".") or "no extension"
                self._clear_visual(
                    f"No preview available for {path.name} ({suffix}).\n\n"
                    "Use “Open Folder” to inspect it with another tool; the file "
                    "is listed with its size on the Files tab."
                )
        except Exception as exc:
            self._clear_visual(f"Could not render {path.name}:\n{exc}")

    def _replace_visual(
        self, widget: QWidget, *, resources: Optional[List[Any]] = None
    ) -> None:
        # Whatever is on screen owns "Add to Map"; dropping the handle here means
        # a new artifact cannot be mapped, or corrected, as the previous one.
        self._series = None
        self._series_source = None
        self._temperature_panel = None
        self._clear_visual()
        self._visual_resources.extend(resources or [])
        self._visual_layout.addWidget(widget)
        widget.show()
        compact = getattr(widget, 'set_compact', None)
        if callable(compact):
            compact(self._compact and not self._details_button.isChecked())

    def _compare_selected_models(self):
        records = self._selected_records()
        if len(records) != 2:
            self._clear_visual('Select exactly two runs with Ctrl-click to compare their models.')
            return
        try:
            import pyvista as pv
            from ..widgets.scientific_sections import ModelComparison
            models = []
            for record in records:
                candidates = [a for a in self._virtual_artifacts(record) if select_renderer(a) == 'vtk']
                if not candidates:
                    raise ValueError(f'{record.label} has no VTK model.')
                artifact = next((a for a in candidates if (a.get('metadata') or {}).get('linked_sections')), candidates[0])
                models.append(pv.read(self._store.locate_run_artifact(record, artifact)))
            view = ModelComparison(*models, names=tuple(r.label for r in records))
            self._replace_visual(view)
            self._tabs.setCurrentWidget(self._visual_page)
        except Exception as exc:
            self._clear_visual(f'Could not compare these models: {exc}')

    def _render_numpy(
        self,
        path: Path,
        renderer: str,
        artifact: Optional[Dict[str, Any]] = None,
    ) -> None:
        artifact = dict(artifact or {})
        title, value_label, log_scale = _artifact_plot_options(artifact, path)
        loaded = np.load(
            path,
            allow_pickle=False,
            mmap_mode="r" if path.suffix.lower() == ".npy" else None,
        )
        keep_open = False
        try:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                names = [
                    name for name in loaded.files
                    if np.asarray(loaded[name]).dtype.kind in "fiu"
                ]
                if not names:
                    raise ValueError("NPZ contains no numeric arrays.")
                metadata = artifact.get("metadata")
                metadata = dict(metadata) if isinstance(metadata, dict) else {}
                requested = str(metadata.get("array_key") or "")
                chosen = requested if requested in names else names[0]
                array = np.array(loaded[chosen], copy=True)
                loaded.close()
            else:
                array = loaded
            # Keep semantic dispatch from the artifact (notably a 2-D curve
            # table).  Shape only upgrades/downgrades the generic array modes.
            if renderer == "array" and array.ndim >= 3:
                renderer = "array_stack"
            elif renderer == "array_stack" and array.ndim < 3:
                renderer = "curve" if array.ndim == 1 else "array"
            if renderer == "curve" or array.ndim == 1:
                values = np.array(array, copy=True)
                view = CurveViewer()
                if values.ndim == 2 and min(values.shape) >= 2:
                    # A compact 2 x N / 3 x N array normally stores x and one
                    # or two series by row; a tall N x K array stores columns.
                    if values.shape[0] <= 4 and values.shape[1] > 2 * values.shape[0]:
                        values = values.T
                    x = values[:, 0]
                    for column in range(1, values.shape[1]):
                        view.add_curve(x, values[:, column], f"{value_label} {column}")
                else:
                    values = values.ravel()
                    view.add_curve(np.arange(values.size), values, value_label)
                self._replace_visual(view); return
            if array.ndim == 2:
                values = np.array(array, copy=True)
                view = ArrayViewer(colormaps=self._colormaps()); view.set_array(
                    values,
                    log=log_scale,
                    value_label=value_label,
                    title=title,
                )
                self._replace_visual(view); return
            if array.ndim >= 3:
                host = QWidget(); layout = QVBoxLayout(host)
                controls = QHBoxLayout(); step = QSpinBox()
                step.setRange(0, array.shape[0] - 1)
                controls.addWidget(QLabel("Step / slice:")); controls.addWidget(step); controls.addStretch(1)
                view = ArrayViewer(colormaps=self._colormaps())
                sample = np.asarray(array).ravel()[::max(1, array.size // 250_000)]
                if log_scale:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sample = np.log10(np.where(sample > 0, sample, np.nan))
                finite = sample[np.isfinite(sample)]
                limits = None
                if finite.size:
                    lo, hi = np.percentile(finite, [2, 98])
                    if hi <= lo:
                        half_span = 0.5 if log_scale else max(abs(float(lo)) * 0.05, 0.5)
                        lo, hi = float(lo) - half_span, float(hi) + half_span
                    limits = (float(lo), float(hi))
                def show_slice(value: int) -> None:
                    # Copy only the visible slice: the widget never owns a view
                    # of the file mapping, so Delete Run can close it first.
                    view.set_array(
                        np.array(array[value], copy=True),
                        autoscale=limits is None,
                        log=log_scale,
                        value_label=value_label,
                        title=f"{title} — slice {value + 1}",
                    )
                    if limits is not None:
                        view.set_levels(*limits)
                step.valueChanged.connect(show_slice); show_slice(0)
                layout.addLayout(controls); layout.addWidget(view, stretch=1)
                self._replace_visual(host, resources=[loaded])
                keep_open = True
                return
            raise ValueError(f"Unsupported array shape {array.shape}.")
        finally:
            if not keep_open:
                _close_array_resource(loaded)

    def _render_curve_file(self, path: Path) -> None:
        values = np.genfromtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None,
                               names=True, dtype=float, encoding="utf-8")
        names = list(values.dtype.names or [])
        if len(names) < 2:
            raise ValueError("Curve table needs at least two numeric columns.")
        view = CurveViewer()
        for name in names[1:]:
            view.add_curve(np.asarray(values[names[0]]), np.asarray(values[name]), name)
        self._replace_visual(view)

    def _render_table(self, path: Path) -> None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            rows = list(csv.reader(stream, delimiter=delimiter))[:5000]
        columns = max((len(row) for row in rows), default=0)
        table = QTableWidget(len(rows), columns)
        for row_index, row in enumerate(rows):
            for col, value in enumerate(row):
                table.setItem(row_index, col, QTableWidgetItem(value))
        self._replace_visual(table)

    def _render_mesh_file(self, path: Path) -> None:
        """Show a standalone PyGIMLi mesh, coloured by its region markers.

        A mesh on its own carries no field, but the geometry and the region
        layout are exactly what someone reviewing a mesh-building run wants to
        check, and it is the main output of that module.
        """
        from PyHydroGeophysX.core.mesh_serialization import read_bms
        from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView

        # Through read_bms, not pg.load: PyGIMLi cannot open a path the Windows
        # ANSI codepage cannot spell (a Project under a Chinese folder name), and
        # pg.load then hands back a matrix instead of failing.
        mesh = read_bms(path)
        markers = np.asarray(mesh.cellMarkers(), dtype=float)
        view = MeshResultView(colormaps=self._colormaps())
        view.show_field(
            mesh, markers,
            title=f"{path.name} — {mesh.cellCount()} cells, region markers",
        )
        self._replace_visual(view)

    def _step_titles(self, model: Any, mesh: Any) -> List[str]:
        """Acquisition dates for the steps, when the run recorded them.

        A saved run carries the titles its own panels used, so a result reopened
        months later is still headed by the survey date rather than by an index.
        """
        if self._current is None:
            return []
        summary = self._current.summary or {}
        for key in ("step_titles", "time_labels"):
            titles = summary.get(key)
            if isinstance(titles, list) and titles:
                return [str(t) for t in titles]
        # A single inversion is headed as the ERT page heads it, not as step 1
        # of a series it is not part of: with the survey's time when the run
        # recorded one.
        if np.ndim(model) == 1 or (np.ndim(model) == 2 and 1 in np.shape(model)):
            from PyHydroGeophysX.data_processing.survey_timing import format_time

            when = format_time(summary.get("survey_time") or "")
            return [f"Resistivity · {when}" if when else "Resistivity"]
        return []

    def _with_map_export(self, view: QWidget) -> QWidget:
        """Put the section view beside the actions that apply to it.

        A saved result is exactly the thing that should be corrected, compared and
        mapped without being inverted again: this is where someone opens a result a
        colleague handed over. The temperature correction is the panel the ERT page
        has, used the same way - set it, press Apply - next to the section it
        changes; the splitter lets it be dragged narrower or out of the way.
        """
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view, stretch=1)
        # Carried on the host rather than assigned here: the caller swaps the page
        # in afterwards, and that swap clears the page-level handles.
        panel = temperature_panel.TemperatureOptions(
            shared=getattr(self.state, "temperature_settings", None))
        host.temperature_panel = panel
        # Right of the section and under its toolbar, where the ERT page puts it
        # too. As wide as the panel needs, measured rather than guessed: with the
        # horizontal bar off, anything narrower cuts off Remove.
        side = ContentWidthScrollArea()
        side.setWidget(panel)
        view.mesh_view.add_side_panel(side)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.map_export_button())
        layout.addLayout(row)
        return host

    def _series_survey_times(self, n_steps: int):
        """``(days, dates)`` for the loaded series, from what the run recorded."""
        summary = (self._current.summary if self._current is not None else {}) or {}
        return temperature_panel.series_survey_times(summary, n_steps)

    def apply_temperature_spec(self, spec: Optional[Dict[str, Any]]) -> tuple:
        """Correct the displayed series, or put the inverted models back.

        What the panel's Apply and Remove call, and callable without them - by
        a test, or by a caller that already knows what it wants. The correction is
        the same code the ERT page uses, so a run reads the same on both pages.

        Returns ``(ok, message)``.
        """
        source = getattr(self, "_series_source", None)
        panel = getattr(self, "_temperature_panel", None)
        if self._series is None or not source:
            return False, "Open a saved model result in the Visualization tab first."
        models = source["models"]
        if models is None:
            return False, ("This run corrected its own sections and its uncorrected "
                           "models are missing, so they cannot be corrected again.")
        if spec is None:
            # Put the raw inverted models back rather than leaving a corrected
            # section on screen with nothing saying so.
            self._apply_series(models, None)
            self.log("Temperature correction removed; showing the inverted models.",
                     "info")
            return True, ""
        days, dates = self._series_survey_times(
            self._n_series_steps(source["mesh"], models))
        try:
            corrected, report = temperature_panel.correct_series(
                source["mesh"], models, spec, days=days, dates=dates)
        except Exception as exc:  # noqa: BLE001 - report it, never lose the result
            self.log(f"Temperature correction not applied: {exc}", "warn")
            if panel is not None:
                panel.show_problem(f"Not applied: {exc}")
            return False, str(exc)
        self._apply_series(corrected, report)
        self.log(f"Temperature correction: {report['note']}", "success")
        return True, report["note"]

    @staticmethod
    def _n_series_steps(mesh: Any, models: Any) -> int:
        """How many surveys a series holds, whichever way round it is stored."""
        models = np.asarray(models)
        if models.ndim == 1:
            return 1
        n_cells = int(mesh.cellCount())
        return int(models.shape[1] if models.shape[0] == n_cells else models.shape[0])

    def _connect_temperature_panel(self, shown: Any,
                                   made: Optional[Dict[str, Any]]) -> None:
        """Hook the panel beside a freshly opened series up to that series.

        ``shown`` is what the series view was given - the run's saved models -
        and ``made`` the correction the run applied to them itself, if it did.
        """
        panel = self._temperature_panel
        source = self._series_source
        if panel is None or not source:
            return
        n_steps = self._n_series_steps(source["mesh"], shown)
        days, dates = self._series_survey_times(n_steps)
        panel.set_context(n_steps, dates is not None, days=days, dates=dates)
        panel.applyRequested.connect(self.apply_temperature_spec)
        panel.removeRequested.connect(lambda: self.apply_temperature_spec(None))
        panel.set_available(True)
        if made is None:
            panel.show_applied(None)
            return
        # Corrected by the run itself: the section and the panel both say so.
        self._apply_series(shown, made)
        if source["models"] is None:
            panel.set_available(False, (
                f"This run corrected its own sections "
                f"({str(made.get('note', '')).replace('degC', '°C')}) and its "
                f"uncorrected models are missing, so the correction cannot be "
                f"changed here."))

    def _apply_series(self, models: Any, report: Optional[Dict[str, Any]]) -> None:
        """Redraw the series and say, on the page, what it is now showing."""
        source = getattr(self, "_series_source", None)
        if self._series is None or not source:
            return
        source["correction"] = report
        # The step and display mode stay where the user had them: a correction
        # changes the values, not which survey is being looked at.
        self._series.update_models(models, temperature_panel.title_suffix(report))
        panel = getattr(self, "_temperature_panel", None)
        if panel is not None:
            panel.show_applied(report)

    def map_snapshot(self):
        """What "Add to Map" takes from this page: exactly what is on screen.

        Including the display mode - a percentage-change section is added as a
        percentage-change survey, not silently as resistivity.
        """
        from PyHydroGeophysX.qt_apps.project_map import mesh_snapshot

        series = getattr(self, "_series", None)
        if series is None:
            raise ValueError(
                "Open a saved model result in the Visualization tab first.")
        mesh = getattr(series, "_mesh", None)
        values = series.current_values()
        if mesh is None or values is None:
            raise ValueError("The selected result has no model to add.")
        if series.step_count() == 1:
            # Named as the ERT page names a single inversion on the map: "ERT",
            # plus the temperature it was corrected to, if it was.
            correction = (getattr(self, "_series_source", None) or {}).get("correction")
            return mesh_snapshot(mesh, values, "ERT",
                                 "ERT" + temperature_panel.title_suffix(correction),
                                 coverage=series.current_coverage())
        title = series.current_title() or "ERT"
        if series.current_mode() == "change":
            return mesh_snapshot(mesh, values, "ERT", f"ERT change {title}",
                                 coverage=series.current_coverage(), units="%")
        return mesh_snapshot(mesh, values, "ERT", f"ERT {title}",
                             coverage=series.current_coverage())

    def _render_mesh_bundle(self, artifact: Dict[str, Any]) -> None:
        if self._current is None or self._store is None:
            return
        bundle = dict((artifact.get("metadata") or {}).get("bundle") or {})
        try:
            from PyHydroGeophysX.core.mesh_serialization import read_bms

            # Read as the ERT page reads the same bundle (_load_model_bundle):
            # staged through an ASCII path when the Project's own path is one
            # PyGIMLi cannot open, and without the cell-neighbour table.
            mesh = read_bms(self._store.locate_run_artifact(self._current, bundle["mesh"]),
                            neighbours=False)
            model_key = "models" if "models" in bundle else "model"
            model = np.load(
                self._store.locate_run_artifact(self._current, bundle[model_key]),
                allow_pickle=False,
            )
            coverage = None
            if bundle.get("coverage"):
                coverage = np.load(
                    self._store.locate_run_artifact(self._current, bundle["coverage"]),
                    allow_pickle=False,
                )
            # The same series view the processing modules use, so a result someone
            # else inverted is read here with the same controls - step through the
            # surveys, switch to percentage change, clip, smooth, add to the map -
            # instead of being a set of pictures to look at. A single model takes
            # the same path and simply hides the series controls.
            from PyHydroGeophysX.qt_apps.widgets.series_view import TimeLapseSeriesView

            titles = self._step_titles(model, mesh)
            # The studio state's colormap choices, shared with the ERT page: a
            # section reopened here is drawn in the colours chosen there.
            view = TimeLapseSeriesView(colormaps=self._colormaps())
            view.set_series(mesh, model, coverage=coverage, titles=titles)
            host = self._with_map_export(view)
            # After the swap, not before: _replace_visual drops the previous
            # page's handles so a stale one cannot be mapped or corrected.
            self._replace_visual(host)
            self._series = view
            self._temperature_panel = host.temperature_panel
            # Keep the models as inverted, so a temperature correction can be
            # applied, changed and taken off again without reloading the run. A
            # run made while the correction was chosen before the inversion
            # corrected its own series; that one starts from its raw models, or
            # the temperature would be taken out twice.
            inverted, made = temperature_panel.run_correction(
                self._current.summary, model,
                lambda path: self._store.locate_run_artifact(self._current, path))
            self._series_source = {"mesh": mesh, "models": inverted,
                                   "coverage": coverage, "titles": titles,
                                   "correction": None}
            self._connect_temperature_panel(model, made)
        except Exception as exc:
            self._clear_visual(f"Mesh viewer is unavailable or the bundle is incomplete:\n{exc}")

    def _save_current_run(self) -> None:
        """Put the selected run into the Project's history."""
        if self._current is None or self._store is None:
            return
        # Any label or notes typed above are part of what is being saved; without
        # this they would be written only on the next edit, after the record has
        # already gone to disk.
        try:
            self._store.update_run(
                self._current.run_id,
                label=self._label.text(),
                notes=self._notes.toPlainText(),
            )
            record = self._store.save_run(self._current.run_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save run", str(exc))
            return
        self.log(f"Saved '{_run_title(record)}' to {self._store.root}", "success")
        self.refresh()

    def _save_current_run_named(self) -> None:
        """Save to Project: confirm or change the run's name, then save it.

        The button's own path. The assistant saves through
        :meth:`_save_current_run`, which never opens a dialog.
        """
        if self._current is None or self._store is None:
            return
        self._commit_label()          # a name typed above counts
        dialog = SaveRunsDialog(self, [self._current], title="Save Run to Project")
        if dialog.exec() != QDialog.Accepted:
            return
        name = dialog.names().get(self._current.run_id)
        if name:
            self._label.setText(name)
        self._save_current_run()

    def _save_metadata(self) -> None:
        if self._current is None or self._store is None or self._store.read_only:
            return
        try:
            self._current = self._store.update_run(
                self._current.run_id, label=self._label.text(), notes=self._notes.toPlainText()
            )
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Save metadata", str(exc))

    def _delete_run(self) -> None:
        if self._current is None or self._store is None:
            return
        record_id = self._current.run_id
        pending = record_id in self._unsaved_ids
        size_value = self._size_cache.get(self._current.run_id)
        if size_value is None:
            size_value = self._store.run_size(self._current)
            self._size_cache[self._current.run_id] = size_value
        size = _human_size(size_value)
        answer = QMessageBox.question(
            self,
            "Discard unsaved run" if pending else "Delete run permanently",
            f"Permanently delete '{_run_title(self._current)}' and {size} of files?\n\n"
            + ("This run was never saved to the Project. This cannot be undone."
               if pending else "This cannot be undone."),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        # Windows refuses to unlink a file another handle still maps, so the
        # view has to be gone before the run directory is removed, not merely
        # queued for removal. Flush the widgets this call retired and no
        # others: passing no receiver would deliver every pending deferred
        # delete in the application, which reaches objects this module does not
        # own and cannot reason about.
        for widget in self._clear_visual():
            QApplication.sendPostedEvents(widget, QEvent.DeferredDelete)
        QApplication.processEvents()
        gc.collect()
        try:
            self._store.delete_run(self._current.run_id)
        except Exception as exc:
            QMessageBox.warning(self, "Delete run", str(exc)); return
        self._current = None
        self._size_cache.pop(record_id, None)
        self.refresh()


    # -- agent command interface ----------------------------------------------
    _AGENT_TABS = ("overview", "metrics", "visualization", "files", "report")

    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "get_status", "args": {},
                 "desc": ("The Project shown, how many runs (and unsaved ones) it holds, "
                          "the filter, and the run selected with its artifacts.")},
                {"name": "list_runs",
                 "args": {"status": "str (optional)", "search": "str (optional)",
                          "limit": "int (default 30)"},
                 "desc": "Runs, newest first: id, label, module, operation, status, when, unsaved."},
                {"name": "select_run", "args": {"run": "id or label", "compare": "id or label (optional)"},
                 "desc": ("Select a run to show its overview, metrics and results; with "
                          "'compare', two runs side by side in Metrics.")},
                {"name": "get_overview", "args": {"max_chars": "int (default 4000)"},
                 "desc": "The selected run's overview as text: status, timing, numbers, record."},
                {"name": "set_filter", "args": {"search": "str", "status": "str"},
                 "desc": "Filter the run list by text and/or status (All statuses to clear)."},
                {"name": "show_tab", "args": {"tab": list(self._AGENT_TABS)},
                 "desc": "Show the Overview, Metrics, Visualization or Files tab."},
                {"name": "show_artifact", "args": {"artifact": "index or name"},
                 "desc": "Show one of the selected run's artifacts in Visualization."},
                {"name": "set_label_notes", "args": {"label": "str", "notes": "str"},
                 "desc": "Name the selected run and/or write its notes (current Project only)."},
                {"name": "save_run", "args": {},
                 "desc": "Add the selected unsaved run to the Project's history."},
                {"name": "use_current_project", "args": {},
                 "desc": "Show the Project that new computations are written to."},
                {"name": "refresh", "args": {}, "desc": "Re-read the run history from disk."},
            ],
            "note": "Deleting or discarding a run is left to the user (Delete Run…).",
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "get_status": self._agent_status,
            "list_runs": lambda: self._agent_list(args.get("status"), args.get("search"),
                                                  args.get("limit", 30)),
            "select_run": lambda: self._agent_select(args.get("run"), args.get("compare")),
            "get_overview": lambda: self._agent_overview(args.get("max_chars", 4000)),
            "set_filter": lambda: self._agent_filter(args),
            "show_tab": lambda: self._agent_show_tab(args.get("tab")),
            "show_artifact": lambda: self._agent_show_artifact(args.get("artifact")),
            "set_label_notes": lambda: self._agent_label_notes(args),
            "save_run": self._agent_save,
            "use_current_project": lambda: (self.use_current_store(), self._agent_status())[1],
            "refresh": lambda: (self.refresh(), self._agent_status())[1],
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers)}
        return handler()

    def _agent_record(self, record: RunRecord) -> Dict[str, Any]:
        return {"run_id": record.run_id, "label": _run_title(record),
                "module": record.module_key, "operation": record.operation_id,
                "status": record.status, "when": _local_time(record.created_at),
                "took": _duration(record.created_at, record.finished_at),
                "unsaved": record.run_id in self._unsaved_ids,
                "folder": str(record.run_dir)}

    def _agent_status(self) -> Dict[str, Any]:
        current = self._current
        return {
            "status": "ok",
            "project": str(self._store.root) if self._store is not None else None,
            "read_only": bool(self._store.read_only) if self._store is not None else None,
            "runs": len(self._records),
            "unsaved": len(self._unsaved_ids),
            "filter": {"search": self._search.text(), "status": self._status.currentText()},
            "selected": self._agent_record(current) if current is not None else None,
            "compared": [r.run_id for r in self._selected_records()[1:2]],
            "artifacts": [self._artifact.itemText(i) for i in range(self._artifact.count())],
            "artifact": self._artifact.currentText(),
            "tab": self._AGENT_TABS[self._tabs.currentIndex()]
            if 0 <= self._tabs.currentIndex() < len(self._AGENT_TABS) else "",
        }

    def _agent_list(self, status=None, search=None, limit=30) -> Dict[str, Any]:
        records = sorted(self._records.values(), key=lambda r: r.created_at or "", reverse=True)
        if status and status != "All statuses":
            records = [r for r in records if r.status == status]
        if search:
            needle = str(search).lower()
            records = [r for r in records if needle in " ".join(
                [r.label, r.notes, r.module_key, r.operation_id, r.workflow_id, r.run_id]).lower()]
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 30
        return {"status": "ok", "total": len(records),
                "runs": [self._agent_record(r) for r in records[:limit]]}

    def _agent_resolve(self, run) -> tuple:
        key = str(run or "").strip()
        if not key:
            return None, "Provide 'run' (an id or a label)."
        if key in self._records:
            return self._records[key], ""
        found = [r for r in self._records.values()
                 if key.lower() in (r.label or "").lower() or key.lower() in _run_title(r).lower()]
        if len(found) == 1:
            return found[0], ""
        if not found:
            return None, f"No run matches '{key}'."
        return None, (f"'{key}' matches {len(found)} runs; use an id: "
                      + ", ".join(r.run_id for r in found[:8]))

    def _agent_items(self, run_ids) -> Dict[str, QTreeWidgetItem]:
        wanted, found = set(run_ids), {}

        def walk(item: QTreeWidgetItem) -> None:
            run_id = item.data(0, _RUN_ROLE)
            if run_id in wanted:
                found[run_id] = item
            for index in range(item.childCount()):
                walk(item.child(index))

        for index in range(self._tree.topLevelItemCount()):
            walk(self._tree.topLevelItem(index))
        return found

    def _agent_select(self, run, compare=None) -> Dict[str, Any]:
        records = []
        for which in [run] + ([compare] if compare else []):
            record, problem = self._agent_resolve(which)
            if record is None:
                return {"status": "failed", "error": problem}
            records.append(record)
        items = self._agent_items([r.run_id for r in records])
        missing = [r.run_id for r in records if r.run_id not in items]
        if missing:
            return {"status": "failed", "error": f"Not in the list: {', '.join(missing)}."}
        self._tree.blockSignals(True)
        self._tree.clearSelection()
        # The run asked for first is the one shown; the tree reports a
        # selection in the order it was made.
        for record in records:
            item = items[record.run_id]
            item.setHidden(False)
            item.setSelected(True)
        self._tree.scrollToItem(items[records[0].run_id])
        self._tree.blockSignals(False)
        self._selection_changed()
        return self._agent_status()

    def _agent_overview(self, max_chars=4000) -> Dict[str, Any]:
        if self._current is None:
            return {"status": "failed", "error": "No run is selected."}
        try:
            limit = max(200, int(max_chars))
        except (TypeError, ValueError):
            limit = 4000
        text = self._overview.toPlainText()
        return {"status": "ok", "run_id": self._current.run_id,
                "overview": text[:limit], "truncated": len(text) > limit}

    def _agent_filter(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "status" in args:
            index = self._status.findText(str(args["status"] or "All statuses"))
            if index < 0:
                return {"status": "failed", "error": f"Unknown status '{args['status']}'.",
                        "statuses": [self._status.itemText(i) for i in range(self._status.count())]}
            self._status.setCurrentIndex(index)
        if "search" in args:
            self._search.setText(str(args["search"] or ""))
        return self._agent_status()

    def _agent_show_tab(self, tab) -> Dict[str, Any]:
        name = str(tab or "").lower()
        if name not in self._AGENT_TABS:
            return {"status": "failed", "error": f"Unknown tab '{tab}'.", "tabs": list(self._AGENT_TABS)}
        self._tabs.setCurrentIndex(self._AGENT_TABS.index(name))
        return {"status": "ok", "tab": name}

    def _agent_show_artifact(self, artifact) -> Dict[str, Any]:
        names = [self._artifact.itemText(i) for i in range(self._artifact.count())]
        if not names:
            return {"status": "failed", "error": "The selected run has no artifacts to show."}
        index = -1
        if isinstance(artifact, (int, float)) or str(artifact).strip().isdigit():
            index = int(artifact)
        else:
            key = str(artifact or "").lower()
            matches = [i for i, name in enumerate(names) if key and key in name.lower()]
            index = matches[0] if len(matches) == 1 else -1
        if not 0 <= index < len(names):
            return {"status": "failed", "error": f"No single artifact matches '{artifact}'.",
                    "artifacts": names}
        self._tabs.setCurrentIndex(self._AGENT_TABS.index("visualization"))
        self._artifact.setCurrentIndex(index)
        return {"status": "ok", "artifact": names[index]}

    def _agent_label_notes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._current is None or self._store is None:
            return {"status": "failed", "error": "No run is selected."}
        if self._store.read_only or self._store is not self.state.results_store:
            return {"status": "failed", "error": "This Project is open read-only."}
        if "label" not in args and "notes" not in args:
            return {"status": "failed", "error": "Give 'label' and/or 'notes'."}
        if "label" in args:
            self._label.setText(str(args["label"] or ""))
        if "notes" in args:
            self._notes.setPlainText(str(args["notes"] or ""))
        run_id = self._current.run_id
        self._save_metadata()
        record = self._records.get(run_id)
        return {"status": "ok", "run": self._agent_record(record) if record else run_id}

    def _agent_save(self) -> Dict[str, Any]:
        if self._current is None:
            return {"status": "failed", "error": "No run is selected."}
        if self._current.run_id not in self._unsaved_ids:
            return {"status": "ok", "detail": "This run is already in the Project's history."}
        if self._current.status == "running":
            return {"status": "failed", "error": "A running computation cannot be saved yet."}
        run_id = self._current.run_id
        self._save_current_run()
        saved = run_id not in self._unsaved_ids
        return ({"status": "ok", "run_id": run_id} if saved else
                {"status": "failed", "error": "The run could not be saved; see the log."})

__all__ = ["ModelViewerModule"]
