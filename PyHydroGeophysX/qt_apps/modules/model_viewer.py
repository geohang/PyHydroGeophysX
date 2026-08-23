"""Read-only-first browser for durable studio Result Store runs."""

from __future__ import annotations

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
    QComboBox,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
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
from PyHydroGeophysX.qt_apps.results_store import ResultsStore, RunRecord
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.curve_viewer import CurveViewer
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.workers import TaskWorker


_RUN_ROLE = Qt.UserRole

#: Symbol, colour and wording per Result Store status. A run list is scanned, not
#: read, so the outcome has to survive peripheral vision.
_STATUS_DISPLAY = {
    "success": ("✓", "#2e7d32", "Succeeded"),
    "failed": ("✕", "#c62828", "Failed"),
    "cancelled": ("⊘", "#ef6c00", "Cancelled"),
    "interrupted": ("⚠", "#ef6c00", "Interrupted"),
    "running": ("●", "#1565c0", "Running"),
    "unknown": ("?", "#616161", "Unknown"),
}

#: Metrics worth putting in the run summary before the raw record.
_HEADLINE_METRICS = ("chi2", "rrms", "mean_chi2", "iterations", "n_data", "lambda")

#: Heading for runs this session produced that are not in the Project yet.
_UNSAVED_GROUP = "⬤ Unsaved (this session)"
_UNSAVED_COLOUR = "#1565c0"


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
    """Name an artifact by its file, not by its internal id.

    ``artifact_id`` values like ``output:outputs/qc.png`` are how the record
    addresses a file; a chooser should show what the file is.
    """
    path_value = str(artifact.get("path") or "")
    kind = _pretty_kind(artifact.get("kind", ""))
    if not path_value:
        return f"{kind} (in record)"
    label = f"{Path(path_value).name} — {kind}"
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
    label = str(record.label or "").strip()
    default_prefix = f"{record.operation_id} · "
    if label and not label.startswith(default_prefix):
        return label
    suffix = str(record.run_id).rpartition("_")[2]
    return f"Run {suffix}" if suffix else str(record.run_id)


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
        self._size_cache: Dict[str, int] = {}
        self._visual_resources: List[Any] = []
        self._quality_ok = False

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self._path = QLabel("Project: (not available)")
        self._path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self._path, stretch=1)
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
            "All statuses", "success", "failed", "cancelled", "interrupted", "running", "unknown"
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
        left_layout.addWidget(self._tree, stretch=1)
        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setVisible(False)
        left_layout.addWidget(self._hint)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        edit_row = QHBoxLayout()
        self._label = QLineEdit()
        self._label.setPlaceholderText("Run label")
        self._size = QLabel("Size: —")
        save_meta = QPushButton("Save label/notes")
        save_meta.setToolTip("Name this run so you can tell it apart later.")
        save_meta.clicked.connect(self._save_metadata)
        self._save_run = QPushButton("Save to Project")
        self._save_run.setToolTip(
            "Add this run to the Project's history. Until then it exists only as "
            "a folder this session wrote, and closing the studio will ask."
        )
        self._save_run.setVisible(False)
        self._save_run.clicked.connect(self._save_current_run)
        self._open_run = QPushButton("Open Folder")
        self._open_run.setToolTip("Open this run's folder in the system file manager.")
        self._open_run.clicked.connect(self.open_run_folder)
        self._delete = QPushButton("Delete Run…")
        self._delete.clicked.connect(self._delete_run)
        edit_row.addWidget(self._label, stretch=1)
        edit_row.addWidget(self._size)
        edit_row.addWidget(save_meta)
        edit_row.addWidget(self._save_run)
        edit_row.addWidget(self._open_run)
        edit_row.addWidget(self._delete)
        right_layout.addLayout(edit_row)
        self._notes = QTextEdit()
        self._notes.setPlaceholderText("Notes")
        self._notes.setMaximumHeight(75)
        right_layout.addWidget(self._notes)

        self._tabs = QTabWidget()
        self._overview = QTextEdit(); self._overview.setReadOnly(True)
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
        visual_layout.addLayout(visual_bar)
        self._visual_host = QWidget()
        self._visual_layout = QVBoxLayout(self._visual_host)
        self._visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.addWidget(self._visual_host, stretch=1)
        self._files = QTableWidget(0, 5)
        self._files.setHorizontalHeaderLabels(["Kind", "Format", "Path", "Exists", "Size"])
        self._tabs.addTab(self._overview, "Overview")
        self._tabs.addTab(self._metrics_page, "Metrics")
        self._tabs.addTab(self._visual_page, "Visualization")
        self._tabs.addTab(self._files, "Files")
        right_layout.addWidget(self._tabs, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)
        self.use_current_store()

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

    def refresh(self) -> None:
        if self._store is None:
            return
        read_only = " — read-only" if self._store.read_only else ""
        self._path.setText(f"Project: {self._store.root}{read_only}")
        self._path.setToolTip(
            "Runs are browsed read-only; new computations still go to the active Project."
            if self._store.read_only
            else "New computations are written here, one folder per run."
        )
        self._tree.clear()
        # Unsaved runs lead the list. They are the ones that disappear if the
        # session ends without a decision, so they should not be found by
        # scrolling past a year of history.
        unsaved = self._store.list_unsaved_runs() if not self._store.read_only else []
        self._unsaved_ids = {record.run_id for record in unsaved}
        self._records = {item.run_id: item for item in [*unsaved, *self._store.list_runs()]}
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
                zip(path, (module_name, operation_name, date_name))
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
            item.setForeground(1, QColor(colour))
            tooltip = [f"Run ID: {record.run_id}", f"Folder: {record.run_dir}"]
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
            groups[path[2]].addChild(item)
            for key in path:
                groups[key].setExpanded(True)

        for key, group in groups.items():
            total = counts.get(key, 0)
            group.setText(0, f"{group.text(0)}  ({total})")
        for column in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(column)
        if not self._records:
            self._reset_details()
        self._apply_filter()

    def _reset_details(self) -> None:
        """Leave no stale run on screen when the list becomes empty."""
        self._current = None
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
            self._hint.setText(
                "No runs in this Project yet. Run a computation from any module; it "
                "appears here under “Unsaved” until you save it to the Project."
            )
            self._hint.setVisible(True)
            return
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
        if not selected:
            return
        self._current = selected[0]
        record = self._current
        self._label.setText(record.label)
        self._notes.setPlainText(record.notes)
        editable = bool(
            self._store is self.state.results_store
            and self._store is not None and not self._store.read_only
        )
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
        artifacts = [dict(item) for item in record.artifacts]
        registered = {str(item.get("path") or "") for item in artifacts}
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
        for key in ("model_bundle", "fixed_model_bundle"):
            bundle = record.summary.get(key)
            if isinstance(bundle, dict) and bundle:
                artifacts.insert(0, {
                    "artifact_id": f"{record.run_id}:{key}",
                    "kind": "ert_model_bundle",
                    "format": "bundle",
                    "path": "",
                    "metadata": {"bundle": bundle},
                })
        return artifacts

    def _populate_artifacts(self, record: RunRecord) -> None:
        assert self._store is not None
        self._current_artifacts = self._virtual_artifacts(record)
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
                self._files.setItem(row, col, cell)
        self._files.resizeColumnsToContents()
        self._artifact.blockSignals(False)
        if self._current_artifacts:
            self._artifact.setCurrentIndex(0)
            self._render_selected_artifact(0)
        else:
            self._clear_visual("This run has no registered artifacts.")

    def _clear_visual(self, message: str = "") -> None:
        while self._visual_layout.count():
            item = self._visual_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for resource in self._visual_resources:
            _close_array_resource(resource)
        self._visual_resources.clear()
        if message:
            label = QLabel(message)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)
            self._visual_layout.addWidget(label)

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
            elif renderer == "vtk":
                from PyHydroGeophysX.qt_apps.widgets.model3d_view import VTKVolumeView
                view = VTKVolumeView(); view.show_file(path)
                self._replace_visual(view)
            elif renderer == "mesh":
                self._render_mesh_file(path)
            elif renderer == "table":
                self._render_table(path)
            elif renderer == "json":
                text = QTextEdit(); text.setReadOnly(True)
                text.setPlainText(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
                self._replace_visual(text)
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
        self._clear_visual()
        self._visual_resources.extend(resources or [])
        self._visual_layout.addWidget(widget)

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
                view = ArrayViewer(); view.set_array(
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
                view = ArrayViewer()
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
        import pygimli as pg
        from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView

        mesh = pg.load(str(path))
        markers = np.asarray(mesh.cellMarkers(), dtype=float)
        view = MeshResultView()
        view.show_field(
            mesh, markers,
            title=f"{path.name} — {mesh.cellCount()} cells, region markers",
        )
        self._replace_visual(view)

    def _render_mesh_bundle(self, artifact: Dict[str, Any]) -> None:
        if self._current is None or self._store is None:
            return
        bundle = dict((artifact.get("metadata") or {}).get("bundle") or {})
        try:
            import pygimli as pg
            from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
            mesh = pg.load(str(self._store.locate_run_artifact(self._current, bundle["mesh"])))
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
            if model.ndim == 1:
                view = MeshResultView(); view.show_field(mesh, model, coverage=coverage)
                self._replace_visual(view)
            else:
                host = QWidget(); layout = QVBoxLayout(host); step = QSpinBox()
                n_steps = model.shape[1] if model.shape[0] == mesh.cellCount() else model.shape[0]
                step.setRange(0, n_steps - 1)
                view = MeshResultView()
                def show(value: int) -> None:
                    values = model[:, value] if model.shape[0] == mesh.cellCount() else model[value]
                    cov = None
                    if coverage is not None:
                        cov = coverage[:, value] if coverage.ndim > 1 and coverage.shape[0] == mesh.cellCount() else coverage[value] if coverage.ndim > 1 else coverage
                    view.show_field(mesh, values, coverage=cov, title=f"Time step {value + 1}")
                step.valueChanged.connect(show); show(0)
                row = QHBoxLayout(); row.addWidget(QLabel("Time step:")); row.addWidget(step); row.addStretch(1)
                layout.addLayout(row); layout.addWidget(view, stretch=1)
                self._replace_visual(host)
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
        self.log(f"Saved '{record.label}' to {self._store.root}", "success")
        self.refresh()

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
            f"Permanently delete '{self._current.label}' and {size} of files?\n\n"
            + ("This run was never saved to the Project. This cannot be undone."
               if pending else "This cannot be undone."),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._clear_visual()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()
        gc.collect()
        try:
            self._store.delete_run(self._current.run_id)
        except Exception as exc:
            QMessageBox.warning(self, "Delete run", str(exc)); return
        self._current = None
        self._size_cache.pop(record_id, None)
        self.refresh()


__all__ = ["ModelViewerModule"]
