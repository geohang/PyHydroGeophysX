"""The studio's borehole data: one store, edited on the Boreholes page.

The wells, their lithology logs, water levels and borehole geophysical logs are
the studio's one :class:`BoreholeStore` (:func:`store_for`), loaded and edited
on the Boreholes page with :class:`BoreholeDataPanel`. They are compared with
the surveys on the Project Map, where the page adds them as a wells layer; the
method pages do not draw them.

Reading and drawing are the package's
(:mod:`PyHydroGeophysX.data_processing.boreholes`); this module only edits and
remembers.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import boreholes as bh
from PyHydroGeophysX.qt_apps import theme

__all__ = ["BoreholeStore", "BoreholeDataPanel", "store_for"]

_TABLE_FILTER = "Tables (*.csv *.txt *.xlsx *.xls);;All files (*)"
_LOG_FILTER = "Logs (*.las *.csv *.txt *.xlsx *.xls);;LAS (*.las);;Tables (*.csv *.txt *.xlsx *.xls)"


class BoreholeStore(QObject):
    """The studio's wells, lithology, water levels, geophysical logs and legend.

    Loaded and edited on the Boreholes page; ``changed`` fires after any edit,
    and the page's views redraw on it.
    """

    changed = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.data = bh.BoreholeData()
        self.sources: Dict[str, str] = {}

    def has_wells(self) -> bool:
        return bool(self.data.wells)

    def describe(self) -> str:
        """What is loaded, in one line."""
        data = self.data
        counts = [(len(data.wells), "well"), (len(data.lithology), "log interval"),
                  (len(data.water_levels), "water level"),
                  (len({(g.well_id, g.name) for g in data.logs}), "geophysical log")]
        if not any(n for n, _ in counts):
            return "No boreholes loaded."
        return ", ".join(f"{n} {what}{'s' if n != 1 else ''}" for n, what in counts) + "."

    def notify(self) -> None:
        self.changed.emit()


_FALLBACK: Optional[BoreholeStore] = None


def store_for(state: Any) -> BoreholeStore:
    """The studio's one borehole store, kept in ``state.borehole_settings``."""
    global _FALLBACK
    settings = getattr(state, "borehole_settings", None)
    if isinstance(settings, dict):
        store = settings.get("store")
        if not isinstance(store, BoreholeStore):
            store = BoreholeStore()
            settings["store"] = store
        return store
    if _FALLBACK is None:
        _FALLBACK = BoreholeStore()
    return _FALLBACK


def _table(headers: Sequence[str], height: int = 130) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setMinimumHeight(height)
    table.setMaximumHeight(height + 60)
    return table


def _cell(value: Any) -> QTableWidgetItem:
    if isinstance(value, float):
        # Enough digits for a UTM northing to the centimetre.
        text = "" if not np.isfinite(value) else f"{value:.10g}"
    elif isinstance(value, _dt.datetime):
        text = value.strftime("%Y-%m-%d %H:%M") if (value.hour, value.minute) != (0, 0) \
            else value.strftime("%Y-%m-%d")
    else:
        text = "" if value is None else str(value)
    return QTableWidgetItem(text)


def _float(text: str) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        return float("nan")


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    theme.set_tone(label, "hint")
    return label


# ---------------------------------------------------------------------------
# The Boreholes page's panel: the data
# ---------------------------------------------------------------------------

class BoreholeDataPanel(QWidget):
    """Load and edit the studio's boreholes; draws nothing itself.

    Two group boxes in the order the work is done: the data (load buttons, then
    the wells, lithology, water levels and geophysical logs, each in a table,
    all editable but the logs) and the lithology legend. What any one section
    shows of them is that page's choice (:class:`SectionWellsPanel`).
    """

    def __init__(self, store: BoreholeStore, parent: Optional[QWidget] = None, *,
                 log: Optional[Callable[[str, str], None]] = None) -> None:
        super().__init__(parent)
        self._store = store
        self._log = log or (lambda message, level="info": None)
        self._syncing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        load_box = QGroupBox("Load")
        load_layout = QVBoxLayout(load_box)
        rows = (
            (("Wells…", "wells", "Well locations: well_id, easting, northing, ground_elevation, "
                                 "crs, datum_offset, source (CSV or XLSX)."),
             ("Lithology…", "lithology", "Lithology intervals: well_id, top, bottom, unit, "
                                         "description, length_unit (m or ft).")),
            (("Water levels…", "water", "Water levels: well_id, datetime, and depth_below_ground "
                                        "or head_elevation; optional reference, "
                                        "reference_height, length_unit."),
             ("Geophysical logs…", "logs", "Borehole geophysics - gamma, conductivity, "
                                           "resistivity, temperature ... - as a LAS 2.0 file, or a "
                                           "table of well_id, depth and one column per curve "
                                           "('gamma [API]').")),
        )
        for row in rows:
            line = QHBoxLayout()
            for label, kind, tip in row:
                button = QPushButton(label)
                button.setToolTip(tip)
                button.clicked.connect(lambda _=False, k=kind: self._load(k))
                line.addWidget(button)
            load_layout.addLayout(line)
        tools = QHBoxLayout()
        templates = QPushButton("Templates…")
        templates.setToolTip("Write the column templates, each with an example row, to a folder.")
        templates.clicked.connect(self._write_templates)
        clear = QPushButton("Clear")
        clear.setToolTip("Remove every well, log and water level.")
        clear.clicked.connect(self._clear)
        tools.addWidget(templates)
        tools.addWidget(clear)
        tools.addStretch(1)
        load_layout.addLayout(tools)
        self._summary = _hint("")
        load_layout.addWidget(self._summary)
        layout.addWidget(load_box)

        data_box = QGroupBox("Data")
        data_layout = QVBoxLayout(data_box)
        self._tabs = QTabWidget()
        self._wells = _table(["Well", "Easting", "Northing", "Ground (m)", "Datum offset (m)",
                              "CRS", "Source"])
        self._lith = _table(["Well", "Top (m)", "Bottom (m)", "Unit", "Description"])
        self._levels = _table(["Well", "Date", "Depth (m)", "Head (m)", "Reference"])
        self._logs = _table(["Well", "Curve", "Unit", "Depths (m)", "Source"])
        self._logs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for table in (self._wells, self._lith, self._levels):
            table.itemChanged.connect(self._on_table_edited)
        self._tabs.addTab(self._wells, "Wells")
        self._tabs.addTab(self._lith, "Lithology")
        self._tabs.addTab(self._levels, "Water levels")
        self._tabs.addTab(self._logs, "Geophysical logs")
        self._tabs.currentChanged.connect(self._sync_row_buttons)
        data_layout.addWidget(self._tabs)
        row_buttons = QHBoxLayout()
        self._add = QPushButton("Add row")
        self._add.clicked.connect(self._add_row)
        self._remove = QPushButton("Remove row")
        self._remove.clicked.connect(self._remove_rows)
        row_buttons.addWidget(self._add)
        row_buttons.addWidget(self._remove)
        row_buttons.addStretch(1)
        data_layout.addLayout(row_buttons)
        layout.addWidget(data_box)

        legend_box = QGroupBox("Lithology legend")
        lform = QVBoxLayout(legend_box)
        self._legend = _table(["Unit", "Colour", "Label"], height=100)
        self._legend.itemChanged.connect(self._on_legend_edited)
        self._legend.cellDoubleClicked.connect(self._pick_colour)
        lform.addWidget(self._legend)
        lform.addWidget(_hint("Double-click a colour to change it; edit a label to rename it "
                              "in the legend."))
        layout.addWidget(legend_box)

        store.changed.connect(self._sync_from_store)
        self._sync_from_store()

    # -- store -> widgets ----------------------------------------------------
    def _sync_from_store(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._fill_tables()
            self._summary.setText(self._store.describe())
            self._sync_row_buttons()
        finally:
            self._syncing = False

    def _sync_row_buttons(self, *_args: Any) -> None:
        editable = self._tabs.currentWidget() is not self._logs
        self._add.setEnabled(editable)
        self._remove.setEnabled(True)

    def _fill_tables(self) -> None:
        data = self._store.data
        rows = {
            self._wells: [(w.well_id, w.easting, w.northing, w.ground_elevation, w.datum_offset,
                           w.crs, w.source) for w in data.wells.values()],
            self._lith: [(i.well_id, i.top, i.bottom, i.unit, i.description) for i in data.lithology],
            self._levels: [(w.well_id, w.time, w.depth, w.head, w.reference) for w in data.water_levels],
            self._logs: [(g.well_id, g.name, g.unit,
                          f"{g.depth.min():.2f}-{g.depth.max():.2f}" if g.depth.size else "",
                          g.source) for g in data.logs],
        }
        for table, values in rows.items():
            table.blockSignals(True)
            table.setRowCount(len(values))
            for r, row in enumerate(values):
                for c, value in enumerate(row):
                    table.setItem(r, c, _cell(value))
            table.resizeColumnsToContents()
            table.blockSignals(False)
        self._legend.blockSignals(True)
        self._legend.setRowCount(len(data.legend))
        for r, (unit, (colour, label)) in enumerate(data.legend.items()):
            unit_item = QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            swatch = QTableWidgetItem("")
            swatch.setFlags(swatch.flags() & ~Qt.ItemIsEditable)
            swatch.setBackground(QColor(colour))
            swatch.setToolTip(colour)
            self._legend.setItem(r, 0, unit_item)
            self._legend.setItem(r, 1, swatch)
            self._legend.setItem(r, 2, QTableWidgetItem(label))
        self._legend.blockSignals(False)

    # -- widgets -> store ----------------------------------------------------
    def _on_table_edited(self, _item: QTableWidgetItem) -> None:
        if self._syncing:
            return
        data = self._store.data
        wells = {}
        for r in range(self._wells.rowCount()):
            get = lambda c: (self._wells.item(r, c).text() if self._wells.item(r, c) else "")
            well_id = get(0).strip()
            if not well_id:
                continue
            offset = _float(get(4))
            wells[well_id] = bh.Well(well_id, _float(get(1)), _float(get(2)), _float(get(3)),
                                     get(5).strip(), offset if np.isfinite(offset) else 0.0,
                                     get(6).strip())
        lith = []
        for r in range(self._lith.rowCount()):
            get = lambda c: (self._lith.item(r, c).text() if self._lith.item(r, c) else "")
            top, bottom = _float(get(1)), _float(get(2))
            if get(0).strip() and np.isfinite(top) and np.isfinite(bottom):
                lith.append(bh.LithologyInterval(get(0).strip(), min(top, bottom), max(top, bottom),
                                                 get(3).strip() or "unknown", get(4).strip()))
        levels = []
        for r in range(self._levels.rowCount()):
            get = lambda c: (self._levels.item(r, c).text() if self._levels.item(r, c) else "")
            depth, head = _float(get(2)), _float(get(3))
            if get(0).strip() and (np.isfinite(depth) or np.isfinite(head)):
                levels.append(bh.WaterLevel(get(0).strip(), bh._parse_time(get(1)), depth, head,
                                            get(4).strip()))
        data.wells, data.lithology, data.water_levels = wells, lith, levels
        data.complete_legend()
        self._store.notify()

    def _on_legend_edited(self, item: QTableWidgetItem) -> None:
        if self._syncing or item.column() != 2:
            return
        unit = self._legend.item(item.row(), 0).text()
        colour, _ = self._store.data.legend.get(unit, (theme.LIGHT["border_blue"], unit))
        self._store.data.legend[unit] = (colour, item.text())
        self._store.notify()

    def _pick_colour(self, row: int, column: int) -> None:
        if column != 1:
            return
        unit = self._legend.item(row, 0).text()
        colour, label = self._store.data.legend.get(unit, (theme.LIGHT["border_blue"], unit))
        chosen = QColorDialog.getColor(QColor(colour), self, f"Colour for {unit}")
        if chosen.isValid():
            self._store.data.legend[unit] = (chosen.name(), label)
            self._store.notify()

    def _add_row(self) -> None:
        table = self._tabs.currentWidget()
        table.blockSignals(True)
        table.insertRow(table.rowCount())
        table.blockSignals(False)
        table.scrollToBottom()

    def _remove_rows(self) -> None:
        table = self._tabs.currentWidget()
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        if table is self._logs:
            keys = {(table.item(r, 0).text(), table.item(r, 1).text()) for r in rows}
            self._store.data.logs = [g for g in self._store.data.logs
                                     if (g.well_id, g.name) not in keys]
            self._store.notify()
            return
        table.blockSignals(True)
        for r in rows:
            table.removeRow(r)
        table.blockSignals(False)
        self._on_table_edited(QTableWidgetItem())

    # -- files ---------------------------------------------------------------
    def load_file(self, kind: str, path: str) -> int:
        """Read ``path`` as ``kind`` ('wells', 'lithology', 'water', 'logs'); returns rows read."""
        data = self._store.data
        if kind == "wells":
            rows = bh.read_wells(path)
            data.wells.update({w.well_id: w for w in rows})
        elif kind == "lithology":
            rows = bh.read_lithology(path)
            ids = {r.well_id for r in rows}
            data.lithology = [i for i in data.lithology if i.well_id not in ids] + rows
        elif kind == "logs":
            rows = bh.read_geophysical_logs(path)
            keys = {(g.well_id, g.name) for g in rows}
            data.logs = [g for g in data.logs if (g.well_id, g.name) not in keys] + rows
        else:
            rows = bh.read_water_levels(path)
            data.water_levels = data.water_levels + rows
        data.complete_legend()
        self._store.sources[kind] = Path(path).name
        self._store.notify()
        return len(rows)

    def _load(self, kind: str) -> None:
        names = {"wells": "wells", "lithology": "lithology logs", "water": "water levels",
                 "logs": "geophysical logs"}
        path, _ = QFileDialog.getOpenFileName(self, f"Load {names[kind]}", "",
                                              _LOG_FILTER if kind == "logs" else _TABLE_FILTER)
        if not path:
            return
        try:
            n = self.load_file(kind, path)
        except Exception as exc:  # noqa: BLE001 - a bad table is reported, not raised
            self._log(f"Could not read {names[kind]} from {Path(path).name}: {exc}", "error")
            return
        what = "curves" if kind == "logs" else "rows"
        self._log(f"Loaded {n} {names[kind]} {what} from {Path(path).name}.", "success")

    def _write_templates(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Write borehole templates to")
        if not folder:
            return
        paths = bh.write_templates(folder)
        self._log(f"Wrote {', '.join(p.name for p in paths)} to {folder}.", "success")

    def _clear(self) -> None:
        self._store.data = bh.BoreholeData()
        self._store.sources = {}
        self._store.notify()
