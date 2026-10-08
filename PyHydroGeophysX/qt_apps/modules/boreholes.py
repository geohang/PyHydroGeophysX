"""Boreholes page: the wells, their logs and water levels, in one place.

The studio's borehole data are loaded and edited here - well locations,
lithology logs, water levels and borehole geophysical logs - and looked at in
three views: the wells on a map, the logs side by side, and the water levels
through time. Added to the Project Map, they are compared there with every
survey around them; the method pages do not draw them.

Reading and drawing are the package's
(:mod:`PyHydroGeophysX.data_processing.boreholes`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import boreholes as bh
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import ContentWidthScrollArea
from PyHydroGeophysX.qt_apps.widgets import borehole_panel, length_units
from PyHydroGeophysX.qt_apps.widgets.readout import toolbar_row

MAP, LOGS, WATER = "Map", "Logs", "Water levels"


class _FigurePane(QWidget):
    """A matplotlib figure with its toolbar: one of the page's views.

    ``_fig`` is the name the assistant's capture looks for, so a captured view
    is re-rendered at print resolution rather than grabbed off the screen.
    """

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # The cursor position goes to a readout of its own: the toolbar's label
        # would re-lay out the page on every mouse move (see widgets.readout).
        bar, self.toolbar = toolbar_row(self.canvas, self)
        layout.addWidget(bar)
        layout.addWidget(self.canvas, stretch=1)
        self._placeholder = placeholder

    @property
    def figure(self):
        return self._fig

    def drawn(self) -> None:
        """Show the figure just drawn from scratch; Home and Back go to it."""
        self.canvas.draw_idle()
        self.toolbar.update()

    def empty(self, text: Optional[str] = None) -> None:
        self._fig.clear()
        self._fig.text(0.5, 0.5, text or self._placeholder, ha="center", va="center",
                       fontsize=11, color="#8e8e93", wrap=True)
        self.drawn()


class BoreholesModule(BaseModule):
    module_key = "boreholes"
    module_title = "Boreholes"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._store = borehole_panel.store_for(state)
        self._selected: List[str] = []
        self._with_data: set = set()          # wells that had data when last looked at
        self._list_syncing = False

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._map = _FigurePane("Load the wells' locations (Wells… on the right) to see them here.")
        self._logs = _FigurePane("Choose wells with a lithology or geophysical log on the right.")
        self._water = _FigurePane("Load water levels (Water levels… on the right) to see them "
                                  "through time.")
        self._tabs.addTab(self._map, MAP)
        self._tabs.addTab(self._logs, LOGS)
        self._tabs.addTab(self._water, WATER)
        self._tabs.currentChanged.connect(lambda *_: self._redraw())
        root.addWidget(self._tabs, stretch=1)
        self._map.canvas.mpl_connect("button_press_event", self._on_map_click)

        column = QWidget()
        column_layout = QVBoxLayout(column)
        intro = QLabel("Load the wells, logs and water levels here. Add them to the Project "
                       "Map to compare each well with the surveys around it.")
        intro.setWordWrap(True)
        theme.set_tone(intro, "hint")
        column_layout.addWidget(intro)
        column_layout.addWidget(borehole_panel.BoreholeDataPanel(self._store, log=self.log))
        column_layout.addWidget(self._build_show_group())
        column_layout.addWidget(self._build_map_group())
        column_layout.addStretch(1)
        side = ContentWidthScrollArea(minimum=450, maximum=520)
        side.setWidget(column)
        root.addWidget(side)

        self._store.changed.connect(self._on_store_changed)
        length_units.notifier().changed.connect(lambda *_: self._redraw())
        self._on_store_changed()

    # -- the right column ------------------------------------------------------
    def _build_show_group(self) -> QGroupBox:
        box = QGroupBox("Show")
        layout = QVBoxLayout(box)
        hint = QLabel("Ticked wells are drawn on the Logs and Water levels views and "
                      "highlighted on the map; click a well on the map to tick it.")
        hint.setWordWrap(True)
        theme.set_tone(hint, "hint")
        layout.addWidget(hint)
        self._well_list = QListWidget()
        self._well_list.setMinimumHeight(140)
        self._well_list.itemChanged.connect(self._on_list_changed)
        layout.addWidget(self._well_list)
        buttons = QHBoxLayout()
        for label, rule in (("With data", "data"), ("All", "all"), ("None", "none")):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, r=rule: self._select(r))
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        form = QFormLayout()
        self._water_mode = QComboBox()
        self._water_mode.addItem("Range of all readings", "range")
        self._water_mode.addItem("Nearest to a date", "date")
        self._water_mode.currentIndexChanged.connect(self._on_water_mode)
        form.addRow("Water level on the logs", self._water_mode)
        self._date = QDateEdit()
        self._date.setCalendarPopup(True)
        self._date.setDisplayFormat("yyyy-MM-dd")
        self._date.setDate(QDate.currentDate())
        self._date.setEnabled(False)
        self._date.dateChanged.connect(lambda *_: self._redraw())
        form.addRow("Date", self._date)
        layout.addLayout(form)
        return box

    def _build_map_group(self) -> QGroupBox:
        box = QGroupBox("Project Map")
        layout = QVBoxLayout(box)
        hint = QLabel("Put the wells on the Project Map with their logs and water levels. "
                      "There, clicking a well sets its log beside every survey around it.")
        hint.setWordWrap(True)
        theme.set_tone(hint, "hint")
        layout.addWidget(hint)
        button = self.map_export_button()
        button.setText("Add wells to the Map…")
        button.setToolTip("Save the wells, their logs and water levels as a layer of the "
                          "Project Map. Adding them again after an edit adds the new version.")
        layout.addWidget(button)
        return box

    def map_snapshot(self):
        """The wells as a Project Map layer (``widgets/map_export.py`` calls this)."""
        from PyHydroGeophysX.qt_apps.project_map import wells_snapshot

        return wells_snapshot(self._store.data)

    def _on_water_mode(self, *_args: Any) -> None:
        self._date.setEnabled(self._water_mode.currentData() == "date")
        self._redraw()

    def _when(self):
        if self._water_mode.currentData() != "date":
            return None
        import datetime as dt

        d = self._date.date()
        return dt.datetime(d.year(), d.month(), d.day())

    # -- selection -------------------------------------------------------------
    def _well_ids(self) -> List[str]:
        data = self._store.data
        ids = set(data.wells) | {i.well_id for i in data.lithology} \
            | {w.well_id for w in data.water_levels} | {g.well_id for g in data.logs}
        return sorted(ids)

    def _has_data(self, well_id: str) -> bool:
        data = self._store.data
        return bool(data.intervals(well_id) or data.levels(well_id) or data.well_logs(well_id))

    def _select(self, rule: str) -> None:
        ids = self._well_ids()
        if rule == "all":
            self._selected = ids
        elif rule == "none":
            self._selected = []
        else:
            self._selected = [w for w in ids if self._has_data(w)]
        self._fill_list()
        self._redraw()

    def _fill_list(self) -> None:
        self._list_syncing = True
        try:
            self._well_list.clear()
            for well_id in self._well_ids():
                parts = [name for name, ok in (
                    ("location", well_id in self._store.data.wells),
                    ("lithology", bool(self._store.data.intervals(well_id))),
                    ("water levels", bool(self._store.data.levels(well_id))),
                    ("geophysics", bool(self._store.data.well_logs(well_id)))) if ok]
                item = QListWidgetItem(f"{well_id}  ·  {', '.join(parts)}")
                item.setData(Qt.UserRole, well_id)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if well_id in self._selected else Qt.Unchecked)
                self._well_list.addItem(item)
        finally:
            self._list_syncing = False

    def _on_list_changed(self, _item: QListWidgetItem) -> None:
        if self._list_syncing:
            return
        self._selected = [self._well_list.item(i).data(Qt.UserRole)
                          for i in range(self._well_list.count())
                          if self._well_list.item(i).checkState() == Qt.Checked]
        self._redraw()

    def _on_store_changed(self) -> None:
        ids = self._well_ids()
        with_data = {w for w in ids if self._has_data(w)}
        # A well that has just gained a log or water levels is ticked; what the
        # user ticked or unticked before stays as it was.
        gained = sorted(with_data - self._with_data)
        self._selected = [w for w in self._selected if w in ids] + \
            [w for w in gained if w not in self._selected]
        self._with_data = with_data
        self._fill_list()
        self._redraw()

    def _on_map_click(self, event: Any) -> None:
        if self._map.toolbar.mode:      # the click pans or zooms, it does not pick
            return
        if event.inaxes is None or event.xdata is None:
            return
        wells = [w for w in self._store.data.wells.values()
                 if np.isfinite(w.easting) and np.isfinite(w.northing)]
        if not wells:
            return
        ax = event.inaxes
        pts = ax.transData.transform([(w.easting, w.northing) for w in wells])
        dist = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
        k = int(np.argmin(dist))
        if dist[k] > 12:
            return
        well_id = wells[k].well_id
        if well_id in self._selected:
            self._selected.remove(well_id)
        else:
            self._selected.append(well_id)
        self._fill_list()
        self._redraw()

    # -- drawing ---------------------------------------------------------------
    def _redraw(self) -> None:
        view = self._tabs.tabText(self._tabs.currentIndex())
        data = self._store.data
        unit = length_units.current()
        ink = theme.PALETTE["canvas_text"]
        if view == MAP:
            pane = self._map
            if not data.wells:
                pane.empty()
                return
            pane.figure.clear()
            ax = pane.figure.add_subplot(111)
            bh.draw_well_map(ax, data, selected=self._selected, length_unit=unit,
                             color=theme.DATA_COLOR, ink=ink)
            crs = sorted({w.crs for w in data.wells.values() if w.crs})
            ax.set_title("Wells" + (f" ({', '.join(crs)})" if crs else "")
                         + " - filled: ticked on the right", fontsize=10)
            pane.figure.tight_layout()
            pane.drawn()
        elif view == LOGS:
            pane = self._logs
            wells = [w for w in self._selected if data.intervals(w) or data.well_logs(w)
                     or data.levels(w)]
            if not wells:
                pane.empty()
                return
            bh.draw_well_logs(pane.figure, data, wells, when=self._when(), length_unit=unit,
                              water_color=theme.DATA_COLOR, edge_color=ink)
            pane.figure.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12)
            pane.drawn()
        else:
            pane = self._water
            wells = [w for w in self._selected if data.levels(w)]
            if not wells:
                pane.empty("None of the ticked wells has water levels." if data.water_levels
                           else None)
                return
            pane.figure.clear()
            ax = pane.figure.add_subplot(111)
            bh.draw_water_levels(ax, data, wells, length_unit=unit, when=self._when())
            ax.set_title("Depth to water", fontsize=10)
            pane.figure.autofmt_xdate()
            pane.figure.tight_layout()
            pane.drawn()

    # -- assistant -------------------------------------------------------------
    def agent_views(self) -> Dict[str, QWidget]:
        return {"map": self._map, "logs": self._logs, "water_levels": self._water}

    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "load_wells", "args": {"path": "str"},
                 "desc": "Load well locations (CSV/XLSX: well_id, easting, northing, "
                         "ground_elevation, crs, datum_offset, source)."},
                {"name": "load_lithology", "args": {"path": "str"},
                 "desc": "Load lithology intervals (well_id, top, bottom, unit, description, "
                         "length_unit)."},
                {"name": "load_water_levels", "args": {"path": "str"},
                 "desc": "Load water levels (well_id, datetime, depth_below_ground or "
                         "head_elevation)."},
                {"name": "load_logs", "args": {"path": "str"},
                 "desc": "Load borehole geophysical logs: a LAS 2.0 file, or a table of "
                         "well_id, depth and one column per curve."},
                {"name": "select_wells", "args": {"wells": "list[str] | 'all' | 'with_data'"},
                 "desc": "Choose the wells drawn on the Logs and Water levels views."},
                {"name": "show", "args": {"view": "map | logs | water_levels"},
                 "desc": "Bring a view to the front (then capture_view it)."},
                {"name": "get_status", "args": {}, "desc": "What is loaded and shown."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        kinds = {"load_wells": "wells", "load_lithology": "lithology",
                 "load_water_levels": "water", "load_logs": "logs"}
        if action in kinds:
            return self._agent_load(kinds[action], args.get("path"))
        if action == "select_wells":
            wells = args.get("wells", "with_data")
            if wells in ("all", "with_data"):
                self._select("all" if wells == "all" else "data")
            else:
                unknown = sorted(set(map(str, wells)) - set(self._well_ids()))
                if unknown:
                    return {"status": "failed", "error": f"Unknown wells: {', '.join(unknown)}.",
                            "wells": self._well_ids()}
                self._selected = [str(w) for w in wells]
                self._fill_list()
                self._redraw()
            return {"status": "ok", "selected": list(self._selected)}
        if action == "show":
            names = {"map": MAP, "logs": LOGS, "water_levels": WATER}
            view = str(args.get("view", "")).strip().lower()
            if view not in names:
                return {"status": "failed", "error": f"view is one of {', '.join(names)}."}
            self._tabs.setCurrentIndex([MAP, LOGS, WATER].index(names[view]))
            return {"status": "ok", "view": view}
        if action == "get_status":
            return self._agent_status()
        return {"status": "failed", "error": f"Unknown action '{action}'.",
                "valid_actions": [a["name"] for a in self.agent_describe()["actions"]]}

    def _agent_load(self, kind: str, path: Any) -> Dict[str, Any]:
        if not path or not Path(str(path)).is_file():
            return {"status": "failed", "error": f"File not found: {path}"}
        panel = self.findChild(borehole_panel.BoreholeDataPanel)
        try:
            n = panel.load_file(kind, str(path))
        except Exception as exc:  # noqa: BLE001 - a bad table is reported, not raised
            return {"status": "failed", "error": str(exc)}
        return {"status": "ok", "read": n, **self._agent_status()}

    def _agent_status(self) -> Dict[str, Any]:
        data = self._store.data
        return {
            "status": "ok",
            "loaded": self._store.describe(),
            "wells": self._well_ids(),
            "selected": list(self._selected),
            "view": self._tabs.tabText(self._tabs.currentIndex()),
            "geophysical_logs": sorted({f"{g.well_id}: {g.label}" for g in data.logs}),
        }
