"""Survey-wide views of an EM dataset: where the stations are, and what survives.

Two widgets, both fed by
:func:`PyHydroGeophysX.data_processing.em1d.survey_summary`, which reads a
project in one pass.

:class:`EMSurveyView` answers "what does this quality-control setting cost me"
before a run rather than after one: a map of the survey coloured by whatever
you are worried about, and the same stations as a table you can sort. Clicking
a station loads it into the sounding plot.

:class:`EMMetadataView` answers "what will actually be modelled": the geometry,
waveform, gate windows and electronics the reader found, and the gate selection
it applied.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: Columns of the station table, as (heading, row key, format).
_COLUMNS = (
    ("Station", "station", None),
    ("Line", "line", "%d"),
    ("Easting", "x", "%.1f"),
    ("Northing", "y", "%.1f"),
    ("Elev (m)", "elevation", "%.1f"),
    ("Tx-Rx (m)", "rx_tx_distance", "%.2f"),
    ("LM kept", "LM_gates_kept", "%d"),
    ("LM held", "LM_gates_held", "%d"),
    ("LM std", "LM_median_std", "%.3f"),
    ("HM kept", "HM_gates_kept", "%d"),
    ("HM held", "HM_gates_held", "%d"),
    ("HM std", "HM_median_std", "%.3f"),
    ("Gates", "gates_kept", "%d"),
)

#: What the map can colour stations by, as (label, row key).
_COLOUR_FIELDS = (
    ("Gates kept", "gates_kept"),
    ("Line", "line"),
    ("Elevation", "elevation"),
    ("Tx-Rx distance", "rx_tx_distance"),
    ("LM gates kept", "LM_gates_kept"),
    ("HM gates kept", "HM_gates_kept"),
)


#: Rows whose bare value invites the wrong reading. The tree shows a field name
#: and a number with none of the surrounding argument, so the few fields where
#: that is not enough carry the argument as a tooltip.
_FIELD_NOTES = {
    "data_factor_applied": (
        "False because the factor is already present in the stored voltages, "
        "not because it was overlooked. The values a project records are the "
        "ones its own inversion consumed, factor included, so applying it here "
        "would count it twice."),
    "response_sign": (
        "The sign convention relating the recorded voltage to dB/dt. An "
        "offset-loop configuration reverses sign at early time on its own; that "
        "is a property of the measurement, not of this setting."),
    "data_scale": (
        "A single multiplier on the modelled response, either 1 or a value "
        "calibration estimated. It absorbs a constant gain, so it cannot hide a "
        "shape error."),
    "tx_rx_sep": (
        "The transmitter-to-receiver distance for this station. "
        "'tx_rx_sep_nominal' is what the instrument specification states; the "
        "two differ because the frame flexes."),
    "gate_window_shape": (
        "1 is a tapered (Tukey) window and 2 is a square one. The response is "
        "integrated across the window rather than sampled at the gate centre."),
    "gate_window_par": (
        "The tapered fraction of a Tukey window, so 0.667 leaves a third of the "
        "width flat on top."),
    "auto_scale": (
        "Whether a scale factor was estimated from the data rather than fixed "
        "at 1."),
    "source_moment": "Transmitter current times loop area times turns.",
    "rx_coil_area": "Effective receiver coil area, turns included, in square metres.",
}


def _finite(values: List[Any]) -> np.ndarray:
    array = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    return array


class _SortingItem(QTableWidgetItem):
    """A cell that sorts on its value rather than on its formatted text."""

    def __init__(self, text: str, value: Any) -> None:
        super().__init__(text)
        self.setFlags(self.flags() & ~Qt.ItemIsEditable)
        self._value = value

    def __lt__(self, other: "QTableWidgetItem") -> bool:
        mine, theirs = self._value, getattr(other, "_value", None)
        try:
            return float(mine) < float(theirs)
        except (TypeError, ValueError):
            return str(mine) < str(theirs)


class EMSurveyView(QWidget):
    """Map and table of every station, and what the current QC keeps of it."""

    #: Emitted with the 0-based sounding index of a station the user picked.
    stationPicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: List[Dict[str, Any]] = []
        #: Set while the table is being rebuilt, so that clearing and
        #: refilling it does not read back as the user picking a station.
        self._filling = False

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(6, 4, 6, 2)
        row.addWidget(QLabel("Colour by:"))
        self._colour_by = QComboBox()
        for label, key in _COLOUR_FIELDS:
            self._colour_by.addItem(label, key)
        self._colour_by.currentIndexChanged.connect(lambda _i: self._draw_map())
        row.addWidget(self._colour_by)
        row.addSpacing(12)
        self._totals = QLabel("")
        self._totals.setToolTip(
            "What the gate selection above the plot keeps, over the whole "
            "survey. A station whose every gate is dropped disappears from the "
            "inversion entirely, so it is counted separately.")
        row.addWidget(self._totals)
        row.addStretch(1)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Easting", units="m")
        self._plot.setLabel("left", "Northing", units="m")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setAspectLocked(True)
        self._scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#333333", width=0.4))
        self._scatter.sigClicked.connect(self._on_points_clicked)
        self._plot.addItem(self._scatter)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_row_selected)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self._plot)
        split.addWidget(self._table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(controls)
        layout.addWidget(split, stretch=1)

    # -- population ----------------------------------------------------------
    def set_summary(self, summary: Optional[Dict[str, Any]]) -> None:
        """Show one survey, or clear when given nothing."""
        self._rows = list((summary or {}).get("rows", []))
        totals = (summary or {}).get("totals", {})
        if totals:
            emptied = int(totals.get("stations_emptied", 0))
            self._totals.setText(
                "%d stations, %d with data%s   |   %d of %d gates kept" % (
                    totals.get("stations", 0),
                    totals.get("stations_with_data", 0),
                    (" (%d emptied)" % emptied) if emptied else "",
                    totals.get("gates_kept", 0), totals.get("gates_held", 0)))
        else:
            self._totals.setText("")
        self._fill_table()
        self._draw_map()

    def _fill_table(self) -> None:
        self._filling = True
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._rows))
        for index, row in enumerate(self._rows):
            for column, (_, key, fmt) in enumerate(_COLUMNS):
                value = row.get(key)
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    text = ""
                elif fmt is None:
                    text = str(value)
                else:
                    text = fmt % value
                item = _SortingItem(text, value)
                # A station the selection emptied cannot be inverted, so it is
                # greyed rather than left looking like the rest.
                if not row.get("gates_kept"):
                    item.setForeground(QColor("#999999"))
                self._table.setItem(index, column, item)
        # Enabling sorting applies whatever indicator the header carries, which
        # would reorder a freshly filled table before it is ever shown. Clearing
        # the indicator first leaves it in file order, which is the order the
        # survey was walked in and the one the sounding numbers follow.
        self._table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self._table.setSortingEnabled(True)
        self._filling = False

    def _draw_map(self) -> None:
        if not self._rows:
            self._scatter.setData([])
            return
        x = _finite([r.get("x") for r in self._rows])
        y = _finite([r.get("y") for r in self._rows])
        key = str(self._colour_by.currentData())
        values = _finite([r.get(key) for r in self._rows])
        good = np.isfinite(x) & np.isfinite(y)
        if not good.any():
            self._scatter.setData([])
            return
        finite = values[np.isfinite(values)]
        low = float(finite.min()) if finite.size else 0.0
        high = float(finite.max()) if finite.size else 1.0
        span = (high - low) or 1.0
        colormap = pg.colormap.get("viridis")
        spots = []
        for index in np.flatnonzero(good):
            value = values[index]
            colour = (QColor("#bbbbbb") if not np.isfinite(value)
                      else colormap.map(float((value - low) / span), mode="qcolor"))
            spots.append({
                "pos": (x[index], y[index]), "brush": pg.mkBrush(colour),
                "data": int(index), "size": 8,
            })
        self._scatter.setData(spots)
        self._plot.setTitle("%s: %g to %g" % (
            self._colour_by.currentText(), low, high))

    # -- picking -------------------------------------------------------------
    def _sounding_index(self, row_index: int) -> Optional[int]:
        """Position among the stations an inversion would see.

        A station the selection emptied is absent from the loader's list, so the
        table's own row number is not the sounding number once any station has
        been dropped.
        """
        if not (0 <= row_index < len(self._rows)):
            return None
        if not self._rows[row_index].get("gates_kept"):
            return None
        return sum(1 for row in self._rows[:row_index] if row.get("gates_kept"))

    def _on_points_clicked(self, _scatter, points) -> None:
        if not len(points):
            return
        index = self._sounding_index(int(points[0].data()))
        if index is not None:
            self.stationPicked.emit(index)

    def _on_row_selected(self) -> None:
        if self._filling:
            return
        items = self._table.selectedItems()
        if not items:
            return
        # The table sorts, so the visual row is not the data row; the station
        # column carries the identity, so find it back through that.
        station = self._table.item(items[0].row(), 0).text()
        for index, row in enumerate(self._rows):
            if str(row.get("station")) == station:
                picked = self._sounding_index(index)
                if picked is not None:
                    self.stationPicked.emit(picked)
                return


class EMMetadataView(QTreeWidget):
    """What the reader found, and what the forward will therefore model."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Field", "Value"])
        self.setAlternatingRowColors(True)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)

    def set_metadata(self, data: Optional[Dict[str, Any]]) -> None:
        """Show one sounding's acquisition description, or clear."""
        self.clear()
        if not data:
            return
        for title, payload in (
            ("Instrument and geometry", data.get("forward_metadata") or {}),
            ("System (as modelled)", data.get("system") or {}),
            ("Acquisition protocol", data.get("protocol") or {}),
            ("Inversion settings in the file", data.get("inversion_defaults") or {}),
        ):
            if not payload:
                continue
            parent = QTreeWidgetItem([title, ""])
            self.addTopLevelItem(parent)
            self._add_mapping(parent, payload)
            parent.setExpanded(True)

    def _add_mapping(self, parent: QTreeWidgetItem, mapping: Dict[str, Any]) -> None:
        for key in sorted(mapping):
            value = mapping[key]
            if isinstance(value, dict):
                child = QTreeWidgetItem([str(key), ""])
                parent.addChild(child)
                self._add_mapping(child, value)
            else:
                child = QTreeWidgetItem([str(key), _render(value)])
                note = _FIELD_NOTES.get(str(key))
                if note:
                    for column in (0, 1):
                        child.setToolTip(column, note)
                parent.addChild(child)


def _render(value: Any) -> str:
    """A cell's worth of text for a value of unknown shape.

    Long arrays are summarised rather than printed: a waveform has dozens of
    nodes and a gate list has hundreds, and a cell showing the first few with a
    count is more use than one showing all of them unreadably.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=object).ravel()
        if array.size == 0:
            return "(empty)"
        shown = ", ".join(_render(item) for item in array[:4])
        return shown if array.size <= 4 else "%s, ... (%d values)" % (shown, array.size)
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        if value and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return "%.6g" % value
        return ("%.6f" % value).rstrip("0").rstrip(".")
    return str(value)


class EMSignalNoiseView(QWidget):
    """Measured signal against absolute noise, along one survey line.

    A station returns fewer usable gates for two reasons that call for opposite
    readings, and the relative error the file records is the one divided by the
    other, so it rises either way and cannot separate them. Drawn apart they can
    be read directly.

    A signal that falls while the noise floor holds is evidence about the
    ground: a resistive half-space returns dB/dt going as ``rho ** -1.5``, and
    its currents also diffuse away sooner, so resistive ground is genuinely
    quieter and its late gates genuinely go under the floor. A noise floor that
    rises under a steady signal is an instrument or an environment, and reading
    it as resistive ground would invent structure out of a data problem.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._summary: Optional[Dict[str, Any]] = None

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(6, 4, 6, 2)
        row.addWidget(QLabel("Line:"))
        self._line = QComboBox()
        self._line.setToolTip(
            "Which survey line to draw. Distance runs from that line's own "
            "first station.")
        self._line.currentIndexChanged.connect(lambda _i: self._redraw())
        row.addWidget(self._line)
        row.addSpacing(12)
        row.addWidget(QLabel("Smoothing:"))
        self._smooth = QSpinBox()
        self._smooth.setRange(1, 201)
        self._smooth.setValue(21)
        self._smooth.setSuffix(" stations")
        self._smooth.setToolTip(
            "Width of a running mean over stations. Cosmetic: the "
            "station-to-station scatter obscures the trend this is drawn to "
            "show. Set it to 1 to plot the values themselves.")
        self._smooth.valueChanged.connect(lambda _v: self._redraw())
        row.addWidget(self._smooth)
        row.addSpacing(12)
        self._caption = QLabel("")
        row.addWidget(self._caption)
        row.addStretch(1)

        self._plots: Dict[str, pg.PlotWidget] = {}
        stack = QSplitter(Qt.Vertical)
        for name in ("LM", "HM"):
            plot = pg.PlotWidget()
            plot.setLogMode(x=False, y=True)
            # In log mode the tick values are already exponents, and pyqtgraph
            # will still factor an SI prefix out of them on top, which reads as
            # "4 x 10^1 (x1e-09)" for four times ten to the minus eight.
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", "%s (V)" % name)
            plot.addLegend(offset=(-10, 10), labelTextSize="8pt")
            self._plots[name] = plot
            stack.addWidget(plot)
        self._plots["HM"].setLabel("bottom", "Distance along line (m)")
        self._plots["LM"].setXLink(self._plots["HM"])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(controls)
        layout.addWidget(stack, stretch=1)

    def set_summary(self, summary: Optional[Dict[str, Any]]) -> None:
        """Show one survey, keeping the chosen line where the new one has it."""
        self._summary = summary
        previous = self._line.currentData()
        numbers = sorted({int(r["line"])
                          for r in (summary or {}).get("rows", [])})
        self._line.blockSignals(True)
        self._line.clear()
        for value in numbers:
            self._line.addItem("Line %d" % value, value)
        index = self._line.findData(previous)
        self._line.setCurrentIndex(max(0, index))
        self._line.blockSignals(False)
        self._redraw()

    def _redraw(self) -> None:
        for plot in self._plots.values():
            plot.clear()
        rows = [r for r in (self._summary or {}).get("rows", [])
                if int(r["line"]) == self._line.currentData()]
        if not rows:
            self._caption.setText("")
            return
        distance = _along_line(rows)
        width = int(self._smooth.value())
        notes = []
        for name, plot in self._plots.items():
            signal = _finite([r.get("%s_signal" % name) for r in rows])
            noise = _finite([r.get("%s_noise" % name) for r in rows])
            at = _finite([r.get("%s_reference_time" % name) for r in rows])
            plot.setLabel("left", "%s%s (V)" % (
                name, "" if not np.isfinite(np.nanmedian(at))
                else " at %.1f us" % (np.nanmedian(at) * 1e6)))
            if not np.isfinite(signal).any():
                continue
            plot.addItem(pg.PlotDataItem(
                x=distance, y=_running_mean(signal, width),
                pen=pg.mkPen("#1f77b4", width=2), name="|signal|"))
            plot.addItem(pg.PlotDataItem(
                x=distance, y=_running_mean(noise, width),
                pen=pg.mkPen("#c62828", width=2), name="absolute noise"))
            ratio = np.nanmedian(signal) / np.nanmedian(noise)
            notes.append("%s median SNR %.1f" % (name, ratio))
        self._caption.setText("%d stations   |   %s" % (
            len(rows), ",   ".join(notes)))


def _along_line(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Cumulative distance from a line's own first station.

    Falls back to position in the file where the map coordinates are missing,
    so a survey without them still plots against something monotonic.
    """
    x = _finite([r.get("x") for r in rows])
    y = _finite([r.get("y") for r in rows])
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.arange(len(rows), dtype=float)
    steps = np.hypot(np.diff(x), np.diff(y))
    steps[~np.isfinite(steps)] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def _running_mean(values: np.ndarray, width: int) -> np.ndarray:
    """A centred running mean that steps over gaps rather than spreading them.

    A NaN inside a plain convolution takes its whole window with it, which on a
    survey holding a few dummy stations erases a stretch of the trend around
    each one.
    """
    values = np.asarray(values, dtype=float).ravel()
    width = max(1, int(width))
    if width == 1 or values.size < 2:
        return values
    good = np.isfinite(values)
    window = np.ones(min(width, values.size))
    total = np.convolve(np.where(good, values, 0.0), window, mode="same")
    count = np.convolve(good.astype(float), window, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)
