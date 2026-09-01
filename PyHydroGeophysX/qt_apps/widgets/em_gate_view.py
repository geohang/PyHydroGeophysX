"""One sounding, gate by gate: what the file holds and what the run will see.

The sounding plot elsewhere draws the gates that survive the current selection,
which is what an inversion consumes and less than a reader needs. This draws all
of them, each marked with the test that removed it, so "why did this station
come back with four gates" is answered by looking rather than by changing a
setting and watching a count move. Fed by
:func:`PyHydroGeophysX.data_processing.em1d.gate_report`.

Values are drawn as magnitude, so a gate whose voltage is negative is ringed
rather than silently dropped from the picture.

Both axes hold log10 and are labelled in seconds and volts by
:mod:`.log_scale_axis`. pyqtgraph's own ``setLogMode`` re-maps a
``PlotDataItem`` and leaves a bare ``ScatterPlotItem`` or ``ErrorBarItem``
alone, and this plot draws all three, so the transform is applied once at the
source instead of reaching some items and not others.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .log_scale_axis import label_axis_in_physical_units

#: How each verdict is drawn, as verdict -> (colour, symbol, filled). Kept gates
#: are the only filled ones, so the picture reads before the legend does.
_VERDICT_STYLE = {
    "kept": ("#1f77b4", "o", True),
    "flagged out": ("#9e9e9e", "o", False),
    "noisy": ("#ef6c00", "s", False),
    "after a noisy one": ("#ffb74d", "s", False),
    "reversed sign": ("#c62828", "t", False),
    "dummy": ("#555555", "x", False),
}

#: Per-moment line colour for the modelled response.
_MODEL_COLOUR = {"LM": "#2e7d32", "HM": "#6a1b9a"}

#: A gate the loader kept and the inversion then threw out. Drawn over the kept
#: marker rather than replacing it, because both facts are true of that gate and
#: the second one only exists once a fit has run.
_REJECTED_PEN = "#d81b60"

_COLUMNS = ("Moment", "Gate", "Centre (us)", "Open (us)", "Close (us)",
            "Value (V)", "Rel. std", "Flag", "Verdict")


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


class EMGateView(QWidget):
    """Every gate of one sounding, with the verdict the selection gave it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._report: Optional[Dict[str, Any]] = None
        self._model: Optional[Dict[str, Dict[str, Any]]] = None

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(6, 4, 6, 2)
        self._show_dropped = QCheckBox("Show dropped gates")
        self._show_dropped.setChecked(True)
        self._show_dropped.setToolTip(
            "Draw the gates the current selection removes, each marked with the "
            "test that removed it. Turning this off leaves the gates the "
            "inversion will actually see.")
        self._show_dropped.toggled.connect(lambda _b: self._redraw())
        row.addWidget(self._show_dropped)
        self._show_windows = QCheckBox("Gate windows")
        self._show_windows.setToolTip(
            "Draw each gate as a bar from its open time to its close time. The "
            "response is integrated across that width, so a late gate averages "
            "over a span comparable to its own centre time.")
        self._show_windows.toggled.connect(lambda _b: self._redraw())
        row.addWidget(self._show_windows)
        row.addSpacing(12)
        self._caption = QLabel("")
        row.addWidget(self._caption)
        row.addStretch(1)
        for label, slot in (("Export CSV", self._export_csv_dialog),
                            ("Export PNG", self._export_png_dialog)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setLabel("left", "|dB/dt| (V)")
        for side in ("bottom", "left"):
            label_axis_in_physical_units(self._plot.getAxis(side))
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._plot.addLegend(offset=(-10, 10), labelTextSize="8pt")

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)

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
    def set_report(self, report: Optional[Dict[str, Any]]) -> None:
        """Show one station's gates, or clear when given nothing."""
        self._report = report
        self._fill_table()
        self._redraw()

    def set_model(self, moments: Optional[Dict[str, Dict[str, Any]]]) -> None:
        """Overlay a modelled response, keyed by moment name.

        Each entry needs ``times`` and ``pred``, and may carry ``fit_mask``:
        which of the gates the loader kept the fit actually used. Passing
        ``None`` clears the overlay, which is what loading a station should do:
        a response computed for the previous one says nothing about this one.
        """
        self._model = moments
        self._redraw()

    # -- drawing -------------------------------------------------------------
    def _redraw(self) -> None:
        self._plot.clear()
        self._legend.clear()
        if not self._report:
            self._caption.setText("")
            return
        show_dropped = self._show_dropped.isChecked()
        # The legend would otherwise carry one entry per moment per verdict,
        # which is twelve rows saying six things.
        labelled: Set[str] = set()
        kept_total = held_total = rejected_total = 0
        for name, moment in (self._report.get("moments") or {}).items():
            status = np.asarray(moment["status"], dtype=object)
            values = np.asarray(moment["values"], dtype=float)
            times = np.asarray(moment["times"], dtype=float)
            kept_total += int(moment.get("kept", 0))
            held_total += int(moment.get("held", 0))
            magnitude = np.abs(values)
            drawable = (np.isfinite(times) & (times > 0.0)
                        & np.isfinite(values) & (magnitude > 0.0))
            # Substituting 1.0 keeps log10 quiet on the entries the mask will
            # discard anyway; nothing reads those positions.
            log_t = np.log10(np.where(drawable, times, 1.0))
            log_v = np.log10(np.where(drawable, magnitude, 1.0))
            for verdict, (colour, symbol, filled) in _VERDICT_STYLE.items():
                if verdict != "kept" and not show_dropped:
                    continue
                pick = drawable & (status == verdict)
                if not pick.any():
                    continue
                self._plot.addItem(pg.ScatterPlotItem(
                    x=log_t[pick], y=log_v[pick], symbol=symbol,
                    size=9 if verdict == "kept" else 8,
                    pen=pg.mkPen(colour, width=1.4),
                    brush=pg.mkBrush(colour) if filled else None,
                    name=_label(verdict, labelled)))
            self._draw_errors(moment, log_t, log_v, drawable & (status == "kept"))
            rejected_here = self._draw_rejected(
                name, status, log_t, log_v, drawable, labelled)
            rejected_total += rejected_here
            if show_dropped:
                self._draw_negatives(log_t, log_v, values, drawable, labelled)
            if self._show_windows.isChecked():
                shown = drawable if show_dropped else drawable & (status == "kept")
                self._draw_windows(moment, log_v, shown)
            self._draw_model(name, labelled)
        caption = "Station %s (line %s): %d of %d gates kept, %d dropped." % (
            self._report.get("station", "?"), self._report.get("line", "?"),
            kept_total, held_total, held_total - kept_total)
        if rejected_total:
            caption += "  The fit rejected %d of the %d it was given." % (
                rejected_total, kept_total)
        self._caption.setText(caption)

    def _draw_rejected(self, name: str, status: np.ndarray, log_t: np.ndarray,
                       log_v: np.ndarray, drawable: np.ndarray,
                       labelled: Set[str]) -> int:
        """Ring the gates the inversion discarded as outliers.

        ``fit_mask`` runs over the gates the loader kept, in their own order, so
        the positions map back through the kept ones. A length that does not
        match means the overlay belongs to a different selection than the report
        does, and nothing is drawn rather than something misleading.
        """
        item = (self._model or {}).get(name) or {}
        mask = item.get("fit_mask")
        if mask is None:
            return 0
        mask = np.asarray(mask, dtype=bool).ravel()
        kept = np.flatnonzero(status == "kept")
        if mask.size != kept.size or mask.all():
            return 0
        pick = kept[~mask]
        pick = pick[drawable[pick]]
        if not pick.size:
            return int((~mask).sum())
        self._plot.addItem(pg.ScatterPlotItem(
            x=log_t[pick], y=log_v[pick], symbol="x", size=16,
            pen=pg.mkPen(_REJECTED_PEN, width=2.0), brush=None,
            name=_label("rejected by the fit", labelled)))
        return int((~mask).sum())

    def _draw_errors(self, moment: Dict[str, Any], log_t: np.ndarray,
                     log_v: np.ndarray, pick: np.ndarray) -> None:
        """Error bars on the kept gates, from the recorded stack scatter.

        The stored error is relative, so a bar is a fraction of the gate's own
        magnitude. That fraction is symmetric in the value and therefore
        asymmetric on a log axis, so both halves are computed rather than one
        being mirrored: a 60% error reaches 0.2 of a decade up and 0.4 down.
        """
        std = np.asarray(moment.get("relative_std"), dtype=float)
        if std.size != log_t.size:
            return
        pick = pick & np.isfinite(std) & (std > 0.0)
        if not pick.any():
            return
        # An error at or above 100% would put the lower bar at minus infinity.
        # Clipping draws it as a long bar instead, which is the honest picture.
        floor = np.clip(1.0 - std[pick], 1e-3, None)
        self._plot.addItem(pg.ErrorBarItem(
            x=log_t[pick], y=log_v[pick],
            top=np.log10(1.0 + std[pick]), bottom=-np.log10(floor),
            pen=pg.mkPen("#1f77b4", width=1.0), beam=0.01))

    def _draw_negatives(self, log_t: np.ndarray, log_v: np.ndarray,
                        values: np.ndarray, drawable: np.ndarray,
                        labelled: Set[str]) -> None:
        """Ring the gates whose voltage is negative.

        The axis shows magnitude, so without this a reversed gate sits exactly
        where a positive one of the same size would.
        """
        pick = drawable & (values < 0.0)
        if not pick.any():
            return
        self._plot.addItem(pg.ScatterPlotItem(
            x=log_t[pick], y=log_v[pick], symbol="o", size=15,
            pen=pg.mkPen("#c62828", width=1.2, style=Qt.DotLine), brush=None,
            name=_label("negative value", labelled)))

    def _draw_windows(self, moment: Dict[str, Any], log_v: np.ndarray,
                      pick: np.ndarray) -> None:
        """Draw each gate as a bar spanning its integration window."""
        open_t = np.asarray(moment.get("open"), dtype=float)
        close_t = np.asarray(moment.get("close"), dtype=float)
        if open_t.size != log_v.size or close_t.size != log_v.size:
            return
        pick = (pick & np.isfinite(open_t) & np.isfinite(close_t)
                & (open_t > 0.0) & (close_t > open_t))
        if not pick.any():
            return
        low = np.log10(open_t[pick])
        high = np.log10(close_t[pick])
        centre = 0.5 * (low + high)
        self._plot.addItem(pg.ErrorBarItem(
            x=centre, y=log_v[pick], left=centre - low, right=high - centre,
            pen=pg.mkPen("#607d8b", width=1.0), beam=0.0))

    def _draw_model(self, name: str, labelled: Set[str]) -> None:
        """Overlay the modelled response for one moment, when there is one."""
        item = (self._model or {}).get(name)
        if not item:
            return
        times = np.asarray(item.get("times"), dtype=float).ravel()
        pred = np.abs(np.asarray(item.get("pred"), dtype=float)).ravel()
        n = min(times.size, pred.size)
        times, pred = times[:n], pred[:n]
        good = np.isfinite(times) & (times > 0.0) & np.isfinite(pred) & (pred > 0.0)
        if not good.any():
            return
        self._plot.addItem(pg.PlotDataItem(
            x=np.log10(times[good]), y=np.log10(pred[good]),
            pen=pg.mkPen(_MODEL_COLOUR.get(name, "#37474f"), width=1.8),
            name=_label(f"{name} modelled", labelled)))

    # -- export --------------------------------------------------------------
    def export_csv(self, path: str) -> None:
        """Write every gate the station holds, verdict included.

        The whole table rather than the survivors. Which gates were dropped and
        by which test is the reason this view exists, and a file holding only
        what survived cannot be told apart from a station that recorded less.
        """
        import csv

        report = self._report or {}
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["station", "line"] + [c for c in _COLUMNS])
            for moment_name, moment in (report.get("moments") or {}).items():
                status = np.asarray(moment["status"], dtype=object)
                for index in range(status.size):
                    writer.writerow([
                        report.get("station", ""), report.get("line", ""),
                        moment_name, index + 1,
                        _at(moment.get("times"), index) * 1e6,
                        _at(moment.get("open"), index) * 1e6,
                        _at(moment.get("close"), index) * 1e6,
                        _at(moment.get("values"), index),
                        _at(moment.get("relative_std"), index),
                        _at(moment.get("flags"), index),
                        str(status[index]),
                    ])

    def export_png(self, path: str) -> None:
        from pyqtgraph.exporters import ImageExporter

        ImageExporter(self._plot.getPlotItem()).export(str(path))

    def _export_csv_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export gates as CSV", "gates.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def _export_png_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot as PNG", "gates.png", "PNG (*.png)")
        if path:
            self.export_png(path)

    # -- table ---------------------------------------------------------------
    def _fill_table(self) -> None:
        self._table.setSortingEnabled(False)
        rows = []
        for name, moment in ((self._report or {}).get("moments") or {}).items():
            status = np.asarray(moment["status"], dtype=object)
            for index in range(status.size):
                rows.append((
                    name, float(index + 1),
                    _at(moment.get("times"), index) * 1e6,
                    _at(moment.get("open"), index) * 1e6,
                    _at(moment.get("close"), index) * 1e6,
                    _at(moment.get("values"), index),
                    _at(moment.get("relative_std"), index),
                    _at(moment.get("flags"), index),
                    str(status[index]),
                ))
        self._table.setRowCount(len(rows))
        for r, record in enumerate(rows):
            verdict = str(record[-1])
            colour = QColor(_VERDICT_STYLE.get(verdict, ("#333333",))[0])
            for c, value in enumerate(record):
                item = _SortingItem(_cell(c, value), value)
                if verdict != "kept":
                    item.setForeground(colour)
                self._table.setItem(r, c, item)
        self._table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self._table.setSortingEnabled(True)


def _label(name: str, labelled: Set[str]) -> Optional[str]:
    """A legend entry the first time a category is drawn, and none after."""
    if name in labelled:
        return None
    labelled.add(name)
    return name


def _at(array: Any, index: int) -> float:
    values = np.asarray(array, dtype=float).ravel()
    return float(values[index]) if index < values.size else float("nan")


def _cell(column: int, value: Any) -> str:
    """One table cell, formatted for the column it sits in."""
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return ""
    if column in (1, 7):            # gate number and stored flag are integers
        return "%d" % value
    if column == 5:                 # a voltage spans many decades
        return "%.4g" % value
    return "%.3f" % value
