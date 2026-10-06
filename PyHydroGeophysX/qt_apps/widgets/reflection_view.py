"""The Seismic page's Reflection tab: a CMP-stacked section, and the two checks behind it.

One plot at a time, chosen in the toolbar the way the Gather tab chooses its
display:

* **Stacked section** - every shot stacked into one section along the line, in
  two-way time or in depth at the stacking velocity. Events that passed the
  flat-event check are marked; nothing else is.
* **Velocity check** - how well each stacking velocity lines the traces up
  (semblance), with the velocity used drawn on it. The line should run through
  the bright patches.
* **Flat-event check** - the traces from the middle of the line after the
  velocity correction, averaged by offset. A reflection is flat across the
  offsets; the edges of the mutes and ground roll are not.

The data come from :func:`PyHydroGeophysX.data_processing.seismic_shallow.stack_line`.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph import functions as pg_fn
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.flow_layout import FlowLayout, group
from PyHydroGeophysX.qt_apps.widgets.readout import ReadoutLabel
from PyHydroGeophysX.visualization.axis_units import to_display_length

SECTION, VELOCITY, FLATNESS = "Stacked section", "Velocity check", "Flat-event check"
_VIEWS = (SECTION, VELOCITY, FLATNESS)
_STYLES = ("Amplitude image", "Wiggle", "Image + wiggle")
_TIME, _DEPTH = "Time", "Depth"
_PLACEHOLDER = ("Set the traces, clean-up and velocity on the right, then press "
                "<b>Stack the line</b>.<br>The stacked section appears here.")


def _wiggle_paths(data: np.ndarray, y: np.ndarray, xpos: np.ndarray, half: float):
    """Wiggle line and positive-lobe fill for columns of ``data`` at ``xpos``.

    NaN samples (no traces there) draw nothing, rather than a baseline that
    reads as a recorded zero.
    """
    n, k = data.shape
    blank = ~np.isfinite(data)
    data = np.where(blank, 0.0, data)
    ref = np.percentile(np.abs(data), 96, axis=0)
    scale = 0.95 * half / np.maximum(ref, 1e-12)
    exc = xpos[None, :] + np.clip(data * scale, -half, half)
    pos = np.maximum(exc, xpos[None, :])
    exc = np.where(blank, np.nan, exc)
    gap = np.full((1, k), np.nan)
    y_bc = np.broadcast_to(y[:, None], (n, k))
    base = np.broadcast_to(xpos[None, :], (n, k))
    line_x = np.vstack([exc, gap]).reshape(-1, order="F")
    line_y = np.vstack([y_bc, gap]).reshape(-1, order="F")
    fill_x = np.vstack([pos, base, gap]).reshape(-1, order="F")
    fill_y = np.vstack([y_bc, y_bc[::-1], gap]).reshape(-1, order="F")
    return line_x, line_y, pg_fn.arrayToQPath(fill_x, fill_y, connect="finite")


class ReflectionView(QWidget):
    """Shows a :class:`~PyHydroGeophysX.data_processing.seismic_shallow.StackResult`."""

    def __init__(self, parent=None, *, colormaps=None) -> None:
        super().__init__(parent)
        self._result: Any = None
        self._verdict = ""
        self._ray_limit: Optional[float] = None   # depth (m) the velocity was measured to
        self._shown: Optional[dict] = None   # what is on screen, for the readout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bar = FlowLayout(spacing=6)
        self._view_combo = QComboBox()
        self._view_combo.addItems(_VIEWS)
        self._view_combo.setToolTip(
            "Stacked section: the result.\n"
            "Velocity check: is the stacking velocity right?\n"
            "Flat-event check: is an event a reflection, or left over from the mutes?")
        self._view_combo.currentTextChanged.connect(lambda *_: self._render())
        bar.addWidget(group("Show", self._view_combo))
        self._vertical = QComboBox()
        self._vertical.addItems((_TIME, _DEPTH))
        self._vertical.setToolTip("Depth converts two-way time with the stacking velocity, "
                                  "so it is only as right as that velocity.")
        self._vertical.currentTextChanged.connect(lambda *_: self._render())
        self._vertical_group = group("Vertical", self._vertical)
        bar.addWidget(self._vertical_group)
        self._style = QComboBox()
        self._style.addItems(_STYLES)
        self._style.setCurrentText("Image + wiggle")
        self._style.currentTextChanged.connect(lambda *_: self._render())
        # The same colour map as the shot gathers: both are seismic amplitudes.
        self._colormap = cmaps.ColormapChooser(cmaps.SEISMIC_GATHER, cmaps.GATHER, shared=colormaps)
        self._colormap.colormapChanged.connect(lambda *_: self._render())
        self._style_group = group("Display", self._style, self._colormap)
        bar.addWidget(self._style_group)
        self._readout = ReadoutLabel("", alignment=Qt.AlignLeft | Qt.AlignVCenter,
                                     sample="position 00.00 ft, 00.0 ms (00.00 ft deep), 00 traces")
        bar.addWidget(self._readout)
        layout.addLayout(bar)

        self._stack = QStackedLayout()
        self._placeholder = QLabel(_PLACEHOLDER)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        theme.set_tone(self._placeholder, "muted")
        self._stack.addWidget(self._placeholder)
        self._glw = pg.GraphicsLayoutWidget()
        self._stack.addWidget(self._glw)
        holder = QWidget()
        holder.setLayout(self._stack)
        layout.addWidget(holder, stretch=1)

        self._plot = self._glw.addPlot()
        self._plot.invertY(True)
        self._plot.showGrid(x=False, y=False)
        # Where no trace reached, a flat grey: a blank point is not a zero amplitude.
        self._backdrop = QGraphicsRectItem()
        self._backdrop.setPen(QPen(Qt.NoPen))
        self._backdrop.setZValue(-20)
        self._plot.getViewBox().addItem(self._backdrop)
        self._img = pg.ImageItem()
        self._img.setZValue(-10)
        self._plot.addItem(self._img)
        self._fill = QGraphicsPathItem()
        self._fill.setBrush(QBrush(QColor(0, 0, 0, 150)))
        self._fill.setPen(QPen(Qt.NoPen))
        self._fill.setZValue(-5)
        self._plot.getViewBox().addItem(self._fill)
        self._wiggle = pg.PlotCurveItem(pen=pg.mkPen(QColor(15, 15, 15, 220), width=1.0),
                                        connect="finite")
        self._plot.addItem(self._wiggle)
        self._curve = pg.PlotDataItem()
        self._curve.setZValue(20)
        self._plot.addItem(self._curve)
        self._marks: List[pg.InfiniteLine] = []
        self._plot.scene().sigMouseMoved.connect(self._on_move)
        length_units.notifier().changed.connect(lambda *_: self._render())
        theme.notifier().changed.connect(lambda *_: self._render())
        self._sync_controls()

    # -- public API ----------------------------------------------------------
    def show_result(self, result: Any) -> None:
        self._result = result
        self._stack.setCurrentWidget(self._glw)
        self._render()

    def set_ray_limit(self, depth: Optional[float]) -> None:
        """Mark the depth the stacking velocity was measured to (None: no mark).

        With a refraction model's velocity, deeper than its rays the velocity -
        and so the depth scale - is carried down, not measured.
        """
        self._ray_limit = None if depth is None else float(depth)
        self._render()

    def set_verdict(self, text: str) -> None:
        """The stacked section's title: what the flat-event check found."""
        self._verdict = str(text)
        self._render()

    def clear(self) -> None:
        self._result = None
        self._verdict = ""
        self._ray_limit = None
        self._shown = None
        self._plot.setTitle(None)
        self._stack.setCurrentWidget(self._placeholder)
        self._readout.setText("")

    def set_view(self, name: str) -> None:
        self._view_combo.setCurrentText(name)

    def view(self) -> str:
        return self._view_combo.currentText()

    def set_vertical(self, which: str) -> None:
        """``'time'`` or ``'depth'`` for the stacked section."""
        self._vertical.setCurrentText(_DEPTH if str(which).lower() == "depth" else _TIME)

    def has_result(self) -> bool:
        return self._result is not None

    def export_png(self, path: str) -> None:
        from pyqtgraph.exporters import ImageExporter

        ImageExporter(self._plot).export(str(path))

    # -- drawing -------------------------------------------------------------
    def _sync_controls(self) -> None:
        view = self._view_combo.currentText()
        self._vertical_group.setVisible(view == SECTION)
        self._style_group.setVisible(view != VELOCITY)

    def _clear_items(self) -> None:
        for mark in self._marks:
            self._plot.removeItem(mark)
        self._marks = []
        self._curve.setData([], [])
        self._wiggle.setData([], [])
        self._fill.setPath(QPainterPath())

    def _title(self, text: str) -> None:
        self._plot.setTitle(text, color=theme.color("canvas_text"), size="10pt")

    def _render(self) -> None:
        self._sync_controls()
        if self._result is None:
            return
        self._clear_items()
        self._backdrop.setVisible(False)
        view = self._view_combo.currentText()
        if view == VELOCITY:
            self._draw_velocity()
        elif view == FLATNESS:
            self._draw_flatness()
        else:
            self._draw_section()

    def _draw_amplitudes(self, data: np.ndarray, xpos: np.ndarray, width: float,
                         y: np.ndarray) -> None:
        """An amplitude image and/or wiggles, columns at ``xpos`` and rows at ``y``."""
        style = self._style.currentText()
        self._backdrop.setBrush(QBrush(QColor(theme.color("canvas")).darker(107)))
        self._backdrop.setVisible(True)
        finite = np.abs(data[np.isfinite(data) & (data != 0)])
        clip = float(np.percentile(finite, 98)) if finite.size else 1.0
        clip = clip if clip > 0 else 1.0
        dy = float(y[1] - y[0]) if y.size > 1 else 1.0
        rect = QRectF(float(xpos[0]) - width / 2, float(y[0]) - dy / 2,
                      width * xpos.size, dy * y.size)
        self._backdrop.setRect(rect)
        if style != "Wiggle":
            self._img.setVisible(True)
            self._img.setLookupTable(cmaps.lookup_table(self._colormap.colormap(), 256))
            self._img.setImage(np.clip(data, -clip, clip), autoLevels=False)
            self._img.setLevels((-clip, clip))
            self._img.setRect(rect)
        else:
            self._img.setVisible(False)
        if style != "Amplitude image":
            line_x, line_y, path = _wiggle_paths(data, y, xpos, width / 2)
            self._wiggle.setData(line_x, line_y)
            self._fill.setPath(path)
        self._plot.getViewBox().setRange(
            xRange=(float(xpos[0]) - width / 2, float(xpos[-1]) + width / 2),
            yRange=(float(y[0]), float(y[-1])), padding=0.0)

    def _mark(self, y: float, text: str, passes: bool, *, left: bool = False) -> None:
        # The canvas is light in both appearances, so marks take the light palette.
        colour = theme.LIGHT["green"] if passes else theme.LIGHT["muted"]
        where = ({"position": 0.02, "anchors": [(0, 1), (0, 1)]} if left
                 else {"position": 0.98, "anchors": [(1, 1), (1, 1)]})
        line = pg.InfiniteLine(
            pos=y, angle=0, movable=False,
            pen=pg.mkPen(colour, width=2 if passes else 1,
                         style=Qt.SolidLine if passes else Qt.DashLine),
            label=text, labelOpts={**where, "color": colour,
                                   "fill": pg.mkBrush(255, 255, 255, 200)})
        line.setZValue(30)
        self._plot.addItem(line)
        self._marks.append(line)

    def _time_axis(self) -> None:
        self._plot.getAxis("left").setScale(1.0)
        self._plot.setLabel("left", "Two-way time (ms)")

    def _draw_section(self) -> None:
        res = self._result
        st = res.stack
        t_ms = st.t0 * 1e3
        data = np.where(st.fold >= st.min_fold, st.stack, np.nan)
        depth = self._vertical.currentText() == _DEPTH
        if depth:
            z_of_t = np.asarray(res.velocity.depth(st.t0), dtype=float)
            y = np.linspace(0.0, float(z_of_t[-1]), st.t0.size)
            frac = np.interp(y, z_of_t, np.arange(st.t0.size, dtype=float))
            lo = np.clip(np.floor(frac).astype(int), 0, st.t0.size - 2)
            w = (frac - lo)[:, None]
            data = data[lo] * (1 - w) + data[lo + 1] * w
            fold = st.fold[np.clip(np.round(frac).astype(int), 0, st.t0.size - 1)]
            length_units.pyqtgraph_axis(self._plot, "left", "Depth")
        else:
            y = t_ms
            fold = st.fold
            self._time_axis()
        length_units.pyqtgraph_axis(self._plot, "bottom", "Position along the line")
        self._draw_amplitudes(data, st.cmp_x, st.bin_width, y)
        self._title(self._verdict)
        if self._ray_limit is not None:
            if depth:
                at = self._ray_limit
            else:
                at = float(np.interp(self._ray_limit, res.velocity.depth(st.t0), t_ms))
            self._mark(at, "deeper than the rays reach: depth scale carried down", False,
                       left=True)
        sg = res.supergather
        for event in (sg.passing if sg is not None else []):
            t0 = event["t0"]
            at = float(res.velocity.depth(np.array([t0]))[0]) if depth else t0 * 1e3
            self._mark(at, f"{t0 * 1e3:.1f} ms - flat across offsets", True)
        self._shown = {"view": SECTION, "x": st.cmp_x, "width": st.bin_width, "y": y,
                       "fold": fold, "depth": depth}

    def _draw_velocity(self) -> None:
        res = self._result
        t_ms = res.semblance_t0 * 1e3
        vels = res.semblance_velocities
        panel = res.semblance_hyperbolic.T          # (n_t0, n_v)
        self._img.setVisible(True)
        self._img.setLookupTable(cmaps.lookup_table("Greys", 256))
        self._img.setImage(panel, autoLevels=False)
        self._img.setLevels((0.0, max(float(panel.max()), 1e-6)))
        dv = float(vels[1] - vels[0]) if vels.size > 1 else 1.0
        dt = float(t_ms[1] - t_ms[0]) if t_ms.size > 1 else 1.0
        self._img.setRect(QRectF(float(vels[0]) - dv / 2, float(t_ms[0]) - dt / 2,
                                 dv * vels.size, dt * t_ms.size))
        t_line = np.linspace(float(t_ms[0]), float(t_ms[-1]), 200)
        self._curve.setData(np.asarray(res.velocity.rms(t_line * 1e-3), dtype=float), t_line,
                            pen=pg.mkPen(theme.color("primary"), width=2.5))
        self._plot.getAxis("bottom").setScale(1.0)
        self._plot.setLabel("bottom", "Stacking velocity (m/s)")
        self._time_axis()
        self._plot.getViewBox().setRange(xRange=(float(vels[0]), float(vels[-1])),
                                         yRange=(0.0, float(t_ms[-1])), padding=0.0)
        self._title("Darker: the traces line up better at that velocity. "
                    "Blue line: the velocity used.")
        self._shown = {"view": VELOCITY, "v": vels, "t": t_ms, "panel": panel}

    def _draw_flatness(self) -> None:
        sg = self._result.supergather
        if sg is None:
            self._img.setVisible(False)
            self._title("Too few traces in the middle of the line for this check.")
            self._shown = None
            return
        t_ms = sg.t0 * 1e3
        width = float(sg.settings.get("bin_width", 1.0))
        length_units.pyqtgraph_axis(self._plot, "bottom", "Offset")
        self._time_axis()
        self._draw_amplitudes(np.where(sg.binned != 0.0, sg.binned, np.nan), sg.bin_offsets,
                              width, t_ms)
        unit = length_units.current()
        lo, hi = (to_display_length(v, unit) for v in sg.cmp_range)
        self._title(f"Traces from {lo:.1f} to {hi:.1f} {unit} along the line, corrected with the "
                    "velocity and averaged by offset. A reflection lies flat.")
        for event in sg.events:
            label = (f"{event['t0'] * 1e3:.1f} ms - flat" if event["passes"]
                     else f"{event['t0'] * 1e3:.1f} ms - not flat")
            self._mark(event["t0"] * 1e3, label, bool(event["passes"]))
        self._shown = {"view": FLATNESS, "x": sg.bin_offsets, "width": width, "y": t_ms}

    # -- readout -------------------------------------------------------------
    def _on_move(self, scene_pos) -> None:
        shown = self._shown
        if shown is None or not self._plot.sceneBoundingRect().contains(scene_pos):
            return
        p = self._plot.getViewBox().mapSceneToView(scene_pos)
        x, y = float(p.x()), float(p.y())
        unit = length_units.current()
        if shown["view"] == VELOCITY:
            a = int(np.clip(np.searchsorted(shown["t"], y), 0, shown["t"].size - 1))
            b = int(np.clip(np.searchsorted(shown["v"], x), 0, shown["v"].size - 1))
            self._readout.setText(f"{x:.0f} m/s, {y:.1f} ms, semblance {shown['panel'][a, b]:.2f}")
            return
        where = f"{to_display_length(x, unit):.2f} {unit}"
        if shown["view"] == FLATNESS:
            self._readout.setText(f"offset {where}, {y:.1f} ms")
            return
        k = int(np.argmin(np.abs(shown["x"] - x)))
        i = int(np.clip(np.searchsorted(shown["y"], y), 0, shown["y"].size - 1))
        traces = int(shown["fold"][i, k])
        if shown["depth"]:
            when = f"{to_display_length(y, unit):.2f} {unit} deep"
        else:
            depth = float(self._result.velocity.depth(np.array([y * 1e-3]))[0])
            when = f"{y:.1f} ms ({to_display_length(depth, unit):.2f} {unit} deep)"
        self._readout.setText(f"position {where}, {when}, {traces} traces")
