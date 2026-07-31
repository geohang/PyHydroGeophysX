"""Seismic gather viewer that matches the Streamlit web display style.

Renders a shot gather with time on the vertical axis (increasing downward), a
bipolar blue-white-red amplitude image and/or a traditional variable-area wiggle
display, and first-arrival picks drawn as markers colored by source
(auto = red x, manual = green circle, anchor = orange star). Click-to-pick emits
``pointPicked(trace_index, time_s, amplitude)``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph import functions as pg_fn
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGraphicsPathItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

pg.setConfigOptions(imageAxisOrder="row-major")

# Bipolar seismic colour scale (blue negative, white zero, red positive) — the
# same stops the Streamlit app uses.
_SEIS_STOPS = np.array([0.0, 0.48, 0.5, 0.52, 1.0])
_SEIS_COLORS = np.array([
    [8, 48, 107, 255],
    [247, 251, 255, 255],
    [255, 255, 255, 255],
    [255, 245, 240, 255],
    [153, 0, 13, 255],
], dtype=np.ubyte)

# source -> (colour, pyqtgraph symbol, size)
_PICK_STYLES: Dict[str, Tuple[str, str, int]] = {
    "manual": ("#00843d", "o", 13),
    "anchor": ("#ff9f1c", "star", 13),
    "learned": ("#fdae61", "d", 10),
    "auto": ("#d7191c", "x", 10),
}

_STYLES = ["Amplitude image", "Traditional wiggle", "Image + wiggle"]


def first_arrival_onsets(disp: np.ndarray, dt: Optional[float], ratio_thr: float = 5.0,
                         sta_s: float = 0.002, lta_s: float = 0.02) -> np.ndarray:
    """Per-trace first-arrival onset (sample index) via an STA/LTA energy ratio.

    Amplitude-independent, so it follows the refraction first break even when
    later reflections are stronger. Returns one sample index per trace (``nan``
    when no onset crosses the ratio threshold).
    """
    disp = np.asarray(disp, dtype=float)
    if disp.ndim != 2 or disp.size == 0:
        return np.array([])
    n, m = disp.shape
    if dt and dt > 0:
        nsta = max(2, int(round(sta_s / dt)))
        nlta = max(nsta * 4, int(round(lta_s / dt)))
    else:
        nsta = max(2, n // 200)
        nlta = max(nsta * 4, n // 20)
    nlta = min(nlta, max(nsta + 1, n // 2))
    energy = disp * disp
    csum = np.vstack([np.zeros((1, m)), np.cumsum(energy, axis=0)])
    idx = np.arange(n)
    lo_sta = np.maximum(0, idx + 1 - nsta)
    lo_lta = np.maximum(0, idx + 1 - nlta)
    sta = (csum[idx + 1] - csum[lo_sta]) / (idx + 1 - lo_sta)[:, None]
    lta = (csum[idx + 1] - csum[lo_lta]) / (idx + 1 - lo_lta)[:, None]
    ratio = sta / (lta + 1e-15)
    start = nsta
    onsets = np.full(m, np.nan)
    for j in range(m):
        cand = np.where(ratio[start:, j] > ratio_thr)[0]
        if cand.size:
            onsets[j] = float(start + int(cand[0]))

    # Trend-guided refinement: reject early (noise) outliers by re-searching the
    # ratio crossing from just before the local median trend of the picks.
    finite = np.isfinite(onsets)
    if int(finite.sum()) >= 5:
        try:
            from scipy.ndimage import median_filter

            xs = np.arange(m)
            filled = onsets.copy()
            if (~finite).any():
                filled[~finite] = np.interp(xs[~finite], xs[finite], onsets[finite])
            trend = median_filter(filled, size=max(3, min(7, m - (1 - m % 2))), mode="nearest")
            guard = max(nsta * 2, nlta // 2)
            for j in range(m):
                if not np.isfinite(onsets[j]):
                    continue
                if onsets[j] < trend[j] - guard:
                    lo = int(max(start, trend[j] - guard))
                    cand = np.where(ratio[lo:, j] > ratio_thr)[0]
                    onsets[j] = float(lo + int(cand[0])) if cand.size else float(trend[j])
        except Exception:
            pass
    return onsets


class _PickViewBox(pg.ViewBox):
    """ViewBox that turns Ctrl+left-drag into a line-pick gesture.

    Without the Ctrl modifier it behaves like a normal ViewBox (pan/zoom).
    """

    dragStarted = Signal(object)
    dragMoved = Signal(object)
    dragFinished = Signal(object)

    def mouseDragEvent(self, ev, axis=None):  # noqa: N802 - pyqtgraph API
        if ev.button() == Qt.LeftButton and (ev.modifiers() & Qt.ControlModifier):
            ev.accept()
            point = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                self.dragStarted.emit(point)
            elif ev.isFinish():
                self.dragFinished.emit(point)
            else:
                self.dragMoved.emit(point)
        else:
            super().mouseDragEvent(ev, axis)


class SeismicViewer(QWidget):
    """Interactive seismic gather display with web-matched styling."""

    pointPicked = Signal(int, float, float)  # trace_index, time_s, amplitude
    linePicked = Signal(list)  # [(trace_index, time_s, amplitude), ...]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._disp: Optional[np.ndarray] = None     # processed (samples, traces)
        self._time: Optional[np.ndarray] = None     # y axis values (s or samples)
        self._dt: Optional[float] = None
        self._clip_pct = 99.0
        self._pick_mode = False
        self._picks: Dict[int, Tuple[float, str]] = {}
        self._auto_window_on = True
        self._window_max: Optional[float] = None
        self._full_max: Optional[float] = None
        self._drag_anchor: Optional[Tuple[float, float]] = None
        self._shown_ntraces: Optional[int] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Display"))
        self._style_combo = QComboBox()
        self._style_combo.addItems(_STYLES)
        self._style_combo.currentTextChanged.connect(lambda *_: self._render())
        bar.addWidget(self._style_combo)
        bar.addWidget(QLabel("Wiggle/N"))
        self._stride = QSpinBox()
        self._stride.setRange(1, 64)
        self._stride.setValue(1)
        self._stride.valueChanged.connect(lambda *_: self._render())
        bar.addWidget(self._stride)
        bar.addWidget(QLabel("Max time"))
        self._tmax_spin = QDoubleSpinBox()
        self._tmax_spin.setDecimals(4)
        self._tmax_spin.setRange(0.0, 1e9)
        self._tmax_spin.setSuffix(" s")
        self._tmax_spin.setToolTip("Truncate the time axis (e.g. to the longest first arrival).")
        self._tmax_spin.valueChanged.connect(self._on_tmax_changed)
        bar.addWidget(self._tmax_spin)
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setMaximumWidth(60)
        self._auto_btn.setToolTip("Auto-fit the time window to the first arrivals.")
        self._auto_btn.clicked.connect(self._reset_window)
        bar.addWidget(self._auto_btn)
        bar.addStretch(1)
        self._readout = QLabel("trace: -, time: -, amp: -")
        bar.addWidget(self._readout)
        layout.addLayout(bar)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw, stretch=1)
        self._vb = _PickViewBox()
        self._vb.dragStarted.connect(self._drag_start)
        self._vb.dragMoved.connect(self._drag_move)
        self._vb.dragFinished.connect(self._drag_finish)
        self._plot = self._glw.addPlot(viewBox=self._vb)
        self._plot.setLabel("bottom", "Trace")
        self._plot.setLabel("left", "Time (s)")
        self._plot.invertY(True)
        self._plot.setAspectLocked(False)

        self._img = pg.ImageItem()
        self._img.setZValue(-10)
        self._lut = pg.ColorMap(_SEIS_STOPS, _SEIS_COLORS).getLookupTable(0.0, 1.0, 256)
        self._img.setLookupTable(self._lut)
        self._plot.addItem(self._img)

        self._fill_item = QGraphicsPathItem()
        self._fill_item.setBrush(QBrush(QColor(0, 0, 0, 150)))
        self._fill_item.setPen(QPen(Qt.NoPen))
        self._fill_item.setZValue(-5)
        self._vb.addItem(self._fill_item)

        self._wiggle = pg.PlotCurveItem(pen=pg.mkPen(QColor(15, 15, 15, 220), width=1.0), connect="finite")
        self._wiggle.setZValue(0)
        self._plot.addItem(self._wiggle)

        self._scatters: Dict[str, pg.ScatterPlotItem] = {}
        for src, (color, symbol, size) in _PICK_STYLES.items():
            scatter = pg.ScatterPlotItem(
                size=size, symbol=symbol, pen=pg.mkPen("#222222", width=1.4), brush=pg.mkBrush(color))
            scatter.setZValue(10)
            self._plot.addItem(scatter)
            self._scatters[src] = scatter

        self._drag_line = pg.PlotDataItem(pen=pg.mkPen("#00843d", width=2, style=Qt.DashLine))
        self._drag_line.setZValue(25)
        self._plot.addItem(self._drag_line)

        self._plot.scene().sigMouseMoved.connect(self._on_move)
        self._plot.scene().sigMouseClicked.connect(self._on_click)

    # -- public API ----------------------------------------------------------
    def show_gather(self, disp: np.ndarray, dt: Optional[float], clip_pct: float = 99.0) -> None:
        """Display a processed (samples x traces) gather. ``dt`` in seconds."""
        self._disp = np.asarray(disp, dtype=float)
        self._dt = dt if dt and dt > 0 else None
        self._clip_pct = clip_pct
        nsamples, ntraces = self._disp.shape
        if ntraces != self._shown_ntraces:
            self._shown_ntraces = ntraces
            self._suggest_stride(ntraces)
        if self._dt:
            self._time = np.arange(nsamples) * self._dt
            self._plot.setLabel("left", "Time (s)")
        else:
            self._time = np.arange(nsamples, dtype=float)
            self._plot.setLabel("left", "Sample")
        self._full_max = float(self._time[-1]) if self._time.size else float(nsamples)
        self._configure_tmax_spin()
        if self._auto_window_on or self._window_max is None:
            self._window_max = self._estimate_window()
        self._window_max = float(min(self._window_max, self._full_max))
        self._set_tmax_spin_value(self._window_max)
        self._render()
        self._apply_view()

    def _suggest_stride(self, ntraces: int) -> None:
        """Default the wiggle stride so a wide gather stays responsive.

        Only runs when the trace count changes (new data), and it sets the
        visible spin box so the choice is transparent and still user-editable.
        """
        target = 600
        stride = max(1, -(-ntraces // target)) if ntraces > target else 1
        if stride != self._stride.value():
            self._stride.blockSignals(True)
            self._stride.setValue(stride)
            self._stride.blockSignals(False)

    def set_pick_mode(self, enabled: bool) -> None:
        self._pick_mode = bool(enabled)

    def set_picks(self, picks: Dict[int, Tuple[float, str]]) -> None:
        """picks: trace_index -> (time_s, source)."""
        self._picks = dict(picks)
        self._render_picks()
        if self._auto_window_on and self._picks:
            self._window_max = self._estimate_window()
            self._set_tmax_spin_value(self._window_max)
            self._apply_view()

    # -- time window ---------------------------------------------------------
    def _configure_tmax_spin(self) -> None:
        self._tmax_spin.blockSignals(True)
        if self._dt:
            self._tmax_spin.setSuffix(" s"); self._tmax_spin.setDecimals(4)
            self._tmax_spin.setSingleStep(max(self._dt * 5, 0.001))
            lo = self._dt * 5
        else:
            self._tmax_spin.setSuffix(" smp"); self._tmax_spin.setDecimals(0)
            self._tmax_spin.setSingleStep(10)
            lo = 5.0
        self._tmax_spin.setRange(lo, float(self._full_max or 1.0))
        self._tmax_spin.blockSignals(False)

    def _set_tmax_spin_value(self, value: float) -> None:
        self._tmax_spin.blockSignals(True)
        self._tmax_spin.setValue(float(value))
        self._tmax_spin.blockSignals(False)

    def _estimate_window(self) -> float:
        """Best-effort first-arrival window: from picks if present, else from an
        STA/LTA onset estimate (90th percentile, so a few late traces do not blow
        up the window)."""
        full = float(self._full_max or 1.0)
        dt = self._dt or 1.0
        if self._picks:
            times = [t for (t, _s) in self._picks.values() if np.isfinite(t)]
            if times:
                return float(min(full, max(times) * 1.25 + 5.0 * dt))
        if self._disp is not None and self._time is not None and self._time.size:
            onsets = first_arrival_onsets(self._disp, self._dt)
            valid = onsets[np.isfinite(onsets)] if onsets.size else np.array([])
            if valid.size:
                t90 = float(np.percentile(valid, 90)) * dt
                return float(min(full, t90 * 1.4 + 5.0 * dt))
        return full

    def _apply_view(self) -> None:
        if self._disp is None:
            return
        _, m = self._disp.shape
        self._vb.setXRange(-0.5, m - 0.5, padding=0.02)
        self._vb.setYRange(0.0, float(self._window_max or self._full_max or 1.0), padding=0.0)

    def _on_tmax_changed(self, value: float) -> None:
        self._auto_window_on = False
        self._window_max = float(value)
        self._apply_view()

    def _reset_window(self) -> None:
        self._auto_window_on = True
        self._window_max = self._estimate_window()
        self._set_tmax_spin_value(self._window_max)
        self._apply_view()

    def export_png(self, path: str) -> None:
        from pyqtgraph.exporters import ImageExporter

        ImageExporter(self._plot).export(str(path))

    # -- rendering -----------------------------------------------------------
    def _clip(self) -> float:
        finite = np.abs(self._disp[np.isfinite(self._disp)])
        clip = float(np.percentile(finite, self._clip_pct)) if finite.size else 1.0
        return clip if clip > 0 else 1.0

    def _render(self) -> None:
        if self._disp is None:
            return
        nsamples, ntraces = self._disp.shape
        tmax = float(self._time[-1]) if self._time is not None and self._time.size else float(nsamples)
        style = self._style_combo.currentText()
        clip = self._clip()

        if style in ("Amplitude image", "Image + wiggle"):
            self._img.setVisible(True)
            self._img.setImage(np.clip(self._disp, -clip, clip), autoLevels=False)
            self._img.setLevels((-clip, clip))
            self._img.setRect(QRectF(-0.5, 0.0, float(ntraces), tmax))
        else:
            self._img.setVisible(False)

        if style in ("Traditional wiggle", "Image + wiggle"):
            self._draw_wiggle(clip)
        else:
            self._wiggle.setData([], [])
            self._fill_item.setPath(QPainterPath())
        self._render_picks()

    def _draw_wiggle(self, clip: float) -> None:
        # Vectorized: build the wiggle line and its variable-area fill from whole
        # arrays (one QPainterPath via arrayToQPath) instead of a per-sample Python
        # lineTo loop, which is what made a wide gather stutter on every redraw.
        disp, t = self._disp, self._time
        if disp is None or t is None or disp.size == 0:
            self._wiggle.setData([], [])
            self._fill_item.setPath(QPainterPath())
            return
        nsamples, ntraces = disp.shape
        stride = max(1, int(self._stride.value()))
        cols = np.arange(0, ntraces, stride)
        if cols.size == 0:
            self._wiggle.setData([], [])
            self._fill_item.setPath(QPainterPath())
            return

        sub = np.nan_to_num(disp[:, cols])                          # (n, K)
        # per-trace normalization so far-offset (weak) traces stay visible
        tref = np.percentile(np.abs(sub), 96, axis=0)               # (K,)
        scale = 0.46 / np.maximum(tref, 1e-12)
        base = cols.astype(float)                                   # (K,)
        xexc = base[None, :] + np.clip(sub * scale, -0.48, 0.48)    # (n, K)
        pos = np.maximum(xexc, base[None, :])                       # positive lobe

        K = cols.size
        nan_row = np.full((1, K), np.nan)
        t_bc = np.broadcast_to(t[:, None], (nsamples, K))
        base_bc = np.broadcast_to(base[None, :], (nsamples, K))

        # Wiggle line: the excursion per trace, NaN-separated between traces.
        line_x = np.vstack([xexc, nan_row]).reshape(-1, order="F")
        line_y = np.vstack([t_bc, nan_row]).reshape(-1, order="F")
        self._wiggle.setData(line_x, line_y)

        # Fill polygon per trace: down the positive lobe, back up the baseline.
        fill_x = np.vstack([pos, base_bc, nan_row]).reshape(-1, order="F")
        fill_y = np.vstack([t_bc, t_bc[::-1], nan_row]).reshape(-1, order="F")
        try:
            path = pg_fn.arrayToQPath(fill_x, fill_y, connect="finite")
        except Exception:  # noqa: BLE001 - rare: fall back to a safe per-trace build
            path = self._wiggle_fill_fallback(pos, base, t)
        self._fill_item.setPath(path)

    @staticmethod
    def _wiggle_fill_fallback(pos: np.ndarray, base: np.ndarray, t: np.ndarray) -> QPainterPath:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPolygonF

        path = QPainterPath()
        nsamples, K = pos.shape
        for k in range(K):
            poly = QPolygonF()
            for i in range(nsamples):
                poly.append(QPointF(float(pos[i, k]), float(t[i])))
            b = float(base[k])
            for i in range(nsamples - 1, -1, -1):
                poly.append(QPointF(b, float(t[i])))
            path.addPolygon(poly)
        return path

    def _render_picks(self) -> None:
        if self._time is None:
            return
        buckets: Dict[str, list] = {src: [] for src in self._scatters}
        for trace, (time_s, source) in self._picks.items():
            src = source if source in buckets else "auto"
            buckets[src].append((float(trace), float(time_s)))
        for src, scatter in self._scatters.items():
            pts = buckets.get(src, [])
            if pts:
                xs, ys = zip(*pts)
                scatter.setData(list(xs), list(ys))
            else:
                scatter.setData([], [])

    # -- mouse ---------------------------------------------------------------
    def _coords(self, scene_pos) -> Optional[Tuple[int, float, float]]:
        if self._disp is None or not self._plot.sceneBoundingRect().contains(scene_pos):
            return None
        view = self._vb.mapSceneToView(scene_pos)
        nsamples, ntraces = self._disp.shape
        trace = int(round(view.x()))
        if not (0 <= trace < ntraces):
            return None
        t = float(view.y())
        dt = self._dt or 1.0
        sample = int(round(t / dt))
        sample = max(0, min(sample, nsamples - 1))
        return trace, t, float(self._disp[sample, trace])

    def _on_move(self, scene_pos) -> None:
        hit = self._coords(scene_pos)
        if hit is None:
            self._readout.setText("trace: -, time: -, amp: -")
            return
        trace, t, amp = hit
        unit = "s" if self._dt else "smp"
        self._readout.setText(f"trace: {trace}, time: {t:.4g} {unit}, amp: {amp:.3g}")

    def _on_click(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._pick_mode:
            return
        hit = self._coords(event.scenePos())
        if hit is None:
            return
        trace, t, amp = hit
        self.pointPicked.emit(int(trace), float(t), float(amp))

    # -- line pick (Ctrl + drag) ---------------------------------------------
    def _drag_start(self, point) -> None:
        if self._disp is None:
            return
        self._drag_anchor = (float(point.x()), float(point.y()))
        self._drag_line.setData([self._drag_anchor[0]], [self._drag_anchor[1]])

    def _drag_move(self, point) -> None:
        if self._disp is None or self._drag_anchor is None:
            return
        self._drag_line.setData(
            [self._drag_anchor[0], float(point.x())],
            [self._drag_anchor[1], float(point.y())],
        )

    def _drag_finish(self, point) -> None:
        anchor = self._drag_anchor
        self._drag_anchor = None
        self._drag_line.setData([], [])
        if self._disp is None or anchor is None:
            return
        x0, y0 = anchor
        x1, y1 = float(point.x()), float(point.y())
        nsamples, ntraces = self._disp.shape
        lo, hi = sorted((int(round(x0)), int(round(x1))))
        lo, hi = max(0, lo), min(ntraces - 1, hi)
        if hi < lo:
            return
        dt = self._dt or 1.0
        points = []
        for trace in range(lo, hi + 1):
            t = y1 if x1 == x0 else y0 + (y1 - y0) * (trace - x0) / (x1 - x0)
            sample = max(0, min(int(round(t / dt)), nsamples - 1))
            points.append((int(trace), float(t), float(self._disp[sample, trace])))
        if points:
            self.linePicked.emit(points)
