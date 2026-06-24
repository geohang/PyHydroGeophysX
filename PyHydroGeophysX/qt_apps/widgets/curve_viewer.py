"""Interactive multi-curve viewer built on pyqtgraph."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme

_CURVE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


class CurveViewer(QWidget):
    """Plot one or more curves with log toggles, point picking and export."""

    #: Emitted on a left click near a curve: (x, y).
    pointPicked = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._order: List[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.addLegend()
        self._plot = self._plot_widget.getPlotItem()
        layout.addWidget(self._plot_widget, stretch=1)

        self._pick_scatter = pg.ScatterPlotItem(
            size=11, pen=pg.mkPen("#000000", width=1), brush=pg.mkBrush(255, 60, 60, 200), symbol="o"
        )
        self._plot.addItem(self._pick_scatter)
        self._picked: List[Tuple[float, float]] = []

        bar = QHBoxLayout()
        self._readout = QLabel("x: -, y: -")
        self._readout.setMinimumWidth(180)
        bar.addWidget(self._readout, stretch=1)

        self._logx = QCheckBox("log x")
        self._logy = QCheckBox("log y")
        self._pick = QCheckBox("pick")
        self._logx.toggled.connect(self._apply_log)
        self._logy.toggled.connect(self._apply_log)
        bar.addWidget(self._logx)
        bar.addWidget(self._logy)
        bar.addWidget(self._pick)
        for label, slot, icon_name in (
            ("Clear", self.clear_curves, "fa5s.eraser"),
            ("Export CSV", self._export_csv_dialog, "fa5s.file-csv"),
            ("Export PNG", self._export_png_dialog, "fa5s.camera"),
        ):
            btn = QPushButton(label)
            btn.setIcon(theme.icon(icon_name))
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        layout.addLayout(bar)

        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # -- data ----------------------------------------------------------------
    def add_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> str:
        """Add a curve and return its name."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        name = name or f"curve {len(self._order) + 1}"
        color = color or _CURVE_COLORS[len(self._order) % len(_CURVE_COLORS)]
        self._curves[name] = (x, y)
        self._order.append(name)
        self._plot.plot(x, y, pen=pg.mkPen(color, width=2), name=name)
        return name

    def clear_curves(self) -> None:
        self._plot.clear()
        self._plot.addItem(self._pick_scatter)
        self._curves.clear()
        self._order.clear()
        self._picked.clear()
        self._pick_scatter.setData([], [])

    def picked_points(self) -> List[Tuple[float, float]]:
        return list(self._picked)

    # -- log mode ------------------------------------------------------------
    def set_log_x(self, enabled: bool) -> None:
        self._logx.setChecked(bool(enabled))

    def set_log_y(self, enabled: bool) -> None:
        self._logy.setChecked(bool(enabled))

    def _apply_log(self) -> None:
        self._plot.setLogMode(x=self._logx.isChecked(), y=self._logy.isChecked())

    # -- mouse ---------------------------------------------------------------
    def _on_mouse_moved(self, scene_pos) -> None:
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            return
        view_pt = self._plot.vb.mapSceneToView(scene_pos)
        self._readout.setText(f"x: {view_pt.x():.4g}, y: {view_pt.y():.4g}")

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._pick.isChecked():
            return
        if not self._plot.sceneBoundingRect().contains(event.scenePos()):
            return
        view_pt = self._plot.vb.mapSceneToView(event.scenePos())
        snapped = self._snap_to_curve(view_pt.x(), view_pt.y())
        x, y = snapped if snapped is not None else (view_pt.x(), view_pt.y())
        self._picked.append((float(x), float(y)))
        px = [p[0] for p in self._picked]
        py = [p[1] for p in self._picked]
        self._pick_scatter.setData(px, py)
        self.pointPicked.emit(float(x), float(y))

    def _snap_to_curve(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Return the nearest sample on any curve (in x) to the click."""
        best: Optional[Tuple[float, float]] = None
        best_dist = np.inf
        for cx, cy in self._curves.values():
            if cx.size == 0:
                continue
            idx = int(np.argmin(np.abs(cx - x)))
            dist = abs(cx[idx] - x)
            if dist < best_dist:
                best_dist = dist
                best = (float(cx[idx]), float(cy[idx]))
        return best

    # -- export --------------------------------------------------------------
    def export_csv(self, path: str) -> None:
        import csv

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for name in self._order:
                cx, cy = self._curves[name]
                writer.writerow([f"{name}_x", f"{name}_y"])
                for xv, yv in zip(cx, cy):
                    writer.writerow([xv, yv])
                writer.writerow([])

    def export_png(self, path: str) -> None:
        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(self._plot)
        exporter.export(str(path))

    def _export_csv_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export curves as CSV", "curves.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def _export_png_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export plot as PNG", "plot.png", "PNG (*.png)")
        if path:
            self.export_png(path)
