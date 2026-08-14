"""Interactive 2D array/image viewer built on pyqtgraph.

Coordinate convention: arrays are displayed in ``row-major`` order, so a data
coordinate ``(x, y)`` maps directly to ``(col, row)`` of the underlying numpy
array. That matches ``ProfileInterpolator`` which expects ``point=[col, row]``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme

# Display arrays as image[row, col] (numpy convention) rather than the pyqtgraph
# default of image[x, y]. This makes click -> [col, row] mapping correct.
pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


class ArrayViewer(QWidget):
    """Display a 2D array, read out values on hover, pick points, draw a profile."""

    #: Emitted on a left click in pick mode: (col, row, value).
    pointPicked = Signal(float, float, float)
    #: Emitted once a second profile point is set: ([col1, row1], [col2, row2]).
    profileSelected = Signal(list, list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._array: Optional[np.ndarray] = None
        self._extent: Optional[Tuple[float, float, float, float]] = None
        self._pick_mode = False
        self._profile_mode = False
        self._markers: List[Tuple[float, float]] = []
        self._profile_pts: List[List[float]] = []
        self._value_label = "Value"
        self._log_display = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw, stretch=1)

        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setLabel("bottom", "Column (x)")
        self._plot.setLabel("left", "Row (y)")
        self._plot.invertY(True)  # row 0 at top, like an image
        self._plot.setAspectLocked(False)

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        self._hist = pg.HistogramLUTItem(image=self._img)
        try:
            self._hist.gradient.loadPreset("viridis")
        except Exception:
            pass
        self._hist.axis.setLabel(self._value_label)
        self._glw.addItem(self._hist, row=0, col=1)

        self._marker_scatter = pg.ScatterPlotItem(
            size=11, pen=pg.mkPen("#ff3030", width=2), brush=pg.mkBrush(255, 80, 80, 160), symbol="x"
        )
        self._plot.addItem(self._marker_scatter)
        self._profile_scatter = pg.ScatterPlotItem(
            size=13, pen=pg.mkPen("#1565ff", width=2), brush=pg.mkBrush(40, 120, 255, 180), symbol="o"
        )
        self._plot.addItem(self._profile_scatter)
        self._profile_line = pg.PlotDataItem(pen=pg.mkPen("#1565ff", width=2))
        self._plot.addItem(self._profile_line)

        # Readout + action buttons.
        bar = QHBoxLayout()
        self._readout = QLabel("x: -, y: -, value: -")
        self._readout.setMinimumWidth(260)
        bar.addWidget(self._readout, stretch=1)
        for label, slot, icon_name in (
            ("Clear markers", self.clear_markers, "fa5s.times-circle"),
            ("Clear profile", self.clear_profile, "fa5s.eraser"),
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
    def set_array(
        self,
        array: np.ndarray,
        autoscale: bool = True,
        extent: Optional[Tuple[float, float, float, float]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        log: bool = False,
        invert_y: Optional[bool] = None,
        value_label: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Display ``array`` (2D). Levels default to the 2-98 percentile.

        ``extent=(x0, x1, z0, z1)`` maps the array into real-world units (e.g. a
        resistivity section in metres) instead of column/row indices; ``log``
        colours by ``log10`` while the hover read-out still reports the linear
        value; ``x_label`` / ``y_label`` / ``invert_y`` adjust the axes.
        ``value_label`` labels both the colour scale and hover readout.  For a
        logarithmic display, colour-bar ticks are formatted back into the
        original physical values rather than exposing log10 exponents.
        """
        arr = np.asarray(array, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"ArrayViewer needs a 2D array, got shape {arr.shape}.")
        self._array = arr
        self._extent = extent
        self._log_display = bool(log)
        self._value_label = str(value_label or "Value")
        self._hist.axis.setLabel(self._value_label)
        self._hist.axis.setLogMode(self._log_display)
        self._plot.setTitle(str(title or ""))
        self._readout.setText(f"x: -, y: -, {self._value_label}: -")
        if invert_y is not None:
            self._plot.invertY(bool(invert_y))
        if x_label is not None:
            self._plot.setLabel("bottom", x_label)
        if y_label is not None:
            self._plot.setLabel("left", y_label)

        display = arr
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                display = np.log10(np.where(arr > 0, arr, np.nan))
        # setRect leaves a transform on the ImageItem.  Reset it when a later
        # dataset goes back to pixel coordinates, otherwise axes and hover
        # locations silently retain the previous dataset's physical extent.
        self._img.resetTransform()
        self._img.setImage(display, autoLevels=False)
        if extent is not None:
            x0, x1, z0, z1 = (float(v) for v in extent)
            self._img.setRect(QRectF(x0, z0, x1 - x0, z1 - z0))

        if autoscale:
            finite = display[np.isfinite(display)]
            if finite.size:
                lo, hi = np.percentile(finite, [2, 98])
                if hi <= lo:
                    half_span = 0.5 if log else max(abs(float(lo)) * 0.05, 0.5)
                    lo, hi = float(lo) - half_span, float(hi) + half_span
                self._img.setLevels((lo, hi))
                self._hist.setLevels(lo, hi)
            else:
                self._img.setLevels((0.0, 1.0))
                self._hist.setLevels(0.0, 1.0)
            self._plot.autoRange()

    def set_levels(self, lo: float, hi: float) -> None:
        self._img.setLevels((lo, hi))
        self._hist.setLevels(lo, hi)

    def set_colormap(self, name: str) -> None:
        try:
            self._hist.gradient.loadPreset(name)
        except Exception:
            try:
                cmap = pg.colormap.get(name)  # matplotlib / colorcet name (e.g. "turbo")
                if cmap is not None:
                    self._hist.gradient.setColorMap(cmap)
            except Exception:
                pass

    # -- interaction modes ---------------------------------------------------
    def set_pick_mode(self, enabled: bool) -> None:
        self._pick_mode = bool(enabled)
        if enabled:
            self._profile_mode = False

    def set_profile_mode(self, enabled: bool) -> None:
        self._profile_mode = bool(enabled)
        if enabled:
            self._pick_mode = False

    # -- markers / profile ---------------------------------------------------
    def add_marker(self, col: float, row: float) -> None:
        self._markers.append((float(col), float(row)))
        self._refresh_markers()

    def remove_last_marker(self) -> None:
        if self._markers:
            self._markers.pop()
            self._refresh_markers()

    def clear_markers(self) -> None:
        self._markers.clear()
        self._refresh_markers()

    def markers(self) -> List[Tuple[float, float]]:
        return list(self._markers)

    def clear_profile(self) -> None:
        self._profile_pts = []
        self._profile_scatter.setData([], [])
        self._profile_line.setData([], [])

    def set_profile_points(self, p1: List[float], p2: List[float]) -> None:
        self._profile_pts = [list(map(float, p1)), list(map(float, p2))]
        self._refresh_profile()

    def _refresh_markers(self) -> None:
        if self._markers:
            xs, ys = zip(*self._markers)
            self._marker_scatter.setData(list(xs), list(ys))
        else:
            self._marker_scatter.setData([], [])

    def _refresh_profile(self) -> None:
        if not self._profile_pts:
            self._profile_scatter.setData([], [])
            self._profile_line.setData([], [])
            return
        xs = [p[0] for p in self._profile_pts]
        ys = [p[1] for p in self._profile_pts]
        self._profile_scatter.setData(xs, ys)
        if len(self._profile_pts) == 2:
            self._profile_line.setData(xs, ys)
        else:
            self._profile_line.setData([], [])

    # -- mouse ---------------------------------------------------------------
    def _scene_to_array(self, scene_pos):
        """Return ``(col, row, value, x, z)`` under the cursor, or ``None``.

        ``x``/``z`` are data-space coordinates (real units when an ``extent`` is
        set, otherwise column/row); ``col``/``row`` are always array indices.
        """
        if self._array is None:
            return None
        view_pt = self._plot.vb.mapSceneToView(scene_pos)
        x, z = float(view_pt.x()), float(view_pt.y())
        n_rows, n_cols = self._array.shape
        if self._extent is not None:
            x0, x1, z0, z1 = self._extent
            col = int(np.floor((x - x0) / (x1 - x0) * n_cols)) if x1 != x0 else 0
            row = int(np.floor((z - z0) / (z1 - z0) * n_rows)) if z1 != z0 else 0
        else:
            col, row = int(np.floor(x)), int(np.floor(z))
        if 0 <= row < n_rows and 0 <= col < n_cols:
            return col, row, float(self._array[row, col]), x, z
        return None

    def _on_mouse_moved(self, scene_pos) -> None:
        hit = self._scene_to_array(scene_pos)
        if hit is None:
            self._readout.setText(f"x: -, y: -, {self._value_label}: -")
            return
        col, row, value, x, z = hit
        reading = f"{self._value_label}: {value:.4g}"
        if self._extent is not None:
            self._readout.setText(f"x: {x:.1f}, z: {z:.1f}, {reading}")
        else:
            self._readout.setText(f"x(col): {col}, y(row): {row}, {reading}")

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        hit = self._scene_to_array(event.scenePos())
        if hit is None:
            return
        col, row, value, _x, _z = hit
        if self._pick_mode:
            self.add_marker(col, row)
            self.pointPicked.emit(float(col), float(row), float(value))
        elif self._profile_mode:
            if len(self._profile_pts) >= 2:
                self._profile_pts = []
            self._profile_pts.append([float(col), float(row)])
            self._refresh_profile()
            if len(self._profile_pts) == 2:
                self.profileSelected.emit(self._profile_pts[0], self._profile_pts[1])

    # -- export --------------------------------------------------------------
    def export_png(self, path: str) -> None:
        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(self._plot)
        exporter.export(str(path))

    def _export_png_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export view as PNG", "view.png", "PNG (*.png)")
        if path:
            self.export_png(path)
