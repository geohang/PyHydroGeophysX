"""A 3D model viewer for inverted volumes (density, susceptibility, resistivity).

When PyVista + a GL context are available it shows an interactive volume with a
draggable clip plane (rotate + slice through it). Otherwise it falls back to a
matplotlib panel with a plan-view depth slice and a vertical cross-section, each
driven by a slider, so the model is still viewable "at different positions and
depths" headless or without a GPU.

Feed it a regular grid: ``show_model((edges_x, edges_y, edges_z), model3d, ...)``
where each ``edges_*`` is a 1D array of cell edges (length n+1) and ``model3d`` has
shape ``(nx, ny, nz)`` with ``z`` = elevation (increasing upward).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from PyHydroGeophysX.qt_apps.pv_utils import try_import_pyvista


class Model3DView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._edges: Optional[Tuple] = None
        self._model = None
        self._label = "value"
        self._cmap = "turbo"
        self._log = False
        self._plotter = None

        ok, pv, qt_interactor, err = try_import_pyvista()
        self._pv = pv if ok else None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._mode = "mpl"
        if ok:
            try:
                self._plotter = qt_interactor(self)
                self._plotter.set_background("white")
                self._plotter.add_axes()
                bar = QHBoxLayout()
                self._clip_cb = QCheckBox("Clip plane (drag to slice)")
                self._clip_cb.setChecked(True)
                self._clip_cb.toggled.connect(self._redraw_pv)
                reset = QPushButton("Reset view")
                reset.clicked.connect(lambda: self._plotter and self._plotter.reset_camera())
                bar.addWidget(self._clip_cb); bar.addWidget(reset); bar.addStretch(1)
                layout.addLayout(bar)
                layout.addWidget(self._plotter.interactor, stretch=1)
                self._mode = "pyvista"
            except Exception as exc:  # noqa: BLE001 - GL failure -> matplotlib fallback
                self._plotter = None
                self._build_mpl(layout)
        else:
            self._build_mpl(layout)

    # -- matplotlib fallback -------------------------------------------------
    def _build_mpl(self, layout: QVBoxLayout) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(7.5, 3.8), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas, stretch=1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Depth"))
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.valueChanged.connect(self._redraw_mpl)
        row.addWidget(self._z_slider, stretch=1)
        row.addWidget(QLabel("Y position"))
        self._y_slider = QSlider(Qt.Horizontal)
        self._y_slider.valueChanged.connect(self._redraw_mpl)
        row.addWidget(self._y_slider, stretch=1)
        layout.addLayout(row)

    # -- public --------------------------------------------------------------
    def show_model(self, edges: Sequence, model3d, *, label: str = "value",
                   cmap: str = "turbo", log_scale: bool = False) -> None:
        import numpy as np
        self._edges = tuple(np.asarray(e, dtype=float) for e in edges)
        self._model = np.asarray(model3d, dtype=float)
        self._label = label
        self._cmap = cmap
        self._log = bool(log_scale)
        if self._mode == "pyvista":
            self._redraw_pv()
        else:
            _, ny, nz = self._model.shape
            for slider, n, default in ((self._z_slider, nz, nz - 1), (self._y_slider, ny, ny // 2)):
                slider.blockSignals(True)
                slider.setRange(0, max(0, n - 1))
                slider.setValue(max(0, default))
                slider.blockSignals(False)
            # A single-row model (ny == 1) is a 2D section (position x depth) — no sliders.
            self._z_slider.setVisible(ny > 1)
            self._y_slider.setVisible(ny > 1)
            self._redraw_mpl()

    # -- renderers -----------------------------------------------------------
    def _redraw_pv(self, *_) -> None:
        if self._plotter is None or self._pv is None or self._model is None:
            return
        pv = self._pv
        ex, ey, ez = self._edges
        grid = pv.RectilinearGrid(ex, ey, ez)
        grid.cell_data[self._label] = self._model.flatten(order="F")
        self._plotter.clear()
        kw = dict(scalars=self._label, cmap=self._cmap, log_scale=self._log,
                  show_edges=False, scalar_bar_args={"title": self._label})
        try:
            if self._clip_cb.isChecked():
                self._plotter.add_mesh_clip_plane(grid, **kw)
            else:
                self._plotter.add_mesh(grid, **kw)
        except Exception:  # noqa: BLE001 - clip widget can fail; show the plain volume
            self._plotter.add_mesh(grid, **kw)
        try:
            self._plotter.add_mesh(grid.outline(), color="grey")
        except Exception:  # noqa: BLE001
            pass
        self._plotter.add_axes()
        self._plotter.reset_camera()

    def _redraw_mpl(self, *_) -> None:
        import numpy as np
        from matplotlib.colors import LogNorm, Normalize
        if self._model is None:
            return
        ex, ey, ez = self._edges
        m = self._model
        _, ny, nz = m.shape
        zi = int(np.clip(self._z_slider.value(), 0, nz - 1))
        yj = int(np.clip(self._y_slider.value(), 0, ny - 1))
        finite = m[np.isfinite(m)]
        vmin = float(np.nanpercentile(finite, 2)) if finite.size else 0.0
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
        if self._log:
            norm = LogNorm(max(vmin, 1e-9), max(vmax, 1e-9))
        else:
            norm = Normalize(vmin, vmax if vmax > vmin else vmin + 1.0)
        zc = 0.5 * (ez[:-1] + ez[1:])
        yc = 0.5 * (ey[:-1] + ey[1:])
        self._fig.clear()
        if ny == 1:
            # 2D section: position along the line (x) vs elevation/depth (z).
            ax = self._fig.add_subplot(111)
            im = ax.pcolormesh(ex, ez, m[:, 0, :].T, cmap=self._cmap, norm=norm, shading="auto")
            ax.set_title("Resistivity section")
            ax.set_xlabel("position along line (m)"); ax.set_ylabel("elevation (m)")
            self._fig.colorbar(im, ax=ax, shrink=0.85, label=self._label)
            self._canvas.draw_idle()
            return
        ax1 = self._fig.add_subplot(121)
        ax2 = self._fig.add_subplot(122)
        im1 = ax1.pcolormesh(ex, ey, m[:, :, zi].T, cmap=self._cmap, norm=norm, shading="auto")
        ax1.set_title(f"Depth slice  z = {zc[zi]:.0f}")
        ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_aspect("equal", "box")
        self._fig.colorbar(im1, ax=ax1, shrink=0.85, label=self._label)
        im2 = ax2.pcolormesh(ex, ez, m[:, yj, :].T, cmap=self._cmap, norm=norm, shading="auto")
        ax2.set_title(f"Cross-section  y = {yc[yj]:.0f}")
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("elevation (m)")
        self._fig.colorbar(im2, ax=ax2, shrink=0.85, label=self._label)
        self._canvas.draw_idle()
