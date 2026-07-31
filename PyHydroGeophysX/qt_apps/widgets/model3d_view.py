"""Reusable 3D viewers for regular arrays and exported VTK volumes.

``Model3DView`` accepts regular array grids used by inversion modules. When
PyVista + a GL context are available it shows an interactive volume with a
draggable clip plane (rotate + slice through it). Otherwise it falls back to a
matplotlib panel with a plan-view depth slice and a vertical cross-section, each
driven by a slider, so the model is still viewable "at different positions and
depths" headless or without a GPU.

``VTKVolumeView`` displays an exported structured/unstructured VTK volume
directly inside a module, with an optional draggable clipping plane.

Feed it a regular grid: ``show_model((edges_x, edges_y, edges_z), model3d, ...)``
where each ``edges_*`` is a 1D array of cell edges (length n+1) and ``model3d`` has
shape ``(nx, ny, nz)`` with ``z`` = elevation (increasing upward).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from PyHydroGeophysX.visualization.pyvista_compat import try_import_pyvista


class VTKVolumeView(QWidget):
    """Interactive viewer for a VTK dataset exported by another workflow."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plotter = None
        self._mesh = None
        self._scalar: Optional[str] = None
        self._cmap = "turbo"
        self._opacity = 0.65

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._info = QLabel(
            "Build the model to inspect its 3D velocity volume here."
        )
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        ok, pv, qt_interactor, err = try_import_pyvista()
        self._pv = pv if ok else None
        self._clip_cb = QCheckBox("Clip plane (drag to inspect the interior)")
        if ok:
            try:
                self._plotter = qt_interactor(self)
                self._plotter.set_background("white")
                self._plotter.add_axes()
                controls = QHBoxLayout()
                self._clip_cb.toggled.connect(self._redraw)
                reset = QPushButton("Reset view")
                reset.clicked.connect(self._reset_camera)
                controls.addWidget(self._clip_cb)
                controls.addWidget(reset)
                controls.addStretch(1)
                layout.addLayout(controls)
                layout.addWidget(self._plotter.interactor, stretch=1)
            except Exception as exc:  # noqa: BLE001 - GL failure -> clean fallback
                self._plotter = None
                err = str(exc)
        if self._plotter is None:
            self._notice = QLabel(
                "Interactive 3D view is unavailable in this session.<br>"
                f"<code>{err}</code><br><br>"
                "If pyvista is missing, install the 3D viewers with "
                "<code>pip install \"pyhydrogeophysx[desktop-3d]\"</code>. Check "
                "<code>conda list numpy</code> first: a conda channel there means "
                "this environment's packages are conda-managed, so use "
                "<code>conda install -c conda-forge pyvista pyvistaqt</code> "
                "instead.<br><br>"
                "The generated figures and files remain available in the next tab."
            )
            self._notice.setWordWrap(True)
            self._notice.setAlignment(Qt.AlignCenter)
            self._notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self._notice, stretch=1)
        else:
            self._notice = None

    @property
    def interactive_available(self) -> bool:
        """Whether the embedded PyVistaQt renderer is available."""
        return self._plotter is not None and self._pv is not None

    def show_file(
        self,
        path: str,
        *,
        scalar_candidates: Sequence[str] = ("Velocity", "velocity"),
        cmap: str = "turbo",
        opacity: float = 0.65,
    ) -> bool:
        """Load and display a VTK dataset, returning whether it was rendered."""
        vtk_path = Path(path)
        if not vtk_path.is_file():
            self._info.setText(f"3D volume file not found: <code>{vtk_path}</code>")
            return False
        if not self.interactive_available:
            self._info.setText(
                f"3D volume saved at <code>{vtk_path}</code>. "
                "Use the Figures & files tab in this session."
            )
            return False
        try:
            self._mesh = self._pv.read(str(vtk_path))
            self._scalar = next(
                (
                    name for name in scalar_candidates
                    if name in getattr(self._mesh, "point_data", {})
                    or name in getattr(self._mesh, "cell_data", {})
                ),
                None,
            )
            self._cmap = str(cmap)
            self._opacity = float(opacity)
            self._info.setText(
                f"<b>{vtk_path.name}</b>  ·  "
                f"{getattr(self._mesh, 'n_points', 0)} points"
                + (f"  ·  scalar: {self._scalar}" if self._scalar else "")
                + "  ·  drag to rotate, wheel to zoom"
            )
            self._redraw()
            return True
        except Exception as exc:  # noqa: BLE001 - VTK backend-specific failures
            self._info.setText(f"Could not display <code>{vtk_path.name}</code>: {exc}")
            return False

    def _reset_camera(self) -> None:
        if self._plotter is not None:
            self._plotter.reset_camera()
            self._refresh()

    def _redraw(self, *_) -> None:
        if not self.interactive_available or self._mesh is None:
            return
        try:
            self._plotter.clear_plane_widgets()
        except Exception:  # noqa: BLE001 - absent on older PyVista releases
            pass
        self._plotter.clear()
        kwargs = {
            "scalars": self._scalar,
            "cmap": self._cmap,
            "opacity": self._opacity,
            "show_edges": False,
            "show_scalar_bar": bool(self._scalar),
        }
        try:
            if self._clip_cb.isChecked():
                self._plotter.add_mesh_clip_plane(self._mesh, **kwargs)
            else:
                self._plotter.add_mesh(self._mesh, **kwargs)
        except Exception:  # noqa: BLE001 - clip widgets can fail on some VTK builds
            self._plotter.add_mesh(self._mesh, **kwargs)
        try:
            self._plotter.add_mesh(self._mesh.outline(), color="grey")
        except Exception:  # noqa: BLE001 - outline is cosmetic
            pass
        self._plotter.add_axes()
        self._plotter.reset_camera()
        self._refresh()
        QTimer.singleShot(0, self._refresh)

    def _refresh(self) -> None:
        if self._plotter is None:
            return
        try:
            interactor = getattr(self._plotter, "interactor", None)
            if interactor is not None:
                interactor.raise_()
                interactor.update()
            self._plotter.render()
        except Exception:  # noqa: BLE001 - refresh is best-effort
            pass


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
