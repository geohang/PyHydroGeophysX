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

import numpy as np

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.coalesce import Coalesced
from PyHydroGeophysX.visualization.axis_units import (
    set_length_axis, set_section_axes, to_display_length)
from PyHydroGeophysX.visualization.pyvista_compat import try_import_pyvista


def _recolour_actors(actors, name: str) -> int:
    """Give every colour-mapped PyVista actor ``name``; return how many took it.

    The lookup table is swapped in place, so the camera, a dragged clip plane
    and the scalar bar all stay as the user left them - nothing is re-added.
    """
    changed = 0
    for actor in actors or ():
        try:
            actor.mapper.lookup_table.cmap = cmaps.to_pyvista(name)
            changed += 1
        except Exception:  # noqa: BLE001 - an actor without a table is left alone
            pass
    return changed


class VTKVolumeView(QWidget):
    """Interactive viewer for a VTK dataset exported by another workflow.

    ``colormaps`` is the studio state's shared colormap dict and
    ``colormap_key`` names what the volume is (a velocity model, say), so its
    colour map is the one chosen for that quantity everywhere else.
    """

    def __init__(self, parent=None, *, colormaps=None,
                 colormap_key: str = cmaps.MODEL3D) -> None:
        super().__init__(parent)
        self._plotter = None
        self._mesh = None
        self._scalar: Optional[str] = None
        self._cmap = "turbo"
        self._opacity = 0.65
        self._base_colormap_key = colormap_key
        self._scalar_cmaps = {}
        self._default_cmap = 'turbo'
        self._actors: list = []   # the colour-mapped actors now on screen
        self._field = QComboBox(self)
        self._field.setToolTip('Physical property or categorical labels to display')
        self._field.currentTextChanged.connect(self._on_field_changed)
        self._colormap = cmaps.ColormapChooser(
            colormap_key, self._cmap, shared=colormaps, parent=self)
        self._colormap.colormapChanged.connect(self._on_colormap_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._info = QLabel(
            "Open a model to inspect its 3D properties here."
        )
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        ok, pv, qt_interactor, err = try_import_pyvista()
        self._pv = pv if ok else None
        self._clip_cb = QCheckBox("Clip plane (drag to inspect the interior)")
        if ok:
            try:
                # Rendered when something changes, not on a timer: pyvistaqt
                # re-renders five times a second by default, on the UI thread
                # and whether the view is on screen or not.
                self._plotter = qt_interactor(self, auto_update=False)
                self._plotter.set_background("white")
                self._plotter.add_axes()
                controls = QHBoxLayout()
                self._clip_cb.toggled.connect(self._redraw)
                reset = QPushButton("Reset view")
                reset.clicked.connect(self._reset_camera)
                controls.addWidget(self._clip_cb)
                controls.addWidget(reset)
                controls.addWidget(self._field)
                controls.addWidget(self._colormap)
                controls.addStretch(1)
                layout.addLayout(controls)
                layout.addWidget(self._plotter.interactor, stretch=1)
            except Exception as exc:  # noqa: BLE001 - GL failure -> clean fallback
                self._plotter = None
                err = str(exc)
        if self._plotter is None:
            self._field.hide()
            self._colormap.hide()   # nothing here to colour
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
        scalar_cmaps: Optional[dict] = None,
    ) -> bool:
        """Load and display a VTK dataset, returning whether it was rendered.

        ``cmap`` is the volume's default colour map: what it is drawn with until
        a map is chosen for this view's quantity.
        """
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
            self._source_name = vtk_path.name
            fields = list(dict.fromkeys(
                name for data in (self._mesh.point_data, self._mesh.cell_data)
                for name in data if np.asarray(data[name]).ndim == 1
                and np.issubdtype(np.asarray(data[name]).dtype, np.number)))
            self._scalar = next((name for name in scalar_candidates if name in fields),
                                fields[0] if fields else None)
            self._field.blockSignals(True)
            self._field.clear()
            self._field.addItems(fields)
            self._field.setCurrentText(self._scalar or '')
            self._field.blockSignals(False)
            self._scalar_cmaps = dict(scalar_cmaps or {})
            self._default_cmap = str(cmap)
            self._choose_field_colormap()
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
            "cmap": cmaps.to_pyvista(self._cmap),
            "opacity": self._opacity,
            "show_edges": False,
            "show_scalar_bar": bool(self._scalar),
            "scalar_bar_args": {
                "title": self._scalar or "",
                "title_font_size": 14,
                "label_font_size": 12,
                "fmt": "%.3g",
            },
        }
        categorical = bool(self._scalar and self._scalar.lower().endswith(' id'))
        self._colormap.setEnabled(not categorical)
        if categorical:
            # Map sparse IDs to compact colour positions on a plotting array;
            # the dataset and its original IDs remain unchanged.
            values = np.asarray(self._mesh[self._scalar])
            labels, indices = np.unique(values, return_inverse=True)
            kwargs.update(scalars=indices, cmap='tab20', n_colors=max(1, len(labels)),
                          clim=(-0.5, len(labels) - 0.5),
                          annotations={float(i): str(v) for i, v in enumerate(labels)})
            kwargs['scalar_bar_args']['n_labels'] = 0
        try:
            if self._clip_cb.isChecked():
                actor = self._plotter.add_mesh_clip_plane(self._mesh, **kwargs)
            else:
                actor = self._plotter.add_mesh(self._mesh, **kwargs)
        except Exception:  # noqa: BLE001 - clip widgets can fail on some VTK builds
            actor = self._plotter.add_mesh(self._mesh, **kwargs)
        self._actors = [actor] if self._scalar else []
        try:
            self._plotter.add_mesh(self._mesh.outline(), color="grey")
        except Exception:  # noqa: BLE001 - outline is cosmetic
            pass
        self._plotter.add_axes()
        self._plotter.reset_camera()
        self._refresh()
        QTimer.singleShot(0, self._refresh)

    def _on_field_changed(self, name: str) -> None:
        if not name or self._mesh is None:
            return
        self._scalar = name
        self._info.setText(f'<b>{self._source_name}</b> · scalar: {name} · drag to rotate, wheel to zoom')
        self._choose_field_colormap()
        camera = self._plotter.camera_position if self._plotter is not None else None
        self._redraw()
        if camera is not None:
            self._plotter.camera_position = camera
            self._refresh()

    def _choose_field_colormap(self) -> None:
        default = self._scalar_cmaps.get(self._scalar, self._default_cmap)
        key = self._base_colormap_key
        if self._scalar_cmaps and self._scalar:
            key = f'{key}:{self._scalar}'
        self._cmap = self._colormap.set_target(key, default)

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

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Render once the view is on screen, which the timer used to see to."""
        super().showEvent(event)
        if self._plotter is not None:
            QTimer.singleShot(0, self._refresh)

    @property
    def colormap_chooser(self) -> "cmaps.ColormapChooser":
        return self._colormap

    def _on_colormap_changed(self, name: str) -> None:
        """Recolour the volume on screen, keeping the camera and the clip plane."""
        self._cmap = name
        if _recolour_actors(self._actors, name):
            self._refresh()


class Model3DView(QWidget):
    """A regular 3-D grid: a PyVista volume, or matplotlib slices without one.

    ``colormaps`` is the studio state's shared colormap dict and
    ``colormap_key`` names the quantity shown; a page showing different
    quantities in turn passes ``colormap_key`` to :meth:`show_model`.
    """

    def __init__(self, parent=None, *, colormaps=None,
                 colormap_key: str = cmaps.MODEL3D) -> None:
        super().__init__(parent)
        self._edges: Optional[Tuple] = None
        self._model = None
        self._label = "value"
        self._cmap = "turbo"
        self._log = False
        self._plotter = None
        self._actors: list = []   # the colour-mapped actors now on screen
        self._colormap = cmaps.ColormapChooser(
            colormap_key, self._cmap, shared=colormaps, parent=self)
        self._colormap.colormapChanged.connect(self._on_colormap_changed)
        # A page can switch the quantity per model, so the choice waits for one.
        self._colormap.setEnabled(False)

        ok, pv, qt_interactor, err = try_import_pyvista()
        self._pv = pv if ok else None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._mode = "mpl"
        if ok:
            try:
                # Rendered on change, not on pyvistaqt's timer (as VTKVolumeView).
                self._plotter = qt_interactor(self, auto_update=False)
                self._plotter.set_background("white")
                self._plotter.add_axes()
                bar = QHBoxLayout()
                self._clip_cb = QCheckBox("Clip plane (drag to slice)")
                self._clip_cb.setChecked(True)
                self._clip_cb.toggled.connect(self._redraw_pv)
                reset = QPushButton("Reset view")
                reset.clicked.connect(lambda: self._plotter and self._plotter.reset_camera())
                bar.addWidget(self._clip_cb); bar.addWidget(reset)
                bar.addWidget(self._colormap); bar.addStretch(1)
                layout.addLayout(bar)
                layout.addWidget(self._plotter.interactor, stretch=1)
                self._mode = "pyvista"
            except Exception as exc:  # noqa: BLE001 - GL failure -> matplotlib fallback
                self._plotter = None
                self._build_mpl(layout)
        else:
            self._build_mpl(layout)
        # View > Length Units. Only the matplotlib panels have length axes; the
        # PyVista volume carries an orientation marker and no scale.
        length_units.notifier().changed.connect(self._on_length_unit_changed)

    def _on_length_unit_changed(self, _unit: str) -> None:
        """Redraw the slices in the studio's new length unit."""
        if self._mode == "mpl" and self._model is not None:
            self._redraw_mpl()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Render once the view is on screen, which the timer used to see to."""
        super().showEvent(event)
        if self._plotter is not None:
            QTimer.singleShot(0, self._render)

    def _render(self) -> None:
        try:
            self._plotter.render()
        except Exception:  # noqa: BLE001 - a repaint is best-effort
            pass

    # -- matplotlib fallback -------------------------------------------------
    def _build_mpl(self, layout: QVBoxLayout) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(7.5, 3.8), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas, stretch=1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Depth"))
        # A slider moved puts the new slice into the images already drawn, once
        # per turn of the event loop: redrawing both panels, their colour bars
        # and the model's percentiles per mouse move is what made a drag stutter.
        self._slice_moved = Coalesced(lambda: self._show_slices(), self)
        self._mpl_drawn = None
        self._mpl_scale = None
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.valueChanged.connect(self._slice_moved.request)
        row.addWidget(self._z_slider, stretch=1)
        row.addWidget(QLabel("Y position"))
        self._y_slider = QSlider(Qt.Horizontal)
        self._y_slider.valueChanged.connect(self._slice_moved.request)
        row.addWidget(self._y_slider, stretch=1)
        row.addWidget(self._colormap)
        layout.addLayout(row)

    # -- public --------------------------------------------------------------
    def show_model(self, edges: Sequence, model3d, *, label: str = "value",
                   cmap: str = "turbo", log_scale: bool = False,
                   colormap_key: Optional[str] = None) -> None:
        """Show a model on a regular grid.

        ``cmap`` is its default colour map, used until one is chosen for
        ``colormap_key`` - the quantity shown, when the page shows several.
        """
        import numpy as np
        parsed_edges = tuple(np.asarray(e, dtype=float) for e in edges)
        parsed_model = np.asarray(model3d, dtype=float)
        if len(parsed_edges) != 3 or parsed_model.ndim != 3:
            raise ValueError("A 3D model needs three edge arrays and shape (nx, ny, nz).")
        expected = tuple(int(edge.size - 1) for edge in parsed_edges)
        if expected != parsed_model.shape:
            raise ValueError(
                f"Model shape {parsed_model.shape} does not match edge-defined shape {expected}."
            )
        self._edges = parsed_edges
        self._model = parsed_model
        self._label = label
        self._cmap = self._colormap.set_target(
            colormap_key or self._colormap.key(), str(cmap))
        self._colormap.setEnabled(True)
        self._log = bool(log_scale)
        if self._mode == "pyvista":
            self._redraw_pv()
        else:
            self._mpl_drawn = None
            self._mpl_scale = None
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
        self._plotter.clear()
        self._actors = []
        values = self._model
        if self._log:
            values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
            if not np.isfinite(values).any():
                self._plotter.add_text(
                    "No positive finite values for logarithmic display.",
                    position="upper_left",
                )
                self._plotter.add_axes()
                return
        grid.cell_data[self._label] = values.flatten(order="F")
        kw = dict(scalars=self._label, cmap=cmaps.to_pyvista(self._cmap), log_scale=self._log,
                  show_edges=False, scalar_bar_args={"title": self._label})
        try:
            if self._clip_cb.isChecked():
                actor = self._plotter.add_mesh_clip_plane(grid, **kw)
            else:
                actor = self._plotter.add_mesh(grid, **kw)
        except Exception:  # noqa: BLE001 - clip widget can fail; show the plain volume
            actor = self._plotter.add_mesh(grid, **kw)
        self._actors = [actor]
        try:
            self._plotter.add_mesh(grid.outline(), color="grey")
        except Exception:  # noqa: BLE001
            pass
        self._plotter.add_axes()
        self._plotter.reset_camera()

    def _mpl_norm(self):
        """A colour norm over the whole model, or None when it has nothing to show.

        The limits are computed once per model; the norm is new each call,
        because the images drawn with one listen to it and would outlive the
        figure they were cleared from if it were shared.
        """
        from matplotlib.colors import LogNorm, Normalize

        if self._mpl_scale is None:
            m = self._model
            finite_mask = np.isfinite(m)
            if self._log:
                finite_mask &= m > 0
            finite = m[finite_mask]
            if finite.size == 0:
                self._mpl_scale = ()
            else:
                vmin = float(np.nanpercentile(finite, 2))
                vmax = float(np.nanpercentile(finite, 98))
                if self._log:
                    vmin = max(vmin, 1e-12)
                    vmax = max(vmax, vmin * 1.01)
                elif not vmax > vmin:
                    vmax = vmin + 1.0
                self._mpl_scale = (vmin, vmax)
        if not self._mpl_scale:
            return None
        return (LogNorm if self._log else Normalize)(*self._mpl_scale)

    def _slice_indices(self):
        import numpy as np

        _, ny, nz = self._model.shape
        return (int(np.clip(self._z_slider.value(), 0, nz - 1)),
                int(np.clip(self._y_slider.value(), 0, ny - 1)))

    def _show_slices(self) -> None:
        """Put the slices the sliders select into the panels already drawn."""
        drawn = self._mpl_drawn
        if self._model is None:
            return
        if drawn is None or drawn.get("depth") is None:
            self._redraw_mpl()
            return
        zi, yj = self._slice_indices()
        drawn["depth"].set_array(self._model[:, :, zi].T)
        drawn["depth"].axes.set_title(self._slice_title("Depth slice", "z", drawn["zc"][zi]))
        drawn["section"].set_array(self._model[:, yj, :].T)
        drawn["section"].axes.set_title(self._slice_title("Cross-section", "y", drawn["yc"][yj]))
        self._canvas.draw_idle()

    @staticmethod
    def _slice_title(kind: str, axis: str, position: float) -> str:
        """``'Depth slice  z = -12 m'``: where a slice is, in the studio's unit."""
        return f"{kind}  {axis} = {to_display_length(position):.0f} {length_units.current()}"

    def _recolour_mpl(self) -> bool:
        """Give the panels on screen the colour map now chosen; False if none are."""
        drawn = self._mpl_drawn
        if drawn is None or not drawn.get("images"):
            return False
        cmap = cmaps.to_matplotlib(self._cmap)
        for image, bar in drawn["images"]:
            image.set_cmap(cmap)
            bar.update_normal(image)
        self._canvas.draw_idle()
        return True

    def _redraw_mpl(self, *_) -> None:
        if self._model is None:
            return
        ex, ey, ez = self._edges
        m = self._model
        cmap = cmaps.to_matplotlib(self._cmap)
        _, ny, nz = m.shape
        zi, yj = self._slice_indices()
        norm = self._mpl_norm()
        self._mpl_drawn = None
        self._fig.clear()
        if norm is None:
            ax = self._fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No positive finite values for logarithmic display."
                if self._log else "No finite model values to display.",
                ha="center", va="center", transform=ax.transAxes, wrap=True,
            )
            ax.axis("off")
            self._canvas.draw_idle()
            return
        zc = 0.5 * (ez[:-1] + ez[1:])
        yc = 0.5 * (ey[:-1] + ey[1:])
        if ny == 1:
            # 2D section: position along the line (x) vs elevation/depth (z).
            ax = self._fig.add_subplot(111)
            im = ax.pcolormesh(ex, ez, m[:, 0, :].T, cmap=cmap, norm=norm, shading="auto")
            field_name = self._label.strip() or "Model"
            ax.set_title(
                "Resistivity section"
                if "resist" in field_name.lower() else f"{field_name} section"
            )
            # A line whose soundings carry no elevations hangs from z = 0 and
            # reads as depth.
            set_section_axes(ax, z=ez, xlabel="position along line",
                             elevation_name="elevation", depth_name="depth")
            bar = self._fig.colorbar(im, ax=ax, shrink=0.85, label=self._label)
            self._mpl_drawn = {"images": [(im, bar)]}
            self._canvas.draw_idle()
            return
        ax1 = self._fig.add_subplot(121)
        ax2 = self._fig.add_subplot(122)
        im1 = ax1.pcolormesh(ex, ey, m[:, :, zi].T, cmap=cmap, norm=norm, shading="auto")
        ax1.set_title(self._slice_title("Depth slice", "z", zc[zi]))
        set_length_axis(ax1, "x", "x"); set_length_axis(ax1, "y", "y")
        ax1.set_aspect("equal", "box")
        bar1 = self._fig.colorbar(im1, ax=ax1, shrink=0.85, label=self._label)
        im2 = ax2.pcolormesh(ex, ez, m[:, yj, :].T, cmap=cmap, norm=norm, shading="auto")
        ax2.set_title(self._slice_title("Cross-section", "y", yc[yj]))
        set_section_axes(ax2, z=ez, xlabel="x",
                         elevation_name="elevation", depth_name="depth")
        bar2 = self._fig.colorbar(im2, ax=ax2, shrink=0.85, label=self._label)
        self._mpl_drawn = {"images": [(im1, bar1), (im2, bar2)], "depth": im1,
                           "section": im2, "zc": zc, "yc": yc}
        self._canvas.draw_idle()

    # -- colour map ------------------------------------------------------------
    @property
    def colormap_chooser(self) -> "cmaps.ColormapChooser":
        return self._colormap

    def _on_colormap_changed(self, name: str) -> None:
        """Recolour the model on screen from the grid already held.

        The PyVista volume swaps its lookup table in place, keeping the camera
        and the clip plane; the matplotlib slices take the new map in place too,
        at the same depth and position.
        """
        self._cmap = name
        if self._model is None:
            return
        if self._mode == "pyvista":
            if _recolour_actors(self._actors, name) and self._plotter is not None:
                try:
                    self._plotter.render()
                except Exception:  # noqa: BLE001 - a repaint is best-effort
                    pass
        elif not self._recolour_mpl():
            self._redraw_mpl()
