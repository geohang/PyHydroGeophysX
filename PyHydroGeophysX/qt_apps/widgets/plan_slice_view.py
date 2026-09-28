"""Plan-view (map) depth-slice viewer for stitched EM soundings.

Given each sounding's map coordinate ``(x, y)`` and its recovered resistivity per
layer, this shows a horizontal-plane map of resistivity at a depth the user picks
with a slider. When the soundings lie on a 2D spread it interpolates a filled map
(and overlays the sounding points); when they are collinear (a single flight
line) it shows the points coloured by resistivity along the line. NaN cells
(below the depth of investigation) are simply not drawn.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets.coalesce import Coalesced


class PlanSliceView(QWidget):
    """Resistivity depth slices in plan view.

    ``colormaps`` is the studio state's shared colormap dict; the slices share
    the EM sections' colour map unless another ``colormap_key`` is given.
    """

    def __init__(self, parent=None, *, colormaps=None,
                 colormap_key: str = cmaps.EM_SECTION) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(6.2, 5.0), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas, stretch=1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Depth"))
        self._z = QSlider(Qt.Horizontal)
        # A drag emits a value per mouse move, and each used to rebuild the map
        # from scratch. Now each one relabels the depth at once and asks for a
        # drawing; the requests of one turn of the event loop make one, and
        # while the handle is held that one is a preview - the soundings
        # recoloured on the map already drawn. The interpolated surface is drawn
        # once the handle is let go.
        self._z.valueChanged.connect(self._on_depth_changed)
        self._z.sliderReleased.connect(self._request_full)
        row.addWidget(self._z, stretch=1)
        self._z_label = QLabel("—")
        self._z_label.setMinimumWidth(56)
        row.addWidget(self._z_label)
        # The colour map, beside the depth it is read at; turbo until chosen.
        self._colormap = cmaps.ColormapChooser(colormap_key, "turbo", shared=colormaps)
        self._colormap.colormapChanged.connect(lambda _name: self._request_full())
        row.addWidget(self._colormap)
        layout.addLayout(row)

        self._xy = None       # (n_pos, 2) sounding map coordinates
        self._res = None      # (n_pos, n_depth) resistivity, surface-ordered in depth
        self._depths = None   # (n_depth,) depth centres (m, positive down)
        self._label = "value"
        self._log = True
        self._x_label = "Easting (m)"
        self._y_label = "Northing (m)"
        self._pending_draw = Coalesced(lambda: self._flush(), self)
        self._full_due = False
        # Kept per data set: the colour scale spans every depth, and a
        # triangulation serves every depth with the same soundings above it.
        self._scale = None
        self._triangulations: dict = {}
        self._drawn = None    # the artists of the last full drawing

    # -- public --------------------------------------------------------------
    def show_slices(self, xy, res, depths, *, label: str = "value", log_scale: bool = True,
                    x_label: str = "Easting (m)", y_label: str = "Northing (m)") -> None:
        import numpy as np
        self._xy = np.asarray(xy, dtype=float)
        self._res = np.asarray(res, dtype=float)
        self._depths = np.asarray(depths, dtype=float).ravel()
        if self._xy.ndim != 2 or self._xy.shape[1] < 2:
            raise ValueError("Plan slices need xy coordinates with shape (n_soundings, 2).")
        if self._res.ndim != 2:
            raise ValueError("Plan slices need values with shape (n_soundings, n_depths).")
        if self._res.shape[0] != self._xy.shape[0]:
            raise ValueError("Coordinate and sounding counts do not match.")
        if self._res.shape[1] != self._depths.size or self._depths.size == 0:
            raise ValueError("Depth coordinates must match the value columns and cannot be empty.")
        self._label = label
        self._log = bool(log_scale)
        self._x_label = x_label
        self._y_label = y_label
        n = self._depths.size
        self._z.blockSignals(True)
        self._z.setRange(0, max(0, n - 1))
        self._z.setValue(int(min(2, max(0, n - 1))))  # a shallow-ish default layer
        self._z.blockSignals(False)
        self._scale = None
        self._triangulations = {}
        self._drawn = None
        self._request_full()

    @property
    def colormap_chooser(self) -> "cmaps.ColormapChooser":
        return self._colormap

    def flush_redraw(self) -> None:
        """Draw now what a pending redraw would draw, for code reading the figure."""
        self._pending_draw.flush()

    # -- rendering -----------------------------------------------------------
    def _on_depth_changed(self, _value: int = 0) -> None:
        if self._depths is not None and self._depths.size:
            j = min(max(self._z.value(), 0), self._depths.size - 1)
            self._z_label.setText(f"{self._depths[j]:.0f} m")
        if self._z.isSliderDown():
            self._pending_draw.request()
        else:
            self._request_full()

    def _request_full(self) -> None:
        self._full_due = True
        self._pending_draw.request()

    def _flush(self) -> None:
        full, self._full_due = self._full_due, False
        if full or not self._preview():
            self._redraw()

    def _colour_scale(self):
        """A norm and the contour levels, over every depth of the data set.

        The limits are computed once per data set; the norm is new each call.
        Every artist drawn with a norm listens to it, so one shared across
        drawings would keep the artists of cleared figures alive and have them
        answer its changes.
        """
        import numpy as np
        from matplotlib.colors import LogNorm, Normalize

        if self._scale is None:
            finite_mask = np.isfinite(self._res)
            if self._log:
                finite_mask &= self._res > 0
            finite_all = self._res[finite_mask]
            if finite_all.size:
                vmin = float(np.nanpercentile(finite_all, 2))
                vmax = float(np.nanpercentile(finite_all, 98))
            else:
                vmin, vmax = 1.0, 100.0
            if self._log:
                vmin, vmax = max(vmin, 1e-6), max(vmax, vmin * 1.1 + 1e-6)
            elif not vmax > vmin:
                vmax = vmin + 1.0
            # Explicit levels across the range the colours cover. A level *count*
            # under a log norm is handed to a decade locator, so a survey spanning
            # less than two decades comes back as one or two flat bands and a
            # colour bar labelled only in powers of ten.
            levels = (np.geomspace(vmin, vmax, 15) if self._log
                      else np.linspace(vmin, vmax, 15))
            self._scale = (vmin, vmax, levels)
        vmin, vmax, levels = self._scale
        return (LogNorm if self._log else Normalize)(vmin, vmax), levels

    def _soundings_at(self, j: int):
        """The values at depth ``j`` and which soundings have one worth drawing."""
        import numpy as np

        vals = self._res[:, j]
        x, y = self._xy[:, 0], self._xy[:, 1]
        good = np.isfinite(vals) & np.isfinite(x) & np.isfinite(y)
        if self._log:
            good &= vals > 0
        return vals, good

    def _preview(self) -> bool:
        """Recolour the soundings already drawn for the depth now selected.

        The interpolated surface belongs to the depth it was drawn for, so it is
        hidden until the full drawing replaces it. False when there is nothing
        to recolour and only a full drawing will do.
        """
        import numpy as np

        drawn = self._drawn
        if drawn is None or self._res is None:
            return False
        j = int(np.clip(self._z.value(), 0, self._depths.size - 1))
        vals, good = self._soundings_at(j)
        if not good.any():
            return False
        drawn["scatter"].set_offsets(np.column_stack([self._xy[good, 0], self._xy[good, 1]]))
        drawn["scatter"].set_array(vals[good])
        if drawn["surface"] is not None:
            drawn["surface"].set_visible(False)
        # Lighter while the handle moves: the point outlines are most of a
        # paint with thousands of soundings, and the layout does not change
        # with the depth. The full drawing puts both back.
        drawn["scatter"].set_edgecolor("none")
        self._fig.set_layout_engine(None)
        drawn["axes"].set_title(f"Resistivity at depth {self._depths[j]:.0f} m")
        self._canvas.draw_idle()
        return True

    def _redraw(self, *_) -> None:
        import numpy as np
        if self._res is None or self._depths is None or self._xy is None:
            return
        self._drawn = None
        j = int(np.clip(self._z.value(), 0, self._depths.size - 1))
        vals, good = self._soundings_at(j)
        norm, levels = self._colour_scale()
        x, y = self._xy[:, 0], self._xy[:, 1]
        # Treat the soundings as a line (scatter only, no misleading filled map)
        # unless they genuinely spread in 2D: compare the minor vs major PCA axis.
        collinear = True
        if good.sum() >= 3:
            pts = np.column_stack([x[good], y[good]])
            sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
            collinear = (sv[0] <= 0) or (sv[1] / sv[0] < 0.08)

        self._fig.clear()
        self._fig.set_layout_engine("tight")      # a drag preview turns it off
        ax = self._fig.add_subplot(111)
        if not good.any():
            ax.text(
                0.5, 0.5,
                "No positive finite values at this depth."
                if self._log else "No finite values at this depth.",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_xlabel(self._x_label)
            if self._y_label:
                ax.set_ylabel(self._y_label)
            ax.set_title(f"Depth {self._depths[j]:.0f} m")
            self._z_label.setText(f"{self._depths[j]:.0f} m")
            self._canvas.draw_idle()
            return
        cmap = cmaps.to_matplotlib(self._colormap.colormap())
        mappable = None
        if good.sum() >= 4 and not collinear:
            try:  # a filled map when the soundings actually spread in 2D
                mappable = ax.tricontourf(self._triangulation(good), vals[good],
                                          levels=levels, extend="both",
                                          cmap=cmap, norm=norm)
            except Exception:  # noqa: BLE001 - degenerate triangulation -> points only
                mappable = None
        surface = mappable
        sc = ax.scatter(x[good], y[good], c=vals[good], cmap=cmap, norm=norm,
                        s=90, edgecolor="#333333", linewidth=0.5, zorder=3)
        if mappable is None:
            mappable = sc
        ax.set_xlabel(self._x_label)
        if self._y_label:
            ax.set_ylabel(self._y_label)
        ax.set_title(f"Resistivity at depth {self._depths[j]:.0f} m")
        if not collinear:
            ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        bar = self._fig.colorbar(mappable, ax=ax, label=self._label)
        if self._log:
            # Explicit levels are arbitrary reals, and the default formatter
            # renders those as "2.0986 x 10^2". The reader wants 210.
            from matplotlib.ticker import FuncFormatter, LogLocator

            bar.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            bar.ax.yaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))
        self._z_label.setText(f"{self._depths[j]:.0f} m")
        self._drawn = {"axes": ax, "scatter": sc, "surface": surface}
        self._canvas.draw_idle()

    def _triangulation(self, good):
        """The Delaunay triangulation of the soundings in ``good``, kept for reuse.

        Soundings drop out only below their depth of investigation, so most
        depths share the set above them, and with it the triangulation.
        """
        from matplotlib.tri import Triangulation

        key = good.tobytes()
        if key not in self._triangulations:
            self._triangulations[key] = Triangulation(self._xy[good, 0], self._xy[good, 1])
        return self._triangulations[key]
