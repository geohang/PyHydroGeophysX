"""Embedded matplotlib canvas that draws a pyGIMLi inversion result on its mesh.

The inverted resistivity/velocity is rendered with pyGIMLi's own mesh plotting
(``pg.show``) so it looks like a proper geophysical section — coloured cells on
the real triangular inversion mesh, a colorbar, coverage masking, optional cell
edges, and true topography — instead of a re-gridded raster image. A matplotlib
navigation toolbar provides zoom / pan / save, and the canvas rescales with the
window.
"""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget


class MeshResultView(QWidget):
    """Show a pyGIMLi inversion manager's model drawn on its mesh."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(7.5, 4.2), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._mgr = None
        self._mesh = None         # field mode: a pyGIMLi mesh ...
        self._values = None       # ... and a per-cell value array
        self._coverage = None     # optional coverage mask/array for the field
        self._kind = "ert"
        self._title = ""

        bar = QHBoxLayout()
        bar.addWidget(self._toolbar, stretch=1)
        self._show_mesh = QCheckBox("Show mesh")
        self._show_mesh.setChecked(True)
        self._show_mesh.setToolTip("Overlay the inversion mesh cell boundaries.")
        self._show_mesh.toggled.connect(self._redraw)
        bar.addWidget(self._show_mesh)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self._canvas, stretch=1)

    def show_model(self, mgr, kind: str = "ert") -> None:
        """Display the inverted model from a pyGIMLi manager (``ert`` or ``srt``)."""
        self._mgr = mgr
        self._mesh = None
        self._values = None
        self._coverage = None
        self._kind = kind
        self._title = ""
        self._redraw()

    def show_field(self, mesh, values, kind: str = "ert", coverage=None, title: str = "") -> None:
        """Display a raw ``(mesh, per-cell values)`` pair — e.g. one time step of a
        time-lapse result, where there is no single pyGIMLi manager to hold it."""
        import numpy as np
        self._mgr = None
        self._mesh = mesh
        self._values = np.asarray(values, dtype=float)
        self._coverage = None if coverage is None else np.asarray(coverage)
        self._kind = kind
        self._title = title or ""
        self._redraw()

    def _resolve_source(self):
        """Return ``(mesh, values, coverage)`` from either a manager or a raw field."""
        import numpy as np
        if self._mgr is not None:
            mgr = self._mgr
            if self._kind == "srt":
                values = np.asarray(mgr.velocity, dtype=float)
                # Only show velocity where rays actually sample the model. The raw
                # ``coverage()`` (ray density, range ~0..1e6) breaks pg.show when used
                # as alpha, so use ``standardizedCoverage()`` (0/1) as a boolean mask;
                # fall back to thresholding the raw coverage.
                coverage = None
                for method, thresh in (("standardizedCoverage", 0.5), ("coverage", 0.0)):
                    fn = getattr(mgr, method, None)
                    if fn is None:
                        continue
                    try:
                        cov = np.asarray(fn(), dtype=float)
                    except Exception:  # noqa: BLE001
                        continue
                    if cov.size == values.size and np.isfinite(cov).any():
                        mask = cov > thresh
                        if mask.any() and not mask.all():  # a real, partial mask
                            coverage = mask
                            break
                return mgr.paraDomain, values, coverage
            values = np.asarray(mgr.model, dtype=float)
            coverage = None
            try:  # coverage (sensitivity) masking is an ERT concept
                cov = np.asarray(mgr.coverage(), dtype=float)
                if cov.size and np.isfinite(cov).any() and float(np.nanmax(cov)) > float(np.nanmin(cov)):
                    coverage = cov
            except Exception:  # noqa: BLE001 - coverage is optional
                pass
            return mgr.paraDomain, values, coverage
        if self._mesh is not None and self._values is not None:
            return self._mesh, self._values, self._coverage
        return None, None, None

    def _redraw(self) -> None:
        import numpy as np
        import pygimli as pg

        mesh, values, coverage = self._resolve_source()
        if mesh is None or values is None:
            return
        if self._kind == "srt":
            cmap, log_scale, label = "turbo", False, "Velocity (m/s)"
        else:
            cmap, log_scale, label = "Spectral_r", True, "Resistivity (Ω·m)"

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        try:
            # Draw the model on the mesh via pyGIMLi but build the colorbar with
            # matplotlib: pyGIMLi's own colorbar hits a divide-by-zero on some
            # velocity models. colorBar=False avoids that.
            show_kw = dict(ax=ax, colorBar=False, cMap=cmap, logScale=log_scale,
                           showMesh=self._show_mesh.isChecked())
            if coverage is not None and np.asarray(coverage).size == np.asarray(values).size:
                show_kw["coverage"] = coverage
            try:
                pg.show(mesh, values, **show_kw)
            except Exception:  # noqa: BLE001 - coverage masking can still fail; retry plain
                show_kw.pop("coverage", None)
                ax.clear()
                pg.show(mesh, values, **show_kw)
            mappable = next(
                (c for c in ax.collections if getattr(c, "get_array", lambda: None)() is not None),
                None,
            )
            if mappable is not None:
                cbar = self._fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
                cbar.set_label(label)
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Elevation (m)")
            if self._title:
                ax.set_title(self._title)
        except Exception as exc:  # noqa: BLE001 - never crash the UI on a draw error
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Could not draw model:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
            ax.axis("off")
        self._canvas.draw_idle()
