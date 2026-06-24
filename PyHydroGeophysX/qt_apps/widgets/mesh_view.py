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
        self._kind = "ert"

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
        self._kind = kind
        self._redraw()

    def _redraw(self) -> None:
        if self._mgr is None:
            return
        import numpy as np
        import pygimli as pg

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        mgr = self._mgr
        try:
            if self._kind == "srt":
                values = np.asarray(mgr.velocity, dtype=float)
                cmap, log_scale, label = "turbo", False, "Velocity (m/s)"
            else:
                values = np.asarray(mgr.model, dtype=float)
                cmap, log_scale, label = "Spectral_r", True, "Resistivity (Ω·m)"
            # Draw the model on the mesh via pyGIMLi but build the colorbar with
            # matplotlib: pyGIMLi's own colorbar hits a divide-by-zero on some
            # velocity models. colorBar=False avoids that.
            show_kw = dict(ax=ax, colorBar=False, cMap=cmap, logScale=log_scale,
                           showMesh=self._show_mesh.isChecked())
            if self._kind != "srt":
                # Coverage (sensitivity) masking is an ERT concept; the SRT ray
                # coverage normalizes to NaN alpha and breaks the draw.
                try:
                    cov = np.asarray(mgr.coverage(), dtype=float)
                    if cov.size and np.isfinite(cov).any() and float(np.nanmax(cov)) > float(np.nanmin(cov)):
                        show_kw["coverage"] = cov
                except Exception:  # noqa: BLE001 - coverage is optional
                    pass
            try:
                pg.show(mgr.paraDomain, values, **show_kw)
            except Exception:  # noqa: BLE001 - coverage masking can still fail; retry plain
                show_kw.pop("coverage", None)
                ax.clear()
                pg.show(mgr.paraDomain, values, **show_kw)
            mappable = next(
                (c for c in ax.collections if getattr(c, "get_array", lambda: None)() is not None),
                None,
            )
            if mappable is not None:
                cbar = self._fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
                cbar.set_label(label)
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Elevation (m)")
        except Exception as exc:  # noqa: BLE001 - never crash the UI on a draw error
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Could not draw model:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
            ax.axis("off")
        self._canvas.draw_idle()
