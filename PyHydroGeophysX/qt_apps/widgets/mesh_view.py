"""Embedded matplotlib canvas that draws a pyGIMLi inversion result on its mesh.

The inverted resistivity/velocity is rendered with pyGIMLi's own mesh plotting
(``pg.show``) so it looks like a proper geophysical section — coloured cells on
the real triangular inversion mesh, a colorbar, coverage masking, optional cell
edges, and true topography — instead of a re-gridded raster image. A matplotlib
navigation toolbar provides zoom / pan / save, and the canvas rescales with the
window.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX._internal.utils import velocity_of
from PyHydroGeophysX.visualization.ert_style import (
    ERT_RESISTIVITY_LABEL,
    ert_model_plot_kwargs,
)


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

        self._smooth = QCheckBox("Smooth")
        self._smooth.setToolTip(
            "Draw the model on a once-subdivided mesh, interpolating cell centres. "
            "This is a display choice only: it adds no resolution and the exported "
            "model is unchanged. Turn “Show mesh” off to see the effect.")
        self._smooth.toggled.connect(self._redraw)
        bar.addWidget(self._smooth)

        # Sensitivity controls. ERT resolution falls off with depth and away from
        # the line, so part of every section is decoration; these say how much.
        self._show_cov = QCheckBox("Sensitivity")
        self._show_cov.setToolTip(
            "Draw the coverage (log10 cumulative sensitivity) instead of the model, "
            "so you can see which parts of the section the data actually constrain.")
        self._show_cov.toggled.connect(self._redraw)
        bar.addWidget(self._show_cov)

        self._mask_low = QCheckBox("Hide below")
        self._mask_low.setToolTip(
            "Blank the cells whose coverage falls under the threshold, rather than "
            "letting poorly constrained cells read as real structure.")
        self._mask_low.toggled.connect(self._redraw)
        bar.addWidget(self._mask_low)

        self._cov_threshold = QDoubleSpinBox()
        self._cov_threshold.setRange(-10.0, 10.0)
        self._cov_threshold.setDecimals(2)
        self._cov_threshold.setSingleStep(0.25)
        self._cov_threshold.setValue(-2.0)
        self._cov_threshold.setToolTip(
            "Coverage cut in log10 units. The status line under the plot reports the "
            "range for the current result and how much of the section survives.")
        self._cov_threshold.valueChanged.connect(self._redraw)
        bar.addWidget(self._cov_threshold)

        self._rays = QCheckBox("Rays")
        self._rays.setToolTip(
            "Overlay the first-arrival ray paths. Where no ray passes, the velocity "
            "is regularization rather than data. Travel-time results only.")
        self._rays.toggled.connect(self._redraw)
        self._rays.setVisible(False)  # shown once an SRT result arrives
        bar.addWidget(self._rays)

        # Travel time answers "was this cell sampled?" with a yes or no: a ray
        # either passed through or it did not. A log-sensitivity threshold is an
        # ERT question, so SRT gets this instead of "Sensitivity" and
        # "Hide below".
        self._hide_uncovered = QCheckBox("Hide uncovered")
        self._hide_uncovered.setChecked(True)
        self._hide_uncovered.setToolTip(
            "Blank the cells no ray passes through. Their velocity comes from the "
            "regularization pulling on neighbours, not from the travel times, so "
            "showing them invites reading structure into the smoothing. Untick to "
            "see the full inverted domain.")
        self._hide_uncovered.toggled.connect(self._redraw)
        self._hide_uncovered.setVisible(False)  # shown once an SRT result arrives
        bar.addWidget(self._hide_uncovered)

        self._cov_note = QLabel("")
        self._cov_note.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._cov_note)

    def show_model(self, mgr, kind: str = "ert") -> None:
        """Display the inverted model from a pyGIMLi manager (``ert`` or ``srt``)."""
        self._mgr = mgr
        self._mesh = None
        self._values = None
        self._coverage = None
        self._kind = kind
        self._title = ""
        self._sync_controls()
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
        self._sync_controls()
        self._redraw()

    def _resolve_source(self):
        """Return ``(mesh, values, coverage, field)``.

        ``coverage`` is what to hand pyGIMLi as the default alpha mask, and
        ``field`` is a scalar sensitivity array for the sensitivity view and the
        threshold. They differ for travel time, where the default display is
        already limited to ray-covered cells but the ray density itself is still
        worth plotting.
        """
        import numpy as np
        if self._mgr is not None:
            mgr = self._mgr
            if self._kind == "srt":
                values = np.asarray(velocity_of(mgr), dtype=float)
                # Raw coverage() is ray density over ~0..1e6 with exact zeros
                # where nothing passed, so it breaks pg.show as an alpha channel.
                # standardizedCoverage() gives the 0/1 mask to display with, and
                # log10 of the raw density is the field to threshold on.
                coverage, field = None, None
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
                try:
                    raw = np.asarray(mgr.coverage(), dtype=float)
                    if raw.size == values.size:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            field = np.where(raw > 0, np.log10(np.abs(raw)), np.nan)
                except Exception:  # noqa: BLE001 - coverage is optional
                    field = None
                return mgr.paraDomain, values, coverage, field
            values = np.asarray(mgr.model, dtype=float)
            coverage = None
            try:  # coverage (sensitivity) masking is an ERT concept
                cov = np.asarray(mgr.coverage(), dtype=float)
                if cov.size and np.isfinite(cov).any() and float(np.nanmax(cov)) > float(np.nanmin(cov)):
                    coverage = cov
            except Exception:  # noqa: BLE001 - coverage is optional
                pass
            return mgr.paraDomain, values, coverage, coverage
        if self._mesh is not None and self._values is not None:
            field = self._coverage
            if field is not None and np.asarray(field).dtype == bool:
                field = None
            return self._mesh, self._values, self._coverage, field
        return None, None, None, None

    @staticmethod
    def _subdivide(mesh, values, coverage):
        """Resample onto a once-subdivided mesh, for display only.

        ``createH2`` splits every triangle into four and ``pg.interpolate`` fills
        the new cell centres from the old ones. No resolution is added; it only
        stops the coarse inversion cells reading as blocky structure.
        """
        import numpy as np
        import pygimli as pg

        fine = mesh.createH2()
        centres = [c.center() for c in fine.cells()]
        fine_values = np.asarray(pg.interpolate(mesh, values, centres), dtype=float)
        if fine_values.size != fine.cellCount() or not np.isfinite(fine_values).all():
            return mesh, values, coverage  # fall back rather than draw holes
        fine_coverage = None
        if coverage is not None and np.asarray(coverage).size == np.asarray(values).size:
            fine_coverage = np.asarray(
                pg.interpolate(mesh, np.asarray(coverage, dtype=float), centres),
                dtype=float)
            if fine_coverage.size != fine.cellCount():
                fine_coverage = None
        return fine, fine_values, fine_coverage

    def _redraw(self) -> None:
        import numpy as np
        import pygimli as pg

        mesh, values, coverage, scalar_cov = self._resolve_source()
        if mesh is None or values is None:
            return

        srt = self._kind == "srt"
        self._sync_controls()
        show_coverage = (not srt) and self._show_cov.isChecked() and scalar_cov is not None
        self._describe_coverage(scalar_cov, values)

        # Decide what is plotted and what masks it on the source mesh, then
        # smooth the pair together; doing it the other way round leaves the mask
        # and the values on different meshes.
        plot_values = np.asarray(scalar_cov, dtype=float) if show_coverage else values
        mask = coverage
        if srt:
            # The ray mask is already what _resolve_source returned; the only
            # question is whether to apply it.
            if not self._hide_uncovered.isChecked():
                mask = None
        elif self._mask_low.isChecked() and scalar_cov is not None:
            cov = np.asarray(scalar_cov, dtype=float)
            if cov.size == np.asarray(plot_values).size:
                # A user threshold overrides whatever mask the caller supplied:
                # it is the one the reader chose, and it applies here too.
                mask = cov >= float(self._cov_threshold.value())
        elif show_coverage:
            mask = None  # do not mask the sensitivity plot by itself

        if self._smooth.isChecked():
            try:
                mesh, plot_values, mask = self._subdivide(mesh, plot_values, mask)
            except Exception:  # noqa: BLE001 - smoothing is cosmetic, never fatal
                pass
        values, coverage = plot_values, mask

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if show_coverage:
            label = ("Ray coverage (log10)" if self._kind == "srt"
                     else "Coverage (log10 cumulative sensitivity)")
            show_kw = dict(ax=ax, colorBar=False, cMap="viridis", logScale=False,
                           showMesh=self._show_mesh.isChecked())
        elif self._kind == "srt":
            cmap, log_scale, label = "turbo", False, "Velocity (m/s)"
            show_kw = dict(ax=ax, colorBar=False, cMap=cmap, logScale=log_scale,
                           showMesh=self._show_mesh.isChecked())
        else:
            label = ERT_RESISTIVITY_LABEL
            show_kw = ert_model_plot_kwargs(show_mesh=self._show_mesh.isChecked())
            show_kw.update(ax=ax, colorBar=False)

        try:
            # Draw the model on the mesh via pyGIMLi but build the colorbar with
            # matplotlib: pyGIMLi's own colorbar hits a divide-by-zero on some
            # velocity models. colorBar=False avoids that.
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
            # isHidden(), not isVisible(): the latter is False whenever an
            # ancestor has not been shown, which would skip the overlay in any
            # embedded or offscreen use.
            if self._rays.isChecked() and not self._rays.isHidden():
                self._draw_rays(ax)
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

    def _sync_controls(self) -> None:
        """Show the controls that mean something for the result on screen.

        Travel time and ERT ask different questions of the same panel. ERT has a
        continuous sensitivity worth plotting and thresholding; travel time has
        ray paths worth drawing and a yes-or-no coverage, so it gets those two
        controls and not the ERT pair.
        """
        srt = self._kind == "srt"
        has_rays = srt and (
            callable(getattr(self._mgr, "drawRayPaths", None))
            or callable(getattr(self._mgr, "getRayPaths", None)))
        self._rays.setVisible(bool(has_rays))
        self._hide_uncovered.setVisible(srt)
        for widget in (self._show_cov, self._mask_low, self._cov_threshold):
            widget.setVisible(not srt)

    def _draw_rays(self, ax) -> None:
        """Overlay first-arrival ray paths on the velocity section.

        ``drawRayPaths`` adds a LineCollection and is the maintained route; the
        manual fallback exists because it depends on the manager still holding
        the forward operator that produced the model.
        """
        import numpy as np

        mgr = self._mgr
        if mgr is None:
            return
        # zorder has to beat the filled cells and the mesh edges, or the paths are
        # drawn underneath and the overlay silently does nothing.
        style = dict(color="w", lw=0.6, alpha=0.75, zorder=10)
        try:
            mgr.drawRayPaths(ax=ax, **style)
            return
        except Exception:  # noqa: BLE001 - fall back to plotting the paths myself
            pass
        try:
            for path in mgr.getRayPaths():
                arr = np.asarray(path, dtype=float)
                if arr.ndim == 2 and arr.shape[0] > 1:
                    ax.plot(arr[:, 0], arr[:, 1], **style)
        except Exception:  # noqa: BLE001 - the overlay is optional
            pass

    def _describe_coverage(self, coverage, values) -> None:
        """Say how much of the section the data actually constrain."""
        import numpy as np

        if coverage is None:
            self._cov_note.setText("")
            for widget in (self._show_cov, self._mask_low, self._cov_threshold):
                widget.setEnabled(False)
            self._hide_uncovered.setEnabled(False)
            return
        for widget in (self._show_cov, self._mask_low, self._cov_threshold):
            widget.setEnabled(True)
        self._hide_uncovered.setEnabled(True)
        cov = np.asarray(coverage, dtype=float)
        finite = cov[np.isfinite(cov)]
        if finite.size == 0 or cov.size != np.asarray(values).size:
            self._cov_note.setText("")
            return

        if self._kind == "srt":
            # For travel time the field is log10 ray density and the unsampled
            # cells are the NaNs, so the count is the whole story: a threshold
            # on ray density is not a quantity anyone reads.
            sampled = int(finite.size)
            total = int(cov.size)
            share = 100.0 * sampled / max(total, 1)
            self._cov_note.setText(
                f"{sampled} of {total} cells have rays through them "
                f"({share:.0f} %)."
                + ("" if self._hide_uncovered.isChecked()
                   else "  The rest are shown, but their velocity is smoothing "
                        "rather than data."))
            return

        cut = float(self._cov_threshold.value())
        kept = int((finite >= cut).sum())
        share = 100.0 * kept / finite.size
        unsampled = int(cov.size - finite.size)
        self._cov_note.setText(
            f"Coverage {finite.min():.2f} to {finite.max():.2f} (median "
            f"{np.median(finite):.2f})."
            + (f" {unsampled} cells unsampled." if unsampled else "")
            + f" At the {cut:.2f} cut, {kept} of {finite.size} cells survive "
              f"({share:.0f} %)."
            + ("" if self._mask_low.isChecked() else "  Tick “Hide below” to apply it."))
