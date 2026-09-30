"""Embedded matplotlib canvas that draws a pyGIMLi inversion result on its mesh.

The inverted resistivity/velocity is rendered with pyGIMLi's own mesh plotting
(``pg.show``) so it looks like a proper geophysical section — coloured cells on
the real triangular inversion mesh, a colorbar, coverage masking, optional cell
edges, and true topography — instead of a re-gridded raster image. A matplotlib
navigation toolbar provides zoom / pan / save, and the canvas rescales with the
window.
"""

from __future__ import annotations

import hashlib

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX._internal.utils import velocity_of
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.coalesce import Coalesced
from PyHydroGeophysX.qt_apps.widgets.readout import navigation_toolbar
from PyHydroGeophysX.visualization.axis_units import set_section_axes, vertical_axis
from PyHydroGeophysX.visualization.ert_style import (
    ERT_RESISTIVITY_LABEL,
    ert_model_plot_kwargs,
)


def _array_key(array):
    """A cheap identity for an array's contents, to key what was derived from it."""
    if array is None:
        return None
    import numpy as np

    data = np.ascontiguousarray(array)
    return (data.dtype.str, data.shape,
            hashlib.blake2b(data.tobytes(), digest_size=16).digest())


def _same_array(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:            # equal_nan means nothing for a boolean mask
        return bool(np.array_equal(a, b))


def _refine(mesh):
    """``mesh`` with every triangle split into four, and its cell centres."""
    fine = mesh.createH2()
    return fine, [cell.center() for cell in fine.cells()]


def _interpolate(mesh, values, centres, fine_count: int):
    """``values`` on ``mesh`` interpolated to ``centres``; None if that fails.

    For display only: ``pg.interpolate`` fills the new cell centres from the
    old ones, which stops the coarse inversion cells reading as blocky
    structure without adding any resolution.
    """
    import numpy as np
    import pygimli as pg

    try:
        fine = np.asarray(pg.interpolate(mesh, np.asarray(values, dtype=float), centres),
                          dtype=float)
    except Exception:  # noqa: BLE001 - smoothing is cosmetic, never fatal
        return None
    return fine if fine.size == fine_count else None


class MeshResultView(QWidget):
    """Show a pyGIMLi inversion manager's model drawn on its mesh.

    ``colormaps`` is the studio state's shared colormap dict (see
    :mod:`~PyHydroGeophysX.qt_apps.widgets.colormaps`): a resistivity section
    drawn here and one reopened on another page then use the same choice.
    """

    def __init__(self, parent=None, *, colormaps=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(7.5, 4.2))
        # Every drawing starts from the same layout state (see _redraw_now): the
        # subplot parameters as they were before pyGIMLi's tight_layout moved them.
        self._subplot_defaults = {
            name: getattr(self._fig.subplotpars, name)
            for name in ("left", "right", "bottom", "top", "wspace", "hspace")}
        self._canvas = FigureCanvasQTAgg(self._fig)
        # Below this a section is not readable anyway, and matplotlib gives up on
        # fitting the title and the colorbar around an equal-aspect axes - which
        # is how the title ends up cut off by the top of the panel.
        self._canvas.setMinimumHeight(280)
        # The cursor position goes to a readout of its own: the toolbar's label
        # would re-lay out the page on every mouse move (see widgets.readout).
        self._toolbar, self._coords = navigation_toolbar(self._canvas, self)
        self._mgr = None
        self._mesh = None         # field mode: a pyGIMLi mesh ...
        self._values = None       # ... and a per-cell value array
        self._coverage = None     # optional coverage mask/array for the field
        self._kind = "ert"
        self._title = ""
        # Redraw requests of one turn of the event loop become one drawing (see
        # _redraw), and what a drawing derives from the result - its coverage,
        # the subdivided mesh, the contour grid - is kept until the result
        # changes, under a version that every new result bumps.
        self._pending_draw = Coalesced(lambda: self._redraw_now(), self)
        self._version = 0
        self._derived: dict = {}
        self._drawn = None        # what the figure on screen was drawn from

        bar = QHBoxLayout()
        bar.addWidget(self._toolbar)
        bar.addWidget(self._coords, stretch=1)
        self._show_mesh = QCheckBox("Show mesh")
        self._show_mesh.setChecked(True)
        self._show_mesh.setToolTip("Overlay the inversion mesh cell boundaries.")
        self._show_mesh.toggled.connect(self._redraw)
        bar.addWidget(self._show_mesh)

        # How smooth the section is drawn. Subdividing interpolates the model onto
        # a finer mesh and still draws cells; "Contour" leaves the mesh behind and
        # draws filled contours on a regular grid, which is the continuous image
        # traditional resistivity software produces. Both are display choices: no
        # resolution is added and the exported model is unchanged.
        bar.addWidget(QLabel("Smooth"))
        self._smooth = QComboBox()
        self._smooth.addItem("Off", 0)
        self._smooth.addItem("Cells ×1", 1)
        self._smooth.addItem("Cells ×2", 2)
        self._smooth.addItem("Contour", -1)
        self._smooth.setToolTip(
            "How smoothly the model is drawn. “Cells” interpolates it onto a "
            "subdivided mesh and still draws cells; “Contour” draws filled "
            "contours on a regular grid instead, for the continuous image "
            "traditional resistivity software produces. Display only - no "
            "resolution is added and the exported model is unchanged. Turn "
            "“Show mesh” off to see the effect.\n\n"
            "Contours cannot reproduce the soft per-cell sensitivity fade, so "
            "they draw the whole section: use “Hide below” or “Clean cut” to trim "
            "it to what the data resolve.")
        self._smooth.currentIndexChanged.connect(self._on_smooth_changed)
        bar.addWidget(self._smooth)

        self._levels = QSpinBox()
        self._levels.setRange(4, 128)
        self._levels.setValue(40)
        self._levels.setPrefix("levels ")
        self._levels.setMaximumWidth(110)
        self._levels.setKeyboardTracking(False)
        self._levels.setToolTip(
            "Number of filled contour bands. A high count reads as a continuous "
            "image; a low one bands the section, which is easier to read a value "
            "off but invents edges where the model is smooth.")
        self._levels.setVisible(False)
        self._levels.valueChanged.connect(self._redraw)
        bar.addWidget(self._levels)

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

        # Blanking cells can only cut on cell boundaries, and an inversion mesh has
        # ten-metre triangles at depth, so the cut comes out as a saw-tooth with
        # islands hanging off it. Clipping the drawing to the coverage envelope cuts
        # on that line instead: the clean shape traditional software produces.
        self._clean_cut = QCheckBox("Clean cut")
        self._clean_cut.setToolTip(
            "Clip the section to a smooth envelope at the coverage cut, instead of "
            "blanking cell by cell. The edge then follows how deep the survey sees "
            "rather than where the mesh happens to put a triangle. Needs "
            "“Hide below”.")
        self._clean_cut.setEnabled(False)
        self._clean_cut.toggled.connect(self._redraw)
        bar.addWidget(self._clean_cut)

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

        # Colour limits. Autoscaling every result to its own extremes is right
        # for a single model and wrong for a series: each time step gets its own
        # scale, so the change everyone is looking for is exactly what the
        # rescaling hides. Locking freezes the limits across steps.
        self._lock_range = QCheckBox("Lock range")
        self._lock_range.setToolTip(
            "Freeze the colour limits instead of rescaling to each result. Time "
            "steps can only be compared on one scale; without this, every step "
            "is stretched to its own min/max and the change between them cannot "
            "be read off the colours.")
        self._lock_range.toggled.connect(self._on_lock_toggled)
        bar.addWidget(self._lock_range)

        self._cmin = QDoubleSpinBox()
        self._cmax = QDoubleSpinBox()
        for box, name in ((self._cmin, "Lower"), (self._cmax, "Upper")):
            box.setRange(-1.0e9, 1.0e9)
            box.setDecimals(3)
            box.setMaximumWidth(96)
            box.setKeyboardTracking(False)  # redraw on commit, not per keystroke
            box.setEnabled(False)
            box.setToolTip(f"{name} colour limit, applied when “Lock range” is "
                           "ticked. While it is unticked these track the result "
                           "on screen, so ticking the box keeps what you see.")
            box.valueChanged.connect(self._on_limit_changed)
            bar.addWidget(box)

        # The colour map, one per quantity - resistivity, change, velocity,
        # coverage - each shared with every other view of it, and each opening
        # on the map this view always drew it with. The row above is already as
        # wide as the panel can afford, so the choice has a row of its own,
        # under the colour limits it belongs with.
        self._colormap = cmaps.ColormapChooser(
            cmaps.RESISTIVITY, ert_model_plot_kwargs()["cMap"], shared=colormaps)
        self._colormap.colormapChanged.connect(self._on_colormap_changed)
        # Which quantity it is set for is only known once a result is drawn; a
        # choice made before that would be filed under the wrong one.
        self._colormap.setEnabled(False)
        colour_row = QHBoxLayout()
        colour_row.setContentsMargins(0, 0, 0, 0)
        colour_row.addStretch(1)
        colour_row.addWidget(QLabel("Colour map"))
        colour_row.addWidget(self._colormap)

        self._cov_note = QLabel("")
        self._cov_note.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addLayout(colour_row)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._cov_note)
        self._side = None          # splitter holding the canvas and a side panel
        # View > Length Units: the axes change, the model on the mesh does not.
        length_units.notifier().changed.connect(self._on_length_unit_changed)

    def add_side_panel(self, widget: QWidget) -> None:
        """Put ``widget`` to the right of the section, under the toolbar.

        For tools that act on the result on screen, such as the temperature
        correction, so they sit next to what they change. Under the toolbar
        rather than beside the whole view: the toolbar alone already sets how
        narrow this view can get, and a panel beside it would add its own width
        on top - wider than many screens. The splitter lets it be dragged
        narrower or closed.
        """
        if self._side is None:
            layout = self.layout()
            index = layout.indexOf(self._canvas)
            layout.removeWidget(self._canvas)
            self._side = QSplitter(Qt.Horizontal)
            self._side.addWidget(self._canvas)
            self._side.setCollapsible(0, False)
            layout.insertWidget(index, self._side, 1)
        self._side.addWidget(widget)
        self._side.setStretchFactor(0, 1)
        self._side.setStretchFactor(self._side.count() - 1, 0)

    def show_model(self, mgr, kind: str = "ert") -> None:
        """Display the inverted model from a pyGIMLi manager (``ert`` or ``srt``)."""
        self._mgr = mgr
        self._mesh = None
        self._values = None
        self._coverage = None
        self._kind = kind
        self._title = ""
        self._new_result()
        # A standalone model is a new quantity on a new mesh, so limits carried
        # over from whatever was shown before would be meaningless.
        self._lock_range.setChecked(False)
        self._sync_controls()
        self._redraw()

    def set_color_range(self, vmin: float, vmax: float, lock: bool = True) -> None:
        """Fix the colour limits — e.g. to the range over every time step.

        Callers that hold a whole series use this to put each step on one scale;
        the viewer on its own only ever sees one step and cannot know the range
        of the others.
        """
        lo, hi = float(vmin), float(vmax)
        if not (hi > lo):
            return
        self._set_limit_boxes(lo, hi)
        if self._lock_range.isChecked() != bool(lock):
            self._lock_range.setChecked(bool(lock))   # its handler asks for the redraw
        elif lock:
            self._redraw()

    def show_field(self, mesh, values, kind: str = "ert", coverage=None, title: str = "") -> None:
        """Display a raw ``(mesh, per-cell values)`` pair — e.g. one time step of a
        time-lapse result, where there is no single pyGIMLi manager to hold it.

        ``kind='change'`` renders a signed percentage change rather than a model:
        a diverging map on a linear scale, centred on zero.
        """
        import numpy as np
        values = np.asarray(values, dtype=float)
        coverage = None if coverage is None else np.asarray(coverage)
        # The same field handed in again - a caller refreshing, or switching the
        # colour mode - keeps what was derived from it.
        same = (self._mgr is None and mesh is self._mesh
                and _same_array(values, self._values)
                and _same_array(coverage, self._coverage))
        self._mgr = None
        self._mesh = mesh
        self._values = values
        self._coverage = coverage
        self._kind = kind
        self._title = title or ""
        if not same:
            self._new_result()
        self._sync_controls()
        self._redraw()

    def _new_result(self) -> None:
        """Forget everything derived from the previous result."""
        self._version += 1
        self._derived = {}

    def _cached(self, key, compute):
        """``compute()``, once per result: kept until the result changes."""
        full = (self._version, key)
        if full not in self._derived:
            self._derived[full] = compute()
        return self._derived[full]

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

    def _smoothed(self, mesh, values, mask, level: int, role: str):
        """Resample onto a ``level`` times subdivided mesh, for display only.

        ``createH2`` splits every triangle into four and the values are
        interpolated onto the new cells: no resolution is added, it only stops
        the coarse inversion cells reading as blocky structure. The finer meshes
        and the values on them do not change with the coverage cut or the
        colours, so they are kept; a moved cut interpolates only its mask.
        """
        import numpy as np

        mask_key = _array_key(mask)
        for step in range(level):
            try:
                fine, centres = self._cached(("fine mesh", step), lambda m=mesh: _refine(m))
            except Exception:  # noqa: BLE001 - smoothing is cosmetic, never fatal
                break
            count = fine.cellCount()
            fine_values = self._cached(
                ("fine values", step, role),
                lambda m=mesh, v=values, c=centres, n=count: _interpolate(m, v, c, n))
            if fine_values is None or not np.isfinite(fine_values).all():
                break  # fall back rather than draw holes
            fine_mask = None
            if mask is not None and np.asarray(mask).size == np.asarray(values).size:
                fine_mask = self._cached(
                    ("fine mask", step, mask_key),
                    lambda m=mesh, k=mask, c=centres, n=count: _interpolate(m, k, c, n))
            mesh, values, mask = fine, fine_values, fine_mask
        return mesh, values, mask

    def _redraw(self) -> None:
        """Ask for the section to be drawn again, once the current event is handled.

        Every change to the picture lands here - a result arriving, a tick box,
        the colour limits, a caller setting several of them in a row - and the
        requests of one turn of the event loop become one drawing. ``draw_idle``
        merges only the final paint: the subdivision, the coverage statistics
        and pyGIMLi rebuilding every cell all ran once per request before it,
        so locking the colour range drew the section twice and a step of a
        series three times.
        """
        self._pending_draw.request()

    def flush_redraw(self) -> None:
        """Draw now what a pending redraw would draw.

        For code that reads the figure straight after changing it - a capture of
        the view - rather than waiting for the event loop to get there.
        """
        self._pending_draw.flush()

    def _redraw_now(self) -> None:
        import numpy as np
        import pygimli as pg

        mesh, values, coverage, scalar_cov = self._cached("source", self._resolve_source)
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

        # A clean cut replaces the per-cell mask rather than adding to it: keeping
        # both would draw the saw-tooth edge inside the smooth outline.
        clip_polygon = None
        clip_key = None
        if (mask is not None and not show_coverage
                and self._clean_cut.isChecked() and self._clean_cut.isEnabled()):
            clip_key = (_array_key(mask), float(self._cov_threshold.value()))
            clip_polygon = self._cached(("clip",) + clip_key,
                                        lambda m=mesh, k=mask: self._clip_polygon(m, k))
            if clip_polygon is not None:
                mask = None
            else:
                clip_key = None

        smooth_level = int(self._smooth.currentData() or 0)
        contour = smooth_level < 0
        source_mesh = mesh          # the contour grid interpolates from this one
        role = "coverage" if show_coverage else "model"
        mask_key = _array_key(mask)
        if smooth_level > 0:
            mesh, plot_values, mask = self._smoothed(mesh, plot_values, mask,
                                                     smooth_level, role)
        values, coverage = plot_values, mask

        show_kw, label = self._style(show_coverage, values)
        # The sensitivity view is a different quantity in different units, so a
        # lock set on the model does not follow it there.
        locked = self._lock_range.isChecked() and not show_coverage
        rays = self._rays.isChecked() and not self._rays.isHidden()
        # Everything the drawing depends on but its colours and its coverage
        # mask. While it holds, a new colour map or colour range re-colours the
        # cells already drawn, and a moved coverage cut re-masks them, rather
        # than having pyGIMLi rebuild every one. Contour bands are cut at the
        # colour limits, so there the limits belong to the drawing.
        shape = (self._version, self._kind, self._title, role, clip_key, smooth_level,
                 self._show_mesh.isChecked(), rays, length_units.current(),
                 (int(self._levels.value()), show_kw.get("cMin"), show_kw.get("cMax"))
                 if contour else None)
        drawn = self._drawn
        # Unlocking hands the limits back to pyGIMLi's autoscale, which only a
        # drawing from scratch computes.
        if drawn is not None and drawn["shape"] == shape and (locked or not drawn["locked"]):
            if drawn["mask"] == mask_key or self._remask(coverage, drawn):
                if self._recolour(show_kw, drawn):
                    drawn.update(locked=locked, mask=mask_key)
                    return

        self._drawn = None
        self._fig.clear()
        # The subplot parameters and the layout engine both outlive clear(), and
        # pyGIMLi's tight_layout moves the first and swaps the second: each
        # drawing then started from where the last one's layout ended, and the
        # section shifted and changed size from one redraw to the next. Start
        # every drawing from the same place - with no layout engine, since
        # pyGIMLi calls tight_layout itself and matplotlib warns every time that
        # is done to a constrained figure. The layout is put back once the
        # drawing is finished, in _relayout.
        self._fig.subplotpars.update(**self._subplot_defaults)
        self._fig.set_layout_engine(None)
        ax = self._fig.add_subplot(111)
        draw_kw = dict(show_kw, ax=ax)

        try:
            # Draw the model on the mesh via pyGIMLi but build the colorbar with
            # matplotlib: pyGIMLi's own colorbar hits a divide-by-zero on some
            # velocity models. colorBar=False avoids that.
            mappable, cbar = None, None
            if contour:
                mappable = self._draw_contour(ax, source_mesh, values, coverage, show_kw, role)
            drew_contour = mappable is not None
            if mappable is None:
                if contour:
                    ax.clear()   # the contour attempt left partial artists behind
                if coverage is not None and np.asarray(coverage).size == np.asarray(values).size:
                    draw_kw["coverage"] = coverage
                try:
                    pg.show(mesh, values, **draw_kw)
                except Exception:  # noqa: BLE001 - coverage masking can still fail; retry plain
                    draw_kw.pop("coverage", None)
                    ax.clear()
                    pg.show(mesh, values, **draw_kw)
                mappable = next(
                    (c for c in ax.collections
                     if getattr(c, "get_array", lambda: None)() is not None),
                    None,
                )
            if clip_polygon is not None:
                from PyHydroGeophysX.visualization.section_clip import (
                    clip_axes_to_polygon,
                )
                clip_axes_to_polygon(ax, clip_polygon, outline=True, tighten=True)
            if mappable is not None:
                cbar = self._fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
                cbar.set_label(label)
                if drew_contour:
                    # A contour colorbar ticks on its own level boundaries, which
                    # with forty bands are arbitrary numbers like 1.48038e2. Put
                    # readable values on it instead.
                    self._set_contour_ticks(cbar, bool(show_kw.get("logScale")))
            # isHidden(), not isVisible(): the latter is False whenever an
            # ancestor has not been shown, which would skip the overlay in any
            # embedded or offscreen use.
            if rays:
                self._draw_rays(ax)
            # Elevation when the survey had elevations; depth, positive down,
            # when every electrode sits at z = 0 - a line read without them.
            mode, reference = self._cached(
                "vertical axis", lambda m=source_mesh: vertical_axis(mesh=m))
            set_section_axes(ax, vertical=mode, depth_reference=reference)
            if self._title:
                ax.set_title(self._title)
            self._drawn = {"shape": shape, "mask": mask_key, "locked": locked,
                           "mappable": mappable, "colorbar": cbar, "contour": drew_contour,
                           "masked_in_place": (not drew_contour and smooth_level == 0
                                               and clip_polygon is None),
                           "log": bool(show_kw.get("logScale"))}
        except Exception as exc:  # noqa: BLE001 - never crash the UI on a draw error
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Could not draw model:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
            ax.axis("off")
        self._relayout()
        self._canvas.draw_idle()

    def _style(self, show_coverage: bool, values):
        """``pg.show`` keywords and colorbar label for what is about to be drawn.

        Colours included: the map chosen for the quantity and the colour limits,
        locked or tracking the result.
        """
        show_mesh = self._show_mesh.isChecked()
        if show_coverage:
            label = ("Ray coverage (log10)" if self._kind == "srt"
                     else "Coverage (log10 cumulative sensitivity)")
            show_kw = dict(colorBar=False, cMap="viridis", logScale=False, showMesh=show_mesh)
        elif self._kind == "srt":
            label = "Velocity (m/s)"
            show_kw = dict(colorBar=False, cMap="turbo", logScale=False, showMesh=show_mesh)
        elif self._kind == "change":
            # A signed change reads off a diverging map centred on zero; on an
            # off-centre scale the sign is decided by the colours rather than by
            # the numbers.
            label = "Change from baseline (%)"
            show_kw = dict(colorBar=False, cMap="RdBu_r", logScale=False, showMesh=show_mesh)
            span = self._symmetric_span(values)
            if span is not None:
                show_kw["cMin"], show_kw["cMax"] = -span, span
        else:
            label = ERT_RESISTIVITY_LABEL
            show_kw = ert_model_plot_kwargs(show_mesh=show_mesh)
            show_kw.update(colorBar=False)
        # The map chosen for this quantity, or the one above when none has been:
        # a name PyGIMLi and matplotlib both take, so the default draws exactly
        # as it always did.
        key = (cmaps.COVERAGE if show_coverage
               else cmaps.VELOCITY if self._kind == "srt"
               else cmaps.RESISTIVITY_CHANGE if self._kind == "change"
               else cmaps.RESISTIVITY)
        show_kw["cMap"] = cmaps.to_matplotlib(
            self._colormap.set_target(key, show_kw["cMap"]))
        self._colormap.setEnabled(True)
        self._apply_color_limits(show_kw, values, lockable=not show_coverage)
        return show_kw, label

    @staticmethod
    def _remask(mask, drawn: dict) -> bool:
        """Put a new coverage mask on the cells already drawn; False if it cannot.

        Only for cells drawn one per model cell - not subdivided, not contoured,
        not clipped - where the mask maps onto them one to one. pyGIMLi's own
        ``addCoverageAlpha`` sets it, exactly as ``pg.show`` would have.
        """
        import numpy as np
        import pygimli as pg

        mappable = drawn.get("mappable")
        add_alpha = getattr(pg.viewer.mpl, "addCoverageAlpha", None)
        if (mask is None or mappable is None or add_alpha is None
                or not drawn.get("masked_in_place")
                or np.asarray(mask).size != len(mappable.get_paths())):
            return False
        try:
            add_alpha(mappable, mask)
        except Exception:  # noqa: BLE001 - draw from scratch instead
            return False
        return True

    def _recolour(self, show_kw: dict, drawn: dict) -> bool:
        """Give what is drawn new colours in place; False when only a redraw can.

        pyGIMLi keeps each cell's value on the collection it draws and puts the
        coverage fade in its alpha, so a new map or new limits are the
        collection's own to apply - the fade, the mesh lines and the layout stay.
        """
        mappable, cbar = drawn.get("mappable"), drawn.get("colorbar")
        if mappable is None or cbar is None:
            return False
        try:
            mappable.set_cmap(show_kw["cMap"])
            if show_kw.get("cMin") is not None and show_kw.get("cMax") is not None:
                mappable.set_clim(show_kw["cMin"], show_kw["cMax"])
            cbar.update_normal(mappable)
            if drawn.get("contour"):
                self._set_contour_ticks(cbar, bool(drawn.get("log")))
        except Exception:  # noqa: BLE001 - draw from scratch instead
            return False
        self._canvas.draw_idle()
        return True

    def _relayout(self) -> None:
        """Re-run the figure layout once pyGIMLi has finished drawing.

        ``pg.show`` positions its axes itself, which leaves matplotlib holding a
        placeholder layout engine and the axes reaching to 96 % of the figure
        height. On an equal-aspect section - which is most of them - the title
        then lands above the canvas and is cut off by its top edge. Handing the
        figure back to a real layout engine here puts that space back, and keeps
        it correct when the panel is resized.
        """
        try:
            self._fig.set_layout_engine("constrained")
            self._fig.draw_without_rendering()
        except Exception:  # noqa: BLE001 - fall back rather than lose the figure
            try:
                self._fig.tight_layout()
            except Exception:  # noqa: BLE001 - layout is cosmetic, never fatal
                pass

    def _draw_contour(self, ax, mesh, values, coverage, show_kw, role: str = "model"):
        """Draw the model as filled contours on a regular grid.

        The mesh view is honest about where the model's degrees of freedom are,
        which is why it is the default; but a section is also read as a picture of
        the ground, and for that the cell edges are an artefact of the inversion,
        not of the site. This is the continuous rendering traditional resistivity
        software produces: the model interpolated onto a grid and drawn as filled
        bands, blanked above the ground surface and outside the coverage.

        Returns the mappable for the colorbar, or None if the section could not be
        gridded - in which case the caller falls back to the mesh drawing rather
        than showing nothing. The gridding is the costly part and depends only on
        the result, so it is kept for the next redraw; ``role`` says which of the
        result's fields ``values`` is.
        """
        import numpy as np
        from matplotlib.colors import LogNorm, Normalize
        from scipy.interpolate import griddata

        from PyHydroGeophysX.core import section_geometry

        def layout():
            centers = section_geometry.cell_centers(mesh)
            surface = section_geometry.surface_line(mesh)
            x, z = centers[:, 0], centers[:, 1]
            # A fixed grid count rather than a cell size: the point is a smooth
            # picture, and the resolution of the model is set by the mesh either way.
            xi = np.linspace(float(x.min()), float(x.max()), 500)
            zi = np.linspace(float(z.min()), float(z.max()), 250)
            grid_x, grid_z = np.meshgrid(xi, zi)
            # Blank the air: the convex hull of the cell centres reaches above a
            # concave hillside, and a contour drawn there is interpolation into
            # the sky.
            top = section_geometry.surface_elevation_at(xi, surface)
            return x, z, xi, zi, grid_x, grid_z, top

        try:
            x, z, xi, zi, grid_x, grid_z, top = self._cached(("contour grid",), layout)
        except Exception:  # noqa: BLE001 - no geometry, no grid
            return None
        field = np.asarray(values, dtype=float).ravel()
        if x.size != field.size:
            return None

        def gridded():
            grid = griddata((x, z), field, (grid_x, grid_z), method="linear")
            return np.where(grid_z <= top[None, :], grid, np.nan)

        grid = self._cached(("contour values", role), gridded)
        if not np.isfinite(grid).any():
            return None

        if coverage is not None and np.asarray(coverage).size == field.size:
            if np.asarray(coverage).dtype == bool:
                weight = np.asarray(coverage, dtype=float).ravel()
                blanked = self._cached(
                    ("contour mask", _array_key(coverage)),
                    lambda: griddata((x, z), weight, (grid_x, grid_z), method="linear"))
                grid = np.where(np.nan_to_num(blanked) >= 0.5, grid, np.nan)

        lo = show_kw.get("cMin")
        hi = show_kw.get("cMax")
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            return None
        lo = float(lo if lo is not None else np.nanmin(finite))
        hi = float(hi if hi is not None else np.nanmax(finite))
        if hi <= lo:
            hi = lo + abs(lo) * 0.01 + 1.0e-9
        count = int(self._levels.value())
        if show_kw.get("logScale") and lo > 0.0:
            norm = LogNorm(vmin=lo, vmax=hi)
            levels = np.logspace(np.log10(lo), np.log10(hi), count)
        else:
            norm = Normalize(vmin=lo, vmax=hi)
            levels = np.linspace(lo, hi, count)

        filled = ax.contourf(grid_x, grid_z, grid, levels=levels, norm=norm,
                             cmap=show_kw.get("cMap", "Spectral_r"), extend="both")
        if self._show_mesh.isChecked():
            try:
                import pygimli as pg

                pg.viewer.mpl.drawMeshBoundaries(ax, mesh, hideMesh=False)
            except Exception:  # noqa: BLE001 - the overlay is optional
                pass
        ax.set_xlim(float(xi.min()), float(xi.max()))
        ax.set_ylim(float(zi.min()), float(np.nanmax(top)))
        return filled

    @staticmethod
    def _set_contour_ticks(cbar, log_scale: bool) -> None:
        """Put round numbers on a filled-contour colorbar."""
        import numpy as np
        from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter

        try:
            lo, hi = cbar.mappable.get_clim()
            if log_scale and lo > 0.0:
                ticks = LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)).tick_values(lo, hi)
            else:
                ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
            ticks = [t for t in np.asarray(ticks, dtype=float) if lo <= t <= hi]
            if len(ticks) >= 2:
                cbar.set_ticks(ticks)
                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                cbar.ax.yaxis.set_major_formatter(formatter)
        except Exception:  # noqa: BLE001 - ticks are cosmetic, never fatal
            pass

    def _on_length_unit_changed(self, _unit: str) -> None:
        """Redraw the axes in the studio's new length unit."""
        self._redraw()

    def _on_smooth_changed(self, _index: int = 0) -> None:
        """Show the contour-band count only while contours are being drawn."""
        self._levels.setVisible(int(self._smooth.currentData() or 0) < 0)
        self._redraw()

    def _on_colormap_changed(self, _name: str) -> None:
        """Redraw what is on screen in the new colours; nothing is reloaded."""
        self._redraw()

    @property
    def colormap_chooser(self) -> "cmaps.ColormapChooser":
        """The chooser beside the section, for callers that set it directly."""
        return self._colormap

    def _clip_polygon(self, mesh, mask):
        """Envelope polygon for the clean cut, or None if it cannot be built.

        Built on the pre-smoothing mesh, because that is the one ``mask`` belongs
        to; the polygon itself is in data coordinates and clips whichever mesh ends
        up being drawn.
        """
        try:
            from PyHydroGeophysX.core.section_geometry import coverage_envelope_polygon

            return coverage_envelope_polygon(
                mesh, mask, float(self._cov_threshold.value()))
        except Exception:  # noqa: BLE001 - the clip is cosmetic, never fatal
            return None

    # -- colour limits -------------------------------------------------------

    def _on_lock_toggled(self, locked: bool) -> None:
        self._cmin.setEnabled(locked)
        self._cmax.setEnabled(locked)
        self._redraw()

    def _on_limit_changed(self, _value: float) -> None:
        if self._lock_range.isChecked():
            self._redraw()

    def _set_limit_boxes(self, lo: float, hi: float) -> None:
        """Write the limit boxes without provoking a redraw from their signals."""
        span = abs(hi - lo)
        step = max(span / 50.0, 1.0e-6)
        for box, value in ((self._cmin, lo), (self._cmax, hi)):
            box.blockSignals(True)
            box.setSingleStep(step)
            box.setValue(float(value))
            box.blockSignals(False)

    @staticmethod
    def _symmetric_span(values):
        """Half-width of a zero-centred scale, robust to a few extreme cells."""
        import numpy as np
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        span = float(np.nanpercentile(np.abs(finite), 98.0))
        return span if span > 0.0 else None

    def _apply_color_limits(self, show_kw: dict, values, lockable: bool) -> None:
        """Honour a locked range, or track the displayed one when unlocked.

        Keeping the boxes in step with the autoscale while unlocked is what makes
        the checkbox mean "keep this": whatever is on screen is already in them.
        """
        import numpy as np

        if not self._lock_range.isChecked():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                lo = float(show_kw.get("cMin", np.nanmin(finite)))
                hi = float(show_kw.get("cMax", np.nanmax(finite)))
                if hi > lo:
                    self._set_limit_boxes(lo, hi)
            return
        if not lockable:
            # Locked, but this view is not the locked quantity. Leave the boxes
            # as they are: overwriting them here would hand the model back a
            # range in coverage units when the reader unticks "Sensitivity".
            return

        lo, hi = float(self._cmin.value()), float(self._cmax.value())
        if hi <= lo:
            return  # an inverted range would raise rather than draw
        if show_kw.get("logScale") and lo <= 0.0:
            # A log colour scale cannot start at or below zero; keep the upper
            # limit the user set and pull the lower one just above zero.
            positive = np.asarray(values, dtype=float)
            positive = positive[np.isfinite(positive) & (positive > 0.0)]
            lo = float(np.nanmin(positive)) if positive.size else hi / 1000.0
            if lo >= hi:
                return
        show_kw["cMin"], show_kw["cMax"] = lo, hi

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

        srt = self._kind == "srt"
        if coverage is None:
            self._cov_note.setText("")
            for widget in (self._show_cov, self._mask_low, self._cov_threshold,
                           self._hide_uncovered, self._clean_cut):
                widget.setEnabled(False)
            return
        for widget in (self._show_cov, self._mask_low, self._cov_threshold):
            widget.setEnabled(True)
        self._hide_uncovered.setEnabled(True)
        # The clean cut reshapes a cut that is already being applied; on its own it
        # would have no boundary to follow.
        self._clean_cut.setEnabled(
            self._hide_uncovered.isChecked() if srt else self._mask_low.isChecked())
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
            + ("" if self._mask_low.isChecked() else "  Tick “Hide below” to apply it.")
            + ("  Clipped to the smooth envelope of that cut."
               if self._mask_low.isChecked() and self._clean_cut.isChecked() else ""))
