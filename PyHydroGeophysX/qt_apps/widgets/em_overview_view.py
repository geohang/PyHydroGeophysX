"""Map + section overview of an EM line inversion.

This is the panel a reader looks at first: where the soundings are, and what the
earth under the selected survey line looks like. It deliberately shows no
observed curves — those live in the Sounding tab — so the result reads as an
interpretation product rather than a QC plot.

The layout mirrors what field acquisition software (TEMcompany/TEM2Go) puts on
screen: a plan map with every sounding and the current line picked out, and
below it the position x depth resistivity section for that line with the
per-sounding data misfit drawn over it. A survey with several survey lines is
plotted one line at a time, because chaining the lines end to end puts an
artificial jump in the middle of the section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.inversion.em1d_lci import DOI_SENSITIVITY_THRESHOLD
from PyHydroGeophysX.visualization.basemap import (
    TILE_SOURCES,
    basemap_image,
    fit_local_transform,
)

#: Colours shared with the acquisition-software convention: every sounding in
#: black, the line being sectioned in red, its zero-distance end in blue.
_ALL_COLOR = "#111111"
_LINE_COLOR = "#d62728"
_START_COLOR = "#1f77b4"


class EMOverviewView(QWidget):
    """Plan map plus resistivity section for one line-inversion result."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(11.0, 7.4))
        self._canvas = FigureCanvasQTAgg(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setContentsMargins(6, 2, 6, 2)
        self._line_label = QLabel("Survey line:")
        self._line = QComboBox()
        self._line.setToolTip(
            "Which survey line to section. Lines are sectioned one at a time "
            "because their along-line distances are not continuous with each "
            "other; 'All lines' chains them and marks the joins.")
        self._line.currentIndexChanged.connect(self._redraw)
        row.addWidget(self._line_label)
        row.addWidget(self._line)
        row.addSpacing(18)
        self._basemap = QCheckBox("Basemap")
        self._basemap.setToolTip(
            "Draw map tiles under the soundings. Tiles are fetched once and cached "
            "on disk, so a line already looked at stays available offline. Needs "
            "the survey's longitude/latitude, which a TEMcompany project carries; "
            "the axes stay in projected metres, so distances are unaffected.")
        self._basemap.toggled.connect(self._on_basemap_toggled)
        self._basemap_source = QComboBox()
        self._basemap_source.addItems(list(TILE_SOURCES))
        self._basemap_source.setEnabled(False)
        self._basemap_source.currentIndexChanged.connect(self._on_basemap_toggled)
        row.addWidget(self._basemap)
        row.addWidget(self._basemap_source)
        row.addSpacing(18)
        # Same idea as the ERT view's coverage cut: how far down the data reach
        # is a judgement, so it is a dial on the picture rather than something
        # frozen into the result.
        self._below_doi = QComboBox()
        for label, key in (("Fade", "fade"), ("Hide", "hide"), ("Show", "show")):
            self._below_doi.addItem(f"Below DOI: {label}", key)
        self._below_doi.setToolTip(
            "What to do with the cells the data do not constrain. The cut is the "
            "depth of investigation: below it, moving the whole remaining column "
            "by one decade in resistivity would not move the predicted response by "
            "more than the threshold, counted in the gates' own error bars.\n\n"
            "Fade keeps the model visible but washed out, so the section stays "
            "continuous and it is still obvious which part is regularization. "
            "Hide blanks it. Show draws everything at full strength, which is only "
            "honest with the depth-of-investigation line to read it against.")
        self._below_doi.currentIndexChanged.connect(self._redraw)
        self._style = QComboBox()
        for label, key in (("Smooth", "smooth"), ("Layer cells", "cells")):
            self._style.addItem(label, key)
        self._style.setToolTip(
            "Smooth interpolates the recovered models between soundings and down "
            "each layer stack, the way acquisition software draws a section: the "
            "structure reads continuously instead of as a wall of blocks. It adds "
            "no resolution, and the exported model is unchanged.\n\n"
            "Layer cells draws one rectangle per layer per sounding, which is what "
            "the inversion actually solved for.")
        self._style.currentIndexChanged.connect(self._redraw)
        self._doi_threshold = QDoubleSpinBox()
        self._doi_threshold.setRange(0.5, 500.0)
        self._doi_threshold.setDecimals(1)
        self._doi_threshold.setSingleStep(5.0)
        self._doi_threshold.setValue(float(DOI_SENSITIVITY_THRESHOLD))
        self._doi_threshold.setPrefix("σ ≥ ")
        self._doi_threshold.setToolTip(
            "Cumulated sensitivity a depth has to carry to be shown. Raise it for "
            "a more conservative section, lower it to see what the deeper part of "
            "the model looks like. The default reproduces the depths of "
            "investigation a TEMcompany project reported for its own survey to "
            "within about one layer thickness; anything from 20 to 30 sits within "
            "a few metres of that.")
        self._doi_threshold.valueChanged.connect(self._redraw)
        row.addWidget(self._style)
        row.addWidget(self._below_doi)
        row.addWidget(self._doi_threshold)
        row.addStretch(1)
        self._row = QWidget()
        self._row.setLayout(row)
        layout.addWidget(self._row)
        layout.addWidget(self._canvas, stretch=1)

        self._result: Optional[Dict[str, Any]] = None
        self._res: Optional[np.ndarray] = None      # (n_pos, n_layers) surface-ordered
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._sensitivity: Optional[np.ndarray] = None
        self._transform = None      # projected metres -> Web Mercator, or None
        self._tiles: Optional[Dict[str, Any]] = None

    # -- public --------------------------------------------------------------
    def show_result(self, result: Dict[str, Any], *, x=None, y=None,
                    lon=None, lat=None) -> None:
        """Display an :func:`PyHydroGeophysX.workflows.em1d.invert_line` result.

        ``x`` / ``y`` are the per-sounding map coordinates (easting/northing).
        Without them the map panel is dropped and the section uses the full
        figure, so a data file that carries no coordinates still gets a section.
        ``lon`` / ``lat`` for the same soundings enable the tile basemap; without
        them the basemap control is disabled and the map is drawn on its grid.
        """
        self._result = dict(result)
        model = np.asarray(result["model3d"], dtype=float)[:, 0, :]
        self._res = model[:, ::-1]                  # surface-ordered in depth
        n_pos = self._res.shape[0]
        sensitivity = np.asarray(result.get("sensitivity", []), dtype=float)
        self._sensitivity = (sensitivity if sensitivity.shape == self._res.shape
                             else None)
        self._below_doi.setEnabled(self._sensitivity is not None)
        self._doi_threshold.setEnabled(self._sensitivity is not None)
        if self._sensitivity is not None and "doi_threshold" in result:
            self._doi_threshold.blockSignals(True)
            self._doi_threshold.setValue(float(result["doi_threshold"]))
            self._doi_threshold.blockSignals(False)
        self._x = self._coordinate(x, n_pos)
        self._y = self._coordinate(y, n_pos)
        self._tiles = None
        self._transform = None
        if self._x is not None and self._y is not None:
            longitude = self._coordinate(lon, n_pos)
            latitude = self._coordinate(lat, n_pos)
            if longitude is not None and latitude is not None:
                self._transform = fit_local_transform(
                    self._x, self._y, longitude, latitude)
        available = self._transform is not None
        self._basemap.setEnabled(available)
        if not available:
            self._basemap.blockSignals(True)
            self._basemap.setChecked(False)
            self._basemap.blockSignals(False)
            self._basemap.setToolTip(
                "No longitude/latitude for these soundings, so map tiles cannot "
                "be placed. A TEMcompany project folder carries them.")

        lines = np.asarray(result.get("line_numbers", []), dtype=int).ravel()
        if lines.size < n_pos:
            lines = np.zeros(n_pos, dtype=int)
        self._result["line_numbers"] = lines[:n_pos]
        unique = np.unique(self._result["line_numbers"])

        self._line.blockSignals(True)
        self._line.clear()
        for value in unique:
            self._line.addItem(f"Line {int(value)}", int(value))
        if unique.size > 1:
            self._line.addItem("All lines", -1)
        self._line.setCurrentIndex(0)
        self._line.blockSignals(False)
        self._line.setVisible(unique.size > 1)
        self._line_label.setVisible(unique.size > 1)
        self._row.setVisible(unique.size > 1 or available)
        self._redraw()

    def _on_basemap_toggled(self, *_) -> None:
        """Fetching tiles blocks, so only do it when the map is actually wanted."""
        self._basemap_source.setEnabled(self._basemap.isChecked())
        self._tiles = None
        self._redraw()

    def save_figure(self, path) -> Optional[str]:
        """Write the current figure to *path*; return the path, or None."""
        if self._result is None:
            return None
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(target, dpi=160)
            return str(target)
        except Exception:  # noqa: BLE001 - export is a convenience, not the result
            return None

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _coordinate(values, n_pos: int) -> Optional[np.ndarray]:
        if values is None:
            return None
        array = np.asarray(values, dtype=float).ravel()
        if array.size < n_pos or not np.isfinite(array[:n_pos]).all():
            return None
        return array[:n_pos]

    def _selection(self) -> np.ndarray:
        lines = self._result["line_numbers"]
        chosen = self._line.currentData()
        if chosen is None or int(chosen) < 0:
            return np.ones(lines.size, dtype=bool)
        return lines == int(chosen)

    @staticmethod
    def _cell_edges(distance: np.ndarray) -> np.ndarray:
        """Cell boundaries midway between soundings, half a step at each end."""
        if distance.size == 1:
            return np.array([distance[0] - 0.5, distance[0] + 0.5])
        step = float(np.median(np.diff(distance)))
        return np.concatenate([[distance[0] - step / 2.0],
                               0.5 * (distance[:-1] + distance[1:]),
                               [distance[-1] + step / 2.0]])

    # -- rendering -----------------------------------------------------------
    def _redraw(self, *_) -> None:
        from matplotlib.colors import LogNorm, Normalize
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        if self._result is None or self._res is None:
            return
        result = self._result
        selected = self._selection()
        if not selected.any():
            return
        res = self._masked_resistivity()[selected]
        lines = result["line_numbers"][selected]
        depth_edges = np.asarray(result["depth_edges"], dtype=float).ravel()
        depth_centre = 0.5 * (depth_edges[:-1] + depth_edges[1:])
        chi2 = np.asarray(result.get("chi2_list", []), dtype=float).ravel()
        chi2 = chi2[:selected.size][selected] if chi2.size >= selected.size else None
        positions = np.asarray(result["positions"], dtype=float).ravel()[:selected.size]
        distance = positions[selected] - float(positions[selected][0])
        edges = self._cell_edges(distance)
        has_map = self._x is not None and self._y is not None

        log_scale = bool(result.get("log_scale", True))
        vmin, vmax = self._colour_range(log_scale)
        norm = LogNorm(vmin, vmax) if log_scale else Normalize(vmin, vmax)

        self._fig.clear()
        if has_map:
            grid = self._fig.add_gridspec(
                2, 2, height_ratios=[1.0, 1.25], width_ratios=[1.5, 1.0],
                hspace=0.36, wspace=0.05,
                left=0.075, right=0.865, top=0.945, bottom=0.085)
            self._draw_map(self._fig.add_subplot(grid[0, 0]),
                           self._fig.add_subplot(grid[0, 1]), selected, chi2)
            ax = self._fig.add_subplot(grid[1, :])
        else:
            grid = self._fig.add_gridspec(
                1, 1, left=0.085, right=0.865, top=0.92, bottom=0.12)
            ax = self._fig.add_subplot(grid[0, 0])

        cmap = result.get("cmap", "turbo")
        doi = self._doi_depths()
        here = doi[:selected.size][selected] if doi is not None else None
        mode = self._below_doi_mode()
        fading = mode == "fade" and here is not None
        bottom = self._depth_limit(res if np.isfinite(res).any() else self._res[selected],
                                   depth_centre, depth_edges,
                                   margin=2.5 if fading else 1.3)
        if str(self._style.currentData()) == "smooth":
            self._draw_smooth(ax, distance, depth_centre, self._res[selected],
                              here if mode != "show" else None,
                              norm, cmap, bottom, hide=(mode == "hide"),
                              groups=lines)
        else:
            if fading:
                ax.pcolormesh(edges, depth_edges, self._res[selected].T, cmap=cmap,
                              norm=norm, shading="auto", alpha=0.22, zorder=1)
            ax.pcolormesh(edges, depth_edges, res.T, cmap=cmap,
                          norm=norm, shading="auto", zorder=2)
        from matplotlib.cm import ScalarMappable
        mesh = ScalarMappable(norm=norm, cmap=cmap)
        ax.set_xlim(edges[0], edges[-1])
        if here is not None and mode != "hide":
            self._draw_doi_line(ax, distance, here)
        ax.set_ylim(bottom, 0.0)
        ax.set_xlabel("Distance along line (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(self._section_title(selected, lines, distance), fontsize=11)
        for boundary in np.flatnonzero(np.diff(lines)) + 1:
            ax.axvline(0.5 * (distance[boundary - 1] + distance[boundary]),
                       color="white", lw=1.4, ls="--", alpha=0.9)
        self._hatch_unconstrained(ax, selected, edges)

        box = ax.get_position()
        cax = self._fig.add_axes([0.895, box.y0, 0.018, box.height])
        bar = self._fig.colorbar(mesh, cax=cax, extend="both", ax=ax)
        bar.set_label(result.get("label", "resistivity (Ω·m)"))
        if log_scale:
            bar.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 3.0)))
            bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            bar.ax.yaxis.set_minor_formatter(NullFormatter())

        self._canvas.draw_idle()

    def _draw_basemap(self, ax, x0: float, y0: float) -> str:
        """Put map tiles under the survey; return the attribution to display.

        The axes limits are settled first (a margin around the soundings) so the
        imagery is fetched for the extent that will actually be shown, and the
        limits are then pinned: ``imshow`` would otherwise widen them and leave
        the map floating in a larger frame on the next draw.
        """
        if not self._basemap.isChecked() or self._transform is None:
            return ""
        x, y = self._x - x0, self._y - y0
        pad = 0.12 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
        x_limits = (float(x.min()) - pad, float(x.max()) + pad)
        y_limits = (float(y.min()) - pad, float(y.max()) + pad)
        a, b = self._transform
        # The fit was made in absolute projected metres; the axes are offset.
        shifted = (a, b + a * complex(x0, y0))
        if self._tiles is None:
            self._tiles = basemap_image(
                x_limits, y_limits, transform=shifted,
                source=self._basemap_source.currentText())
        if not self._tiles:
            return ""
        ax.imshow(self._tiles["image"], extent=self._tiles["extent"],
                  origin="upper", interpolation="bilinear", zorder=0)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        return str(self._tiles.get("attribution", ""))

    def _hatch_unconstrained(self, ax, selected: np.ndarray,
                             edges: np.ndarray) -> None:
        """Hatch the soundings that contributed no data of their own.

        Outlier rejection can take every gate of a station, and a station can
        arrive with too few gates to fit. Its column is still drawn, because the
        lateral constraint gives it a model, but that model came from its
        neighbours and not from a measurement under it. Colour alone would not
        say so.
        """
        from matplotlib.patches import Rectangle

        counts = np.asarray(self._result.get("data_count_list") or [], dtype=float)
        if counts.size < selected.size:
            return
        empty = counts[:selected.size][selected] <= 0
        if not empty.any():
            return
        bottom, top = ax.get_ylim()
        for index in np.flatnonzero(empty):
            ax.add_patch(Rectangle(
                (edges[index], min(bottom, top)),
                edges[index + 1] - edges[index], abs(top - bottom),
                facecolor="none", edgecolor="white", hatch="///",
                lw=0.0, alpha=0.55, zorder=4))

    def _below_doi_mode(self) -> str:
        return str(self._below_doi.currentData() or "fade")

    def _masked_resistivity(self) -> np.ndarray:
        """The model with the poorly constrained cells taken out.

        The inversion hands back the whole model plus the cumulated sensitivity
        behind it, so the cut is applied here and can be moved without solving
        again. Falls back to whatever the inversion already blanked when no
        sensitivity came with the result.
        """
        if self._sensitivity is None or self._below_doi_mode() == "show":
            return self._res
        masked = self._res.copy()
        masked[self._sensitivity < float(self._doi_threshold.value())] = np.nan
        return masked

    def _doi_depths(self) -> Optional[np.ndarray]:
        """Per-sounding depth of investigation at the threshold now selected."""
        if self._sensitivity is None or self._result is None:
            return None
        edges = np.asarray(self._result["depth_edges"], dtype=float).ravel()
        threshold = float(self._doi_threshold.value())
        depths = np.zeros(self._sensitivity.shape[0], dtype=float)
        for index, row in enumerate(self._sensitivity):
            resolved = np.flatnonzero(row >= threshold)
            depths[index] = (float(edges[min(resolved[-1] + 1, edges.size - 1)])
                             if resolved.size else 0.0)
        return depths

    def _colour_range(self, log_scale: bool) -> "tuple[float, float]":
        """One colour scale for the whole survey, robust to railed cells.

        Every line is drawn on the same scale, so a section that looks
        conductive really is more conductive than its neighbour rather than
        merely autoscaled differently. Percentiles keep a handful of cells stuck
        at the resistivity bound from flattening everything else, and the span is
        capped at four decades for the same reason.

        Taken from the cells that are actually drawn: an unconstrained deep cell
        railed at the resistivity bound would otherwise set the top of a scale
        nothing on the picture reaches.
        """
        shown = self._masked_resistivity()
        finite = shown[np.isfinite(shown)]
        if not finite.size:
            return (1.0, 100.0)
        vmin = float(np.percentile(finite, 5))
        vmax = float(np.percentile(finite, 95))
        if log_scale:
            vmin = max(vmin, 1e-3)
            vmax = min(max(vmax, vmin * 1.5), vmin * 1e4)
        else:
            vmax = max(vmax, vmin + 1.0)
        return (vmin, vmax)

    #: Opacity of the part of the model the data do not resolve.
    _FADE_ALPHA = 0.30

    @staticmethod
    def _draw_smooth(ax, distance: np.ndarray, depth_centre: np.ndarray,
                     res: np.ndarray, doi: Optional[np.ndarray],
                     norm, cmap, bottom: float, *, hide: bool = False,
                     groups: Optional[np.ndarray] = None) -> None:
        """Draw the section as a continuous image rather than a wall of blocks.

        Resistivity is interpolated in log10, which is the quantity the inversion
        solves for and the scale the colours use, first down each sounding's
        layer stack and then across the line. Adding no resolution, only removing
        the staircase that the fixed layer grid and the station spacing put on
        the picture.

        Building the image as RGBA rather than drawing two meshes lets the part
        below the depth of investigation fade out smoothly with everything else,
        instead of showing a hard edge where one mesh stops and the other starts.
        """
        from matplotlib import colormaps

        if distance.size < 2 or not np.isfinite(res).any():
            return
        width, height = 700, 420
        xs = np.linspace(float(distance[0]), float(distance[-1]), width)
        zs = np.linspace(float(depth_centre[0]), float(bottom), height)

        # Down each sounding first: only its own finite layers take part, so a
        # blanked cell does not drag its neighbours towards it.
        columns = np.full((res.shape[0], height), np.nan, dtype=float)
        for index, row in enumerate(res):
            finite = np.isfinite(row) & (row > 0)
            if finite.sum() >= 2:
                columns[index] = np.interp(zs, depth_centre[finite],
                                           np.log10(row[finite]))
            elif finite.sum() == 1:
                columns[index] = np.log10(row[finite][0])
        usable = np.flatnonzero(np.isfinite(columns).any(axis=1))
        if usable.size < 2:
            return
        # Then across the line, one survey line at a time. Interpolating over a
        # join would draw a smooth transition between two places that are not
        # next to each other, which is the one thing a continuous fill must not
        # invent. The gap between lines stays empty.
        if groups is None or np.size(groups) != distance.size:
            groups = np.zeros(distance.size, dtype=int)
        groups = np.asarray(groups).ravel()
        grid = np.full((height, width), np.nan, dtype=float)
        for value in np.unique(groups[usable]):
            members = usable[groups[usable] == value]
            if not members.size:
                continue
            positions = distance[members]
            pad = (0.5 * float(np.median(np.diff(positions)))
                   if positions.size > 1 else 0.5)
            span = ((xs >= positions[0] - pad) & (xs <= positions[-1] + pad))
            if not span.any():
                continue
            for level in range(height):
                values = columns[members, level]
                good = np.isfinite(values)
                if good.sum() >= 2:
                    grid[level, span] = np.interp(
                        xs[span], positions[good], values[good])
                elif good.sum() == 1:
                    grid[level, span] = float(values[good][0])

        rgba = colormaps[cmap](norm(np.power(10.0, grid)))
        rgba[..., 3] = np.where(np.isfinite(grid), 1.0, 0.0)
        if doi is not None and np.isfinite(doi).any():
            # Nearest sounding rather than a ramp between two, so the edge of the
            # faded band sits exactly under the stepped depth-of-investigation
            # line instead of cutting diagonally across it.
            nearest = np.abs(xs[:, None] - distance[None, :]).argmin(axis=1)
            reach = np.nan_to_num(doi)[nearest]
            below = zs[:, None] > reach[None, :]
            rgba[..., 3] *= np.where(
                below, 0.0 if hide else EMOverviewView._FADE_ALPHA, 1.0)
        ax.imshow(rgba, extent=(xs[0], xs[-1], zs[-1], zs[0]), origin="upper",
                  aspect="auto", interpolation="bilinear", zorder=1)

    @staticmethod
    def _draw_doi_line(ax, distance: np.ndarray, doi: np.ndarray) -> None:
        """Where the data stop constraining the model, as a step per sounding.

        Drawn as a step rather than a smooth curve because the depth of
        investigation belongs to a sounding, not to the distance between two.
        """
        from matplotlib import patheffects

        if not np.isfinite(doi).any():
            return
        ax.step(distance, doi, where="mid", color="black", lw=1.2, zorder=5,
                path_effects=[patheffects.withStroke(linewidth=2.8,
                                                     foreground="white", alpha=0.85)])

    @staticmethod
    def _depth_limit(res: np.ndarray, depth_centre: np.ndarray,
                     depth_edges: np.ndarray, *, margin: float = 1.3) -> float:
        """Crop the depth axis to a little past where the soundings resolve.

        The layer grid runs far deeper than a ground TDEM system sees, so drawing
        all of it would leave most of the panel empty. The margin is larger when
        the unresolved part is being faded rather than blanked, to leave that
        band something to occupy, and there is a floor so a line that resolves
        almost nothing still gets a readable axis instead of collapsing to a
        centimetre scale.
        """
        reach = [float(depth_centre[np.isfinite(row)].max())
                 for row in res if np.isfinite(row).any()]
        if not reach:
            return float(depth_edges[-1])
        limit = float(np.percentile(reach, 90)) * float(margin)
        return float(min(float(depth_edges[-1]), max(limit, 10.0)))

    def _section_title(self, selected: np.ndarray, lines: np.ndarray,
                       distance: np.ndarray) -> str:
        chosen = self._line.currentData()
        which = ("all lines" if chosen is None or int(chosen) < 0
                 else f"Line {int(chosen)}")
        return (f"Resistivity section — {which}   "
                f"({int(selected.sum())} soundings, {float(distance[-1]):.0f} m)")

    def _draw_map(self, ax, panel, selected: np.ndarray, chi2) -> None:
        x, y = self._x, self._y
        x0, y0 = float(np.nanmin(x)), float(np.nanmin(y))
        ax.set_aspect("equal", "box")
        # Imagery first: it sets the extent the markers are then drawn into, and
        # it decides how the markers have to be styled to stay readable.
        attribution = self._draw_basemap(ax, x0, y0)
        # The sectioned line is a wide translucent band under the markers, so
        # every sounding stays visible even when the whole survey is sectioned.
        ax.plot(x[selected] - x0, y[selected] - y0, "-",
                color=_LINE_COLOR, lw=6.0, alpha=0.55 if attribution else 0.40,
                solid_capstyle="round", zorder=2, label="On the section")
        ax.plot(x - x0, y - y0, "o", color=_ALL_COLOR, ms=3.4,
                mec="white" if attribution else "none",
                mew=0.7 if attribution else 0.0,
                label="All soundings", zorder=3, ls="none")
        ax.plot(x[selected][0] - x0, y[selected][0] - y0, "s", color=_START_COLOR,
                ms=9, mec="white", mew=0.8, label="Section start (0 m)", zorder=4)
        ax.set_xlabel(f"Easting − {x0:,.0f} m", fontsize=9)
        ax.set_ylabel(f"Northing − {y0:,.0f} m", fontsize=9)
        ax.set_title("Survey map", fontsize=11)
        ax.tick_params(labelsize=8)
        # Grid lines help on a plain background and only clutter imagery.
        ax.grid(not attribution, alpha=0.3)
        if attribution:
            ax.text(0.99, 0.01, attribution, transform=ax.transAxes, fontsize=6.5,
                    ha="right", va="bottom", color="white",
                    bbox=dict(facecolor="black", alpha=0.35, pad=1.5,
                              edgecolor="none"))

        panel.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        panel.legend(handles, labels, loc="upper left", fontsize=9, frameon=False,
                     bbox_to_anchor=(0.0, 1.0))
        panel.text(0.0, 0.56, self._summary(selected, chi2), fontsize=8.5,
                   va="top", ha="left", color="#333333", linespacing=1.8,
                   transform=panel.transAxes)

    def _summary(self, selected: np.ndarray, chi2) -> str:
        result = self._result
        report = result.get("lci_report") or {}
        coupling = {"simultaneous": "simultaneous LCI",
                    "sequential": "block-coordinate LCI"}.get(
                        str(report.get("mode", "")), "independent 1D")
        rows = [
            f"{result.get('method', 'EM')}"
            + (" · joint LM+HM" if result.get("joint_moments") else "")
            + f" · {coupling}",
            f"soundings: {int(selected.sum())} of {selected.size}",
            f"layers: {int(result.get('n_layers', self._res.shape[1]))}"
            f"   grid depth: 0–{float(result['depth_edges'][-1]):.0f} m",
        ]
        if chi2 is not None and np.isfinite(chi2).any():
            # One number only. How the fit varies along the line belongs on the
            # Inversion quality page, where it can be read against the
            # convergence history instead of covering the model.
            rows.append(f"median χ² per sounding: {np.nanmedian(chi2):.1f}"
                        "   (see Inversion quality)")
        outliers = dict(result.get("outliers") or {})
        if outliers.get("enabled"):
            rows.append(f"gates kept: {outliers.get('kept')} of {outliers.get('n_start')}"
                        f" (cut beyond {float(outliers.get('threshold', 0)):g}σ)")
        counts = np.asarray(result.get("data_count_list") or [], dtype=float)
        empty = int((counts[:selected.size][selected] <= 0).sum()) if counts.size >= selected.size else 0
        vmin, vmax = self._colour_range(bool(result.get("log_scale", True)))
        rows.append(f"colour range: {vmin:.4g}–{vmax:.4g} Ω·m, shared by every line")
        doi = self._doi_depths()
        if doi is not None and self._below_doi_mode() != "show":
            # Quote the depth over the soundings that have one. Averaging in the
            # stations that were left with no data would report a shallow survey
            # when what actually happened is that some columns hold nothing.
            here = doi[:selected.size][selected]
            resolved = here[here > 0]
            verb = "faded" if self._below_doi_mode() == "fade" else "blanked"
            if resolved.size:
                rows.append(
                    f"{verb} below the DOI (σ ≥ {self._doi_threshold.value():g}): "
                    f"median {np.median(resolved):.0f} m, "
                    f"{np.percentile(resolved, 10):.0f}–"
                    f"{np.percentile(resolved, 90):.0f} m")
            blank = int((here <= 0).sum())
            if blank:
                rows.append(
                    f"{blank} of {here.size} column(s) resolve nothing at this "
                    "threshold; lower it to see what they hold")
        elif doi is None:
            rows.append("cells below the depth of investigation are left blank")
        if empty:
            rows.append(f"hatched: {empty} sounding(s) with no data of their own")
        return "\n".join(rows)
