"""Plan-view maps of a sounding survey: resistivity at chosen depths, between the soundings.

A section shows one line; the question "where is the conductor?" is answered in
plan. Each map here is one depth below ground, kriged from the soundings that
resolve that depth and filled across the survey's outline, the gaps between
lines included. The kriging is a picture of the soundings, not a measurement
between them, so every map comes with the kriging standard deviation of each
cell, drawn as a map of its own: it is small beside a sounding and grows into
the gaps and wherever the soundings do not reach that depth.

The kriging is ordinary kriging (Matheron, 1963) with an isotropic variogram
fitted per depth, done by :mod:`PyHydroGeophysX.core.plan_interpolation`, the
package's own NumPy/SciPy kriging, so it needs no geostatistics package.

Matheron, G. (1963). Principles of geostatistics. Economic Geology 58, 1246-1266.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: Depths (m) a map is drawn at, of which the resolved ones are offered.
CANDIDATE_DEPTHS_M = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
                      60.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0)

#: A depth is mapped when at least this share of the soundings resolve it.
MIN_RESOLVED_SHARE = 0.3


def _bottoms(models: np.ndarray, depth_edges: np.ndarray) -> np.ndarray:
    """Depth to the base of each sounding's deepest resolved (finite) layer."""
    bottoms = np.full(models.shape[0], np.nan)
    for s, row in enumerate(models):
        finite = np.flatnonzero(np.isfinite(row) & (row > 0))
        if finite.size:
            bottoms[s] = depth_edges[min(finite[-1] + 1, depth_edges.size - 1)]
    return bottoms


def choose_depths(models: np.ndarray, depth_edges: np.ndarray, count: int = 6) -> List[float]:
    """Up to ``count`` depths that enough soundings resolve, spread from shallow to deep.

    Examples
    --------
    >>> edges = np.array([0., 1., 2., 4., 8., 16., 32.])
    >>> models = np.tile([10., 20., 30., 40., 50., np.nan], (5, 1))
    >>> choose_depths(models, edges, count=4)
    [1.0, 3.0, 7.5, 15.0]
    """
    models = np.asarray(models, float)
    edges = np.asarray(depth_edges, float).ravel()
    bottoms = _bottoms(models, edges)
    reach = bottoms[np.isfinite(bottoms)]
    if not reach.size:
        return []
    usable = [d for d in CANDIDATE_DEPTHS_M
              if edges[0] < d < edges[-1] and np.mean(reach > d) >= MIN_RESOLVED_SHARE]
    if len(usable) <= count:
        return [float(d) for d in usable]
    picks = np.unique(np.round(np.linspace(0, len(usable) - 1, count)).astype(int))
    return [float(usable[i]) for i in picks]


def masking_distance(x: np.ndarray, y: np.ndarray,
                     lines: Optional[np.ndarray] = None) -> float:
    """The survey's spacing scale: 0.6 of the wider gaps between lines.

    It sets the variogram's lag window (:func:`depth_variogram`) and the margin
    around a map, and it is the distance a map is blanked beyond when
    :func:`resistivity_depth_slices` is asked to blank at all. The gap is the
    distance from a sounding to the nearest sounding on another line, taken at
    its 90th percentile: a walked survey runs its lines in pairs, and on one
    TEM2Go day the median gap was the 11 m inside a pair while pairs stood
    20 m apart. Without line numbers, or on one line, it is ten along-line
    spacings. Never under 5 m.

    Examples
    --------
    >>> x = np.r_[np.arange(10.), np.arange(10.)]; y = np.r_[np.zeros(10), np.full(10, 30.)]
    >>> masking_distance(x, y, np.r_[np.ones(10), 2 * np.ones(10)])
    18.0
    """
    from scipy.spatial import cKDTree

    points = np.column_stack([x, y])
    nearest, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(nearest[:, 1])) if len(points) > 1 else 1.0
    gap = None
    if lines is not None and len(np.unique(lines)) > 1:
        gaps = []
        for line in np.unique(lines):
            mine, others = lines == line, lines != line
            distance, _ = cKDTree(points[others]).query(points[mine])
            gaps.append(distance)
        gap = float(np.percentile(np.concatenate(gaps), 90))
    distance = 0.6 * gap if gap is not None else 10.0 * spacing
    return float(max(distance, 5.0))


def depth_variogram(points: np.ndarray, logs: np.ndarray, reach: float) -> Dict[str, Any]:
    """The variogram of log10 resistivity at one depth, fitted when the pairs allow it.

    Fitted by :func:`PyHydroGeophysX.core.plan_interpolation.fit_variogram`
    (spherical, exponential or Gaussian with a nugget, pair-count weighted least
    squares, the lowest misfit kept) to lags out to eight masking distances, the
    separations that decide a node's estimate, or half the survey's extent when
    that is shorter. Half the extent alone, the usual limit, puts every pair
    that matters into the first bin or two of a survey kilometres long. When
    the soundings give too few lags, the model is an exponential one with no
    nugget, the sample variance as its sill and three masking distances as its
    range, and ``fitted`` is False.

    >>> pts = np.column_stack([np.repeat(np.arange(10.0) * 5, 10), np.tile(np.arange(10.0) * 5, 10)])
    >>> fit = depth_variogram(pts, np.sin(pts[:, 0] / 15) + np.cos(pts[:, 1] / 15), 5.0)
    >>> fit["fitted"], fit["model"] in ("spherical", "exponential", "gaussian"), fit["sill"] > fit["nugget"]
    (True, True, True)
    """
    from scipy.spatial.distance import pdist

    from PyHydroGeophysX.core.plan_interpolation import empirical_variogram, fit_variogram

    every = max(1, len(points) // 500)
    extent = 0.5 * float(pdist(points[::every]).max()) if len(points[::every]) > 1 else 0.0
    lag = min(8.0 * reach, extent) if extent > 0 else 8.0 * reach
    try:
        experimental = empirical_variogram(points, logs, max_lag=lag)
        fit = fit_variogram(experimental["lags"], experimental["gamma"], experimental["counts"])
    except ValueError:
        return {"model": "exponential", "nugget": 0.0, "sill": max(float(np.var(logs)), 1e-12),
                "range": 3.0 * reach, "fitted": False}
    return {"model": fit["model"], "nugget": float(fit["nugget"]), "sill": float(fit["sill"]),
            "range": float(fit["range"]), "rmse": float(fit["rmse"]), "fitted": True}


def resistivity_depth_slices(x: Sequence[float], y: Sequence[float], models: np.ndarray,
                             depth_edges: Sequence[float],
                             depths: Optional[Sequence[float]] = None, *,
                             lines: Optional[Sequence[int]] = None,
                             max_distance: Optional[float] = None,
                             cells: int = 220) -> Dict[str, Any]:
    """Resistivity grids at ``depths``, kriged across the survey's outline.

    ``models`` is (n_soundings, n_layers), surface first, in ohm-m, NaN below
    each sounding's depth of investigation, as a line inversion returns it.
    Each depth takes the layer that contains it at every sounding resolving
    it; a sounding below its depth of investigation there is left out rather
    than trusted. Log10 resistivity is ordinary-kriged (Matheron, 1963) by
    :func:`PyHydroGeophysX.core.plan_interpolation.ordinary_kriging` with a
    variogram fitted to that depth's soundings (:func:`depth_variogram`), so
    the weight a sounding gets follows how far resistivity is seen to vary over
    that distance on this survey.

    Every node inside the convex hull of all the soundings is filled, the same
    outline at every depth, so the gaps between lines and the parts of a deep
    map no sounding reaches carry the kriging estimate from farther soundings;
    the standard deviation says how weak it is there. Only outside the outline
    is a node left NaN. ``max_distance`` blanks, in addition, nodes farther
    than that from every sounding resolving the depth.

    Returns the grids in ohm-m under ``slices``, the kriging standard deviation
    of log10 resistivity, in decades, under ``std``, and each depth's variogram
    under ``variograms`` (None for a depth with nothing to map).

    Raises
    ------
    ValueError
        When fewer than three soundings carry coordinates, or they lie along a
        single straight line: a plan-view map needs soundings spread in two
        directions.
    """
    from scipy.spatial import Delaunay, cKDTree

    from PyHydroGeophysX.core.plan_interpolation import ordinary_kriging, prepare_samples

    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    models = np.asarray(models, float)
    edges = np.asarray(depth_edges, float).ravel()
    placed = np.isfinite(x) & np.isfinite(y)
    if placed.sum() < 3:
        raise ValueError("fewer than three soundings carry coordinates")
    centred = np.column_stack([x[placed] - x[placed].mean(), y[placed] - y[placed].mean()])
    spread = np.linalg.svd(centred, compute_uv=False)
    if spread[-1] < 1e-3 * spread[0] or spread[-1] < 1.0:
        raise ValueError("the soundings lie along one line, so a plan-view map would be "
                         "an extrapolation away from it")
    line_numbers = None if lines is None else np.asarray(lines).ravel()[placed]
    scale = masking_distance(x[placed], y[placed], line_numbers)
    depths = list(depths) if depths is not None else choose_depths(models[placed], edges)
    if not depths:
        raise ValueError("no depth is resolved by enough soundings to map")

    pad = 0.5 * scale
    x0, x1 = x[placed].min() - pad, x[placed].max() + pad
    y0, y1 = y[placed].min() - pad, y[placed].max() + pad
    cell = max(x1 - x0, y1 - y0) / float(cells)
    xs = x0 + (np.arange(int(np.ceil((x1 - x0) / cell))) + 0.5) * cell
    ys = y0 + (np.arange(int(np.ceil((y1 - y0) / cell))) + 0.5) * cell
    grid_x, grid_y = np.meshgrid(xs, ys)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    outline = Delaunay(np.column_stack([x[placed], y[placed]])).find_simplex(nodes) >= 0

    slices, spreads, variograms, used = [], [], [], []
    for depth in depths:
        layer = int(np.clip(np.searchsorted(edges, depth, side="right") - 1, 0, models.shape[1] - 1))
        values = models[:, layer]
        keep = placed & np.isfinite(values) & (values > 0)
        used.append(int(keep.sum()))
        surface = np.full(len(nodes), np.nan)
        error = np.full(len(nodes), np.nan)
        variogram = None
        if keep.sum() >= 3:
            points, logs = prepare_samples(np.column_stack([x[keep], y[keep]]),
                                           np.log10(values[keep]))
            wanted = outline.copy()
            if max_distance:
                wanted &= cKDTree(points).query(nodes)[0] <= float(max_distance)
            if len(points) >= 3 and wanted.any():
                variogram = depth_variogram(points, logs, scale)
                estimate, variance = ordinary_kriging(points, logs, nodes[wanted], variogram)
                surface[wanted], error[wanted] = estimate, np.sqrt(variance)
        slices.append(10.0 ** surface.reshape(grid_x.shape))
        spreads.append(error.reshape(grid_x.shape))
        variograms.append(variogram)
    return {"x": xs, "y": ys, "depths": [float(d) for d in depths], "slices": slices,
            "std": spreads, "variograms": variograms, "soundings_used": used,
            "max_distance": float(max_distance) if max_distance else None,
            "cell": float(cell),
            "method": "ordinary kriging of log10 resistivity, with a variogram fitted "
                      "to the soundings resolving each depth"}


def utm_lonlat(x: Sequence[float], y: Sequence[float], coordinate_system: str):
    """Longitude and latitude of UTM ``(x, y)``, or None when that is not what they are.

    ``coordinate_system`` is read as "UTM zone 15T" or "UTM zone 15N"; a band
    letter from N up, or none, is the northern hemisphere. A survey's own
    longitude and latitude are not always the same point as its ``(x, y)``: a
    TEM2Go project places each sounding midway between loop and coil and
    records the GPS fix of the instrument, 7 m apart on one survey and on
    either side as the lines run back and forth. Converting the sounding
    coordinates themselves is what keeps a basemap registered to them. Needs
    pyproj; None without it.

    >>> lon, lat = utm_lonlat([617680.0], [4613620.0], "UTM zone 15T")
    >>> round(float(lon[0]), 3), round(float(lat[0]), 3)
    (-91.586, 41.666)
    >>> utm_lonlat([1.0], [2.0], "local metric (tangent plane)") is None
    True
    """
    import re

    match = re.search(r"UTM\s*zone\s*(\d{1,2})\s*([A-Za-z])?", str(coordinate_system or ""))
    if match is None:
        return None
    try:
        from pyproj import Transformer
    except ImportError:
        return None
    zone, band = int(match.group(1)), (match.group(2) or "N").upper()
    epsg = (32600 if band >= "N" else 32700) + zone
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
        np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon, float), np.asarray(lat, float)


def write_ascii_grids(grids: Dict[str, Any], folder: Any, stem: str = "resistivity") -> List[str]:
    """Each depth slice as an ESRI ASCII grid (``.asc``), which any GIS opens.

    Resistivity (ohm-m) goes to ``<stem>_<depth>m.asc`` and, when the grids
    carry it, the kriging standard deviation of log10 resistivity (decades) to
    ``<stem>_<depth>m_std.asc``. Cells are square, the lower-left corner is the
    grid's first node less half a cell, and masked cells are written as -9999.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cell = float(grids["cell"])
    header = ("ncols {}\nnrows {}\n"
              f"xllcorner {grids['x'][0] - 0.5 * cell:.3f}\n"
              f"yllcorner {grids['y'][0] - 0.5 * cell:.3f}\n"
              f"cellsize {cell:.4f}\nNODATA_value -9999\n")
    layers = [("", grids["slices"])]
    if grids.get("std"):
        layers.append(("_std", grids["std"]))
    written = []
    for suffix, stack in layers:
        for depth, values in zip(grids["depths"], stack):
            path = folder / f"{stem}_{depth:g}m{suffix}.asc"
            rows = np.where(np.isfinite(values), values, -9999.0)[::-1]   # north row first
            with path.open("w", encoding="ascii") as stream:
                stream.write(header.format(rows.shape[1], rows.shape[0]))
                np.savetxt(stream, rows, fmt="%.4g")
            written.append(str(path))
    return written


def _scale_bar(ax, unit: str) -> None:
    """A round-numbered scale bar in the lower left, in the figure's length unit."""
    from PyHydroGeophysX.visualization.axis_units import length_factor

    factor = length_factor(unit)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    target = 0.25 * (x1 - x0) * factor
    magnitude = 10 ** np.floor(np.log10(target))
    length = max(m for m in (1, 2, 5, 10) if m * magnitude <= target) * magnitude
    start_x, start_y = x0 + 0.05 * (x1 - x0), y0 + 0.06 * (y1 - y0)
    ax.plot([start_x, start_x + length / factor], [start_y, start_y], color="black", lw=3,
            solid_capstyle="butt")
    ax.plot([start_x, start_x + length / factor], [start_y, start_y], color="white", lw=1.2,
            solid_capstyle="butt")
    ax.text(start_x + 0.5 * length / factor, start_y + 0.02 * (y1 - y0), f"{length:g} {unit}",
            ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def _plan_panels(grids: Dict[str, Any], stack: Sequence[np.ndarray], x: Sequence[float],
                 y: Sequence[float], path: Any, *, basemap, unit: str, norm, cmap: str,
                 label: str, notes: Sequence[str], title: str, panel_size: float,
                 secondary=None, alpha: float = 0.78) -> str:
    """One map per depth of ``stack`` on a common colour scale, over ``basemap``.

    ``alpha`` is the maps' opacity over the basemap; the footer ``notes`` are
    wrapped to the figure's width, with room kept for them below the panels.
    """
    import textwrap

    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    from PyHydroGeophysX.visualization.axis_units import length_factor

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    factor = length_factor(unit)
    count = len(grids["depths"])
    cols = min(3, count)
    rows = int(np.ceil(count / cols))
    width = max(x.max() - x.min(), 1.0)
    height = max(y.max() - y.min(), 1.0)
    aspect = min(max(height / width, 0.4), 2.0)
    size = (panel_size * cols + 1.2, panel_size * aspect * rows + 0.9)
    # About 17 characters an inch at 7 pt.
    footer = textwrap.fill("  ".join(n for n in notes if n), width=int(17 * size[0]))
    lines = footer.count("\n") + 1
    fig = Figure(figsize=(size[0], size[1] + 0.12 * (lines - 1)))
    fig.set_layout_engine("constrained",
                          rect=(0, (0.12 * lines + 0.04) / fig.get_figheight(), 1, 1))
    axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False)).ravel()
    image = None
    ticks = FuncFormatter(lambda value, _pos: f"{value * factor:,.0f}")
    for ax, depth, values in zip(axes, grids["depths"], stack):
        if basemap is not None:
            ax.imshow(basemap["image"], extent=basemap["extent"], origin="upper",
                      interpolation="bilinear", zorder=0)
        image = ax.pcolormesh(grids["x"], grids["y"], np.ma.masked_invalid(values),
                              shading="nearest", cmap=cmap, norm=norm,
                              alpha=alpha if basemap is not None else 1.0, zorder=1)
        ax.plot(x, y, ".", color="black", ms=1.5, alpha=0.6, zorder=2)
        ax.set_xlim(grids["x"][0], grids["x"][-1])
        ax.set_ylim(grids["y"][0], grids["y"][-1])
        ax.set_aspect("equal")
        # Three figures: 1 m is 3.28 ft, not 3.28084.
        ax.set_title(f"{depth * factor:.3g} {unit} below ground", fontsize=10)
        # Full projected coordinates run to seven digits; more than four
        # of them along an axis overlap.
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticks)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(f"Easting ({unit})", fontsize=8)
        ax.set_ylabel(f"Northing ({unit})", fontsize=8)
        ax.text(0.97, 0.95, "N\n↑", transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        _scale_bar(ax, unit)
    for ax in axes[count:]:
        ax.set_visible(False)
    bar = fig.colorbar(image, ax=axes[:count].tolist(), label=label, shrink=0.85)
    if secondary is not None:
        secondary(bar)
    fig.text(0.01, 0.004, footer, fontsize=7, ha="left", va="bottom")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(str(path), dpi=200)
    return str(path)


def _footer(basemap, coordinate_label: str) -> List[str]:
    """What every map's footer says about the coordinates and the basemap."""
    notes = [coordinate_label]
    if basemap is not None and basemap.get("attribution"):
        notes.append(str(basemap["attribution"]))
    return notes


def plot_resistivity_depth_slices(grids: Dict[str, Any], x: Sequence[float], y: Sequence[float],
                                  path: Any, *, basemap: Optional[Dict[str, Any]] = None,
                                  unit: str = "m", norm=None, cmap: str = "turbo_r",
                                  title: str = "", panel_size: float = 4.2,
                                  coordinate_label: str = "") -> str:
    """One resistivity map per depth on a common colour scale, over ``basemap`` when given."""
    import matplotlib.colors as mcolors

    from PyHydroGeophysX.visualization.axis_units import length_factor

    values = np.concatenate([s[np.isfinite(s)] for s in grids["slices"]] or [np.array([1.0])])
    if norm is None:
        low, high = (np.percentile(values, [2, 98]) if values.size > 1 else (1.0, 10.0))
        norm = mcolors.LogNorm(vmin=float(low), vmax=float(max(high, low * 1.01)))
    if grids.get("max_distance"):
        method = (f"Kriged between soundings (log10 resistivity); blank beyond "
                  f"{grids['max_distance'] * length_factor(unit):.0f} {unit} from a sounding "
                  f"resolving that depth.")
    else:
        method = ("Kriged across the survey outline (log10 resistivity); soundings below "
                  "their depth of investigation are left out at that depth. How firm each "
                  "cell is: see the kriging uncertainty maps.")
    return _plan_panels(grids, grids["slices"], x, y, path, basemap=basemap, unit=unit,
                        norm=norm, cmap=cmap, label="Resistivity (ohm-m)",
                        notes=[method] + _footer(basemap, coordinate_label),
                        title=title, panel_size=panel_size)


def plot_kriging_uncertainty(grids: Dict[str, Any], x: Sequence[float], y: Sequence[float],
                             path: Any, *, basemap: Optional[Dict[str, Any]] = None,
                             unit: str = "m", cmap: str = "YlOrRd", title: str = "",
                             panel_size: float = 4.2, coordinate_label: str = "") -> str:
    """The kriging standard deviation of each depth map, on one scale for all depths.

    In decades of log10 resistivity, with the multiplicative factor it stands
    for (``10 ** std``) on the colour bar's other side: 0.1 decade is a factor
    of 1.26 either way.
    """
    import matplotlib.colors as mcolors

    values = np.concatenate([s[np.isfinite(s)] for s in grids["std"]] or [np.array([0.0])])
    top = float(np.percentile(values, 99)) if values.size > 1 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max(top, 1e-3))

    def factors(bar):
        side = bar.ax.secondary_yaxis("left", functions=(lambda s: s, lambda s: s))
        side.set_yticks(bar.get_ticks())
        side.set_yticklabels([f"×{10 ** t:.2f}" for t in bar.get_ticks()])
        side.tick_params(labelsize=7)

    notes = ["Kriging standard deviation of log10 resistivity: small beside a sounding "
             "resolving that depth, growing into the gaps between lines and where the "
             "soundings do not reach the depth. It is the interpolation's uncertainty "
             "only, on top of each sounding's model."]
    return _plan_panels(grids, grids["std"], x, y, path, basemap=basemap, unit=unit,
                        norm=norm, cmap=cmap,
                        label="Kriging std (decades); factor either way at left",
                        notes=notes + _footer(basemap, coordinate_label),
                        title=title, panel_size=panel_size, secondary=factors, alpha=0.9)


def survey_plan_grids(result: Dict[str, Any], *, source: Any = None,
                      basemap_file: Any = None, basemap: str = "auto") -> Dict[str, Any]:
    """The kriged depth maps of a survey's line inversion, and a basemap to draw them over.

    ``result`` is what :func:`PyHydroGeophysX.inversion.em1d_line.invert_line`
    returns (``model3d``), or a TDEM agent's survey results, whose
    ``recovered_resistivity`` is the same model surface first; either way NaN
    below each sounding's depth of investigation, with the soundings' ``x``,
    ``y``, ``depth_edges`` and ``line_numbers``, and ``coordinate_system``,
    ``longitude`` and ``latitude`` to place a basemap by.

    The basemap is ``basemap_file`` when given, else a georeferenced
    image kept beside ``source`` (a TEM2Go controller saves its field map under
    ``Maps``), else satellite tiles; ``basemap`` picks a tile source, or
    ``'none'`` for no basemap at all.

    Raises
    ------
    ValueError
        When the soundings carry no map coordinates or cannot make a plan-view
        map (see :func:`resistivity_depth_slices`).
    """
    from PyHydroGeophysX.visualization import basemap as maps

    model = (np.asarray(result["model3d"], dtype=float)[:, 0, ::-1] if "model3d" in result
             else np.atleast_2d(np.asarray(result["recovered_resistivity"], dtype=float)))
    x = np.asarray(result.get("x", []), dtype=float).ravel()
    y = np.asarray(result.get("y", []), dtype=float).ravel()
    if x.size != model.shape[0] or y.size != model.shape[0]:
        raise ValueError("the soundings carry no map coordinates")
    grids = resistivity_depth_slices(x, y, model, result["depth_edges"],
                                     lines=result.get("line_numbers"))
    background, choice = None, str(basemap or "auto")
    system = str(result.get("coordinate_system") or "")
    # The sounding points' own longitude and latitude where their grid is
    # known; the recorded GPS fixes otherwise (see utm_lonlat).
    converted = utm_lonlat(x, y, system)
    lon, lat = converted if converted is not None else (
        np.asarray(result.get("longitude", []), dtype=float).ravel(),
        np.asarray(result.get("latitude", []), dtype=float).ravel())
    transform = (maps.fit_local_transform(x, y, lon, lat)
                 if lon.size == x.size and lat.size == x.size else None)
    limits = ((grids["x"][0], grids["x"][-1]), (grids["y"][0], grids["y"][-1]))
    if choice != "none" and transform is not None:
        images = [basemap_file] if basemap_file else []
        if source:
            folder = Path(source) if Path(source).is_dir() else Path(source).parent
            images += maps.find_world_file_images(folder, folder.parent)
        # The image named first; one that does not cover this survey (a
        # neighbouring day's, say) gives way to the survey's own.
        for image in images:
            background = maps.local_basemap_image(image, *limits, transform=transform)
            if background is not None:
                break
        if background is None and choice in ("auto", *maps.TILE_SOURCES):
            background = maps.basemap_image(
                *limits, transform=transform, max_tiles=16, timeout=4.0,
                source="Satellite" if choice == "auto" else choice)
    resolved = model[np.isfinite(model) & (model > 0)]
    return {"grids": grids, "x": x, "y": y, "basemap": background, "coordinate_system": system,
            "resistivity_range": (float(resolved.min()), float(resolved.max()))}


def draw_survey_plan_maps(maps: Dict[str, Any], output_dir: Any, *, unit: str = "m",
                          cmap: Any = "turbo_r", name: str = "",
                          stem: str = "tdem") -> Dict[str, Any]:
    """Draw and save what :func:`survey_plan_grids` computed.

    Writes ``<stem>_depth_slices.png`` (resistivity, on the colour scale of the
    survey's sections), ``<stem>_depth_slice_uncertainty.png`` (the kriging
    standard deviation) and the grids of both under ``maps/`` for GIS. Returns
    their paths and a summary for a report: per depth, the soundings used,
    the variogram and the median and 90th-percentile standard deviation.
    """
    import matplotlib.colors as mcolors

    grids, x, y = maps["grids"], maps["x"], maps["y"]
    background, system = maps.get("basemap"), maps.get("coordinate_system") or ""
    low, high = maps["resistivity_range"]
    label = f"Coordinates: {system}." if system else ""
    prefix = f"{name}: " if name else ""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    figure = plot_resistivity_depth_slices(
        grids, x, y, folder / f"{stem}_depth_slices.png", basemap=background, unit=unit,
        norm=mcolors.LogNorm(vmin=low, vmax=max(high, low * 1.01)), cmap=cmap,
        title=f"{prefix}resistivity in plan view", coordinate_label=label)
    uncertainty = plot_kriging_uncertainty(
        grids, x, y, folder / f"{stem}_depth_slice_uncertainty.png", basemap=background,
        unit=unit, title=f"{prefix}uncertainty of the plan-view maps", coordinate_label=label)
    return {
        "map_figure": figure,
        "map_uncertainty_figure": uncertainty,
        "map_grids": write_ascii_grids(grids, folder / "maps", f"{stem}_resistivity"),
        "depth_slices": {
            "depths": grids["depths"], "soundings_used": grids["soundings_used"],
            "max_distance": grids["max_distance"], "cell": grids["cell"],
            "method": grids["method"], "coordinate_system": system,
            "variograms": grids["variograms"],
            # Kriging standard deviation of log10 resistivity over each map's
            # cells: the median, and the 90th percentile, which the gaps
            # between lines and the unreached corners of a deep map set.
            "std_decades": [
                None if not np.isfinite(s).any() else
                {"median": float(np.nanmedian(s)), "p90": float(np.nanpercentile(s, 90))}
                for s in grids["std"]],
            "basemap": None if background is None else {
                "source": background.get("source"),
                "attribution": background.get("attribution")},
        },
    }


__all__ = [
    "CANDIDATE_DEPTHS_M",
    "choose_depths",
    "depth_variogram",
    "draw_survey_plan_maps",
    "masking_distance",
    "plot_kriging_uncertainty",
    "plot_resistivity_depth_slices",
    "resistivity_depth_slices",
    "survey_plan_grids",
    "write_ascii_grids",
]
