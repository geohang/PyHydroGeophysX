"""Pictures of the data a workflow run reads, drawn while the run reads them.

A load step used to read for forty seconds and then say "Loaded 1 ERT survey",
which left the user nothing to look at and nothing to check. The studio's Live
tab shows every image a run writes the moment it appears, so each step that
reads a method's data now draws what it read, under ``raw_data/`` in the run
folder: an apparent-resistivity pseudosection per ERT survey, the travel-time
curves of a seismic line, the decay curves of TEM soundings, and the apparent
resistivity and phase of MT sites. The same figures open the method's section
of the report, as the data the models were fitted to.

Every function here draws on a detached figure (see
:func:`._figstyle.detached_figure`), so nothing here can block a headless run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ._figstyle import detached_figure

#: Where a run keeps these figures, under its output folder.
RAW_DIR = "raw_data"

#: Most surveys, soundings or sites one step draws; a long series is sampled.
MAX_DRAWN = 6

#: Resolution of these figures: they are looked at on screen while a run works.
DPI = 150


def chosen(count: int, limit: int = MAX_DRAWN) -> List[int]:
    """Up to ``limit`` indices spread evenly over ``count``, first and last included.

    Examples
    --------
    >>> chosen(3)
    [0, 1, 2]
    >>> chosen(20)
    [0, 4, 8, 11, 15, 19]
    >>> chosen(0)
    []
    """
    if count <= limit:
        return list(range(max(0, count)))
    return sorted({int(round(v)) for v in np.linspace(0, count - 1, limit)})


def _save(figure: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    return str(path)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))[:60]


# ---------------------------------------------------------------------------
# ERT
# ---------------------------------------------------------------------------
def ert_readings(std: Any) -> Dict[str, Any]:
    """What one loaded ERT survey holds, and where its readings sit on a pseudosection.

    The apparent resistivity is the one the inversion will see: the survey is
    converted as the inversion converts it (``ert_io.standard_to_pg``, which
    forms ``rhoa = R * k`` when the file reports resistance). Where that
    conversion cannot be made, the values are the file's own, and
    ``converted`` is False.

    Parameters
    ----------
    std : StandardERT
        A survey as ``ERTLoaderAgent`` returns it.

    Returns
    -------
    dict
        ``n_electrodes``, ``n_readings``, ``points`` (``(x, pseudo-depth, rhoa)``
        rows), ``rhoa`` (the positive finite values), ``n_nonpositive`` and
        ``converted``.
    """
    from PyHydroGeophysX.data_processing import ert_io

    electrodes = list(getattr(std, "electrodes", None) or [])
    observations = list(getattr(std, "observations", None) or [])
    points = np.zeros((0, 3))
    converted = False
    try:
        container = ert_io.standard_to_pg(std)
    except Exception:  # noqa: BLE001 - the inversion reports why; the picture still helps
        container = None
    if container is not None:
        points = ert_io.container_pseudosection(container)
        converted = True
    elif electrodes:
        index = {int(e.id): k for k, e in enumerate(electrodes)}
        rows = [(index.get(int(o.quad.A), -1), index.get(int(o.quad.B), -1),
                 index.get(int(o.quad.M), -1), index.get(int(o.quad.N), -1),
                 np.nan if o.app_res is None else float(o.app_res)) for o in observations]
        if rows:
            a, b, m, n, values = (np.asarray(col) for col in zip(*rows))
            points = ert_io.pseudosection_points([float(e.x) for e in electrodes],
                                                 a, b, m, n, values)
    values = points[:, 2] if points.size else np.zeros(0)
    usable = np.isfinite(values) & (values > 0)
    return {"n_electrodes": len(electrodes), "n_readings": len(observations),
            "points": points[usable], "rhoa": values[usable],
            "n_nonpositive": int(np.count_nonzero(np.isfinite(values) & (values <= 0))),
            "converted": converted}


def ert_pseudosection(readings: Mapping[str, Any], path: Path, *, title: str,
                      unit: Optional[str] = None, cmap: str = "viridis") -> Optional[str]:
    """An apparent-resistivity pseudosection of one survey, as a PNG.

    Readings at their midpoint and pseudo-depth (positive down), coloured on
    a log scale clipped to the 3rd-97th percentile as the ERT page colours
    them. Returns the path, or None when no reading has a usable value.
    """
    from matplotlib.colors import LogNorm

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    points = np.asarray(readings.get("points"), dtype=float)
    if points.ndim != 2 or not points.shape[0]:
        return None
    x, depth, rhoa = points[:, 0], points[:, 1], points[:, 2]
    low, high = np.percentile(rhoa, [3, 97])
    if not high > low:
        low, high = low / 2.0, high * 2.0
    figure = detached_figure((8.0, 3.6))
    ax = figure.add_subplot(111)
    shown = ax.scatter(x, depth, c=np.clip(rhoa, low, high), cmap=cmap,
                       norm=LogNorm(vmin=low, vmax=high), s=22, marker="o",
                       edgecolors="#394b59", linewidths=0.3)
    span = max(float(np.ptp(x)), 1e-6)
    ax.set_xlim(float(x.min()) - 0.03 * span, float(x.max()) + 0.03 * span)
    ax.set_ylim(max(float(depth.max()) * 1.08, 0.02), 0.0)
    set_length_axis(ax, "x", "x", unit=unit)
    set_length_axis(ax, "y", "Pseudo-depth", unit=unit)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=10)
    figure.colorbar(shown, ax=ax, label="Apparent resistivity (Ω·m)", pad=0.015)
    return _save(figure, path)


def ert_pseudosections(series: Sequence[Mapping[str, Any]], path: Path, *,
                       titles: Sequence[str], unit: Optional[str] = None,
                       cmap: str = "viridis") -> Optional[str]:
    """Several surveys' pseudosections on one shared log colour scale, as a PNG.

    The report's picture of the data: with a scale per survey, as the Live tab
    draws them one at a time, every survey looks alike and the change between
    them - what a monitoring survey is for - cannot be seen. Up to three
    panels a row. Returns None when no survey has a usable reading.
    """
    from matplotlib.colors import LogNorm

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    drawn = [(np.asarray(r.get("points"), dtype=float), title)
             for r, title in zip(series, titles) if np.size(r.get("points"))]
    if not drawn:
        return None
    values = np.concatenate([points[:, 2] for points, _ in drawn])
    low, high = np.percentile(values, [3, 97])
    if not high > low:
        low, high = low / 2.0, high * 2.0
    columns = min(3, len(drawn))
    rows = int(np.ceil(len(drawn) / columns))
    figure = detached_figure((5.2 * columns + 1.0, 2.9 * rows + 0.4))
    axes = np.atleast_1d(figure.subplots(rows, columns, squeeze=False)).ravel()
    depth = max(float(points[:, 1].max()) for points, _ in drawn) * 1.08
    x_low = min(float(points[:, 0].min()) for points, _ in drawn)
    x_high = max(float(points[:, 0].max()) for points, _ in drawn)
    pad = 0.03 * max(x_high - x_low, 1e-6)
    shown = None
    for k, (ax, (points, title)) in enumerate(zip(axes, drawn)):
        shown = ax.scatter(points[:, 0], points[:, 1], c=np.clip(points[:, 2], low, high),
                           cmap=cmap, norm=LogNorm(vmin=low, vmax=high), s=10,
                           edgecolors="none")
        ax.set_xlim(x_low - pad, x_high + pad)
        ax.set_ylim(depth, 0.0)
        set_length_axis(ax, "x", "x", unit=unit, labelled=k >= len(drawn) - columns)
        set_length_axis(ax, "y", "Pseudo-depth", unit=unit, labelled=k % columns == 0)
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9)
    for ax in axes[len(drawn):]:
        ax.set_visible(False)
    figure.colorbar(shown, ax=axes[:len(drawn)].tolist(), label="Apparent resistivity (Ω·m)",
                    shrink=0.9, pad=0.015)
    return _save(figure, path)


def describe_ert(readings: Mapping[str, Any]) -> str:
    """One clause on a survey's readings, for the load step's summary.

    >>> describe_ert({'n_electrodes': 48, 'n_readings': 900,
    ...               'rhoa': np.array([12.0, 80.0, 450.0]), 'n_nonpositive': 0})
    '48 electrodes, 900 readings, apparent resistivity 12 to 450 Ω·m (median 80)'
    """
    rhoa = np.asarray(readings.get("rhoa"), dtype=float)
    text = f"{readings.get('n_electrodes', 0)} electrodes, {readings.get('n_readings', 0)} readings"
    if rhoa.size:
        text += (f", apparent resistivity {rhoa.min():.3g} to {rhoa.max():.3g} Ω·m "
                 f"(median {np.median(rhoa):.3g})")
    if readings.get("n_nonpositive"):
        text += f", {readings['n_nonpositive']} not positive"
    return text


# ---------------------------------------------------------------------------
# Seismic refraction
# ---------------------------------------------------------------------------
def traveltime_table(data: Any) -> Dict[str, np.ndarray]:
    """Shot and geophone positions and travel times from a pyGIMLi travel-time container."""
    sensors = np.asarray(data.sensors(), dtype=float)
    shots = np.asarray(data["s"], dtype=int)
    geophones = np.asarray(data["g"], dtype=int)
    return {"shot_x": sensors[shots, 0], "geophone_x": sensors[geophones, 0],
            "shot": shots, "time_s": np.asarray(data["t"], dtype=float)}


def traveltime_curves(table: Mapping[str, np.ndarray], path: Path, *, title: str,
                      unit: Optional[str] = None,
                      predicted: Optional[np.ndarray] = None) -> str:
    """Travel time against geophone position, one curve per shot, as a PNG.

    Triangles mark the shots. With ``predicted`` the model's times are drawn
    as lines over the picks, which is how the evaluation shows the fit.
    """
    from matplotlib import colormaps

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    shots = np.asarray(table["shot"])
    order = np.unique(shots)
    colours = colormaps["viridis"](np.linspace(0.05, 0.9, max(1, order.size)))
    figure = detached_figure((8.0, 4.2))
    ax = figure.add_subplot(111)
    for colour, shot in zip(colours, order):
        on = shots == shot
        x = np.asarray(table["geophone_x"])[on]
        t = np.asarray(table["time_s"])[on] * 1e3
        rank = np.argsort(x)
        if predicted is None:
            ax.plot(x[rank], t[rank], "-o", ms=2.5, lw=0.9, color=colour)
        else:
            ax.plot(x[rank], t[rank], "o", ms=2.5, color=colour)
            ax.plot(x[rank], np.asarray(predicted)[on][rank] * 1e3, "-", lw=1.0, color=colour)
        ax.plot([np.asarray(table["shot_x"])[on][0]], [0.0], "v", ms=6, color=colour,
                markeredgecolor="#1d1d1f", markeredgewidth=0.4)
    set_length_axis(ax, "x", "Geophone position", unit=unit)
    ax.set_ylabel("Travel time (ms)")
    ax.set_ylim(bottom=0.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=10)
    return _save(figure, path)


# ---------------------------------------------------------------------------
# TDEM
# ---------------------------------------------------------------------------
def decays(sounding: Mapping[str, Any]) -> List[tuple]:
    """``(label, times, |response|, relative std or None)`` per moment of one sounding."""
    moments = sounding.get("moments") or {}
    found = []
    for name, moment in (moments.items() if moments else [("", sounding)]):
        times = np.asarray(moment.get("times", []), dtype=float)
        values = np.asarray(moment.get("response", moment.get("values", [])), dtype=float)
        n = min(times.size, values.size)
        if not n:
            continue
        std = moment.get("relative_std")
        std = np.asarray(std, dtype=float)[:n] if std is not None and np.size(std) >= n else None
        found.append((str(name), times[:n], np.abs(values[:n]), std))
    return found


def tdem_decays(soundings: Sequence[Mapping[str, Any]], path: Path, *, title: str,
                labels: Optional[Sequence[str]] = None,
                layout: Optional[Mapping[str, Any]] = None,
                unit: Optional[str] = None) -> Optional[str]:
    """Decay curves of a few soundings, log-log, beside the station layout when known.

    ``layout`` holds every station's ``x``, ``y`` and ``line_numbers``; the
    drawn soundings are ringed on it. Returns the path, or None when no
    sounding has data.
    """
    from matplotlib import colormaps

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    curves = [(k, decays(sounding)) for k, sounding in enumerate(soundings)]
    curves = [(k, found) for k, found in curves if found]
    if not curves:
        return None
    x = np.asarray((layout or {}).get("x", []), dtype=float)
    y = np.asarray((layout or {}).get("y", []), dtype=float)
    mapped = x.size > 1 and x.size == y.size and np.isfinite(x).any()
    figure = detached_figure((10.5 if mapped else 7.0, 4.4))
    grid = figure.add_gridspec(1, 2 if mapped else 1, width_ratios=[1.5, 1] if mapped else None)
    ax = figure.add_subplot(grid[0, 0])
    colours = colormaps["viridis"](np.linspace(0.05, 0.9, max(1, len(curves))))
    markers = {"LM": "o", "HM": "s"}
    moments = set()
    for colour, (k, found) in zip(colours, curves):
        name = labels[k] if labels and k < len(labels) else f"Sounding {k + 1}"
        for order, (moment, times, values, std) in enumerate(found):
            keep = values > 0
            moments.add(moment)
            # One legend entry per sounding; the marker tells the moments apart.
            ax.loglog(times[keep] * 1e3, values[keep], "-" + markers.get(moment, "o"), ms=3,
                      lw=0.9, color=colour, label=name if order == 0 else None,
                      alpha=1.0 if moment != "LM" or len(found) == 1 else 0.75)
            if std is not None:
                band = std[keep] * values[keep]
                ax.fill_between(times[keep] * 1e3, np.clip(values[keep] - band, values[keep] * 0.05,
                                                           None),
                                values[keep] + band, color=colour, alpha=0.12, lw=0)
    ax.set_xlabel("Time after turn-off (ms)")
    ax.set_ylabel("|Response|")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, ncol=2 if len(curves) > 3 else 1, loc="lower left")
    if {"LM", "HM"} <= moments:
        title += " (circles LM, squares HM)"
    ax.set_title(title, fontsize=10)
    if mapped:
        plan = figure.add_subplot(grid[0, 1])
        lines = np.asarray((layout or {}).get("line_numbers", np.zeros(x.size)))
        plan.scatter(x, y, c=lines if lines.size == x.size else None, cmap="tab20", s=6)
        picked = [k for k, _ in curves if k < x.size]
        drawn = np.asarray((layout or {}).get("drawn", picked), dtype=int)
        drawn = drawn[(drawn >= 0) & (drawn < x.size)]
        plan.scatter(x[drawn], y[drawn], s=60, facecolors="none",
                     edgecolors=colours[:drawn.size], linewidths=1.4)
        set_length_axis(plan, "x", "Easting", unit=unit)
        set_length_axis(plan, "y", "Northing", unit=unit)
        plan.ticklabel_format(useOffset=False, style="plain")
        plan.tick_params(axis="x", labelrotation=30)
        plan.set_aspect("equal", adjustable="datalim")
        plan.grid(True, alpha=0.25)
        plan.set_title(f"{x.size} stations; the drawn ones ringed", fontsize=9)
    figure.tight_layout()
    return _save(figure, path)


# ---------------------------------------------------------------------------
# MT
# ---------------------------------------------------------------------------
def mt_soundings(transfer_functions: Sequence[Any], path: Path, *, title: str) -> Optional[str]:
    """Apparent resistivity and phase of up to three sites per row, as a PNG."""
    from PyHydroGeophysX.visualization import plot_mt_sounding

    sites = list(transfer_functions)
    if not sites:
        return None
    columns = min(3, len(sites))
    rows = int(np.ceil(len(sites) / columns))
    figure = detached_figure((4.2 * columns, 5.2 * rows))
    grid = figure.add_gridspec(2 * rows, columns, height_ratios=[3, 2] * rows)
    for k, tf in enumerate(sites):
        row, column = divmod(k, columns)
        ax_rho = figure.add_subplot(grid[2 * row, column])
        ax_phase = figure.add_subplot(grid[2 * row + 1, column], sharex=ax_rho)
        plot_mt_sounding(tf, axes=(ax_rho, ax_phase), title=getattr(tf, "station", None)
                         or f"Site {k + 1}")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Gravity and magnetics
# ---------------------------------------------------------------------------
def gravmag_stations(x: Any, y: Any, value: Any, path: Path, *, title: str, label: str,
                     unit: Optional[str] = None) -> str:
    """The station values as read, as a map, as a PNG."""
    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    x, y, value = (np.asarray(v, dtype=float) for v in (x, y, value))
    low, high = np.nanpercentile(value, [2, 98])
    figure = detached_figure((6.4, 5.2))
    ax = figure.add_subplot(111)
    shown = ax.scatter(x, y, c=np.clip(value, low, high), cmap="viridis", s=14,
                       edgecolors="none")
    figure.colorbar(shown, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal", adjustable="datalim")
    ax.ticklabel_format(useOffset=False, style="plain")
    set_length_axis(ax, "x", "Easting", unit=unit)
    set_length_axis(ax, "y", "Northing", unit=unit)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=10)
    figure.tight_layout()
    return _save(figure, path)


def figure_path(output_dir: Any, stem: str) -> Path:
    """``<output_dir>/raw_data/<stem>.png``, the stem made safe for a file name.

    >>> figure_path('run', 'ert 01: a.ohm').as_posix()
    'run/raw_data/ert_01__a_ohm.png'
    """
    return Path(output_dir, RAW_DIR, _slug(stem) + ".png")
