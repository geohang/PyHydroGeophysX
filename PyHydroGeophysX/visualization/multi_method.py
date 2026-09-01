"""Visualization helpers for multi-method geophysical workflows."""

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# result field
# ---------------------------------------------------------------------------
def _result_field(result, preferred):
    for key in preferred:
        if result is not None and hasattr(result, key):
            value = getattr(result, key)
            if value is not None:
                return np.asarray(value, dtype=float)
    return None


# ---------------------------------------------------------------------------
# as 1d
# ---------------------------------------------------------------------------
def _as_1d(values, name):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


# ---------------------------------------------------------------------------
# plot on mesh
# ---------------------------------------------------------------------------
def _plot_on_mesh(ax, mesh, values, title, cmap="viridis"):
    try:
        import pygimli as pg

        arr = np.asarray(values, dtype=float).ravel()
        if mesh is not None and hasattr(mesh, "cellCount") and mesh.cellCount() == arr.size:
            pg.show(mesh, data=arr, ax=ax, cMap=cmap)
            ax.set_title(title)
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# plot multi method panel
# ---------------------------------------------------------------------------
def plot_multi_method_panel(
    ert_result: Any,
    srt_result: Any,
    em_result: Any,
    mesh: Any = None,
) -> Any:
    """Plot side-by-side ERT/SRT/EM model panels."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("ERT", _result_field(ert_result, ["final_model", "recovered_model"]), "plasma"),
        ("SRT", _result_field(srt_result, ["final_model"]), "viridis"),
        ("EM", _result_field(em_result, ["recovered_conductivity", "final_model"]), "cividis"),
    ]

    for ax, (name, values, cmap) in zip(axes, panels):
        if values is None:
            ax.text(0.5, 0.5, f"No {name} result", ha="center", va="center")
            ax.set_title(name)
            ax.axis("off")
            continue

        flat = np.asarray(values, dtype=float).ravel()
        if not _plot_on_mesh(ax, mesh, flat, f"{name} Model", cmap=cmap):
            ax.plot(flat, lw=1.8)
            ax.set_title(f"{name} Model")
            ax.set_xlabel("Cell Index")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# plot hydro vs geophys
# ---------------------------------------------------------------------------
def plot_hydro_vs_geophys(
    hydro_wc: Any,
    inverted_wc: Any,
    mesh: Any = None,
) -> Any:
    """Compare hydrological water content to geophysics-derived water content."""
    hydro_wc = _as_1d(hydro_wc, "hydro_wc")
    inverted_wc = _as_1d(inverted_wc, "inverted_wc")

    if hydro_wc.size != inverted_wc.size:
        x_src = np.linspace(0.0, 1.0, inverted_wc.size)
        x_tgt = np.linspace(0.0, 1.0, hydro_wc.size)
        inverted_wc = np.interp(x_tgt, x_src, inverted_wc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if not _plot_on_mesh(axes[0], mesh, hydro_wc, "Hydrological Water Content", cmap="Blues"):
        axes[0].plot(hydro_wc, color="tab:blue", lw=1.8)
        axes[0].set_title("Hydrological Water Content")
        axes[0].set_xlabel("Cell Index")
        axes[0].set_ylabel("Water Content")
        axes[0].grid(True, alpha=0.3)

    if not _plot_on_mesh(axes[1], mesh, inverted_wc, "Geophysical-Derived Water Content", cmap="Oranges"):
        axes[1].plot(inverted_wc, color="tab:orange", lw=1.8)
        axes[1].set_title("Geophysical-Derived Water Content")
        axes[1].set_xlabel("Cell Index")
        axes[1].set_ylabel("Water Content")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# plot cross section with wells
# ---------------------------------------------------------------------------
def plot_cross_section_with_wells(
    result: Any,
    mesh: Any,
    well_data: Optional[Dict[str, np.ndarray]] = None,
) -> Any:
    """Plot a model cross-section and overlay optional well picks."""
    values = _result_field(result, ["final_model", "recovered_conductivity", "recovered_model"])
    if values is None:
        raise ValueError("result has no plottable model field.")

    flat = np.asarray(values, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(9, 5))

    plotted = _plot_on_mesh(ax, mesh, flat, "Cross-Section with Wells", cmap="viridis")
    if not plotted:
        ax.plot(flat, lw=1.8, label="Model")
        ax.set_xlabel("Cell Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    if well_data is not None:
        x = np.asarray(well_data.get("x", []), dtype=float)
        z = np.asarray(well_data.get("z", []), dtype=float)
        labels = well_data.get("labels", None)

        if x.size and z.size and x.size == z.size:
            ax.scatter(x, z, c="k", s=30, marker="o", label="Wells")
            if labels is not None and len(labels) == x.size:
                for xi, zi, label in zip(x, z, labels):
                    ax.text(xi, zi, str(label), fontsize=8, ha="left", va="bottom")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot em data fit
# ---------------------------------------------------------------------------
def plot_em_data_fit(
    times: Any,
    observed: Any,
    predicted: Any,
    uncertainties: Any = None,
    true_data: Any = None,
    chi2: Optional[float] = None,
    time_scale: float = 1e3,
    time_label: str = "Time (ms)",
    data_label: str = "|Response|",
    ax: Any = None,
) -> Any:
    """Plot log-log EM data fit similar to TDEM workflow examples."""
    t = _as_1d(times, "times")
    dobs = _as_1d(observed, "observed")
    dpred = _as_1d(predicted, "predicted")
    if t.size != dobs.size or dobs.size != dpred.size:
        raise ValueError("times, observed, and predicted must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    ax.loglog(t * time_scale, np.abs(dobs), "ko", markersize=6, label="Observed")
    ax.loglog(t * time_scale, np.abs(dpred), "r-", lw=2, label="Predicted")

    if true_data is not None:
        dtrue = _as_1d(true_data, "true_data")
        if dtrue.size != t.size:
            raise ValueError("true_data must have the same length as times.")
        ax.loglog(t * time_scale, np.abs(dtrue), "b--", lw=1.5, alpha=0.8, label="True")

    if uncertainties is not None:
        unc = _as_1d(uncertainties, "uncertainties")
        if unc.size != t.size:
            raise ValueError("uncertainties must have the same length as times.")
        lower = np.maximum(np.abs(dobs) - unc, np.finfo(float).tiny)
        upper = np.abs(dobs) + unc
        ax.fill_between(t * time_scale, lower, upper, alpha=0.2, color="gray", label="Uncertainty")

    title = "Data Fit"
    if chi2 is not None:
        title = f"Data Fit (chi2 = {chi2:.2f})"
    ax.set_title(title)
    ax.set_xlabel(time_label)
    ax.set_ylabel(data_label)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best")

    if own_fig:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# signal against noise along a line
# ---------------------------------------------------------------------------
def plot_signal_and_noise(
    summary: Any,
    line: Optional[int] = None,
    moments: Any = ("LM", "HM"),
    smooth: int = 21,
    axes: Any = None,
) -> Any:
    """Draw the measured signal and the absolute noise along a survey line.

    Takes what :func:`PyHydroGeophysX.data_processing.em1d.survey_summary`
    returns. One panel per moment, both on a log axis, plotted against distance
    along the line.

    The pair is the point. A station returns fewer usable gates for two reasons
    that call for opposite readings, and the relative error the file records is
    the one divided by the other, so it rises either way and cannot separate
    them. Drawn apart they can be read directly: a signal that falls while the
    noise holds is evidence about the ground, because a resistive half-space
    returns dB/dt going as rho**(-3/2) and is genuinely quieter. A noise floor
    that rises under a steady signal is an instrument or an environment and says
    nothing about the ground.

    ``smooth`` is the width of a running mean over stations, which is cosmetic:
    station-to-station scatter obscures the trend the figure is drawn to show.
    Set it to 1 to plot the values themselves.

    ``line`` selects one survey line, and defaults to the first the summary
    holds. Distance runs from that line's own first station.
    """
    rows = list((summary or {}).get("rows", []))
    if not rows:
        raise ValueError("the summary holds no stations to plot.")
    numbers = sorted({int(r["line"]) for r in rows})
    chosen = numbers[0] if line is None else int(line)
    if chosen not in numbers:
        raise ValueError(
            f"the summary holds no line {chosen}; it has {numbers}.")
    on_line = [r for r in rows if int(r["line"]) == chosen]
    names = [str(m) for m in moments
             if any(f"{m}_signal" in r for r in on_line)]
    if not names:
        raise ValueError(
            f"no moment among {list(moments)} carries a signal column. "
            "survey_summary records them only for the moments it was asked for.")

    x, y = _line_distance(on_line)
    del y
    if axes is None:
        fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.5 * len(names)),
                                 sharex=True, squeeze=False)
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure
    for ax, name in zip(axes, names):
        signal = np.asarray([r.get(f"{name}_signal", np.nan) for r in on_line],
                            dtype=float)
        noise = np.asarray([r.get(f"{name}_noise", np.nan) for r in on_line],
                           dtype=float)
        ref = np.asarray([r.get(f"{name}_reference_time", np.nan)
                          for r in on_line], dtype=float)
        ax.semilogy(x, _running_mean(signal, smooth), color="#1f77b4",
                    lw=1.8, label="|signal|")
        ax.semilogy(x, _running_mean(noise, smooth), color="#c62828",
                    lw=1.8, label="absolute noise")
        at = np.nanmedian(ref)
        ax.set_ylabel("%s%s  (V)" % (
            name, "" if not np.isfinite(at) else " at %.1f us" % (at * 1e6)))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Distance along line %d (m)" % chosen)
    fig.tight_layout()
    return fig, axes


def _line_distance(rows: Any) -> Any:
    """Cumulative distance along a line, from its own first station.

    Falls back to the station's order in the file where the map coordinates are
    missing, so a survey without them still plots against something monotonic.
    """
    x = np.asarray([r.get("x", np.nan) for r in rows], dtype=float)
    y = np.asarray([r.get("y", np.nan) for r in rows], dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or x.size < 2:
        return np.arange(len(rows), dtype=float), None
    steps = np.hypot(np.diff(x), np.diff(y))
    steps[~np.isfinite(steps)] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)]), None


def _running_mean(values: Any, width: int) -> Any:
    """A centred running mean that ignores the gaps rather than spreading them.

    A NaN inside a plain convolution takes its whole window with it, which on a
    survey with a few dummy stations erases a stretch of the trend around each.
    """
    values = np.asarray(values, dtype=float).ravel()
    width = max(1, int(width))
    if width == 1 or values.size < 2:
        return values
    good = np.isfinite(values)
    filled = np.where(good, values, 0.0)
    window = np.ones(min(width, values.size))
    total = np.convolve(filled, window, mode="same")
    count = np.convolve(good.astype(float), window, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


# ---------------------------------------------------------------------------
# plan-view slices at fixed depth
# ---------------------------------------------------------------------------
def plot_depth_slices(
    cells: Any,
    depths: Any = (5.0, 15.0, 30.0, 50.0),
    basemap: Any = None,
    extent: Any = None,
    max_distance: Optional[float] = None,
    drop_below_doi: bool = True,
    grid: int = 400,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ncols: int = 2,
    cmap: str = "turbo",
    show_stations: bool = False,
) -> Any:
    """Map the recovered resistivity at fixed depths, over a basemap.

    ``cells`` is the per-cell table a line inversion writes, needing ``x``,
    ``y``, ``depth_center_m`` and ``resistivity_ohm_m``, and using ``below_doi``
    when it carries one. A pandas frame or any mapping of columns will do.

    Each requested depth is snapped to the nearest layer the model actually has,
    and the panel is titled with the depth used rather than the depth asked for.
    Interpolating between layer centres would invent a resolution the layering
    does not have.

    ``max_distance`` is what keeps the picture honest. A survey of a few lines
    leaves most of the map with no station anywhere near it, and a triangulated
    interpolation will happily fill that space from stations hundreds of metres
    away. Anything further than this from a station is left blank, so the ground
    that was measured is the ground that is coloured.

    One width serves every panel, so that a ribbon growing or shrinking between
    them means the coverage changed rather than the rule did. The mask is about
    where a station is, and a survey traces the same track at every depth; how
    far a sounding sees sideways is a question about resolution, which belongs
    in how the answer is read rather than in which ground gets coloured.

    Left unset it is half the deepest slice requested, floored at three times
    the station spacing, and the figure states the number it used.

    ``drop_below_doi`` removes cells the run marked as unresolved before
    interpolating, so a depth below the investigation depth over part of the
    survey shows a gap there instead of a colour.

    Colour is log10 resistivity, with the range taken from the slices drawn so
    that the panels are comparable to each other.

    ``show_stations`` marks each sounding. It is off because a shallow slice is
    a ribbon a few metres wide and a marker per station hides the colour it is
    there to mark; the ribbon already traces the survey.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    x, y, depth, rho, below = _cell_columns(cells)
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(rho) & (rho > 0)
    if not usable.any():
        raise ValueError(
            "no cells to draw; every one was non-finite or non-positive.")
    # The layers a depth snaps to are the ones the model has, before the
    # investigation-depth cut. Snapping against what survives the cut instead
    # would answer a request for fifty metres with an eight-metre slice and say
    # so only in the panel title.
    available = np.unique(depth[usable & np.isfinite(depth)])
    resolved = usable.copy()
    if drop_below_doi and below is not None:
        resolved &= ~below
    x, y, depth, rho = x[usable], y[usable], depth[usable], rho[usable]
    resolved = resolved[usable]
    if not available.size:
        raise ValueError("the cells carry no finite depth to slice at.")
    wanted = np.atleast_1d(np.asarray(depths, dtype=float)).ravel()
    chosen = [float(available[int(np.argmin(np.abs(available - d)))])
              for d in wanted]

    spacing = _median_station_spacing(x, y)

    if extent is None:
        pad_x = 0.05 * (np.ptp(x) or 1.0)
        pad_y = 0.05 * (np.ptp(y) or 1.0)
        extent = (x.min() - pad_x, x.max() + pad_x,
                  y.min() - pad_y, y.max() + pad_y)
    west, east, south, north = (float(v) for v in extent)
    gx = np.linspace(west, east, int(grid))
    gy = np.linspace(south, north, int(grid))
    mesh_x, mesh_y = np.meshgrid(gx, gy)

    reach = (float(max_distance) if max_distance is not None
             else max(3.0 * spacing, 0.5 * max(chosen)))
    panels = []
    for at in chosen:
        on_layer = (depth == at) & resolved
        if not on_layer.any():
            # Nothing survives the cut at this depth. An empty panel that says
            # so beats quietly drawing a shallower layer in its place.
            panels.append((at, np.full(mesh_x.shape, np.nan), 0))
            continue
        points = np.column_stack([x[on_layer], y[on_layer]])
        values = _interpolate_layer(points, np.log10(rho[on_layer]),
                                    mesh_x, mesh_y)
        far = cKDTree(points).query(
            np.column_stack([mesh_x.ravel(), mesh_y.ravel()]))[0]
        values[far.reshape(mesh_x.shape) > reach] = np.nan
        panels.append((at, values, int(on_layer.sum())))

    finite = np.concatenate([p[1][np.isfinite(p[1])] for p in panels]
                           + [np.log10(rho[resolved])])
    low = np.log10(vmin) if vmin else float(np.nanpercentile(finite, 2))
    high = np.log10(vmax) if vmax else float(np.nanpercentile(finite, 98))

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(panels) / ncols))
    # Sized from the ground rather than from a fixed shape: the axes lock to an
    # equal aspect, so a wide survey in a square panel is mostly white paper.
    span_x, span_y = east - west, north - south
    width = 6.4
    height = max(2.4, width * (span_y / span_x if span_x else 1.0)) + 0.9
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width * ncols, height * nrows),
                             squeeze=False)
    flat = axes.ravel()
    mesh = None
    for ax, (at, values, count) in zip(flat, panels):
        if basemap is not None:
            ax.imshow(basemap, extent=(west, east, south, north),
                      origin="upper", zorder=0)
        mesh = ax.pcolormesh(gx, gy, values, vmin=low, vmax=high, cmap=cmap,
                             shading="auto", alpha=0.82, zorder=2)
        if show_stations:
            # Off by default: at a shallow slice the ribbon is only a few
            # metres wide, and a marker per station covers the colour it is
            # supposed to be marking.
            ax.plot(x[depth == at], y[depth == at], ".", ms=1.2,
                    color="white", alpha=0.55, zorder=3)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect("equal", "box")
        ax.set_title(
            "%.1f m depth   (%d stations)" % (at, count) if count else
            "%.1f m depth   (nothing resolved at this depth)" % at,
            fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
    for ax in flat[len(panels):]:
        ax.axis("off")
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=list(flat), shrink=0.85, pad=0.02)
        bar.set_label("Resistivity (ohm-m)")
        ticks = np.linspace(low, high, 5)
        bar.set_ticks(ticks)
        bar.set_ticklabels(["%.0f" % (10.0 ** t) for t in ticks])
    # Stated once, since it is one number for every panel: a ribbon that
    # changes width between them is the coverage changing, not the rule. Placed
    # under the lowest axes rather than at the foot of the figure, which on a
    # wide survey sits a long way below the last panel.
    drawn = [ax for ax in flat if ax.get_title()]
    floor = 0.05
    if drawn:
        # The tight bounding box, so the note clears the tick labels and the
        # axis label rather than landing on top of them.
        fig.canvas.draw()
        boxes = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in drawn]
        floor = min(fig.transFigure.inverted().transform(b.p0)[1]
                    for b in boxes)
    fig.text(0.5, max(0.004, floor - 0.02),
             "Coloured within %.0f m of a sounding; further ground was not "
             "measured." % reach,
             ha="center", va="top", fontsize=8, color="0.35")
    return fig, axes


def _interpolate_layer(points: Any, values: Any, mesh_x: Any,
                       mesh_y: Any) -> Any:
    """Interpolate one layer, falling back where a triangulation cannot exist.

    A survey walked in a straight line gives collinear stations, and a Delaunay
    triangulation of collinear points is not defined; the linear interpolant
    raises rather than returning anything. That is an ordinary survey geometry
    and not an error, so it falls back to nearest-neighbour, which is the more
    honest answer there in any case: a line has no width to interpolate across,
    so every point beside it can only carry the value of the station nearest to
    it. The distance mask then decides how far that value is allowed to reach.
    """
    from scipy.interpolate import griddata

    try:
        from scipy.spatial import QhullError
    except ImportError:  # scipy < 1.8 kept it in a private module
        from scipy.spatial.qhull import QhullError

    try:
        return griddata(points, values, (mesh_x, mesh_y), method="linear")
    except (QhullError, ValueError):
        return griddata(points, values, (mesh_x, mesh_y), method="nearest")


def _cell_columns(cells: Any) -> Any:
    """The four columns a slice needs, from a frame or any column mapping."""
    def column(name, required=True):
        try:
            values = cells[name]
        except (KeyError, TypeError, IndexError):
            if required:
                raise ValueError(
                    f"the cell table has no '{name}' column; a depth slice "
                    "needs x, y, depth_center_m and resistivity_ohm_m.")
            return None
        return np.asarray(getattr(values, "to_numpy", lambda: values)(),
                          dtype=float).ravel()

    below = column("below_doi", required=False)
    return (column("x"), column("y"), column("depth_center_m"),
            column("resistivity_ohm_m"),
            None if below is None else below.astype(bool))


def _median_station_spacing(x: Any, y: Any) -> float:
    """Distance from a station to its nearest neighbour, at the median.

    Taken over the unique positions rather than the cells, since every station
    contributes one row per layer and the duplicates would all read as zero.
    """
    from scipy.spatial import cKDTree

    points = np.unique(np.column_stack([x, y]), axis=0)
    if points.shape[0] < 2:
        return 1.0
    nearest = cKDTree(points).query(points, k=2)[0][:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest)) if nearest.size else 1.0


# ---------------------------------------------------------------------------
# plot em residuals
# ---------------------------------------------------------------------------
def plot_em_residuals(
    times: Any,
    observed: Any,
    predicted: Any,
    uncertainties: Any,
    sigma_bound: float = 2.0,
    time_scale: float = 1e3,
    time_label: str = "Time (ms)",
    ax: Any = None,
) -> Any:
    """Plot normalized residuals with +/-sigma bounds."""
    t = _as_1d(times, "times")
    dobs = _as_1d(observed, "observed")
    dpred = _as_1d(predicted, "predicted")
    unc = _as_1d(uncertainties, "uncertainties")
    if not (t.size == dobs.size == dpred.size == unc.size):
        raise ValueError("times, observed, predicted, and uncertainties must have the same length.")

    if np.any(unc <= 0):
        raise ValueError("uncertainties must be positive.")

    residual = (dobs - dpred) / unc

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    ax.semilogx(t * time_scale, residual, "ro-", lw=1.4, markersize=4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(sigma_bound, color="gray", ls="--", lw=0.8)
    ax.axhline(-sigma_bound, color="gray", ls="--", lw=0.8)
    ax.fill_between(t * time_scale, -sigma_bound, sigma_bound, alpha=0.12, color="green")
    ax.set_xlabel(time_label)
    ax.set_ylabel("Normalized Residual")
    ax.set_title("Data Residuals")
    ax.grid(True, alpha=0.4)

    if own_fig:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot em fit and residuals
# ---------------------------------------------------------------------------
def plot_em_fit_and_residuals(
    times: Any,
    observed: Any,
    predicted: Any,
    uncertainties: Any,
    true_data: Any = None,
    chi2: Optional[float] = None,
    time_scale: float = 1e3,
) -> Any:
    """Create side-by-side EM data fit and residual plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_em_data_fit(
        times=times,
        observed=observed,
        predicted=predicted,
        uncertainties=uncertainties,
        true_data=true_data,
        chi2=chi2,
        time_scale=time_scale,
        ax=axes[0],
    )
    plot_em_residuals(
        times=times,
        observed=observed,
        predicted=predicted,
        uncertainties=uncertainties,
        time_scale=time_scale,
        ax=axes[1],
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# plot time lapse panel
# ---------------------------------------------------------------------------
def plot_time_lapse_panel(
    models: Sequence[Any],
    mesh: Any = None,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 4,
    cmap: str = "viridis",
) -> Any:
    """Plot a grid of time-lapse model snapshots."""
    if not models:
        raise ValueError("models must contain at least one model.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")

    data_arrays = []
    for obj in models:
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj, dtype=float)
        else:
            arr = _result_field(obj, ["final_model", "recovered_model", "recovered_conductivity"])
            if arr is None:
                arr = np.asarray(obj, dtype=float)
        data_arrays.append(np.asarray(arr, dtype=float))

    n_models = len(data_arrays)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    flattened = axes.ravel()
    for i, ax in enumerate(flattened):
        if i >= n_models:
            ax.axis("off")
            continue

        arr = data_arrays[i]
        title = titles[i] if titles is not None and i < len(titles) else f"Timestep {i + 1}"

        if arr.ndim == 2:
            im = ax.imshow(arr, cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("X Index")
            ax.set_ylabel("Z Index")
            ax.set_title(title)
            continue

        flat = arr.ravel()
        if not _plot_on_mesh(ax, mesh, flat, title, cmap=cmap):
            ax.plot(flat, lw=1.5)
            ax.set_xlabel("Cell Index")
            ax.set_ylabel("Value")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# plot petrophysical scatter
# ---------------------------------------------------------------------------
def plot_petrophysical_scatter(
    x: Any,
    y: Any,
    color: Any = None,
    xlabel: str = "Porosity (-)",
    ylabel: str = "Property",
    color_label: str = "Saturation (-)",
    cmap: str = "Blues",
    fit_line: bool = True,
    ax: Any = None,
) -> Any:
    """Plot petrophysical scatter diagnostics with optional trend line."""
    xv = _as_1d(x, "x")
    yv = _as_1d(y, "y")
    if xv.size != yv.size:
        raise ValueError("x and y must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    if color is None:
        ax.scatter(xv, yv, s=24, c="tab:blue", alpha=0.8)
    else:
        cv = _as_1d(color, "color")
        if cv.size != xv.size:
            raise ValueError("color must have the same length as x and y.")
        scatter = ax.scatter(xv, yv, c=cv, s=24, cmap=cmap, alpha=0.9)
        fig.colorbar(scatter, ax=ax, label=color_label)

    if fit_line and xv.size >= 2:
        mask = np.isfinite(xv) & np.isfinite(yv)
        if np.count_nonzero(mask) >= 2:
            coeff = np.polyfit(xv[mask], yv[mask], 1)
            xline = np.linspace(np.nanmin(xv[mask]), np.nanmax(xv[mask]), 200)
            yline = coeff[0] * xline + coeff[1]
            ax.plot(xline, yline, "k--", lw=1.2, label="Linear fit")
            ax.legend(loc="best")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Petrophysical Relationship")
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot layered profiles
# ---------------------------------------------------------------------------
def plot_layered_profiles(
    depth_edges: Any,
    profiles: Dict[str, Sequence[float]],
    colors: Optional[Sequence[str]] = None,
    xscale: str = "linear",
) -> Any:
    """Plot one or more layered profiles as step-like vertical columns."""
    z = _as_1d(depth_edges, "depth_edges")
    if z.size < 2:
        raise ValueError("depth_edges must contain at least two values.")

    labels = list(profiles.keys())
    if not labels:
        raise ValueError("profiles must contain at least one entry.")

    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 6), squeeze=False)
    axes = axes.ravel()

    for i, label in enumerate(labels):
        values = _as_1d(profiles[label], label)
        if values.size != z.size - 1:
            raise ValueError(f"profile '{label}' length must be len(depth_edges)-1.")
        ax = axes[i]
        color = colors[i] if colors is not None and i < len(colors) else None

        for j, v in enumerate(values):
            ax.fill_betweenx([z[j], z[j + 1]], np.finfo(float).tiny, v, alpha=0.35, color=color)
            ax.plot([v, v], [z[j], z[j + 1]], color=color if color else "k", lw=1.6)

        ax.set_xlabel(label)
        ax.set_ylabel("Depth (m)")
        ax.set_xscale(xscale)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, axes
