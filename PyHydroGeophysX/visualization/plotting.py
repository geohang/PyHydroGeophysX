"""2D plotting utilities for geophysical models and data."""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_pg_show(ax, mesh, values, **kwargs):
    """Attempt to plot with pygimli; return True on success."""
    try:
        import pygimli as pg

        arr = np.asarray(values, dtype=float).ravel()
        if mesh is not None and hasattr(mesh, "cellCount") and mesh.cellCount() == arr.size:
            pg.show(mesh, data=arr, ax=ax, **kwargs)
            return True
    except Exception:
        pass
    return False


def _get_cmap(name: Optional[str] = None):
    """Return a matplotlib colormap, trying palettable first."""
    if name is not None:
        return plt.get_cmap(name) if isinstance(name, str) else name
    try:
        from palettable.lightbartlein.diverging import BlueDarkRed18_18_r
        return BlueDarkRed18_18_r.mpl_colormap
    except ImportError:
        return plt.get_cmap("RdBu_r")


# ---------------------------------------------------------------------------
# Single model cross-section
# ---------------------------------------------------------------------------

def plot_model_section(
    mesh: Any,
    values: np.ndarray,
    *,
    ax: Any = None,
    cmap: Any = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    log_scale: bool = False,
    label: str = "",
    xlabel: str = "Distance (m)",
    ylabel: str = "Elevation (m)",
    title: str = "",
    coverage: Optional[np.ndarray] = None,
    orientation: str = "vertical",
) -> Tuple:
    """Plot a 2D model cross-section on a PyGIMLi mesh.

    Parameters
    ----------
    mesh : pygimli.Mesh
        The mesh to plot on.
    values : array-like
        Cell values (resistivity, velocity, water content, etc.).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if *None*.
    cmap : str or Colormap, optional
        Colormap.  Defaults to ``BlueDarkRed18_18_r`` if available.
    cmin, cmax : float, optional
        Color limits.
    log_scale : bool
        Use logarithmic color scaling.
    label : str
        Colorbar label.
    coverage : array-like, optional
        Coverage array for masking low-sensitivity cells.
    orientation : str
        Colorbar orientation (``'vertical'`` or ``'horizontal'``).

    Returns
    -------
    fig, ax, cbar
    """
    import pygimli as pg

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    cmap = _get_cmap(cmap)
    kw = dict(
        cMap=cmap, logScale=log_scale, label=label,
        xlabel=xlabel, ylabel=ylabel, orientation=orientation, pad=0.3,
    )
    if cmin is not None:
        kw["cMin"] = cmin
    if cmax is not None:
        kw["cMax"] = cmax
    if coverage is not None:
        kw["coverage"] = np.asarray(coverage).ravel() > -1

    arr = np.asarray(values, dtype=float).ravel()
    ax, cbar = pg.show(mesh, arr, ax=ax, **kw)
    if title:
        ax.set_title(title)
    return fig, ax, cbar


# ---------------------------------------------------------------------------
# Multi-panel time-lapse snapshots
# ---------------------------------------------------------------------------

def plot_timelapse_snapshots(
    mesh: Any,
    models: Sequence[np.ndarray],
    *,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 4,
    cmap: Any = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    log_scale: bool = False,
    label: str = "",
    coverage: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    figsize_per_panel: Tuple[float, float] = (4.0, 2.5),
) -> Tuple:
    """Plot a grid of time-lapse model snapshots.

    Parameters
    ----------
    mesh : pygimli.Mesh
        Mesh shared by all snapshots.
    models : sequence of array-like
        Model arrays for each timestep.
    titles : sequence of str, optional
        Panel titles.  Defaults to ``'Timestep 1'``, ``'Timestep 2'``, ...
    ncols : int
        Number of columns.
    cmap, cmin, cmax, log_scale, label :
        Passed to ``pg.show``.
    coverage : array or sequence of arrays, optional
        Coverage mask(s).  If a single 1-D array it is reused for all panels.
        If 2-D, ``coverage[i]`` is used for panel *i*.
    figsize_per_panel : tuple
        (width, height) per subplot panel.

    Returns
    -------
    fig, axes
    """
    import pygimli as pg

    n = len(models)
    nrows = int(np.ceil(n / ncols))
    cmap = _get_cmap(cmap)
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Prepare coverage
    if coverage is not None:
        cov_arr = np.asarray(coverage)
        single_cov = cov_arr.ndim == 1
    else:
        cov_arr = None
        single_cov = False

    last_cbar = None
    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        if idx >= n:
            ax.axis("off")
            continue

        arr = np.asarray(models[idx], dtype=float).ravel()
        t = titles[idx] if titles is not None and idx < len(titles) else f"Timestep {idx + 1}"

        kw = dict(
            cMap=cmap, logScale=log_scale, label=label,
            orientation="vertical", pad=0.3,
        )
        if cmin is not None:
            kw["cMin"] = cmin
        if cmax is not None:
            kw["cMax"] = cmax
        if cov_arr is not None:
            c = cov_arr if single_cov else cov_arr[idx]
            kw["coverage"] = np.asarray(c).ravel() > -1

        # Axis labels only on edges
        if col == 0:
            kw["ylabel"] = "Elevation (m)"
        else:
            ax.set_yticks([])
        if row == nrows - 1:
            kw["xlabel"] = "Distance (m)"
        else:
            ax.set_xticks([])

        ax_out, cbar = pg.show(mesh, arr, ax=ax, **kw)
        ax.set_title(t, fontsize=10)
        # Remove individual colorbars except for one reference
        if last_cbar is not None:
            try:
                cbar.remove()
            except Exception:
                pass
        last_cbar = cbar

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Difference / ratio maps
# ---------------------------------------------------------------------------

def plot_difference_map(
    mesh: Any,
    model_a: np.ndarray,
    model_b: np.ndarray,
    *,
    mode: str = "difference",
    ax: Any = None,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    label: str = "",
    title: str = "",
    coverage: Optional[np.ndarray] = None,
) -> Tuple:
    """Plot the difference or ratio between two models.

    Parameters
    ----------
    mesh : pygimli.Mesh
    model_a, model_b : array-like
        Two model arrays.  ``result = model_b - model_a`` (difference) or
        ``model_b / model_a`` (ratio).
    mode : ``'difference'`` | ``'ratio'`` | ``'percent_change'``
    symmetric : bool
        If *True*, center the colorbar at zero (difference) or one (ratio).
    coverage : array-like, optional
        Coverage mask.

    Returns
    -------
    fig, ax, cbar
    """
    import pygimli as pg

    a = np.asarray(model_a, dtype=float).ravel()
    b = np.asarray(model_b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("model_a and model_b must have the same size.")

    if mode == "difference":
        vals = b - a
    elif mode == "ratio":
        vals = np.where(np.abs(a) > 1e-30, b / a, np.nan)
    elif mode == "percent_change":
        vals = np.where(np.abs(a) > 1e-30, 100.0 * (b - a) / np.abs(a), np.nan)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'difference', 'ratio', or 'percent_change'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    kw = dict(cMap=cmap, label=label or mode.replace("_", " ").title(),
              orientation="vertical", pad=0.3)
    if symmetric:
        vmax = np.nanmax(np.abs(vals))
        if mode == "ratio":
            kw["cMin"] = 1.0 / max(vmax, 1.01)
            kw["cMax"] = max(vmax, 1.01)
        else:
            kw["cMin"] = -vmax
            kw["cMax"] = vmax
    if coverage is not None:
        kw["coverage"] = np.asarray(coverage).ravel() > -1

    ax, cbar = pg.show(mesh, vals, ax=ax, **kw)
    if title:
        ax.set_title(title)
    return fig, ax, cbar


# ---------------------------------------------------------------------------
# Convergence / misfit
# ---------------------------------------------------------------------------

def plot_convergence(
    chi2_history: Sequence[float],
    *,
    ax: Any = None,
    target_chi2: float = 1.0,
    ylabel: str = r"$\chi^2$",
    title: str = "Inversion Convergence",
) -> Tuple:
    """Plot chi-squared convergence curve.

    Parameters
    ----------
    chi2_history : sequence of float
        Chi-squared value per iteration.
    target_chi2 : float
        Target misfit (plotted as a dashed line).

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    iters = np.arange(1, len(chi2_history) + 1)
    ax.semilogy(iters, chi2_history, "ko-", markersize=5, lw=1.5)
    ax.axhline(target_chi2, color="red", ls="--", lw=1, label=f"Target ({target_chi2})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Pseudosection (apparent resistivity matrix)
# ---------------------------------------------------------------------------

def plot_pseudosection_matrix(
    data_matrix: np.ndarray,
    *,
    ax: Any = None,
    cmap: Any = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlabel: str = "Time",
    ylabel: str = "Measurement #",
    label: str = r"Apparent resistivity ($\Omega\cdot$m)",
    title: str = "",
) -> Tuple:
    """Plot a time-lapse apparent resistivity matrix as a heatmap.

    Parameters
    ----------
    data_matrix : 2-D array
        Shape ``(n_times, n_measurements)`` or similar.

    Returns
    -------
    fig, ax, im
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    cmap = _get_cmap(cmap)
    im = ax.imshow(data_matrix.T if data_matrix.shape[0] < data_matrix.shape[1] else data_matrix,
                   aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax, im


# ---------------------------------------------------------------------------
# Electrode / sensor layout
# ---------------------------------------------------------------------------

def plot_electrode_layout(
    positions: Dict[str, np.ndarray],
    *,
    ax: Any = None,
    color_by: str = "z",
    cmap: str = "terrain",
    title: str = "Electrode Layout",
) -> Tuple:
    """Scatter-plot electrode positions colored by elevation.

    Parameters
    ----------
    positions : dict
        Must contain ``'x'`` and ``'y'`` keys; optionally ``'z'``.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    x = np.asarray(positions["x"])
    y = np.asarray(positions["y"])
    c = np.asarray(positions.get(color_by, np.zeros_like(x)))

    sc = ax.scatter(x, y, c=c, cmap=cmap, s=80, edgecolors="black", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label=f"{color_by.upper()} (m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Surface / topography
# ---------------------------------------------------------------------------

def plot_topography(
    topo_grid: np.ndarray,
    *,
    profile_endpoints: Optional[List[Tuple[float, float]]] = None,
    ax: Any = None,
    cmap: str = "terrain",
    title: str = "Surface Topography",
) -> Tuple:
    """Plot a 2-D topography grid with optional profile line overlay.

    Parameters
    ----------
    topo_grid : 2-D array
        Elevation raster.
    profile_endpoints : list of (row, col) tuples, optional
        If two points are given, draw the profile line.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.imshow(topo_grid, cmap=cmap, origin="lower")
    fig.colorbar(im, ax=ax, label="Elevation (m)")
    if profile_endpoints is not None and len(profile_endpoints) >= 2:
        p1, p2 = profile_endpoints[0], profile_endpoints[1]
        ax.plot(p1[1], p1[0], "ro", markersize=8, label="Start")
        ax.plot(p2[1], p2[0], "bo", markersize=8, label="End")
        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], "r--", lw=1.5)
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Monitoring-point time-series
# ---------------------------------------------------------------------------

def plot_monitoring_timeseries(
    times: np.ndarray,
    series: Dict[str, np.ndarray],
    *,
    true_series: Optional[Dict[str, np.ndarray]] = None,
    uncertainties: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ax: Any = None,
    ylabel: str = "Value",
    title: str = "Monitoring Point Time Series",
) -> Tuple:
    """Plot estimated (and optionally true) time-series at monitoring points.

    Parameters
    ----------
    times : array-like
        Time axis.
    series : dict of str -> array
        Estimated values keyed by point name.
    true_series : dict of str -> array, optional
        True / reference values for comparison (dashed lines).
    uncertainties : dict of str -> (lower, upper), optional
        Uncertainty bounds per point for shading.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    t = np.asarray(times)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, vals) in enumerate(series.items()):
        c = colors[i % len(colors)]
        ax.plot(t, vals, "o-", color=c, label=f"{name} (est.)", markersize=4)
        if true_series is not None and name in true_series:
            ax.plot(t, true_series[name], "--", color=c, label=f"{name} (true)")
        if uncertainties is not None and name in uncertainties:
            lo, hi = uncertainties[name]
            ax.fill_between(t, lo, hi, alpha=0.2, color=c)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Apparent-resistivity pseudosection with topography (SimPEG)
# ---------------------------------------------------------------------------

def _convert_pygimli_to_simpeg(data_obj):
    """Convert a PyGimli ERT DataContainer to SimPEG DC survey + apparent resistivity.

    Parameters
    ----------
    data_obj : pygimli.DataContainer or str
        PyGimli ERT data object, or file path to a ``.dat`` file.

    Returns
    -------
    dc_data : SimPEG Data object
        SimPEG DC data object containing the survey and apparent resistivity.
    topo_xyz : np.ndarray
        Topography array of shape (n_electrodes, 3) with (x, 0, z) columns.
    """
    import pygimli as pg
    from pygimli.physics import ert as pgert
    from SimPEG import data as simpeg_data
    from SimPEG.electromagnetics.static import resistivity as dc

    # Load if path
    if isinstance(data_obj, (str, os.PathLike)):
        data_obj = pgert.load(str(data_obj))

    # Extract electrode positions
    xx = np.asarray(pg.x(data_obj.sensorPositions()))
    yy = np.asarray(pg.y(data_obj.sensorPositions()))

    # Extract ABMN indices and data
    a_idx = np.asarray(data_obj["a"], dtype=int)
    b_idx = np.asarray(data_obj["b"], dtype=int)
    m_idx = np.asarray(data_obj["m"], dtype=int)
    n_idx = np.asarray(data_obj["n"], dtype=int)
    rhoa = np.asarray(data_obj["rhoa"])

    # Build SimPEG source list
    # Group by unique current electrode pairs (A, B)
    # Use 2D coordinates (x, z) for 2D pseudosection compatibility
    source_dict = {}
    for i in range(len(a_idx)):
        a_loc = np.array([xx[a_idx[i]], yy[a_idx[i]]])
        b_loc = np.array([xx[b_idx[i]], yy[b_idx[i]]])
        m_loc = np.array([xx[m_idx[i]], yy[m_idx[i]]])
        n_loc = np.array([xx[n_idx[i]], yy[n_idx[i]]])

        key = (a_idx[i], b_idx[i])
        if key not in source_dict:
            source_dict[key] = {
                "a_loc": a_loc,
                "b_loc": b_loc,
                "m_locs": [],
                "n_locs": [],
                "rhoa_vals": [],
            }
        source_dict[key]["m_locs"].append(m_loc)
        source_dict[key]["n_locs"].append(n_loc)
        source_dict[key]["rhoa_vals"].append(rhoa[i])

    source_list = []
    dobs_list = []
    for key, info in source_dict.items():
        m_locs = np.array(info["m_locs"])
        n_locs = np.array(info["n_locs"])
        rx = dc.receivers.Dipole(m_locs, n_locs)
        src = dc.sources.Dipole([rx], info["a_loc"], info["b_loc"])
        source_list.append(src)
        dobs_list.extend(info["rhoa_vals"])

    survey = dc.Survey(source_list)
    dc_data_out = simpeg_data.Data(survey, dobs=np.array(dobs_list))

    # Build topography: (x, z) for 2D profile
    topo_xyz = np.column_stack([xx, yy])

    return dc_data_out, topo_xyz


def plot_apparent_resistivity_pseudosection(
    data_obj: Any,
    *,
    ax: Any = None,
    plot_type: str = "scatter",
    cmap: Any = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    scale: str = "linear",
    label: str = r"Apparent resistivity ($\Omega\cdot$m)",
    title: str = "",
    scatter_marker: str = "s",
    scatter_size: float = 10,
    mask_topography: bool = True,
    show_colorbar: bool = True,
    cbar_opts: Optional[Dict] = None,
    figsize: Tuple[float, float] = (12, 5),
    data_locations: bool = False,
    clean_axes: bool = False,
    xlabel: str = "x (m)",
    ylabel: str = "Elevation (m)",
) -> Tuple:
    """Plot apparent resistivity pseudosection with topography using SimPEG.

    Converts PyGimli ERT data to SimPEG format and uses SimPEG's
    ``plot_pseudosection`` to render a pseudosection that honours surface
    topography.

    Parameters
    ----------
    data_obj : pygimli.DataContainer, str, or SimPEG Data
        PyGimli ERT data (or file path to a ``.dat``), or a pre-converted
        SimPEG ``Data`` object.  When a PyGimli object or path is passed the
        conversion is done automatically.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if *None*.
    plot_type : ``'scatter'`` | ``'pcolor'`` | ``'contourf'``
        Pseudosection rendering style.
    cmap : str or Colormap, optional
        Colormap.  Defaults to ``BlueDarkRed18_18`` (warm-to-cool).
    cmin, cmax : float, optional
        Color limits.
    scale : ``'linear'`` | ``'log'``
        Color scale.
    label : str
        Colorbar label.
    title : str
        Axes title.
    scatter_marker : str
        Marker style when *plot_type='scatter'*.
    scatter_size : float
        Marker size when *plot_type='scatter'*.
    mask_topography : bool
        If *True*, mask the region above topography.
    show_colorbar : bool
        Show the colorbar.
    cbar_opts : dict, optional
        Extra keyword arguments forwarded to the colorbar.
    figsize : tuple
        Figure size (only used when *ax* is *None*).
    data_locations : bool
        Show electrode locations on the plot.
    clean_axes : bool
        If *True*, remove spines and ticks for a clean look.
    xlabel, ylabel : str
        Axis labels.

    Returns
    -------
    fig, ax, cbar_or_mappable
    """
    import os as _os

    from SimPEG.electromagnetics.static.utils.static_utils import (
        plot_pseudosection,
    )

    # Determine if we need to convert from PyGimli
    try:
        from SimPEG import data as simpeg_data
        if isinstance(data_obj, simpeg_data.Data):
            dc_data = data_obj
            topo_xyz = None
        else:
            dc_data, topo_xyz = _convert_pygimli_to_simpeg(data_obj)
    except Exception:
        dc_data, topo_xyz = _convert_pygimli_to_simpeg(data_obj)

    # Default colormap: warm-to-cool resistivity map
    if cmap is None:
        try:
            from palettable.lightbartlein.diverging import BlueDarkRed18_18
            cmap = BlueDarkRed18_18.mpl_colormap
        except ImportError:
            cmap = plt.get_cmap("RdBu")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    clim = None
    if cmin is not None and cmax is not None:
        clim = [cmin, cmax]

    if cbar_opts is None:
        cbar_opts = {"location": "right", "shrink": 0.8, "aspect": 30}

    pcolor_opts = {"cmap": cmap}
    contourf_opts = {"cmap": cmap}
    scatter_opts = {"cmap": cmap, "marker": scatter_marker, "s": scatter_size}

    ax, cc = plot_pseudosection(
        dc_data,
        plot_type=plot_type,
        ax=ax,
        scale=scale,
        cbar_label=label,
        mask_topography=mask_topography,
        pcolor_opts=pcolor_opts,
        contourf_opts=contourf_opts,
        scatter_opts=scatter_opts,
        data_locations=data_locations,
        cbar_opts=cbar_opts,
        clim=clim,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if clean_axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig, ax, cc


def plot_apparent_resistivity_timelapse(
    data_objs: Any,
    *,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 4,
    plot_type: str = "scatter",
    cmap: Any = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    scale: str = "linear",
    label: str = r"Apparent resistivity ($\Omega\cdot$m)",
    scatter_marker: str = "s",
    scatter_size: float = 10,
    mask_topography: bool = True,
    figsize_per_panel: Tuple[float, float] = (4.0, 2.5),
    clean_axes: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 100,
) -> Tuple:
    """Plot a multi-panel time-lapse apparent resistivity pseudosection.

    Parameters
    ----------
    data_objs : sequence
        List of PyGimli DataContainers, file paths, or SimPEG Data objects.
    titles : sequence of str, optional
        Panel titles.
    ncols : int
        Number of columns.
    plot_type : str
        ``'scatter'``, ``'pcolor'``, or ``'contourf'``.
    cmap, cmin, cmax, scale, label :
        Colormap and scale parameters.
    figsize_per_panel : tuple
        (width, height) per subplot.
    clean_axes : bool
        Remove spines and ticks.
    save_path : str, optional
        If given, save the figure to this path.
    dpi : int
        Resolution for saving.

    Returns
    -------
    fig, axes
    """
    from SimPEG.electromagnetics.static.utils.static_utils import (
        plot_pseudosection,
    )

    # Default colormap
    if cmap is None:
        try:
            from palettable.lightbartlein.diverging import BlueDarkRed18_18
            cmap = BlueDarkRed18_18.mpl_colormap
        except ImportError:
            cmap = plt.get_cmap("RdBu")

    n = len(data_objs)
    nrows = int(np.ceil(n / ncols))
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    clim = None
    if cmin is not None and cmax is not None:
        clim = [cmin, cmax]

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        if idx >= n:
            ax.axis("off")
            continue

        # Convert data
        try:
            from SimPEG import data as simpeg_data
            if isinstance(data_objs[idx], simpeg_data.Data):
                dc_data = data_objs[idx]
            else:
                dc_data, _ = _convert_pygimli_to_simpeg(data_objs[idx])
        except Exception:
            dc_data, _ = _convert_pygimli_to_simpeg(data_objs[idx])

        scatter_opts = {"cmap": cmap, "marker": scatter_marker, "s": scatter_size}
        pcolor_opts = {"cmap": cmap}
        contourf_opts = {"cmap": cmap}

        ax, cc = plot_pseudosection(
            dc_data,
            plot_type=plot_type,
            ax=ax,
            scale=scale,
            cbar_label=label,
            mask_topography=mask_topography,
            pcolor_opts=pcolor_opts,
            contourf_opts=contourf_opts,
            scatter_opts=scatter_opts,
            data_locations=False,
            cbar_opts={"location": "right", "shrink": 0.8, "aspect": 30},
            clim=clim,
        )

        t = titles[idx] if titles and idx < len(titles) else f"Timestep {idx + 1}"
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        if clean_axes:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        # Remove individual colorbars except last
        if idx < n - 1:
            for child in ax.get_children():
                if hasattr(child, "colorbar") and child.colorbar:
                    try:
                        child.colorbar.remove()
                    except Exception:
                        pass
                    break

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes


# ---------------------------------------------------------------------------
# Sensitivity / coverage map
# ---------------------------------------------------------------------------

def plot_coverage(
    mesh: Any,
    coverage: np.ndarray,
    *,
    ax: Any = None,
    cmap: str = "YlGn",
    threshold: Optional[float] = None,
    title: str = "Data Coverage",
) -> Tuple:
    """Plot a coverage / sensitivity map.

    Parameters
    ----------
    mesh : pygimli.Mesh
    coverage : array-like
        Coverage values per cell.
    threshold : float, optional
        If given, overlay a contour at this level.

    Returns
    -------
    fig, ax, cbar
    """
    import pygimli as pg

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    arr = np.asarray(coverage, dtype=float).ravel()
    ax, cbar = pg.show(mesh, arr, ax=ax, cMap=cmap, label="Coverage",
                       orientation="vertical", pad=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax, cbar
