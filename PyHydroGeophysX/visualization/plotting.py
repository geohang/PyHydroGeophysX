"""2D plotting utilities for geophysical models and data."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    mesh,
    values: np.ndarray,
    *,
    ax=None,
    cmap=None,
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
    mesh,
    models: Sequence[np.ndarray],
    *,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 4,
    cmap=None,
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
    mesh,
    model_a: np.ndarray,
    model_b: np.ndarray,
    *,
    mode: str = "difference",
    ax=None,
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
    ax=None,
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
    ax=None,
    cmap=None,
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
    ax=None,
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
    ax=None,
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
    ax=None,
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
# Sensitivity / coverage map
# ---------------------------------------------------------------------------

def plot_coverage(
    mesh,
    coverage: np.ndarray,
    *,
    ax=None,
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
