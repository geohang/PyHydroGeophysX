"""Plan-view (map) gridding of scattered geophysical values.

A depth slice of a stitched TEM/AEM inversion, a gravity or magnetic station
product, an ERT layer average, or any other "one value per map position" result
is a scattered point set. Reading such a slice as coloured dots leaves the space
between soundings to the eye; this module fills it explicitly with a stated
interpolator so the layer can be read as an image and exported to GIS.

The entry point is :func:`plan_grid`. It is deliberately method-agnostic: it
takes map coordinates, one value per coordinate, and a choice of interpolator,
so the same code serves EM depth slices, potential-field station grids and any
future point product.

Ordinary kriging is implemented here on top of numpy/scipy (an empirical
semivariogram, a fitted spherical/exponential/Gaussian model, and the ordinary
kriging system with a Lagrange multiplier) rather than through ``gstools``, so
the map view carries no optional dependency.
:mod:`PyHydroGeophysX.core.kriging_3d` remains the 3D, gstools-backed path for
volume interpolation.

Two properties matter for honest maps and are enforced rather than left to the
caller:

* Values that live on a multiplicative scale -- resistivity above all -- must be
  interpolated in log space. Pass ``log_values=True``; the grid comes back in
  the original units.
* Interpolators extrapolate happily beyond the data. ``clip_to_hull`` (on by
  default) blanks everything outside the convex hull of the samples, and
  ``max_distance`` additionally blanks cells farther than a stated distance from
  any sample, which is what keeps a wide line spacing from reading as coverage.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import RBFInterpolator, griddata
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares
from scipy.spatial import Delaunay, cKDTree


METHODS = ('kriging', 'idw', 'linear', 'cubic', 'nearest', 'rbf')
VARIOGRAM_MODELS = ('spherical', 'exponential', 'gaussian')

#: Above this many samples the kriging system is solved on local neighbourhoods
#: instead of globally, which bounds both memory and solve time.
GLOBAL_KRIGING_LIMIT = 700

#: Neighbours used by local kriging and by inverse-distance weighting.
DEFAULT_NEIGHBORS = 24

#: Smallest nugget, as a fraction of the sill, allowed for a Gaussian model.
#: A Gaussian variogram is smooth at the origin, and without a nugget its
#: kriging matrix is numerically singular -- on a 320-sounding survey the
#: condition number falls from 4e19 to 2e4 once this floor applies, and the
#: wild over- and undershoot between lines disappears with it.
GAUSSIAN_NUGGET_FLOOR = 0.01


# ---------------------------------------------------------------------------
# sample preparation
# ---------------------------------------------------------------------------
def prepare_samples(xy: np.ndarray,
                    values: np.ndarray,
                    log_values: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Drop unusable samples and average exact coordinate duplicates.

    Duplicated positions make the kriging system singular, and a value that is
    zero or negative has no logarithm, so both are resolved here once instead of
    inside every interpolator.

    Args:
        xy: ``(n, 2)`` map coordinates.
        values: ``(n,)`` value at each coordinate.
        log_values: Also drop non-positive values, which cannot be logged.

    Returns:
        Cleaned ``(m, 2)`` coordinates and ``(m,)`` values, ``m <= n``.
    """
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Plan gridding needs map coordinates with shape (n, 2).')
    if values.size != len(xy):
        raise ValueError('Plan gridding needs one value for every map coordinate.')
    keep = np.isfinite(values) & np.isfinite(xy).all(axis=1)
    if log_values:
        keep &= values > 0
    xy, values = xy[keep], values[keep]
    if not len(values):
        raise ValueError('No positive finite values to interpolate in this layer.'
                         if log_values else 'No finite values to interpolate in this layer.')
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    if len(unique) != len(xy):
        inverse = inverse.ravel()
        totals = np.bincount(inverse, weights=values, minlength=len(unique))
        counts = np.bincount(inverse, minlength=len(unique))
        xy, values = unique, totals / counts
    return xy, values


def spread_ratio(xy: np.ndarray) -> float:
    """Return the minor/major principal-axis ratio of a point set.

    A ratio near zero means the positions are effectively collinear -- a single
    flight or tow line -- and a filled plan map of them would show the
    interpolator rather than the survey.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return 0.0
    singular = np.linalg.svd(xy - xy.mean(axis=0), compute_uv=False)
    return 0.0 if singular[0] <= 0 else float(singular[1] / singular[0])


# ---------------------------------------------------------------------------
# variogram
# ---------------------------------------------------------------------------
def variogram_function(model: str, nugget: float, sill: float, range_: float):
    """Return ``gamma(h)`` for a named theoretical semivariogram.

    ``range_`` is the practical range for every model: the exponential and
    Gaussian forms use the conventional ``3h/a`` scaling so gamma reaches 95% of
    the sill at ``h = range_``, which keeps fitted ranges comparable across
    models.
    """
    if model not in VARIOGRAM_MODELS:
        raise ValueError(f'Unknown variogram model {model!r}; expected one of {VARIOGRAM_MODELS}.')
    nugget = max(float(nugget), 0.0)
    partial = max(float(sill) - nugget, 0.0)
    range_ = max(float(range_), 1e-12)

    def gamma(h):
        h = np.asarray(h, dtype=float)
        if model == 'spherical':
            ratio = np.clip(h / range_, 0.0, 1.0)
            shape = 1.5 * ratio - 0.5 * ratio ** 3
        elif model == 'exponential':
            shape = 1.0 - np.exp(-3.0 * h / range_)
        else:
            shape = 1.0 - np.exp(-3.0 * (h / range_) ** 2)
        # gamma(0) = 0 keeps kriging an exact interpolator at the samples.
        return np.where(h > 0, nugget + partial * shape, 0.0)

    return gamma


def empirical_variogram(xy: np.ndarray,
                        values: np.ndarray,
                        n_bins: int = 15,
                        max_lag: Optional[float] = None,
                        max_points: int = 1200,
                        seed: int = 0) -> Dict[str, np.ndarray]:
    """Bin the omnidirectional semivariance of a scattered sample set.

    Args:
        xy: ``(n, 2)`` coordinates, already cleaned by :func:`prepare_samples`.
        values: ``(n,)`` values in the space that will be interpolated.
        n_bins: Number of lag bins between zero and ``max_lag``.
        max_lag: Largest separation to bin. Defaults to half the largest
            separation present, the usual limit beyond which bins hold too few
            independent pairs to mean anything.
        max_points: Random subsample size above which the pair cloud is
            estimated from a subset instead of all ``n(n-1)/2`` pairs.
        seed: Seed for that subsample, so a redraw reproduces the same fit.

    Returns:
        Dict with ``lags``, ``gamma`` and ``counts`` for the occupied bins.
    """
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    if len(values) < 3:
        raise ValueError('A variogram needs at least three samples.')
    if len(values) > max_points:
        chosen = np.random.default_rng(seed).choice(len(values), max_points, replace=False)
        xy, values = xy[chosen], values[chosen]
    i, j = np.triu_indices(len(values), k=1)
    separation = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
    semivariance = 0.5 * (values[i] - values[j]) ** 2
    limit = float(max_lag) if max_lag else float(separation.max()) / 2.0
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('All samples share one position; a variogram needs spread.')
    inside = (separation > 0) & (separation <= limit)
    # A bin means nothing without pairs in it, so a small survey gets fewer and
    # wider bins rather than a cloud of empty ones.
    n_bins = int(np.clip(int(n_bins), 3, max(3, int(np.count_nonzero(inside)) // 5)))
    edges = np.linspace(0.0, limit, n_bins + 1)
    index = np.clip(np.digitize(separation[inside], edges) - 1, 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    lag_sum = np.bincount(index, weights=separation[inside], minlength=n_bins)
    gamma_sum = np.bincount(index, weights=semivariance[inside], minlength=n_bins)
    # Prefer well-populated bins, but keep sparse ones rather than fail outright
    # on a small survey.
    occupied = counts >= 5
    if occupied.sum() < 3:
        occupied = counts > 0
    if occupied.sum() < 3:
        raise ValueError(f'{len(values)} samples give too few distinct separations to fit a '
                         'variogram. Krige a denser layer, or pick a deterministic method '
                         'such as inverse distance.')
    return {'lags': lag_sum[occupied] / counts[occupied],
            'gamma': gamma_sum[occupied] / counts[occupied],
            'counts': counts[occupied].astype(float)}


def fit_variogram(lags: np.ndarray,
                  gamma: np.ndarray,
                  counts: Optional[np.ndarray] = None,
                  model: str = 'auto') -> Dict[str, object]:
    """Fit a theoretical semivariogram to binned experimental values.

    Bins are weighted by their pair count, so a well-populated short lag counts
    for more than a sparse long one. With ``model='auto'`` every supported model
    is fitted and the lowest-RMSE one is returned.

    Returns:
        Dict with ``model``, ``nugget``, ``sill``, ``range``, ``rmse`` and the
        experimental ``lags`` / ``gamma`` / ``counts`` used, so the fit can be
        plotted and judged rather than trusted.
    """
    lags = np.asarray(lags, dtype=float).ravel()
    gamma = np.asarray(gamma, dtype=float).ravel()
    counts = (np.ones_like(lags) if counts is None
              else np.asarray(counts, dtype=float).ravel())
    if lags.size != gamma.size or lags.size != counts.size or lags.size < 3:
        raise ValueError('Variogram fitting needs at least three matching lag/gamma bins.')
    candidates = VARIOGRAM_MODELS if model in ('auto', None, '') else (str(model),)
    weights = np.sqrt(counts / counts.max())
    scale = float(np.max(gamma)) or 1.0
    span = float(np.max(lags))
    best = None
    for name in candidates:
        def residual(parameters, name=name):
            nugget, partial, range_ = parameters
            model_gamma = variogram_function(name, nugget, nugget + partial, range_)(lags)
            return (model_gamma - gamma) * weights

        floor = GAUSSIAN_NUGGET_FLOOR * scale if name == 'gaussian' else 0.0
        try:
            solution = least_squares(
                residual, [max(0.1 * scale, floor), 0.9 * scale, 0.5 * span],
                # A range beyond twice the largest reliable lag is not identifiable
                # from these pairs, and an unbounded one destabilises the solve.
                bounds=([floor, 0.0, span * 1e-3], [1.5 * scale, 4.0 * scale, 2.0 * span]),
                max_nfev=400)
        except Exception:  # noqa: BLE001 - one failed candidate must not lose the others
            continue
        nugget, partial, range_ = (float(v) for v in solution.x)
        if name == 'gaussian':
            # Hold the fitted sill and move variance into the nugget, since the
            # total sill is what the pair cloud actually determines.
            sill = nugget + partial
            nugget = max(nugget, GAUSSIAN_NUGGET_FLOOR * sill)
            partial = sill - nugget
        fitted = variogram_function(name, nugget, nugget + partial, range_)(lags)
        rmse = float(np.sqrt(np.mean((fitted - gamma) ** 2)))
        if best is None or rmse < best['rmse']:
            best = {'model': name, 'nugget': nugget, 'sill': nugget + partial,
                    'range': range_, 'rmse': rmse}
    if best is None:
        raise ValueError('No variogram model could be fitted to these samples.')
    return {**best, 'lags': lags, 'gamma': gamma, 'counts': counts}


def auto_variogram(xy: np.ndarray, values: np.ndarray, model: str = 'auto') -> Dict[str, object]:
    """Estimate and fit a semivariogram in one step."""
    experimental = empirical_variogram(xy, values)
    return fit_variogram(experimental['lags'], experimental['gamma'],
                         experimental['counts'], model=model)


# ---------------------------------------------------------------------------
# interpolators
# ---------------------------------------------------------------------------
def ordinary_kriging(xy: np.ndarray,
                     values: np.ndarray,
                     targets: np.ndarray,
                     variogram: Dict[str, object],
                     neighbors: Optional[int] = None,
                     chunk: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging of ``values`` onto ``targets``.

    The unbiasedness constraint is imposed with a Lagrange multiplier, so the
    returned spread is the full ordinary-kriging variance
    ``sum(w_i * gamma(x_i, x0)) + mu``, in the squared units of ``values`` --
    squared log10 units when the caller works in log space.

    Small surveys are solved once against every sample; larger ones are solved
    on the ``neighbors`` nearest samples per target, which bounds cost without
    visibly changing a map whose variogram range is shorter than the survey.

    Returns:
        ``(estimate, variance)``, each of length ``len(targets)``.
    """
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    if len(values) < 3:
        raise ValueError('Kriging needs at least three samples.')
    gamma = variogram_function(str(variogram['model']), variogram['nugget'],
                               variogram['sill'], variogram['range'])
    # A tiny ridge keeps near-coincident samples from making the system
    # singular; it acts as a negligible measurement-error nugget.
    ridge = 1e-9 * max(float(variogram['sill']), 1.0)
    estimate = np.empty(len(targets))
    variance = np.empty(len(targets))
    if neighbors is None and len(values) <= GLOBAL_KRIGING_LIMIT:
        count = len(values)
        left = np.ones((count + 1, count + 1))
        left[:count, :count] = gamma(np.hypot(xy[:, 0, None] - xy[None, :, 0],
                                              xy[:, 1, None] - xy[None, :, 1]))
        left[:count, :count] += ridge * np.eye(count)
        left[count, count] = 0.0
        factorized = lu_factor(left)
        for start in range(0, len(targets), chunk):
            block = targets[start:start + chunk]
            right = np.ones((count + 1, len(block)))
            right[:count] = gamma(np.hypot(xy[:, 0, None] - block[None, :, 0],
                                           xy[:, 1, None] - block[None, :, 1]))
            weights = lu_solve(factorized, right)
            estimate[start:start + chunk] = values @ weights[:count]
            variance[start:start + chunk] = (
                np.einsum('ij,ij->j', weights[:count], right[:count]) + weights[count])
        return estimate, np.maximum(variance, 0.0)
    count = int(min(neighbors or DEFAULT_NEIGHBORS, len(values)))
    _, nearest = cKDTree(xy).query(targets, k=count)
    nearest = nearest.reshape(len(targets), count)
    identity = ridge * np.eye(count)
    for start in range(0, len(targets), chunk):
        index = nearest[start:start + chunk]
        block = targets[start:start + chunk]
        points = xy[index]
        size = len(index)
        left = np.ones((size, count + 1, count + 1))
        left[:, :count, :count] = gamma(np.linalg.norm(
            points[:, :, None, :] - points[:, None, :, :], axis=-1)) + identity
        left[:, count, count] = 0.0
        right = np.ones((size, count + 1, 1))
        right[:, :count, 0] = gamma(np.linalg.norm(points - block[:, None, :], axis=-1))
        weights = np.linalg.solve(left, right)
        estimate[start:start + size] = np.einsum('ck,ck->c', values[index], weights[:, :count, 0])
        variance[start:start + size] = (
            np.einsum('ck,ck->c', weights[:, :count, 0], right[:, :count, 0]) + weights[:, count, 0])
    return estimate, np.maximum(variance, 0.0)


def inverse_distance(xy: np.ndarray,
                     values: np.ndarray,
                     targets: np.ndarray,
                     power: float = 2.0,
                     neighbors: Optional[int] = None) -> np.ndarray:
    """Inverse-distance weighted interpolation over the nearest samples."""
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    count = int(min(neighbors or DEFAULT_NEIGHBORS, len(values)))
    distance, index = cKDTree(xy).query(targets, k=count)
    distance = distance.reshape(len(targets), count)
    index = index.reshape(len(targets), count)
    exact = distance <= 0
    weights = np.where(exact, 1.0, 1.0 / np.maximum(distance, 1e-12) ** float(power))
    # A target sitting on a sample takes that sample's value, not a blend.
    hit = exact.any(axis=1)
    weights[hit] = exact[hit].astype(float)
    return np.einsum('ck,ck->c', values[index], weights) / weights.sum(axis=1)


# ---------------------------------------------------------------------------
# plan grid
# ---------------------------------------------------------------------------
def _grid_axes(xy: np.ndarray,
               resolution: int,
               padding: float,
               bounds: Optional[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Square-celled cell-centre axes covering the survey."""
    if bounds is not None:
        x0, x1, y0, y1 = (float(v) for v in bounds)
    else:
        x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
        y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
        pad = max(float(padding), 0.0) * max(x1 - x0, y1 - y0)
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    if not (x1 > x0 and y1 > y0):
        raise ValueError('The survey has no two-dimensional extent to grid.')
    resolution = int(max(resolution, 8))
    # Square cells keep map distances isotropic and make the ESRI ASCII export --
    # which carries a single cell size -- valid without resampling.
    cell = max(x1 - x0, y1 - y0) / resolution
    x = x0 + (np.arange(int(np.ceil((x1 - x0) / cell))) + 0.5) * cell
    y = y0 + (np.arange(int(np.ceil((y1 - y0) / cell))) + 0.5) * cell
    return x, y


def plan_grid(xy: np.ndarray,
              values: np.ndarray,
              method: str = 'kriging',
              resolution: int = 140,
              log_values: bool = False,
              variogram: Union[str, Dict[str, object], None] = 'auto',
              neighbors: Optional[int] = None,
              power: float = 2.0,
              smoothing: float = 0.0,
              clip_to_hull: bool = True,
              max_distance: Optional[float] = None,
              padding: float = 0.04,
              bounds: Optional[Sequence[float]] = None,
              min_spread: float = 0.02) -> Dict[str, object]:
    """Interpolate one scattered map layer onto a regular plan grid.

    Args:
        xy: ``(n, 2)`` map coordinates of the samples.
        values: ``(n,)`` value per sample, in physical units.
        method: One of :data:`METHODS`. ``'kriging'`` is ordinary kriging with a
            fitted variogram; ``'idw'`` is inverse-distance weighting;
            ``'linear'``, ``'cubic'`` and ``'nearest'`` use triangulation-based
            :func:`scipy.interpolate.griddata`; ``'rbf'`` is a thin-plate spline.
        resolution: Cells along the longer map axis. Cells are square.
        log_values: Interpolate ``log10(value)`` and return physical units.
            Required for anything log-normal, resistivity above all.
        variogram: ``'auto'`` fits nugget, sill and range; a model name restricts
            the fit to that shape; a dict is used as given. Kriging only.
        neighbors: Samples used per target by kriging and IDW.
        power: IDW distance exponent.
        smoothing: Thin-plate-spline smoothing; 0 interpolates exactly.
        clip_to_hull: Blank cells outside the convex hull of the samples, so the
            map never presents extrapolation as if it were data.
        max_distance: Additionally blank cells farther than this from any sample,
            in the units of ``xy``. ``None`` or 0 disables it.
        padding: Fraction of the survey extent added around it, before clipping.
        bounds: Explicit ``(xmin, xmax, ymin, ymax)`` instead of the data extent.
        min_spread: Smallest minor/major axis ratio accepted. Below it the
            samples are treated as a single line and gridding is refused.

    Returns:
        Dict with ``grid`` ``(ny, nx)`` in physical units and NaN where blanked,
        ``x`` / ``y`` cell centres, ``x_edges`` / ``y_edges`` for
        ``pcolormesh``, ``cell_size``, ``variance`` (kriging only, in
        interpolated-space squared units, NaN where blanked), ``variogram`` (the
        fit actually used), ``n_samples``, ``coverage`` and the settings applied.
    """
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f'Unknown interpolation method {method!r}; expected one of {METHODS}.')
    xy, values = prepare_samples(xy, values, log_values=log_values)
    minimum = 4 if method in ('cubic', 'rbf') else 3
    if len(values) < minimum:
        raise ValueError(f'{method} gridding needs at least {minimum} usable samples; '
                         f'this layer has {len(values)}.')
    if spread_ratio(xy) < float(min_spread):
        raise ValueError('These positions are effectively a single line. A plan map of them '
                         'would show the interpolator, not the survey; use the section view.')
    work = np.log10(values) if log_values else values
    x, y = _grid_axes(xy, resolution, padding, bounds)
    mesh_x, mesh_y = np.meshgrid(x, y)
    targets = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    keep = np.ones(len(targets), dtype=bool)
    if clip_to_hull:
        try:
            keep &= Delaunay(xy).find_simplex(targets) >= 0
        except Exception as exc:  # noqa: BLE001 - a degenerate hull is a data problem
            raise ValueError(f'Cannot outline the survey for clipping: {exc}') from exc
    if max_distance:
        keep &= cKDTree(xy).query(targets)[0] <= float(max_distance)
    if not keep.any():
        raise ValueError('Every grid cell was blanked. Widen the blanking distance '
                         'or turn off convex-hull clipping.')

    wanted = targets[keep]
    fit = None
    variance = None
    if method == 'kriging':
        fit = dict(variogram) if isinstance(variogram, dict) else auto_variogram(
            xy, work, model=variogram or 'auto')
        estimate, spread = ordinary_kriging(xy, work, wanted, fit, neighbors=neighbors)
        variance = np.full(len(targets), np.nan)
        variance[keep] = spread
    elif method == 'idw':
        estimate = inverse_distance(xy, work, wanted, power=power, neighbors=neighbors)
    elif method == 'rbf':
        estimate = RBFInterpolator(xy, work, kernel='thin_plate_spline',
                                   smoothing=float(smoothing),
                                   neighbors=min(len(work), 64))(wanted)
    else:
        estimate = griddata(xy, work, wanted, method=method)

    flat = np.full(len(targets), np.nan)
    flat[keep] = estimate
    grid = flat.reshape(mesh_x.shape)
    if log_values:
        grid = np.power(10.0, grid)
    cell = float(x[1] - x[0]) if x.size > 1 else float(y[1] - y[0])
    edges = [np.r_[axis - cell / 2, axis[-1] + cell / 2] for axis in (x, y)]
    return {'grid': grid, 'x': x, 'y': y, 'x_edges': edges[0], 'y_edges': edges[1],
            'cell_size': cell,
            'variance': None if variance is None else variance.reshape(mesh_x.shape),
            'variogram': fit, 'method': method, 'log_values': bool(log_values),
            'n_samples': int(len(values)), 'clipped': bool(clip_to_hull),
            'max_distance': None if not max_distance else float(max_distance),
            'coverage': float(np.count_nonzero(keep)) / keep.size}


def write_plan_grid(result: Dict[str, object], path, nodata: float = -9999.0) -> str:
    """Write a :func:`plan_grid` result to ``.asc`` (ESRI ASCII) or ``.csv``.

    The ASCII grid carries the cell size and lower-left corner, so it opens
    directly in QGIS/ArcGIS in whatever frame the coordinates were given in. The
    CSV is a plain ``x,y,value`` table of the cells that were not blanked.
    """
    from pathlib import Path

    path = Path(path)
    grid = np.asarray(result['grid'], dtype=float)
    x = np.asarray(result['x'], dtype=float)
    y = np.asarray(result['y'], dtype=float)
    cell = float(result['cell_size'])
    if path.suffix.lower() == '.csv':
        mesh_x, mesh_y = np.meshgrid(x, y)
        finite = np.isfinite(grid)
        table = np.column_stack([mesh_x[finite], mesh_y[finite], grid[finite]])
        np.savetxt(path, table, delimiter=',', header='x,y,value', comments='', fmt='%.6g')
        return str(path)
    if path.suffix.lower() not in ('.asc', '.txt'):
        raise ValueError('Plan grids export as .asc (ESRI ASCII) or .csv.')
    # ESRI ASCII rows run north to south.
    body = np.where(np.isfinite(grid), grid, float(nodata))[::-1]
    header = (f'ncols {grid.shape[1]}\nnrows {grid.shape[0]}\n'
              f'xllcorner {x[0] - cell / 2:.6f}\nyllcorner {y[0] - cell / 2:.6f}\n'
              f'cellsize {cell:.6f}\nNODATA_value {nodata:g}')
    np.savetxt(path, body, header=header, comments='', fmt='%.6g')
    return str(path)
