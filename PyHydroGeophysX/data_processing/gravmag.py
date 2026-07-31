"""Gravity/magnetics preprocessing, QC, gridding, and profiles."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

def regional_residual(x: np.ndarray, y: np.ndarray, value: np.ndarray,
                       degree: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a polynomial regional trend of ``degree`` (1..3); return (regional, residual)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    v = np.asarray(value, dtype=float)
    terms = [np.ones_like(x)]
    for d in range(1, int(degree) + 1):
        for i in range(d + 1):
            terms.append((x ** (d - i)) * (y ** i))
    A = np.column_stack(terms)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    regional = A @ coef
    return regional, v - regional


def spatially_balanced_indices(x: np.ndarray, y: np.ndarray, max_stations: int) -> np.ndarray:
    """Return deterministic farthest-point indices for a spatially balanced subset.

    The previous evenly spaced file-row selection could over-sample a survey
    segment when input rows were ordered by flight line or acquisition order.
    Farthest-point selection starts near the survey centroid and repeatedly adds
    the station furthest from the selected set, preserving map coverage without
    a random seed.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    cap = max(1, int(max_stations))
    if n <= cap:
        return np.arange(n, dtype=int)
    sx = max(float(np.ptp(x)), 1.0)
    sy = max(float(np.ptp(y)), 1.0)
    xy = np.column_stack(((x - float(x.min())) / sx, (y - float(y.min())) / sy))
    center = np.mean(xy, axis=0)
    first = int(np.argmin(np.sum((xy - center) ** 2, axis=1)))
    selected = np.empty(cap, dtype=int)
    selected[0] = first
    min_dist2 = np.sum((xy - xy[first]) ** 2, axis=1)
    min_dist2[first] = -1.0
    for i in range(1, cap):
        chosen = int(np.argmax(min_dist2))
        selected[i] = chosen
        min_dist2 = np.minimum(min_dist2, np.sum((xy - xy[chosen]) ** 2, axis=1))
        min_dist2[selected[:i + 1]] = -1.0
    return selected


def qc_products(x: np.ndarray, y: np.ndarray, value: np.ndarray, *, detrend: int = 1,
                nx: int = 120, ny: int = 120) -> Dict[str, Any]:
    """Calculate observed, regional and residual products for map/profile QC."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    value = np.asarray(value, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(value)
    if int(good.sum()) < 3:
        raise ValueError("Need at least three finite stations for QC products.")
    x, y, value = x[good], y[good], value[good]
    degree = max(0, int(detrend))
    if degree == 0:
        regional = np.zeros_like(value)
        residual = value.copy()
    else:
        regional, residual = regional_residual(x, y, value, degree=degree)
    fields = {"Observed": value, "Regional": regional, "Residual": residual}
    grids = {name: grid_data(x, y, values, nx=nx, ny=ny) for name, values in fields.items()}
    stats = {
        name: {"min": float(np.nanmin(values)), "max": float(np.nanmax(values)),
               "mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
        for name, values in fields.items()
    }
    return {"x": x, "y": y, "fields": fields, "grids": grids, "stats": stats,
            "detrend": degree}


def grid_data(x: np.ndarray, y: np.ndarray, value: np.ndarray,
              nx: int = 120, ny: int = 120, method: str = "linear") -> Dict[str, np.ndarray]:
    """Grid scattered station values onto a regular map. Returns xx, yy, zz."""
    from scipy.interpolate import griddata
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    v = np.asarray(value, dtype=float)
    xi = np.linspace(float(x.min()), float(x.max()), int(nx))
    yi = np.linspace(float(y.min()), float(y.max()), int(ny))
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata((x, y), v, (xx, yy), method=method)
    if not np.all(np.isfinite(zz)):
        zz_n = griddata((x, y), v, (xx, yy), method="nearest")
        zz = np.where(np.isfinite(zz), zz, zz_n)
    return {"xx": xx, "yy": yy, "zz": zz}


def extract_profile(grid: Dict[str, np.ndarray], p1: Sequence[float], p2: Sequence[float],
                    n: int = 200) -> Dict[str, np.ndarray]:
    """Sample a gridded field along the line p1 -> p2 (bilinear)."""
    from scipy.interpolate import RegularGridInterpolator
    xx, yy, zz = grid["xx"], grid["yy"], grid["zz"]
    xi = xx[0, :]; yi = yy[:, 0]
    interp = RegularGridInterpolator((yi, xi), zz, bounds_error=False, fill_value=np.nan)
    t = np.linspace(0.0, 1.0, int(n))
    px = p1[0] + t * (p2[0] - p1[0])
    py = p1[1] + t * (p2[1] - p1[1])
    vals = interp(np.column_stack([py, px]))
    dist = np.hypot(px - p1[0], py - p1[1])
    return {"distance": dist, "x": px, "y": py, "value": vals}

__all__ = [
    "regional_residual",
    "spatially_balanced_indices",
    "qc_products",
    "grid_data",
    "extract_profile",
]
