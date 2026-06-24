"""Gravity / magnetics anomaly tools and analytic forward modeling (Qt-free).

Pure numpy/scipy + analytic potential-field formulas (no SimPEG), so it runs
anywhere the desktop app does. Provides:

* anomaly separation (polynomial regional / residual),
* gridding of scattered station data and profile extraction,
* analytic forward modeling of buried bodies: gravity of a sphere and a right
  rectangular prism (Nagy 1966), and the total-field magnetic anomaly of an
  induced/uniformly magnetized sphere (a dipole),
* export of grids to ``.npy`` / VTK and configs to JSON.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX.qt_apps import io_utils

LogFn = Callable[[str], None]

_G = 6.67430e-11        # gravitational constant (m^3 kg^-1 s^-2)
_MGAL = 1.0e5           # m/s^2 -> mGal
_MU0_4PI = 1.0e-7       # mu0 / 4pi (T*m/A)


def _noop(_msg: str) -> None:
    return None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Anomaly separation + gridding + profiles
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Analytic forward modeling
# ---------------------------------------------------------------------------
def gravity_sphere(xobs: np.ndarray, yobs: np.ndarray, body: Dict[str, Any]) -> np.ndarray:
    """Vertical gravity (mGal) of a buried sphere. z positive down, obs at z=0."""
    x0 = float(body.get("x0", 0.0)); y0 = float(body.get("y0", 0.0))
    z0 = float(body.get("z0", 10.0)); R = float(body.get("radius", 5.0))
    drho = float(body.get("density_contrast", 300.0))
    mass = (4.0 / 3.0) * np.pi * R ** 3 * drho
    dx = np.asarray(xobs, float) - x0
    dy = np.asarray(yobs, float) - y0
    r = np.sqrt(dx ** 2 + dy ** 2 + z0 ** 2)
    gz = _G * mass * z0 / np.clip(r ** 3, 1e-12, None)
    return gz * _MGAL


def gravity_prism(xobs: np.ndarray, yobs: np.ndarray, body: Dict[str, Any]) -> np.ndarray:
    """Vertical gravity (mGal) of a right rectangular prism (Nagy 1966). z down."""
    drho = float(body.get("density_contrast", 300.0))
    x1 = float(body["x1"]); x2 = float(body["x2"])
    y1 = float(body["y1"]); y2 = float(body["y2"])
    z1 = float(body["z1"]); z2 = float(body["z2"])
    xo = np.atleast_1d(np.asarray(xobs, float)); yo = np.atleast_1d(np.asarray(yobs, float))
    out = np.zeros(xo.shape, dtype=float)
    for k, (xq, yq) in enumerate(zip(xo.ravel(), yo.ravel())):
        X = [x1 - xq, x2 - xq]; Y = [y1 - yq, y2 - yq]; Z = [z1, z2]
        total = 0.0
        for i in range(2):
            for j in range(2):
                for m in range(2):
                    mu = (-1.0) ** (i + j + m)
                    xi, yj, zk = X[i], Y[j], Z[m]
                    r = np.sqrt(xi ** 2 + yj ** 2 + zk ** 2)
                    arg1 = yj + r if (yj + r) > 1e-12 else 1e-12
                    arg2 = xi + r if (xi + r) > 1e-12 else 1e-12
                    term = (xi * np.log(arg1) + yj * np.log(arg2)
                            - zk * np.arctan2(xi * yj, zk * r + 1e-30))
                    total += mu * term
        out.ravel()[k] = _G * drho * total * _MGAL
    return out.reshape(np.asarray(xobs, float).shape)


def magnetic_dipole(xobs: np.ndarray, yobs: np.ndarray, body: Dict[str, Any],
                    field: Dict[str, Any]) -> np.ndarray:
    """Total-field magnetic anomaly (nT) of an induced/magnetized sphere (a dipole)."""
    x0 = float(body.get("x0", 0.0)); y0 = float(body.get("y0", 0.0))
    z0 = float(body.get("z0", 10.0)); R = float(body.get("radius", 5.0))
    chi = float(body.get("susceptibility", 0.05))
    B0_nt = float(field.get("strength", 50000.0))
    inc = np.radians(float(field.get("inclination", 60.0)))
    dec = np.radians(float(field.get("declination", 0.0)))
    # Ambient field unit vector (x=east, y=north, z=down).
    bhat = np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), np.sin(inc)])
    H0 = (B0_nt * 1e-9) / (4.0 * np.pi * _MU0_4PI)   # A/m
    moment = chi * H0 * (4.0 / 3.0) * np.pi * R ** 3  # A*m^2
    m_vec = moment * bhat
    dx = np.asarray(xobs, float) - x0
    dy = np.asarray(yobs, float) - y0
    dz = -z0 * np.ones_like(dx)   # obs above the body (body at +z down)
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    rhat = np.stack([dx, dy, dz], axis=-1) / np.clip(r[..., None], 1e-12, None)
    m_dot_r = rhat @ m_vec
    b_dip = _MU0_4PI * (3.0 * m_dot_r[..., None] * rhat - m_vec) / np.clip(r[..., None] ** 3, 1e-30, None)
    dT = (b_dip @ bhat) * 1e9   # T -> nT
    return dT


def forward_bodies(xobs: np.ndarray, yobs: np.ndarray, kind: str,
                   bodies: List[Dict[str, Any]], field: Optional[Dict[str, Any]] = None,
                   log: LogFn = _noop) -> np.ndarray:
    """Sum the anomaly of a list of bodies. kind = 'gravity' or 'magnetics'."""
    total = np.zeros(np.asarray(xobs, float).shape, dtype=float)
    for b in bodies:
        bt = str(b.get("type", "sphere")).lower()
        if kind == "gravity":
            total = total + (gravity_prism(xobs, yobs, b) if bt == "prism"
                             else gravity_sphere(xobs, yobs, b))
        else:
            total = total + magnetic_dipole(xobs, yobs, b, field or {})
    log(f"{kind} forward: {len(bodies)} body(ies)")
    return total


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def save_grid(grid: Dict[str, np.ndarray], out_dir: Path, name: str = "anomaly",
              log: LogFn = _noop) -> List[str]:
    """Save a grid to npy + CSV + VTK (best-effort). Return written paths."""
    out = io_utils.ensure_dir(out_dir)
    paths: List[str] = []
    xx, yy, zz = grid["xx"], grid["yy"], grid["zz"]
    np.save(out / f"{name}_grid.npy", np.asarray(zz, float)); paths.append(str(out / f"{name}_grid.npy"))
    rows = list(zip(xx.ravel().tolist(), yy.ravel().tolist(), np.asarray(zz, float).ravel().tolist()))
    io_utils.write_csv(out / f"{name}_grid.csv", rows, header=["x", "y", name])
    paths.append(str(out / f"{name}_grid.csv"))
    try:
        import pyvista as pv
        sg = pv.StructuredGrid(np.asarray(xx, float), np.asarray(yy, float),
                               np.zeros_like(np.asarray(xx, float)))
        sg[name] = np.asarray(zz, float).ravel(order="F")
        sg.save(str(out / f"{name}_grid.vtk"))
        paths.append(str(out / f"{name}_grid.vtk"))
    except Exception as exc:  # noqa: BLE001 - VTK is best-effort
        log(f"VTK export skipped: {exc}")
    return paths


def build_gravmag_config(kind: str, settings: Dict[str, Any], bodies: List[Dict[str, Any]],
                         field: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "created_time": _utc_now(),
        "kind": kind,
        "settings": dict(settings),
        "bodies": [dict(b) for b in bodies],
        "field": dict(field) if field else {},
    }
