"""Analytic gravity and magnetic forward operators."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop

LogFn = Callable[[str], None]

_G = 6.67430e-11        # gravitational constant (m^3 kg^-1 s^-2)


_MGAL = 1.0e5           # m/s^2 -> mGal


_MU0_4PI = 1.0e-7       # mu0 / 4pi (T*m/A)


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

__all__ = [
    "gravity_sphere",
    "gravity_prism",
    "magnetic_dipole",
    "forward_bodies",
]
