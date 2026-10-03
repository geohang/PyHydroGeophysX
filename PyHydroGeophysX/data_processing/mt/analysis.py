"""What a site's transfer functions say about the ground: dimensionality and depth.

- :func:`phase_tensor` - Caldwell et al. (2004): ``Phi = X^-1 Y`` for
  ``Z = X + iY`` is free of galvanic distortion; its principal phases, skew
  ``beta``, angle ``alpha``, strike ``alpha - beta`` and ellipticity.
  ``|beta| > 3 deg`` is the usual sign of a 3D response (Booker 2014).
- :func:`swift_skew`, :func:`bahr_skew`, :func:`swift_strike` - the classical
  rotational invariants (Swift 1967; Bahr 1988).
- :func:`induction_arrows` - the tipper as arrows, in the Parkinson
  convention (real arrows point towards conductors) or Wiese's.
- :func:`niblett_bostick` - an approximate resistivity-depth profile from
  apparent resistivity and phase (Bostick 1977; Jones 1983).
- :func:`estimate_static_shift`, :func:`static_shift_from_layers`,
  :func:`apply_static_shift` - static shift: the frequency-independent
  factor small near-surface bodies multiply apparent resistivity by, found
  against a reference (often a TEM sounding at the site, Sternberg et al.
  1988) and divided out.

Caldwell, T. G., Bibby, H. M. & Brown, C. (2004). The magnetotelluric phase
tensor. Geophysical Journal International, 158(2), 457-469.
https://doi.org/10.1111/j.1365-246X.2004.02281.x

Booker, J. R. (2014). The magnetotelluric phase tensor: a critical review.
Surveys in Geophysics, 35(1), 7-40. https://doi.org/10.1007/s10712-013-9234-2

Swift, C. M. (1967). A magnetotelluric investigation of an electrical
conductivity anomaly in the southwestern United States. PhD thesis, MIT.

Bahr, K. (1988). Interpretation of the magnetotelluric impedance tensor:
regional induction and local telluric distortion. Journal of Geophysics,
62(2), 119-127.

Bostick, F. X. (1977). A simple almost exact method of MT analysis. Workshop
on Electrical Methods in Geothermal Exploration, U.S. Geological Survey.

Jones, A. G. (1983). On the equivalence of the "Niblett" and "Bostick"
transformations in the magnetotelluric method. Journal of Geophysics, 53(1),
72-73.

Sternberg, B. K., Washburne, J. C. & Pellerin, L. (1988). Correction for the
static shift in magnetotellurics using transient electromagnetic soundings.
Geophysics, 53(11), 1459-1468. https://doi.org/10.1190/1.1442426
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .forward1d import apparent_resistivity_1d
from .transfer_function import MU0, TransferFunction


def _invariants(phi: np.ndarray) -> Dict[str, np.ndarray]:
    p11, p12, p21, p22 = phi[..., 0, 0], phi[..., 0, 1], phi[..., 1, 0], phi[..., 1, 1]
    pi1 = 0.5 * np.hypot(p11 - p22, p12 + p21)
    pi2 = 0.5 * np.hypot(p11 + p22, p12 - p21)
    alpha = 0.5 * np.degrees(np.arctan2(p12 + p21, p11 - p22))
    beta = 0.5 * np.degrees(np.arctan2(p12 - p21, p11 + p22))
    phi_max = np.degrees(np.arctan(pi2 + pi1))
    phi_min = np.degrees(np.arctan(pi2 - pi1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ellipticity = (phi_max - phi_min) / (phi_max + phi_min)
    return {"phi_max": phi_max, "phi_min": phi_min, "alpha": alpha, "beta": beta,
            "strike": alpha - beta, "ellipticity": ellipticity}


def phase_tensor(tf: TransferFunction, *, n_realizations: int = 0,
                 seed: int = 0) -> Dict[str, np.ndarray]:
    """The phase tensor and its invariants (degrees), per frequency.

    Returns ``phi`` ``(n, 2, 2)`` (in the TF's frame) and ``phi_max``,
    ``phi_min``, ``alpha``, ``beta`` (skew), ``strike`` and ``ellipticity``;
    ``alpha`` and ``strike`` are azimuths clockwise from north, modulo 180
    (strike has the usual 90-degree ambiguity). With ``n_realizations``,
    their standard errors (``*_err``) come from perturbing ``Z`` by its errors.
    """
    if not tf.has_impedance:
        raise ValueError("the phase tensor needs an impedance")
    phi = _phase_tensor(tf.z)
    rotation = np.asarray(tf.rotation, dtype=float)
    result = {"phi": phi, **_invariants(phi)}
    for key in ("alpha", "strike"):
        result[key] = (result[key] + rotation) % 180.0
    if n_realizations and tf.z_err is not None:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_realizations):
            noise = (rng.standard_normal(tf.z.shape) + 1j * rng.standard_normal(tf.z.shape)) / np.sqrt(2)
            draws.append(_invariants(_phase_tensor(tf.z + noise * np.nan_to_num(tf.z_err))))
        for key in ("phi_max", "phi_min", "alpha", "beta", "strike", "ellipticity"):
            values = np.stack([d[key] for d in draws])
            if key in ("alpha", "strike"):
                # angles wrap at 180 degrees: spread of the doubled angle
                doubled = np.radians(2 * values)
                spread = np.sqrt(-2 * np.log(np.clip(np.abs(np.mean(np.exp(1j * doubled), axis=0)), 1e-12, 1)))
                result[key + "_err"] = np.degrees(spread) / 2
            else:
                result[key + "_err"] = np.nanstd(values, axis=0)
    return result


def _phase_tensor(z: np.ndarray) -> np.ndarray:
    X, Y = z.real, z.imag
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.linalg.solve(X, Y)


def swift_skew(tf: TransferFunction) -> np.ndarray:
    """``|Zxx + Zyy| / |Zxy - Zyx|``; near 0 for 1D and 2D (Swift 1967)."""
    z = tf.z
    return np.abs(z[:, 0, 0] + z[:, 1, 1]) / np.abs(z[:, 0, 1] - z[:, 1, 0])


def bahr_skew(tf: TransferFunction) -> np.ndarray:
    """Bahr's phase-sensitive skew; above about 0.3 the response is 3D (Bahr 1988)."""
    z = tf.z
    s1, s2 = z[:, 0, 0] + z[:, 1, 1], z[:, 0, 1] + z[:, 1, 0]
    d1, d2 = z[:, 0, 0] - z[:, 1, 1], z[:, 0, 1] - z[:, 1, 0]
    commutator = lambda a, b: np.imag(np.conj(a) * b)
    return np.sqrt(np.abs(commutator(d1, s2) - commutator(s1, d2))) / np.abs(d2)


def swift_strike(tf: TransferFunction) -> np.ndarray:
    """The strike (deg clockwise from north, +-90 ambiguity) that minimizes ``|Zxx|^2 + |Zyy|^2``."""
    z = tf.z
    d, s = z[:, 0, 0] - z[:, 1, 1], z[:, 0, 1] + z[:, 1, 0]
    # |Zxx'|^2 + |Zyy'|^2 is smallest a quarter turn of 4 theta from where atan2 points.
    angle = 0.25 * np.degrees(np.arctan2(2 * np.real(d * np.conj(s)), np.abs(d) ** 2 - np.abs(s) ** 2)) + 45.0
    return (angle + np.asarray(tf.rotation)) % 180.0


def induction_arrows(tf: TransferFunction, *, convention: str = "parkinson") -> Dict[str, np.ndarray]:
    """Real and imaginary induction arrows: lengths and azimuths (deg clockwise from north).

    ``"parkinson"`` reverses the real arrow so it points at current
    concentrations; ``"wiese"`` keeps the tipper's sign.
    """
    if not tf.has_tipper:
        raise ValueError("induction arrows need a tipper")
    t = tf.tipper[:, 0, :]
    sign = -1.0 if convention == "parkinson" else 1.0
    if convention not in ("parkinson", "wiese"):
        raise ValueError("convention must be 'parkinson' or 'wiese'")
    rotation = np.asarray(tf.rotation)
    out = {}
    for part, values in (("real", t.real), ("imag", t.imag)):
        x, y = sign * values[:, 0], sign * values[:, 1]
        out[f"{part}_length"] = np.hypot(x, y)
        out[f"{part}_azimuth"] = (np.degrees(np.arctan2(y, x)) + rotation) % 360.0
    return out


def niblett_bostick(tf: TransferFunction, *, mode: str = "det") -> Dict[str, np.ndarray]:
    """``depth`` (m) and ``resistivity`` (ohm m) by the Niblett-Bostick transform.

    ``mode`` is ``"det"`` (the determinant impedance), ``"xy"`` or ``"yx"``.
    ``rho_NB = rho_a (pi / (2 phi) - 1)``, ``depth = sqrt(rho_a / (omega mu0))``.
    """
    if mode == "det":
        z = tf.determinant()
    elif mode in ("xy", "yx"):
        z = tf.z[:, 0, 1] if mode == "xy" else -tf.z[:, 1, 0]
    else:
        raise ValueError("mode must be 'det', 'xy' or 'yx'")
    omega = tf.angular_frequency
    rho_a = np.abs(z) ** 2 / (omega * MU0)
    phase = np.angle(z)
    phase = np.where(phase > np.pi / 2, phase - np.pi, np.where(phase < 0, phase + np.pi, phase))
    with np.errstate(invalid="ignore", divide="ignore"):
        resistivity = rho_a * (np.pi / (2 * phase) - 1)
    depth = np.sqrt(rho_a / (omega * MU0))
    valid = np.isfinite(resistivity) & (resistivity > 0)
    return {"depth": depth, "resistivity": np.where(valid, resistivity, np.nan),
            "apparent_resistivity": rho_a, "phase": np.degrees(phase)}


def apply_static_shift(tf: TransferFunction, factor_x: float = 1.0,
                       factor_y: Optional[float] = None) -> TransferFunction:
    """Divide the apparent resistivities of the Ex row by ``factor_x`` and Ey's by ``factor_y``.

    ``factor_y`` defaults to ``factor_x``. The phases are unchanged; errors
    and covariances scale with the rows.
    """
    fy = factor_x if factor_y is None else factor_y
    scale = 1.0 / np.sqrt(np.array([float(factor_x), float(fy)]))
    out = tf.copy()
    out.z = tf.z * scale[None, :, None]
    if tf.z_err is not None:
        out.z_err = tf.z_err * scale[None, :, None]
    if tf.z_residual_covariance is not None:
        out.z_residual_covariance = tf.z_residual_covariance * np.outer(scale, scale)[None]
    notes = list(out.metadata.get("notes", []))
    notes.append(f"static shift removed: rho_a of Ex row / {factor_x:g}, Ey row / {fy:g}")
    out.metadata["notes"] = notes
    out.metadata["static_shift"] = {"x": float(factor_x), "y": float(fy)}
    return out


def estimate_static_shift(tf: TransferFunction, reference_rho_a: Any, *,
                          period_range: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """Static-shift factors of the Ex and Ey rows against a reference apparent resistivity.

    ``reference_rho_a`` is an array at ``tf.frequency`` or a callable of
    frequency. The factor is the geometric mean of ``rho_a / reference`` over
    ``period_range`` (s), where the reference holds - at the short periods a
    TEM sounding sees. Returns ``{"x", "y", "x_spread", "y_spread", "n"}``.
    """
    reference = reference_rho_a(tf.frequency) if callable(reference_rho_a) else np.asarray(reference_rho_a, float)
    rho = tf.apparent_resistivity()
    keep = np.isfinite(reference) & (reference > 0)
    if period_range is not None:
        keep &= (tf.period >= min(period_range)) & (tf.period <= max(period_range))
    if not np.any(keep):
        raise ValueError("no periods where the reference and the data overlap")
    result: Dict[str, float] = {"n": int(np.count_nonzero(keep))}
    for name, (i, j) in (("x", (0, 1)), ("y", (1, 0))):
        logs = np.log10(rho[keep, i, j] / reference[keep])
        logs = logs[np.isfinite(logs)]
        result[name] = float(10 ** np.median(logs)) if logs.size else float("nan")
        result[name + "_spread"] = float(np.std(logs)) if logs.size else float("nan")
    return result


def static_shift_from_layers(tf: TransferFunction, resistivity: Any, thickness: Any, *,
                             period_range: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """Static-shift factors against the MT response of a layered model, such as a TEM inversion's.

    The model's apparent resistivity is computed at the site's frequencies
    (:func:`.forward1d.apparent_resistivity_1d`); ``period_range`` should keep
    to the periods whose skin depth the model resolves.
    """
    reference, _ = apparent_resistivity_1d(resistivity, thickness, tf.frequency)
    return estimate_static_shift(tf, reference, period_range=period_range)
