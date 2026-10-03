"""The magnetotelluric transfer functions of one site: impedance and tipper.

One container serves every reader, writer and processing step, so a site read
from an EDI file, an EMTF XML file or a Z-file, or estimated from time series,
is the same object. Its conventions are fixed, and every reader converts to
them:

- **Impedance in ohms (SI)**, ``E [V/m] / H [A/m]``. Files usually carry field
  units, (mV/km)/nT, which are ``1 / (4 pi 1e-4)`` times larger.
- **Time dependence e^{+i omega t}**, the SEG EDI standard's: a half-space
  gives a phase of +45 degrees for Zxy and -135 degrees for Zyx.
- **Rotation** is the azimuth of the x axis, in degrees clockwise from north,
  for every frequency. Rotating ``Z' = R Z R^T`` and ``T' = T R^T``.
- **Errors** are one standard error of the real part and of the imaginary
  part alike - the square root of the EDI ``Z**.VAR``, Gamble's average of
  the two variances. When the full covariance is known (EMTF XML, Z-files and
  the robust estimator here: inverse signal power and residual covariance),
  it is kept, and errors are derived from it, so they rotate correctly.

The apparent resistivity is ``|Z|^2 / (omega mu0)``, and the phase
``atan2(Im Z, Re Z)`` in degrees; their errors follow from the impedance error
to first order, ``drho / rho = 2 dZ / |Z|`` and ``dphi = dZ / |Z|`` radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence

import numpy as np

MU0 = 4e-7 * np.pi
#: (mV/km)/nT -> ohm. ``E 1e-6 V/m`` over ``H = B / mu0`` with ``B 1e-9 T``.
FIELD_TO_OHM = 1e3 * MU0

#: Which impedance element holds which output/input pair.
Z_COMPONENTS = ("xx", "xy", "yx", "yy")
_Z_INDEX = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}


def _as_rotation(value: Any, n: int) -> np.ndarray:
    angle = np.asarray(0.0 if value is None else value, dtype=float).reshape(-1)
    if angle.size == 1:
        angle = np.full(n, float(angle[0]))
    if angle.size != n:
        raise ValueError(f"rotation has {angle.size} values for {n} frequencies")
    return np.where(np.isfinite(angle), angle, 0.0)


def rotation_matrix(angle_deg: Any) -> np.ndarray:
    """``R`` for a frame rotated clockwise by ``angle_deg``: shape ``(..., 2, 2)``.

    A vector's components in the rotated frame are ``R v``: the new x axis,
    at azimuth theta, is ``(cos theta, sin theta)`` in the old (north, east)
    frame.
    """
    theta = np.radians(np.asarray(angle_deg, dtype=float))
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1)], axis=-2)


@dataclass
class TransferFunction:
    """Impedance and tipper of one MT site, against frequency.

    ``z`` is ``(n, 2, 2)`` complex ohms, ``tipper`` ``(n, 1, 2)`` complex and
    dimensionless; either may be None. ``z_err`` and ``tipper_err`` are the
    standard errors of each real and imaginary part. ``inverse_signal_power``
    ``(n, 2, 2)`` and the residual covariances - ``z_residual_covariance``
    ``(n, 2, 2)`` for (Ex, Ey) and ``tipper_residual_covariance`` ``(n, 1, 1)``
    for Hz - are the full error model where a file or the estimator supplies
    it: ``var(Z_ij) = residual_ii S_jj``. Values that are missing are NaN.
    """

    frequency: np.ndarray
    z: Optional[np.ndarray] = None
    z_err: Optional[np.ndarray] = None
    tipper: Optional[np.ndarray] = None
    tipper_err: Optional[np.ndarray] = None
    rotation: Any = 0.0
    inverse_signal_power: Optional[np.ndarray] = None
    z_residual_covariance: Optional[np.ndarray] = None
    tipper_residual_covariance: Optional[np.ndarray] = None
    station: str = ""
    latitude: float = float("nan")
    longitude: float = float("nan")
    elevation: float = float("nan")
    #: Local coordinates (m), when the site belongs to a projected survey.
    x: float = float("nan")
    y: float = float("nan")
    declination: float = float("nan")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, dtype=float).reshape(-1)
        n = self.frequency.size
        if np.any(~np.isfinite(self.frequency)) or np.any(self.frequency <= 0):
            raise ValueError("frequencies must be finite and positive")
        self.rotation = _as_rotation(self.rotation, n)

        def _check(name, value, shape, dtype):
            if value is None:
                return None
            array = np.asarray(value, dtype=dtype)
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
            return array

        self.z = _check("z", self.z, (n, 2, 2), complex)
        self.tipper = _check("tipper", self.tipper, (n, 1, 2), complex)
        self.z_err = _check("z_err", self.z_err, (n, 2, 2), float)
        self.tipper_err = _check("tipper_err", self.tipper_err, (n, 1, 2), float)
        self.inverse_signal_power = _check(
            "inverse_signal_power", self.inverse_signal_power, (n, 2, 2), complex)
        self.z_residual_covariance = _check(
            "z_residual_covariance", self.z_residual_covariance, (n, 2, 2), complex)
        self.tipper_residual_covariance = _check(
            "tipper_residual_covariance", self.tipper_residual_covariance, (n, 1, 1), complex)
        if self.z is not None and self.z_err is None:
            self.z_err = self._errors_from_covariance(self.z_residual_covariance)
        if self.tipper is not None and self.tipper_err is None:
            self.tipper_err = self._errors_from_covariance(self.tipper_residual_covariance)

    # -- basic views ---------------------------------------------------------
    @property
    def n_frequencies(self) -> int:
        return int(self.frequency.size)

    @property
    def period(self) -> np.ndarray:
        return 1.0 / self.frequency

    @property
    def angular_frequency(self) -> np.ndarray:
        return 2.0 * np.pi * self.frequency

    @property
    def has_impedance(self) -> bool:
        return self.z is not None and bool(np.any(np.isfinite(self.z)))

    @property
    def has_tipper(self) -> bool:
        return self.tipper is not None and bool(np.any(np.isfinite(self.tipper)))

    @property
    def z_field_units(self) -> Optional[np.ndarray]:
        """The impedance in (mV/km)/nT."""
        return None if self.z is None else self.z / FIELD_TO_OHM

    def _errors_from_covariance(self, residual: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """``sqrt(residual_ii S_jj)`` per element, or None without a covariance."""
        if residual is None or self.inverse_signal_power is None:
            return None
        res = np.real(np.diagonal(residual, axis1=1, axis2=2))       # (n, outputs)
        sig = np.real(np.diagonal(self.inverse_signal_power, axis1=1, axis2=2))  # (n, 2)
        return np.sqrt(np.abs(res[:, :, None] * sig[:, None, :]))

    # -- derived quantities --------------------------------------------------
    def apparent_resistivity(self) -> np.ndarray:
        """``|Z|^2 / (omega mu0)`` in ohm-m, ``(n, 2, 2)``."""
        self._require_z()
        return np.abs(self.z) ** 2 / (self.angular_frequency[:, None, None] * MU0)

    def phase(self) -> np.ndarray:
        """Impedance phase in degrees, ``(n, 2, 2)``; Zyx sits near -135 for a half-space."""
        self._require_z()
        return np.degrees(np.angle(self.z))

    def apparent_resistivity_err(self) -> np.ndarray:
        """First-order standard error of the apparent resistivity, ohm-m."""
        self._require_z()
        err = self._z_err_or_nan()
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * self.apparent_resistivity() * err / np.abs(self.z)

    def phase_err(self) -> np.ndarray:
        """First-order standard error of the phase, degrees."""
        self._require_z()
        err = self._z_err_or_nan()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.degrees(np.clip(err / np.abs(self.z), 0.0, np.pi))

    def determinant(self) -> np.ndarray:
        """The rotation-invariant determinant impedance ``sqrt(Zxx Zyy - Zxy Zyx)``, ``(n,)``.

        The branch is the one with the phase of Zxy, so a half-space gives +45.
        """
        self._require_z()
        det = np.sqrt(self.z[:, 0, 0] * self.z[:, 1, 1] - self.z[:, 0, 1] * self.z[:, 1, 0])
        flip = np.real(det * np.conj(self.z[:, 0, 1] - self.z[:, 1, 0])) < 0
        return np.where(flip, -det, det)

    def _z_err_or_nan(self) -> np.ndarray:
        return self.z_err if self.z_err is not None else np.full(self.z.shape, np.nan)

    def _require_z(self) -> None:
        if self.z is None:
            raise ValueError(f"site {self.station or '?'} has no impedance")

    # -- transformations -----------------------------------------------------
    def copy(self) -> "TransferFunction":
        copies = {name: (None if value is None else np.array(value, copy=True))
                  for name, value in self._array_fields().items()}
        return replace(self, metadata=dict(self.metadata), **copies)

    def _array_fields(self) -> Dict[str, Optional[np.ndarray]]:
        return dict(frequency=self.frequency, z=self.z, z_err=self.z_err, tipper=self.tipper,
                    tipper_err=self.tipper_err, rotation=self.rotation,
                    inverse_signal_power=self.inverse_signal_power,
                    z_residual_covariance=self.z_residual_covariance,
                    tipper_residual_covariance=self.tipper_residual_covariance)

    def select(self, mask: Any) -> "TransferFunction":
        """The frequencies a boolean mask or an index array selects."""
        index = np.arange(self.n_frequencies)[np.asarray(mask)]
        picked = {name: (None if value is None else np.asarray(value)[index])
                  for name, value in self._array_fields().items()}
        return replace(self, metadata=dict(self.metadata), **picked)

    def sorted_by_period(self) -> "TransferFunction":
        """Frequencies in order of increasing period."""
        return self.select(np.argsort(self.period))

    def rotated(self, angle: Any) -> "TransferFunction":
        """The site with its x axis at azimuth ``angle`` (degrees from north, clockwise).

        ``angle`` is one value or one per frequency. With a full covariance the
        errors are recomputed from the rotated covariance; otherwise each
        rotated variance is ``sum (R_ik R_jl)^2 var_kl`` from the errors in the
        frame the site was read in - which assumes the elements independent, an
        approximation that does not depend on the path the rotations took.
        """
        target = _as_rotation(angle, self.n_frequencies)
        R = rotation_matrix(target - self.rotation)            # (n, 2, 2)
        RT = np.swapaxes(R, 1, 2)
        out = self.copy()
        out.rotation = target
        if self.z is not None:
            out.z = R @ self.z @ RT
        if self.tipper is not None:
            out.tipper = self.tipper @ RT
        if self.inverse_signal_power is not None:
            out.inverse_signal_power = R @ self.inverse_signal_power @ RT
        if self.z_residual_covariance is not None:
            out.z_residual_covariance = R @ self.z_residual_covariance @ RT
        covariance = out.inverse_signal_power is not None
        # Without a covariance the errors are always rotated from the frame the
        # site was read in, so rotating there and back returns them unchanged;
        # chained, each independent-variance rotation would widen them.
        base_rotation, base_z_err, base_t_err = self.metadata.get(
            "_error_frame", (self.rotation, self.z_err, self.tipper_err))
        R0 = rotation_matrix(target - base_rotation)
        if not covariance:
            out.metadata["_error_frame"] = (base_rotation, base_z_err, base_t_err)
        if self.z is not None:
            from_cov = out._errors_from_covariance(out.z_residual_covariance) if covariance else None
            out.z_err = from_cov if from_cov is not None else _rotate_variance(base_z_err, R0, R0)
        if self.tipper is not None:
            from_cov = (out._errors_from_covariance(out.tipper_residual_covariance)
                        if covariance else None)
            out.tipper_err = (from_cov if from_cov is not None else _rotate_variance(
                base_t_err, np.ones((R0.shape[0], 1, 1)), R0))
        return out

    def conjugated(self) -> "TransferFunction":
        """The site under the opposite time convention (e^{-i omega t} <-> e^{+i omega t})."""
        out = self.copy()
        for name in ("z", "tipper", "inverse_signal_power", "z_residual_covariance",
                     "tipper_residual_covariance"):
            value = getattr(out, name)
            if value is not None:
                setattr(out, name, np.conj(value))
        return out

    def summary(self) -> Dict[str, Any]:
        """A JSON-safe description of the site, for logs and workflow results."""
        return {
            "station": self.station,
            "latitude": _finite_or_none(self.latitude),
            "longitude": _finite_or_none(self.longitude),
            "elevation": _finite_or_none(self.elevation),
            "n_frequencies": self.n_frequencies,
            "period_range_s": [float(self.period.min()), float(self.period.max())],
            "impedance": self.has_impedance,
            "tipper": self.has_tipper,
            "full_covariance": self.inverse_signal_power is not None,
            "rotation_deg": float(np.median(self.rotation)),
            "source_format": self.metadata.get("source_format", ""),
        }


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def _rotate_variance(err: Optional[np.ndarray], left: np.ndarray, right: np.ndarray):
    """Rotate standard errors as independent variances: ``sum (L_ik R_jl)^2 var_kl``."""
    if err is None:
        return None
    var = np.asarray(err, dtype=float) ** 2
    var = np.where(np.isfinite(var), var, 0.0)
    rotated = np.einsum("nik,njl,nkl->nij", left ** 2, right ** 2, var)
    missing = ~np.isfinite(np.asarray(err, dtype=float))
    rotated = np.sqrt(rotated)
    if missing.any():
        # An element built from a missing one is missing too.
        reach = np.einsum("nik,njl,nkl->nij", left ** 2, right ** 2, missing.astype(float)) > 0
        rotated[reach] = np.nan
    return rotated


def from_apparent_resistivity(
    frequency: Sequence[float],
    rho: Dict[str, Sequence[float]],
    phase_deg: Dict[str, Sequence[float]],
    *,
    rho_err: Optional[Dict[str, Sequence[float]]] = None,
    phase_err_deg: Optional[Dict[str, Sequence[float]]] = None,
    **site: Any,
) -> TransferFunction:
    """A site from apparent resistivities and phases, keyed ``"xy"``, ``"yx"``, ...

    ``|Z| = sqrt(rho omega mu0)``, ``Z = |Z| e^{i phi}``. The impedance error is
    taken from the resistivity error, ``dZ = |Z| drho / (2 rho)``, or else from
    the phase error, ``dZ = |Z| dphi``. Elements not given are NaN.
    """
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    n = frequency.size
    omega = 2.0 * np.pi * frequency
    z = np.full((n, 2, 2), np.nan + 1j * np.nan)
    z_err = np.full((n, 2, 2), np.nan)
    for component, values in rho.items():
        i, j = _Z_INDEX[component.lower()]
        rho_c = np.asarray(values, dtype=float)
        phase_c = np.radians(np.asarray(phase_deg[component], dtype=float))
        modulus = np.sqrt(np.abs(rho_c) * omega * MU0)
        z[:, i, j] = modulus * np.exp(1j * phase_c)
        error = np.full(n, np.nan)
        if rho_err and component in rho_err:
            error = modulus * np.asarray(rho_err[component], dtype=float) / (2.0 * np.abs(rho_c))
        if phase_err_deg and component in phase_err_deg:
            from_phase = modulus * np.radians(np.asarray(phase_err_deg[component], dtype=float))
            error = np.where(np.isfinite(error), error, from_phase)
        z_err[:, i, j] = error
    return TransferFunction(frequency=frequency, z=z, z_err=z_err, **site)


def resolve_sign_convention(tf: TransferFunction, convention: str) -> TransferFunction:
    """``tf`` in e^{+i omega t}, given the convention it was read in.

    ``"+"`` keeps it, ``"-"`` conjugates it, and ``"auto"`` conjugates it only
    when, at four frequencies in five (of three or more), the phases of Zxy and
    Zyx lie closer to (-45, +135) - a half-space under e^{-i omega t} - than to
    (+45, -135). Distances rather than quadrants, because the phases of real
    data stray past 90 degrees at long periods. The choice is recorded in
    ``metadata["sign_convention"]`` and, when it changed the data, in the notes.
    """
    convention = str(convention).strip().lower()
    if convention in ("+", "plus", "+1", "exp(+iwt)"):
        tf.metadata["sign_convention"] = "+"
        return tf
    if convention in ("-", "minus", "-1", "exp(-iwt)"):
        out = tf.conjugated()
        out.metadata["sign_convention"] = "-"
        out.metadata.setdefault("notes", []).append("Conjugated from e^{-i omega t}.")
        return out
    if convention != "auto":
        raise ValueError("sign_convention must be '+', '-' or 'auto'")
    if tf.z is None:
        tf.metadata["sign_convention"] = "+"
        return tf
    phase_xy = np.degrees(np.angle(tf.z[:, 0, 1]))
    phase_yx = np.degrees(np.angle(tf.z[:, 1, 0]))
    valid = np.isfinite(phase_xy) & np.isfinite(phase_yx)

    def apart(a, b):
        return np.abs((a - b + 180.0) % 360.0 - 180.0)

    opposite = (apart(phase_xy, -45.0) + apart(phase_yx, 135.0)
                < apart(phase_xy, 45.0) + apart(phase_yx, -135.0))
    if valid.sum() >= 3 and opposite[valid].mean() >= 0.8:
        out = tf.conjugated()
        out.metadata["sign_convention"] = "-"
        out.metadata.setdefault("notes", []).append(
            "The phases lie in the quadrants of e^{-i omega t}; the data were conjugated.")
        return out
    tf.metadata["sign_convention"] = "+"
    return tf
