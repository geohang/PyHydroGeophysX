"""The impedance of a layered earth: the 1D MT forward problem.

For layers of resistivity ``rho_k`` and thickness ``h_k`` over a half-space,
the impedance at the surface follows from the bottom up (Wait's recursion):

    k_j = sqrt(i omega mu0 / rho_j),  Z_N = i omega mu0 / k_N,
    Z_j = z_j (Z_{j+1} + z_j tanh(k_j h_j)) / (z_j + Z_{j+1} tanh(k_j h_j)),

with ``z_j = i omega mu0 / k_j`` the layer's intrinsic impedance, in ohms and
e^{+i omega t}, so a half-space has ``rho_a = rho`` and a phase of 45 degrees.
The derivative of ``log rho_a`` and the phase with respect to ``log rho_j`` is
the same recursion differentiated, for the inversion.

Wait, J. R. (1954). On the relation between telluric currents and the Earth's
magnetic field. Geophysics, 19(2), 281-289. https://doi.org/10.1190/1.1437994
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .transfer_function import MU0


def _layers(resistivity: Any, thickness: Any) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(resistivity, dtype=float).reshape(-1)
    h = np.asarray(thickness if thickness is not None else [], dtype=float).reshape(-1)
    if h.size != rho.size - 1:
        raise ValueError(f"{rho.size} resistivities need {rho.size - 1} thicknesses, got {h.size}")
    if np.any(rho <= 0) or np.any(h <= 0):
        raise ValueError("resistivities and thicknesses must be positive")
    return rho, h


def _tanh(x: np.ndarray) -> np.ndarray:
    """tanh for Re x >= 0 without overflow: thick layers give exactly 1."""
    e = np.exp(-2.0 * x)
    return (1.0 - e) / (1.0 + e)


def impedance_1d(resistivity: Any, thickness: Any, frequency: Any) -> np.ndarray:
    """Surface impedance (ohm, e^{+i omega t}) of a layered earth at each frequency."""
    z, _ = _impedance_and_derivative(resistivity, thickness, frequency, derivative=False)
    return z


def _impedance_and_derivative(resistivity, thickness, frequency, derivative=True):
    rho, h = _layers(resistivity, thickness)
    omega = 2 * np.pi * np.asarray(frequency, dtype=float).reshape(-1)
    iwm = 1j * omega * MU0
    k = np.sqrt(iwm[:, None] / rho[None, :])
    intrinsic = iwm[:, None] / k
    n = rho.size
    Z = intrinsic[:, -1].copy()
    dZ = np.zeros((omega.size, n), dtype=complex) if derivative else None
    if derivative:
        # d z_j / d log rho_j = z_j / 2 for a half-space bottom.
        dZ[:, -1] = intrinsic[:, -1] / 2
    for j in range(n - 2, -1, -1):
        zj = intrinsic[:, j]
        t = _tanh(k[:, j] * h[j])
        num = Z + zj * t
        den = zj + Z * t
        Z_new = zj * num / den
        if derivative:
            # Chain rule: below-layer terms through Z, this layer through z_j and k_j.
            dZ_below = zj * (den - num * t) / den**2
            dZ[:, j + 1:] = dZ[:, j + 1:] * dZ_below[:, None]
            dz_j = zj / 2                                   # d z_j / d log rho_j
            dt_j = (1 - t**2) * (-k[:, j] / 2) * h[j]       # d tanh(k h) / d log rho_j
            dnum = dz_j * t + zj * dt_j
            dden = dz_j + Z * dt_j
            dZ[:, j] = (dz_j * num + zj * dnum) / den - zj * num * dden / den**2
        Z = Z_new
    return Z, dZ


def apparent_resistivity_1d(resistivity: Any, thickness: Any, frequency: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``(rho_a, phase_deg)`` of a layered earth."""
    Z = impedance_1d(resistivity, thickness, frequency)
    omega = 2 * np.pi * np.asarray(frequency, dtype=float).reshape(-1)
    return np.abs(Z) ** 2 / (omega * MU0), np.degrees(np.angle(Z))


def sensitivity_1d(resistivity: Any, thickness: Any, frequency: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Z`` and the derivatives of ``log10 rho_a`` and phase (rad) to ``log rho_j``.

    Returns ``(Z, d_log10_rho_a, d_phase)``, the latter two ``(n_freq, n_layers)``.
    """
    Z, dZ = _impedance_and_derivative(resistivity, thickness, frequency, derivative=True)
    # log rho_a = 2 log|Z| + const; d log|Z| = Re(dZ / Z); d phase = Im(dZ / Z).
    ratio = dZ / Z[:, None]
    return Z, 2 * ratio.real / np.log(10), ratio.imag
