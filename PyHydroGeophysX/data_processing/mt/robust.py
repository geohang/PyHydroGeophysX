"""Robust regression of one output's Fourier coefficients on two inputs.

The transfer function ``b`` in ``y = X b + r`` - y one band's Ex (or Ey, Hz)
coefficients, X the Hx, Hy ones - is estimated as the instrumental variable
``b = (R^H W X)^{-1} R^H W y``, with R = X for a single site and the remote
site's Hx, Hy for a remote reference (Gamble et al. 1979), and W diagonal
weights that a regression M-estimator sets from the residuals (Egbert &
Booker 1986; Chave et al. 1987):

1. least squares, then iterations with Huber weights ``min(1, c / |r_i|/s)``
   (c = 1.5) until ``b`` changes by less than ``tolerance``; the scale ``s``
   is re-estimated every time from the winsorized squared residuals, divided
   by ``1 - exp(-c^2)``, their expectation for complex Gaussian residuals;
2. a few iterations with the redescending weights
   ``exp(-exp(u0 (|r_i|/s - u0)))`` (u0 = 2.8), which reject outliers outright,
   with the scale correction computed for those weights;
3. optionally, bounded influence: points whose leverage (hat value) exceeds
   ``leverage_cutoff`` times its mean ``p/n`` are downweighted in proportion
   (after Chave & Thomson 2004).

The error model is EMTF's: ``var(b_j) = sigma^2 S_jj`` with the residual
variance ``sigma^2`` and ``S = (R^H W X)^{-1} (R^H W^2 R) (X^H W R)^{-1}``, the
inverse signal power (``(X^H X)^{-1}`` for least squares).

Gamble, T. D., Goubau, W. M. & Clarke, J. (1979). Magnetotellurics with a
remote magnetic reference. Geophysics, 44(1), 53-68.
https://doi.org/10.1190/1.1440923

Chave, A. D., Thomson, D. J. & Ander, M. E. (1987). On the robust estimation
of power spectra, coherences, and transfer functions. Journal of Geophysical
Research, 92(B1), 633-648. https://doi.org/10.1029/JB092iB01p00633

Chave, A. D. & Thomson, D. J. (2004). Bounded influence magnetotelluric
response function estimation. Geophysical Journal International, 157(3),
988-1006. https://doi.org/10.1111/j.1365-246X.2004.02203.x
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np


@dataclass
class RegressionConfig:
    huber: float = 1.5
    redescend: float = 2.8
    max_iterations: int = 10
    redescend_iterations: int = 2
    tolerance: float = 0.005
    leverage_cutoff: Optional[float] = None


@dataclass
class RegressionResult:
    coefficients: np.ndarray
    weights: np.ndarray
    residual_variance: float
    inverse_signal_power: np.ndarray
    residuals: np.ndarray
    coherence: float
    iterations: int


def _redescend(a: np.ndarray, u0: float) -> np.ndarray:
    return np.exp(-np.exp(np.minimum(u0 * (a - u0), 50.0)))


@lru_cache(maxsize=8)
def _redescend_correction(u0: float) -> float:
    """``E[w a^2] / E[w]`` for ``a = |r| / sigma`` of complex Gaussian residuals."""
    a = np.linspace(0.0, 12.0, 24001)
    pdf = 2 * a * np.exp(-a**2)
    w = _redescend(a, u0)
    integral = lambda f: float(np.sum((f[1:] + f[:-1]) * np.diff(a)) / 2)
    return integral(w * a**2 * pdf) / integral(w * pdf)


def _solve(y, X, R, w):
    RW = R.conj().T * w
    return np.linalg.solve(RW @ X, RW @ y)


def robust_regression(y: np.ndarray, X: np.ndarray, R: Optional[np.ndarray] = None,
                      config: Optional[RegressionConfig] = None) -> RegressionResult:
    """Estimate ``b`` in ``y = X b`` robustly; ``R`` is the remote reference (default X)."""
    cfg = config or RegressionConfig()
    y = np.asarray(y, dtype=complex)
    X = np.asarray(X, dtype=complex)
    R = X if R is None else np.asarray(R, dtype=complex)
    n, p = X.shape
    if n <= p:
        raise ValueError(f"{n} points cannot determine {p} coefficients")
    w = np.ones(n)
    b = _solve(y, X, R, w)
    r = y - X @ b
    sigma2 = float(np.mean(np.abs(r) ** 2))
    c = cfg.huber
    iterations = 0
    for iterations in range(1, cfg.max_iterations + 1):
        a = np.abs(r) / np.sqrt(sigma2)
        w = np.where(a <= c, 1.0, c / np.maximum(a, 1e-300))
        b_new = _solve(y, X, R, w)
        r = y - X @ b_new
        sigma2 = float(np.mean(np.minimum(np.abs(r) ** 2, c * c * sigma2)) / (1 - np.exp(-c * c)))
        change = np.max(np.abs(b_new - b) / np.maximum(np.abs(b), 1e-300))
        b = b_new
        if change < cfg.tolerance:
            break
    u0 = cfg.redescend
    for _ in range(cfg.redescend_iterations):
        a = np.abs(r) / np.sqrt(sigma2)
        w = _redescend(a, u0)
        if w.sum() <= p:
            break
        b = _solve(y, X, R, w)
        r = y - X @ b
        sigma2 = float(np.sum(w * np.abs(r) ** 2) / np.sum(w) / _redescend_correction(u0))
    if cfg.leverage_cutoff:
        XW = X.conj().T * w
        hat = w * np.einsum("ij,jk,ik->i", X.conj(), np.linalg.inv(XW @ X), X).real
        limit = cfg.leverage_cutoff * p / n
        w = w * np.minimum(1.0, limit / np.maximum(hat, 1e-300))
        b = _solve(y, X, R, w)
        r = y - X @ b
    RW = R.conj().T * w
    A = np.linalg.inv(RW @ X)
    S = A @ ((R.conj().T * w**2) @ R) @ A.conj().T
    effective = float(np.sum(w))
    sigma2 *= effective / max(effective - p, 1.0)
    power = float(np.sum(w * np.abs(y) ** 2))
    coherence = float(1.0 - np.sum(w * np.abs(r) ** 2) / power) if power > 0 else float("nan")
    return RegressionResult(b, w, sigma2, S, r, coherence, iterations)
