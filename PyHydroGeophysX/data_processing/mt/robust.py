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
3. optionally, bounded influence, after Chave & Thomson (2004): the M-estimate
   bounds the influence of a large residual but not of a large input - a
   burst of noise in the magnetic channels gives its points a large leverage
   and they pull the estimate however small their residuals. With
   ``bounded_influence``, every iteration also weights each point by its
   leverage, the hat value ``h_i = w_i x_i (X^H W X)^{-1} x_i^H`` (x_i a row):
   points above the ``leverage_probability`` point of the hat values'
   distribution for complex Gaussian inputs, Beta(p, n - p), get
   ``(cutoff / h_i)^2``. The leverage weights accumulate from iteration to
   iteration, so a point once downweighted stays so and none keeps more
   leverage than the cutoff. ``leverage_cutoff`` is a single-pass form: hat
   values above that many times their mean ``p/n`` are downweighted in
   proportion, after the M-estimate.

4. optionally, noise classes (the noise-weighted estimate of Chen 2020): when
   the noise grows with the signal - the residual power ``|r_i|^2`` rising
   with the input power ``q_i = |x_i|^2``, from more than one source - the
   residuals are heavy-tailed for every M-estimate. The points are split
   into ``k`` classes by k-medians on ``(q_i, |r_i|^2)``; in each class a
   bisquare line gives the noise power ``a + b q_i``, and every row of the
   regression is divided by its square root, so the noise is the same
   everywhere; the M-estimate is then made again. ``k`` grows from 1 until
   the normalised residuals' stable index ``alpha`` (McCulloch's quantile
   estimate; 2 is Gaussian) reaches ``noise_alpha``; failing that, the
   ``k`` with the largest ``alpha`` is kept, and none if no ``k`` beats the
   unweighted residuals.

The error model is EMTF's: ``var(b_j) = sigma^2 S_jj`` with the residual
variance ``sigma^2`` and ``S = (R^H W X)^{-1} (R^H W^2 R) (X^H W R)^{-1}``, the
inverse signal power (``(X^H X)^{-1}`` for least squares). With
``jackknife_groups``, the variance is also estimated without that model, by
the delete-a-group jackknife (Chave & Thomson 1989): the points' windows, in
time order, are cut into that many groups, the estimate is made again without
each group at the final weights, and the spread of those estimates is the
variance. A group holds whole windows, so the correlation between a window's
harmonics - which the parametric error needs inflated for - is carried by it.
Both variances are of the complex coefficient, ``E|db|^2``: the sum of its
real and imaginary parts' variances.

Chave, A. D. & Thomson, D. J. (1989). Some comments on magnetotelluric
response function estimation. Journal of Geophysical Research, 94(B10),
14215-14225. https://doi.org/10.1029/JB094iB10p14215

Gamble, T. D., Goubau, W. M. & Clarke, J. (1979). Magnetotellurics with a
remote magnetic reference. Geophysics, 44(1), 53-68.
https://doi.org/10.1190/1.1440923

Chave, A. D., Thomson, D. J. & Ander, M. E. (1987). On the robust estimation
of power spectra, coherences, and transfer functions. Journal of Geophysical
Research, 92(B1), 633-648. https://doi.org/10.1029/JB092iB01p00633

Chave, A. D. & Thomson, D. J. (2004). Bounded influence magnetotelluric
response function estimation. Geophysical Journal International, 157(3),
988-1006. https://doi.org/10.1111/j.1365-246X.2004.02203.x

Chen, H. (2020). Study on the processing method of magnetotelluric impedance
estimation for non-Gaussian noise. Undergraduate thesis, School of
Geosciences and Info-Physics, Central South University (advisor R. Guo).

McCulloch, J. H. (1986). Simple consistent estimators of stable distribution
parameters. Communications in Statistics - Simulation and Computation,
15(4), 1109-1136. https://doi.org/10.1080/03610918608812563
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np


@dataclass
class RegressionConfig:
    huber: float = 1.5
    redescend: float = 2.8
    max_iterations: int = 10
    redescend_iterations: int = 2
    tolerance: float = 0.005
    leverage_cutoff: Optional[float] = None
    bounded_influence: bool = False
    leverage_probability: float = 0.999
    jackknife_groups: int = 0
    #: Up to this many noise classes (0: none); see the module notes.
    noise_classes: int = 0
    noise_alpha: float = 1.8


@dataclass
class RegressionResult:
    coefficients: np.ndarray
    weights: np.ndarray
    residual_variance: float
    inverse_signal_power: np.ndarray
    residuals: np.ndarray
    coherence: float
    iterations: int
    #: ``E|b_j - b|^2`` from the delete-a-group jackknife, when it was asked for.
    jackknife_variance: Optional[np.ndarray] = None
    #: The residuals' stable index (2 is Gaussian), and the noise classes used.
    alpha: float = float("nan")
    noise_classes: int = 0


#: McCulloch's quantile ratio ``(x95 - x05) / (x75 - x25)`` of the symmetric
#: stable law of index alpha (from scipy's ``levy_stable.ppf``; 6.3138 at
#: alpha 1 is the Cauchy law's ``tan(0.45 pi)``).
_STABLE_NU = np.array([(2.0, 2.4387), (1.95, 2.4733), (1.9, 2.5128), (1.8, 2.6099), (1.7, 2.7394),
                       (1.6, 2.9140), (1.5, 3.1498), (1.4, 3.4656), (1.3, 3.8865), (1.2, 4.4509),
                       (1.1, 5.2229), (1.0, 6.3138), (0.9, 7.9285), (0.8, 10.4791), (0.7, 14.8938),
                       (0.6, 23.6122), (0.5, 44.6351)])


def stable_alpha(residuals: np.ndarray) -> float:
    """The stable index of complex residuals (real and imaginary parts pooled), McCulloch's way."""
    parts = np.concatenate([np.real(residuals), np.imag(residuals)]) if np.iscomplexobj(residuals) \
        else np.asarray(residuals, float)
    parts = parts[np.isfinite(parts)]
    if parts.size < 20:
        return float("nan")
    q05, q25, q75, q95 = np.quantile(parts, [0.05, 0.25, 0.75, 0.95])
    if q75 <= q25:
        return float("nan")
    nu = (q95 - q05) / (q75 - q25)
    # alpha falls as nu rises; below the Gaussian's nu the tails are lighter still.
    return float(np.interp(nu, _STABLE_NU[:, 1], _STABLE_NU[:, 0]))


def _k_medians(features: np.ndarray, k: int, rng: np.random.Generator, *, sample: int = 20000,
               iterations: int = 30, replicates: int = 3) -> np.ndarray:
    """Class labels by k-medians (k-means with the city-block distance), fitted on a sample."""
    n = len(features)
    fit = features if n <= sample else features[rng.choice(n, sample, replace=False)]

    def distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.stack([np.abs(points - c).sum(axis=1) for c in centers], axis=1)

    best_cost, best_centers = np.inf, None
    for _ in range(replicates):
        centers = [fit[rng.integers(len(fit))]]
        for _ in range(1, k):
            d = distances(fit, np.array(centers)).min(axis=1)
            centers.append(fit[rng.choice(len(fit), p=d / d.sum())] if d.sum() > 0 else fit[rng.integers(len(fit))])
        centers = np.array(centers)
        for _ in range(iterations):
            label = distances(fit, centers).argmin(axis=1)
            moved = np.array([np.median(fit[label == j], axis=0) if np.any(label == j) else centers[j]
                              for j in range(k)])
            if np.allclose(moved, centers):
                break
            centers = moved
        cost = distances(fit, centers).min(axis=1).sum()
        if cost < best_cost:
            best_cost, best_centers = cost, centers
    return distances(features, best_centers).argmin(axis=1)


def _bisquare_line(q: np.ndarray, p: np.ndarray, iterations: int = 20) -> Tuple[float, float]:
    """``p = a + b q`` fitted with Tukey's bisquare weights (c = 4.685 MADs)."""
    A = np.column_stack([np.ones_like(q), q])
    w = np.ones_like(q)
    coef = np.zeros(2)
    for _ in range(iterations):
        Aw = A * w[:, None]
        try:
            coef = np.linalg.lstsq(Aw.T @ A, Aw.T @ p, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        r = p - A @ coef
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not scale > 0:
            break
        u = r / (4.685 * scale)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    return float(coef[0]), float(coef[1])


def _noise_power(q: np.ndarray, p: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Each point's noise power from ``k`` classes of ``(q, |r|^2)``; 0 for a class too small to keep."""
    features = np.column_stack([q / max(np.median(q), 1e-300), p / max(np.median(p), 1e-300)])
    label = np.zeros(len(q), dtype=int) if k == 1 else _k_medians(features, k, rng)
    power = np.zeros(len(q))
    smallest = max(10, int(0.01 * len(q)))
    for j in range(k):
        members = label == j
        if members.sum() < smallest:
            continue                    # a class this rare holds little signal: left out
        a, b = _bisquare_line(q[members], p[members])
        floor = 0.1 * np.median(p[members])
        power[members] = np.maximum(a + b * q[members], max(floor, 1e-300))
    return power


def _leverage_weights(X: np.ndarray, weights: np.ndarray, probability: float) -> np.ndarray:
    """``min(1, (cutoff / h_i)^2)`` for the hat values of ``X`` at ``weights``."""
    from scipy.stats import beta

    p = X.shape[1]
    A = np.linalg.inv((X.conj().T * weights) @ X)
    hat = weights * np.einsum("ij,jk,ik->i", X, A, X.conj()).real
    effective = float(np.sum(weights > 0))
    cutoff = float(beta.ppf(probability, p, max(effective - p, 1.0)))
    return np.square(np.minimum(1.0, cutoff / np.maximum(hat, 1e-300)))


def _jackknife_variance(Rh: np.ndarray, Xy: np.ndarray, w: np.ndarray, groups: int,
                        blocks: Optional[np.ndarray] = None) -> np.ndarray:
    """Delete-a-group jackknife variance of the IV estimate at fixed weights ``w``.

    With ``blocks`` (a label per point) every group holds whole blocks -
    neighbours in label order - and there are no more groups than blocks;
    without, the points are cut into runs in their order.
    """
    p = Rh.shape[0]
    n = Xy.shape[0]
    groups = int(min(groups, n // (4 * p)))
    if blocks is not None:
        labels, member = np.unique(np.asarray(blocks), return_inverse=True)
        groups = min(groups, labels.size)
    if groups < 3:
        return np.full(p, np.nan)
    if blocks is None:
        edges = np.linspace(0, n, groups + 1).astype(int)
        parts = [(Rh[:, a:b] * w[a:b]) @ Xy[a:b] for a, b in zip(edges[:-1], edges[1:])]
    else:
        group = (member.ravel() * groups) // labels.size
        parts = [(Rh[:, group == g] * w[group == g]) @ Xy[group == g] for g in range(groups)]
    total = np.sum(parts, axis=0)
    estimates = []
    for part in parts:
        G = total - part
        try:
            estimates.append(np.linalg.solve(G[:, :p], G[:, p]))
        except np.linalg.LinAlgError:
            continue
    if len(estimates) < 3:
        return np.full(p, np.nan)
    estimates = np.array(estimates)
    k = len(estimates)
    return (k - 1) / k * np.sum(np.abs(estimates - estimates.mean(axis=0)) ** 2, axis=0)


def _redescend(a: np.ndarray, u0: float, out: Optional[np.ndarray] = None) -> np.ndarray:
    """``exp(-exp(u0 (a - u0)))``, written into ``out`` when given (it may be ``a``)."""
    out = np.subtract(a, u0, out=out)
    out *= u0
    np.minimum(out, 50.0, out=out)
    np.exp(out, out=out)
    np.negative(out, out=out)
    return np.exp(out, out=out)


@lru_cache(maxsize=8)
def _redescend_correction(u0: float) -> float:
    """``E[w a^2] / E[w]`` for ``a = |r| / sigma`` of complex Gaussian residuals."""
    a = np.linspace(0.0, 12.0, 24001)
    pdf = 2 * a * np.exp(-a**2)
    w = _redescend(a, u0)
    integral = lambda f: float(np.sum((f[1:] + f[:-1]) * np.diff(a)) / 2)
    return integral(w * a**2 * pdf) / integral(w * pdf)


def robust_regression(y: np.ndarray, X: np.ndarray, R: Optional[np.ndarray] = None,
                      config: Optional[RegressionConfig] = None,
                      blocks: Optional[np.ndarray] = None) -> RegressionResult:
    """Estimate ``b`` in ``y = X b`` robustly; ``R`` is the remote reference (default X).

    ``blocks`` labels each point with its window (its start time, say): the
    jackknife then deletes whole windows, in groups of neighbours in label
    order, instead of runs of points that may split a window's harmonics.
    """
    cfg = config or RegressionConfig()
    y = np.asarray(y, dtype=complex)
    X = np.asarray(X, dtype=complex)
    R = X if R is None else np.asarray(R, dtype=complex)
    plain = _m_estimate(y, X, R, cfg, blocks)
    plain.alpha = stable_alpha(plain.residuals)
    if cfg.noise_classes <= 0 or not plain.alpha < cfg.noise_alpha:
        return plain
    # Noise classes (module notes): rows divided by their noise level, the
    # M-estimate made again, for k = 1, 2, ... until the residuals are close
    # enough to Gaussian.
    q = np.sum(np.abs(X) ** 2, axis=1)
    p = np.abs(plain.residuals) ** 2
    rng = np.random.default_rng(0)
    best = plain
    for k in range(1, int(cfg.noise_classes) + 1):
        power = _noise_power(q, p, k, rng)
        keep = power > 0
        if keep.sum() <= 4 * X.shape[1]:
            continue
        # Weights 1/power, scaled to a mean of one: the rows keep their units.
        weight = 1.0 / power[keep]
        scale = np.sqrt(weight / weight.mean())[:, None]
        fit = _m_estimate(y[keep] * scale[:, 0], X[keep] * scale, R[keep] * scale, cfg,
                          None if blocks is None else np.asarray(blocks)[keep])
        fit.alpha = stable_alpha(fit.residuals)
        if not fit.alpha > best.alpha:
            continue
        # Back to every point, in the data's units: residuals of the new
        # coefficients, and each point's weight (robust times noise class).
        weights = np.zeros(len(y))
        weights[keep] = fit.weights * scale[:, 0] ** 2
        fit.residuals = y - X @ fit.coefficients
        fit.weights = weights
        fit.noise_classes = k
        best = fit
        if best.alpha >= cfg.noise_alpha:
            break
    return best


def _m_estimate(y: np.ndarray, X: np.ndarray, R: np.ndarray, cfg: RegressionConfig,
                blocks: Optional[np.ndarray] = None) -> RegressionResult:
    """The M-estimate (with bounded influence and the jackknife when asked) of ``y = X b``."""
    n, p = X.shape
    if n <= p:
        raise ValueError(f"{n} points cannot determine {p} coefficients")
    # A long burst recording gives millions of points per band, so every array
    # the iterations need is made once and rewritten in place: each solve is
    # then one pass over the points and each residual another, with no allocation.
    Rh = np.ascontiguousarray(R.conj().T)
    Xy = np.column_stack([X, y])
    weighted = np.empty_like(Rh)
    r = np.empty(n, dtype=complex)
    r2, scratch, w = np.empty(n), np.empty(n), np.ones(n)

    def solve(weights: np.ndarray) -> np.ndarray:
        G = np.multiply(Rh, weights, out=weighted) @ Xy
        return np.linalg.solve(G[:, :p], G[:, p])

    def residuals(b: np.ndarray) -> None:
        """``r = y - X b`` and ``r2 = |r|^2``."""
        np.matmul(Xy, np.append(-b, 1.0), out=r)
        np.add(np.square(r.real, out=r2), np.square(r.imag, out=scratch), out=r2)

    leverage = np.ones(n) if cfg.bounded_influence else None

    def bound_leverage() -> None:
        """Weight ``w`` (the residual weights) by leverage too: the hat values at the weights so far
        scale the leverage weights, which only accumulate - recomputed afresh, a point already
        downweighted has a small hat value, gets its full weight back, and the weights oscillate."""
        nonlocal leverage
        if leverage is not None:
            leverage = leverage * _leverage_weights(X, w * leverage, cfg.leverage_probability)
            np.multiply(w, leverage, out=w)

    b = solve(w)
    residuals(b)
    sigma2 = float(np.mean(r2))
    if not sigma2 > 1e-24 * max(float(np.mean(np.abs(y) ** 2)), 1e-300):
        # An exact fit: nothing to weight, and no residual to scale by.
        A = np.linalg.inv((Rh @ X))
        S = A @ (Rh @ R) @ A.conj().T
        zero = np.zeros(p) if cfg.jackknife_groups else None
        return RegressionResult(b, w, 0.0, S, r.copy(), 1.0, 0, zero)
    c = cfg.huber
    iterations = 0
    for iterations in range(1, cfg.max_iterations + 1):
        # min(1, c / a) for a = |r| / sigma: full weight within c sigmas.
        np.sqrt(np.divide(r2, sigma2, out=w), out=w)
        with np.errstate(divide="ignore"):
            np.divide(c, w, out=w)
        np.minimum(w, 1.0, out=w)
        bound_leverage()
        b_new = solve(w)
        residuals(b_new)
        sigma2 = float(np.mean(np.minimum(r2, c * c * sigma2, out=scratch)) / (1 - np.exp(-c * c)))
        change = np.max(np.abs(b_new - b) / np.maximum(np.abs(b), 1e-300))
        b = b_new
        if change < cfg.tolerance:
            break
    u0 = cfg.redescend
    # Leverage weights settle over several rounds, so with them the redescending
    # stage runs until the estimate settles, not for the usual two iterations.
    rounds = cfg.redescend_iterations if leverage is None else max(cfg.redescend_iterations,
                                                                    cfg.max_iterations)
    for k in range(rounds):
        _redescend(np.sqrt(np.divide(r2, sigma2, out=w), out=w), u0, out=w)
        bound_leverage()
        if w.sum() <= p:
            break
        b_new = solve(w)
        residuals(b_new)
        sigma2 = float(np.sum(np.multiply(w, r2, out=scratch)) / np.sum(w) / _redescend_correction(u0))
        change = np.max(np.abs(b_new - b) / np.maximum(np.abs(b), 1e-300))
        b = b_new
        if leverage is not None and k + 1 >= cfg.redescend_iterations and change < cfg.tolerance:
            break
    if cfg.leverage_cutoff:
        XW = X.conj().T * w
        hat = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XW @ X), X.conj()).real
        limit = cfg.leverage_cutoff * p / n
        w = w * np.minimum(1.0, limit / np.maximum(hat, 1e-300))
        b = solve(w)
        residuals(b)
    A = np.linalg.inv((np.multiply(Rh, w, out=weighted) @ Xy)[:, :p])
    S = A @ (np.multiply(Rh, np.square(w, out=scratch), out=weighted) @ R) @ A.conj().T
    effective = float(np.sum(w))
    sigma2 *= effective / max(effective - p, 1.0)
    power_points = np.square(y.real, out=scratch)
    power_points += np.square(y.imag)
    power = float(np.sum(power_points * w))
    coherence = float(1.0 - np.sum(w * r2) / power) if power > 0 else float("nan")
    jackknife = _jackknife_variance(Rh, Xy, w, cfg.jackknife_groups, blocks) if cfg.jackknife_groups else None
    return RegressionResult(b, w, sigma2, S, r, coherence, iterations, jackknife)
