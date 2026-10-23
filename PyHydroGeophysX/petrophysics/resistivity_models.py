"""
Simplified Waxman-Smits model for converting between water content and resistivity.

This implementation follows the Waxman-Smits model that expresses conductivity as:
    
    σ = σsat * S^n + σs * S^(n-1)
    
where:
- σ is the electrical conductivity of the formation
- σsat is the conductivity at full saturation without surface effects (1/rhos)
- σs is the surface conductivity
- S is the water saturation (S = θ/φ where θ is water content and φ is porosity)
- n is the saturation exponent

The resistivity is the reciprocal of conductivity: ρ = 1/σ
"""
import warnings
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# WS Model
# ---------------------------------------------------------------------------
def WS_Model(
    saturation: Any,
    porosity: Any,
    sigma_w: Any,
    m: Any,
    n: Any,
    sigma_s: Any = 0,
) -> Any:
    """
    Convert saturation to resistivity using the implemented Waxman-Smits form.

    Uses σ = (S_w^n/F) * (σ_w + σ_s/S_w), where F = φ^(-m).
    Here σ_s is inside the formation-factor scaling; it is not interchangeable
    with the unscaled sigma_sur parameter used by the other converters.

    Args:
        saturation (array): Saturation fraction, clipped to [0.001, 1].
        porosity (array): Porosity fraction (φ), positive and broadcast-compatible.
        sigma_w (float): Pore water conductivity (S/m).
        m (float): Cementation exponent
        n (float): Saturation exponent
        sigma_s (float): Surface-conductivity coefficient (S/m). Default is 0.

    Returns:
        array: Resistivity in ohm-m, clipped to [0.1, 1e6].

    Warns:
        RuntimeWarning: Once per call if saturation, the conductivity floor
            (1e-6 S/m), or resistivity is clipped. Bounds and numerical outputs
            are unchanged. NaN values propagate; they are not repaired.
    """
    clipped = []
    raw_saturation = np.asarray(saturation)
    if np.any((raw_saturation < 0.001) | (raw_saturation > 1)):
        clipped.append("saturation to [0.001, 1]")
    # Clip saturation to physically meaningful range (avoid division by zero)
    saturation = np.clip(saturation, 0.001, 1.0)

    # Calculate formation factor F = φ^(-m)
    formation_factor = porosity**(-m)

    # Calculate conductivity using Waxman-Smits model
    # σ = (S_w^n/F)(σ_w + σ_s/S_w)
    sigma = (saturation**n / formation_factor) * (sigma_w + sigma_s/saturation)

    # Add minimum conductivity threshold to prevent division by zero
    sigma_min = 1e-6
    if np.any(sigma < sigma_min):
        clipped.append("conductivity to >= 1e-6 S/m (resistivity <= 1e6 ohm-m)")
    sigma = np.maximum(sigma, sigma_min)

    # Convert conductivity to resistivity
    resistivity = 1.0 / sigma

    # Clip resistivity to physically reasonable range
    if np.any((resistivity < 0.1) | (resistivity > 1e6)):
        clipped.append("resistivity to [0.1, 1e6] ohm-m")
    resistivity = np.clip(resistivity, 0.1, 1e6)
    if clipped:
        warnings.warn("WS_Model clipped " + "; ".join(clipped) + ".",
                      RuntimeWarning, stacklevel=2)

    return resistivity



# ---------------------------------------------------------------------------
# water content to resistivity
# ---------------------------------------------------------------------------
def water_content_to_resistivity(
    water_content: Any,
    rhos: Any,
    n: Any,
    porosity: Any,
    sigma_sur: Any = 0,
) -> Any:
    """
    Convert water content to resistivity using Waxman-Smits model.

    Args:
        water_content (array): Volumetric water content (θ)
        rhos (float): Saturated resistivity without surface effects
        n (float): Saturation exponent
        porosity (array): Porosity values (φ)
        sigma_sur (float): Surface conductivity. Default is 0 (no surface effects).

    Returns:
        array: Resistivity values
    """
    # Calculate saturation with minimum threshold to avoid division issues
    saturation = water_content / porosity
    saturation = np.clip(saturation, 0.001, 1.0)

    # Calculate conductivity using Waxman-Smits model
    sigma_sat = 1.0 / rhos
    sigma = sigma_sat * saturation**n + sigma_sur * saturation**(n-1)

    # Add minimum conductivity threshold to prevent division by zero
    sigma_min = 1e-6
    sigma = np.maximum(sigma, sigma_min)

    # Convert conductivity to resistivity
    resistivity = 1.0 / sigma

    # Clip resistivity to physically reasonable range
    resistivity = np.clip(resistivity, 0.1, 1e6)

    return resistivity


# ---------------------------------------------------------------------------
# resistivity to water content
# ---------------------------------------------------------------------------
def resistivity_to_water_content(
    resistivity: Any,
    rhos: Any,
    n: Any,
    porosity: Any,
    sigma_sur: Any = 0,
) -> Any:
    """
    Convert resistivity to water content using Waxman-Smits model.
    
    Args:
        resistivity (array): Resistivity values
        rhos (float): Saturated resistivity without surface effects
        n (float): Saturation exponent
        porosity (array): Porosity values
        sigma_sur (float): Surface conductivity. Default is 0 (no surface effects).
    
    Returns:
        array: Volumetric water content values
    """
    # The saturated resistivity is given, so solve Waxman-Smits with it directly.
    # Going through resistivity_to_saturation with m=0 and rho_fluid=rhos would
    # rely on porosity**-0 == 1, which fails because that function clamps m to
    # at least 1: the saturated resistivity would become rhos / porosity.
    saturation = _waxman_smits_saturation(resistivity, rhos, n, sigma_sur)
    if saturation.size == 1:
        # The solver keeps the input's shape, so a single value may sit in a
        # (1, 1) array; it still comes back as a float.
        saturation = float(saturation.reshape(-1)[0])

    # Convert saturation to water content
    water_content = saturation * porosity

    return water_content



# ---------------------------------------------------------------------------
# resistivity to saturation
# ---------------------------------------------------------------------------
def resistivity_to_saturation(
    resistivity: Any,
    porosity: Any,
    m: Any,
    rho_fluid: Any,
    n: Any,
    sigma_sur: Any = 0,
    a: Any = 1.0,
) -> Any:
    """
    Convert resistivity to saturation using Waxman-Smits model.
    
    The function calculates saturated resistivity using Archie's law:
    rhos = a * rho_fluid * porosity^(-m)
    
    Then solves the Waxman-Smits equation:
    1/rho = sigma_sat * S^n + sigma_sur * S^(n-1)
    where sigma_sat = 1/rhos
    
    Args:
        resistivity (array): Resistivity values (ohm-m)
        porosity (array): Porosity values (fraction, 0-1)
        m (float): Cementation exponent (typically 1.3-2.5)
        rho_fluid (float): Fluid resistivity (ohm-m)
        n (float): Saturation exponent (typically 1.8-2.2)
        sigma_sur (float): Surface conductivity (S/m). Default is 0 (no surface effects)
        a (float): Tortuosity factor. Default is 1.0
        
    Returns:
        array: Saturation values (fraction, 0-1), shaped like the inputs
        broadcast together (NumPy rules, so a (cells, times) resistivity takes
        a scalar, (cells, 1) or (cells, times) porosity); a float for one value.

    Raises:
        ValueError: sigma_sur is negative.
    """
    # Convert inputs to arrays and broadcast them with NumPy's rules. Matching
    # lengths only by the first axis rejected any 2-D input.
    resistivity, porosity, m_arr = np.broadcast_arrays(
        *(np.atleast_1d(value).astype(float) for value in (resistivity, porosity, m)))

    # Clip porosity to avoid extremes, and the cementation exponent with it: a
    # wide prior sampled per cell otherwise reaches m < 1.
    porosity = np.clip(porosity, 1e-3, 0.99)
    m_arr = np.clip(m_arr, 1.0, 4.0)

    # Saturated resistivity (Archie)
    rhos = a * rho_fluid * porosity**(-m_arr)
    sat = _waxman_smits_saturation(resistivity, rhos, n, sigma_sur)

    # Return scalar if inputs were scalar
    if sat.size == 1:
        return float(sat.reshape(-1)[0])
    return sat


def _waxman_smits_saturation(resistivity: Any, rhos: Any, n: Any,
                             sigma_sur: Any = 0, *, clip_exponent=True) -> np.ndarray:
    """Solve ``1/rho = S**n / rhos + sigma_sur * S**(n-1)`` for S, per value.

    ``resistivity_to_saturation`` derives ``rhos`` from Archie's law first;
    ``resistivity_to_water_content`` is given it. Always returns an array.
    A negative ``sigma_sur`` raises ValueError.
    """
    resistivity, rhos, sigma_sur, n_arr = np.broadcast_arrays(*[
        np.atleast_1d(value).astype(float)
        for value in (resistivity, rhos, sigma_sur, n)
    ])
    shape = resistivity.shape
    resistivity, rhos, sigma_sur, n_arr = (
        value.ravel() for value in (resistivity, rhos, sigma_sur, n_arr))
    if np.any(sigma_sur < 0):
        raise ValueError("sigma_sur (surface conductivity, S/m) must not be negative.")

    # Preserve the newer APIs' exponent bounds for broad sampled priors;
    # the legacy rhos API continues to use the exponent supplied by its caller.
    if clip_exponent:
        n_arr = np.clip(n_arr, 1.0, 4.0)

    sigma_sat = 1.0 / rhos
    sigma_obs = 1.0 / resistivity

    # Initial guess via Archie's law
    # No artificial dry-end floor: a small positive saturation can be the
    # exact Archie solution, rather than a failed numerical solve.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        S0 = np.exp(np.minimum((np.log(rhos) - np.log(resistivity)) / n_arr, 0.0))

    # Compute saturation for each point. The zero-surface-conductivity branch
    # is already solved analytically.
    sat = S0.copy()
    cells = np.flatnonzero(sigma_sur != 0)
    if cells.size:
        sat[cells] = _surface_conduction_roots(
            sigma_sat[cells], sigma_sur[cells], n_arr[cells], sigma_obs[cells], S0[cells])
    return np.clip(sat, 0.0, 1.0).reshape(shape)


def _surface_conduction_roots(A, B, n, C, fallback):
    """Bounded roots, with a relative conductivity residual as stopping rule.

    Solve in t=-log(S), where small saturations are resolved without negative
    Newton iterates or underflow in the powers. For n>1 the residual decreases
    monotonically; the upper bracket makes each conduction term at most C/2.
    n=1 is linear. Outside the attainable range use the physical endpoint,
    retaining the public APIs' saturation clipping convention. Without bulk
    conduction (A=0, rhos=inf) and n>1 the root is closed-form.
    """
    result = np.array(fallback, dtype=float, copy=True)
    valid = (np.isfinite(A) & np.isfinite(B) & np.isfinite(n) & np.isfinite(C)
             & (A > 0) & (B > 0) & (C > 0) & (n > 0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # C = B S**(n-1) gives S = (C/B)**(1/(n-1)), and S = 1 once C reaches
        # A + B = B, the conductivity at full saturation. The Archie guess for
        # these cells is S = 1 whatever the resistivity, so it is not kept.
        surface_only = (np.isfinite(B) & np.isfinite(n) & np.isfinite(C)
                        & (A == 0) & (B > 0) & (C > 0) & (n > 1))
        ratio = C[surface_only] / B[surface_only]
        result[surface_only] = np.where(
            ratio >= 1, 1.0, ratio ** (1.0 / (n[surface_only] - 1.0)))
        linear = valid & (n == 1)
        result[linear] = np.clip((C[linear] - B[linear]) / A[linear], 0, 1)
        ids = np.flatnonzero(valid & (n > 1))
        a, b = np.log(A[ids]) - np.log(C[ids]), np.log(B[ids]) - np.log(C[ids])
        exponent = n[ids]
        lo = np.zeros(ids.size)
        hi = np.maximum.reduce([lo, (a + np.log(2)) / exponent,
                                (b + np.log(2)) / (exponent - 1)])
        t = hi / 2
        active = np.logaddexp(a, b) > 0
        t[~active] = 0  # observed conductivity at/above full saturation
        for _ in range(100):
            if not active.any():
                break
            log_a, log_b = a - exponent * t, b - (exponent - 1) * t
            residual = np.logaddexp(log_a, log_b)
            active &= np.abs(residual) > 1e-12
            lo = np.where(active & (residual > 0), t, lo)
            hi = np.where(active & (residual < 0), t, hi)
            slope = exponent * np.exp(log_a - residual) + (exponent - 1) * np.exp(log_b - residual)
            step = t + residual / slope
            inside = np.isfinite(step) & (step > lo) & (step < hi)
            t = np.where(active, np.where(inside, step, (lo + hi) / 2), t)
        result[ids] = np.exp(-t)
        # Do not silently return an unconverged guess if numerical conditions
        # defeat the safeguarded iteration. Scalar bracketing is exceptional.
        retry = list(ids[active]) + list(np.flatnonzero(valid & (n < 1)))
        for index in retry:
            result[index] = _bracketed_surface_root(A[index], B[index], n[index], C[index])
    return result


def _bracketed_surface_root(A, B, n, C):
    """Rare scalar fallback, also preserving the legacy API's 0<n<1 inputs.

    Below n=1 the conductivity is nonmonotonic. Split at its minimum and
    prefer the larger saturation when both roots are physical (the branch
    connected to the Archie starting guess). Do not return a failed iterate.
    """
    from scipy.optimize import brentq
    import warnings

    a, b = np.log(A) - np.log(C), np.log(B) - np.log(C)
    residual = lambda t: np.logaddexp(a - n * t, b - (n - 1) * t)
    if n > 1:
        upper = max(0., (a + np.log(2)) / n, (b + np.log(2)) / (n - 1))
        points = [0., upper]
    else:
        turn = max(0., np.log(A) + np.log(n) - np.log(B) - np.log1p(-n))
        upper = max(turn, -b / (1 - n), 0.) + 1.
        points = [0., turn, upper]
    for lower, upper in zip(points[:-1], points[1:]):
        f_lower, f_upper = residual(lower), residual(upper)
        # At a conductivity minimum the root only touches zero. Allow for
        # rounding in the logs instead of requiring a floating-point sign flip.
        if abs(f_lower) <= 1e-12:
            return float(np.exp(-lower))
        if abs(f_upper) <= 1e-12:
            return float(np.exp(-upper))
        if f_lower * f_upper < 0:
            t = brentq(residual, lower, upper, xtol=1e-13, rtol=1e-14)
            return float(np.exp(-t))
    warnings.warn("Waxman-Smits parameters have no saturation root in [0, 1]; returning NaN.",
                  RuntimeWarning, stacklevel=3)
    return np.nan


# ---------------------------------------------------------------------------
# resistivity to porosity
# ---------------------------------------------------------------------------
def resistivity_to_porosity(
    resistivity: Any,
    saturation: Any,
    m: Any,
    rho_fluid: Any,
    n: Any,
    sigma_sur: Any = 0,
    a: Any = 1.0,
) -> Any:
    """
    Convert resistivity to porosity using Waxman-Smits model, given known saturation.
    
    The function solves the Waxman-Smits equation for porosity:
    1/rho = sigma_sat * S^n + sigma_sur * S^(n-1)
    where sigma_sat = 1/rhos and rhos = a * rho_fluid * porosity^(-m)
    
    Rearranging: porosity = [(1/rho - sigma_sur * S^(n-1)) * (a * rho_fluid) / S^n]^(1/m)

    Porosity only adds conduction, so where ``1/rho <= sigma_sur * S**(n-1)``
    the surface term alone conducts at least as well as the rock is measured
    to and no porosity satisfies the equation. Those values are NaN, with a
    RuntimeWarning naming the parameters and the resistivity range a porosity
    can explain.

    Args:
        resistivity (array): Resistivity values (ohm-m)
        saturation (array): Saturation values (fraction, 0-1)
        m (float): Cementation exponent (typically 1.3-2.5)
        rho_fluid (float): Fluid resistivity (ohm-m)
        n (float): Saturation exponent (typically 1.8-2.2)
        sigma_sur (float): Surface conductivity (S/m). Default is 0 (no surface effects)
        a (float): Tortuosity factor. Default is 1.0

    Returns:
        array: Porosity values (fraction), clipped to [0.001, 0.99], shaped
        like the inputs broadcast together; NaN where no porosity exists. A
        float when resistivity and saturation are scalars.
    """
    porosity, no_root = _waxman_smits_porosity(
        resistivity, saturation, m, rho_fluid, n, sigma_sur, a)
    if no_root.any():
        warnings.warn(_no_porosity_message(
            resistivity, saturation, m, rho_fluid, n, sigma_sur, a, no_root),
            RuntimeWarning, stacklevel=2)

    # Return scalar if input was scalar
    if np.isscalar(resistivity) and np.isscalar(saturation) and porosity.size == 1:
        return float(porosity.reshape(-1)[0])

    return porosity


def _waxman_smits_porosity(resistivity, saturation, m, rho_fluid, n,
                           sigma_sur=0, a=1.0):
    """Closed-form porosity and the mask of values with no porosity root.

    Returns ``(porosity, no_root)`` broadcast together, without warning; the
    Monte Carlo collects ``no_root`` over its draws and reports it once.
    """
    rho, saturation, m, rho_fluid, n, sigma_sur, a = np.broadcast_arrays(*(
        np.atleast_1d(value).astype(float)
        for value in (resistivity, saturation, m, rho_fluid, n, sigma_sur, a)))
    saturation = np.clip(saturation, 0.001, 1.0)  # Avoid extreme values

    # 1/rho = (porosity^m / (a * rho_fluid)) * S^n + sigma_sur * S^(n-1), so
    # porosity^m = (1/rho - sigma_sur * S^(n-1)) * (a * rho_fluid) / S^n.
    # With sigma_sur = 0 this is Archie's law.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        term = 1.0 / rho - sigma_sur * saturation ** (n - 1)
        porosity = (term * a * rho_fluid / saturation ** n) ** (1.0 / m)
    no_root = ~(term > 0) & ~np.isnan(term)
    porosity = np.where(no_root, np.nan, np.clip(porosity, 0.001, 0.99))
    return porosity, no_root


def _span(values) -> str:
    """One value, or the range the values cover."""
    values = np.asarray(values, dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    return f"{low:.4g}" if np.isclose(low, high) else f"{low:.4g}-{high:.4g}"


def _no_porosity_message(resistivity, saturation, m, rho_fluid, n, sigma_sur, a,
                         no_root) -> str:
    rho, saturation, m, rho_fluid, n, sigma_sur, a = (
        np.broadcast_to(np.atleast_1d(value).astype(float), no_root.shape)[no_root]
        for value in (resistivity, saturation, m, rho_fluid, n, sigma_sur, a))
    saturation = np.clip(saturation, 0.001, 1.0)
    with np.errstate(divide="ignore"):
        surface = sigma_sur * saturation ** (n - 1)
        limit = 1.0 / surface
        lowest = 1.0 / (saturation ** n / (a * rho_fluid) + surface)
    return (
        f"Waxman-Smits has no porosity for {int(no_root.sum())} of {no_root.size} "
        f"resistivity value(s) ({_span(rho)} ohm m): with saturation S = {_span(saturation)}, "
        f"n = {_span(n)} and sigma_sur = {_span(sigma_sur)} S/m, surface conduction "
        f"sigma_sur * S**(n-1) alone explains resistivity only up to {_span(limit)} ohm m, "
        f"and porosity adds conduction. With m = {_span(m)}, rho_fluid = {_span(rho_fluid)} "
        f"ohm m and a = {_span(a)}, porosities in (0, 1] explain {_span(lowest)} to "
        f"{_span(limit)} ohm m. These porosities are NaN; the surface conductivity is too "
        f"high for this resistivity at this saturation.")



# ---------------------------------------------------------------------------
# resistivity to saturation2
# ---------------------------------------------------------------------------
def resistivity_to_saturation2(
    resistivity: Any,
    rhos: Any,
    n: Any,
    sigma_sur: Any = 0,
) -> Any:
    """
    Convert resistivity to saturation using Waxman-Smits model.

    Surface conduction is solved with a safeguarded root finder. The
    zero-surface case uses Archie's analytic solution without a dry-end floor.
    Saturations outside the attainable range for n >= 1 are clipped to [0, 1].
    For 0 < n < 1, the larger physical root is preferred; if no physical root
    exists, the result is NaN with a RuntimeWarning.
    
    Args:
        resistivity (array): Resistivity values
        rhos (float): Saturated resistivity without surface effects
        n (float): Saturation exponent
        sigma_sur (float): Surface conductivity. Default is 0 (no surface effects).
    
    Returns:
        array: Saturation values
    """
    # Keep the legacy entry point and its return-type convention. Unlike the
    # newer Archie-parameter API, this function never clipped the caller's n.
    saturation = _waxman_smits_saturation(
        resistivity, rhos, n, sigma_sur, clip_exponent=False)
    
    # Return scalar if input was scalar
    if np.isscalar(resistivity):
        return float(saturation[0])
    
    return saturation

