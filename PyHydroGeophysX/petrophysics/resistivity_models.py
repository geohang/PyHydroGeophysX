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
from typing import Any

import numpy as np
from scipy.optimize import fsolve


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
    """
    # Clip saturation to physically meaningful range (avoid division by zero)
    saturation = np.clip(saturation, 0.001, 1.0)

    # Calculate formation factor F = φ^(-m)
    formation_factor = porosity**(-m)

    # Calculate conductivity using Waxman-Smits model
    # σ = (S_w^n/F)(σ_w + σ_s/S_w)
    sigma = (saturation**n / formation_factor) * (sigma_w + sigma_s/saturation)

    # Add minimum conductivity threshold to prevent division by zero
    sigma_min = 1e-6
    sigma = np.maximum(sigma, sigma_min)

    # Convert conductivity to resistivity
    resistivity = 1.0 / sigma

    # Clip resistivity to physically reasonable range
    resistivity = np.clip(resistivity, 0.1, 1e6)

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
    # This used to go through resistivity_to_saturation with m=0 and
    # rho_fluid=rhos, relying on porosity**-0 == 1; once that function clamped m
    # to at least 1 the saturated resistivity became rhos / porosity and every
    # water content came back too high (0.10 returned as 0.18).
    saturation = _waxman_smits_saturation(resistivity, rhos, n, sigma_sur)
    if saturation.size == 1:
        # The solver keeps the input's shape now, so a single value may sit in
        # a (1, 1) array; it still comes back as a float, as it always did.
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
    A negative ``sigma_sur`` raises ValueError; it used to be answered with
    the Archie saturation as if it were zero.
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
        # A + B = B, the conductivity at full saturation. These cells used to
        # keep the Archie guess, which is S = 1 whatever the resistivity.
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
    
    Args:
        resistivity (array): Resistivity values (ohm-m)
        saturation (array): Saturation values (fraction, 0-1)
        m (float): Cementation exponent (typically 1.3-2.5)
        rho_fluid (float): Fluid resistivity (ohm-m)
        n (float): Saturation exponent (typically 1.8-2.2)
        sigma_sur (float): Surface conductivity (S/m). Default is 0 (no surface effects)
        a (float): Tortuosity factor. Default is 1.0
        
    Returns:
        array: Porosity values (fraction, 0-1)
    """
    # Convert inputs to arrays
    resistivity_array = np.atleast_1d(resistivity).astype(float)
    saturation_array = np.atleast_1d(saturation)
    sigma_sur_array = np.atleast_1d(sigma_sur)
    n_array = np.atleast_1d(n)
    m_array = np.atleast_1d(m)
    
    # Ensure all arrays have compatible shapes
    max_length = max(len(resistivity_array), len(saturation_array))
    
    if len(saturation_array) == 1 and max_length > 1:
        saturation_array = np.full(max_length, saturation_array[0])
    if len(sigma_sur_array) == 1 and max_length > 1:
        sigma_sur_array = np.full(max_length, sigma_sur_array[0])
    if len(n_array) == 1 and max_length > 1:
        n_array = np.full(max_length, n_array[0])
    if len(m_array) == 1 and max_length > 1:
        m_array = np.full(max_length, m_array[0])
    if len(resistivity_array) == 1 and max_length > 1:
        resistivity_array = np.full(max_length, resistivity_array[0])
    
    # Validate saturation values
    saturation_array = np.clip(saturation_array, 0.001, 1.0)  # Avoid extreme values
    
    # Initialize porosity array
    porosity = np.zeros_like(resistivity_array)

    if np.all(sigma_sur_array == 0):
        # Same Archie formula as the scalar loop; no root solving is needed.
        porosity = np.clip(((a * rho_fluid) / (resistivity_array * saturation_array**n_array))
                           ** (1.0 / m_array), 0.001, 0.99)
        return float(porosity[0]) if np.isscalar(resistivity) and np.isscalar(saturation) else porosity
    
    # Solve for each resistivity-saturation pair
    for i in range(len(resistivity_array)):
        rho_val = resistivity_array[i]
        S_val = saturation_array[i]
        sigma_sur_val = sigma_sur_array[i]
        n_val = n_array[i]
        m_val = m_array[i]
        
        if sigma_sur_val == 0:
            # Without surface conductivity, use simplified Archie's law
            # 1/rho = (1/rhos) * S^n = (porosity^m / (a * rho_fluid)) * S^n
            # porosity^m = (a * rho_fluid) / (rho * S^n)
            # porosity = [(a * rho_fluid) / (rho * S^n)]^(1/m)
            
            porosity_val = ((a * rho_fluid) / (rho_val * S_val**n_val))**(1.0/m_val)
            
        else:
            # With surface conductivity, solve numerically
            # 1/rho = (porosity^m / (a * rho_fluid)) * S^n + sigma_sur * S^(n-1)
            # Rearranging: porosity^m = (1/rho - sigma_sur * S^(n-1)) * (a * rho_fluid) / S^n
            
            conductivity_term = 1.0/rho_val - sigma_sur_val * S_val**(n_val-1)
            
            if conductivity_term > 0:
                # Direct calculation if the term is positive
                porosity_val = (conductivity_term * a * rho_fluid / S_val**n_val)**(1.0/m_val)
            else:
                # If negative (which shouldn't happen physically), use numerical solver
                def func(phi):
                    if phi <= 0:
                        return 1e10  # Large penalty for non-physical values
                    rhos = a * rho_fluid * phi**(-m_val)
                    sigma_sat = 1.0 / rhos
                    return sigma_sat * S_val**n_val + sigma_sur_val * S_val**(n_val-1) - 1.0/rho_val
                
                # Initial guess using simplified formula
                initial_guess = max(0.05, ((a * rho_fluid) / (rho_val * S_val**n_val))**(1.0/m_val))
                
                try:
                    solution = fsolve(func, initial_guess)
                    porosity_val = solution[0]
                except Exception:
                    # If numerical solution fails, use simplified formula
                    porosity_val = ((a * rho_fluid) / (rho_val * S_val**n_val))**(1.0/m_val)
        
        porosity[i] = porosity_val
    
    # Ensure porosity is physically meaningful
    porosity = np.clip(porosity, 0.001, 0.99)
    
    # Return scalar if input was scalar
    if np.isscalar(resistivity) and np.isscalar(saturation):
        return float(porosity[0])
    
    return porosity



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

