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
        saturation = float(saturation[0])

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
        array: Saturation values (fraction, 0-1)
    """
    # Convert inputs to arrays and broadcast
    resistivity = np.atleast_1d(resistivity).astype(float)
    porosity    = np.atleast_1d(porosity).astype(float)
    m_arr       = np.atleast_1d(m).astype(float)
    L = max(map(len, (resistivity, porosity, m_arr)))
    def _b(x): return np.full(L, x[0]) if len(x)==1 else x
    resistivity, porosity, m_arr = map(_b, (resistivity, porosity, m_arr))

    # Clip porosity to avoid extremes, and the cementation exponent with it: a
    # wide prior sampled per cell otherwise reaches m < 1.
    porosity = np.clip(porosity, 1e-3, 0.99)
    m_arr = np.clip(m_arr, 1.0, 4.0)

    # Saturated resistivity (Archie)
    rhos = a * rho_fluid * porosity**(-m_arr)
    sat = _waxman_smits_saturation(resistivity, rhos, n, sigma_sur)

    # Return scalar if inputs were scalar
    if sat.size == 1:
        return float(sat[0])
    return sat


def _waxman_smits_saturation(resistivity: Any, rhos: Any, n: Any,
                             sigma_sur: Any = 0) -> np.ndarray:
    """Solve ``1/rho = S**n / rhos + sigma_sur * S**(n-1)`` for S, per value.

    ``resistivity_to_saturation`` derives ``rhos`` from Archie's law first;
    ``resistivity_to_water_content`` is given it. Always returns an array.
    """
    resistivity = np.atleast_1d(resistivity).astype(float)
    rhos        = np.atleast_1d(rhos).astype(float)
    sigma_sur   = np.atleast_1d(sigma_sur).astype(float)
    n_arr       = np.atleast_1d(n).astype(float)
    L = max(map(len, (resistivity, rhos, sigma_sur, n_arr)))
    def _b(x): return np.full(L, x[0]) if len(x)==1 else x
    resistivity, rhos, sigma_sur, n_arr = map(_b, (resistivity, rhos, sigma_sur, n_arr))

    # Below n = 1 the residual below evaluates 0**(n-1) at the bracket's lower
    # end (a divide-by-zero) and the initial guess raises a ratio to the power
    # 1/n (an overflow as n -> 0). Callers that sample n from a wide prior
    # otherwise hit both per cell.
    n_arr = np.clip(n_arr, 1.0, 4.0)

    sigma_sat = 1.0 / rhos
    sigma_obs = 1.0 / resistivity

    # Initial guess via Archie's law
    S0 = np.clip((rhos / resistivity)**(1.0 / n_arr), 1e-3, 1.0)

    # Compute saturation for each point. The zero-surface-conductivity branch
    # is already solved analytically.
    sat = S0.copy()
    cells = np.flatnonzero(sigma_sur != 0)
    if cells.size:
        sat[cells] = _surface_conduction_roots(
            sigma_sat[cells], sigma_sur[cells], n_arr[cells], sigma_obs[cells], S0[cells])
    return np.clip(sat, 0.0, 1.0)


def _surface_conduction_roots(A, B, n, C, fallback):
    """Roots of ``A S**n + B S**(n-1) = C`` on ``[0, 1]``, for all cells at once.

    The residual rises monotonically in S for ``n >= 1``, so each root is
    bracketed by [0, 1] whenever one exists there. Newton steps are taken
    inside the shrinking bracket, and a step that would leave it is replaced by
    bisection, until every root is known to 1e-14. This replaced a scalar
    brentq per cell (xtol 1e-6), about 13 us of interpreter work a cell - most
    of a Monte Carlo water-content estimate - with a handful of array passes;
    the roots are now tighter than before, not looser. As before, a cell with
    no root in [0, 1] (the residual has one sign there, or is not finite) keeps
    the Archie guess ``fallback``.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f_lo = A * 0.0 ** n + B * 0.0 ** (n - 1) - C
        f_hi = A + B - C
        bracketed = f_lo * f_hi <= 0
        root = np.where(f_lo == 0, 0.0, np.where(f_hi == 0, 1.0, fallback))
        active = bracketed & (f_lo != 0) & (f_hi != 0)
        lo, hi = np.zeros_like(A), np.ones_like(A)
        x = np.clip(fallback, 1e-12, 1.0)
        for _ in range(200):
            if not active.any():
                break
            fx = A * x ** n + B * x ** (n - 1) - C
            below = fx < 0
            lo = np.where(active & below, x, lo)
            hi = np.where(active & ~below, x, hi)
            step = x - fx / (A * n * x ** (n - 1) + B * (n - 1) * x ** (n - 2))
            inside = np.isfinite(step) & (step > lo) & (step < hi)
            nxt = np.where(inside, step, 0.5 * (lo + hi))
            settled = (np.abs(nxt - x) <= 1e-14) | (hi - lo <= 1e-14)
            x = np.where(active, nxt, x)
            active &= ~settled
    return np.where(bracketed & (f_lo != 0) & (f_hi != 0), x, root)


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
    
    Args:
        resistivity (array): Resistivity values
        rhos (float): Saturated resistivity without surface effects
        n (float): Saturation exponent
        sigma_sur (float): Surface conductivity. Default is 0 (no surface effects).
    
    Returns:
        array: Saturation values
    """
    # Convert inputs to arrays
    resistivity_array = np.atleast_1d(resistivity).astype(float)
    sigma_sur_array = np.atleast_1d(sigma_sur)
    n_array = np.atleast_1d(n)
    
    # Ensure all arrays have compatible shapes
    if len(sigma_sur_array) == 1 and len(resistivity_array) > 1:
        sigma_sur_array = np.full_like(resistivity_array, sigma_sur_array[0])
    if len(n_array) == 1 and len(resistivity_array) > 1:
        n_array = np.full_like(resistivity_array, n_array[0])
    
    # Calculate sigma_sat
    sigma_sat = 1.0 / rhos
    
    # First calculate saturation without surface conductivity (Archie's law)
    # This provides an initial guess for numerical solution
    S_initial = (rhos / resistivity_array) ** (1.0/n_array)
    S_initial = np.clip(S_initial, 0.01, 1.0)
    
    # Initialize saturation array
    saturation = S_initial.copy()
    
    # Solve for each resistivity value
    for i in np.flatnonzero(sigma_sur_array != 0):
        if sigma_sur_array[i] == 0:
            # If no surface conductivity, use Archie's law
            saturation[i] = S_initial[i]
        else:
            # With surface conductivity, solve numerically
            n_val = n_array[i]
            
            def func(S):
                return sigma_sat * S**n_val + sigma_sur_array[i] * S**(n_val-1) - 1.0/resistivity_array[i]
            
            solution = fsolve(func, S_initial[i])
            saturation[i] = solution[0]
    
    # Ensure saturation is physically meaningful
    saturation = np.clip(saturation, 0.0, 1.0)
    
    # Return scalar if input was scalar
    if np.isscalar(resistivity):
        return float(saturation[0])
    
    return saturation

