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
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root_scalar



def WS_Model(saturation, porosity, sigma_w, m, n, sigma_s=0):
    """
    Convert water content to resistivity using Waxman-Smits model.

    Based on equation: σ = (S_w^n/F)σ_w + σ_s
    where F = φ^(-m) (formation factor from Archie's law)

    Args:
        saturation (array): Saturation (S_w)
        porosity (array): Porosity values (φ)
        sigma_w (float): Pore water conductivity (σ_w)
        m (float): Cementation exponent
        n (float): Saturation exponent
        sigma_s (float): Surface conductivity (σ_s). Default is 0 (no surface effects).

    Returns:
        array: Resistivity values
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



def water_content_to_resistivity(water_content, rhos, n, porosity, sigma_sur=0):
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


def resistivity_to_water_content(resistivity, rhos, n, porosity, sigma_sur=0):
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
    # Calculate saturation
    saturation = resistivity_to_saturation(resistivity, rhos, n, sigma_sur)
    
    # Convert saturation to water content
    water_content = saturation * porosity
    
    return water_content



def resistivity_to_saturation(resistivity, porosity, m, rho_fluid,
                              n, sigma_sur=0, a=1.0):
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
    sigma_sur   = np.atleast_1d(sigma_sur).astype(float)
    n_arr       = np.atleast_1d(n).astype(float)
    m_arr       = np.atleast_1d(m).astype(float)
    L = max(map(len, (resistivity, porosity, sigma_sur, n_arr, m_arr)))
    def _b(x): return np.full(L, x[0]) if len(x)==1 else x
    resistivity, porosity, sigma_sur, n_arr, m_arr = map(_b,
        (resistivity, porosity, sigma_sur, n_arr, m_arr))

    # Clip porosity to avoid extremes
    porosity = np.clip(porosity, 1e-3, 0.99)

    # Saturated resistivity (Archie)
    rhos = a * rho_fluid * porosity**(-m_arr)
    sigma_sat = 1.0 / rhos
    sigma_obs = 1.0 / resistivity

    # Initial guess via Archie's law
    S0 = np.clip((rhos / resistivity)**(1.0 / n_arr), 1e-3, 1.0)

    # Solver settings
    tol, maxiter = 1e-6, 50

    def _solve(i):
        if sigma_sur[i] == 0:
            return S0[i]
        A, B, ni, Ci = sigma_sat[i], sigma_sur[i], n_arr[i], sigma_obs[i]
        # Residual and derivative
        def f(S):      return A * S**ni + B * S**(ni-1) - Ci
        def fprime(S): return A * ni * S**(ni-1) + B * (ni-1) * S**(ni-2)
        try:
            sol = root_scalar(f,
                              fprime=fprime,
                              bracket=[0.0, 1.0],
                              method='brentq',
                              xtol=tol,
                              maxiter=maxiter)
            return sol.root if sol.converged else S0[i]
        except:
            return S0[i]

    # Compute saturation for each point
    sat = np.array([_solve(i) for i in range(L)], dtype=float)
    sat = np.clip(sat, 0.0, 1.0)

    # Return scalar if inputs were scalar
    if sat.size == 1:
        return float(sat[0])
    return sat


def resistivity_to_porosity(resistivity, saturation, m, rho_fluid, n, sigma_sur=0, a=1.0):
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
    resistivity_array = np.atleast_1d(resistivity)
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
                except:
                    # If numerical solution fails, use simplified formula
                    porosity_val = ((a * rho_fluid) / (rho_val * S_val**n_val))**(1.0/m_val)
        
        porosity[i] = porosity_val
    
    # Ensure porosity is physically meaningful
    porosity = np.clip(porosity, 0.001, 0.99)
    
    # Return scalar if input was scalar
    if np.isscalar(resistivity) and np.isscalar(saturation):
        return float(porosity[0])
    
    return porosity



def resistivity_to_saturation2(resistivity, rhos, n, sigma_sur=0):
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
    resistivity_array = np.atleast_1d(resistivity)
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
    saturation = np.zeros_like(resistivity_array)
    
    # Solve for each resistivity value
    for i in range(len(resistivity_array)):
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

