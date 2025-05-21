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
    # Calculate saturation
    saturation = water_content / porosity
    saturation = np.clip(saturation, 0.0, 1.0)
    
    # Calculate conductivity using Waxman-Smits model
    sigma_sat = 1.0 / rhos
    sigma = sigma_sat * saturation**n + sigma_sur * saturation**(n-1)
    
    # Convert conductivity to resistivity
    resistivity = 1.0 / sigma
    
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


def resistivity_to_saturation(resistivity, rhos, n, sigma_sur=0):
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

