"""
Simplified Waxman-Smits model for converting between water content and resistivity.

This implementation follows the Waxman-Smits model that expresses conductivity as:
    
    σ = σsat * S^n + σs * S^(n-1)  (Equation 1)
    
where:
- σ (sigma) is the bulk electrical conductivity of the formation [Siemens/meter, S/m].
- σsat (sigma_sat) is the conductivity of the formation when fully saturated with water,
  assuming no contribution from surface conductivity (i.e., clean sand term).
  It's related to the resistivity of the saturating fluid (water) and formation factor.
  σsat = 1 / (F * ρw), where F is formation factor, ρw is water resistivity.
  In this model, σsat is taken as 1/rhos, where rhos is `saturated resistivity without surface effects`.
- σs (sigma_s, or sigma_sur in code) is the surface conductivity contribution [S/m].
  This term accounts for the additional conductivity pathway along the grain surfaces,
  which becomes significant in shaly or clayey formations, especially at low water salinities.
- S is the water saturation (fraction of pore volume occupied by water).
  Calculated as S = θ / φ, where θ (theta) is volumetric water content and φ (phi) is porosity.
  S ranges from 0 (completely dry) to 1 (fully saturated).
- n is the saturation exponent, an empirical parameter often close to 2. It describes how
  the connectivity of the conductive water phase changes with saturation.

The resistivity (ρ, rho) is the reciprocal of conductivity: ρ = 1 / σ [ohm-meters, Ω·m].

This model simplifies the full Waxman-Smits by directly using σsat and σs,
rather than deriving them from CEC (Cation Exchange Capacity) and other parameters.
If sigma_sur = 0, the model reduces to Archie's Law for the shaly term being zero:
    σ = σsat * S^n  => (1/ρ) = (1/rhos) * S^n => ρ = rhos * S^(-n)
which is a form of Archie's law where rhos = F * ρw.
The (n-1) exponent for the surface conductivity term is a common simplification/variation.
Other forms might use different exponents for the surface conductivity term.
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
        array: Resistivity values (ρ) in Ω·m.
    """
    # Calculate water saturation (S) from volumetric water content (θ) and porosity (φ).
    # S = θ / φ
    saturation = water_content / porosity
    # Ensure saturation is within the physical bounds [0, 1].
    # Clipping handles cases where water_content might slightly exceed porosity due to numerical precision
    # or if water_content < 0 (though physically unlikely for content).
    saturation = np.clip(saturation, 0.0, 1.0)
    
    # Calculate conductivity of the formation when fully saturated (clean sand term, σsat).
    # rhos is the resistivity of the formation when fully saturated with conductive fluid,
    # assuming no surface conductivity effects. So, sigma_sat = 1 / rhos.
    sigma_sat = 1.0 / rhos # Units: S/m if rhos is in Ω·m.

    # Calculate bulk conductivity (σ) using the simplified Waxman-Smits model.
    # σ = σsat * S^n + σs * S^(n-1)
    # sigma_sur (σs) is the surface conductivity term.
    # n is the saturation exponent.
    # The term S^(n-1) for surface conductivity implies that surface conduction paths
    # are also dependent on water saturation, but possibly differently than bulk pore water paths.
    # SUGGESTION: Ensure inputs (water_content, porosity) are broadcastable if they are arrays.
    # Numpy handles broadcasting if shapes are compatible.
    sigma = sigma_sat * saturation**n + sigma_sur * saturation**(n-1)
    
    # Convert bulk conductivity (σ) to bulk resistivity (ρ).
    # ρ = 1 / σ
    # Handle potential division by zero if sigma is 0 (e.g., if saturation is 0 and sigma_sur is 0).
    # np.divide handles this by returning inf, which might be acceptable or could be replaced by a large number/NaN.
    with np.errstate(divide='ignore'): # Suppress division by zero warnings if sigma can be zero.
        resistivity = 1.0 / sigma
    # If sigma is zero (e.g. dry, no surface cond), resistivity is infinite.
    # Depending on application, inf might be capped or handled.
    # For example: resistivity[np.isinf(resistivity)] = some_large_finite_number
    
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
        array: Volumetric water content values (θ). This will be a fraction (e.g., 0.0 to porosity value).
    """
    # First, convert bulk resistivity (ρ) to water saturation (S) using the inverse of the Waxman-Smits model.
    # This step typically requires solving a non-linear equation, implemented in `resistivity_to_saturation`.
    saturation = resistivity_to_saturation(resistivity, rhos, n, sigma_sur)
    
    # Then, calculate volumetric water content (θ) from saturation (S) and porosity (φ).
    # θ = S * φ
    # This assumes that porosity represents the total interconnected pore space available for water.
    water_content = saturation * porosity
    # The result (water_content) should range from 0 up to the value of porosity.
    # SUGGESTION: Consider if clipping water_content (e.g., water_content = np.clip(water_content, 0, porosity))
    # is needed if saturation calculation could yield S > 1 due to numerical issues or model limits,
    # though `resistivity_to_saturation` already clips S to [0,1].
    
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
        array: Saturation values (S), ranging from 0.0 to 1.0.
    """
    # Ensure inputs that can be scalar or array are treated as arrays for consistent processing.
    # np.atleast_1d converts scalars to 1-element arrays, leaves arrays unchanged.
    resistivity_array = np.atleast_1d(resistivity)
    sigma_sur_input = np.atleast_1d(sigma_sur) # Surface conductivity
    n_input = np.atleast_1d(n) # Saturation exponent

    # Broadcast scalar inputs (sigma_sur, n) to match the shape of resistivity_array if needed.
    # This allows providing a single sigma_sur or n for an array of resistivities.
    sigma_sur_array = sigma_sur_input
    if sigma_sur_input.shape == (1,) and resistivity_array.shape != (1,):
        sigma_sur_array = np.full_like(resistivity_array, sigma_sur_input[0], dtype=np.common_type(resistivity_array, sigma_sur_input))

    n_array = n_input
    if n_input.shape == (1,) and resistivity_array.shape != (1,):
        n_array = np.full_like(resistivity_array, n_input[0], dtype=np.common_type(resistivity_array, n_input))

    # Calculate conductivity of the formation when fully saturated (clean sand term).
    sigma_sat = 1.0 / rhos # rhos = resistivity of 100% saturated rock without surface effects.

    # --- Initial Guess for Saturation (S_initial) ---
    # For the numerical solver (fsolve), a good initial guess improves convergence and stability.
    # Here, Archie's law (Waxman-Smits without surface conductivity) is used to get S_initial.
    # Archie: ρ = rhos * S^(-n)  => S = (rhos / ρ)^(1/n)
    # This assumes sigma_sur = 0 for the initial guess.
    # Clipping S_initial to [0.01, 1.0]:
    #   - Avoids S=0, which can cause issues with S^(n-1) if n-1 is negative (e.g. n<1, though typically n~2).
    #   - 0.01 is a small, non-zero saturation.
    #   - Max 1.0 is physically correct.
    S_initial = (rhos / resistivity_array)**(1.0 / n_array)
    S_initial = np.clip(S_initial, 0.01, 1.0) # Clip for robustness in fsolve.

    # Initialize an array to store the calculated saturation values.
    saturation_results = np.zeros_like(resistivity_array, dtype=float) # Ensure float for results

    # --- Solve for Saturation ---
    # Iterate over each element if inputs were arrays. If all inputs are scalar, this loop runs once.
    # This element-wise solution is necessary because fsolve is not vectorized for this kind of problem.
    # SUGGESTION: For very large arrays, performance could be a concern. Vectorized root-finding
    # or pre-computed lookup tables might be faster if fsolve becomes a bottleneck.
    for i in range(resistivity_array.size):
        # Handle single-element arrays from np.atleast_1d if original inputs were scalar
        current_resistivity = resistivity_array.flat[i]
        current_sigma_sur = sigma_sur_array.flat[i]
        current_n = n_array.flat[i]
        current_S_initial = S_initial.flat[i]

        if current_sigma_sur == 0:
            # If surface conductivity is zero, the model simplifies to Archie's Law.
            # S_initial is already the Archie's solution, so use that.
            saturation_results.flat[i] = current_S_initial
        else:
            # If surface conductivity is non-zero, need to solve the full Waxman-Smits equation numerically.
            # The equation to solve for S is:
            # σ_bulk = σ_sat * S^n + σ_sur * S^(n-1)
            # Where σ_bulk = 1 / current_resistivity.
            # We need to find S such that: func(S) = σ_sat * S^n + σ_sur * S^(n-1) - (1/current_resistivity) = 0
            
            # Define the function to find the root of.
            # `S_val` is the saturation value being solved for.
            def waxman_smits_func_for_fsolve(S_val):
                # Ensure S_val is within reasonable bounds for calculation, e.g., to avoid S**(negative) if S is zero.
                S_val = max(S_val, 1e-9) # Prevent S=0 if n-1 < 0
                term1 = sigma_sat * S_val**current_n
                term2 = current_sigma_sur * S_val**(current_n - 1)
                return term1 + term2 - (1.0 / current_resistivity)
            
            # Use fsolve to find the root (S) of waxman_smits_func_for_fsolve.
            # `current_S_initial` is used as the starting estimate for the solver.
            # `fsolve` returns an array, so take the first element for the solution.
            solution_array = fsolve(waxman_smits_func_for_fsolve, current_S_initial, xtol=1e-6) # xtol for tolerance
            saturation_results.flat[i] = solution_array[0]

    # Ensure calculated saturation is within physical bounds [0, 1].
    # Clipping handles any numerical inaccuracies from fsolve or edge cases.
    saturation_clipped = np.clip(saturation_results, 0.0, 1.0)

    # If the original resistivity input was a scalar, return a scalar float.
    # Otherwise, return the NumPy array of saturation values.
    if np.isscalar(resistivity):
        return float(saturation_clipped[0]) # Convert single-element array to Python float
    else:
        # If original inputs were arrays, reshape result to match resistivity_array's shape.
        return np.reshape(saturation_clipped, resistivity_array.shape)

