"""
Petrophysical models for electrical resistivity.

This module implements the Waxman-Smits model to relate subsurface electrical
resistivity to water content, porosity, and other formation properties.
The model accounts for both electrolytic conduction through pore water and
surface conduction along grain-water interfaces.

The core Waxman-Smits equation for formation conductivity (σ) is:
    σ_formation = (1/F*) * σ_water * S^n + B * Qv * S^(n-1) / F*
    where F* is formation factor, σ_water is water conductivity, S is saturation,
    n is saturation exponent, B is equivalent conductance of clay cations,
    Qv is cation exchange capacity per unit pore volume.

A simplified form used here (as described in the original module docstring) is:
    σ = σ_fully_saturated * S^n + σ_surface_effects * S^(n-1)
    where:
    - σ is the bulk electrical conductivity of the formation.
    - σ_fully_saturated (σ_sat) is the conductivity of the rock when fully saturated
      with conductive fluid, assuming no surface conduction effects. This can be
      equated to (1/F*) * σ_water from Archie's law, or 1/rhos where rhos is
      the resistivity of the rock fully saturated with this water (no surface effects).
    - σ_surface_effects (σ_s or sigma_sur) represents the contribution from surface
      conduction, related to B * Qv / F*.
    - S is the water saturation (S = θ/φ, where θ is volumetric water content
      and φ is porosity).
    - n is the saturation exponent (often close to 2).

Resistivity (ρ) is the reciprocal of conductivity (ρ = 1/σ).
"""
import numpy as np
from scipy.optimize import fsolve
from typing import Union, cast # Used for type hinting

# Define a type alias for inputs that can be scalar or array-like
ScalarOrArray = Union[float, np.ndarray]


def water_content_to_resistivity(water_content: ScalarOrArray,
                                 rhos: float,
                                 n: float,
                                 porosity: ScalarOrArray,
                                 sigma_sur: float = 0.0) -> ScalarOrArray:
    """
    Convert volumetric water content to bulk resistivity using a simplified Waxman-Smits model.

    Args:
        water_content (ScalarOrArray): Volumetric water content (θ) [m³/m³].
                                       Can be a scalar or a NumPy array.
        rhos (float): Resistivity of the rock when fully saturated with water,
                      assuming no surface conduction effects (1/σ_sat) [ohm·m].
                      This represents the Archie part of the rock resistivity.
        n (float): Saturation exponent (dimensionless). Typically around 2.
        porosity (ScalarOrArray): Porosity (φ) of the rock [m³/m³].
                                  Can be a scalar or a NumPy array (must be broadcastable
                                  with `water_content`).
        sigma_sur (float, optional): Surface conductivity term (σs) [S/m].
                                     Represents the contribution of clay/surface conduction.
                                     Defaults to 0.0, which reduces the model to Archie's Law
                                     if n-1 term is handled appropriately or if saturation is 1.
                                     (Note: The original model has S^(n-1); if S is very low and n>1,
                                     this term can become very large if sigma_sur is non-zero).

    Returns:
        ScalarOrArray: Calculated bulk resistivity (ρ) [ohm·m]. Returns a scalar if inputs
                       are scalar, otherwise a NumPy array.

    Raises:
        ValueError: If `rhos` is zero or negative.
        ZeroDivisionError: If calculated conductivity `sigma` is zero.
                           (Note: `1.0 / sigma` can also produce `inf`).
    """
    if rhos <= 0:
        raise ValueError("rhos (fully saturated resistivity) must be positive.")
    if np.any(np.asarray(porosity) <= 0) or np.any(np.asarray(porosity) > 1):
        # Potential Issue: Porosity should be within (0, 1]. Zero porosity is problematic.
        print("Warning: Porosity values are not within the typical range (0, 1].")

    # Ensure inputs are NumPy arrays for vectorized operations
    wc_arr = np.atleast_1d(water_content)
    por_arr = np.atleast_1d(porosity)

    # Calculate water saturation (S = θ/φ)
    # Avoid division by zero if porosity is zero; result in NaN or Inf handled by clip.
    # Or, handle porosity = 0 explicitly: if porosity is 0, saturation is undefined or 0.
    # Let's assume if porosity is 0, saturation is 0 (if wc is also 0) or undefined (NaN).
    # np.divide handles this by producing nan for 0/0 and inf for x/0 (with warnings).
    saturation = np.divide(wc_arr, por_arr, out=np.full_like(wc_arr, np.nan), where=por_arr!=0)

    # Clip saturation to physically meaningful range [0, 1].
    # Handles cases where wc > porosity due to measurement errors or wc < 0.
    saturation = np.clip(saturation, 0.0, 1.0)
    
    # Calculate conductivity of the rock fully saturated with water (σ_sat = 1/rhos)
    sigma_sat = 1.0 / rhos
    
    # Calculate bulk conductivity (σ) using the Waxman-Smits type equation:
    # σ = σ_sat * S^n + σ_sur * S^(n-1)
    # Potential Issue: If S=0 and n-1 < 0 (i.e., n < 1), S^(n-1) term is undefined (division by zero).
    # Standard Archie/Waxman-Smits typically assumes n >= 1 (often n approx 2).
    # If n=1, S^(n-1) = S^0 = 1 (for S>0). If S=0 and n=1, 0^0 is usually 1 in this context.
    # If S=0 and n>1, S^(n-1) = 0.
    # If S=0 and n<1, S^(n-1) -> inf.
    # Let's handle S=0 for the surface term carefully.

    sigma_bulk_part = sigma_sat * (saturation ** n)

    sigma_surface_part = np.zeros_like(saturation)
    # Only calculate surface term if sigma_sur is non-zero and saturation > 0,
    # or if n=1 (then S^0 = 1, so it contributes sigma_sur).
    # This avoids 0^(negative number) issues.
    if sigma_sur != 0:
        if n == 1:
            # S^(n-1) = S^0 = 1 for S > 0. For S = 0, 0^0 is context-dependent, often 1.
            # If we assume 0^0 = 0 for physical reasons (no surface cond. if no water film):
            # sigma_surface_part[saturation > 0] = sigma_sur
            # However, power function np.power(0,0) is 1. Let's rely on that default.
            sigma_surface_part = sigma_sur * (saturation ** (n - 1)) # This will be sigma_sur * 1.0
        else: # n != 1
            # Calculate S^(n-1) only where S > 0 to avoid 0^(negative) if n < 1.
            # If n > 1, 0^(n-1) is 0, which is fine.
            # If n < 1, S^(n-1) can be problematic at S=0.
            # However, saturation is clipped to [0,1]. If S=0, term is 0 unless n-1 is negative.
            # For typical n values (e.g., n=2), n-1=1, so S^(n-1) = S.
            # The term sigma_sur * S^(n-1)
            # If S is 0, and n-1 is positive, this term is 0.
            # If S is 0, and n-1 is 0 (n=1), this term is sigma_sur. (Handled above)
            # If S is 0, and n-1 is negative (n<1), this term is Inf. This should be avoided or means model is misapplied.
            # Assuming n >= 1 for physical validity of this simplified form.
            # If S is very small but non-zero, and n-1 < 0, S^(n-1) can be very large.
            # This is a known behavior of the model for low S if n is not chosen carefully or if sigma_sur is large.

            # To prevent issues with 0^(negative power) if n < 1, we can mask:
            non_zero_satur = saturation > 1e-9 # Threshold to avoid numerical issues with very small S
            if n < 1: # n-1 is negative
                 sigma_surface_part[non_zero_satur] = sigma_sur * (saturation[non_zero_satur] ** (n - 1))
                 # For S=0, this term would be infinite. It's left as 0 here assuming no surface path.
                 # This implies that if n<1, the model might not be well-behaved at S->0 with sigma_sur > 0.
            else: # n > 1 (n-1 > 0) or n=1 (handled by np.power(0,0)=1)
                 sigma_surface_part = sigma_sur * (saturation ** (n - 1))


    sigma = sigma_bulk_part + sigma_surface_part

    # Convert bulk conductivity to bulk resistivity (ρ = 1/σ)
    # Handle potential division by zero if sigma is zero (e.g., if water_content and sigma_sur are both zero).
    # np.divide will produce inf if sigma is 0, which might be desired (infinite resistivity).
    resistivity = np.full_like(sigma, np.inf) # Default to infinite resistivity
    non_zero_sigma = sigma != 0
    resistivity[non_zero_sigma] = 1.0 / sigma[non_zero_sigma]
    
    # If original input was scalar, return scalar
    if np.isscalar(water_content) and np.isscalar(porosity):
        return resistivity[0]
    return resistivity


def resistivity_to_water_content(resistivity: ScalarOrArray,
                                 rhos: float,
                                 n: float,
                                 porosity: ScalarOrArray,
                                 sigma_sur: float = 0.0) -> ScalarOrArray:
    """
    Convert bulk resistivity to volumetric water content using the simplified Waxman-Smits model.

    This function first converts resistivity to saturation using `resistivity_to_saturation`,
    and then calculates water content as saturation multiplied by porosity.

    Args:
        resistivity (ScalarOrArray): Bulk resistivity (ρ) [ohm·m].
        rhos (float): Resistivity of the rock when fully saturated (1/σ_sat) [ohm·m].
        n (float): Saturation exponent (dimensionless).
        porosity (ScalarOrArray): Porosity (φ) of the rock [m³/m³].
        sigma_sur (float, optional): Surface conductivity term (σs) [S/m]. Defaults to 0.0.

    Returns:
        ScalarOrArray: Calculated volumetric water content (θ) [m³/m³].
    """
    # Calculate saturation (S) from resistivity
    saturation = resistivity_to_saturation(resistivity, rhos, n, sigma_sur)
    
    # Ensure porosity is a NumPy array for vectorized multiplication
    por_arr = np.atleast_1d(porosity)
    sat_arr = np.atleast_1d(saturation) # resistivity_to_saturation should return array if input is array

    # Calculate water content (θ = S * φ)
    water_content = sat_arr * por_arr # Broadcasting should handle if shapes are compatible
                                     # e.g. saturation (N,) and porosity (scalar) or porosity (N,)
    
    # Match output type (scalar/array) to resistivity input type for consistency,
    # assuming porosity might be scalar.
    if np.isscalar(resistivity) and water_content.size == 1:
        return water_content[0]
    return water_content


def resistivity_to_saturation(resistivity: ScalarOrArray,
                              rhos: float,
                              n: Union[float, np.ndarray], # n can be array here
                              sigma_sur: Union[float, np.ndarray] = 0.0 # sigma_sur can be array
                             ) -> ScalarOrArray:
    """
    Convert bulk resistivity to water saturation using the simplified Waxman-Smits model.

    The relationship is given by: 1/ρ = (1/rhos) * S^n + σ_sur * S^(n-1).
    This equation is solved numerically for S using `scipy.optimize.fsolve`.
    If `sigma_sur` is zero, the equation simplifies to Archie's Law for saturation.

    Args:
        resistivity (ScalarOrArray): Bulk resistivity (ρ) [ohm·m].
        rhos (float): Resistivity of the rock when fully saturated (1/σ_sat) [ohm·m].
        n (Union[float, np.ndarray]): Saturation exponent (dimensionless). Can be scalar or array
                                      broadcastable with resistivity.
        sigma_sur (Union[float, np.ndarray], optional): Surface conductivity term (σs) [S/m].
                                                        Can be scalar or array. Defaults to 0.0.

    Returns:
        ScalarOrArray: Calculated water saturation (S) [m³/m³], clipped to [0, 1].

    Raises:
        ValueError: If `rhos` is non-positive or if numerical solution fails for any point.
    """
    if rhos <= 0:
        raise ValueError("rhos (fully saturated resistivity) must be positive.")

    # Ensure inputs are NumPy arrays for consistent processing and broadcasting.
    # `np.atleast_1d` converts scalars to 1-element arrays.
    resistivity_arr = np.atleast_1d(resistivity)
    sigma_sur_arr = np.atleast_1d(sigma_sur)
    n_arr = np.atleast_1d(n)

    # Broadcast sigma_sur and n to match the shape of resistivity_array if they are scalar
    # This simplifies element-wise operations in the loop.
    if sigma_sur_arr.shape != resistivity_arr.shape and sigma_sur_arr.size == 1:
        sigma_sur_arr = np.full_like(resistivity_arr, sigma_sur_arr[0], dtype=np.float64)
    if n_arr.shape != resistivity_arr.shape and n_arr.size == 1:
        n_arr = np.full_like(resistivity_arr, n_arr[0], dtype=np.float64)

    # Check for shape compatibility after attempting broadcast
    if sigma_sur_arr.shape != resistivity_arr.shape or n_arr.shape != resistivity_arr.shape:
        raise ValueError("Shapes of resistivity, n, and sigma_sur must be compatible for broadcasting or be identical.")

    # Calculate conductivity of the rock fully saturated with water (σ_sat = 1/rhos)
    sigma_sat = 1.0 / rhos
    
    # Initial guess for saturation (S_initial) using Archie's Law (ignoring sigma_sur).
    # This is robust when sigma_sur is small or zero.
    # S_initial = (σ_bulk / σ_sat)^(1/n) = ( (1/ρ_bulk) / (1/ρ_sat) )^(1/n) = (ρ_sat / ρ_bulk)^(1/n)
    # Handle potential division by zero or negative results from resistivity_arr if it contains non-positive values.
    # Resistivity should be positive. If resistivity_arr can be zero or negative, it implies an issue.
    # For Archie's, if resistivity_arr < rhos, S_initial > 1. If resistivity_arr is very large, S_initial -> 0.
    # Clipping S_initial to [0.01, 1.0] provides a bounded starting point for fsolve.
    # (0.01 instead of 0.0 to avoid issues if n-1 < 0 in the full equation at S=0).

    # Suppress warnings for invalid operations (e.g., division by zero if resistivity_arr contains 0)
    # These will result in inf/nan which are handled by clipping or fsolve.
    with np.errstate(divide='ignore', invalid='ignore'):
        s_initial_guess = (rhos / resistivity_arr) ** (1.0 / n_arr)
    
    s_initial_guess = np.clip(s_initial_guess, 0.01, 1.0) # Clip initial guess
    # For elements where resistivity_arr was non-positive, s_initial_guess might be NaN.
    # Replace NaNs in initial guess with a sensible default (e.g., 0.5 or 0.01).
    s_initial_guess[np.isnan(s_initial_guess)] = 0.01
    
    # Initialize output saturation array
    saturation_result = np.zeros_like(resistivity_arr, dtype=float)

    # Iterate and solve for saturation for each element if inputs are arrays.
    # np.nditer could be used for more complex broadcasting, but direct indexing is fine if shapes match.
    for i in range(resistivity_arr.size):
        # Current values for this element
        current_resistivity = resistivity_arr[i]
        current_sigma_sur = sigma_sur_arr[i]
        current_n = n_arr[i]
        current_s_initial = s_initial_guess[i]

        # If resistivity is non-physical (e.g., zero or negative), result is ill-defined.
        # We can assign NaN or a boundary value (0 or 1). Let's try NaN, then clip later.
        if current_resistivity <= 0:
            saturation_result[i] = np.nan # Non-physical resistivity
            continue

        target_conductivity = 1.0 / current_resistivity

        if np.isclose(current_sigma_sur, 0.0):
            # Simplified case: Archie's Law (sigma_sur = 0)
            # S^n = (target_conductivity / sigma_sat)
            # Handle cases where target_conductivity/sigma_sat might be negative (non-physical)
            # or where sigma_sat is zero (should not happen if rhos > 0).
            ratio = target_conductivity / sigma_sat
            if ratio < 0: # Non-physical
                saturation_result[i] = np.nan
            elif current_n == 0: # Avoid 0th root; S^0 = 1 implies ratio must be 1.
                 saturation_result[i] = 1.0 if np.isclose(ratio,1.0) else np.nan # Or 0.0 if ratio is 0? Undefined.
            else:
                saturation_result[i] = ratio ** (1.0 / current_n)
        else:
            # Full Waxman-Smits form, requires numerical solution for S:
            # sigma_sat * S^n + sigma_sur * S^(n-1) - target_conductivity = 0
            # Potential Issue: fsolve might fail or find non-physical roots if parameters are extreme
            # or if the initial guess is poor. The function must be well-behaved.
            # The term S^(n-1) can cause issues if S=0 and n-1 is negative (n<1).
            # We clipped S_initial to be >0.
            
            def waxman_smits_func(S_var: float) -> float:
                # S_var is the variable (saturation) we are solving for.
                # Clip S_var within [0,1] during function evaluation for stability, though fsolve might go outside.
                S_eval = np.clip(S_var, 0.0, 1.0)

                term1 = sigma_sat * (S_eval ** current_n)
                term2 = 0.0
                # Handle S^(n-1) carefully for S=0 cases
                if S_eval == 0.0:
                    if current_n > 1: # n-1 > 0
                        term2 = 0.0
                    elif current_n == 1: # n-1 = 0, S^0 = 1 (conventionally for S>0, 0^0=1 in numpy)
                        term2 = current_sigma_sur # sigma_sur * 1.0
                    else: # n < 1, n-1 < 0. 0 to a negative power is problematic (inf).
                          # This indicates an issue with model applicability or parameters.
                          # Return a large number to steer fsolve away if it tries S=0.
                          # However, S_eval is already clipped, so S_eval=0 means S_var was <=0.
                          # If fsolve provides S_var=0, and n<1, this implies infinite conductivity from surface term.
                          # This case should be rare with typical n values (>=1).
                          # If S_eval is truly 0 from clipping, this term is problematic.
                          # Let's assume n >= 1 based on typical usage.
                          # If n < 1 and S_eval=0, this term should contribute massively if current_sigma_sur > 0.
                          # For fsolve, better to ensure function is defined.
                          # If we expect S to be very small, S^(n-1) can be huge if n-1<0.
                          # Let's assume n >= 1 for physical formulation of this simplified model.
                           pass # term2 remains 0, or handle as error if n<1 is expected.
                else: # S_eval > 0
                    term2 = current_sigma_sur * (S_eval ** (current_n - 1))

                return term1 + term2 - target_conductivity
            
            try:
                # `fsolve` attempts to find roots of `waxman_smits_func`.
                # `xtol` can be adjusted for precision; `full_output=False` for only solution.
                solution_array, infodict, ier, mesg = fsolve(waxman_smits_func, current_s_initial, xtol=1.49012e-8, full_output=True)
                if ier == 1: # ier=1 means solution converged.
                    saturation_result[i] = solution_array[0]
                else:
                    # Solution did not converge.
                    # print(f"Warning: Saturation solver did not converge for resistivity {current_resistivity} (index {i}). Message: {mesg}")
                    # Fallback: use initial Archie guess, or set to NaN. Using NaN is safer.
                    saturation_result[i] = np.nan
            except Exception as e:
                # Catch any other errors during fsolve.
                # print(f"Error during fsolve for resistivity {current_resistivity} (index {i}): {e}")
                saturation_result[i] = np.nan
    
    # Final clip to ensure all computed/fallback values are within physical bounds [0, 1].
    # NaNs will remain NaNs.
    saturation_final = np.clip(saturation_result, 0.0, 1.0)
    
    # Return scalar if the original resistivity input was scalar.
    if np.isscalar(resistivity):
        # Cast to standard float if it's a 0-dim array scalar.
        return float(saturation_final[0]) if saturation_final.ndim == 0 or saturation_final.size == 1 else saturation_final[0]
    
    return saturation_final

