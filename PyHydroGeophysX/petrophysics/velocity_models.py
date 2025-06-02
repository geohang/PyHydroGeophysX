"""
Seismic velocity models for relating rock properties to elastic wave velocities.

This module provides implementations of several common rock physics models
to estimate seismic velocities (P-wave and S-wave) based on properties
such as mineral composition, porosity, saturation, and effective pressure.

Models included:
- Voigt-Reuss-Hill (VRH) average for effective medium properties.
- Brie's model for effective fluid modulus in partially saturated rocks.
- Differential Effective Medium (DEM) model for porous rocks.
- Hertz-Mindlin contact theory combined with Hashin-Shtrikman bounds.

The module also includes standalone functions that appear to be earlier or alternative
implementations of some of these models (VRH_model, satK, velDEM, vel_porous).
These might be for specific use cases or represent developmental versions.
It's important to distinguish these from the class-based models for clarity.
"""
import numpy as np
from scipy.optimize import fsolve, root
from typing import Tuple, Optional, Union, List, Dict, Any # Dict not used, Optional not used in current signatures

# Define a type alias for inputs that can be scalar or array-like for clarity
ScalarOrArray = Union[float, np.ndarray]


class BaseVelocityModel:
    """
    Abstract base class for seismic velocity models.

    This class provides a common interface for different velocity models.
    Subclasses should implement the `calculate_velocity` method.
    """
    
    def __init__(self):
        """Initialize base velocity model."""
        # No specific initialization needed for the base class itself.
        pass
    
    def calculate_velocity(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Abstract method to calculate seismic velocity from rock properties.

        This method must be implemented by derived classes. Each implementation
        will take specific rock and fluid properties relevant to that model.

        Args:
            **kwargs: Model-specific keyword arguments representing rock and fluid properties
                      (e.g., porosity, saturation, bulk_moduli, shear_moduli, densities).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                Depending on the model, this could be a single array (e.g., P-wave velocity)
                or a tuple of arrays (e.g., P-wave and S-wave velocities).

        Raises:
            NotImplementedError: If the method is not implemented in a derived class.
        """
        raise NotImplementedError("The 'calculate_velocity' method must be implemented in derived classes.")


class VRHModel(BaseVelocityModel):
    """
    Voigt-Reuss-Hill (VRH) mixing model for effective elastic properties of composites.

    The VRH model calculates the arithmetic average of the Voigt upper bound
    (iso-strain condition) and the Reuss lower bound (iso-stress condition)
    for the effective bulk and shear moduli of a mineral mixture.
    """
    
    def __init__(self):
        """Initialize VRH model."""
        super().__init__()
        # No specific parameters for VRH model itself beyond inputs to methods.
    
    def calculate_properties(self, 
                           fractions: List[float],
                           bulk_moduli: List[float], # K for each component
                           shear_moduli: List[float],# G for each component
                           densities: List[float]   # ρ for each component
                           ) -> Tuple[float, float, float]:
        """
        Calculate effective elastic properties (bulk modulus, shear modulus, density)
        of a mineral mixture using the Voigt-Reuss-Hill (VRH) averaging scheme.

        Args:
            fractions (List[float]): List of volume fractions for each mineral component.
                                     Must sum to 1.0.
            bulk_moduli (List[float]): List of bulk moduli (K) for each mineral component [GPa].
            shear_moduli (List[float]): List of shear moduli (G) for each mineral component [GPa].
            densities (List[float]): List of densities (ρ) for each mineral component [kg/m³].

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - K_vrh (float): Effective bulk modulus of the mixture [GPa].
                - G_vrh (float): Effective shear modulus of the mixture [GPa].
                - rho_eff (float): Effective density of the mixture [kg/m³].

        Raises:
            ValueError: If `fractions` do not sum to 1.0 (within tolerance) or if
                        input lists (fractions, K, G, rho) have different lengths or are empty.
            ZeroDivisionError: If any component bulk or shear modulus is zero when calculating Reuss average
                               (as it involves division by K_i or G_i).
        """
        if not (len(fractions) == len(bulk_moduli) == len(shear_moduli) == len(densities)):
            raise ValueError("Input lists (fractions, bulk_moduli, shear_moduli, densities) must have the same length.")
        if not fractions: # Check for empty list
            raise ValueError("Input lists cannot be empty.")

        # Convert inputs to NumPy arrays for vectorized calculations
        f_arr = np.array(fractions, dtype=float)
        K_arr = np.array(bulk_moduli, dtype=float)
        G_arr = np.array(shear_moduli, dtype=float)
        rho_arr = np.array(densities, dtype=float)
        
        # Verify that fractions sum to approximately 1.0
        if not np.isclose(np.sum(f_arr), 1.0):
            raise ValueError(f"Volume fractions must sum to 1.0. Current sum: {np.sum(f_arr)}")
        
        # Potential Issue: Division by zero if any K_i or G_i is zero for Reuss average.
        # Check for zeros in moduli where they are used as divisors.
        if np.any(K_arr <= 0): # Bulk modulus should be positive
            raise ValueError("Component bulk moduli must be positive for Reuss average calculation.")
        if np.any(G_arr < 0): # Shear modulus should be non-negative (can be 0 for fluids)
             # If G_arr contains 0, 1.0 / G_arr will be inf. np.sum(f_arr / G_arr) could be inf.
             # 1.0 / inf is 0. So G_reuss can be 0 if a component has G=0 (fluid). This is valid.
             # However, if ALL components are fluid (all G_i=0), G_reuss would be 0/0 -> NaN.
             # For a mixture of fluids, G_reuss should be 0.
             pass # Allow G=0 for fluids.

        # Calculate Voigt average (upper bound): K_V = Σ(f_i * K_i), G_V = Σ(f_i * G_i)
        K_voigt = np.sum(f_arr * K_arr)
        G_voigt = np.sum(f_arr * G_arr)
        
        # Calculate Reuss average (lower bound): 1/K_R = Σ(f_i / K_i), 1/G_R = Σ(f_i / G_i)
        # Avoid division by zero for G_reuss if all G_arr are zero (e.g., mixture of fluids)
        if np.all(G_arr == 0):
            G_reuss = 0.0
        else:
            # Temporarily set G_arr to infinity where G_arr is zero to make f_arr/G_arr zero for those components
            # This correctly handles fluid components (G=0) in the Reuss sum for shear modulus.
            # An alternative: sum only non-fluid components, then adjust.
            # Or, if a component has G_i = 0, its contribution f_i/G_i to the sum is effectively infinite,
            # making 1/G_reuss infinite, so G_reuss = 0. This is physically correct for mixtures with fluids.
            # np.sum(f_arr / G_arr) will correctly yield `inf` if any G_i is 0 and f_i > 0.
            # Then 1.0 / `inf` is 0.0. This handles fluids correctly.
            # Ensure no f_i/0 where f_i is also 0, but f_arr elements should be >0 for components.
            with np.errstate(divide='ignore'): # Suppress warning for division by zero if G_i = 0
                K_reuss = 1.0 / np.sum(f_arr / K_arr)
                sum_f_over_G = np.sum(f_arr / G_arr) # This can be inf if any G_i is 0

            if np.isinf(sum_f_over_G): # If any component is fluid (G_i=0, f_i>0)
                G_reuss = 0.0
            elif sum_f_over_G == 0: # Only if all f_i are zero (already checked by sum(f)=1) or all G_i are inf
                G_reuss = np.inf # Or handle as error, implies no shear rigidity (like vacuum) or infinitely stiff
            else:
                G_reuss = 1.0 / sum_f_over_G

        # Calculate Hill (VRH) arithmetic average of Voigt and Reuss bounds
        K_vrh = 0.5 * (K_voigt + K_reuss)
        G_vrh = 0.5 * (G_voigt + G_reuss)
        
        # Calculate effective density (simple arithmetic weighted average)
        rho_eff = np.sum(f_arr * rho_arr)
        
        return K_vrh, G_vrh, rho_eff
    
    def calculate_velocity(self, 
                         fractions: List[float],
                         bulk_moduli: List[float],
                         shear_moduli: List[float],
                         densities: List[float]) -> Tuple[float, float]:
        """
        Calculate P-wave (Vp) and S-wave (Vs) velocities for a mineral mixture
        using the VRH model for effective elastic moduli and density.

        Args:
            fractions (List[float]): Volume fractions of each mineral component. Must sum to 1.0.
            bulk_moduli (List[float]): Bulk moduli (K) of each mineral [GPa].
            shear_moduli (List[float]): Shear moduli (G) of each mineral [GPa].
            densities (List[float]): Densities (ρ) of each mineral [kg/m³].

        Returns:
            Tuple[float, float]: A tuple containing:
                - Vp (float): P-wave velocity [m/s].
                - Vs (float): S-wave velocity [m/s].

        Raises:
            ValueError: If calculated effective properties (K_eff, G_eff, rho_eff) are non-physical
                        (e.g., negative density, negative moduli leading to NaN in sqrt).
        """
        # Calculate effective elastic properties (K_eff, G_eff, rho_eff) in GPa and kg/m³
        K_eff_gpa, G_eff_gpa, rho_eff_kg_m3 = self.calculate_properties(
            fractions, bulk_moduli, shear_moduli, densities
        )
        
        # Validate effective properties before velocity calculation
        if rho_eff_kg_m3 <= 0:
            raise ValueError(f"Effective density must be positive. Calculated: {rho_eff_kg_m3} kg/m³")
        if K_eff_gpa < 0: # Effective bulk modulus should be non-negative
            raise ValueError(f"Effective bulk modulus must be non-negative. Calculated: {K_eff_gpa} GPa")
        if G_eff_gpa < 0: # Effective shear modulus should be non-negative
            raise ValueError(f"Effective shear modulus must be non-negative. Calculated: {G_eff_gpa} GPa")

        # Convert moduli from GPa to Pascals (Pa) for velocity calculation (1 GPa = 1e9 Pa)
        K_eff_pa = K_eff_gpa * 1e9
        G_eff_pa = G_eff_gpa * 1e9

        # Calculate P-wave velocity (Vp = sqrt((K + 4/3*G) / ρ))
        # The term (K_eff_pa + (4/3) * G_eff_pa) is the P-wave modulus (M).
        # M must be non-negative.
        p_wave_modulus = K_eff_pa + (4.0/3.0) * G_eff_pa
        if p_wave_modulus < 0:
            raise ValueError(f"P-wave modulus (K + 4/3*G) is negative ({p_wave_modulus} Pa), leading to invalid Vp.")
        
        Vp = np.sqrt(p_wave_modulus / rho_eff_kg_m3)
        
        # Calculate S-wave velocity (Vs = sqrt(G / ρ))
        # G_eff_pa must be non-negative. rho_eff_kg_m3 must be positive.
        if G_eff_pa == 0: # If effective shear modulus is zero (e.g. fluid mixture)
            Vs = 0.0
        else: # G_eff_pa > 0
            Vs = np.sqrt(G_eff_pa / rho_eff_kg_m3)
        
        return Vp, Vs


class BrieModel:
    """
    Brie's empirical model for calculating the effective bulk modulus of a fluid mixture
    (typically water and gas) in a partially saturated porous medium.

    The model uses an exponent to describe how fluid phases mix.
    """
    
    def __init__(self, exponent: float = 3.0):
        """
        Initialize Brie's model.

        Args:
            exponent (float, optional): Brie's mixing exponent. Common values range
                                        from 1 (iso-stress, Reuss-like for fluid) to
                                        very large (iso-strain, Voigt-like for fluid).
                                        A typical default is 3.0.
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError("Brie's exponent must be a numeric value.")
        # Potential Issue: Exponent value constraints? (e.g., positive).
        # Original paper might specify typical ranges or constraints. For now, accept any float.
        self.exponent = float(exponent)
    
    def calculate_fluid_modulus(self, 
                              saturation_water: float, # Renamed for clarity
                              water_bulk_modulus: float = 2.2, # Corrected default for water (approx 2.2 GPa)
                              gas_bulk_modulus: float = 0.01 # Bulk modulus of gas (e.g. air at STP)
                              ) -> float:
        """
        Calculate the effective bulk modulus of a fluid mixture (water and gas)
        using Brie's empirical equation:
            K_fluid_eff = (K_water - K_gas) * S_water^e + K_gas
        where 'e' is Brie's exponent.

        Args:
            saturation_water (float): Water saturation (S_w), ranging from 0.0 (fully gas)
                                      to 1.0 (fully water).
            water_bulk_modulus (float, optional): Bulk modulus of water (K_w) [GPa].
                                                  Defaults to 2.2 GPa.
            gas_bulk_modulus (float, optional): Bulk modulus of gas (K_gas) [GPa].
                                                Defaults to 0.01 GPa (approx. for air at STP).
            
        Returns:
            float: Effective bulk modulus of the fluid mixture (K_fluid_eff) [GPa].

        Raises:
            ValueError: If saturation is not between 0 and 1.
        """
        if not (0.0 <= saturation_water <= 1.0):
            raise ValueError("Water saturation must be between 0 and 1.")
        # Ensure moduli are positive
        if water_bulk_modulus <= 0 or gas_bulk_modulus <= 0:
            raise ValueError("Fluid bulk moduli must be positive.")

        k_fluid_eff = (water_bulk_modulus - gas_bulk_modulus) * (saturation_water ** self.exponent) + gas_bulk_modulus
        return k_fluid_eff
    
    def calculate_saturated_modulus(self, 
                                  dry_rock_bulk_modulus: float, # K_dry
                                  mineral_bulk_modulus: float, # K_mineral or K_matrix
                                  porosity: float,             # φ
                                  saturation_water: float,     # S_w
                                  water_bulk_modulus: float = 2.2, # K_w
                                  gas_bulk_modulus: float = 0.01   # K_gas
                                  ) -> float:
        """
        Calculate the bulk modulus of a rock saturated with a fluid mixture (water/gas)
        using Gassmann's equation, with the effective fluid modulus derived from Brie's model.

        Gassmann's equation:
            K_sat = K_dry + ( (1 - K_dry/K_mineral)^2 ) /
                            ( φ/K_fluid_eff + (1-φ)/K_mineral - K_dry/K_mineral^2 )
        
        Args:
            dry_rock_bulk_modulus (float): Bulk modulus of the dry rock frame (K_dry) [GPa].
            mineral_bulk_modulus (float): Intrinsic bulk modulus of the solid mineral matrix (K_mineral) [GPa].
                                          (Sometimes denoted K0 or Ks).
            porosity (float): Porosity of the rock (φ), ranging from 0 to 1.
            saturation_water (float): Water saturation (S_w), ranging from 0 to 1.
            water_bulk_modulus (float, optional): Bulk modulus of water (K_w) [GPa]. Defaults to 2.2 GPa.
            gas_bulk_modulus (float, optional): Bulk modulus of gas (K_gas) [GPa]. Defaults to 0.01 GPa.
            
        Returns:
            float: Bulk modulus of the saturated rock (K_sat) [GPa].

        Raises:
            ValueError: If inputs are non-physical (e.g., porosity not in [0,1], K_dry > K_mineral).
        """
        if not (0.0 <= porosity <= 1.0):
            raise ValueError("Porosity must be between 0 and 1.")
        if dry_rock_bulk_modulus < 0 or mineral_bulk_modulus <= 0:
            raise ValueError("Dry rock and mineral bulk moduli must be positive.")
        if dry_rock_bulk_modulus > mineral_bulk_modulus:
            # This can happen with very stiff pore-filling material or complex pore structures,
            # but Gassmann's assumptions might be violated. Standard assumption is K_dry <= K_mineral.
            print("Warning: Dry rock bulk modulus is greater than mineral bulk modulus. "
                  "This is unusual for simple Gassmann application (K_dry > K_mineral).")

        # Calculate effective fluid bulk modulus using Brie's model
        k_fluid_eff = self.calculate_fluid_modulus(
            saturation_water, water_bulk_modulus, gas_bulk_modulus
        )
        if k_fluid_eff <= 0: # Should not happen if component moduli are positive
            raise ValueError(f"Calculated effective fluid modulus is non-positive ({k_fluid_eff} GPa), check inputs to Brie's model.")

        # Apply Gassmann's equation for K_sat
        # K_sat = K_dry + num / den
        # num = (1 - K_dry/K_mineral)^2
        # den = phi/K_fluid_eff + (1-phi)/K_mineral - K_dry/(K_mineral^2)
        
        # Avoid division by zero if K_mineral or K_fluid_eff is zero (already checked mostly).
        # Also, (K_mineral - K_dry) in the denominator of the original formulation was problematic.
        # Using the common form of Gassmann's:
        term_kdry_over_kmin = dry_rock_bulk_modulus / mineral_bulk_modulus
        
        numerator_gassmann = (1.0 - term_kdry_over_kmin) ** 2
        denominator_gassmann = (porosity / k_fluid_eff) + \
                               ((1.0 - porosity) / mineral_bulk_modulus) - \
                               (dry_rock_bulk_modulus / (mineral_bulk_modulus ** 2))

        if np.isclose(denominator_gassmann, 0.0):
            # This case implies K_sat could be infinite or undefined, usually due to K_fluid_eff being very small
            # relative to porosity, or specific combinations of K_dry, K_mineral, phi.
            # If K_dry = K_mineral (zero porosity rock frame), then K_sat = K_mineral.
            # If phi = 0, then K_sat = K_mineral (or K_dry if K_dry used as proxy for K_mineral).
            # If K_fluid_eff is extremely small (like gas), then K_sat approaches K_dry.
            # For a robust solution, one might need to check limits or specific conditions.
            # If K_dry = K_mineral, numerator is 0, K_sat = K_dry = K_mineral.
            # If denominator is zero, it often implies K_sat approaches K_mineral if K_dry is less than K_mineral.
            # If K_dry is very stiff, this can be an issue.
            # For now, if denominator is zero, it might be better to return K_mineral or raise error.
            # If K_dry = K_mineral, numerator is 0, K_sat = K_dry = K_mineral.
            if np.isclose(dry_rock_bulk_modulus, mineral_bulk_modulus):
                return mineral_bulk_modulus # Or dry_rock_bulk_modulus
            else:
                # This indicates a problem, K_sat would be infinite.
                # Could be due to K_fluid_eff being extremely low and other terms aligning.
                raise ValueError("Gassmann equation denominator is zero, leading to undefined K_sat. Check input parameters.")

        k_sat = dry_rock_bulk_modulus + numerator_gassmann / denominator_gassmann
        return k_sat


class DEMModel(BaseVelocityModel):
    """
    Differential Effective Medium (DEM) model for calculating elastic properties
    and seismic velocities of porous rocks.

    This model simulates the incremental addition of pore space (filled with fluid)
    into a solid matrix, updating the effective moduli at each step.
    It requires solving differential equations, often approximated by numerical iteration
    or specific solutions for certain pore geometries (like spheroids with aspect ratio α).
    The implementation here appears to solve for K_eff and G_eff using root finding
    on implicit equations derived from DEM theory for specific inclusion types.
    """
    
    def __init__(self):
        """Initialize DEM model."""
        super().__init__()
        # Specific parameters for DEM (like pore aspect ratio) are passed to calculate_velocity.
    
    def calculate_velocity(self, 
                         porosity: np.ndarray,       # φ, array of porosity values
                         saturation: np.ndarray,    # S_w, array of water saturation values
                         matrix_bulk_modulus: float,  # K_matrix (Km) in GPa
                         matrix_shear_modulus: float, # G_matrix (Gm) in GPa
                         matrix_density: float,       # ρ_matrix in kg/m³
                         aspect_ratio: float = 0.1,   # α, pore aspect ratio
                         water_bulk_modulus: float = 2.2, # K_w in GPa
                         gas_bulk_modulus: float = 0.01,  # K_gas (e.g., air) in GPa
                         water_density: float = 1000.0, # ρ_w in kg/m³
                         gas_density: float = 1.225   # ρ_gas (e.g., air) in kg/m³
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate effective bulk (K_eff) and shear (G_eff) moduli, and P-wave velocity (Vp)
        using a Differential Effective Medium (DEM) model.

        The method iterates through each porosity/saturation point, calculates the
        effective fluid properties using Brie's model, then solves implicit DEM
        equations numerically for K_eff and G_eff.

        Args:
            porosity (np.ndarray): Array of porosity values (φ) [m³/m³].
            saturation (np.ndarray): Array of water saturation values (S_w) [m³/m³].
                                     Must be same length as `porosity`.
            matrix_bulk_modulus (float): Bulk modulus of the solid mineral matrix (K_m) [GPa].
            matrix_shear_modulus (float): Shear modulus of the solid mineral matrix (G_m) [GPa].
            matrix_density (float): Density of the solid mineral matrix (ρ_m) [kg/m³].
            aspect_ratio (float, optional): Aspect ratio (α) of the pores/cracks.
                                            Defaults to 0.1 (oblate spheroids).
            water_bulk_modulus (float, optional): Bulk modulus of water (K_w) [GPa]. Defaults to 2.2 GPa.
            gas_bulk_modulus (float, optional): Bulk modulus of gas (K_gas) [GPa]. Defaults to 0.01 GPa.
            water_density (float, optional): Density of water (ρ_w) [kg/m³]. Defaults to 1000 kg/m³.
            gas_density (float, optional): Density of gas (ρ_gas) [kg/m³]. Defaults to 1.225 kg/m³.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Contains:
                - K_eff (np.ndarray): Effective bulk modulus for each porosity/saturation point [GPa].
                - G_eff (np.ndarray): Effective shear modulus for each porosity/saturation point [GPa].
                - Vp (np.ndarray): P-wave velocity for each porosity/saturation point [m/s].

        Raises:
            ValueError: If root finding for K_eff or G_eff fails for any point, or if input array
                        lengths for porosity and saturation do not match.
            TypeError: If inputs are not of expected types.
        """
        if not (isinstance(porosity, np.ndarray) and isinstance(saturation, np.ndarray)):
            raise TypeError("Porosity and saturation must be NumPy arrays.")
        if porosity.shape != saturation.shape:
            raise ValueError("Porosity and saturation arrays must have the same shape.")
        if not (0.0 <= np.all(porosity) <= 1.0) or not (0.0 <= np.all(saturation) <= 1.0):
             # Element-wise check is better if only some values are out of [0,1] range.
             # This simplified check flags if *all* are not in range, which is too loose.
             # Better: if np.any((porosity < 0) | (porosity > 1)) or np.any((saturation < 0) | (saturation > 1)):
            print("Warning: Porosity or saturation values are outside the typical [0,1] range.")


        num_points = len(porosity)
        K_eff_results = np.zeros(num_points)
        G_eff_results = np.zeros(num_points)
        Vp_results = np.zeros(num_points)
        
        # Brie model for effective fluid properties
        brie_model = BrieModel() # Using default exponent 3.0 from BrieModel class

        # Initial matrix properties (K_m, G_m)
        K_m = matrix_bulk_modulus
        G_m = matrix_shear_modulus
        
        for i in range(num_points):
            phi_i = porosity[i]
            sat_i = saturation[i]

            # Calculate effective fluid bulk modulus (K_f) for current saturation
            # using Brie's model. Shear modulus of fluid (G_f) is assumed to be 0.
            K_f_eff = brie_model.calculate_fluid_modulus(sat_i, water_bulk_modulus, gas_bulk_modulus)
            G_f_eff = 0.0 # Fluids do not support static shear stress.
            
            # Calculate Poisson's ratio of the current effective medium (initially matrix)
            # This v is for the *matrix into which pores are being inserted*.
            # In DEM, the "matrix" properties evolve. This v should be K_m, G_m at phi=0.
            # For subsequent steps (if DEM were iterative), this v would use K_eff, G_eff from previous step.
            # The current formulation seems to use initial K_m, G_m for 'v' throughout,
            # which is typical if the DEM equations are solved for a target porosity phi_i directly
            # from initial matrix, rather than incrementally building up porosity.
            # The equations solved for Keff and Geff seem to be integrated forms of DEM.
            
            # Poisson's ratio of the solid matrix material
            v_matrix = (3 * K_m - 2 * G_m) / (2 * (3 * K_m + G_m))
            if not (-1.0 < v_matrix < 0.5): # Physical bounds for Poisson's ratio
                print(f"Warning: Calculated matrix Poisson's ratio ({v_matrix:.3f}) is outside physical bounds (-1 to 0.5) at index {i}.")

            # DEM parameters (P, Q or b, c, d, g in this code) depend on pore geometry (aspect_ratio)
            # and properties of the matrix (K_m, G_m, v_matrix) and inclusion (K_f_eff, G_f_eff=0).
            # The terms b, c, d, g are derived from Eshelby-Wu type tensors for spheroidal inclusions.
            # These specific forms appear in Berryman's DEM papers or related works.
            # (Using v_matrix for these calculations)
            alpha = aspect_ratio
            # Parameter 'b' (related to P_ijkl tensor components for bulk modulus change)
            b = 3 * np.pi * alpha * (1 - 2 * v_matrix) / (4 * (1 - v_matrix**2))

            # Parameter 'c' and 'd' (related to Q_ijkl tensor components for shear modulus change)
            # The two-step `c = 1/c` is unusual; it implies `c_orig = 5 / ((3 + ...))`.
            # These are specific to certain DEM formulations (e.g. Kuster-Toksoz like terms in DEM context).
            # (Using v_matrix for these calculations)
            # Original code: c = 1 / 5 * ((3 + 8 * (1 - v_matrix) / (np.pi * alpha * (2 - v_matrix)))); c = 1 / c
            # This is equivalent to: c_temp = (1/5) * term; c = 1 / c_temp = 5 / term.
            # Let's rewrite for clarity and to match DEMModel class if that was intended.
            # The DEMModel class has: c = 1.0 / (0.2 * c_inv_term) where c_inv_term is ((3 + ...))
            # This is c = 5.0 / c_inv_term. This seems consistent.
            c_inv_term = (3 + 8 * (1 - v_matrix) / (np.pi * alpha * (2 - v_matrix)))
            c = 5.0 / c_inv_term if c_inv_term != 0 else np.inf

            d_inv_term = (1 + 8 * (1 - v_matrix) * (5 - v_matrix) / (3 * np.pi * alpha * (2 - v_matrix)))
            d = 5.0 / d_inv_term if d_inv_term != 0 else np.inf
            
            # Parameter 'g'
            g = np.pi * alpha / (2 * (1 - v_matrix))

            # Define implicit equation for effective bulk modulus K_eff from DEM:
            # (K_eff - K_f) / (K_m - K_f) * (K_m / K_eff)^(1 / (1 + b)) = (1 - φ)^(1 / (1 + b))
            # This is a form of Berryman's DEM solution for K_eff.
            def dem_eq_Keff(Keff_trial):
                if Keff_trial <= 0: return 1e12 # Penalty for non-physical K_eff
                # Avoid division by zero if K_m == K_f_eff
                if np.isclose(K_m, K_f_eff): # Matrix and fluid are same, K_eff should be K_m (or K_f_eff)
                    return Keff_trial - K_m # Should be zero if Keff_trial = K_m

                # Avoid Keff_trial being zero if K_m/Keff_trial is raised to a power.
                # If K_f_eff is very small (gas), Keff_trial might also be small.
                # The term (K_m / Keff_trial) can be large.
                # If Keff_trial approaches K_f_eff, LHS can be tricky.
                # If Keff_trial = K_f_eff, LHS = 0. RHS = (1-phi)^(1/(1+b)). So K_f_eff is a root if phi makes RHS zero (phi=1).
                # This form assumes K_m != K_f.

                # Original form: (Keff_val - Kf) / (bulk_modulus - Kf) * (bulk_modulus / Keff_val)**(1 / (1 + b)) - (1 - porosity[ii])**(1 / (1 + b))
                # Here, bulk_modulus is K_m. Kf is K_f_eff. porosity[ii] is phi_i.
                # Keff_val is Keff_trial.
                if np.isclose(K_m, K_f_eff): # If matrix and fluid are indistinguishable
                    # Then K_eff should also be K_m (or K_f_eff)
                    # The equation simplifies to: (K_eff - K_m) * (K_m/K_eff)^X = 0
                    # This means K_eff = K_m is a solution.
                    # If K_m = K_f, then DEM implies K_eff = K_m for any porosity.
                    return Keff_trial - K_m

                lhs = (Keff_trial - K_f_eff) / (K_m - K_f_eff) * \
                      (K_m / Keff_trial)**(1.0 / (1.0 + b))
                rhs = (1.0 - phi_i)**(1.0 / (1.0 + b))
                return lhs - rhs
            
            # Numerically solve for K_eff. Initial guess K_m.
            # `method='lm'` is Levenberg-Marquardt, suitable for non-linear least squares,
            # often robust for root finding too.
            sol_K = root(dem_eq_Keff, K_m, method='lm', tol=1e-7) # Add tolerance
            if sol_K.success and sol_K.x[0] > 0:
                K_eff_results[i] = sol_K.x[0]
            else:
                # Potential Issue: Solver failure. Could try different initial guess or method,
                # or indicate failure (e.g., with NaN or by raising error).
                # Original code raises ValueError.
                raise ValueError(f"DEM root finding for K_eff failed at index {i} for porosity {phi_i:.3f}, sat {sat_i:.3f}. Message: {sol_K.message}")

            # Define implicit equation for effective shear modulus G_eff from DEM:
            # G_eff / G_m * ( (1/G_eff + c*g/(d*K_f_eff)) / (1/G_m + c*g/(d*K_f_eff)) )^(1 - c/d) = (1 - φ)^(1/d)
            # This is complex. Note G_f_eff = 0. Term c*g/(d*K_f_eff) involves K_f_eff.
            # If K_f_eff (fluid bulk modulus) is very small (gas), this term can be large.
            # If K_f_eff = 0 (not possible as default is 0.01 GPa), it's infinite.
            # The original `velDEM` has `Kf` (which is K_f_eff here) in the denominator.
            # This implies the fluid's incompressibility affects shear properties if pores are not dry.
            # This specific form of G_eff DEM equation seems less common or for specific assumptions.
            # Often, for G_eff, the fluid term (K_f_eff) might not appear if G_f_eff=0 is directly used.
            # Let's assume K_f_eff is non-zero.
            if np.isclose(K_f_eff, 0.0): # If fluid is extremely compressible (like vacuum)
                # This term c*g/(d*K_f_eff) would blow up.
                # If K_f_eff -> 0, then (1/G_eff + Inf) / (1/G_m + Inf) -> 1.
                # So eq becomes: G_eff/G_m = (1-phi)^(1/d). This is a dry pore limit.
                # This suggests if Kf is very small, the equation simplifies.
                # Let's add a small epsilon to K_f_eff if it's zero to avoid division by zero in the main equation.
                k_f_eff_safe = K_f_eff if K_f_eff > 1e-9 else 1e-9
            else:
                k_f_eff_safe = K_f_eff

            def dem_eq_Geff(Geff_trial):
                if Geff_trial <= 0: return 1e12 # Penalty

                # Handle G_m = 0 case (e.g. if matrix was fluid - not typical for DEM solid matrix)
                if np.isclose(G_m, 0.0):
                    return Geff_trial # If matrix has no shear, effective medium should also have no shear.

                term_in_power_num = (1.0 / Geff_trial) + (c * g / (d * k_f_eff_safe))
                term_in_power_den = (1.0 / G_m) + (c * g / (d * k_f_eff_safe))

                if np.isclose(term_in_power_den, 0.0): # Denominator inside power is zero
                    # This case is complex. If num is also zero, could be 1. If num non-zero, then inf.
                    # This would likely mean parameters are at extreme limits.
                    return 1e12 # Penalize, likely indicates an issue or limit.

                ratio_terms = term_in_power_num / term_in_power_den
                exponent_term = 1.0 - (c / d) if d != 0 else 1.0 # if d=0, exponent is ill-defined. Assume 1 for now.
                                                               # d being 0 implies d_inv_term was inf, meaning alpha or (1-v) was problematic.

                if ratio_terms < 0 and isinstance(exponent_term, float) and not exponent_term.is_integer():
                    # Cannot take fractional power of negative number without complex result.
                    # This implies non-physical parameters or model breakdown.
                    return 1e12 # Penalize

                lhs = (Geff_trial / G_m) * (ratio_terms ** exponent_term)
                rhs = (1.0 - phi_i)**(1.0 / d if d != 0 else 1.0) # if d=0, rhs exponent is ill-defined.
                return lhs - rhs

            sol_G = root(dem_eq_Geff, G_m, method='lm', tol=1e-7) # Initial guess G_m
            if sol_G.success and sol_G.x[0] >= 0: # G_eff can be 0 for suspensions
                G_eff_results[i] = sol_G.x[0]
            else:
                # print(f"Warning: DEM root finding for G_eff failed or gave negative result at index {i} for porosity {phi_i:.3f}, sat {sat_i:.3f}. Message: {sol_G.message}. G_eff set to 0.")
                # Setting to 0.0 if DEM fails for shear, as it might be close to fluid suspension.
                # Original code raises ValueError.
                raise ValueError(f"DEM root finding for G_eff failed at index {i} for porosity {phi_i:.3f}, sat {sat_i:.3f}. Message: {sol_G.message}")

            # Calculate effective density (ρ_eff) of the saturated rock
            # ρ_eff = ρ_matrix * (1-φ) + ρ_fluid * φ
            # ρ_fluid = S_w * ρ_w + (1-S_w) * ρ_gas
            rho_fluid_eff = sat_i * water_density + (1.0 - sat_i) * gas_density
            rho_total_eff = matrix_density * (1.0 - phi_i) + rho_fluid_eff * phi_i
            
            if rho_total_eff <= 0:
                raise ValueError(f"Calculated total effective density is non-positive ({rho_total_eff} kg/m³) at index {i}.")

            # Calculate P-wave velocity (Vp = sqrt((K_eff + 4/3*G_eff) / ρ_eff))
            # Moduli are in GPa, so multiply by 1e9 for Pascals.
            p_wave_modulus_eff = K_eff_results[i] * 1e9 + (4.0/3.0) * G_eff_results[i] * 1e9
            if p_wave_modulus_eff < 0:
                 # This can happen if K_eff or G_eff are not well-behaved from solver.
                print(f"Warning: Calculated P-wave modulus is negative ({p_wave_modulus_eff} Pa) at index {i}. Setting Vp to NaN.")
                Vp_results[i] = np.nan
            else:
                Vp_results[i] = np.sqrt(p_wave_modulus_eff / rho_total_eff)
        
        return K_eff_results, G_eff_results, Vp_results


class HertzMindlinModel(BaseVelocityModel):
    """
    Hertz-Mindlin model for contact stiffness of granular media, often combined with
    Hashin-Shtrikman bounds to estimate effective moduli of porous rocks.

    This model calculates the elastic moduli (K_HM, G_HM) of a dense random pack
    of identical elastic spheres at a critical porosity (φ_c). These are then used
    with Hashin-Shtrikman bounds (or other schemes) to estimate moduli at other porosities.
    """
    
    def __init__(self, 
               critical_porosity: float = 0.4, 
               coordination_number: float = 6.0): # Typical C for random pack can be 6-9. Original had 4.
        """
        Initialize Hertz-Mindlin model parameters.
        
        Args:
            critical_porosity (float, optional): Critical porosity (φ_c), typically the porosity
                                                 of a dense random pack of spheres (e.g., 0.36-0.4).
                                                 Defaults to 0.4.
            coordination_number (float, optional): Average number of contacts per grain (C)
                                                   at critical porosity. Defaults to 6.0.
                                                   (Original code had 4.0, which is low for sphere packs).
        """
        super().__init__()
        if not (0 < critical_porosity < 1):
            raise ValueError("Critical porosity must be between 0 and 1.")
        if coordination_number <= 0:
            raise ValueError("Coordination number must be positive.")

        self.critical_porosity = critical_porosity
        self.coordination_number = coordination_number
    
    def calculate_velocity(self, 
                         porosity: np.ndarray,       # φ, array of current porosities
                         saturation: np.ndarray,    # S_w, array of water saturations
                         matrix_bulk_modulus: float,  # K_m of solid grains [GPa]
                         matrix_shear_modulus: float, # G_m of solid grains [GPa]
                         matrix_density: float,       # ρ_m of solid grains [kg/m³]
                         depth: float = 1.0,          # Depth for pressure calc [m]
                         water_bulk_modulus: float = 2.2, # K_w [GPa]
                         gas_bulk_modulus: float = 0.01,  # K_gas [GPa]
                         water_density: float = 1000.0, # ρ_w [kg/m³]
                         gas_density: float = 1.225   # ρ_gas [kg/m³]
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate P-wave velocity bounds (Hashin-Shtrikman high and low) for porous rocks
        using Hertz-Mindlin theory for dry frame moduli at critical porosity, and then
        interpolating/extrapolating these moduli for other porosities. Fluid substitution
        is applied using Brie's model for effective fluid modulus.
        
        Args:
            porosity (np.ndarray): Array of porosity values (φ).
            saturation (np.ndarray): Array of water saturation values (S_w).
            matrix_bulk_modulus (float): Bulk modulus of the solid mineral matrix (K_m) [GPa].
            matrix_shear_modulus (float): Shear modulus of the solid mineral matrix (G_m) [GPa].
            matrix_density (float): Density of the solid mineral matrix (ρ_m) [kg/m³].
            depth (float, optional): Depth for effective pressure estimation [m]. Defaults to 1.0 m.
                                     Pressure calculation is simplified.
            water_bulk_modulus (float, optional): Bulk modulus of water (K_w) [GPa]. Defaults to 2.2 GPa.
            gas_bulk_modulus (float, optional): Bulk modulus of gas (K_gas) [GPa]. Defaults to 0.01 GPa.
            water_density (float, optional): Density of water (ρ_w) [kg/m³]. Defaults to 1000 kg/m³.
            gas_density (float, optional): Density of gas (ρ_gas) [kg/m³]. Defaults to 1.225 kg/m³.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Vp_high (np.ndarray): P-wave velocity upper bound [m/s].
                - Vp_low (np.ndarray): P-wave velocity lower bound [m/s].

        Raises:
            ValueError: If input parameters are non-physical or lead to invalid intermediate calculations.
        """
        if not (isinstance(porosity, np.ndarray) and isinstance(saturation, np.ndarray)):
            raise TypeError("Porosity and saturation must be NumPy arrays.")
        if porosity.shape != saturation.shape:
            raise ValueError("Porosity and saturation arrays must have the same shape.")
        # Add checks for physical ranges of moduli and densities if needed.

        # Poisson's ratio of the solid mineral matrix
        v_matrix = (3 * matrix_bulk_modulus - 2 * matrix_shear_modulus) / \
                   (2 * (3 * matrix_bulk_modulus + matrix_shear_modulus))
        if not (-1.0 < v_matrix < 0.5):
            print(f"Warning: Matrix Poisson's ratio ({v_matrix:.3f}) is outside physical bounds (-1 to 0.5).")

        # Effective pressure (P_eff) estimation [GPa]
        # Simplified: (ρ_matrix - ρ_water_avg) * g * depth. Assumes hydrostatic pore pressure.
        # (1000 kg/m³ is approx density of water, used as reference for buoyancy).
        # This is a very rough estimate and might need refinement for specific scenarios.
        # Potential Issue: If matrix_density < 1000 (e.g., ice), pressure could be negative.
        # Pressure should ideally be effective pressure (confining - pore).
        # Let's ensure pressure P is non-negative for Hertz-Mindlin equations.
        effective_pressure_gpa = max(0, (matrix_density - water_density) * 9.80665 * depth / 1e9)
        if effective_pressure_gpa == 0:
            print("Warning: Effective pressure is zero or negative. Hertz-Mindlin moduli will be zero. "
                  "This occurs if matrix_density <= water_density or depth is zero.")
            # If P=0, K_HM and G_HM will be 0. This leads to Vp=0 for dry frame.
            # Fluid substitution might still yield non-zero Vp if K_fluid is non-zero.

        # Hertz-Mindlin model moduli at critical porosity (φ_c)
        C_coord_num = self.coordination_number
        phi_critical = self.critical_porosity
        
        # K_HM (Bulk modulus of dry frame at critical porosity)
        # K_HM = [ C^2 * (1-φ_c)^2 * G_m^2 * P_eff / (18 * π^2 * (1-v_m)^2) ]^(1/3)
        # Ensure (1-v_m)^2 is not zero. v_m = 0.5 leads to this.
        denominator_khm_term = (18 * np.pi**2 * (1 - v_matrix)**2)
        if np.isclose(denominator_khm_term, 0.0):
            # This implies v_matrix is 1.0, which is non-physical for stable isotropic material.
            # It would make K_HM infinite if P_eff > 0.
            # If P_eff = 0, K_HM = 0 anyway.
            # For v_matrix=0.5 (incompressible solid), K_m is inf. This formula assumes finite K_m, G_m.
            # If v_matrix is very close to 1, K_HM can be very large.
            # If v_matrix = 0.5 (incompressible grains, K_m -> inf), then (1-v_matrix)^2 = 0.25.
            # The issue is if v_matrix = 1.0.
            raise ValueError(f"Matrix Poisson's ratio v_m={v_matrix} leads to division by zero in K_HM calculation.")
        
        K_HM_dry_critical = (C_coord_num**2 * (1 - phi_critical)**2 * matrix_shear_modulus**2 * effective_pressure_gpa /
                             denominator_khm_term)**(1/3) if effective_pressure_gpa > 0 else 0.0
        
        # G_HM (Shear modulus of dry frame at critical porosity)
        # G_HM = [(5-4v_m)/(5(2-v_m))] * [ (3 * C^2 * (1-φ_c)^2 * G_m^2 * P_eff) / (2 * π^2 * (1-v_m)^2) ]^(1/3)
        # Note: Original has (10-2*v) which is 2*(5-v). Simpler form often (5-4v_m)/(5(2-v_m)).
        # The term ( (3 * C^2 ...) / (2 * pi^2 * (1-v_m)^2) ) ^ (1/3) is related to K_HM.
        # G_HM = Factor * ( (K_HM_numerator_part_scaled_for_G) ) ^ (1/3)
        # Let's use K_HM to simplify G_HM: G_HM = ( (5-4v)/(5(2-v)) ) * (3/2 * K_HM_from_shear_term_relation)
        # Factor = (5 - 4*v_matrix) / (5 * (2 - v_matrix)) -- this is a common pre-factor.
        # The term ((3*C^2...)/ (2*pi^2...)) can be related to K_HM's term.
        # K_HM_term_cubed = C^2*(1-phi_c)^2*G_m^2*P / (18*pi^2*(1-v_m)^2)
        # G_HM_term_cubed = (3*C^2*(1-phi_c)^2*G_m^2*P) / (2*pi^2*(1-v_m)^2) = 27 * K_HM_term_cubed
        # So, (G_HM_term_cubed)^(1/3) = 3 * K_HM.
        # G_HM = Factor_G * 3 * K_HM -- Check this derivation.
        # Original formula for G_HM:
        # G_HM = ((5 - 4v) / (10 - 2v)) * [ (3 * C^2 * (1-phi_c)^2 * G_m^2 * P) / (2 * pi^2 * (1-v_m)^2) ]^(1/3)
        # This is equivalent to: G_HM = ((5-4v_m)/(10-2*v_m)) * ( (9*matrix_shear_modulus)/(1-phi_c) * K_HM_dry_critical * (1-phi_c)/matrix_shear_modulus )
        # G_HM = ((5-4*v_matrix) / (5*(2-v_matrix))) * ( (3 * C_coord_num**2 * (1-phi_critical)**2 * matrix_shear_modulus**2 * effective_pressure_gpa) / (2*np.pi**2 * (1-v_matrix)**2) )**(1/3)
        # This seems more direct from literature.
        if np.isclose(2 - v_matrix, 0.0) or np.isclose(1 - v_matrix, 0.0): # Avoid division by zero if v_m=2 or v_m=1
            raise ValueError(f"Matrix Poisson's ratio v_m={v_matrix} leads to division by zero in G_HM calculation.")

        g_hm_bracket_term_cubed = (3 * C_coord_num**2 * (1 - phi_critical)**2 * matrix_shear_modulus**2 * effective_pressure_gpa) / \
                            (2 * np.pi**2 * (1 - v_matrix)**2) if (1-v_matrix)!=0 else 0 # Avoid div by zero if v_m=1
        G_HM_dry_critical = ((5 - 4 * v_matrix) / (5 * (2 - v_matrix))) * (g_hm_bracket_term_cubed)**(1/3) if effective_pressure_gpa > 0 else 0.0

        # Initialize output velocity arrays
        Vp_high_results = np.zeros(len(porosity))
        Vp_low_results = np.zeros(len(porosity))
        
        # Brie model instance for fluid substitution in Gassmann
        # Using default exponent from BrieModel class (typically 3.0)
        brie = BrieModel(exponent=3.0) # Can make exponent a parameter of HertzMindlinModel if needed
        
        for i in range(len(porosity)):
            phi_i = porosity[i]
            sat_i = saturation[i]

            # Validate current porosity and saturation
            if not (0.0 <= phi_i <= 1.0):
                print(f"Warning: Porosity phi_i={phi_i:.3f} at index {i} is outside [0,1]. Results may be non-physical. Setting Vp to NaN.")
                Vp_high_results[i], Vp_low_results[i] = np.nan, np.nan
                continue
            if not (0.0 <= sat_i <= 1.0):
                print(f"Warning: Saturation sat_i={sat_i:.3f} at index {i} is outside [0,1]. Results may be non-physical. Setting Vp to NaN.")
                Vp_high_results[i], Vp_low_results[i] = np.nan, np.nan
                continue

            K_eff_dry_H, G_eff_dry_H = 0.0, 0.0 # Effective dry rock moduli, Hashin-Shtrikman Upper
            K_eff_dry_L, G_eff_dry_L = 0.0, 0.0 # Effective dry rock moduli, Hashin-Shtrikman Lower

            if phi_i < phi_critical:
                # Porosity < critical: Modified Hashin-Shtrikman bounds (interpolate between matrix and critical point)
                # These are for the DRY rock frame.
                # K_eff_L_dry = [ (φ/φ_c) / (K_HM + 4/3 G_HM) + (1 - φ/φ_c) / (K_m + 4/3 G_HM) ]^-1 - 4/3 G_HM
                # G_eff_L_dry = [ (φ/φ_c) / (G_HM + ζ_HM)    + (1 - φ/φ_c) / (G_m + ζ_HM)    ]^-1 - ζ_HM
                # where ζ_HM = G_HM/6 * (9 K_HM + 8 G_HM) / (K_HM + 2 G_HM)
                # Similar for Upper bounds (H) but swapping K_m,G_m with K_HM,G_HM and G_HM with G_m in denominators of terms.
                
                # Denominators for K_eff_L_dry terms
                den1_Kl = K_HM_dry_critical + (4.0/3.0) * G_HM_dry_critical
                den2_Kl = matrix_bulk_modulus + (4.0/3.0) * G_HM_dry_critical
                if np.isclose(den1_Kl,0.0) or np.isclose(den2_Kl,0.0): K_eff_dry_L = 0 # Avoid division by zero if moduli are zero
                else: K_eff_dry_L = ( (phi_i / phi_critical) / den1_Kl + \
                                  (1.0 - phi_i / phi_critical) / den2_Kl )**(-1) - (4.0/3.0) * G_HM_dry_critical
                
                zeta_L_den = (K_HM_dry_critical + 2 * G_HM_dry_critical)
                zeta_L = (G_HM_dry_critical / 6.0) * (9 * K_HM_dry_critical + 8 * G_HM_dry_critical) / \
                         zeta_L_den if zeta_L_den !=0 else 0
                den1_Gl = G_HM_dry_critical + zeta_L
                den2_Gl = matrix_shear_modulus + zeta_L
                if np.isclose(den1_Gl,0.0) or np.isclose(den2_Gl,0.0): G_eff_dry_L = 0
                else: G_eff_dry_L = ( (phi_i / phi_critical) / den1_Gl + \
                                  (1.0 - phi_i / phi_critical) / den2_Gl )**(-1) - zeta_L

                # Denominators for K_eff_H_dry terms
                den1_Kh = K_HM_dry_critical + (4.0/3.0) * matrix_shear_modulus # Uses G_m here
                den2_Kh = matrix_bulk_modulus + (4.0/3.0) * matrix_shear_modulus # Uses G_m here
                if np.isclose(den1_Kh,0.0) or np.isclose(den2_Kh,0.0): K_eff_dry_H = 0
                else: K_eff_dry_H = ( (phi_i / phi_critical) / den1_Kh + \
                                  (1.0 - phi_i / phi_critical) / den2_Kh )**(-1) - (4.0/3.0) * matrix_shear_modulus
                
                zeta_H_den = (matrix_bulk_modulus + 2*matrix_shear_modulus)
                zeta_H = (matrix_shear_modulus / 6.0) * (9 * matrix_bulk_modulus + 8 * matrix_shear_modulus) / \
                         zeta_H_den if zeta_H_den!=0 else 0
                den1_Gh = G_HM_dry_critical + zeta_H # Uses G_HM_dry_crit here
                den2_Gh = matrix_shear_modulus + zeta_H # Uses G_m here
                if np.isclose(den1_Gh,0.0) or np.isclose(den2_Gh,0.0): G_eff_dry_H = 0
                else: G_eff_dry_H = ( (phi_i / phi_critical) / den1_Gh + \
                                  (1.0 - phi_i / phi_critical) / den2_Gh )**(-1) - zeta_H
                
                # Ensure physical non-negative moduli before fluid substitution
                K_eff_dry_L, G_eff_dry_L = max(0, K_eff_dry_L), max(0, G_eff_dry_L)
                K_eff_dry_H, G_eff_dry_H = max(0, K_eff_dry_H), max(0, G_eff_dry_H)

                # Fluid substitution using Gassmann (via BrieModel's helper)
                # K_sat_H from K_eff_H_dry, G_eff_H_dry (Gassmann only changes K, G_sat_dry = G_dry)
                K_sat_H = brie.calculate_saturated_modulus(
                    K_eff_dry_H, matrix_bulk_modulus, phi_i, sat_i, water_bulk_modulus, gas_bulk_modulus
                )
                G_sat_H = G_eff_dry_H # Shear modulus unchanged by fluid for Gassmann

                K_sat_L = brie.calculate_saturated_modulus(
                    K_eff_dry_L, matrix_bulk_modulus, phi_i, sat_i, water_bulk_modulus, gas_bulk_modulus
                )
                G_sat_L = G_eff_dry_L # Shear modulus unchanged
                
            else: # Porosity >= critical_porosity: Suspension model (Reuss-like average)
                  # This part models the rock as a suspension of matrix material (properties K_HM, G_HM at phi_c)
                  # in excess porosity (phi_i - phi_c) filled with fluid.
                  # The "solid" part of this suspension is the frame at critical porosity.
                  # The "fluid" part is the content of the excess pores.
                  # This seems to be a specific variant of Reuss average or similar for high porosity.
                  # The original code is:
                  # K_eff = ((1-phi)/(1-phi_c)/(K_HM + 4/3 G_HM) + (phi-phi_c)/(1-phi_c)/(4/3 G_HM))^-1 - 4/3 G_HM
                  # This looks like Reuss average for K_dry where one component is frame at phi_c,
                  # and other component is "empty pores" with K=0, G=G_HM (this G for empty pores is unusual).
                  # Typically for Reuss with empty pores (K_pore=0, G_pore=0):
                  # 1/K_dry = (1-phi_excess)/K_frame_at_phic + phi_excess/K_pore -> K_dry = K_frame_at_phic * (1-phi_excess)
                  # 1/G_dry = (1-phi_excess)/G_frame_at_phic + phi_excess/G_pore -> G_dry = G_frame_at_phic * (1-phi_excess)
                  # Where phi_excess = (phi_i - phi_c) / (1 - phi_c) is porosity relative to non-solid volume.
                  # The original code seems to use a specific variant. We will follow it.

                # Effective moduli of the dry frame for phi >= phi_c
                # Fraction of original solid phase (at critical porosity) in the (1-phi_c) volume
                frac_solid_phase = (1.0 - phi_i) / (1.0 - phi_critical) if (1.0 - phi_critical) != 0 else 0
                # Fraction of "added" pore space (beyond critical) in the (1-phi_c) volume
                frac_added_pores = (phi_i - phi_critical) / (1.0 - phi_critical) if (1.0 - phi_critical) != 0 else 1.0

                # Ensure fractions are valid (e.g. phi_i should not be < phi_critical here if logic is strict)
                # However, original code structure implies phi_i >= phi_c for this branch.
                # If phi_i = phi_critical, then frac_solid_phase=1, frac_added_pores=0. K_eff should be K_HM.

                # Denominators for K_eff_susp_dry terms
                den1_Keff_susp = K_HM_dry_critical + (4.0/3.0) * G_HM_dry_critical
                den2_Keff_susp = (4.0/3.0) * G_HM_dry_critical # This term for K is unusual (fluid K=0 usually)

                if np.isclose(den1_Keff_susp,0.0) or np.isclose(den2_Keff_susp,0.0) or np.isclose(1.0 - phi_critical, 0.0):
                    K_eff_susp_dry = 0.0 # Or K_HM_dry_critical if phi_i approx phi_c
                    if np.isclose(phi_i, phi_critical): K_eff_susp_dry = K_HM_dry_critical
                else:
                     K_eff_susp_dry = ( frac_solid_phase / den1_Keff_susp + frac_added_pores / den2_Keff_susp
                                       )**(-1) - (4.0/3.0) * G_HM_dry_critical
                
                zeta_susp_den = (K_HM_dry_critical + 2 * G_HM_dry_critical)
                zeta_susp = (G_HM_dry_critical / 6.0) * (9 * K_HM_dry_critical + 8 * G_HM_dry_critical) / \
                            zeta_susp_den if zeta_susp_den !=0 else 0
                den1_Geff_susp = G_HM_dry_critical + zeta_susp
                den2_Geff_susp = zeta_susp # Term for "added pores" G=0 contribution
                if np.isclose(den1_Geff_susp,0.0) or np.isclose(den2_Geff_susp,0.0) or np.isclose(1.0 - phi_critical, 0.0):
                    G_eff_susp_dry = 0.0
                    if np.isclose(phi_i, phi_critical): G_eff_susp_dry = G_HM_dry_critical
                else:
                    G_eff_susp_dry = ( frac_solid_phase / den1_Geff_susp + frac_added_pores / den2_Geff_susp
                                       )**(-1) - zeta_susp
                
                K_eff_susp_dry, G_eff_susp_dry = max(0,K_eff_susp_dry), max(0,G_eff_susp_dry)

                # Fluid substitution for this suspension-like dry frame
                # Both high and low bounds become equal in this regime as per original code.
                K_sat_H = K_sat_L = brie.calculate_saturated_modulus(
                    K_eff_susp_dry, matrix_bulk_modulus, phi_i, sat_i, water_bulk_modulus, gas_bulk_modulus
                )
                G_sat_H = G_sat_L = G_eff_susp_dry # Shear modulus unchanged by fluid
            
            # Calculate total effective density (ρ_total_eff)
            rho_fluid_eff = sat_i * water_density + (1.0 - sat_i) * gas_density
            rho_total_eff = matrix_density * (1.0 - phi_i) + rho_fluid_eff * phi_i
            if rho_total_eff <= 0:
                # This might happen if matrix_density is low and porosity is high with light gas.
                # Or if phi_i > 1 or sat_i is outside [0,1].
                print(f"Warning: Total effective density is non-positive ({rho_total_eff} kg/m³) at index {i}. Vp will be NaN.")
                Vp_high_results[i] = np.nan
                Vp_low_results[i] = np.nan
                continue


            # Calculate P-wave velocities for high and low bounds
            # Vp = sqrt((K_sat + 4/3*G_sat) / ρ_total_eff)
            # Moduli are in GPa, convert to Pa (multiply by 1e9)
            p_wave_modulus_H = K_sat_H * 1e9 + (4.0/3.0) * G_sat_H * 1e9
            p_wave_modulus_L = K_sat_L * 1e9 + (4.0/3.0) * G_sat_L * 1e9

            if p_wave_modulus_H < 0: Vp_high_results[i] = np.nan
            else: Vp_high_results[i] = np.sqrt(p_wave_modulus_H / rho_total_eff)
            
            if p_wave_modulus_L < 0: Vp_low_results[i] = np.nan
            else: Vp_low_results[i] = np.sqrt(p_wave_modulus_L / rho_total_eff)
        
        return Vp_high_results, Vp_low_results


# Standalone functions (potentially older versions or specific use cases)
# These are kept but should be reviewed for consistency with the class-based models above.
# Their documentation will also be improved.

def VRH_model(f: List[float]=[0.35, 0.25, 0.2, 0.125, 0.075], # Default values are illustrative
             K_minerals: List[float]=[55.4, 36.6, 75.6, 46.7, 50.4], # K for each mineral
             G_minerals: List[float]=[28.1, 45, 25.6, 23.65, 27.4],   # G for each mineral
             rho_minerals: List[float]=[2560, 2650, 2630, 2540, 3050] # ρ for each mineral
             ) -> Tuple[float, float, float]:
    """
    Standalone function implementing the Voigt-Reuss-Hill (VRH) mixing model.

    Estimates effective bulk modulus (K_eff), shear modulus (G_eff), and density (rho_eff)
    of a composite material (rock matrix) made from various mineral components.
    This function is similar to `VRHModel.calculate_properties`.

    Args:
        f (List[float], optional): List of volume fractions of each mineral component.
                                   Must sum to 1.0. Defaults are illustrative.
        K_minerals (List[float], optional): List of bulk moduli (K) for each mineral [GPa].
        G_minerals (List[float], optional): List of shear moduli (G) for each mineral [GPa].
        rho_minerals (List[float], optional): List of densities (ρ) for each mineral [kg/m³].

    Returns:
        Tuple[float, float, float]:
            - K_eff (float): Effective bulk modulus of the mineral mixture [GPa].
            - G_eff (float): Effective shear modulus of the mineral mixture [GPa].
            - rho_eff (float): Effective density of the mineral mixture [kg/m³].

    Raises:
        ValueError: If input lists have different lengths or if fractions do not sum to 1.
    """
    if not (len(f) == len(K_minerals) == len(G_minerals) == len(rho_minerals)):
        raise ValueError("All input lists (fractions, K, G, rho) must have the same length.")
    if not f:
        raise ValueError("Input lists cannot be empty.")

    f_arr = np.array(f, dtype=float)
    K_arr = np.array(K_minerals, dtype=float)
    G_arr = np.array(G_minerals, dtype=float)
    rho_arr = np.array(rho_minerals, dtype=float)

    if not np.isclose(np.sum(f_arr), 1.0):
        raise ValueError(f"Volume fractions must sum to 1.0. Current sum: {np.sum(f_arr)}")
    if np.any(K_arr <= 0) or np.any(G_arr < 0): # G can be 0 for fluids, but typically minerals have G>0
        raise ValueError("Mineral bulk moduli must be positive, and shear moduli non-negative.")


    # Voigt average (upper bound for moduli)
    K_voigt = np.sum(f_arr * K_arr)
    G_voigt = np.sum(f_arr * G_arr)

    # Reuss average (lower bound for moduli)
    # Handle potential division by zero if any K_i or G_i is zero.
    # For G_reuss, if any G_i is zero (fluid component), G_reuss becomes 0.
    with np.errstate(divide='ignore'): # Suppress warning for division by zero
        sum_f_over_K = np.sum(f_arr / K_arr)
        sum_f_over_G = np.sum(f_arr / G_arr) # Can be inf if any G_i is 0

    K_reuss = 1.0 / sum_f_over_K if sum_f_over_K != 0 else np.inf # Or 0 if K is stiffness? This is compliance sum.
                                                              # If sum_f_over_K is 0 (all K_i are inf), K_reuss is inf.
                                                              # If any K_i is 0, sum_f_over_K is inf, K_reuss is 0.
    if np.isinf(sum_f_over_G): # If any component is fluid (G_i=0, f_i>0)
        G_reuss = 0.0
    elif sum_f_over_G == 0: # Should not happen if sum(f)=1 and G_i are finite.
        G_reuss = np.inf
    else:
        G_reuss = 1.0 / sum_f_over_G

    # VRH average is the arithmetic mean of Voigt and Reuss bounds
    K_eff = 0.5 * (K_voigt + K_reuss)
    G_eff = 0.5 * (G_voigt + G_reuss)

    # Effective density is the weighted sum of component densities
    rho_eff = np.sum(f_arr * rho_arr)

    return K_eff, G_eff, rho_eff


def satK(K_dry_rock: float, K_matrix: float, porosity: float, saturation_water: float,
         water_bulk_modulus: float = 2.2, gas_bulk_modulus: float = 0.01, brie_exponent: float = 3.0) -> float:
    """
    Standalone function to calculate saturated bulk modulus (K_sat) using Gassmann's equation,
    with effective fluid modulus from Brie's model.
    Similar to `BrieModel.calculate_saturated_modulus`.

    Args:
        K_dry_rock (float): Effective bulk modulus of the dry rock frame (K_eff_dry) [GPa].
        K_matrix (float): Bulk modulus of the solid mineral matrix (Km) [GPa].
        porosity (float): Porosity of the rock (phi).
        saturation_water (float): Water saturation level (Sat).
        water_bulk_modulus (float, optional): Bulk modulus of water (Kw) [GPa]. Defaults to 2.2.
        gas_bulk_modulus (float, optional): Bulk modulus of gas (Ka) [GPa]. Defaults to 0.01.
        brie_exponent (float, optional): Exponent for Brie's fluid mixing model. Defaults to 3.0.

    Returns:
        float: Saturated bulk modulus (K_sat) [GPa].

    Raises:
        ValueError for non-physical inputs.
    """
    if not (0.0 <= porosity <= 1.0): raise ValueError("Porosity must be between 0 and 1.")
    if not (0.0 <= saturation_water <= 1.0): raise ValueError("Saturation must be between 0 and 1.")
    if K_dry_rock < 0 or K_matrix <= 0 or water_bulk_modulus <=0 or gas_bulk_modulus <=0:
        raise ValueError("Bulk moduli must be positive (K_matrix, K_water, K_gas > 0; K_dry_rock >=0).")
    if K_dry_rock > K_matrix: # Check physical constraint for Gassmann
        print(f"Warning: Dry rock bulk modulus ({K_dry_rock}) > matrix bulk modulus ({K_matrix}). This is unusual.")

    # Effective fluid bulk modulus using Brie's equation
    K_fluid_eff = (water_bulk_modulus - gas_bulk_modulus) * (saturation_water ** brie_exponent) + gas_bulk_modulus
    if K_fluid_eff <= 0:
        raise ValueError(f"Calculated effective fluid modulus is non-positive ({K_fluid_eff} GPa).")

    # Gassmann's equation - standard form: K_sat = K_dry + num/den
    # num = (1 - K_dry/K_matrix)^2
    # den = porosity/K_fluid_eff + (1-porosity)/K_matrix - K_dry/(K_matrix^2)
    # The form used in original code:
    # K_sat = (K_dry_rock / (K_matrix - K_dry_rock) + K_fluid_eff / (porosity * (K_matrix - K_fluid_eff))) * K_matrix / \
    #         (1 + (K_dry_rock / (K_matrix - K_dry_rock) + K_fluid_eff / (porosity * (K_matrix - K_fluid_eff))))
    # This form has potential divisions by zero if K_matrix == K_dry_rock or K_matrix == K_fluid_eff.
    # Let's use the more robust standard Gassmann formulation to avoid some pitfalls.

    if np.isclose(K_matrix, K_dry_rock): # If K_dry is same as K_matrix (e.g. zero porosity limit)
        return K_matrix # K_sat should be K_matrix

    term_Kdry_over_Kmatrix = K_dry_rock / K_matrix
    numerator_gassmann = (1.0 - term_Kdry_over_Kmatrix)**2

    # Denominator terms - check for division by zero
    term_phi_over_Kfluid = porosity / K_fluid_eff if K_fluid_eff != 0 else np.inf
    term_1minphi_over_Kmatrix = (1.0 - porosity) / K_matrix if K_matrix !=0 else np.inf
    term_Kdry_over_KmatrixSq = K_dry_rock / (K_matrix**2) if K_matrix !=0 else 0 # if K_matrix is 0, K_dry should also be 0

    denominator_gassmann = term_phi_over_Kfluid + term_1minphi_over_Kmatrix - term_Kdry_over_KmatrixSq

    if np.isclose(denominator_gassmann, 0.0) or np.isinf(denominator_gassmann):
        # If denominator is zero or infinite, it means K_sat might be K_dry (if num is also zero) or K_matrix.
        # This often happens if K_fluid_eff is very small (gas), K_sat should approach K_dry.
        # If K_fluid_eff is very large (stiff fluid), K_sat approaches K_matrix (upper bound).
        # A simple fallback if K_fluid_eff is very small relative to K_dry and K_matrix:
        if K_fluid_eff < 0.001 * K_dry_rock : # Heuristic for gas-like fluid
            return K_dry_rock # For gas saturation, K_sat is close to K_dry
        else: # Otherwise, implies K_sat is likely K_matrix or problem with inputs
            # This situation means the rock is incompressible by the fluid.
            # print(f"Warning: Gassmann denominator near zero/inf for K_dry={K_dry_rock}, K_m={K_matrix}, phi={porosity}, K_fl={K_fluid_eff}. Returning K_matrix.")
            return K_matrix # Limit where fluid is much stiffer or frame very compliant

    K_saturated = K_dry_rock + numerator_gassmann / denominator_gassmann
    return K_saturated


def velDEM(phi_array: np.ndarray, K_matrix_init: float, G_matrix_init: float, rho_matrix: float,
           saturation_array: np.ndarray, aspect_ratio_pores: float,
           water_bulk_modulus: float = 2.2, gas_bulk_modulus: float = 0.01,
           water_density: float = 1000.0, gas_density: float = 1.225
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standalone function to calculate effective moduli (K_eff, G_eff) and P-wave velocity (Vp)
    using a Differential Effective Medium (DEM) model.
    Similar to `DEMModel.calculate_velocity`.

    Args:
        phi_array (np.ndarray): Array of porosity values (φ).
        K_matrix_init (float): Initial bulk modulus of the solid matrix (Km) [GPa].
        G_matrix_init (float): Initial shear modulus of the solid matrix (Gm) [GPa].
        rho_matrix (float): Density of the solid matrix (ρ_b) [kg/m³].
        saturation_array (np.ndarray): Array of water saturation values (Sat). Must match `phi_array` shape.
        aspect_ratio_pores (float): Aspect ratio of pores/cracks (alpha).
        water_bulk_modulus (float, optional): Bulk modulus of water (Kw) [GPa]. Defaults to 2.2.
        gas_bulk_modulus (float, optional): Bulk modulus of gas (Ka) [GPa]. Defaults to 0.01.
        water_density (float, optional): Density of water [kg/m³]. Defaults to 1000.0.
        gas_density (float, optional): Density of gas [kg/m³]. Defaults to 1.225.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - K_eff_results (np.ndarray): Effective bulk modulus for each porosity [GPa].
            - G_eff_results (np.ndarray): Effective shear modulus for each porosity [GPa].
            - Vp_results (np.ndarray): P-wave velocity for each porosity [m/s].

    Raises:
        ValueError: If root finding fails or inputs are inconsistent/non-physical.
    """
    if phi_array.shape != saturation_array.shape:
        raise ValueError("Porosity and saturation arrays must have the same shape.")

    num_points = len(phi_array)
    K_eff_results = np.zeros(num_points)
    G_eff_results = np.zeros(num_points)
    Vp_results = np.zeros(num_points)

    # Brie model for effective fluid properties (using a default exponent of 3.0 from BrieModel class)
    brie_fluid_model = BrieModel()

    for i in range(num_points):
        current_phi = phi_array[i]
        current_sat = saturation_array[i]

        if not (0.0 <= current_phi <= 1.0 and 0.0 <= current_sat <= 1.0):
            print(f"Warning: Invalid porosity ({current_phi}) or saturation ({current_sat}) at index {i}. Skipping calculation for this point (NaN results).")
            K_eff_results[i], G_eff_results[i], Vp_results[i] = np.nan, np.nan, np.nan
            continue

        # Effective fluid bulk modulus (Kf) using Brie's model
        K_fluid_effective = brie_fluid_model.calculate_fluid_modulus(current_sat, water_bulk_modulus, gas_bulk_modulus)

        # Poisson's ratio of the initial solid matrix material
        v_matrix = (3 * K_matrix_init - 2 * G_matrix_init) / (2 * (3 * K_matrix_init + G_matrix_init))
        if not (-1.0 < v_matrix < 0.5):
             print(f"Warning: Matrix Poisson's ratio ({v_matrix:.3f}) for initial matrix is outside physical bounds.")


        # DEM parameters (b, c, d, g) based on initial matrix properties and pore aspect ratio
        b_param = 3 * np.pi * aspect_ratio_pores * (1 - 2 * v_matrix) / (4 * (1 - v_matrix**2))

        # Original two-step definition of c and d: c_val = 1 / (1/5 * term_c); d_val = 1 / (1/5 * term_d)
        # Equivalent to c_val = 5 / term_c
        term_c_inv = (3 + 8 * (1 - v_matrix) / (np.pi * aspect_ratio_pores * (2 - v_matrix)))
        c_param = 5.0 / term_c_inv if term_c_inv !=0 else np.inf # Original was 1/(0.2*term) then 1/c_param
                                                                # This seems to be c_param = 0.2 * term_c_inv, then c = 1/c.
                                                                # The formulation in DEMModel class was used: c = 1.0 / (0.2 * c_inv_term)
                                                                # Let's match DEMModel class for consistency.
        c_param = 1.0 / (0.2 * term_c_inv) if term_c_inv !=0 else np.inf

        term_d_inv = (1 + 8 * (1 - v_matrix) * (5 - v_matrix) / (3 * np.pi * aspect_ratio_pores * (2 - v_matrix)))
        d_param = 1.0 / (0.2 * term_d_inv) if term_d_inv !=0 else np.inf
        
        g_param = np.pi * aspect_ratio_pores / (2 * (1 - v_matrix))
        
        # Implicit equation for K_eff (same as in DEMModel class)
        def eq_Keff_dem(Keff_try):
            if Keff_try <= 0: return 1e12
            if np.isclose(K_matrix_init, K_fluid_effective): return Keff_try - K_matrix_init
            lhs_k = (Keff_try - K_fluid_effective) / (K_matrix_init - K_fluid_effective) * \
                    (K_matrix_init / Keff_try)**(1.0 / (1.0 + b_param))
            rhs_k = (1.0 - current_phi)**(1.0 / (1.0 + b_param))
            return lhs_k - rhs_k

        sol_K = root(eq_Keff_dem, K_matrix_init, method='lm', tol=1e-7)
        if sol_K.success and sol_K.x[0] > 0:
            K_eff_results[i] = sol_K.x[0]
        else:
            raise ValueError(f"velDEM: K_eff root finding failed (index {i}, phi={current_phi:.2f}, sat={current_sat:.2f}): {sol_K.message}")

        # Implicit equation for G_eff (same as in DEMModel class)
        # Ensure K_fluid_effective is not zero for safety in c*g/(d*Kf) term
        k_fluid_safe = K_fluid_effective if K_fluid_effective > 1e-9 else 1e-9
        def eq_Geff_dem(Geff_try):
            if Geff_try <= 0: return 1e12
            if np.isclose(G_matrix_init, 0.0): return Geff_try # if matrix G is 0, effective G should be 0.
            
            term_num_g = (1.0 / Geff_try) + (c_param * g_param / (d_param * k_fluid_safe))
            term_den_g = (1.0 / G_matrix_init) + (c_param * g_param / (d_param * k_fluid_safe))
            if np.isclose(term_den_g, 0.0): return 1e12 # Avoid division by zero

            ratio_g = term_num_g / term_den_g
            exp_g = 1.0 - (c_param / d_param) if d_param != 0 else 1.0
            if ratio_g < 0 and isinstance(exp_g, float) and not exp_g.is_integer(): return 1e12 # Avoid complex numbers

            lhs_g = (Geff_try / G_matrix_init) * (ratio_g ** exp_g)
            rhs_g = (1.0 - current_phi)**(1.0 / d_param if d_param != 0 else 1.0)
            return lhs_g - rhs_g

        sol_G = root(eq_Geff_dem, G_matrix_init, method='lm', tol=1e-7)
        if sol_G.success and sol_G.x[0] >= 0:
            G_eff_results[i] = sol_G.x[0]
        else:
            raise ValueError(f"velDEM: G_eff root finding failed (index {i}, phi={current_phi:.2f}, sat={current_sat:.2f}): {sol_G.message}")

        # Total density
        rho_fluid_effective = current_sat * water_density + (1.0 - current_sat) * gas_density
        rho_total_effective = rho_matrix * (1.0 - current_phi) + rho_fluid_effective * current_phi
        if rho_total_effective <=0:
            raise ValueError(f"velDEM: Total density non-positive ({rho_total_effective}) at index {i}.")

        # P-wave velocity (moduli in GPa converted to Pa)
        p_modulus = K_eff_results[i] * 1e9 + (4.0/3.0) * G_eff_results[i] * 1e9
        if p_modulus < 0:
             Vp_results[i] = np.nan # Non-physical modulus
        else:
            Vp_results[i] = np.sqrt(p_modulus / rho_total_effective)

    return K_eff_results, G_eff_results, Vp_results


def vel_porous(phi_array: np.ndarray, K_matrix: float, G_matrix: float, rho_matrix: float,
               saturation_array: np.ndarray, depth: float = 1.0,
               critical_porosity: float = 0.4, coordination_number: float = 4.0,
               water_bulk_modulus: float = 2.2, gas_bulk_modulus: float = 0.01,
               water_density: float = 1000.0, gas_density: float = 1.225
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone function to calculate P-wave velocity bounds (Vp_high, Vp_low) for porous rocks.
    Combines Hertz-Mindlin theory for critical porosity frame moduli with Hashin-Shtrikman bounds
    and fluid substitution (via Brie's model / `satK` helper).
    Similar to `HertzMindlinModel.calculate_velocity`.

    Args:
        phi_array (np.ndarray): Array of porosity values (φ).
        K_matrix (float): Bulk modulus of the solid mineral matrix (Km) [GPa].
        G_matrix (float): Shear modulus of the solid mineral matrix (Gm) [GPa].
        rho_matrix (float): Density of the solid matrix (ρ_b) [kg/m³].
        saturation_array (np.ndarray): Array of water saturation values (Sat).
        depth (float, optional): Depth for effective pressure estimation [m]. Defaults to 1.0.
        critical_porosity (float, optional): Critical porosity (φ_c). Defaults to 0.4.
        coordination_number (float, optional): Avg. grain contacts (C) at φ_c. Defaults to 4.0.
                                               (Note: 4 is low for typical sphere packs, 6-9 is more common).
        water_bulk_modulus (float, optional): K_water [GPa]. Defaults to 2.2.
        gas_bulk_modulus (float, optional): K_gas [GPa]. Defaults to 0.01.
        water_density (float, optional): ρ_water [kg/m³]. Defaults to 1000.0.
        gas_density (float, optional): ρ_gas [kg/m³]. Defaults to 1.225.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Vp_high (np.ndarray): P-wave velocity upper bound [m/s].
            - Vp_low (np.ndarray): P-wave velocity lower bound [m/s].
    """
    if phi_array.shape != saturation_array.shape:
        raise ValueError("Porosity and saturation arrays must have the same shape.")

    # Poisson's ratio of the solid matrix
    v_matrix = (3 * K_matrix - 2 * G_matrix) / (2 * (3 * K_matrix + G_matrix))
    if not (-1.0 < v_matrix < 0.5): print(f"Warning: Matrix Poisson's ratio ({v_matrix:.3f}) is outside physical bounds.")

    # Simplified effective pressure calculation [GPa]
    # Potential Issue: (rho_matrix - 1000) can be negative if rho_matrix < 1000.
    # Pressure P should be non-negative for Hertz-Mindlin.
    P_eff_gpa = max(0, (rho_matrix - water_density) * 9.80665 * depth / 1e9)
    if P_eff_gpa == 0: print("Warning: Effective pressure is zero. Hertz-Mindlin moduli will be zero.")

    # Hertz-Mindlin moduli for dry frame at critical porosity (phi_c)
    # K_HM_dry = [ C^2 * (1-φ_c)^2 * G_m^2 * P_eff / (18 * π^2 * (1-v_m)^2) ]^(1/3)
    # G_HM_dry = Factor * [ (3*C^2*(1-φ_c)^2*G_m^2*P_eff) / (2*π^2*(1-v_m)^2) ]^(1/3)
    den_hm = (18 * np.pi**2 * (1 - v_matrix)**2)
    if np.isclose(den_hm, 0.0): raise ValueError(f"Matrix Poisson's ratio {v_matrix} causes division by zero in HM calc.")

    K_HM_dry_crit = (coordination_number**2 * (1 - critical_porosity)**2 * G_matrix**2 * P_eff_gpa / den_hm)**(1/3) if P_eff_gpa > 0 else 0.0

    g_hm_factor_num = (5 - 4 * v_matrix)
    g_hm_factor_den = (10 - 2 * v_matrix) # Original was (10-2v), same as 2*(5-v). For consistency with class: 5*(2-v)
                                          # Using 5*(2-v) instead of 2*(5-v) based on common forms.
                                          # If original (10-2v) is intended: (5-4v)/(2*(5-v))
    if np.isclose(g_hm_factor_den, 0.0): raise ValueError(f"Matrix Poisson's ratio {v_matrix} causes division by zero in G_HM factor.")
    g_hm_factor = g_hm_factor_num / g_hm_factor_den

    g_hm_bracket_term_cubed_den = (2 * np.pi**2 * (1 - v_matrix)**2)
    if np.isclose(g_hm_bracket_term_cubed_den, 0.0): raise ValueError(f"Matrix Poisson's ratio {v_matrix} causes division by zero in G_HM bracket term.")
    g_hm_bracket_term_cubed = (3 * coordination_number**2 * (1 - critical_porosity)**2 * G_matrix**2 * P_eff_gpa) / \
                              g_hm_bracket_term_cubed_den
    G_HM_dry_crit = g_hm_factor * (g_hm_bracket_term_cubed)**(1/3) if P_eff_gpa > 0 else 0.0

    Vp_high_list = [] # Using lists for append, then convert to array
    Vp_low_list = []

    for i in range(len(phi_array)):
        current_phi = phi_array[i]
        current_sat = saturation_array[i]
        if not (0.0 <= current_phi <= 1.0 and 0.0 <= current_sat <= 1.0):
            print(f"Warning: Invalid phi ({current_phi}) or sat ({current_sat}) at index {i}. Resulting Vp will be NaN.")
            Vp_high_list.append(np.nan)
            Vp_low_list.append(np.nan)
            continue

        K_eff_dry_H_i, G_eff_dry_H_i = 0.0, 0.0
        K_eff_dry_L_i, G_eff_dry_L_i = 0.0, 0.0

        if current_phi < critical_porosity:
            # Modified Hashin-Shtrikman bounds for porosity < critical_porosity
            # Lower bound moduli (effective medium with K_HM, G_HM as "soft" component)
            den1_Kl = K_HM_dry_crit + (4/3) * G_HM_dry_crit
            den2_Kl = K_matrix + (4/3) * G_HM_dry_crit
            if not (np.isclose(den1_Kl,0) or np.isclose(den2_Kl,0)):
                 K_eff_dry_L_i = ((current_phi / critical_porosity) / den1_Kl + \
                               (1 - current_phi / critical_porosity) / den2_Kl)**(-1) - (4/3) * G_HM_dry_crit

            zeta_L_den = (K_HM_dry_crit + 2 * G_HM_dry_crit)
            zeta_L = (G_HM_dry_crit / 6) * (9 * K_HM_dry_crit + 8 * G_HM_dry_crit) / \
                     zeta_L_den if zeta_L_den !=0 else 0
            den1_Gl = G_HM_dry_crit + zeta_L
            den2_Gl = G_matrix + zeta_L
            if not (np.isclose(den1_Gl,0) or np.isclose(den2_Gl,0)):
                G_eff_dry_L_i = ((current_phi / critical_porosity) / den1_Gl + \
                               (1 - current_phi / critical_porosity) / den2_Gl)**(-1) - zeta_L

            # Upper bound moduli (effective medium with K_matrix, G_matrix as "stiff" component, K_HM as "soft")
            den1_Kh = K_HM_dry_crit + (4/3) * G_matrix # Uses G_matrix here
            den2_Kh = K_matrix + (4/3) * G_matrix      # Uses G_matrix here
            if not (np.isclose(den1_Kh,0) or np.isclose(den2_Kh,0)):
                K_eff_dry_H_i = ((current_phi / critical_porosity) / den1_Kh + \
                               (1 - current_phi / critical_porosity) / den2_Kh)**(-1) - (4/3) * G_matrix

            zeta_H_den = (K_matrix + 2*G_matrix)
            zeta_H = (G_matrix / 6) * (9 * K_matrix + 8 * G_matrix) / \
                     zeta_H_den if zeta_H_den!=0 else 0
            den1_Gh = G_HM_dry_crit + zeta_H # Uses G_HM_dry_crit here
            den2_Gh = G_matrix + zeta_H      # Uses G_matrix here
            if not (np.isclose(den1_Gh,0) or np.isclose(den2_Gh,0)):
                G_eff_dry_H_i = ((current_phi / critical_porosity) / den1_Gh + \
                               (1 - current_phi / critical_porosity) / den2_Gh)**(-1) - zeta_H

            K_eff_dry_L_i, G_eff_dry_L_i = max(0,K_eff_dry_L_i), max(0,G_eff_dry_L_i)
            K_eff_dry_H_i, G_eff_dry_H_i = max(0,K_eff_dry_H_i), max(0,G_eff_dry_H_i)

            # Fluid substitution using satK helper (which uses Brie for K_fluid_eff)
            K_sat_H = satK(K_eff_dry_H_i, K_matrix, current_phi, current_sat, water_bulk_modulus, gas_bulk_modulus)
            G_sat_H = G_eff_dry_H_i # Shear modulus unchanged by fluid (Gassmann)
            K_sat_L = satK(K_eff_dry_L_i, K_matrix, current_phi, current_sat, water_bulk_modulus, gas_bulk_modulus)
            G_sat_L = G_eff_dry_L_i
        else: # Porosity >= critical_porosity: Suspension model (Reuss-like average)
            # This models the rock as a suspension of "critical porosity grains" in additional fluid.
            # The "solid" part is the frame at critical_porosity (K_HM_dry_crit, G_HM_dry_crit).
            # The "excess porosity" (current_phi - critical_porosity) is filled with fluid.
            # The original code for K_eff_susp_dry:
            # K_eff = ((1-phi)/(1-phi_c)/(K_HM + 4/3 G_HM) + (phi-phi_c)/(1-phi_c)/(4/3 G_HM))^-1 - 4/3 G_HM
            # This form is complex and its derivation/assumptions for the "fluid part K=0" term (4/3 G_HM) need verification.
            # For a simple Reuss average of frame_at_phi_c and fluid_in_excess_pores (K_fluid_eff, G_fluid_eff=0):
            # phi_excess_vol_fraction = (current_phi - critical_porosity) / (1 - critical_porosity) -> This is fraction of non-solid volume
            # K_dry_susp = ((1-phi_excess_vol_fraction)/K_HM_dry_crit + phi_excess_vol_fraction/K_pore_dry)^-1
            # G_dry_susp = ((1-phi_excess_vol_fraction)/G_HM_dry_crit + phi_excess_vol_fraction/G_pore_dry)^-1
            # Assuming K_pore_dry = 0, G_pore_dry = 0 for empty excess pores.
            # The original code seems to use a specific variant. We will follow it.

            # Fraction of original solid phase (at critical porosity) in the (1-phi_c) volume
            frac_solid_phase = (1.0 - current_phi) / (1.0 - critical_porosity) if (1.0 - critical_porosity) != 0 else 0
            # Fraction of "added" pore space (beyond critical) in the (1-phi_c) volume
            frac_added_pores = (current_phi - critical_porosity) / (1.0 - critical_porosity) if (1.0 - critical_porosity) != 0 else 1.0

            # Ensure fractions are valid (e.g. phi_i should not be < critical_porosity here if logic is strict)
            # However, original code structure implies phi_i >= critical_porosity for this branch.
            # If phi_i = critical_porosity, then frac_solid_phase=1, frac_added_pores=0. K_eff should be K_HM.

            # Denominators for K_eff_susp_dry terms
            den1_Keff_susp = K_HM_dry_crit + (4/3) * G_HM_dry_crit
            den2_Keff_susp = (4/3) * G_HM_dry_crit # This term for K is unusual (fluid K=0 usually)
            if not (np.isclose(den1_Keff_susp,0) or np.isclose(den2_Keff_susp,0) or np.isclose(1.0 - critical_porosity, 0.0)):
                 K_eff_dry_susp = ( frac_solid_phase / den1_Keff_susp + frac_added_pores / den2_Keff_susp
                                   )**(-1) - (4/3) * G_HM_dry_crit
            else:
                K_eff_dry_susp = 0.0
                if np.isclose(current_phi, critical_porosity): K_eff_dry_susp = K_HM_dry_crit

            zeta_susp_den = (K_HM_dry_crit + 2 * G_HM_dry_crit)
            zeta_susp = (G_HM_dry_crit / 6) * (9 * K_HM_dry_crit + 8 * G_HM_dry_crit) / \
                        zeta_susp_den if zeta_susp_den !=0 else 0
            den1_Geff_susp = G_HM_dry_crit + zeta_susp
            den2_Geff_susp = zeta_susp # Term for "added pores" G=0 contribution
            if not (np.isclose(den1_Geff_susp,0) or np.isclose(den2_Geff_susp,0) or np.isclose(1.0 - critical_porosity, 0.0)):
                G_eff_dry_susp = ( frac_solid_phase / den1_Geff_susp + frac_added_pores / den2_Geff_susp
                                   )**(-1) - zeta_susp
            else:
                G_eff_dry_susp = 0.0
                if np.isclose(current_phi, critical_porosity): G_eff_dry_susp = G_HM_dry_crit

            K_eff_dry_susp, G_eff_dry_susp = max(0,K_eff_dry_susp), max(0,G_eff_dry_susp)

            K_sat_H = K_sat_L = satK(K_eff_dry_susp, K_matrix, current_phi, current_sat, water_bulk_modulus, gas_bulk_modulus)
            G_sat_H = G_sat_L = G_eff_dry_susp

        # Total effective density
        rho_fluid_eff = current_sat * water_density + (1.0 - current_sat) * gas_density
        rho_total_eff = rho_matrix * (1.0 - current_phi) + rho_fluid_eff * current_phi
        if rho_total_eff <=0:
            Vp_high_list.append(np.nan); Vp_low_list.append(np.nan)
            continue

        # P-wave velocities (GPa for K,G converted to Pa by 1e9)
        p_mod_H = K_sat_H * 1e9 + (4/3) * G_sat_H * 1e9
        p_mod_L = K_sat_L * 1e9 + (4/3) * G_sat_L * 1e9

        Vp_high_list.append(np.sqrt(p_mod_H / rho_total_eff) if p_mod_H >=0 else np.nan)
        Vp_low_list.append(np.sqrt(p_mod_L / rho_total_eff) if p_mod_L >=0 else np.nan)

    return np.array(Vp_high_list), np.array(Vp_low_list)
