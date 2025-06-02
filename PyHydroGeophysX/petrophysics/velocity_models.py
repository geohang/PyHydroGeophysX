"""
Seismic velocity models for relating rock properties to elastic wave velocities.
"""
import numpy as np
from scipy.optimize import fsolve, root
from typing import Tuple, Optional, Union, List, Dict, Any


class BaseVelocityModel:
    """Base class for seismic velocity models."""
    
    def __init__(self):
        """Initialize base velocity model."""
        # This base class constructor currently does nothing.
        # It's a placeholder for any common initialization logic for derived models.
        pass
    
    def calculate_velocity(self, **kwargs) -> np.ndarray:
        """
        Calculate seismic velocity from rock properties.
        
        Args:
            **kwargs: Rock properties specific to each model
            
        Returns:
            Seismic velocity values (Vp, Vs, or both)
        """
        # This method must be implemented by any class that inherits from BaseVelocityModel.
        # It serves as a template for what velocity calculation methods should look like.
        raise NotImplementedError("Velocity calculation must be implemented in derived classes")


class VRHModel(BaseVelocityModel):
    """
    Voigt-Reuss-Hill (VRH) mixing model for effective elastic properties of composites.
    The VRH model calculates the arithmetic average of the Voigt and Reuss bounds.
    - Voigt bound assumes uniform strain across components (iso-strain), leading to an upper bound for moduli.
    - Reuss bound assumes uniform stress across components (iso-stress), leading to a lower bound for moduli.
    The VRH average is often a good empirical estimate for isotropic multi-component materials.
    It's commonly used for estimating the elastic properties of mineral mixtures forming a rock matrix.
    """
    
    def __init__(self):
        """Initialize VRH model."""
        super().__init__() # Call to the base class constructor.
    
    def calculate_properties(self, 
                           fractions: List[float],    # f_i
                           bulk_moduli: List[float],  # K_i
                           shear_moduli: List[float], # G_i
                           densities: List[float]     # ρ_i
                           ) -> Tuple[float, float, float]: # K_vrh, G_vrh, ρ_eff
        """
        Calculate effective elastic properties (bulk modulus, shear modulus, density)
        of a composite material using the Voigt-Reuss-Hill (VRH) averaging scheme.

        Args:
            fractions (List[float]): List of volume fractions of each component (mineral).
                                     These must sum to 1.0.
            bulk_moduli (List[float]): List of bulk moduli for each component [GPa].
            shear_moduli (List[float]): List of shear moduli for each component [GPa].
            densities (List[float]): List of densities for each component [kg/m³].
            
        Returns:
            Tuple[float, float, float]:
                - K_vrh: Effective bulk modulus of the composite [GPa].
                - G_vrh: Effective shear modulus of the composite [GPa].
                - rho_eff: Effective density of the composite [kg/m³].
        """
        # Convert input lists to NumPy arrays for vectorized calculations.
        f = np.array(fractions)
        K = np.array(bulk_moduli)
        G = np.array(shear_moduli)
        rho = np.array(densities)
        
        # Validate that the sum of volume fractions is close to 1.0.
        if not np.isclose(np.sum(f), 1.0):
            raise ValueError("Volume fractions must sum to 1.0.")
        
        # --- Voigt Average (Upper Bound) ---
        # K_V = Σ (f_i * K_i)
        # G_V = Σ (f_i * G_i)
        K_voigt = np.sum(f * K) # Weighted arithmetic mean for bulk modulus.
        G_voigt = np.sum(f * G) # Weighted arithmetic mean for shear modulus.
        
        # --- Reuss Average (Lower Bound) ---
        # 1/K_R = Σ (f_i / K_i)  => K_R = (Σ (f_i / K_i))^(-1)
        # 1/G_R = Σ (f_i / G_i)  => G_R = (Σ (f_i / G_i))^(-1)
        K_reuss = 1.0 / np.sum(f / K) # Weighted harmonic mean for bulk modulus.
        G_reuss = 1.0 / np.sum(f / G) # Weighted harmonic mean for shear modulus.
        
        # --- Hill (VRH) Average ---
        # K_VRH = (K_V + K_R) / 2
        # G_VRH = (G_V + G_R) / 2
        K_vrh = 0.5 * (K_voigt + K_reuss) # Arithmetic mean of Voigt and Reuss bulk moduli.
        G_vrh = 0.5 * (G_voigt + G_reuss) # Arithmetic mean of Voigt and Reuss shear moduli.
        
        # --- Effective Density ---
        # ρ_eff = Σ (f_i * ρ_i)
        # The effective density is the simple weighted average of component densities.
        rho_eff = np.sum(f * rho)
        
        return K_vrh, G_vrh, rho_eff
    
    def calculate_velocity(self, 
                         fractions: List[float],    # f_i
                         bulk_moduli: List[float],  # K_i
                         shear_moduli: List[float], # G_i
                         densities: List[float]     # ρ_i
                         ) -> Tuple[float, float]: # Vp, Vs
        """
        Calculate P-wave (Vp) and S-wave (Vs) velocities based on the effective
        elastic properties determined by the VRH model.
        
        Args:
            fractions (List[float]): Volume fractions of each mineral component.
            bulk_moduli (List[float]): Bulk moduli of each mineral component [GPa].
            shear_moduli (List[float]): Shear moduli of each mineral component [GPa].
            densities (List[float]): Densities of each mineral component [kg/m³].
            
        Returns:
            Tuple[float, float]:
                - Vp: P-wave velocity [m/s].
                - Vs: S-wave velocity [m/s].
        """
        # First, calculate the effective elastic moduli (K_eff, G_eff) and density (rho_eff)
        # of the mineral mixture using the VRH averaging method.
        K_eff, G_eff, rho_eff = self.calculate_properties(
            fractions, bulk_moduli, shear_moduli, densities
        )
        
        # Convert effective moduli from GPa to Pascals (Pa) for velocity calculation,
        # as standard velocity equations use SI units (Pa for pressure/modulus, kg/m³ for density).
        # 1 GPa = 1e9 Pa.
        K_eff_pa = K_eff * 1e9 # Effective bulk modulus in Pa.
        G_eff_pa = G_eff * 1e9 # Effective shear modulus in Pa.
        
        # --- Calculate P-wave Velocity (Vp) ---
        # Vp = sqrt((K_eff + 4/3 * G_eff) / ρ_eff)
        # This equation assumes an isotropic elastic medium.
        Vp = np.sqrt((K_eff_pa + (4.0/3.0) * G_eff_pa) / rho_eff)
        
        # --- Calculate S-wave Velocity (Vs) ---
        # Vs = sqrt(G_eff / ρ_eff)
        # This also assumes an isotropic elastic medium.
        # If G_eff_pa is zero (e.g., for a fluid), Vs will be zero.
        Vs = np.sqrt(G_eff_pa / rho_eff)
        
        return Vp, Vs


class BrieModel:
    """
    Brie's model for calculating the effective bulk modulus of a partially saturated medium.
    """
    
    def __init__(self, exponent: float = 3.0):
        """
        Initialize Brie's model.
        Brie's model is an empirical formula to calculate the effective bulk modulus of a fluid mixture
        (typically water and gas) in porous media as a function of water saturation.
        It is often used as an alternative to Reuss averaging for fluid mixtures, especially
        when patchy saturation is suspected or to better fit observed velocity-saturation trends.
        
        Args:
            exponent (float): Brie's exponent (often denoted 'e' or 'p'). This is an empirical fitting parameter.
                              A common default value is 3, but it can vary significantly (e.g., 2 to 20 or more)
                              depending on the rock and fluid properties and how well the model fits data.
                              Higher exponents give a sharper transition from gas-like to water-like behavior
                              as saturation increases.
        """
        self.exponent = exponent # Store the Brie exponent.
    
    def calculate_fluid_modulus(self, 
                              saturation: float,          # S_w (water saturation)
                              water_modulus: float = 2.0, # K_w (bulk modulus of water)
                              gas_modulus: float = 0.01   # K_g (bulk modulus of gas/air)
                              ) -> float:                 # K_fl (effective fluid bulk modulus)
        """
        Calculate the effective bulk modulus of the pore fluid mixture (e.g., water and gas)
        using Brie's empirical equation.
        
        The equation is: K_fl = (K_w - K_g) * S_w^e + K_g
        where:
            K_fl is the effective fluid bulk modulus.
            K_w is the bulk modulus of water.
            K_g is the bulk modulus of gas (e.g., air, methane).
            S_w is the water saturation (fraction, 0 to 1).
            e is Brie's exponent (self.exponent).

        Args:
            saturation (float): Water saturation (S_w), ranging from 0 (fully gas) to 1 (fully water).
            water_modulus (float, optional): Bulk modulus of water (K_w) [GPa]. Defaults to 2.0 GPa.
                                             (Typical value for water is ~2.2 GPa, can vary with P, T, salinity).
            gas_modulus (float, optional): Bulk modulus of gas (K_g) [GPa]. Defaults to 0.01 GPa.
                                           (Gas modulus is very low, sensitive to P, T. 0.01 GPa is a typical placeholder for air at near-surface).
            
        Returns:
            float: Effective bulk modulus of the fluid mixture (K_fl) [GPa].
        """
        # Ensure saturation is within physical bounds [0, 1] for the calculation.
        # Though Brie's formula itself doesn't strictly require S_w in [0,1] mathematically,
        # it's physically defined for this range.
        S_w = np.clip(saturation, 0.0, 1.0)

        # Brie's equation: K_fl = (K_w - K_g) * S_w^e + K_g
        # This models a non-linear mixing behavior. When S_w = 0, K_fl = K_g. When S_w = 1, K_fl = K_w.
        effective_K_fluid = (water_modulus - gas_modulus) * S_w ** self.exponent + gas_modulus
        return effective_K_fluid
    
    def calculate_saturated_modulus(self, 
                                  dry_modulus: float,       # K_dry (Bulk modulus of the dry rock frame)
                                  mineral_modulus: float,   # K_m (Bulk modulus of the solid mineral matrix)
                                  porosity: float,          # φ (phi)
                                  saturation: float,        # S_w (Water saturation)
                                  water_modulus: float = 2.0, # K_w
                                  gas_modulus: float = 0.01   # K_g
                                  ) -> float:                 # K_sat (Bulk modulus of the rock saturated with fluid mixture)
        """
        Calculate the bulk modulus of a rock saturated with a fluid mixture (K_sat),
        using Gassmann's equation with the effective fluid modulus derived from Brie's model.

        This method first calculates the effective fluid bulk modulus (K_fl) using Brie's model,
        then substitutes this K_fl into Gassmann's equation to find K_sat.
        
        Gassmann's equation:
        K_sat = K_dry + ( (1 - K_dry/K_m)^2 ) / ( φ/K_fl + (1-φ)/K_m - K_dry/K_m^2 )

        This can be rewritten as shown in the code, which might be a specific algebraic form or rearrangement.
        The provided code uses a form:
        K_sat = [ K_dry/(K_m - K_dry) + K_fl/(φ*(K_m-K_fl)) ] / [ 1 + K_dry/(K_m - K_dry) + K_fl/(φ*(K_m-K_fl)) ] * K_m
        This form needs to be verified against standard Gassmann forms or cited if it's a specific variant.
        Let's assume the implemented formula is a known variant.
        SUGGESTION: Double-check the algebraic form of Gassmann's equation used here.
        A common form is K_sat = K_dry + alpha^2 * M, where alpha = (1 - K_dry/K_mineral) and M is related to K_fl, K_mineral, porosity.
        The standard Gassmann equation relates K_sat to K_dry, K_mineral (K0 or Km), K_fluid (Kf), and porosity (phi):
        K_sat / (K_mineral - K_sat) - K_dry / (K_mineral - K_dry) = K_fluid / (phi * (K_mineral - K_fluid))
        Or, more commonly: K_sat = K_dry + ((1 - K_dry/K_mineral)^2) / (phi/K_fluid + (1-phi)/K_mineral - K_dry/(K_mineral^2))

        Args:
            dry_modulus (float): Bulk modulus of the dry rock frame (K_dry) [GPa].
            mineral_modulus (float): Intrinsic bulk modulus of the solid mineral phase (K_m or K_0) [GPa].
            porosity (float): Porosity (φ) of the rock, as a fraction (0 to 1).
            saturation (float): Water saturation (S_w), as a fraction (0 to 1).
            water_modulus (float, optional): Bulk modulus of water (K_w) [GPa]. Defaults to 2.0 GPa.
            gas_modulus (float, optional): Bulk modulus of gas (K_g) [GPa]. Defaults to 0.01 GPa.
            
        Returns:
            float: Saturated bulk modulus (K_sat) of the rock [GPa].
        """
        # Step 1: Calculate the effective bulk modulus of the fluid mixture (K_fl) using Brie's model.
        fluid_modulus = self.calculate_fluid_modulus( # K_fl
            saturation, water_modulus, gas_modulus
        )
        
        # Step 2: Apply Gassmann's equation to get the saturated bulk modulus (K_sat).
        # The formula used here is:
        # TermA = K_dry / (K_mineral - K_dry)
        # TermB = K_fluid / (porosity * (K_mineral - K_fluid))
        # K_sat = (TermA + TermB) / (1 + TermA + TermB) * K_mineral
        # This needs to be equivalent to the standard Gassmann equation.
        # Assumptions of Gassmann's equation:
        #   - Isotropic, elastic, homogeneous mineral matrix and rock frame.
        #   - Pores are interconnected and filled with a frictionless fluid.
        #   - Low frequency (seismic range), where wave-induced pore pressures are equilibrated.
        #   - Shear modulus (G) is assumed to be unaffected by fluid saturation (G_sat = G_dry).

        # To avoid division by zero if K_mineral is very close to K_dry or K_fluid:
        # SUGGESTION: Add epsilon checks or use a more numerically stable form of Gassmann if needed.
        # For example, K_m - K_dry should not be zero. K_m - K_fl should not be zero. Porosity should not be zero.
        if np.isclose(mineral_modulus, dry_modulus): # K_m == K_dry implies solid rock, porosity should be 0.
            # If K_m == K_dry, then K_sat should be K_m.
            # The formula might become unstable.
            # If porosity is 0, K_sat = K_m.
            # This case should be handled carefully or rely on physical constraints (porosity > 0 for Gassmann usually).
            # If K_dry is very close to K_m, this implies porosity is very low or pores are very stiff.
             return mineral_modulus # Or handle as per physical expectation for this edge case.

        if np.isclose(porosity, 0.0): # If porosity is zero, K_sat = K_mineral
            return mineral_modulus

        term1_num = mineral_modulus - dry_modulus
        term2_num_fl = mineral_modulus - fluid_modulus

        if np.isclose(term1_num, 0.0) or np.isclose(term2_num_fl, 0.0) or np.isclose(fluid_modulus, mineral_modulus) :
             # This condition could mean K_dry = K_mineral (no pores), or K_fluid = K_mineral (problematic).
             # Gassmann might not be applicable or simplifies.
             # If K_dry = K_mineral, then K_sat = K_mineral.
             # If K_fluid = K_mineral, the denominator term in Gassmann involving (K_mineral - K_fluid) becomes problematic.
             # This specific formula's stability needs checking.
             # For now, let's assume inputs are physically reasonable for Gassmann.
             pass


        # Numerator part of the main fraction in the specific Gassmann form used
        # Let A = dry_modulus / (mineral_modulus - dry_modulus)
        # Let B = fluid_modulus / (porosity * (mineral_modulus - fluid_modulus))
        # K_sat = (A + B) / (1 + A + B) * mineral_modulus
        # This appears to be a rearranged form.
        # Standard Gassmann: K_sat = K_dry + (1 - K_dry/K_m)^2 / (phi/K_fl + (1-phi)/K_m - K_dry/K_m^2)

        # Calculate terms for the specific formula used:
        # TermX = K_dry / (K_m - K_dry)
        term_X = dry_modulus / (mineral_modulus - dry_modulus)
        # TermY = K_fl / (φ * (K_m - K_fl))
        term_Y = fluid_modulus / (porosity * (mineral_modulus - fluid_modulus))

        # Saturated bulk modulus using the specific formula
        saturated_K = (term_X + term_Y) / (1 + term_X + term_Y) * mineral_modulus
        # SUGGESTION: This form of Gassmann's equation should be verified or cited.
        # It can be derived from one of Wood's equations or specialized forms.
        # If K_dry = 0 (e.g. suspension), then K_sat = K_fl / (1 + (1-1/K_m)*K_fl/(phi*(K_m-K_fl))) * K_m (needs check)
        
        return saturated_K


class DEMModel(BaseVelocityModel):
    """
    Differential Effective Medium (DEM) model for calculating elastic properties
    and seismic velocities of porous rocks.
    """
    
    def __init__(self):
        """Initialize DEM model."""
        super().__init__() # Call base class constructor.
        # DEM model typically requires properties of the solid matrix and inclusion phase (fluid).
        # These are often passed during the calculation method.
        # No specific initialization parameters stored in this version of the class itself.
    
    def calculate_velocity(self, 
                         porosity: np.ndarray,       # φ (phi), array of porosity values
                         saturation: np.ndarray,     # S_w, array of water saturation values
                         bulk_modulus: float,        # K_m, initial bulk modulus of solid matrix [GPa]
                         shear_modulus: float,       # G_m, initial shear modulus of solid matrix [GPa]
                         mineral_density: float,     # ρ_m, density of solid matrix [kg/m³]
                         aspect_ratio: float = 0.1   # α (alpha), aspect ratio of pores/cracks
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # K_eff, G_eff, Vp
        """
        Calculate effective P-wave velocity (Vp) along with effective bulk (Keff) and
        shear (Geff) moduli using a Differential Effective Medium (DEM) model.
        The DEM model incrementally adds inclusions (pores filled with fluid) into a solid matrix,
        updating the effective properties at each step. This implementation solves
        differential equations (often simplified to algebraic forms for specific inclusion shapes)
        numerically for each porosity point.

        The specific DEM formulation used here (parameterization of b, c, d, g) seems to be based on
        Wu's (1966) or similar work for spheroidal inclusions, adapted for DEM.
        The equations for Keff and Geff are solved iteratively using `scipy.optimize.root`.
        The fluid properties are calculated using Brie's model for the fluid mixture.

        Args:
            porosity (np.ndarray): Array of porosity values (φ), fraction from 0 to 1.
            saturation (np.ndarray): Array of water saturation values (S_w), fraction from 0 to 1.
                                     Assumes porosity and saturation arrays are of the same length.
            bulk_modulus (float): Bulk modulus of the solid mineral matrix (K_m) [GPa].
            shear_modulus (float): Shear modulus of the solid mineral matrix (G_m) [GPa].
            mineral_density (float): Density of the solid mineral matrix (ρ_m) [kg/m³].
            aspect_ratio (float, optional): Aspect ratio (α) of the pores/inclusions.
                                            α = (short axis / long axis). For spheres, α=1.
                                            For cracks, α << 1. Defaults to 0.1 (oblate spheroids).
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Keff: Array of effective bulk moduli [GPa] for each porosity.
                - Geff: Array of effective shear moduli [GPa] for each porosity.
                - Vp: Array of P-wave velocities [m/s] for each porosity.
        """
        # Ensure inputs are numpy arrays for consistency, though type hints already specify.
        # porosity = np.asarray(porosity)
        # saturation = np.asarray(saturation)

        # Initialize output arrays for effective moduli and Vp, same length as porosity array.
        Keff_results = np.zeros_like(porosity, dtype=float)
        Geff_results = np.zeros_like(porosity, dtype=float)
        Vp_results = np.zeros_like(porosity, dtype=float)
        
        # --- Fluid Properties (assumed constant here, could be arrays if varying per point) ---
        K_water = 2.0  # Bulk modulus of water (K_w) [GPa]. Typical value.
                       # SUGGESTION: Make K_water and K_air parameters of the method or class.
        K_air = 0.01   # Bulk modulus of air/gas (K_a) [GPa]. Typical value for air at STP.
        
        # Iterate over each porosity/saturation point to calculate effective properties.
        # DEM is typically solved incrementally, but here it seems to be solved for each target porosity φ_i
        # starting from a solid matrix (φ=0) up to φ_i. The equations used are often solutions
        # to the DEM differential equations for a given final porosity.
        for ii in range(len(porosity)):
            current_phi = porosity[ii]
            current_S_w = saturation[ii]
            
            # Calculate effective fluid bulk modulus (Kf) using Brie's model for the water-gas mixture.
            # Using a fixed Brie exponent of 3 here.
            # SUGGESTION: The Brie exponent could be a parameter of the DEMModel or this method.
            Kf = (K_water - K_air) * current_S_w**3 + K_air # Effective fluid bulk modulus [GPa]
            
            # Calculate Poisson's ratio (v) of the solid matrix.
            # v = (3*K_m - 2*G_m) / (2 * (3*K_m + G_m))
            v_matrix = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))
            
            # --- DEM Parameters (p, q, or similar, here b, c, d, g) ---
            # These parameters depend on the matrix properties (Poisson's ratio v_matrix)
            # and inclusion geometry (aspect_ratio α).
            # The specific forms for b, c, d, g are from a particular DEM solution (e.g., Wu, 1966; Mavko et al.).
            # These relate to Eshelby's tensor components or related shape factors for spheroidal inclusions.
            # b relates to bulk modulus change.
            b_param = 3 * np.pi * aspect_ratio * (1 - 2 * v_matrix) / (4 * (1 - v_matrix**2))
            # c, d, g relate to shear modulus change.
            # The 1/c and 1/d forms are noted as "Two-step calculation as in velDEM" (a specific software/code).
            c_param_inv_step1 = (3 + 8 * (1 - v_matrix) / (np.pi * aspect_ratio * (2 - v_matrix)))
            c_param = 1.0 / ( (1.0/5.0) * c_param_inv_step1 )

            d_param_inv_step1 = (1 + 8 * (1 - v_matrix) * (5 - v_matrix) / (3 * np.pi * aspect_ratio * (2 - v_matrix)))
            d_param = 1.0 / ( (1.0/5.0) * d_param_inv_step1 )

            g_param = np.pi * aspect_ratio / (2 * (1 - v_matrix))
            
            # --- Solve for Effective Bulk Modulus (Keff) ---
            # The DEM equation for Keff is often of the form:
            # (1 - φ) * dK_eff / dφ = (K_inclusion - K_eff) * P_factor
            # The algebraic form used here is a solution to such a differential equation.
            # (Keff - Kf) / (K_matrix - Kf) * (K_matrix / Keff)^(1/(1+b_param)) = (1 - current_phi)^(1/(1+b_param))
            # This needs to be solved for Keff.
            def equation_Keff(Keff_val_iter):
                if Keff_val_iter <= 0: return 1e6 # Penalize non-physical values.
                # Avoid division by zero if K_matrix == Kf
                if np.isclose(bulk_modulus, Kf):
                    # Special case or limit needs to be handled if K_matrix = K_fluid.
                    # This might imply the equation simplifies or DEM assumptions break.
                    # For now, assume they are different enough.
                    # If K_matrix = Kf, then Keff should also be Kf. The equation might become 0=0 or problematic.
                    # If Keff_val_iter is also Kf, this term is 1 or 0/0.
                    # (Keff - Kf) part dominates.
                    if np.isclose(Keff_val_iter, Kf): return 0.0 # This is a solution if K_matrix = Kf

                # (Keff - Kf) / (Km - Kf)
                term1_K = (Keff_val_iter - Kf) / (bulk_modulus - Kf)
                # (Km / Keff)^(1/(1+b))
                term2_K = (bulk_modulus / Keff_val_iter)**(1 / (1 + b_param))
                # (1 - φ)^(1/(1+b))
                rhs_K = (1 - current_phi)**(1 / (1 + b_param))
                return term1_K * term2_K - rhs_K
            
            # Solve for Keff using scipy.optimize.root with Levenberg-Marquardt ('lm') method.
            # Initial guess is the matrix bulk modulus.
            # SUGGESTION: Consider other root-finding methods or initial guesses if 'lm' fails or is slow.
            # `fsolve` could also be used here.
            result_K = root(equation_Keff, bulk_modulus, method='lm', tol=1e-7) # Added tolerance
            if result_K.success:
                Keff_results[ii] = result_K.x[0]
            else: # If solver fails to converge.
                Keff_results[ii] = np.nan # Store NaN on failure
                print(f"Warning: Root finding for Keff failed at index {ii} for porosity {current_phi:.3f}: {result_K.message}")
                # raise ValueError(f"Root finding for Keff failed at index {ii}: {result_K.message}") # Or raise error

            # --- Solve for Effective Shear Modulus (Geff) ---
            # Similar DEM-derived algebraic equation for Geff.
            # Geff / Gm * [ (1/Geff + c*g/(d*Kf)) / (1/Gm + c*g/(d*Kf)) ]^(1 - c/d) = (1 - φ)^(1/d)
            # This is more complex due to coupling with Kf (fluid bulk modulus, shear modulus of fluid is 0).
            def equation_Geff(Geff_val_iter):
                if Geff_val_iter <= 0: return 1e6 # Penalize non-physical values.
                # Term: (1/Geff + c*g/(d*Kf))
                term_num_G = (1 / Geff_val_iter + c_param * g_param / (d_param * Kf))
                # Term: (1/Gm + c*g/(d*Kf))
                term_den_G = (1 / shear_modulus + c_param * g_param / (d_param * Kf))
                # (num/den)^(1 - c/d)
                ratio_terms_G = (term_num_G / term_den_G)**(1 - c_param / d_param)
                # Geff / Gm
                factor_G = Geff_val_iter / shear_modulus
                # (1 - φ)^(1/d)
                rhs_G = (1 - current_phi)**(1 / d_param)
                return factor_G * ratio_terms_G - rhs_G

            # Solve for Geff using scipy.optimize.root. Initial guess is matrix shear modulus.
            result_G = root(equation_Geff, shear_modulus, method='lm', tol=1e-7) # Added tolerance
            if result_G.success:
                Geff_results[ii] = result_G.x[0]
            else: # If solver fails.
                Geff_results[ii] = np.nan # Store NaN
                print(f"Warning: Root finding for Geff failed at index {ii} for porosity {current_phi:.3f}: {result_G.message}")
                # raise ValueError(f"Root finding for Geff failed at index {ii}: {result_G.message}")

            # --- Calculate Total Density (ρ_total) ---
            # ρ_total = ρ_matrix * (1-φ) + ρ_fluid * φ
            # ρ_fluid = S_w * ρ_water + (1-S_w) * ρ_air
            rho_air = 1.225   # Density of air [kg/m³] at standard conditions.
                              # SUGGESTION: Make rho_air, rho_water parameters or class attributes.
            rho_water = 1000  # Density of water [kg/m³].
            rho_fluid_eff = current_S_w * rho_water + (1 - current_S_w) * rho_air # Effective fluid density.
            rho_total = mineral_density * (1 - current_phi) + rho_fluid_eff * current_phi # Total bulk density.
            
            # --- Calculate P-wave Velocity (Vp) ---
            # Vp = sqrt((K_eff + 4/3 * G_eff) / ρ_total)
            # Ensure Keff and Geff are in Pascals (Pa) from GPa.
            if not np.isnan(Keff_results[ii]) and not np.isnan(Geff_results[ii]): # Only if Keff, Geff solved successfully
                Vp_results[ii] = np.sqrt((Keff_results[ii] * 1e9 + (4.0/3.0) * Geff_results[ii] * 1e9) / rho_total)
            else:
                Vp_results[ii] = np.nan # Propagate NaN if moduli calculation failed.

        return Keff_results, Geff_results, Vp_results
        
        return Keff, Geff, Vp


class HertzMindlinModel(BaseVelocityModel):
    """
    Hertz-Mindlin model and Hashin-Shtrikman bounds for seismic velocity in porous rocks.
    """
    
    def __init__(self, 
               critical_porosity: float = 0.4,  # φ_c (phi_c)
               coordination_number: float = 4.0 # C or N
               ):
        """
        Initialize Hertz-Mindlin model.
        The Hertz-Mindlin model calculates the effective bulk and shear moduli of a random pack
        of identical elastic spheres under confining pressure. It's a contact mechanics theory.
        This model is often used to estimate the properties of the dry rock frame at critical porosity (φ_c),
        which is the porosity at which grains just form a load-bearing framework (e.g., ~0.36-0.4 for spheres).
        These properties (K_HM, G_HM) are then used as endpoints in Hashin-Shtrikman bounds or other models
        to predict properties at other porosities.

        Args:
            critical_porosity (float, optional): Critical porosity (φ_c) of the granular material.
                                                 This is the porosity at which the granular pack starts to
                                                 become load-bearing. Defaults to 0.4.
            coordination_number (float, optional): Average number of contacts per grain (C).
                                                   For random sphere packs, C can range from ~6-9.
                                                   Defaults to 4.0, which seems low for typical random packs.
                                                   Commonly used values are often higher (e.g., 6-8).
                                                   SUGGESTION: Default coordination_number might need review.
                                                   Dvorkin and Nur (1996) use C that depends on (1-phi_c).
        """
        super().__init__() # Call base class constructor.
        self.critical_porosity = critical_porosity # φ_c
        self.coordination_number = coordination_number # C
    
    def calculate_velocity(self, 
                         porosity: np.ndarray,       # φ, array of porosity values
                         saturation: np.ndarray,     # S_w, array of water saturation values
                         bulk_modulus: float,        # K_m, bulk modulus of solid matrix [GPa]
                         shear_modulus: float,       # G_m, shear modulus of solid matrix [GPa]
                         mineral_density: float,     # ρ_m, density of solid matrix [kg/m³]
                         depth: float = 1.0          # Depth in meters, for effective pressure calculation
                         ) -> Tuple[np.ndarray, np.ndarray]: # Vp_high, Vp_low
        """
        Calculate P-wave velocity bounds (high and low) for porous rocks using a combination of
        Hertz-Mindlin theory for the dry frame at critical porosity, Hashin-Shtrikman bounds
        for interpolating/extrapolating moduli to other porosities, and Gassmann's equation
        (via Brie's model for fluid modulus) for fluid substitution.

        The model distinguishes two porosity regimes:
        1. φ < φ_c (critical porosity): Modified Hashin-Shtrikman bounds are used to interpolate
           between the properties of the solid mineral (at φ=0) and the Hertz-Mindlin pack (at φ=φ_c).
        2. φ >= φ_c: A suspension model (e.g., Reuss bound for K, or other effective medium theory for G)
           is used to model the rock as a suspension of grains in fluid, or by extrapolating
           from the Hertz-Mindlin point. The code here seems to use a specific formulation.

        Args:
            porosity (np.ndarray): Array of porosity values (φ), fraction from 0 to 1.
            saturation (np.ndarray): Array of water saturation values (S_w), fraction from 0 to 1.
            bulk_modulus (float): Bulk modulus of the solid mineral matrix (K_m) [GPa].
            shear_modulus (float): Shear modulus of the solid mineral matrix (G_m) [GPa].
            mineral_density (float): Density of the solid mineral matrix (ρ_m) [kg/m³].
            depth (float, optional): Depth below surface [m], used to estimate effective confining pressure.
                                     Defaults to 1.0 m.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Vp_high: Array of P-wave velocities for the upper Hashin-Shtrikman bound [m/s].
                - Vp_low: Array of P-wave velocities for the lower Hashin-Shtrikman bound [m/s].
        """
        # --- Calculate Properties of the Solid Matrix ---
        # Poisson's ratio (v_m) of the solid mineral matrix.
        v_matrix = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))
        
        # --- Estimate Effective Confining Pressure (P_eff) ---
        # P_eff = (ρ_matrix - ρ_fluid_pore) * g * depth
        # Assuming ρ_fluid_pore is density of water (1000 kg/m³).
        # This is a simplified hydrostatic effective pressure. Units: GPa.
        # SUGGESTION: Pore pressure could be an input if known, for more accurate P_eff.
        # (mineral_density - 1000 kg/m^3) * 9.8 m/s^2 * depth_m / 1e9 (to convert Pa to GPa)
        effective_pressure_GPa = (mineral_density - 1000.0) * 9.8 * depth / 1e9
        # Ensure pressure is non-negative, as HM theory requires P > 0.
        effective_pressure_GPa = max(effective_pressure_GPa, 1e-6) # Small positive floor for pressure [GPa]
        
        # --- Hertz-Mindlin Model for Dry Frame at Critical Porosity (φ_c) ---
        # These are K_HM and G_HM, the effective bulk and shear moduli of the grain pack
        # at critical porosity, subjected to effective pressure P.
        C = self.coordination_number
        phi_c = self.critical_porosity
        
        # K_HM = [ C^2 * (1-φ_c)^2 * G_m^2 * P_eff / (18 * π^2 * (1-v_m)^2) ]^(1/3)
        K_HM = (C**2 * (1 - phi_c)**2 * shear_modulus**2 * effective_pressure_GPa /
               (18 * np.pi**2 * (1 - v_matrix)**2))**(1/3)
        
        # G_HM = ( (5-4v_m)/(5*(2-v_m)) ) * [ (3 * C^2 * (1-φ_c)^2 * G_m^2 * P_eff) / (2 * π^2 * (1-v_m)^2) ]^(1/3)
        # The code form: G_HM = ((5-4v)/(10-2v)) * [term]^(1/3) is equivalent to ( (5-4v)/(5*(2-v)) )
        # Note: The original code's G_HM formula has a slight difference in constant factor compared to some texts.
        # ( (5-4v)/(10-2v) ) is ( (5-4v_m) / (5*(2-v_m)) ) simplified by factor of 2.
        # Or, ( (5-4v_m)/(10(1-v_m)) ) * K_HM * (3*(1-v_m)/(2-v_m)) - check source (e.g. Mavko et al.)
        # The term `((5-4v)/(10-2v))` is equivalent to `( (5-4v_m) / (5*(2-v_m)) )` if it's `(5-4v)/(2*(5-v))`.
        # A common form is G_HM = ( (2+3v_m-v_m^2) / (5*(2-v_m)) ) * K_HM * (if K_HM is related to normal stiffness)
        # Or more directly: G_HM = Factor * ( (C^2 * (1-φ_c)^2 * G_m^2 * P_eff) / (..))^1/3)
        # The factor (5-4v)/(10-2v) seems to be from Dvorkin and Nur (1996) for G_HM/K_HM ratio.
        # Let's assume the formula is as intended from a specific source.
        G_HM = ((5 - 4 * v_matrix) / (10 - 2 * v_matrix) * # This is (5-4v_m)/(5*(2-v_m)) * (5/2) = (5-4v_m)/(2*(2-v_m))
               ((3 * C**2 * (1 - phi_c)**2 * shear_modulus**2) * effective_pressure_GPa /
                (2 * np.pi**2 * (1 - v_matrix)**2)))**(1/3)
        
        # Initialize output arrays for P-wave velocities (high and low bounds).
        Vp_high_bound = np.zeros_like(porosity, dtype=float)
        Vp_low_bound = np.zeros_like(porosity, dtype=float)
        
        # Create BrieModel instance for calculating effective fluid modulus later (for Gassmann).
        # Using default Brie exponent of 3.0.
        # SUGGESTION: Brie exponent could be a parameter.
        brie_fluid_model = BrieModel()
        
        # --- Calculate Velocities for each Porosity/Saturation Point ---
        for i in range(len(porosity)):
            current_phi = porosity[i]
            current_S_w = saturation[i]

            K_eff_H_dry, G_eff_H_dry = 0, 0 # Effective moduli for High bound, dry rock
            K_eff_L_dry, G_eff_L_dry = 0, 0 # Effective moduli for Low bound, dry rock

            if current_phi < phi_c:
                # --- Regime 1: Porosity < Critical Porosity (φ < φ_c) ---
                # Use modified Hashin-Shtrikman bounds to interpolate dry rock moduli
                # between the solid mineral (at φ=0) and the Hertz-Mindlin aggregate (at φ=φ_c).
                # K_m, G_m are moduli of mineral phase (φ=0 point).
                # K_HM, G_HM are moduli of Hertz-Mindlin pack (φ=φ_c point).
                
                # Lower Hashin-Shtrikman bound for K_dry (using G_HM as shear modulus of softer phase for K bound)
                # K_eff_L_dry = [ (φ/φ_c) / (K_HM + 4/3 G_HM) + (1 - φ/φ_c) / (K_m + 4/3 G_HM) ]^(-1) - 4/3 G_HM
                K_eff_L_dry = (current_phi / phi_c / (K_HM + (4.0/3.0) * G_HM) +
                             (1 - current_phi / phi_c) / (bulk_modulus + (4.0/3.0) * G_HM))**(-1) - (4.0/3.0) * G_HM
                
                # Lower Hashin-Shtrikman bound for G_dry
                # zeta_L = G_HM/6 * (9*K_HM + 8*G_HM)/(K_HM + 2*G_HM) (shear modulus of comparison material for G_HM)
                zeta_L_G_HM = G_HM / 6.0 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM)
                G_eff_L_dry = (current_phi / phi_c / (G_HM + zeta_L_G_HM) +
                             (1 - current_phi / phi_c) / (shear_modulus + zeta_L_G_HM))**(-1) - zeta_L_G_HM
                
                # Upper Hashin-Shtrikman bound for K_dry (using G_m as shear modulus of stiffer phase for K bound)
                # K_eff_H_dry = [ (φ/φ_c) / (K_HM + 4/3 G_m) + (1 - φ/φ_c) / (K_m + 4/3 G_m) ]^(-1) - 4/3 G_m
                K_eff_H_dry = (current_phi / phi_c / (K_HM + (4.0/3.0) * shear_modulus) +
                             (1 - current_phi / phi_c) / (bulk_modulus + (4.0/3.0) * shear_modulus))**(-1) - (4.0/3.0) * shear_modulus
                
                # Upper Hashin-Shtrikman bound for G_dry
                # zeta_H = G_m/6 * (9*K_m + 8*G_m)/(K_m + 2*G_m) (shear modulus of comparison material for G_m)
                zeta_H_Gm = shear_modulus / 6.0 * (9 * bulk_modulus + 8 * shear_modulus) / (bulk_modulus + 2 * shear_modulus)
                G_eff_H_dry = (current_phi / phi_c / (G_HM + zeta_H_Gm) + # Note: Original code uses G_HM here, typically should be K_HM for HS bound consistency if G_HM is the "inclusion". This formula is specific.
                             (1 - current_phi / phi_c) / (shear_modulus + zeta_H_Gm))**(-1) - zeta_H_Gm
                
                # --- Fluid Substitution using Gassmann (via Brie for K_fluid) ---
                # K_sat_H and K_sat_L are saturated bulk moduli for upper and lower bounds respectively.
                K_sat_H = brie_fluid_model.calculate_saturated_modulus(
                    K_eff_H_dry, bulk_modulus, current_phi, current_S_w
                )
                K_sat_L = brie_fluid_model.calculate_saturated_modulus(
                    K_eff_L_dry, bulk_modulus, current_phi, current_S_w
                )
                # Shear moduli (G_eff_H_dry, G_eff_L_dry) are assumed unchanged by fluid saturation (Gassmann's assumption).
                G_sat_H = G_eff_H_dry
                G_sat_L = G_eff_L_dry
                
            else: # current_phi >= phi_c
                # --- Regime 2: Porosity >= Critical Porosity (φ >= φ_c) ---
                # Model as a suspension, or extrapolate from the Hertz-Mindlin point (φ_c).
                # The formulas here represent a specific model for this regime (e.g., modified Reuss for K, specific for G).
                # This is effectively the "soft sand" or "unconsolidated" model part.
                # K_dry for suspension (Reuss-like average for K, using K_HM, G_HM as one end-member at phi_c, and fluid properties at phi=1)
                # The term `4/3 * G_HM` in denominator suggests it's related to M_HM (P-wave modulus of HM pack).
                # Effective K_dry: ( (1-φ)/(1-φ_c) / (K_HM + 4/3 G_HM) + (φ-φ_c)/(1-φ_c) / (K_fluid_dry + 4/3 G_fluid_dry) )^-1 - 4/3 G_fluid_dry
                # Here, K_fluid_dry=0, G_fluid_dry=0 (for dry pores). So it simplifies.
                # The code implies K_fluid_dry is not zero but related to 4/3*G_HM. This is specific.
                # K_eff_dry_suspension = ((1 - current_phi) / (1 - phi_c) / (K_HM + (4.0/3.0) * G_HM) +
                #                       (current_phi - phi_c) / (1 - phi_c) / ((4.0/3.0) * G_HM) # Assuming K_pore_dry=0
                #                       )**(-1) - (4.0/3.0) * G_HM
                # This seems to be a specific form of Hashin-Shtrikman lower bound for a material with porosity (phi-phi_c)/(1-phi_c)
                # made of fluid (K=0, G=0) in a host K_HM, G_HM. Let's follow the code's exact formulation.
                K_eff_dry_suspension = ((1 - current_phi) / (1 - phi_c) / (K_HM + (4.0/3.0) * G_HM) +
                                      (current_phi - phi_c) / (1 - phi_c) / ((4.0/3.0) * G_HM) # K_inclusion=0
                                      )**(-1) - (4.0/3.0) * G_HM
                
                # Effective G_dry for suspension
                # zeta_susp = G_HM / 6.0 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM) # Using G_HM for comparison material
                # G_eff_dry_suspension = ((1 - current_phi) / (1 - phi_c) / (G_HM + zeta_susp) +
                #                       (current_phi - phi_c) / (1 - phi_c) / zeta_susp # G_inclusion=0
                #                       )**(-1) - zeta_susp
                # The code has `onede` which is `zeta_L_G_HM` from the phi < phi_c block.
                # This implies using the same comparison modulus zeta for this regime too.
                zeta_susp = G_HM / 6.0 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM) # Renaming `onede` for clarity
                G_eff_dry_suspension = ((1 - current_phi) / (1 - phi_c) / (G_HM + zeta_susp) + # Original code uses 'onede'
                                      (current_phi - phi_c) / (1 - phi_c) / zeta_susp # G_inclusion=0
                                      )**(-1) - zeta_susp
                
                # Fluid substitution for this single effective dry frame.
                # For this regime, high and low bounds for K_sat are considered same.
                K_sat_H = K_sat_L = brie_fluid_model.calculate_saturated_modulus(
                    K_eff_dry_suspension, bulk_modulus, current_phi, current_S_w
                )
                # Shear modulus is assumed same for high and low bounds in this regime.
                G_sat_H = G_sat_L = G_eff_dry_suspension
            
            # --- Calculate Total Density (ρ_total) ---
            # Same calculation as in DEMModel.
            rho_air_const = 1.225   # Density of air [kg/m³]
            rho_water_const = 1000  # Density of water [kg/m³]
            rho_fluid_effective = current_S_w * rho_water_const + (1 - current_S_w) * rho_air_const
            rho_total_bulk = mineral_density * (1 - current_phi) + rho_fluid_effective * current_phi
            
            # --- Calculate P-wave Velocities (Vp_high, Vp_low) ---
            # Vp = sqrt((K_sat + 4/3 * G_sat) / ρ_total_bulk)
            # Ensure moduli are in Pascals (Pa) from GPa.
            Vp_high_bound[i] = np.sqrt((K_sat_H * 1e9 + (4.0/3.0) * G_sat_H * 1e9) / rho_total_bulk)
            Vp_low_bound[i] = np.sqrt((K_sat_L * 1e9 + (4.0/3.0) * G_sat_L * 1e9) / rho_total_bulk)
        
        return Vp_high, Vp_low


def VRH_model(f=[0.35, 0.25, 0.2, 0.125, 0.075],
             K=[55.4, 36.6, 75.6, 46.7, 50.4],
             G=[28.1, 45, 25.6, 23.65, 27.4],
             rho=[2560, 2650, 2630, 2540, 3050]):
    """
    Implements the Voigt-Reuss-Hill (VRH) mixing model to estimate the effective bulk modulus (Km),
    shear modulus (Gm), and density (rho_b) of a composite material made from various minerals.

    Parameters:
    f (list): Fraction of each mineral in the composite (must sum to 1).
    K (list): Bulk modulus of each mineral (GPa).
    G (list): Shear modulus of each mineral (GPa).
    rho (list): Density of each mineral (kg/m^3).

    Returns:
    Km (float): Effective bulk modulus of the composite material (GPa).
    Gm (float): Effective shear modulus of the composite material (GPa).
    rho_b (float): Effective density of the composite material (kg/m^3).
    """
    # This is a standalone function, not part of the VRHModel class, but implements the same logic.
    # It directly calculates effective moduli and density from mineral properties.
    # Physical basis: Voigt-Reuss-Hill average for elastic moduli.
    # Voigt: Assumes uniform strain. K_V = Σ(f_i * K_i), G_V = Σ(f_i * G_i)
    # Reuss: Assumes uniform stress. 1/K_R = Σ(f_i / K_i), 1/G_R = Σ(f_i / G_i)
    # Hill (VRH): Arithmetic average of Voigt and Reuss bounds. K_VRH = (K_V + K_R)/2.
    # Density: Simple weighted average by volume fraction. ρ_eff = Σ(f_i * ρ_i).

    # Convert input lists to numpy arrays for vectorized calculations.
    f_arr = np.array(f)        # f_i: volume fractions
    K_arr = np.array(K)        # K_i: bulk moduli of components [GPa]
    G_arr = np.array(G)        # G_i: shear moduli of components [GPa]
    rho_arr = np.array(rho)    # ρ_i: densities of components [kg/m³]

    # --- Calculate Effective Bulk Modulus (Km_vrh) using VRH average ---
    K_voigt = np.sum(f_arr * K_arr)       # Voigt upper bound for K
    K_reuss = 1.0 / np.sum(f_arr / K_arr) # Reuss lower bound for K
    Km_vrh = 0.5 * (K_voigt + K_reuss)    # VRH average for K [GPa]

    # --- Calculate Effective Shear Modulus (Gm_vrh) using VRH average ---
    G_voigt = np.sum(f_arr * G_arr)       # Voigt upper bound for G
    G_reuss = 1.0 / np.sum(f_arr / G_arr) # Reuss lower bound for G
    Gm_vrh = 0.5 * (G_voigt + G_reuss)    # VRH average for G [GPa]

    # --- Calculate Effective Density (rho_b_eff) ---
    # This is a simple volume-weighted average of component densities.
    rho_b_eff = np.sum(f_arr * rho_arr) # Effective density [kg/m³]

    return Km_vrh, Gm_vrh, rho_b_eff


def satK(Keff, Km, phi, Sat):
    """
    Calculate the saturated bulk modulus (K_sat) based on Brie's equation.

    Parameters:
    Keff (float): Effective bulk modulus of the dry rock (GPa).
    Km (float): Bulk modulus of the matrix (GPa).
    phi (float): Porosity of the rock.
    Sat (float): Saturation level of the fluid in the pores.

    Returns:
    float: Saturated bulk modulus (K_sat) [GPa].
    """
    # This standalone function calculates saturated bulk modulus (K_sat).
    # It first calculates effective fluid bulk modulus (Kfl) using Brie's equation,
    # then uses this Kfl in a specific form of Gassmann's equation.
    # This is similar to `BrieModel.calculate_saturated_modulus`.
    # SUGGESTION: This function duplicates logic from the BrieModel class.
    # Consider using an instance of BrieModel here or consolidating.

    # --- Brie's Equation for Effective Fluid Modulus (Kfl) ---
    # Parameters for Brie's equation:
    Kw_fluid = 2.0  # Bulk modulus of water (K_w) [GPa]. Default value.
                    # SUGGESTION: Make Kw_fluid and Ka_fluid parameters of this function.
    Ka_fluid = 0.01 # Bulk modulus of air/gas (K_a) [GPa]. Default value.
    brie_exponent = 3 # Brie's exponent 'e'. Default value.
                      # SUGGESTION: Make brie_exponent a parameter.

    # Ensure saturation (Sat) is clipped to [0,1] for physical meaning in Brie's eq.
    S_clipped = np.clip(Sat, 0.0, 1.0)
    Kfl = (Kw_fluid - Ka_fluid) * S_clipped ** brie_exponent + Ka_fluid # Effective fluid bulk modulus [GPa]

    # --- Gassmann's Equation (specific algebraic form) ---
    # K_sat = [ Keff/(Km - Keff) + Kfl/(phi*(Km-Kfl)) ] / [ 1 + Keff/(Km - Keff) + Kfl/(phi*(Km-Kfl)) ] * Km
    # Where:
    # Keff = Effective bulk modulus of the dry rock frame [GPa]
    # Km   = Bulk modulus of the solid mineral matrix [GPa]
    # phi  = Porosity (fraction)
    # Kfl  = Effective bulk modulus of the pore fluid [GPa] (calculated above)
    # K_sat= Saturated bulk modulus of the rock [GPa]
    # SUGGESTION: Verify this algebraic form of Gassmann's equation. It appears different from the most common textbook forms.
    # It might be a specific rearrangement or subject to certain assumptions.
    # Numerical stability should be checked for edge cases (e.g., Km close to Keff, phi close to 0).

    # Calculate intermediate terms for clarity and to avoid division by zero if possible.
    # Denominator term for Keff part: (Km - Keff)
    den1 = Km - Keff
    # Denominator term for Kfl part: phi * (Km - Kfl)
    den2 = phi * (Km - Kfl)

    # Handle potential division by zero or unstable conditions.
    if np.isclose(den1, 0.0) or np.isclose(den2, 0.0) or np.isclose(phi, 0.0):
        # If Km=Keff (no effective porosity for frame stiffness) or phi=0, Ksat should be Km.
        # If Km=Kfl (fluid has same stiffness as matrix), formula might simplify or be problematic.
        # A robust implementation would handle these edge cases based on physical limits.
        # For example, if phi is very close to 0, K_sat should approach Km.
        if np.isclose(phi, 0.0): return Km
        # If other problematic divisions, result might be ill-defined by this specific formula.
        # Returning NaN or raising an error might be appropriate.
        # print(f"Warning: Potential instability in satK calculation (den1={den1}, den2={den2}, phi={phi}).")
        # Fallback or error based on how these cases should be handled.
        # For now, proceed with calculation which might result in inf/NaN if not careful.
        pass

    term_A = Keff / den1             # Keff / (Km - Keff)
    term_B = Kfl / den2             # Kfl / (phi * (Km - Kfl))

    K_sat_calculated = (term_A + term_B) * Km / (1 + term_A + term_B)

    return K_sat_calculated


def velDEM(phi, Km, Gm, rho_b, Sat, alpha):
    """
    Calculate effective bulk modulus (Keff), shear modulus (Geff), and P-wave velocity (Vp)
    for a rock with varying porosity (phi) based on the DEM model, taking into account
    the saturation (Sat) and the crack aspect ratio (alpha).

    Parameters:
    phi (np.array): Array of porosities.
    Km (float): Initial bulk modulus of the material (GPa).
    Gm (float): Initial shear modulus of the material (GPa).
    rho_b (float): Density of the solid phase (kg/m^3).
    Sat (float): Saturation level of the fluid in the cracks (0 to 1, where 1 is fully saturated).
    alpha (float): Crack aspect ratio (α). Typically << 1 for cracks.

    Returns:
    Keff_results (np.array): Effective bulk modulus for each porosity value [GPa].
    Geff_results (np.array): Effective shear modulus for each porosity value [GPa].
    Vp_results (np.array): P-wave velocity for each porosity value [m/s].
    """
    # This is a standalone function implementing the DEM model, similar to the DEMModel class.
    # Physical Basis: Differential Effective Medium theory. Pores are added incrementally to the matrix.
    # The effective moduli of the (matrix + previous pores) become the matrix for the next increment of pores.
    # This leads to differential equations whose solutions (often algebraic for simple pore shapes) give K_eff, G_eff.
    # Assumes pores are spheroidal inclusions with aspect ratio α.
    # Fluid properties within pores are calculated using Brie's model.
    # Equations for b, c, d, g parameters are specific to DEM formulations (e.g., Wu 1966, Mavko et al. for Eshelby-based DEM).
    # SUGGESTION: Consolidate with DEMModel class or clarify distinction.

    # Initialize output arrays, same size as the input porosity array `phi`.
    Keff_results = np.zeros_like(phi, dtype=float)
    Geff_results = np.zeros_like(phi, dtype=float)
    Vp_results = np.zeros_like(phi, dtype=float)

    # --- Fluid Properties (Constants) ---
    # These are hardcoded here. For more flexibility, they could be function parameters.
    Kw_fluid = 2.0  # Bulk modulus of water (K_w) [GPa].
    Ka_fluid = 0.01 # Bulk modulus of air/gas (K_a) [GPa].
    # Brie exponent is implicitly 3 in the Kf calculation below.

    # Iterate through each porosity and saturation value provided in the input arrays.
    for ii in range(len(phi)):
        current_phi = phi[ii]   # Porosity for current point
        current_Sat = Sat[ii] # Water saturation for current point

        # --- Calculate Effective Fluid Bulk Modulus (Kf) ---
        # Using Brie's equation with a fixed exponent of 3.
        Kf = (Kw_fluid - Ka_fluid) * current_Sat**3 + Ka_fluid # [GPa]

        # --- Calculate Poisson's Ratio (v_matrix) of the Solid Matrix ---
        # v_matrix = (3*Km - 2*Gm) / (2*(3*Km + Gm))
        v_matrix = (3 * Km - 2 * Gm) / (2 * (3 * Km + Gm))

        # --- DEM Shape Parameters (b, c, d, g) ---
        # These parameters depend on matrix Poisson's ratio (v_matrix) and pore aspect ratio (alpha).
        # They are derived from Eshelby's tensor solutions for spheroidal inclusions and used in DEM equations.
        b_param = 3 * np.pi * alpha * (1 - 2 * v_matrix) / (4 * (1 - v_matrix**2))

        # The 1/c and 1/d forms are noted as "Two-step calculation as in velDEM" (a specific software).
        # This means c_calc = 1 / ( (1/5) * ( ... ) )
        c_calc_inv_term = (1.0/5.0) * (3 + 8 * (1 - v_matrix) / (np.pi * alpha * (2 - v_matrix)))
        c_param = 1.0 / c_calc_inv_term

        d_calc_inv_term = (1.0/5.0) * (1 + 8 * (1 - v_matrix) * (5 - v_matrix) / (3 * np.pi * alpha * (2 - v_matrix)))
        d_param = 1.0 / d_calc_inv_term

        g_param = np.pi * alpha / (2 * (1 - v_matrix))

        # --- Solve for Effective Bulk Modulus (Keff) Numerically ---
        # The DEM equation for Keff (often a solution to the DEM differential form):
        # (Keff - Kf) / (Km - Kf) * (Km / Keff)^(1/(1+b_param)) - (1 - current_phi)^(1/(1+b_param)) = 0
        def equation_Keff_root(Keff_iter):
            if Keff_iter <= 0: # Constraint for physical meaningfulness
                return 1e6  # Return a large penalty if Keff is non-positive
            # Handle potential division by zero if Km == Kf or Keff_iter == 0
            if np.isclose(Km, Kf): # If matrix and fluid moduli are same, Keff should be same.
                return Keff_iter - Kf
            if np.isclose(Keff_iter, 0.0): Keff_iter = 1e-9 # Avoid division by zero in (Km/Keff_iter)

            term1 = (Keff_iter - Kf) / (Km - Kf)
            term2 = (Km / Keff_iter)**(1.0 / (1.0 + b_param))
            rhs = (1.0 - current_phi)**(1.0 / (1.0 + b_param))
            return term1 * term2 - rhs

        # Use scipy.optimize.root (with Levenberg-Marquardt) to find Keff. Initial guess is Km.
        result_Keff_solve = root(equation_Keff_root, Km, method='lm', tol=1e-7)
        if result_Keff_solve.success:
            Keff_results[ii] = result_Keff_solve.x[0]
        else:
            Keff_results[ii] = np.nan # Mark as NaN if solver fails
            print(f"Warning: Root finding for Keff failed at index {ii} for porosity {current_phi:.3f}: {result_Keff_solve.message}")
            # raise ValueError(f"Root finding for Keff failed at index {ii}: {result_Keff_solve.message}") # Alternative: raise error
            
        # --- Solve for Effective Shear Modulus (Geff) Numerically ---
        # DEM equation for Geff:
        # Geff/Gm * [ (1/Geff + c*g/(d*Kf)) / (1/Gm + c*g/(d*Kf)) ]^(1-c/d) - (1-phi)^(1/d) = 0
        # Note: Shear modulus of fluid is 0. Kf (fluid bulk modulus) appears due to coupling in DEM for non-spherical pores.
        def equation_Geff_root(Geff_iter):
            if Geff_iter <= 0: # Constraint
                return 1e6
            if np.isclose(Kf, 0.0): Kf_eff_for_G = 1e-9 # Avoid division by zero if Kf is effectively zero
            else: Kf_eff_for_G = Kf

            term_num_G = (1.0 / Geff_iter + c_param * g_param / (d_param * Kf_eff_for_G))
            term_den_G = (1.0 / Gm + c_param * g_param / (d_param * Kf_eff_for_G))
            # Avoid division by zero or log of zero if term_den_G is zero or ratio is negative
            if np.isclose(term_den_G, 0.0) or (term_num_G / term_den_G < 0 and (1.0 - c_param / d_param) % 1 != 0):
                return 1e6 # Penalize problematic conditions

            ratio_terms_G = (term_num_G / term_den_G)**(1.0 - c_param / d_param)
            factor_G = Geff_iter / Gm
            rhs_G = (1.0 - current_phi)**(1.0 / d_param)
            return factor_G * ratio_terms_G - rhs_G

        # Use scipy.optimize.root to find Geff. Initial guess is Gm.
        result_Geff_solve = root(equation_Geff_root, Gm, method='lm', tol=1e-7)
        if result_Geff_solve.success:
            Geff_results[ii] = result_Geff_solve.x[0]
        else:
            Geff_results[ii] = np.nan # Mark as NaN
            print(f"Warning: Root finding for Geff failed at index {ii} for porosity {current_phi:.3f}: {result_Geff_solve.message}")
            # raise ValueError(f"Root finding for Geff failed at index {ii}: {result_Geff_solve.message}")

        # --- Calculate Total Bulk Density (rhototal) ---
        # ρ_total = ρ_matrix * (1-φ) + ρ_fluid * φ
        # ρ_fluid = S_w * ρ_water + (1-S_w) * ρ_air
        rho_air_const = 1.225  # Density of air [kg/m³] at standard conditions.
                               # SUGGESTION: Make rho_air_const, rho_water_const parameters.
        rho_water_const = 1000   # Density of water [kg/m³].
        rho_fluid_eff = current_Sat * rho_water_const + (1.0 - current_Sat) * rho_air_const # Effective fluid density
        rhototal = rho_b * (1.0 - current_phi) + rho_fluid_eff * current_phi # Total bulk density

        # --- Calculate P-wave Velocity (Vp) ---
        # Vp = sqrt((K_eff + 4/3 * G_eff) / ρ_total)
        # Moduli Keff, Geff are in GPa, so convert to Pa ( * 1e9 ) for velocity in m/s.
        if not np.isnan(Keff_results[ii]) and not np.isnan(Geff_results[ii]): # Check if Keff, Geff were successfully computed
            Vp_results[ii] = np.sqrt((Keff_results[ii] * 1e9 + (4.0 / 3.0) * Geff_results[ii] * 1e9) / rhototal)
        else:
            Vp_results[ii] = np.nan # Propagate NaN if Keff or Geff failed

    return Keff_results, Geff_results, Vp_results


def vel_porous(phi, Km, Gm, rho_b, Sat, depth=1):
    """
    Calculate P-wave velocity (Vp) for a rock with varying porosity (phi) based on the 
    Hertz-Mindlin model and Hashin-Shtrikman bounds, taking into account the saturation (Sat).

    Parameters:
    phi (np.array): Array of porosities.
    Km (float): Bulk modulus of the solid phase (GPa).
    Gm (float): Shear modulus of the solid phase (GPa).
    rho_b (float): Density of the solid phase (kg/m^3).
    Sat (float): Saturation level of the fluid in the pores (0 to 1, where 1 is fully saturated).
    depth (float): depth for pressure estimation (m)

    Returns:
    Vp_h (np.array): P-wave velocity for each porosity value (upper Hashin-Shtrikman bound) (m/s).
    Vp_l_results (np.array): P-wave velocity for each porosity value (lower Hashin-Shtrikman bound) [m/s].
    """
    # This is a standalone function, similar to HertzMindlinModel class.
    # It combines Hertz-Mindlin contact theory for a critical porosity state (φ_c)
    # with Hashin-Shtrikman bounds for other porosities, and Gassmann fluid substitution (via satK function).
    # Physical Basis:
    # 1. Hertz-Mindlin: Calculates K_HM, G_HM for a pack of spheres at critical porosity φ_c under pressure P.
    #    Assumes elastic spheres, small deformations at contacts.
    # 2. Hashin-Shtrikman (HS) Bounds: Provide upper and lower bounds for effective moduli of a two-phase composite.
    #    Here, used to model rock with porosity φ < φ_c as a composite of solid mineral (phase 1) and
    #    Hertz-Mindlin material (phase 2, representing porous fraction at φ_c).
    #    For φ >= φ_c, a suspension model is used (grains in fluid, or extrapolation from φ_c).
    # 3. Gassmann's Equation (via `satK`): Used for fluid substitution to get saturated moduli from dry frame moduli.
    # Parameters for Hertz-Mindlin part:
    C_coord_num = 4.0      # Coordination number (average contacts per grain). Default in function.
                           # SUGGESTION: Make C and phi_c parameters of the function.
    phi_critical = 0.4   # Critical porosity (φ_c). Default in function.

    # --- Calculate Properties of Solid Matrix ---
    v_matrix = (3 * Km - 2 * Gm) / (2 * (3 * Km + Gm))  # Poisson's ratio of solid mineral (v_m)

    # --- Estimate Effective Confining Pressure (P_eff) ---
    # P_eff = (ρ_matrix - ρ_fluid_pore) * g * depth. Using ρ_water = 1000 kg/m³. Units: GPa.
    P_eff_GPa = (rho_b - 1000.0) * 9.8 * depth / 1e9
    P_eff_GPa = max(P_eff_GPa, 1e-6) # Ensure positive pressure for Hertz-Mindlin calculations [GPa].

    # --- Hertz-Mindlin Moduli at Critical Porosity (K_HM, G_HM) ---
    # K_HM = [ C^2 * (1-φ_c)^2 * G_m^2 * P_eff / (18*π^2*(1-v_m)^2) ]^(1/3)
    K_HM = (C_coord_num**2 * (1 - phi_critical)**2 * Gm**2 * P_eff_GPa /
           (18 * np.pi**2 * (1 - v_matrix)**2))**(1.0/3.0)
    # G_HM = Factor * K_HM or Factor * [ C^2 * ... P_eff ]^(1/3)
    # The factor (5-4v)/(10-2v) from original code for G_HM.
    G_HM_factor_term = ( (3 * C_coord_num**2 * (1 - phi_critical)**2 * Gm**2 * P_eff_GPa) /
                         (2 * np.pi**2 * (1 - v_matrix)**2) )**(1.0/3.0)
    G_HM = (5 - 4 * v_matrix) / (10 - 2 * v_matrix) * G_HM_factor_term

    # Initialize lists to store results for each porosity value.
    Vp_h_list = [] # Upper bound Vp
    Vp_l_list = [] # Lower bound Vp

    # Iterate over each input porosity value.
    for ii in range(len(phi)):
        current_phi = phi[ii]
        current_Sat = Sat[ii] # Corresponding saturation for this porosity point.

        K_sat_H_eff, K_sat_L_eff = 0, 0 # Effective saturated bulk moduli (High, Low)
        G_eff_H_final, G_eff_L_final = 0, 0 # Effective shear moduli (High, Low) (assumed same as dry for Gassmann)

        if current_phi < phi_critical:
            # --- Regime 1: Porosity φ < Critical Porosity φ_c ---
            # Use modified Hashin-Shtrikman bounds for dry frame moduli (K_dry, G_dry).
            # The "phases" are:
            #   Phase 1: Solid mineral (Km, Gm) with volume fraction (1 - current_phi/phi_critical).
            #   Phase 2: Hertz-Mindlin aggregate (K_HM, G_HM) with volume fraction (current_phi/phi_critical).

            # Effective moduli of the dry rock frame (K_eff_L_dry, G_eff_L_dry, K_eff_H_dry, G_eff_H_dry)
            # Lower bound for K_dry (K_eff_L_dry)
            # HS Lower Bound: K_L = K1 + f2 / (1/(K2-K1) + f1/(K1+4/3 G1)) where K1,G1 are softer phase. Here K_HM, G_HM.
            # The formula used seems a direct application of HS bounds for a two-phase composite.
            # K_eff_L = ( (φ/φ_c)/(K_HM + 4/3 G_HM) + (1-φ/φ_c)/(Km + 4/3 G_HM) )^(-1) - 4/3 G_HM
            K_eff_L_dry = (current_phi / phi_critical / (K_HM + (4.0/3.0) * G_HM) +
                         (1 - current_phi / phi_critical) / (Km + (4.0/3.0) * G_HM)
                         )**(-1) - (4.0/3.0) * G_HM

            # Parameter zeta for shear modulus bounds (related to bulk and shear moduli of the reference phase).
            # For lower bound G_dry, reference is the softer phase (Hertz-Mindlin material).
            zeta_L = G_HM / 6.0 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM) # `onede` in original code
            G_eff_L_dry = (current_phi / phi_critical / (G_HM + zeta_L) +
                         (1 - current_phi / phi_critical) / (Gm + zeta_L)
                         )**(-1) - zeta_L

            # Upper bound for K_dry (K_eff_H_dry)
            # K_eff_H = ( (φ/φ_c)/(K_HM + 4/3 Gm) + (1-φ/φ_c)/(Km + 4/3 Gm) )^(-1) - 4/3 Gm
            K_eff_H_dry = (current_phi / phi_critical / (K_HM + (4.0/3.0) * Gm) + # Using Gm as the stiffer component's shear for K bound
                         (1 - current_phi / phi_critical) / (Km + (4.0/3.0) * Gm)
                         )**(-1) - (4.0/3.0) * Gm

            # For upper bound G_dry, reference is the stiffer phase (solid mineral).
            zeta_H = Gm / 6.0 * (9 * Km + 8 * Gm) / (Km + 2 * Gm) # `onede` in original for upper
            G_eff_H_dry = (current_phi / phi_critical / (G_HM + zeta_H) + # Original code uses G_HM here.
                                                                       # HS theory would use K2,G2 for the "inclusion" phase.
                                                                       # This specific formulation needs to be checked against source.
                         (1 - current_phi / phi_critical) / (Gm + zeta_H)
                         )**(-1) - zeta_H

            # --- Fluid Substitution (Gassmann via satK function) ---
            # Calculate saturated bulk moduli using the dry frame moduli bounds.
            K_sat_H_eff = satK(K_eff_H_dry, Km, current_phi, current_Sat) # Saturated K for High bound dry frame
            K_sat_L_eff = satK(K_eff_L_dry, Km, current_phi, current_Sat) # Saturated K for Low bound dry frame
            # Shear moduli are assumed unchanged by fluid (Gassmann's assumption).
            G_eff_H_final = G_eff_H_dry
            G_eff_L_final = G_eff_L_dry
        else: # current_phi >= phi_critical
            # --- Regime 2: Porosity φ >= Critical Porosity φ_c (Suspension Regime) ---
            # Model as a suspension of grains in fluid, or an extrapolation from the φ_c point.
            # The dry frame moduli (K_eff_dry_susp, G_eff_dry_susp) are calculated first.
            # These formulas effectively treat the Hertz-Mindlin aggregate (at φ_c) as the "solid" phase,
            # and additional porosity (φ - φ_c) is added, filled with a "dry fluid" (K=0, G=0 for dry frame).
            # This is akin to Hashin-Shtrikman lower bound for adding zero-moduli pores to the HM aggregate.

            # Effective K_dry for the suspension part (relative to HM point)
            # ( (1-φ')/(K_HM + 4/3 G_HM) + φ'/(K_fluid_dry + 4/3 G_fluid_dry) )^(-1) - 4/3 G_fluid_dry
            # where φ' = (φ - φ_c) / (1 - φ_c) is porosity relative to the HM aggregate.
            # K_fluid_dry=0, G_fluid_dry=0.
            # The formula in code: ((1-φ)/(1-φ_c) / (K_HM+4/3G_HM) + (φ-φ_c)/(1-φ_c) / (4/3G_HM) )^(-1) - 4/3G_HM
            # This seems to be Reuss-type average for P-wave modulus, then converted back to K.
            # This specific formulation should be referenced.
            K_eff_dry_susp = ((1 - current_phi) / (1 - phi_critical) / (K_HM + (4.0/3.0) * G_HM) +
                            (current_phi - phi_critical) / (1 - phi_critical) / ((4.0/3.0) * G_HM) # K_inclusion_dry=0
                           )**(-1) - (4.0/3.0) * G_HM

            # Effective G_dry for the suspension part
            zeta_susp_ref = G_HM / 6.0 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM) # `onede` in original code.
            G_eff_dry_susp = ((1 - current_phi) / (1 - phi_critical) / (G_HM + zeta_susp_ref) +
                            (current_phi - phi_critical) / (1 - phi_critical) / zeta_susp_ref # G_inclusion_dry=0
                           )**(-1) - zeta_susp_ref

            # Fluid substitution. For this regime, high and low bounds are considered the same.
            K_sat_H_eff = K_sat_L_eff = satK(K_eff_dry_susp, Km, current_phi, current_Sat)
            G_eff_H_final = G_eff_L_final = G_eff_dry_susp

        # --- Calculate Total Bulk Density (rhototal) ---
        rho_air_const = 1.225  # Density of air [kg/m³]
        rho_water_const = 1000 # Density of water [kg/m³]
        rho_fluid_eff = current_Sat * rho_water_const + (1.0 - current_Sat) * rho_air_const
        rhototal_bulk = rho_b * (1.0 - current_phi) + rho_fluid_eff * current_phi

        # --- Calculate P-wave Velocities (High and Low Bounds) ---
        # Vp = sqrt((K_sat + 4/3 G_sat) / ρ_total)
        # Moduli are in GPa, convert to Pa (*1e9) for velocity in m/s.
        Vp_h_list.append(np.sqrt((K_sat_H_eff * 1e9 + (4.0/3.0) * G_eff_H_final * 1e9) / rhototal_bulk))
        Vp_l_list.append(np.sqrt((K_sat_L_eff * 1e9 + (4.0/3.0) * G_eff_L_final * 1e9) / rhototal_bulk))

    return np.array(Vp_h_list), np.array(Vp_l_list)