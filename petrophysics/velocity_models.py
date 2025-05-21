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
        pass
    
    def calculate_velocity(self, **kwargs) -> np.ndarray:
        """
        Calculate seismic velocity from rock properties.
        
        Args:
            **kwargs: Rock properties specific to each model
            
        Returns:
            Seismic velocity values (Vp, Vs, or both)
        """
        raise NotImplementedError("Velocity calculation must be implemented in derived classes")


class VRHModel(BaseVelocityModel):
    """
    Voigt-Reuss-Hill (VRH) mixing model for effective elastic properties of composites.
    """
    
    def __init__(self):
        """Initialize VRH model."""
        super().__init__()
    
    def calculate_properties(self, 
                           fractions: List[float],
                           bulk_moduli: List[float],
                           shear_moduli: List[float],
                           densities: List[float]) -> Tuple[float, float, float]:
        """
        Calculate effective elastic properties using the VRH model.
        
        Args:
            fractions: Volume fractions of each mineral (must sum to 1)
            bulk_moduli: Bulk moduli of each mineral (GPa)
            shear_moduli: Shear moduli of each mineral (GPa)
            densities: Densities of each mineral (kg/m³)
            
        Returns:
            Effective bulk modulus (GPa), effective shear modulus (GPa), and effective density (kg/m³)
        """
        # Convert inputs to numpy arrays
        f = np.array(fractions)
        K = np.array(bulk_moduli)
        G = np.array(shear_moduli)
        rho = np.array(densities)
        
        # Verify that fractions sum to 1
        if not np.isclose(np.sum(f), 1.0):
            raise ValueError("Volume fractions must sum to 1")
        
        # Calculate Voigt and Reuss averages for bulk modulus
        K_voigt = np.sum(f * K)
        K_reuss = 1.0 / np.sum(f / K)
        
        # Calculate Voigt and Reuss averages for shear modulus
        G_voigt = np.sum(f * G)
        G_reuss = 1.0 / np.sum(f / G)
        
        # Calculate Hill (VRH) averages
        K_vrh = 0.5 * (K_voigt + K_reuss)
        G_vrh = 0.5 * (G_voigt + G_reuss)
        
        # Calculate effective density (simple weighted average)
        rho_eff = np.sum(f * rho)
        
        return K_vrh, G_vrh, rho_eff
    
    def calculate_velocity(self, 
                         fractions: List[float],
                         bulk_moduli: List[float],
                         shear_moduli: List[float],
                         densities: List[float]) -> Tuple[float, float]:
        """
        Calculate P-wave and S-wave velocities using the VRH model.
        
        Args:
            fractions: Volume fractions of each mineral (must sum to 1)
            bulk_moduli: Bulk moduli of each mineral (GPa)
            shear_moduli: Shear moduli of each mineral (GPa)
            densities: Densities of each mineral (kg/m³)
            
        Returns:
            P-wave velocity (m/s) and S-wave velocity (m/s)
        """
        # Calculate effective properties
        K_eff, G_eff, rho_eff = self.calculate_properties(
            fractions, bulk_moduli, shear_moduli, densities
        )
        
        # Convert GPa to Pa for velocity calculations
        K_eff_pa = K_eff * 1e9
        G_eff_pa = G_eff * 1e9
        
        # Calculate P-wave velocity
        Vp = np.sqrt((K_eff_pa + 4/3 * G_eff_pa) / rho_eff)
        
        # Calculate S-wave velocity
        Vs = np.sqrt(G_eff_pa / rho_eff)
        
        return Vp, Vs


class BrieModel:
    """
    Brie's model for calculating the effective bulk modulus of a partially saturated medium.
    """
    
    def __init__(self, exponent: float = 3.0):
        """
        Initialize Brie's model.
        
        Args:
            exponent: Brie's exponent (default: 3.0)
        """
        self.exponent = exponent
    
    def calculate_fluid_modulus(self, 
                              saturation: float,
                              water_modulus: float = 2.0,
                              gas_modulus: float = 0.01) -> float:
        """
        Calculate effective fluid bulk modulus using Brie's equation.
        
        Args:
            saturation: Water saturation (0 to 1)
            water_modulus: Bulk modulus of water (GPa, default: 2.0)
            gas_modulus: Bulk modulus of gas (GPa, default: 0.01)
            
        Returns:
            Effective fluid bulk modulus (GPa)
        """
        return (water_modulus - gas_modulus) * saturation ** self.exponent + gas_modulus
    
    def calculate_saturated_modulus(self, 
                                  dry_modulus: float,
                                  mineral_modulus: float,
                                  porosity: float,
                                  saturation: float,
                                  water_modulus: float = 2.0,
                                  gas_modulus: float = 0.01) -> float:
        """
        Calculate the saturated bulk modulus based on Brie's equation.
        
        Args:
            dry_modulus: Bulk modulus of the dry rock (GPa)
            mineral_modulus: Bulk modulus of the mineral matrix (GPa)
            porosity: Porosity of the rock
            saturation: Water saturation (0 to 1)
            water_modulus: Bulk modulus of water (GPa, default: 2.0)
            gas_modulus: Bulk modulus of gas (GPa, default: 0.01)
            
        Returns:
            Saturated bulk modulus (GPa)
        """
        # Calculate effective fluid modulus
        fluid_modulus = self.calculate_fluid_modulus(
            saturation, water_modulus, gas_modulus
        )
        
        # Apply Gassmann's equation
        numerator = dry_modulus / (mineral_modulus - dry_modulus) + fluid_modulus / (porosity * (mineral_modulus - fluid_modulus))
        denominator = 1 + (dry_modulus / (mineral_modulus - dry_modulus) + fluid_modulus / (porosity * (mineral_modulus - fluid_modulus)))
        
        return (numerator / denominator) * mineral_modulus


class DEMModel(BaseVelocityModel):
    """
    Differential Effective Medium (DEM) model for calculating elastic properties
    and seismic velocities of porous rocks.
    """
    
    def __init__(self):
        """Initialize DEM model."""
        super().__init__()
    
    def calculate_velocity(self, 
                         porosity: np.ndarray,
                         saturation: np.ndarray,
                         bulk_modulus: float,
                         shear_modulus: float,
                         mineral_density: float,
                         aspect_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate P-wave velocity using the DEM model.
        
        Args:
            porosity: Porosity values (array)
            saturation: Saturation values (array)
            bulk_modulus: Initial bulk modulus of the solid matrix (GPa)
            shear_modulus: Initial shear modulus of the solid matrix (GPa)
            mineral_density: Density of the solid matrix (kg/m³)
            aspect_ratio: Aspect ratio of pores (default: 0.1)
            
        Returns:
            Effective bulk modulus (GPa), effective shear modulus (GPa), and P-wave velocity (m/s)
        """
        # Initialize arrays for calculated values
        Keff = np.zeros(len(porosity))
        Geff = np.zeros(len(porosity))
        Vp = np.zeros(len(porosity))
        
        # Constants for fluid moduli
        Kw = 2.0  # Bulk modulus of water (GPa)
        Ka = 0.01  # Bulk modulus of air (GPa)
        
        # Process each porosity/saturation point
        for ii in range(len(porosity)):
            # Brie's equation for fluid bulk modulus
            Kf = (Kw - Ka) * saturation[ii]**3 + Ka
            
            # Calculate Poisson's ratio
            v = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))
            
            # Calculate DEM parameters - matching velDEM implementation
            b = 3 * np.pi * aspect_ratio * (1 - 2 * v) / (4 * (1 - v**2))
            c = 1 / 5 * ((3 + 8 * (1 - v) / (np.pi * aspect_ratio * (2 - v))))
            c = 1 / c  # Two-step calculation as in velDEM
            d = 1 / 5 * ((1 + 8 * (1 - v) * (5 - v) / (3 * np.pi * aspect_ratio * (2 - v))))
            d = 1 / d  # Two-step calculation as in velDEM
            g = np.pi * aspect_ratio / (2 * (1 - v))
            
            # Define equation for effective bulk modulus
            def equation_Keff(Keff_val):
                if Keff_val <= 0:
                    return 1e6  # Return large value for invalid K_eff
                return (Keff_val - Kf) / (bulk_modulus - Kf) * (bulk_modulus / Keff_val)**(1 / (1 + b)) - (1 - porosity[ii])**(1 / (1 + b))
            
            # Solve for effective bulk modulus

            result_K = root(equation_Keff, bulk_modulus, method='lm')
            if result_K.success:
                Keff[ii] = result_K.x[0]
            else:
                raise ValueError(f"Root finding for Keff failed at index {ii}: {result_K.message}")

            
            # Define equation for effective shear modulus
            def equation_Geff(Geff_val):
                if Geff_val <= 0:
                    return 1e6
                return Geff_val / shear_modulus * ((1 / Geff_val + c * g / (d * Kf)) / (1 / shear_modulus + c * g / (d * Kf)))**(1 - c / d) - (1 - porosity[ii])**(1 / d)
            
            # Solve for effective shear modulus

            result_G = root(equation_Geff, shear_modulus, method='lm')
            if result_G.success:
                Geff[ii] = result_G.x[0]
            else:
                raise ValueError(f"Root finding for Geff failed at index {ii}: {result_G.message}")

            
            # Calculate total density considering porosity and saturation
            rho_a = 1.225   # Density of air (kg/m³)
            rho_w = 1000    # Density of water (kg/m³)
            rhototal = mineral_density * (1 - porosity[ii]) + (saturation[ii] * rho_w + (1 - saturation[ii]) * rho_a) * porosity[ii]
            
            # Calculate P-wave velocity
            Vp[ii] = np.sqrt((Keff[ii] + 4/3 * Geff[ii]) * 1e9 / rhototal)
        
        return Keff, Geff, Vp


class HertzMindlinModel(BaseVelocityModel):
    """
    Hertz-Mindlin model and Hashin-Shtrikman bounds for seismic velocity in porous rocks.
    """
    
    def __init__(self, 
               critical_porosity: float = 0.4, 
               coordination_number: float = 4.0):
        """
        Initialize Hertz-Mindlin model.
        
        Args:
            critical_porosity: Critical porosity at which the rock loses matrix connectivity
            coordination_number: Average number of contacts per grain
        """
        super().__init__()
        self.critical_porosity = critical_porosity
        self.coordination_number = coordination_number
    
    def calculate_velocity(self, 
                         porosity: np.ndarray,
                         saturation: np.ndarray,
                         bulk_modulus: float,
                         shear_modulus: float,
                         mineral_density: float,
                         depth: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate P-wave velocity for porous rocks.
        
        Args:
            porosity: Porosity values (array)
            saturation: Saturation values (array)
            bulk_modulus: Bulk modulus of the solid matrix (GPa)
            shear_modulus: Shear modulus of the solid matrix (GPa)
            mineral_density: Density of the solid matrix (kg/m³)
            depth: Depth for pressure estimation (m, default: 1.0)
            
        Returns:
            Tuple of high-bound P-wave velocity (m/s) and low-bound P-wave velocity (m/s)
        """
        # Poisson's ratio
        v = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))
        
        # Pressure estimation
        P = (mineral_density - 1000) * 9.8 * depth / 1e9  # GPa
        
        # Hertz-Mindlin model at critical porosity
        C = self.coordination_number
        phi_c = self.critical_porosity
        
        # Calculate Hertz-Mindlin bulk modulus
        K_HM = (C**2 * (1 - phi_c)**2 * shear_modulus**2 / 
               (18 * np.pi**2 * (1 - v)**2) * P)**(1/3)
        
        # Calculate Hertz-Mindlin shear modulus
        G_HM = ((5 - 4 * v) / (10 - 2 * v) * 
               ((3 * C**2 * (1 - phi_c)**2 * shear_modulus**2) * P / 
                (2 * np.pi**2 * (1 - v)**2)))**(1/3)
        
        # Initialize velocity arrays
        Vp_high = np.zeros(len(porosity))
        Vp_low = np.zeros(len(porosity))
        
        # Create Brie model for fluid substitution
        brie_model = BrieModel()
        
        # Calculate velocities for each porosity/saturation point
        for i in range(len(porosity)):
            if porosity[i] < phi_c:
                # Below critical porosity, use modified Hashin-Shtrikman bounds
                
                # Lower bound for bulk modulus
                K_eff_L = (porosity[i] / phi_c / (K_HM + 4/3 * G_HM) + 
                          (1 - porosity[i] / phi_c) / (bulk_modulus + 4/3 * G_HM))**(-1) - 4/3 * G_HM
                
                # Lower bound for shear modulus
                zeta = G_HM / 6 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM)
                G_eff_L = (porosity[i] / phi_c / (G_HM + zeta) + 
                          (1 - porosity[i] / phi_c) / (shear_modulus + zeta))**(-1) - zeta
                
                # Upper bound for bulk modulus
                K_eff_H = (porosity[i] / phi_c / (K_HM + 4/3 * shear_modulus) + 
                          (1 - porosity[i] / phi_c) / (bulk_modulus + 4/3 * shear_modulus))**(-1) - 4/3 * shear_modulus
                
                # Upper bound for shear modulus
                zeta = shear_modulus / 6 * (9 * bulk_modulus + 8 * shear_modulus) / (bulk_modulus + 2 * shear_modulus)
                G_eff_H = (porosity[i] / phi_c / (G_HM + zeta) + 
                          (1 - porosity[i] / phi_c) / (shear_modulus + zeta))**(-1) - zeta
                
                # Apply fluid substitution
                K_sat_H = brie_model.calculate_saturated_modulus(
                    K_eff_H, bulk_modulus, porosity[i], saturation[i]
                )
                K_sat_L = brie_model.calculate_saturated_modulus(
                    K_eff_L, bulk_modulus, porosity[i], saturation[i]
                )
                
            else:
                # Above critical porosity, suspension model
                K_eff = ((1 - porosity[i]) / (1 - phi_c) / (K_HM + 4/3 * G_HM) + 
                        (porosity[i] - phi_c) / (1 - phi_c) / (4/3 * G_HM))**(-1) - 4/3 * G_HM
                
                zeta = G_HM / 6 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM)
                G_eff = ((1 - porosity[i]) / (1 - phi_c) / (G_HM + zeta) + 
                        (porosity[i] - phi_c) / (1 - phi_c) / zeta)**(-1) - zeta
                
                K_sat_H = K_sat_L = brie_model.calculate_saturated_modulus(
                    K_eff, bulk_modulus, porosity[i], saturation[i]
                )
                G_eff_H = G_eff_L = G_eff
            
            # Calculate total density
            rho_air = 1.225
            rho_water = 1000
            rho_total = mineral_density * (1 - porosity[i]) + (saturation[i] * rho_water + (1 - saturation[i]) * rho_air) * porosity[i]
            
            # Calculate velocities
            Vp_high[i] = np.sqrt((K_sat_H + 4/3 * G_eff_H) * 1e9 / rho_total)
            Vp_low[i] = np.sqrt((K_sat_L + 4/3 * G_eff_L) * 1e9 / rho_total)
        
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
    # Convert input lists to numpy arrays for vectorized operations
    f = np.array(f)
    K = np.array(K)
    G = np.array(G)
    rho = np.array(rho)

    # Calculate effective bulk modulus (Km) using the VRH model
    # Voigt average for bulk modulus is the weighted sum of the moduli
    # Reuss average for bulk modulus is the harmonic mean of the moduli
    # VRH average is the arithmetic mean of Voigt and Reuss averages
    Km = 1 / 2 * (np.sum(f * K) + (np.sum(f / K)) ** (-1))

    # Calculate effective shear modulus (Gm) using the VRH model
    # Similar to bulk modulus but for shear moduli
    Gm = 1 / 2 * (np.sum(f * G) + (np.sum(f / G)) ** (-1))

    # Calculate effective density (rho_b) as the weighted sum of the densities
    rho_b = np.sum(f * rho)

    return Km, Gm, rho_b


def satK(Keff, Km, phi, Sat):
    """
    Calculate the saturated bulk modulus (K_sat) based on Brie's equation.

    Parameters:
    Keff (float): Effective bulk modulus of the dry rock (GPa).
    Km (float): Bulk modulus of the matrix (GPa).
    phi (float): Porosity of the rock.
    Sat (float): Saturation level of the fluid in the pores.

    Returns:
    float: Saturated bulk modulus (GPa).
    """
    Kw = 2  # Bulk modulus of water (GPa)
    Ka = 0.01  # Bulk modulus of air (GPa)
    Kfl = (Kw - Ka) * Sat ** 3 + Ka  # Effective fluid bulk modulus
    K_sat = (Keff / (Km - Keff) + Kfl / (phi * (Km - Kfl))) * Km / (1 + (Keff / (Km - Keff) + Kfl / (phi * (Km - Kfl))))
    return K_sat


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
    alpha (float): Crack aspect ratio.

    Returns:
    Keff1 (np.array): Effective bulk modulus for each porosity value (GPa).
    Geff1 (np.array): Effective shear modulus for each porosity value (GPa).
    Vp (np.array): P-wave velocity for each porosity value (m/s).
    """
    # Initialize arrays for the calculated values
    Keff1 = np.zeros(len(phi))
    Geff1 = np.zeros(len(phi))
    Vp = np.zeros(len(phi))

    # Constants for fluid moduli
    Kw = 2  # Bulk modulus of water (GPa)
    Ka = 0.01  # Bulk modulus of air (GPa)

    for ii in range(len(phi)):
        Kf = (Kw - Ka) * Sat[ii]**3 + Ka  # Brie's equation for fluid bulk modulus
        
        # Poisson's ratio for the rock
        v = (3 * Km - 2 * Gm) / (2 * (3 * Km + Gm))

        # Parameters b, c, and d for equations
        b = 3 * np.pi * alpha * (1 - 2 * v) / (4 * (1 - v**2))
        c = 1 / 5 * ((3 + 8 * (1 - v) / (np.pi * alpha * (2 - v))))
        c = 1 / c
        d = 1 / 5 * ((1 + 8 * (1 - v) * (5 - v) / (3 * np.pi * alpha * (2 - v))))
        d = 1 / d
        g = np.pi * alpha / (2 * (1 - v))
        
        # Solve for effective bulk modulus Keff using fsolve
        def equation_Keff(Keff):
            if Keff < 0:
                return 1e6  # Return a large value to force root away from invalid values
            return (Keff - Kf) / (Km - Kf) * (Km / Keff)**(1 / (1 + b)) - (1 - phi[ii])**(1 / (1 + b))

        # Use fsolve to find Keff
        result_Keff = root(equation_Keff, Km, method='lm')
        if result_Keff.success:
            Keff1[ii] = result_Keff.x[0]
        else:
            raise ValueError(f"Root finding for Keff failed at index {ii}: {result_Keff.message}")
            
        # Solve for effective shear modulus Geff using fsolve
        def equation_Geff(Geff):
            if Geff < 0:
                return 1e6
            return Geff / Gm * ((1 / Geff + c * g / (d * Kf)) / (1 / Gm + c * g / (d * Kf)))**(1 - c / d) - (1 - phi[ii])**(1 / d)

        # Use root to find Geff
        result_Geff = root(equation_Geff, Gm, method='lm')
        if result_Geff.success:
            Geff1[ii] = result_Geff.x[0]
        else:
            raise ValueError(f"Root finding for Geff failed at index {ii}: {result_Geff.message}")

        # Total density calculation considering porosity and saturation
        rho_a = 1.225  # Density of air (kg/m^3)
        rho_w = 1000   # Density of water (kg/m^3)
        rhototal = rho_b * (1 - phi[ii]) + (Sat[ii] * rho_w + (1 - Sat[ii]) * rho_a) * phi[ii]

        # P-wave velocity calculation
        Vp[ii] = np.sqrt((Keff1[ii] + 4 / 3 * Geff1[ii]) * 1e9 / rhototal)   # in m/s

    return Keff1, Geff1, Vp


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
    Vp_l (np.array): P-wave velocity for each porosity value (lower Hashin-Shtrikman bound) (m/s).
    """
    # Hertz-Mindlin model in critical porosity
    C = 4  # The number of contacts
    phi_c = 0.4  # The critical porosity
    v = (3 * Km - 2 * Gm) / (2 * (3 * Km + Gm))  # Poisson's ratio
    P = (rho_b - 1000) * 9.8 * depth / (1e9)
    K_HM = (C ** 2 * (1 - phi_c) ** 2 * Gm ** 2 / (18 * np.pi ** 2 * (1 - v) ** 2) * P) ** (1 / 3)
    G_HM = (5 - 4 * v) / (10 - 2 * v) * ((3 * C ** 2 * (1 - phi_c) ** 2 * Gm ** 2) * P / (2 * np.pi ** 2 * (1 - v) ** 2)) ** (1 / 3)

    Vp_h = []
    Vp_l = []

    for ii in range(len(phi)):
        if phi[ii] < phi_c:
            Keff_L = (phi[ii] / phi_c / (K_HM + 4 / 3 * G_HM) + (1 - phi[ii] / phi_c) / (Km + 4 / 3 * G_HM)) ** (-1) - 4 / 3 * G_HM
            onede = G_HM / 6 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM)
            Geff_L = (phi[ii] / phi_c / (G_HM + onede) + (1 - phi[ii] / phi_c) / (Gm + onede)) ** (-1) - onede

            Keff_H = (phi[ii] / phi_c / (K_HM + 4 / 3 * Gm) + (1 - phi[ii] / phi_c) / (Km + 4 / 3 * Gm)) ** (-1) - 4 / 3 * Gm
            onede = Gm / 6 * (9 * Km + 8 * Gm) / (Km + 2 * Gm)
            Geff_H = (phi[ii] / phi_c / (G_HM + onede) + (1 - phi[ii] / phi_c) / (Gm + onede)) ** (-1) - onede

            Sh = satK(Keff_H, Km, phi[ii], Sat[ii])
            Sl = satK(Keff_L, Km, phi[ii], Sat[ii])
        else:
            Keff = ((1 - phi[ii]) / (1 - phi_c) / (K_HM + 4 / 3 * G_HM) + (phi[ii] - phi_c) / (1 - phi_c) / (4 / 3 * G_HM)) ** (-1) - 4 / 3 * G_HM
            onede = G_HM / 6 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM)
            Geff = ((1 - phi[ii]) / (1 - phi_c) / (G_HM + onede) + (phi[ii] - phi_c) / (1 - phi_c) / (onede)) ** (-1) - onede

            Sh = satK(Keff, Km, phi[ii], Sat[ii])
            Sl = satK(Keff, Km, phi[ii], Sat[ii])
            Geff_L = Geff
            Geff_H = Geff

        rho_a = 1.225
        rho_w = 1000
        rhototal = rho_b * (1 - phi[ii]) + (Sat[ii] * rho_w + (1 - Sat[ii]) * rho_a) * phi[ii]
        Vp_h.append(np.sqrt((Sh + 4 / 3 * Geff_H) * 1e9 / rhototal))
        Vp_l.append(np.sqrt((Sl + 4 / 3 * Geff_L) * 1e9 / rhototal))

    return np.array(Vp_h), np.array(Vp_l)
