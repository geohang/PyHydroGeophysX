"""
Seismic velocity models for relating rock properties to elastic wave velocities.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# water content to velocity
# ---------------------------------------------------------------------------
def water_content_to_velocity(water_content: np.ndarray, 
                               v_dry: float = 3500.0,
                               v_sat: float = 4500.0,
                               porosity: Union[float, np.ndarray] = 0.3,
                               model: str = 'linear') -> np.ndarray:
    """
    Convert water content to seismic P-wave velocity using empirical relationships.
    
    These are simplified mixing rules, not calibrated site predictions.
    Saturation is clipped to [0, 1] and output velocity to [200, 8000] m/s.
    
    Args:
        water_content: Volumetric water content (θ), typically 0-porosity
        v_dry: Dry velocity for the linear model only (m/s), default 3500.
        v_sat: Saturated velocity for the linear model only (m/s), default 4500.
        porosity: Porosity values (φ), can be scalar or array
        model: Velocity model type:
            - 'linear': Linear interpolation between dry and saturated velocities
            - 'wyllie': Wyllie time-average equation
            - 'raymer': Raymer-Hunt-Gardner equation
            
    Returns:
        P-wave velocity array (m/s)
    """
    # Calculate saturation from water content and porosity
    # S = θ / φ
    saturation = np.clip(water_content / porosity, 0.0, 1.0)
    
    if model == 'linear':
        # Linear interpolation between dry and saturated velocities
        # Vp = V_dry + S * (V_sat - V_dry)
        velocity = v_dry + saturation * (v_sat - v_dry)
        
    elif model == 'wyllie':
        # Wyllie time-average equation
        # 1/Vp = φ*S/V_water + φ*(1-S)/V_air + (1-φ)/V_mineral
        v_water = 1500.0  # P-wave velocity in water (m/s)
        v_air = 340.0     # P-wave velocity in air (m/s)
        v_mineral = 6000.0  # P-wave velocity in mineral (m/s)
        
        # Fluid velocity (weighted by saturation)
        v_fluid = 1.0 / (saturation / v_water + (1 - saturation) / v_air + 1e-10)
        
        # Time-average
        slowness = porosity / v_fluid + (1 - porosity) / v_mineral
        velocity = 1.0 / (slowness + 1e-10)
        
    elif model == 'raymer':
        # Raymer-Hunt-Gardner equation (simplified)
        # Vp = (1 - φ)^2 * V_mineral + φ * V_fluid
        v_water = 1500.0
        v_air = 340.0
        v_mineral = 6000.0
        
        v_fluid = saturation * v_water + (1 - saturation) * v_air
        velocity = (1 - porosity)**2 * v_mineral + porosity * v_fluid
        
    else:
        raise ValueError(f"Unknown velocity model: {model}. Use 'linear', 'wyllie', or 'raymer'.")
    
    # Clip to physically reasonable range
    velocity = np.clip(velocity, 200.0, 8000.0)
    
    return velocity


def velocity_to_water_content(velocity: np.ndarray,
                             v_dry: float = 3500.0,
                             v_sat: float = 4500.0,
                             porosity: Union[float, np.ndarray] = 0.3,
                             model: str = 'linear') -> np.ndarray:
    """Invert the empirical relationship used by water_content_to_velocity.

    Supports linear, Wyllie and Raymer models with the same fixed constituent
    velocities as the forward function. Porosity must be known. Observations
    outside the model's saturation range, or at the forward function's clipping
    limits (200/8000 m/s), are rejected rather than silently assigned 0/1
    saturation. Hertz-Mindlin and DEM require a separately calibrated inverse.
    """
    v, phi = np.broadcast_arrays(np.asarray(velocity, dtype=float),
                                 np.asarray(porosity, dtype=float))
    if (not np.isfinite(v).all() or not np.isfinite(phi).all()
            or np.any((phi <= 0.) | (phi > 1.))):
        raise ValueError("Velocity and porosity must be finite, with 0 < porosity <= 1.")
    if np.any((v <= 200.) | (v >= 8000.)):
        raise ValueError("Velocity at or beyond the forward clipping limits cannot be uniquely inverted.")
    model = str(model).lower()
    if model == 'linear':
        if not np.isfinite([v_dry, v_sat]).all() or v_dry == v_sat:
            raise ValueError("Linear inverse requires finite, distinct v_dry and v_sat.")
        saturation = (v - v_dry) / (v_sat - v_dry)
    elif model == 'wyllie':
        # Include the two numerical epsilons used by the forward slowness rule.
        dry_slowness = phi * (1. / 340. + 1e-10) + (1. - phi) / 6000. + 1e-10
        saturation = (1. / v - dry_slowness) / (phi * (1. / 1500. - 1. / 340.))
    elif model == 'raymer':
        dry_velocity = (1. - phi) ** 2 * 6000. + phi * 340.
        saturation = (v - dry_velocity) / (phi * (1500. - 340.))
    else:
        raise ValueError("Empirical inverse model must be 'linear', 'wyllie', or 'raymer'.")
    if np.any((saturation < -1e-10) | (saturation > 1. + 1e-10)):
        raise ValueError("Velocity lies outside the supplied model's physical saturation range.")
    return phi * np.clip(saturation, 0., 1.)


# ---------------------------------------------------------------------------
# Base Velocity Model
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# VRHModel
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Brie Model
# ---------------------------------------------------------------------------
def _gassmann_bulk_modulus(dry_modulus, mineral_modulus, porosity, fluid_modulus):
    """Gassmann substitution in GPa, including dry-fluid and mineral endpoints.

    Algebraically equivalent to Kdry + (1-Kdry/Km)**2 /
    (phi/Kfluid + (1-phi)/Km - Kdry/Km**2), without division by Kfluid
    or Km-Kdry. All public substitution routes use this implementation.
    """
    kd, km, phi, kf = np.broadcast_arrays(*(
        np.asarray(value, dtype=float)
        for value in (dry_modulus, mineral_modulus, porosity, fluid_modulus)))
    alpha = 1 - kd / km
    denominator = phi + (alpha - phi) * kf / km
    # A pore-free solid has Kdry=Km, hence alpha=denominator=0 and no
    # substitution increment. Do not hide other singular, inconsistent inputs.
    mineral_limit = (alpha == 0) & (denominator == 0)
    increment = np.divide(alpha**2 * kf, denominator,
                          out=np.zeros_like(phi), where=~mineral_limit)
    return kd + increment


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
        
        return _gassmann_bulk_modulus(dry_modulus, mineral_modulus, porosity, fluid_modulus)


# ---------------------------------------------------------------------------
# DEMModel
# ---------------------------------------------------------------------------
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
        from scipy.optimize import root

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


# ---------------------------------------------------------------------------
# Hertz Mindlin Model
# ---------------------------------------------------------------------------
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
            coordination_number: Average number of contacts per grain; default 4.
                About 9 is a common idealized random-packing choice, requiring
                calibration (Mavko et al., 2009, The Rock Physics Handbook).
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
                           depth: float = 1.0, *,
                           effective_pressure=None,
                           fluid_modulus=None,
                           fluid_density=None,
                           return_moduli: bool = False):
        """Calculate high/low P-wave bounds, optionally with elastic moduli.

        Parameters
        ----------
        porosity, saturation : array-like
            Fractions in [0, 1]. All numerical inputs broadcast together.
        bulk_modulus, shear_modulus : array-like
            Positive mineral moduli in GPa.
        mineral_density : array-like
            Positive mineral density in kg/m3.
        depth : array-like
            Depth in metres, default 1. Ignored when effective_pressure is set.
        effective_pressure : array-like, optional
            Nonnegative grain-contact pressure in Pa, replacing the historical
            (mineral_density - 1000) * 9.8 * depth estimate. This is distinct
            from the pore-fluid pressure passed to batzle_wang_brine.
        fluid_modulus : array-like, optional
            Liquid bulk modulus in GPa, replacing water's default 2.0 in
            Brie's liquid/gas mixture (gas=0.01 GPa, exponent=3). An explicit
            value <= 1e-12 GPa selects the dry-frame limit at any saturation,
            bypassing fluid mixing and substitution.
        fluid_density : array-like, optional
            Liquid density in kg/m3, default 1000; used in bulk density only.
            The historical pressure estimate is unchanged by this parameter.
        return_moduli : bool
            False preserves (Vp_high, Vp_low). True adds a dictionary with
            Vs_low, Vs_high, K_dry_low, K_dry_high, G_dry_low, G_dry_high,
            K_sat_low, K_sat_high and density. Moduli are GPa, velocities m/s,
            density kg/m3. Output arrays have the broadcast input shape.

        Notes
        -----
        Coordination number remains 4. A value around 9 is commonly used
        for idealized random packings, but should be calibrated to the rock
        (Mavko, Mukerji and Dvorkin, 2009, The Rock Physics Handbook, ch. 5).
        Zero effective pressure is handled by the zero-contact-stiffness
        limit. No change is made to the caller's saturation or porosity.
        """
        pressure = ((np.asarray(mineral_density) - 1000) * 9.8 * np.asarray(depth)
                    if effective_pressure is None else effective_pressure)
        inputs = np.broadcast_arrays(*[
            np.asarray(value, dtype=float) for value in (
                porosity, saturation, bulk_modulus, shear_modulus,
                mineral_density, pressure,
                2.0 if fluid_modulus is None else fluid_modulus,
                1000.0 if fluid_density is None else fluid_density)
        ])
        phi, sat, km, gm, rho_m, pressure, k_liquid, rho_liquid = inputs
        if any(np.any(~np.isfinite(value)) for value in inputs):
            raise ValueError("Model inputs must be finite.")
        if np.any((phi < 0) | (phi > 1) | (sat < 0) | (sat > 1)):
            raise ValueError("Porosity and saturation must be in [0, 1].")
        if np.any((km <= 0) | (gm <= 0) | (rho_m <= 0) | (rho_liquid <= 0)):
            raise ValueError("Mineral moduli and densities must be positive.")
        if np.any((pressure < 0) | (k_liquid < 0)):
            raise ValueError("Effective pressure and fluid modulus must be nonnegative.")
        phi_c, contacts = self.critical_porosity, self.coordination_number
        if not np.isfinite(phi_c) or not 0 < phi_c < 1:
            raise ValueError("critical_porosity must be in (0, 1).")
        if not np.isfinite(contacts) or contacts <= 0:
            raise ValueError("coordination_number must be positive.")

        v = (3 * km - 2 * gm) / (2 * (3 * km + gm))
        p_gpa = pressure / 1e9
        k_hm = (contacts**2 * (1 - phi_c)**2 * gm**2
                / (18 * np.pi**2 * (1 - v)**2) * p_gpa)**(1/3)
        # Mavko, Mukerji & Dvorkin (2009), RPHtools/hertzmind.m;
        # independently agrees with rockphypy.GM.hertzmindlin(..., f=1).
        # The prefactor MUST stay outside the cube root. Historical versions
        # used (10 - 2*v) and cubed-rooted the prefactor, overstating stiffness.
        shear_prefactor = (5 - 4*v) / (5 * (2 - v))
        shear_contact_term = (3 * contacts**2 * (1 - phi_c)**2 * gm**2 * p_gpa
                              / (2 * np.pi**2 * (1 - v)**2))
        g_hm = shear_prefactor * shear_contact_term**(1/3)
        z_hm = np.divide(g_hm * (9*k_hm + 8*g_hm), 6*(k_hm + 2*g_hm),
                         out=np.zeros_like(g_hm), where=(k_hm + 2*g_hm) > 0)
        z_m = gm / 6 * (9*km + 8*gm) / (km + 2*gm)
        k_low, k_high, g_low, g_high = [np.zeros_like(phi) for _ in range(4)]

        below = phi < phi_c
        f = phi[below] / phi_c
        # Positive contact stiffness uses the original modified HS equations.
        contact = below & (pressure > 0)
        fc = phi[contact] / phi_c
        k_low[contact] = (fc / (k_hm[contact] + 4/3*g_hm[contact])
                         + (1-fc) / (km[contact] + 4/3*g_hm[contact]))**-1 - 4/3*g_hm[contact]
        g_low[contact] = (fc / (g_hm[contact] + z_hm[contact])
                         + (1-fc) / (gm[contact] + z_hm[contact]))**-1 - z_hm[contact]
        k_high[below] = (f / (k_hm[below] + 4/3*gm[below])
                        + (1-f) / (km[below] + 4/3*gm[below]))**-1 - 4/3*gm[below]
        g_high[below] = (f / (g_hm[below] + z_m[below])
                        + (1-f) / (gm[below] + z_m[below]))**-1 - z_m[below]

        suspension = ~below & (pressure > 0)
        f = (1-phi[suspension]) / (1-phi_c)
        k_low[suspension] = (f / (k_hm[suspension] + 4/3*g_hm[suspension])
                            + (1-f) / (4/3*g_hm[suspension]))**-1 - 4/3*g_hm[suspension]
        g_low[suspension] = (f / (g_hm[suspension] + z_hm[suspension])
                            + (1-f) / z_hm[suspension])**-1 - z_hm[suspension]
        k_high[~below], g_high[~below] = k_low[~below], g_low[~below]
        # Exact endpoints avoid cancellation and the zero-pressure 0/0 limit.
        for value, mineral in ((k_low, km), (k_high, km), (g_low, gm), (g_high, gm)):
            value[phi == 0] = mineral[phi == 0]
            value[phi == 1] = 0.0

        k_fluid = BrieModel().calculate_fluid_modulus(sat, water_modulus=k_liquid)
        if fluid_modulus is not None:
            k_fluid = np.where(k_liquid <= 1e-12, 0.0, k_fluid)

        k_sat_low = _gassmann_bulk_modulus(k_low, km, phi, k_fluid)
        k_sat_high = _gassmann_bulk_modulus(k_high, km, phi, k_fluid)
        density = rho_m * (1-phi) + (sat*rho_liquid + (1-sat)*1.225) * phi
        vp_high = np.sqrt((k_sat_high + 4/3*g_high) * 1e9 / density)
        vp_low = np.sqrt((k_sat_low + 4/3*g_low) * 1e9 / density)
        if not return_moduli:
            return vp_high, vp_low
        return vp_high, vp_low, {
            "Vs_low": np.sqrt(g_low * 1e9 / density),
            "Vs_high": np.sqrt(g_high * 1e9 / density),
            "K_dry_low": k_low, "K_dry_high": k_high,
            "G_dry_low": g_low, "G_dry_high": g_high,
            "K_sat_low": k_sat_low, "K_sat_high": k_sat_high,
            "density": density,
        }


# ---------------------------------------------------------------------------
# VRH model
# ---------------------------------------------------------------------------
def VRH_model(
    f: Any = [0.35, 0.25, 0.2, 0.125, 0.075],
    K: Any = [55.4, 36.6, 75.6, 46.7, 50.4],
    G: Any = [28.1, 45, 25.6, 23.65, 27.4],
    rho: Any = [2560, 2650, 2630, 2540, 3050],
) -> Any:
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


# ---------------------------------------------------------------------------
# sat K
# ---------------------------------------------------------------------------
def satK(
    Keff: Any,
    Km: Any,
    phi: Any,
    Sat: Any,
) -> Any:
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
    return BrieModel().calculate_saturated_modulus(Keff, Km, phi, Sat)


# ---------------------------------------------------------------------------
# vel DEM
# ---------------------------------------------------------------------------
def velDEM(
    phi: Any,
    Km: Any,
    Gm: Any,
    rho_b: Any,
    Sat: Any,
    alpha: Any,
) -> Any:
    """
    Calculate effective bulk modulus (Keff), shear modulus (Geff), and P-wave velocity (Vp)
    for a rock with varying porosity (phi) based on the DEM model, taking into account
    the saturation (Sat) and the crack aspect ratio (alpha).

    Parameters:
    phi (np.array): Array of porosities.
    Km (float): Initial bulk modulus of the material (GPa).
    Gm (float): Initial shear modulus of the material (GPa).
    rho_b (float): Density of the solid phase (kg/m^3).
    Sat (array): Saturation fractions with one entry per phi value; scalars are not broadcast here.
    alpha (float): Crack aspect ratio.

    Returns:
    Keff1 (np.array): Effective bulk modulus for each porosity value (GPa).
    Geff1 (np.array): Effective shear modulus for each porosity value (GPa).
    Vp (np.array): P-wave velocity for each porosity value (m/s).
    """
    from scipy.optimize import root

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
            
        # Solve for effective shear modulus with root(method="lm").
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


# ---------------------------------------------------------------------------
# vel porous
# ---------------------------------------------------------------------------
def vel_porous(
    phi: Any,
    Km: Any,
    Gm: Any,
    rho_b: Any,
    Sat: Any,
    depth: Any = 1,
) -> Any:
    """
    Calculate P-wave velocity (Vp) for a rock with varying porosity (phi) based on the 
    Hertz-Mindlin model and Hashin-Shtrikman bounds, taking into account the saturation (Sat).

    Compatibility wrapper over HertzMindlinModel.calculate_velocity with
    critical porosity 0.4 and coordination number 4. Both return modes of
    that method and this function share the contact, HS and Gassmann
    implementation, including the corrected Hertz-Mindlin shear prefactor.
    Historical incorrect shear values are intentionally not preserved.

    Parameters:
    phi (np.array): Array of porosities.
    Km (float): Bulk modulus of the solid phase (GPa).
    Gm (float): Shear modulus of the solid phase (GPa).
    rho_b (float): Density of the solid phase (kg/m^3).
    Sat (array): Saturation fractions; scalars and arrays broadcast with phi.
    depth (float): depth for pressure estimation (m)

    Returns:
    Vp_h (np.array): P-wave velocity for each porosity value (upper Hashin-Shtrikman bound) (m/s).
    Vp_l (np.array): P-wave velocity for each porosity value (lower Hashin-Shtrikman bound) (m/s).
    """
    return HertzMindlinModel().calculate_velocity(
        porosity=phi, saturation=Sat, bulk_modulus=Km, shear_modulus=Gm,
        mineral_density=rho_b, depth=depth,
    )
