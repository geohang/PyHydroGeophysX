"""
Petrophysical Models for Hydrological and Geophysical Parameter Conversion.

This subpackage provides a collection of petrophysical models to relate
hydrological properties (e.g., water content, saturation, porosity) to
geophysical parameters (e.g., electrical resistivity, seismic velocity).
It includes established empirical and theoretical models.

Modules:
- `resistivity_models.py`: Contains models and functions for electrical resistivity.
- `velocity_models.py`: Contains models and functions for seismic velocity.

Key functionalities include converting between water content/saturation and resistivity,
and calculating seismic velocities based on various rock physics models like Voigt-Reuss-Hill,
Brie, Differential Effective Medium (DEM), and Hertz-Mindlin contact theory.
"""

from .resistivity_models import (
    water_content_to_resistivity,
    resistivity_to_water_content,
    resistivity_to_saturation
)
from .velocity_models import (
    BaseVelocityModel,
    VRHModel,
    BrieModel,
    DEMModel,
    HertzMindlinModel,
    # Standalone functions - to be evaluated for deprecation
    VRH_model,
    satK,
    velDEM,
    vel_porous
)

__all__ = [
    # From resistivity_models
    'water_content_to_resistivity',
    'resistivity_to_water_content',
    'resistivity_to_saturation',
    # From velocity_models (Classes)
    'BaseVelocityModel',
    'VRHModel',
    'BrieModel',
    'DEMModel',
    'HertzMindlinModel',
    # From velocity_models (Standalone functions - for potential deprecation)
    'VRH_model',
    'satK',
    'velDEM',
    'vel_porous',
]
