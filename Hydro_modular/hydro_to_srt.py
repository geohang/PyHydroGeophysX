"""
Module for converting hydrologic model output to seismic travel times.
"""
import os
import numpy as np
import pygimli as pg
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager
from typing import Tuple, Optional, Dict, Any, Union, List

from watershed_geophysics.core.interpolation import ProfileInterpolator
from watershed_geophysics.petrophysics.velocity_models import HertzMindlinModel, DEMModel
from watershed_geophysics.forward.srt_forward import SeismicForwardModeling


def hydro_to_srt(
    water_content: np.ndarray,
    porosity: np.ndarray,
    mesh: pg.Mesh,
    profile_interpolator: ProfileInterpolator,
    marker_labels: List[int],
    vel_parameters: Dict[str, Any],
    sensor_spacing: float = 1.0,
    sensor_start: float = 0.0,
    num_sensors: int = 72,
    shot_distance: float = 5,
    noise_level: float = 0.05,
    noise_abs: float = 0.00001,
    save_path: Optional[str] = None,
    mesh_markers: Optional[np.ndarray] = None,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[pg.DataContainer, np.ndarray]:
    """
    Convert hydrologic model output to seismic travel times.
    
    This function performs the complete workflow from water content to synthetic SRT data:
    1. Interpolates water content to mesh
    2. Calculates saturation
    3. Converts saturation to seismic velocities using petrophysical models
    4. Creates sensor array along surface profile
    5. Performs forward modeling to generate synthetic travel time data
    
    Args:
        water_content: Water content array (nlay, ny, nx) or mesh values
        porosity: Porosity array (nlay, ny, nx) or mesh values
        mesh: PyGIMLI mesh
        profile_interpolator: ProfileInterpolator for surface interpolation
        marker_labels: Layer marker labels [top, middle, bottom]
        vel_parameters: Dictionary of velocity parameters:
            {
                'top': {'bulk_modulus': 30.0, 'shear_modulus': 20.0, 'mineral_density': 2650, 'depth': 1.0},
                'mid': {'bulk_modulus': 50.0, 'shear_modulus': 35.0, 'mineral_density': 2670, 'aspect_ratio': 0.05},
                'bot': {'bulk_modulus': 55.0, 'shear_modulus': 50.0, 'mineral_density': 2680, 'aspect_ratio': 0.03}
            }
        sensor_spacing: Spacing between sensors
        sensor_start: Starting position of sensor array
        num_sensors: Number of sensors
        shot_distance: Distance between shot points
        noise_level: Relative noise level for synthetic data
        noise_abs: Absolute noise level for synthetic data
        save_path: Path to save synthetic data (None = don't save)
        mesh_markers: Mesh cell markers (None = get from mesh)
        verbose: Whether to display verbose information
        seed: Random seed for noise generation
        
    Returns:
        Tuple of (synthetic SRT data container, velocity model)
    """
    # Get mesh markers if not provided
    if mesh_markers is None:
        mesh_markers = np.array(mesh.cellMarkers())
    
    # Get mesh centers
    mesh_centers = np.array(mesh.cellCenters())
    
    # 1. If water_content is a 3D array (layer data), interpolate to mesh
    if water_content.ndim > 1 and water_content.shape[0] > 1:
        # Get structure from profile interpolator
        structure = profile_interpolator.interpolate_layer_data([water_content])
        
        # Set up layer IDs based on marker labels
        ID_layers = np.zeros_like(structure[0])
        ID_layers[:marker_labels[0]] = marker_labels[0]  # Top layer
        ID_layers[marker_labels[0]:marker_labels[1]] = marker_labels[1]  # Middle layer
        ID_layers[marker_labels[1]:] = marker_labels[2]  # Bottom layer
        
        # Interpolate water content to mesh
        wc_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=water_content,
            depth_values=structure,
            mesh_x=mesh_centers[:, 0],
            mesh_y=mesh_centers[:, 1],
            mesh_markers=mesh_markers,
            ID=ID_layers,
            layer_markers=marker_labels
        )
        
        # Interpolate porosity to mesh
        porosity_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=porosity,
            depth_values=structure,
            mesh_x=mesh_centers[:, 0],
            mesh_y=mesh_centers[:, 1],
            mesh_markers=mesh_markers,
            ID=ID_layers,
            layer_markers=marker_labels
        )
    else:
        # Already mesh values
        wc_mesh = water_content
        porosity_mesh = porosity
    
    # 2. Calculate saturation
    # Ensure porosity is not zero to avoid division by zero
    porosity_safe = np.maximum(porosity_mesh, 0.01)
    saturation = np.clip(wc_mesh / porosity_safe, 0.0, 1.0)
    
    # 3. Convert to seismic velocity using petrophysical model
    # Initialize velocity models
    hm_model = HertzMindlinModel(critical_porosity=0.4, coordination_number=6.0)
    dem_model = DEMModel()
    
    # Initialize velocity model
    velocity_mesh = np.zeros_like(wc_mesh)
    
    # Get parameters for each layer
    top_params = vel_parameters.get('top', {})
    mid_params = vel_parameters.get('mid', {})
    bot_params = vel_parameters.get('bot', {})
    
    # Process top layer (Hertz-Mindlin model)
    top_mask = (mesh_markers == marker_labels[0])
    if np.any(top_mask):
        Vp_high, Vp_low = hm_model.calculate_velocity(
            porosity=porosity_mesh[top_mask],
            saturation=saturation[top_mask],
            bulk_modulus=top_params.get('bulk_modulus', 30.0),
            shear_modulus=top_params.get('shear_modulus', 20.0),
            mineral_density=top_params.get('mineral_density', 2650),
            depth=top_params.get('depth', 1.0)
        )
        # Use average of high and low bounds
        velocity_mesh[top_mask] = (Vp_high + Vp_low) / 2
    
    # Process middle layer (DEM model)
    mid_mask = (mesh_markers == marker_labels[1])
    if np.any(mid_mask):
        _, _, Vp = dem_model.calculate_velocity(
            porosity=porosity_mesh[mid_mask],
            saturation=saturation[mid_mask],
            bulk_modulus=mid_params.get('bulk_modulus', 50.0),
            shear_modulus=mid_params.get('shear_modulus', 35.0),
            mineral_density=mid_params.get('mineral_density', 2670),
            aspect_ratio=mid_params.get('aspect_ratio', 0.05)
        )
        velocity_mesh[mid_mask] = Vp
    
    # Process bottom layer (DEM model)
    bot_mask = (mesh_markers == marker_labels[2])
    if np.any(bot_mask):
        _, _, Vp = dem_model.calculate_velocity(
            porosity=porosity_mesh[bot_mask],
            saturation=saturation[bot_mask],
            bulk_modulus=bot_params.get('bulk_modulus', 55.0),
            shear_modulus=bot_params.get('shear_modulus', 50.0),
            mineral_density=bot_params.get('mineral_density', 2680),
            aspect_ratio=bot_params.get('aspect_ratio', 0.03)
        )
        velocity_mesh[bot_mask] = Vp
    
    if verbose:
        print(f"Velocity range: {np.min(velocity_mesh)} - {np.max(velocity_mesh)} m/s")
    
    # 4. Create sensor positions along profile
    sensors = np.linspace(sensor_start, 
                        sensor_start + (num_sensors - 1) * sensor_spacing, 
                        num_sensors)
    
    # 5. Perform forward modeling to create synthetic seismic data
    synth_data, _ = SeismicForwardModeling.create_synthetic_data(
        sensor_x=sensors,
        surface_points=np.column_stack((profile_interpolator.L_profile,
                                       profile_interpolator.surface_profile)),
        mesh=mesh,
        velocity_model=velocity_mesh,
        slowness=False,
        shot_distance=shot_distance,
        noise_level=noise_level,
        noise_abs=noise_abs,
        save_path=save_path,
        show_data=verbose,
        verbose=verbose,
        seed=seed
    )
    
    # Draw first picks if verbose is enabled
    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        SeismicForwardModeling.draw_first_picks(ax, synth_data)
        plt.tight_layout()
        plt.show()
    
    return synth_data, velocity_mesh