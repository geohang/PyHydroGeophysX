"""
Module for converting hydrologic model output to seismic travel times.
"""
import os
import numpy as np
import pygimli as pg
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager
from typing import Tuple, Optional, Dict, Any, Union, List

from PyHydroGeophysX.core.interpolation import ProfileInterpolator
from PyHydroGeophysX.petrophysics.velocity_models import HertzMindlinModel, DEMModel
from PyHydroGeophysX.forward.srt_forward import SeismicForwardModeling


def hydro_to_srt(
    water_content: np.ndarray,
    porosity: np.ndarray,
    mesh: pg.Mesh,
    profile_interpolator: ProfileInterpolator,
    layer_idx: Union[int, List[int]],
    structure: np.ndarray,
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
        Tuple[pg.DataContainer, np.ndarray]:
            - synth_data: PyGIMLi DataContainer with synthetic travel time data and estimated errors.
            - velocity_mesh: NumPy array of the cell P-wave velocity model [m/s] used for the forward simulation.
    """
    # --- Parameter and Data Preparation ---

    # Get cell markers from the mesh if not explicitly provided.
    # These markers link cells to layer-specific petrophysical parameters.
    if mesh_markers is None:
        mesh_markers = np.array(mesh.cellMarkers())
    
    # Get coordinates of cell centers for interpolation if needed.
    mesh_centers = np.array(mesh.cellCenters()) # Shape: (num_cells, 2) for 2D mesh
    
    # --- Step 1: Interpolate Hydrological Model Output to SRT Mesh (if necessary) ---
    # This section is identical to `hydro_to_ert`, converting gridded hydro output (water_content, porosity)
    # to values per cell on the geophysical mesh using the `profile_interpolator`.
    wc_mesh: np.ndarray     # Water content values per SRT mesh cell
    porosity_mesh: np.ndarray # Porosity values per SRT mesh cell

    if water_content.ndim > 1 and water_content.shape[0] > 1: # Heuristic for gridded layer data
        if verbose: print("Input water content and porosity are multi-dimensional; interpolating to SRT mesh.")

        # Interpolate 3D hydro data to the 2D profile.
        water_content_profile = profile_interpolator.interpolate_3d_data(water_content)
        porosity_profile = profile_interpolator.interpolate_3d_data(porosity)

        # Create layer IDs for profile data to guide mesh interpolation.
        ID_layers = porosity_profile.copy()
        ID_layers[:layer_idx[1]] = marker_labels[0]  # Top layer
        ID_layers[layer_idx[1]:layer_idx[2]] = marker_labels[1]  # Middle layer
        ID_layers[layer_idx[2]:] = marker_labels[2]  # Bottom layer
        if verbose: print(f"ID_layers for profile interpolation (SRT): {np.unique(ID_layers)}")

        # Interpolate from profile to the SRT mesh cells.
        wc_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=water_content_profile, depth_values=structure,
            mesh_x=mesh_centers[:, 0], mesh_y=mesh_centers[:, 1],
            mesh_markers=mesh_markers, ID=ID_layers, layer_markers=marker_labels
        )
        porosity_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=porosity_profile, depth_values=structure,
            mesh_x=mesh_centers[:, 0], mesh_y=mesh_centers[:, 1],
            mesh_markers=mesh_markers, ID=ID_layers, layer_markers=marker_labels
        )
    else: # Assume data is already per SRT mesh cell
        if verbose: print("Input water content and porosity are 1D; assuming mapped to SRT mesh cells.")
        wc_mesh = water_content.ravel()
        porosity_mesh = porosity.ravel()
        if wc_mesh.shape[0] != mesh.cellCount() or porosity_mesh.shape[0] != mesh.cellCount():
            raise ValueError("1D water_content/porosity length must match mesh cell count.")

    # --- Step 2: Calculate Saturation ---
    # S = θ / φ. Ensure porosity is not zero to avoid division errors.
    porosity_safe = np.maximum(porosity_mesh, 1e-4) # Floor porosity to avoid division by zero.
    saturation = np.clip(wc_mesh / porosity_safe, 0.0, 1.0) # Clip saturation to [0, 1].

    # --- Step 3: Convert Saturation to Seismic P-wave Velocity (Vp) using Petrophysical Models ---
    # Initialize petrophysical velocity models.
    # HertzMindlinModel: Typically for unconsolidated sediments, near-surface.
    #   - critical_porosity: Porosity at which material transitions from grain-supported to fluid-supported.
    #   - coordination_number: Average number of contacts per grain.
    #   SUGGESTION: These (0.4, 6.0) are common but could be parameters.
    hm_model = HertzMindlinModel(critical_porosity=0.4, coordination_number=6.0)
    # DEMModel (Differential Effective Medium): Models effect of adding pores/cracks into a solid matrix.
    dem_model = DEMModel()
    
    # Initialize velocity array for the mesh cells.
    velocity_mesh = np.zeros_like(wc_mesh, dtype=float) # Store Vp in m/s.
    
    # Retrieve layer-specific petrophysical parameters from `vel_parameters` dictionary.
    # These include elastic moduli (bulk, shear), density, and potentially geometric factors like aspect ratio.
    # Default values are provided if specific layer parameters are missing.
    top_params = vel_parameters.get('top', {})
    mid_params = vel_parameters.get('mid', {})
    bot_params = vel_parameters.get('bot', {})
    
    # --- Calculate Vp for each layer based on its assigned petrophysical model and parameters ---
    # Top layer: Assumed to be modeled by Hertz-Mindlin.
    top_mask = (mesh_markers == marker_labels[0]) # Boolean mask for top layer cells.
    if np.any(top_mask):
        # `hm_model.calculate_velocity` returns (Vp_high_bound, Vp_low_bound).
        # Here, the average of these bounds is used.
        # Parameters: porosity, saturation, matrix K, matrix G, matrix density, depth (for pressure).
        Vp_high, Vp_low = hm_model.calculate_velocity(
            porosity=porosity_mesh[top_mask],
            saturation=saturation[top_mask],
            bulk_modulus=top_params.get('bulk_modulus', 30.0),    # K_matrix [GPa]
            shear_modulus=top_params.get('shear_modulus', 20.0),  # G_matrix [GPa]
            mineral_density=top_params.get('mineral_density', 2650), # rho_matrix [kg/m^3]
            depth=top_params.get('depth', 1.0)                   # Effective depth for pressure [m]
        )
        velocity_mesh[top_mask] = (Vp_high + Vp_low) / 2.0 # Average Vp [m/s]
    
    # Middle layer: Assumed to be modeled by DEM.
    mid_mask = (mesh_markers == marker_labels[1]) # Boolean mask for middle layer cells.
    if np.any(mid_mask):
        # `dem_model.calculate_velocity` returns (K_eff, G_eff, Vp). We need Vp.
        # Parameters: porosity, saturation, matrix K, matrix G, matrix density, pore aspect_ratio.
        _Keff_mid, _Geff_mid, Vp_mid = dem_model.calculate_velocity(
            porosity=porosity_mesh[mid_mask],
            saturation=saturation[mid_mask],
            bulk_modulus=mid_params.get('bulk_modulus', 50.0),
            shear_modulus=mid_params.get('shear_modulus', 35.0),
            mineral_density=mid_params.get('mineral_density', 2670),
            aspect_ratio=mid_params.get('aspect_ratio', 0.05) # Pore aspect ratio
        )
        velocity_mesh[mid_mask] = Vp_mid # Vp [m/s]
    
    # Bottom layer: Assumed to be modeled by DEM.
    bot_mask = (mesh_markers == marker_labels[2]) # Boolean mask for bottom layer cells.
    if np.any(bot_mask):
        _Keff_bot, _Geff_bot, Vp_bot = dem_model.calculate_velocity(
            porosity=porosity_mesh[bot_mask],
            saturation=saturation[bot_mask],
            bulk_modulus=bot_params.get('bulk_modulus', 55.0),
            shear_modulus=bot_params.get('shear_modulus', 50.0),
            mineral_density=bot_params.get('mineral_density', 2680),
            aspect_ratio=bot_params.get('aspect_ratio', 0.03)
        )
        velocity_mesh[bot_mask] = Vp_bot # Vp [m/s]
    
    if verbose:
        # Print min/max of finite velocity values to avoid issues with NaNs from failed petro calculations.
        finite_velocities = velocity_mesh[np.isfinite(velocity_mesh)]
        if finite_velocities.size > 0:
            print(f"Calculated P-wave velocity model range: {np.min(finite_velocities):.2f} - {np.max(finite_velocities):.2f} m/s")
        else:
            print("Warning: Velocity model contains no finite values after petrophysical conversion.")

    # --- Step 4: Create Sensor (Geophone) Positions along the Profile ---
    # `sensors` array holds the x-coordinates of geophones.
    sensor_x_coords = np.linspace(sensor_start,
                                  sensor_start + (num_sensors - 1) * sensor_spacing,
                                  num_sensors)

    # The `surface_points` for `create_synthetic_data` should be the actual (x,y) coordinates
    # of the surface where sensors are placed. This is obtained from the `profile_interpolator`.
    # `profile_interpolator.L_profile` are distances along profile, `surface_profile` are elevations.
    # These need to be mapped to the coordinate system of `mesh`.
    # Assuming `L_profile` corresponds to x-coordinates for a flat or gently dipping profile setup.
    # If profile is not aligned with x-axis, this might need adjustment.
    # For `SeismicForwardModeling.create_synthetic_data`, `surface_points` are used to drape sensors.
    actual_surface_coords = np.column_stack((profile_interpolator.L_profile,
                                             profile_interpolator.surface_profile))

    # --- Step 5: Perform Seismic Forward Modeling (Generate Synthetic Travel Times) ---
    # Use the class method from `SeismicForwardModeling` to generate synthetic data.
    # This encapsulates scheme creation, mesh handling (if not provided), forward calculation, and noise addition.
    # The `mesh` passed here is the geophysical mesh. `velocity_mesh` is Vp for its cells.
    # `slowness=False` indicates `velocity_mesh` contains velocities, not slownesses.
    # `shot_distance` controls spacing of shots in `createRAData` within `create_synthetic_data`.
    synthetic_srt_data, _used_mesh = SeismicForwardModeling.create_synthetic_data(
        sensor_x=sensor_x_coords,         # 1D array of sensor x-positions for scheme creation.
        surface_points=actual_surface_coords, # 2D surface for draping sensors.
        mesh=mesh,                        # The geophysical mesh.
        velocity_model=velocity_mesh,     # Vp model for mesh cells [m/s].
        slowness=False,                   # Indicate that `velocity_model` is indeed velocity.
        shot_distance=shot_distance,      # Spacing between shots.
        noise_level=noise_level,          # Relative noise level for travel times.
        noise_abs=noise_abs,              # Absolute noise level for travel times [s].
        save_path=save_path,              # Optional path to save data.
        show_data=verbose,                # Show data plot if verbose. (Original was `verbose`, changed to `show_data` if distinct control is desired)
                                          # Keeping `verbose` as per original signature for `show_data` flag.
        verbose=verbose,                  # Verbosity for the simulation process itself.
        seed=seed                         # Random seed for noise generation.
    )
    # `_used_mesh` is the mesh potentially modified or created by `create_synthetic_data`, can be ignored if `mesh` is just passed through.
    
    return synthetic_srt_data, velocity_mesh # Return synthetic data and the Vp model used.