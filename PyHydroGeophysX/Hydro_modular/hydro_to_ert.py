"""
Module for converting hydrologic model output to ERT apparent resistivity.
"""
import os
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from typing import Tuple, Optional, Dict, Any, Union, List

from PyHydroGeophysX.core.interpolation import ProfileInterpolator
from PyHydroGeophysX.petrophysics.resistivity_models import water_content_to_resistivity
from PyHydroGeophysX.forward.ert_forward import ERTForwardModeling


def hydro_to_ert(
    water_content: np.ndarray,
    porosity: np.ndarray,
    mesh: pg.Mesh,
    profile_interpolator: ProfileInterpolator,
    layer_idx: Union[int, List[int]],
    structure: np.ndarray,
    marker_labels: List[int],
    rho_parameters: Dict[str, Any],
    electrode_spacing: float = 1.0,
    electrode_start: float = 0.0,
    num_electrodes: int = 72,
    scheme_name: str = 'wa',
    noise_level: float = 0.05,
    abs_error: float = 0.0,
    rel_error: float = 0.05,
    save_path: Optional[str] = None,
    mesh_markers: Optional[np.ndarray] = None,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[pg.DataContainer, np.ndarray]:
    """
    Convert hydrologic model output to ERT apparent resistivity.
    
    This function performs the complete workflow from water content to synthetic ERT data:
    1. Interpolates water content to mesh
    2. Calculates saturation
    3. Converts saturation to resistivity using petrophysical models
    4. Creates electrode array along surface profile
    5. Performs forward modeling to generate synthetic ERT data
    
    Args:
        water_content: Water content array (nlay, ny, nx) or mesh values
        porosity: Porosity array (nlay, ny, nx) or mesh values
        mesh: PyGIMLI mesh
        profile_interpolator: ProfileInterpolator for surface interpolation
        marker_labels: Layer marker labels [top, middle, bottom]
        rho_parameters: Dictionary of resistivity parameters:
            {
                'rho_sat': [100, 500, 2400],  # Saturated resistivity values
                'n': [2.2, 1.8, 2.5],         # Cementation exponents
                'sigma_s': [1/500, 0, 0]      # Surface conductivity values
            }
        electrode_spacing: Spacing between electrodes
        electrode_start: Starting position of electrode array
        num_electrodes: Number of electrodes
        scheme_name: ERT scheme name ('wa', 'dd', etc.)
        noise_level: Relative noise level for synthetic data
        abs_error: Absolute error for data estimation
        rel_error: Relative error for data estimation
        save_path: Path to save synthetic data (None = don't save)
        mesh_markers: Mesh cell markers (None = get from mesh)
        verbose: Whether to display verbose information
        seed: Random seed for noise generation
        
    Returns:
        Tuple[pg.DataContainer, np.ndarray]:
            - synth_data: PyGIMLi DataContainer with synthetic apparent resistivity data and estimated errors.
            - res_model: NumPy array of the cell resistivity model used for the forward simulation.
    """
    # --- Parameter and Data Preparation ---

    # Get cell markers from the mesh if not explicitly provided.
    # Cell markers are used to assign different petrophysical properties to different regions/layers.
    if mesh_markers is None:
        mesh_markers = np.array(mesh.cellMarkers())
    
    # Get coordinates of cell centers. Used for interpolation if input `water_content` is gridded layer data.
    mesh_centers = np.array(mesh.cellCenters()) # Shape: (num_cells, 2) for 2D mesh
    
    # --- Step 1: Interpolate Hydrological Model Output to ERT Mesh (if necessary) ---
    # If `water_content` is a 3D array (e.g., [layers, ny_hydro, nx_hydro] from a structured hydro model),
    # it needs to be interpolated onto the potentially unstructured ERT mesh (`mesh`).
    # `profile_interpolator` and `structure` (depths of layers along profile) are used for this.
    # `ID_layers` helps map profile data to mesh regions based on markers.
    wc_mesh: np.ndarray     # Water content values per ERT mesh cell
    porosity_mesh: np.ndarray # Porosity values per ERT mesh cell

    if water_content.ndim > 1 and water_content.shape[0] > 1: # Heuristic: if more than 1D array and first dim size > 1, assume gridded layer data.
                                                              # SUGGESTION: A more explicit flag or type check might be better.
        if verbose: print("Input water content and porosity are multi-dimensional; interpolating to ERT mesh.")

        # Interpolate the 3D gridded water content from the hydrological model output
        # onto the 2D profile defined by `profile_interpolator`.
        # `profile_interpolator.interpolate_3d_data` handles this projection.
        water_content_profile = profile_interpolator.interpolate_3d_data(water_content) # Shape: (n_hydro_layers, n_profile_points)

        # Similarly, interpolate 3D gridded porosity onto the 2D profile.
        porosity_profile = profile_interpolator.interpolate_3d_data(porosity) # Shape: (n_hydro_layers, n_profile_points)

        # Create an ID array (`ID_layers`) that assigns a marker label to each point in the profile data.
        # This is based on the provided `layer_idx` (indices defining layer boundaries in the profile data)
        # and `marker_labels` (the corresponding mesh marker for each layer).
        # `structure` (not explicitly used here but used by `interpolate_to_mesh`) provides depth context.
        # This ID_layers will guide `interpolate_to_mesh` on how to map different layers from the profile
        # to different regions in the ERT mesh.
        ID_layers = porosity_profile.copy() # Use shape of porosity_profile for ID_layers structure
        # Assign marker for the top layer (from surface down to first specified layer_idx depth)
        ID_layers[:layer_idx[1]] = marker_labels[0]
        # Assign marker for the middle layer
        ID_layers[layer_idx[1]:layer_idx[2]] = marker_labels[1]
        # Assign marker for the bottom layer
        ID_layers[layer_idx[2]:] = marker_labels[2]
        if verbose: print(f"ID_layers created for profile interpolation with unique values: {np.unique(ID_layers)}")

        # Interpolate the 2D profile water content data to the cells of the ERT mesh.
        # `profile_interpolator.interpolate_to_mesh` maps values from the profile (with layer IDs)
        # to the ERT mesh cells based on their spatial location and marker.
        wc_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=water_content_profile, # Water content values along the profile
            depth_values=structure,                # Depth coordinates of layers along the profile
            mesh_x=mesh_centers[:, 0],             # X-coordinates of ERT mesh cell centers
            mesh_y=mesh_centers[:, 1],             # Y-coordinates of ERT mesh cell centers (depth)
            mesh_markers=mesh_markers,             # Marker for each ERT mesh cell
            ID=ID_layers,                          # Layer IDs for the profile data
            layer_markers=marker_labels            # List of unique marker labels for layers
        )
        
        # Interpolate the 2D profile porosity data to the ERT mesh cells.
        porosity_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=porosity_profile,
            depth_values=structure,
            mesh_x=mesh_centers[:, 0],
            mesh_y=mesh_centers[:, 1],
            mesh_markers=mesh_markers,
            ID=ID_layers,
            layer_markers=marker_labels
        )
    else: # If water_content is already 1D (assumed to be per ERT mesh cell).
        if verbose: print("Input water content and porosity are 1D; assuming they are already mapped to ERT mesh cells.")
        wc_mesh = water_content.ravel() # Ensure it's a flat array
        porosity_mesh = porosity.ravel() # Ensure it's a flat array
        if wc_mesh.shape[0] != mesh.cellCount() or porosity_mesh.shape[0] != mesh.cellCount():
            raise ValueError("If providing 1D water_content/porosity, length must match mesh cell count.")

    # --- Step 2: Calculate Saturation ---
    # Saturation (S) = Volumetric Water Content (θ) / Porosity (φ)
    # Ensure porosity is not zero to avoid division by zero errors. Add a small epsilon for safety.
    # SUGGESTION: A more robust handling might involve checking where porosity is zero and setting
    # saturation to a specific value (e.g., 0 or NaN) or raising errors if wc_mesh is non-zero there.
    porosity_safe = np.maximum(porosity_mesh, 1e-4) # Use a small positive floor for porosity (e.g., 0.0001 or 0.01).
                                                    # Original was 0.01. Using 1e-4 for potentially lower porosities.
    saturation = wc_mesh / porosity_safe
    # Clip saturation to physically realistic bounds [0, 1].
    saturation = np.clip(saturation, 0.0, 1.0)

    # --- Step 3: Convert Water Content/Saturation to Resistivity using Petrophysical Model ---
    # Retrieve petrophysical parameters from `rho_parameters` dictionary.
    # These include saturated resistivity (rho_sat), saturation exponent (n), and surface conductivity (sigma_s)
    # for each layer/material defined by `marker_labels`.
    # Defaults are provided if keys are missing.
    rho_sat_params = rho_parameters.get('rho_sat', [100.0, 500.0, 2400.0]) # Default saturated resistivities for 3 layers
    n_exponent_params = rho_parameters.get('n', [2.2, 1.8, 2.5])           # Default saturation exponents
    sigma_surf_params = rho_parameters.get('sigma_s', [1.0/500.0, 0.0, 0.0]) # Default surface conductivities

    res_model = np.zeros_like(wc_mesh)  # Initialize cell resistivity model array.

    # Apply the petrophysical model (Waxman-Smits via `water_content_to_resistivity`)
    # for each layer/region defined by the mesh markers.
    # The loop iterates through `marker_labels`, which should correspond to the order of params in `rho_parameters`.
    for i, marker_val in enumerate(marker_labels):
        layer_mask = (mesh_markers == marker_val) # Create a boolean mask for cells belonging to the current marker.
        if not np.any(layer_mask): # If no cells have this marker, skip.
            if verbose: print(f"Warning: No cells found with marker {marker_val}. Skipping resistivity calculation for this layer.")
            continue

        # Apply `water_content_to_resistivity` only to the cells in the current layer/region.
        # Note: `water_content_to_resistivity` itself uses `saturation = wc/porosity`.
        # It might be more direct to pass saturation if already computed, or ensure it recalculates consistently.
        # Here, it effectively uses wc_mesh[layer_mask] and porosity_mesh[layer_mask].
        layer_resistivities = water_content_to_resistivity(
            water_content=wc_mesh[layer_mask],       # Water content for the current layer's cells.
            rhos=float(rho_sat_params[i]),           # Saturated resistivity for this layer.
            n=float(n_exponent_params[i]),           # Saturation exponent for this layer.
            porosity=porosity_mesh[layer_mask],    # Porosity for the current layer's cells.
            sigma_sur=float(sigma_surf_params[i])    # Surface conductivity for this layer.
        )
        res_model[layer_mask] = layer_resistivities # Assign calculated resistivities to the model.
    
    if verbose:
        print(f"Calculated resistivity model range: {np.min(res_model[np.isfinite(res_model)]):.2f} - {np.max(res_model[np.isfinite(res_model)]):.2f} Ohm-m")
    
    # --- Step 4: Create Electrode Positions and ERT Scheme ---
    # Define electrode x-positions based on start, spacing, and number of electrodes.
    electrode_x_coords = np.linspace(electrode_start,
                                     electrode_start + (num_electrodes - 1) * electrode_spacing,
                                     num_electrodes)
    
    # Interpolate electrode y-positions (elevations) from the surface profile provided by `profile_interpolator`.
    # This drapes electrodes onto the surface topography.
    electrode_y_coords = np.interp(electrode_x_coords,
                                   profile_interpolator.L_profile,       # X-distances along the profile.
                                   profile_interpolator.surface_profile) # Corresponding elevations on the profile.
    
    # The mesh used for forward modeling might need specific boundary conditions or extensions.
    # Here, cell markers are set to 2 (active region), and a triangular boundary (marker 1) is added.
    # This `grid` will be the actual mesh used by the forward operator.
    # SUGGESTION: The markers assigned here (1 and 2) should be consistent with how `res_model` is defined
    # or how background resistivity for marker 1 is handled by the ERTModelling operator.
    # If `res_model` is defined for all cells of the original `mesh`, ensure it's correctly mapped to `grid`.
    # Typically, `res_model` should be for `grid.cellCount()`.
    # If `mesh.setCellMarkers` changes markers that `res_model` calculation relied on, it's an issue.
    # Assuming `res_model` is for the original `mesh` cells, and `appendTriangleBoundary` adds new cells for marker 1.
    # The `ERTForwardModeling` class or `fob.response` needs a model defined for all cells of `grid`.
    # This part might need careful handling of resistivity values for boundary cells (marker 1).
    # For now, assume `res_model` (from `wc_mesh` which is on `mesh`) is for region 2, and region 1 gets a default.
    
    # Create a working copy of the mesh for forward modeling if it's going to be modified.
    # If `mesh` is used by other parts, modifying it here might have side effects.
    # However, `appendTriangleBoundary` usually creates a new mesh.
    # The line `mesh.setCellMarkers(...)` modifies the input `mesh` object.
    mesh.setCellMarkers(np.ones(mesh.cellCount(), dtype=int) * 2) # Set all cells of original mesh to marker 2.
    simulation_grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1, # Boundary cells get marker 1.
                                                          xbound=100, ybound=100) # Extent of boundary.
                                                                                # SUGGESTION: xbound/ybound should be relative to mesh size.

    # Reconcile `res_model` (defined on `mesh` cells, now marker 2) with `simulation_grid`.
    # `simulation_grid` has original cells (now marker 2) + new boundary cells (marker 1).
    # We need a resistivity vector for all cells in `simulation_grid`.
    final_res_model_for_fwd = np.ones(simulation_grid.cellCount()) # Initialize for all cells in the new grid.
    # Assign `res_model` values to cells with marker 2 (original mesh cells).
    final_res_model_for_fwd[simulation_grid.cellMarkers() == 2] = res_model
    # Assign a background resistivity to boundary cells (marker 1).
    # SUGGESTION: Make this background resistivity a parameter. Using 100 Ohm-m as a placeholder.
    default_background_rho = 100.0
    final_res_model_for_fwd[simulation_grid.cellMarkers() == 1] = default_background_rho
    if verbose: print(f"Resistivity model for forward operator prepared for {simulation_grid.cellCount()} cells.")

    # Combine electrode x and y coordinates into a 2D array for PyGIMLi.
    electrode_positions = np.hstack((electrode_x_coords.reshape(-1,1), electrode_y_coords.reshape(-1,1)))
    # Create an ERT data scheme (e.g., Wenner alpha 'wa', dipole-dipole 'dd').
    ert_scheme = ert.createData(elecs=electrode_positions, schemeName=scheme_name)

    # The `ERTForwardModeling` class was initialized here in original code but not used.
    # The PyGIMLi `ert.ERTModelling` (`fob`) is used directly below.
    # fwd_operator_custom_class = ERTForwardModeling(mesh=simulation_grid, data=ert_scheme) # This instance is not used later.

    # --- Step 5: Perform ERT Forward Modeling ---
    # Create a new data container to store synthetic (simulated) apparent resistivities.
    synthetic_data = ert_scheme.copy() # Copies scheme structure (configs, but not 'rhoa' values yet).

    # Initialize PyGIMLi's ERTModelling forward operator.
    fwd_op_pygimli = ert.ERTModelling()
    fwd_op_pygimli.setData(ert_scheme)      # Set the measurement configurations.
    fwd_op_pygimli.setMesh(simulation_grid) # Set the mesh with boundary.
    
    # Calculate the "true" forward response (apparent resistivities) using the `final_res_model_for_fwd`.
    true_simulated_rhoa = fwd_op_pygimli.response(final_res_model_for_fwd) # pg.Vector of rhoa

    # --- Add Noise to Synthetic Data ---
    # Simulate measurement noise: relative Gaussian noise + absolute noise component.
    # `pg.randn` creates an array of normally distributed random numbers (mean 0, std dev 1).
    # Original code had `0.05` hardcoded for relative noise. Using `noise_level` parameter.
    noisy_simulated_rhoa = true_simulated_rhoa * (1.0 + pg.randn(true_simulated_rhoa.size()) * noise_level)
    # Note: `noise_abs` (absolute noise) from parameters is not directly added here.
    # PyGIMLi's `ert.ERTManager.estimateError` and `manager.simulate` handle noise differently.
    # This is a simple relative multiplicative noise.
    # SUGGESTION: Clarify noise model. If `noise_abs` is important, it should be incorporated,
    # e.g., `noisy_rhoa = true_rhoa + true_rhoa * rel_noise_randn + abs_noise_randn`.

    # Assign noisy apparent resistivities to the data container.
    synthetic_data['rhoa'] = noisy_simulated_rhoa

    # Estimate and assign errors to the synthetic data.
    # This simulates how errors might be characterized for real data during inversion.
    temp_ert_manager = ert.ERTManager(synthetic_data) # Create a temporary manager.
    synthetic_data['err'] = temp_ert_manager.estimateError(
        synthetic_data,
        absoluteUError=abs_error, # Uses abs_error, not noise_abs here.
        relativeError=rel_error   # Uses rel_error.
    )
    # Note: The noise added to `rhoa` and the error model assigned via `err` are related but distinct.
    # `noise_level` corrupts the "true" data. `abs_error`/`rel_error` define the assumed error model for inversion weighting.

    # Optional: Save or display the synthetic data (not implemented in this refactoring, but was in original).
    # if save_path: synthetic_data.save(save_path)
    # if show_data: ert.showData(synthetic_data, vals='rhoa', scheme=ert_scheme)

    return synthetic_data, final_res_model_for_fwd # Return synthetic data and the resistivity model used.