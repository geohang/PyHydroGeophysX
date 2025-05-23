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
        Tuple of (synthetic ERT data container, resistivity model)
    """
    # Get mesh markers if not provided
    if mesh_markers is None:
        mesh_markers = np.array(mesh.cellMarkers())
    
    # Get mesh centers
    mesh_centers = np.array(mesh.cellCenters())
    
    # 1. If water_content is a 3D array (layer data), interpolate to mesh
    if water_content.ndim > 1 and water_content.shape[0] > 1:
        # Get structure from profile interpolator
       
        # Step 4: Interpolate data to profile
        # Initialize profile interpolator

        # Interpolate water content to profile
        water_content_profile = profile_interpolator.interpolate_3d_data(water_content)

        # Interpolate porosity to profile
        porosity_profile = profile_interpolator.interpolate_3d_data(porosity)


        # Set up layer IDs based on marker labels
        ID_layers = porosity_profile.copy()
        ID_layers[:layer_idx[1]] = marker_labels[0]  # Top layer
        ID_layers[layer_idx[1]:layer_idx[2]] = marker_labels[1]  # Middle layer
        ID_layers[layer_idx[2]:] = marker_labels[2]  # Bottom layer
        print(ID_layers)

        # Interpolate water content to mesh
        wc_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=water_content_profile,
            depth_values=structure,
            mesh_x=mesh_centers[:, 0],
            mesh_y=mesh_centers[:, 1],
            mesh_markers=mesh_markers,
            ID=ID_layers,
            layer_markers=marker_labels
        )
        
        # Interpolate porosity to mesh
        porosity_mesh = profile_interpolator.interpolate_to_mesh(
            property_values=porosity_profile,
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
    
    # 3. Convert to resistivity using petrophysical model
    rho_sat = rho_parameters.get('rho_sat', [100, 500, 2400])
    n_values = rho_parameters.get('n', [2.2, 1.8, 2.5])
    sigma_s = rho_parameters.get('sigma_s', [1/500, 0, 0])
    
    res_model = np.zeros_like(wc_mesh)  # Initialize resistivity array
    
    # Calculate resistivity for each layer based on marker labels
    for i, marker in enumerate(marker_labels):
        mask = (mesh_markers == marker)
        layer_res = water_content_to_resistivity(
            wc_mesh[mask],
            float(rho_sat[i]),
            float(n_values[i]),
            porosity_mesh[mask],
            sigma_s[i]
        )
        res_model[mask] = layer_res
    
    if verbose:
        print(f"Resistivity range: {np.min(res_model)} - {np.max(res_model)} Ohm-m")
    
    # 4. Create electrode positions along profile
    xpos = np.linspace(electrode_start, 
                      electrode_start + (num_electrodes - 1) * electrode_spacing, 
                      num_electrodes)
    
    # Interpolate elevations from profile
    ypos = np.interp(xpos, 
                    profile_interpolator.L_profile, 
                    profile_interpolator.surface_profile)
    
    mesh.setCellMarkers(np.ones(mesh.cellCount())*2)
    grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1,
                                            xbound=100, ybound=100)
    
    pos = np.hstack((xpos.reshape(-1,1),ypos.reshape(-1,1)))
    schemeert = ert.createData(elecs=pos,schemeName=scheme_name)
    fwd_operator = ERTForwardModeling(mesh=grid, data=schemeert)


    # 5. Perform forward modeling to create synthetic ERT data
    synth_data = schemeert.copy()
    fob = ert.ERTModelling()
    fob.setData(schemeert)
    fob.setMesh(grid)
    dr = fob.response(res_model)

    dr *= 1. + pg.randn(dr.size()) * 0.05
    ert_manager = ert.ERTManager(synth_data)
    synth_data['rhoa'] = dr
    synth_data['err'] = ert_manager.estimateError(synth_data, absoluteUError=0.0, relativeError=0.05)
    
    return synth_data, res_model