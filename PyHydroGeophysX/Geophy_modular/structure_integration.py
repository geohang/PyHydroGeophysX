"""
Structure integration module for constrained geophysical inversion.
"""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from typing import Tuple, List, Optional, Union, Dict, Any


def integrate_velocity_interface(ertData, smooth_x, smooth_z, paraBoundary=2, 
                               quality=28, paraMaxCellSize=30, paraDepth=30.0):
    """
    Integrate velocity interface into mesh for constrained ERT inversion.
    
    Args:
        ertData: PyGIMLi ERT data container
        smooth_x: X coordinates of velocity interface
        smooth_z: Z coordinates of velocity interface
        paraBoundary: Extra boundary size (default: 2)
        quality: Mesh quality parameter (default: 28)
        paraMaxCellSize: Maximum cell size (default: 30)
        paraDepth: Maximum depth of the model (default: 30.0)
        
    Returns:
        markers: Cell markers array
        meshafter: Mesh with interface structure
    """
    # Create the initial parameter mesh
    geo = mt.createParaMeshPLC(ertData, quality=quality, paraMaxCellSize=paraMaxCellSize,
                              paraBoundary=paraBoundary, paraDepth=paraDepth,
                              boundaryMaxCellSize=500)
    
    # Stack x and z coordinates for the interface
    interface_points = np.vstack((smooth_x, smooth_z)).T
    
    # Extend the interface line beyond the data range by paraBoundary
    input_points = np.vstack((
        np.array([[interface_points[0][0] - paraBoundary, interface_points[0][1]]]),
        interface_points,
        np.array([[interface_points[-1][0] + paraBoundary, interface_points[-1][1]]])
    ))
    
    # Create a polygon line for the interface
    interface_line = mt.createPolygon(input_points.tolist(), isClosed=False,
                                     interpolate='linear', marker=99)
    
    # Add the interface to the geometry
    geo_with_interface = geo + interface_line
    
    # Create a mesh from the combined geometry
    meshafter = mt.createMesh(geo_with_interface, quality=quality)
    
    # Initialize all markers to 1 (outside region)
    markers = np.ones(meshafter.cellCount())
    
    # Identify the survey area
    survey_left = ertData.sensors()[0][0] - paraBoundary
    survey_right = ertData.sensors()[-1][0] + paraBoundary
    
    # Process each cell
    for i in range(meshafter.cellCount()):
        cell_x = meshafter.cell(i).center().x()
        cell_y = meshafter.cell(i).center().y()
        
        # Only modify markers within the survey area
        if cell_x >= survey_left and cell_x <= survey_right:
            # Interpolate the interface height at this x position
            interface_y = np.interp(cell_x, input_points[:, 0], input_points[:, 1])
            
            # Set marker based on position relative to interface
            if abs(cell_y) < abs(interface_y):
                markers[i] = 2  # Below interface
            else:
                markers[i] = 3  # Above interface
    
    # Keep original markers for outside cells
    markers[meshafter.cellMarkers()==1] = 1
    
    # Set the updated markers
    meshafter.setCellMarkers(markers)
    
    return markers, meshafter


def create_ert_mesh_with_structure(ertData, interface_data, **kwargs):
    """
    Create ERT mesh with structure interface for constrained inversion.
    
    Args:
        ertData: PyGIMLi ERT data container
        interface_data: Interface data (can be a tuple of (x, z) or a dictionary with smooth_x, smooth_z)
        **kwargs: Additional parameters including:
            - paraBoundary: Extra boundary size (default: 2)
            - quality: Mesh quality parameter (default: 28)
            - paraMaxCellSize: Maximum cell size (default: 30)
            - paraDepth: Maximum depth (default: 30.0)
            
    Returns:
        meshafter: Mesh with interface structure
        markers: Cell markers array
        regions: Dictionary with region definitions
    """
    # Set default parameters
    params = {
        'paraBoundary': 2,
        'quality': 28, 
        'paraMaxCellSize': 30,
        'paraDepth': 30.0
    }
    params.update(kwargs)
    
    # Extract interface coordinates
    if isinstance(interface_data, tuple) and len(interface_data) == 2:
        smooth_x, smooth_z = interface_data
    elif isinstance(interface_data, dict) and 'smooth_x' in interface_data and 'smooth_z' in interface_data:
        smooth_x = interface_data['smooth_x']
        smooth_z = interface_data['smooth_z']
    else:
        raise ValueError("Interface data must be a (x, z) tuple or a dictionary with 'smooth_x' and 'smooth_z' keys")
    
    # Create mesh with interface
    markers, meshafter = integrate_velocity_interface(
        ertData, smooth_x, smooth_z,
        paraBoundary=params['paraBoundary'],
        quality=params['quality'],
        paraMaxCellSize=params['paraMaxCellSize'],
        paraDepth=params['paraDepth']
    )
    
    # Define regions based on markers
    regions = {
        1: {"name": "boundary", "marker": 1, "description": "Outside survey area"},
        2: {"name": "lower_layer", "marker": 2, "description": "Below velocity interface"},
        3: {"name": "upper_layer", "marker": 3, "description": "Above velocity interface"}
    }
    
    return meshafter, markers, regions


def create_joint_inversion_mesh(ertData, ttData, velocity_threshold=1200, **kwargs):
    """
    Create a mesh for joint ERT-seismic inversion by first inverting seismic data,
    extracting the velocity interface, and then creating a constrained ERT mesh.
    
    Args:
        ertData: PyGIMLi ERT data container
        ttData: PyGIMLi seismic travel time data container
        velocity_threshold: Threshold for velocity interface (default: 1200)
        **kwargs: Additional parameters including:
            - seismic_params: Dictionary of seismic inversion parameters
            - mesh_params: Dictionary of mesh generation parameters
            
    Returns:
        joint_mesh: Mesh suitable for constrained joint inversion
        seismic_manager: TravelTimeManager with seismic inversion results
        structure_data: Structure interface data
    """
    # Import required modules
    from pygimli.physics import traveltime as tt
    from watershed_geophysics.Geophy_modular.seismic_processor import (
        process_seismic_tomography, extract_velocity_structure
    )
    
    # Extract parameter dictionaries
    seismic_params = kwargs.get('seismic_params', {})
    mesh_params = kwargs.get('mesh_params', {})
    
    # Create mesh for seismic inversion if not provided
    if 'mesh' not in seismic_params:
        # Use ERT data to create a suitable mesh
        ert_manager = pg.physics.ert.ERTManager(ertData)
        seismic_mesh = ert_manager.createMesh(
            data=ertData, 
            quality=seismic_params.get('quality', 31),
            paraMaxCellSize=seismic_params.get('paraMaxCellSize', 5),
            paraBoundary=seismic_params.get('paraBoundary', 0.1),
            paraDepth=seismic_params.get('paraDepth', 30.0)
        )
        seismic_params['mesh'] = seismic_mesh
    
    # Process seismic tomography
    seismic_manager = process_seismic_tomography(ttData, **seismic_params)
    
    # Extract velocity interface
    smooth_x, smooth_z, structure_data = extract_velocity_structure(
        seismic_manager.paraDomain,
        seismic_manager.model.array(),
        threshold=velocity_threshold,
        interval=kwargs.get('interface_interval', 5.0)
    )
    
    # Create ERT mesh with interface structure
    joint_mesh, markers, regions = create_ert_mesh_with_structure(
        ertData, 
        (smooth_x, smooth_z),
        **mesh_params
    )
    
    return joint_mesh, seismic_manager, structure_data