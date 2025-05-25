"""
Mesh Utilities for Geophysical Modeling and Inversion.

This module provides a collection of utility functions and a class for creating,
manipulating, and preparing meshes, primarily for use with the PyGIMLi library.
It includes functionalities such as:
- Creating meshes from defined geological layer boundaries.
- Extracting velocity interfaces from geophysical data on a mesh.
- Adding velocity interfaces to existing mesh geometries for structured inversion.
- A `MeshCreator` class that encapsulates some of these mesh creation processes.

The module leverages PyGIMLi for mesh data structures and meshing algorithms,
and SciPy for interpolation and signal processing tasks.
"""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from typing import Tuple, List, Optional, Union
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def create_mesh_from_layers(surface: np.ndarray,
                          line1: np.ndarray,
                          line2: np.ndarray,
                          bottom_depth: float = 30.0,
                          quality: float = 28,
                          area: float = 40) -> Tuple[pg.Mesh, np.ndarray, np.ndarray]:
    """
    Create a 2D mesh from three defined layer boundaries (surface, line1, line2)
    and a specified bottom depth. It also returns cell centers and markers.
    
    The function defines three main regions based on these boundaries:
    1. From `surface` down to `line1`.
    2. From `line1` down to `line2`.
    3. From `line2` down to the `bottom_depth`.

    Args:
        surface (np.ndarray): Coordinates of the top surface, as an array of [x, z] pairs.
                              Example: `[[x1,z1], [x2,z2], ..., [xn,zn]]`.
        line1 (np.ndarray): Coordinates of the first subsurface layer boundary, as [x, z] pairs.
        line2 (np.ndarray): Coordinates of the second subsurface layer boundary, as [x, z] pairs.
        bottom_depth (float, optional): Absolute elevation of the bottom of the mesh. 
                                      The current implementation uses this value directly as the
                                      Z-coordinate for the mesh bottom. For instance, if surface elevations
                                      are around 100m and `bottom_depth` is 30.0, the mesh bottom will be at Z=30.0.
                                      A previous commented line `min_surface_elev - bottom_depth` suggested
                                      it might have been intended as a depth relative to the minimum surface
                                      elevation, but this is not the current behavior. Defaults to 30.0.
        quality (float, optional): Mesh quality parameter for `mt.createMesh`. Higher values generally
                                 result in better quality meshes but may take longer to generate. 
                                 Defaults to 28.
        area (float, optional): Maximum cell area constraint for `mt.createMesh`. Defaults to 40.
        
    Returns:
        Tuple[pg.Mesh, np.ndarray, np.ndarray, pg.meshtools.Geometry]:
            - mesh (pg.Mesh): The generated PyGIMLi mesh.
            - mesh_centers (np.ndarray): Array of cell center coordinates (x, z) for each cell in the mesh.
            - markers (np.ndarray): Array of integer markers assigned to each cell in the mesh.
                                    The markers are hardcoded during polygon creation:
                                    - Layer 1 (surface to line1): marker 2 (from `layer1` and `Gline2` contributions)
                                    - Layer 2 (line1 to line2): marker 3 (from `Gline1`)
                                    - Layer 3 (line2 to bottom): marker 2 (from `layer2`)
                                    This assignment strategy might need review for consistency if specific
                                    marker values are critical for subsequent modeling steps.
            - geom (pg.meshtools.Geometry): The combined geometry object used to create the mesh.
    """
    # `bottom_elev` is set to `bottom_depth`. This means `bottom_depth` is treated as an absolute Z-coordinate.
    # The commented-out part `min_surface_elev - bottom_depth` would imply `bottom_depth` is a relative depth.
    # min_surface_elev = np.nanmin(surface[:,1]) # This line is not used for bottom_elev calculation currently.
    bottom_elev = bottom_depth 
    
    # Create reversed versions of line1 and line2 for constructing closed polygons
    line1r = line1.copy()
    line1r[:,0] = np.flip(line1[:,0])
    line1r[:,1] = np.flip(line1[:,1])
    
    line2r = line2.copy()
    line2r[:,0] = np.flip(line2[:,0])
    line2r[:,1] = np.flip(line2[:,1])
    
    # Create polygons for each layer. Markers are hardcoded here.
    # Polygon for the top layer (surface down to line1).
    # Note: The way these polygons are constructed seems to define regions rather than just layers.
    # `layer1` seems to define the region above `line1`, but its interaction with Gline2 is complex.
    # The marker assignment (e.g. marker=2 for layer1) should be carefully considered for its final effect on cell markers.

    # The "surface layer" polygon is defined by the surface line.
    # It's not closed, suggesting it's part of a larger geometry definition.
    # Area=0.1 is a meshing parameter for this specific polygon.
    layer1_poly = mt.createPolygon(surface,
                                   isClosed=False, 
                                   marker=2, # Hardcoded marker for this part of the geometry
                                   boundaryMarker=-1,
                                   interpolate='linear', 
                                   area=0.1) # Area constraint for meshing this polygon
    
    # Polygon for the "middle layer" (between line1 and line2).
    # This forms a closed polygon by combining line1 and the reversed line2 (line2r).
    middle_layer_poly = mt.createPolygon(np.vstack((line1, line2r)),
                                         isClosed=True, 
                                         marker=3, # Hardcoded marker for this layer
                                         boundaryMarker=1,
                                         interpolate='linear', 
                                         area=1) # Area constraint
    
    # `Gline2` seems to define the overall extent or a background region.
    # It connects the start of the surface, goes down to `bottom_elev` at the x-range of `line2`,
    # then across at `bottom_elev`, and back up to the end of the surface.
    # This is also marked with marker=2.
    outer_boundary_poly = mt.createPolygon([[surface[0,0], surface[0,1]], # Start of surface
                                           [line2[0,0], bottom_elev],    # Bottom-left based on line2's x-start
                                           [line2[-1,0], bottom_elev],   # Bottom-right based on line2's x-end
                                           [surface[-1,0], surface[-1,1]]], # End of surface
                                          isClosed=False, # Should this be closed to form the overall domain?
                                          marker=2, # Hardcoded marker
                                          boundaryMarker=1,
                                          interpolate='linear', 
                                          area=2) # Area constraint
    
    # Polygon for the "bottom layer" (from line2r down to `bottom_elev`).
    # This forms a closed region using the reversed line2, and lines extending down to `bottom_elev`.
    bottom_layer_poly = mt.createPolygon(np.vstack((line2r,
                                                    [[line2[0,0], line2[0,1]], # Point from original line2 start
                                                     [line2[0,0], bottom_elev], # Down from line2 start
                                                     [line2[-1,0], bottom_elev], # Across at bottom_elev
                                                     [line2[-1,0], line2[-1,1]]])), # Point from original line2 end (connects back to line2r start)
                                         isClosed=True, 
                                         marker=2, # Hardcoded marker for this layer
                                         area=2,   # Area constraint
                                         boundaryMarker=1)
    
    # Combine all geometric parts. The order might matter for how markers are assigned in overlapping regions.
    geom = layer1_poly + bottom_layer_poly + middle_layer_poly + outer_boundary_poly
    
    # Create mesh
    mesh = mt.createMesh(geom, quality=quality, area=area)
    
    # Get cell centers and markers
    mesh_centers = np.array(mesh.cellCenters())
    markers = np.array(mesh.cellMarkers())
    
    return mesh, mesh_centers, markers,geom







def extract_velocity_interface(mesh, velocity_data, threshold=1200, interval=4.0):
    """
    Extract an interface (e.g., bedrock) from velocity data on a mesh, defined by a threshold.
    
    The function bins cells by their x-coordinates, then for each bin, it searches vertically
    for where the `velocity_data` crosses a given `threshold`. Linear interpolation is used
    to find the precise depth of this crossing. The resulting interface points are then
    optionally smoothed using cubic interpolation and a Savitzky-Golay filter.
    
    Args:
        mesh (pg.Mesh): The PyGIMLi mesh object containing the cell structure.
        velocity_data (np.ndarray): Array of velocity values, one per cell in `mesh`.
        threshold (float, optional): The velocity value that defines the interface. 
                                   Defaults to 1200.
        interval (float, optional): Spacing (width) of the x-coordinate bins used to 
                                  search for the interface. Defaults to 4.0.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - x_dense (np.ndarray): Array of x-coordinates for the smoothed interface.
            - z_dense (np.ndarray): Array of z-coordinates (depths) for the smoothed interface.
                                    Note: The Savitzky-Golay filter parameters (`window_length=31`, 
                                    `polyorder=3`) are empirically chosen and might need tuning 
                                    for different datasets to achieve optimal smoothing.
    """
    # Get cell center coordinates
    cell_centers = mesh.cellCenters() # Returns array of [x, z] for each cell
    x_coords = cell_centers[:,0]
    z_coords = cell_centers[:,1]
    
    # Get x-range for complete boundary
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    # Create bins across the entire x-range
    x_bins = np.arange(x_min, x_max + interval, interval)
    
    # Arrays to store interface points
    interface_x = []
    interface_z = []
    
    # For each bin, find the velocity interface
    for i in range(len(x_bins)-1):
        # Get all cells in this x-range
        bin_indices = np.where((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]))[0]
        
        if len(bin_indices) > 0:
            # Get velocity values and depths for this bin
            bin_velocities = velocity_data[bin_indices]
            bin_depths = z_coords[bin_indices]
            
            # Sort by depth
            sort_indices = np.argsort(bin_depths)
            bin_velocities = bin_velocities[sort_indices]
            bin_depths = bin_depths[sort_indices]
            
            # Find where velocity crosses the threshold
            for j in range(1, len(bin_velocities)):
                if (bin_velocities[j-1] < threshold and bin_velocities[j] >= threshold) or \
                   (bin_velocities[j-1] >= threshold and bin_velocities[j] < threshold):
                    # Linear interpolation for exact interface depth
                    v1 = bin_velocities[j-1]
                    v2 = bin_velocities[j]
                    z1 = bin_depths[j-1]
                    z2 = bin_depths[j]
                    
                    # Calculate the interpolated z-value where velocity = threshold
                    ratio = (threshold - v1) / (v2 - v1)
                    interface_depth = z1 + ratio * (z2 - z1)
                    
                    interface_x.append((x_bins[i] + x_bins[i+1]) / 2)
                    interface_z.append(interface_depth)
                    break
    
    # Ensure we have interface points for the entire range
    # If first point is missing, extrapolate from the first available points
    if len(interface_x) > 0 and interface_x[0] > x_min + interval:
        interface_x.insert(0, x_min)
        # Use the slope of the first two points to extrapolate
        if len(interface_x) > 2:
            slope = (interface_z[1] - interface_z[0]) / (interface_x[1] - interface_x[0])
            interface_z.insert(0, interface_z[0] - slope * (interface_x[1] - x_min))
        else:
            interface_z.insert(0, interface_z[0])
    
    # If last point is missing, extrapolate from the last available points
    if len(interface_x) > 0 and interface_x[-1] < x_max - interval:
        interface_x.append(x_max)
        # Use the slope of the last two points to extrapolate
        if len(interface_x) > 2:
            slope = (interface_z[-1] - interface_z[-2]) / (interface_x[-1] - interface_x[-2])
            interface_z.append(interface_z[-1] + slope * (x_max - interface_x[-1]))
        else:
            interface_z.append(interface_z[-1])
    
    # Create a dense interpolation grid for smoothing
    x_dense = np.linspace(x_min, x_max, 500)  # 500 points for smooth curve
    
    # Apply cubic interpolation for smoother interface
    if len(interface_x) > 3:
        try:
            interp_func = interp1d(interface_x, interface_z, kind='cubic', 
                                  bounds_error=False, fill_value="extrapolate")
            z_dense = interp_func(x_dense)
            
            # Apply additional smoothing
            z_dense = savgol_filter(z_dense, window_length=31, polyorder=3)
        except:
            # Fall back to linear interpolation if cubic fails
            interp_func = interp1d(interface_x, interface_z, kind='linear',
                                  bounds_error=False, fill_value="extrapolate")
            z_dense = interp_func(x_dense)
    else:
        # Not enough points for cubic interpolation
        interp_func = interp1d(interface_x, interface_z, kind='linear',
                              bounds_error=False, fill_value="extrapolate")
        z_dense = interp_func(x_dense)
    
    return x_dense, z_dense


def add_velocity_interface(ertData, smooth_x, smooth_z, paraBoundary=2, boundary=1):
    """
    Add a velocity interface line to a mesh geometry derived from ERT data,
    and assign cell markers based on cell position relative to this interface
    and the survey area.

    The function performs the following steps:
    1. Creates an initial mesh geometry based on ERT sensor positions (`ertData`).
    2. Defines the velocity interface as a polygon line using `smooth_x` and `smooth_z`.
    3. Extends this interface line slightly beyond the survey data range.
    4. Adds the interface line to the initial geometry.
    5. Creates a new mesh (`meshafter`) from this combined geometry.
    6. Assigns markers to cells in `meshafter`:
        - Marker 1: Cells outside the defined survey area (laterally).
        - Marker 2: Cells inside the survey area and determined to be *below* the velocity interface.
        - Marker 3: Cells inside the survey area and determined to be *above* the velocity interface.
    The determination of "above" or "below" assumes a coordinate system where Z values (depths)
    are typically negative, or at least that smaller absolute Z values are shallower.
    The condition `abs(cell_y) < abs(interface_y)` means the cell center is shallower (less deep)
    than the interface at that x-location, thus it's marked as "above" (marker 3).
    Conversely, if `abs(cell_y) >= abs(interface_y)`, it's deeper or at the interface, marked as "below" (marker 2).
    It's assumed that the Z-axis points upwards (depths are negative or smaller values are shallower).
    
    Args:
        ertData (pg.DataContainer): ERT data container, used to get sensor positions for defining
                                   the initial mesh and survey area. It's assumed that
                                   `ertData.sensors()` are sorted by x-coordinate.
        smooth_x (np.ndarray): Array of x-coordinates for the velocity interface line.
        smooth_z (np.ndarray): Array of z-coordinates (depths) for the velocity interface line.
        paraBoundary (float, optional): Size of the parameter boundary used in `mt.createParaMeshPLC`
                                      and for extending the interface line. Defaults to 2.
        boundary (int, optional): This parameter is currently unused in the function body.
                                   Original intent might have been for boundary conditions in mesh creation.
        
    Returns:
        Tuple[np.ndarray, pg.Mesh]:
            - markers (np.ndarray): Array of cell markers assigned to the final mesh `meshafter`.
            - meshafter (pg.Mesh): The created PyGIMLi mesh with updated cell markers reflecting
                                   the regions defined by the velocity interface and survey area.
    """
    # Create the initial parameter mesh geometry based on ERT sensor locations
    geo = mt.createParaMeshPLC(ertData, quality=32, paraMaxCellSize=30, # quality and cell size for primary mesh
                               paraBoundary=paraBoundary, paraDepth=30.0,
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
    meshafter = mt.createMesh(geo_with_interface, quality=28)
    
    # Initialize all markers to 1 (outside region)
    markers = np.ones(meshafter.cellCount())
    
    # Identify the survey area
    survey_left = ertData.sensors()[0][0] - paraBoundary
    survey_right = ertData.sensors()[-1][0] + paraBoundary
    
    # Process each cell
    for i in range(meshafter.cellCount()):
        cell_x = meshafter.cell(i).center().x()
        cell_y = meshafter.cell(i).center().y()
        
        # Only modify markers for cells considered within the lateral extent of the survey area.
        if cell_x >= survey_left and cell_x <= survey_right:
            # Interpolate the z-coordinate (depth/elevation) of the velocity interface 
            # at the current cell's x-center.
            interface_y_at_cell_x = np.interp(cell_x, input_points[:, 0], input_points[:, 1])
            
            # Assign markers based on cell center position relative to the interface.
            # Assumes Z-axis points upwards (depths are negative, or smaller Z values are shallower).
            # `abs(cell_y) < abs(interface_y)`: True if cell_y is shallower than interface_y.
            # E.g., cell_y = -5, interface_y = -10. abs(-5) < abs(-10) is 5 < 10, so cell is above.
            # E.g., cell_y = -10, interface_y = -5. abs(-10) < abs(-5) is 10 < 5 (False), so cell is below.
            if abs(cell_y) < abs(interface_y_at_cell_x):
                markers[i] = 3  # Cell is shallower than (above) the interface.
            else:
                markers[i] = 2  # Cell is deeper than or at (below) the interface.
    
    # Ensure cells that were originally marked as 1 (outside region by meshafter.cellMarkers())
    # retain marker 1. This might be redundant if survey_left/right logic correctly handles this.
    # However, it explicitly preserves the original "outside" marker if meshafter assigned it.
    markers[meshafter.cellMarkers() == 1] = 1
    
    # Set the updated markers
    meshafter.setCellMarkers(markers)
    
    return markers, meshafter





class MeshCreator:
    """Class for creating and managing meshes for geophysical inversion."""
    
    def __init__(self, quality: float = 28, area: float = 40):
        """
        Initialize MeshCreator with quality and area parameters.
        
        Args:
            quality: Mesh quality parameter (higher is better)
            area: Maximum cell area
        """
        self.quality = quality
        self.area = area
    
    def create_from_layers(self, surface: np.ndarray, 
                          layers: List[np.ndarray],
                          bottom_depth: float = 30.0,
                          markers: List[int] = None) -> pg.Mesh:
        """
        Create a mesh from surface and layer boundaries.
        
        Args:
            surface (np.ndarray): Coordinates of the top surface, as an array of [x, z] pairs.
            layers (List[np.ndarray]): List of layer boundary coordinates. Each element is an array
                                      of [x, z] pairs defining a subsurface boundary.
            bottom_depth (float, optional): Absolute elevation of the bottom of the mesh.
                                          Passed to `create_mesh_from_layers`. Defaults to 30.0.
            markers (List[int], optional): List of markers for each layer. This parameter is currently
                                         NOT USED by this method, as the underlying `create_mesh_from_layers`
                                         has hardcoded marker assignments. If provided, it will be ignored
                                         for the 2-layer case. Defaults to None.
            
        Returns:
            pg.Mesh: The generated PyGIMLi mesh.
            pg.meshtools.Geometry: The geometry object from which the mesh was created (returned by the 
                                   standalone `create_mesh_from_layers` function).

        Note:
            This method currently only supports exactly 2 subsurface layer boundaries (plus the surface).
            It calls the standalone `create_mesh_from_layers` function, and thus shares its behavior
            regarding `bottom_depth` interpretation (absolute elevation) and the hardcoded layer marker
            assignments (typically resulting in markers 2, 3, 2 for the three regions).
            The normalization code for elevations (`surface_norm`, `layers_norm`) using `max_ele` has been
            commented out, meaning original elevation values are used directly.
        """
        if len(layers) < 1:
            raise ValueError("At least one layer boundary (in the `layers` list) is required.")
            
        # The `markers` parameter is defined but not actually used to control polygon markers
        # in the current 2-layer implementation, as `create_mesh_from_layers` has hardcoded markers.
        # If `markers` were to be used, it would require modifying `create_mesh_from_layers`
        # or implementing the polygon creation logic here.
        # if markers is None:
        #     markers = [2] * (len(layers) + 1) # Default: layer above first boundary is 2
        #     if len(layers) > 0:
        #         markers[1] = 3  # Layer between first and second boundary is 3
        
        # Normalization of elevation by max_ele was previously commented out.
        # Keeping it commented to reflect current behavior: absolute elevations are used.
        # max_ele = np.nanmax(surface[:,1])
        surface_processed = surface.copy()
        # surface_processed[:,1] = surface_processed[:,1]  #- max_ele # No normalization applied
        
        layers_processed = []
        for layer in layers:
            layer_processed = layer.copy()
            # layer_processed[:,1] = layer_processed[:,1] # - max_ele # No normalization applied
            layers_processed.append(layer_processed)
        
        # Create mesh using the standalone function, currently specific to 2 layers.
        if len(layers_processed) == 2:
            # `create_mesh_from_layers` returns: mesh, mesh_centers, cell_markers, geom
            mesh, _, _, geom = create_mesh_from_layers(
                surface_processed, layers_processed[0], layers_processed[1], 
                bottom_depth, self.quality, self.area
            )
            return mesh, geom # Return the mesh and the geometry object
        else:
            # If not 2 layers, raise NotImplementedError.
            raise NotImplementedError("Mesh creation from layers is currently implemented only for exactly 2 subsurface layer boundaries.")
    
    def create_from_ert_data(self, data, max_depth: float = 30.0, quality: float = 34):
        """
        Create a mesh suitable for ERT inversion from ERT data.
        
        Args:
            data: PyGIMLI ERT data object
            max_depth: Maximum depth of the mesh
            quality: Mesh quality parameter
            
        Returns:
            PyGIMLI mesh for ERT inversion
        """
        from pygimli.physics import ert
        ert_manager = ert.ERTManager(data)
        return ert_manager.createMesh(data=data, quality=quality)
