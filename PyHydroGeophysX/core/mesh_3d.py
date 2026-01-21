"""
3D Mesh utilities for geophysical modeling and inversion.

This module provides tools for creating 3D meshes using GMSH and PyGIMLi
for applications such as 3D ERT forward modeling and inversion.
"""

import numpy as np
import pandas as pd
import subprocess
import os
from typing import Tuple, List, Optional, Union, Dict
import pygimli as pg
import pygimli.meshtools as mt

try:
    from pygimli.meshtools import readGmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class Mesh3DCreator:
    """
    Class for creating 3D meshes for geophysical modeling.
    
    This class provides methods for creating complex 3D meshes with electrode
    positions embedded, suitable for ERT, SRT, and other geophysical applications.
    
    Example
    -------
    >>> creator = Mesh3DCreator()
    >>> geometry = creator.create_box_geometry(
    ...     length=10.0, width=5.0, height=3.0,
    ...     electrode_positions=electrode_positions
    ... )
    >>> mesh = creator.generate_mesh(geometry, output_name='my_mesh')
    """
    
    def __init__(self, 
                 gmsh_path: str = 'gmsh',
                 mesh_directory: str = './mesh_output',
                 elec_refinement: float = 0.05,
                 node_refinement: float = 0.1,
                 attractor_distance: float = 0.3):
        """
        Initialize 3D mesh creator.
        
        Parameters
        ----------
        gmsh_path : str
            Path to gmsh executable (default: 'gmsh' assumes it's in PATH)
        mesh_directory : str
            Directory for mesh output files
        elec_refinement : float
            Mesh refinement size at electrode positions (in meters)
        node_refinement : float
            Mesh refinement size at corner nodes (in meters)
        attractor_distance : float
            Maximum distance for attractor field effect (in meters)
        """
        self.gmsh_path = gmsh_path
        self.mesh_directory = mesh_directory
        self.elec_refinement = elec_refinement
        self.node_refinement = node_refinement
        self.attractor_distance = attractor_distance
        
        # Create output directory if it doesn't exist
        os.makedirs(mesh_directory, exist_ok=True)
    
    def create_electrode_grid(self,
                              x_positions: np.ndarray,
                              y_positions: np.ndarray,
                              z_positions: np.ndarray = None,
                              surface_elevation: callable = None) -> pd.DataFrame:
        """
        Create a grid of electrode positions.
        
        Parameters
        ----------
        x_positions : np.ndarray
            X coordinates of electrodes
        y_positions : np.ndarray
            Y coordinates of electrodes  
        z_positions : np.ndarray, optional
            Z coordinates of electrodes. If None, uses surface_elevation or 0
        surface_elevation : callable, optional
            Function that returns z given (x, y) coordinates
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['n', 'x', 'y', 'z'] for electrode positions
        """
        # Flatten inputs if they are 2D grids
        x_flat = np.array(x_positions).ravel()
        y_flat = np.array(y_positions).ravel()
        
        n_electrodes = len(x_flat)
        
        elec_df = pd.DataFrame({
            'n': np.arange(1, n_electrodes + 1, dtype=int),
            'x': x_flat,
            'y': y_flat,
            'z': np.zeros(n_electrodes)
        })
        
        if z_positions is not None:
            elec_df['z'] = np.array(z_positions).ravel()
        elif surface_elevation is not None:
            elec_df['z'] = [surface_elevation(x, y) for x, y in zip(x_flat, y_flat)]
        
        return elec_df.round(6)
    
    def create_surface_electrode_array(self,
                                       nx: int,
                                       ny: int,
                                       dx: float,
                                       dy: float,
                                       x_offset: float = 0.0,
                                       y_offset: float = 0.0,
                                       z: float = 0.0) -> pd.DataFrame:
        """
        Create a regular surface electrode array.
        
        Parameters
        ----------
        nx : int
            Number of electrodes in x direction
        ny : int
            Number of electrodes in y direction
        dx : float
            Electrode spacing in x direction (meters)
        dy : float
            Electrode spacing in y direction (meters)
        x_offset : float
            Starting x coordinate
        y_offset : float
            Starting y coordinate
        z : float or np.ndarray
            Z coordinate(s) for electrodes
            
        Returns
        -------
        pd.DataFrame
            Electrode positions with columns ['n', 'x', 'y', 'z']
        """
        x_coords = np.linspace(x_offset, x_offset + (nx - 1) * dx, nx)
        y_coords = np.linspace(y_offset, y_offset + (ny - 1) * dy, ny)
        
        # Create meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)
        
        n_electrodes = nx * ny
        
        elec_df = pd.DataFrame({
            'n': np.arange(1, n_electrodes + 1, dtype=int),
            'x': X.ravel(),
            'y': Y.ravel(),
            'z': np.full(n_electrodes, z) if np.isscalar(z) else z.ravel()
        })
        
        return elec_df.round(6)
    
    def create_borehole_electrode_array(self,
                                        borehole_x: float,
                                        borehole_y: float,
                                        z_positions: np.ndarray,
                                        electrode_start_number: int = 1) -> pd.DataFrame:
        """
        Create electrode array for a single borehole.
        
        Parameters
        ----------
        borehole_x : float
            X coordinate of borehole
        borehole_y : float
            Y coordinate of borehole
        z_positions : np.ndarray
            Z coordinates of electrodes (negative for depth)
        electrode_start_number : int
            Starting electrode number
            
        Returns
        -------
        pd.DataFrame
            Electrode positions with columns ['n', 'x', 'y', 'z']
        """
        n_electrodes = len(z_positions)
        
        elec_df = pd.DataFrame({
            'n': np.arange(electrode_start_number, electrode_start_number + n_electrodes, dtype=int),
            'x': np.full(n_electrodes, borehole_x),
            'y': np.full(n_electrodes, borehole_y),
            'z': z_positions
        })
        
        return elec_df.round(6)
    
    def create_crosshole_electrode_array(self,
                                         borehole_positions: List[Tuple[float, float]],
                                         z_positions: np.ndarray) -> pd.DataFrame:
        """
        Create electrode arrays for multiple boreholes (crosshole setup).
        
        Parameters
        ----------
        borehole_positions : list of tuples
            List of (x, y) coordinates for each borehole
        z_positions : np.ndarray
            Z coordinates of electrodes in each borehole
            
        Returns
        -------
        pd.DataFrame
            Combined electrode positions for all boreholes
        """
        all_electrodes = []
        current_n = 1
        
        for bh_x, bh_y in borehole_positions:
            bh_elec = self.create_borehole_electrode_array(
                bh_x, bh_y, z_positions, electrode_start_number=current_n
            )
            all_electrodes.append(bh_elec)
            current_n += len(z_positions)
        
        return pd.concat(all_electrodes, ignore_index=True)
    
    def write_gmsh_geo_file(self,
                            geometry: Dict,
                            electrode_positions: pd.DataFrame,
                            output_name: str,
                            neumann_nodes: pd.DataFrame = None,
                            include_volume: bool = True) -> str:
        """
        Write a GMSH .geo file for the 3D mesh.
        
        Parameters
        ----------
        geometry : dict
            Dictionary containing geometry parameters:
            - 'length': length in x direction
            - 'width': width in y direction  
            - 'height': height in z direction
            - 'origin': (x, y, z) origin point (default: (0, 0, 0))
        electrode_positions : pd.DataFrame
            Electrode positions with columns ['n', 'x', 'y', 'z']
        output_name : str
            Base name for output files
        neumann_nodes : pd.DataFrame, optional
            Additional nodes for Neumann boundary conditions
        include_volume : bool
            Whether to include volume definition
            
        Returns
        -------
        str
            Path to the created .geo file
        """
        geo_file = os.path.join(self.mesh_directory, f'{output_name}.geo')
        
        # Extract geometry parameters
        length = geometry['length']
        width = geometry['width']
        height = geometry['height']
        origin = geometry.get('origin', (0, 0, 0))
        
        with open(geo_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("// 3D Mesh generated by PyHydroGeophysX\n")
            f.write(f"// Domain: {length} x {width} x {height} meters\n\n")
            
            # Write electrode nodes
            f.write("// Electrode nodes\n")
            point_id = 1
            for _, row in electrode_positions.iterrows():
                f.write(f"Point({point_id}) = {{{row['x']}, {row['y']}, {row['z']}, {self.elec_refinement}}};\n")
                point_id += 1
            
            n_electrodes = len(electrode_positions)
            
            # Write Neumann nodes if provided
            if neumann_nodes is not None:
                f.write("\n// Neumann boundary nodes\n")
                for _, row in neumann_nodes.iterrows():
                    f.write(f"Point({point_id}) = {{{row['x']}, {row['y']}, {row['z']}, {self.elec_refinement}}};\n")
                    point_id += 1
            
            n_total_special_points = point_id - 1
            
            # Write corner nodes for the box
            f.write("\n// Domain corner nodes\n")
            corner_start = 501
            corners = [
                (origin[0], origin[1], origin[2]),
                (origin[0] + length, origin[1], origin[2]),
                (origin[0] + length, origin[1] + width, origin[2]),
                (origin[0], origin[1] + width, origin[2]),
                (origin[0], origin[1], origin[2] + height),
                (origin[0] + length, origin[1], origin[2] + height),
                (origin[0] + length, origin[1] + width, origin[2] + height),
                (origin[0], origin[1] + width, origin[2] + height),
            ]
            
            for i, (x, y, z) in enumerate(corners):
                f.write(f"Point({corner_start + i}) = {{{x}, {y}, {z}, {self.node_refinement}}};\n")
            
            # Write lines connecting corners
            f.write("\n// Lines\n")
            line_id = 1
            
            # Bottom face lines
            bottom_lines = [(501, 502), (502, 503), (503, 504), (504, 501)]
            for p1, p2 in bottom_lines:
                f.write(f"Line({line_id}) = {{{p1}, {p2}}};\n")
                line_id += 1
            
            # Top face lines  
            top_lines = [(505, 506), (506, 507), (507, 508), (508, 505)]
            for p1, p2 in top_lines:
                f.write(f"Line({line_id}) = {{{p1}, {p2}}};\n")
                line_id += 1
            
            # Vertical lines
            vertical_lines = [(501, 505), (502, 506), (503, 507), (504, 508)]
            for p1, p2 in vertical_lines:
                f.write(f"Line({line_id}) = {{{p1}, {p2}}};\n")
                line_id += 1
            
            # Write line loops
            f.write("\n// Line Loops\n")
            line_loops = [
                [1, 2, 3, 4],      # Bottom face
                [5, 6, 7, 8],      # Top face
                [1, 10, -5, -9],   # Front face
                [2, 11, -6, -10],  # Right face
                [3, 12, -7, -11],  # Back face
                [4, 9, -8, -12]    # Left face
            ]
            
            for i, loop in enumerate(line_loops, 1):
                f.write(f"Line Loop({i}) = {{{loop[0]}, {loop[1]}, {loop[2]}, {loop[3]}}};\n")
            
            # Write surfaces
            f.write("\n// Surfaces\n")
            for i in range(1, 7):
                f.write(f"Plane Surface({i}) = {{{i}}};\n")
            
            # Write surface loop
            f.write("\n// Surface Loop\n")
            f.write("Surface Loop(1) = {1, 2, 3, 4, 5, 6};\n")
            
            # Write volume
            if include_volume:
                f.write("\n// Volume\n")
                f.write("Volume(1) = {1};\n")
                
                # Embed electrode points in volume
                f.write(f"\nPoint{{1:{n_electrodes}}} In Volume{{1}};\n")
            
            # Write attractor field for mesh refinement near electrodes
            f.write("\n// Mesh refinement fields\n")
            f.write("Field[1] = Attractor;\n")
            f.write(f"Field[1].NodesList = {{1:{n_total_special_points}}};\n")
            f.write("Field[2] = Threshold;\n")
            f.write("Field[2].IField = 1;\n")
            f.write(f"Field[2].LcMin = {self.elec_refinement};\n")
            f.write(f"Field[2].LcMax = {self.node_refinement};\n")
            f.write(f"Field[2].DistMin = {self.elec_refinement};\n")
            f.write(f"Field[2].DistMax = {self.attractor_distance};\n")
            f.write("Field[3] = Min;\n")
            f.write("Field[3].FieldsList = {2};\n")
            f.write("Background Field = 3;\n")
            f.write("Mesh.CharacteristicLengthExtendFromBoundary = 0;\n")
            
            # Write physical groups
            f.write("\n// Physical groups\n")
            f.write(f"Physical Point(99) = {{1:{n_total_special_points}}};\n")
            f.write("Physical Surface(1) = {1:6};\n")
            if include_volume:
                f.write("Physical Volume(2) = {1};\n")
        
        return geo_file
    
    def generate_gmsh_mesh(self, geo_file: str, dimension: int = 3) -> str:
        """
        Call GMSH to generate mesh from .geo file.
        
        Parameters
        ----------
        geo_file : str
            Path to .geo file
        dimension : int
            Mesh dimension (2 or 3)
            
        Returns
        -------
        str
            Path to generated .msh file
        """
        msh_file = geo_file.replace('.geo', '.msh')
        
        try:
            subprocess.run(
                [self.gmsh_path, geo_file, f'-{dimension}', '-format', 'msh2'],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GMSH failed to generate mesh: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError(f"GMSH executable not found at: {self.gmsh_path}")
        
        return msh_file
    
    def convert_to_pygimli(self, msh_file: str, save_formats: List[str] = None) -> pg.Mesh:
        """
        Convert GMSH mesh to PyGIMLi format.
        
        Parameters
        ----------
        msh_file : str
            Path to .msh file
        save_formats : list of str, optional
            List of formats to save ('bms', 'vtk')
            
        Returns
        -------
        pg.Mesh
            PyGIMLi mesh object
        """
        if not GMSH_AVAILABLE:
            raise ImportError("readGmsh not available. Please install pygimli with gmsh support.")
        
        mesh = readGmsh(msh_file, verbose=True)
        
        if save_formats:
            base_name = msh_file.replace('.msh', '')
            if 'bms' in save_formats:
                mesh.save(f'{base_name}.bms')
            if 'vtk' in save_formats:
                mesh.exportVTK(f'{base_name}.vtk')
        
        return mesh
    
    def create_box_mesh(self,
                        length: float,
                        width: float,
                        height: float,
                        electrode_positions: pd.DataFrame,
                        output_name: str = 'box_mesh',
                        origin: Tuple[float, float, float] = (0, 0, 0),
                        neumann_nodes: pd.DataFrame = None) -> pg.Mesh:
        """
        Create a complete 3D box mesh with embedded electrodes.
        
        Parameters
        ----------
        length : float
            Box length in x direction (meters)
        width : float
            Box width in y direction (meters)
        height : float
            Box height in z direction (meters)
        electrode_positions : pd.DataFrame
            Electrode positions with columns ['n', 'x', 'y', 'z']
        output_name : str
            Base name for output files
        origin : tuple
            (x, y, z) origin point
        neumann_nodes : pd.DataFrame, optional
            Additional nodes for Neumann boundary conditions
            
        Returns
        -------
        pg.Mesh
            PyGIMLi 3D mesh
        """
        geometry = {
            'length': length,
            'width': width,
            'height': height,
            'origin': origin
        }
        
        # Write geo file
        geo_file = self.write_gmsh_geo_file(
            geometry, electrode_positions, output_name, neumann_nodes
        )
        
        # Generate mesh
        msh_file = self.generate_gmsh_mesh(geo_file)
        
        # Convert to PyGIMLi
        mesh = self.convert_to_pygimli(msh_file, save_formats=['bms', 'vtk'])
        
        return mesh
    
    def create_prism_mesh_from_2d(self,
                                  mesh2d: pg.Mesh,
                                  z_vector: np.ndarray,
                                  surface_marker: int = 1,
                                  subsurface_marker: int = 2) -> pg.Mesh:
        """
        Create a 3D prism mesh by extruding a 2D mesh along z-axis.
        
        This follows the PyGIMLi approach using triangular prisms.
        
        Parameters
        ----------
        mesh2d : pg.Mesh
            2D triangular mesh in x-y plane
        z_vector : np.ndarray
            Z coordinates for extrusion (negative for depth)
        surface_marker : int
            Marker for surface/boundary cells
        subsurface_marker : int
            Marker for subsurface cells
            
        Returns
        -------
        pg.Mesh
            3D prism mesh
        """
        mesh = mt.createMesh3D(
            mesh2d, z_vector,
            pg.core.MARKER_BOUND_HOMOGEN_NEUMANN,
            pg.core.MARKER_BOUND_MIXED
        )
        
        return mesh
    
    def create_surface_mesh_with_topography(self,
                                            electrode_positions: pd.DataFrame,
                                            topography: callable = None,
                                            boundary_extension: float = 1.4,
                                            area: float = 0.4,
                                            quality: float = 34.3) -> pg.Mesh:
        """
        Create a 2D surface mesh suitable for 3D extension.
        
        Parameters
        ----------
        electrode_positions : pd.DataFrame
            Electrode positions with columns ['x', 'y', 'z']
        topography : callable, optional
            Function returning elevation for (x, y)
        boundary_extension : float
            Fraction to extend boundary (1.4 = 40% extension)
        area : float
            Maximum triangle area
        quality : float
            Mesh quality parameter
            
        Returns
        -------
        pg.Mesh
            2D surface mesh
        """
        # Get unique x-y positions
        xy_positions = electrode_positions[['x', 'y']].drop_duplicates().values
        
        # Create rectangle around electrodes
        rect = mt.createRectangle(pnts=xy_positions, minBBOffset=boundary_extension, marker=2)
        
        # Add electrode positions as nodes
        for pos in xy_positions:
            rect.createNode(*pos, 0)
        
        # Create mesh
        mesh = mt.createMesh(rect, quality=quality, area=area)
        
        return mesh
    
    def create_3d_mesh_with_topography(self,
                                        electrode_positions: pd.DataFrame,
                                        topography_data: np.ndarray = None,
                                        topography_func: callable = None,
                                        topography_x: np.ndarray = None,
                                        topography_y: np.ndarray = None,
                                        para_depth: float = 20.0,
                                        boundary_extension: float = 1.4,
                                        boundary_depth: float = 5.0,
                                        para_max_cell_size: float = 5.0,
                                        surface_quality: float = 34,
                                        mesh_quality: float = 1.3,
                                        dz_fine: float = 0.5,
                                        dz_coarse: float = 2.0,
                                        use_prism_mesh: bool = True,
                                        markers: Dict = None) -> pg.Mesh:
        """
        Create a 3D mesh that follows topography with electrode positions.
        
        This method creates meshes that respect surface topography, essential for
        accurate ERT forward modeling in areas with significant terrain variation.
        
        Parameters
        ----------
        electrode_positions : pd.DataFrame
            Electrode positions with columns ['n', 'x', 'y', 'z'].
            The 'z' values should represent surface elevations at electrode locations.
        topography_data : np.ndarray, optional
            2D array of surface elevations. Shape should be (ny, nx).
        topography_func : callable, optional
            Function that returns elevation for given (x, y) coordinates.
            Alternative to topography_data.
        topography_x : np.ndarray, optional
            X coordinates for topography_data grid
        topography_y : np.ndarray, optional
            Y coordinates for topography_data grid
        para_depth : float
            Depth of the parameter domain below surface (meters)
        boundary_extension : float
            Fraction to extend boundary beyond electrodes (1.4 = 40% extension)
        boundary_depth : float
            Additional depth for boundary region (meters)
        para_max_cell_size : float
            Maximum cell size in parameter domain (meters)
        surface_quality : float
            Quality parameter for surface mesh triangulation
        mesh_quality : float
            Quality parameter for 3D mesh generation
        dz_fine : float
            Fine vertical spacing in electrode region (meters)
        dz_coarse : float
            Coarse vertical spacing in boundary region (meters)
        use_prism_mesh : bool
            If True, creates prism mesh (efficient for layered structures).
            If False, uses tetrahedral mesh.
        markers : dict, optional
            Dictionary with depth ranges for different markers.
            Example: {'surface': (0, 5, 2), 'middle': (5, 15, 3), 'deep': (15, 30, 1)}
            Format: (min_depth, max_depth, marker_value)
            
        Returns
        -------
        pg.Mesh
            3D mesh following topography
            
        Examples
        --------
        >>> creator = Mesh3DCreator()
        >>> # Create electrodes on topographic surface
        >>> electrodes = creator.create_surface_electrode_array(
        ...     nx=10, ny=6, dx=5.0, dy=5.0
        ... )
        >>> # Apply topography to electrodes
        >>> electrodes['z'] = my_topo_function(electrodes['x'], electrodes['y'])
        >>> # Create mesh
        >>> mesh = creator.create_3d_mesh_with_topography(
        ...     electrode_positions=electrodes,
        ...     topography_func=my_topo_function,
        ...     para_depth=20.0
        ... )
        """
        from scipy.interpolate import RegularGridInterpolator, interp2d
        
        # Get electrode bounds
        x_min, x_max = electrode_positions['x'].min(), electrode_positions['x'].max()
        y_min, y_max = electrode_positions['y'].min(), electrode_positions['y'].max()
        
        # Calculate boundary extension
        dx = x_max - x_min
        dy = y_max - y_min
        ext_x = dx * (boundary_extension - 1) / 2
        ext_y = dy * (boundary_extension - 1) / 2
        
        # Extended domain bounds
        x_min_ext = x_min - ext_x
        x_max_ext = x_max + ext_x
        y_min_ext = y_min - ext_y
        y_max_ext = y_max + ext_y
        
        # Create topography interpolator if data is provided
        if topography_func is None and topography_data is not None:
            if topography_x is None or topography_y is None:
                raise ValueError("topography_x and topography_y must be provided with topography_data")
            
            topo_interpolator = RegularGridInterpolator(
                (topography_y, topography_x), topography_data,
                method='linear', bounds_error=False,
                fill_value=np.mean(electrode_positions['z'])
            )
            topography_func = lambda x, y: topo_interpolator((y, x))
        
        # If no topography provided, use electrode z values to create one
        if topography_func is None:
            from scipy.interpolate import LinearNDInterpolator
            points = electrode_positions[['x', 'y']].values
            z_vals = electrode_positions['z'].values
            topo_interp = LinearNDInterpolator(points, z_vals, fill_value=np.mean(z_vals))
            topography_func = lambda x, y: topo_interp(x, y)
        
        # Get surface elevations
        z_surface = electrode_positions['z'].values
        z_max = np.max(z_surface)
        z_min_surface = np.min(z_surface)
        z_bottom = z_min_surface - para_depth - boundary_depth
        
        if use_prism_mesh:
            return self._create_prism_mesh_with_topography(
                electrode_positions=electrode_positions,
                topography_func=topography_func,
                x_bounds=(x_min_ext, x_max_ext),
                y_bounds=(y_min_ext, y_max_ext),
                z_max=z_max,
                z_bottom=z_bottom,
                para_depth=para_depth,
                boundary_depth=boundary_depth,
                surface_quality=surface_quality,
                para_max_cell_size=para_max_cell_size,
                dz_fine=dz_fine,
                dz_coarse=dz_coarse,
                markers=markers
            )
        else:
            return self._create_tetrahedral_mesh_with_topography(
                electrode_positions=electrode_positions,
                topography_func=topography_func,
                para_depth=para_depth,
                para_max_cell_size=para_max_cell_size,
                mesh_quality=mesh_quality
            )
    
    def _create_prism_mesh_with_topography(self,
                                            electrode_positions: pd.DataFrame,
                                            topography_func: callable,
                                            x_bounds: Tuple[float, float],
                                            y_bounds: Tuple[float, float],
                                            z_max: float,
                                            z_bottom: float,
                                            para_depth: float,
                                            boundary_depth: float,
                                            surface_quality: float,
                                            para_max_cell_size: float,
                                            dz_fine: float,
                                            dz_coarse: float,
                                            markers: Dict = None) -> pg.Mesh:
        """Create prism mesh with topography (internal method)."""
        
        # Get unique x-y positions
        xy_positions = electrode_positions[['x', 'y']].drop_duplicates().values
        
        # Create rectangle around electrodes
        rect = mt.createRectangle(
            pnts=xy_positions, 
            minBBOffset=1.4, 
            marker=2
        )
        
        # Add electrode positions as nodes
        for pos in xy_positions:
            rect.createNode(*pos, 0)
        
        # Create 2D mesh
        mesh2d = mt.createMesh(rect, quality=surface_quality, area=para_max_cell_size)
        
        # Add boundary region
        mesh2d_with_bnd = mt.appendTriangleBoundary(
            mesh2d, boundary=boundary_depth, isSubSurface=False, marker=1
        )
        
        # Create z-discretization vector
        # Fine resolution in electrode region, coarse in boundary
        z_top_layers = np.arange(0, para_depth * 0.2, dz_coarse)
        z_mid_layers = np.arange(z_top_layers[-1] if len(z_top_layers) > 0 else 0, 
                                  para_depth, dz_fine)
        z_bot_layers = np.arange(z_mid_layers[-1] if len(z_mid_layers) > 0 else para_depth, 
                                  para_depth + boundary_depth + 0.1, dz_coarse)
        
        z_vec_relative = -np.concatenate([
            z_top_layers, 
            z_mid_layers[1:] if len(z_mid_layers) > 1 else [],
            z_bot_layers[1:] if len(z_bot_layers) > 1 else []
        ])
        
        # Create 3D prism mesh
        mesh3d = mt.createMesh3D(
            mesh2d_with_bnd, z_vec_relative,
            pg.core.MARKER_BOUND_HOMOGEN_NEUMANN,
            pg.core.MARKER_BOUND_MIXED
        )
        
        # Apply topography: shift z-coordinates based on surface elevation
        # Get node positions and adjust z based on local surface elevation
        nodes = mesh3d.nodes()
        for node in nodes:
            x, y, z = node.x(), node.y(), node.z()
            # Get local surface elevation
            try:
                surface_z = float(topography_func(x, y))
                if np.isnan(surface_z):
                    surface_z = z_max
            except:
                surface_z = z_max
            
            # Shift z-coordinate (z is negative depth relative to surface)
            new_z = surface_z + z  # z is already negative
            node.setPos(pg.Pos(x, y, new_z))
        
        # Set markers based on depth if provided
        if markers is not None:
            for c in mesh3d.cells():
                cell_x = c.center().x()
                cell_y = c.center().y()
                cell_z = c.center().z()
                
                # Get local surface elevation
                try:
                    surface_z = float(topography_func(cell_x, cell_y))
                    if np.isnan(surface_z):
                        surface_z = z_max
                except:
                    surface_z = z_max
                
                depth = surface_z - cell_z  # Positive depth below surface
                
                # Assign marker based on depth ranges
                for name, (d_min, d_max, marker) in markers.items():
                    if d_min <= depth < d_max:
                        c.setMarker(marker)
                        break
        else:
            # Default: mark boundary cells
            for c in mesh3d.cells():
                cell_x = c.center().x()
                cell_y = c.center().y()
                cell_z = c.center().z()
                
                try:
                    surface_z = float(topography_func(cell_x, cell_y))
                    if np.isnan(surface_z):
                        surface_z = z_max
                except:
                    surface_z = z_max
                
                depth = surface_z - cell_z
                
                # Mark boundary region
                if depth < 0 or depth > para_depth:
                    c.setMarker(1)
                else:
                    c.setMarker(2)
        
        return mesh3d
    
    def _create_tetrahedral_mesh_with_topography(self,
                                                  electrode_positions: pd.DataFrame,
                                                  topography_func: callable,
                                                  para_depth: float,
                                                  para_max_cell_size: float,
                                                  mesh_quality: float) -> pg.Mesh:
        """Create tetrahedral mesh with topography using PyGIMLi PLC (internal method)."""
        from pygimli.physics import ert
        
        # Create sensor array
        sensors = electrode_positions[['x', 'y', 'z']].values
        
        # Create ERT data container (just for mesh generation)
        data = ert.createData(sensors, schemeName='dd')
        
        # Create 3D PLC mesh
        plc = mt.createParaMeshPLC3D(
            data,
            paraDepth=para_depth,
            paraMaxCellSize=para_max_cell_size,
            surfaceMeshQuality=34
        )
        
        # Generate mesh
        mesh = mt.createMesh(plc, quality=mesh_quality)
        
        return mesh
    
    def apply_topography_to_electrodes(self,
                                        electrode_positions: pd.DataFrame,
                                        topography_data: np.ndarray = None,
                                        topography_func: callable = None,
                                        topography_x: np.ndarray = None,
                                        topography_y: np.ndarray = None) -> pd.DataFrame:
        """
        Apply topography elevation to electrode positions.
        
        Parameters
        ----------
        electrode_positions : pd.DataFrame
            Electrode positions with columns ['n', 'x', 'y', 'z']
        topography_data : np.ndarray, optional
            2D array of surface elevations
        topography_func : callable, optional
            Function that returns elevation for (x, y)
        topography_x : np.ndarray, optional
            X coordinates for topography grid
        topography_y : np.ndarray, optional
            Y coordinates for topography grid
            
        Returns
        -------
        pd.DataFrame
            Updated electrode positions with z from topography
        """
        from scipy.interpolate import RegularGridInterpolator
        
        electrodes = electrode_positions.copy()
        
        if topography_func is not None:
            electrodes['z'] = [topography_func(x, y) 
                               for x, y in zip(electrodes['x'], electrodes['y'])]
        elif topography_data is not None:
            if topography_x is None or topography_y is None:
                raise ValueError("topography_x and topography_y required with topography_data")
            
            interp = RegularGridInterpolator(
                (topography_y, topography_x), topography_data,
                method='linear', bounds_error=False,
                fill_value=np.mean(topography_data)
            )
            electrodes['z'] = interp(electrodes[['y', 'x']].values)
        
        return electrodes
    
    def visualize_mesh(self, mesh_or_path: Union[pg.Mesh, str], 
                       show_edges: bool = True,
                       show_electrodes: bool = True,
                       electrode_positions: pd.DataFrame = None,
                       cmap: str = 'RdBu',
                       **kwargs) -> None:
        """
        Visualize the 3D mesh using PyVista.
        
        Parameters
        ----------
        mesh_or_path : pg.Mesh or str
            PyGIMLi mesh or path to VTK file
        show_edges : bool
            Whether to show mesh edges
        show_electrodes : bool
            Whether to show electrode positions
        electrode_positions : pd.DataFrame, optional
            Electrode positions to display
        cmap : str
            Colormap for visualization
        **kwargs
            Additional arguments passed to pv.Plotter
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization. Install with: pip install pyvista")
        
        pv.set_plot_theme("document")
        
        if isinstance(mesh_or_path, str):
            mesh = pv.read(mesh_or_path)
        else:
            # Convert PyGIMLi mesh to VTK file temporarily
            temp_vtk = os.path.join(self.mesh_directory, '_temp_viz.vtk')
            mesh_or_path.exportVTK(temp_vtk)
            mesh = pv.read(temp_vtk)
        
        plotter = pv.Plotter(**kwargs)
        plotter.add_mesh(
            mesh, 
            show_edges=show_edges, 
            cmap=cmap,
            **{k: v for k, v in kwargs.items() if k not in ['notebook']}
        )
        
        if show_electrodes and electrode_positions is not None:
            points = electrode_positions[['x', 'y', 'z']].values
            plotter.add_points(points, color='red', point_size=10)
        
        plotter.show_bounds(
            mesh=mesh, 
            grid='back', 
            location='outer', 
            ticks='both',
            font_size=12
        )
        
        plotter.show()


def create_3d_ert_mesh_from_modflow(model_grid: Dict,
                                    electrode_positions: pd.DataFrame,
                                    output_name: str = 'ert_3d_mesh',
                                    mesh_directory: str = './mesh_output',
                                    refinement_factor: float = 1.0) -> pg.Mesh:
    """
    Create a 3D ERT mesh compatible with MODFLOW grid structure.
    
    Parameters
    ----------
    model_grid : dict
        MODFLOW grid information containing:
        - 'nrow': number of rows
        - 'ncol': number of columns  
        - 'nlay': number of layers
        - 'delr': row spacing
        - 'delc': column spacing
        - 'top': top elevation array
        - 'botm': bottom elevation array
    electrode_positions : pd.DataFrame
        Electrode positions with columns ['n', 'x', 'y', 'z']
    output_name : str
        Base name for output files
    mesh_directory : str
        Directory for output files
    refinement_factor : float
        Factor for mesh refinement (smaller = finer mesh)
        
    Returns
    -------
    pg.Mesh
        3D mesh suitable for ERT forward modeling
    """
    # Calculate domain dimensions
    length = model_grid['ncol'] * model_grid['delr']
    width = model_grid['nrow'] * model_grid['delc']
    
    top_elevation = np.max(model_grid['top'])
    bot_elevation = np.min(model_grid['botm'])
    height = top_elevation - bot_elevation
    
    # Create mesh creator with appropriate refinement
    creator = Mesh3DCreator(
        mesh_directory=mesh_directory,
        elec_refinement=0.05 * refinement_factor,
        node_refinement=0.1 * refinement_factor,
        attractor_distance=0.3 * refinement_factor
    )
    
    # Create the mesh
    mesh = creator.create_box_mesh(
        length=length,
        width=width,
        height=height,
        electrode_positions=electrode_positions,
        output_name=output_name,
        origin=(0, 0, bot_elevation)
    )
    
    return mesh


def interpolate_modflow_to_3d_mesh(mesh: pg.Mesh,
                                   modflow_data: np.ndarray,
                                   model_grid: Dict,
                                   method: str = 'nearest') -> np.ndarray:
    """
    Interpolate MODFLOW 3D data to mesh cell centers.
    
    Parameters
    ----------
    mesh : pg.Mesh
        PyGIMLi 3D mesh
    modflow_data : np.ndarray
        3D array of MODFLOW data (nlay, nrow, ncol)
    model_grid : dict
        MODFLOW grid information
    method : str
        Interpolation method ('nearest', 'linear')
        
    Returns
    -------
    np.ndarray
        Interpolated values at mesh cell centers
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Get mesh cell centers
    cell_centers = np.array(mesh.cellCenters())
    
    # Create MODFLOW grid coordinates
    nlay, nrow, ncol = modflow_data.shape
    
    # X coordinates (column centers)
    x_coords = np.arange(0.5, ncol) * model_grid['delr']
    # Y coordinates (row centers)
    y_coords = np.arange(0.5, nrow) * model_grid['delc']
    # Z coordinates (layer centers) - need to handle variable thickness
    z_coords = np.zeros(nlay)
    for k in range(nlay):
        if k == 0:
            z_coords[k] = (model_grid['top'].mean() + model_grid['botm'][k].mean()) / 2
        else:
            z_coords[k] = (model_grid['botm'][k-1].mean() + model_grid['botm'][k].mean()) / 2
    
    # Create interpolator
    # Note: Need to flip data axes to match coordinate order
    interpolator = RegularGridInterpolator(
        (z_coords[::-1], y_coords, x_coords),
        modflow_data[::-1, :, :],
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Interpolate to mesh cell centers
    mesh_values = interpolator(cell_centers[:, [2, 1, 0]])
    
    return mesh_values


def create_3d_ert_data_container(electrode_positions: pd.DataFrame,
                                 scheme: str = 'dd',
                                 dimension: int = 3) -> pg.DataContainer:
    """
    Create a 3D ERT data container for forward modeling.
    
    Parameters
    ----------
    electrode_positions : pd.DataFrame
        Electrode positions with columns ['x', 'y', 'z']
    scheme : str
        Measurement scheme ('dd' for dipole-dipole, 'wa' for Wenner-alpha, etc.)
    dimension : int
        Dimension for geometric factor calculation (should be 3 for 3D)
        
    Returns
    -------
    pg.DataContainer
        ERT data container with electrode positions and measurement scheme
    """
    from pygimli.physics import ert
    
    # Create sensor positions array
    sensors = electrode_positions[['x', 'y', 'z']].values
    
    # Create data container
    data = ert.createData(sensors, schemeName=scheme)
    
    # Calculate 3D geometric factors
    data['k'] = ert.geometricFactors(data, dim=dimension)
    
    return data


def export_electrodes_to_csv(electrode_positions: pd.DataFrame,
                             output_path: str,
                             include_neumann: pd.DataFrame = None) -> None:
    """
    Export electrode positions to CSV file.
    
    Parameters
    ----------
    electrode_positions : pd.DataFrame
        Electrode positions
    output_path : str
        Path for output CSV file
    include_neumann : pd.DataFrame, optional
        Additional Neumann boundary nodes to include
    """
    if include_neumann is not None:
        all_electrodes = pd.concat([electrode_positions, include_neumann], ignore_index=True)
    else:
        all_electrodes = electrode_positions
    
    all_electrodes.to_csv(output_path, index=False)
