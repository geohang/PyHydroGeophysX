"""
Forward modeling utilities for Seismic Refraction Tomography (SRT).
"""
import numpy as np
import pygimli as pg
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager
from typing import Tuple, Optional, Union, List, Dict, Any


class SeismicForwardModeling:
    """Class for forward modeling of Seismic Refraction Tomography (SRT) data."""
    
    def __init__(self, mesh: pg.Mesh, scheme: Optional[pg.DataContainer] = None):
        """
        Initialize seismic forward modeling.
        
        Args:
            mesh: PyGIMLI mesh for forward modeling
            scheme: Seismic data scheme
        """
        self.mesh = mesh
        self.scheme = scheme
        self.manager = TravelTimeManager()
        
        if scheme is not None:
            self.manager.setData(scheme)
        
        self.manager.setMesh(mesh)
    
    def set_scheme(self, scheme: pg.DataContainer) -> None:
        """
        Set seismic data scheme for forward modeling.
        
        Args:
            scheme: Seismic data scheme
        """
        self.scheme = scheme
        self.manager.setData(scheme)
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set mesh for forward modeling.
        
        Args:
            mesh: PyGIMLI mesh
        """
        self.mesh = mesh
        self.manager.setMesh(mesh)
    
    def forward(self, velocity_model: np.ndarray, slowness: bool = True) -> np.ndarray:
        """
        Compute forward response for a given velocity model.
        
        Args:
            velocity_model: Velocity model values (or slowness if slowness=True)
            slowness: Whether velocity_model is slowness (1/v)
            
        Returns:
            Forward response (travel times)
        """
        if not slowness:
            # Convert velocity to slowness
            slowness_values = 1.0 / velocity_model
        else:
            slowness_values = velocity_model
        
        # Calculate response
        response = self.manager.fop.response(slowness_values)
        
        return response

    
    @classmethod
    def create_synthetic_data(cls, 
                            sensor_x: np.ndarray, 
                            surface_points: Optional[np.ndarray] = None,
                            mesh: pg.Mesh = None, 
                            velocity_model: Optional[np.ndarray] = None,
                            slowness: bool = False,
                            shot_distance: float = 5,
                            noise_level: float = 0.05, 
                            noise_abs: float = 0.00001,
                            save_path: Optional[str] = None, 
                            show_data: bool = False,
                            verbose: bool = False,
                            seed: Optional[int] = None) -> Tuple[pg.DataContainer, pg.Mesh]:
        """
        Create synthetic seismic data using forward modeling.
        
        This method simulates a seismic survey by placing geophones along a surface,
        creating a measurement scheme, and performing forward modeling to generate
        synthetic travel time data.
        
        Args:
            sensor_x: X-coordinates of geophones
            surface_points: Surface coordinates for placing geophones [[x,y],...] 
                            If None, geophones will be placed on flat surface
            mesh: Mesh for forward modeling
            velocity_model: Velocity model values
            slowness: Whether velocity_model is slowness (1/v)
            shot_distance: Distance between shots
            noise_level: Level of relative noise to add
            noise_abs: Level of absolute noise to add
            save_path: Path to save synthetic data (if None, does not save)
            show_data: Whether to display data after creation
            verbose: Whether to show verbose output
            seed: Random seed for noise generation
            
        Returns:
            Tuple of (synthetic seismic data container, simulation mesh)
        """

        
        # Create seismic scheme (Refraction Data)
        scheme = tt.createRAData(sensor_x, shotDistance=shot_distance)
        
        # If surface points are provided, place geophones on the surface
        if surface_points is not None:
            sensor_positions = np.zeros((len(sensor_x), 2))
            
            # Find the closest point on the surface for each geophone
            for i, sx in enumerate(sensor_x):
                distances = np.abs(surface_points[:, 0] - sx)
                index = np.argmin(distances)
                sensor_positions[i, 0] = surface_points[index, 0]
                sensor_positions[i, 1] = surface_points[index, 1]
                
            # Set sensor positions
            scheme.setSensors(sensor_positions)
        
        # Initialize manager
        manager = TravelTimeManager()
        
        # If no mesh is provided, create a simple one
        if mesh is None:
            if surface_points is not None:
                # Create a mesh based on the surface profile
                x_min, x_max = np.min(sensor_positions[:, 0]) - 10, np.max(sensor_positions[:, 0]) + 10
                y_min = np.min(sensor_positions[:, 1]) - 20
                
                # Create a simple grid
                mesh = pg.createGrid(
                    x=np.linspace(x_min, x_max, 50),
                    y=np.linspace(y_min, 0, 20)
                )
                mesh = pg.meshtools.appendTriangleBoundary(mesh, marker=1, xbound=50, ybound=50)
            else:
                # Create a simple mesh for flat surface
                x_min, x_max = np.min(sensor_x) - 10, np.max(sensor_x) + 10
                
                # Create a simple grid
                mesh = pg.createGrid(
                    x=np.linspace(x_min, x_max, 50),
                    y=np.linspace(-20, 0, 20)
                )
                mesh = pg.meshtools.appendTriangleBoundary(mesh, marker=1, xbound=50, ybound=50)
        
        # Create velocity model if not provided
        if velocity_model is None:
            # Create a simple velocity model that increases with depth
            velocity_model = np.ones(mesh.cellCount())
            centers = np.array(mesh.cellCenters())
            
            # Velocity increases with depth
            for i, center in enumerate(centers):
                velocity_model[i] = 500 + 50 * abs(center[1])
        
        # Prepare slowness model
        if not slowness:
            slowness_values = 1.0 / velocity_model
        else:
            slowness_values = velocity_model
        
        # Simulate data
        synth_data = manager.simulate(
            slowness=slowness_values,
            scheme=scheme,
            mesh=mesh,
            noiseLevel=noise_level,
            noiseAbs=noise_abs,
            verbose=verbose
        )
        
        # Save data if a path is provided
        if save_path is not None:
            synth_data.save(save_path)
        
        # Display data if requested
        if show_data:
            pg.plt.figure(figsize=(10, 6))
            tt.drawFirstPicks(pg.plt.gca(), synth_data)
            pg.plt.show()
        
        return synth_data, mesh
    

    @staticmethod
    def draw_first_picks(ax, data, tt=None, plotva=False, **kwargs):
        """Plot first arrivals as lines.
        
        Parameters
        ----------
        ax : matplotlib.axes
            axis to draw the lines in
        data : :gimliapi:`GIMLI::DataContainer`
            data containing shots ("s"), geophones ("g") and traveltimes ("t")
        tt : array, optional
            traveltimes to use instead of data("t")
        plotva : bool, optional
            plot apparent velocity instead of traveltimes
        
        Return
        ------
        ax : matplotlib.axes
            the modified axis
        """
        # Extract coordinates
        px = pg.x(data)
        gx = np.array([px[int(g)] for g in data("g")])
        sx = np.array([px[int(s)] for s in data("s")])
        
        # Get traveltimes
        if tt is None:
            tt = np.array(data("t"))
        if plotva:
            tt = np.absolute(gx - sx) / tt
        
        # Find unique source positions    
        uns = np.unique(sx)
        
        # Override kwargs with clean, minimalist style
        kwargs['color'] = 'black'
        kwargs['linestyle'] = '--'
        kwargs['linewidth'] = 0.9
        kwargs['marker'] = None  # No markers on the lines
        
        # Plot for each source
        for i, si in enumerate(uns):
            ti = tt[sx == si]
            gi = gx[sx == si]
            ii = gi.argsort()
            
            # Plot line
            ax.plot(gi[ii], ti[ii], **kwargs)
            
            # Add source marker as black square at top
            ax.plot(si, 0.0, 's', color='black', markersize=4, 
                    markeredgecolor='black', markeredgewidth=0.5)
        
        # Clean grid style
        ax.grid(True, linestyle='-', linewidth=0.2, color='lightgray')
        
        # Set proper axis labels with units
        if plotva:
            ax.set_ylabel("Apparent velocity (m s$^{-1}$)")
        else:
            ax.set_ylabel("Traveltime (s)")
        
        ax.set_xlabel("Distance (m)")
        
        # Invert y-axis for traveltimes
        ax.invert_yaxis()

        return ax