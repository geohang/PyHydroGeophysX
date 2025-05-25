"""
Seismic Refraction Tomography (SRT) Forward Modeling Utilities.

This module provides tools for performing forward modeling of seismic travel times,
a core component of Seismic Refraction Tomography. It includes a class
`SeismicForwardModeling` to manage the modeling process and a static method
for creating synthetic datasets, along with a utility for plotting first arrivals.

The functionalities rely on PyGIMLi for mesh handling and travel time calculations.
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
        Initialize the SeismicForwardModeling class.
        
        Args:
            mesh (pg.Mesh): A PyGIMLi mesh object that defines the subsurface
                            discretization for the forward model.
            scheme (Optional[pg.DataContainer], optional): A PyGIMLi DataContainer 
                that defines the seismic survey geometry (shot and receiver positions).
                If provided, it is set for the forward operator. Defaults to None.
        """
        self.mesh = mesh
        self.scheme = scheme
        self.manager = TravelTimeManager()
        
        if scheme is not None:
            self.manager.setData(scheme)
        
        self.manager.setMesh(mesh)
    
    def set_scheme(self, scheme: pg.DataContainer) -> None:
        """
        Set or update the seismic data scheme (survey geometry) for the forward modeling.
        
        Args:
            scheme (pg.DataContainer): A PyGIMLi DataContainer defining the sensor and shot
                                       positions for the seismic survey.
        """
        self.scheme = scheme
        self.manager.setData(scheme)
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set or update the mesh used for the forward modeling.
        
        Args:
            mesh (pg.Mesh): A PyGIMLi mesh object representing the subsurface model domain.
        """
        self.mesh = mesh
        self.manager.setMesh(mesh)
    
    def forward(self, velocity_model: np.ndarray, slowness: bool = True) -> np.ndarray:
        """
        Compute seismic travel times (forward response) for a given velocity or slowness model.
        
        The forward calculation is performed using the PyGIMLi `TravelTimeManager`.
        
        Args:
            velocity_model (np.ndarray): An array representing the velocity model.
                                       The values can either be velocities (e.g., m/s) or
                                       slowness (e.g., s/m), depending on the `slowness` argument.
                                       The array should correspond to the cells of the mesh.
            slowness (bool, optional): If True, `velocity_model` is interpreted as slowness values (s/m).
                                     If False, `velocity_model` is interpreted as velocity values (m/s)
                                     and will be converted to slowness (1/velocity) internally.
                                     Defaults to True.
            
        Returns:
            np.ndarray: An array of calculated travel times corresponding to the source-receiver
                        pairs defined in the data scheme.
        """
        if not slowness:
            # Convert velocity model to slowness model (s/m)
            # Add a small epsilon to velocity_model to avoid division by zero if any velocity is zero.
            slowness_values = 1.0 / (velocity_model + 1e-12) 
        else:
            # Input is already slowness
            slowness_values = velocity_model
        
        # Calculate travel times using the forward operator (fop) from the manager
        # The `response` method of the ERTModelling (which fop is an instance of, via TravelTimeManager)
        # takes the model (slowness in this case) and returns the data (travel times).
        travel_times = self.manager.fop.response(slowness_values)
        
        return travel_times

    
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
            sensor_x (np.ndarray): 1D array of X-coordinates for geophone locations.
            surface_points (Optional[np.ndarray], optional): A 2D array of [x,y] coordinates
                defining the surface topography. If provided, geophone y-coordinates will be
                interpolated/snapped to this surface. If None, geophones are placed on a flat
                surface (y=0). Defaults to None.
            mesh (Optional[pg.Mesh], optional): A PyGIMLi mesh to be used for the forward simulation.
                If None, a simple rectangular mesh is automatically generated based on
                sensor locations and `surface_points`. Defaults to None.
            velocity_model (Optional[np.ndarray], optional): An array of velocity values (m/s)
                or slowness values (s/m) for each cell in the `mesh`. If None, a simple
                default velocity model (increasing with depth) is generated. Defaults to None.
            slowness (bool, optional): If True, `velocity_model` is treated as slowness.
                If False (default), it's treated as velocity and converted to slowness.
                Defaults to False.
            shot_distance (float, optional): Distance between shot points in the created survey scheme.
                Defaults to 5.
            noise_level (float, optional): Relative noise level to add to the synthetic travel times
                (fraction, e.g., 0.05 for 5%). Defaults to 0.05.
            noise_abs (float, optional): Absolute noise level (in seconds) to add to synthetic travel times.
                Defaults to 0.00001.
            save_path (Optional[str], optional): If provided, the path to save the generated
                synthetic data (PyGIMLi DataContainer format). Defaults to None (no saving).
            show_data (bool, optional): If True, displays the generated travel time data using
                `tt.drawFirstPicks`. Defaults to False.
            verbose (bool, optional): If True, enables verbose output during simulation.
                Defaults to False.
            seed (Optional[int], optional): Random seed for noise generation, ensuring reproducibility.
                Defaults to None (no fixed seed).
            
        Returns:
            Tuple[pg.DataContainer, pg.Mesh]:
                - synth_data (pg.DataContainer): The generated synthetic seismic data (travel times).
                - mesh (pg.Mesh): The mesh used for the simulation (either provided or auto-generated).
        """

        
        # Create seismic scheme (Refraction Data)
        scheme = tt.createRAData(sensor_x, shotDistance=shot_distance)
        
        # If surface points are provided, adjust sensor Y-coordinates to conform to the surface.
        # This assumes sensor_x are the target x-locations, and y is found by interpolation/snapping.
        if surface_points is not None:
            sensor_positions_on_surface = np.zeros((len(sensor_x), 2))
            # Interpolate y-coordinate for each sensor_x from the surface_points
            # Ensure surface_points are sorted by x for np.interp
            sorted_surface_indices = np.argsort(surface_points[:, 0])
            surface_x_sorted = surface_points[sorted_surface_indices, 0]
            surface_y_sorted = surface_points[sorted_surface_indices, 1]
            
            interp_y_values = np.interp(sensor_x, surface_x_sorted, surface_y_sorted)
            sensor_positions_on_surface[:, 0] = sensor_x
            sensor_positions_on_surface[:, 1] = interp_y_values
            
            scheme.setSensors(sensor_positions_on_surface)
        # If surface_points is None, scheme created by createRAData already has flat (y=0) sensor positions.
        
        # Initialize TravelTimeManager
        manager = TravelTimeManager() # Note: cls() cannot be used here as it's a @classmethod
                                      # If an instance of SeismicForwardModeling was needed, it would be cls(mesh, scheme)
                                      # However, TravelTimeManager.simulate is a standalone utility.
        
        # If no mesh is provided, create a default one.
        if mesh is None:
            # Determine mesh boundaries based on sensor positions
            if scheme.sensorCount() > 0:
                current_sensor_pos = np.array([scheme.sensorPosition(i).array() for i in range(scheme.sensorCount())])
                x_min_sensors, x_max_sensors = np.min(current_sensor_pos[:,0]), np.max(current_sensor_pos[:,0])
                y_min_sensors, y_max_sensors = np.min(current_sensor_pos[:,1]), np.max(current_sensor_pos[:,1])
            else: # Fallback if scheme has no sensors (should not happen with createRAData)
                x_min_sensors, x_max_sensors = np.min(sensor_x) -10, np.max(sensor_x) + 10
                y_max_sensors = 0
                y_min_sensors = -20 # Default depth if no surface info

            mesh_x_coords = np.linspace(x_min_sensors - 10, x_max_sensors + 10, 50)
            # Mesh depth goes from 20 units below lowest sensor/surface point up to highest sensor/surface point
            mesh_y_coords = np.linspace(y_min_sensors - 20, y_max_sensors, 20) 
            
            mesh = pg.createGrid(x=mesh_x_coords, y=mesh_y_coords)
            # Append boundary for better numerical stability in forward modeling
            mesh = pg.meshtools.appendTriangleBoundary(mesh, marker=1, xbound=50, ybound=50)
        
        # Create a default velocity model if not provided
        if velocity_model is None:
            velocity_model = np.ones(mesh.cellCount())
            cell_centers = mesh.cellCenters().array() # Get cell centers as numpy array
            # Simple velocity model: velocity increases with depth (absolute y-value, assuming y is negative downwards)
            # Example: 500 m/s at surface (y=0), increasing by 50 m/s per unit depth.
            velocity_model = 500 + 50 * np.abs(cell_centers[:, 1]) 
        
        # Convert velocity model to slowness if necessary
        if not slowness:
            slowness_model = 1.0 / (velocity_model + 1e-12) # Add epsilon to avoid division by zero
        else:
            slowness_model = velocity_model
        
        # Simulate travel time data using the manager
        synth_data = manager.simulate(
            slowness=slowness_model, # Expects slowness
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
        ax (matplotlib.axes.Axes): The matplotlib axes object on which to plot.
        data (pg.DataContainer): A PyGIMLi DataContainer object that includes sensor positions
                                 and travel time data. It must contain 's' (source index),
                                 'g' (geophone/receiver index), and 't' (travel time) tokens.
        tt (Optional[np.ndarray], optional): An array of travel times to plot. If provided,
                                             these values will be used instead of `data("t")`.
                                             Defaults to None.
        plotva (bool, optional): If True, plots apparent velocity (distance/traveltime)
                                 instead of travel times. Defaults to False.
        **kwargs: Additional keyword arguments passed directly to `ax.plot()` for line styling.
        
        Returns:
            matplotlib.axes.Axes: The modified matplotlib axes object with the plot.
        
        Note:
            This method provides a convenient way to visualize seismic first arrival picks,
            often used for assessing data quality or comparing observed and modeled travel times.
        """
        # Extract sensor positions (x-coordinates) from the data container
        # pg.x(data) gets all unique x-positions of sensors.
        sensor_x_coords = pg.x(data) 
        # Get x-coordinates for receivers (geophones) and sources (shots) for each data point
        geophone_x = np.array([sensor_x_coords[int(g_idx)] for g_idx in data("g")])
        source_x = np.array([sensor_x_coords[int(s_idx)] for s_idx in data("s")])
        
        # Get travel times: use provided `tt` array or data("t") from the container
        if tt is None:
            travel_times_to_plot = np.array(data("t"))
        else:
            travel_times_to_plot = np.array(tt) # Ensure it's a NumPy array

        if plotva: # If plotting apparent velocity
            # Calculate apparent velocity: distance / travel_time
            # Add small epsilon to travel_times_to_plot to avoid division by zero
            apparent_velocity = np.abs(geophone_x - source_x) / (travel_times_to_plot + 1e-12)
            y_values_to_plot = apparent_velocity
            ax.set_ylabel("Apparent Velocity (m/s)")
            ax.invert_yaxis() # Typically, higher velocity is plotted upwards
        else: # Plotting travel times
            y_values_to_plot = travel_times_to_plot
            ax.set_ylabel("Traveltime (s)")
            ax.invert_yaxis() # Travel time plots often have time increasing downwards
        
        # Find unique source positions to plot each shot gather separately
        unique_source_x_coords = np.unique(source_x)
        
        # Default plotting style for lines if not overridden by kwargs
        plot_kwargs = {
            'color': 'black',
            'linestyle': '--',
            'linewidth': 0.9,
            'marker': None # No markers on the lines themselves
        }
        plot_kwargs.update(kwargs) # Allow user to override defaults
        
        # Plot data for each source
        for i, current_source_x in enumerate(unique_source_x_coords):
            # Select data for the current source
            source_mask = (source_x == current_source_x)
            current_geophone_x = geophone_x[source_mask]
            current_y_values = y_values_to_plot[source_mask]
            
            # Sort by geophone position for connected line plot
            sort_indices = current_geophone_x.argsort()
            
            # Plot the line for this shot gather
            ax.plot(current_geophone_x[sort_indices], current_y_values[sort_indices], **plot_kwargs)
            
            # Add a marker for the source position at y=0 (or min y if apparent velocity)
            # Source marker (black square at y=0 or bottom of y-axis for velocity)
            source_marker_y_pos = 0.0 if not plotva else ax.get_ylim()[0] # bottom for VA
            ax.plot(current_source_x, source_marker_y_pos, 
                    marker='s', color='black', markersize=5, 
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='None') # Ensure only marker is plotted
        
        # Apply a clean grid style
        ax.grid(True, linestyle='-', linewidth=0.2, color='lightgray', alpha=0.7)
        
        # Set common x-axis label
        ax.set_xlabel("Distance (m)")
        
        # Standard y-axis configuration (inversion handled above based on plotva)
        # ax.invert_yaxis() was handled above

        return ax