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
        Initialize seismic forward modeling for Seismic Refraction Tomography (SRT).
        
        Args:
            mesh (pg.Mesh): PyGIMLi mesh representing the subsurface discretization
                            for the travel time calculation. The mesh should have cells
                            where velocity (or slowness) values will be defined.
            scheme (Optional[pg.DataContainer], optional): PyGIMLi data container
                                                            that defines the seismic survey geometry,
                                                            including shot and receiver positions.
                                                            If provided, it's set to the manager.
                                                            Defaults to None.
        """
        self.mesh = mesh      # Store the mesh.
        self.scheme = scheme  # Store the data scheme (shot/receiver configurations).
        
        # Initialize PyGIMLi's TravelTimeManager.
        # This manager class handles the setup for travel time calculations,
        # including associating the mesh and data scheme with the forward operator.
        self.manager = TravelTimeManager()

        # If a scheme (DataContainer) is provided, set it to the manager.
        # The scheme contains sensor positions and defines which sensor is a shot/receiver for each datum.
        if scheme is not None:
            self.manager.setData(scheme)
        
        # Set the mesh for the manager. The forward calculation (ray tracing) will occur on this mesh.
        self.manager.setMesh(mesh)
        # The TravelTimeManager internally creates a forward operator (fop) accessible via self.manager.fop.
    
    def set_scheme(self, scheme: pg.DataContainer) -> None:
        """
        Set seismic data scheme for forward modeling.
        
        Args:
            scheme (pg.DataContainer): Seismic data scheme (shot/receiver configurations and data).
        """
        self.scheme = scheme # Update the stored scheme.
        # Set the new data scheme on the PyGIMLi TravelTimeManager.
        self.manager.setData(scheme)
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set mesh for forward modeling.
        
        Args:
            mesh (pg.Mesh): PyGIMLI mesh for the forward simulation.
        """
        self.mesh = mesh # Update the stored mesh.
        # Set the new mesh on the PyGIMLi TravelTimeManager.
        # This will also update the mesh in the internal forward operator (fop).
        self.manager.setMesh(mesh)
    
    def forward(self, velocity_model: np.ndarray, slowness: bool = True) -> np.ndarray:
        """
        Compute forward response for a given velocity model.
        
        Args:
            velocity_model: Velocity model values (or slowness if slowness=True)
            slowness: Whether velocity_model is slowness (1/v)
            
        Returns:
            np.ndarray: NumPy array of calculated travel times for each shot-receiver pair
                        defined in the scheme. Units are typically seconds.
        """
        # The forward operator in PyGIMLi's travel time tomography typically works with slowness (1/velocity).
        # Slowness units: s/m if velocity is in m/s.

        slowness_model_values: np.ndarray
        if not slowness:
            # If the input `velocity_model` contains velocity values, convert them to slowness.
            # slowness = 1 / velocity
            # Add a small epsilon to velocity_model to prevent division by zero if any velocity is 0.
            # SUGGESTION: Handle velocity_model == 0 more explicitly if it can occur and has physical meaning.
            epsilon = 1e-12 # A small number to avoid division by zero.
            slowness_model_values = 1.0 / (velocity_model + epsilon)
        else:
            # If the input `velocity_model` already contains slowness values.
            slowness_model_values = velocity_model
        
        # Ensure the model is a flat NumPy array for the operator if not already.
        # PyGIMLi's fop.response usually expects a 1D array or pg.Vector matching cell count.
        if isinstance(slowness_model_values, np.ndarray):
            slowness_model_input = slowness_model_values.ravel()
        else: # Assuming it might be pg.Vector or compatible
            slowness_model_input = slowness_model_values

        # Perform the forward calculation (travel time simulation).
        # `self.manager.fop` is the actual forward operator (e.g., FMMModelling instance).
        # The `response` method calculates travel times based on the slowness model and the scheme.
        simulated_travel_times = self.manager.fop.response(slowness_model_input) # Returns a pg.RVector
        
        # Convert to NumPy array if it's not already (though fop.response usually returns RVector, which behaves like numpy array).
        # If `simulated_travel_times` is an RVector, `.array()` is not needed for direct use with numpy functions.
        # However, explicitly converting to a numpy array can be safer for type consistency.
        # For PyGIMLi RVector, direct use in numpy operations is often fine.
        return np.asarray(simulated_travel_times) # Ensure it's a NumPy array for return consistency.

    
    @classmethod # Indicates this is a class method, can be called on the class itself (SeismicForwardModeling.create_synthetic_data(...)).
    def create_synthetic_data(cls,  # `cls` refers to the class itself
                            sensor_x: np.ndarray,  # X-coordinates of geophones/sensors
                            surface_points: Optional[np.ndarray] = None, # Optional surface topography [[x,y],...]
                            mesh: Optional[pg.Mesh] = None,      # Optional user-provided PyGIMLi mesh
                            velocity_model: Optional[np.ndarray] = None, # Velocity model (m/s) for each mesh cell
                            slowness: bool = False,            # Flag: True if `velocity_model` is actually slowness (s/m)
                            shot_distance: float = 5,          # Distance between shots (if scheme is auto-generated)
                            noise_level: float = 0.05,         # Relative noise level (fraction, e.g., 0.05 for 5%)
                            noise_abs: float = 0.00001,        # Absolute noise level (e.g., in seconds)
                            save_path: Optional[str] = None,   # Path to save the synthetic data container
                            show_data: bool = False,           # Whether to display the data (e.g., travel time picks)
                            verbose: bool = False,             # Whether to show verbose output during simulation
                            seed: Optional[int] = None         # Random seed for noise generation
                            ) -> Tuple[pg.DataContainer, pg.Mesh]: # Returns the data container and the mesh used
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
        # Note: `seed` parameter is not used in this method currently.
        # SUGGESTION: If `seed` is intended for PyGIMLi's random noise generation in `manager.simulate`,
        # it should be passed or set globally, e.g., `pg.misc.core.gscheme(seed)`.
        if seed is not None:
            np.random.seed(seed) # For any numpy based noise if added manually.
            # PyGIMLi's manager.simulate might have its own way or use global pg seed.
            # pg.misc.core.gscheme(seed) # Example for global PyGIMLi seed

        
        # --- Create Seismic Measurement Scheme (DataContainer) ---
        # `tt.createRAData` creates a Refraction Tomography data scheme.
        # It defines shot positions based on sensor_x and shot_distance.
        # Each sensor acts as a shot, and all other sensors act as receivers for that shot.
        # `sensor_x` provides the x-coordinates for placing sensors.
        # `shot_distance` is the distance between shot points if not all sensors are shots.
        # If `shotDistance` is 0 or less, every sensor position is a shot.
        # If > 0, shots are created at multiples of this distance along the sensor line.
        # The function returns a pg.DataContainer with 's' (shot) and 'g' (geophone) tokens.
        scheme = tt.createRAData(sensors=sensor_x, shotDistance=shot_distance)
        
        # --- Adjust Sensor Positions for Surface Topography ---
        # If `surface_points` (an array of [x,y] coordinates defining topography) are provided,
        # map the 1D `sensor_x` positions onto this 2D surface.
        if surface_points is not None:
            sensor_positions_2d = np.zeros((len(sensor_x), 2)) # Initialize array for (x,y) sensor positions
            
            # For each sensor's x-coordinate, find the closest x in surface_points
            # and use the corresponding y-value from surface_points.
            # This effectively "drapes" the sensors onto the provided surface.
            for i, sx_val in enumerate(sensor_x):
                # Calculate horizontal distances from current sensor_x to all surface_points x-coordinates.
                distances_to_surface_x = np.abs(surface_points[:, 0] - sx_val)
                # Find the index of the surface point with the minimum horizontal distance.
                closest_surface_point_idx = np.argmin(distances_to_surface_x)
                # Assign the (x,y) of this closest surface point as the sensor's position.
                sensor_positions_2d[i, 0] = surface_points[closest_surface_point_idx, 0]
                sensor_positions_2d[i, 1] = surface_points[closest_surface_point_idx, 1]
                
            # Update the sensor positions in the scheme.
            # This also updates shot positions if they are tied to sensor locations.
            scheme.setSensors(sensor_positions_2d)
        # If surface_points is None, sensors are assumed to be on a flat surface (y=0 implicitly by createRAData from 1D sensor_x).
        
        # --- Initialize TravelTimeManager ---
        # The TravelTimeManager is a PyGIMLi class that handles the forward simulation (ray tracing).
        manager = TravelTimeManager() # Uses Finite Difference Method (FMM) by default.
        
        # --- Prepare Mesh for Forward Modeling ---
        # If no mesh is provided by the user, create a default one.
        current_mesh: pg.Mesh
        if mesh is None:
            # Determine mesh boundaries based on sensor locations and extend them.
            min_x_sensor = np.min(scheme.sensorPositions()[:,0]) # Min x of actual sensor positions
            max_x_sensor = np.max(scheme.sensorPositions()[:,0]) # Max x
            min_y_sensor = np.min(scheme.sensorPositions()[:,1]) # Min y (depth, could be 0 for flat)

            # Define mesh extent with some padding around sensors.
            mesh_x_min = min_x_sensor - 10
            mesh_x_max = max_x_sensor + 10
            mesh_y_min = min_y_sensor - 20 # Extend mesh deeper than lowest sensor
            mesh_y_max = min_y_sensor + 1 if np.allclose(min_y_sensor, 0) else max(0, min_y_sensor + 5) # Ensure mesh top is at or slightly above surface
                                                                                                    # Handles cases for y=0 or y<0 surfaces

            # Create a simple rectangular grid.
            # SUGGESTION: Consider using `pg.meshtools.createParaMesh` for better mesh refinement near sensors,
            # especially if topography is present. A simple grid might not be optimal.
            # Number of cells (50 in x, 20 in y) is fixed; might need adjustment based on survey scale.
            current_mesh = pg.createGrid(
                x=np.linspace(mesh_x_min, mesh_x_max, 50),
                y=np.linspace(mesh_y_min, mesh_y_max, 20) # y typically negative downwards for PyGIMLi
            )
            # Append a triangular boundary to the mesh for numerical stability and to define far-field conditions.
            # `xbound`, `ybound` define the extent of this boundary region.
            # Marker 1 is assigned to these boundary cells.
            current_mesh = pg.meshtools.appendTriangleBoundary(current_mesh, marker=1, xbound=50, ybound=50) # Original xbound/ybound were fixed.
                                                                                                        # SUGGESTION: xbound/ybound should be relative to mesh size.
        else:
            current_mesh = mesh # Use user-provided mesh.
        
        # --- Prepare Velocity/Slowness Model ---
        # If no velocity_model is provided, create a default one (linearly increasing velocity with depth).
        slowness_model_for_sim: np.ndarray
        if velocity_model is None:
            # Create a simple velocity model: V(z) = 500 + 50 * |depth|
            # This is applied to each cell based on its center depth.
            temp_velocity_model = np.ones(current_mesh.cellCount()) # Initialize with 1 m/s
            cell_centers_y = current_mesh.cellCenter()[:, 1] # Get y-coordinates (depths) of cell centers
            
            # Apply depth-dependent velocity. Assumes y is negative downwards.
            for i, cell_y_center in enumerate(cell_centers_y):
                temp_velocity_model[i] = 500.0 + 50.0 * abs(cell_y_center)

            # Convert this velocity model to slowness for the simulation.
            slowness_model_for_sim = 1.0 / temp_velocity_model
        else:
            # If a velocity_model is provided by the user.
            if not slowness:
                # Convert it to slowness if `slowness` flag is False.
                slowness_model_for_sim = 1.0 / np.asarray(velocity_model) # Ensure numpy array
            else:
                # Use it directly if it's already slowness.
                slowness_model_for_sim = np.asarray(velocity_model) # Ensure numpy array
        
        # --- Simulate Travel Time Data ---
        # `manager.simulate` performs the forward modeling (ray tracing) using the provided
        # slowness model, measurement scheme, and mesh. It can also add noise.
        # `slowness` here refers to the model parameter passed to simulate.
        synthetic_data_container = manager.simulate(
            slowness=slowness_model_for_sim.ravel(), # Ensure flat array of slowness for each cell
            scheme=scheme,                  # Measurement configurations
            mesh=current_mesh,              # Simulation mesh
            noiseLevel=noise_level,         # Relative noise level (fractional)
            noiseAbs=noise_abs,             # Absolute noise level (in seconds)
            verbose=verbose                 # Verbosity flag
        )
        # The result `synthetic_data_container` will have a 't' token with noisy travel times.
        
        # --- Optional Save and Display ---
        if save_path is not None:
            # Save the generated synthetic data container to a file.
            synthetic_data_container.save(save_path)
        
        if show_data:
            # Display the travel time data (first picks).
            # This requires matplotlib.
            pg.plt.figure(figsize=(10, 6)) # Create a figure.
            # `tt.drawFirstPicks` is a utility to plot travel time vs distance for each shot.
            tt.drawFirstPicks(ax=pg.plt.gca(), data=synthetic_data_container) # Pass current axes and data.
            pg.plt.show() # Show the plot.
        
        return synthetic_data_container, current_mesh # Return the data and the mesh used for simulation.
    

    @staticmethod # This method does not depend on instance state, so it's a static method.
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
        # Extract sensor positions (x-coordinates) from the data container.
        # `pg.x(data)` gets all unique x-positions of sensors defined in the data container.
        # It's assumed here that sensors are primarily defined by their x-coordinate for plotting.
        sensor_x_coords = pg.x(data) # Unique sorted sensor x-positions.
        
        # Get geophone (receiver) x-positions for each datum.
        # `data("g")` gives the sensor index for the geophone of each measurement.
        geophone_x_positions = np.array([sensor_x_coords[int(g_idx)] for g_idx in data("g")])
        # Get shot (source) x-positions for each datum.
        # `data("s")` gives the sensor index for the shot of each measurement.
        shot_x_positions = np.array([sensor_x_coords[int(s_idx)] for s_idx in data("s")])

        # --- Prepare Travel Times or Apparent Velocities for Plotting ---
        travel_times_to_plot: np.ndarray
        if tt is None:
            # If no specific travel times (`tt`) are provided, use the 't' token from the data container.
            travel_times_to_plot = np.array(data("t"))
        else:
            # If `tt` (an array of travel times) is provided, use it.
            travel_times_to_plot = np.asarray(tt) # Ensure it's a NumPy array.

        y_axis_label = "Traveltime (s)" # Default y-axis label.
        if plotva: # If apparent velocity is requested instead of travel time.
            # Calculate apparent velocity: V_app = |x_geophone - x_shot| / travel_time
            # This is a simple horizontal apparent velocity.
            # Avoid division by zero if travel_time is zero (add epsilon or handle).
            distance_shot_geophone = np.abs(geophone_x_positions - shot_x_positions)
            epsilon_time = 1e-12 # Small epsilon to prevent division by zero travel time.
            travel_times_to_plot = distance_shot_geophone / (travel_times_to_plot + epsilon_time)
            y_axis_label = "Apparent velocity (m s$^{-1}$)" # Update y-axis label.
        
        # --- Plotting ---
        # Find unique shot positions to iterate through each shot gather.
        unique_shot_x = np.unique(shot_x_positions)
        
        # Override or set default plotting keyword arguments for a clean style.
        # These control the appearance of the travel time pick lines.
        plot_style_kwargs = {
            'color': 'black',
            'linestyle': '--',
            'linewidth': 0.9,
            'marker': None  # No markers on the lines themselves.
        }
        plot_style_kwargs.update(kwargs) # Allow user to override defaults via **kwargs.
        
        # Plot travel times (or apparent velocities) for each unique shot position.
        for i, current_shot_x in enumerate(unique_shot_x):
            # Select data corresponding to the current shot.
            shot_mask = (shot_x_positions == current_shot_x)
            current_travel_times = travel_times_to_plot[shot_mask]
            current_geophone_x = geophone_x_positions[shot_mask]

            # Sort by geophone position for connected line plot.
            sort_indices = current_geophone_x.argsort()
            sorted_geophone_x = current_geophone_x[sort_indices]
            sorted_travel_times = current_travel_times[sort_indices]
            
            # Plot the travel time curve for the current shot.
            ax.plot(sorted_geophone_x, sorted_travel_times, **plot_style_kwargs)
            
            # Add a marker for the shot position itself.
            # Plotted at y=0 (or top of y-axis if inverted for travel times) to indicate shot location.
            # If plotting apparent velocity, y=0 might not be meaningful for shot; consider adjusting.
            # Assuming y=0 is a reference line for shots.
            shot_marker_y_pos = 0.0
            if not plotva and ax.get_ylim()[0] > ax.get_ylim()[1]: # If y-axis is inverted (typical for travel time)
                 shot_marker_y_pos = ax.get_ylim()[1] # Place at the "top" (min value) of inverted y-axis
            elif plotva and ax.get_ylim()[0] < ax.get_ylim()[1]: # If y-axis is normal (typical for velocity)
                 shot_marker_y_pos = ax.get_ylim()[0] # Place at the "bottom" (min value) for velocity plots

            ax.plot(current_shot_x, shot_marker_y_pos,
                    marker='s', # Square marker for shot
                    color='black', markersize=4,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='None') # Ensure no line connects shot markers if called in a loop by mistake
        
        # --- Styling the Plot ---
        # Add a light grid for better readability.
        ax.grid(True, linestyle='-', linewidth=0.2, color='lightgray')
        
        # Set axis labels.
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel("Distance (m)") # X-axis is typically distance along the survey line.
        
        # Invert y-axis if plotting travel times (smaller times at top).
        # Do not invert if plotting apparent velocity (larger velocities usually at top or as per data range).
        if not plotva:
            ax.invert_yaxis()

        return ax