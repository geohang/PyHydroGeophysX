"""
Forward modeling utilities for Seismic Refraction Tomography (SRT) using PyGIMLi.

This module provides a class `SeismicForwardModeling` to encapsulate SRT forward
modeling operations, primarily focused on travel time calculations. It includes
functionality for generating synthetic seismic refraction data and a helper
function for plotting first arrival travel times.
"""
import numpy as np
import pygimli as pg
import pygimli.physics.traveltime as tt # Travel time modeling specific tools
from pygimli.physics import TravelTimeManager # Main manager class for travel time physics
from typing import Tuple, Optional, Union, List, Dict, Any # Dict not used currently


class SeismicForwardModeling:
    """
    A class to perform Seismic Refraction Tomography (SRT) forward modeling using PyGIMLi.

    This class wraps PyGIMLi's `TravelTimeManager` and its associated forward operator
    to compute synthetic travel times based on a given velocity (or slowness) model
    and measurement scheme. It also provides a method to generate synthetic datasets.
    """
    
    def __init__(self, mesh: pg.Mesh, scheme: Optional[pg.DataContainer] = None):
        """
        Initialize the SeismicForwardModeling class.

        Args:
            mesh (pg.Mesh): A PyGIMLi mesh object representing the subsurface model domain.
                            The mesh should be suitable for seismic travel time calculation.
            scheme (Optional[pg.DataContainer], optional):
                A PyGIMLi DataContainer object defining the seismic survey scheme
                (shot and receiver positions). If provided, it's set on the
                TravelTimeManager. Defaults to None.
        """
        if not isinstance(mesh, pg.Mesh):
            raise TypeError("mesh must be a PyGIMLi pg.Mesh object.")
        if scheme is not None and not isinstance(scheme, pg.DataContainer):
            raise TypeError("scheme must be a PyGIMLi pg.DataContainer object or None.")

        self.mesh: pg.Mesh = mesh
        self.scheme: Optional[pg.DataContainer] = scheme
        
        # Initialize PyGIMLi's TravelTimeManager
        self.manager = TravelTimeManager()
        
        # Associate the mesh with the manager. The manager internally sets up the forward operator (fop).
        self.manager.setMesh(self.mesh)
        # Potential Issue: Mesh quality and type (e.g., triangular elements) are important for
        # travel time calculations. Consider adding checks or recommendations.
        
        # If a data scheme is provided, associate it with the manager.
        if self.scheme is not None:
            self.manager.setData(self.scheme)
            # This sets the scheme for the manager's forward operator (fop).
    
    def set_scheme(self, scheme: pg.DataContainer) -> None:
        """
        Set or update the seismic data scheme for forward modeling.

        Args:
            scheme (pg.DataContainer): The PyGIMLi DataContainer defining the seismic
                                       shot and receiver configurations.
        """
        if not isinstance(scheme, pg.DataContainer):
            raise TypeError("scheme must be a PyGIMLi pg.DataContainer object.")
        self.scheme = scheme
        self.manager.setData(self.scheme) # Updates the scheme in the TravelTimeManager
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set or update the PyGIMLi mesh for forward modeling.

        Args:
            mesh (pg.Mesh): The PyGIMLi mesh object.
        """
        if not isinstance(mesh, pg.Mesh):
            raise TypeError("mesh must be a PyGIMLi pg.Mesh object.")
        self.mesh = mesh
        self.manager.setMesh(self.mesh) # Updates the mesh in the TravelTimeManager
        # Potential Issue: If a scheme was already set, ensure its compatibility
        # with the new mesh (e.g., sensor positions).
    
    def forward(self, model_values: np.ndarray, is_slowness: bool = True) -> np.ndarray:
        """
        Compute the forward seismic response (travel times) for a given model.

        The model can be provided as either velocity [m/s] or slowness [s/m].

        Args:
            model_values (np.ndarray): A 1D NumPy array representing the seismic property
                                     (slowness or velocity) for each cell in the mesh.
            is_slowness (bool, optional): If True, `model_values` are interpreted as slowness [s/m].
                                       If False, `model_values` are interpreted as velocity [m/s]
                                       and will be converted to slowness (1/velocity).
                                       Defaults to True.

        Returns:
            np.ndarray: A NumPy array of the computed travel times [s] for the
                        shot-receiver pairs defined in the scheme.

        Raises:
            RuntimeError: If the data scheme is not set on the manager.
            ValueError: If `model_values` contains non-positive velocities when `is_slowness` is False.
        """
        if self.scheme is None:
            # The manager's forward operator (fop) might not be correctly initialized
            # or might not know the number of data points if scheme is missing.
            raise RuntimeError("Seismic data scheme is not set. Call set_scheme() before forward modeling.")

        slowness_model_values: np.ndarray
        if not is_slowness:
            # Convert velocity model to slowness model (s = 1/v)
            # Potential Issue: Division by zero or negative velocities.
            if np.any(model_values <= 0):
                raise ValueError("Velocity model contains non-positive values, cannot convert to slowness.")
            slowness_model_values = 1.0 / model_values
        else:
            slowness_model_values = model_values
            if np.any(slowness_model_values <= 0):
                # Slowness should also be positive for physical travel times.
                print("Warning: Slowness model contains non-positive values. This may lead to errors or non-physical travel times.")
        
        # PyGIMLi's travel time forward operator expects slowness.
        # The `self.manager.fop` is the forward operator instance.
        # `response()` takes the slowness model as a pg.Vector or NumPy array.
        # Ensure the model_values array is flat, as expected by fop.response.
        try:
            # Ensure model is 1D array (ravel) and of type pg.Vector if required by specific PyGIMLi versions,
            # though fop.response often handles numpy arrays directly.
            # For safety, one could convert: pg.Vector(slowness_model_values.ravel())
            travel_times_pg = self.manager.fop.response(slowness_model_values.ravel())
            return np.array(travel_times_pg) # Convert to NumPy array
        except Exception as e:
            # Catch potential errors from PyGIMLi's forward calculation
            raise RuntimeError(f"PyGIMLi forward modeling (response calculation) failed: {e}")

    
    @classmethod
    def create_synthetic_data(cls, 
                            sensor_x_coords: np.ndarray, 
                            surface_topography_points: Optional[np.ndarray] = None, # e.g., [[x1,z1], [x2,z2]...]
                            fwd_mesh: Optional[pg.Mesh] = None, 
                            velocity_model_values: Optional[np.ndarray] = None,
                            model_is_slowness: bool = False, # Input model type
                            shot_spacing: float = 5.0, # Renamed from shot_distance for clarity
                            noise_relative_level: float = 0.05, 
                            noise_absolute_val: float = 1e-5, # Changed from 1e-5 for more realistic time unit
                            save_data_path: Optional[str] = None, 
                            show_travel_time_data: bool = False, # Renamed for clarity
                            verbose_simulation: bool = False, # Renamed for clarity
                            random_seed_value: Optional[int] = None, # Renamed for clarity
                            mesh_x_boundary_ext: float = 50.0, # Renamed, more descriptive
                            mesh_y_boundary_ext: float = 50.0  # Renamed
                            ) -> Tuple[pg.DataContainer, pg.Mesh]:
        """
        Create synthetic seismic refraction tomography (SRT) data.

        This class method simulates an SRT survey by:
        1. Defining geophone positions (potentially on a given surface topography).
        2. Creating a measurement scheme with shots at specified intervals.
        3. Generating or using a provided mesh for the forward calculation.
        4. Assigning velocity/slowness values to the mesh cells.
        5. Computing synthetic travel times using the SRT forward operator.
        6. Optionally adding Gaussian noise to the synthetic travel times.
        7. Optionally saving the data and displaying a plot of the first arrivals.
        
        Args:
            sensor_x_coords (np.ndarray): 1D NumPy array of X-coordinates for geophone locations [m].
            surface_topography_points (Optional[np.ndarray], optional):
                2D NumPy array of [x, z] coordinates defining the surface topography.
                If provided, geophone Z-coordinates will be interpolated from this surface.
                If None, geophones are placed on a flat surface (Z=0). Defaults to None.
            fwd_mesh (Optional[pg.Mesh], optional): A pre-defined PyGIMLi mesh for forward modeling.
                                                    If None, a simple 2D mesh is created. Defaults to None.
            velocity_model_values (Optional[np.ndarray], optional):
                Seismic property values for each cell of `fwd_mesh`.
                Interpreted as velocity [m/s] or slowness [s/m] based on `model_is_slowness`.
                If None, a default velocity model (increasing with depth) is used. Defaults to None.
            model_is_slowness (bool, optional): If True, `velocity_model_values` are slowness [s/m].
                                                If False, they are velocity [m/s]. Defaults to False.
            shot_spacing (float, optional): Distance between shot points along the geophone line [m].
                                           Defaults to 5.0.
            noise_relative_level (float, optional): Relative level of Gaussian noise to add to travel times
                                                    (e.g., 0.05 for 5%). Defaults to 0.05.
            noise_absolute_val (float, optional): Absolute level of Gaussian noise to add to travel times [s].
                                                 Defaults to 1e-5 s (0.01 ms).
            save_data_path (Optional[str], optional): File path to save the generated synthetic data.
                                                      If None, data is not saved. Defaults to None.
            show_travel_time_data (bool, optional): If True, displays a plot of the first arrival picks.
                                                   Defaults to False.
            verbose_simulation (bool, optional): If True, PyGIMLi's TravelTimeManager may print progress.
                                                 Defaults to False.
            random_seed_value (Optional[int], optional): Seed for random number generator for reproducible noise.
                                                        If None, noise varies. Defaults to None.
            mesh_x_boundary_ext (float, optional): Horizontal boundary extension for auto-generated mesh [m].
                                                   Defaults to 50.0.
            mesh_y_boundary_ext (float, optional): Vertical boundary extension for auto-generated mesh [m].
                                                   Defaults to 50.0.
            
        Returns:
            Tuple[pg.DataContainer, pg.Mesh]:
                - synth_srt_data (pg.DataContainer): Generated synthetic SRT data with travel times and error estimates.
                - simulation_mesh (pg.Mesh): The PyGIMLi mesh used for the simulation.
        
        Raises:
            ValueError: For invalid inputs (e.g., empty sensor_x_coords, model/mesh dimension mismatch).
        """
        if not isinstance(sensor_x_coords, np.ndarray) or sensor_x_coords.ndim != 1 or sensor_x_coords.size == 0:
            raise ValueError("sensor_x_coords must be a 1D NumPy array with at least one geophone position.")

        if random_seed_value is not None:
            np.random.seed(random_seed_value)
            pg.Core.RNG().setSeed(random_seed_value) # For PyGIMLi's internal RNG

        # Create seismic data scheme (DataContainer for SRT)
        # tt.createRAData generates a scheme with shots and receivers based on sensor_x and shot_spacing.
        # It assumes geophones are also shots unless specified otherwise.
        # 'RA' likely stands for Refraction Array or similar.
        try:
            scheme = tt.createRAData(sensors=sensor_x_coords, shotDistance=shot_spacing)
        except Exception as e:
            raise RuntimeError(f"Failed to create seismic scheme with PyGIMLi: {e}")
        
        # Adjust sensor Z-coordinates if surface topography is provided
        if surface_topography_points is not None:
            if not isinstance(surface_topography_points, np.ndarray) or surface_topography_points.ndim != 2 or surface_topography_points.shape[1] != 2:
                raise ValueError("surface_topography_points must be a 2D NumPy array of [x, z] coordinates.")
            
            sensor_positions_adjusted = np.zeros((len(sensor_x_coords), 2))
            # Interpolate Z for each sensor X from the provided surface points
            # This simple interpolation finds the Z of the nearest X in surface_topography_points.
            # More sophisticated interpolation (e.g., linear) could be used if surface_topography_points are sparse.
            for i, sx_coord in enumerate(sensor_x_coords):
                distances_to_surface_x = np.abs(surface_topography_points[:, 0] - sx_coord)
                closest_surface_point_idx = np.argmin(distances_to_surface_x)
                sensor_positions_adjusted[i, 0] = surface_topography_points[closest_surface_point_idx, 0] # Use actual surface X
                sensor_positions_adjusted[i, 1] = surface_topography_points[closest_surface_point_idx, 1] # Interpolated Z
            
            scheme.setSensors(sensor_positions_adjusted)
            # Note: Shot positions also need to be updated if they are derived from sensor positions
            # and topography is applied. createRAData might place shots at sensor locations by default.
            # If shots are separate, their Z would also need adjustment.
            # For now, assuming shots are at sensors and setSensors updates all relevant positions.

        simulation_mesh: pg.Mesh
        # Prepare mesh for forward modeling
        if fwd_mesh is not None:
            if not isinstance(fwd_mesh, pg.Mesh):
                raise TypeError("fwd_mesh must be a PyGIMLi pg.Mesh object or None.")
            simulation_mesh = fwd_mesh
            # As in ERT, user should handle markers if specific regions are needed.
            # Defaulting all to marker 1 for SRT is common if model is cell-based.
            if simulation_mesh.cellCount() > 0 and not any(c.marker() !=0 for c in simulation_mesh.cells()):
                 simulation_mesh.setCellMarkers(pg.Vector(simulation_mesh.cellCount(), 1))
        else:
            # Auto-create a 2D mesh if none provided
            min_x_sensor = np.min(scheme.sensorPositions()[:,0]) # Use actual sensor positions after topography
            max_x_sensor = np.max(scheme.sensorPositions()[:,0])
            min_z_sensor = np.min(scheme.sensorPositions()[:,1]) # Min Z (could be max elevation if Z is up)
            # max_z_sensor = np.max(scheme.sensorPositions()[:,1]) # Max Z (usually surface, e.g. 0)

            # Define mesh extent slightly larger than survey line
            mesh_x_coords = np.linspace(min_x_sensor - mesh_x_boundary_ext / 5.0, 
                                        max_x_sensor + mesh_x_boundary_ext / 5.0, 
                                        50) # Example discretization
            # Mesh depth should extend below lowest sensor/topography point.
            # If Z is elevation, y_coords might go from (min_z_sensor - depth_extent) to (max_z_sensor + surface_buffer).
            # Assuming Z is depth-like (increases downwards) or elevation (increases upwards, surface near 0).
            # Original code: y=np.linspace(-20, 0, 20) for flat.
            # For topography: y_min = np.min(sensor_positions[:, 1]) - 20
            mesh_z_min = min_z_sensor - mesh_y_boundary_ext / 2.0 # Extends deeper
            mesh_z_max = np.max(scheme.sensorPositions()[:,1]) + mesh_y_boundary_ext / 10.0 # Small buffer above highest sensor
            mesh_z_coords = np.linspace(mesh_z_min, mesh_z_max, 25) # Example discretization for Z

            grid_core = pg.createGrid(x=mesh_x_coords, y=mesh_z_coords, marker=1) # Default marker 1 for cells
            simulation_mesh = pg.meshtools.appendTriangleBoundary(grid_core, marker=2, # Boundary marker different from cells
                                                                  xbound=mesh_x_boundary_ext, ybound=mesh_y_boundary_ext)
        
        # Prepare slowness/velocity model for the simulation_mesh
        if velocity_model_values is None:
            print("No velocity_model_values provided. Using default model (500 m/s + 50*abs(depth)).")
            slowness_for_fwd = np.ones(simulation_mesh.cellCount())
            cell_centers_np = np.array(simulation_mesh.cellCenters()) # Get cell centers as NumPy array
            # Velocity increases with depth (abs(z_coord) assuming z is neg depth or elevation)
            # If z is positive depth, then (500 + 50*z_coord)
            # Assuming PyGIMLi's y is depth-like for createGrid, or z for general mesh.
            # Let's assume cell_centers_np[:,1] is the vertical coordinate.
            # If y is elevation (surface at 0, deeper is negative): 500 + 50 * abs(centers_y)
            # If y is depth (surface at 0, deeper is positive): 500 + 50 * centers_y
            # Default createGrid y goes from -20 to 0. So abs(y) makes sense.
            for i, cell_center_y in enumerate(cell_centers_np[:, 1]):
                velocity = 500.0 + 50.0 * abs(cell_center_y)
                slowness_for_fwd[i] = 1.0 / velocity
        else:
            if len(velocity_model_values) != simulation_mesh.cellCount():
                raise ValueError(f"Length of velocity_model_values ({len(velocity_model_values)}) must match "
                                 f"cell count of simulation_mesh ({simulation_mesh.cellCount()}).")
            if model_is_slowness:
                slowness_for_fwd = np.asarray(velocity_model_values, dtype=float).ravel()
                if np.any(slowness_for_fwd <= 0): print("Warning: Slowness model contains non-positive values.")
            else: # Input is velocity
                velo_np = np.asarray(velocity_model_values, dtype=float).ravel()
                if np.any(velo_np <= 0): raise ValueError("Velocity model contains non-positive values.")
                slowness_for_fwd = 1.0 / velo_np
        
        # Initialize TravelTimeManager for simulation
        # The class `cls` is used here to call its own constructor if this were part of the class.
        # However, create_synthetic_data is a @classmethod, so `cls` refers to `SeismicForwardModeling`.
        # An instance is not strictly needed if using `manager.simulate` static-like.
        # The original created a new `manager = TravelTimeManager()`. This is cleaner.
        sim_manager = TravelTimeManager(verbose=verbose_simulation)
        
        # Simulate synthetic travel time data
        # `manager.simulate` is a high-level utility that sets mesh, scheme, slowness, and runs fop.
        try:
            synth_srt_data = sim_manager.simulate(
                slowness=slowness_for_fwd, # Expects slowness
                scheme=scheme,
                mesh=simulation_mesh,
                noiseLevel=noise_relative_level, # Relative noise
                noiseAbs=noise_absolute_val,          # Absolute noise in seconds
                seed=random_seed_value # Pass seed to simulate for reproducibility
            )
        except Exception as e:
            raise RuntimeError(f"PyGIMLi simulate() method failed: {e}")
        
        # `simulate` already adds noise and populates 't' and 'err'.
        
        if save_data_path:
            try:
                synth_srt_data.save(save_data_path)
                # print(f"Synthetic SRT data saved to: {save_data_path}")
            except Exception as e:
                print(f"Error saving synthetic SRT data to '{save_data_path}': {e}")
        
        if show_travel_time_data:
            try:
                # Use the static plotting method also defined in this file for consistency.
                # Need a matplotlib axes object.
                import matplotlib.pyplot as plt # Import locally for plotting
                fig, ax = plt.subplots(figsize=(10,6))
                cls.draw_first_picks(ax, synth_srt_data) # Call staticmethod with cls
                ax.set_title("Synthetic First Arrival Travel Times")
                plt.show()
            except Exception as e:
                print(f"Could not display travel time data plot: {e}")
        
        return synth_srt_data, simulation_mesh
    

    @staticmethod # This method does not depend on instance state, so it's a staticmethod.
    def draw_first_picks(ax: Any, # matplotlib.axes.Axes
                         data: pg.DataContainer, 
                         travel_times: Optional[np.ndarray] = None, # Renamed from tt for clarity
                         plot_app_velocity: bool = False, # Renamed from plotva
                         **kwargs: Any # Pass through to ax.plot
                        ) -> Any: # matplotlib.axes.Axes
        """
        Plot first arrival travel times (or apparent velocities) for seismic refraction data.

        This function visualizes travel time curves for each shot point.
        
        Args:
            ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
            data (pg.DataContainer): PyGIMLi DataContainer containing the seismic survey layout
                                     (sensor positions 'x', 'z'; shot 's'; geophone 'g' indices).
                                     It should also contain travel times 't' if `travel_times` is None.
            travel_times (Optional[np.ndarray], optional):
                Array of travel times [s] to plot. If None, uses 't' from `data`. Defaults to None.
            plot_app_velocity (bool, optional): If True, plots apparent velocity (distance/travel_time)
                                                instead of travel times. Defaults to False.
            **kwargs (Any): Additional keyword arguments passed directly to `ax.plot()`
                            for customizing line appearance (e.g., color, linestyle).
                            Default style is black dashed lines.
        
        Returns:
            matplotlib.axes.Axes: The modified matplotlib Axes object with the plot.
        """
        # Extract sensor (geophone) X-coordinates from the data container
        # pg.x(data) typically returns all unique X positions of sensors.
        # 's' and 'g' in data are indices referring to these sensor positions.
        # A more robust way to get all unique sensor X positions:
        # sensor_x_all = np.unique(np.array(data.sensorPositions()[:,0]))
        # However, pg.x(data) is a shortcut if data is simple.
        # For safety, directly use sensorPositions:
        sensor_x_all = np.array(data.sensorPositions()[:,0]) # Get X directly from sensor positions
        
        # Get X-coordinates for geophones (g) and shots (s) for each measurement
        try:
            receiver_indices = np.array(data['g'], dtype=int)
            shot_indices = np.array(data['s'], dtype=int)
        except KeyError as e:
            raise KeyError(f"Data container missing required fields ('g' or 's'): {e}")

        gx = sensor_x_all[receiver_indices] # X-coordinates of geophones for each datum
        sx = sensor_x_all[shot_indices]     # X-coordinates of shots for each datum
        
        # Get travel times to plot
        if travel_times is None:
            if 't' not in data:
                raise ValueError("Travel times 't' not found in data container and not provided via 'travel_times' argument.")
            current_travel_times = np.array(data['t'])
        else:
            if len(travel_times) != len(data['s']): # Check consistency
                raise ValueError("Provided 'travel_times' length must match the number of data points in 'data'.")
            current_travel_times = np.array(travel_times)
        
        if plot_app_velocity:
            # Calculate apparent velocity: distance / time
            # Distance is |geophone_x - shot_x|
            # Avoid division by zero if time is zero (should not happen for physical data)
            delta_x = np.abs(gx - sx)
            # Prevent division by zero or very small travel times leading to huge velocities
            valid_tt_mask = current_travel_times > 1e-9 # Threshold for valid time
            plot_values = np.full_like(current_travel_times, np.nan) # Default to NaN
            plot_values[valid_tt_mask] = delta_x[valid_tt_mask] / current_travel_times[valid_tt_mask]
            y_label = "Apparent Velocity (m/s)"
        else:
            plot_values = current_travel_times
            y_label = "Travel Time (s)"
        
        # Find unique shot positions to iterate through shots
        unique_shot_x_coords = np.unique(sx)
        
        # Default plotting style if not overridden by kwargs
        plot_style = {
            'color': 'black',
            'linestyle': '--',
            'linewidth': 0.9,
            'marker': None # No markers on the lines themselves by default
        }
        plot_style.update(kwargs) # Allow user to override defaults
        
        # Plot travel time (or apparent velocity) curves for each shot
        for shot_x_val in unique_shot_x_coords:
            shot_mask = (sx == shot_x_val) # Mask for data from the current shot
            
            current_shot_plot_values = plot_values[shot_mask]
            current_shot_geophone_x = gx[shot_mask]
            
            # Sort by geophone position for smooth line plotting
            sort_indices = np.argsort(current_shot_geophone_x)
            ax.plot(current_shot_geophone_x[sort_indices], current_shot_plot_values[sort_indices], **plot_style)
            
            # Add a marker for the shot position itself (typically at t=0 or top of y-axis)
            # Plotting shot marker at y=0 (or slightly offset if y-axis is inverted and starts at 0)
            # If apparent velocity, y=0 might not be appropriate. Plot on actual axis.
            # For travel times, y=0 makes sense.
            # The original code plots shot marker at y=0.0
            y_shot_marker = 0.0
            if plot_app_velocity and ax.get_ylim()[0] > 0 : # If y-axis min is positive (typical for velocity)
                 y_shot_marker = ax.get_ylim()[0] # Place at bottom of current y-axis view for visibility
            
            ax.plot(shot_x_val, y_shot_marker, marker='s', color='black', markersize=5, 
                    markeredgecolor='black', markeredgewidth=0.5, linestyle='None')
        
        # Apply styling to the plot
        ax.grid(True, linestyle='-', linewidth=0.3, color='lightgrey', alpha=0.7)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Distance (m)")
        
        # Invert y-axis if plotting travel times (common convention)
        if not plot_app_velocity:
            ax.invert_yaxis()

        return ax