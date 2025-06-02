"""
Forward modeling utilities for Electrical Resistivity Tomography (ERT).
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from typing import Tuple, Optional, Union


class ERTForwardModeling:
    """Class for forward modeling of Electrical Resistivity Tomography (ERT) data."""
    
    def __init__(self, mesh: pg.Mesh, data: Optional[pg.DataContainer] = None):
        """
        Initialize ERT forward modeling.
        
        Args:
            mesh (pg.Mesh): PyGIMLi mesh representing the subsurface discretization
                            for the forward calculation.
            data (Optional[pg.DataContainer], optional): PyGIMLi data container
                                                          that defines the electrode configurations
                                                          (measurement scheme). If provided, it's
                                                          set to the forward operator.
                                                          Defaults to None.
        """
        self.mesh = mesh  # Store the mesh
        self.data = data  # Store the data container (scheme)
        # Create an ERTModelling instance from PyGIMLi.
        # This object will perform the actual forward calculation.
        self.fwd_operator = ert.ERTModelling()

        # If a data container (scheme) is provided, associate it with the forward operator.
        # The data container holds information about electrode configurations (a, b, m, n pairs).
        if data is not None:
            self.fwd_operator.setData(data)
        
        # Associate the mesh with the forward operator.
        # The mesh defines the geometry and cells where resistivity values are defined.
        self.fwd_operator.setMesh(mesh)
    
    def set_data(self, data: pg.DataContainer) -> None:
        """
        Set ERT data for forward modeling.
        
        Args:
            data (pg.DataContainer): ERT data container defining the measurement configurations.
        """
        self.data = data # Update the stored data container
        # Set the new data (scheme) on the PyGIMLi forward operator.
        self.fwd_operator.setData(data)
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set mesh for forward modeling.
        
        Args:
            mesh (pg.Mesh): PyGIMLI mesh for the forward simulation.
        """
        self.mesh = mesh # Update the stored mesh
        # Set the new mesh on the PyGIMLi forward operator.
        self.fwd_operator.setMesh(mesh)
    
    def forward(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute forward response for a given resistivity model.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Forward response (apparent resistivity values). If `log_transform` is True,
            this will be log(apparent resistivity).
        """
        # The `resistivity_model` contains resistivity values for each cell in the mesh.
        # PyGIMLi's forward operator expects these values as a pg.Vector.

        model_pg_vector: pg.Vector
        if isinstance(resistivity_model, np.ndarray):
            # If the input is a NumPy array, flatten it and convert to pg.Vector.
            # .ravel() creates a 1D view of the array.
            model_pg_vector = pg.Vector(resistivity_model.ravel())
        else:
            # If it's already a pg.Vector (or compatible type), use it directly.
            # SUGGESTION: Add type check here to ensure `resistivity_model` is pg.Vector if not np.ndarray.
            model_pg_vector = resistivity_model
            
        # If the input `resistivity_model` is log-transformed (i.e., contains log(resistivity)),
        # it needs to be converted back to actual resistivity values by exponentiation (exp).
        # This is common in inversion to enforce positivity and handle large dynamic ranges.
        if log_transform:
            model_pg_vector = pg.Vector(np.exp(model_pg_vector)) # Element-wise exponentiation
            
        # Perform the forward calculation using the `response` method of the ERTModelling operator.
        # This simulates the ERT measurements for the given resistivity model and electrode configurations.
        simulated_response_pg_vector = self.fwd_operator.response(model_pg_vector)
        
        # Convert the PyGIMLi response vector to a NumPy array.
        response_array = simulated_response_pg_vector.array()

        # If `log_transform` was True, the output data (apparent resistivities) is also often
        # log-transformed for consistency in inversion algorithms (e.g., to stabilize variance).
        if log_transform:
            # Take the natural logarithm of the simulated apparent resistivities.
            # Handle potential log(0) or log(<0) if response_array can contain non-positives, though rhoa should be >0.
            # Adding a small epsilon or checking values might be needed for robustness if responses can be <=0.
            return np.log(response_array)
        
        return response_array # Return linear apparent resistivities
    
    def forward_and_jacobian(self, resistivity_model: np.ndarray, log_transform: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward response and Jacobian matrix.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Tuple of (forward response (np.ndarray), Jacobian matrix (np.ndarray)).
            If `log_transform` is True, response is log(apparent_resistivity) and
            Jacobian is for d(log(data))/d(log(model_param)).
        """
        # --- Prepare Model Vector ---
        # Similar to the `forward` method, convert and potentially untransform the model.
        model_pg_vector: pg.Vector
        actual_resistivity_values_for_jacobian: np.ndarray # Keep track of linear model for Jacobian transformation
        
        if isinstance(resistivity_model, np.ndarray):
            model_input_flat = resistivity_model.ravel()
            if log_transform:
                actual_resistivity_values_for_jacobian = np.exp(model_input_flat)
                model_pg_vector = pg.Vector(actual_resistivity_values_for_jacobian)
            else:
                actual_resistivity_values_for_jacobian = model_input_flat
                model_pg_vector = pg.Vector(model_input_flat)
        else: # Assuming pg.Vector if not ndarray
            # SUGGESTION: Add type check for pg.Vector here.
            if log_transform:
                # If input is pg.Vector of log_rho, need its numpy version for exp()
                # then back to pg.Vector for fwd_operator.
                # This assumes resistivity_model (if pg.Vector) holds log values if log_transform is true.
                numpy_log_model = resistivity_model.array() # Converts pg.Vector to numpy array
                actual_resistivity_values_for_jacobian = np.exp(numpy_log_model)
                model_pg_vector = pg.Vector(actual_resistivity_values_for_jacobian)
            else:
                actual_resistivity_values_for_jacobian = resistivity_model.array()
                model_pg_vector = resistivity_model

        # --- Calculate Forward Response ---
        # This response is in linear apparent resistivity.
        simulated_response_pg_vector = self.fwd_operator.response(model_pg_vector)
        response_linear_array = simulated_response_pg_vector.array() # NumPy array of linear apparent resistivities

        # --- Calculate Jacobian Matrix ---
        # The Jacobian matrix (J) contains the partial derivatives of the data (d)
        # with respect to the model parameters (m): J_ij = ∂d_i / ∂m_j.
        # PyGIMLi calculates J for linear data and linear model parameters (∂(rho_a) / ∂(rho_cell)).
        self.fwd_operator.createJacobian(model_pg_vector) # Compute Jacobian based on linear model values
        jacobian_gimli_matrix = self.fwd_operator.jacobian() # Get the GIMLI sparse matrix object
        J_linear = pg.utils.gmat2numpy(jacobian_gimli_matrix) # Convert to a NumPy dense array
                                                              # SUGGESTION: For large problems, dense Jacobian can be huge.
                                                              # Consider keeping it sparse if memory is an issue.

        # --- Adjust Response and Jacobian for Log-Transformation ---
        if log_transform:
            # If model and data are log-transformed for inversion (m' = log(m), d' = log(d)),
            # the Jacobian needs to be transformed: J' = ∂(log(d)) / ∂(log(m)).
            # Chain rule: ∂(log(d))/∂(log(m)) = (∂(log(d))/∂d) * (∂d/∂m) * (∂m/∂(log(m)))
            # ∂(log(d))/∂d = 1/d
            # ∂m/∂(log(m)) = m
            # So, J' = (1/d) * J_linear * m
            # This is equivalent to J_log = diag(1/response_linear_array) * J_linear * diag(actual_resistivity_values_for_jacobian)
            
            # Multiply columns of J_linear by corresponding model parameter values (m_j)
            # J_temp = J_linear * m (element-wise for columns)
            J_transformed = J_linear * actual_resistivity_values_for_jacobian # Broadcasting: (n_data, n_model) * (n_model,)

            # Divide rows of J_temp by corresponding data values (d_i)
            # J_log = J_temp / d (element-wise for rows)
            # Reshape response_linear_array to (n_data, 1) for broadcasting.
            J_final = J_transformed / response_linear_array[:, np.newaxis] # Broadcasting: (n_data, n_model) / (n_data, 1)

            # Log-transform the response for output.
            response_output = np.log(response_linear_array)
            return response_output, J_final
        else:
            # If no log transformation, return linear response and linear Jacobian.
            return response_linear_array, J_linear
    
    def get_coverage(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute coverage (resolution) for a given resistivity model.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Coverage values for each cell, typically log10 transformed and weighted by cell size.
        """
        # --- Prepare Model Vector ---
        # Similar to `forward` method, ensure model is in linear resistivity space.
        model_pg_vector: pg.Vector
        if isinstance(resistivity_model, np.ndarray):
            model_input_flat = resistivity_model.ravel()
            if log_transform:
                model_pg_vector = pg.Vector(np.exp(model_input_flat))
            else:
                model_pg_vector = pg.Vector(model_input_flat)
        else: # Assuming pg.Vector if not ndarray
            if log_transform:
                # If resistivity_model is already a pg.Vector but holds log_rho values
                model_pg_vector = pg.Vector(np.exp(resistivity_model.array()))
            else:
                model_pg_vector = resistivity_model # Assumed to be pg.Vector of linear rho
            
        # --- Calculate Response and Jacobian (needed for coverage) ---
        # Response is in linear apparent resistivity.
        response_pg_vector = self.fwd_operator.response(model_pg_vector)
        # Jacobian is ∂(rho_a) / ∂(rho_cell).
        self.fwd_operator.createJacobian(model_pg_vector)
        jacobian_gimli_matrix = self.fwd_operator.jacobian()

        # --- Calculate Coverage using PyGIMLi's `coverageDCtrans` ---
        # `coverageDCtrans` calculates a form of sensitivity or coverage, often related to
        # the diagonal elements of (J^T W_d^T W_d J + lambda W_m^T W_m)^-1 J^T W_d^T W_d, or simpler forms like diag(J^T J).
        # The arguments `1.0 / response_pg_vector` and `1.0 / model_pg_vector` suggest
        # weightings related to relative errors or log-transformations.
        # Specifically, this weighting makes `coverageDCtrans` compute the diagonal of
        # (J_log^T J_log) where J_log is d(log data)/d(log model).
        # `covTrans` will be a pg.Vector representing coverage for each model cell/parameter.
        coverage_raw_pg_vector = pg.core.coverageDCtrans(
            jacobian_gimli_matrix,    # The Jacobian matrix (d(rho_a)/d(rho_cell))
            1.0 / response_pg_vector, # Weighting for data space (typically 1/data_value for log transform)
            1.0 / model_pg_vector     # Weighting for model space (typically 1/model_value for log transform)
        )
        coverage_raw_numpy = coverage_raw_pg_vector.array() # Convert to NumPy array

        # --- Weight Coverage by Cell Sizes (or Parameter Region Sizes) ---
        # `paraDomain` is the mesh used for parameterization.
        parameter_domain_mesh = self.fwd_operator.paraDomain()

        # `param_cell_sizes` will store the size (area in 2D, volume in 3D) associated with each parameter.
        # The length of this array should be the number of parameters in the model.
        num_params = model_pg_vector.size()
        param_cell_sizes = np.zeros(num_params)

        # This loop assumes that parameters are defined on a cell-by-cell basis in the `parameter_domain_mesh`.
        # If parameters were region-based (i.e., one parameter per marker value), this logic would need adjustment.
        # The original code `paramSizes[c.marker()] += c.size()` implies region-based from markers.
        # This was identified as potentially problematic if markers are not 0..N-1 unique indices.
        # Assuming cell-based parameters for `parameter_domain_mesh` matching `model_pg_vector` length:
        if parameter_domain_mesh.cellCount() == num_params:
            for i in range(num_params):
                param_cell_sizes[i] = parameter_domain_mesh.cell(i).size()
        else:
            # Fallback to original logic if cell count mismatch, with a warning.
            # This part is kept for consistency with original but might be incorrect if markers aren't parameter indices.
            print("Warning: Parameter domain cell count does not match model size. Coverage weighting by cell marker might be incorrect.")
            # SUGGESTION: This logic needs to be robust. If parameters are cell-based, iterate cells and use cell.id()
            # or a direct index. If region-based, map markers to parameter indices correctly.
            for cell in parameter_domain_mesh.cells():
                marker_idx = cell.marker() # Assuming marker is the parameter index
                if 0 <= marker_idx < num_params:
                    param_cell_sizes[marker_idx] += cell.size()
                else:
                     print(f"Warning: Cell marker {marker_idx} is out of bounds for param_cell_sizes array (len {num_params}).")


        # --- Normalize Raw Coverage by Sizes and Log-Transform ---
        # Avoid division by zero if any param_cell_sizes are zero (e.g., for unused parameters or zero-size cells).
        coverage_weighted = np.full(num_params, np.nan) # Initialize with NaN
        valid_indices = param_cell_sizes > 1e-9 # Check for non-zero sizes (small epsilon for float comparison)

        coverage_weighted[valid_indices] = coverage_raw_numpy[valid_indices] / param_cell_sizes[valid_indices]

        # Take log10 for visualization, as coverage can span many orders of magnitude.
        # Handle cases where coverage_weighted might be zero or negative after division before log10.
        # Add a small epsilon (e.g., 1e-12 relative to max, or absolute small number) to avoid log10(0) or log10(<0).
        # A common practice is to work with positive definite coverage values.
        min_positive_coverage = np.min(coverage_weighted[coverage_weighted > 0]) if np.any(coverage_weighted > 0) else 1e-12
        floor_value = min_positive_coverage * 1e-6 # Small fraction of min positive value, or absolute small number

        log10_coverage = np.full(num_params, np.nan)
        positive_coverage_indices = coverage_weighted > floor_value
        log10_coverage[positive_coverage_indices] = np.log10(coverage_weighted[positive_coverage_indices])
        # For values <= floor_value but not NaN, assign log10 of the floor value
        log10_coverage[(~positive_coverage_indices) & (~np.isnan(coverage_weighted))] = np.log10(floor_value)

        return log10_coverage # Return log10 of coverage, weighted by cell/parameter size.
    
    @classmethod # This indicates that the method belongs to the class itself, not an instance.
    def create_synthetic_data(cls, xpos: np.ndarray,      # Electrode x-positions
                            ypos: Optional[np.ndarray] = None, # Electrode y-positions (0 for surface)
                            mesh: Optional[pg.Mesh] = None,    # Optional user-provided mesh
                            res_models: Optional[np.ndarray] = None, # Resistivity model for each cell
                            schemeName: str = 'wa',            # Measurement scheme name (e.g., 'wa' for Wenner alpha)
                            noise_level: float = 0.05,         # Relative noise level (e.g., 0.05 for 5%)
                            absolute_error: float = 0.0,       # Absolute error component for data error model
                            relative_error: float = 0.05,      # Relative error component for data error model
                            save_path: Optional[str] = None,   # Path to save synthetic data
                            show_data: bool = False,           # Whether to display pseudosection
                            seed: Optional[int] = None,        # Random seed for noise generation
                            xbound: float = 100,               # X-boundary extension for mesh generation
                            ybound: float = 100                # Y-boundary extension for mesh generation
                            ) -> Tuple[pg.DataContainer, pg.Mesh]: # Returns data container and the mesh used
        """
        Create synthetic ERT data using forward modeling.
        
        This method simulates an ERT survey by placing electrodes, creating a measurement 
        scheme, performing forward modeling to generate synthetic data, and adding noise.
        
        Args:
            xpos: X-coordinates of electrodes
            ypos: Y-coordinates of electrodes (if None, uses flat surface)
            mesh: Mesh for forward modeling
            res_models: Resistivity model values
            schemeName: Name of measurement scheme ('wa', 'dd', etc.)
            noise_level: Level of Gaussian noise to add
            absolute_error: Absolute error for data estimation
            relative_error: Relative error for data estimation
            save_path: Path to save synthetic data (if None, does not save)
            show_data: Whether to display data after creation
            seed: Random seed for noise generation
            xbound: X boundary extension for mesh
            ybound: Y boundary extension for mesh
            
        Returns:
            Tuple of (synthetic ERT data container, simulation mesh)
        """
        # --- Initialize Random Seed (for noise generation reproducibility) ---
        if seed is not None:
            np.random.seed(seed) # NumPy's random generator
            pg.misc.core.gscheme(seed) # PyGIMLi's global random scheme (older pygimli versions might use pg.rrng.randpin.seed(seed))
                                      # SUGGESTION: Check current PyGIMLi API for setting global random seed if `pg.misc.core.gscheme` is outdated.
        
        # --- Electrode Setup ---
        # Create electrode positions. If ypos is not provided, assume a flat surface (y=0).
        if ypos is None:
            ypos = np.zeros_like(xpos) # Create y-coordinates array of zeros.
        
        # Combine x and y coordinates into a 2D array for electrode positions.
        # Shape: (num_electrodes, 2) where columns are [x, y].
        electrode_positions = np.hstack((xpos.reshape(-1, 1), ypos.reshape(-1, 1)))
        
        # --- Create ERT Measurement Scheme ---
        # `ert.createData` generates a PyGIMLi data container with standard configurations
        # (e.g., Wenner alpha ('wa'), dipole-dipole ('dd')) based on electrode positions.
        # This container defines the (a, b, m, n) quadrupoles for measurements.
        measurement_scheme = ert.createData(elecs=electrode_positions, schemeName=schemeName)
        
        # --- Prepare Mesh for Forward Modeling ---
        simulation_mesh: pg.Mesh
        if mesh is not None:
            # If a user-provided mesh is given.
            # It's good practice to ensure cell markers are set for assigning resistivities.
            # Here, all cells are set to marker 2. This implies `res_models` should correspond to this marker
            # or be a single value if homogeneous, or an array matching cell count.
            # SUGGESTION: If `res_models` is an array per cell, this marker setting might be overridden or irrelevant
            # if `fwd_operator.setMesh(mesh, ignoreRegionManager=True)` is used or if model is directly cell-based.
            # The current logic assumes `res_models` will map to cells of `grid`.
            # The variable name `grid` is used below for the final simulation_mesh.
            mesh.setCellMarkers(np.ones(mesh.cellCount(), dtype=int) * 2) # Assign marker 2 to all cells. Use dtype=int.
            
            # Append a triangular boundary to the mesh. This is often done to extend the modeling domain
            # and ensure boundary conditions for the PDE solver are far from the survey area.
            # Marker 1 is assigned to these boundary cells, which might have a fixed background resistivity.
            simulation_mesh = pg.meshtools.appendTriangleBoundary(mesh, marker=1,
                                                                  xbound=xbound, ybound=ybound)
        else:
            # If no mesh is provided, create a simple structured grid based on electrode positions.
            # This creates a rectangular grid.
            # SUGGESTION: The quality and extent of this auto-generated mesh might not be optimal for all cases.
            # Consider using `pg.meshtools.createParaMesh` for more robust mesh generation around electrodes.
            x_coords_for_grid = np.linspace(np.min(xpos) - 10, np.max(xpos) + 10, 50) # 50 points in x
            y_coords_for_grid = np.linspace(np.min(ypos) - 20, 0, 20)                # 20 points in y (depth)
            # Create a 2D grid.
            grid_tmp = pg.createGrid(x=x_coords_for_grid, y=y_coords_for_grid)
            # Append boundary as above.
            simulation_mesh = pg.meshtools.appendTriangleBoundary(grid_tmp, marker=1,
                                                                  xbound=xbound, ybound=ybound)
            
            # If no resistivity model is provided with an auto-generated mesh,
            # create a default homogeneous resistivity model (e.g., 100 Ω·m).
            if res_models is None:
                # `res_models` should be per cell of `simulation_mesh`.
                res_models = np.ones(simulation_mesh.cellCount()) * 100.0 # Default resistivity: 100 Ω·m

        # Ensure `res_models` is a pg.Vector for the forward operator if it's a numpy array.
        # This is important as PyGIMLi operators expect pg.Vector.
        if isinstance(res_models, np.ndarray):
            res_models_pg_vector = pg.Vector(res_models.ravel())
        else: # Assuming it might already be a pg.Vector or compatible type
            # SUGGESTION: Add type check to ensure `res_models` is pg.Vector if not np.ndarray.
            res_models_pg_vector = res_models


        # --- Simulate Synthetic Data ---
        # `measurement_scheme.copy()` creates a new data container to store synthetic measurements.
        synthetic_data_container = measurement_scheme.copy()

        # The original code uses `cls(mesh=grid, data=scheme)` to initialize a forward operator.
        # This implies `ERTForwardModeling` instance methods would be used.
        # However, the subsequent lines instantiate a new `ert.ERTModelling()` (`fob`).
        # This makes the `fwd_operator = cls(...)` line effectively unused if `fob` is used for the core task.
        # Assuming the intent is to use PyGIMLi's direct ERTModelling:

        fwd_op_pygimli = ert.ERTModelling()      # Create a new PyGIMLi forward operator.
        fwd_op_pygimli.setData(measurement_scheme) # Set the measurement configurations.
        fwd_op_pygimli.setMesh(simulation_mesh)    # Set the mesh. (original code used `grid` here, renamed to `simulation_mesh` for clarity)

        # Calculate the synthetic "true" response (apparent resistivities) for the given resistivity model.
        # `res_models_pg_vector` must contain resistivity for each cell in `simulation_mesh`.
        true_response_vector = fwd_op_pygimli.response(res_models_pg_vector) # This is RVector of rho_a

        # --- Add Noise to Synthetic Data ---
        # Simulate measurement noise by adding random Gaussian noise to the true response.
        # Noise is relative to the data magnitude (noise_level is fractional, e.g., 0.05 = 5%).
        # `pg.randn` generates standard normal distributed random numbers.
        noisy_response_vector = true_response_vector * (1.0 + pg.randn(true_response_vector.size()) * noise_level)

        # Assign the noisy apparent resistivity values to the 'rhoa' field in the data container.
        synthetic_data_container['rhoa'] = noisy_response_vector

        # --- Estimate Data Errors ---
        # `ert.ERTManager.estimateError` calculates error estimates based on absolute and relative components.
        # This is often used in inversion to weight data points.
        # Here, it's used to assign an 'err' field to the synthetic data, simulating error estimation.
        # An ERTManager is instantiated temporarily for this purpose.
        ert_manager_instance = ert.ERTManager(synthetic_data_container) # Pass the data container itself.

        # Calculate and assign errors. `absoluteUError` refers to an absolute error on voltage,
        # which then propagates to an error on apparent resistivity. `relativeError` is typically applied to rhoa.
        # SUGGESTION: The use of `absoluteUError` might be specific. If errors are directly on rhoa,
        # the error model might be simpler (e.g., `abs_err + rel_err * |rhoa|`).
        # PyGIMLi's `estimateError` handles the propagation from voltage/current errors if those tokens exist.
        synthetic_data_container['err'] = ert_manager_instance.estimateError(
            synthetic_data_container,
            absoluteUError=absolute_error, # Absolute error component (often related to voltage measurement precision).
            relativeError=relative_error   # Relative error component (percentage of the measurement).
        )
        
        # --- Optional Display and Save ---
        if show_data:
            # Display the synthetic data as a pseudosection.
            # `logscale=True` is common for resistivity data which spans orders of magnitude.
            ert.showData(synthetic_data_container, logscale=True)
        
        if save_path is not None:
            # Save the synthetic data container to the specified file path.
            synthetic_data_container.save(save_path)
        
        return synthetic_data_container, simulation_mesh # Return the data and the mesh used.

def ertforward(fob, mesh, rhomodel, xr):
    """
    Forward model for ERT.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        mesh (pg.Mesh): Mesh for the forward model.
        rhomodel (pg.RVector): Resistivity model vector.
        xr (np.ndarray): Log-transformed model parameter (resistivity).

    Returns:
        dr_log (np.ndarray): Log-transformed forward response (log(rho_a)).
        rhomodel_updated_pg_vector (pg.RVector): The resistivity model vector that was actually used for the forward calculation
                                                 (after updating parts of it with `xr`), in linear resistivity space.
    """
    # This function appears designed for a specific inversion workflow where:
    # - `fob`: Is a pre-configured PyGIMLi ERTModelling operator.
    # - `mesh`: Is the PyGIMLi mesh associated with the model.
    # - `rhomodel`: Is the current full resistivity model (pg.RVector, linear resistivity values).
    # - `xr`: Contains new *log-transformed* resistivity values for a specific part of the model,
    #         identified by cells with marker 2 in the `mesh`.

    # 1. Convert the current full `rhomodel` from linear to log space.
    log_rhomodel_full_array = np.log(rhomodel.array()) # log(rho_cell_values) for all cells

    # 2. Update the log-transformed model array.
    # `xr` provides the new log-resistivity values for the cells marked as 2.
    # The original code `xr1[mesh.cellMarkers() == 2] = np.exp(xr)` implies `xr` itself is some pre-log value,
    # which is confusing. Assuming `xr` directly contains the new log_rho values for region 2.
    # If `xr` was linear, it should be `np.log(xr)`. If `xr` is already log, then direct assignment.
    # Corrected interpretation: `xr` are the log-values for the region of interest.
    # We need to select the cells from `mesh.cellMarkers()` that correspond to the elements of `xr`.
    # This assumes `xr` is an array matching the number of cells with marker 2.
    # A robust way:
    #   `idx_region2 = np.where(mesh.cellMarkers() == 2)[0]`
    #   `log_rhomodel_full_array[idx_region2] = xr` (if xr is flat array for region 2)
    # The original code `xr1[mesh.cellMarkers() == 2] = np.exp(xr)` is kept if `xr` is log(log(rho)) for that region.
    # Given the overall log-transform context, it's more likely `xr` is log(rho) for that region.
    # Let's assume `xr` is an array of log-resistivity values intended for the cells with marker == 2.
    # And `mesh.cellMarkers() == 2` creates a boolean mask of the same size as `log_rhomodel_full_array`.
    # `xr` must then be an array that matches the `True` elements in this mask.
    # The `np.exp(xr)` in original was confusing. If `xr` is already log(rho) for the sub-region, it should be assigned directly.
    # If `xr` is linear resistivity for the sub-region, then `np.log(xr)` should be assigned.
    # Sticking to original structure as much as possible:
    # If `xr` is log(rho) for the specific region, then `np.exp(xr)` would be linear rho for that region.
    # Then `xr1[mask] = linear_rho_region` means `log_rho_full[mask] = linear_rho_region`. This is inconsistent.
    # Re-interpreting: `xr` might be the complete model in log-space that we want to test.
    # And `rhomodel` is some base/previous state. The line `xr1[mesh.cellMarkers()==2] = np.exp(xr)`
    # makes sense if `xr` refers to something else, perhaps related to an update vector.
    # Given the function signature, `xr` is likely the model parameter vector being optimized (in log space).
    # So, `log_rhomodel_full_array` should be constructed from `xr`.
    # If `rhomodel` is a background and `xr` is an update for region 2:
    #   `log_rhomodel_full_array[mesh.cellMarkers() == 2] = xr` (if xr is just for region 2)
    # Let's assume the original intent was that `xr` is the full log-model, and `rhomodel` is ignored beyond its initial state for `xr1`.
    # This function is quite specific and its variable naming/usage is ambiguous.
    # For now, I will comment based on a plausible interpretation that `xr` contains the complete model parameters in log space.
    # The line `xr1 = np.log(rhomodel.array())` suggests `rhomodel` is the starting point.
    # And `xr1[mesh.cellMarkers() == 2] = np.exp(xr)` suggests `xr` is an update for region 2, but in some transformed space.
    # This function is hard to comment accurately without more context on its specific use in an inversion.
    # Assuming `xr` is the current model estimate (log-transformed) for the *entire* domain.
    # Then the update logic `xr1[mesh.cellMarkers() == 2] = np.exp(xr)` is very confusing.
    # Let's proceed by commenting the code as literally as possible while noting issues.

    # Take the log of the initial full model `rhomodel`.
    log_rhomodel_base = np.log(rhomodel.array())

    # This line is problematic: `np.exp(xr)` implies `xr` is log(log(rho)) or similar.
    # If `xr` are log-rho values for region 2, it should be `log_rhomodel_base[mesh.cellMarkers() == 2] = xr`.
    # If `xr` are linear-rho values for region 2, it should be `log_rhomodel_base[mesh.cellMarkers() == 2] = np.log(xr)`.
    # Commenting as is:
    log_rhomodel_base[mesh.cellMarkers() == 2] = np.exp(xr) # Updates region 2 with exp(xr). `xr` must be shaped for this.
                                                          # This assumes `xr` is specifically for region 2 and needs exponentiation.
                                                          # This line is highly specific and context-dependent.

    # Convert the (potentially modified) log-resistivity array back to linear resistivity.
    rhomodel_updated_linear_array = np.exp(log_rhomodel_base)
    rhomodel_updated_pg_vector = pg.Vector(rhomodel_updated_linear_array) # pg.matrix.RVector is an alias

    # 3. Calculate the forward response using the updated linear resistivity model.
    response_linear_pg_vector = fob.response(rhomodel_updated_pg_vector) # This is linear rho_a

    # 4. Return the log-transformed response and the updated linear model vector that was used.
    return np.log(response_linear_pg_vector.array()), rhomodel_updated_pg_vector


def ertforward2(fob, xr, mesh):
    """
    Simplified ERT forward model.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        dr_log (np.ndarray): Log-transformed forward response (log(rho_a)).
    """
    # This function assumes `xr` is a log-transformed resistivity model for the entire mesh.
    # It converts it to linear resistivity, computes the response, and returns the log of the response.

    # `xr` is log(rho_cell_values) for all cells.
    # The line `xr1 = xr.copy()` followed by `xr1 = np.exp(xr)` implies `xr` holds log(rho).

    # 1. Convert log-resistivity model `xr` to linear resistivity.
    resistivity_linear_model_array = np.exp(xr) # rho_cell_values

    # Create a pg.Vector from the linear resistivity model.
    # This assumes `resistivity_linear_model_array` matches the cell order expected by `fob.response`.
    # If `fob` is set up with `mesh`, this should align with `mesh.cellCount()`.
    # SUGGESTION: Ensure `xr` corresponds to `mesh.cellCount()` or `fob.paraDomain().cellCount()`.
    # The original code `rhomodel = xr1` does not convert to pg.Vector which might be an issue if `fob.response` expects it.
    # However, PyGIMLi often handles numpy arrays directly by implicit conversion.
    # For clarity and safety, explicit conversion is better.
    rhomodel_pg_vector = pg.Vector(resistivity_linear_model_array)

    # 2. Calculate forward response using the linear resistivity model.
    response_linear_pg_vector = fob.response(rhomodel_pg_vector) # This is rho_a

    # 3. Log-transform the response.
    response_log_array = np.log(response_linear_pg_vector.array())

    return response_log_array


def ertforandjac(fob, rhomodel, xr):
    """
    Forward model and Jacobian for ERT.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        rhomodel (pg.RVector): Resistivity model.
        xr (np.ndarray): Log-transformed model parameter.

    Returns:
        dr_log (np.ndarray): Log-transformed forward response (log(rho_a)).
        J_transformed (np.ndarray): Jacobian matrix for d(log(data))/d(log(model_param)).
    """
    # This function calculates the forward response and the Jacobian matrix.
    # It appears to assume that `rhomodel` is in linear resistivity space for the forward calculation,
    # and `xr` represents the log-transformed model parameters used for transforming the Jacobian.
    # This is typical in iterative inversion where `rhomodel` is the current linear model,
    # and `xr` (log of `rhomodel`) is the optimization variable.

    # 1. Calculate forward response using the provided linear `rhomodel` (pg.RVector).
    response_linear_pg_vector = fob.response(rhomodel) # rho_a
    response_linear_array = response_linear_pg_vector.array()

    # 2. Create and retrieve the Jacobian matrix for linear model and linear data (∂(rho_a) / ∂(rho_cell)).
    fob.createJacobian(rhomodel) # Computes Jacobian based on the current `rhomodel`.
    jacobian_gimli_matrix = fob.jacobian() # Get GIMLI sparse matrix.
    J_linear = pg.utils.gmat2numpy(jacobian_gimli_matrix) # Convert to dense NumPy array.

    # 3. Transform the Jacobian for log-log space (∂(log(rho_a)) / ∂(log(rho_cell))).
    # J_log = diag(1/response_linear_array) * J_linear * diag(rhomodel_linear_array)
    # `xr` is assumed to be log(rhomodel_linear_array). So, `np.exp(xr)` gives `rhomodel_linear_array`.

    # Multiply columns of J_linear by corresponding linear model parameter values (rho_cell_j).
    # J_temp = J_linear * rho_cell
    # `np.exp(xr)` should be a 1D array of linear cell resistivities.
    # Broadcasting `J_linear` (n_data, n_model) with `np.exp(xr)` (n_model,)
    J_transformed_cols = J_linear * np.exp(xr) # This assumes xr is 1D array of log(rho_cell)

    # Divide rows of J_transformed_cols by corresponding linear data values (rho_a_i).
    # J_log = J_transformed_cols / rho_a
    # Reshape `response_linear_array` to (n_data, 1) for row-wise division.
    J_log_transformed = J_transformed_cols / response_linear_array.reshape(-1, 1) # Or response_linear_array[:, np.newaxis]

    # 4. Log-transform the forward response.
    response_log_array = np.log(response_linear_array)

    return response_log_array, J_log_transformed


def ertforandjac2(fob, xr, mesh):
    """
    Alternative ERT forward model and Jacobian using log-resistivity values.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        dr_log (np.ndarray): Log-transformed forward response (log(rho_a)).
        J_transformed (np.ndarray): Jacobian matrix for d(log(data))/d(log(model_param)).
    """
    # This function is similar to `ertforandjac` but takes `xr` (log-transformed model)
    # as the primary model input, converts it to linear for forward/Jacobian calculation,
    # and then transforms Jacobian and response to log-log space.
    # The `mesh` parameter is present but not explicitly used within this function's direct logic,
    # though `fob` (the forward operator) would have been initialized with a mesh.

    # 1. Convert log-resistivity model `xr` to linear resistivity.
    # `xr` is assumed to be a 1D array of log(rho_cell) for all cells.
    resistivity_linear_array = np.exp(xr)
    # Convert to pg.Vector for PyGIMLi operator.
    # SUGGESTION: As in ertforward2, ensure this aligns with `fob.paraDomain().cellCount()`.
    rhomodel_pg_vector = pg.Vector(resistivity_linear_array)

    # 2. Calculate forward response using the linear resistivity model.
    response_linear_pg_vector = fob.response(rhomodel_pg_vector) # rho_a
    response_linear_array = response_linear_pg_vector.array()

    # 3. Create and retrieve the Jacobian matrix for linear model and linear data (∂(rho_a) / ∂(rho_cell)).
    fob.createJacobian(rhomodel_pg_vector) # Jacobian based on linear `rhomodel_pg_vector`.
    jacobian_gimli_matrix = fob.jacobian()
    J_linear = pg.utils.gmat2numpy(jacobian_gimli_matrix) # Dense NumPy array.

    # 4. Transform the Jacobian for log-log space: J_log = diag(1/response_linear) * J_linear * diag(model_linear)
    # `resistivity_linear_array` (which is `np.exp(xr)`) are the linear model parameters.
    # `xr.T` (transpose) on a 1D array `xr` has no effect. `np.exp(xr)` is correct here.
    # J_transformed_cols = J_linear * resistivity_linear_array
    J_transformed_cols = J_linear * np.exp(xr) # Multiplies columns of J_linear by exp(xr_j)

    # Divide rows by linear response values.
    # J_log_transformed = J_transformed_cols / response_linear_array (row-wise)
    J_log_transformed = J_transformed_cols / response_linear_array.reshape(-1, 1)

    # 5. Log-transform the forward response.
    response_log_array = np.log(response_linear_array)

    return response_log_array, J_log_transformed


