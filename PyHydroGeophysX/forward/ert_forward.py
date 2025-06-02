"""
Forward modeling utilities for Electrical Resistivity Tomography (ERT) using PyGIMLi.

This module provides a class `ERTForwardModeling` to encapsulate ERT forward
modeling operations, including Jacobian calculation and synthetic data generation.
It also contains several standalone functions for forward modeling and Jacobian
computation, which might be older or specialized versions.
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert # Electrical Resistivity Tomography functionalities
from typing import Tuple, Optional, Union, Any # Any for flexible pg.Vector/np.ndarray type hints


class ERTForwardModeling:
    """
    A class to perform ERT forward modeling using PyGIMLi.

    This class wraps PyGIMLi's `ert.ERTModelling` operator, providing methods
    to compute synthetic ERT data (apparent resistivities), calculate Jacobian matrices,
    and estimate data coverage (resolution). It handles log transformations
    for models and data if specified.
    """
    
    def __init__(self, mesh: pg.Mesh, data_scheme: Optional[pg.DataContainer] = None):
        """
        Initialize the ERTForwardModeling class.

        Args:
            mesh (pg.Mesh): A PyGIMLi mesh object representing the subsurface model domain.
                            The mesh should be suitable for ERT simulation (e.g., 2D or 3D).
            data_scheme (Optional[pg.DataContainer], optional):
                A PyGIMLi DataContainer object defining the ERT measurement scheme
                (electrode configurations, etc.). If provided, it's set on the
                forward operator. Defaults to None.
        """
        if not isinstance(mesh, pg.Mesh):
            raise TypeError("mesh must be a PyGIMLi pg.Mesh object.")
        if data_scheme is not None and not isinstance(data_scheme, pg.DataContainer):
            raise TypeError("data_scheme must be a PyGIMLi pg.DataContainer object or None.")

        self.mesh: pg.Mesh = mesh
        self.data_scheme: Optional[pg.DataContainer] = data_scheme # Store the data scheme

        # Initialize the ERTModelling operator from PyGIMLi
        self.fwd_operator = ert.ERTModelling()
        
        # Associate the mesh with the forward operator
        self.fwd_operator.setMesh(self.mesh)
        
        # If a data scheme is provided, associate it with the forward operator
        if self.data_scheme is not None:
            self.fwd_operator.setData(self.data_scheme)
        # Potential Issue: If no data_scheme is provided initially, some operations like
        # response calculation might fail if the operator expects it.
        # User must call set_data() later if not provided here.
    
    def set_data(self, data_scheme: pg.DataContainer) -> None:
        """
        Set or update the ERT data scheme for forward modeling.

        Args:
            data_scheme (pg.DataContainer): The PyGIMLi DataContainer defining
                                            the measurement configurations.
        """
        if not isinstance(data_scheme, pg.DataContainer):
            raise TypeError("data_scheme must be a PyGIMLi pg.DataContainer object.")
        self.data_scheme = data_scheme
        self.fwd_operator.setData(self.data_scheme)
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set or update the PyGIMLi mesh for forward modeling.

        Args:
            mesh (pg.Mesh): The PyGIMLi mesh object.
        """
        if not isinstance(mesh, pg.Mesh):
            raise TypeError("mesh must be a PyGIMLi pg.Mesh object.")
        self.mesh = mesh
        self.fwd_operator.setMesh(self.mesh)
        # Potential Issue: If a data_scheme was already set, ensure its compatibility
        # with the new mesh (e.g., electrode positions relative to mesh).
        # PyGIMLi might handle this internally, but it's a point of attention.
    
    def forward(self, resistivity_model: Union[np.ndarray, pg.Vector],
                log_transform_model: bool = True,
                log_transform_response: bool = True) -> np.ndarray:
        """
        Compute the forward ERT response (apparent resistivities) for a given resistivity model.

        Args:
            resistivity_model (Union[np.ndarray, pg.Vector]):
                The resistivity model values for each cell in the mesh [ohm·m].
                Can be a NumPy array or a PyGIMLi RVector.
            log_transform_model (bool, optional):
                If True, `resistivity_model` is assumed to be log10-resistivity or
                ln-resistivity. The method will exponentiate it (base e for ln)
                before using it in the forward calculation. Defaults to True.
                (Note: Original code uses `np.exp`, implying natural log for model).
            log_transform_response (bool, optional):
                If True, the computed apparent resistivities will be log-transformed (natural log)
                before being returned. Defaults to True.

        Returns:
            np.ndarray: A NumPy array of the computed forward response values
                        (apparent resistivities). Log-transformed if specified.

        Raises:
            RuntimeError: If the forward operator fails (e.g., if data scheme not set).
        """
        # Ensure the data scheme is set on the operator, crucial for response calculation.
        if self.data_scheme is None:
            raise RuntimeError("ERT data scheme is not set. Call set_data() before forward modeling.")

        # Convert NumPy array model to PyGIMLi RVector if necessary.
        # Ensure it's flattened, as ERTModelling expects a 1D vector for cell properties.
        if isinstance(resistivity_model, np.ndarray):
            model_pg = pg.Vector(resistivity_model.ravel())
        elif isinstance(resistivity_model, pg.Vector):
            model_pg = resistivity_model
        else:
            raise TypeError("resistivity_model must be a NumPy array or PyGIMLi RVector.")
            
        # If the input model is log-transformed, exponentiate it.
        if log_transform_model:
            # Assuming natural log for the model, as np.exp is used.
            # If model was log10, use 10**model_pg.
            actual_resistivity_model = pg.Vector(np.exp(np.array(model_pg))) # Convert to np.array for np.exp
        else:
            actual_resistivity_model = model_pg
            
        # Perform the forward calculation using PyGIMLi's ERTModelling.
        response_pg_vector = self.fwd_operator.response(actual_resistivity_model)
        response_np_array = np.array(response_pg_vector) # Convert RVector response to NumPy array

        # Log-transform the response if requested.
        if log_transform_response:
            # Handle potential non-positive resistivity values before log:
            # np.log will issue warnings for <=0. Can result in -inf or nan.
            # This might indicate issues with model or simulation yielding non-physical results.
            if np.any(response_np_array <= 0):
                print("Warning: Calculated apparent resistivities contain non-positive values. Log transform will result in -inf or NaN for these.")
            with np.errstate(divide='ignore', invalid='ignore'): # Suppress log(0) warnings
                log_response = np.log(response_np_array)
            return log_response

        return response_np_array
    
    def forward_and_jacobian(self,
                             resistivity_model: Union[np.ndarray, pg.Vector],
                             log_transform_model: bool = True,
                             log_transform_response: bool = True
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward ERT response and the Jacobian matrix.

        The Jacobian J_ij = ∂d_i / ∂m_j, where d_i is the i-th data point and
        m_j is the j-th model parameter (cell resistivity).
        If log transformations are used, the Jacobian is adjusted accordingly:
        - If model is log(m) and data is log(d): J_log = (∂log(d) / ∂log(m)) = (m/d) * (∂d/∂m)

        Args:
            resistivity_model (Union[np.ndarray, pg.Vector]):
                Resistivity model values [ohm·m] or log-resistivity.
            log_transform_model (bool, optional): If True, input model is log-transformed (natural log).
                                                  Defaults to True.
            log_transform_response (bool, optional): If True, returned response and Jacobian
                                                     are for log-transformed data. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - response (np.ndarray): Forward response (apparent resistivities), potentially log-transformed.
                - jacobian_matrix (np.ndarray): The Jacobian matrix, adjusted for log transforms if specified.
                                                Shape (n_data, n_model_cells).

        Raises:
            RuntimeError: If data scheme not set or Jacobian creation fails.
        """
        if self.data_scheme is None:
            raise RuntimeError("ERT data scheme is not set. Call set_data() before Jacobian calculation.")

        # Prepare the model in actual resistivity values (ρ)
        if isinstance(resistivity_model, np.ndarray):
            model_input_np_form = resistivity_model.ravel() # Keep original form if needed for scaling
        elif isinstance(resistivity_model, pg.Vector):
            model_input_np_form = np.array(resistivity_model)
        else:
            raise TypeError("resistivity_model must be a NumPy array or PyGIMLi RVector.")
            
        if log_transform_model:
            actual_resistivity_values_np = np.exp(model_input_np_form)
        else:
            actual_resistivity_values_np = model_input_np_form

        actual_resistivity_pg_model = pg.Vector(actual_resistivity_values_np)

        # Calculate forward response d = f(ρ)
        response_pg_vector = self.fwd_operator.response(actual_resistivity_pg_model)
        response_np_linear = np.array(response_pg_vector) # This is d (actual apparent resistivities)

        # Create and retrieve Jacobian J = ∂d/∂ρ (sensitivity of actual data to actual resistivity)
        self.fwd_operator.createJacobian(actual_resistivity_pg_model)
        jacobian_gmat = self.fwd_operator.jacobian() # This is a GMat object
        jacobian_np_linear = pg.utils.gmat2numpy(jacobian_gmat) # Convert GMat to NumPy array
        
        # Initialize final outputs
        final_response = response_np_linear
        final_jacobian = jacobian_np_linear

        # Adjust response and Jacobian based on log transformation choices
        if log_transform_model and log_transform_response:
            # Jacobian for d(log d) / d(log ρ) = (ρ/d) * (∂d/∂ρ)
            # Here, actual_resistivity_values_np is ρ.
            # response_np_linear is d.
            # jacobian_np_linear is ∂d/∂ρ.

            # Avoid division by zero if response_np_linear (d) or actual_resistivity_values_np (ρ) contain zeros.
            if np.any(response_np_linear <= 0):
                print("Warning: Non-positive apparent resistivities computed. Jacobian for log-log transform may contain NaN/Inf.")
            if np.any(actual_resistivity_values_np <= 0) and log_transform_model: # Check rho only if model was log (so rho came from exp)
                print("Warning: Non-positive resistivities in model after exponentiation. Jacobian for log-log transform may contain NaN/Inf.")

            # Scaling factor matrix for Jacobian: diag(ρ/d)
            # For J_new = diag(1/d) * J_old * diag(ρ)
            # J_old is (n_data, n_model). diag(ρ) is (n_model, n_model). diag(1/d) is (n_data, n_data).

            # Create diagonal matrix of rho values
            diag_rho = np.diag(actual_resistivity_values_np)
            # Jacobian scaled by model parameters (J * rho)
            J_times_rho = jacobian_np_linear @ diag_rho

            # Inverse of data vector, prepared for diagonal matrix
            inv_d = np.zeros_like(response_np_linear)
            valid_d_mask = response_np_linear > 1e-12 # Avoid division by zero or very small numbers
            inv_d[valid_d_mask] = 1.0 / response_np_linear[valid_d_mask]
            # For d <= 0, inv_d remains 0. Multiplying by 0 will make that row of Jacobian zero.
            # This might be acceptable if non-positive responses are errors or non-physical.
            diag_inv_d = np.diag(inv_d)

            final_jacobian = diag_inv_d @ J_times_rho # diag(1/d) @ J @ diag(rho)
            with np.errstate(divide='ignore', invalid='ignore'):
                final_response = np.log(response_np_linear) # log(d)

        elif log_transform_model: # Model is log(ρ), data is d. Jacobian is ∂d/∂log(ρ) = ρ * ∂d/∂ρ
            # final_jacobian = jacobian_np_linear * actual_resistivity_values_np # Element-wise row scaling
            final_jacobian = jacobian_np_linear @ np.diag(actual_resistivity_values_np)
            # final_response remains response_np_linear

        elif log_transform_response: # Model is ρ, data is log(d). Jacobian is ∂log(d)/∂ρ = (1/d) * ∂d/∂ρ
            inv_d = np.zeros_like(response_np_linear)
            valid_d_mask = response_np_linear > 1e-12
            inv_d[valid_d_mask] = 1.0 / response_np_linear[valid_d_mask]
            # final_jacobian = inv_d[:, np.newaxis] * jacobian_np_linear # Column-wise scaling by 1/d
            final_jacobian = np.diag(inv_d) @ jacobian_np_linear
            with np.errstate(divide='ignore', invalid='ignore'):
                final_response = np.log(response_np_linear)

        return final_response, final_jacobian
    
    def get_coverage(self, resistivity_model: Union[np.ndarray, pg.Vector],
                     log_transform_model: bool = True) -> np.ndarray:
        """
        Compute data coverage (approximate resolution or sensitivity density) for the mesh cells.

        This uses PyGIMLi's `coverageDCtrans` which typically computes the diagonal
        of J^T * diag(1/d^2) * J, often related to sensitivity or resolution matrix diagonal.
        The result is then weighted by cell sizes and log10 transformed.

        Args:
            resistivity_model (Union[np.ndarray, pg.Vector]):
                Resistivity model values [ohm·m] or log-resistivity.
            log_transform_model (bool, optional): If True, input model is log-transformed (natural log).
                                                  Defaults to True.

        Returns:
            np.ndarray: An array of log10-transformed coverage values for each cell in the parameter domain.

        Raises:
            RuntimeError: If data scheme not set or Jacobian/response calculation fails.
        """
        if self.data_scheme is None:
            raise RuntimeError("ERT data scheme is not set. Call set_data() before get_coverage().")

        # Prepare actual resistivity model (ρ)
        if isinstance(resistivity_model, np.ndarray):
            model_input_np_form = resistivity_model.ravel()
        elif isinstance(resistivity_model, pg.Vector):
            model_input_np_form = np.array(resistivity_model)
        else:
            raise TypeError("resistivity_model must be a NumPy array or PyGIMLi RVector.")
            
        if log_transform_model:
            actual_resistivity_values_np = np.exp(model_input_np_form)
        else:
            actual_resistivity_values_np = model_input_np_form
        
        actual_resistivity_pg_model = pg.Vector(actual_resistivity_values_np)
        
        # Calculate response d=f(ρ) and Jacobian J=∂d/∂ρ
        response_pg_vector = self.fwd_operator.response(actual_resistivity_pg_model)
        self.fwd_operator.createJacobian(actual_resistivity_pg_model)
        jacobian_gmat = self.fwd_operator.jacobian() # GMat object
        
        # Prepare weights for coverageDCtrans: dataError = 1/response, modelWeight = 1/model (actual resistivity)
        # These weights are specific to how coverageDCtrans is defined.
        response_for_weighting = np.array(response_pg_vector)
        response_for_weighting[response_for_weighting <= 1e-12] = 1e-12 # Avoid division by zero or very small numbers

        model_for_weighting = actual_resistivity_values_np.copy() # Use the actual resistivity values
        model_for_weighting[model_for_weighting <= 1e-12] = 1e-12

        # Calculate coverage using pg.core.coverageDCtrans
        # This function computes diag(J^T * diag(dataError^2) * J) * modelWeight^2 or similar.
        # The exact formulation: sum_i (J_ij * dataError_i)^2 * modelWeight_j^2
        # With dataError = 1/d and modelWeight = 1/m, this is sum_i (J_ij / (d_i * m_j))^2
        # This seems to be related to normalized sensitivities.
        coverage_values_pg = pg.core.coverageDCtrans(
            jacobian_gmat,
            pg.Vector(1.0 / response_for_weighting),
            pg.Vector(1.0 / model_for_weighting)
        )
        coverage_values_np = np.array(coverage_values_pg)

        # Normalize coverage by cell sizes.
        # The original code `paramSizes[c.marker()] += c.size()` is problematic
        # if cell markers are not contiguous parameter indices from 0.
        # A more robust approach is to get cell sizes directly corresponding to parameter indices.
        parameter_mesh = self.fwd_operator.paraDomain() # Mesh defining the parameters
        if parameter_mesh.cellCount() != len(coverage_values_np):
            print(f"Warning: Coverage vector length ({len(coverage_values_np)}) does not match "
                  f"parameter mesh cell count ({parameter_mesh.cellCount()}). "
                  "Cell size normalization might be incorrect or skipped.")
            # Fallback: do not normalize by size if counts mismatch.
            normalized_coverage = coverage_values_np
        else:
            cell_sizes_for_norm = np.array([cell.size() for cell in parameter_mesh.cells()])
            # Avoid division by zero if any cell size is zero (should not happen for valid mesh cells)
            cell_sizes_for_norm[cell_sizes_for_norm <= 1e-12] = 1e-12
            normalized_coverage = coverage_values_np / cell_sizes_for_norm

        # Log10 transform the normalized coverage.
        # Handle non-positive values before log10 to avoid warnings/errors.
        # Coverage values (sum of squares) should ideally be positive.
        valid_coverage_mask = normalized_coverage > 1e-12 # Use a small threshold
        log10_transformed_coverage = np.full_like(normalized_coverage, np.nan) # Initialize with NaN
        log10_transformed_coverage[valid_coverage_mask] = np.log10(normalized_coverage[valid_coverage_mask])

        return log10_transformed_coverage
    
    @classmethod # This should be a classmethod as it uses `cls`
    def create_synthetic_data(cls, xpos_electrodes: np.ndarray,
                            ypos_electrodes: Optional[np.ndarray] = None,
                            fwd_mesh: Optional[pg.Mesh] = None,
                            resistivity_values: Optional[Union[np.ndarray, pg.Vector]] = None,
                            scheme_name: str = 'wa', # Wenner alpha is common
                            noise_level_relative: float = 0.05, # Relative noise
                            noise_abs_val: float = 0.0, # Absolute noise component (e.g. uV) for error estimation
                            # The original 'relative_error' parameter was for ert.estimateError.
                            # Let's assume it's the same as noise_level_relative for error estimation if not specified.
                            relative_error_for_estimation: Optional[float] = None,
                            save_path: Optional[str] = None, 
                            show_plot: bool = False, # Renamed for clarity
                            random_seed: Optional[int] = None,
                            mesh_x_bound: float = 100.0, # Boundary extension for mesh creation
                            mesh_y_bound: float = 100.0  # Boundary extension for mesh creation
                            ) -> Tuple[pg.DataContainer, pg.Mesh]:
        """
        Create synthetic ERT data using forward modeling.

        This class method simulates an ERT survey by:
        1. Defining electrode positions.
        2. Creating a measurement scheme (e.g., Wenner, Dipole-Dipole).
        3. Generating or using a provided mesh for the forward calculation.
        4. Assigning resistivity values to the mesh cells.
        5. Computing synthetic apparent resistivities using the ERT forward operator.
        6. Optionally adding Gaussian noise to the synthetic data.
        7. Optionally saving the data and displaying a pseudosection.
        
        Args:
            xpos_electrodes (np.ndarray): 1D NumPy array of X-coordinates for electrodes [m].
            ypos_electrodes (Optional[np.ndarray], optional): 1D NumPy array of Y-coordinates (depth/elevation)
                                                              for electrodes [m]. If None, assumes a flat
                                                              surface (all Y=0). Defaults to None.
            fwd_mesh (Optional[pg.Mesh], optional): A pre-defined PyGIMLi mesh to use for forward modeling.
                                                    If None, a simple rectangular mesh is created based on
                                                    electrode positions. Defaults to None.
            resistivity_values (Optional[Union[np.ndarray, pg.Vector]], optional):
                Resistivity values [ohm·m] for each cell of the `fwd_mesh`.
                If None, a default homogeneous resistivity (100 ohm·m) is used.
                Must match the number of cells in `fwd_mesh`. Defaults to None.
            scheme_name (str, optional): Name of the ERT array configuration to use (e.g., 'wa' for Wenner alpha,
                                         'dd' for dipole-dipole, 'schlumberger', 'gr' for gradient).
                                         Passed to `ert.createData`. Defaults to 'wa'.
            noise_level_relative (float, optional): Relative level of Gaussian noise to add to the
                                                    synthetic data (e.g., 0.05 for 5% noise).
                                                    Noise is calculated as `data * (1 + N(0,1)*noise_level_relative)`.
                                                    Defaults to 0.05.
            noise_abs_val (float, optional): Absolute error component used for error estimation
                                             by `ert.ERTManager().estimateError()`. Typically in units of the
                                             measured quantity if that were voltage (e.g., uV), or ohm-m if data is already rhoa.
                                             Defaults to 0.0.
            relative_error_for_estimation (Optional[float], optional): Relative error percentage for
                                                                    `ert.ERTManager().estimateError()`.
                                                                    If None, `noise_level_relative` is used. Defaults to None.
            save_path (Optional[str], optional): File path to save the generated synthetic data
                                                 (e.g., 'synthetic_data.dat'). If None, data is not saved.
                                                 Defaults to None.
            show_plot (bool, optional): If True, displays a pseudosection of the generated data.
                                        Defaults to False.
            random_seed (Optional[int], optional): Seed for the random number generator to ensure
                                                   reproducible noise. If None, noise will vary.
                                                   Defaults to None.
            mesh_x_bound (float, optional): Horizontal boundary extension factor/distance if a new mesh
                                            is created. Used by `pg.meshtools.appendTriangleBoundary`.
                                            Defaults to 100.0.
            mesh_y_bound (float, optional): Vertical boundary extension factor/distance if a new mesh
                                            is created. Defaults to 100.0.
            
        Returns:
            Tuple[pg.DataContainer, pg.Mesh]: A tuple containing:
                - synth_data (pg.DataContainer): The generated synthetic ERT data with 'rhoa' and 'err' fields.
                - simulation_mesh (pg.Mesh): The PyGIMLi mesh used for the forward simulation.

        Raises:
            ValueError: If `xpos_electrodes` is empty or if `ypos_electrodes` (if provided)
                        does not match `xpos_electrodes` length, or if resistivity_values
                        length doesn't match mesh cell count.
            TypeError: If inputs are of incorrect type.
            RuntimeError: If PyGIMLi scheme creation or forward modeling fails.
        """
        if not isinstance(xpos_electrodes, np.ndarray) or xpos_electrodes.ndim != 1 or xpos_electrodes.size == 0:
            raise ValueError("xpos_electrodes must be a 1D NumPy array with at least one electrode position.")
        if ypos_electrodes is not None:
            if not isinstance(ypos_electrodes, np.ndarray) or ypos_electrodes.shape != xpos_electrodes.shape:
                raise ValueError("ypos_electrodes, if provided, must be a 1D NumPy array of the same shape as xpos_electrodes.")

        # Set random seed for NumPy and PyGIMLi's RNG if provided for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            pg.Core.RNG().setSeed(random_seed) # More modern way for global GIMLi RNG

        # Create electrode positions: array of [x, y] pairs
        if ypos_electrodes is None:
            ypos_electrodes = np.zeros_like(xpos_electrodes) # Flat surface at y=0

        # Ensure ypos_electrodes is 1D if xpos_electrodes is 1D, for consistent hstack
        electrode_positions = np.vstack((xpos_electrodes.ravel(), ypos_electrodes.ravel())).T

        # Create ERT survey scheme (DataContainer with electrode configurations)
        try:
            scheme = ert.createData(elecs=electrode_positions, schemeName=scheme_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create ERT scheme '{scheme_name}' with PyGIMLi: {e}")

        simulation_mesh: pg.Mesh
        # Prepare mesh for forward modeling
        if fwd_mesh is not None:
            if not isinstance(fwd_mesh, pg.Mesh):
                raise TypeError("fwd_mesh must be a PyGIMLi pg.Mesh object or None.")
            simulation_mesh = fwd_mesh
            # The original code set all cell markers to 2. This might override user's intent.
            # A common practice is to have region markers for different resistivities.
            # If `resistivity_values` is per-cell, markers aren't strictly needed for fwd model assignment,
            # but might be used by inversion later.
            # For now, let's comment out the forced marker setting. User should prepare mesh markers if needed.
            # simulation_mesh.setCellMarkers(pg.Vector(simulation_mesh.cellCount(), 2))
            
            # Append triangle boundary for forward modeling if it's a 2D mesh
            if simulation_mesh.dim() == 2 :
                simulation_mesh = pg.meshtools.appendTriangleBoundary(simulation_mesh, marker=1, # Boundary marker
                                                                    xbound=mesh_x_bound, ybound=mesh_y_bound)
        else:
            # Create a default 2D mesh if none is provided
            x_min_elec, x_max_elec = np.min(xpos_electrodes), np.max(xpos_electrodes)
            y_min_elec, y_max_elec = np.min(ypos_electrodes), np.max(ypos_electrodes)
            
            mesh_x_coords = np.linspace(x_min_elec - mesh_x_bound/5.0, x_max_elec + mesh_x_bound/5.0, 50)
            mesh_y_coords = np.linspace(y_min_elec - mesh_y_bound/2.0, y_max_elec + mesh_y_bound/10.0, 20)

            grid_core = pg.createGrid(x=mesh_x_coords, y=mesh_y_coords, marker=2) # Assign a default marker for core cells
            simulation_mesh = pg.meshtools.appendTriangleBoundary(grid_core, marker=1,
                                                                  xbound=mesh_x_bound, ybound=mesh_y_bound)

        # Prepare resistivity model for the simulation_mesh
        if resistivity_values is None:
            print("No resistivity_values provided. Using homogeneous 100 ohm·m for the simulation mesh.")
            current_resistivity_model = np.ones(simulation_mesh.cellCount()) * 100.0
        elif len(resistivity_values) != simulation_mesh.cellCount():
            raise ValueError(f"Length of resistivity_values ({len(resistivity_values)}) must match cell count "
                             f"of the simulation_mesh ({simulation_mesh.cellCount()}).")
        else:
            current_resistivity_model = resistivity_values

        # Convert to PyGIMLi RVector if it's a NumPy array
        if isinstance(current_resistivity_model, np.ndarray):
            res_model_pg = pg.Vector(current_resistivity_model.ravel())
        elif isinstance(current_resistivity_model, pg.Vector):
            res_model_pg = current_resistivity_model
        else:
            raise TypeError("resistivity_values must be a NumPy array or PyGIMLi RVector.")

        # Create a DataContainer for synthetic data, copying scheme configurations
        synth_data = scheme.copy() # Copies electrode positions and configurations

        # Initialize a new ERTModelling operator for this specific simulation
        # Avoid using `cls` to instantiate if this is a helper method not tied to ERTForwardModeling instance state.
        # The original code used `cls(mesh=grid, data=scheme)` then `fob = ert.ERTModelling()`.
        # Simpler to just use `ert.ERTModelling()` directly here.
        fwd_op_sim = ert.ERTModelling()
        fwd_op_sim.setMesh(simulation_mesh)
        fwd_op_sim.setData(scheme)

        # Compute forward response (apparent resistivities)
        simulated_rhoa_pg = fwd_op_sim.response(res_model_pg)
        simulated_rhoa_np = np.array(simulated_rhoa_pg)

        # Add Gaussian noise to the synthetic apparent resistivities
        if noise_level_relative > 0:
            # Ensure noise is added in a way that doesn't make resistivities non-positive if possible
            # Noise: N(0, noise_level_relative). Data_noisy = Data_true * (1 + noise)
            noise_values = pg.randn(len(simulated_rhoa_np)) * noise_level_relative
            simulated_rhoa_noisy = simulated_rhoa_np * (1.0 + noise_values)
            # Potential issue: if noise_level_relative is large, (1+noise) could be negative.
            # This could make rhoa negative, which is non-physical.
            # Clipping to a small positive value if that happens.
            simulated_rhoa_noisy[simulated_rhoa_noisy <= 1e-6] = 1e-6
        else:
            simulated_rhoa_noisy = simulated_rhoa_np

        synth_data['rhoa'] = simulated_rhoa_noisy # Store apparent resistivity

        # Estimate and assign errors using ERTManager
        # `absoluteUError` in estimateError refers to voltage error, not resistivity error.
        # If we only have rhoa, we use `absoluteError` (for rhoa) and `relativeError`.
        err_manager = ert.ERTManager() # Temporary manager for error estimation

        # Use `relative_error_for_estimation` if provided, otherwise default to `noise_level_relative`
        rel_err_param = relative_error_for_estimation if relative_error_for_estimation is not None else noise_level_relative

        synth_data['err'] = err_manager.estimateError(synth_data,
                                                      absoluteError=noise_abs_val, # Absolute error in ohm-m
                                                      relativeError=rel_err_param)

        if show_plot:
            try:
                ert.showData(synth_data, vals=synth_data['rhoa'], label=r"Apparent Resistivity ($\Omega$m)", cMap="viridis", logScale=True)
                # For plots to show in non-interactive environments:
                # import matplotlib.pyplot as plt
                # plt.show()
            except Exception as e:
                print(f"Could not display data plot: {e}")

        if save_path:
            try:
                synth_data.save(save_path)
                # print(f"Synthetic ERT data saved to: {save_path}")
            except Exception as e:
                print(f"Error saving synthetic data to '{save_path}': {e}")

        return synth_data, simulation_mesh

# Standalone functions - These appear to be alternative or older implementations.
# They should be reviewed for consistency with the ERTForwardModeling class methods.
# If they are redundant or outdated, they could be deprecated or removed.
# If they serve a distinct purpose, their docstrings should clarify this.

def ertforward(fwd_op: ert.ERTModelling,
               mesh_with_markers: pg.Mesh,
               rhomodel_background: pg.Vector,
               xr_active_log: np.ndarray
              ) -> Tuple[np.ndarray, pg.Vector]:
    """
    Performs ERT forward modeling, updating only a specific region of the model.

    This function takes an existing background resistivity model (`rhomodel_background`),
    updates a portion of it (where cell markers == 2 in `mesh_with_markers`)
    with new log-transformed resistivity values (`xr_active_log`), and then computes
    the forward response. The response is returned as log-transformed apparent resistivities.

    Args:
        fwd_op (ert.ERTModelling): Initialized PyGIMLi ERTModelling operator.
                                   Must have data scheme and full forward mesh already set.
                                   The full forward mesh should be consistent with `rhomodel_background`.
        mesh_with_markers (pg.Mesh): A PyGIMLi mesh object whose cell markers define the active region.
                                     Cells with marker 2 will be updated. This mesh's cell indexing
                                     must correspond to `rhomodel_background`.
        rhomodel_background (pg.RVector): Full resistivity model vector [ohm·m] for all cells
                                          in the mesh associated with `fwd_op`.
        xr_active_log (np.ndarray): 1D NumPy array of log-transformed (natural log) resistivity values
                                for the active region cells (marker == 2). The length of `xr_active_log`
                                must match the number of cells marked as 2.

    Returns:
        Tuple[np.ndarray, pg.RVector]:
            - dr_log (np.ndarray): Log-transformed (natural log) forward response (apparent resistivities).
            - rhomodel_updated_pg (pg.RVector): The updated full resistivity model vector [ohm·m] after
                                                modifying the active region.
    """
    # Convert the background resistivity model to a NumPy array for easier manipulation.
    rhomodel_current_np = np.array(rhomodel_background)

    # Identify cells in the active region using markers from `mesh_with_markers`.
    # It's crucial that `mesh_with_markers` is the same mesh that `rhomodel_background` refers to.
    cell_markers = np.array(mesh_with_markers.cellMarkers())
    active_cell_mask = (cell_markers == 2)

    if np.sum(active_cell_mask) != len(xr_active_log):
        raise ValueError(f"Length of xr_active_log ({len(xr_active_log)}) does not match the number of cells "
                         f"with marker 2 ({np.sum(active_cell_mask)}) in mesh_with_markers.")

    # Update the resistivity values in the active region.
    # `xr_active_log` contains log-transformed values, so exponentiate them.
    rhomodel_current_np[active_cell_mask] = np.exp(xr_active_log)

    rhomodel_updated_pg = pg.Vector(rhomodel_current_np) # Convert updated model back to RVector

    # Compute forward response with the updated model
    response_pg = fwd_op.response(rhomodel_updated_pg)
    response_np = np.array(response_pg)

    # Log-transform the response, handling non-positive values
    if np.any(response_np <= 0):
        print("Warning (ertforward function): Response contains non-positive values. Log transform will result in -inf or NaN.")
    with np.errstate(divide='ignore', invalid='ignore'):
        log_response_np = np.log(response_np)

    return log_response_np, rhomodel_updated_pg


def ertforward2(fwd_op: ert.ERTModelling,
                xr_log_model_full: np.ndarray,
                mesh: Optional[pg.Mesh] = None # Argument kept for signature consistency, but often unused if fwd_op is pre-configured
               ) -> np.ndarray:
    """
    Simplified ERT forward model calculation from a full log-transformed resistivity model.

    Assumes `xr_log_model_full` is a complete vector of log-transformed (natural log)
    resistivity values for all cells in the mesh already associated with `fwd_op`.
    The function exponentiates these values, computes the response, and returns the
    natural log of the response.

    Args:
        fwd_op (ert.ERTModelling): Initialized PyGIMLi ERTModelling operator, with mesh and data scheme set.
        xr_log_model_full (np.ndarray): 1D NumPy array of log-transformed (natural log)
                                        resistivity values for all model cells.
        mesh (Optional[pg.Mesh]): This argument is present in the original signature but typically
                                  not used if `fwd_op` is already configured with its mesh and
                                  `xr_log_model_full` corresponds to that mesh's cells.
                                  If provided, it could be used for validation.

    Returns:
        np.ndarray: Log-transformed (natural log) forward response (apparent resistivities).
    """
    # Convert log-domain model to actual resistivity values: ρ = exp(log_ρ)
    actual_resistivity_values_np = np.exp(xr_log_model_full)

    # Ensure model is in PyGIMLi RVector format for the forward operator
    rhomodel_pg_linear = pg.Vector(actual_resistivity_values_np)

    response_pg = fwd_op.response(rhomodel_pg_linear)
    response_np = np.array(response_pg)

    if np.any(response_np <= 0):
        print("Warning (ertforward2 function): Response contains non-positive values. Log transform will result in -inf or NaN.")
    with np.errstate(divide='ignore', invalid='ignore'):
        log_response_np = np.log(response_np)

    return log_response_np


def ertforandjac(fwd_op: ert.ERTModelling,
                 rhomodel_linear_pg: pg.Vector, # Full model in linear resistivity [ohm·m]
                 xr_active_log_full: np.ndarray    # Log-resistivity for the entire model [log(ohm·m)]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward model and Jacobian calculation for ERT, with log-transform adjustments.

    This function computes the forward response `d` using `rhomodel_linear_pg` (which should be ρ).
    It then calculates the Jacobian J_lin = ∂d/∂ρ based on `rhomodel_linear_pg`.
    Finally, it transforms this Jacobian to J_loglog = ∂log(d)/∂log(ρ) = (ρ/d) * J_lin,
    using `xr_active_log_full` to derive ρ for scaling (ρ = exp(xr_active_log_full)).
    The response `d` is also returned as log(d).

    Args:
        fwd_op (ert.ERTModelling): Initialized PyGIMLi ERTModelling operator.
        rhomodel_linear_pg (pg.RVector): Full resistivity model vector in linear scale [ohm·m].
                                         This is ρ, used to compute response and base Jacobian.
        xr_active_log_full (np.ndarray): 1D NumPy array of log-transformed (natural log) resistivity values
                                         for ALL model cells. This is log(ρ). Used for Jacobian scaling.
                                         Length must match number of cells in `fwd_op`'s mesh.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dr_log (np.ndarray): Log-transformed (natural log) forward response, log(d).
            - J_loglog (np.ndarray): Adjusted Jacobian matrix for log-data versus log-model changes.
    """
    # 1. Compute response d = f(ρ) using the linear resistivity model `rhomodel_linear_pg`
    response_pg = fwd_op.response(rhomodel_linear_pg)
    response_np_linear = np.array(response_pg) # This is 'd'

    # 2. Compute base Jacobian J_lin = ∂d/∂ρ using the linear resistivity model `rhomodel_linear_pg`
    fwd_op.createJacobian(rhomodel_linear_pg)
    jacobian_gmat = fwd_op.jacobian()
    jacobian_np_linear = pg.utils.gmat2numpy(jacobian_gmat) # This is J_lin = ∂d/∂ρ

    # 3. Adjust Jacobian and response for log-log transformation.
    # J_loglog = ∂(log d) / ∂(log ρ) = (ρ/d) * (∂d/∂ρ) = diag(ρ/d) * J_lin if J is row vector of gradients
    # Or more generally, J_loglog_ij = (ρ_j / d_i) * (∂d_i / ∂ρ_j)
    # This is achieved by: J_loglog = diag(1/d) @ J_lin @ diag(ρ)

    # ρ values from the provided log-transformed full model array
    rho_values_for_scaling_np = np.exp(xr_active_log_full)

    # Check for issues before scaling
    if np.any(response_np_linear <= 0):
        print("Warning (ertforandjac function): Linear response `d` contains non-positive values. Jacobian scaling may yield NaN/Inf.")
    if np.any(rho_values_for_scaling_np <= 0):
        print("Warning (ertforandjac function): Model resistivities `ρ` (from exp(xr_active_log_full)) contain non-positive values. Jacobian scaling may yield NaN/Inf.")

    # Create diagonal matrix diag(ρ)
    diag_rho = np.diag(rho_values_for_scaling_np)
    # J_lin @ diag(ρ) effectively scales columns of J_lin by corresponding ρ_j
    jacobian_times_rho = jacobian_np_linear @ diag_rho

    # Create diagonal matrix diag(1/d)
    inv_d_values = np.zeros_like(response_np_linear)
    valid_d_mask = response_np_linear > 1e-12 # Avoid division by zero or very small numbers
    inv_d_values[valid_d_mask] = 1.0 / response_np_linear[valid_d_mask]
    # For d <= 0, inv_d_values remains 0. This means corresponding rows in J_loglog will be zero.
    diag_inv_d = np.diag(inv_d_values)

    J_loglog = diag_inv_d @ jacobian_times_rho

    # Log-transform the linear response: log(d)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_response_np = np.log(response_np_linear)

    return log_response_np, J_loglog


def ertforandjac2(fwd_op: ert.ERTModelling,
                  xr_log_model_full: np.ndarray,
                  mesh: Optional[pg.Mesh] = None # Typically unused if fwd_op is fully configured
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternative ERT forward model and Jacobian calculation using log-resistivity values.

    This function is very similar to `ertforandjac`. It assumes the input `xr_log_model_full`
    is the natural log of resistivity for all cells. It calculates d = f(exp(log_ρ))
    and then transforms the Jacobian to J_loglog = (ρ/d) * (∂d/∂ρ).

    The main difference from `ertforandjac` seems to be how the linear model `rhomodel`
    is derived (from `xr_log_model_full` directly) rather than being passed in.
    The Jacobian transformation logic is intended to be the same (for log-data vs log-model).

    Args:
        fwd_op (ert.ERTModelling): Initialized PyGIMLi ERTModelling operator.
        xr_log_model_full (np.ndarray): 1D NumPy array of log-transformed (natural log)
                                        resistivity values for all model cells.
        mesh (Optional[pg.Mesh]): PyGIMLi mesh. Unused in the original logic if `xr_log_model_full`
                                  is for the full model set in `fwd_op`. Retained for signature.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dr_log (np.ndarray): Log-transformed (natural log) forward response.
            - J_loglog (np.ndarray): Adjusted Jacobian matrix for log-log space.
    """
    # Convert full log-model to linear resistivity model: ρ = exp(log_ρ)
    actual_resistivity_values_np = np.exp(xr_log_model_full)
    rhomodel_linear_pg = pg.Vector(actual_resistivity_values_np)

    # Calculate response d = f(ρ)
    response_pg = fwd_op.response(rhomodel_linear_pg)
    response_np_linear = np.array(response_pg) # This is 'd'

    # Calculate base Jacobian J_lin = ∂d/∂ρ
    fwd_op.createJacobian(rhomodel_linear_pg)
    jacobian_gmat = fwd_op.jacobian()
    jacobian_np_linear = pg.utils.gmat2numpy(jacobian_gmat) # J_lin = ∂d/∂ρ

    # Adjust Jacobian for log-log space: J_loglog = (ρ/d) * J_lin
    # rho_values_for_scaling is actual_resistivity_values_np

    # Original line: J = np.exp(xr.T)*J. This is mathematically incorrect for matrix multiplication.
    # If xr is (N,) and J is (M,N), exp(xr.T) is still (N,).
    # `np.exp(xr.T) * J` would be element-wise multiplication if J was also (N,), or broadcasting.
    # Correct scaling for J_loglog = diag(1/d) @ J_lin @ diag(ρ):

    diag_rho = np.diag(actual_resistivity_values_np)
    jacobian_times_rho = jacobian_np_linear @ diag_rho

    if np.any(response_np_linear <= 0):
        print("Warning (ertforandjac2 function): Linear response `d` contains non-positive values. Jacobian scaling may yield NaN/Inf.")

    inv_d_values = np.zeros_like(response_np_linear)
    valid_d_mask = response_np_linear > 1e-12
    inv_d_values[valid_d_mask] = 1.0 / response_np_linear[valid_d_mask]
    diag_inv_d = np.diag(inv_d_values)

    J_loglog = diag_inv_d @ jacobian_times_rho

    # Log-transform the linear response
    with np.errstate(divide='ignore', invalid='ignore'):
        log_response_np = np.log(response_np_linear)

    return log_response_np, J_loglog


