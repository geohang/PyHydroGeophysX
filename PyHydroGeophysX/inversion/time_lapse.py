"""
Time-lapse ERT inversion functionality.
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import lsqr
from typing import Optional, Union, List, Dict, Any, Tuple
from scipy.linalg import block_diag


from .base import InversionBase, TimeLapseInversionResult
from ..forward.ert_forward import ertforward2, ertforandjac2
from ..solvers.solver import generalized_solver


def _calculate_jacobian(fwd_operators, model, mesh, size):
    """
    Calculate Jacobian matrix for multi-time model.
    
    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        
    Returns:
        obs_stacked (np.ndarray): Stacked observed (predicted) data for all timesteps (log-transformed).
                                  Shape: (total_num_data, 1).
        J_block_diag (scipy.sparse.csr_matrix or np.ndarray): Block diagonal Jacobian matrix.
                                                              Each block is the Jacobian for one timestep.
                                                              Shape: (total_num_data, total_num_model_params).
    """
    # Reshape the flattened model vector `model` into a 2D array where
    # columns represent timesteps and rows represent cells/parameters.
    # Order 'F' (Fortran-like) means column-major order: model = [m_t1_cell1, m_t1_cell2, ..., m_t2_cell1, ...]
    # So, model_reshaped will have shape (num_cells, num_timesteps).
    model_reshaped = np.reshape(model, (-1, size), order='F') # (-1 implies inferring num_cells)
    
    obs_list = [] # List to store observed data (log-transformed predictions) for each timestep.
    jacobian_blocks = [] # List to store Jacobian matrices for each timestep.
    
    # Iterate through each timestep.
    for i in range(size):
        # `model_reshaped[:, i]` is the log-resistivity model for the i-th timestep.
        # `fwd_operators[i]` is the forward operator for the i-th timestep's data configuration.
        # `ertforandjac2` calculates log(response) and Jacobian d(log(response))/d(log(model_param)).
        dr_log, Jr_log_log = ertforandjac2(fwd_operators[i], model_reshaped[:, i], mesh)

        obs_list.append(dr_log) # Store log-transformed predicted data.
        jacobian_blocks.append(Jr_log_log) # Store the Jacobian for this timestep.
        # SUGGESTION: If Jacobian matrices (Jr_log_log) are large and dense,
        # converting them to sparse format before `block_diag` might be more memory efficient.
        # e.g., `jacobian_blocks.append(csr_matrix(Jr_log_log))`

    # --- Construct the full block diagonal Jacobian matrix ---
    # `block_diag` from `scipy.linalg` (or `scipy.sparse` for sparse matrices)
    # creates a block diagonal matrix from the list of Jacobian blocks.
    # This assumes that model parameters for different timesteps are independent in terms of data misfit
    # (i.e., data at time t_i only depends on model at t_i).
    # Spatial regularization will couple parameters within each timestep's model block.
    # Temporal regularization will couple parameters across different timestep blocks.
    if jacobian_blocks: # Check if list is not empty
        J_block_diag = block_diag(*jacobian_blocks) # The * unpacks the list of matrices.
                                                  # If Jr are dense, J_block_diag is dense.
                                                  # If Jr are sparse, J_block_diag is sparse.
                                                  # For inversion, sparse format is usually preferred.
    else: # Should not happen if size > 0
        J_block_diag = np.array([]).reshape(0, model_reshaped.shape[0]*size) # Empty with correct columns if no timesteps
        # Or handle as an error.

    # --- Stack observations into a single column vector ---
    # obs_list contains arrays of log-transformed data for each timestep.
    # These are stacked vertically to form one long vector of observations.
    if obs_list: # Check if list is not empty
        obs_stacked = obs_list[0].reshape(-1, 1) # Start with the first timestep's data, ensure column vector.
        for i in range(size - 1): # Iterate for remaining timesteps.
            obs_stacked = np.vstack((obs_stacked, obs_list[i + 1].reshape(-1, 1)))
    else: # Should not happen if size > 0
        obs_stacked = np.array([]).reshape(0,1) # Empty column vector

    return obs_stacked, J_block_diag


def _calculate_forward(fwd_operators, model, mesh, size):
    """
    Calculate forward response for multi-time model.
    
    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        
    Returns:
        obs_stacked (np.ndarray): Stacked observed (predicted) data for all timesteps (log-transformed).
                                  Shape: (total_num_data, 1).
    """
    # Reshape the model vector similar to _calculate_jacobian.
    # Model is flattened column-major: parameters for t1, then t2, etc.
    # model_reshaped will have shape (num_cells_per_timestep, num_timesteps).
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs_list = [] # List to store predicted data for each timestep.
    
    # Iterate through each timestep.
    for i in range(size):
        # `model_reshaped[:, i]` is the log-resistivity model for the i-th timestep.
        # `fwd_operators[i]` is the forward operator for the i-th data.
        # `ertforward2` computes log(response) given log-model.
        dr_log = ertforward2(fwd_operators[i], model_reshaped[:, i], mesh)
        obs_list.append(dr_log) # Append log-transformed predicted data.
    
    # Stack the list of observation arrays into a single column vector.
    # This matches the structure expected by the inversion algorithm.
    if obs_list: # Check if list is not empty
        obs_stacked = obs_list[0].reshape(-1, 1) # Start with the first, ensure column vector.
        for i in range(size - 1):
            obs_stacked = np.vstack((obs_stacked, obs_list[i + 1].reshape(-1, 1)))
    else: # Should not happen if size > 0
        obs_stacked = np.array([]).reshape(0,1)

    return obs_stacked


def _calculate_forward_separate(fwd_operators, model, mesh, size):
    """
    Calculate forward response for multi-time model without stacking.
    
    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        
    Returns:
        obs_list (List[np.ndarray]): A list where each element is a NumPy array of
                                     log-transformed predicted data for a single timestep.
    """
    # Reshape the model vector as in other helper functions.
    # model_reshaped will have shape (num_cells_per_timestep, num_timesteps).
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs_list = [] # List to store predicted data for each timestep separately.
    
    # Iterate through each timestep.
    for i in range(size):
        # Calculate log-transformed predicted data for the current timestep's model.
        dr_log = ertforward2(fwd_operators[i], model_reshaped[:, i], mesh)
        obs_list.append(dr_log) # Append the array for this timestep to the list.
    
    # Returns a list of arrays, unlike the other helpers that stack them.
    # This might be useful if subsequent processing needs per-timestep data.
    return obs_list


class TimeLapseERTInversion(InversionBase):
    """Time-lapse ERT inversion class."""
    
    def __init__(self, data_files: List[str], measurement_times: List[float],
                mesh: Optional[pg.Mesh] = None, **kwargs):
        """
        Initialize time-lapse ERT inversion.
        
        Args:
            data_files: List of paths to ERT data files
            measurement_times: List of measurement times
            mesh: Mesh for inversion (created if None)
            **kwargs: Additional parameters including:
                - `lambda_val` (float): Spatial regularization parameter (smoothness within each timestep).
                - `alpha` (float): Temporal regularization parameter (smoothness between timesteps).
                - `decay_rate` (float): Decay rate for temporal regularization weights (exponential decay).
                - `method` (str): Solver for linear system (e.g., 'cgls', 'lsqr').
                - `model_constraints` (Tuple[float,float]): Min/max resistivity bounds (linear scale).
                - `max_iterations` (int): Max iterations for the main Gauss-Newton loop.
                - `absoluteUError` (float): Absolute data error component.
                - `relativeError` (float): Relative data error component.
                - `lambda_rate` (float): Reduction factor for spatial lambda per iteration.
                - `lambda_min` (float): Minimum spatial lambda.
                - `inversion_type` (str): Norm for data/model misfit ('L1', 'L2', 'L1L2').
        """
        # Store paths to data files and corresponding measurement times.
        self.data_files = data_files
        self.measurement_times = np.array(measurement_times) # Convert to numpy array for potential calculations.
        
        # Validate that the number of data files matches the number of measurement times.
        if len(data_files) != len(measurement_times):
            raise ValueError("Number of data files must match the number of measurement times.")

        # Load the first dataset to initialize the parent `InversionBase` class.
        # `InversionBase` likely requires a sample data container for some initial setup.
        first_data_container = ert.load(data_files[0])

        # Call parent class initializer.
        super().__init__(first_data_container, mesh, **kwargs) # `data` attribute in base will be first_data_container.

        # --- Set Time-Lapse Specific Default Parameters ---
        # These defaults are applied if not overridden by user-provided `kwargs`.
        timelapse_defaults = {
            'lambda_val': 100.0,      # Spatial regularization strength.
            'alpha': 10.0,            # Temporal regularization strength (relative to lambda_val or absolute).
            'decay_rate': 0.0,        # Exponential decay rate for temporal weights (0 means uniform weights).
            'method': 'cgls',         # Default linear solver.
            'absoluteUError': 0.0,    # Default absolute data error.
            'relativeError': 0.05,    # Default relative data error (5%).
            'lambda_rate': 0.8,       # Factor to reduce spatial lambda each iteration (e.g., lambda_new = lambda_old * 0.8).
            'lambda_min': 1.0,        # Minimum value for spatial lambda.
            'inversion_type': 'L2',   # Default to L2 norm (least squares). Options: 'L1', 'L2', 'L1L2' (hybrid).
            'model_constraints': (0.0001, 10000.0), # Min/max resistivity values [ohm.m].
                                                  # SUGGESTION: These constraints are quite wide. Might need adjustment based on expected geology.
        }
        
        # Update `self.parameters` (from InversionBase) with these defaults.
        for key, value in timelapse_defaults.items():
            if key not in self.parameters: # Only set if not already provided in kwargs.
                self.parameters[key] = value
        
        # --- Initialize Time-Lapse Specific Attributes ---
        self.size = len(data_files) # Number of time steps in the lapse sequence.
        
        self.fwd_operators = [] # List to store PyGIMLi forward operators for each timestep.
        self.datasets = []      # List to store PyGIMLi data containers for each timestep.
        self.rhos1 = None       # Will store all observed log-apparent resistivities stacked into one vector.
        self.Wd = None          # Combined data weighting matrix for all timesteps.
        self.Wm = None          # Combined spatial regularization matrix (block diagonal).
        self.Wt = None          # Temporal regularization matrix.
    
    def setup(self):
        """Set up time-lapse ERT inversion (load data, create operators, matrices, etc.)"""
        # --- Mesh Creation (if not provided during __init__) ---
        # `self.data` here refers to the first dataset loaded in __init__.
        if self.mesh is None:
            print("Mesh not provided, creating default mesh based on the first dataset.")
            ert_manager_setup = ert.ERTManager(self.data) # Use the first dataset for mesh creation guidance.
            self.mesh = ert_manager_setup.createMesh(data=self.data, quality=34)
            # SUGGESTION: Ensure this mesh is appropriate for ALL datasets if their geometries differ significantly.
            # Or, consider creating a combined mesh / using individual meshes if necessary.

        # --- Load All Datasets and Initialize Forward Operators ---
        observed_rhoa_list = []    # List to store apparent resistivities for each dataset.
        data_error_list = []       # List to store error estimates for each dataset.
        geometric_factors_k = [] # Store geometric factors if needed.

        for i, data_file_path in enumerate(self.data_files):
            # Load the current ERT data file.
            current_data_container = ert.load(data_file_path)
            self.datasets.append(current_data_container) # Store the loaded data container.
            
            # --- Handle Geometric Factors (k) ---
            # If 'k' is not present or all zeros, try to calculate it.
            # This is crucial if data is in resistance ('r') and needs conversion to rhoa.
            if 'k' not in current_data_container.tInfos or np.all(current_data_container['k'] == 0.0):
                if not geometric_factors_k: # If k hasn't been calculated from the first dataset yet
                    print(f"Calculating geometric factors for dataset {i} (and assuming same for others if needed).")
                    current_data_container['k'] = ert.createGeometricFactors(current_data_container, numerical=True) # Calculate k
                    geometric_factors_k = current_data_container['k'].array() # Store for potential reuse
                else: # Reuse k from the first dataset if subsequent ones are missing it.
                      # This assumes all datasets use compatible configurations for k.
                    print(f"Warning: Dataset {i} missing geometric factors. Reusing from first dataset. Ensure configurations are compatible.")
                    current_data_container['k'] = geometric_factors_k
            
            # --- Get Apparent Resistivity (rhoa) ---
            # If 'rhoa' is present and non-zero, use it. Otherwise, calculate from resistance 'r' and 'k'.
            if 'rhoa' in current_data_container.tInfos and np.any(current_data_container['rhoa']):
                observed_rhoa_list.append(current_data_container['rhoa'].array())
            elif 'r' in current_data_container.tInfos and 'k' in current_data_container.tInfos:
                print(f"Calculating rhoa from r and k for dataset {i}.")
                rhoa_calculated = current_data_container['r'].array() * current_data_container['k'].array()
                current_data_container['rhoa'] = rhoa_calculated # Store it back
                observed_rhoa_list.append(rhoa_calculated)
            else:
                raise ValueError(f"Dataset {i} ({data_file_path}) contains neither 'rhoa' nor 'r'/'k' for resistivity calculation.")

            # --- Get or Estimate Data Errors ---
            # Similar logic as in single-time ERTInversion setup.
            if 'err' in current_data_container.tInfos and np.any(current_data_container['err']):
                data_error_list.append(current_data_container['err'].array())
            else:
                print(f"Estimating errors for dataset {i} using provided error parameters.")
                temp_ert_manager = ert.ERTManager(current_data_container)
                estimated_err = temp_ert_manager.estimateError(
                    current_data_container,
                    absoluteUError=self.parameters['absoluteUError'],
                    relativeError=self.parameters['relativeError']
                ).array()
                current_data_container['err'] = estimated_err # Store back
                data_error_list.append(estimated_err)
            
            # --- Create and Configure Forward Operator for Current Timestep ---
            fwd_op_timestep = ert.ERTModelling()
            fwd_op_timestep.setData(current_data_container) # Set specific data scheme for this timestep.
            fwd_op_timestep.setMesh(self.mesh)             # Use the common mesh for all timesteps.
            self.fwd_operators.append(fwd_op_timestep)     # Store the configured operator.

        # --- Aggregate Data and Errors ---
        # Stack all observed apparent resistivities into a single column vector (log-transformed).
        # `observed_rhoa_list` contains numpy arrays, one for each timestep.
        # Ensure all rhoa are positive before log. Add epsilon if necessary.
        # SUGGESTION: Add check for non-positive rhoa values before log.
        epsilon_rhoa = 1e-9 # Small value to prevent log(0)
        stacked_rhoa_linear = np.concatenate(observed_rhoa_list).ravel() # Flatten into 1D array
        self.rhos1 = np.log(stacked_rhoa_linear + epsilon_rhoa).reshape(-1, 1) # Log and reshape to column vector.

        # Stack all data errors and create the combined data weighting matrix Wd.
        # Assumes `data_error_list` contains fractional errors for linear data.
        stacked_data_errors_linear = np.concatenate(data_error_list).ravel()
        # Using the same potentially problematic weighting formula as in single ERT.
        # Wd_diag = 1.0 / log(error_fractional + 1)
        # SUGGESTION: Re-evaluate this weighting. Standard is 1.0 / error_fractional (if error_fractional is std dev of log data).
        self.Wd = np.diag(1.0 / np.log(stacked_data_errors_linear + 1.0))

        # --- Create Model Regularization Matrices ---
        # Spatial Regularization Matrix (Wm):
        # This matrix imposes smoothness constraints *within* each timestep's model.
        # It's created as a block diagonal matrix, where each block is the smoothness
        # matrix for a single timestep.
        rm_template = self.fwd_operators[0].regionManager() # Use region manager from first operator as template.
        constraint_matrix_single_time = pg.matrix.RSparseMapMatrix()
        rm_template.setConstraintType(1) # 1st order smoothness.
        rm_template.fillConstraints(constraint_matrix_single_time)

        Wm_r_single_time_coo = pg.utils.sparseMatrix2coo(constraint_matrix_single_time)
        constraint_weights_single_time = rm_template.constraintWeights().array()
        Wm_r_single_time_weighted = diags(constraint_weights_single_time).dot(Wm_r_single_time_coo)
        Wm_r_single_time_dense = Wm_r_single_time_weighted.todense() # Dense matrix for single timestep.
                                                                    # SUGGESTION: Keep sparse for block_diag.

        # Create a list of these single-timestep spatial regularization matrices.
        list_of_Wm_blocks = [Wm_r_single_time_dense] * self.size # Repeat for each timestep.
        # Create the full block diagonal Wm matrix.
        # If Wm_r_single_time_dense is kept sparse (e.g., CSR), block_diag will be more efficient.
        self.Wm = block_diag(*list_of_Wm_blocks) # scipy.linalg.block_diag for dense, or scipy.sparse.block_diag.
                                                 # If inputs are dense, output is dense.
                                                 # SUGGESTION: Use sparse block_diag if Wm_r_single_time_dense is sparse.
                                                 # `from scipy.sparse import block_diag as sparse_block_diag`

        # Temporal Regularization Matrix (Wt):
        # This matrix imposes smoothness constraints *between* corresponding cells at different timesteps.
        # It penalizes m_t - m_{t-1}.
        num_cells_per_timestep = self.fwd_operators[0].paraDomain().cellCount()
        time_differences = np.diff(self.measurement_times) # Differences between measurement times.

        # Calculate temporal weights, potentially decaying with time difference.
        # w_temporal_j = exp(-decay_rate * Δt_j)
        temporal_weights_flat = []
        for dt_idx in range(self.size - 1): # For each time interface (t_i, t_{i+1})
            weight_for_interface = np.exp(-self.parameters['decay_rate'] * time_differences[dt_idx])
            temporal_weights_flat.extend([weight_for_interface] * num_cells_per_timestep)
        temporal_weights_flat = np.array(temporal_weights_flat)

        # Construct Wt:
        # Wt has (size-1)*num_cells rows and size*num_cells columns.
        # Each row group [idx : idx+num_cells] corresponds to differences m_i - m_{i+1}.
        # It has blocks of [I, -I] for adjacent timesteps.
        Wt_构造 = np.zeros((num_cells_per_timestep * (self.size - 1), num_cells_per_timestep * self.size))
        identity_block = np.eye(num_cells_per_timestep)
        for i in range(self.size - 1): # Iterate through time interfaces
            row_start_idx = i * num_cells_per_timestep
            # For m_i part:
            Wt_构造[row_start_idx : row_start_idx + num_cells_per_timestep,
                   i * num_cells_per_timestep : (i+1) * num_cells_per_timestep] = identity_block
            # For -m_{i+1} part:
            Wt_构造[row_start_idx : row_start_idx + num_cells_per_timestep,
                   (i+1) * num_cells_per_timestep : (i+2) * num_cells_per_timestep] = -identity_block

        # Apply temporal weights.
        # self.Wt should be a sparse matrix for efficiency.
        self.Wt = diags(temporal_weights_flat).dot(csr_matrix(Wt_构造)) # Convert Wt_构造 to sparse first.
                                                                    # Using csr_matrix for efficient dot product.
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> TimeLapseInversionResult:
        """
        Run time-lapse ERT inversion.
        
        Args:
            initial_model: Initial model parameters (if None, a homogeneous model is used)
            
        Returns:
            TimeLapseInversionResult: An object containing the final inverted models for all timesteps,
                                      coverage, predicted data, and iteration statistics.
        """
        # Ensure setup has been called to initialize data, operators, and matrices.
        if not self.fwd_operators: # Check if list of forward operators is empty.
            self.setup()
        
        # Initialize TimeLapseInversionResult object to store outcomes.
        result = TimeLapseInversionResult()
        result.timesteps = self.measurement_times # Store measurement times for context.
        
        # --- Initial Model (mr) ---
        # The model `mr` is a flattened vector of log-resistivities for all cells and all timesteps.
        # Order: [log(rho_cell1_t1), log(rho_cell2_t1), ..., log(rho_cellN_t1), log(rho_cell1_t2), ...]
        num_cells_per_timestep = self.fwd_operators[0].paraDomain().cellCount()
        
        current_log_model_flat: np.ndarray # This will be the model vector used in iterations.
        if initial_model is None:
            # If no initial model provided, create a default one:
            # For each timestep, use the median of its observed log-apparent resistivities
            # as the homogeneous starting log-resistivity for that timestep's model.
            initial_log_rho_timesteps = []
            for i in range(self.size): # self.size is number of timesteps
                # Extract observed rhoa for the current dataset, log it, and find median.
                current_data_rhoa_linear = self.datasets[i]['rhoa'].array()
                # Add epsilon for safety before log, though rhoa should be positive.
                median_log_rhoa_timestep = np.median(np.log(current_data_rhoa_linear + 1e-9))
                initial_log_rho_timesteps.append(median_log_rhoa_timestep)
            
            # Repeat median for all cells in that timestep, then flatten and make column vector.
            # `np.repeat` creates [med1,med1,..(Ncells)..med2,med2,...]
            current_log_model_flat = np.log(np.repeat(initial_log_rho_timesteps, num_cells_per_timestep)).reshape(-1, 1)
            # The above line seems to take log of medians again. If `initial_log_rho_timesteps` are already log,
            # then it should be: `current_log_model_flat = np.repeat(initial_log_rho_timesteps, num_cells_per_timestep).reshape(-1,1)`
            # Correcting based on `median_log_rhoa_timestep` already being log:
            current_log_model_flat = np.repeat(initial_log_rho_timesteps, num_cells_per_timestep).reshape(-1,1)

        else: # User provided an initial_model.
            # Expected shape: (num_cells_per_timestep, num_timesteps) in linear resistivity.
            if initial_model.shape != (num_cells_per_timestep, self.size):
                raise ValueError(f"Provided initial_model shape {initial_model.shape} does not match "
                                 f"expected shape ({num_cells_per_timestep}, {self.size}).")
            # Flatten in Fortran order (column-major) to match [t1_allcells, t2_allcells, ...]
            # and then log-transform. Add epsilon for log safety.
            current_log_model_flat = np.log(initial_model.flatten(order='F') + 1e-9).reshape(-1, 1)

        # --- Reference Model (mr_R) ---
        # Used for regularization, typically penalizes deviation from this model.
        # Here, the initial model itself is used as the reference model.
        log_reference_model_flat = current_log_model_flat.copy()

        # --- Regularization Parameters ---
        # Spatial regularization strength (lambda).
        current_spatial_lambda = self.parameters['lambda_val']
        # Temporal regularization strength (alpha).
        current_temporal_alpha = self.parameters['alpha']

        # --- Model Constraints (Bounds in Log Space) ---
        min_rho_linear, max_rho_linear = self.parameters['model_constraints']
        min_log_rho = np.log(min_rho_linear)
        max_log_rho = np.log(max_rho_linear)
        print(f"Model log-resistivity bounds: min={min_log_rho:.2f}, max={max_log_rho:.2f}") # Debug print

        # --- Iteration Tracking Variables ---
        iteration_stats_list = [] # To store [chi2, model_misfit, temporal_misfit] per iteration.
        previous_chi2 = np.inf    # Chi-squared from previous iteration, for dPhi calculation.

        # --- Inversion Type and IRLS Parameters ---
        # Determine inversion norm type (L1, L2, or L1L2 hybrid).
        inversion_norm_type = self.parameters['inversion_type'].upper()
        if inversion_norm_type not in ['L1', 'L2', 'L1L2']:
            print(f"Warning: Invalid inversion_type '{inversion_norm_type}'. Defaulting to 'L2'.")
            inversion_norm_type = 'L2'

        # Parameters for Iteratively Reweighted Least Squares (IRLS) if L1 or L1L2 norm is used.
        # Epsilon for L1 norm stability (prevents division by zero in weights).
        l1_epsilon_irls = 1e-4
        # Max IRLS iterations (outer loop). Gauss-Newton (GN) iterations are inner loop.
        max_irls_iterations = 1
        if inversion_norm_type == 'L1': max_irls_iterations = 5
        if inversion_norm_type == 'L1L2': max_irls_iterations = 8
        # Tolerance for IRLS convergence (relative change in model).
        irls_convergence_tolerance = 1e-3 if inversion_norm_type == 'L1' else 1e-2
        # Threshold for L1L2 hybrid norm (Huber-like transition between L2 and L1).
        l1l2_hybrid_threshold_c = 2.0

        log_model_previous_irls = current_log_model_flat.copy() # For checking IRLS convergence.

        # --- Main Inversion Loop (Outer IRLS loop, Inner Gauss-Newton loop) ---
        for irls_iter_count in range(max_irls_iterations):
            if inversion_norm_type in ['L1', 'L1L2']:
                print(f'------------------- IRLS Iteration: {irls_iter_count + 1} ---------------------------')
            
            # Gauss-Newton iterations (inner loop)
            for gn_iter_count in range(self.parameters['max_iterations']):
                print(f'------------------- Gauss-Newton Iteration (ERT Total Iteration): {gn_iter_count} (IRLS {irls_iter_count+1}) ----')
                
                # 1. Forward Modeling and Jacobian Calculation for the current model `current_log_model_flat`.
                # `_calculate_jacobian` returns stacked log-predicted data and block-diagonal log-log Jacobian.
                log_predicted_data_stacked, jacobian_log_log_block_diag = _calculate_jacobian(
                    self.fwd_operators, current_log_model_flat, self.mesh, self.size
                )
                # log_predicted_data_stacked is already a column vector.
                
                # 2. Data Misfit Calculation
                # `data_residual_log` = log(observed_data_stacked) - log(predicted_data_stacked)
                data_residual_log = self.rhos1 - log_predicted_data_stacked # self.rhos1 is already (N_total_data, 1)
                
                # --- Norm-Specific Calculations for Misfit, Gradient, Hessian Approx. ---
                # Initialize components for L2 case, then modify if L1/L1L2.
                # Data term components:
                current_data_misfit_term: float
                data_weighting_matrix_for_hessian = self.Wd.T @ self.Wd # Wd^T * Wd
                data_part_of_gradient = jacobian_log_log_block_diag.T @ data_weighting_matrix_for_hessian @ (-data_residual_log)

                # Spatial model term components:
                spatial_model_diff = self.Wm @ current_log_model_flat # Wm * m (if m_ref=0) or Wm * (m-m_ref)
                                                                    # Here, current_log_model_flat is deviation from 0 if m_ref=0.
                                                                    # If m_ref is non-zero (log_reference_model_flat), then:
                                                                    # spatial_model_diff = self.Wm @ (current_log_model_flat - log_reference_model_flat)
                                                                    # Assuming m_ref = 0 for simplicity here based on `mr.T @ Wm.T @ Wm @ mr` form.
                                                                    # The code uses `mr` (current model) directly in regularization terms.
                                                                    # This means reference model is implicitly zero in log space (i.e. 1 ohm.m linear).
                                                                    # This needs to be consistent with how `mr_R` was used in single ERT.
                                                                    # For time-lapse, often m_ref is previous timestep or initial model.
                                                                    # The code uses `mr` in `fmert` and `ftert` directly.
                                                                    # This means it regularizes ||m|| and ||Δm_temporal||.
                                                                    # Let's assume reference model is zero for now for these terms.
                spatial_weighting_matrix_for_hessian = self.Wm.T @ self.Wm
                spatial_part_of_gradient = current_spatial_lambda * spatial_weighting_matrix_for_hessian @ current_log_model_flat

                # Temporal model term components:
                temporal_model_diff = self.Wt @ current_log_model_flat # Wt * m
                temporal_weighting_matrix_for_hessian = self.Wt.T @ self.Wt
                temporal_part_of_gradient = current_temporal_alpha * temporal_weighting_matrix_for_hessian @ current_log_model_flat

                if inversion_norm_type == 'L2':
                    current_data_misfit_term = float(data_residual_log.T @ data_weighting_matrix_for_hessian @ data_residual_log)
                    current_spatial_reg_term = float(current_spatial_lambda * (current_log_model_flat.T @ spatial_weighting_matrix_for_hessian @ current_log_model_flat))
                    current_temporal_reg_term = float(current_temporal_alpha * (current_log_model_flat.T @ temporal_weighting_matrix_for_hessian @ current_log_model_flat))

                elif inversion_norm_type == 'L1':
                    # IRLS weights for L1 norm. Rd for data, Rs for spatial, Rt for temporal.
                    Rd_irls = diags(1.0 / np.sqrt(data_residual_log.flatten()**2 + l1_epsilon_irls))
                    Rs_irls = diags(1.0 / np.sqrt(spatial_model_diff.flatten()**2 + l1_epsilon_irls))
                    Rt_irls = diags(1.0 / np.sqrt(temporal_model_diff.flatten()**2 + l1_epsilon_irls))
                    
                    # Update matrices for Hessian and gradient.
                    data_weighting_matrix_for_hessian = self.Wd.T @ Rd_irls @ self.Wd
                    spatial_weighting_matrix_for_hessian = self.Wm.T @ Rs_irls @ self.Wm
                    temporal_weighting_matrix_for_hessian = self.Wt.T @ Rt_irls @ self.Wt
                    
                    # Recalculate misfit terms and gradient parts with IRLS weights.
                    current_data_misfit_term = float(data_residual_log.T @ data_weighting_matrix_for_hessian @ data_residual_log)
                    current_spatial_reg_term = float(current_spatial_lambda * (spatial_model_diff.T @ Rs_irls @ spatial_model_diff))
                    current_temporal_reg_term = float(current_temporal_alpha * (temporal_model_diff.T @ Rt_irls @ temporal_model_diff))
                    
                    data_part_of_gradient = jacobian_log_log_block_diag.T @ data_weighting_matrix_for_hessian @ (-data_residual_log)
                    spatial_part_of_gradient = current_spatial_lambda * spatial_weighting_matrix_for_hessian @ current_log_model_flat
                    temporal_part_of_gradient = current_temporal_alpha * temporal_weighting_matrix_for_hessian @ current_log_model_flat
                    
                else:  # L1L2 hybrid norm
                    # Hybrid L1-L2 weights for data misfit term (Rd_hybrid).
                    # Epsilon for stability, potentially annealed.
                    effective_l1_epsilon = l1_epsilon_irls * (1 + 10 * np.exp(-gn_iter_count / 5.0)) # Annealing epsilon
                    hybrid_data_weights = []
                    for val_abs_resid in np.abs(data_residual_log.flatten()):
                        norm_val = val_abs_resid / np.sqrt(effective_l1_epsilon)
                        if norm_val > l1l2_hybrid_threshold_c: # L1-like behavior for large residuals
                            hybrid_data_weights.append(l1l2_hybrid_threshold_c / norm_val)
                        else: # L2-like behavior for small residuals
                            hybrid_data_weights.append(1.0)
                    Rd_hybrid = diags(hybrid_data_weights)
                    
                    # Model and temporal terms use pure L1 weights (Rs_l1, Rt_l1).
                    model_weights_l1 = 1.0 / np.sqrt(spatial_model_diff.flatten()**2 + l1_epsilon_irls)
                    model_weights_l1 = np.maximum(model_weights_l1, 1e-10) # Floor to avoid extreme weights.
                    Rs_l1 = diags(model_weights_l1)
                    
                    temporal_weights_l1 = 1.0 / np.sqrt(temporal_model_diff.flatten()**2 + l1_epsilon_irls)
                    temporal_weights_l1 = np.maximum(temporal_weights_l1, 1e-10) # Floor.
                    Rt_l1 = diags(temporal_weights_l1)

                    # Update matrices for Hessian and gradient using hybrid/L1 weights.
                    data_weighting_matrix_for_hessian = self.Wd.T @ Rd_hybrid @ self.Wd
                    spatial_weighting_matrix_for_hessian = self.Wm.T @ Rs_l1 @ self.Wm
                    temporal_weighting_matrix_for_hessian = self.Wt.T @ Rt_l1 @ self.Wt
                    
                    # Recalculate misfit terms and gradient parts.
                    current_data_misfit_term = float(data_residual_log.T @ data_weighting_matrix_for_hessian @ data_residual_log)
                    current_spatial_reg_term = float(current_spatial_lambda * (spatial_model_diff.T @ Rs_l1 @ spatial_model_diff))
                    current_temporal_reg_term = float(current_temporal_alpha * (temporal_model_diff.T @ Rt_l1 @ temporal_model_diff))

                    data_part_of_gradient = jacobian_log_log_block_diag.T @ data_weighting_matrix_for_hessian @ (-data_residual_log)
                    spatial_part_of_gradient = current_spatial_lambda * spatial_weighting_matrix_for_hessian @ current_log_model_flat
                    temporal_part_of_gradient = current_temporal_alpha * temporal_weighting_matrix_for_hessian @ current_log_model_flat

                # --- Total Gradient and Objective Function ---
                # Gradient of the objective function: ∇Φ = J^T Wd^T Wd (Jm - d) + λ Wm^T Wm m + α Wt^T Wt m
                # (assuming m_ref = 0 for regularization terms for now)
                total_gradient = data_part_of_gradient + spatial_part_of_gradient + temporal_part_of_gradient

                # Total objective function value: Φ = Φ_data + Φ_spatial_model + Φ_temporal_model
                current_total_objective = current_data_misfit_term + current_spatial_reg_term + current_temporal_reg_term

                # --- Chi-Squared and Convergence Check ---
                # Normalized data misfit (chi-squared for L2, pseudo-chi-squared for L1/L1L2).
                # For L2, this is (d-Gm)^T Wd^T Wd (d-Gm) / N_data.
                # For L1/L1L2, the "chi2" here is based on the L2 measure of misfit, not the L1 objective part.
                chi2_current_iter = float(data_residual_log.T @ self.Wd.T @ self.Wd @ data_residual_log) / len(log_predicted_data_stacked)
                
                relative_chi2_change = abs(chi2_current_iter - previous_chi2) / previous_chi2 if gn_iter_count > 0 else 1.0
                previous_chi2 = chi2_current_iter # Update for next iteration.
                
                print(f'ERT chi2: {chi2_current_iter:.4f}')
                print(f'dPhi (relative chi2 change): {relative_chi2_change:.4f}')
                print(f'Objective terms: data={current_data_misfit_term:.2f}, spatial_reg={current_spatial_reg_term:.2f}, temporal_reg={current_temporal_reg_term:.2f}')
                
                iteration_stats_list.append([chi2_current_iter, current_spatial_reg_term, current_temporal_reg_term]) # Store stats.
                
                # Convergence criteria for Gauss-Newton loop:
                # Target chi2 (e.g., 1.0-1.5) or small change in chi2.
                # `nn > 5` condition for dPhi seems to be `gn_iter_count > 5`.
                if (chi2_current_iter < 1.5) or (relative_chi2_change < 0.01 and gn_iter_count > 5):
                    print(f"Gauss-Newton converged at iteration {gn_iter_count}.")
                    break # Exit GN loop.
                
                # --- Approximate Hessian Matrix (H) ---
                # H ≈ J^T Wd_eff^T Wd_eff J + λ Wm_eff^T Wm_eff + α Wt_eff^T Wt_eff
                # Wd_eff, Wm_eff, Wt_eff are effective/IRLS weights depending on norm type.
                # J is jacobian_log_log_block_diag.
                # Wm is self.Wm, Wt is self.Wt.
                # Lambda is current_spatial_lambda, Alpha is current_temporal_alpha.
                
                # For L2: H_L2 = J^T(Wd^T Wd)J + λ(Wm^T Wm) + α(Wt^T Wt)
                # For L1 (IRLS): H_L1 = J^T(Wd^T Rd Wd)J + λ(Wm^T Rs Wm) + α(Wt^T Rt Wt)
                # For L1L2 (Hybrid): H_Hybrid = J^T(Wd^T Rd_hybrid Wd)J + λ(Wm^T Rs_l1 Wm) + α(Wt^T Rt_l1 Wt) + small_diag_loading
                
                # `data_weighting_matrix_for_hessian` already contains (Wd^T R_eff Wd)
                # `spatial_weighting_matrix_for_hessian` already contains (Wm^T R_eff Wm)
                # `temporal_weighting_matrix_for_hessian` already contains (Wt^T R_eff Wt)
                
                hessian_approx = (jacobian_log_log_block_diag.T @ data_weighting_matrix_for_hessian @ jacobian_log_log_block_diag +
                                 current_spatial_lambda * spatial_weighting_matrix_for_hessian +
                                 current_temporal_alpha * temporal_weighting_matrix_for_hessian)

                if inversion_norm_type == 'L1L2': # Add small diagonal damping for L1L2 hybrid
                    hessian_approx += l1_epsilon_irls * np.eye(jacobian_log_log_block_diag.shape[1])

                # Delete large Jacobian matrix to free memory if possible (though it's used in gradient calc for line search).
                # `del jacobian_log_log_block_diag` # If not needed later in loop/line search.
                                                   # It *is* needed if gradient is re-evaluated in line search.
                                                   # The provided line search re-evaluates forward model, not full gradient.

                # --- Solve for Model Update (Δm) ---
                # Solve H Δm = -∇Φ
                # `total_gradient` is ∇Φ.
                model_update_log = generalized_solver(
                    hessian_approx, -total_gradient.ravel(), # ravel gradient to 1D
                    method=self.parameters['method'],
                    use_gpu=self.parameters.get('use_gpu', False),
                    parallel=self.parameters.get('parallel', False),
                    n_jobs=self.parameters.get('n_jobs', -1)
                ).reshape(-1, 1) # Reshape solution back to column vector.
                
                # --- Line Search ---
                # Find optimal step length `mu_LS` to ensure decrease in objective function.
                # Objective function at current model `current_log_model_flat` is `current_total_objective`.
                step_length_mu = 1.0 # Start with full step.
                line_search_succeeded = False
                best_log_model_in_ls = current_log_model_flat.copy() # Fallback if line search fails.
                best_objective_in_ls = current_total_objective
                
                # Line search for L1L2 is simplified in original code (accepts full step after clipping).
                if inversion_norm_type == 'L1L2':
                    temp_next_log_model = current_log_model_flat + model_update_log
                    temp_next_log_model = np.clip(temp_next_log_model, min_log_rho, max_log_rho) # Apply bounds
                    best_log_model_in_ls = temp_next_log_model
                    line_search_succeeded = True # Effectively, no line search, just take the step and clip.
                else: # Standard backtracking line search for L1 and L2.
                    for ls_iter_count in range(20): # Max 20 line search steps.
                        # Tentative model: m_k + mu * Δm_k
                        temp_next_log_model = current_log_model_flat + step_length_mu * model_update_log
                        # Apply model constraints (bounds).
                        temp_next_log_model = np.clip(temp_next_log_model, min_log_rho, max_log_rho)
                        
                        try:
                            # Calculate objective function for this tentative model.
                            # Requires forward modeling.
                            next_log_predicted_data = _calculate_forward(
                                self.fwd_operators, temp_next_log_model, self.mesh, self.size
                            ).reshape(-1, 1)
                            next_data_residual_log = self.rhos1 - next_log_predicted_data
                            
                            # Calculate new objective function terms based on norm type.
                            # This reuses Rd, Rs, Rt from the main loop's current IRLS state for L1.
                            if inversion_norm_type == 'L2':
                                next_data_misfit = float(next_data_residual_log.T @ (self.Wd.T @ self.Wd) @ next_data_residual_log)
                                next_spatial_reg = float(current_spatial_lambda * (temp_next_log_model.T @ (self.Wm.T @ self.Wm) @ temp_next_log_model))
                                next_temporal_reg = float(current_temporal_alpha * (temp_next_log_model.T @ (self.Wt.T @ self.Wt) @ temp_next_log_model))
                            else:  # L1 (uses Rd, Rs, Rt from current IRLS iteration)
                                next_data_misfit = float(next_data_residual_log.T @ (self.Wd.T @ Rd_irls @ self.Wd) @ next_data_residual_log)
                                next_spatial_diff = self.Wm @ temp_next_log_model
                                next_spatial_reg = float(current_spatial_lambda * (next_spatial_diff.T @ Rs_irls @ next_spatial_diff))
                                next_temporal_diff = self.Wt @ temp_next_log_model
                                next_temporal_reg = float(current_temporal_alpha * (next_temporal_diff.T @ Rt_irls @ next_temporal_diff))
                            
                            next_objective_value_ls = next_data_misfit + next_spatial_reg + next_temporal_reg
                            
                            # Armijo condition (simplified: check for sufficient decrease)
                            # f(m_k + mu*p_k) <= f(m_k) + c1 * mu * grad_f_k^T p_k
                            # Here, p_k = model_update_log, grad_f_k = total_gradient.
                            # `current_total_objective` is f(m_k).
                            # The original code had `fgoal = fc_r - 1e-4 * mu_LS * (d_mr.T.dot(gc_r1_reshaped))`.
                            # `gc_r1_reshaped` was an alternative gradient. Using `total_gradient`.
                            # Need `model_update_log` to be 1D for dot product with 1D gradient for `armijo_rhs_term`.
                            armijo_rhs_term = (total_gradient.ravel().T @ model_update_log.ravel()) # g^T p
                            if armijo_rhs_term > 0: # Should be <0 for descent direction. If >0, something is wrong (e.g. H not pos-def).
                                print("Warning: Line search direction is not a descent direction. Using full step with caution.")
                                # armijo_target = current_total_objective # Effectively disable Armijo check, just ensure reduction.
                                armijo_target = current_total_objective + 1e-4 * step_length_mu * armijo_rhs_term # Still use for consistency if it happens
                            else:
                                armijo_target = current_total_objective + 1e-4 * step_length_mu * armijo_rhs_term

                            if next_objective_value_ls < armijo_target: # Sufficient decrease.
                                best_objective_in_ls = next_objective_value_ls
                                best_log_model_in_ls = temp_next_log_model.copy()
                                line_search_succeeded = True
                                print(f"Line search success at step {ls_iter_count+1} with mu={step_length_mu:.3e}, obj={best_objective_in_ls:.2e}")
                                break # Exit line search.
                                
                        except Exception as e_ls: # Catch errors during forward model in line search.
                            print(f"Line search iteration {ls_iter_count+1} with mu={step_length_mu:.3e} failed: {str(e_ls)}")
                        
                        step_length_mu *= 0.5 # Reduce step length.
                
                # --- Update Model ---
                if line_search_succeeded:
                    current_log_model_flat = best_log_model_in_ls
                    # Update lambda (spatial regularization strength) if line search was good.
                    # This is a simple cooling schedule for lambda.
                    if current_spatial_lambda > self.parameters['lambda_min']:
                        current_spatial_lambda *= self.parameters['lambda_rate']
                        current_spatial_lambda = max(current_spatial_lambda, self.parameters['lambda_min']) # Ensure not below min.
                        print(f"Spatial lambda updated to: {current_spatial_lambda:.2f}")
                else: # Line search failed to find a better model.
                    print('Line search failed. Using conservative step along negative gradient.')
                    # Take a very small, conservative step along the negative gradient direction.
                    # This is a fallback to ensure some progress or to escape a problematic region.
                    # Norm of gradient for scaling:
                    norm_total_gradient = np.linalg.norm(total_gradient)
                    if norm_total_gradient > 1e-12: # Avoid division by zero if gradient is effectively zero.
                         current_log_model_flat = current_log_model_flat - 0.01 * (total_gradient / norm_total_gradient)
                    # Apply bounds after this fallback step.
                    current_log_model_flat = np.clip(current_log_model_flat, min_log_rho, max_log_rho)
            
            # --- IRLS Convergence Check (for L1 or L1L2 norms) ---
            if inversion_norm_type in ['L1', 'L1L2'] and irls_iter_count > 0:
                # Calculate relative change in the model vector from previous IRLS iteration.
                model_change_irls = np.linalg.norm(current_log_model_flat - log_model_previous_irls) / np.linalg.norm(log_model_previous_irls)
                print(f"IRLS relative model change: {model_change_irls:.4e}")
                # Stop IRLS if model change is small or if data fit (chi2) is already good.
                if model_change_irls < irls_convergence_tolerance or chi2_current_iter < 1.5:
                    print(f"IRLS converged after {irls_iter_count + 1} iterations.")
                    break # Exit IRLS loop.
            
            if inversion_norm_type in ['L1', 'L1L2']:
                log_model_previous_irls = current_log_model_flat.copy() # Store for next IRLS change calculation.

        # --- Final Processing and Storing Results ---
        # Reshape the final flat log-model vector back to (num_cells, num_timesteps) Fortran order.
        final_log_model_reshaped = np.reshape(current_log_model_flat, (-1, self.size), order='F')
        # Convert back to linear resistivity space.
        final_linear_model_timesteps = np.exp(final_log_model_reshaped)

        # Compute coverage for a representative timestep (e.g., middle one).
        # This provides an indication of spatial resolution for one of the timeframes.
        # SUGGESTION: Coverage might change over time if acquisition geometry changes.
        # Could compute and store for all, or allow user to specify which timestep.
        mid_timestep_idx = self.size // 2 # Integer division for middle index.
        # Get the linear model for this middle timestep.
        model_for_coverage_linear = pg.Vector(final_linear_model_timesteps[:, mid_timestep_idx])

        # Calculate response and Jacobian for this specific timestep's model.
        fwd_op_mid = self.fwd_operators[mid_timestep_idx]
        response_mid_linear = fwd_op_mid.response(model_for_coverage_linear)
        fwd_op_mid.createJacobian(model_for_coverage_linear)
        jacobian_mid_gimli = fwd_op_mid.jacobian()

        # Calculate coverage using PyGIMLi's utility.
        # Weights 1/response and 1/model are typical for log-transform style sensitivity.
        coverage_values_mid_log10 = pg.core.coverageDCtrans(
            jacobian_mid_gimli,
            1.0 / response_mid_linear,         # Weight by 1/data
            1.0 / model_for_coverage_linear  # Weight by 1/model
        )
        
        # Normalize coverage by cell sizes and take log10.
        # Same logic as in ERTInversion.get_coverage().
        param_domain_mid = fwd_op_mid.paraDomain()
        param_sizes_mid = np.zeros(model_for_coverage_linear.size())
        if param_domain_mid.cellCount() == model_for_coverage_linear.size():
             for i in range(model_for_coverage_linear.size()):
                param_sizes_mid[i] = param_domain_mid.cell(i).size()
        else: # Fallback using markers, with warning
            print("Warning: Mismatch in cell count for coverage cell size calculation (mid-timestep).")
            for c_cell in param_domain_mid.cells():
                if 0 <= c_cell.marker() < len(param_sizes_mid):
                    param_sizes_mid[c_cell.marker()] += c_cell.size()

        # Avoid division by zero for param_sizes_mid.
        coverage_norm_mid = np.full_like(coverage_values_mid_log10.array(), np.nan)
        valid_idx_cov = param_sizes_mid > 1e-9
        coverage_norm_mid[valid_idx_cov] = coverage_values_mid_log10.array()[valid_idx_cov] / param_sizes_mid[valid_idx_cov]

        # Log10 transform, handling non-positives carefully.
        final_coverage_log10 = np.full_like(coverage_norm_mid, np.nan)
        min_pos_cov = np.min(coverage_norm_mid[coverage_norm_mid > 0]) if np.any(coverage_norm_mid > 0) else 1e-12
        floor_val_cov = min_pos_cov * 1e-6

        idx_pos_cov = coverage_norm_mid > floor_val_cov
        final_coverage_log10[idx_pos_cov] = np.log10(coverage_norm_mid[idx_pos_cov])
        final_coverage_log10[(~idx_pos_cov) & (~np.isnan(coverage_norm_mid))] = np.log10(floor_val_cov)

        # Store results in the TimeLapseInversionResult object.
        result.final_models = final_linear_model_timesteps # Shape (n_cells, n_timesteps)
        # Store the single coverage map (e.g., for middle timestep) in all_coverage, or could be list of coverages.
        # Original code copies it for each timestep.
        result.all_coverage = [final_coverage_log10.copy()] * self.size
        result.mesh = param_domain_mid # Store the mesh used.
        result.all_chi2 = np.array(iteration_stats_list) # Store [chi2, spatial_reg, temporal_reg] for each GN iter.

        print('End of time-lapse inversion.')
        return result
