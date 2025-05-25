"""
Time-lapse ERT inversion functionality.
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from scipy.sparse import diags, csr_matrix # Though csr_matrix is not explicitly used, it's related to sparse operations
from scipy.linalg import block_diag # Corrected import for block_diag
from typing import Optional, Union, List, Dict, Any, Tuple


from .base import InversionBase, TimeLapseInversionResult
from ..forward.ert_forward import ertforward2, ertforandjac2
from ..solvers.solver import generalized_solver # Corrected import path


def _calculate_jacobian(fwd_operators: List[ert.ERTModelling], model: np.ndarray, mesh: pg.Mesh, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the combined Jacobian matrix for a multi-time (time-lapse) model.

    This is an internal helper function. It iterates through each timestep,
    calculates the Jacobian for that individual timestep's model slice,
    and then assembles these individual Jacobians into a larger block diagonal matrix.
    The forward responses for each timestep are also calculated and stacked.
    
    Args:
        fwd_operators (List[ert.ERTModelling]): List of PyGIMLi ERTModelling forward operators,
                                               one for each timestep.
        model (np.ndarray): The current model parameters, a flattened 1D array
                            representing log-transformed resistivities for all timesteps,
                            concatenated (e.g., [m_t1, m_t2, ..., m_tn]).
        mesh (pg.Mesh): The PyGIMLi mesh used for the forward modeling. Assumed to be
                        the same for all timesteps.
        size (int): The number of timesteps in the time-lapse sequence.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - obs_stacked (np.ndarray): Stacked (concatenated) log-transformed observed data
                                        (forward responses) for all timesteps. Shape (N_total_data, 1).
            - J_block_diag (np.ndarray): The combined block diagonal Jacobian matrix.
                                         Each block is the Jacobian for one timestep: d(log_data_i)/d(log_model_i).
                                         Shape (N_total_data, N_total_model_params).
    """
    # Reshape the flattened model vector into a 2D array (cells_per_timestep, num_timesteps)
    # 'F' (Fortran-like) order means model parameters for timestep 0 are in column 0, etc.
    model_reshaped_time_columns = np.reshape(model, (-1, size), order='F')
    
    obs_list = []  # To store observed data (forward responses) for each timestep
    jacobian_blocks = []  # To store Jacobian matrices for each timestep

    for i in range(size):  # Iterate through each timestep
        # Calculate forward response (dr_i) and Jacobian (Jr_i) for the i-th timestep's model slice.
        # ertforandjac2 expects log-model and returns log-data and d(log-data)/d(log-model) Jacobian.
        dr_i, Jr_i = ertforandjac2(fwd_operators[i], model_reshaped_time_columns[:, i], mesh)
        obs_list.append(dr_i)
        jacobian_blocks.append(Jr_i)
        
    # Create a block diagonal matrix from the individual Jacobian blocks.
    # This structure arises because data from timestep `i` only depends on model parameters from timestep `i`.
    J_block_diag = block_diag(*jacobian_blocks)  # Uses scipy.linalg.block_diag
    
    # Stack the observed data (forward responses) from all timesteps into a single column vector.
    obs_stacked = np.concatenate([obs.reshape(-1, 1) for obs in obs_list])
    
    return obs_stacked, J_block_diag


def _calculate_forward(fwd_operators: List[ert.ERTModelling], model: np.ndarray, mesh: pg.Mesh, size: int) -> np.ndarray:
    """
    Calculate the forward response for a multi-time (time-lapse) model and stack them.

    This is an internal helper function. It iterates through each timestep,
    calculates the forward response for that individual timestep's model slice using
    `ertforward2` (which implies log-transformed inputs/outputs), and then
    stacks these responses into a single column vector.
    
    Args:
        fwd_operators (List[ert.ERTModelling]): List of PyGIMLi ERTModelling forward operators.
        model (np.ndarray): The current model parameters (flattened 1D array of log-transformed resistivities).
        mesh (pg.Mesh): The PyGIMLi mesh.
        size (int): The number of timesteps.
        
    Returns:
        np.ndarray: Stacked (concatenated) log-transformed observed data (forward responses)
                    for all timesteps. Shape (N_total_data, 1).
    """
    # Reshape model to (cells_per_timestep, num_timesteps)
    model_reshaped_time_columns = np.reshape(model, (-1, size), order='F')
    obs_list = []  # To store responses for each timestep
    
    for i in range(size):  # Iterate through each timestep
        # ertforward2 expects log-model and returns log-data
        dr_i = ertforward2(fwd_operators[i], model_reshaped_time_columns[:, i], mesh)
        obs_list.append(dr_i)
    
    # Stack the observed data from all timesteps into a single column vector
    obs_stacked = np.concatenate([obs.reshape(-1, 1) for obs in obs_list])
    
    return obs_stacked


def _calculate_forward_separate(fwd_operators: List[ert.ERTModelling], model: np.ndarray, mesh: pg.Mesh, size: int) -> List[np.ndarray]:
    """
    Calculate forward responses for a multi-time model, returning separate responses for each timestep.

    This is an internal helper function. It iterates through each timestep
    and calculates the forward response for that timestep's model slice using `ertforward2`,
    returning a list of these (log-transformed) responses.
    
    Args:
        fwd_operators (List[ert.ERTModelling]): List of PyGIMLi ERTModelling forward operators.
        model (np.ndarray): The current model parameters (flattened 1D array of log-transformed resistivities).
        mesh (pg.Mesh): The PyGIMLi mesh.
        size (int): The number of timesteps.
        
    Returns:
        List[np.ndarray]: A list where each element is a NumPy array of log-transformed
                          observed data (forward response) for a single timestep.
    """
    # Reshape model to (cells_per_timestep, num_timesteps)
    model_reshaped_time_columns = np.reshape(model, (-1, size), order='F')
    obs_list = []  # To store responses for each timestep
    
    for i in range(size):  # Iterate through each timestep
        # ertforward2 expects log-model and returns log-data
        dr_i = ertforward2(fwd_operators[i], model_reshaped_time_columns[:, i], mesh)
        obs_list.append(dr_i)  # Append individual timestep response
    
    return obs_list


class TimeLapseERTInversion(InversionBase):
    """Time-lapse ERT inversion class."""
    
    def __init__(self, data_files: List[str], measurement_times: List[float],
                mesh: Optional[pg.Mesh] = None, **kwargs):
        """
        Initialize time-lapse ERT inversion.
        
        Args:
            data_files (List[str]): List of paths to ERT data files, one for each timestep.
            measurement_times (List[float]): List of measurement times corresponding to each data file.
            mesh (Optional[pg.Mesh], optional): PyGIMLi mesh for inversion. If None, it will be
                                                created during setup based on the first data file.
                                                Defaults to None.
            **kwargs: Additional parameters for the inversion process, including:
                - `lambda_val` (float): Spatial regularization parameter.
                - `alpha` (float): Temporal regularization parameter.
                - `decay_rate` (float): Decay rate for temporal regularization weights.
                - `method` (str): Solver method for the linear system (e.g., 'cgls', 'lsqr').
                - `model_constraints` (Tuple[float, float]): Min and max physical model parameter bounds (e.g., resistivity).
                - `max_iterations` (int): Maximum number of Gauss-Newton iterations per IRLS step.
                - `absoluteUError` (float): Absolute data error component.
                - `relativeError` (float): Relative data error component.
                - `lambda_rate` (float): Rate at which lambda is reduced during iterations.
                - `lambda_min` (float): Minimum lambda value.
                - `inversion_type` (str): Type of norm for regularization ('L1', 'L2', 'L1L2').
        """
        # Load ERT data
        self.data_files = data_files
        self.measurement_times = np.array(measurement_times)
        
        # Validate input
        if len(data_files) != len(measurement_times):
            raise ValueError("Number of data files must match number of measurement times")
        
        # Load first dataset to initialize base class (data attribute of base class)
        # This `self.data` will hold the first timestep's data for convenience (e.g. mesh creation if not provided)
        data = ert.load(data_files[0])
        
        # Call parent initializer with first dataset
        super().__init__(data, mesh, **kwargs)
        
        # Set time-lapse specific default parameters
        tl_defaults = {
            'lambda_val': 100.0,  # Spatial regularization
            'alpha': 10.0,        # Temporal regularization
            'decay_rate': 0.0,    # For temporal weighting, if used
            'method': 'cgls',     # Linear solver method
            'absoluteUError': 0.0, # Absolute data error
            'relativeError': 0.05,# Relative data error
            'lambda_rate': 0.8,   # Lambda reduction factor per iteration
            'lambda_min': 1.0,    # Minimum lambda
            'inversion_type': 'L2',  # Norm type: 'L1', 'L2', or 'L1L2' (hybrid)
            'model_constraints':(1e-4, 1e4),  # Min and max physical resistivity values
            'tolerance': 0.01, # Convergence tolerance for dPhi (relative change in chi-squared)
            # max_iterations is inherited from InversionBase default_parameters
        }
        
        # Update parameters with time-lapse defaults if not provided in kwargs
        for key, value in tl_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Number of timesteps
        self.size = len(data_files)
        
        # Initialize internal variables specific to time-lapse
        self.fwd_operators = [] # List to store forward operator for each timestep
        self.datasets = []      # List to store loaded data for each timestep
        self.rhos1 = None       # Stacked log-transformed observed apparent resistivities
        self.Wd = None          # Combined data weighting matrix (inverse of data error)
        self.Wm = None          # Combined spatial model regularization matrix
        self.Wt = None          # Combined temporal model regularization matrix
    
    def setup(self):
        """Set up time-lapse ERT inversion (load all data, create operators, regularization matrices)."""
        # Create mesh if not provided (based on the first data file used in super().__init__)
        if self.mesh is None:
            ert_manager = ert.ERTManager(self.data) # self.data is from the first timestep
            self.mesh = ert_manager.createMesh(data=self.data, quality=34) # Default quality
        
        # Load all datasets and process them
        rhos_all_timesteps = [] # List to store physical apparent resistivities for each timestep
        dataerr_all_timesteps = [] # List to store data errors for each timestep
        k_factors_first = [] # To store geometric factors from the first dataset if needed by others
        
        for i, fname in enumerate(self.data_files):
            # Load individual ERT data file
            data_i = ert.load(fname)
            self.datasets.append(data_i) # Store the loaded DataContainer
            
            # Handle geometric factors (k)
            # If 'k' is all zeros, try to compute it. If already computed for first dataset, reuse.
            if np.all(data_i['k'] == 0.0):
                if not k_factors_first: # If k_factors_first is empty (first time)
                    data_i['k'] = ert.createGeometricFactors(data_i, numerical=True)
                    k_factors_first = data_i['k'].array() # Store for potential reuse
                else:
                    data_i['k'] = k_factors_first # Reuse k from first dataset
            
            # Get apparent resistivity (rhoa)
            # If 'rhoa' is not present or all zeros, calculate from resistance 'r' and 'k'
            if 'rhoa' in data_i.standardTokens() and np.any(data_i['rhoa'] != 0.0):
                rhos_all_timesteps.append(data_i['rhoa'].array())
            elif 'r' in data_i.standardTokens() and 'k' in data_i.standardTokens():
                rhos_all_timesteps.append(data_i['r'].array() * data_i['k'].array())
            else:
                raise ValueError(f"Could not determine apparent resistivity for data file: {fname}")

            # Get or estimate data errors ('err')
            if 'err' in data_i.standardTokens() and np.any(data_i['err'] != 0.0):
                dataerr_all_timesteps.append(data_i['err'].array())
            else: # Estimate error if not present
                ert_manager_i = ert.ERTManager(data_i)
                data_i['err'] = ert_manager_i.estimateError(
                    data_i,
                    absoluteUError=self.parameters['absoluteUError'],
                    relativeError=self.parameters['relativeError']
                )
                dataerr_all_timesteps.append(data_i['err'].array())
            
            # Create and configure forward operator for this timestep
            fwd_operator_i = ert.ERTModelling()
            fwd_operator_i.setData(data_i) # Set specific data scheme for this operator
            fwd_operator_i.setMesh(self.mesh) # Use the common mesh
            self.fwd_operators.append(fwd_operator_i)
        
        # Stack all observed apparent resistivities and log-transform them
        rhos_stacked_physical = np.concatenate(rhos_all_timesteps)
        self.rhos1 = np.log(rhos_stacked_physical.reshape(-1, 1)) # Ensure column vector

        # Data error weighting matrix (Wd)
        # Inverse of log-transformed relative errors (1 / log(1 + rel_err))
        dataerr_stacked = np.concatenate(dataerr_all_timesteps)
        # Add small epsilon to avoid log(0) if error is exactly 0, though relativeError usually prevents this.
        self.Wd = np.diag(1.0 / (np.log(dataerr_stacked + 1.0) + 1e-9)) 
        
        # Spatial model regularization matrix (Wm)
        # Based on the first timestep's forward operator (assuming mesh and regions are consistent)
        rm = self.fwd_operators[0].regionManager()
        rm.setConstraintType(1) # First-order smoothness constraints
        Ctmp_spatial = pg.matrix.RSparseMapMatrix()
        rm.fillConstraints(Ctmp_spatial)
        Wm_r_single_timestep = pg.utils.sparseMatrix2coo(Ctmp_spatial)
        constraint_weights_spatial = rm.constraintWeights().array()
        Wm_r_single_timestep = diags(constraint_weights_spatial).dot(Wm_r_single_timestep)
        Wm_r_single_timestep = Wm_r_single_timestep.todense() # Convert to dense for block_diag if it's small enough
        
        # Create block diagonal spatial regularization matrix for all timesteps
        self.Wm = block_diag(*[Wm_r_single_timestep for _ in range(self.size)])
        
        # Temporal model regularization matrix (Wt)
        # Penalizes differences between model parameters at consecutive timesteps.
        num_cells = self.fwd_operators[0].paraDomain().cellCount()
        time_differences = np.diff(self.measurement_times)
        
        # Calculate temporal weights (w_temp) based on decay rate and time differences
        temporal_weights = []
        if self.size > 1:
            for i in range(self.size - 1):
                weight = np.exp(-self.parameters['decay_rate'] * time_differences[i])
                temporal_weights.extend([weight] * num_cells)
        
        # Construct Wt matrix: Wt * m = [m_t1-m_t2; m_t2-m_t3; ...]
        # It has shape (num_cells * (num_timesteps - 1), num_cells * num_timesteps)
        if self.size > 1:
            Wt_sparse = np.zeros((num_cells * (self.size - 1), num_cells * self.size))
            for i in range(self.size - 1): # For each time interface
                row_start_idx = i * num_cells
                # Entries for m_ti
                Wt_sparse[row_start_idx : row_start_idx + num_cells, 
                          i * num_cells : (i + 1) * num_cells] = np.eye(num_cells)
                # Entries for -m_t{i+1}
                Wt_sparse[row_start_idx : row_start_idx + num_cells, 
                          (i + 1) * num_cells : (i + 2) * num_cells] = -np.eye(num_cells)
            self.Wt = diags(temporal_weights).dot(Wt_sparse) # Apply weights
        else: # Single timestep, no temporal regularization
            self.Wt = np.zeros((0, num_cells)) # Empty matrix

    def run(self, initial_model: Optional[np.ndarray] = None) -> TimeLapseInversionResult:
        """
        Run time-lapse ERT inversion using Gauss-Newton framework,
        with options for L1, L2, or L1-L2 hybrid norms for regularization.

        The inversion minimizes an objective function typically of the form:
        Phi(m) = Phi_d(m) + lambda * Phi_m_spatial(m) + alpha * Phi_m_temporal(m)
        where:
        - Phi_d is the data misfit term.
        - Phi_m_spatial is the spatial regularization term (smoothness within each time's model).
        - Phi_m_temporal is the temporal regularization term (smoothness/coupling between time steps).
        - lambda and alpha are regularization parameters.
        - m is the concatenated model vector for all timesteps [m_t1, m_t2, ..., m_tn].

        For L1 or L1L2 norms, an Iteratively Reweighted Least Squares (IRLS) approach is used,
        which involves an outer loop for updating weights and an inner Gauss-Newton loop.
        
        Args:
            initial_model (Optional[np.ndarray]): Initial model parameters.
                If 2D (n_cells, n_timesteps), it's used directly (after log-transform).
                If None, a homogeneous model is created based on the median of observed
                apparent resistivities for each timestep. Defaults to None.
            
        Returns:
            TimeLapseInversionResult: Object containing the inversion results, including
                                      final models for all timesteps, coverage, mesh, and
                                      iteration statistics.
        """
        # Ensure setup has been called (loads data, creates matrices and operators)
        if not self.fwd_operators:
            self.setup()
        
        # Initialize result object to store inversion outputs
        result = TimeLapseInversionResult()
        result.timesteps = self.measurement_times # Store measurement times
        
        num_cells_per_timestep = self.fwd_operators[0].paraDomain().cellCount()
        
        # Set up initial model (mr, log-transformed) if not provided
        if initial_model is None:
            initial_rhos_per_timestep = []
            for i in range(self.size): # For each timestep
                # Use median of observed apparent resistivity for this timestep as initial guess
                # Check if 'rhoa' exists and has non-zero values
                if 'rhoa' in self.datasets[i].standardTokens() and np.any(self.datasets[i]['rhoa'] > 0):
                    median_rhoa_i = np.median(self.datasets[i]['rhoa'].array())
                else: # Fallback if 'rhoa' is missing or zero
                    median_rhoa_i = 100.0 # Default homogeneous resistivity
                initial_rhos_per_timestep.append(median_rhoa_i)
            
            # Create the full initial model: repeat median for each cell, then log-transform
            # Result `mr` is a flat column vector [log(m_t1), log(m_t2), ...]
            mr = np.log(np.repeat(initial_rhos_per_timestep, num_cells_per_timestep).reshape(-1, 1))
        else: # Use provided initial model
            if initial_model.shape != (num_cells_per_timestep, self.size):
                raise ValueError(f"Provided initial_model shape {initial_model.shape} "
                                 f"does not match expected ({num_cells_per_timestep}, {self.size})")
            # Flatten in column-major ('F') order to get [m_t1, m_t2, ...] and log-transform
            mr = np.log(initial_model.flatten(order='F').reshape(-1, 1))
        
        # Reference model for regularization (m_ref). Here, it's the initial model.
        mr_R = mr.copy()
        
        # Regularization parameters
        Lambda_spatial = self.parameters['lambda_val'] # Spatial regularization weight
        alpha_temporal = self.parameters['alpha']      # Temporal regularization weight
        
        # Model constraints (min/max physical resistivity values, converted to log space)
        min_resistivity_phys, max_resistivity_phys = self.parameters['model_constraints']
        min_mr_log = np.log(min_resistivity_phys) # Minimum log-resistivity
        max_mr_log = np.log(max_resistivity_phys) # Maximum log-resistivity

        Err_tot = [] # To store [chi2, spatial_reg_term, temporal_reg_term] for each iteration
        chi2_old = np.inf # Initialize old chi-squared for dPhi calculation
        
        inversion_type = self.parameters['inversion_type'].upper()
        
        # Parameters for L1/L1L2 norms (IRLS)
        if inversion_type in ['L1', 'L1L2']:
            l1_epsilon = 1e-4 # Small constant to stabilize IRLS weights
            # Max IRLS iterations: fewer for pure L1, more for L1L2 which might need more adjustment
            irls_iter_max = 5 if inversion_type == 'L1' else 8 
            irls_tol = 1e-3 if inversion_type == 'L1' else 1e-2 # Tolerance for IRLS convergence (model change)
            threshold_c_hybrid = 2.0  # Threshold for Huber-like weighting in L1L2 data misfit
            mr_previous_irls = mr.copy() # Store model for IRLS convergence check

        # --- Start of Iteratively Reweighted Least Squares (IRLS) Loop ---
        # (This loop runs once if inversion_type is 'L2')
        for irls_iter in range(1 if inversion_type == 'L2' else irls_iter_max):
            if inversion_type in ['L1', 'L1L2']:
                print(f'------------------- IRLS Iteration: {irls_iter + 1} ---------------------------')
            
            # --- Start of Gauss-Newton (Inner) Inversion Loop ---
            # TODO: Consider making print statements conditional via a `verbose` parameter in self.parameters
            for nn in range(self.parameters['max_iterations']):
                print(f'-------------------ERT (Gauss-Newton) Iteration: {nn} ---------------------------')
                
                # Calculate Jacobian (Jr) and predicted data (dr) for the current model (mr)
                dr, Jr = _calculate_jacobian(self.fwd_operators, mr, self.mesh, self.size)
                dr = dr.reshape(-1, 1) # Ensure dr is a column vector (log-transformed predicted data)
                
                # Data misfit vector (d_obs - d_pred, in log space)
                dataerror_log = self.rhos1 - dr # self.rhos1 is log(observed_data)
                
                # --- Objective Function and Gradient Calculation (depends on norm type) ---
                if inversion_type == 'L2':
                    # L2-norm objective function terms
                    phi_d_L2 = float(dataerror_log.T @ self.Wd.T @ self.Wd @ dataerror_log)
                    phi_m_spatial_L2 = float(Lambda_spatial * (mr.T @ self.Wm.T @ self.Wm @ mr))
                    phi_m_temporal_L2 = float(alpha_temporal * (mr.T @ self.Wt.T @ self.Wt @ mr))
                    
                    # Gradient for L2-norm objective function
                    grad_phi_d_L2 = Jr.T @ self.Wd.T @ self.Wd @ dataerror_log * -1 
                    grad_phi_m_spatial_L2 = Lambda_spatial * self.Wm.T @ self.Wm @ mr
                    grad_phi_m_temporal_L2 = alpha_temporal * self.Wt.T @ self.Wt @ mr
                        
                elif inversion_type == 'L1': # Pure L1 norm (IRLS)
                    # Weighting matrix Rd for data misfit (|| Wd * (d_obs - d_pred) ||_1)
                    Rd_L1 = diags(1.0 / np.sqrt(dataerror_log.flatten()**2 + l1_epsilon))
                    # Weighting matrix Rs for spatial regularization (|| Wm * m ||_1)
                    model_spatial_diff_L1 = self.Wm @ mr
                    Rs_L1 = diags(1.0 / np.sqrt(model_spatial_diff_L1.flatten()**2 + l1_epsilon))
                    # Weighting matrix Rt for temporal regularization (|| Wt * m ||_1)
                    model_temporal_diff_L1 = self.Wt @ mr
                    Rt_L1 = diags(1.0 / np.sqrt(model_temporal_diff_L1.flatten()**2 + l1_epsilon))
                    
                    # Weighted L1-norm approximations for objective function terms
                    phi_d_L2 = float(dataerror_log.T @ (self.Wd.T @ Rd_L1 @ self.Wd) @ dataerror_log) # Weighted L2 approx of L1
                    phi_m_spatial_L2 = float(Lambda_spatial * (model_spatial_diff_L1.T @ Rs_L1 @ model_spatial_diff_L1))
                    phi_m_temporal_L2 = float(alpha_temporal * (model_temporal_diff_L1.T @ Rt_L1 @ model_temporal_diff_L1))
                    
                    # Gradient for IRLS L1-norm objective function
                    grad_phi_d_L2 = Jr.T @ self.Wd.T @ Rd_L1 @ self.Wd @ dataerror_log * -1
                    grad_phi_m_spatial_L2 = Lambda_spatial * self.Wm.T @ Rs_L1 @ (self.Wm @ mr)
                    grad_phi_m_temporal_L2 = alpha_temporal * self.Wt.T @ Rt_L1 @ (self.Wt @ mr)
                    
                else:  # L1L2 hybrid norm
                    # Data misfit: Hybrid L1-L2 weights (Huber-like)
                    effective_epsilon_data_hybrid = l1_epsilon * (1 + 10*np.exp(-nn/5)) # Epsilon can vary with GN iteration
                    data_weights_hybrid_vals = []
                    for val_de in dataerror_log.flatten():
                        norm_val_de = np.abs(val_de) / np.sqrt(effective_epsilon_data_hybrid)
                        data_weights_hybrid_vals.append(threshold_c_hybrid / norm_val_de if norm_val_de > threshold_c_hybrid else 1.0)
                    Rd_hybrid = diags(data_weights_hybrid_vals)
                    
                    # Model and temporal regularization: Pure L1 weights
                    model_spatial_diff_hybrid = self.Wm @ mr
                    spatial_weights_L1_hybrid = 1.0 / np.sqrt(model_spatial_diff_hybrid.flatten()**2 + l1_epsilon)
                    spatial_weights_L1_hybrid = np.maximum(spatial_weights_L1_hybrid, 1e-10) # Stability
                    Rs_L1_hybrid = diags(spatial_weights_L1_hybrid)
                    
                    model_temporal_diff_hybrid = self.Wt @ mr
                    temporal_weights_L1_hybrid = 1.0 / np.sqrt(model_temporal_diff_hybrid.flatten()**2 + l1_epsilon)
                    temporal_weights_L1_hybrid = np.maximum(temporal_weights_L1_hybrid, 1e-10)
                    Rt_L1_hybrid = diags(temporal_weights_L1_hybrid)
                    
                    # Objective function terms for L1L2 hybrid
                    phi_d_L2 = float(dataerror_log.T @ (self.Wd.T @ Rd_hybrid @ self.Wd) @ dataerror_log)
                    phi_m_spatial_L2 = float(Lambda_spatial * (model_spatial_diff_hybrid.T @ Rs_L1_hybrid @ model_spatial_diff_hybrid))
                    phi_m_temporal_L2 = float(alpha_temporal * (model_temporal_diff_hybrid.T @ Rt_L1_hybrid @ model_temporal_diff_hybrid))
                    
                    # Gradient for L1L2 hybrid norm
                    grad_phi_d_L2 = Jr.T @ self.Wd.T @ Rd_hybrid @ self.Wd @ dataerror_log * -1
                    grad_phi_m_spatial_L2 = Lambda_spatial * self.Wm.T @ Rs_L1_hybrid @ (self.Wm @ mr)
                    grad_phi_m_temporal_L2 = alpha_temporal * self.Wt.T @ Rt_L1_hybrid @ (self.Wt @ mr)
                
                # Total gradient of the objective function
                gradient_total_objective = grad_phi_d_L2 + grad_phi_m_spatial_L2 + grad_phi_m_temporal_L2
                
                # Total objective function value (sum of weighted L2 norms)
                objective_function_total = phi_d_L2 + phi_m_spatial_L2 + phi_m_temporal_L2
                
                # Chi-squared (based on unweighted L2 norm of data misfit for consistent reporting)
                chi2_current_iter = float(dataerror_log.T @ self.Wd.T @ self.Wd @ dataerror_log) / len(dr)
                dPhi_chi2_relative_change = abs(chi2_current_iter - chi2_old) / chi2_old if nn > 0 and chi2_old > 1e-9 else 1.0
                chi2_old = chi2_current_iter # Update for next iteration's dPhi calculation
                
                print(f'ERT chi2: {chi2_current_iter}')
                print(f'dPhi (chi2 relative change): {dPhi_chi2_relative_change}')
                print(f'Objective Function Terms: DataMisfit(weighted)={phi_d_L2}, SpatialReg(weighted)={phi_m_spatial_L2}, TemporalReg(weighted)={phi_m_temporal_L2}, TotalObjective={objective_function_total}')
                
                # Store iteration statistics
                Err_tot.append([chi2_current_iter, phi_m_spatial_L2, phi_m_temporal_L2])
                
                # Check for Gauss-Newton convergence
                if (chi2_current_iter < self.parameters['min_chi2']) or \
                   (dPhi_chi2_relative_change < self.parameters['tolerance'] and nn > 5): # Allow some iterations before dPhi check
                    print(f"Gauss-Newton convergence reached at iteration {nn}")
                    break
                
                # --- Assemble Approximate Hessian (H_approx) and Solve Linear System ---
                if inversion_type == 'L2':
                    H_approx = (Jr.T @ self.Wd.T @ self.Wd @ Jr + 
                                Lambda_spatial * self.Wm.T @ self.Wm + 
                                alpha_temporal * self.Wt.T @ self.Wt)
                elif inversion_type == 'L1': # IRLS Hessian for L1
                    H_approx = (Jr.T @ self.Wd.T @ Rd_L1 @ self.Wd @ Jr + 
                                Lambda_spatial * self.Wm.T @ Rs_L1 @ self.Wm + 
                                alpha_temporal * self.Wt.T @ Rt_L1 @ self.Wt)
                else:  # L1L2 hybrid
                    H_approx = (Jr.T @ self.Wd.T @ Rd_hybrid @ self.Wd @ Jr + 
                                Lambda_spatial * self.Wm.T @ Rs_L1_hybrid @ self.Wm + 
                                alpha_temporal * self.Wt.T @ Rt_L1_hybrid @ self.Wt + 
                                l1_epsilon * np.eye(Jr.shape[1])) # Damping for stability
                
                del Jr  # Free Jacobian memory

                # Solve for model update step (d_mr): H_approx * d_mr = -gradient_total_objective
                d_mr_step = generalized_solver(
                    H_approx, -gradient_total_objective, 
                    method=self.parameters['method'],
                    use_gpu=self.parameters.get('use_gpu', False),
                    parallel=self.parameters.get('parallel', False),
                    n_jobs=self.parameters.get('n_jobs', -1)
                )
                d_mr_step = d_mr_step.reshape(-1, 1) # Ensure column vector
                
                # --- Line Search ---
                mu_LS = 1.0  # Initial step length
                line_search_success = False
                mr_current_iter_start = mr.copy() # Model at start of this GN iteration
                objective_current_iter_start = objective_function_total # Objective at start of this GN iteration
                
                if inversion_type == 'L1L2': # Simpler step for L1L2 (could be more sophisticated)
                    mr_candidate = mr + d_mr_step
                    mr_candidate = np.clip(mr_candidate, min_mr_log, max_mr_log)
                    line_search_success = True # Or re-evaluate objective if needed
                else: # Armijo-type line search for L2 and L1
                    for iarm in range(20): # Max line search iterations
                        mr_candidate = mr_current_iter_start + mu_LS * d_mr_step
                        mr_candidate = np.clip(mr_candidate, min_mr_log, max_mr_log)
                        
                        try:
                            dr_candidate = _calculate_forward(self.fwd_operators, mr_candidate, self.mesh, self.size)
                            dr_candidate = dr_candidate.reshape(-1, 1)
                            dataerror_candidate = self.rhos1 - dr_candidate
                            
                            # Objective function terms for candidate model
                            if inversion_type == 'L2':
                                fd_cand = float(dataerror_candidate.T @ self.Wd.T @ self.Wd @ dataerror_candidate)
                                fm_s_cand = float(Lambda_spatial * (mr_candidate.T @ self.Wm.T @ self.Wm @ mr_candidate))
                                fm_t_cand = float(alpha_temporal * (mr_candidate.T @ self.Wt.T @ self.Wt @ mr_candidate))
                            else:  # L1 (using weights Rd_L1, Rs_L1, Rt_L1 from current IRLS iteration)
                                fd_cand = float(dataerror_candidate.T @ (self.Wd.T @ Rd_L1 @ self.Wd) @ dataerror_candidate)
                                m_s_diff_cand = self.Wm @ mr_candidate
                                fm_s_cand = float(Lambda_spatial * (m_s_diff_cand.T @ Rs_L1 @ m_s_diff_cand))
                                m_t_diff_cand = self.Wt @ mr_candidate
                                fm_t_cand = float(alpha_temporal * (m_t_diff_cand.T @ Rt_L1 @ m_t_diff_cand))
                            
                            objective_candidate = fd_cand + fm_s_cand + fm_t_cand
                            
                            # Armijo condition for sufficient decrease
                            # Term: c1 * mu_LS * gradient^T * search_direction
                            # Here, search_direction is d_mr_step, gradient is gradient_total_objective
                            # For descent, grad^T * step should be negative.
                            armijo_check_term = 1e-4 * mu_LS * (gradient_total_objective.T @ d_mr_step)[0,0]
                            
                            if objective_candidate < objective_current_iter_start + armijo_check_term:
                                mr = mr_candidate # Update model
                                line_search_success = True
                                break # Exit line search
                                
                        except Exception as e: # Catch errors during forward modeling in line search
                            print(f"Line search iteration {iarm} with mu_LS={mu_LS} failed: {str(e)}")
                        
                        mu_LS *= 0.5 # Reduce step size
                
                # Post-line search model update
                if line_search_success:
                    if inversion_type == 'L1L2': # If L1L2, mr was already updated with full step
                        mr = mr_candidate 
                    # For L2/L1, mr was updated inside loop if successful
                    # Lambda reduction strategy (typically for L2 or when line search is consistently successful)
                    if Lambda_spatial > self.parameters['lambda_min'] and inversion_type == 'L2':
                        Lambda_spatial *= self.parameters['lambda_rate']
                        if Lambda_spatial < self.parameters['lambda_min']: Lambda_spatial = self.parameters['lambda_min']
                else: # Line search failed
                    print("Line search failed to improve objective function. Model not updated in this GN iteration.")
                    # TODO: Consider alternative strategies: e.g., stop, increase lambda, or very small step.
                    # For now, model `mr` from start of this GN iter is retained if L2/L1.
            
            # --- End of Gauss-Newton Iteration nn ---

            # Check IRLS convergence (if applicable)
            if inversion_type in ['L1', 'L1L2'] and irls_iter > 0 :
                # Relative change in model parameters from previous IRLS iteration
                irls_model_change = np.linalg.norm(mr - mr_previous_irls) / (np.linalg.norm(mr_previous_irls) + 1e-9)
                print(f"IRLS relative model change: {irls_model_change}")
                if irls_model_change < irls_tol or chi2_current_iter < self.parameters['min_chi2']:
                    print(f"IRLS converged after {irls_iter + 1} iterations.")
                    break # Exit IRLS loop
            
            if inversion_type in ['L1', 'L1L2']:
                mr_previous_irls = mr.copy() # Store model for next IRLS iteration
        
        # --- End of IRLS Loop ---
        
        # Final model processing:
        # Reshape final log-model `mr` to (num_cells_per_timestep, num_timesteps)
        final_model_log = np.reshape(mr, (-1, self.size), order='F')
        # Convert to physical scale (resistivity)
        final_model_physical = np.exp(final_model_log)
        
        # Compute coverage for a representative timestep (e.g., middle one)
        # This gives a general idea of spatial resolution.
        representative_timestep_idx = self.size // 2
        model_for_coverage_calc = pg.Vector(final_model_physical[:, representative_timestep_idx])
        fwd_op_for_coverage_calc = self.fwd_operators[representative_timestep_idx]
        
        response_for_coverage_calc = fwd_op_for_coverage_calc.response(model_for_coverage_calc)
        fwd_op_for_coverage_calc.createJacobian(model_for_coverage_calc)
        jacobian_for_coverage_calc = fwd_op_for_coverage_calc.jacobian()
        
        # Calculate raw coverage values
        coverage_values_raw = pg.core.coverageDCtrans(
            jacobian_for_coverage_calc, 
            1.0 / response_for_coverage_calc, 
            1.0 / model_for_coverage_calc
        )
        
        # Weight coverage by cell sizes
        parameter_domain_mesh_final = fwd_op_for_coverage_calc.paraDomain()
        num_cells_final = parameter_domain_mesh_final.cellCount()
        
        if len(model_for_coverage_calc) != num_cells_final:
            raise ValueError(f"Model size for coverage ({len(model_for_coverage_calc)}) "
                             f"mismatches cell count ({num_cells_final}).")

        cell_sizes_final = np.array([cell.size() for cell in parameter_domain_mesh_final.cells()])
        cell_sizes_final[cell_sizes_final == 0] = 1e-12 # Avoid division by zero

        if coverage_values_raw.size() != num_cells_final:
             raise ValueError(f"Raw coverage size ({coverage_values_raw.size()}) "
                              f"mismatches cell count ({num_cells_final}).")

        # Final coverage (log10 of sensitivity density per cell)
        # Note: Original `paramSizes[c.marker()]` implies region-based, but model `mr` is cell-based.
        # Correcting to use individual cell_sizes for cell-based model coverage.
        final_coverage_log10 = np.log10(coverage_values_raw.array() / cell_sizes_final)
        
        # Store results
        result.final_models = final_model_physical # Shape: (num_cells, num_timesteps)
        # Store the representative coverage for all timesteps.
        # For a more detailed analysis, coverage could be computed for each timestep if data/schemes vary.
        result.all_coverage = [final_coverage_log10.copy() for _ in range(self.size)] 
        result.mesh = parameter_domain_mesh_final # Store the mesh used for inversion
        result.all_chi2 = Err_tot # List of [chi2, spatial_reg_term, temporal_reg_term] per iteration
        
        print('End of inversion')
        return result

[end of PyHydroGeophysX/inversion/time_lapse.py]
