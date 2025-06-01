"""
Time-lapse Electrical Resistivity Tomography (ERT) inversion functionality.

This module provides the `TimeLapseERTInversion` class for performing
time-lapse ERT inversions. It builds upon the base inversion framework and
incorporates temporal regularization to link inversions across different
timesteps. Helper functions for Jacobian calculation and forward modeling
in a time-lapse context are also included.
"""
import os # For path joining
import numpy as np
import pygimli as pg
from pygimli.physics import ert # ERT specific PyGIMLi tools
from scipy.sparse import diags, csr_matrix, block_diag as sparse_block_diag # For sparse matrices
# from scipy.sparse.linalg import lsqr # LSQR solver, if used directly. Not used in current code.
from typing import Optional, Union, List, Dict, Any, Tuple 
# from scipy.linalg import block_diag # For dense block_diag, sparse_block_diag is better if matrices are sparse.


from .base import InversionBase, TimeLapseInversionResult # Base classes
# Need to import ERTInversion to use its get_coverage method if it's not static or refactored.
from .ert_inversion import ERTInversion # For potential use of its methods like get_coverage
from ..forward.ert_forward import ertforward2, ertforandjac2 # Forward/Jacobian helpers from forward modeling module
from ..solvers.linear_solvers import generalized_solver # Custom or wrapped linear system solver.
# Note: Original path was ..solvers.solver. Corrected to ..solvers.linear_solvers based on typical structure.


# Helper functions for time-lapse operations (prefixed with underscore for internal use)
# These functions seem tailored for log-transformed models and data.
def _calculate_jacobian(fwd_operators: List[ert.ERTModelling],
                        model_all_times_log: np.ndarray, # Combined model vector (log-transformed)
                        mesh: pg.Mesh, # Assumed common mesh for all time steps
                        num_timesteps: int
                       ) -> Tuple[np.ndarray, csr_matrix]: # Jacobian returned as sparse
    """
    Calculate the combined Jacobian matrix for a multi-timestep (time-lapse) model.

    The input `model_all_times_log` is assumed to be log-transformed and flattened.
    It's reshaped to (n_cells, num_timesteps). The Jacobian for each timestep
    (J_i = ∂log(d_i)/∂log(ρ_i)) is computed using `ertforandjac2` and then
    assembled into a block diagonal sparse matrix, J_total = diag(J_1, J_2, ..., J_N).
    Log-transformed responses from each timestep are stacked vertically.

    Args:
        fwd_operators (List[ert.ERTModelling]): List of initialized PyGIMLi ERTModelling
                                                operators, one for each timestep. Each should
                                                have its respective data scheme set.
        model_all_times_log (np.ndarray): A 1D array of log-transformed resistivity model
                                          parameters for all cells and all timesteps, typically
                                          ordered (Fortran-style): [log(ρ_t1_c1)...log(ρ_t1_cN), log(ρ_t2_c1)...].
        mesh (pg.Mesh): The PyGIMLi mesh used for the inversion (assumed common for all timesteps).
        num_timesteps (int): The number of timesteps in the time-lapse sequence.

    Returns:
        Tuple[np.ndarray, csr_matrix]:
            - stacked_log_responses (np.ndarray): Stacked log-transformed forward responses from all
                                                 timesteps, as a column vector.
            - J_block_diag_sparse (csr_matrix): The block diagonal Jacobian matrix (log-log sensitivities).
    
    Raises:
        ValueError: If model_all_times_log cannot be reshaped as expected.
    """
    num_cells = mesh.cellCount()
    expected_size = num_cells * num_timesteps
    if model_all_times_log.size != expected_size:
        raise ValueError(f"Model size ({model_all_times_log.size}) does not match expected "
                         f"num_cells ({num_cells}) * num_timesteps ({num_timesteps}) = {expected_size}.")

    # Reshape flat model vector to (num_cells, num_timesteps) using Fortran order.
    # Each column model_reshaped_log[:, i] is the log-model for timestep i.
    model_reshaped_log = np.reshape(model_all_times_log, (num_cells, num_timesteps), order='F')
    
    all_responses_log: List[np.ndarray] = []
    jacobian_blocks_csr_list: List[csr_matrix] = [] # Store individual Jacobians as CSR
    
    for i in range(num_timesteps):
        current_log_model_i = model_reshaped_log[:, i] # Log-model for current timestep i
        # `ertforandjac2` returns (log_response, log_log_Jacobian)
        log_dr_i, Jr_i_loglog = ertforandjac2(fwd_operators[i], current_log_model_i, mesh)
        
        all_responses_log.append(log_dr_i)
        # Ensure Jacobian is in sparse CSR format for efficient block diagonal construction
        if isinstance(Jr_i_loglog, np.ndarray):
            jacobian_blocks_csr_list.append(csr_matrix(Jr_i_loglog))
        elif isinstance(Jr_i_loglog, csr_matrix): # Already sparse
            jacobian_blocks_csr_list.append(Jr_i_loglog)
        else: # Should ideally be ndarray or csr_matrix from ertforandjac2
            # This case might occur if ertforandjac2 returns other sparse types or dense GMat-like objects
            # that are not directly ndarray or csr_matrix. Forcing to CSR.
            print(f"Warning: Jacobian for timestep {i} is of unexpected type {type(Jr_i_loglog)}. Converting to CSR.")
            jacobian_blocks_csr_list.append(csr_matrix(np.asarray(Jr_i_loglog))) 
    
    # Stack all log-transformed responses into a single column vector
    stacked_log_responses = np.concatenate([resp.reshape(-1, 1) for resp in all_responses_log], axis=0)
    
    # Create a block diagonal sparse matrix for the combined Jacobian
    J_block_diag_sparse = sparse_block_diag(jacobian_blocks_csr_list, format='csr')
    
    return stacked_log_responses, J_block_diag_sparse


def _calculate_forward(fwd_operators: List[ert.ERTModelling],
                       model_all_times_log: np.ndarray, 
                       mesh: pg.Mesh,
                       num_timesteps: int
                      ) -> np.ndarray:
    """
    Calculate the stacked log-transformed forward responses for a multi-timestep model.

    Args:
        fwd_operators (List[ert.ERTModelling]): List of ERTModelling operators.
        model_all_times_log (np.ndarray): 1D array of log-transformed model parameters for all timesteps.
        mesh (pg.Mesh): The PyGIMLi mesh.
        num_timesteps (int): Number of timesteps.

    Returns:
        np.ndarray: Stacked log-transformed forward responses as a column vector.
    """
    num_cells = mesh.cellCount()
    if model_all_times_log.size != num_cells * num_timesteps: # Check size consistency
        raise ValueError("Model size does not match num_cells * num_timesteps in _calculate_forward.")

    model_reshaped_log = np.reshape(model_all_times_log, (num_cells, num_timesteps), order='F')
    all_log_responses_list: List[np.ndarray] = []
    
    for i in range(num_timesteps):
        current_log_model_i = model_reshaped_log[:, i]
        # `ertforward2` computes log-transformed response from log-transformed model
        log_dr_i = ertforward2(fwd_operators[i], current_log_model_i, mesh)
        all_log_responses_list.append(log_dr_i)
    
    # Stack all log-transformed responses
    stacked_log_responses = np.concatenate([resp.reshape(-1, 1) for resp in all_log_responses_list], axis=0)
    
    return stacked_log_responses


def _calculate_forward_separate(fwd_operators: List[ert.ERTModelling],
                                model_all_times_log: np.ndarray, 
                                mesh: pg.Mesh,
                                num_timesteps: int
                               ) -> List[np.ndarray]:
    """
    Calculate forward responses for each timestep separately in a time-lapse model.
    Returns a list of response arrays, not stacked.

    Args:
        fwd_operators (List[ert.ERTModelling]): List of ERTModelling operators.
        model_all_times_log (np.ndarray): 1D array of log-transformed model parameters for all timesteps.
        mesh (pg.Mesh): The PyGIMLi mesh.
        num_timesteps (int): Number of timesteps.

    Returns:
        List[np.ndarray]: A list where each element is the log-transformed forward response
                          for a single timestep.
    """
    num_cells = mesh.cellCount()
    if model_all_times_log.size != num_cells * num_timesteps: # Check size consistency
        raise ValueError("Model size does not match num_cells * num_timesteps in _calculate_forward_separate.")

    model_reshaped_log = np.reshape(model_all_times_log, (num_cells, num_timesteps), order='F')
    all_log_responses_separate_list: List[np.ndarray] = []
    
    for i in range(num_timesteps):
        current_log_model_i = model_reshaped_log[:, i]
        log_dr_i = ertforward2(fwd_operators[i], current_log_model_i, mesh)
        all_log_responses_separate_list.append(log_dr_i) # Keep as separate arrays
    
    return all_log_responses_separate_list


class TimeLapseERTInversion(InversionBase):
    """
    Performs time-lapse Electrical Resistivity Tomography (ERT) inversion.

    This class extends `InversionBase` to handle multiple ERT datasets collected
    over time. It incorporates temporal regularization to link the inversions of
    consecutive timesteps, promoting smoother changes or enforcing specific
    temporal behaviors in the resistivity models.
    """
    
    def __init__(self, data_files: List[str], 
                 measurement_times: Union[List[float], np.ndarray], 
                 mesh: Optional[pg.Mesh] = None, **kwargs: Any):
        """
        Initialize the TimeLapseERTInversion class.

        Args:
            data_files (List[str]): A list of file paths, where each path points to an
                                    ERT data file for a specific timestep.
            measurement_times (Union[List[float], np.ndarray]): A list or NumPy array of time values
                                                               (e.g., hours, days) corresponding to each data file.
                                                               Must be the same length as `data_files`.
            mesh (Optional[pg.Mesh], optional): A PyGIMLi mesh for the inversion. If None,
                                                a mesh is created based on the first dataset
                                                during `setup()`. Defaults to None.
            **kwargs (Any): Additional keyword arguments for configuring the inversion.
                            These override defaults in `InversionBase` and time-lapse specific defaults.
                            Key time-lapse parameters include:
                            - `alpha` (float): Temporal regularization strength.
                            - `decay_rate` (float): Decay rate for temporal weights (exp(-decay*dt)).
                            - `inversion_type` (str): Norm for misfit ('L1', 'L2', 'L1L2').
        
        Raises:
            ValueError: If `data_files` is empty or if lengths of `data_files` and `measurement_times` mismatch.
            RuntimeError: If loading the first data file fails.
        """
        if not data_files:
            raise ValueError("data_files list cannot be empty.")
        if len(data_files) != len(measurement_times):
            raise ValueError("The number of data files must match the number of measurement times.")
        
        self.data_files = data_files
        self.measurement_times = np.array(measurement_times, dtype=float) # Ensure NumPy array
        
        # Load the first dataset to initialize the InversionBase class.
        try:
            first_data = ert.load(data_files[0])
        except Exception as e:
            raise RuntimeError(f"Failed to load the first ERT data file '{data_files[0]}': {e}")
        
        super().__init__(first_data, mesh, **kwargs) # `self.data` will hold first_data initially
        
        # Time-lapse specific default parameters
        timelapse_defaults = {
            'lambda_val': 100.0,       # Spatial regularization strength (can be overridden)
            'alpha': 10.0,             # Temporal regularization strength
            'decay_rate': 0.0,         # Decay rate for temporal weights (0 means constant weights, i.e., simple difference)
            'inversion_type': 'L2',    # Norm for data misfit and regularization ('L1', 'L2', or 'L1L2')
            # Parameters inherited from InversionBase (via kwargs or its defaults) will be used if not set here.
            # e.g. 'method', 'absoluteUError', 'relativeError', 'lambda_rate', 'lambda_min', 'model_constraints'.
            # We ensure these are present in self.parameters by merging with InversionBase's defaults.
        }
        
        # Update self.parameters with time-lapse specific defaults, only if not already set by user via kwargs
        # or by InversionBase defaults (if that had more specific ones).
        for key, default_value in timelapse_defaults.items():
            if key not in self.parameters: 
                self.parameters[key] = default_value
        
        self.num_timesteps = len(data_files) 
        
        # Initialize attributes for time-lapse specific components
        self.fwd_operators: List[ert.ERTModelling] = [] 
        self.datasets: List[pg.DataContainer] = []      
        self.log_rhoa_obs_stacked: Optional[np.ndarray] = None 
        self.Wd_stacked: Optional[csr_matrix] = None  # Combined data weighting matrix (sparse)
        self.Wm_stacked: Optional[csr_matrix] = None  # Combined spatial regularization matrix (sparse)
        self.Wt_temporal: Optional[csr_matrix] = None # Temporal regularization matrix (sparse)
        
        # Result object should be TimeLapseInversionResult
        self.result = TimeLapseInversionResult() # Overwrites InversionBase's result object
        self.result.meta['inversion_type'] = f"TimeLapseERT_{self.parameters['inversion_type']}"
        self.result.meta['data_files'] = self.data_files
        self.result.meta['measurement_times'] = self.measurement_times
        self.result.meta.update(self.parameters) # Store all effective parameters used for this inversion
    
    def setup(self) -> None:
        """
        Set up the time-lapse ERT inversion environment.

        This involves:
        - Creating/validating the common inversion mesh.
        - Loading all time-lapse datasets and their error models.
        - Initializing a PyGIMLi `ERTModelling` forward operator for each dataset.
        - Preparing observed data (stacking log-transformed apparent resistivities).
        - Constructing combined data weighting matrix (`self.Wd_stacked`).
        - Constructing combined spatial regularization matrix (`self.Wm_stacked`).
        - Constructing temporal regularization matrix (`self.Wt_temporal`).
        """
        super().setup() # Handles mesh creation if self.mesh is None, and stores basic meta

        if self.mesh is None: 
             raise RuntimeError("Mesh was not created or provided during setup process.")

        all_log_rhoa_obs_list: List[np.ndarray] = []
        all_data_weights_list: List[np.ndarray] = [] 
        
        first_k_factors: Optional[np.ndarray] = None # Store k-factors from first dataset

        for i, data_filepath in enumerate(self.data_files):
            try:
                current_data = ert.load(data_filepath)
                self.datasets.append(current_data)
            except Exception as e:
                raise RuntimeError(f"Failed to load ERT data from '{data_filepath}' for timestep {i}: {e}")

            # Handle geometric factors ('k')
            # If 'k' is missing or all zeros, try to use from first dataset or recalculate.
            if 'k' not in current_data or np.all(np.asarray(current_data['k']) == 0.0):
                if first_k_factors is not None and len(first_k_factors) == current_data.size():
                    current_data['k'] = first_k_factors
                else:
                    if first_k_factors is not None: # Mismatch case
                        print(f"Warning: Geometric factor 'k' length from first dataset ({len(first_k_factors)}) "
                              f"does not match current dataset size ({current_data.size()}) for timestep {i}. "
                              "Recalculating 'k' for current dataset. Ensure this is intended.")
                    current_data['k'] = ert.createGeometricFactors(current_data, numerical=True)
                if i == 0: first_k_factors = np.asarray(current_data['k'])
            
            # Get apparent resistivity (rhoa)
            k_vals_curr = np.asarray(current_data['k']) if 'k' in current_data else None # Ensure k_vals_curr is defined
            # Potential Issue: if 'k' was not in current_data and not derivable, k_vals_curr is None.

            if 'rhoa' in current_data and np.any(np.asarray(current_data['rhoa']) != 0.0):
                rhoa_i_linear = np.asarray(current_data['rhoa'])
            elif 'r' in current_data and k_vals_curr is not None:
                rhoa_i_linear = np.asarray(current_data['r']) * k_vals_curr
            else:
                raise ValueError(f"Dataset {data_filepath} (timestep {i}) lacks 'rhoa' or ('r' and 'k') fields.")
            
            if np.any(rhoa_i_linear <= 0):
                print(f"Warning: Timestep {i}: Observed rhoa contains non-positive values. Log transform will have -inf/NaN.")
            with np.errstate(divide='ignore', invalid='ignore'): # Suppress log(0) warnings
                all_log_rhoa_obs_list.append(np.log(rhoa_i_linear))

            # Data errors and weights
            if 'err' in current_data and np.any(np.asarray(current_data['err']) != 0.0):
                relative_errors_i = np.asarray(current_data['err'])
                if np.any(relative_errors_i <= 0):
                    raise ValueError(f"Data 'err' in file '{data_filepath}' has non-positive relative errors.")
                all_data_weights_list.append(1.0 / relative_errors_i)
            else:
                global_rel_err = self.parameters['relativeError']
                if global_rel_err <= 0:
                     print(f"Warning: Global 'relativeError' is non-positive for timestep {i}. Using unit weights (error=1).")
                     all_data_weights_list.append(np.ones(current_data.size()))
                else:
                     all_data_weights_list.append(np.full(current_data.size(), 1.0 / global_rel_err))
            
            fwd_op_i = ert.ERTModelling(data=current_data, mesh=self.mesh, verbose=False)
            self.fwd_operators.append(fwd_op_i)
        
        self.log_rhoa_obs_stacked = np.concatenate(all_log_rhoa_obs_list).reshape(-1, 1)
        self.Wd_stacked = diags(np.concatenate(all_data_weights_list), format="csr")
        
        rm_template = self.fwd_operators[0].regionManager()
        if rm_template is None: raise RuntimeError("RegionManager is None in the first forward operator.")
        
        constraints_single_time = pg.matrix.RSparseMapMatrix()
        rm_template.setConstraintType(1) 
        rm_template.fillConstraints(constraints_single_time)
        Wm_single_sparse = pg.utils.sparseMatrix2coo(constraints_single_time).tocsr()
        
        weights_constraints_pg = rm_template.constraintWeights()
        if weights_constraints_pg is not None and weights_constraints_pg.size() > 0:
            weights_np = weights_constraints_pg.array()
            if weights_np.size == Wm_single_sparse.shape[0]:
                Wm_single_sparse = diags(weights_np).dot(Wm_single_sparse)
            else:
                print("Warning: Spatial constraint weights size mismatch for template. Using unweighted.")
        
        self.Wm_stacked = sparse_block_diag([Wm_single_sparse] * self.num_timesteps, format='csr')
        
        num_model_cells = self.mesh.cellCount()
        if self.num_timesteps > 1:
            num_temporal_constraints = num_model_cells * (self.num_timesteps - 1)
            time_diffs_np = np.diff(self.measurement_times)
            if len(time_diffs_np) != self.num_timesteps - 1:
                 raise ValueError("Measurement times issue for temporal differences.")

            temporal_weights_per_link_flat: List[float] = []
            for dt_val_loop in time_diffs_np:
                weight_val = np.exp(-self.parameters['decay_rate'] * dt_val_loop)
                temporal_weights_per_link_flat.extend([weight_val] * num_model_cells)
            
            row_idx_list, col_idx_list, val_list = [], [], []
            current_row_offset = 0 
            for t_loop in range(self.num_timesteps - 1):
                for cell_loop in range(num_model_cells):
                    col_idx_t = t_loop * num_model_cells + cell_loop
                    col_idx_tplus1 = (t_loop + 1) * num_model_cells + cell_loop
                    row_idx_list.extend([current_row_offset, current_row_offset])
                    col_idx_list.extend([col_idx_t, col_idx_tplus1])
                    val_list.extend([1.0, -1.0])
                    current_row_offset += 1
            
            Wt_unweighted = csr_matrix((val_list, (row_idx_list, col_idx_list)),
                                        shape=(num_temporal_constraints, num_model_cells * self.num_timesteps))
            if len(temporal_weights_per_link_flat) == Wt_unweighted.shape[0]:
                 self.Wt_temporal = diags(temporal_weights_per_link_flat, format="csr").dot(Wt_unweighted) 
            else: 
                 self.Wt_temporal = Wt_unweighted 
                 if self.num_timesteps > 1:
                      print(f"Warning: Temporal weights size mismatch. Using unweighted Wt.")
        else: 
            self.Wt_temporal = csr_matrix((0, num_model_cells))

        self.result.meta['spatial_reg_matrix_shape'] = self.Wm_stacked.shape
        self.result.meta['temporal_reg_matrix_shape'] = self.Wt_temporal.shape
    
    def run(self, initial_model_linear_3d: Optional[np.ndarray] = None) -> TimeLapseInversionResult:
        """
        Run the time-lapse ERT inversion.

        Args:
            initial_model_linear_3d (Optional[np.ndarray], optional):
                Initial guess for the resistivity model [ohm·m] (linear scale).
                Should be a 2D array of shape (num_cells, num_timesteps).
                If None, a homogeneous model for each timestep is derived from the median
                of its respective observed apparent resistivities. Defaults to None.

        Returns:
            TimeLapseInversionResult: Object containing the results of the time-lapse inversion.
        
        Raises:
            RuntimeError: If `setup()` has not been called or if inversion encounters critical errors.
            ValueError: If `initial_model_linear_3d` has incorrect shape.
        """
        # Ensure setup has been completed
        if not self.fwd_operators or self.Wd_stacked is None or self.Wm_stacked is None or self.Wt_temporal is None or self.log_rhoa_obs_stacked is None:
            print("Time-lapse inversion setup not complete. Calling setup()...")
            self.setup()
            if not self.fwd_operators: 
                 raise RuntimeError("Setup failed to initialize necessary components for time-lapse inversion.")
        
        if not isinstance(self.result, TimeLapseInversionResult):
             self.result = TimeLapseInversionResult()
        self.result.timesteps = self.measurement_times 
        self.result.meta.update(self.parameters) 

        num_model_cells = self.fwd_operators[0].paraDomain().cellCount()
        
        mr_log: np.ndarray
        if initial_model_linear_3d is None:
            initial_rho_per_timestep = []
            for i in range(self.num_timesteps):
                current_data_rhoa = self.datasets[i]['rhoa'].array()
                valid_rhoa = current_data_rhoa[np.isfinite(current_data_rhoa) & (current_data_rhoa > 0)]
                median_rho_i = np.median(valid_rhoa) if valid_rhoa.size > 0 else 100.0 
                if median_rho_i <= 0 or not np.isfinite(median_rho_i): median_rho_i = 100.0 
                initial_rho_per_timestep.append(np.full(num_model_cells, median_rho_i))
            initial_model_stacked_linear = np.array(initial_rho_per_timestep).T.flatten(order='F')
            mr_log = np.log(initial_model_stacked_linear).reshape(-1, 1)
        else:
            initial_model_np = np.asarray(initial_model_linear_3d)
            if initial_model_np.shape == (num_model_cells, self.num_timesteps):
                mr_log = np.log(np.maximum(initial_model_np.flatten(order='F'), 1e-9)).reshape(-1, 1) 
            elif initial_model_np.size == num_model_cells * self.num_timesteps:
                mr_log = np.log(np.maximum(initial_model_np.ravel(), 1e-9)).reshape(-1, 1)
            else:
                raise ValueError(f"Initial model shape {initial_model_np.shape} is not compatible.")
            if np.any(initial_model_np <=0): 
                 print("Warning: initial_model_linear_3d contains non-positive values. Using log(max(val, 1e-9)).")


        mr_ref_log = mr_log.copy()
        
        current_lambda_spatial = self.parameters['lambda_val'] 
        current_alpha_temporal = self.parameters['alpha']    
        
        min_rho_constr, max_rho_constr = self.parameters['model_constraints']
        if min_rho_constr <= 0 or max_rho_constr <=0 : raise ValueError("Model constraints must be positive.")
        log_min_rho_constr = np.log(min_rho_constr)
        log_max_rho_constr = np.log(max_rho_constr)
        if log_min_rho_constr >= log_max_rho_constr:
            raise ValueError(f"Log-transformed model constraints are invalid.")

        iteration_chi2_history: List[float] = [] 
        chi2_previous_iter = np.inf
        
        inversion_type = self.parameters.get('inversion_type', 'L2').upper()
        irls_max_iter = 1
        if inversion_type in ['L1', 'L1L2']:
            l1_epsilon = self.parameters.get('l1_epsilon', 1e-4) 
            irls_max_iter = self.parameters.get('irls_iter_max', 5 if inversion_type == 'L1' else 8)
            irls_conv_tolerance = self.parameters.get('irls_tol', 1e-3 if inversion_type == 'L1' else 1e-2)
            l1l2_threshold_c = self.parameters.get('l1l2_threshold_c', 2.0)  
        
        mr_log_previous_irls = mr_log.copy() 

        for irls_iter_count in range(irls_max_iter):
            if inversion_type != 'L2':
                print(f"--- IRLS Iteration: {irls_iter_count + 1} / {irls_max_iter} ---")
            
            for iter_count in range(self.parameters['max_iterations']):
                print(f"--- Main Iteration: {iter_count + 1} (IRLS {irls_iter_count + 1}) --- Lambda_sp: {current_lambda_spatial:.2e}, Lambda_t: {current_alpha_temporal:.2e} ---")
                sys.stdout.flush()
                
                log_dr_pred_stacked, J_loglog_block_diag = _calculate_jacobian(
                    self.fwd_operators, mr_log.ravel(), self.mesh, self.num_timesteps
                ) 
                
                residual_log_data = self.log_rhoa_obs_stacked - log_dr_pred_stacked
                
                Rd_irls = diags(np.ones(self.Wd_stacked.shape[0]),format="csr") 
                Rs_irls = diags(np.ones(self.Wm_stacked.shape[0]),format="csr") 
                num_tc_check = self.Wt_temporal.shape[0]
                Rt_irls = diags(np.ones(num_tc_check), format="csr") if num_tc_check > 0 else csr_matrix((0,0)) # Handle empty Wt

                if inversion_type == 'L1':
                    Rd_irls_vals = 1.0 / np.sqrt(residual_log_data.flatten()**2 + l1_epsilon)
                    Rd_irls = diags(Rd_irls_vals, format="csr")
                    model_spatial_diff = self.Wm_stacked @ mr_log 
                    Rs_irls_vals = 1.0 / np.sqrt(model_spatial_diff.flatten()**2 + l1_epsilon)
                    Rs_irls = diags(Rs_irls_vals, format="csr")
                    if num_tc_check > 0:
                        model_temporal_diff = self.Wt_temporal @ mr_log 
                        Rt_irls_vals = 1.0 / np.sqrt(model_temporal_diff.flatten()**2 + l1_epsilon)
                        Rt_irls = diags(Rt_irls_vals, format="csr")

                elif inversion_type == 'L1L2':
                    eff_eps_l1l2 = l1_epsilon * (1 + 10 * np.exp(-iter_count / 5.0)) 
                    data_w_l1l2 = [min(1.0, l1l2_threshold_c / (np.abs(val) / np.sqrt(eff_eps_l1l2) + 1e-9)) 
                                   if np.abs(val) > 1e-9 else 1.0 for val in residual_log_data.flatten()]
                    Rd_irls = diags(data_w_l1l2, format="csr")
                    model_spatial_diff = self.Wm_stacked @ mr_log
                    Rs_irls_vals = 1.0 / np.sqrt(model_spatial_diff.flatten()**2 + l1_epsilon)
                    Rs_irls = diags(Rs_irls_vals, format="csr")
                    if num_tc_check > 0:
                        model_temporal_diff = self.Wt_temporal @ mr_log
                        Rt_irls_vals = 1.0 / np.sqrt(model_temporal_diff.flatten()**2 + l1_epsilon)
                        Rt_irls = diags(Rt_irls_vals, format="csr")

                WdT_eff_Wd_eff = self.Wd_stacked.T @ Rd_irls @ self.Wd_stacked 
                data_misfit_val = float(residual_log_data.T @ WdT_eff_Wd_eff @ residual_log_data)
                
                model_diff_for_reg_spatial = mr_log - mr_ref_log 
                WmT_eff_Wm_eff = self.Wm_stacked.T @ Rs_irls @ self.Wm_stacked
                model_spatial_reg_val = current_lambda_spatial * float(model_diff_for_reg_spatial.T @ WmT_eff_Wm_eff @ model_diff_for_reg_spatial)
                                                                                             
                model_temporal_reg_val = 0.0
                if num_tc_check > 0:
                    WtT_eff_Wt_eff = self.Wt_temporal.T @ Rt_irls @ self.Wt_temporal
                    model_temporal_reg_val = current_alpha_temporal * float((self.Wt_temporal @ mr_log).T @ Rt_irls @ (self.Wt_temporal @ mr_log))

                objective_func_current = data_misfit_val + model_spatial_reg_val + model_temporal_reg_val
                
                chi2_val = float(residual_log_data.T @ (self.Wd_stacked.T @ self.Wd_stacked) @ residual_log_data) / len(self.log_rhoa_obs_stacked)
                dPhi_val = abs(chi2_val - chi2_previous_iter) / chi2_previous_iter if iter_count > 0 and chi2_previous_iter > 1e-9 else 1.0
                
                print(f"ObjFunc: {objective_func_current:.4e}, Chi²: {chi2_val:.4f}, dPhi: {dPhi_val:.4f}")
                print(f"DataMisfit (w): {data_misfit_val:.3e}, SpatialReg (w): {model_spatial_reg_val:.3e}, TemporalReg (w): {model_temporal_reg_val:.3e}")
                
                iteration_chi2_history.append(chi2_val) 
                
                target_chi_sq_param = self.parameters.get('min_chi2', 1.0)
                conv_tolerance_param = self.parameters.get('tolerance', 0.01)
                if chi2_val < target_chi_sq_param :
                    print(f"Convergence: Chi² ({chi2_val:.2f}) < target ({target_chi_sq_param}).")
                    break 
                if dPhi_val < conv_tolerance_param and iter_count > 3 : 
                    print(f"Convergence: Relative Chi² change ({dPhi_val:.4f}) < tolerance ({conv_tolerance_param}).")
                    break
                if iter_count == self.parameters['max_iterations'] - 1:
                    print("Max Gauss-Newton iterations reached.")
                    break
                chi2_previous_iter = chi2_val

                H_data = J_loglog_block_diag.T @ WdT_eff_Wd_eff @ J_loglog_block_diag
                H_model_spatial_eff = current_lambda_spatial * WmT_eff_Wm_eff
                H_model_temporal_eff = 0.0
                if num_tc_check > 0:
                    H_model_temporal_eff = current_alpha_temporal * (self.Wt_temporal.T @ Rt_irls @ self.Wt_temporal)
                
                Hessian_approx_total = H_data + H_model_spatial_eff + H_model_temporal_eff
                if inversion_type == 'L1L2' and isinstance(Hessian_approx_total, csr_matrix): 
                    Hessian_approx_total += l1_epsilon * sparse_block_diag([pg.matrix.Identity(num_model_cells)]*self.num_timesteps, format="csr")
                elif inversion_type == 'L1L2': 
                     Hessian_approx_total += l1_epsilon * np.eye(J_loglog_block_diag.shape[1])

                grad_data = - (J_loglog_block_diag.T @ WdT_eff_Wd_eff @ residual_log_data)
                grad_spatial_reg = current_lambda_spatial * (WmT_eff_Wm_eff @ model_diff_for_reg_spatial)
                grad_temporal_reg = 0.0
                if num_tc_check > 0:
                    grad_temporal_reg = current_alpha_temporal * ((self.Wt_temporal.T @ Rt_irls @ self.Wt_temporal) @ mr_log)
                
                full_gradient = grad_data + grad_spatial_reg + grad_temporal_reg
                
                try:
                    model_update_log = generalized_solver(
                        Hessian_approx_total, -full_gradient.reshape(-1,1), 
                        method=self.parameters['method'],
                        use_gpu=self.parameters.get('use_gpu', False),
                        parallel=self.parameters.get('parallel', False),
                        n_jobs=self.parameters.get('n_jobs', -1)
                    ).reshape(-1,1)
                except Exception as e:
                    print(f"Linear solver failed in main loop (iter {iter_count+1}): {e}. Stopping.")
                    break 

                mu_LS = 1.0
                line_search_succeeded_main = False
                directional_derivative = (full_gradient.T @ model_update_log).item() 

                for ls_main_iter in range(20): 
                    mr_trial_log = mr_log + mu_LS * model_update_log
                    mr_trial_log = np.clip(mr_trial_log, log_min_rho_constr, log_max_rho_constr)
                    
                    try:
                        log_dr_trial_stacked = _calculate_forward(self.fwd_operators, mr_trial_log.ravel(), self.mesh, self.num_timesteps)
                        res_log_trial_stacked = self.log_rhoa_obs_stacked - log_dr_trial_stacked
                        
                        data_misfit_trial = float(res_log_trial_stacked.T @ WdT_eff_Wd_eff @ res_log_trial_stacked)
                        model_spatial_diff_trial = mr_trial_log - mr_ref_log
                        model_spatial_reg_trial = current_lambda_spatial * float(model_spatial_diff_trial.T @ WmT_eff_Wm_eff @ model_spatial_diff_trial)
                        
                        model_temporal_reg_trial_val = 0.0
                        if self.Wt_temporal.shape[0] > 0: 
                             model_temporal_diff_trial = self.Wt_temporal @ mr_trial_log 
                             model_temporal_reg_trial_val = current_alpha_temporal * float(model_temporal_diff_trial.T @ (Rt_irls if Rt_irls is not None and Rt_irls.shape[0]>0 else csr_matrix(np.eye(self.Wt_temporal.shape[0]))) @ model_temporal_diff_trial)
                        
                        obj_func_trial = data_misfit_trial + model_spatial_reg_trial + model_temporal_reg_trial_val
                        
                        armijo_c1 = 1e-4 
                        if obj_func_trial <= objective_func_current + armijo_c1 * mu_LS * directional_derivative:
                            mr_log = mr_trial_log
                            line_search_succeeded_main = True
                            break 
                    except Exception as e:
                        print(f"Error in line search evaluation (iter {ls_main_iter}, mu_LS {mu_LS:.1e}): {e}")
                    
                    mu_LS *= 0.5
                    if mu_LS < 1e-4: break
                
                if not line_search_succeeded_main:
                    print("Line search failed in main loop. Model not updated.")
                
                lambda_update_rate = self.parameters['lambda_rate']
                min_lambda_allowed = self.parameters['lambda_min']
                if lambda_update_rate < 1.0 and current_lambda_spatial > min_lambda_allowed :
                    if chi2_val < target_chi_sq_param * 1.5 or dPhi_val < 0.05 : 
                        current_lambda_spatial = max(current_lambda_spatial * lambda_update_rate, min_lambda_allowed)
            
            if inversion_type != 'L2': 
                irls_model_change = np.linalg.norm(mr_log - mr_log_previous_irls) / (np.linalg.norm(mr_log_previous_irls) + 1e-9) 
                print(f"IRLS iteration {irls_iter_count + 1} model change: {irls_model_change:.4e}")
                if irls_model_change < irls_conv_tolerance or chi2_val < target_chi_sq_param :
                    print(f"IRLS converged after {irls_iter_count + 1} iterations.")
                    break 
                mr_log_previous_irls = mr_log.copy() 
        
        final_model_linear_3d = np.exp(mr_log.reshape((num_model_cells, self.num_timesteps), order='F'))
        
        self.result.final_models = final_model_linear_3d
        self.result.all_chi2 = iteration_chi2_history 
        self.result.mesh = self.mesh

        if self.num_timesteps > 0:
            rep_timestep_idx = self.num_timesteps // 2
            rep_model_log_for_coverage = mr_log.reshape((num_model_cells, self.num_timesteps), order='F')[:, rep_timestep_idx]
            try:
                from .ert_inversion import ERTInversion as SingleERTInv 
                temp_ert_inv_for_coverage = SingleERTInv(data_file=self.data_files[rep_timestep_idx], mesh=self.mesh)
                temp_ert_inv_for_coverage.fwd_operator = self.fwd_operators[rep_timestep_idx]
                
                # Call the get_coverage method from the ERTInversion instance
                # This method was added to ERTInversion in a previous step.
                coverage_rep = temp_ert_inv_for_coverage.get_coverage(rep_model_log_for_coverage, is_log_model=True)

                if coverage_rep is not None:
                    self.result.all_coverage = [coverage_rep.copy() for _ in range(self.num_timesteps)] 
                else: self.result.all_coverage = []
            except Exception as e:
                print(f"Warning: Could not compute representative coverage for time-lapse: {e}")
                self.result.all_coverage = []
        
        print('Time-Lapse ERT inversion finished.')
        return self.result
