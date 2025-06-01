"""
Single-time Electrical Resistivity Tomography (ERT) inversion functionality.

This module provides the `ERTInversion` class, which performs a standard
single-time ERT inversion using PyGIMLi. It handles data loading, mesh creation
(if not provided), forward modeling, Jacobian computation, and an iterative
Gauss-Newton like optimization scheme with regularization and model constraints.
"""
import os # For checking file existence in __init__
import numpy as np
import pygimli as pg
from pygimli.physics import ert # For ERT specific functionalities like ert.load, ERTManager, ERTModelling
from scipy.sparse import diags # For creating diagonal matrices efficiently
import sys # For flushing print statements, useful in some environments
from typing import Optional, Union, Dict, Any, Tuple

from .base import InversionBase, InversionResult # Base classes for inversion structure
# ertforward2 and ertforandjac2 are helper functions for forward modeling and Jacobian calculation,
# likely handling log-transformations.
from ..forward.ert_forward import ertforward2, ertforandjac2
# generalized_solver is a custom or wrapped linear system solver.
from ..solvers.linear_solvers import generalized_solver # Corrected relative import path


class ERTInversion(InversionBase):
    """
    Performs a single-time Electrical Resistivity Tomography (ERT) inversion.

    This class inherits from `InversionBase` and implements the specific steps
    for ERT inversion, including setting up PyGIMLi's ERT forward operator,
    data weighting, model regularization, and an iterative inversion loop.
    """
    
    def __init__(self, data_file: str, mesh: Optional[pg.Mesh] = None, **kwargs: Any):
        """
        Initialize the ERTInversion class.

        Args:
            data_file (str): Path to the ERT data file (readable by `pygimli.ert.load`).
            mesh (Optional[pg.Mesh], optional): A pre-defined PyGIMLi mesh for the inversion.
                                                If None, a mesh will be automatically created
                                                based on the data during the `setup()` phase.
                                                Defaults to None.
            **kwargs (Any): Additional keyword arguments to override default inversion parameters.
                            See `InversionBase` and ERT-specific defaults below. Common parameters include:
                            - `lambda_val` (float): Initial regularization parameter.
                            - `method` (str): Solver method for the linear system (e.g., 'cgls', 'lsqr').
                            - `model_constraints` (Tuple[float, float]): Min and max bounds for resistivity [ohm·m].
                            - `max_iterations` (int): Maximum number of inversion iterations.
                            - `absoluteUError` (float): Absolute error in voltage (U) for data error estimation [V].
                            - `relativeError` (float): Relative error percentage for data error estimation (e.g., 0.05 for 5%).
                            - `lambda_rate` (float): Factor to reduce lambda at each iteration if chi^2 improves.
                            - `lambda_min` (float): Minimum value for lambda.
                            - `use_gpu` (bool): Whether to attempt GPU acceleration (requires CuPy and compatible solver).
                            - `parallel` (bool): Whether to use parallel CPU computation for certain solver steps.
                            - `n_jobs` (int): Number of CPU jobs for parallel execution.
        
        Raises:
            FileNotFoundError: If the `data_file` does not exist.
            Exception: If `pygimli.ert.load` fails to load the data.
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"ERT data file not found: {data_file}")
        try:
            data = ert.load(data_file) # Load ERT data using PyGIMLi
        except Exception as e:
            raise RuntimeError(f"Failed to load ERT data from '{data_file}': {e}")
        
        # Call the parent class's initializer (InversionBase)
        super().__init__(data, mesh, **kwargs) # data and mesh are stored in self.data and self.mesh
        
        # Define ERT-specific default parameters that can be overridden by kwargs
        ert_specific_defaults = {
            'lambda_val': 10.0,        # Initial regularization strength
            'method': 'cgls',          # Default linear solver method
            'absoluteUError': 0.0,     # Absolute voltage error for error model (if err not in data)
            'relativeError': 0.05,     # Relative error for error model (if err not in data)
            'lambda_rate': 1.0,        # Factor to multiply lambda by (typically <=1 to reduce lambda)
                                       # A value of 1.0 means lambda is not changed by this rate.
                                       # Should be < 1 for lambda reduction strategies (e.g. 0.8).
            'lambda_min': 1.0,         # Minimum lambda value to prevent it from becoming too small.
            'use_gpu': False,          # GPU acceleration for solver (if supported)
            'parallel': False,         # Parallel CPU computation for solver (if supported)
            'n_jobs': -1               # Number of CPU jobs for parallel execution
        }
        
        # Update self.parameters with ERT-specific defaults if not already provided by user in kwargs
        for key, default_value in ert_specific_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = default_value
        
        # Initialize attributes specific to ERT inversion
        self.fwd_operator: Optional[ert.ERTModelling] = None # PyGIMLi forward operator
        self.Wd: Optional[Union[np.ndarray, diags]] = None  # Data weighting matrix ( Wd.T * Wd = Cd^-1 ), can be sparse
        self.Wm_r: Optional[Union[np.ndarray, pg.matrix.RSparseMatrix]] = None   # Model regularization/weighting matrix (spatial constraints)
                                                                                 # Often denoted as W_m or R in literature. Kept as sparse.
        self.log_rhoa_obs: Optional[np.ndarray] = None  # Log-transformed observed apparent resistivities
        
        # Ensure the InversionResult object is initialized for this specific inversion type
        self.result = InversionResult() # Base result type is sufficient for single-time ERT
        self.result.meta['inversion_type'] = 'ERTStandard'
        self.result.meta['data_file'] = data_file
        self.result.meta.update(self.parameters) # Store all used parameters

    def setup(self) -> None:
        """
        Set up the ERT inversion environment.

        This involves:
        - Creating an inversion mesh if not already provided.
        - Initializing the PyGIMLi `ERTModelling` forward operator.
        - Preparing observed data (log-transforming apparent resistivities).
        - Calculating data error weighting matrix (`self.Wd`).
        - Creating the model regularization matrix (`self.Wm_r`).
        """
        super().setup() # Calls InversionBase.setup() which handles basic mesh storage in self.result.meta

        # Create an inversion mesh if one wasn't provided during initialization.
        # `ert.ERTManager` can create a suitable mesh based on electrode configurations.
        if self.mesh is None:
            # Potential Issue: `ERTManager` needs data with sensor positions.
            # `self.data` should have been loaded in __init__.
            if self.data is None: # Should not happen if __init__ succeeded.
                raise RuntimeError("Data container is not available for automatic mesh creation.")
            try:
                ert_manager = ert.ERTManager(self.data, verbose=False)
                # Quality 34 is a common value for ERT meshes.
                # `paraDepth` could be a parameter if more control is needed.
                self.mesh = ert_manager.createMesh(quality=34.0) 
                self.result.mesh = self.mesh # Update mesh in result object
                print("Inversion mesh automatically created based on data.")
            except Exception as e:
                raise RuntimeError(f"Failed to automatically create inversion mesh: {e}")
        
        # Initialize the PyGIMLi ERT forward modeling operator
        self.fwd_operator = ert.ERTModelling()
        self.fwd_operator.setData(self.data) # Set the measurement scheme
        # Associate the mesh with the forward operator.
        # This mesh should have cell markers if region-based regularization or parameterization is used.
        # The paraDomain is the part of the mesh where parameters are defined (usually all cells).
        self.fwd_operator.setMesh(self.mesh, ignoreRegionManager=True) # ignoreRegionManager might be needed if mesh has no regions defined
                                                                    # or if we handle regions manually.
                                                                    # Default behavior usually works.
        
        # Prepare observed data: log-transform apparent resistivities (rhoa)
        # Inversion often works better with log-transformed data and model parameters for ERT
        # due to the wide range of resistivity values.
        if 'rhoa' not in self.data:
            raise ValueError("Data container must contain 'rhoa' (apparent resistivity) values.")
        
        observed_rhoa_linear = self.data['rhoa'].array() # Get rhoa as NumPy array
        if np.any(observed_rhoa_linear <= 0):
            # Non-positive apparent resistivities are non-physical and problematic for log transform.
            # This might indicate issues with data or geometric factors.
            # Consider filtering these data points or raising a more critical error.
            print("Warning: Observed apparent resistivities contain non-positive values. These will become -inf or NaN after log transform.")
        
        with np.errstate(divide='ignore', invalid='ignore'): # Suppress log(0) warnings
            self.log_rhoa_obs = np.log(observed_rhoa_linear)
        self.log_rhoa_obs = self.log_rhoa_obs.reshape(-1, 1) # Ensure it's a column vector
        
        # Data error weighting matrix (Wd)
        # Wd is typically diagonal, with elements 1/error_i or 1/(log_error_i) if data is logged.
        # If 'err' field (relative error) is present in data, use it. Otherwise, estimate.
        if 'err' in self.data and np.any(self.data['err'].array() != 0.0): # Check if 'err' exists and is not all zero
            # Assuming self.data['err'] stores relative errors (e.g., 0.05 for 5%)
            relative_errors = self.data['err'].array()
            # Absolute error in log space: error_log_d approx error_relative_d for small errors.
            # So, weight = 1 / error_relative_d
            if np.any(relative_errors <= 0):
                # This would lead to non-positive weights or division by zero.
                raise ValueError("Data 'err' field contains non-positive relative errors, which is invalid for weighting.")
            data_weights = 1.0 / relative_errors
        else:
            # Estimate errors if not present or all zero
            print("Estimating data errors using 'absoluteUError' and 'relativeError' parameters, as 'err' field in data is zero or missing.")
            ert_mng = ert.ERTManager() # Temporary manager for error estimation utility
            # `estimateError` calculates absolute error in ohm-m based on U/I and relative error.
            # If data is `rhoa`, it assumes this `rhoa` was derived from R and k.
            # The error is typically: err_abs = sqrt((abs_U_err/I)^2 * k^2 + (rel_err * R)^2 * k^2)
            # Or more simply, if error is on rhoa: err_abs_rhoa = abs_rhoa_err + rel_err_rhoa * rhoa
            # The original code used: `np.log(Delta_rhoa_rhoa + 1)` for the denominator of weights.
            # If `Delta_rhoa_rhoa` is a relative error (e.g., 0.05 from parameters), this weight is `1/log(1.05)`.
            # This is approximately `1/0.05` for small relative errors (since log(1+x) ~ x).
            # So, this implies the weights are `1 / relativeErrorParameter`.
            if self.parameters['relativeError'] <= 0:
                print("Warning: Parameter 'relativeError' is non-positive. Using identity weights for data (no specific error weighting).")
                data_weights = np.ones(self.data.size())
            else:
                data_weights = np.full(self.data.size(), 1.0 / self.parameters['relativeError'])
        
        self.Wd = diags(data_weights) # Sparse diagonal matrix for data weighting
        
        # Model regularization matrix (Wm_r or R)
        # This matrix imposes smoothness or other constraints on the model.
        # `RegionManager` from PyGIMLi can create standard smoothness constraints.
        rm = self.fwd_operator.regionManager()
        if rm is None: 
            raise RuntimeError("RegionManager not initialized in ERTModelling operator. Ensure mesh is set correctly.")
        
        constraint_matrix_sparse = pg.matrix.RSparseMapMatrix() # Empty sparse matrix
        rm.setConstraintType(1) # First-order smoothness constraints (small differences between adjacent cells)
        rm.fillConstraints(constraint_matrix_sparse) # Fill with constraints based on mesh connectivity
        
        # Convert to SciPy COO sparse matrix format
        self.Wm_r = pg.utils.sparseMatrix2coo(constraint_matrix_sparse)
        
        # Apply constraint weights if defined in regions (e.g., for different geological units)
        constraint_weights_pg = rm.constraintWeights()
        if constraint_weights_pg is not None and constraint_weights_pg.size() > 0:
            constraint_weights_np = constraint_weights_pg.array()
            # Ensure correct application of weights. If Wm_r is (nConstraints, nModelCells),
            # weights are typically per constraint row.
            if constraint_weights_np.size == self.Wm_r.shape[0]:
                 self.Wm_r = diags(constraint_weights_np).dot(self.Wm_r)
            else: # Mismatch in size
                 print(f"Warning: Constraint weights size ({constraint_weights_np.size}) does not match "
                       f"number of constraints ({self.Wm_r.shape[0]}). Not applying constraint weights.")
        
        # Keep as CSR sparse matrix for efficiency in products like Wm_r.T @ Wm_r
        self.Wm_r = self.Wm_r.tocsr() 
        # Original code converted to dense: self.Wm_r = self.Wm_r.todense()
        # This can be very memory intensive for large meshes. Sparse is preferred.
        
        # Store setup parameters in result meta for record-keeping
        self.result.meta['mesh_cell_count'] = self.mesh.cellCount()
        self.result.meta['data_point_count'] = self.data.size()
        # Note: self.parameters are already stored in self.result.meta by InversionBase.__init__
    
    def run(self, initial_model_linear: Optional[np.ndarray] = None) -> InversionResult:
        """
        Run the ERT inversion using an iterative Gauss-Newton like approach.

        Args:
            initial_model_linear (Optional[np.ndarray], optional):
                Initial guess for the resistivity model [ohm·m] (linear scale).
                If None, a homogeneous model based on the median of observed apparent
                resistivities is used. Should be a 1D array or a 2D column vector.
                Defaults to None.

        Returns:
            InversionResult: An object containing all results of the inversion process.
        
        Raises:
            RuntimeError: If `setup()` has not been called successfully.
            ValueError: If initial model dimensions are incorrect.
        """
        # Ensure setup has been completed (operators and matrices are ready)
        if self.fwd_operator is None or self.Wd is None or self.Wm_r is None or self.log_rhoa_obs is None:
            print("Inversion setup not complete. Calling setup()...")
            self.setup()
            # Re-check after setup, as setup itself could fail if data/mesh is problematic.
            if self.fwd_operator is None or self.Wd is None or self.Wm_r is None or self.log_rhoa_obs is None:
                 raise RuntimeError("Setup failed to initialize necessary components for inversion.")

        # Initialize the model (mr is log-transformed resistivity: log(ρ))
        mr: np.ndarray # Type hint for log-model
        num_model_cells = self.fwd_operator.paraDomain().cellCount()

        if initial_model_linear is None:
            # Default initial model: homogeneous, based on median of log-transformed data
            # Convert median(exp(log_rhoa_obs)) back to log for mr
            median_rhoa_linear = np.median(np.exp(self.log_rhoa_obs))
            # Handle cases where median_rhoa_linear might be non-positive due to data issues
            if median_rhoa_linear <= 0:
                print(f"Warning: Median apparent resistivity ({median_rhoa_linear:.2e}) is non-positive. "
                      "Using default initial model of 100 ohm·m.")
                median_rhoa_linear = 100.0
            
            initial_model_log = np.log(np.full((num_model_cells, 1), median_rhoa_linear))
            mr = initial_model_log
        else:
            # Use user-provided initial model
            initial_model_np = np.asarray(initial_model_linear).ravel() # Ensure 1D
            if initial_model_np.shape[0] != num_model_cells:
                raise ValueError(f"Initial model size ({initial_model_np.shape[0]}) does not match "
                                 f"mesh cell count ({num_model_cells}).")
            if np.any(initial_model_np <= 0):
                # If provided model is linear but has non-positive values, it's problematic for log.
                # Add small epsilon, or require user to provide log-model if they want to start from very low resistivities.
                print("Warning: Provided initial_model_linear contains non-positive values. Adding epsilon (1e-6) before log transform.")
                mr = np.log(initial_model_np + 1e-6).reshape(-1, 1)
            else:
                mr = np.log(initial_model_np).reshape(-1, 1)
        
        # Reference model (m_ref) for regularization is the initial model (in log space)
        mr_ref = mr.copy()
        
        # Regularization parameter (lambda)
        current_lambda_val = self.parameters['lambda_val'] # This is λ
        
        # Model parameter constraints (min/max resistivity, converted to log space)
        min_resistivity_constraint, max_resistivity_constraint = self.parameters['model_constraints']
        # Ensure constraints are positive before log transform
        if min_resistivity_constraint <= 0 or max_resistivity_constraint <=0:
            raise ValueError("Model constraints (min/max resistivity) must be positive.")
        log_min_rho_constraint = np.log(min_resistivity_constraint)
        log_max_rho_constraint = np.log(max_resistivity_constraint)
        
        # Initial chi-squared value (can be set to a large number or calculated)
        chi2_current = np.inf # Initialize with a large value
        
        # Store parameters used in this run in the result object
        self.result.meta['inversion_run_params'] = {
            'initial_lambda': current_lambda_val,
            'lambda_reduction_rate': self.parameters['lambda_rate'],
            'min_lambda': self.parameters['lambda_min'],
            'solver_method': self.parameters['method'],
            'max_iterations': self.parameters['max_iterations'],
            'target_chi_squared': self.parameters.get('min_chi2', 1.0), # target_chi_squared from InversionBase defaults
            'convergence_tolerance_dPhi': self.parameters.get('tolerance', 0.01) # from InversionBase defaults
        }

        # Main inversion iteration loop
        for iter_num in range(self.parameters['max_iterations']):
            print(f"--- ERT Inversion Iteration: {iter_num + 1} --- Lambda: {current_lambda_val:.2e} ---")
            sys.stdout.flush() # Ensure print output appears in some environments
            
            # Forward modeling (d_pred = f(m)) and Jacobian (J) computation
            # ertforandjac2 expects log-model (mr) and returns log-data (log_dr_pred) and J_loglog.
            # J_loglog = (ρ/d) * (∂d/∂ρ) = ∂log(d)/∂log(ρ)
            log_dr_pred, J_loglog = ertforandjac2(self.fwd_operator, mr.ravel(), self.mesh) # mr must be 1D for this func
            log_dr_pred = log_dr_pred.reshape(-1, 1) # Ensure column vector
            
            # Data misfit calculation (in log space)
            # residual_log_data = log_d_obs - log_d_pred
            residual_log_data = self.log_rhoa_obs - log_dr_pred
            
            # Weighted data misfit term: || Wd * (log_d_obs - log_d_pred) ||^2
            weighted_residual_log_data = self.Wd @ residual_log_data # Wd is sparse diagonal
            data_misfit_term = np.sum(weighted_residual_log_data**2)

            # Model regularization term: lambda * || Wm_r * (m - m_ref) ||^2
            # (where m and m_ref are log-resistivities)
            model_difference_log = mr - mr_ref
            # Wm_r is CSR sparse matrix. Product with dense vector (mr - mr_ref) results in dense vector.
            weighted_model_diff = self.Wm_r @ model_difference_log 
            model_reg_term = current_lambda_val * np.sum(weighted_model_diff**2)
            
            # Total objective function value
            objective_func_val = data_misfit_term + model_reg_term
            
            # Compute chi-squared (χ²) data misfit normalized by number of data points
            # χ² = (1/N) * Σ ( (d_obs_i - d_pred_i) / error_i )^2
            # If Wd_ii = 1/error_log_i, then data_misfit_term is already sum of squared weighted residuals.
            chi2_new = data_misfit_term / len(self.log_rhoa_obs) # This is (Phi_d / N)
            
            # Relative change in chi-squared (dPhi)
            # Avoid division by zero if chi2_current is 0 (or very small if it was previous chi2_new).
            dPhi = abs(chi2_new - chi2_current) / chi2_current if iter_num > 0 and chi2_current > 1e-9 else 1.0
            
            print(f"Objective Function: {objective_func_val:.4e}, Data Misfit Term: {data_misfit_term:.4e}, Model Reg Term: {model_reg_term:.4e}")
            print(f"Chi-squared (χ²): {chi2_new:.4f}, Relative χ² change (dPhi): {dPhi:.4f}")
            
            # Store iteration results
            self.result.iteration_models.append(np.exp(mr.ravel())) # Store model in linear scale
            self.result.iteration_chi2.append(chi2_new)
            # Store log-space residuals, or could convert to linear space if preferred
            self.result.iteration_data_errors.append(residual_log_data.ravel()) 
            
            # Check convergence criteria
            target_chi_sq = self.parameters.get('min_chi2', 1.0) # min_chi2 is target_chi_squared
            conv_tolerance = self.parameters.get('tolerance', 0.01)
            if chi2_new < target_chi_sq:
                print(f"Convergence achieved: Chi-squared ({chi2_new:.2f}) is below target ({target_chi_sq}).")
                break
            if dPhi < conv_tolerance and iter_num > 0: # Ensure dPhi is meaningful (not first iter)
                print(f"Convergence achieved: Relative Chi-squared change ({dPhi:.4f}) is below tolerance ({conv_tolerance}).")
                break
            if iter_num == self.parameters['max_iterations'] - 1: # Reached max iterations
                print("Maximum number of iterations reached.")
                break

            chi2_current = chi2_new # Update chi-squared for next iteration's dPhi calculation
            
            # System matrix for Gauss-Newton step: H = J_loglog^T @ Wd^T @ Wd @ J_loglog + lambda * Wm_r^T @ Wm_r
            # Gradient: grad = -J_loglog^T @ Wd^T @ Wd @ residual_log_data + lambda * Wm_r^T @ Wm_r @ model_difference_log
            
            # Left-hand side (Hessian approximation)
            # H = Jr.T @ Wd^T @ Wd @ Jr + lambda * Wm_r^T @ Wm_r
            # Note: J_loglog is (n_data, n_model), Wd is (n_data, n_data) diagonal (sparse)
            # Wm_r is (n_constraints, n_model) (sparse)
            WdT_Wd = self.Wd.T @ self.Wd # This is diag(1/err_log_i^2) (sparse)
            H_data_part = J_loglog.T @ WdT_Wd @ J_loglog # Result can be dense or sparse depending on J_loglog
            
            # Term2: lambda * Wm_r.T @ Wm_r
            H_model_part = current_lambda_val * (self.Wm_r.T @ self.Wm_r) # Sparse product
            
            Hessian_approx = H_data_part + H_model_part # Can be sparse or dense
            
            # Right-hand side (negative gradient of objective function)
            # grad_data = -J_loglog.T @ Wd^T @ Wd @ residual_log_data
            grad_data_part = - (J_loglog.T @ WdT_Wd @ residual_log_data)
            # grad_model_reg = -lambda * Wm_r^T @ Wm_r @ model_difference_log
            grad_model_reg_part = - (current_lambda_val * (self.Wm_r.T @ self.Wm_r @ model_difference_log))
            
            neg_gradient = grad_data_part + grad_model_reg_part
            neg_gradient = neg_gradient.reshape(-1,1) # Ensure column vector
            
            # Solve the linear system H * d_mr = -grad for model update d_mr
            try:
                model_update_log = generalized_solver(
                    Hessian_approx, neg_gradient, 
                    method=self.parameters['method'],
                    use_gpu=self.parameters.get('use_gpu', False), 
                    parallel=self.parameters.get('parallel', False), 
                    n_jobs=self.parameters.get('n_jobs', -1)    
                )
                model_update_log = model_update_log.reshape(-1,1) # Ensure column vector
            except Exception as e:
                print(f"Linear solver failed at iteration {iter_num+1}: {e}. Stopping inversion.")
                break

            # Line search to find optimal step size (mu_LS)
            mu_LS = 1.0 # Initial step length
            num_line_search_max_steps = 20 # Max attempts for line search
            armijo_c1 = 1e-4 # Constant for Armijo condition
            
            # Gradient term for Armijo: grad_phi^T @ p_k
            # Here, gradient_phi = -neg_gradient, p_k = model_update_log
            # So, term is (-neg_gradient)^T @ model_update_log
            # This should be negative for a descent direction if model_update_log is from H*p = -g
            # Original code: fgoal = fc_r - 1e-4 * mu_LS * (d_mr.T @ gc_r1_col)
            # where gc_r1 was an alternative gradient. d_mr was model_update.
            # If model_update_log is the search direction p, and full_gradient = -neg_gradient,
            # then directional_derivative = full_gradient.T @ model_update_log.
            # This should be negative if p is a descent direction.
            directional_derivative = (-neg_gradient.T @ model_update_log).item()
            
            mr_next = mr # Default to current model if line search fails
            line_search_succeeded = False
            for ls_iter in range(num_line_search_max_steps):
                mr_trial = mr + mu_LS * model_update_log
                mr_trial = np.clip(mr_trial, log_min_rho_constraint, log_max_rho_constraint) # Apply constraints
                
                # Recalculate objective function with mr_trial
                try:
                    log_dr_trial = ertforward2(self.fwd_operator, mr_trial.ravel(), self.mesh).reshape(-1,1)
                    res_log_trial = self.log_rhoa_obs - log_dr_trial
                    weighted_res_log_trial = self.Wd @ res_log_trial
                    data_misfit_trial = np.sum(weighted_res_log_trial**2)
                    
                    model_diff_log_trial = mr_trial - mr_ref
                    weighted_model_diff_trial = self.Wm_r @ model_diff_log_trial
                    model_reg_trial = current_lambda_val * np.sum(weighted_model_diff_trial**2)
                    
                    obj_func_trial = data_misfit_trial + model_reg_trial

                    # Armijo condition: f(x + mu*p) <= f(x) + c1 * mu * (grad_f(x)^T * p)
                    # Here, grad_f(x)^T * p = directional_derivative (which should be < 0 for minimization)
                    if obj_func_trial <= objective_func_val + armijo_c1 * mu_LS * directional_derivative:
                        mr_next = mr_trial
                        line_search_succeeded = True
                        # print(f"Line search success: mu_LS={mu_LS:.2e}, New ObjF={obj_func_trial:.4e} (Old: {objective_func_val:.4e})")
                        break
                    else:
                        mu_LS *= 0.5 # Reduce step size
                except Exception as e:
                    print(f"Error during line search (iter {ls_iter}, mu_LS {mu_LS:.1e}): {e}. Trying smaller step.")
                    mu_LS *= 0.5 # Reduce step size on error too
                
                if mu_LS < 1e-4: # Minimum step size to prevent infinite loop
                    if not line_search_succeeded: print('Line search: Step size too small, using previous model state.')
                    break 
            
            if not line_search_succeeded:
                print("Line search failed to find sufficient decrease. Model not updated in this iteration.")
                # Optionally, could decide to stop inversion or try different strategy if LS fails repeatedly.
                # For now, mr remains unchanged from previous iteration.
            else:
                 mr = mr_next # Update model only if line search succeeded

            # Update lambda (regularization parameter)
            # Reduce lambda if chi2 is decreasing and getting close to target, or if dPhi is small.
            # Increase lambda if chi2 is too high or increasing (not implemented here, simple reduction).
            lambda_update_rate = self.parameters['lambda_rate']
            min_lambda_allowed = self.parameters['lambda_min']
            if current_lambda_val > min_lambda_allowed and lambda_update_rate < 1.0: # Only if rate is for reduction
                # Example strategy: reduce if chi2 is good or improvement is slow
                if chi2_new < target_chi_sq * 1.5 or dPhi < 0.05 : 
                    current_lambda_val = max(current_lambda_val * lambda_update_rate, min_lambda_allowed)
        
        # After loop finishes (convergence or max_iterations)
        # Store final model and related results
        final_model_linear = np.exp(mr.ravel()) # Convert final log-model to linear scale
        self.result.final_model = final_model_linear
        
        # Compute final predicted data using the final linear model
        try:
            final_response_pg = self.fwd_operator.response(pg.Vector(final_model_linear))
            self.result.predicted_data = np.array(final_response_pg)
        except Exception as e:
            print(f"Warning: Could not compute final predicted data: {e}")
            self.result.predicted_data = None
        
        # Compute and store final coverage
        try:
            # Pass the log-transformed model `mr` to get_coverage if it expects log model
            self.result.coverage = self.get_coverage(model_for_coverage=mr.ravel(), is_log_model=True)
        except Exception as e:
            print(f"Warning: Could not compute final coverage: {e}")
            self.result.coverage = None 
        
        self.result.mesh = self.mesh # Ensure mesh is in results (already done in InversionBase.setup)
        
        print(f'ERT inversion finished after {iter_num + 1} iterations. Final Chi-squared: {chi2_new:.4f}')
        return self.result

    # Helper method to get coverage, adapted from ERTForwardModeling class, using instance's fwd_operator
    def get_coverage(self, model_for_coverage: np.ndarray, is_log_model: bool) -> Optional[np.ndarray]:
        """
        Computes coverage (sensitivity/resolution proxy) for the current forward operator setup and a given model.
        This method is similar to `ERTForwardModeling.get_coverage` but uses the instance's attributes.

        Args:
            model_for_coverage (np.ndarray): The model parameters (1D array, typically log-transformed resistivity
                                             if `is_log_model` is True) for which to calculate coverage.
            is_log_model (bool): If True, `model_for_coverage` is assumed to be log-transformed (natural log)
                                 and will be exponentiated before use.

        Returns:
            Optional[np.ndarray]: An array of log10-transformed coverage values for each cell,
                                  or None if calculation fails.
        """
        if self.fwd_operator is None or self.fwd_operator.paraDomain() is None:
            print("Warning: Forward operator or its parameter domain (mesh) not set up. Cannot calculate coverage.")
            return None
        
        actual_resistivity_pg: pg.Vector
        if is_log_model:
            actual_resistivity_pg = pg.Vector(np.exp(model_for_coverage))
        else:
            actual_resistivity_pg = pg.Vector(model_for_coverage)

        try:
            response_pg = self.fwd_operator.response(actual_resistivity_pg)
            self.fwd_operator.createJacobian(actual_resistivity_pg)
            jacobian_gmat = self.fwd_operator.jacobian()

            # Prepare weights for coverageDCtrans: 1/response and 1/model (actual resistivity)
            response_arr = np.array(response_pg)
            response_arr_safe = np.where(response_arr > 1e-12, response_arr, 1e-12) # Avoid division by zero/small
            
            model_arr_safe = np.array(actual_resistivity_pg)
            model_arr_safe[model_arr_safe <= 1e-12] = 1e-12

            # pg.core.coverageDCtrans(GIMatrix J, RVector dataErrorWeights, RVector modelParameterWeights)
            # Here, dataErrorWeights = 1.0 / response_arr_safe (weights inversely proportional to data magnitude)
            # modelParameterWeights = 1.0 / model_arr_safe (weights inversely proportional to model parameter magnitude)
            # This calculates sum_i (J_ij * (1/d_i))^2 * (1/m_j)^2 or similar, which is different from original class.
            # Original ERTForwardModeling.get_coverage used:
            # covTrans = pg.core.coverageDCtrans(jacobian, 1.0 / response, 1.0 / model)
            # This implies dataError = 1/response and modelWeight = 1/model.
            # Let's stick to that for consistency with the example.
            coverage_values_pg = pg.core.coverageDCtrans(jacobian_gmat, 
                                                       pg.Vector(1.0 / response_arr_safe), 
                                                       pg.Vector(1.0 / model_arr_safe))
            coverage_values_np = np.array(coverage_values_pg)
            
            parameter_mesh = self.fwd_operator.paraDomain()
            if parameter_mesh.cellCount() != len(coverage_values_np):
                print(f"Warning: Coverage vector length ({len(coverage_values_np)}) does not match "
                      f"parameter mesh cell count ({parameter_mesh.cellCount()}). "
                      "Cell size normalization might be incorrect/skipped.")
                normalized_coverage = coverage_values_np # No normalization if counts mismatch
            else:
                cell_sizes_for_norm = np.array([cell.size() for cell in parameter_mesh.cells()])
                cell_sizes_for_norm[cell_sizes_for_norm <= 1e-12] = 1e-12 # Avoid division by zero
                normalized_coverage = coverage_values_np / cell_sizes_for_norm
            
            # Log10 transform, handling non-positive values
            valid_cov_mask = normalized_coverage > 1e-12 # Threshold for log10
            final_coverage_log10 = np.full_like(normalized_coverage, np.nan) # Init with NaN
            final_coverage_log10[valid_cov_mask] = np.log10(normalized_coverage[valid_cov_mask])
            return final_coverage_log10
        except Exception as e:
            print(f"Error during coverage calculation in ERTInversion: {e}")
            return None
