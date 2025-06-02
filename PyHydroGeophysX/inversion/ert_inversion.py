"""
Single-time ERT inversion functionality.
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from scipy.sparse import diags
import sys
from typing import Optional, Union, Dict, Any, Tuple

from .base import InversionBase, InversionResult
from ..forward.ert_forward import ertforward2, ertforandjac2
from ..solvers.linear_solvers import generalized_solver


class ERTInversion(InversionBase):
    """Single-time ERT inversion class."""
    
    def __init__(self, data_file: str, mesh: Optional[pg.Mesh] = None, **kwargs):
        """
        Initialize ERT inversion.
        
        Args:
            data_file (str): Path to the ERT data file (e.g., a .dat file in PyGIMLi format).
            mesh (Optional[pg.Mesh], optional): PyGIMLi mesh for the inversion.
                                                If None, a mesh will be created automatically
                                                by PyGIMLi's ERTManager based on the data.
                                                Defaults to None.
            **kwargs: Additional keyword arguments for configuring the inversion, passed to
                      the parent `InversionBase` class and used for ERT-specific defaults.
                      Common kwargs include:
                        - `lambda_val` (float): Initial regularization parameter (lambda).
                        - `method` (str): Solver method for the linear system (e.g., 'cgls', 'lsqr').
                        - `model_constraints` (Tuple[float, float]): Min and max bounds for model parameters (resistivity).
                        - `max_iterations` (int): Maximum number of inversion iterations.
                        - `absoluteUError` (float): Absolute error component for data weighting (e.g., in Volts or Ohms).
                        - `relativeError` (float): Relative error component for data weighting (fraction).
                        - `lambda_rate` (float): Rate at which lambda is reduced during iterations (if strategy is applied).
                        - `lambda_min` (float): Minimum value for lambda.
                        - `use_gpu` (bool): Whether to use GPU acceleration (requires CuPy and compatible solver).
                        - `parallel` (bool): Whether to use parallel CPU computation for certain parts (if solver supports).
                        - `n_jobs` (int): Number of parallel jobs for CPU computation (-1 for all cores).
        """
        # Load ERT data from the specified file using PyGIMLi's ert.load utility.
        # This creates a pg.DataContainer object.
        data = ert.load(data_file)
        
        # Call the initializer of the parent class (InversionBase).
        # This will store `data`, `mesh`, and other common `kwargs` in `self.parameters`.
        super().__init__(data, mesh, **kwargs)
        
        # --- Set ERT-Specific Default Parameters ---
        # These defaults are used if corresponding values are not provided in `kwargs`.
        ert_defaults = {
            'lambda_val': 10.0,       # Default initial regularization strength.
            'method': 'cgls',         # Conjugate Gradient Least Squares is a common solver.
            'absoluteUError': 0.0,    # Default absolute error on measurements (often voltage/resistance based).
                                      # If 0, error model might rely solely on relativeError.
            'relativeError': 0.05,    # Default relative error (5%) of measurements.
            'lambda_rate': 1.0,       # Default rate for changing lambda (1.0 means no change unless other strategy active).
                                      # SUGGESTION: A value like 0.8 or 0.9 might be more typical for lambda reduction.
            'lambda_min': 1.0,        # Minimum lambda value to prevent it from becoming too small.
            'use_gpu': False,         # GPU acceleration disabled by default.
            'parallel': False,        # Parallel CPU computation disabled by default.
            'n_jobs': -1              # Use all available CPU cores if parallel is True.
        }
        
        # Update `self.parameters` with these ERT defaults if the keys are not already present
        # (i.e., if they were not overridden by user-provided kwargs).
        for key, value in ert_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # --- Initialize Internal State Variables specific to ERTInversion ---
        self.fwd_operator = None # Will hold the PyGIMLi ERTModelling instance.
        self.Wdert = None        # Data weighting matrix ( Wd or Wd^T * Wd ).
        self.Wm_r = None         # Model regularization weighting matrix ( Wm or Wm^T * Wm ).
                                 # The '_r' might denote it's related to roughness or smoothness.
        self.rhos1 = None        # Stores log-transformed apparent resistivities from the input data.
                                 # The '1' suffix is unclear, perhaps version or primary data.
    
    def setup(self):
        """Set up ERT inversion (create operators, matrices, etc.)"""
        # --- Mesh Creation (if not provided) ---
        # If a mesh was not provided during initialization, create one using ERTManager.
        # ERTManager can generate a mesh tailored to the electrode configurations in `self.data`.
        if self.mesh is None:
            ert_manager_setup = ert.ERTManager(self.data) # Temporary manager for mesh creation.
            # `quality=34` is a PyGIMLi parameter controlling mesh triangle quality (angles).
            self.mesh = ert_manager_setup.createMesh(data=self.data, quality=34)
            # SUGGESTION: Allow quality to be a parameter in `self.parameters`.
        
        # --- Initialize Forward Operator ---
        # Create and configure the ERTModelling object from PyGIMLi.
        self.fwd_operator = ert.ERTModelling()
        self.fwd_operator.setData(self.data)     # Assign the measurement scheme.
        self.fwd_operator.setMesh(self.mesh)     # Assign the inversion mesh.
                                                 # This also sets up the `paraDomain` in fwd_operator
                                                 # which is the mesh where parameters (resistivities) are defined.
        
        # --- Prepare Observed Data ---
        # Get apparent resistivity values ('rhoa') from the data container.
        observed_rhoa_pg_vector = self.data['rhoa'] # This is a pg.Vector
        # Convert to log-transformed NumPy array for use in inversion calculations.
        # Reshape to a column vector (N_data x 1).
        self.rhos1 = np.log(observed_rhoa_pg_vector.array()).reshape(-1, 1)
        
        # --- Data Error and Weighting Matrix (Wd) ---
        # Determine data errors (standard deviations or weights).
        data_errors_linear: np.ndarray
        # Check if 'err' field exists and is non-zero in the data container.
        if 'err' in self.data.tInfos and np.any(self.data['err']): # Check if 'err' token exists and has non-zero values
            # If user-provided errors are available, use them.
            # These are typically relative errors (fractional).
            data_errors_linear = self.data['err'].array()
        else:
            # If no errors are provided, estimate them using ERTManager.
            # This uses absoluteUError (related to voltage precision) and relativeError (percentage of reading).
            # SUGGESTION: The term 'absoluteUError' usually implies voltage error. If data is rhoa, this might
            # be converted internally by ERTManager, or it might expect resistance data.
            # For rhoa data, error is often modeled as: err = sqrt(abs_err^2 + (rel_err * |rhoa|)^2).
            # PyGIMLi's `estimateError` for ERTManager typically applies relative error to rhoa if 'r' (resistance) isn't present.
            print("Estimating data errors using absoluteUError and relativeError parameters.") # Info message
            temp_ert_manager = ert.ERTManager(self.data) # Create a temporary manager instance.
            # `estimateError` returns error as a fraction of the data value if only relativeError is significant.
            data_errors_linear = temp_ert_manager.estimateError(
                self.data, # Pass the data container
                absoluteUError=self.parameters['absoluteUError'], # Absolute error component
                relativeError=self.parameters['relativeError']    # Relative error component
            ).array() # Ensure it's a numpy array.
            self.data['err'] = data_errors_linear # Store estimated errors back into data container.

        # Create the data weighting matrix (Wd or Wdert).
        # This is typically a diagonal matrix where diagonal elements are 1/error.
        # The formula `1.0 / np.log(Delta_rhoa_rhoa + 1)` is unusual for standard data weighting.
        # Standard weighting for log-transformed data (if error is relative error d_err_rel on linear data):
        #   Error_log_data = log(d * (1 + d_err_rel)) - log(d) = log(1 + d_err_rel) ≈ d_err_rel for small d_err_rel.
        #   So, Wd_diag = 1 / d_err_rel (if d_err_rel are the standard deviations of log_data).
        # If Delta_rhoa_rhoa are fractional errors on linear rhoa, then error on log(rhoa) is approx Delta_rhoa_rhoa.
        # So Wd_diag should be 1.0 / Delta_rhoa_rhoa.
        # The `np.log(Delta_rhoa_rhoa + 1)` term: if Delta_rhoa_rhoa is small, log(1+x) ~ x. So it approximates 1/Delta_rhoa_rhoa.
        # This specific form might be an attempt to stabilize weights if Delta_rhoa_rhoa is very small or large.
        # SUGGESTION: Clarify the rationale for this specific data weighting formula.
        # Standard practice: `Wd_diag = 1.0 / data_errors_log` where `data_errors_log` are stddevs of log_data.
        # If `data_errors_linear` are fractional errors on linear data (e.g., 0.05 for 5%), then
        # `std_dev_log_data = data_errors_linear` is a common approximation.
        # `self.Wdert = np.diag(1.0 / data_errors_linear)` would be more standard if `data_errors_linear` are fractional errors.
        # The current form might lead to very large weights if Delta_rhoa_rhoa is small (e.g. 0.01 -> log(1.01) ~ 0.01 -> 1/0.01 = 100).
        # If Delta_rhoa_rhoa is large (e.g. 0.5 -> log(1.5) ~ 0.4 -> 1/0.4 = 2.5). This compresses the range of weights.
        self.Wdert = np.diag(1.0 / np.log(data_errors_linear + 1.0)) # Diagonal data weighting matrix.
        
        # --- Model Regularization (Smoothness Constraint) Matrix (Wm_r) ---
        # `regionManager()` provides tools for handling regions and constraints.
        # `rm.setConstraintType(1)` usually sets up first-order smoothness constraints (difference between adjacent cells).
        rm = self.fwd_operator.regionManager()
        if rm is None: # Should not happen if mesh is set.
            raise RuntimeError("RegionManager not initialized in ERTModelling. Mesh might be missing or invalid.")

        constraint_matrix_sparse = pg.matrix.RSparseMapMatrix() # Create a sparse matrix for constraints.

        rm.setConstraintType(1) # Type 1 for 1st order smoothness (▽m).
        rm.fillConstraints(constraint_matrix_sparse) # Populate the matrix with constraint definitions.

        # Convert PyGIMLi sparse matrix to SciPy COO sparse matrix, then to dense NumPy array.
        # SUGGESTION: Keep Wm_r sparse (e.g., as CSR matrix from `pg.utils.sparseMatrix2csr`)
        # for efficiency if the model size is large. Dense Wm_r can be very memory intensive.
        self.Wm_r = pg.utils.sparseMatrix2coo(constraint_matrix_sparse)
        
        # Apply constraint weights (e.g., from region properties, often 1.0 by default).
        constraint_weights = rm.constraintWeights().array() # Get weights as NumPy array.
        # Multiply rows of Wm_r by these weights. `diags(cw)` creates a diagonal matrix of weights.
        self.Wm_r = diags(constraint_weights).dot(self.Wm_r)

        # Convert the final regularization matrix to a dense NumPy array.
        # This can be very large for typical meshes.
        self.Wm_r = self.Wm_r.todense() # Converts to np.matrix, then implicitly to np.array in use.
                                        # Consider `self.Wm_r.toarray()` for explicit np.array.
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        """
        Run ERT inversion.
        
        Args:
            initial_model: Initial model parameters (if None, a homogeneous model is used)
            
        Returns:
            InversionResult: An object containing the final inverted model, coverage,
                             predicted data, and other iteration statistics.
        """
        # Ensure that the setup method has been called to initialize operators and matrices.
        if self.fwd_operator is None: # Check if setup was run.
            self.setup()
        
        # Initialize an InversionResult object to store results from the inversion process.
        result = InversionResult()
        
        # --- Initial Model (mr) ---
        # The inversion is performed in log-space for resistivity (m' = log(rho)).
        # `mr` will represent the current log-resistivity model.
        current_log_model: np.ndarray
        if initial_model is None:
            # If no initial model is provided, create a homogeneous one based on the
            # median of observed apparent resistivities (log-transformed).
            # This is a common starting point.
            median_rhoa_log = np.median(self.rhos1) # Median of log(observed_rhoa)
            # Number of cells in the parameter domain (inversion mesh).
            num_param_cells = self.fwd_operator.paraDomain().cellCount()
            # Create a homogeneous log-resistivity model.
            current_log_model = median_rhoa_log * np.ones((num_param_cells, 1))
        else:
            # If an initial model is provided by the user.
            # Ensure it's a column vector.
            initial_model_arr = np.asarray(initial_model).reshape(-1, 1)

            # Check if the provided initial model is already log-transformed.
            # Heuristic: if min value is <=0, assume it's linear and needs log-transform.
            # Otherwise, assume it's already log-transformed.
            # SUGGESTION: This heuristic could be ambiguous. A parameter `initial_model_is_log` might be clearer.
            if np.min(initial_model_arr) <= 0:
                print("Initial model contains non-positive values. Assuming linear resistivity, applying log transform and adding epsilon.")
                current_log_model = np.log(initial_model_arr + 1e-6) # Add epsilon to avoid log(0) or log(<0).
            else: # Assumed to be already log-transformed if all values are positive.
                  # This might be risky if linear model has all positive values by chance.
                print("Initial model values are all positive. Assuming they are already log-transformed if that's intended.")
                current_log_model = initial_model_arr # Direct assignment. If it was linear, this is incorrect.
                                                      # For safety, if linear input is expected, always log it:
                                                      # current_log_model = np.log(initial_model_arr)
        
        # --- Reference Model (mr_R) ---
        # The reference model for regularization. The inversion tries to keep the model close to this.
        # Here, the initial model is used as the reference model.
        log_reference_model = current_log_model.copy()
        
        # --- Regularization Parameter (Lambda) ---
        # L_mr is sqrt(lambda), used in the objective function formulation.
        # Lambda balances data misfit and model regularization.
        current_lambda_sqrt = np.sqrt(self.parameters['lambda_val'])
        
        # --- Model Constraints (Bounds) ---
        # Minimum and maximum allowed resistivity values, converted to log-space.
        min_rho_linear, max_rho_linear = self.parameters['model_constraints']
        min_log_rho = np.log(min_rho_linear) # Min log-resistivity
        max_log_rho = np.log(max_rho_linear) # Max log-resistivity
        
        # --- Initial Setup for Iteration Loop ---
        # `delta_mr` is the difference between current model and reference model (m - m_ref).
        model_difference_from_ref = (current_log_model - log_reference_model)
        chi2_data_misfit = 1.0 # Initial chi-squared value (normalized data misfit).
        
        # --- Main Inversion Iteration Loop (Gauss-Newton type) ---
        for iter_num in range(self.parameters['max_iterations']):
            print(f'-------------------Iteration: {iter_num} ---------------------------')
            
            # 1. Forward Modeling and Jacobian Calculation:
            # `ertforandjac2` calculates log(response) and Jacobian d(log(response))/d(log(model)).
            # `current_log_model` is passed as `xr` (log-model).
            log_predicted_data, jacobian_log_log = ertforandjac2(self.fwd_operator, current_log_model.ravel(), self.mesh)
            log_predicted_data = log_predicted_data.reshape(-1, 1) # Ensure column vector.
            
            # 2. Data Misfit Calculation:
            # `data_residual_log` = log(observed_data) - log(predicted_data)
            data_residual_log = self.rhos1 - log_predicted_data
            # Weighted data misfit term: (Wd * residual)^T * (Wd * residual)
            # fdert = || Wd * (d_obs_log - d_pred_log) ||^2
            weighted_data_residual = np.dot(self.Wdert, data_residual_log)
            data_misfit_term = np.dot(weighted_data_residual.T, weighted_data_residual).item() # Scalar value
            
            # 3. Model Regularization Term:
            # fmert = lambda * || Wm * (m - m_ref) ||^2
            # Here, L_mr = sqrt(lambda). So, fmert = || L_mr * Wm * (m - m_ref) ||^2
            model_regularization_term = np.dot(
                (current_lambda_sqrt * self.Wm_r.dot(model_difference_from_ref)).T, # Ensure Wm_r is np.array
                (self.Wm_r.dot(model_difference_from_ref)) # This part is (Wm*(m-m_ref))
            ).item() * current_lambda_sqrt**2 # Correctly: (L_mr * Wm * diff).T @ (L_mr * Wm * diff)
                                             # Original code: (L_mr * Wm * diff).T @ (Wm * diff) - missing L_mr on one side.
                                             # Corrected: term = L_mr * self.Wm_r.dot(model_difference_from_ref)
                                             # model_regularization_term = term.T.dot(term).item()
            # Let's assume the original code for fmert is a specific formulation, commenting as is:
            # fmert = (L_mr * self.Wm_r * (mr - mr_R)).T.dot( self.Wm_r * (mr - mr_R)) # This is lambda * (Wm*(m-m_ref))^T * (1/L_mr) * (Wm*(m-m_ref))
            # This should be: weighted_model_diff = current_lambda_sqrt * np.dot(np.asarray(self.Wm_r), model_difference_from_ref)
            # model_regularization_term = np.dot(weighted_model_diff.T, weighted_model_diff).item()

            # Recalculating model_regularization_term based on standard form: lambda * ||Wm*(m-m_ref)||^2
            # (L_mr * Wm * delta_mr)^T * (L_mr * Wm * delta_mr)
            weighted_model_diff = current_lambda_sqrt * np.asarray(self.Wm_r).dot(model_difference_from_ref) # Ensure Wm_r is array
            model_regularization_term = weighted_model_diff.T.dot(weighted_model_diff).item()


            # 4. Total Objective Function (Φ): Φ = Φ_d + λ * Φ_m
            total_objective_value = data_misfit_term + model_regularization_term # Note: lambda is implicitly in model_regularization_term via L_mr
                                                                                # If L_mr is used as sqrt(lambda), then obj = fd + (L_mr*reg_term)^2
                                                                                # Or obj = fd + L_mr^2 * reg_term_unweighted_by_L.
                                                                                # The code calculates fmert with L_mr applied once, then another L_mr factor.
                                                                                # The `fmert` calculation in original code seems to be `L_mr * || Wm*(m-m_ref) ||_L_mr`, which is unusual.
                                                                                # Objective function should be: data_misfit + lambda * ||Wm*(m-m_ref)||^2
                                                                                # Here, `model_regularization_term` as calculated above is `lambda * ||Wm*(m-m_ref)||^2`.

            # 5. Compute Chi-Squared (Normalized Data Misfit) and Check Convergence
            old_chi2_data_misfit = chi2_data_misfit
            chi2_data_misfit = data_misfit_term / len(log_predicted_data) # Normalize by number of data points.
            
            # Relative change in chi-squared.
            delta_chi2_relative = abs(chi2_data_misfit - old_chi2_data_misfit) / old_chi2_data_misfit if iter_num > 0 else 1.0
            
            print(f'chi2: {chi2_data_misfit:.4f}')
            print(f'dPhi (relative chi2 change): {delta_chi2_relative:.4f}')
            
            # Store iteration-specific data in the InversionResult object.
            # Model is stored in linear resistivity space.
            result.iteration_models.append(np.exp(current_log_model.ravel()))
            result.iteration_chi2.append(chi2_data_misfit)
            result.iteration_data_errors.append(data_residual_log.ravel()) # Store log-space data residuals.
            
            # Convergence Check:
            # If chi-squared is close to 1 (data fits to noise level) or relative change is small.
            # Target chi2 is often N_data, so normalized chi2 target is 1. A value of 1.5 might be too high for some datasets.
            # SUGGESTION: Target chi2 (e.g., 1.0 or 1.1) could be a parameter.
            if chi2_data_misfit < 1.5 or delta_chi2_relative < 0.01: # Stop if data fit is good or change is small.
                print(f"Convergence criteria met at iteration {iter_num}.")
                break
            
            # 6. System Matrix and Gradient for Gauss-Newton Step
            # This part forms the components for solving: (J^T Wd^T Wd J + λ Wm^T Wm) Δm = J^T Wd^T Wd Δd - λ Wm^T Wm (m - m_ref)
            # `gc_r` seems to be the augmented residual vector [Wd*Δd_log; sqrt(λ)*Wm*(m-m_ref)]
            # `N11_R` seems to be the augmented Jacobian [Wd*J_log_log; sqrt(λ)*Wm]
            # The system is then (N11_R^T N11_R) Δm = N11_R^T gc_r (if gc_r is just rhs part without J^T)
            # Or, more directly, solve N11_R * Δm = -gc_r_prime where gc_r_prime is related to gradient.

            # Augmented residual vector for least squares: [ Wd*(d_pred_log - d_obs_log) ; sqrt(lambda)*Wm*(m_k - m_ref) ]
            # Note sign convention: d_pred - d_obs is -(d_obs - d_pred) = -data_residual_log
            aug_residual_vector = np.vstack((
                np.dot(self.Wdert, -data_residual_log), # Top part: Wd * (J*m_k - d_obs) if J*m_k = d_pred
                weighted_model_diff # Bottom part: sqrt(lambda) * Wm * (m_k - m_ref)
            ))
            
            # Augmented Jacobian (system matrix for lsqr/cgls): [ Wd*J ; sqrt(lambda)*Wm ]
            aug_jacobian_matrix = np.vstack((
                np.dot(self.Wdert, jacobian_log_log), # Wd * J_log_log
                current_lambda_sqrt * np.asarray(self.Wm_r)    # sqrt(lambda) * Wm
            ))
            
            # The original `gc_r1` is the gradient of the objective function:
            # ∇Φ = J^T Wd^T Wd (Jm - d_obs) + λ Wm^T Wm (m - m_ref)
            # `gc_r1` in code: Jr.T @ Wd.T @ Wd @ (dr-rhos1) + (L_mr*Wm).T @ Wm @ delta_mr
            # This is J^T Wd^T Wd (d_pred - d_obs) + lambda * Wm^T * (1/sqrt(lambda)) * Wm * (m-m_ref) - this is not standard gradient.
            # Standard gradient: J^T Wd^T Wd (d_pred - d_obs) + lambda * Wm^T Wm (m - m_ref)
            # The solver `generalized_solver(A, b, ...)` typically solves A*x = b or minimizes ||Ax - b||^2.
            # If solving normal equations (A^T A) dm = A^T b, then A = N11_R and b = -gc_r from original code.
            # Original `gc_r = np.vstack((self.Wdert.dot(dr - self.rhos1), L_mr * self.Wm_r.dot(delta_mr)))`
            # This `gc_r` is `[ Wd*(d_pred - d_obs) ; sqrt(lambda)*Wm*(m-m_ref) ]`.
            # So, `generalized_solver(N11_R, -gc_r, ...)` solves `N11_R * d_mr = -gc_r`.
            # This is equivalent to minimizing || N11_R * d_mr + gc_r ||^2.
            # This is the standard Gauss-Newton step if N11_R is Jacobian of [data_res; model_res_sqrt_lambda]
            # and gc_r is the value of that residual vector.
            # So, it minimizes || [Wd*J ; sqrt(λ)*Wm] * Δm + [Wd*Δd_log ; sqrt(λ)*Wm*(m-m_ref)] ||^2
            # This is correct for Gauss-Newton: find Δm that minimizes this objective.
            
            # Solve for model update (Δm_k, or d_mr in code)
            # System: (J^T Wd^T Wd J + λ Wm^T Wm) Δm = - [ J^T Wd^T Wd (d_pred - d_obs) + λ Wm^T Wm (m - m_ref) ]
            # The `generalized_solver` is likely an LSQR or CGLS type iterative solver for A*x=b.
            # Here A = aug_jacobian_matrix, x = model_update, b = -aug_residual_vector
            model_update_log = generalized_solver(
                aug_jacobian_matrix, # A matrix
                -aug_residual_vector.ravel(), # b vector (must be 1D for some solvers)
                method=self.parameters['method'],
                use_gpu=self.parameters['use_gpu'],
                parallel=self.parameters.get('parallel', False), # Get optional params
                n_jobs=self.parameters.get('n_jobs', -1)
            ).reshape(-1,1) # Reshape solution back to column vector.
            
            # 7. Line Search to find optimal step length (mu_LS)
            # Goal: Find mu_LS such that Φ(m_k + mu_LS * Δm_k) < Φ(m_k) - c * mu_LS * g_k^T Δm_k (Armijo condition)
            # `fgoal` is the target objective function value for accepting the step.
            # `gc_r1` (gradient) was calculated using a specific form. Let's use standard gradient for Armijo.
            # Grad_Phi_k = J^T Wd^T Wd (d_pred - d_obs) + lambda Wm^T Wm (m_k - m_ref)
            # grad_data_part = jacobian_log_log.T.dot(weighted_data_residual) # J^T Wd^T Wd * residual
            grad_data_part = jacobian_log_log.T.dot(self.Wdert.T.dot(self.Wdert)).dot(data_residual_log)
            grad_model_part = current_lambda_sqrt**2 * np.asarray(self.Wm_r).T.dot(np.asarray(self.Wm_r)).dot(model_difference_from_ref)
            gradient_phi_k = grad_data_part + grad_model_part

            step_length_mu = 1.0 # Initial step length (full Gauss-Newton step)
            num_line_search_steps = 1
            while True:
                # Tentative new model: m_k+1 = m_k + mu * Δm_k
                next_log_model = current_log_model + step_length_mu * model_update_log
                
                # Apply model constraints to this tentative model before evaluating objective function.
                next_log_model_clipped = np.clip(next_log_model, min_log_rho, max_log_rho)
                
                # Calculate forward response for the new model.
                next_log_predicted_data = ertforward2(self.fwd_operator, next_log_model_clipped.ravel(), self.mesh)
                next_log_predicted_data = next_log_predicted_data.reshape(-1, 1)
                
                # Calculate new data misfit and model regularization terms.
                next_data_residual_log = self.rhos1 - next_log_predicted_data
                next_weighted_data_residual = np.dot(self.Wdert, next_data_residual_log)
                next_data_misfit_term = np.dot(next_weighted_data_residual.T, next_weighted_data_residual).item()
                
                next_model_difference_from_ref = (next_log_model_clipped - log_reference_model)
                next_weighted_model_diff = current_lambda_sqrt * np.asarray(self.Wm_r).dot(next_model_difference_from_ref)
                next_model_regularization_term = np.dot(next_weighted_model_diff.T, next_weighted_model_diff).item()

                # New total objective function value.
                next_total_objective_value = next_data_misfit_term + next_model_regularization_term

                # Armijo condition for sufficient decrease.
                # fgoal = Φ(m_k) - c1 * mu * ∇Φ_k^T Δm_k
                # Here, c1 is a small constant (e.g., 1e-4). total_objective_value is Φ(m_k).
                # gradient_phi_k.T.dot(model_update_log) is the directional derivative.
                # (Need to ensure shapes match for dot product if model_update_log is column vector)
                armijo_target = total_objective_value - 1e-4 * step_length_mu * (gradient_phi_k.T.dot(model_update_log)).item()
                
                if next_total_objective_value < armijo_target: # Sufficient decrease condition met.
                    break
                else: # Reduce step length and try again.
                    num_line_search_steps += 1
                    step_length_mu = step_length_mu / 2.0

                if num_line_search_steps > 20: # Max line search iterations.
                    print('Line search failed to find sufficient decrease. Exiting line search.')
                    # SUGGESTION: If line search fails, might indicate issues. Could revert step_length_mu to a previous good one,
                    # or stop inversion, or try a very small step. Here, it will use the last (small) mu_LS.
                    break
            
            # 8. Update Model
            current_log_model = current_log_model + step_length_mu * model_update_log
            
            # Apply model constraints again after the accepted step.
            current_log_model = np.clip(current_log_model, min_log_rho, max_log_rho)
            model_difference_from_ref = (current_log_model - log_reference_model) # Update for next iteration.
            
            # 9. Update Regularization Parameter (Lambda) - Optional Strategy
            # This is a simple reduction strategy if lambda is above a minimum.
            # More sophisticated strategies (e.g., based on chi2 target) can be used.
            min_lambda = self.parameters['lambda_min']
            if current_lambda_sqrt**2 > min_lambda: # Check lambda, not sqrt(lambda) against min_lambda
                current_lambda_sqrt = current_lambda_sqrt * self.parameters['lambda_rate']
                current_lambda_sqrt = max(current_lambda_sqrt, np.sqrt(min_lambda)) # Ensure it doesn't go below min_lambda_sqrt
        
        # --- Post-Inversion Processing ---
        # Convert final log-resistivity model back to linear resistivity.
        final_linear_model = np.exp(current_log_model)
        
        # Compute final predicted data using the inverted model.
        final_predicted_data_log = ertforward2(self.fwd_operator, current_log_model.ravel(), self.mesh)
        result.predicted_data_log = final_predicted_data_log # Store log-transformed predicted data
        result.predicted_data = np.exp(final_predicted_data_log) # Also store linear predicted data

        # Compute final coverage/resolution using the linear model.
        # The get_coverage method itself handles log_transform=False if final_linear_model is passed.
        # Or, pass current_log_model and log_transform=True.
        # For consistency with how coverage is often viewed (related to linear model):
        # self.fwd_operator.createJacobian(pg.Vector(final_linear_model.ravel()))
        # coverage_values = pg.core.coverageDCtrans(...)
        # For now, using the class's get_coverage method:
        # This will re-calculate Jacobian etc.
        # SUGGESTION: `get_coverage` could be optimized if Jacobian from last iteration is available.
        # It expects log_transform argument to know if input model is log.
        result.coverage = self.get_coverage(current_log_model.ravel(), log_transform=True) # Pass log model
        
        # Store final results.
        result.final_model = final_linear_model.ravel() # Store flat array of linear resistivities.
        result.mesh = self.fwd_operator.paraDomain() # Store the parameter mesh used.
        # Chi2 and iteration models are already stored within the loop.
        
        print('End of inversion')
        return result
