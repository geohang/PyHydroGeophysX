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
            data_file: Path to ERT data file
            mesh: Mesh for inversion (created if None)
            **kwargs: Additional parameters including:
                - lambda_val: Regularization parameter
                - method: Solver method ('cgls', 'lsqr', etc.)
                - model_constraints: (min, max) model parameter bounds
                - max_iterations: Maximum iterations
                - absoluteUError: Absolute data error
                - relativeError: Relative data error
                - lambda_rate: Lambda reduction rate
                - lambda_min: Minimum lambda value
                - use_gpu: Whether to use GPU acceleration (requires CuPy)
                - parallel: Whether to use parallel CPU computation
                - n_jobs: Number of parallel jobs (-1 for all cores)
        """
        # Load ERT data
        data = ert.load(data_file)
        
        # Call parent initializer
        super().__init__(data, mesh, **kwargs)
        
        # Set ERT-specific default parameters

        ert_defaults = {
            'lambda_val': 10.0,
            'method': 'cgls',
            'absoluteUError': 0.0,
            'relativeError': 0.05,
            'lambda_rate': 1.0,
            'lambda_min': 1.0,
            'use_gpu': False,      # Add GPU acceleration option
            'parallel': False,     # Add parallel computation option
            'n_jobs': -1           # Number of parallel jobs (-1 means all available cores)
        }
        
        # Update parameters with ERT defaults
        for key, value in ert_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize internal variables
        self.fwd_operator = None
        self.Wdert = None  # Data weighting matrix
        self.Wm_r = None   # Model weighting matrix
        self.rhos1 = None  # Log-transformed apparent resistivities
    
    def setup(self):
        """Set up ERT inversion (create operators, matrices, etc.)"""
        # Create mesh if not provided
        if self.mesh is None:
            ert_manager = ert.ERTManager(self.data)
            self.mesh = ert_manager.createMesh(data=self.data, quality=34)
        
        # Initialize forward operator
        self.fwd_operator = ert.ERTModelling()
        self.fwd_operator.setData(self.data)
        self.fwd_operator.setMesh(self.mesh)
        
        # Prepare data
        rhos = self.data['rhoa']
        self.rhos1 = np.log(rhos.array())
        self.rhos1 = self.rhos1.reshape(self.rhos1.shape[0], 1)
        
        # Data error matrix
        if np.all(self.data['err']) != 0.0:
            # If data has error values, use them
            Delta_rhoa_rhoa = self.data['err'].array()
        else:
            # Otherwise, estimate error
            ert_manager = ert.ERTManager(self.data)
            Delta_rhoa_rhoa = ert_manager.estimateError(
                self.data,
                absoluteUError=self.parameters['absoluteUError'],
                relativeError=self.parameters['relativeError']
            )
        
        # Create data weighting matrix
        self.Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))
        
        # Create model regularization matrix
        rm = self.fwd_operator.regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)
        self.Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
        cw = rm.constraintWeights().array()
        self.Wm_r = diags(cw).dot(self.Wm_r)
        self.Wm_r = self.Wm_r.todense()
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        """
        Run ERT inversion.
        
        Args:
            initial_model: Initial model parameters (if None, a homogeneous model is used)
            
        Returns:
            InversionResult with inversion results
        """
        # Make sure setup has been called
        if self.fwd_operator is None:
            self.setup()
        
        # Initialize result object
        result = InversionResult()
        
        # Set up initial model if not provided
        if initial_model is None:
            rhomodel = np.median(np.exp(self.rhos1)) * np.ones((self.fwd_operator.paraDomain.cellCount(), 1))
            mr = np.log(rhomodel)
        else:
            if initial_model.ndim == 1:
                initial_model = initial_model.reshape(-1, 1)
            if np.min(initial_model) <= 0:
                # Assume linear model values, convert to log
                mr = np.log(initial_model + 1e-6)
            else:
                mr = np.log(initial_model)
        
        # Reference model is the initial model
        mr_R = mr.copy()
        
        # Reference model is the initial model (in log space)
        mr_R = mr.copy()
        
        # Regularization parameter (lambda), square root is often used in objective function formulation
        L_mr = np.sqrt(self.parameters['lambda_val'])
        
        # Model constraints (min/max resistivity values, converted to log space)
        min_resistivity, max_resistivity = self.parameters['model_constraints']
        min_mr = np.log(min_resistivity) # Minimum log-resistivity
        max_mr = np.log(max_resistivity) # Maximum log-resistivity
        
        # Initial setup for the inversion
        # delta_mr is the difference between the current model and the reference model (in log space)
        delta_mr = (mr - mr_R) 
        chi2_ert = 1.0 # Initial chi-squared value (will be updated)
        
        # Main inversion loop
        # TODO: Consider making print statements conditional via a `verbose` parameter.
        for nn in range(self.parameters['max_iterations']):
            print(f'-------------------Iteration: {nn} ---------------------------')
            
            # Forward modeling and Jacobian calculation:
            # `ertforandjac2` computes the forward response (dr) and Jacobian (Jr)
            # for the current log-transformed model (mr).
            # dr is log(apparent resistivity), Jr is d(log(data))/d(log(model)).
            dr, Jr = ertforandjac2(self.fwd_operator, mr, self.mesh)
            dr = dr.reshape(dr.shape[0], 1) # Ensure dr is a column vector
            
            # Data misfit calculation:
            # `dataerror_ert` is the difference between observed (self.rhos1) and predicted (dr) log-resistivities.
            # `fdert` is the squared L2-norm of the weighted data misfit: ||Wd * (d_obs - d_pred)||^2
            dataerror_ert = self.rhos1 - dr
            fdert = (np.dot(self.Wdert, dataerror_ert)).T.dot(np.dot(self.Wdert, dataerror_ert))
            
            # Model regularization term calculation:
            # `fmert` is the squared L2-norm of the weighted model regularization term:
            # ||lambda * Wm * (m - m_ref)||^2
            # Here, delta_mr = mr - mr_R (current model - reference model in log space)
            fmert = (L_mr * self.Wm_r.dot(mr - mr_R)).T.dot( L_mr * self.Wm_r.dot(mr - mr_R))
            
            # Total objective function calculation:
            # `fc_r` is the sum of data misfit and model regularization.
            fc_r = fdert + fmert
            
            # Chi-squared calculation and convergence check:
            # Chi-squared is the data misfit normalized by the number of data points.
            old_chi2 = chi2_ert
            chi2_ert = fdert[0,0] / len(dr) # fdert is a 1x1 matrix, access its scalar value
            # dPhi is the relative change in chi-squared, used for convergence check.
            dPhi = abs(chi2_ert - old_chi2) / old_chi2 if nn > 0 else 1.0
            
            print(f'chi2: {chi2_ert}')
            print(f'dPhi: {dPhi}')
            
            # Store iteration data for later analysis/plotting
            result.iteration_models.append(np.exp(mr.ravel())) # Store model in physical scale
            result.iteration_chi2.append(chi2_ert)
            result.iteration_data_errors.append(dataerror_ert.ravel()) # Store data error
            
            # Check for convergence based on chi-squared target or relative change
            if chi2_ert < self.parameters['min_chi2'] or dPhi < self.parameters['tolerance']:
                print(f"Convergence reached: chi2={chi2_ert}, dPhi={dPhi}")
                break
            
            # Gradient and system matrix (approximate Hessian) assembly:
            # `gc_r` is the gradient of the objective function.
            # `N11_R` is the approximate Hessian matrix (Gauss-Newton approximation).
            # gc_r has two parts: data misfit gradient and model regularization gradient.
            # For L2 norm: Gradient = J^T * Wd^T * Wd * (d_pred - d_obs) + lambda^2 * Wm^T * Wm * (m - m_ref)
            # The term `(dr - self.rhos1)` is -(d_obs - d_pred).
            # The `gc_r1` is an explicit calculation of the gradient.
            # `gc_r` seems to be constructing parts of the normal equations system:
            # [ Wd*J         ] [dm] = [-Wd*(d_pred - d_obs)]
            # [ Lm*Wm        ]        [-Lm*Wm*(m - m_ref)  ]
            # This is the form A*x = b for the least squares problem: min ||Ax - b||^2
            
            # Construct the components for the least-squares system matrix (N11_R) and right-hand side (gc_r)
            # N11_R corresponds to the matrix A in Ax=b
            N11_R = np.vstack((self.Wdert.dot(Jr), L_mr * self.Wm_r))
            # gc_r corresponds to the vector b in Ax=b
            # Note the negative sign for the data part: -Wd*(d_pred - d_obs) = Wd*(d_obs - d_pred)
            # and for the model part: -Lm*Wm*(m - m_ref)
            gc_r_data_part = -self.Wdert.dot(dr - self.rhos1)
            gc_r_model_part = -L_mr * self.Wm_r.dot(mr - mr_R)
            gc_r = np.vstack((gc_r_data_part, gc_r_model_part))
            
            # Ensure gc_r is a column vector
            gc_r = np.array(gc_r).reshape(-1, 1)
            
            # Alternative gradient formulation (explicit gradient of the objective function):
            # Used later in line search, so it's important.
            # grad_f = J^T Wd^T Wd (f(m) - d_obs) + lambda Wm^T Wm (m - m_ref)
            grad_data_misfit = Jr.T.dot(self.Wdert.T.dot(self.Wdert)).dot(dr - self.rhos1)
            grad_model_reg = (L_mr**2 * self.Wm_r.T).dot(self.Wm_r).dot(mr - mr_R) # L_mr is sqrt(lambda), so L_mr^2 is lambda
            gc_r1 = grad_data_misfit + grad_model_reg
            
            # Solving the linear system for model update (d_mr):
            # N11_R * d_mr = gc_r  (This is A*x=b form, not (A^T A)*x = A^T b form)
            # The solver finds d_mr that minimizes || N11_R * d_mr - gc_r ||^2
            # This d_mr is the step to update the current model `mr`.
            d_mr = generalized_solver(
                N11_R, gc_r, # System is A*x = b, where x is d_mr
                method=self.parameters['method'],
                use_gpu=self.parameters['use_gpu'],
                parallel=self.parameters.get('parallel', False),
                n_jobs=self.parameters.get('n_jobs', -1)
            )
            
            # Line search procedure to find optimal step size (mu_LS):
            # Ensures that the model update reduces the objective function.
            mu_LS = 1.0 # Initial step length (full step)
            iarm = 1    # Armijo rule iteration counter
            while True:
                mr1_try = mr + mu_LS * d_mr # Candidate updated model
                
                # Forward model with the candidate model
                dr_try = ertforward2(self.fwd_operator, mr1_try, self.mesh)
                dr_try = dr_try.reshape(dr_try.shape[0], 1)
                
                # Calculate objective function for the candidate model
                dataerror_ert_try = self.rhos1 - dr_try
                fdert_try = (np.dot(self.Wdert, dataerror_ert_try)).T.dot(np.dot(self.Wdert, dataerror_ert_try))
                fmert_try = (L_mr * self.Wm_r.dot(mr1_try - mr_R)).T.dot(L_mr * self.Wm_r.dot(mr1_try - mr_R))
                ft_r_try = fdert_try + fmert_try # Objective function value for mr1_try
                
                # Armijo condition: Check for sufficient decrease in the objective function
                # f(m + mu*dm) <= f(m) + c1 * mu * grad_f^T * dm
                # Here, c1 is a small constant (e.g., 1e-4), gc_r1 is grad_f.
                # The term d_mr.T.dot(gc_r1) is related to the directional derivative.
                # Since d_mr is typically a descent direction, d_mr.T @ gc_r1 should be negative.
                # So, -1e-4 * mu_LS * (d_mr.T @ gc_r1) will be a positive quantity.
                # We want ft_r_try <= fc_r - (positive_decrease_term)
                fgoal = fc_r - 1e-4 * mu_LS * (d_mr.T.dot(gc_r1.reshape(gc_r1.shape[0],1)))[0,0] 
                
                if ft_r_try[0,0] < fgoal: # Sufficient decrease achieved
                    break
                else: # Reduce step size
                    iarm += 1
                    mu_LS = mu_LS / 2
                
                if iarm > 20: # Maximum line search iterations
                    print('Line search FAIL EXIT')
                    # TODO: Decide behavior on line search fail (e.g., stop iteration, try smaller lambda)
                    break 
            
            # Update model with the accepted step
            mr = mr + mu_LS * d_mr
            delta_mr = mr - mr_R # Update delta_mr for the next iteration's regularization term
            
            # Apply model constraints (clipping in log space)
            mr = np.clip(mr, min_mr, max_mr)
            
            # Lambda (regularization parameter) update strategy:
            # Reduce lambda if chi-squared is decreasing, to fit data more closely.
            # This is a simple reduction strategy; more sophisticated ones exist.
            lambda_current_val_sq = L_mr**2
            lambda_min_val = self.parameters['lambda_min']
            if lambda_current_val_sq > lambda_min_val: # Only reduce if above minimum
                L_mr = L_mr * self.parameters['lambda_rate'] # lambda_rate typically <= 1
                if L_mr**2 < lambda_min_val: # Don't go below min_lambda
                    L_mr = np.sqrt(lambda_min_val)
        
        # Process final model (convert back to physical scale)
        final_model_physical = np.exp(mr)
        
        # Compute final forward response with the final physical model
        dr_final_physical = self.fwd_operator.response(pg.Vector(final_model_physical.ravel()))
        
        # Compute coverage using the final physical model
        self.fwd_operator.createJacobian(pg.Vector(final_model_physical.ravel()))
        jacobian_final_physical = self.fwd_operator.jacobian()
        
        # coverageDCtrans typically uses physical model and data for weighting
        cov_trans = pg.core.coverageDCtrans(
            jacobian_final_physical,
            1.0 / dr_final_physical, # Inverse of physical data
            1.0 / pg.Vector(final_model_physical.ravel()) # Inverse of physical model
        )
        
        # Weight by cell sizes for coverage calculation.
        # The model is cell-based, meaning one resistivity value per mesh cell.
        # Therefore, paramSizes should be an array of individual cell sizes.
        mesh_para_domain = self.fwd_operator.paraDomain()
        num_cells = mesh_para_domain.cellCount()
        param_sizes = np.zeros(num_cells)
        
        # Ensure the final_model length matches the number of cells for consistency.
        if len(final_model_physical.ravel()) != num_cells:
            raise ValueError(
                f"Mismatch between final model size ({len(final_model_physical.ravel())}) "
                f"and number of cells in parameter domain ({num_cells})."
            )
            
        for i, cell in enumerate(mesh_para_domain.cells()):
            # param_sizes[i] = cell.size() # Direct assignment if model parameters map 1:1 to cells by ID
            # The original code `paramSizes[c.marker()] += c.size()` implies a region-based model
            # (one parameter per unique marker). However, the inversion loop updates `mr` which
            # is typically cell-based (size = mesh.cellCount()).
            # Assuming `mr` and `final_model` are cell-based:
            if i < len(param_sizes): # Check bounds, though loop should be fine
                 param_sizes[i] = cell.size()
            else:
                print(f"Warning: Cell index {i} out of bounds for param_sizes (size {len(param_sizes)}). This might indicate an issue with mesh or model parameterization.")

        # Ensure param_sizes does not contain zeros to avoid division by zero issues.
        param_sizes[param_sizes == 0] = 1e-12 # Replace zero sizes with a tiny number

        if cov_trans.size() != num_cells:
            # This would indicate a mismatch in how coverage is calculated vs. cell count
             raise ValueError(f"Size of covTrans ({cov_trans.size()}) does not match number of cells ({num_cells}).")

        final_coverage_values = np.log10(cov_trans.array() / param_sizes)
        
        # Store results
        result.final_model = final_model_physical.ravel()
        result.coverage = final_coverage_values
        result.predicted_data = dr_final_physical.array()
        result.mesh = mesh_para_domain # Store the mesh used for parameterization
        
        print('End of inversion')
        return result
        # Reference model is the initial model (in log space)
        mr_R = mr.copy()
        
        # Regularization parameter (lambda), square root is often used in objective function formulation
        L_mr = np.sqrt(self.parameters['lambda_val'])
        
        # Model constraints (min/max resistivity values, converted to log space)
        min_resistivity, max_resistivity = self.parameters['model_constraints']
        min_mr = np.log(min_resistivity) # Minimum log-resistivity
        max_mr = np.log(max_resistivity) # Maximum log-resistivity
        
        # Initial setup for the inversion
        # delta_mr is the difference between the current model and the reference model (in log space)
        # This variable is updated within the loop after `mr` is updated.
        # delta_mr = (mr - mr_R) # Initial delta_mr before loop starts
        
        chi2_ert = 1.0 # Initial chi-squared value (will be updated)
        
        # Main inversion loop
        # TODO: Consider making print statements conditional via a `verbose` parameter.
        for nn in range(self.parameters['max_iterations']):
            print(f'-------------------Iteration: {nn} ---------------------------')
            
            # Update delta_mr based on the current model `mr` from previous iteration or initial model
            delta_mr = mr - mr_R

            # Forward modeling and Jacobian calculation:
            # `ertforandjac2` computes the forward response (dr) and Jacobian (Jr)
            # for the current log-transformed model (mr).
            # dr is log(apparent resistivity), Jr is d(log(data))/d(log(model)).
            dr, Jr = ertforandjac2(self.fwd_operator, mr, self.mesh)
            dr = dr.reshape(dr.shape[0], 1) # Ensure dr is a column vector
            
            # Data misfit calculation:
            # `dataerror_ert` is the difference between observed (self.rhos1) and predicted (dr) log-resistivities.
            # `fdert` is the squared L2-norm of the weighted data misfit: ||Wd * (d_obs - d_pred)||^2
            dataerror_ert = self.rhos1 - dr
            fdert = (np.dot(self.Wdert, dataerror_ert)).T.dot(np.dot(self.Wdert, dataerror_ert))
            
            # Model regularization term calculation:
            # `fmert` is the squared L2-norm of the weighted model regularization term:
            # ||lambda * Wm * (m - m_ref)||^2
            # Here, delta_mr = mr - mr_R (current model - reference model in log space)
            fmert = (L_mr * self.Wm_r.dot(delta_mr)).T.dot(L_mr * self.Wm_r.dot(delta_mr))
            
            # Total objective function calculation:
            # `fc_r` is the sum of data misfit and model regularization.
            fc_r = fdert + fmert
            
            # Chi-squared calculation and convergence check:
            # Chi-squared is the data misfit normalized by the number of data points.
            old_chi2 = chi2_ert
            chi2_ert = fdert[0,0] / len(dr) # fdert is a 1x1 matrix, access its scalar value
            # dPhi is the relative change in chi-squared, used for convergence check.
            dPhi = abs(chi2_ert - old_chi2) / old_chi2 if nn > 0 else 1.0
            
            print(f'chi2: {chi2_ert}')
            print(f'dPhi: {dPhi}')
            
            # Store iteration data for later analysis/plotting
            result.iteration_models.append(np.exp(mr.ravel())) # Store model in physical scale
            result.iteration_chi2.append(chi2_ert)
            result.iteration_data_errors.append(dataerror_ert.ravel()) # Store data error
            
            # Check for convergence based on chi-squared target or relative change
            if chi2_ert < self.parameters['min_chi2'] or dPhi < self.parameters['tolerance']:
                print(f"Convergence reached: chi2={chi2_ert}, dPhi={dPhi}")
                break
            
            # Gradient and system matrix (approximate Hessian) assembly:
            # `gc_r` is the gradient of the objective function.
            # `N11_R` is the approximate Hessian matrix (Gauss-Newton approximation).
            # gc_r has two parts: data misfit gradient and model regularization gradient.
            # For L2 norm: Gradient = J^T * Wd^T * Wd * (d_pred - d_obs) + lambda^2 * Wm^T * Wm * (m - m_ref)
            # The term `(dr - self.rhos1)` is -(d_obs - d_pred).
            # The `gc_r1` is an explicit calculation of the gradient.
            # `gc_r` seems to be constructing parts of the normal equations system for solving for model update `d_mr`:
            # [ Wd*J         ] [d_mr] = [-Wd*(d_pred - d_obs)]  which is [ Wd*J         ] [d_mr] = [ Wd*(d_obs - d_pred) ]
            # [ Lm*Wm        ]          [-Lm*Wm*(m - m_ref)  ]          [ Lm*Wm        ]          [ -Lm*Wm*delta_mr   ]
            # This is the form A*x = b for the least squares problem: min ||Ax - b||^2
            
            # Construct the components for the least-squares system matrix (N11_R is A)
            N11_R = np.vstack((self.Wdert.dot(Jr), L_mr * self.Wm_r))
            # Construct the right-hand side vector (gc_r is b)
            # Note the negative sign for the data part: -Wd*(d_pred - d_obs) = Wd*(d_obs - d_pred)
            # and for the model part: -Lm*Wm*(m - m_ref) = -Lm*Wm*delta_mr
            gc_r_data_part = self.Wdert.dot(self.rhos1 - dr) # Wd * (d_obs - d_pred)
            gc_r_model_part = -L_mr * self.Wm_r.dot(delta_mr) # -Lm * Wm * (m - m_ref)
            gc_r = np.vstack((gc_r_data_part, gc_r_model_part))
            
            # Ensure gc_r is a column vector
            gc_r = np.array(gc_r).reshape(-1, 1)
            
            # Explicit gradient of the objective function (used in line search goal calculation):
            # grad_f = J^T Wd^T Wd (f(m) - d_obs) + lambda Wm^T Wm (m - m_ref)
            # Note: dr - self.rhos1 = d_pred - d_obs
            grad_data_misfit = Jr.T.dot(self.Wdert.T.dot(self.Wdert)).dot(dr - self.rhos1)
            grad_model_reg = (L_mr**2 * self.Wm_r.T).dot(self.Wm_r).dot(delta_mr) # L_mr is sqrt(lambda_val), so L_mr^2 is lambda_val
            gc_r1_gradient_objective = grad_data_misfit + grad_model_reg # This is grad(Phi)
            
            # Solving the linear system for model update (d_mr):
            # N11_R * d_mr = gc_r  (This is A*x=b form, where x is d_mr)
            # The solver finds d_mr that minimizes || N11_R * d_mr - gc_r ||^2
            # This d_mr is the step to update the current model `mr`.
            # The RHS should be `gc_r` as constructed above. The original code used `-gc_r` which might
            # be due to how `gc_r` was defined or how the solver expects it.
            # If `gc_r` is defined as `b` in `Ax=b`, then solver takes `A, b`.
            # If solver expects `Ax+b=0`, then it would be `A, -b`.
            # Given `generalized_solver(A, b, ...)` it implies `Ax=b`.
            # The previous `gc_r` was `[ Wd*(d_pred - d_obs); Lm*Wm*(m-m_ref)]`.
            # If `b` is `[-Wd*(d_pred-d_obs); -Lm*Wm*(m-m_ref)]`, then `generalized_solver(N11_R, b)` is correct.
            # The current `gc_r` is `[Wd*(d_obs-d_pred); -Lm*Wm*delta_mr]`. This seems correct for RHS.
            d_mr = generalized_solver(
                N11_R, gc_r, 
                method=self.parameters['method'],
                use_gpu=self.parameters['use_gpu'],
                parallel=self.parameters.get('parallel', False),
                n_jobs=self.parameters.get('n_jobs', -1)
            )
            
            # Line search procedure to find optimal step size (mu_LS):
            # Ensures that the model update reduces the objective function (Armijo condition).
            mu_LS = 1.0 # Initial step length (full step)
            iarm = 1    # Armijo rule iteration counter
            while True:
                mr1_try = mr + mu_LS * d_mr # Candidate updated model (in log space)
                
                # Forward model with the candidate model
                dr_try = ertforward2(self.fwd_operator, mr1_try, self.mesh)
                dr_try = dr_try.reshape(dr_try.shape[0], 1)
                
                # Calculate objective function for the candidate model
                dataerror_ert_try = self.rhos1 - dr_try
                fdert_try = (np.dot(self.Wdert, dataerror_ert_try)).T.dot(np.dot(self.Wdert, dataerror_ert_try))
                fmert_try = (L_mr * self.Wm_r.dot(mr1_try - mr_R)).T.dot(L_mr * self.Wm_r.dot(mr1_try - mr_R))
                ft_r_try = fdert_try + fmert_try # Objective function value for mr1_try
                
                # Armijo condition: Check for sufficient decrease in the objective function
                # f(m + mu*dm) <= f(m) + c1 * mu * grad_f^T * dm
                # Here, c1 is a small constant (e.g., 1e-4), gc_r1_gradient_objective is grad_f.
                # The term d_mr.T @ gc_r1_gradient_objective is the directional derivative.
                # Since d_mr is a descent direction, (d_mr.T @ gc_r1_gradient_objective) should be negative.
                fgoal = fc_r[0,0] - 1e-4 * mu_LS * (d_mr.T.dot(gc_r1_gradient_objective.reshape(gc_r1_gradient_objective.shape[0],1)))[0,0] 
                
                if ft_r_try[0,0] < fgoal: # Sufficient decrease achieved
                    break
                else: # Reduce step size if condition not met
                    iarm += 1
                    mu_LS = mu_LS / 2
                
                if iarm > 20: # Maximum line search iterations
                    print('Line search FAIL EXIT')
                    # TODO: Decide behavior on line search fail (e.g., stop iteration, try smaller lambda, accept step anyway)
                    break 
            
            # Update model with the accepted step size
            mr = mr + mu_LS * d_mr
            # Note: `delta_mr` will be recomputed at the start of the next iteration based on the new `mr`.
            
            # Apply model constraints (clipping in log space)
            mr = np.clip(mr, min_mr, max_mr)
            
            # Lambda (regularization parameter) update strategy:
            # Reduce lambda if chi-squared is decreasing, to fit data more closely.
            # This is a simple reduction strategy; more sophisticated ones exist.
            lambda_current_val_sq = L_mr**2 # Current lambda squared
            lambda_min_val = self.parameters['lambda_min']
            if lambda_current_val_sq > lambda_min_val: # Only reduce if current lambda is above minimum
                L_mr = L_mr * self.parameters['lambda_rate'] # lambda_rate typically <= 1 (e.g., 0.8 to 1.0)
                if L_mr**2 < lambda_min_val: # Ensure lambda does not go below its minimum allowed value
                    L_mr = np.sqrt(lambda_min_val)
        
        # Process final model (convert back to physical scale: resistivity)
        final_model_physical = np.exp(mr)
        
        # Compute final forward response with the final physical model
        dr_final_physical = self.fwd_operator.response(pg.Vector(final_model_physical.ravel()))
        
        # Compute coverage using the final physical model
        self.fwd_operator.createJacobian(pg.Vector(final_model_physical.ravel()))
        jacobian_final_physical = self.fwd_operator.jacobian()
        
        # coverageDCtrans typically uses physical model and data for weighting
        cov_trans = pg.core.coverageDCtrans(
            jacobian_final_physical,
            1.0 / dr_final_physical, # Inverse of physical data
            1.0 / pg.Vector(final_model_physical.ravel()) # Inverse of physical model
        )
        
        # Weight by cell sizes for coverage calculation.
        # The model is cell-based, meaning one resistivity value per mesh cell.
        # Therefore, paramSizes should be an array of individual cell sizes.
        mesh_para_domain = self.fwd_operator.paraDomain()
        num_cells = mesh_para_domain.cellCount()
        param_sizes = np.zeros(num_cells)
        
        # Ensure the final_model length matches the number of cells for consistency.
        if len(final_model_physical.ravel()) != num_cells:
            raise ValueError(
                f"Mismatch between final model size ({len(final_model_physical.ravel())}) "
                f"and number of cells in parameter domain ({num_cells})."
            )
            
        for i, cell_obj in enumerate(mesh_para_domain.cells()):
            # Assuming model parameters map 1:1 to cells by their order/ID.
            # The original code `paramSizes[c.marker()] += c.size()` is for region-based models.
            # For cell-based models (which this inversion seems to be, as `mr` is cell-based),
            # each cell's size should correspond to its parameter.
            param_sizes[i] = cell_obj.size()

        # Ensure param_sizes does not contain zeros to avoid division by zero issues.
        param_sizes[param_sizes == 0] = 1e-12 # Replace zero sizes with a tiny number

        if cov_trans.size() != num_cells:
            # This would indicate a mismatch in how coverage is calculated vs. cell count
             raise ValueError(f"Size of covTrans ({cov_trans.size()}) does not match number of cells ({num_cells}).")

        final_coverage_values = np.log10(cov_trans.array() / param_sizes)
        
        # Store results
        result.final_model = final_model_physical.ravel()
        result.coverage = final_coverage_values
        result.predicted_data = dr_final_physical.array()
        result.mesh = mesh_para_domain # Store the mesh used for parameterization
        
        print('End of inversion')
        return result
