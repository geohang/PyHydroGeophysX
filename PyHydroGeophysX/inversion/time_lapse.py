"""
Time-lapse ERT inversion functionality.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pygimli as pg
import scipy.sparse as sp
from pygimli.physics import ert
from scipy.linalg import block_diag as dense_block_diag
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsqr

from ..forward.ert_forward import ertforandjac2, ertforward2
from ..solvers.linear_solvers import generalized_solver
from .base import InversionBase, TimeLapseInversionResult


def _sparse_temporal_difference_matrix(cell_count: int, size: int, dtype):
    """Build sparse first differences between adjacent model blocks."""
    time_difference = diags(
        (np.ones(size - 1, dtype=dtype), -np.ones(size - 1, dtype=dtype)),
        (0, 1),
        shape=(size - 1, size),
        format="csr",
        dtype=dtype,
    )
    return sp.kron(
        time_difference,
        sp.eye(cell_count, format="csr", dtype=dtype),
        format="csr",
    )


# ---------------------------------------------------------------------------
# calculate jacobian
# ---------------------------------------------------------------------------
def _calculate_jacobian(fwd_operators, model, mesh, size, as_sparse: bool = False,
                        dtype=np.float64):
    """
    Calculate Jacobian matrix for multi-time model.
    
    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        
    Returns:
        obs: Observed data for all timesteps
        J: Jacobian matrix
    """
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs = []
    
    jac_blocks = []
    
    for i in range(size):
        dr, Jr = ertforandjac2(fwd_operators[i], model_reshaped[:, i], mesh)
        dr = dr.astype(dtype, copy=False)
        obs.append(dr)
        jac_blocks.append(csr_matrix(Jr, dtype=dtype) if as_sparse else Jr.astype(dtype, copy=False))
    
    # Stack observations
    obs_stacked = np.vstack([o.reshape(-1, 1) for o in obs]).astype(dtype, copy=False)
    
    if as_sparse:
        J = sparse_block_diag(jac_blocks, format="csr", dtype=dtype)
    else:
        J = dense_block_diag(*jac_blocks).astype(dtype, copy=False)
    
    return obs_stacked, J


# ---------------------------------------------------------------------------
# calculate forward
# ---------------------------------------------------------------------------
def _calculate_forward(fwd_operators, model, mesh, size):
    """
    Calculate forward response for multi-time model.
    
    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        
    Returns:
        obs: Observed data for all timesteps
    """
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs = []
    
    for i in range(size):
        dr = ertforward2(fwd_operators[i], model_reshaped[:, i], mesh)
        obs.append(dr)
    
    # Stack observations
    return np.vstack([response.reshape(-1, 1) for response in obs])


# ---------------------------------------------------------------------------
# calculate forward separate
# ---------------------------------------------------------------------------
def _calculate_forward_separate(fwd_operators, model, mesh, size):
    """
    Calculate forward response for multi-time model without stacking.
    
    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        
    Returns:
        obs: List of observed data for each timestep
    """
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs = []
    
    for i in range(size):
        dr = ertforward2(fwd_operators[i], model_reshaped[:, i], mesh)
        obs.append(dr)
    
    return obs


# ---------------------------------------------------------------------------
# Time Lapse ERTInversion
# ---------------------------------------------------------------------------
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
                - lambda_val: Regularization parameter
                - alpha: Temporal regularization parameter
                - decay_rate: Temporal decay rate
                - method: Solver method ('cgls', 'lsqr', etc.)
                - model_constraints: (min, max) model parameter bounds
                - max_iterations: Maximum iterations
                - absoluteError: Absolute resistance error floor [Ohm] (default 0.0001)
                - relativeError: Relative data error
                - lambda_rate: Lambda reduction rate
                - lambda_min: Minimum lambda value
                - save_memory: Use sparse operators to reduce RAM consumption
        """
        # Load ERT data
        self.data_files = data_files
        self.measurement_times = np.array(measurement_times)
        
        # Validate input
        if len(data_files) != len(measurement_times):
            raise ValueError("Number of data files must match number of measurement times")
        
        # Load first dataset to initialize base class
        data = ert.load(data_files[0])
        
        # Call parent initializer with first dataset
        super().__init__(data, mesh, **kwargs)
        
        # Set time-lapse specific default parameters
        tl_defaults = {
            'lambda_val': 100.0,
            'alpha': 10.0,
            'decay_rate': 0.0,
            'method': 'cgls',
            'absoluteError': 0.0001,
            'relativeError': 0.05,
            # Cooling is off by default. It used to be 0.8, which moved lambda on
            # every iteration and left the final chi2 attributable to no single
            # value; the caller relaxes lambda between converged runs instead.
            'lambda_rate': 1.0,
            'lambda_min': 1.0,
            'inversion_type': 'L2',  # 'L1', 'L2', or 'L1L2'
            'model_constraints':(0.0001,10000.0),  # min and max resistivity
            'save_memory': False,  # use sparse operators to reduce RAM
            # Stopping. Both were hard-coded (chi2 < 1.5, dPhi < 0.01 after 5
            # iterations); a lambda sweep needs them configurable so a flattened
            # misfit means the lambda is spent, not that the budget ran out.
            'target_chi_squared': 1.0,
            'convergence_tolerance': 0.01,
            'min_iterations': 5,
            'verbose': True,
        }
        
        # Update parameters with time-lapse defaults
        for key, value in tl_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        self.use_sparse = bool(self.parameters.get('save_memory', False))
        self.dtype = np.float32 if self.use_sparse else np.float64
        
        # Number of timesteps
        self.size = len(data_files)
        
        # Initialize internal variables
        self.fwd_operators = []
        self.datasets = []
        self.rhos1 = None
        self.Wd = None
        self.Wm = None
        self.Wt = None
    
    def setup(self):
        """Set up time-lapse ERT inversion (load data, create operators, matrices, etc.)"""
        # Create mesh if not provided
        if self.mesh is None:
            ert_manager = ert.ERTManager(self.data)
            self.mesh = ert_manager.createMesh(data=self.data, quality=34)
        
        # Load all datasets and process
        rhos = []
        dataerr = []
        k = []
        
        for i, fname in enumerate(self.data_files):
            # Load data
            dataert = ert.load(fname)
            self.datasets.append(dataert)
            
            # Handle geometric factors
            if np.all(dataert['k'] == 0.0):
                if len(k) == 0:
                    dataert['k'] = ert.createGeometricFactors(dataert, numerical=True)
                    k = dataert['k'].array()
                else:
                    dataert['k'] = k
            
            # Get apparent resistivity
            if np.all(dataert['rhoa']) != 0.0:
                rhos.append(dataert['rhoa'].array())
            else:
                rhos.append(dataert['r'].array() * k)
            
            # Get or estimate data errors
            if np.all(dataert['err']) != 0.0:
                dataerr.append(np.clip(dataert['err'].array(), 0.01, 0.50))
            else:
                # Seb's per-measurement formula: err_i = relativeError + absoluteError / |r_i|
                abs_e = float(self.parameters['absoluteError'])
                rel_e = float(self.parameters['relativeError'])
                if 'r' in dataert.dataMap():
                    r_abs = np.abs(dataert['r'].array())
                elif 'k' in dataert.dataMap():
                    r_abs = np.abs(dataert['rhoa'].array()) / np.maximum(
                        np.abs(dataert['k'].array()), 1e-10)
                else:
                    raise RuntimeError(
                        f"Dataset {fname}: cannot estimate error without 'r' or 'k'.")
                err_i = rel_e + abs_e / np.maximum(r_abs, 1e-10)
                dataerr.append(np.clip(err_i, 0.01, 0.50))
            
            # Create forward operator
            fwd_operator = ert.ERTModelling()
            fwd_operator.setData(dataert)
            fwd_operator.setMesh(self.mesh)
            self.fwd_operators.append(fwd_operator)
        
        # Stack all data
        rhos = np.array(rhos)
        rhos_temp = rhos[0]
        for i in range(self.size - 1):
            rhos_temp = np.hstack((rhos_temp, rhos[i + 1]))
        
        rhos_temp = rhos_temp.reshape((-1, 1)).astype(self.dtype, copy=False)
        self.rhos1 = np.log(rhos_temp).astype(self.dtype, copy=False)

        del rhos_temp  # Delete after use
        del rhos  # Delete after use

        # Data error and weighting matrix
        dataerr = np.array(dataerr)
        err_temp = np.hstack(dataerr)
        data_weights = (1.0 / np.log(err_temp + 1)).astype(self.dtype, copy=False)
        if self.use_sparse:
            self.Wd = diags(data_weights, dtype=self.dtype)
            self.Wd_sq = self.Wd.multiply(self.Wd).astype(self.dtype, copy=False)
        else:
            self.Wd = np.diag(data_weights)
            self.Wd_sq = (self.Wd.T @ self.Wd).astype(self.dtype, copy=False)
        
        # Create model regularization matrix
        rm = self.fwd_operators[0].regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)
        Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
        cw = rm.constraintWeights().array().astype(self.dtype, copy=False)
        Wm_r = diags(cw).dot(Wm_r)
        
        if self.use_sparse:
            Wm_r = Wm_r.tocsr().astype(self.dtype, copy=False)
            self.Wm = sparse_block_diag([Wm_r for _ in range(self.size)], format="csr", dtype=self.dtype)
        else:
            Wm_dense = Wm_r.todense().astype(self.dtype, copy=False)
            self.Wm = dense_block_diag(*[Wm_dense for _ in range(self.size)]).astype(self.dtype, copy=False)
        
        # Create temporal regularization matrix
        cell_count = self.fwd_operators[0].paraDomain.cellCount()
        tdiff = np.diff(self.measurement_times)
        temporal_weights = np.repeat(
            np.exp(-self.parameters['decay_rate'] * tdiff),
            cell_count,
        ).astype(self.dtype, copy=False)
        if self.use_sparse:
            Wt = _sparse_temporal_difference_matrix(
                cell_count,
                self.size,
                self.dtype,
            )
        else:
            # Dense mode remains faster when constructed directly. Converting
            # a very large sparse Kronecker product back to dense is costly.
            Wt = np.zeros(
                (cell_count * (self.size - 1), cell_count * self.size),
                dtype=self.dtype,
            )
            identity = np.eye(cell_count, dtype=self.dtype)
            for i in range(self.size - 1):
                idx = i * cell_count
                Wt[idx:idx + cell_count, idx:idx + cell_count] = identity
                Wt[
                    idx:idx + cell_count,
                    idx + cell_count:idx + 2 * cell_count,
                ] = -identity
        self.Wt = diags(temporal_weights, dtype=self.dtype).dot(Wt)
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> TimeLapseInversionResult:
        """
        Run time-lapse ERT inversion.
        
        Args:
            initial_model: Initial model parameters (if None, a homogeneous model is used)
            
        Returns:
            TimeLapseInversionResult with inversion results
        """
        # Make sure setup has been called
        if not self.fwd_operators:
            self.setup()
        
        use_sparse = self.use_sparse

        def _as_col(vec):
            arr = np.asarray(vec)
            if arr.dtype != self.dtype:
                arr = arr.astype(self.dtype, copy=False)
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        def _matvec(mat, vec):
            res = mat.dot(_as_col(vec)) if sp.issparse(mat) else mat @ _as_col(vec)
            res_arr = np.asarray(res)
            return res_arr if res_arr.ndim > 1 else res_arr.reshape(-1, 1)

        def _ttm(mat, vec):
            return _matvec(mat.transpose(), _matvec(mat, vec))

        def _apply_data_weights(weight_mat, vec):
            weighted = _matvec(self.Wd, vec)
            weighted = _matvec(weight_mat, weighted)
            return _matvec(self.Wd.transpose(), weighted)

        def _quad(weighted_vec, vec):
            return (_as_col(vec).T @ _as_col(weighted_vec)).item()
        
        # Initialize result object
        result = TimeLapseInversionResult()
        result.timesteps = self.measurement_times
        
        # Set up initial model if not provided
        cell_count = self.fwd_operators[0].paraDomain.cellCount()
        
        if initial_model is None:
            # Create initial model with median resistivity for each time step
            initial_rhos = []
            for i in range(self.size):
                if hasattr(self.datasets[i], 'rhoa') and np.any(self.datasets[i]['rhoa'] > 0):
                    initial_rhos.append(np.median(self.datasets[i]['rhoa'].array()))
                else:
                    # Use default value if no apparent resistivity data
                    initial_rhos.append(100.0)
            
            mr = np.log(np.repeat(initial_rhos, cell_count).reshape(-1, 1)).astype(self.dtype, copy=False)
        else:
            # Use provided initial model
            if initial_model.shape != (cell_count, self.size):
                raise ValueError(f"Initial model should have shape ({cell_count}, {self.size})")
            
            # Flatten in column-major order and log-transform
            mr = np.log(initial_model.flatten(order='F').reshape(-1, 1)).astype(self.dtype, copy=False)
        
        # Reference model is the initial model
        mr_R = mr.copy()
        
        # Regularization parameters
        Lambda = self.parameters['lambda_val']
        alpha = self.parameters['alpha']
        
        # Model constraints
        min_mr, max_mr = self.parameters['model_constraints']
        min_mr = np.log(min_mr)
        max_mr = np.log(max_mr)

        target_chi2 = float(self.parameters.get('target_chi_squared', 1.0))
        dphi_tol = float(self.parameters.get('convergence_tolerance', 0.01))
        min_iterations = int(self.parameters.get('min_iterations', 5))
        verbose = bool(self.parameters.get('verbose', True))
        stop_reason = 'iteration_cap'

        if verbose:
            print(min_mr, max_mr)

        # Track errors for each iteration
        Err_tot = []
        chi2_old = np.inf

        # Choose inversion type
        inversion_type = self.parameters['inversion_type'].upper()
        if inversion_type not in ['L1', 'L2', 'L1L2']:
            if verbose:
                print(f"Invalid inversion type {inversion_type}, defaulting to L2")
            inversion_type = 'L2'
        
        # L1-specific parameters
        if inversion_type in ['L1', 'L1L2']:
            l1_epsilon = 1e-4
            irls_iter_max = 5 if inversion_type == 'L1' else 8
            irls_tol = 1e-3 if inversion_type == 'L1' else 1e-2
            threshold_c = 2.0  # For L1L2 hybrid
        
        # IRLS iterations for L1-norm
        for irls_iter in range(1 if inversion_type == 'L2' else irls_iter_max):
            if inversion_type in ['L1', 'L1L2'] and verbose:
                print(f'------------------- IRLS Iteration: {irls_iter + 1} ---------------------------')

            # Main inversion loop
            for nn in range(self.parameters['max_iterations']):
                if verbose:
                    print(f'-------------------ERT Iteration: {nn} ---------------------------')
                
                # Forward modeling and Jacobian computation
                dr, Jr = _calculate_jacobian(
                    self.fwd_operators, mr, self.mesh, self.size,
                    as_sparse=use_sparse, dtype=self.dtype
                )
                dr = dr.reshape(-1, 1)
                
                # Data misfit calculation
                dataerror_ert = _as_col(self.rhos1 - dr)
                
                # Handle different norms
                if inversion_type == 'L2':
                    # Standard L2 norm
                    data_weighted = _matvec(self.Wd_sq, dataerror_ert)
                    fdert = _quad(data_weighted, dataerror_ert)
                    
                    # Gradient computation with memory management
                    grad_data = -_matvec(Jr.transpose(), data_weighted)
                    
                    model_term = _ttm(self.Wm, mr)
                    fmert = Lambda * _quad(model_term, mr)
                    grad_model = Lambda * model_term
                    
                    temp_term = _ttm(self.Wt, mr)
                    ftert = alpha * _quad(temp_term, mr)
                    grad_temporal = alpha * temp_term
                        


                    
                elif inversion_type == 'L1':
                    # L1 norm using IRLS
                    Rd = diags(1.0 / np.sqrt(dataerror_ert.flatten()**2 + l1_epsilon))
                    
                    model_diff = _matvec(self.Wm, mr)
                    Rs = diags(1.0 / np.sqrt(model_diff.flatten()**2 + l1_epsilon))
                    
                    temp_diff = _matvec(self.Wt, mr)
                    Rt = diags(1.0 / np.sqrt(temp_diff.flatten()**2 + l1_epsilon))
                    
                    # Objective functions with weighted L1 norms
                    data_weighted = _apply_data_weights(Rd, dataerror_ert)
                    fdert = _quad(data_weighted, dataerror_ert)
                    
                    model_weighted = _matvec(Rs, model_diff)
                    fmert = Lambda * _quad(model_weighted, model_diff)
                    
                    temp_weighted = _matvec(Rt, temp_diff)
                    ftert = alpha * _quad(temp_weighted, temp_diff)
                    
                    # Gradient computation
                    grad_data = -_matvec(Jr.transpose(), data_weighted)
                    grad_model = Lambda * _matvec(self.Wm.transpose(), model_weighted)
                    grad_temporal = alpha * _matvec(self.Wt.transpose(), temp_weighted)
                    
                else:  # L1L2 hybrid
                    # Compute hybrid L1-L2 weights for data misfit
                    effective_epsilon = l1_epsilon * (1 + 10*np.exp(-nn/5))
                    norm_values = (
                        np.abs(dataerror_ert.flatten())
                        / np.sqrt(effective_epsilon)
                    )
                    data_weights = np.ones_like(norm_values)
                    outlier_mask = norm_values > threshold_c
                    data_weights[outlier_mask] = (
                        threshold_c / norm_values[outlier_mask]
                    )
                    
                    Rd = diags(data_weights)
                    
                    # Model and temporal weights (pure L1)
                    model_diff = _matvec(self.Wm, mr)
                    model_weights = 1.0 / np.sqrt(model_diff.flatten()**2 + l1_epsilon)
                    model_weights = np.maximum(model_weights, 1e-10)
                    Rs = diags(model_weights)
                    
                    temp_diff = _matvec(self.Wt, mr)
                    temp_weights = 1.0 / np.sqrt(temp_diff.flatten()**2 + l1_epsilon)
                    temp_weights = np.maximum(temp_weights, 1e-10)
                    Rt = diags(temp_weights)
                    
                    # Objective functions
                    data_weighted = _apply_data_weights(Rd, dataerror_ert)
                    fdert = _quad(data_weighted, dataerror_ert)
                    
                    model_weighted = _matvec(Rs, model_diff)
                    fmert = Lambda * _quad(model_weighted, model_diff)
                    
                    temp_weighted = _matvec(Rt, temp_diff)
                    ftert = alpha * _quad(temp_weighted, temp_diff)
                    
                    # Gradient computation
                    grad_data = -_matvec(Jr.transpose(), data_weighted)
                    grad_model = Lambda * _matvec(self.Wm.transpose(), model_weighted)
                    grad_temporal = alpha * _matvec(self.Wt.transpose(), temp_weighted)
                
                # Total gradient
                gc_r = grad_data + grad_model + grad_temporal
                
                # Total objective function
                ftot = fdert + fmert + ftert
                
                # Compute chi-squared and check convergence
                chi2_ert = _quad(_matvec(self.Wd_sq, dataerror_ert), dataerror_ert) / len(dr)
                dPhi = abs(chi2_ert - chi2_old) / chi2_old if nn > 0 else 1.0
                chi2_old = chi2_ert
                
                if verbose:
                    print(f'ERT chi2: {chi2_ert}')
                    print(f'dPhi: {dPhi}')
                    print(f'ERTphi_d: {fdert}, ERTphi_m: {fmert}, ERTphi_t: {ftert}')

                # Store iteration data
                Err_tot.append([chi2_ert, fmert, ftert])

                # Check for convergence
                if chi2_ert < target_chi2:
                    stop_reason = 'target'
                    if verbose:
                        print(f"Convergence reached at iteration {nn}")
                    break
                if dPhi < dphi_tol and nn > min_iterations:
                    stop_reason = 'plateau'
                    if verbose:
                        print(f"Convergence reached at iteration {nn}")
                    break
                
                # Compute Hessian (or approximation)
                if inversion_type == 'L2':
                    # Standard Gauss-Newton Hessian
                    if use_sparse:
                        H = (Jr.transpose().dot(self.Wd_sq.dot(Jr)) + 
                             Lambda * self.Wm.transpose().dot(self.Wm) + 
                             alpha * self.Wt.transpose().dot(self.Wt))
                    else:
                        H = (Jr.T @ self.Wd_sq @ Jr + 
                             Lambda * self.Wm.T @ self.Wm + 
                             alpha * self.Wt.T @ self.Wt)
                elif inversion_type == 'L1':
                    # IRLS modified Hessian
                    if use_sparse:
                        weighted_J = Rd.dot(self.Wd.dot(Jr))
                        weighted_J = self.Wd.transpose().dot(weighted_J)
                        H = (Jr.transpose().dot(weighted_J) + 
                             Lambda * self.Wm.transpose().dot(Rs.dot(self.Wm)) + 
                             alpha * self.Wt.transpose().dot(Rt.dot(self.Wt)))
                    else:
                        H = (Jr.T @ self.Wd.T @ Rd @ self.Wd @ Jr + 
                             Lambda * self.Wm.T @ Rs @ self.Wm + 
                             alpha * self.Wt.T @ Rt @ self.Wt)
                else:  # L1L2
                    # Hybrid Hessian with damping
                    if use_sparse:
                        weighted_J = Rd.dot(self.Wd.dot(Jr))
                        weighted_J = self.Wd.transpose().dot(weighted_J)
                        H = (Jr.transpose().dot(weighted_J) + 
                             Lambda * self.Wm.transpose().dot(Rs.dot(self.Wm)) + 
                             alpha * self.Wt.transpose().dot(Rt.dot(self.Wt)) + 
                             l1_epsilon * sp.eye(Jr.shape[1], format='csr', dtype=self.dtype))
                    else:
                        H = (Jr.T @ self.Wd.T @ Rd @ self.Wd @ Jr + 
                             Lambda * self.Wm.T @ Rs @ self.Wm + 
                             alpha * self.Wt.T @ Rt @ self.Wt + 
                             l1_epsilon * np.eye(Jr.shape[1]))
                
                # After using Jr for gradient computation
                del Jr  # No longer needed

                # Solve for model update
                d_mr = generalized_solver(
                    H, -gc_r, 
                    method=self.parameters['method'],
                    use_gpu=self.parameters.get('use_gpu', False),
                    parallel=self.parameters.get('parallel', False),
                    n_jobs=self.parameters.get('n_jobs', -1)
                )
                d_mr = d_mr.reshape(-1, 1)
                
                # Line search
                mu_LS = 1.0
                success = False
                best_mr = mr.copy()
                best_f = ftot
                
                # Different line search strategies based on inversion type
                if inversion_type == 'L1L2':
                    # Trust region approach for L1L2
                    mr1 = mr + d_mr
                    mr1 = np.clip(mr1, min_mr, max_mr)
                    success = True
                else:
                    # Standard line search for L2 and L1
                    for iarm in range(20):
                        mr1 = mr + mu_LS * d_mr
                        mr1 = np.clip(mr1, min_mr, max_mr)
                        
                        try:
                            dr_new = _calculate_forward(self.fwd_operators, mr1, self.mesh, self.size)
                            dr_new = dr_new.reshape(-1, 1)
                            dataerror_new = _as_col(self.rhos1 - dr_new)
                            
                            # Compute new objective function
                            if inversion_type == 'L2':
                                data_weighted_new = _matvec(self.Wd_sq, dataerror_new)
                                fdert_new = _quad(data_weighted_new, dataerror_new)
                                model_term_new = _ttm(self.Wm, mr1)
                                fmert_new = Lambda * _quad(model_term_new, mr1)
                                temp_term_new = _ttm(self.Wt, mr1)
                                ftert_new = alpha * _quad(temp_term_new, mr1)
                            else:  # L1
                                data_weighted_new = _apply_data_weights(Rd, dataerror_new)
                                fdert_new = _quad(data_weighted_new, dataerror_new)
                                model_diff_new = _matvec(self.Wm, mr1)
                                model_weighted_new = _matvec(Rs, model_diff_new)
                                fmert_new = Lambda * _quad(model_weighted_new, model_diff_new)
                                temp_diff_new = _matvec(self.Wt, mr1)
                                temp_weighted_new = _matvec(Rt, temp_diff_new)
                                ftert_new = alpha * _quad(temp_weighted_new, temp_diff_new)
                            
                            ftot_new = fdert_new + fmert_new + ftert_new
                            
                            if ftot_new < ftot:
                                best_f = ftot_new
                                best_mr = mr1.copy()
                                success = True
                                break
                                
                        except Exception as e:
                            if verbose:
                                print(f"Line search iteration {iarm} failed: {str(e)}")

                        mu_LS *= 0.5
                
                # Update model
                if success:
                    mr = best_mr
                    if Lambda > self.parameters['lambda_min']:
                        Lambda *= self.parameters['lambda_rate']
                else:
                    # Take conservative step along negative gradient
                    mr = mr - 0.01 * gc_r / np.linalg.norm(gc_r)
                    mr = np.clip(mr, min_mr, max_mr)
            
            # Check IRLS convergence
            if inversion_type in ['L1', 'L1L2'] and irls_iter > 0:
                irls_change = np.linalg.norm(mr - mr_previous) / np.linalg.norm(mr_previous)
                if verbose:
                    print(f"IRLS relative change: {irls_change}")
                if irls_change < irls_tol or chi2_ert < target_chi2:
                    if verbose:
                        print(f"IRLS converged after {irls_iter + 1} iterations")
                    break
            
            if inversion_type in ['L1', 'L1L2']:
                mr_previous = mr.copy()
        
        # Process final results
        # Reshape to (cells, timesteps)
        final_model = np.reshape(mr, (-1, self.size), order='F').astype(self.dtype, copy=False)
        final_model = np.exp(final_model).astype(self.dtype if self.use_sparse else np.float64, copy=False)
        
        # Compute coverage for middle time step
        mid_idx = self.size // 2
        dr = self.fwd_operators[mid_idx].response(pg.Vector(final_model[:, mid_idx]))
        self.fwd_operators[mid_idx].createJacobian(pg.Vector(final_model[:, mid_idx]))
        
        covTrans = pg.core.coverageDCtrans(
            self.fwd_operators[mid_idx].jacobian(), 
            1.0 / dr,
            1.0 / pg.Vector(final_model[:, mid_idx])
        )
        
        paramSizes = np.zeros(len(final_model[:, mid_idx]))
        mesh2 = self.fwd_operators[mid_idx].paraDomain
        
        for c in mesh2.cells():
            paramSizes[c.marker()] += c.size()
            
        FinalJ = np.log10(covTrans / paramSizes)
        
        # Store results
        result.final_models = final_model
        result.all_coverage = [FinalJ.copy() for _ in range(self.size)]
        result.mesh = mesh2
        result.all_chi2 = Err_tot
        # Why the loop ended, so a caller driving lambda can tell "this lambda is
        # spent" apart from "this run ran out of iterations".
        result.meta['stop_reason'] = stop_reason
        result.meta['iterations'] = len(Err_tot)
        result.meta['chi2'] = float(Err_tot[-1][0]) if Err_tot else float('nan')
        result.meta['lambda'] = float(self.parameters['lambda_val'])
        result.meta['final_lambda'] = float(Lambda)
        result.meta['chi2_history'] = [float(row[0]) for row in Err_tot]

        if verbose:
            print('End of inversion')
        return result


# Artifact/export orchestration promoted from qt_apps. This public module owns
# the API while the private sibling keeps those helpers separate from the
# numerical TimeLapseERTInversion class.
from ._time_lapse_workflow import (  # noqa: E402
    BackendUnavailable,
    build_timelapse_config,
    default_times,
    run_timelapse_ert,
)
