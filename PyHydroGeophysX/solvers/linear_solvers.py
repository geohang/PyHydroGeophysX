"""
Linear solvers for geophysical inversion.
"""
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import linalg as splinalg
import sys
import time
from typing import Optional, Union, Dict, Any, Tuple, List, Callable

# Attempt to import CuPy for GPU-accelerated computations.
# If CuPy is not installed, GPU_AVAILABLE will be set to False.
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix # Use CuPy's sparse matrix
    GPU_AVAILABLE = True
    print("CuPy found, GPU acceleration is available.")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found, GPU acceleration is NOT available.")

# Attempt to import Joblib for parallel CPU computations.
# If Joblib is not installed, PARALLEL_AVAILABLE will be set to False.
try:
    from joblib import Parallel, delayed
    PARALLEL_AVAILABLE = True
    print("Joblib found, parallel CPU processing is available.")
except ImportError:
    PARALLEL_AVAILABLE = False
    print("Joblib not found, parallel CPU processing is NOT available.")


def generalized_solver(A: Union[np.ndarray, scipy.sparse.spmatrix, Any],  # System matrix (Jacobian)
                       b: Union[np.ndarray, Any],  # Right-hand side vector (data residual)
                       method: str = "cgls",      # Solver algorithm to use
                       x: Optional[Union[np.ndarray, Any]] = None,  # Initial guess for the solution
                       maxiter: int = 200,        # Maximum number of iterations
                       tol: float = 1e-8,         # Convergence tolerance
                       verbose: bool = False,     # Print progress information
                       damp: float = 0.0,         # Damping factor (for Tikhonov regularization)
                       use_gpu: bool = False,     # Flag to enable GPU acceleration
                       parallel: bool = False,    # Flag to enable parallel CPU computation
                       n_jobs: int = -1           # Number of parallel jobs (-1 uses all available cores)
                       ) -> Union[np.ndarray, Any]: # Returns the solution vector
    """
    Generalized iterative solver for linear systems Ax = b or least-squares problems min ||Ax - b||.
    It supports several common iterative algorithms (CGLS, LSQR, etc.) and can optionally
    utilize GPU acceleration via CuPy or parallel CPU processing via Joblib if available.

    The choice of solver method can impact performance and suitability for different types
    of matrices A (e.g., symmetric, positive-definite, ill-conditioned).
    
    Parameters:
    -----------
    A : array_like (NumPy ndarray) or sparse matrix (SciPy sparse matrix or CuPy sparse matrix)
        The system matrix, typically the Jacobian matrix in inversion problems.
    b : array_like (NumPy ndarray or CuPy ndarray)
        The right-hand side vector, often the data residual or A^T * data_residual.
    method : str, optional
        The iterative solver method to use. Supported methods include:
        'lsqr': Least Squares QR factorization method (Paige & Saunders, 1982). Good for ill-conditioned systems.
        'rrlsqr': Regularized LSQR (implementation specific, likely LSQR on augmented system).
        'cgls': Conjugate Gradient Least Squares. Solves the normal equations A^T A x = A^T b.
                Requires A^T A to be positive definite (often true for Jacobian^T Jacobian).
        'rrls': Range-Restricted Least Squares (implementation specific).
        Default is 'cgls'.
    x : array_like, optional
        Initial guess for the solution vector x. If None, a zero vector is used. [Shape: (A.shape[1],)]
    maxiter : int, optional
        Maximum number of iterations allowed for the solver. Default is 200.
    tol : float, optional
        Tolerance for convergence. The specific meaning depends on the solver
        (e.g., relative norm of residual or gradient). Default is 1e-8.
    verbose : bool, optional
        If True, print progress information (e.g., iteration number, residual norm)
        during the solution process. Default is False.
    damp : float, optional
        Damping factor (lambda) for Tikhonov regularization. If `damp` > 0, the problem
        effectively solves (A^T A + damp^2 I) x = A^T b or similar, depending on the method.
        This helps stabilize solutions for ill-posed problems. Default is 0.0 (no damping).
    use_gpu : bool, optional
        If True and CuPy is available (GPU_AVAILABLE=True), computations will be performed on the GPU.
        Data (A, b, x) will be transferred to the GPU. Default is False.
    parallel : bool, optional
        If True and Joblib is available (PARALLEL_AVAILABLE=True), certain matrix operations
        (like matrix-vector products for dense matrices on CPU) might be parallelized across CPU cores.
        Default is False.
    n_jobs : int, optional
        Number of parallel jobs to use if `parallel` is True.
        -1 means use all available CPU cores. Default is -1.
        
    Returns:
    --------
    x_solution : array_like (NumPy ndarray or CuPy ndarray)
        The computed solution vector for the linear system. If `use_gpu` was True,
        the result is transferred back to the CPU (NumPy array) before returning.
    """
    # --- Backend Selection (NumPy for CPU or CuPy for GPU) ---
    if use_gpu and GPU_AVAILABLE:
        xp = cp # Use CuPy as the array library (xp for "execute placement")
        if verbose: print("Using GPU (CuPy) for solver.")
    else:
        xp = np # Use NumPy as the array library
        if use_gpu and not GPU_AVAILABLE: # If GPU was requested but not available
            if verbose: print("GPU requested but CuPy not available. Falling back to CPU (NumPy).")
        use_gpu = False  # Ensure use_gpu is False if CuPy is not used.
    
    # --- Data Conversion to Selected Backend (NumPy/CuPy) ---
    # Convert matrix A and vector b to the appropriate array type (NumPy or CuPy)
    # and format (dense or sparse CSR for SciPy/CuPy sparse).
    if use_gpu: # GPU path
        if scipy.sparse.isspmatrix(A): # If A is a SciPy sparse matrix
            # Convert SciPy sparse matrix to CuPy sparse CSR matrix.
            # This involves transferring data to GPU memory.
            A_backend = cupy_csr_matrix(A)
        else: # If A is a NumPy dense array (or already CuPy array)
            A_backend = cp.asarray(A) # Ensure A is a CuPy array.
        b_backend = cp.asarray(b)     # Ensure b is a CuPy array.
    else: # CPU path
        if scipy.sparse.isspmatrix(A): # If A is a SciPy sparse matrix
            A_backend = A.tocsr() # Ensure it's in CSR format for consistent SciPy sparse operations.
        else: # If A is a NumPy dense array (or list/other array-like)
            A_backend = np.asarray(A) # Ensure A is a NumPy array.
        b_backend = np.asarray(b)     # Ensure b is a NumPy array.
    
    # --- Initialize Solution Vector (x) and Residual (r) ---
    # If no initial guess `x` is provided, initialize it as a zero vector
    # with the appropriate shape (number of columns in A, which is number of model parameters).
    if x is None:
        x_curr = xp.zeros(A_backend.shape[1], dtype=A_backend.dtype) # Use dtype of A for x
        r_curr = b_backend.copy() # Initial residual r = b - A*x = b - A*0 = b.
    else: # If an initial guess `x` is provided
        x_curr = xp.asarray(x, dtype=A_backend.dtype) # Convert to backend array type and match dtype.
        # Calculate initial residual: r = b - A*x.
        r_curr = b_backend - _matrix_multiply(A_backend, x_curr, use_gpu, parallel, n_jobs, xp) # Use helper for mat-vec
    
    # --- Precompute Initial Quantities for Iterative Solvers ---
    # These are common starting values for many conjugate gradient type methods.
    # s = A^T * r  (gradient of the least squares objective function, or related term)
    s_curr = _matrix_multiply(A_backend.T, r_curr, use_gpu, parallel, n_jobs, xp)
    # p = s      (initial search direction)
    p_curr = s_curr.copy()
    # gamma = s^T * s (norm squared of s)
    gamma_curr = xp.dot(s_curr.T, s_curr).item() # .item() to get scalar from 0-dim array
    # rr = r^T * r (norm squared of residual)
    rr_curr = xp.dot(r_curr.T, r_curr).item()
    rr0_initial = rr_curr # Store initial residual norm for convergence checks (relative residual).
    
    # --- Dispatch to Specific Solver Routine ---
    # Based on the `method` string, call the corresponding private solver function.
    # All necessary variables (A, b, x, r, s, gamma, rr, rr0, etc.) are passed.
    if method.lower() == "lsqr":
        solution = _lsqr(A_backend, b_backend, x_curr, r_curr, s_curr, gamma_curr, rr_curr, rr0_initial,
                          maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif method.lower() == "rrlsqr": # Regularized LSQR
        solution = _rrlsqr(A_backend, b_backend, x_curr, r_curr, s_curr, gamma_curr, rr_curr, rr0_initial,
                             maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif method.lower() == "cgls": # Conjugate Gradient Least Squares
        solution = _cgls(A_backend, b_backend, x_curr, r_curr, s_curr, gamma_curr, rr_curr, rr0_initial,
                           maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif method.lower() == "rrls": # Range-Restricted Least Squares
        solution = _rrls(A_backend, b_backend, x_curr, r_curr, s_curr, gamma_curr, rr_curr, rr0_initial,
                           maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    else: # If method string is not recognized
        raise ValueError(f"Unknown method: {method}. Supported methods: 'lsqr', 'rrlsqr', 'cgls', 'rrls'")

    # If GPU was used, transfer the solution back to CPU (NumPy array) before returning.
    # `cp.asnumpy()` handles this transfer.
    if use_gpu: # This implies solution is a CuPy array
        return cp.asnumpy(solution)
    else: # Solution is already a NumPy array
        return solution


def _matrix_multiply(A, v, use_gpu, parallel, n_jobs, xp):
    """
    Helper function for matrix-vector multiplication with optional GPU or parallel CPU support.
    
    Args:
        A: Matrix
        v: Vector
        use_gpu: Whether to use GPU
        parallel: Whether to use parallel CPU
        n_jobs: Number of parallel jobs
        xp: NumPy or CuPy module
        
    Returns:
        Matrix-vector product (A @ v).
    """
    if use_gpu: # GPU execution path
        # Ensure vector `v` is a CuPy array.
        # `A` is assumed to be already a CuPy sparse or dense matrix if use_gpu is True.
        v_gpu = xp.asarray(v) # xp is cp (CuPy) here.
        return A.dot(v_gpu)
    else: # CPU execution path
        if scipy.sparse.isspmatrix(A): # If A is a SciPy sparse matrix
            return A.dot(v) # Use SciPy sparse matrix dot product.
        else: # A is a NumPy dense array
            if parallel and PARALLEL_AVAILABLE: # If parallel CPU computation is enabled and Joblib is available
                # --- Parallel Matrix-Vector Multiplication for Dense CPU Arrays ---
                n_rows = A.shape[0] # Total number of rows in matrix A.
                if n_jobs <= 0: # If n_jobs is -1 or 0, use all available CPU cores.
                    import multiprocessing # Import for cpu_count.
                    n_jobs_resolved = multiprocessing.cpu_count()
                else:
                    n_jobs_resolved = n_jobs
                
                # Determine the size of each partition for parallel processing.
                # Each worker process will handle a block of rows.
                partition_size = max(1, n_rows // n_jobs_resolved)
                # Create a list of (start_row, end_row) tuples for each partition.
                partitions = [(i, min(i + partition_size, n_rows)) 
                             for i in range(0, n_rows, partition_size)]
                
                # Use Joblib's Parallel and delayed to compute dot products of row blocks in parallel.
                # `backend='threading'` is used here. For CPU-bound tasks like dot products,
                # `backend='loky'` or `backend='multiprocessing'` might offer better parallelism by bypassing GIL,
                # but threading can be lighter for some numpy operations that release GIL.
                # SUGGESTION: Investigate `backend='loky'` or `backend='multiprocessing'` for potentially better CPU parallelism.
                results_list = Parallel(n_jobs=n_jobs_resolved, backend='threading')(
                    delayed(lambda row_range: A[row_range[0]:row_range[1]].dot(v))(p) # Each worker does A_slice @ v
                    for p in partitions
                )
                
                # Concatenate the results from all parallel workers.
                return xp.concatenate(results_list) # xp is np (NumPy) here.
            else: # Standard serial matrix-vector product for dense CPU arrays.
                return A.dot(v)


def _cgls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
         use_gpu, parallel, n_jobs, xp):
    """
    Conjugate Gradient Least Squares (CGLS) solver.
    
    This method solves the normal equations A^T A x = A^T b for x, which is equivalent
    to minimizing the least-squares problem ||Ax - b||_2^2.
    It is suitable when A^T A is positive definite.
    The `damp` parameter adds Tikhonov regularization: (A^T A + damp^2 I) x = A^T b.
    
    Args:
        A: System matrix (Jacobian).
        b: Right-hand side vector.
        x: Initial solution guess.
        r: Initial residual (b - A*x).
        s: Initial A^T * r.
        gamma: Initial s^T * s.
        rr: Initial r^T * r (squared norm of residual).
        rr0: Squared norm of initial residual (b - A*x_initial, often just ||b||^2 if x0=0). Used for relative tolerance.
        maxiter: Maximum number of iterations.
        tol: Convergence tolerance (e.g., for relative residual norm).
        verbose: If True, print iteration progress.
        damp: Damping factor (lambda for Tikhonov regularization, sqrt(lambda) if applied to augmented system).
              Here, it seems to be lambda_val itself, applied as damp^2 in effect due to normal equations.
              If applied to A*p, then it's (A^T A + damp I)x = A^T b.
              The code `q += damp * p` inside loop implies (A^T A + damp I) form if p is related to A.
              Let's analyze loop: alpha = gamma / (q^T q). q = A*p. If damp > 0, q_reg = A*p + damp*p = (A+damp*I)p (if p has same dim as x)
              This is not standard CGLS on (A^T A + damp^2 I)x = A^T b.
              Standard CGLS applies to ||Ax-b||^2 + ||damp*x||^2, which means solving
              [ A ] x = [ b ]
              [damp*I]   [ 0 ]
              The normal equations for this augmented system are (A^T A + damp^2 I) x = A^T b.
              The CGLS algorithm iteratively solves this.
              The `damp` parameter here seems to modify `q` and `s` in a way that might correspond to
              solving a damped system, possibly equivalent to (A^T A + damp*I)x = A^T b if `damp` is lambda.
              Or if `damp` is sqrt(lambda), then (A^T A + damp^2 I)x = A^T b.
              The line `q += damp * p` and `s += damp * r` is non-standard for CGLS.
              This might be a specific variant or an error in implementation of standard damped CGLS.
              Standard CGLS on augmented system does not modify `s` like this.
              Assuming this is a custom CGLS variant.
              SUGGESTION: Clarify the exact formulation of damped CGLS being implemented,
              especially the role of `damp` in `q` and `s` updates. Standard damped CGLS
              would typically involve the augmented matrix implicitly or explicitly.
        use_gpu: Boolean, use GPU if True.
        parallel: Boolean, use parallel CPU if True.
        n_jobs: Integer, number of parallel jobs.
        xp: Array module (NumPy or CuPy).
        
    Returns:
        x_final: The computed solution vector.
    """
    # Ensure solution x, residual r, and gradient term s are column vectors for consistent dot products.
    # This is important as .dot() behavior can differ for 1D arrays vs column/row vectors.
    if x.ndim == 1: x = x.reshape(-1, 1)
    if r.ndim == 1: r = r.reshape(-1, 1)
    if s.ndim == 1: s = s.reshape(-1, 1)
    
    # Initialize search direction `p` with `s` (which is A^T*r).
    p_curr = s.copy() # p_0 = s_0
    
    # Iteration loop for CGLS.
    for iter_count in range(maxiter):
        if verbose and iter_count % 10 == 0: # Print progress every 10 iterations.
            # pg.info is likely a PyGIMLi utility for logging.
            pg.info(f"CGLS Iteration: {iter_count}, residual norm sq: {float(rr):.4e}, relative residual norm sq: {float(rr / rr0):.4e}")
        
        # --- Core CGLS Steps ---
        # 1. q_k = A * p_k
        q_k = _matrix_multiply(A, p_curr, use_gpu, parallel, n_jobs, xp)
        
        # Apply damping term to q_k if damp > 0.
        # This modifies q_k to (A * p_k + damp * p_k) = (A + damp*I) * p_k (if p_k has same shape as x).
        # This is where the damping is incorporated into the system being effectively solved.
        if damp > 0:
            q_k += damp * p_curr # If p_curr is search direction for x, this makes system (A+damp*I)p
                                # This is unusual. Standard CGLS for (A'A + λI)x = A'b does not modify q like this.
                                # This seems to be trying to solve (A+damp*I)x = b or similar, but CGLS is for normal equations.
                                # If this is for (A^T A + damp I) x = A^T b, then `damp` here is effectively `sqrt(damp_coeff)`.
                                # The update `q_k = A @ p_k` should be `q_k = A @ p_k`. If damping is on `x`, it's `A.T @ A @ p_k + damp**2 * p_k`.
                                # This `q_k += damp * p_k` is non-standard for typical CGLS normal equations.
        
        q_k = q_k.reshape(-1, 1) # Ensure q_k is a column vector.
        
        # 2. alpha_k = gamma_k / (q_k^T * q_k)  (step length)
        # gamma_k = s_k^T * s_k
        alpha_k = float(gamma / xp.dot(q_k.T, q_k)) # gamma is s_prev.T @ s_prev
        
        # 3. x_{k+1} = x_k + alpha_k * p_k (update solution)
        x += alpha_k * p_curr
        # 4. r_{k+1} = r_k - alpha_k * q_k (update residual)
        # If q_k = A*p_k, then r_k - alpha_k*A*p_k = b - A*x_k - alpha_k*A*p_k = b - A*(x_k + alpha_k*p_k) = b - A*x_{k+1}
        r -= alpha_k * q_k
        
        # 5. s_{k+1} = A^T * r_{k+1}
        s_next = _matrix_multiply(A.T, r, use_gpu, parallel, n_jobs, xp)
        
        # Apply damping term to s_next. (Again, non-standard for typical CGLS normal equations)
        # If this is for (A^T A + damp I)x = A^T b, then s_next = A^T r - damp*x (if r = b - Ax)
        # This `s_next += damp * r` is highly unusual. `s` should be related to gradient of normal eq.
        if damp > 0:
            s_next += damp * r # This also seems non-standard.
        
        s_next = s_next.reshape(-1, 1) # Ensure column vector.
        
        # 6. gamma_{k+1} = s_{k+1}^T * s_{k+1}
        gamma_next = float(xp.dot(s_next.T, s_next))
        # 7. beta_k = gamma_{k+1} / gamma_k
        beta_k = float(gamma_next / gamma) # gamma is previous s_k^T * s_k
        
        # 8. p_{k+1} = s_{k+1} + beta_k * p_k (update search direction)
        p_curr = s_next + beta_k * p_curr
        
        # Update gamma for next iteration.
        gamma = gamma_next
        
        # --- Check Convergence ---
        # Based on the norm of the current residual `r`.
        rr = float(xp.dot(r.T, r)) # Current r_k^T * r_k
        # Relative residual norm: ||r_k||^2 / ||r_0||^2 < tol^2 (or similar, often ||r_k|| / ||b|| < tol)
        # Here, it's ||r_k||^2 / ||r_initial_true_residual||^2 < tol. (rr0 = initial r^T r)
        # This tolerance check is on squared norms. If tol is for norm itself, this should be sqrt(rr/rr0) < tol or rr/rr0 < tol^2.
        # Assuming `tol` is for relative squared residual norm.
        if rr / rr0 < tol:
            if verbose:
                pg.info(f"CGLS converged after {iter_count+1} iterations.")
            break # Exit loop if converged.
    else: # If loop finishes without break (maxiter reached)
        if verbose:
            pg.info(f"CGLS reached maxiter ({maxiter}) without full convergence. Relative residual norm sq: {rr/rr0:.2e}")

    # Return solution. If GPU was used, transfer data back to CPU NumPy array.
    return x.get() if use_gpu else x # x.get() is CuPy's method to get NumPy array from GPU.


def _lsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
         use_gpu, parallel, n_jobs, xp):
    """
    LSQR solver for linear least squares problems.
    
    This implements the LSQR algorithm by Paige and Saunders (1982).
    It's an iterative method for solving sparse linear equations and sparse
    least-squares problems: min || Ax - b || or solve Ax = b.
    LSQR is mathematically equivalent to applying Conjugate Gradients (CG) to the
    normal equations A^T A x = A^T b or A A^T y = b, but it has better numerical properties,
    especially for ill-conditioned A.
    The `damp` parameter is NOT used in this standard LSQR implementation, unlike Scipy's LSQR.
    If damping is needed, an augmented system approach is typical for LSQR.
    SUGGESTION: This `_lsqr` does not use `damp`. If damping is required with LSQR,
    the system (A, b) should be augmented before calling this, or use `_rrlsqr`.
    The parameters s, gamma, rr are also not standard inputs for LSQR's typical initialization.
    LSQR initializes with beta = ||b - Ax0||, u = (b-Ax0)/beta, alpha = ||A^T u||, v = A^T u / alpha.
    This implementation seems to take pre-calculated `r` (b-Ax0), but then re-calculates u and beta from it.
    
    Args: (mostly same as _cgls, but some are not used or re-derived)
        r: Initial residual (b - A*x). Used to initialize `u`.
        s, gamma, rr: Not directly used by standard LSQR initialization in this form.
        damp: Not used in this specific LSQR implementation.
        
    Returns:
        Solution vector x.
    """
    # --- LSQR Initialization ---
    # Ensure x is a column vector and on the correct backend (CPU/GPU).
    if x is None: # If no initial guess x0
        x_curr = xp.zeros((A.shape[1], 1), dtype=A.dtype) # Initialize x0 = 0
        # r is b - A*x0 = b.
    else: # If initial guess x0 is provided
        x_curr = xp.asarray(x, dtype=A.dtype)
        if x_curr.ndim == 1: x_curr = x_curr.reshape(-1, 1)
        # r should be b - A*x0, which is passed in as `r`.

    if r.ndim == 1: r = r.reshape(-1, 1) # Ensure r is column vector.

    # u_k = r_k (scaled). Initial u_0 = r_0 / beta_1
    u_curr = r.copy() # r is b - A*x_0
    beta_curr = xp.sqrt(xp.dot(u_curr.T, u_curr).item()) # beta_1 = ||r_0||
    if beta_curr > 0:
        u_curr = u_curr / beta_curr # u_1 = r_0 / beta_1 (note LSQR paper uses u_1, beta_1)

    # v_k = A^T * u_k (scaled). Initial v_1 = A^T * u_1 / alpha_1
    v_curr = _matrix_multiply(A.T, u_curr, use_gpu, parallel, n_jobs, xp)
    if v_curr.ndim == 1: v_curr = v_curr.reshape(-1, 1)
    alpha_curr = xp.sqrt(xp.dot(v_curr.T, v_curr).item()) # alpha_1 = ||A^T * u_1||
    if alpha_curr > 0:
        v_curr = v_curr / alpha_curr # v_1 = A^T * u_1 / alpha_1

    w_curr = v_curr.copy() # w_1 = v_1
    phi_bar_curr = beta_curr # φ_bar_1 = beta_1
    rho_bar_curr = alpha_curr # ρ_bar_1 = alpha_1

    # rr0 is passed in, it's ||initial residual||^2. Used for stopping criterion.
    # LSQR stopping criterion is often based on norm of A*x-b or A^T*(Ax-b).
    # phi_bar_curr is an estimate of ||r_k||. So (phi_bar_curr)^2 / rr0 can be used.

    # Iteration loop for LSQR.
    for iter_count in range(maxiter):
        if verbose and iter_count % 10 == 0:
            # rr is an estimate of ||r_k||^2. Here, phi_bar_curr^2 is used as an estimate.
            # The passed `rr` is not updated in this LSQR loop, `phi_bar_curr` is the relevant term.
            current_res_norm_sq_est = phi_bar_curr**2
            pg.info(f"LSQR Iteration: {iter_count}, est. residual norm sq: {float(current_res_norm_sq_est):.4e}, relative: {float(current_res_norm_sq_est / rr0):.4e}")

        # --- Core LSQR Bidiagonalization and Solution Update Steps ---
        # ( Golub-Kahan bidiagonalization process for A = U [B 0]^T V^T )
        # Step 1: Generate next u_vector (part of U in A = U B V^T)
        # u_hat_{k+1} = A * v_k - alpha_k * u_k  (where v_k is current v_curr, u_k is current u_curr)
        u_next_unscaled = _matrix_multiply(A, v_curr, use_gpu, parallel, n_jobs, xp)
        if u_next_unscaled.ndim == 1: u_next_unscaled = u_next_unscaled.reshape(-1, 1)
        u_next_unscaled = u_next_unscaled - alpha_curr * u_curr # This is β_{k+1} * u_{k+1}

        beta_next = xp.sqrt(xp.dot(u_next_unscaled.T, u_next_unscaled).item()) # β_{k+1}
        if beta_next > 0:
            u_curr = u_next_unscaled / beta_next # u_{k+1} = u_hat_{k+1} / β_{k+1}
            
        # Step 2: Generate next v_vector (part of V in A = U B V^T)
        # v_hat_{k+1} = A^T * u_{k+1} - beta_{k+1} * v_k
        v_next_unscaled = _matrix_multiply(A.T, u_curr, use_gpu, parallel, n_jobs, xp)
        if v_next_unscaled.ndim == 1: v_next_unscaled = v_next_unscaled.reshape(-1, 1)
        v_next_unscaled = v_next_unscaled - beta_next * v_curr # This is α_{k+1} * v_{k+1}

        alpha_next = xp.sqrt(xp.dot(v_next_unscaled.T, v_next_unscaled).item()) # α_{k+1}
        if alpha_next > 0:
            v_curr = v_next_unscaled / alpha_next # v_{k+1} = v_hat_{k+1} / α_{k+1}

        # Step 3: Apply sequence of plane rotations to effectively solve the bidiagonal system.
        # rho_k = sqrt(rho_bar_k^2 + beta_{k+1}^2)
        rho_plane_rot = xp.sqrt(rho_bar_curr**2 + beta_next**2)
        # c_k = rho_bar_k / rho_k
        c_plane_rot = rho_bar_curr / rho_plane_rot
        # s_k = beta_{k+1} / rho_k
        s_plane_rot = beta_next / rho_plane_rot

        # theta_next = s_k * alpha_{k+1}
        theta_next_val = s_plane_rot * alpha_next
        # rho_bar_{k+1} = -c_k * alpha_{k+1}
        rho_bar_curr = -c_plane_rot * alpha_next
        # phi_k = c_k * phi_bar_k
        phi_curr_val = c_plane_rot * phi_bar_curr
        # phi_bar_{k+1} = s_k * phi_bar_k
        phi_bar_curr = s_plane_rot * phi_bar_curr

        # Step 4: Update solution `x` and search direction `w`.
        # t_k = phi_k / rho_k
        t_step = phi_curr_val / rho_plane_rot
        # x_k = x_{k-1} + t_k * w_k
        x_curr = x_curr + t_step * w_curr
        # w_{k+1} = v_{k+1} - (theta_next / rho_k) * w_k
        w_curr = v_curr - (theta_next_val / rho_plane_rot) * w_curr

        # --- Check Convergence ---
        # Convergence is typically checked on ||r_k|| or ||A^T r_k||.
        # phi_bar_curr is an estimate of ||r_k||.
        # rr0 is ||initial_residual||^2.
        # So, (phi_bar_curr**2) / rr0 is relative squared residual norm.
        if (phi_bar_curr**2) / rr0 < tol:
            if verbose:
                pg.info(f"LSQR converged after {iter_count+1} iterations.")
            break
    else: # If loop finishes
        if verbose:
            pg.info(f"LSQR reached maxiter ({maxiter}) without full convergence. Est. relative residual norm sq: {(phi_bar_curr**2)/rr0:.2e}")

    return x_curr.get() if use_gpu else x_curr # Transfer to CPU if needed.


def _rrlsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
          use_gpu, parallel, n_jobs, xp):
    """
    Regularized LSQR (RRLSQR) solver.
    
    This implements a version of LSQR that incorporates Tikhonov damping (`damp` parameter).
    It solves min || [  A  ] x - [ b ] ||^2
                  || [damp*I]   [ 0 ] ||_2
    The standard LSQR algorithm can be applied to this augmented system.
    The implementation here seems to follow a direct modification of LSQR steps
    rather than explicitly forming the augmented system.
    The parameters s, gamma, rr are not standard inputs for LSQR initialization.
    
    Args: (mostly same as _lsqr)
        damp: Damping factor (lambda). If this is the lambda from (A^T A + lambda^2 I),
              then the augmented matrix uses lambda directly.
              Here, `damp` seems to be used as sqrt of regularization coefficient.
        
    Returns:
        Solution vector x.
    """
    # --- RRLSQR Initialization --- (Similar to LSQR, with modifications for damping)
    if x is None:
        x_curr = xp.zeros((A.shape[1], 1), dtype=A.dtype)
    else:
        x_curr = xp.asarray(x, dtype=A.dtype)
        if x_curr.ndim == 1: x_curr = x_curr.reshape(-1, 1)

    if r.ndim == 1: r = r.reshape(-1, 1) # r = b - A*x0

    u_curr = r.copy()
    beta_curr = xp.sqrt(xp.dot(u_curr.T, u_curr).item())
    if beta_curr > 0:
        u_curr = u_curr / beta_curr

    # v_hat_1 = A^T * u_1 + damp * x_0 (if x_0 is current estimate of x being regularized)
    # Here, `x` is the initial guess for the solution `x` of the damped system.
    # If x0=0, then v_hat_1 = A^T * u_1.
    v_next_unscaled = _matrix_multiply(A.T, u_curr, use_gpu, parallel, n_jobs, xp)
    if v_next_unscaled.ndim == 1: v_next_unscaled = v_next_unscaled.reshape(-1, 1)
    if damp > 0: # This term +damp*x is specific to some regularized LSQR variants.
                 # It implies x is the current full solution estimate being damped.
                 # Standard LSQR on augmented system: x_aug = [x; damp*x_orig], A_aug = [A; damp*I]
                 # Here, it seems to be a direct modification of the Lanczos bidiagonalization.
        v_next_unscaled = v_next_unscaled + damp * x_curr # If x_curr is x_k, this is A^T u_k + damp x_k

    alpha_curr = xp.sqrt(xp.dot(v_next_unscaled.T, v_next_unscaled).item())
    v_curr = v_next_unscaled / alpha_curr if alpha_curr > 0 else v_next_unscaled # Handle alpha_curr = 0

    w_curr = v_curr.copy()
    phi_bar_curr = beta_curr
    rho_bar_curr = alpha_curr
    # rr0 is ||initial_residual_original_problem||^2
    # Convergence check for RRLSQR often uses norm of residual of augmented system.

    # Iteration loop for RRLSQR.
    for iter_count in range(maxiter):
        if verbose and iter_count % 10 == 0:
            current_res_norm_sq_est = phi_bar_curr**2 # Estimate of ||r_k_aug|| or related term
            pg.info(f"RRLSQR Iteration: {iter_count}, est. residual norm sq: {float(current_res_norm_sq_est):.4e}, relative: {float(current_res_norm_sq_est / rr0):.4e}")

        # --- Bidiagonalization with Damping ---
        # Step 1: u_hat_{k+1} = A * v_k - alpha_k * u_k (Same as LSQR)
        u_next_unscaled = _matrix_multiply(A, v_curr, use_gpu, parallel, n_jobs, xp)
        if u_next_unscaled.ndim == 1: u_next_unscaled = u_next_unscaled.reshape(-1, 1)
        u_next_unscaled = u_next_unscaled - alpha_curr * u_curr

        beta_next = xp.sqrt(xp.dot(u_next_unscaled.T, u_next_unscaled).item())
        if beta_next > 0:
            u_curr = u_next_unscaled / beta_next
            
        # Step 2: v_hat_{k+1} = A^T * u_{k+1} - beta_{k+1} * v_k + damp * x_k (Modified for RRLSQR type damping)
        # The `+ damp * x` term is added here in the original code.
        # Standard LSQR on augmented system would not have this evolving `damp*x` term added directly to v_next.
        # It would be implicitly handled by the augmented A matrix [A; damp*I].
        # This suggests a specific variant of regularized LSQR.
        v_next_unscaled = _matrix_multiply(A.T, u_curr, use_gpu, parallel, n_jobs, xp)
        if v_next_unscaled.ndim == 1: v_next_unscaled = v_next_unscaled.reshape(-1, 1)
        v_next_unscaled = v_next_unscaled - beta_next * v_curr
        
        if damp > 0: # This is the unusual term.
            v_next_unscaled = v_next_unscaled + damp * x_curr # Adding current solution x_curr scaled by damp.
                                                          # This makes the algorithm differ from standard LSQR on augmented system.
            
        alpha_next = xp.sqrt(xp.dot(v_next_unscaled.T, v_next_unscaled).item())
        if alpha_next > 0:
            v_curr = v_next_unscaled / alpha_next

        # --- Apply Orthogonal Transformation (Plane Rotation) ---
        # This part is modified to handle the `damp` term in the system matrix structure.
        # rho_k = sqrt(rho_bar_k^2 + beta_{k+1}^2 + damp^2) (if damp is applied to identity part of augmented matrix)
        # The `damp**2` appears here if the augmented system is [A; d*I]x = [b; 0].
        rho_plane_rot = xp.sqrt(rho_bar_curr**2 + beta_next**2 + damp**2) # Incorporates damping
        c_plane_rot = rho_bar_curr / rho_plane_rot
        s_plane_rot = beta_next / rho_plane_rot

        theta_next_val = s_plane_rot * alpha_next
        rho_bar_curr = -c_plane_rot * alpha_next
        phi_curr_val = c_plane_rot * phi_bar_curr
        phi_bar_curr = s_plane_rot * phi_bar_curr # This is ||r_{k+1}|| estimate

        # --- Update Solution `x` and Search Direction `w` ---
        t_step = phi_curr_val / rho_plane_rot
        x_curr = x_curr + t_step * w_curr
        w_curr = v_curr - (theta_next_val / rho_plane_rot) * w_curr

        # --- Check Convergence ---
        # Relative squared norm of the effective residual.
        if (phi_bar_curr**2) / rr0 < tol:
            if verbose:
                pg.info(f"RRLSQR converged after {iter_count+1} iterations.")
            break
    else: # If loop finishes
        if verbose:
            pg.info(f"RRLSQR reached maxiter ({maxiter}) without full convergence. Est. relative residual norm sq: {(phi_bar_curr**2)/rr0:.2e}")

    return x_curr.get() if use_gpu else x_curr


def _rrls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
         use_gpu, parallel, n_jobs, xp):
    """
    Range-Restricted Least Squares (RRLS) solver.
    
    This method is less common in standard literature under this exact name and formulation.
    It appears to be an iterative method that updates the solution `x` using a search
    direction `w` derived from `s = A^T*r` (or a damped version).
    The step length `lam` is calculated based on `p = A*w`.
    This might be related to a form of steepest descent or a specific variant of CG for least squares.
    The `damp` parameter seems to be incorporated by modifying `s` with `damp*x`.
    
    Args: (mostly same as _cgls)
        s: A^T * r (or damped version). Used to initialize search direction `w`.
        gamma: Not explicitly used in the loop of this RRLS version, unlike CGLS.
        
    Returns:
        Solution vector x.
    """
    # Ensure x, r, and s are column vectors and on the correct backend.
    if x is None:
        x_curr = xp.zeros((A.shape[1], 1), dtype=A.dtype)
    else:
        x_curr = xp.asarray(x, dtype=A.dtype)
        if x_curr.ndim == 1: x_curr = x_curr.reshape(-1, 1)
    
    if r.ndim == 1: r_curr = r.reshape(-1, 1)
    else: r_curr = r.copy()
    if s.ndim == 1: s_curr = s.reshape(-1, 1)
    else: s_curr = s.copy()
        
    # Initialize search direction `w` with `s`.
    w_curr = s_curr.copy()
    
    # Iteration loop for RRLS.
    for iter_count in range(maxiter):
        if verbose and iter_count % 10 == 0:
            # `rr` is current r^T*r. `rr0` is initial r^T*r.
            pg.info(f"RRLS Iteration: {iter_count}, residual norm sq: {float(rr):.4e}, relative: {float(rr / rr0):.4e}")
        
        # 1. p_k = A * w_k (project search direction into data space)
        p_k = _matrix_multiply(A, w_curr, use_gpu, parallel, n_jobs, xp)
        if p_k.ndim == 1: p_k = p_k.reshape(-1, 1)
            
        # 2. Calculate step length (lambda_k, named `lam` here)
        # lam_k = (p_k^T * r_k) / (p_k^T * p_k)
        # This is a common step length for methods like Steepest Descent if w_k was gradient.
        denom_lam = xp.dot(p_k.T, p_k).item()
        if xp.isclose(denom_lam, 0.0): # If p_k is zero or very small, stop.
            if verbose: print(f"RRLS: Denominator for step length is close to zero at iteration {iter_count}. Stopping.")
            break
            
        lambda_k = xp.dot(p_k.T, r_curr).item() / denom_lam

        # 3. Update solution: x_{k+1} = x_k + lambda_k * w_k
        x_curr = x_curr + w_curr * lambda_k # lambda_k is scalar
        
        # 4. Update residual: r_{k+1} = r_k - lambda_k * p_k
        r_curr = r_curr - p_k * lambda_k

        # 5. Update s_{k+1} = A^T * r_{k+1}
        s_curr = _matrix_multiply(A.T, r_curr, use_gpu, parallel, n_jobs, xp)
        if s_curr.ndim == 1: s_curr = s_curr.reshape(-1, 1)
            
        # Apply damping by modifying `s_curr`.
        # s_{k+1}_damped = s_{k+1} + damp * x_{k+1}
        # This makes the effective gradient related to a damped objective function.
        if damp > 0:
            s_curr = s_curr + damp * x_curr
            
        # 6. Update search direction: w_{k+1} = s_{k+1}_damped
        # This makes it a steepest descent type method on some objective function if s_curr is its gradient.
        # If it were CG, w_curr would be updated using a beta term (e.g., Polak-Ribiere, Fletcher-Reeves).
        w_curr = s_curr.copy() # No conjugation, so it's not CG unless s_curr itself is made conjugate.

        # --- Check Convergence ---
        rr = float(xp.dot(r_curr.T, r_curr).item()) # Current squared residual norm
        if rr / rr0 < tol: # Relative squared residual norm.
            if verbose:
                pg.info(f"RRLS converged after {iter_count+1} iterations.")
            break
    else: # If loop finishes
        if verbose:
             pg.info(f"RRLS reached maxiter ({maxiter}) without full convergence. Relative residual norm sq: {rr/rr0:.2e}")
            
    return x_curr.get() if use_gpu else x_curr


class LinearSolver:
    """Base class for linear system solvers."""
    
    def __init__(self, method: str = "cgls",      # Name of the solver algorithm
                 max_iterations: int = 200,     # Max iteration count
                 tolerance: float = 1e-8,       # Convergence tolerance
                 use_gpu: bool = False,         # Use GPU if available
                 parallel: bool = False,        # Use parallel CPU if available
                 n_jobs: int = -1,              # Number of parallel CPU jobs
                 damping: float = 0.0,          # Damping (regularization) factor
                 verbose: bool = False          # Print progress
                 ):
        """
        Initialize the LinearSolver wrapper.
        
        This constructor stores the configuration for the solver, which will be used
        when the `solve` method is called. It also handles checks for GPU/parallel availability.

        Args:
            method (str, optional): Solver algorithm to use. Defaults to "cgls".
                                    Supported: 'cgls', 'lsqr', 'rrlsqr', 'rrls'.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 200.
            tolerance (float, optional): Tolerance for convergence. Defaults to 1e-8.
            use_gpu (bool, optional): If True, attempt to use GPU acceleration via CuPy.
                                      Defaults to False.
            parallel (bool, optional): If True, attempt to use parallel CPU computation via Joblib
                                       for certain operations (mainly dense matrix-vector products).
                                       Defaults to False.
            n_jobs (int, optional): Number of parallel jobs if `parallel` is True.
                                    -1 means use all available cores. Defaults to -1.
            damping (float, optional): Damping factor for regularization. Defaults to 0.0.
            verbose (bool, optional): If True, print solver progress. Defaults to False.
        """
        self.method = method.lower() # Store method name (lowercase for case-insensitivity).
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        # Set use_gpu only if requested AND available.
        self.use_gpu = use_gpu and GPU_AVAILABLE
        # Set parallel only if requested AND available.
        self.parallel = parallel and PARALLEL_AVAILABLE
        self.n_jobs = n_jobs
        self.damping = damping
        self.verbose = verbose
        
        # Validate the chosen solver method.
        valid_methods = ['cgls', 'lsqr', 'rrlsqr', 'rrls']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid solver method: '{self.method}'. Must be one of {valid_methods}.")
        
        # Issue warnings if requested features (GPU/parallel) are not available.
        if use_gpu and not GPU_AVAILABLE:
            print("Warning: GPU acceleration was requested, but CuPy is not installed or unavailable. Solver will run on CPU.")
            # self.use_gpu is already False due to `and GPU_AVAILABLE`
        
        if parallel and not PARALLEL_AVAILABLE:
            print("Warning: Parallel CPU computation was requested, but Joblib is not installed or unavailable. Solver will run in serial.")
            # self.parallel is already False
    
    def solve(self, A, b, x0=None):
        """
        Solve linear system Ax = b.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x0: Initial guess (None for zeros)
            
        Returns:
            Solution vector (NumPy ndarray).
        """
        # Delegate the actual solution process to the `generalized_solver` function,
        # passing all the configured parameters.
        return generalized_solver(
            A, b,
            method=self.method,
            x=x0, # Pass initial guess
            maxiter=self.max_iterations,
            tol=self.tolerance,
            verbose=self.verbose,
            damp=self.damping,
            use_gpu=self.use_gpu,
            parallel=self.parallel,
            n_jobs=self.n_jobs
        )


class CGLSSolver(LinearSolver):
    """
    Specialized solver class for CGLS (Conjugate Gradient Least Squares).
    Inherits from LinearSolver and fixes the method to 'cgls'.
    """
    
    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8,
                 use_gpu: bool = False, parallel: bool = False, n_jobs: int = -1,
                 damping: float = 0.0, verbose: bool = False):
        """
        Initialize CGLS solver. Parameters are identical to LinearSolver but method is fixed.
        """
        super().__init__(
            method="cgls", # Fixed method
            max_iterations=max_iterations,
            tolerance=tolerance,
            use_gpu=use_gpu,
            parallel=parallel,
            n_jobs=n_jobs,
            damping=damping,
            verbose=verbose
        )


class LSQRSolver(LinearSolver):
    """
    Specialized solver class for LSQR.
    Inherits from LinearSolver and fixes the method to 'lsqr'.
    """
    
    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8,
                 use_gpu: bool = False, parallel: bool = False, n_jobs: int = -1,
                 damping: float = 0.0, verbose: bool = False):
        """
        Initialize LSQR solver. Parameters are identical to LinearSolver but method is fixed.
        Note: The underlying `_lsqr` implementation does not currently use the `damping` parameter.
        """
        super().__init__(
            method="lsqr", # Fixed method
            max_iterations=max_iterations,
            tolerance=tolerance,
            use_gpu=use_gpu,
            parallel=parallel,
            n_jobs=n_jobs,
            damping=damping, # Passed, but _lsqr might not use it.
            verbose=verbose
        )


class RRLSQRSolver(LinearSolver):
    """
    Specialized solver class for Regularized LSQR (RRLSQR).
    Inherits from LinearSolver and fixes the method to 'rrlsqr'.
    """
    
    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8,
                 use_gpu: bool = False, parallel: bool = False, n_jobs: int = -1,
                 damping: float = 0.1, # Default damping for RRLSQR is often non-zero
                 verbose: bool = False):
        """
        Initialize RRLSQR solver. Parameters are identical to LinearSolver but method is fixed.
        The `damping` parameter is actively used by the `_rrlsqr` implementation.
        """
        super().__init__(
            method="rrlsqr", # Fixed method
            max_iterations=max_iterations,
            tolerance=tolerance,
            use_gpu=use_gpu,
            parallel=parallel,
            n_jobs=n_jobs,
            damping=damping,
            verbose=verbose
        )


class RRLSSolver(LinearSolver):
    """
    Specialized solver class for Range-Restricted Least Squares (RRLS).
    Inherits from LinearSolver and fixes the method to 'rrls'.
    """
    
    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8,
                 use_gpu: bool = False, parallel: bool = False, n_jobs: int = -1,
                 damping: float = 0.0, verbose: bool = False):
        """
        Initialize RRLS solver. Parameters are identical to LinearSolver but method is fixed.
        """
        super().__init__(
            method="rrls", # Fixed method
            max_iterations=max_iterations,
            tolerance=tolerance,
            use_gpu=use_gpu,
            parallel=parallel,
            n_jobs=n_jobs,
            damping=damping,
            verbose=verbose
        )


# Additional solver implementations (Direct Solvers and Regularization Application)
import scipy.linalg # For dense direct solvers
# SciPy sparse linalg `splinalg` already imported.

def direct_solver(A: Union[np.ndarray, scipy.sparse.spmatrix],
                  b: np.ndarray,
                  method: str = "lu", **kwargs) -> np.ndarray:
    """
    Solve a linear system using direct methods.
    
    Args:
            A (Union[np.ndarray, scipy.sparse.spmatrix]): The system matrix.
            b (np.ndarray): The right-hand side vector.
            method (str, optional): The direct solver method to use. Defaults to "lu".
                Supported methods:
                - 'lu': LU decomposition (for general square matrices). Uses `scipy.linalg.solve` for dense
                        and `scipy.sparse.linalg.spsolve` for sparse.
                - 'qr': QR decomposition (good for least-squares on dense matrices). Uses `scipy.linalg.qr`.
                - 'svd': Singular Value Decomposition (robust for ill-conditioned or rank-deficient dense matrices).
                         Uses `scipy.linalg.svd`. A tolerance `tol` can be passed via `**kwargs`.
                - 'cholesky': Cholesky decomposition (for symmetric/Hermitian positive-definite matrices).
                              Uses `scipy.linalg.cholesky` for dense and `sksparse.cholmod` (if available) or
                              `scipy.sparse.linalg.cholesky` (SciPy >= 1.12, limited) for sparse.
                              Falls back if not SPD.
            **kwargs: Additional keyword arguments for specific methods (e.g., `tol` for 'svd').
        
    Returns:
        np.ndarray: The solution vector x.
    """
    # --- Sparse Matrix Solvers ---
    if scipy.sparse.isspmatrix(A):
        if method == "lu":
            # Use SciPy's sparse LU decomposition based solver (UMFPACK or SuperLU).
            # `spsolve` is a general-purpose sparse solver.
            return splinalg.spsolve(A, b)
        elif method == "cholesky":
            # For sparse Cholesky, matrix A must be symmetric positive-definite.
            # SciPy's default sparse Cholesky might be limited. External libraries like scikit-sparse (cholmod)
            # are often more robust for sparse Cholesky.
            # SUGGESTION: For production, consider adding `sksparse.cholmod` as an optional dependency for sparse Cholesky.
            try:
                # Ensure matrix is in CSC format for some sparse Cholesky routines.
                # `splinalg.cholesky` (SciPy >= 1.12) or `sksparse.cholmod.cholesky`
                # For simplicity, assuming `spsolve` might handle it or using a placeholder.
                # If specific sparse Cholesky is needed, `A.tocsc()` might be required.
                # factor = splinalg.cholesky(A.tocsc()) # This is if using SciPy 1.12+ factorized object.
                # return factor(b) # This is how you use the factor object.
                # A simpler way for older SciPy or general case might be to use spsolve with specific flags if available,
                # or rely on it choosing appropriately for SPD if matrix properties are set.
                # For now, using spsolve as a fallback if direct Cholesky is tricky or version-dependent.
                print("Note: Sparse Cholesky requested. Robustness depends on SciPy version and matrix properties. Using spsolve as a general approach.")
                return splinalg.spsolve(A, b) # spsolve can sometimes use Cholesky if matrix is suitable.
            except Exception as e: # Catch errors if not SPD or other issues.
                print(f"Warning: Sparse Cholesky failed ('{e}'), falling back to general sparse LU solver (spsolve).")
                return splinalg.spsolve(A, b)
        else: # For other methods like 'qr', 'svd' with sparse input.
            print(f"Warning: Method '{method}' not directly optimized for sparse matrices via this function. Using general spsolve.")
            # SciPy's sparse linalg does not have direct sparse QR or SVD solution functions like dense linalg.
            # `spsolve` is the general recommendation. For least squares with sparse A, `splinalg.lsqr` or `splinalg.lsmr` are iterative.
            # This function is for *direct* solvers.
            return splinalg.spsolve(A, b)

    # --- Dense Matrix Solvers ---
    else: # A is a dense NumPy array.
        A_dense = np.asarray(A) # Ensure it's a NumPy array.
        b_dense = np.asarray(b)

        if method == "lu":
            # LU decomposition based solver for dense matrices. Most general.
            return scipy.linalg.solve(A_dense, b_dense)
        elif method == "qr":
            # QR decomposition solver. Good for least-squares, but `solve` also handles non-square.
            # This is more explicit if QR is specifically desired.
            q, r = scipy.linalg.qr(A_dense) # QR decomposition: A = QR
            # Solve Rx = Q^T b for x using back-substitution (as R is upper triangular).
            return scipy.linalg.solve_triangular(r, q.T @ b_dense)
        elif method == "svd":
            # Singular Value Decomposition solver. Robust for ill-conditioned or rank-deficient matrices.
            # A = U S V^H. x = V S^+ U^H b, where S^+ is pseudo-inverse of S.
            u, s_diag, vh = scipy.linalg.svd(A_dense, full_matrices=False) # Use economy SVD.
            # Filter small singular values to stabilize solution (pseudo-inverse).
            tolerance_svd = kwargs.get('tol', np.finfo(A_dense.dtype).eps * max(A_dense.shape) * s_diag[0]) # Default from np.linalg.pinv
            s_inv_diag = np.where(s_diag > tolerance_svd, 1.0/s_diag, 0.0) # Invert non-zero singular values.
            # x = V @ diag(S_inv) @ U^T @ b
            return vh.T @ (s_inv_diag[:, np.newaxis] * (u.T @ b_dense)) # Ensure correct broadcasting for s_inv_diag
        elif method == "cholesky":
            # Cholesky decomposition for symmetric (or Hermitian) positive-definite dense matrices.
            try:
                # `lower=True` returns L such that A = L L^T.
                L_cholesky = scipy.linalg.cholesky(A_dense, lower=True)
                # Solve Ly = b, then L^T x = y.
                y = scipy.linalg.solve_triangular(L_cholesky, b_dense, lower=True)
                return scipy.linalg.solve_triangular(L_cholesky.T, y, lower=False)
            except scipy.linalg.LinAlgError as e: # Matrix is not SPD or other issue.
                print(f"Warning: Dense Cholesky decomposition failed ('{e}'), matrix might not be symmetric positive-definite. Falling back to LU solver.")
                return scipy.linalg.solve(A_dense, b_dense) # Fallback to LU.
        else:
            raise ValueError(f"Unknown direct solver method for dense matrices: '{method}'.")


class TikhonvRegularization:
    """Tikhonov regularization for ill-posed problems."""
    
    def __init__(self, regularization_matrix: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
                 alpha: float = 1.0,  # Regularization strength parameter (lambda or alpha^2)
                 regularization_type: str = 'identity' # Type of regularization if matrix L is auto-generated
                 ):
        """
        Initialize Tikhonov regularization.
        This class helps in setting up and applying Tikhonov regularization to linear inverse problems.
        The problem is typically min ||Ax - b||^2 + alpha^2 * ||Lx||^2,
        where L is the regularization operator (or identity).

        Args:
            regularization_matrix (Optional[Union[np.ndarray, scipy.sparse.spmatrix]], optional):
                A custom regularization matrix L. If None, a matrix will be generated based on `regularization_type`.
                Defaults to None.
            alpha (float, optional): The regularization parameter (often denoted as λ or α).
                                     It controls the trade-off between data misfit and solution smoothness/smallness.
                                     Defaults to 1.0.
            regularization_type (str, optional): Type of regularization matrix L to generate if `regularization_matrix` is None.
                                                 Supported types:
                                                 - 'identity': L = I (Identity matrix). Penalizes the L2 norm of the solution (||x||^2). (Zeroth-order Tikhonov)
                                                 - 'gradient': L is a finite difference approximation of the first derivative (gradient).
                                                               Penalizes ||∇x||^2, promoting smoothness. (First-order Tikhonov)
                                                 - 'laplacian': L is a finite difference approximation of the second derivative (Laplacian).
                                                                Penalizes ||∇^2 x||^2, promoting flatness/smoothness of gradient. (Second-order Tikhonov)
                                                 Defaults to 'identity'.
        """
        self.alpha = alpha # Regularization strength parameter.
        self.regularization_matrix = regularization_matrix # User-provided L matrix.
        self.regularization_type = regularization_type # Type of L to generate if not provided.
    
    def create_regularization_matrix(self, n: int) -> scipy.sparse.spmatrix:
        """
        Create regularization matrix based on the selected type.
        
        Args:
            n: Size of model vector
            
        Returns:
            Regularization matrix L (SciPy sparse matrix).
        """
        if self.regularization_type.lower() == 'identity':
            # L = I (Identity matrix of size n x n)
            # Penalizes the squared L2-norm of the model parameters: ||x||^2
            # This is zeroth-order Tikhonov regularization.
            return scipy.sparse.eye(n, format='csr') # Use CSR for efficient operations.
        elif self.regularization_type.lower() == 'gradient':
            # L = First-derivative operator (approximates gradient ∇)
            # Penalizes squared L2-norm of the model gradient: ||∇x||^2, promoting smooth solutions.
            # This is first-order Tikhonov regularization.
            # Matrix D has shape (n-1, n) for 1D gradient.
            # D_ij = -1 if i=j, 1 if i=j-1, 0 otherwise (for forward difference).
            # Example for n=4: D = [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]
            # This form is for `diags` creating rows: d[i,i]=-1, d[i,i+1]=1.
            D_grad = scipy.sparse.diags([-1*np.ones(n), np.ones(n-1)], offsets=[0, 1], shape=(n-1, n), format='csr')
            return D_grad
        elif self.regularization_type.lower() == 'laplacian':
            # L = Second-derivative operator (approximates Laplacian ∇^2)
            # Penalizes squared L2-norm of the model Laplacian: ||∇^2 x||^2, promoting flat solutions (smooth gradient).
            # This is second-order Tikhonov regularization.
            # Matrix D2 has shape (n-2, n) for 1D Laplacian.
            # Example for n=4: D2 = [[1, -2, 1, 0], [0, 1, -2, 1]]
            # This form is for `diags` creating rows: d[i,i-1]=1, d[i,i]=-2, d[i,i+1]=1.
            D_lap = scipy.sparse.diags([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], offsets=[-1, 0, 1], shape=(n-2, n), format='csr')
            # The shape should be (n-2, n). Offsets are correct for centered differences.
            # For diags: main diag is k=0. Super-diag is k=1. Sub-diag is k=-1.
            # Corrected for standard 1D Laplacian (central difference):
            # D_lap = scipy.sparse.diags([1, -2, 1], offsets=[0, 1, 2], shape=(n-2, n)).tocsr() # Example for stencil [1 -2 1]
            # This needs to be carefully constructed. A common form:
            # D_lap = scipy.sparse.diags([np.ones(n-2), -2*np.ones(n-1), np.ones(n)], offsets=[0,1,2], shape=(n-2,n)) is wrong.
            # Let's use the common construction from finite differences for [1, -2, 1] stencil:
            D_lap = scipy.sparse.diags([1, -2, 1], offsets=[0, 1, 2], shape=(n - 2, n)).tocsr()
            # This creates:
            # [1 -2  1  0 ... ]
            # [0  1 -2  1 ... ]
            # ...
            # This is a common representation for the discrete 1D Laplacian.
            return D_lap

        else:
            raise ValueError(f"Unknown regularization type: '{self.regularization_type}'. "
                             "Supported types: 'identity', 'gradient', 'laplacian'.")
    
    def apply(self, A, b, solver=None):
        """
        Apply Tikhonov regularization to the linear system.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            solver: Solver to use (None for direct solver)
            
        Returns:
            Regularized solution vector x that minimizes ||Ax-b||^2 + alpha^2 ||Lx||^2.
        """
        num_model_params = A.shape[1] # Number of model parameters (columns in A).
        
        # --- Get or Create Regularization Matrix L ---
        L_matrix: Union[np.ndarray, scipy.sparse.spmatrix]
        if self.regularization_matrix is None:
            # If no custom L matrix is provided, create one based on `regularization_type`.
            L_matrix = self.create_regularization_matrix(num_model_params)
        else:
            # Use the user-provided regularization matrix.
            # Ensure its dimensions are compatible with the model vector.
            if self.regularization_matrix.shape[1] != num_model_params:
                raise ValueError(f"Custom regularization matrix shape ({self.regularization_matrix.shape}) "
                                 f"is not compatible with number of model parameters ({num_model_params}).")
            L_matrix = self.regularization_matrix

        # --- Construct Augmented System for Tikhonov Regularization ---
        # The Tikhonov regularized least-squares problem min ||Ax - b||^2_2 + ||alpha * Lx||^2_2
        # is equivalent to solving the augmented linear least-squares problem:
        #   [  A     ] x = [  b  ]
        #   [alpha*L ]     [  0  ]
        # Let A_aug = [A; alpha*L] and b_aug = [b; 0]. We solve min ||A_aug x - b_aug||^2_2.
        
        # Scale L by sqrt(alpha) if alpha is meant to be alpha^2 in objective function.
        # Or, if alpha is directly the weight for ||Lx||, then use alpha * L.
        # The common formulation is alpha^2 * ||Lx||^2, so use sqrt(alpha) * L here.
        # Assuming self.alpha is the lambda^2 or alpha^2 in the objective function.
        # So, the term added to A is sqrt(self.alpha) * L.
        scaled_L_matrix = np.sqrt(self.alpha) * L_matrix

        # Stack A and scaled_L_matrix vertically to form A_aug.
        # This requires A and scaled_L_matrix to be of the same sparse/dense type or compatible.
        # `scipy.sparse.vstack` handles mixed sparse/dense by converting dense to sparse.
        if scipy.sparse.isspmatrix(A) or scipy.sparse.isspmatrix(scaled_L_matrix):
            A_augmented = scipy.sparse.vstack([A, scaled_L_matrix], format='csr')
        else: # Both are dense
            A_augmented = np.vstack([A, scaled_L_matrix])

        # Construct b_aug: [b; 0].
        # `b` corresponds to A. The part corresponding to scaled_L_matrix is zeros.
        # Length of zeros vector is number of rows in L_matrix.
        zeros_for_reg = np.zeros(L_matrix.shape[0], dtype=b.dtype)
        b_augmented = np.hstack([b.ravel(), zeros_for_reg]) # Ensure b is flat then hstack.
        
        # --- Solve the Augmented System ---
        if solver is None:
            # If no specific solver object is provided, use a default approach.
            # For small to medium systems, direct solution of normal equations (A_aug^T A_aug x = A_aug^T b_aug) can be efficient.
            # For larger systems, iterative solvers like LSQR are preferred for the augmented system.
            # Heuristic: if total elements < 1e6 (e.g., 1000x1000 dense), consider direct.
            # SUGGESTION: This threshold is arbitrary. A better choice depends on available memory and matrix sparsity.
            if A_augmented.shape[0] * A_augmented.shape[1] < 1e6 and not scipy.sparse.isspmatrix(A_augmented):
                try:
                    # Solve normal equations: (A_aug^T * A_aug) * x = A_aug^T * b_aug
                    # This is generally not recommended for ill-conditioned A_aug due to squaring condition number.
                    # However, if A_aug is well-behaved, it can be fast.
                    # Using `direct_solver` which might choose LU or Cholesky.
                    # Note: A_aug.T @ A_aug should be symmetric positive (semi-)definite.
                    # Using method='lu' for robustness with direct_solver.
                    # It's generally better to solve the least squares problem ||A_aug x - b_aug|| directly if possible.
                    # `scipy.linalg.lstsq` for dense, `scipy.sparse.linalg.lsqr/lsmr` for sparse.
                    print("Using direct_solver on normal equations for Tikhonov (small system).")
                    return direct_solver(A_augmented.T @ A_augmented, A_augmented.T @ b_augmented, method="lu")
                except Exception as e_direct: # Fallback if direct solution fails
                    print(f"Direct solver on normal equations failed: {e_direct}. Falling back to LSQR for augmented system.")
                    return splinalg.lsqr(A_augmented, b_augmented)[0] # lsqr returns tuple (x, istop, itn, ...)
            else: # Large system or sparse A_augmented
                # Use LSQR, which is suitable for sparse and potentially ill-conditioned systems.
                print("Using LSQR solver for Tikhonov regularized augmented system (large or sparse system).")
                # `splinalg.lsqr` returns a tuple; solution is the first element.
                return splinalg.lsqr(A_augmented, b_augmented)[0]
        else:
            # If a solver object (e.g., an instance of LinearSolver) is provided, use its `solve` method.
            # This allows using configured iterative solvers (CGLS, LSQR, etc.) on the augmented system.
            return solver.solve(A_augmented, b_augmented)


class IterativeRefinement:
    """
    Iterative refinement to improve the accuracy of a solution to a linear system Ax = b.
    This is useful when the initial solution `x0` is obtained by a method that might
    suffer from precision loss (e.g., due to ill-conditioning or single-precision arithmetic).
    The process:
    1. Compute residual: r = b - A*x_k
    2. Solve for correction: A*c = r
    3. Update solution: x_{k+1} = x_k + c
    Repeat until ||r|| is small enough or max iterations reached.
    """
    
    def __init__(self, max_iterations: int = 5,       # Max refinement steps
                 tolerance: float = 1e-10,          # Target tolerance for norm of residual
                 use_double_precision: bool = True  # Compute residual in higher precision if possible
                 ):
        """
        Initialize iterative refinement parameters.
        
        Args:
            max_iterations (int, optional): Maximum number of refinement iterations. Defaults to 5.
            tolerance (float, optional): Convergence tolerance for the norm of the residual.
                                         Refinement stops if ||b - Ax|| < tolerance. Defaults to 1e-10.
            use_double_precision (bool, optional): If True, computes the residual (b - Ax) using
                                                   double precision arithmetic, even if A, x, b are
                                                   single precision. This can significantly improve
                                                   the accuracy of the refinement. Defaults to True.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_double_precision = use_double_precision
    
    def refine(self, A: Union[np.ndarray, scipy.sparse.spmatrix],
               b: np.ndarray,
               x0: np.ndarray, # Initial solution to be refined
               solver_func: Callable[[Any, Any], Any] # Function to solve Ac=r (e.g., a direct or iterative solver)
               ) -> np.ndarray:
        """
        Perform iterative refinement.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x0: Initial solution
            solver_func: Function that solves A*x = b
            
        Returns:
            Improved solution vector x.
        """
        current_solution_x = x0.copy() # Start with the initial solution.

        for iter_count in range(self.max_iterations):
            # 1. Compute Residual (r = b - A*x)
            # Optionally use higher precision for this calculation to capture small errors accurately.
            # This is crucial as `x` gets closer to the true solution.
            residual: np.ndarray
            if self.use_double_precision and current_solution_x.dtype != np.float64:
                # Convert A, x, b to float64 for residual calculation if not already.
                # This assumes A, b can also be other dtypes.
                # More robustly, ensure A, b are also float64 for this step if x is cast.
                A_double = A.astype(np.float64) if hasattr(A, 'astype') else np.asarray(A, dtype=np.float64)
                b_double = np.asarray(b, dtype=np.float64)
                x_double = current_solution_x.astype(np.float64)
                residual_double = b_double - A_double.dot(x_double)
                residual = residual_double.astype(current_solution_x.dtype) # Cast back to original dtype.
            else: # Use native precision.
                residual = b - A.dot(current_solution_x)
            
            # 2. Check Convergence
            # If the norm of the residual is below the specified tolerance, solution is accurate enough.
            residual_norm = np.linalg.norm(residual)
            if residual_norm < self.tolerance:
                if self.verbose: print(f"Iterative refinement converged at iteration {iter_count+1} with residual norm {residual_norm:.2e}")
                break # Exit loop.
            
            # 3. Solve for Correction (A*c = r)
            # Use the provided `solver_func` to solve for the correction term `c`.
            # `solver_func` should be a callable that takes (Matrix, RHS_vector) and returns solution.
            correction_c = solver_func(A, residual) # Solve A*c = r for c.
            
            # 4. Update Solution (x_{k+1} = x_k + c_k)
            current_solution_x = current_solution_x + correction_c
        else: # If loop finishes without break (max_iterations reached)
             if self.verbose: print(f"Iterative refinement reached max_iterations ({self.max_iterations}). Final residual norm {residual_norm:.2e}")
        
        return current_solution_x


def get_optimal_solver(A: Union[np.ndarray, scipy.sparse.spmatrix],
                      b: np.ndarray,
                      estimate_condition: bool = True,
                      time_limit: Optional[float] = None,    # Not currently used
                      memory_limit: Optional[int] = None   # Not currently used in selection logic beyond a rough check
                      ) -> Tuple[Any, Dict[str, str]]: # Returns (solver_function_or_object, info_dict)
    """
    Automatically select an appropriate linear solver for a given system Ax = b.
    
    This function analyzes properties of matrix A (size, sparsity, condition number if estimated)
    and suggests a suitable direct or iterative solver from SciPy or custom implementations.
    The selection logic prioritizes direct solvers for smaller/denser well-conditioned systems
    and iterative solvers for larger/sparser or ill-conditioned systems.

    Args:
        A (Union[np.ndarray, scipy.sparse.spmatrix]): The system matrix.
        b (np.ndarray): The right-hand side vector.
        estimate_condition (bool, optional): If True, attempt to estimate the condition number of A
                                             to guide solver selection (can be expensive for large matrices).
                                             Defaults to True.
        time_limit (Optional[float], optional): Maximum allowed solution time in seconds.
                                                (Currently NOT IMPLEMENTED in solver selection logic).
        memory_limit (Optional[int], optional): Maximum allowed memory usage in bytes.
                                                (Currently a ROUGH CHECK is implemented).
        
    Returns:
        Tuple[Any, Dict[str, str]]:
            - solver_object: A callable solver function (e.g., `lambda A,b: direct_solver(A,b,'lu')`)
                             or an instance of a `LinearSolver` subclass.
            - solver_info (Dict[str, str]): A dictionary containing information about the selected
                                            solver type and reasoning.
    """
    # --- Get Matrix Properties ---
    is_sparse_matrix = scipy.sparse.isspmatrix(A)
    num_rows, num_cols = A.shape

    # --- Estimate Memory Requirements (Very Rough) ---
    # This is a coarse estimation and might not be accurate for all sparse formats or solver overheads.
    memory_usage_estimate_bytes: float
    if is_sparse_matrix:
        num_non_zeros = A.nnz
        density = num_non_zeros / (num_rows * num_cols)
        # Rough estimate for sparse: (data + indices + indptr) * itemsize. Assume 3 arrays of size nnz for data, indices, plus indptr.
        # Assuming float64 (8 bytes) and int32/64 for indices (avg 6 bytes).
        memory_usage_estimate_bytes = num_non_zeros * (A.dtype.itemsize + 2 * np.dtype(np.int32).itemsize) # Approx.
    else: # Dense matrix
        density = 1.0
        memory_usage_estimate_bytes = num_rows * num_cols * A.dtype.itemsize

    # --- Memory Limit Check ---
    if memory_limit is not None and memory_usage_estimate_bytes > memory_limit:
        # If estimated memory exceeds limit, default to a memory-efficient iterative solver like CGLS.
        # Max iterations for CGLS could be related to num_cols.
        print(f"Estimated memory ({memory_usage_estimate_bytes / 1e6:.1f}MB) exceeds limit. Choosing memory-efficient CGLS.")
        solver = CGLSSolver(max_iterations=min(num_cols, 1000)) # Limit iterations
        return solver, {"type": "cgls", "reason": "memory_limit_exceeded"}
    
    # --- Problem Size and Density Based Selection ---
    # Heuristic: if total elements < 1M and density > 20%, consider it "small and relatively dense".
    is_small_dense_problem = (num_rows * num_cols < 1e6 and density > 0.2)
    
    if is_small_dense_problem and not is_sparse_matrix: # Only for dense matrices for this direct solver path
        # --- Small, Relatively Dense Problem: Attempt Direct Solvers ---
        try:
            well_conditioned = True # Assume well-conditioned by default
            if estimate_condition:
                # Estimate condition number for dense matrices using SVD values.
                # This can be computationally expensive for larger "small" matrices.
                try:
                    singular_values = scipy.linalg.svdvals(A)
                    # Avoid division by zero if any singular value is zero (rank deficient).
                    if singular_values[-1] < np.finfo(singular_values.dtype).eps:
                        condition_number_estimate = np.inf
                    else:
                        condition_number_estimate = singular_values[0] / singular_values[-1]
                    # Threshold for "well-conditioned" (can be application-dependent).
                    # 1e6 is a common heuristic; values much larger suggest ill-conditioning.
                    well_conditioned = condition_number_estimate < 1e6
                    print(f"Estimated condition number (dense SVD): {condition_number_estimate:.2e}")
                except Exception as e_cond_dense: # Catch errors during SVD (e.g., memory, convergence for non-finite)
                    print(f"Warning: Dense condition number estimation failed: {e_cond_dense}. Assuming well-conditioned.")
                    well_conditioned = True
            
            if well_conditioned:
                # For well-conditioned small dense problems, direct solvers are often best.
                # Check for symmetry for Cholesky eligibility.
                if np.allclose(A, A.T): # Check if A is symmetric (A == A^T)
                    try:
                        # Attempt Cholesky decomposition (requires symmetric positive-definite).
                        _ = scipy.linalg.cholesky(A, lower=True) # Test if Cholesky works.
                        print("Selected direct solver: Cholesky (dense, SPD).")
                        return (lambda A_in, b_in: direct_solver(A_in, b_in, method="cholesky")), \
                               {"type": "direct_dense", "method": "cholesky"}
                    except scipy.linalg.LinAlgError: # Not positive-definite.
                        pass # Fall through to LU for symmetric indefinite or other cases.
                
                # Default to LU decomposition for general dense square or over/underdetermined systems via `solve`.
                print("Selected direct solver: LU (dense).")
                return (lambda A_in, b_in: direct_solver(A_in, b_in, method="lu")), \
                       {"type": "direct_dense", "method": "lu"}
            else: # Ill-conditioned small dense problem
                print("Problem is small/dense but estimated as ill-conditioned. Using Tikhonov-regularized direct solver.")
                # Apply Tikhonov regularization with a small alpha.
                # The `direct_solver` within TikhonovRegularization.apply will handle the regularized system.
                tikhonov_reg = TikhonvRegularization(alpha=1e-6, regularization_type='identity')
                # The solver itself becomes the apply method of the Tikhonov object.
                return (lambda A_in, b_in: tikhonov_reg.apply(A_in, b_in, solver=None)), \
                       {"type": "tikhonov_direct", "condition": "ill-conditioned", "alpha": 1e-6}

        except Exception as e_direct_path: # Catch any other errors in the direct solver path.
            print(f"Warning: Attempt to use direct solver for small/dense system failed: {str(e_direct_path)}. Falling back to iterative.")

    # --- Large or Sparse Problem: Use Iterative Solvers ---
    # (Or if small/dense direct path failed or was skipped)
    if is_sparse_matrix and estimate_condition: # Condition number estimation for sparse A
        # For sparse matrices, SVD is too expensive. `scipy.sparse.linalg.svds` gives few SVs.
        # Condition number can be estimated using iterative methods (e.g. `condest` in MATLAB style)
        # or via LU/QR properties if feasible. `splu.rcond` is a cheap estimate from LU.
        try:
            # This requires A to be square for splu.
            # If A is rectangular and sparse, condition estimation is harder.
            # For now, let's assume if it's sparse, we might lean towards iterative anyway.
            # This rcond is 1/cond(A) in 1-norm, so small rcond means large cond.
            if num_rows == num_cols: # splu only for square matrices
                 lu_factor = splinalg.splu(A.tocsc()) # Requires CSC format
                 reciprocal_condition_estimate = lu_factor.rcond()
                 if reciprocal_condition_estimate > 1e-6: # If rcond is not too small
                     print(f"Sparse matrix estimated as well-conditioned (rcond={reciprocal_condition_estimate:.2e}). Considering sparse LU.")
                     # If well-conditioned and sparse, sparse LU might be good if direct solution is desired.
                     # However, iterative solvers are generally preferred for large sparse.
                     # Fall through to iterative, or could return sparse LU here.
                 else:
                     print(f"Sparse matrix estimated as ill-conditioned (rcond={reciprocal_condition_estimate:.2e}). Preferring iterative solver.")
            else: # Rectangular sparse matrix
                 print("Matrix is sparse and rectangular. Preferring iterative solver (LSQR/CGLS family).")
        except Exception as e_cond_sparse:
            print(f"Warning: Sparse condition number estimation failed: {e_cond_sparse}. Proceeding with default iterative solver.")

    # Default to iterative solvers for large/sparse systems.
    # Selection based on matrix properties (square, symmetric, positive-definite).
    if num_rows == num_cols: # Square system
        is_symmetric = False # Default
        if is_sparse_matrix:
            # A cheap way to check for symmetry in sparse matrices: nnz(A - A.T) == 0
            # This can be slow if A is very large.
            # For now, assume not symmetric unless explicitly known.
            # if (A - A.T).nnz == 0: is_symmetric = True
            pass # Cannot reliably check symmetry cheaply for very large sparse.
        else: # Dense matrix (already handled by small_dense_problem path, but as a fallback)
            is_symmetric = np.allclose(A, A.T)
        
        if is_symmetric:
            # For symmetric systems. Try to check for positive-definiteness for CG.
            # Positive definiteness check (e.g., attempting Cholesky or checking eigenvalues) can be expensive.
            # If SPD, Conjugate Gradient (CG) is optimal. SciPy has `splinalg.cg`.
            # If symmetric but indefinite, MINRES or SYMMLQ are options. `splinalg.minres`.
            # For simplicity here, if symmetric, might choose CGLS (as it works on normal equations, always SPD).
            # Or, if we trust it's SPD, could point to a CG wrapper if available.
            print("Matrix is square & (assumed) symmetric. Choosing CGLS (works on normal equations). Consider CG if known SPD.")
            # `splinalg.cg` is for Ax=b where A is SPD. CGLS solves A^T A x = A^T b.
            # This class structure uses CGLSSolver which wraps the custom _cgls.
            solver = CGLSSolver(max_iterations=min(num_cols, 1000), damping=1e-9) # Small damping for stability
            return solver, {"type": "cgls", "matrix_type": "square_symmetric_assumption"}
        else: # General square, non-symmetric system
            # GMRES is a common choice. `splinalg.gmres`.
            # LSQR/CGLS can also solve square systems via least-squares formulation.
            print("Matrix is square & non-symmetric. Choosing CGLS (general applicability). Consider GMRES for direct Ax=b.")
            solver = CGLSSolver(max_iterations=min(num_cols, 1000), damping=1e-9)
            return solver, {"type": "cgls", "matrix_type": "square_nonsymmetric"}
    
    # Rectangular system (or fallback for square if other conditions not met):
    # Use LSQR or CGLS family. RRLSQR is chosen here as a robust default.
    # These are designed for least-squares problems min ||Ax-b||.
    print("Matrix is rectangular or general fallback. Choosing RRLSQR (robust iterative for least-squares).")
    solver = RRLSQRSolver(max_iterations=min(num_cols, 1000), damping=1e-9) # Small default damping
    return solver, {"type": "rrlsqr", "matrix_type": "rectangular_or_general"}