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

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


def generalized_solver(A, b, method="cgls", x=None, maxiter=200, tol=1e-8,
                      verbose=False, damp=0.0, use_gpu=False, parallel=False, n_jobs=-1):
    """
    Generalized solver for Ax = b with optional GPU acceleration and parallelism.
    
    Parameters:
    -----------
    A : array_like or sparse matrix
        The system matrix (Jacobian or forward operator).
    b : array_like
        Right-hand side vector.
    method : str, optional
        Solver method: 'lsqr', 'rrlsqr', 'cgls', or 'rrls'. Default is 'cgls'.
    x : array_like, optional
        Initial guess for the solution. If None, zeros are used.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance.
    verbose : bool, optional
        Print progress information every 10 iterations.
    damp : float, optional
        Damping factor (Tikhonov regularization).
    use_gpu : bool, optional
        Use GPU acceleration with CuPy (if available).
    parallel : bool, optional
        Use parallel CPU computations.
    n_jobs : int, optional
        Number of parallel jobs (if parallel is True).
        
    Returns:
    --------
    x : array_like
        The computed solution vector.
    """
    # Choose the backend (NumPy or CuPy)
    if use_gpu and GPU_AVAILABLE:
        xp = cp
    else:
        xp = np
        use_gpu = False  # Ensure it's turned off if not available
    
    # Convert A and b to appropriate arrays
    if use_gpu:
        if scipy.sparse.isspmatrix(A):
            A = csr_matrix(A)
        else:
            A = cp.asarray(A)
        b = cp.asarray(b)
    else:
        if scipy.sparse.isspmatrix(A):
            A = A.tocsr()
        else:
            A = np.asarray(A)
        b = np.asarray(b)
    
    # Initialize solution and residual
    if x is None:
        x = xp.zeros(A.shape[1])
        r = b.copy()
    else:
        x = xp.asarray(x)
        r = b - A.dot(x)
    
    # Precompute initial quantities
    s = A.T.dot(r)
    p = s.copy()
    gamma = xp.dot(s.T, s)
    rr = xp.dot(r.T, r)
    rr0 = rr
    
    # Choose the solver routine based on method
    if method.lower() == "lsqr":
        return _lsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif method.lower() == "rrlsqr":
        return _rrlsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif method.lower() == "cgls":
        return _cgls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif method.lower() == "rrls":
        return _rrls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'lsqr', 'rrlsqr', 'cgls', 'rrls'")


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
        Matrix-vector product
    """
    if use_gpu:
        v = xp.asarray(v)
        return A.dot(v)
    else:
        if scipy.sparse.isspmatrix(A):
            return A.dot(v)
        else:
            if parallel and PARALLEL_AVAILABLE:
                # Partition matrix rows for parallel processing
                n_rows = A.shape[0]
                if n_jobs <= 0:
                    import multiprocessing
                    n_jobs = multiprocessing.cpu_count()
                
                partition_size = max(1, n_rows // n_jobs)
                partitions = [(i, min(i + partition_size, n_rows)) 
                             for i in range(0, n_rows, partition_size)]
                
                # Compute matrix-vector product in parallel
                results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(lambda row_range: A[row_range[0]:row_range[1]].dot(v))(partition)
                    for partition in partitions
                )
                
                # Combine results
                return xp.concatenate(results)
            else:
                return A.dot(v)


def _cgls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
         use_gpu, parallel, n_jobs, xp):
    """
    CGLS solver for linear least squares problems.
    
    This implements the Conjugate Gradient Least Squares method for solving
    the normal equations A^T A x = A^T b.
    
    Args:
        A: System matrix
        b: Right-hand side vector
        x: Initial solution vector
        r: Initial residual
        s: Initial A^T r
        gamma: Initial s^T s
        rr: Initial r^T r
        rr0: Initial residual norm
        maxiter: Maximum iterations
        tol: Convergence tolerance
        verbose: Whether to print progress
        damp: Damping parameter
        use_gpu: Whether to use GPU acceleration
        parallel: Whether to use parallel computation
        n_jobs: Number of parallel jobs
        xp: NumPy or CuPy module
        
    Returns:
        Solution vector
    """
    # Ensure inputs have correct shape
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    
    # Initialize search direction
    p = s.copy()
    
    for i in range(maxiter):
        if verbose and i % 10 == 0:
            pg.info("CGLS Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
        # Compute A*p
        q = _matrix_multiply(A, p, use_gpu, parallel, n_jobs, xp)
        
        # Add damping if requested
        if damp > 0:
            q += damp * p
        
        # Ensure q is a column vector
        q = q.reshape(-1, 1)
        
        # Compute step size
        alpha = float(gamma / xp.dot(q.T, q))
        
        # Update solution and residual
        x += alpha * p
        r -= alpha * q
        
        # Compute new gradient
        s = _matrix_multiply(A.T, r, use_gpu, parallel, n_jobs, xp)
        
        # Add damping if requested
        if damp > 0:
            s += damp * r
        
        # Ensure s is a column vector
        s = s.reshape(-1, 1)
        
        # Compute new gamma and beta
        gamma_new = float(xp.dot(s.T, s))
        beta = float(gamma_new / gamma)
        
        # Update search direction
        p = s + beta * p
        
        # Update gamma
        gamma = gamma_new
        
        # Check convergence
        rr = float(xp.dot(r.T, r))
        if rr / rr0 < tol:
            if verbose:
                pg.info(f"CGLS converged after {i+1} iterations")
            break
    
    # Return solution (convert back to CPU if on GPU)
    return x.get() if use_gpu and GPU_AVAILABLE else x


def _lsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
         use_gpu, parallel, n_jobs, xp):
    """
    LSQR solver for linear least squares problems.
    
    This implements the LSQR algorithm of Paige and Saunders for solving
    the least squares problem min ||Ax - b||_2.
    
    Args: (same as _cgls)
        
    Returns:
        Solution vector
    """
    # Ensure x and r are column vectors
    if x is None:
        x = xp.zeros((A.shape[1], 1))
    else:
        x = xp.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
    
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    
    # Initialize u and beta
    u = r.copy()
    beta = xp.sqrt(float(xp.dot(u.T, u)))
    if beta > 0:
        u = u / beta
    
    # Initialize v and alpha
    v = _matrix_multiply(A.T, u, use_gpu, parallel, n_jobs, xp)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    alpha = xp.sqrt(float(xp.dot(v.T, v)))
    if alpha > 0:
        v = v / alpha
    
    w = v.copy()
    phi_bar = beta
    rho_bar = alpha
    
    for i in range(maxiter):
        if verbose and i % 10 == 0:
            pg.info("LSQR Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
        # Bidiagonalization
        u_next = _matrix_multiply(A, v, use_gpu, parallel, n_jobs, xp)
        if u_next.ndim == 1:
            u_next = u_next.reshape(-1, 1)
        u_next = u_next - alpha * u
        
        beta = xp.sqrt(float(xp.dot(u_next.T, u_next)))
        if beta > 0:
            u = u_next / beta
            
        v_next = _matrix_multiply(A.T, u, use_gpu, parallel, n_jobs, xp)
        if v_next.ndim == 1:
            v_next = v_next.reshape(-1, 1)
        v_next = v_next - beta * v
        
        alpha = xp.sqrt(float(xp.dot(v_next.T, v_next)))
        if alpha > 0:
            v = v_next / alpha
        
        # Apply orthogonal transformation
        rho = xp.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar
        
        # Update x and w
        t = phi / rho
        x = x + t * w
        w = v - (theta / rho) * w
        
        # Check convergence
        rr = phi_bar**2
        if rr / rr0 < tol:
            if verbose:
                pg.info(f"LSQR converged after {i+1} iterations")
            break
    
    return x.get() if use_gpu and GPU_AVAILABLE else x


def _rrlsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
          use_gpu, parallel, n_jobs, xp):
    """
    Regularized LSQR solver.
    
    This implements a regularized version of the LSQR algorithm.
    
    Args: (same as _lsqr)
        
    Returns:
        Solution vector
    """
    # Ensure x and r are column vectors
    if x is None:
        x = xp.zeros((A.shape[1], 1))
    else:
        x = xp.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
    
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    
    # Initialize u and beta
    u = r.copy()
    beta = xp.sqrt(float(xp.dot(u.T, u)))
    if beta > 0:
        u = u / beta
    
    # Initialize v and alpha with regularization
    v = _matrix_multiply(A.T, u, use_gpu, parallel, n_jobs, xp)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if damp > 0:
        v = v + damp * x
        
    alpha = xp.sqrt(float(xp.dot(v.T, v)))
    if alpha > 0:
        v = v / alpha
    
    w = v.copy()
    phi_bar = beta
    rho_bar = alpha
    
    for i in range(maxiter):
        if verbose and i % 10 == 0:
            pg.info("RRLSQR Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
        # Bidiagonalization with regularization
        u_next = _matrix_multiply(A, v, use_gpu, parallel, n_jobs, xp)
        if u_next.ndim == 1:
            u_next = u_next.reshape(-1, 1)
        u_next = u_next - alpha * u
        
        beta = xp.sqrt(float(xp.dot(u_next.T, u_next)))
        if beta > 0:
            u = u_next / beta
            
        v_next = _matrix_multiply(A.T, u, use_gpu, parallel, n_jobs, xp)
        if v_next.ndim == 1:
            v_next = v_next.reshape(-1, 1)
        v_next = v_next - beta * v
        
        if damp > 0:
            v_next = v_next + damp * x
            
        alpha = xp.sqrt(float(xp.dot(v_next.T, v_next)))
        if alpha > 0:
            v = v_next / alpha
        
        # Apply orthogonal transformation
        rho = xp.sqrt(rho_bar**2 + beta**2 + damp**2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar
        
        # Update x and w
        t = phi / rho
        x = x + t * w
        w = v - (theta / rho) * w
        
        # Check convergence
        rr = phi_bar**2
        if rr / rr0 < tol:
            if verbose:
                pg.info(f"RRLSQR converged after {i+1} iterations")
            break
    
    return x.get() if use_gpu and GPU_AVAILABLE else x


def _rrls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
         use_gpu, parallel, n_jobs, xp):
    """
    Range-Restricted Least Squares solver.
    
    This implements a Range-Restricted Least Squares method.
    
    Args: (same as _cgls)
        
    Returns:
        Solution vector
    """
    # Ensure x, r, and s are column vectors
    if x is None:
        x = xp.zeros((A.shape[1], 1))
    else:
        x = xp.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
    
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
        
    w = s.copy()
    
    for i in range(maxiter):
        if verbose and i % 10 == 0:
            pg.info("RRLS Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
        p = _matrix_multiply(A, w, use_gpu, parallel, n_jobs, xp)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
            
        denom = xp.dot(p.T, p)
        if xp.isclose(denom, 0.0):
            break
            
        lam = xp.dot(p.T, r) / denom
        x = x + w * float(lam)  # Convert lam to scalar
        r = r - p * float(lam)
        
        s = _matrix_multiply(A.T, r, use_gpu, parallel, n_jobs, xp)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
            
        if damp > 0:
            s = s + damp * x
            
        w = s
        rr = float(xp.dot(r.T, r))
        if rr / rr0 < tol:
            if verbose:
                pg.info(f"RRLS converged after {i+1} iterations")
            break
            
    return x.get() if use_gpu and GPU_AVAILABLE else x


class LinearSolver:
    """Base class for linear system solvers."""
    
    def __init__(self, method="cgls", max_iterations=200, tolerance=1e-8, 
                use_gpu=False, parallel=False, n_jobs=-1, damping=0.0,
                verbose=False):
        """
        Initialize solver with computational options.
        
        Args:
            method: Solver method ('cgls', 'lsqr', 'rrlsqr', 'rrls')
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            use_gpu: Whether to use GPU acceleration
            parallel: Whether to use parallel computation
            n_jobs: Number of parallel jobs
            damping: Damping parameter
            verbose: Whether to print progress
        """
        self.method = method.lower()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.parallel = parallel and PARALLEL_AVAILABLE
        self.n_jobs = n_jobs
        self.damping = damping
        self.verbose = verbose
        
        # Check method
        valid_methods = ['cgls', 'lsqr', 'rrlsqr', 'rrls']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")
        
        # Check GPU availability
        if use_gpu and not GPU_AVAILABLE:
            print("Warning: GPU acceleration requested but CuPy not available. Using CPU.")
            self.use_gpu = False
        
        # Check parallel availability
        if parallel and not PARALLEL_AVAILABLE:
            print("Warning: Parallel computation requested but joblib not available. Using serial.")
            self.parallel = False
    
    def solve(self, A, b, x0=None):
        """
        Solve linear system Ax = b.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x0: Initial guess (None for zeros)
            
        Returns:
            Solution vector
        """
        return generalized_solver(
            A, b, method=self.method, x=x0,
            maxiter=self.max_iterations, tol=self.tolerance,
            verbose=self.verbose, damp=self.damping,
            use_gpu=self.use_gpu, parallel=self.parallel, n_jobs=self.n_jobs
        )


class CGLSSolver(LinearSolver):
    """CGLS (Conjugate Gradient Least Squares) solver."""
    
    def __init__(self, max_iterations=200, tolerance=1e-8, use_gpu=False, 
                parallel=False, n_jobs=-1, damping=0.0, verbose=False):
        """
        Initialize CGLS solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            use_gpu: Whether to use GPU acceleration
            parallel: Whether to use parallel computation
            n_jobs: Number of parallel jobs
            damping: Damping parameter
            verbose: Whether to print progress
        """
        super().__init__(
            method="cgls", max_iterations=max_iterations, tolerance=tolerance,
            use_gpu=use_gpu, parallel=parallel, n_jobs=n_jobs,
            damping=damping, verbose=verbose
        )


class LSQRSolver(LinearSolver):
    """LSQR solver for least squares problems."""
    
    def __init__(self, max_iterations=200, tolerance=1e-8, use_gpu=False, 
                parallel=False, n_jobs=-1, damping=0.0, verbose=False):
        """
        Initialize LSQR solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            use_gpu: Whether to use GPU acceleration
            parallel: Whether to use parallel computation
            n_jobs: Number of parallel jobs
            damping: Damping parameter
            verbose: Whether to print progress
        """
        super().__init__(
            method="lsqr", max_iterations=max_iterations, tolerance=tolerance,
            use_gpu=use_gpu, parallel=parallel, n_jobs=n_jobs,
            damping=damping, verbose=verbose
        )


class RRLSQRSolver(LinearSolver):
    """Regularized LSQR solver."""
    
    def __init__(self, max_iterations=200, tolerance=1e-8, use_gpu=False, 
                parallel=False, n_jobs=-1, damping=0.1, verbose=False):
        """
        Initialize regularized LSQR solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            use_gpu: Whether to use GPU acceleration
            parallel: Whether to use parallel computation
            n_jobs: Number of parallel jobs
            damping: Damping parameter (regularization strength)
            verbose: Whether to print progress
        """
        super().__init__(
            method="rrlsqr", max_iterations=max_iterations, tolerance=tolerance,
            use_gpu=use_gpu, parallel=parallel, n_jobs=n_jobs,
            damping=damping, verbose=verbose
        )


class RRLSSolver(LinearSolver):
    """Range-Restricted Least Squares solver."""
    
    def __init__(self, max_iterations=200, tolerance=1e-8, use_gpu=False, 
                parallel=False, n_jobs=-1, damping=0.0, verbose=False):
        """
        Initialize RRLS solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            use_gpu: Whether to use GPU acceleration
            parallel: Whether to use parallel computation
            n_jobs: Number of parallel jobs
            damping: Damping parameter
            verbose: Whether to print progress
        """
        super().__init__(
            method="rrls", max_iterations=max_iterations, tolerance=tolerance,
            use_gpu=use_gpu, parallel=parallel, n_jobs=n_jobs,
            damping=damping, verbose=verbose
        )


# Additional solver implementations
import scipy.linalg
def direct_solver(A, b, method="lu", **kwargs):
    """
    Solve a linear system using direct methods.
    
    Args:
        A: System matrix
        b: Right-hand side vector
        method: Direct solver method ('lu', 'qr', 'svd', 'cholesky')
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Solution vector
    """
    # Handle sparse matrices
    if scipy.sparse.isspmatrix(A):
        if method == "lu":
            # Sparse LU decomposition
            return splinalg.spsolve(A, b)
        elif method == "cholesky":
            # Check if A is symmetric positive definite
            try:
                factor = splinalg.cholesky(A.tocsc())
                return factor.solve(b)
            except:
                print("Warning: Matrix not SPD, falling back to spsolve")
                return splinalg.spsolve(A, b)
        else:
            # Fall back to sparse solve for other methods
            return splinalg.spsolve(A, b)
    else:
        # Dense matrix solvers
        if method == "lu":
            # LU decomposition
            
            return scipy.linalg.solve(A, b)
        elif method == "qr":
            # QR decomposition
            
            q, r = scipy.linalg.qr(A)
            return scipy.linalg.solve_triangular(r, q.T @ b)
        elif method == "svd":
            # SVD decomposition
           
            u, s, vh = scipy.linalg.svd(A, full_matrices=False)
            # Filter small singular values
            tol = kwargs.get('tol', 1e-10)
            s_inv = np.where(s > tol, 1/s, 0)
            return vh.T @ (s_inv * (u.T @ b))
        elif method == "cholesky":
            # Cholesky decomposition
            try:
               
                L = scipy.linalg.cholesky(A, lower=True)
                return scipy.linalg.solve_triangular(
                    L.T, 
                    scipy.linalg.solve_triangular(L, b, lower=True),
                    lower=False
                )
            except:
                print("Warning: Matrix not SPD, falling back to LU")
                return scipy.linalg.solve(A, b)
        else:
            raise ValueError(f"Unknown direct solver method: {method}")


class TikhonvRegularization:
    """Tikhonov regularization for ill-posed problems."""
    
    def __init__(self, regularization_matrix=None, 
                 alpha=1.0, regularization_type='identity'):
        """
        Initialize Tikhonov regularization.
        
        Args:
            regularization_matrix: Custom regularization matrix (if None, one will be generated)
            alpha: Regularization parameter
            regularization_type: Type of regularization ('identity', 'gradient', 'laplacian')
        """
        self.alpha = alpha
        self.regularization_matrix = regularization_matrix
        self.regularization_type = regularization_type
    
    def create_regularization_matrix(self, n):
        """
        Create regularization matrix based on the selected type.
        
        Args:
            n: Size of model vector
            
        Returns:
            Regularization matrix
        """
        if self.regularization_type == 'identity':
            # 0th order Tikhonov (identity matrix)
            return scipy.sparse.eye(n)
        elif self.regularization_type == 'gradient':
            # 1st order Tikhonov (gradient operator)
            D = scipy.sparse.diags([[-1], [1]], offsets=[0, 1], shape=(n-1, n))
            return D
        elif self.regularization_type == 'laplacian':
            # 2nd order Tikhonov (Laplacian operator)
            D = scipy.sparse.diags([[1], [-2], [1]], offsets=[-1, 0, 1], shape=(n-2, n))
            return D
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")
    
    def apply(self, A, b, solver=None):
        """
        Apply Tikhonov regularization to the linear system.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            solver: Solver to use (None for direct solver)
            
        Returns:
            Regularized solution
        """
        m = A.shape[1]
        
        # Create regularization matrix if not provided
        if self.regularization_matrix is None:
            L = self.create_regularization_matrix(m)
        else:
            L = self.regularization_matrix
        
        # Augment system with regularization
        A_aug = scipy.sparse.vstack([A, np.sqrt(self.alpha) * L])
        b_aug = np.hstack([b, np.zeros(L.shape[0])])
        
        # Solve regularized system
        if solver is None:
            # Use direct solver for small systems
            if A.shape[0] * A.shape[1] < 1e6:
                try:
                    return direct_solver(A_aug.T @ A_aug, A_aug.T @ b_aug)
                except:
                    # Fall back to LSQR
                    return splinalg.lsqr(A_aug, b_aug)[0]
            else:
                # Use LSQR for large systems
                return splinalg.lsqr(A_aug, b_aug)[0]
        else:
            # Use provided solver
            return solver.solve(A_aug, b_aug)


class IterativeRefinement:
    """
    Iterative refinement to improve accuracy of a solution to a linear system.
    """
    
    def __init__(self, max_iterations=5, tolerance=1e-10, 
                 use_double_precision=True):
        """
        Initialize iterative refinement.
        
        Args:
            max_iterations: Maximum number of refinement iterations
            tolerance: Convergence tolerance
            use_double_precision: Whether to use double precision for residual
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_double_precision = use_double_precision
    
    def refine(self, A, b, x0, solver_func):
        """
        Perform iterative refinement.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x0: Initial solution
            solver_func: Function that solves A*x = b
            
        Returns:
            Improved solution
        """
        x = x0.copy()
        
        for i in range(self.max_iterations):
            # Compute residual (optionally in higher precision)
            if self.use_double_precision and not isinstance(x, np.float64):
                residual = b - A.dot(x.astype(np.float64))
                residual = residual.astype(x.dtype)
            else:
                residual = b - A.dot(x)
            
            # Check convergence
            if np.linalg.norm(residual) < self.tolerance:
                break
            
            # Solve for correction
            correction = solver_func(A, residual)
            
            # Update solution
            x = x + correction
        
        return x


def get_optimal_solver(A, b, estimate_condition=True, 
                      time_limit=None, memory_limit=None):
    """
    Automatically select the optimal solver for a given linear system.
    
    Args:
        A: System matrix
        b: Right-hand side vector
        estimate_condition: Whether to estimate condition number
        time_limit: Maximum allowed solution time (seconds)
        memory_limit: Maximum allowed memory usage (bytes)
        
    Returns:
        Tuple of (solver_object, solver_info)
    """
    # Get matrix info
    is_sparse = scipy.sparse.isspmatrix(A)
    n, m = A.shape
    
    # Estimate memory requirements
    if is_sparse:
        nnz = A.nnz
        density = nnz / (n * m)
        memory_estimate = nnz * 8 * 3  # Rough estimate for sparse solvers
    else:
        density = 1.0
        memory_estimate = n * m * 8 * 2  # Rough estimate for dense solvers
    
    # Check memory limit
    if memory_limit is not None and memory_estimate > memory_limit:
        # Use iterative method with low memory requirements
        solver = CGLSSolver(max_iterations=min(n, 1000))
        return solver, {"type": "cgls", "reason": "memory_limit"}
    
    # Check problem size
    if n * m < 1e6 and density > 0.2:
        # Small, relatively dense problem
        try:
            # Estimate condition number (if requested)
            if estimate_condition:
                if is_sparse:
                    # For sparse matrices, use cheaper estimator
                    try:
                        import scipy.sparse.linalg as spla
                        lu = spla.splu(A.tocsc())
                        condition_est = lu.rcond
                        well_conditioned = condition_est > 1e-6
                    except:
                        well_conditioned = True  # Assume well-conditioned if estimation fails
                else:
                    # For dense matrices, use SVD-based estimator
                    try:
                        s = scipy.linalg.svdvals(A)
                        condition_number = s[0] / s[-1]
                        well_conditioned = condition_number < 1e6
                    except:
                        well_conditioned = True  # Assume well-conditioned if estimation fails
            else:
                well_conditioned = True
            
            if well_conditioned:
                # Use direct solver for well-conditioned problems
                if is_sparse:
                    solver = lambda A, b: direct_solver(A, b, method="lu")
                    return solver, {"type": "direct_sparse", "method": "lu"}
                else:
                    # Check if matrix is symmetric
                    is_symmetric = np.allclose(A, A.T)
                    if is_symmetric:
                        try:
                            # Check if positive definite
                            scipy.linalg.cholesky(A)
                            solver = lambda A, b: direct_solver(A, b, method="cholesky")
                            return solver, {"type": "direct_dense", "method": "cholesky"}
                        except:
                            pass
                    
                    solver = lambda A, b: direct_solver(A, b, method="lu")
                    return solver, {"type": "direct_dense", "method": "lu"}
            else:
                # Ill-conditioned problem, use regularized solver
                tikhonov = TikhonvRegularization(alpha=1e-6)
                solver = lambda A, b: tikhonov.apply(A, b)
                return solver, {"type": "tikhonov", "condition": "ill"}
                
        except Exception as e:
            # Fall back to iterative solver
            print(f"Warning: Direct solver selection failed: {str(e)}")
    
    # Large or sparse problem, use iterative solver
    # Check if matrix is square
    if n == m:
        # Square system, try conjugate gradient variants
        is_symmetric = False
        if is_sparse:
            # Cheap test for symmetry
            is_symmetric = (A - A.T).nnz == 0
        else:
            is_symmetric = np.allclose(A, A.T)
        
        if is_symmetric:
            # For symmetric systems
            try:
                # Test for positive definiteness
                if is_sparse:
                    # Randomly sample a few values on diagonal
                    import random
                    pos_def = all(A[i,i] > 0 for i in random.sample(range(n), min(10, n)))
                else:
                    pos_def = np.all(np.linalg.eigvalsh(A) > 0)
                
                if pos_def:
                    # Symmetric positive definite, use CG
                    solver = lambda A, b: splinalg.cg(A, b)[0]
                    return solver, {"type": "cg", "matrix": "spd"}
            except:
                pass
            
            # Symmetric but not necessarily positive definite, use MINRES
            solver = lambda A, b: splinalg.minres(A, b)[0]
            return solver, {"type": "minres", "matrix": "symmetric"}
        else:
            # General square system, use GMRES
            solver = lambda A, b: splinalg.gmres(A, b)[0]
            return solver, {"type": "gmres", "matrix": "square"}
    
    # Rectangular or fallback, use LSQR
    solver = RRLSQRSolver(max_iterations=min(n, 1000))
    return solver, {"type": "rrlsqr", "matrix": "rectangular"}