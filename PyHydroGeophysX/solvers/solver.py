# time_lapse_inversion/solver.py
import numpy as np
import scipy.sparse
from joblib import Parallel, delayed
import pygimli as pg

# Optional GPU acceleration with CuPy (if available)
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    gpu_available = True
except ImportError:
    gpu_available = False


def generalized_solver(A, b, method="cgls", x=None, maxiter=2000, tol=1e-8,
                       verbose=False, damp=0.0, use_gpu=False, parallel=False, n_jobs=-1):
    """
    Generalized solver for Ax = b with optional GPU acceleration and parallelism.
    
    Parameters
    ----------
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
        
    Returns
    -------
    x : array_like
        The computed solution vector.
    """
    # Choose the backend (NumPy or CuPy)
    xp = cp if use_gpu and gpu_available else np

    # Convert A and b to appropriate arrays
    if use_gpu and gpu_available:
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
    if method == "lsqr":
        return _lsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose,
                     damp, use_gpu, parallel, n_jobs, xp)
    elif method == "rrlsqr":
        return _rrlsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose,
                       damp, use_gpu, parallel, n_jobs, xp)
    elif method == "cgls":
        return _cgls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose,
                     damp, use_gpu, parallel, n_jobs, xp)
    elif method == "rrls":
        return _rrls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose,
                     damp, use_gpu, parallel, n_jobs, xp)
    else:
        raise ValueError("Unknown method: {}".format(method))


def _matrix_multiply(A, v, use_gpu, parallel, n_jobs, xp):
    """
    Helper routine for matrix-vector multiplication with optional GPU or parallel CPU support.
    """
    if use_gpu:
        v = xp.asarray(v)
        return A.dot(v)
    else:
        if scipy.sparse.isspmatrix(A):
            return A.dot(v)
        else:
            if parallel:
                result = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(xp.dot)(A[i], v) for i in range(A.shape[0])
                )
                return xp.array(result)
            else:
                return A.dot(v)


def _lsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
          use_gpu, parallel, n_jobs, xp):
    """
    LSQR solver with proper shape handling.
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
            pg.info("LSQR Iteration:", i, "residual:", rr, "relative:", rr / rr0)
        
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
        
        # Construct and apply orthogonal transformation
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
        
        rr = phi_bar**2
        if rr / rr0 < tol:
            break
    
    return x.get() if use_gpu and gpu_available else x




def _rrlsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
            use_gpu, parallel, n_jobs, xp):
    """
    Regularized LSQR solver with proper shape handling.
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
            pg.info("RRLSQR Iteration:", i, "residual:", rr, "relative:", rr / rr0)
        
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
        
        # Apply regularization and update solution
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
        
        rr = phi_bar**2
        if rr / rr0 < tol:
            break
    
    return x.get() if use_gpu and gpu_available else x



def _cgls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
          use_gpu, parallel, n_jobs, xp):
    """
    CGLS solver core routine to solve A^T A x = A^T b.
    """
    p = s.copy()

    for i in range(maxiter):
        if verbose and i % 10 == 0:
            pg.info("Iteration:", i, "residual:", rr, "relative:", rr / rr0)

        q = _matrix_multiply(A, p, use_gpu, parallel, n_jobs, xp)
        if damp > 0:
            q += damp * p
        # Ensure q is a column vector and update solution
        q = q.reshape(-1, 1)
        x = x.reshape(-1, 1)
        alfa = gamma / xp.dot(q.T, q)
        x += p.reshape(-1, 1) * alfa
        r -= q * alfa

        # Update s = A^T r (and include damping if needed)
        s = _matrix_multiply(A.T, r, use_gpu, parallel, n_jobs, xp)
        if damp > 0:
            s += damp * r
        newgamma = xp.dot(s.T, s)
        p = s + float(newgamma / gamma) * p
        gamma = float(newgamma)

        rr = xp.dot(r.T, r)
        if rr / rr0 < tol:
            break

    return x.get() if use_gpu and gpu_available else x


def _rrls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp,
          use_gpu, parallel, n_jobs, xp):
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
            pg.info("RRLS Iteration:", i, "residual:", rr, "relative:", rr / rr0)
        
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
            break
            
    return x.get() if use_gpu and gpu_available else x


