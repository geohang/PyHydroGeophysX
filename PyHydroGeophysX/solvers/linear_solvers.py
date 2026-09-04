"""
Linear solvers for geophysical inversion.
"""
import inspect
import math
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
from scipy.sparse import linalg as splinalg

# CuPy must stay lazy. Importing it while a CPU-only workflow is starting loads
# CUDA DLLs into the process before PyGIMLi loads its native solver DLLs. On
# Windows, environment-specific DLL versions can then collide even though the
# run never requested a GPU solver (observed as 0xC06D007F / procedure not
# found). Keep these public names for compatibility, but populate them only when
# a caller explicitly asks for ``use_gpu=True``.
cp = None
csr_matrix = None
GPU_AVAILABLE = False
_GPU_BACKEND_CHECKED = False


def _load_gpu_backend() -> bool:
    """Import CuPy once, and only for an explicitly requested GPU solve."""
    global cp, csr_matrix, GPU_AVAILABLE, _GPU_BACKEND_CHECKED
    if _GPU_BACKEND_CHECKED:
        return bool(GPU_AVAILABLE)
    _GPU_BACKEND_CHECKED = True
    try:
        import cupy as _cp
        from cupyx.scipy.sparse import csr_matrix as _csr_matrix
    except Exception:  # noqa: BLE001 - optional DLL imports can raise OSError
        cp = None
        csr_matrix = None
        GPU_AVAILABLE = False
        return False
    cp = _cp
    csr_matrix = _csr_matrix
    GPU_AVAILABLE = True
    return True

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# SPD solver constants
# ---------------------------------------------------------------------------
# Three inversion call sites (inversion/time_lapse.py, inversion/srt_inversion.py
# and inversion/srt_time_lapse.py) assemble the Gauss-Newton normal matrix
# H = J^T W_d^T W_d J + lambda W_m^T W_m + ... and pass it to this module. Every
# least-squares method here applies A and A.T, so on such a matrix it solves
# H^T H d = H^T (-g): the condition number is squared, and because H is symmetric
# each iteration spends two matrix-vector products where one would do. The
# spd_cholesky and spd_cg methods solve H d = -g directly instead.

_SPD_RIDGE_REL = 1e-12
_SPD_DENSE_COPY_WARN_BYTES = 2 * 1024 ** 3
_SYM_PROBE_PAIRS = 2
_SQUARE_LS_WARNED = False
_CG_RTOL_KEYWORD = None  # type: Optional[str]
_LEAST_SQUARES_METHODS = frozenset(
    {"lsqr", "rrlsqr", "cgls", "rrls", "scipy_lsqr", "scipy_lsmr", "precond_lsmr"}
)
_SPD_METHODS = frozenset({"spd_cholesky", "spd_cg"})


def _info(*args: Any) -> None:
    """Emit solver progress on stderr.

    The verbose branches below used to call ``pg.info`` even though pygimli is
    never imported here, so every one of them raised ``NameError`` the moment a
    caller passed ``verbose=True``. This module is numpy/scipy only, so it must
    not pull in a heavy optional backend just to log. stderr keeps stdout free
    for the machine-readable output of generated workflow scripts.
    """
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------------------
# scalar dot
# ---------------------------------------------------------------------------
def _scalar_dot(xp, a, b):
    """Return scalar dot product regardless of vector shape (works for NumPy/CuPy)."""
    return float(xp.vdot(a.ravel(), b.ravel()))


# ---------------------------------------------------------------------------
# scipy lsmr solve
# ---------------------------------------------------------------------------
def _scipy_lsmr_solve(A, b, maxiter=400, tol=1e-8, damp=0.0, **kwargs):
    """
    Solve min ||Ax - b|| using SciPy's LSMR algorithm.

    LSMR is mathematically equivalent to applying MINRES to the normal
    equations and converges faster than LSQR for ill-conditioned systems.
    """
    if not scipy.sparse.isspmatrix(A):
        A = scipy.sparse.csr_matrix(A)
    b_flat = np.asarray(b, dtype=float).ravel()
    result = splinalg.lsmr(A, b_flat, atol=tol, btol=tol, maxiter=maxiter, damp=damp)
    x = result[0]
    return np.asarray(x, dtype=float).reshape(-1, 1)


# ---------------------------------------------------------------------------
# precond lsmr solve
# ---------------------------------------------------------------------------
def _precond_lsmr_solve(A, b, maxiter=400, tol=1e-8, damp=0.0, **kwargs):
    """
    Solve min ||Ax - b|| using LSMR with column-scaling (Jacobi) preconditioning.

    Computes column norms d_j = ||A[:,j]|| and solves the preconditioned
    system (A D^{-1})(D x) = b, then recovers x = D^{-1} y.  This
    balances parameter sensitivities across the different block rows
    (data, regularization, cross-gradient) of the stacked system.
    """
    if not scipy.sparse.isspmatrix(A):
        A = scipy.sparse.csr_matrix(A)
    b_flat = np.asarray(b, dtype=float).ravel()

    # Column-norm scaling
    col_norms = scipy.sparse.linalg.norm(A, axis=0)
    col_norms = np.asarray(col_norms, dtype=float).ravel()
    col_norms[col_norms < 1e-12] = 1.0  # avoid division by zero

    D_inv = scipy.sparse.diags(1.0 / col_norms, format="csr")
    A_scaled = A.dot(D_inv)

    result = splinalg.lsmr(A_scaled, b_flat, atol=tol, btol=tol, maxiter=maxiter, damp=damp)
    y = result[0]

    # Recover original variables: x = D^{-1} y
    x = y / col_norms
    return np.asarray(x, dtype=float).reshape(-1, 1)


# ---------------------------------------------------------------------------
# SPD helpers
# ---------------------------------------------------------------------------
def _as_dense_base(A: Any) -> Any:
    """Return a dense ``A`` as an ndarray view, and a sparse ``A`` untouched.

    ``time_lapse.py`` assembles ``H`` from a product involving a scipy
    ``spmatrix``, which numpy resolves to ``np.matrix``. ``np.matrix.dot`` on a
    1-D vector returns shape ``(1, n)`` rather than ``(n,)``, which would
    quietly corrupt both the symmetry probe and the CG operator below.
    ``np.asarray`` is a zero-copy view of the same buffer, so an in-place
    factorization still reaches the caller's memory.
    """
    if scipy.sparse.issparse(A):
        return A
    return np.asarray(A)


def _probe_symmetric(A: Any, n_probes: Any = _SYM_PROBE_PAIRS) -> bool:
    """Test ``A == A.T`` with random bilinear probes rather than an n^2 compare.

    ``w^T A v == v^T A w`` for every pair exactly when ``A`` is symmetric. Each
    pair costs two matrix-vector products, against the ``2 * maxiter`` a CGLS
    solve is about to spend, and it never allocates the n-by-n temporary that
    ``np.allclose(A, A.T)`` would. The generator is seeded, so a given matrix
    always receives the same verdict and a surprising one can be reproduced.
    """
    Ab = _as_dense_base(A)
    shape = getattr(Ab, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        return False
    n = int(shape[0])
    try:
        eps = float(np.finfo(Ab.dtype).eps)
    except (TypeError, ValueError):
        eps = float(np.finfo(np.float64).eps)
    # w^T A v sums n^2 products, so rounding accumulates like sqrt(n)*eps*||A||,
    # while a genuinely nonsymmetric matrix differs by O(1). The threshold only
    # has to sit between the two; the cap keeps the float32 path from going so
    # slack that a nonsymmetric matrix reads as symmetric.
    rtol = min(1e-4, max(1e-10, 1e3 * eps * math.sqrt(max(n, 1))))
    rng = np.random.default_rng(0)
    for _ in range(int(n_probes)):
        v = rng.standard_normal(n)
        w = rng.standard_normal(n)
        wav = float(np.dot(w, np.asarray(Ab.dot(v)).ravel()))
        vaw = float(np.dot(v, np.asarray(Ab.dot(w)).ravel()))
        scale = max(abs(wav), abs(vaw))
        if scale <= 0.0:
            return False  # a zero probe learned nothing; stay quiet
        if abs(wav - vaw) > rtol * scale:
            return False
    return True


def _warn_if_least_squares_on_symmetric(A: Any, method: Any) -> None:
    """Warn once per process when a least-squares method gets a normal matrix.

    The gate runs cheapest test first. A rectangular ``A`` -- the stacked systems
    in ``ert_inversion.py`` and ``joint_ert_srt.py`` -- costs one integer
    comparison and returns. Only a square matrix pays for the probe, and only
    until the first warning fires.
    """
    global _SQUARE_LS_WARNED
    if _SQUARE_LS_WARNED:
        return
    shape = getattr(A, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1] or shape[0] < 2:
        return
    if not _probe_symmetric(A):
        return
    _SQUARE_LS_WARNED = True
    warnings.warn(
        "method={0!r} is a least-squares solver, but 'A' is square and "
        "symmetric, so it looks like a Gauss-Newton normal matrix. The solver "
        "therefore works on 'A^T A x = A^T b': the effective condition number "
        "is squared, and each iteration costs two matrix-vector products where "
        "one would do. Pass method='spd_cholesky' for an exact factorization, "
        "or method='spd_cg' when the matrix is too large to factor. Reported "
        "once per process.".format(method),
        RuntimeWarning,
        stacklevel=3,
    )


def _cg_rtol_keyword() -> Any:
    """Name of SciPy's relative-tolerance keyword for ``cg`` on this install.

    SciPy renamed ``tol`` to ``rtol`` in 1.12 and removed ``tol`` in 1.14.
    ``pyproject.toml`` pins ``scipy>=1.8,<3.0``, which straddles both changes.
    """
    global _CG_RTOL_KEYWORD
    if _CG_RTOL_KEYWORD is None:
        try:
            params = inspect.signature(splinalg.cg).parameters
        except (TypeError, ValueError):
            params = {}
        _CG_RTOL_KEYWORD = "rtol" if "rtol" in params else "tol"
    return _CG_RTOL_KEYWORD


def _spd_ridge(A_dense: Any) -> float:
    """Rescue ridge for a matrix that is positive definite only to within rounding.

    A Gauss-Newton normal matrix is positive definite in exact arithmetic, but it
    is assembled as a sum of products, so its smallest eigenvalue can land a
    rounding step below zero and LAPACK's ``potrf`` then refuses it. The ridge is
    relative to the mean absolute diagonal, floored at the working dtype's own
    resolution: ``time_lapse.py`` uses float32 when ``save_memory=True``, where a
    fixed 1e-12 sits three orders below eps (1.19e-7) and would do nothing.

    This is applied only after an unridged factorization has failed. Applying it
    unconditionally is tempting and measurably wrong: on synthetic SPD systems
    the ridge is a few percent of the smallest eigenvalue once ``cond(A)``
    reaches 1e12, and it costs three to four orders of magnitude of accuracy
    (6.4e-3 relative error against 1.1e-5 without it) on exactly the
    ill-conditioned problems this module exists to solve.
    """
    diag = np.abs(np.diagonal(A_dense))
    diag_mean = float(np.mean(diag)) if diag.size else 0.0
    if not np.isfinite(diag_mean) or diag_mean <= 0.0:
        return 0.0
    if np.issubdtype(A_dense.dtype, np.floating):
        eps = float(np.finfo(A_dense.dtype).eps)
    else:
        eps = float(np.finfo(np.float64).eps)
    return max(_SPD_RIDGE_REL, 10.0 * eps) * diag_mean


def _require_square(A: Any, method: Any) -> Any:
    """Shape guard shared by the two SPD methods."""
    shape = getattr(A, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            "method={0!r} requires a square matrix; got {1}. Use a "
            "least-squares method ('cgls', 'scipy_lsmr') for a rectangular "
            "system.".format(method, shape)
        )
    return shape


# ---------------------------------------------------------------------------
# spd_cholesky
# ---------------------------------------------------------------------------
def _spd_cholesky_solve(A, b, damp=0.0, overwrite_a=False, verbose=False, **kwargs):
    """Solve ``A x = b`` exactly for symmetric positive definite ``A``.

    Unlike the least-squares methods in this module, this never forms ``A^T A``,
    so the accuracy of the result is governed by ``cond(A)`` rather than by
    ``cond(A)^2``, and there is no convergence question to get wrong.

    The factorization is attempted unridged. Only if it fails is a small ridge
    added to the diagonal and the factorization retried, and only if that fails
    too does the solve fall back to a symmetric-indefinite (LDL^T) factorization.
    Ridging unconditionally would cost three to four orders of magnitude of
    accuracy on the ill-conditioned problems this exists to solve; see
    ``_spd_ridge``.

    Set ``overwrite_a=True`` when the caller has no further use for ``A``: the
    factorization then runs in the caller's own buffer and allocates nothing,
    which for a dense normal matrix of a few tens of thousands of unknowns is
    the difference between one working copy and none. The tradeoff is that a
    partial factorization has already destroyed ``A`` by the time a failure is
    detected, so neither the ridge retry nor the LDL^T fallback is available;
    that case raises instead.
    """
    shape = _require_square(A, "spd_cholesky")
    b_flat = np.asarray(b, dtype=float).ravel()
    n = int(shape[0])

    if scipy.sparse.issparse(A):
        # SciPy has no sparse Cholesky; ``scipy.sparse.linalg.cholesky`` simply
        # does not exist, which is the bug repaired in ``direct_solver`` below.
        # CHOLMOD lives in scikit-sparse, which is not a dependency. SuperLU
        # ignores symmetry and so does roughly twice the work of a Cholesky, but
        # it is exact and it keeps the matrix sparse, which is the entire point
        # of the ``save_memory=True`` path that produces a sparse H.
        mat = A.tocsc()
        if damp:
            mat = mat + float(damp) * scipy.sparse.eye(n, format="csc", dtype=mat.dtype)
        x = np.asarray(splinalg.spsolve(mat, b_flat), dtype=float)
        if not np.all(np.isfinite(x)):
            # spsolve warns and returns NaN on a singular matrix rather than
            # raising. Do not let that escape as if it were a solution.
            raise np.linalg.LinAlgError(
                "spsolve returned a non-finite solution; the sparse matrix is "
                "singular to working precision."
            )
        if verbose:
            _info("spd_cholesky: sparse LU (SuperLU) path, n =", n)
        return x.reshape(-1, 1)

    dense = _as_dense_base(A)
    idx = np.arange(n)
    damp = float(damp)

    in_place = (
        bool(overwrite_a)
        and dense.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and dense.flags.writeable
        and (dense.flags.c_contiguous or dense.flags.f_contiguous)
    )

    if in_place:
        # LAPACK wants Fortran order. For a C-contiguous array ``dense.T`` is an
        # F-contiguous view of the same buffer, and ``A`` is symmetric, so the
        # transpose is the same matrix with ``lower`` flipped. Recent SciPy
        # already factors a C-ordered symmetric array in place, so this is
        # insurance for the older end of the ``scipy>=1.8`` pin rather than a
        # saving on every install.
        if dense.flags.c_contiguous:
            work, lower = dense.T, False
        else:
            work, lower = dense, True
        if damp:
            dense[idx, idx] += damp
        try:
            factor = scipy.linalg.cho_factor(work, lower=lower, overwrite_a=True)
        except np.linalg.LinAlgError as exc:
            # A partial potrf has already overwritten the buffer, so neither the
            # rescue ridge nor the LDL^T fallback below is reachable from here.
            # This is the price of overwrite_a=True, and the message says so.
            raise np.linalg.LinAlgError(
                "Cholesky factorization failed: 'A' is not positive definite to "
                "working precision. 'A' was consumed by overwrite_a=True, so the "
                "ridge retry and the symmetric-indefinite fallback are both out "
                "of reach. Retry with overwrite_a=False, or use method='spd_cg'."
            ) from exc
        if verbose:
            _info("spd_cholesky: in-place factorization, n =", n)
        return np.asarray(
            scipy.linalg.cho_solve(factor, b_flat), dtype=float
        ).reshape(-1, 1)

    nbytes = int(getattr(dense, "nbytes", 0))
    if nbytes > _SPD_DENSE_COPY_WARN_BYTES:
        warnings.warn(
            "spd_cholesky is copying a {0:.1f} GiB matrix. Pass "
            "overwrite_a=True to factor in place when 'A' is not needed "
            "afterwards, or use method='spd_cg'.".format(nbytes / 1024.0 ** 3),
            RuntimeWarning,
            stacklevel=3,
        )

    def _fresh(extra):
        # np.asfortranarray returns the SAME object for an array that is already
        # F-ordered float64, so a shift applied to its result would reach the
        # caller's matrix even though overwrite_a is off. Copy explicitly.
        out = np.array(dense, dtype=float, order="F", copy=True)
        if damp or extra:
            out[idx, idx] += damp + extra
        return out

    try:
        factor = scipy.linalg.cho_factor(_fresh(0.0), lower=True, overwrite_a=True)
    except np.linalg.LinAlgError:
        ridge = _spd_ridge(dense)
        warnings.warn(
            "Cholesky factorization failed: 'A' is not positive definite to "
            "working precision. Retrying with a ridge of {0:g} on the "
            "diagonal.".format(ridge),
            RuntimeWarning,
            stacklevel=3,
        )
        try:
            factor = scipy.linalg.cho_factor(
                _fresh(ridge), lower=True, overwrite_a=True
            )
        except np.linalg.LinAlgError:
            warnings.warn(
                "The ridged Cholesky factorization also failed; 'A' is "
                "indefinite rather than merely borderline. Falling back to a "
                "symmetric-indefinite (LDL^T) solve.",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.asarray(
                scipy.linalg.solve(
                    _fresh(0.0), b_flat, assume_a="sym", overwrite_a=True
                ),
                dtype=float,
            ).reshape(-1, 1)
        if verbose:
            _info("spd_cholesky: factorization on a copy, n =", n, "ridge =", ridge)
        return np.asarray(
            scipy.linalg.cho_solve(factor, b_flat), dtype=float
        ).reshape(-1, 1)
    if verbose:
        _info("spd_cholesky: factorization on a copy, n =", n)
    return np.asarray(
        scipy.linalg.cho_solve(factor, b_flat), dtype=float
    ).reshape(-1, 1)


# ---------------------------------------------------------------------------
# spd_cg
# ---------------------------------------------------------------------------
def _spd_cg_solve(A, b, x=None, maxiter=200, tol=1e-8, damp=0.0, verbose=False,
                  **kwargs):
    """Solve ``A x = b`` for symmetric positive definite ``A`` by conjugate gradients.

    For matrices too large to factor. CG applies ``A`` once per iteration and
    converges at a rate set by ``sqrt(cond(A))``, where CGLS on the same matrix
    applies it twice and converges at a rate set by ``cond(A)``.

    A caution on ``tol``. ``_cgls`` compares ``r . r`` against ``r0 . r0``, a
    squared residual ratio, whereas SciPy's ``cg`` measures on the norm scale.
    The same numeric ``tol`` is therefore stricter here by a square root: code
    migrating from ``cgls`` and wanting comparable effort should pass
    ``tol=sqrt(old_tol)``.
    """
    shape = _require_square(A, "spd_cg")
    b_flat = np.asarray(b, dtype=float).ravel()
    operator = _as_dense_base(A)
    if damp:
        base = operator
        shift = float(damp)
        operator = splinalg.LinearOperator(
            shape,
            matvec=lambda v: np.asarray(base.dot(v)).ravel()
            + shift * np.asarray(v, dtype=float).ravel(),
            dtype=float,
        )

    x0 = None if x is None else np.asarray(x, dtype=float).ravel()
    seen = {"n": 0}

    def _count(_xk):
        seen["n"] += 1

    # Count iterations only when asked. scipy hands the callback ``xk``, not the
    # residual, so measuring one per iteration would cost an extra mat-vec.
    callback = _count if verbose else None

    # atol=0.0 makes the test purely relative and also silences the changing
    # default that scipy < 1.12 warns about.
    kwds = {_cg_rtol_keyword(): float(tol), "atol": 0.0, "maxiter": int(maxiter)}
    sol, info = splinalg.cg(operator, b_flat, x0=x0, callback=callback, **kwds)

    if info < 0:
        raise RuntimeError(
            "scipy.sparse.linalg.cg reported illegal input or breakdown "
            "(info={0}).".format(info)
        )
    if info > 0:
        # Return the iterate rather than raising. Every caller feeds this into
        # an Armijo line search that rejects a bad step, so raising would turn a
        # slow solve into a failed inversion; staying silent would reproduce the
        # blind spot in _cgls, which reports nothing when it exhausts maxiter.
        warnings.warn(
            "spd_cg did not reach tol={0:g} within maxiter={1}; returning the "
            "current iterate. Note that 'cgls' compares a squared residual "
            "ratio while cg measures on the norm scale, so the same 'tol' is "
            "stricter here; pass tol=sqrt(old_tol) for comparable "
            "effort.".format(float(tol), int(maxiter)),
            RuntimeWarning,
            stacklevel=3,
        )
    if verbose:
        _info("spd_cg: iterations", seen["n"], "info", info)
    return np.asarray(sol, dtype=float).reshape(-1, 1)


# ---------------------------------------------------------------------------
# generalized solver
# ---------------------------------------------------------------------------
def generalized_solver(
    A: Any,
    b: Any,
    method: Any = "cgls",
    x: Any = None,
    maxiter: Any = 200,
    tol: Any = 1e-8,
    verbose: Any = False,
    damp: Any = 0.0,
    use_gpu: Any = False,
    parallel: Any = False,
    n_jobs: Any = -1,
    *,
    overwrite_a: Any = False,
) -> Any:
    """
    Generalized solver for Ax = b with optional GPU acceleration and parallelism.

    Parameters:
    -----------
    A : array_like or sparse matrix
        The system matrix. Two shapes of problem reach this function, and they
        need opposite solver families:

        1. A stacked least-squares system, ``[W_d J; sqrt(lambda) W_m; ...]``,
           with a matching stacked right-hand side. Use a least-squares method.
        2. A Gauss-Newton normal matrix, ``J^T W_d^T W_d J + lambda W_m^T W_m
           + ...``, which is square and symmetric positive definite, paired with
           a gradient right-hand side ``-g``. Use an SPD method. A least-squares
           method applied to this case solves ``A^T A x = A^T b`` instead, which
           squares the condition number and spends two matrix-vector products
           per iteration where one would do; passing one is warned about once
           per process.
    b : array_like
        Right-hand side vector.
    method : str, optional
        Solver method. Default is 'cgls'.
        Iterative:  'lsqr', 'rrlsqr', 'cgls', 'rrls'
        SciPy:      'scipy_lsqr', 'scipy_lsmr', 'precond_lsmr'
        SPD, CPU-only, for a square symmetric positive definite ``A``:
                    'spd_cholesky' (exact factorization), 'spd_cg' (iterative,
                    for a matrix too large to factor)
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
    overwrite_a : bool, optional
        Keyword-only. 'spd_cholesky' only. Factor in the caller's own buffer,
        which allocates nothing but destroys ``A``. Set it only when ``A`` is
        not needed after the call. Ignored by every other method.

    Returns:
    --------
    x : array_like
        The computed solution vector.
    """
    m = method.lower().strip()

    # Report a normal matrix handed to a least-squares method. Placed above the
    # dispatch so it covers all seven, not just the four in the second block.
    if m in _LEAST_SQUARES_METHODS:
        _warn_if_least_squares_on_symmetric(A, m)

    # SPD solvers for a square symmetric positive definite A. These return here,
    # before the GPU backend selection and before the A.T r precomputation that
    # the least-squares path needs and they do not. Keeping them ahead of that
    # boundary is what stops an spd request from importing CuPy.
    if m in _SPD_METHODS:
        if use_gpu:
            warnings.warn(
                "method={0!r} is CPU-only; use_gpu is ignored.".format(method),
                RuntimeWarning,
                stacklevel=2,
            )
        if m == "spd_cholesky":
            return _spd_cholesky_solve(
                A, b, damp=damp, overwrite_a=overwrite_a, verbose=verbose
            )
        return _spd_cg_solve(
            A, b, x=x, maxiter=maxiter, tol=tol, damp=damp, verbose=verbose
        )

    # SciPy-backed solvers (no custom state needed)
    if m == "scipy_lsmr":
        return _scipy_lsmr_solve(A, b, maxiter=maxiter, tol=tol, damp=damp)
    if m == "precond_lsmr":
        return _precond_lsmr_solve(A, b, maxiter=maxiter, tol=tol, damp=damp)
    if m == "scipy_lsqr":
        if not scipy.sparse.isspmatrix(A):
            A = scipy.sparse.csr_matrix(A)
        b_flat = np.asarray(b, dtype=float).ravel()
        result = splinalg.lsqr(A, b_flat, atol=tol, btol=tol, iter_lim=maxiter, damp=damp)
        return np.asarray(result[0], dtype=float).reshape(-1, 1)

    # Choose the backend (NumPy or CuPy). A CPU solve never imports CuPy.
    gpu_available = _load_gpu_backend() if use_gpu else False
    if use_gpu and gpu_available:
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
    gamma = _scalar_dot(xp, s, s)
    rr = _scalar_dot(xp, r, r)
    rr0 = rr

    # Choose the solver routine based on method
    if m == "lsqr":
        return _lsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif m == "rrlsqr":
        return _rrlsqr(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif m == "cgls":
        return _cgls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    elif m == "rrls":
        return _rrls(A, b, x, r, s, gamma, rr, rr0, maxiter, tol, verbose, damp, use_gpu, parallel, n_jobs, xp)
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported least-squares methods: "
            "'lsqr', 'rrlsqr', 'cgls', 'rrls', 'scipy_lsqr', 'scipy_lsmr', "
            "'precond_lsmr'. Supported SPD methods, for a square symmetric "
            "positive definite matrix: 'spd_cholesky', 'spd_cg'."
        )


# ---------------------------------------------------------------------------
# matrix multiply
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# cgls
# ---------------------------------------------------------------------------
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
            _info("CGLS Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
        # Compute A*p
        q = _matrix_multiply(A, p, use_gpu, parallel, n_jobs, xp)
        
        # Add damping if requested
        if damp > 0:
            q += damp * p
        
        # Ensure q is a column vector
        q = q.reshape(-1, 1)
        
        # Compute step size
        denom = _scalar_dot(xp, q, q)
        alpha = gamma / denom
        
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
        gamma_new = _scalar_dot(xp, s, s)
        beta = gamma_new / gamma
        
        # Update search direction
        p = s + beta * p
        
        # Update gamma
        gamma = gamma_new
        
        # Check convergence
        rr = _scalar_dot(xp, r, r)
        if rr / rr0 < tol:
            if verbose:
                _info(f"CGLS converged after {i+1} iterations")
            break
    
    # Return solution (convert back to CPU if on GPU)
    return x.get() if use_gpu else x


# ---------------------------------------------------------------------------
# lsqr
# ---------------------------------------------------------------------------
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
            _info("LSQR Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
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
                _info(f"LSQR converged after {i+1} iterations")
            break
    
    return x.get() if use_gpu else x


# ---------------------------------------------------------------------------
# rrlsqr
# ---------------------------------------------------------------------------
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
            _info("RRLSQR Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
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
                _info(f"RRLSQR converged after {i+1} iterations")
            break
    
    return x.get() if use_gpu else x


# ---------------------------------------------------------------------------
# rrls
# ---------------------------------------------------------------------------
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
            _info("RRLS Iteration:", i, "residual:", float(rr), "relative:", float(rr / rr0))
        
        p = _matrix_multiply(A, w, use_gpu, parallel, n_jobs, xp)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
            
        denom = _scalar_dot(xp, p, p)
        if xp.isclose(denom, 0.0):
            break
            
        lam = _scalar_dot(xp, p, r) / denom
        x = x + w * float(lam)  # Convert lam to scalar
        r = r - p * float(lam)
        
        s = _matrix_multiply(A.T, r, use_gpu, parallel, n_jobs, xp)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
            
        if damp > 0:
            s = s + damp * x
            
        w = s
        rr = _scalar_dot(xp, r, r)
        if rr / rr0 < tol:
            if verbose:
                _info(f"RRLS converged after {i+1} iterations")
            break
            
    return x.get() if use_gpu else x


# ---------------------------------------------------------------------------
# Linear Solver
# ---------------------------------------------------------------------------
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
        gpu_available = _load_gpu_backend() if use_gpu else False
        self.use_gpu = bool(use_gpu and gpu_available)
        self.parallel = parallel and PARALLEL_AVAILABLE
        self.n_jobs = n_jobs
        self.damping = damping
        self.verbose = verbose
        
        # Check method
        valid_methods = ['cgls', 'lsqr', 'rrlsqr', 'rrls']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")
        
        # Check GPU availability
        if use_gpu and not gpu_available:
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


# ---------------------------------------------------------------------------
# CGLSSolver
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LSQRSolver
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# RRLSQRSolver
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# RRLSSolver
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# direct solver
# ---------------------------------------------------------------------------
def direct_solver(
    A: Any,
    b: Any,
    method: Any = "lu",
    **kwargs: Any,
) -> Any:
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
            # SciPy has no sparse Cholesky. This branch used to call
            # ``splinalg.cholesky``, which does not exist, so it raised
            # AttributeError on every call, the bare except swallowed it, and it
            # printed "Matrix not SPD" whether or not the matrix was SPD. SuperLU
            # ignores symmetry but is exact and keeps the matrix sparse.
            return splinalg.splu(A.tocsc()).solve(b)
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
            except np.linalg.LinAlgError:
                # Narrowed from a bare except, which reported a NaN or a shape
                # error as "not SPD" too.
                print("Warning: Matrix not SPD, falling back to LU")
                return scipy.linalg.solve(A, b)
        else:
            raise ValueError(f"Unknown direct solver method: {method}")


# ---------------------------------------------------------------------------
# Tikhonv Regularization
# ---------------------------------------------------------------------------
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
                except Exception:
                    # Fall back to LSQR
                    return splinalg.lsqr(A_aug, b_aug)[0]
            else:
                # Use LSQR for large systems
                return splinalg.lsqr(A_aug, b_aug)[0]
        else:
            # Use provided solver
            return solver.solve(A_aug, b_aug)


# ---------------------------------------------------------------------------
# Iterative Refinement
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# get optimal solver
# ---------------------------------------------------------------------------
def get_optimal_solver(
    A: Any,
    b: Any,
    estimate_condition: Any = True,
    time_limit: Any = None,
    memory_limit: Any = None,
) -> Any:
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
                    except Exception:
                        well_conditioned = True  # Assume well-conditioned if estimation fails
                else:
                    # For dense matrices, use SVD-based estimator
                    try:
                        s = scipy.linalg.svdvals(A)
                        condition_number = s[0] / s[-1]
                        well_conditioned = condition_number < 1e6
                    except Exception:
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
                        except Exception:
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
            except Exception:
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
