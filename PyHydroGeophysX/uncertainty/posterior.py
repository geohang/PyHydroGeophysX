"""Posterior covariance and uncertainty propagation helpers."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from PyHydroGeophysX.solvers.linear_solvers import spd_solve, symmetrize


def linearized_posterior(
    J: Any,
    Cd: Any,
    Cm_prior: Any,
) -> Any:
    """
    Compute linearized Gaussian posterior covariance:

    Cm_post = (J^T Cd^-1 J + Cm_prior^-1)^-1

    None of the three inverses in that expression is formed. Each is applied as
    a Cholesky solve instead, and the result is symmetrized before it is
    returned. This is a correctness matter, not a speed one: a posterior built
    from chained pseudo-inverses is not a covariance when the Jacobian is
    ill-conditioned (it comes out asymmetric, with negative eigenvalues), and
    sampling from it in ``propagate_petro_uncertainty`` below would be invalid.
    """
    J = np.asarray(J, dtype=float)
    Cd = np.asarray(Cd, dtype=float)
    Cm_prior = np.asarray(Cm_prior, dtype=float)
    n_model = J.shape[1]

    # J^T Cd^-1 J, without ever forming Cd^-1.
    if Cd.ndim == 1:
        if Cd.size != J.shape[0] or not np.all(np.isfinite(Cd) & (Cd > 0)):
            raise ValueError("Diagonal data covariance must have one positive finite variance per datum.")
        # Variance has physical units: a fixed floor changes the statistical
        # model on conversion of units and disagrees with the dense diagonal.
        jt_cdinv_j = J.T @ (J / Cd[:, None])
    else:
        jt_cdinv_j = J.T @ spd_solve(Cd, J, what="the data covariance Cd")

    prior_inv = spd_solve(
        Cm_prior, np.eye(n_model), what="the prior covariance Cm_prior"
    )

    lhs = symmetrize(jt_cdinv_j + prior_inv)
    return symmetrize(
        spd_solve(lhs, np.eye(n_model), what="the posterior precision matrix")
    )


#: Most cells :func:`regularized_posterior_std` forms the covariance for: the
#: dense Hessian and its inverse take about 16 n^2 bytes (0.6 GB at 6000 cells).
POSTERIOR_MAX_CELLS = 6000


def regularized_posterior_std(
    J: Any,
    data_weights: Any,
    Wm: Any,
    lam: float,
    alpha: float = 0.0,
    free: Optional[Any] = None,
) -> np.ndarray:
    """Per-cell standard deviation of a regularised Gauss-Newton model, linearised.

    Reading the data weights as Gaussian data errors and the regularisation as
    a Gaussian prior, the model's covariance at the final iterate is

    C = (J^T Wd^2 J + lambda (Wm^T Wm + alpha I))^-1

    (Tarantola, 2005, Inverse Problem Theory, SIAM, eq. 3.38; Aster, Borchers
    and Thurber, 2018, Parameter Estimation and Inverse Problems, ch. 11).
    Returned is sqrt(diag C), in the units of the model parameter (for an
    inversion in log resistivity, a standard deviation of ln rho).

    What it leaves out: the error of the regularisation itself. A smoothness
    prior cannot describe a sharp layer boundary, so where the truth is blocky
    the real error exceeds this: the result is a lower bound there, most of
    all at depth and where coverage is low.

    Parameters
    ----------
    J : array_like
        Jacobian at the final model, data x cells, in the inversion's own
        parameterisation (log data against log model for ERT).
    data_weights : array_like
        Diagonal of Wd, one per datum (1 / the data error).
    Wm : array_like or sparse matrix
        Model weighting (constraint) matrix, rows x cells.
    lam : float
        Regularisation weight lambda, as in the objective above.
    alpha : float
        Smallness weight relative to the smoothness (0 for none).
    free : array_like of bool, optional
        Cells the inversion updated; held cells get a standard deviation of 0.

    Returns
    -------
    numpy.ndarray
        One standard deviation per cell.

    Raises
    ------
    ValueError
        When more than :data:`POSTERIOR_MAX_CELLS` free cells would have to be
        formed, the shapes disagree, or the posterior precision is not positive definite.

    Examples
    --------
    >>> J = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> regularized_posterior_std(J, [10.0, 10.0], np.zeros((1, 2)), 1.0, alpha=1.0).round(4)
    array([0.0995, 0.0995])
    >>> regularized_posterior_std([[1, 0]], [1], [[0, 0]], 1, alpha=0)
    Traceback (most recent call last):
        ...
    ValueError: The posterior is improper: some cells are constrained neither by the data nor by the regularisation.
    """
    import scipy.sparse as sp
    from scipy.linalg import cho_factor, cho_solve

    J = np.asarray(J, dtype=float)
    weights = np.asarray(data_weights, dtype=float).ravel()
    n = J.shape[1]
    if weights.size != J.shape[0]:
        raise ValueError("data_weights needs one weight per datum.")
    keep = np.ones(n, dtype=bool) if free is None else np.asarray(free, dtype=bool).ravel()
    if keep.size != n:
        raise ValueError("free needs one flag per cell.")
    count = int(keep.sum())
    if count > POSTERIOR_MAX_CELLS:
        raise ValueError(f"{count} free cells is more than POSTERIOR_MAX_CELLS "
                         f"({POSTERIOR_MAX_CELLS}); the dense covariance is not formed.")
    std = np.zeros(n)
    if count == 0:
        return std
    Jf = J[:, keep]
    hessian = Jf.T @ (Jf * (weights ** 2)[:, None])
    Wm = sp.csr_matrix(Wm)[:, keep] if sp.issparse(Wm) else np.asarray(Wm, dtype=float)[:, keep]
    smooth = Wm.T @ Wm
    smooth = smooth.toarray() if sp.issparse(smooth) else np.asarray(smooth)
    smooth *= float(lam)
    hessian += smooth
    del smooth
    if alpha:
        hessian[np.diag_indices(count)] += float(lam) * float(alpha)
    try:
        factor = cho_factor(symmetrize(hessian))
    except np.linalg.LinAlgError as exc:
        raise ValueError("The posterior is improper: some cells are constrained neither "
                         "by the data nor by the regularisation.") from exc
    covariance = cho_solve(factor, np.eye(count))
    std[keep] = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return std


def model_resolution_spread(
    R: Any,
) -> Any:
    """Return diagonal resolution spread metrics from a resolution matrix."""
    R = np.asarray(R, dtype=float)
    diag = np.diag(R)
    return {
        "diagonal": diag,
        "spread": 1.0 - np.clip(diag, 0.0, 1.0),
        "mean_resolution": float(np.mean(diag)),
    }


def propagate_petro_uncertainty(
    rho: Any,
    rho_cov: Any,
    petro_func: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Propagate resistivity uncertainty through a petrophysical transform.

    Uses Monte Carlo sampling by default. ``"cov"`` is always a 2-D
    (outputs, outputs) matrix, (1, 1) for one cell or a scalar output.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    rho_cov = np.asarray(rho_cov, dtype=float)

    if rho_cov.ndim == 1:
        if rho_cov.size != rho.size or not np.all(np.isfinite(rho_cov) & (rho_cov >= 0)):
            raise ValueError("Diagonal resistivity covariance must have one nonnegative finite variance per cell.")
        # A zero variance is deterministic, and tiny variances are not noise
        # floors. Match the same covariance supplied as a full diagonal matrix.
        rho_cov = np.diag(rho_cov)

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(rho, rho_cov, size=int(n_samples))

    transformed = np.array([np.asarray(petro_func(s), dtype=float).ravel() for s in samples])

    return {
        "mean": np.mean(transformed, axis=0),
        "std": np.std(transformed, axis=0),
        # np.cov returns a 0-D array for a single output, which np.diag and
        # cov[0, 0] both reject.
        "cov": np.atleast_2d(np.cov(transformed.T)),
        "samples": transformed,
    }
