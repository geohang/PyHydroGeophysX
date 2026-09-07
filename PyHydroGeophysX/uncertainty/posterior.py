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
    returned. This is a correctness matter, not a speed one: building the
    posterior from chained pseudo-inverses returns a matrix that is not a
    covariance. On a Jacobian with condition number 1e5 the old form came back
    asymmetric with two negative eigenvalues out of sixty, and feeding it to
    ``propagate_petro_uncertainty`` below made NumPy report "covariance is not
    symmetric positive-semidefinite" and sample from it anyway.
    """
    J = np.asarray(J, dtype=float)
    Cd = np.asarray(Cd, dtype=float)
    Cm_prior = np.asarray(Cm_prior, dtype=float)
    n_model = J.shape[1]

    # J^T Cd^-1 J, without ever forming Cd^-1.
    if Cd.ndim == 1:
        jt_cdinv_j = J.T @ (J / np.clip(Cd, 1e-12, None)[:, None])
    else:
        jt_cdinv_j = J.T @ spd_solve(Cd, J, what="the data covariance Cd")

    prior_inv = spd_solve(
        Cm_prior, np.eye(n_model), what="the prior covariance Cm_prior"
    )

    lhs = symmetrize(jt_cdinv_j + prior_inv)
    return symmetrize(
        spd_solve(lhs, np.eye(n_model), what="the posterior precision matrix")
    )


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

    Uses Monte Carlo sampling by default.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    rho_cov = np.asarray(rho_cov, dtype=float)

    if rho_cov.ndim == 1:
        rho_cov = np.diag(np.clip(rho_cov, 1e-12, None))

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(rho, rho_cov, size=int(n_samples))

    transformed = np.array([np.asarray(petro_func(s), dtype=float).ravel() for s in samples])

    return {
        "mean": np.mean(transformed, axis=0),
        "std": np.std(transformed, axis=0),
        "cov": np.cov(transformed.T),
        "samples": transformed,
    }
