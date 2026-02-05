"""Posterior covariance and uncertainty propagation helpers."""

from typing import Callable, Dict, Optional

import numpy as np


def linearized_posterior(J, Cd, Cm_prior):
    """
    Compute linearized Gaussian posterior covariance:

    Cm_post = (J^T Cd^-1 J + Cm_prior^-1)^-1
    """
    J = np.asarray(J, dtype=float)
    Cd = np.asarray(Cd, dtype=float)
    Cm_prior = np.asarray(Cm_prior, dtype=float)

    if Cd.ndim == 1:
        Cd_inv = np.diag(1.0 / np.clip(Cd, 1e-12, None))
    else:
        Cd_inv = np.linalg.pinv(Cd)

    Cm_prior_inv = np.linalg.pinv(Cm_prior)

    lhs = J.T @ Cd_inv @ J + Cm_prior_inv
    Cm_post = np.linalg.pinv(lhs)
    return Cm_post


def model_resolution_spread(R):
    """Return diagonal resolution spread metrics from a resolution matrix."""
    R = np.asarray(R, dtype=float)
    diag = np.diag(R)
    return {
        "diagonal": diag,
        "spread": 1.0 - np.clip(diag, 0.0, 1.0),
        "mean_resolution": float(np.mean(diag)),
    }


def propagate_petro_uncertainty(
    rho,
    rho_cov,
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
