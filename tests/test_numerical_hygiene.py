"""Regression checks for symmetric solves and covariance calculations.

Cover solve accuracy, warning/fallback behavior, covariance symmetry and
sampling on well-conditioned and ill-conditioned synthetic systems. These
tests check numerical behavior; they do not establish performance ratios
or guarantee positive definiteness for arbitrary user-supplied matrices.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from PyHydroGeophysX.analysis.sensitivity import compute_resolution_matrix
from PyHydroGeophysX.assimilation.enkf import (
    ESMDA, EnsembleKalmanFilter, HydroGeophysObsOperator)
from PyHydroGeophysX.inversion.em1d_lci import _solve_normal_equations
from PyHydroGeophysX.solvers import spd_solve, symmetrize
from PyHydroGeophysX.uncertainty.posterior import (
    linearized_posterior, propagate_petro_uncertainty)


def _jacobian(n_data: int, n_model: int, cond: float, seed: int = 0) -> np.ndarray:
    """A Jacobian with exactly the requested condition number."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n_data, n_data)))
    V, _ = np.linalg.qr(rng.standard_normal((n_model, n_model)))
    k = min(n_data, n_model)
    return U[:, :k] @ np.diag(np.logspace(0.0, -np.log10(cond), k)) @ V[:, :k].T


# ---------------------------------------------------------------------------
# spd_solve
# ---------------------------------------------------------------------------
def test_spd_solve_matches_the_dense_reference() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 30))
    A = A @ A.T + 30.0 * np.eye(30)
    B = rng.standard_normal((30, 4))
    np.testing.assert_allclose(spd_solve(A, B), np.linalg.solve(A, B), rtol=1e-10)


def test_spd_solve_warns_and_names_the_quantity_when_it_degrades() -> None:
    """A caller should learn which matrix went indefinite, not just that one did."""
    with pytest.warns(RuntimeWarning, match="the prior covariance"):
        spd_solve(np.diag([1.0, 0.0]), np.ones((2, 1)), what="the prior covariance")


def test_symmetrize_averages_the_halves() -> None:
    np.testing.assert_allclose(
        symmetrize(np.array([[1.0, 2.0], [0.0, 1.0]])), [[1.0, 1.0], [1.0, 1.0]]
    )


# ---------------------------------------------------------------------------
# uncertainty.posterior
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cond", [1e2, 1e5, 1e8])
def test_the_posterior_is_a_covariance(cond) -> None:
    """Symmetric, and positive definite. The pinv chain satisfied neither.

    At cond(J) = 1e5 the old form returned a matrix that was asymmetric with two
    negative eigenvalues out of sixty.
    """
    n_model = 60
    J = _jacobian(120, n_model, cond)
    Cm_post = linearized_posterior(J, np.eye(120) * 1e-4, np.eye(n_model) * 1e6)

    np.testing.assert_array_equal(Cm_post, Cm_post.T)
    assert float(np.linalg.eigvalsh(Cm_post).min()) > 0.0
    assert np.all(np.diag(Cm_post) > 0.0)


def test_the_posterior_matches_the_exact_expression() -> None:
    n_model = 40
    J = _jacobian(80, n_model, 1e3)
    Cd, Cm = np.eye(80) * 1e-3, np.eye(n_model) * 5.0
    want = np.linalg.inv(J.T @ np.linalg.inv(Cd) @ J + np.linalg.inv(Cm))
    np.testing.assert_allclose(
        linearized_posterior(J, Cd, Cm), want, rtol=1e-7, atol=1e-12
    )


def test_a_diagonal_data_covariance_takes_the_vector_path() -> None:
    n_model = 20
    J = _jacobian(50, n_model, 1e2)
    diag = np.full(50, 1e-3)
    np.testing.assert_allclose(
        linearized_posterior(J, diag, np.eye(n_model) * 5.0),
        linearized_posterior(J, np.diag(diag), np.eye(n_model) * 5.0),
        rtol=1e-8,
    )


def test_the_posterior_can_be_sampled_from() -> None:
    """The whole point: the next call in this module has to accept the result.

    NumPy used to report "covariance is not symmetric positive-semidefinite" on
    exactly this chain and sample from it anyway.
    """
    n_model = 60
    J = _jacobian(120, n_model, 1e5)
    Cm_post = linearized_posterior(J, np.eye(120) * 1e-4, np.eye(n_model) * 1e6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = propagate_petro_uncertainty(
            np.full(n_model, 100.0), Cm_post, lambda r: r / 100.0,
            n_samples=50, seed=1,
        )
    assert np.all(np.isfinite(out["samples"]))


# ---------------------------------------------------------------------------
# analysis.sensitivity
# ---------------------------------------------------------------------------
def test_the_resolution_matrix_matches_the_exact_expression() -> None:
    n_model = 40
    J = _jacobian(90, n_model, 1e4)
    lam = 1e-6
    got = compute_resolution_matrix(J, np.ones(90), np.eye(n_model), lam)
    jtj = J.T @ J
    want = np.linalg.solve(jtj + lam * np.eye(n_model), jtj)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)


def test_a_heavily_regularized_resolution_matrix_tends_to_zero() -> None:
    """Physical sanity: crank lambda up and the data resolve nothing."""
    n_model = 25
    J = _jacobian(60, n_model, 1e2)
    R = compute_resolution_matrix(J, np.ones(60), np.eye(n_model), 1e12)
    assert float(np.abs(np.trace(R))) < 1e-6


# ---------------------------------------------------------------------------
# assimilation.enkf
# ---------------------------------------------------------------------------
def test_the_kalman_gain_matches_the_explicit_inverse() -> None:
    """The solve has to agree with the inverse it replaced on a well-posed case."""
    rng = np.random.default_rng(3)
    n_state, n_obs, n_members = 10, 6, 40
    A = rng.standard_normal((n_obs, n_state))
    operator = HydroGeophysObsOperator(
        petro_transform=lambda s: s, forward_operator=lambda m: A @ m
    )
    obs_cov = np.eye(n_obs) * 0.05
    ens = rng.standard_normal((n_state, n_members))

    filt = EnsembleKalmanFilter(operator, obs_cov)
    got = filt.update(ens, A @ np.ones(n_state), rng=np.random.default_rng(9))

    # Rebuild the update with an explicit inverse and the same noise draw.
    Y = np.column_stack([operator(ens[:, i]) for i in range(n_members)])
    anom = lambda M: M - M.mean(axis=1, keepdims=True)  # noqa: E731
    Pxy = (anom(ens) @ anom(Y).T) / (n_members - 1)
    Pyy = (anom(Y) @ anom(Y).T) / (n_members - 1) + obs_cov
    K = Pxy @ np.linalg.inv(Pyy)
    noise = np.random.default_rng(9).multivariate_normal(
        np.zeros(n_obs), obs_cov, size=n_members).T
    want = ens + K @ (((A @ np.ones(n_state)).reshape(-1, 1) + noise) - Y)

    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


def test_the_filter_reduces_the_spread_toward_the_truth() -> None:
    rng = np.random.default_rng(5)
    n_state, n_obs, n_members = 8, 8, 200
    A = np.eye(n_obs)
    operator = HydroGeophysObsOperator(
        petro_transform=lambda s: s, forward_operator=lambda m: A @ m
    )
    truth = np.full(n_state, 2.0)
    ens = rng.standard_normal((n_state, n_members)) * 3.0
    filt = EnsembleKalmanFilter(operator, np.eye(n_obs) * 0.01)
    post = filt.update(ens, truth, rng=np.random.default_rng(6))
    before = float(np.linalg.norm(ens.mean(axis=1) - truth))
    after = float(np.linalg.norm(post.mean(axis=1) - truth))
    assert after < before


def test_esmda_runs_every_step() -> None:
    rng = np.random.default_rng(7)
    A = np.eye(5)
    operator = HydroGeophysObsOperator(
        petro_transform=lambda s: s, forward_operator=lambda m: A @ m
    )
    out = ESMDA(operator, np.eye(5) * 0.1, n_steps=3).update(
        rng.standard_normal((5, 30)), np.ones(5), rng=np.random.default_rng(8)
    )
    assert out["history"].shape[0] == 3
    assert np.all(np.isfinite(out["ensemble"]))


# ---------------------------------------------------------------------------
# em1d_lci normal-equation solve
# ---------------------------------------------------------------------------
def test_the_lci_solve_is_exact_on_a_nonsingular_system() -> None:
    rng = np.random.default_rng(0)
    n = 40
    A = rng.standard_normal((n, n))
    gram = sp.csr_matrix(A @ A.T + n * np.eye(n))
    rhs = rng.standard_normal(n)
    np.testing.assert_allclose(gram @ _solve_normal_equations(gram, rhs), rhs, rtol=1e-9)


def test_a_singular_but_consistent_lci_system_still_solves() -> None:
    """MINRES picks this up after the sparse LU returns NaN."""
    gram = sp.csr_matrix(np.diag([2.0, 4.0, 0.0]))
    got = _solve_normal_equations(gram, np.array([2.0, 4.0, 0.0]))
    np.testing.assert_allclose((gram @ got)[:2], [2.0, 4.0], rtol=1e-8)


def test_a_singular_inconsistent_lci_system_raises_rather_than_returning_junk() -> None:
    """spsolve does not raise here, it warns and returns NaN.

    The old guard was a bare ``except`` around spsolve, so it never fired and the
    NaN travelled into the model update. MINRES then reports success on this
    system while the null-direction component runs away to 1e15, so the residual
    has to be checked too.
    """
    gram = sp.csr_matrix(np.diag([1.0, 1.0, 0.0]))
    with pytest.raises(np.linalg.LinAlgError, match="singular to working precision"):
        _solve_normal_equations(gram, np.ones(3))
