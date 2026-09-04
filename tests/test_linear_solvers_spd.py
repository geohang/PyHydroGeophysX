"""Tests for the SPD solvers in ``PyHydroGeophysX.solvers.linear_solvers``.

Three inversion call sites assemble the Gauss-Newton normal matrix
``H = J^T W_d^T W_d J + lambda W_m^T W_m + ...`` and hand it to
``generalized_solver``. Every least-squares method in that module applies ``A``
and ``A.T``, so on such a matrix it works on ``H^T H d = H^T (-g)``: the
condition number is squared, and because ``H`` is symmetric each iteration
spends two matrix-vector products where one would do. ``spd_cholesky`` and
``spd_cg`` solve ``H d = -g`` directly instead.

These tests pin four things: that the SPD methods are correct and that
``spd_cholesky`` is exact where a fixed iteration budget of CGLS is not; that
the ridge rescues a borderline matrix without being charged to a well-behaved
one; that the advisory fires exactly on the case it is meant for; and that none
of it disturbs ``cgls`` or imports CuPy.
"""

from __future__ import annotations

import builtins
import inspect
import warnings

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from PyHydroGeophysX.solvers import linear_solvers
from PyHydroGeophysX.solvers.linear_solvers import generalized_solver

SPD_METHODS = ("spd_cholesky", "spd_cg")


def _spd(n: int, cond: float, seed: int = 0) -> np.ndarray:
    """A dense SPD matrix with exactly the requested 2-norm condition number.

    ``Q diag(w) Q.T`` with ``Q`` orthogonal and ``w`` log-spaced from 1 to
    ``cond``. Building the spectrum by construction is what makes
    "ill-conditioned" a controlled variable: a random Gram matrix has whatever
    condition number it happens to have. The final symmetrization strips the
    O(eps) asymmetry the product leaves behind, so the probe tests start from a
    matrix that is symmetric to the last bit.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    M = (Q * np.logspace(0.0, np.log10(cond), n)) @ Q.T
    return np.ascontiguousarray((M + M.T) / 2.0)


def _system(n: int, cond: float, seed: int = 0):
    """An SPD matrix, a known solution, and the matching right-hand side."""
    A = _spd(n, cond, seed=seed)
    x_true = np.random.default_rng(seed + 1).standard_normal((n, 1))
    return A, x_true, A @ x_true


def _rel_error(got, want) -> float:
    got = np.asarray(got, dtype=float).reshape(-1, 1)
    want = np.asarray(want, dtype=float).reshape(-1, 1)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


@pytest.fixture()
def fresh_advisory(monkeypatch) -> None:
    """Re-arm the once-per-process advisory, and restore it afterwards."""
    monkeypatch.setattr(linear_solvers, "_SQUARE_LS_WARNED", False)


@pytest.fixture()
def quiet_advisory(monkeypatch) -> None:
    """Silence the advisory for tests that are not about it."""
    monkeypatch.setattr(linear_solvers, "_SQUARE_LS_WARNED", True)


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cond", [1e2, 1e6, 1e10])
def test_spd_cholesky_matches_the_dense_reference(cond, quiet_advisory) -> None:
    """Compare against LAPACK's LU on the same system.

    The tolerance scales with the condition number because both routines are
    backward stable, not exact: two stable factorizations of a matrix with
    ``cond(A) = 1e10`` agree only to about ``cond(A) * eps``, so a fixed
    tolerance would be asserting something neither method promises.
    """
    A, _, b = _system(40, cond)
    got = generalized_solver(A.copy(), b, method="spd_cholesky")
    rtol = max(1e-9, 50.0 * cond * float(np.finfo(float).eps))
    np.testing.assert_allclose(got, np.linalg.solve(A, b), rtol=rtol, atol=1e-12)


def test_spd_cg_matches_the_dense_reference(quiet_advisory) -> None:
    n = 40
    A, _, b = _system(n, 1e3)
    got = generalized_solver(A.copy(), b, method="spd_cg", maxiter=20 * n, tol=1e-13)
    np.testing.assert_allclose(got, np.linalg.solve(A, b), rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("cond", [1e4, 1e8, 1e12])
def test_spd_cholesky_beats_cgls_and_the_gap_widens(cond, quiet_advisory) -> None:
    """The finding this module exists for, stated as an assertion.

    CGLS on a normal matrix converges at a rate set by ``cond(A)`` rather than
    ``sqrt(cond(A))``, so a fixed budget lands nowhere near the solution and
    falls further behind as the matrix gets worse.
    """
    A, x_true, b = _system(120, cond)
    exact = generalized_solver(A.copy(), b, method="spd_cholesky")
    iterative = generalized_solver(A.copy(), b, method="cgls", maxiter=100)

    assert _rel_error(exact, x_true) < 1e-4
    assert _rel_error(iterative, x_true) > 0.1
    assert _rel_error(exact, x_true) < 1e-3 * _rel_error(iterative, x_true)


def test_spd_cg_beats_cgls_at_an_equal_matrix_vector_budget(quiet_advisory) -> None:
    """CGLS spends two matrix-vector products per iteration on a symmetric A.

    Compare on the residual rather than the solution error: at a budget this
    small neither method is near the solution, so the residual is the honest
    measure of progress.
    """
    n, budget = 120, 60
    A, _, b = _system(n, 1e5)
    cg = generalized_solver(A.copy(), b, method="spd_cg", maxiter=budget, tol=1e-14)
    cgls = generalized_solver(A.copy(), b, method="cgls", maxiter=budget // 2)

    def resid(step):
        return float(np.linalg.norm(A @ np.asarray(step).reshape(-1, 1) - b))

    assert resid(cg) < resid(cgls)


@pytest.mark.parametrize("method", ["cgls", "spd_cholesky", "spd_cg"])
def test_the_solvers_return_a_float64_column(method, quiet_advisory) -> None:
    """Pin the SPD return contract against the incumbent rather than a guess."""
    n = 25
    A, _, b = _system(n, 1e3)
    got = generalized_solver(A.copy(), b, method=method, maxiter=50)
    assert got.shape == (n, 1)
    assert got.dtype == np.float64


# ---------------------------------------------------------------------------
# the ridge
# ---------------------------------------------------------------------------
def test_a_semidefinite_matrix_is_rescued_by_the_ridge(quiet_advisory) -> None:
    """``diag(1, 1, 0)`` is symmetric, positive semidefinite, exactly singular.

    LAPACK's ``potrf`` refuses it on every platform, so this is deterministic
    rather than dependent on the BLAS in use.
    """
    A = np.diag([1.0, 1.0, 0.0])
    b = np.array([1.0, 2.0, 0.0])

    with pytest.raises(np.linalg.LinAlgError):
        scipy.linalg.cho_factor(A.copy())

    with pytest.warns(RuntimeWarning, match="Retrying with a ridge"):
        got = generalized_solver(A.copy(), b, method="spd_cholesky")

    np.testing.assert_allclose(got.ravel()[:2], [1.0, 2.0], rtol=1e-9)
    assert abs(float(got.ravel()[2])) < 1e-6


def test_the_ridge_is_not_charged_to_a_matrix_that_factors(quiet_advisory) -> None:
    """The ridge is a rescue, not a default.

    Applying it unconditionally is measurably wrong: on a matrix with a
    condition number of 1e12 the ridge is a few percent of the smallest
    eigenvalue and costs three to four orders of magnitude of accuracy. Assert
    the result is bit-comparable to a plain unridged Cholesky.
    """
    A, _, b = _system(60, 1e12)
    got = generalized_solver(A.copy(), b, method="spd_cholesky")
    want = scipy.linalg.cho_solve(scipy.linalg.cho_factor(A.copy()), b)
    np.testing.assert_allclose(got.ravel(), want.ravel(), rtol=1e-12, atol=0.0)


def test_an_indefinite_matrix_falls_back_to_an_ldlt_solve(quiet_advisory) -> None:
    A = np.diag([1.0, -2.0])
    b = np.array([1.0, 2.0])
    with pytest.warns(RuntimeWarning):
        got = generalized_solver(A.copy(), b, method="spd_cholesky")
    np.testing.assert_allclose(got.ravel(), [1.0, -1.0], rtol=1e-12)


def test_a_float32_matrix_solves_and_its_ridge_is_representable(quiet_advisory) -> None:
    """``time_lapse.py`` uses float32 when ``save_memory=True``.

    A ridge fixed at 1e-12 would sit three orders below float32 eps (1.19e-7)
    and do nothing at all, so it is floored at the working dtype's resolution.
    """
    A = _spd(30, 1e3).astype(np.float32)
    assert linear_solvers._spd_ridge(A) > float(np.finfo(np.float32).eps)

    b = np.ones((30, 1))
    got = generalized_solver(A.copy(), b, method="spd_cholesky")
    np.testing.assert_allclose(
        got.ravel(), np.linalg.solve(A.astype(float), b).ravel(), rtol=1e-3
    )


# ---------------------------------------------------------------------------
# overwrite_a
# ---------------------------------------------------------------------------
def test_overwrite_a_consumes_a_c_ordered_buffer(quiet_advisory) -> None:
    """If SciPy ever stops honouring ``overwrite_a`` on a transposed view, the
    saving evaporates silently. Assert it is actually happening."""
    A, _, b = _system(30, 1e4)
    reference = A.copy()
    assert A.flags.c_contiguous

    got = generalized_solver(A, b, method="spd_cholesky", overwrite_a=True)

    np.testing.assert_allclose(got, np.linalg.solve(reference, b), rtol=1e-9)
    assert not np.array_equal(A, reference)


def test_overwrite_a_defaults_to_leaving_the_matrix_alone(quiet_advisory) -> None:
    A, _, b = _system(30, 1e4)
    reference = A.copy()
    generalized_solver(A, b, method="spd_cholesky")
    np.testing.assert_array_equal(A, reference)


def test_overwrite_a_on_a_singular_matrix_raises_rather_than_ridging(
    quiet_advisory,
) -> None:
    """A partial ``potrf`` has already destroyed the buffer, so neither the ridge
    retry nor the LDL^T fallback is reachable. The message must say so."""
    A = np.diag([1.0, 1.0, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="overwrite_a=False"):
        generalized_solver(A, np.ones(3), method="spd_cholesky", overwrite_a=True)


# ---------------------------------------------------------------------------
# input shapes the call sites actually produce
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", SPD_METHODS)
def test_an_np_matrix_input_is_handled(method, quiet_advisory) -> None:
    """``time_lapse.py`` assembles H through a scipy spmatrix product, which
    numpy resolves to ``np.matrix``. ``np.matrix.dot`` on a 1-D vector returns
    shape ``(1, n)``, which would quietly corrupt the CG operator."""
    dense = _spd(20, 1e3)
    b = np.ones((20, 1))
    got = generalized_solver(
        np.matrix(dense.copy()), b, method=method, maxiter=400, tol=1e-13
    )
    assert got.shape == (20, 1)
    np.testing.assert_allclose(got, np.linalg.solve(dense, b), rtol=1e-7, atol=1e-10)


def test_a_sparse_spd_matrix_solves_through_superlu(quiet_advisory) -> None:
    """SciPy has no sparse Cholesky, so the sparse path is SuperLU. Densifying
    instead would defeat the ``save_memory=True`` setting that produces it."""
    dense = _spd(30, 1e4)
    b = np.ones(30)
    got = generalized_solver(
        scipy.sparse.csr_matrix(dense), b, method="spd_cholesky", overwrite_a=True
    )
    assert got.shape == (30, 1)
    np.testing.assert_allclose(got.ravel(), np.linalg.solve(dense, b), rtol=1e-9)


@pytest.mark.parametrize("method", SPD_METHODS)
def test_damp_shifts_the_diagonal(method, quiet_advisory) -> None:
    A, _, _ = _system(25, 1e3)
    b = np.ones((25, 1))
    got = generalized_solver(
        A.copy(), b, method=method, damp=0.5, maxiter=400, tol=1e-13
    )
    want = np.linalg.solve(A + 0.5 * np.eye(25), b)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-10)


# ---------------------------------------------------------------------------
# failure reporting
# ---------------------------------------------------------------------------
def test_spd_cg_warns_and_returns_the_iterate_on_non_convergence(
    quiet_advisory,
) -> None:
    """``_cgls`` reports nothing at all when it exhausts maxiter. Raising here
    would turn a slow solve into a failed inversion, so warn and return."""
    A, _, b = _system(60, 1e10)
    with pytest.warns(RuntimeWarning, match="did not reach"):
        got = generalized_solver(A.copy(), b, method="spd_cg", maxiter=3, tol=1e-14)
    assert got.shape == (60, 1)
    assert np.all(np.isfinite(got))
    assert float(np.linalg.norm(A @ got - b)) < float(np.linalg.norm(b))


@pytest.mark.parametrize("method", SPD_METHODS)
def test_rectangular_input_is_rejected_by_both_spd_methods(method) -> None:
    A = np.random.default_rng(0).standard_normal((30, 12))
    with pytest.raises(ValueError, match="square"):
        generalized_solver(A, np.ones(30), method=method)


def test_unknown_method_names_the_spd_methods() -> None:
    with pytest.raises(ValueError, match="spd_cholesky"):
        generalized_solver(np.eye(3), np.ones(3), method="not_a_method")


# ---------------------------------------------------------------------------
# the advisory
# ---------------------------------------------------------------------------
def test_a_least_squares_method_on_a_symmetric_matrix_warns_once(
    fresh_advisory,
) -> None:
    A, _, b = _system(30, 1e4)
    with pytest.warns(RuntimeWarning, match="square and symmetric"):
        generalized_solver(A.copy(), b, method="cgls", maxiter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generalized_solver(A.copy(), b, method="cgls", maxiter=3)


def test_a_rectangular_system_never_warns(fresh_advisory) -> None:
    """The stacked systems in ``ert_inversion.py`` and ``joint_ert_srt.py`` are
    using the right solver family and must stay silent, at the cost of one
    integer comparison per call."""
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generalized_solver(
            rng.standard_normal((40, 15)), rng.standard_normal((40, 1)),
            method="cgls", maxiter=3,
        )


def test_a_square_nonsymmetric_system_never_warns(fresh_advisory) -> None:
    """The probe has to discriminate, not fire on everything square."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 30))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generalized_solver(A, rng.standard_normal((30, 1)), method="cgls", maxiter=3)


def test_the_probe_leaves_a_and_the_global_random_state_alone(fresh_advisory) -> None:
    """The two ways the advisory could have changed ``cgls`` behaviour."""
    A, _, b = _system(30, 1e4)
    reference = A.copy()

    np.random.seed(1234)
    control = [float(np.random.random()) for _ in range(2)]

    np.random.seed(1234)
    before = float(np.random.random())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generalized_solver(A, b, method="cgls", maxiter=3)
    after = float(np.random.random())

    np.testing.assert_array_equal(A, reference)
    assert [before, after] == control


# ---------------------------------------------------------------------------
# GPU hygiene and scipy compatibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", SPD_METHODS)
def test_the_spd_methods_never_import_cupy(method, monkeypatch, quiet_advisory) -> None:
    """The SPD methods return before the GPU backend selection. Importing CuPy
    for a CPU solve is what the lazy loader in this module exists to prevent."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy.") or name.startswith("cupyx"):
            raise AssertionError("an SPD solve attempted to import CuPy")
        return real_import(name, *args, **kwargs)

    A, _, b = _system(20, 1e3)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = generalized_solver(
            A.copy(), b, method=method, maxiter=400, tol=1e-13, use_gpu=True
        )
    assert got.shape == (20, 1)


@pytest.mark.parametrize("method", SPD_METHODS)
def test_use_gpu_is_ignored_with_a_warning(method, quiet_advisory) -> None:
    A, _, b = _system(20, 1e3)
    with pytest.warns(RuntimeWarning, match="CPU-only"):
        generalized_solver(A.copy(), b, method=method, use_gpu=True, maxiter=400)


def test_the_cg_tolerance_keyword_matches_this_scipy() -> None:
    """SciPy renamed ``tol`` to ``rtol`` in 1.12 and removed ``tol`` in 1.14;
    ``pyproject.toml`` pins ``scipy>=1.8,<3.0``, which straddles both."""
    keyword = linear_solvers._cg_rtol_keyword()
    assert keyword in ("rtol", "tol")
    assert keyword in inspect.signature(scipy.sparse.linalg.cg).parameters
