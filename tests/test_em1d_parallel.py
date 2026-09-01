"""Parallel per-sounding solving, and the analytic Jacobian it shares.

The forward operator here is a cheap analytic stand-in rather than SimPEG: what
is under test is that running the soundings on threads changes nothing about the
arithmetic, and that the Jacobian handed to the optimizer is the derivative of
the residual it is paired with. Neither question needs a physical kernel, and a
synthetic one keeps the test fast and free of an optional dependency.
"""

import numpy as np
import pytest

from PyHydroGeophysX.inversion.em1d_lci import (
    SoundingBlock,
    resolve_worker_count,
    solve_lci,
)


def _synthetic_block(seed: int, n_layers: int = 6, n_data: int = 12):
    """A differentiable stand-in for one sounding, with an exact Jacobian.

    ``pred_i = sum_k A_ik sqrt(sigma_k)`` is nonlinear in the model, so the
    solver takes real nonlinear optimisation steps, and its derivative is one line.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.5, 1.5, size=(n_data, n_layers))
    truth = 1.0 / np.geomspace(30.0, 300.0, n_layers)

    def forward(sigma):
        return A @ np.sqrt(np.asarray(sigma, dtype=float))

    def jacobian(sigma):
        return A * (0.5 / np.sqrt(np.asarray(sigma, dtype=float)))[None, :]

    dobs = forward(truth)
    return SoundingBlock(forward=forward, jacobian=jacobian, dobs=dobs,
                         uncertainty=0.02 * np.abs(dobs) + 1e-12,
                         position=float(seed * 10), line=0)


def test_resolve_worker_count_rules():
    assert resolve_worker_count(1, 0) == 1          # nothing to spread
    assert resolve_worker_count(50, 3) == 3         # explicit wins
    assert resolve_worker_count(2, 99) == 2         # never more than soundings
    auto = resolve_worker_count(64, 0)
    assert 1 <= auto <= 64


@pytest.mark.parametrize("workers", [2, 3, 8, 0])
def test_parallel_solve_reproduces_serial_bitwise(workers):
    """Threads must not perturb the answer, and must not do so repeatedly.

    A race shows up as a result that differs sometimes, so one comparison
    proves little; the solve is repeated and every run is compared bit for bit
    against the serial one.
    """
    blocks = [_synthetic_block(seed) for seed in range(1, 13)]
    serial = solve_lci(blocks, 6, lateral_smoothness=0.5, max_iterations=8,
                       parallel_workers=1, verbose=False)
    for _ in range(3):
        parallel = solve_lci(blocks, 6, lateral_smoothness=0.5, max_iterations=8,
                             parallel_workers=workers, verbose=False)
        assert np.array_equal(parallel.models, serial.models)
        assert parallel.chi2 == serial.chi2
        assert parallel.iterations == serial.iterations
        assert parallel.stop_reason == serial.stop_reason


def test_single_sounding_takes_the_serial_path():
    """One sounding has nothing to spread, and must still solve."""
    out = solve_lci([_synthetic_block(1)], 6, max_iterations=4,
                    parallel_workers=8, verbose=False)
    assert out.models.shape == (1, 6)
    assert np.all(np.isfinite(out.models))


def test_analytic_jacobian_matches_finite_difference(monkeypatch):
    """The Jacobian handed to the optimizer is the derivative of its residual.

    Checked over the whole residual, regularization rows included: those blocks
    are constant, so an error in them would never show up as a wrong forward
    and would only bend where the fit lands. The pair is taken from the
    optimizer call itself, so the test cannot pass by re-deriving either side.
    """
    import scipy.optimize

    from PyHydroGeophysX.inversion.em1d import _occam_1d

    block = _synthetic_block(7, n_layers=5, n_data=10)
    inv = {"n_layers": 5, "smoothness": 0.4, "max_iterations": 3,
           "lateral_reference": np.full(5, 80.0), "lateral_weight": 0.25}
    real_ls = scipy.optimize.least_squares
    seen = {}

    def recording(fun, x0, **kwargs):
        seen.update(fun=fun, jac=kwargs.get("jac"),
                    x0=np.asarray(x0, dtype=float))
        return real_ls(fun, x0, **kwargs)

    monkeypatch.setattr(scipy.optimize, "least_squares", recording)
    _occam_1d(block.forward, block.dobs, block.uncertainty, 5, inv,
              lambda _m: None, block.jacobian)

    fun, jac = seen["fun"], seen["jac"]
    assert jac is not None, "the analytic Jacobian was not passed to the optimizer"
    x = seen["x0"] + 0.05 * np.arange(5)
    analytic = np.asarray(jac(x), dtype=float)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for k in range(x.size):
        up, down = x.copy(), x.copy()
        up[k] += step
        down[k] -= step
        numeric[:, k] = (np.asarray(fun(up)) - np.asarray(fun(down))) / (2 * step)
    scale = max(np.abs(analytic).max(), np.abs(numeric).max(), 1e-30)
    assert np.max(np.abs(analytic - numeric)) / scale < 1e-6


def test_analytic_and_numerical_jacobians_reach_the_same_model():
    """Dropping the analytic Jacobian must change the cost, not the answer."""
    from PyHydroGeophysX.inversion.em1d import _occam_1d

    block = _synthetic_block(3, n_layers=6, n_data=14)
    inv = {"n_layers": 6, "smoothness": 0.3, "max_iterations": 40,
           "starting_resistivity": 50.0}
    args = (block.forward, block.dobs, block.uncertainty, 6, inv, lambda _m: None)
    numeric_res, numeric_chi2, _, _ = _occam_1d(*args)
    analytic_res, analytic_chi2, _, _ = _occam_1d(*args, block.jacobian)
    assert np.allclose(analytic_res, numeric_res, rtol=1e-3)
    assert analytic_chi2 == pytest.approx(numeric_chi2, rel=1e-3)
