"""Show that the linearized step solver, not the inversion, decided the lambda sweep.

A regularization sweep on a sibling 4D ERT project moved the model by 1.757% of
its norm at lambda = 1, 10 and 100. A hundred-fold change in the regularization
weight producing no difference to four significant figures is not credible for
an ill-posed problem. The cause was the solver: the Gauss-Newton normal matrix
``H`` was handed to CGLS, which minimizes ``||A x - b||`` and so works on
``H^T H d = H^T (-g)``. That squares the condition number. A fixed iteration
budget then buys only the leading Krylov directions, and those belong to the
data term, because in a Gauss-Newton normal matrix ``J^T W_d^T W_d J`` outweighs
``lambda W_m^T W_m`` by orders of magnitude. Changing lambda barely moves them,
so the step barely moves, and the inversion looks insensitive to regularization
when it is nothing of the kind.

This script builds that situation from a synthetic Jacobian, solves the single
linearized step at two lambda values a hundred-fold apart with each solver, and
prints how much the step moved. It needs only numpy and scipy, and it is not
part of the test suite: it exists to produce the numbers.

What to look for. Under ``spd_cholesky`` the step moves by a large amount when
lambda moves, which is what an ill-posed problem should do. Under ``cgls`` at a
small budget it barely moves, and CGLS's relative residual shows why: it is not
solving the system at all. Raising the budget recovers some of the sensitivity,
slowly, which is the signature of a squared condition number rather than of a
tolerance that needs adjusting.

Usage::

    python scripts/lambda_sweep_solver_check.py
    python scripts/lambda_sweep_solver_check.py --data-scale 1e2
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from PyHydroGeophysX.solvers.linear_solvers import generalized_solver

METHODS = ("cgls", "spd_cholesky")


def _first_difference_operator(n: int) -> np.ndarray:
    """A first-difference smoothness operator, the usual W_m."""
    W = np.zeros((n - 1, n))
    rows = np.arange(n - 1)
    W[rows, rows] = -1.0
    W[rows, rows + 1] = 1.0
    return W


def _normal_system(n_model, n_data, lam, data_scale, seed=0):
    """Assemble ``H = J^T J + lambda W_m^T W_m`` and the gradient ``-g``.

    The Jacobian's singular values decay over three decades, so ``J^T J`` alone
    is near-singular and the regularization is what makes the step well posed.
    ``data_scale`` sets how far the data term outweighs the regularization,
    which is the variable that decides whether the symptom appears: a real
    Gauss-Newton normal matrix sits at a large value.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n_data, n_data)))
    V, _ = np.linalg.qr(rng.standard_normal((n_model, n_model)))
    k = min(n_data, n_model)
    J = U[:, :k] @ np.diag(data_scale * np.logspace(0.0, -3.0, k)) @ V[:, :k].T
    Wm = _first_difference_operator(n_model)
    residual = rng.standard_normal((n_data, 1))
    H = J.T @ J + lam * (Wm.T @ Wm)
    return np.ascontiguousarray(H), J.T @ residual


def _solve(H, rhs, method, maxiter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step = generalized_solver(
            H.copy(), rhs, method=method, maxiter=maxiter, tol=1e-8
        )
    step = np.asarray(step, dtype=float).reshape(-1, 1)
    residual = float(np.linalg.norm(H @ step - rhs) / np.linalg.norm(rhs))
    return step, residual


def run(n_model, n_data, lam, factor, data_scale, budgets):
    lambdas = (lam, lam * factor)
    print(
        "synthetic Gauss-Newton step: %d parameters, %d data, data-term scale %.0e"
        % (n_model, n_data, data_scale)
    )
    for value in lambdas:
        H, _ = _normal_system(n_model, n_data, value, data_scale)
        cond = float(np.linalg.cond(H))
        print(
            "  lambda = %-10g cond(H) = %.2e   what CGLS works with, cond(H)^2 = %.2e"
            % (value, cond, cond ** 2)
        )

    print(
        "\n%-9s %-14s %-14s %-14s %-14s"
        % ("budget", "cgls step chg", "chol step chg", "cgls resid lo", "cgls resid hi")
    )
    print("-" * 69)
    for maxiter in budgets:
        steps, resids = {}, {}
        for value in lambdas:
            H, rhs = _normal_system(n_model, n_data, value, data_scale)
            for method in METHODS:
                steps[(method, value)], resids[(method, value)] = _solve(
                    H, rhs, method, maxiter
                )

        def change(method):
            low, high = steps[(method, lambdas[0])], steps[(method, lambdas[1])]
            return 100.0 * float(np.linalg.norm(high - low) / np.linalg.norm(low))

        print(
            "%-9d %-14s %-14s %-14.3e %-14.3e"
            % (
                maxiter,
                "%.2f%%" % change("cgls"),
                "%.2f%%" % change("spd_cholesky"),
                resids[("cgls", lambdas[0])],
                resids[("cgls", lambdas[1])],
            )
        )

    print(
        "\n'step chg' is how far the model update moved when lambda changed by a\n"
        "factor of %g. 'cgls resid' is CGLS's relative residual ||H d + g|| / ||g||\n"
        "at each lambda; spd_cholesky's is at machine precision throughout and is\n"
        "not tabulated." % factor
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-model", type=int, default=400)
    parser.add_argument("--n-data", type=int, default=120)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--factor", type=float, default=100.0)
    parser.add_argument(
        "--data-scale",
        type=float,
        default=1e3,
        help="how far the data term outweighs the regularization (default 1e3)",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[50, 100, 300, 1000],
        help="CGLS iteration budgets to tabulate (srt_time_lapse defaults to 300)",
    )
    args = parser.parse_args()
    run(
        args.n_model,
        args.n_data,
        args.lambda_value,
        args.factor,
        args.data_scale,
        args.budgets,
    )


if __name__ == "__main__":
    main()
