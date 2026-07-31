"""Deprecated compatibility import for the canonical linear solver."""

from __future__ import annotations

import warnings

from .linear_solvers import generalized_solver as _generalized_solver


def generalized_solver(*args, **kwargs):
    warnings.warn(
        "PyHydroGeophysX.solvers.solver.generalized_solver is deprecated; "
        "import it from solvers.linear_solvers. This compatibility shim is "
        "deprecated in 0.4.0 and will be removed in 0.5.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _generalized_solver(*args, **kwargs)


__all__ = ["generalized_solver"]
