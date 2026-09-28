"""Deprecated compatibility import for the canonical linear solver."""

from __future__ import annotations

from typing import Any

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path

from . import linear_solvers as _linear_solvers

#: The iteration cap this function had in 0.3.0; linear_solvers' is 200.
_V030_MAXITER = 2000


def generalized_solver(*args, **kwargs):
    _warn_legacy_path(
        "PyHydroGeophysX.solvers.solver.generalized_solver",
        "PyHydroGeophysX.solvers.linear_solvers.generalized_solver, with maxiter=2000 "
        "for this path's iteration cap (its own default is 200)")
    # maxiter is the fifth parameter of both signatures.
    if len(args) < 5 and "maxiter" not in kwargs:
        kwargs["maxiter"] = _V030_MAXITER
    return _linear_solvers.generalized_solver(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name != "gpu_available":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _warn_legacy_path(
        f"{__name__}.gpu_available",
        "solvers.linear_solvers.generalized_solver(..., use_gpu=True), which runs on "
        "the CPU when CuPy is missing")
    # 0.3.0 set this by importing CuPy as this module loaded. CuPy is now loaded
    # only on request, and asking for this name is one.
    return _linear_solvers._load_gpu_backend()


__all__ = ["generalized_solver"]
