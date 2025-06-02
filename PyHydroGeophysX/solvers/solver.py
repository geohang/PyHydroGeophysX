"""
This module serves as a re-exporter for solver implementations from `linear_solvers.py`.

It previously contained duplicated solver implementations, which have now been
consolidated into `PyHydroGeophysX.solvers.linear_solvers`.

**It is strongly recommended to import solver functions directly from `linear_solvers.py`
to ensure clarity and avoid issues if this shim module is deprecated or removed in the future.**

For example, instead of:
`from PyHydroGeophysX.solvers.solver import generalized_solver`

Use:
`from PyHydroGeophysX.solvers.linear_solvers import generalized_solver`

This file is maintained temporarily for backward compatibility but should be considered for removal.
The detailed inline comments explaining the solver algorithms and their parameters are located
in `PyHydroGeophysX/solvers/linear_solvers.py`.
"""

# Attempt to re-export the main solver functions and potentially internal helper methods
# from `linear_solvers.py`. This is primarily for backward compatibility if other parts
# of the PyHydroGeophysX library (or external user code) still import from this `solver.py` module.
try:
    # Import specific solver functions that might have been exposed by this module previously.
    from .linear_solvers import (
        generalized_solver,
        _matrix_multiply,  # Helper, might not have been intended for external use
        _lsqr,             # Internal LSQR implementation
        _rrlsqr,           # Internal RRLSQR implementation
        _cgls,             # Internal CGLS implementation
        _rrls              # Internal RRLS implementation
        # SUGGESTION: Only re-export public API elements like `generalized_solver` and solver classes
        # (`CGLSSolver`, `LSQRSolver`, etc.) rather than internal helper functions (`_matrix_multiply`, `_lsqr`).
        # For this task, re-exporting as originally found to maintain structure.
    )
except ImportError:
    # This ImportError might occur if the file structure changes, during certain test setups,
    # or if `linear_solvers.py` is not found in the same package directory.
    print("Warning: Could not re-export solver components from PyHydroGeophysX.solvers.linear_solvers "
          "within PyHydroGeophysX.solvers.solver.py. "
          "This may indicate an issue with the package structure or imports. "
          "Please update imports to point directly to `linear_solvers.py`.")
    # Allow the module to still load without crashing if re-export fails, but functionality will be missing.
    pass

# SUGGESTION: Consider adding __all__ to explicitly define what gets imported when
# `from PyHydroGeophysX.solvers.solver import *` is used, though explicit imports are better.
# Example:
# __all__ = ['generalized_solver'] # if only generalized_solver is meant to be public from here.

# Final Note: The existence of this file suggests a refactoring has occurred.
# The best long-term solution is to update all parts of the PyHydroGeophysX library
# and any dependent user code to import directly from `linear_solvers.py`,
# after which this `solver.py` file can be safely removed to avoid confusion and redundancy.
