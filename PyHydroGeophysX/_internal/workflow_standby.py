"""A workflow process started before its workflow.

A workflow process spent about a second before its workflow's first line ran:
starting the interpreter, then importing numpy, scipy, matplotlib, pyGIMLi,
numba and SimPEG afresh, every run. The desktop studio keeps one of these
processes started and waiting instead (``qt_apps.workers``), so those libraries
load while nobody is waiting for them, and a run hands its command line over on
standard input::

    {"module": "PyHydroGeophysX.workflows.cli", "argv": ["run", ...], "cwd": "..."}

The process then becomes that run's process and nothing else. It runs the
module exactly as ``python -m`` would, exits when the run does, and can be
cancelled by ending it; a crash ends that run only; everything the run
allocated goes back when it exits. One run per process is what keeps a run
from inheriting whatever the previous one left behind.

Only third-party libraries are loaded ahead. PyHydroGeophysX's own modules are
imported by the run, so a module edited while the studio is open is the one the
next run uses, as it was when every run started from nothing. End of input
before a run arrives - the studio closing, or retiring the process - ends it.
"""

from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
import warnings

#: Loaded ahead, in this order; any that is not installed is skipped. Neither
#: pandas nor anything that loads it: pandas loads pyarrow, and an ERT run keeps
#: pyarrow out of its process (``workflows.cli``), which a process already
#: holding it could not. Nor ``matplotlib.pyplot``'s backend: that is chosen at
#: the first figure a run draws, as it would be in a process of its own.
WARM_MODULES = (
    "numpy",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "scipy.interpolate",
    "scipy.optimize",
    "scipy.spatial",
    "scipy.ndimage",
    "matplotlib",
    "matplotlib.figure",
    "matplotlib.backends.backend_agg",
    "pygimli",
    "pygimli.meshtools",
    "pygimli.physics.ert",
    "pygimli.physics.traveltime",
    "numba",
    "discretize",
    "simpeg",
    "empymod",
    "pyvista",
)


def warm_up(modules=WARM_MODULES) -> list:
    """Import ``modules``, quietly; returns those that could not be.

    Warnings are the run's to see, from the imports it makes itself, and an
    import that fails here is left for the run to fail on, with its message.
    pyarrow is held out while this runs, so that nothing loaded ahead can load
    it on a run's behalf.
    """
    missing = []
    held_out = "pyarrow" not in sys.modules
    if held_out:
        sys.modules["pyarrow"] = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in modules:
                try:
                    importlib.import_module(name)
                except Exception:  # noqa: BLE001 - an optional library, or a broken one
                    missing.append(name)
    finally:
        if held_out and sys.modules.get("pyarrow", False) is None:
            del sys.modules["pyarrow"]
    return missing


def main() -> int:
    # ``python -m`` puts the directory it started in first on the path. Nothing
    # loaded ahead is to be found there, and a run started in its own directory
    # has that one first instead - so it is off the path until a run arrives.
    if sys.path and sys.path[0] in ("", os.getcwd()):
        sys.path.pop(0)
    warm_up()
    line = sys.stdin.readline()
    if not line.strip():
        return 0                      # retired, or the studio has gone
    job = json.loads(line)
    cwd = str(job["cwd"])
    os.chdir(cwd)
    sys.path.insert(0, cwd)
    sys.argv = [str(job["module"]), *[str(arg) for arg in job.get("argv", [])]]
    runpy.run_module(str(job["module"]), run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
