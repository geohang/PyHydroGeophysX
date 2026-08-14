import builtins
import subprocess
import sys

import numpy as np


def test_cpu_solver_does_not_import_cupy(monkeypatch) -> None:
    from PyHydroGeophysX.solvers import linear_solvers

    real_import = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy.") or name.startswith("cupyx"):
            attempted.append(name)
            raise AssertionError("CPU solver attempted to import CuPy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    matrix = np.eye(2)
    rhs = np.array([2.0, 3.0])

    result = linear_solvers.generalized_solver(
        matrix,
        rhs,
        method="cgls",
        maxiter=5,
        use_gpu=False,
    )

    np.testing.assert_allclose(np.asarray(result).ravel(), rhs)
    assert attempted == []


def test_importing_cpu_ert_stack_does_not_eagerly_load_cupy() -> None:
    # Use a clean interpreter so this remains valid even when another GPU test
    # imported CuPy earlier in the pytest session.
    code = (
        "import sys; "
        "import PyHydroGeophysX.inversion.ert_inversion; "
        "print('cupy' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.stdout.strip() == "False"
