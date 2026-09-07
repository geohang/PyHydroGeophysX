"""Compare accuracy and CPU timings against a trusted local Git ref.

Run from the repository root: python scripts/benchmark_monte_carlo.py --ref HEAD
The baseline code is executed in memory; the working tree is never changed.
"""

import argparse
import subprocess
import time
import tracemalloc
import types
from unittest.mock import patch

import numpy as np

import PyHydroGeophysX.petrophysics.resistivity_models as models
from PyHydroGeophysX.petrophysics.monte_carlo import run_petrophysics_monte_carlo


def load_baseline(ref, filename, name):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:PyHydroGeophysX/petrophysics/{filename}.py"], encoding="utf-8")
    module = types.ModuleType(name)
    exec(compile(source, filename + ".py", "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    legacy = load_baseline(args.ref, "monte_carlo", "PyHydroGeophysX.petrophysics.legacy_mc")
    old_models = load_baseline(args.ref, "resistivity_models", "legacy_models")
    replacements = {name: getattr(old_models, name) for name in (
        "resistivity_to_saturation", "resistivity_to_saturation2", "resistivity_to_porosity")}

    for surface in (0., .0001):
        rho = np.random.default_rng(4).uniform(200, 900, (12, 4))
        markers = np.resize([1, 2], 12)
        layers = [{"marker": 1, "rho_sat": {"mean": 80, "std": 4}, "sigma_sur": surface},
                  {"marker": 2, "m": {"mean": 1.5, "std": .05}, "sigma_sur": surface}]
        kwargs = dict(products=["water_content", "porosity"], n_realizations=7,
                      seed=22, return_realizations=True)
        with patch.multiple(models, **replacements):
            previous = legacy.run_petrophysics_monte_carlo(rho, markers, layers, **kwargs)
        current = run_petrophysics_monte_carlo(rho, markers, layers, cell_chunk_size=3, **kwargs)
        errors = []
        for key in ("water_content_all", "porosity_all", "saturation_all"):
            np.testing.assert_allclose(previous[key], current[key], rtol=1e-13, atol=1e-15)
            errors.append(float(np.max(np.abs(previous[key] - current[key]))))
        print(f"surface conductivity={surface}: max sample difference={max(errors):.3g}")

    rho = np.random.default_rng(1).uniform(100, 1000, (8000, 8))
    markers = np.ones(8000, dtype=int)
    layers = [{"marker": 1, "rho_sat": {"mean": 100., "std": 5.}}]
    timings = {}
    for label in ("baseline", "current"):
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            if label == "baseline":
                with patch.multiple(models, **replacements):
                    legacy.run_petrophysics_monte_carlo(rho, markers, layers, n_realizations=30, seed=7)
            else:
                run_petrophysics_monte_carlo(rho, markers, layers, n_realizations=30, seed=7)
            durations.append(time.perf_counter() - start)
        timings[label] = float(np.median(durations))
    print("8000 cells x 8 times x 30 draws, seconds (median of 3):", timings)
    print("speedup:", timings["baseline"] / timings["current"])
    for label in ("baseline", "current"):
        tracemalloc.start()
        if label == "baseline":
            with patch.multiple(models, **replacements):
                legacy.run_petrophysics_monte_carlo(rho, markers, layers, n_realizations=30, seed=7)
        else:
            run_petrophysics_monte_carlo(rho, markers, layers, n_realizations=30, seed=7)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"{label} traced peak allocations: {peak / 1024**2:.2f} MiB (not process RSS)")


if __name__ == "__main__":
    main()
