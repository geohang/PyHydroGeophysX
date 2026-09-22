"""Bounded, isolated ADTLERT installation checks; no Qt or numerical imports at startup."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

STAGES = ("runtime", "single", "timelapse")


def check_stage(stage):
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() in {"1", "TRUE", "YES", "ON"}:
        raise RuntimeError("Unset KMP_DUPLICATE_LIB_OK before validating GPU compatibility.")
    if stage == "runtime":
        from PyHydroGeophysX.qt_apps.adtlert_probe import probe_adtlert_runtime
        return probe_adtlert_runtime()
    import numpy as np
    import pygimli.meshtools as mt
    from pygimli.physics import ert

    # Heterogeneous synthetic observations ensure an actual update/line search,
    # rather than an immediate stop on a perfectly fitting homogeneous model.
    scheme = ert.createData(elecs=np.linspace(0., 23., 12), schemeName="dd")
    world = mt.createWorld(start=[-10., 0.], end=[33., -15.], worldMarker=True)
    block = mt.createRectangle(start=[9., -3.], end=[15., -7.], marker=2)
    simulation_mesh = mt.createMesh(mt.mergePLC([world, block]), quality=32, area=1.5)
    with tempfile.TemporaryDirectory(prefix="phgx_gpu_check_") as temporary:
        datasets, paths = [], []
        for i, resistance in enumerate((45., 70.)):
            data = ert.simulate(simulation_mesh, res=[[1, 150.], [2, resistance]],
                                scheme=scheme, noiseLevel=0., verbose=False)
            data["err"] = np.full(data.size(), .03)
            path = Path(temporary) / f"step_{i}.dat"
            data.save(str(path))
            datasets.append(data)
            paths.append(path.name)
        mesh = mt.createMesh(mt.createParaMeshPLC(datasets[0].sensorPositions(),
                            paraDepth=10., paraDX=.6, boundary=1.), quality=32)
        if stage == "single":
            from .ert_inversion import _ADTLertEngine
            engine = _ADTLertEngine(datasets[0], mesh)
            try:
                result = engine.fit(lam=10., max_iterations=2, plateau_tolerance=0., target_chi2=0.)
                model, response = result.model, result.response
                history, backend = result.convergence, result.metrics["backend"]
                solver = result.metrics["linearized_solver"]
            finally:
                close = getattr(engine._forward, "close", None)
                if callable(close):
                    close()
        elif stage == "timelapse":
            from .windowed import WindowedTimeLapseERTInversion
            inversion = WindowedTimeLapseERTInversion(
                data_dir=temporary, ert_files=paths, measurement_times=[0., 1.],
                window_size=2, mesh=mesh, engine="adtlert", max_iterations=2,
                lambda_val=10., alpha=5., convergence_tolerance=0.,
            )
            if inversion.engine != "adtlert":
                raise RuntimeError("GPU time-lapse selected a CPU fallback; ADTLERT is not verified.")
            result = inversion.run()
            model, response = result.final_models, result.predicted_data
            history, backend = result.iteration_chi2, result.meta["backend"]
            solver = result.meta["linearized_solver"]
        else:
            raise ValueError(f"Unknown stage: {stage}")
        if backend != "adtlert" or solver != "gpu_cgls":
            raise RuntimeError(f"Unexpected backend/solver: {backend}/{solver}")
        if not history or not all(np.isfinite(value) for value in history):
            raise RuntimeError("No finite inversion iteration history produced.")
        if not np.all(np.isfinite(model)) or not np.all(np.isfinite(response)):
            raise RuntimeError("Inversion returned non-finite model or predictions.")
        return {"backend": backend, "linearized_solver": solver, "iterations": len(history),
                "chi2": float(history[-1]), "iteration_chi2": list(map(float, history)),
                "test": "two-iteration synthetic inversion with default line search"}


def stage_payload(stage):
    try:
        return {"ok": True, "result": check_stage(stage)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_stage(stage, timeout=120):
    command = [sys.executable, "-m", "PyHydroGeophysX.inversion.adtlert_diagnostics", "--stage", stage]
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", env=environment,
                                   timeout=timeout)
        entry = {"command": command, "exit_code": completed.returncode,
                 "stdout": completed.stdout, "stderr": completed.stderr, "ok": False}
        for line in reversed(completed.stdout.splitlines()):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and "ok" in payload:
                entry.update(payload)
                break
        if completed.returncode != 0:
            entry["ok"] = False
            entry.setdefault("error", f"Stage process exited with code {completed.returncode}.")
        if not entry["ok"]:
            entry.setdefault("error", "No successful diagnostic result received.")
        return entry
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        return {"ok": False, "command": command, "exit_code": None,
                "error": f"Timed out after {timeout} seconds", "stdout": decode(exc.stdout),
                "stderr": decode(exc.stderr)}
    except OSError as exc:
        return {"ok": False, "command": command, "exit_code": None,
                "error": f"Could not start stage: {exc}", "stdout": "", "stderr": ""}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="Run one stage (used by Studio)")
    parser.add_argument("--report", type=Path, help="Save versions, paths, commands and full logs as JSON")
    parser.add_argument("--timeout", type=float, default=120., help="Seconds allowed per isolated stage")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.stage:
        payload = stage_payload(args.stage)
        print(json.dumps(payload, ensure_ascii=True), flush=True)
        return 0 if payload["ok"] else 1
    import PyHydroGeophysX
    versions = {}
    for name in ("PyHydroGeophysX", "numpy", "scipy", "pygimli", "torch", "cupy-cuda12x", "nvmath-python", "adtlert"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"python": sys.executable, "python_version": sys.version,
              "source": PyHydroGeophysX.__file__, "platform": sys.platform,
              "versions": versions, "stages": {}}
    for stage in STAGES:
        report["stages"][stage] = run_stage(stage, args.timeout)
        print(f"{stage}: {'PASS' if report['stages'][stage]['ok'] else 'FAIL'}", flush=True)
    report["ok"] = all(entry["ok"] for entry in report["stages"].values())
    if args.report:
        from PyHydroGeophysX.data_processing.table_io import write_json
        write_json(args.report, report)
        print(f"Report: {args.report.resolve()}")
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
