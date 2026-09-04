"""
Seismic Refraction Tomography (SRT) inversion functionality.

Uses PyGIMLi's TravelTimeManager for forward modeling and Jacobian
computation. Provides custom Gauss-Newton inversion with the same
architecture as ERTInversion but for travel-time data.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pygimli as pg
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager
from scipy.sparse import diags, issparse

from PyHydroGeophysX._internal.utils import noop as _noop_log
from ..solvers.linear_solvers import generalized_solver
from .base import InversionBase, InversionResult
# The plateau loop, the lambda search and the run/result shapes are method
# agnostic; they were written for ERT and are shared rather than duplicated.
from .ert_inversion import (
    ERTRun,
    ModelResult,
    _fit_to_plateau,
    search_lambda_for_chi2,
)
from .metrics import metrics_from_manager


class _SRTEngine:
    """The in-house Gauss-Newton travel-time inversion, driven at one lambda.

    Set up once and re-run at different lambdas: ``setup()`` builds the forward
    operator and the constraint matrix, which is the expensive part and does not
    depend on lambda. Unlike ERT, the penalty here is on ``Wm * m`` directly, so
    a warm start needs no reference model to pin.
    """

    name = "pyhydro"

    def __init__(self, data, mesh=None, secondary_nodes: int = 3, **kwargs):
        self.container = data
        self._inv = SRTInversion(data, mesh=mesh, verbose=False,
                                 secNodes=int(max(1, secondary_nodes)), **kwargs)
        self._inv.setup()
        self.mesh = getattr(self._inv.fop, "paraDomain", self._inv.mesh)

    def reference_model(self):
        """No separate reference: the penalty is on the model itself."""
        return None

    def ray_paths(self, velocity, *, log=_noop_log):
        """First-arrival ray paths through ``velocity``, or an empty list.

        ``TravelTimeManager.getRayPaths`` rebuilds the Jacobian for the model it
        is given, so this is one extra forward-sensitivity pass on top of the
        inversion. It is worth it: where no ray reaches, the velocity is the
        regularization pulling on neighbours, and the overlay is what makes that
        visible.
        """
        manager = getattr(self._inv, "mgr", None)
        if manager is None or not callable(getattr(manager, "getRayPaths", None)):
            return []
        try:
            paths = manager.getRayPaths(model=np.asarray(velocity, dtype=float))
        except Exception as exc:  # noqa: BLE001 - the overlay is never essential
            log(f"  ray paths unavailable: {exc}")
            return []
        return [np.asarray(p, dtype=float) for p in paths
                if np.asarray(p).ndim == 2 and np.asarray(p).shape[0] > 1]

    def fit(self, *, lam, max_iterations, plateau_tolerance, target_chi2,
            start_model=None, reference_model=None):
        self._inv.parameters["lambda_val"] = float(lam)
        self._inv.parameters["max_iterations"] = int(max_iterations)
        self._inv.parameters["target_chi_squared"] = float(target_chi2)
        self._inv.parameters["convergence_tolerance"] = float(plateau_tolerance)
        res = self._inv.run(initial_model=start_model)
        history = [float(c) for c in res.iteration_chi2]
        return ERTRun(
            lam=float(lam),
            chi2=float(res.meta.get("chi2", history[-1] if history else float("nan"))),
            iterations=int(res.meta.get("iterations", len(history))),
            stop=str(res.meta.get("stop_reason", "plateau")),
            convergence=history,
            model=np.asarray(res.final_model, dtype=float),
            response=np.asarray(res.predicted_data, dtype=float),
            mesh=res.mesh,
            coverage=None if res.coverage is None else np.asarray(res.coverage, float),
            metrics={"chi2": float(res.meta.get("chi2", float("nan"))),
                     "lambda": float(lam), "method": "SRT",
                     "iterations": int(res.meta.get("iterations", len(history))),
                     "n_data": int(self.container.size())},
        )


class _SRTModelResult(ModelResult):
    """A travel-time result that can also hand back its ray paths.

    Only built when paths were captured, so ``getRayPaths`` being present is a
    reliable signal that the overlay has something to draw; a version that
    always existed and sometimes returned nothing would show a control that
    does nothing.
    """

    def __init__(self, *args, ray_paths, **kwargs):
        super().__init__(*args, **kwargs)
        self._ray_paths = ray_paths

    def getRayPaths(self, model=None):  # noqa: N802 - matches the PyGIMLi name
        return self._ray_paths


def _make_srt_result(mesh, velocity, response, coverage, ray_paths):
    if ray_paths:
        return _SRTModelResult(mesh, velocity, response, coverage,
                               velocity=velocity, ray_paths=ray_paths)
    return ModelResult(mesh, velocity, response, coverage, velocity=velocity)


def build_srt_mesh(data, *, mesh_quality: float = 32.0, para_depth: float = 0.0,
                   para_max_cell_size: float = 0.0,
                   log: Callable[[str], None] = _noop_log):
    """Build the travel-time inversion mesh, mirroring the ERT pipeline.

    PyGIMLi sizes the parameter domain from the array length when ``paraDepth``
    is left at 0. For refraction that reaches well past where any ray turns, so
    the deep cells are unconstrained and only slow the run down; capping the
    depth removes them. Cell size and quality trade resolution against cost the
    same way they do for ERT.
    """
    kwargs: Dict[str, Any] = {"quality": float(mesh_quality)}
    if float(para_depth) > 0:
        kwargs["paraDepth"] = float(para_depth)
    if float(para_max_cell_size) > 0:
        kwargs["paraMaxCellSize"] = float(para_max_cell_size)
    mesh = tt.TravelTimeManager().createMesh(data=data, **kwargs)
    log(f"  mesh: {mesh.cellCount()} cells, quality {float(mesh_quality):g}"
        + (f", capped at {float(para_depth):g} m depth" if float(para_depth) > 0
           else ", depth sized from the array"))
    return mesh


def run_srt_manager_inversion(
    travel_time_path: str | Path,
    output_dir: str | Path,
    *,
    engine: str = "pygimli",
    lam: float = 50.0,
    max_iterations: int = 20,
    plateau_tolerance: float = 0.005,
    max_total_iterations: int = 60,
    mesh_quality: float = 32.0,
    para_depth: float = 0.0,
    para_max_cell_size: float = 0.0,
    secondary_nodes: int = 3,
    auto_lambda: bool = False,
    target_chi2: float = 1.0,
    chi2_tolerance: float = 0.2,
    max_lambda_trials: int = 6,
    lambda_warm_start: bool = True,
    log: Callable[[str], None] = _noop_log,
) -> Dict[str, Any]:
    """Invert one travel-time dataset for velocity.

    With ``auto_lambda`` the same machinery as the ERT pipeline applies: each
    lambda is iterated to a plateau before its chi2 counts, the sweep relaxes
    from the requested lambda downward, and each trial continues from the
    nearest lambda already solved. ``engine="pyhydro"`` uses the in-house
    Gauss-Newton solver, which is what the search can drive; ``"pygimli"`` runs
    ``TravelTimeManager`` once and is the historical default.

    The mesh is built once here and handed to whichever engine runs, so the two
    invert the same domain. ``para_depth`` and ``para_max_cell_size`` take 0 to
    mean "let PyGIMLi size it from the array"; ``secondary_nodes`` refines the
    ray tracing without adding unknowns.
    """
    source = Path(travel_time_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        data = tt.load(str(source))
    except Exception:  # noqa: BLE001 - retain the legacy fallback
        data = pg.DataContainer(str(source), "s g")
    if data is None or int(data.size()) == 0:
        raise ValueError(f"No travel-time measurements found in {source}.")
    log(f"Inverting {int(data.size())} travel times…")

    inversion_mesh = build_srt_mesh(
        data, mesh_quality=mesh_quality, para_depth=para_depth,
        para_max_cell_size=para_max_cell_size, log=log)

    if str(engine).lower() == "pyhydro":
        return _run_srt_pipeline(
            data, out, lam=lam, max_iterations=max_iterations,
            plateau_tolerance=plateau_tolerance,
            max_total_iterations=max_total_iterations, auto_lambda=auto_lambda,
            target_chi2=target_chi2, chi2_tolerance=chi2_tolerance,
            max_lambda_trials=max_lambda_trials,
            lambda_warm_start=lambda_warm_start, mesh=inversion_mesh,
            secondary_nodes=secondary_nodes, log=log,
        )

    manager = tt.TravelTimeManager(data)
    manager.invert(data, mesh=inversion_mesh,
                   secNodes=int(max(1, secondary_nodes)), verbose=False)
    metrics, convergence = metrics_from_manager(
        manager, n_data=int(data.size()), method="SRT"
    )
    mesh = manager.paraDomain
    try:
        velocity = np.asarray(manager.velocity, dtype=float)
    except Exception:  # noqa: BLE001 - compatibility with older PyGIMLi
        model = np.asarray(manager.model, dtype=float)
        velocity = model if np.nanmedian(model) > 1.0 else 1.0 / model
    vtk_path = ""
    try:
        mesh["velocity"] = velocity
        vtk_path = str(out / "velocity_model.vtk")
        mesh.exportVTK(vtk_path)
    except Exception:  # noqa: BLE001 - VTK remains a best-effort artifact
        vtk_path = ""
    return {
        "mgr": manager,
        "n": int(data.size()),
        "vtk": vtk_path,
        "metrics": metrics,
        "convergence": convergence,
        "engine": "pygimli",
        "lambda_requested": float(lam),
        "lambda_used": float(lam),
        "auto_lambda_status": "off",
        "auto_lambda_note": "",
        "lambda_trials": [],
        "convergence_track": [],
    }


def _run_srt_pipeline(data, out: Path, *, lam, max_iterations, plateau_tolerance,
                      max_total_iterations, auto_lambda, target_chi2,
                      chi2_tolerance, max_lambda_trials, lambda_warm_start,
                      mesh=None, secondary_nodes: int = 3,
                      log=_noop_log) -> Dict[str, Any]:
    """Plateau-converged travel-time inversion with an optional lambda sweep."""
    engine = _SRTEngine(data, mesh=mesh, secondary_nodes=secondary_nodes)
    requested = float(lam)
    target, tol = float(target_chi2), abs(float(chi2_tolerance))

    def to_plateau(value, start_model=None):
        return _fit_to_plateau(
            engine, lam=value, max_iterations=max_iterations,
            plateau_tolerance=plateau_tolerance, target_chi2=target,
            max_total_iterations=max_total_iterations, start_model=start_model,
            log=log,
        )

    first = to_plateau(requested)
    log(f"  lam {requested:g} -> chi2 {first.chi2:.3f} "
        f"({first.iterations} it, {first.stop})")

    result: Dict[str, Any] = {
        "engine": "pyhydro", "lambda_requested": requested,
        "lambda_used": requested, "auto_lambda_status": "off",
        "auto_lambda_note": "", "lambda_trials": [],
        "convergence_stop": first.stop,
    }
    run = first
    track = [{"stage": "start", "lambda": requested, "n_data": int(data.size()),
              "chi2": [float(c) for c in first.convergence]}]

    best = {"run": first, "chi2": first.chi2, "lam": requested}
    if auto_lambda and first.chi2 == first.chi2 and abs(first.chi2 - target) > tol:
        log(f"  chi2 {first.chi2:.2f} outside {target:g} +/- {tol:g}; "
            f"relaxing lambda (max {int(max_lambda_trials)} trials)")
        solved = {requested: first.model}

        def evaluate(trial_lam: float) -> float:
            seed = None
            if lambda_warm_start and solved:
                nearest = min(solved, key=lambda d: abs(np.log(d / trial_lam)))
                seed = (solved[nearest], nearest)
            trial = to_plateau(float(trial_lam), seed[0] if seed else None)
            solved[float(trial_lam)] = trial.model
            track.append({"stage": "lambda", "lambda": float(trial_lam),
                          "n_data": int(data.size()),
                          "chi2": [float(c) for c in trial.convergence]})
            log(f"  lam {trial_lam:g}"
                + (f" from {seed[1]:g}" if seed else " cold")
                + f" -> chi2 {trial.chi2:.3f} ({trial.iterations} it)")
            if trial.chi2 == trial.chi2 and abs(trial.chi2 - target) < abs(best["chi2"] - target):
                best.update(run=trial, chi2=trial.chi2, lam=float(trial_lam))
            return trial.chi2

        search = search_lambda_for_chi2(
            evaluate, start_lambda=requested, start_chi2=first.chi2,
            target_chi2=target, tolerance=tol, max_trials=max_lambda_trials, log=log,
        )
        result["auto_lambda_status"] = search["status"]
        result["lambda_trials"] = [
            {"lambda": float(t["lambda"]), "chi2": float(t["chi2"])}
            for t in search["trials"]
        ]
        if float(best["lam"]) != requested:
            run = best["run"]
            result["lambda_used"] = float(best["lam"])
            result["auto_lambda_note"] = (
                f"Auto-λ: {requested:g} → {best['lam']:g}, χ² {first.chi2:.2f} → "
                f"{best['chi2']:.2f} in {len(search['trials']) - 1} trial(s).")
        else:
            result["auto_lambda_status"] = "no_improvement"
            result["auto_lambda_note"] = (
                f"Auto-λ: no λ beat {requested:g} (χ² {first.chi2:.2f}).")
        if result["auto_lambda_note"]:
            log("  " + result["auto_lambda_note"])

    velocity = np.asarray(run.model, dtype=float)
    mesh = run.mesh
    # Ray paths for the model that is actually being returned. The in-house
    # solver hands back arrays rather than a manager, so without capturing them
    # here the velocity view has nothing to overlay and hides the control.
    ray_paths = engine.ray_paths(velocity, log=log)
    vtk_path = ""
    try:
        mesh["velocity"] = velocity
        vtk_path = str(out / "velocity_model.vtk")
        mesh.exportVTK(vtk_path)
    except Exception:  # noqa: BLE001 - VTK remains a best-effort artifact
        vtk_path = ""

    result.update({
        # velocity= as well as the model, because the travel-time viewer and the
        # VTK export both look for a manager's ``velocity``.
        "mgr": _make_srt_result(mesh, velocity, run.response, run.coverage,
                                ray_paths),
        "n": int(data.size()),
        "vtk": vtk_path,
        "chi2": run.chi2,
        "metrics": dict(run.metrics),
        "convergence": list(run.convergence),
        "convergence_track": [seg for seg in track if seg["chi2"]],
    })
    return result


# ---------------------------------------------------------------------------
# SRTInversion
# ---------------------------------------------------------------------------
class SRTInversion(InversionBase):
    """
    Seismic Refraction Tomography inversion class.

    Inverts travel-time data for subsurface velocity via log-slowness,
    and uses a Gauss-Newton optimization loop.
    """

    def __init__(self, data_file: str, mesh: Optional[pg.Mesh] = None, **kwargs: Any):
        """
        Initialize SRT inversion.

        Args:
            data_file: Path to travel-time data file (e.g. .dat/.sgt).
            mesh: Mesh for inversion (created if None).
            **kwargs: Additional inversion parameters, aligned with ERT naming where possible:
                - lambda_val: Regularization parameter.
                - method: Linear solver ('cgls', 'lsqr', etc.).
                - model_constraints: (min_velocity, max_velocity) bounds.
                - max_iterations: Maximum GN iterations.
                - target_chi_squared: Stop criterion for chi2 (default 1.0).
                - lambda_rate: Lambda reduction rate per iteration.
                - lambda_min: Minimum lambda value.
                - relativeError: Relative data error.
                - absoluteUError: Absolute data error (ERT-style name).
                - zWeight: Vertical smoothness weighting.
                - vTop: Starting-model velocity near surface.
                - vBottom: Starting-model velocity at depth.
                - paraMaxCellSize: Inversion mesh max cell size.
                - paraDepth: Inversion mesh depth.
                - quality: Mesh quality parameter.
                - line_search_maxiter: Max line-search halvings.
                - line_search_c: Armijo coefficient.
                - solver_maxiter: Linear solver max iterations.
                - solver_tol: Linear solver tolerance.
        """
        # Keep ERT-style name for user-facing consistency.
        if "absoluteUError" in kwargs and "absoluteError" not in kwargs:
            kwargs["absoluteError"] = kwargs["absoluteUError"]
        if "absoluteError" in kwargs and "absoluteUError" not in kwargs:
            kwargs["absoluteUError"] = kwargs["absoluteError"]

        # An already-loaded container is accepted so a caller re-running at
        # several lambdas does not have to round-trip through a file.
        data = data_file if isinstance(data_file, pg.DataContainer) \
            else tt.load(str(data_file))
        if not isinstance(data, pg.DataContainer):
            raise TypeError("Loaded SRT data must be a PyGIMLi DataContainer.")
        required_fields = ("s", "g", "t")
        missing = [field for field in required_fields if field not in data.dataMap()]
        if missing:
            raise ValueError(
                f"SRT data is missing required fields: {missing}. "
                "Expected travel-time data with 's', 'g', and 't'."
            )
        if data.size() == 0:
            raise ValueError(
                "Loaded SRT data has zero valid measurements. "
                "Check file format and sensor indexing."
            )

        super().__init__(data, mesh, **kwargs)

        defaults = {
            "lambda_val": 50.0,
            # 'H' below is the Gauss-Newton normal matrix, which needs a
            # symmetric solver; the old 'cgls' default is a least-squares method
            # and works with the square of its condition number. Pass
            # method='cgls' to reproduce a run from before this became the
            # default.
            "method": "spd_cholesky",
            "zWeight": 0.2,
            "vTop": 500.0,
            "vBottom": 5000.0,
            "model_constraints": (100.0, 10000.0),
            "max_iterations": 20,
            "relativeError": 0.03,
            "absoluteUError": 0.001,
            "absoluteError": 0.001,
            "paraMaxCellSize": 2.0,
            "paraDepth": 40.0,
            "quality": 32,
            # Extra nodes along cell edges for the ray tracer. They sharpen the
            # travel times without adding unknowns to the inversion.
            "secNodes": 3,
            "line_search_maxiter": 12,
            "line_search_c": 1e-4,
            "solver_maxiter": 200,
            "solver_tol": 1e-8,
            "lambda_rate": 1.0,
            "lambda_min": 1.0,
            # Stopping. The plateau cut was hard-coded at 0.01; a lambda search
            # needs it configurable so a flattened misfit is attributable to
            # lambda rather than to where the iterations happened to stop.
            "target_chi_squared": 1.0,
            "convergence_tolerance": 0.005,
            "verbose": True,
        }
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

        self.mgr: Optional[TravelTimeManager] = None
        self.fop = None
        self.t_obs: Optional[np.ndarray] = None
        self.Wd = None
        self.Wd_sq = None
        self.Wd_diag: Optional[np.ndarray] = None
        self.Wm = None
        self._setup_complete = False

    @staticmethod
    def _to_col(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=float)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    @staticmethod
    def _jacobian_to_numpy(jacobian: Any) -> np.ndarray:
        """Convert PyGIMLi Jacobian matrix to a dense NumPy array across versions."""
        try:
            return pg.utils.sparseMatrix2coo(jacobian).toarray().astype(float, copy=False)
        except Exception:
            pass
        try:
            return np.asarray(pg.utils.gmat2numpy(jacobian), dtype=float)
        except Exception:
            pass

        arr = np.asarray(jacobian, dtype=float)
        if arr.ndim == 0:
            raise TypeError(f"Unsupported Jacobian type: {type(jacobian)}")
        return arr

    @staticmethod
    def _cell_centers_xy(mesh: pg.Mesh) -> np.ndarray:
        centers_raw = np.asarray(mesh.cellCenters())
        if centers_raw.ndim == 2 and centers_raw.shape[1] >= 2:
            return centers_raw[:, :2].astype(float)

        centers = np.zeros((mesh.cellCount(), 2), dtype=float)
        for i, center in enumerate(mesh.cellCenters()):
            if hasattr(center, "x") and hasattr(center, "y"):
                centers[i, 0] = float(center.x())
                centers[i, 1] = float(center.y())
            else:
                centers[i, 0] = float(center[0])
                centers[i, 1] = float(center[1])
        return centers

    def _estimate_data_errors(self, t_obs: np.ndarray) -> np.ndarray:
        if "err" in self.data.dataMap():
            err = np.asarray(self.data["err"].array(), dtype=float).ravel()
            valid = np.all(np.isfinite(err)) and np.any(err > 0)
            if valid and err.size == t_obs.size:
                return np.clip(err, 1e-12, None)

        rel = float(self.parameters["relativeError"])
        abs_err = float(self.parameters.get("absoluteUError", self.parameters.get("absoluteError", 0.001)))
        return np.sqrt((rel * np.abs(t_obs)) ** 2 + abs_err**2)

    def _build_initial_velocity(self, n_model: int) -> np.ndarray:
        min_v, max_v = self.parameters["model_constraints"]
        min_v = max(float(min_v), 1e-6)
        max_v = max(float(max_v), min_v + 1e-6)

        if self.mesh is None:
            v = np.full(n_model, float(self.parameters["vTop"]), dtype=float)
            return np.clip(v, min_v, max_v)

        centers = self._cell_centers_xy(self.mesh)
        depth = centers[:, 1].max() - centers[:, 1]
        depth_span = np.ptp(depth)
        if depth_span <= 0:
            depth_norm = np.zeros_like(depth)
        else:
            depth_norm = depth / depth_span

        v_top = float(self.parameters["vTop"])
        v_bottom = float(self.parameters["vBottom"])
        velocity = v_top + depth_norm * (v_bottom - v_top)
        return np.clip(velocity, min_v, max_v)

    def setup(self) -> None:
        if self.mesh is None:
            temp_mgr = TravelTimeManager()
            self.mesh = temp_mgr.createMesh(
                self.data,
                paraMaxCellSize=float(self.parameters["paraMaxCellSize"]),
                quality=int(self.parameters["quality"]),
                paraDepth=float(self.parameters["paraDepth"]),
            )

        self.mgr = TravelTimeManager()
        self.mgr.setData(self.data)
        sec_nodes = int(max(1, self.parameters.get("secNodes", 3)))
        try:
            self.mgr.setMesh(self.mesh, secNodes=sec_nodes)
        except TypeError:  # pragma: no cover - older PyGIMLi without the kwarg
            self.mgr.setMesh(self.mesh)
        self.fop = self.mgr.fop

        self.t_obs = self._to_col(np.asarray(self.data["t"], dtype=float))
        err = self._estimate_data_errors(self.t_obs.ravel())
        self.Wd_diag = 1.0 / np.clip(err, 1e-12, None)
        self.Wd = diags(self.Wd_diag)
        self.Wd_sq = diags(self.Wd_diag**2)

        rm = self.fop.regionManager()
        rm.setZWeight(float(self.parameters["zWeight"]))
        Ctmp = pg.matrix.RSparseMapMatrix()
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)

        # `fillConstraints` already applies region-dependent weighting
        # (including zWeight). Do not multiply constraintWeights again.
        self.Wm = pg.utils.sparseMatrix2coo(Ctmp).tocsr()

        self._setup_complete = True

    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        if not self._setup_complete:
            self.setup()

        if self.fop is None or self.t_obs is None or self.Wm is None or self.Wd_diag is None:
            raise RuntimeError("SRT inversion setup is incomplete.")

        n_model = self.Wm.shape[1]

        if initial_model is None:
            v0 = self._build_initial_velocity(n_model)
        else:
            v0 = np.asarray(initial_model, dtype=float).ravel()
            if v0.size != n_model:
                raise ValueError(f"initial_model must have {n_model} parameters, got {v0.size}.")

        min_v, max_v = self.parameters["model_constraints"]
        min_v = max(float(min_v), 1e-6)
        max_v = max(float(max_v), min_v + 1e-6)

        v0 = np.clip(v0, min_v, max_v)
        m = np.log(1.0 / np.clip(v0, 1e-12, None))

        min_m = np.log(1.0 / max_v)
        max_m = np.log(1.0 / min_v)
        m = np.clip(m, min_m, max_m)

        lam = float(self.parameters["lambda_val"])
        lam_rate = float(self.parameters.get("lambda_rate", 1.0))
        lam_min = float(self.parameters.get("lambda_min", lam))
        target_chi2 = float(self.parameters.get("target_chi_squared", 1.0))
        # Was hard-coded at 0.01. A lambda search needs it configurable: a run
        # that stops on a 1 % plateau may still be descending, and its chi2 then
        # says more about the iteration budget than about lambda.
        dphi_tol = float(self.parameters.get("convergence_tolerance", 0.005))
        verbose = bool(self.parameters.get("verbose", True))
        stop_reason = "iteration_cap"
        line_search_failures = 0

        result = InversionResult()
        prev_chi2: Optional[float] = None

        if verbose:
            print("SRTInversion note: reported chi2 below is evaluated at the start of each iteration (pre-update).")
            print("PyGIMLi inv.iter chi2 is reported after oneStep update.")
        for iteration in range(int(self.parameters["max_iterations"])):
            if verbose:
                print(f"-------------------Iteration: {iteration} ---------------------------")
            slowness = np.exp(m)
            t_pred = self._to_col(np.asarray(self.fop.response(pg.Vector(slowness)), dtype=float))

            self.fop.createJacobian(pg.Vector(slowness))
            J = self._jacobian_to_numpy(self.fop.jacobian())
            if J.ndim == 1:
                J = J.reshape(-1, 1)

            J_log = J * slowness.reshape(1, -1)
            residual = self.t_obs - t_pred

            wd = self.Wd_diag.reshape(-1, 1)
            wd2 = wd**2
            phi_d = float((wd * residual).T.dot(wd * residual).item())

            m_col = self._to_col(m)
            reg_vec = self.Wm.dot(m_col)
            phi_m = float(reg_vec.T.dot(reg_vec).item())

            chi2 = phi_d / max(self.t_obs.size, 1)
            d_phi = 1.0 if prev_chi2 is None else abs(chi2 - prev_chi2) / max(abs(prev_chi2), 1e-12)
            prev_chi2 = chi2
            obj = phi_d + lam * phi_m

            result.iteration_models.append(np.exp(-m).copy())
            result.iteration_chi2.append(float(chi2))
            result.iteration_data_errors.append(residual.ravel().copy())

            if verbose:
                print(f"chi2 (pre-update): {chi2:.6f}")
                print(f"dPhi (pre-update): {d_phi:.6f}")
                print(f"phi_d (pre-update): {phi_d:.6f}")
                print(f"phi_m (pre-update): {phi_m:.6f}")
                print(f"obj (pre-update): {obj:.6f}")
                print(f"lambda: {lam:.6f}")

            if chi2 <= target_chi2:
                stop_reason = "target"
                if verbose:
                    print("Converged: chi2 target reached.")
                break
            if iteration > 0 and d_phi < dphi_tol:
                stop_reason = "plateau"
                if verbose:
                    print("Converged: the misfit has flattened.")
                break

            H_data = J_log.T.dot(wd2 * J_log)
            H_reg = lam * self.Wm.T.dot(self.Wm)
            H = H_data + (H_reg.toarray() if issparse(H_reg) else H_reg)

            g_data = -J_log.T.dot(wd2 * residual)
            g_reg = lam * self.Wm.T.dot(reg_vec)
            g = g_data + g_reg

            # 'H' is the Gauss-Newton normal matrix, so it needs a symmetric
            # solver. A least-squares method such as 'cgls' would work on
            # 'H^T H dm = H^T (-g)' instead, squaring the condition number.
            # overwrite_a=True lets the factorization run in H's own buffer;
            # nothing reads H after this call.
            dm = generalized_solver(
                H,
                -g,
                method=str(self.parameters["method"]),
                maxiter=int(self.parameters.get("solver_maxiter", 200)),
                tol=float(self.parameters.get("solver_tol", 1e-8)),
                overwrite_a=True,
            )
            dm = self._to_col(dm)
            del H

            current_obj = obj
            directional = float(dm.T.dot(g).item())
            step = 1.0
            accepted = False
            accepted_step = step

            for _ in range(int(self.parameters.get("line_search_maxiter", 12))):
                m_trial = np.clip(self._to_col(m) + step * dm, min_m, max_m).ravel()
                s_trial = np.exp(m_trial)
                t_trial = self._to_col(np.asarray(self.fop.response(pg.Vector(s_trial)), dtype=float))
                res_trial = self.t_obs - t_trial

                phi_d_trial = float((wd * res_trial).T.dot(wd * res_trial).item())
                reg_trial = self.Wm.dot(self._to_col(m_trial))
                phi_m_trial = float(reg_trial.T.dot(reg_trial).item())

                obj_trial = phi_d_trial + lam * phi_m_trial
                armijo = current_obj - float(self.parameters.get("line_search_c", 1e-4)) * step * directional

                if obj_trial < armijo:
                    m = m_trial
                    accepted = True
                    accepted_step = step
                    break
                step *= 0.5

            if not accepted:
                accepted_step = 0.1
                line_search_failures += 1
                if verbose:
                    print("Line search FAIL EXIT")
                m = np.clip((self._to_col(m) + accepted_step * dm).ravel(), min_m, max_m)

            if verbose:
                print(f"accepted_step: {accepted_step:.6f}")
                print(f"update_norm: {float(np.linalg.norm(dm)):.6e}")

            if lam_rate > 0:
                lam = max(lam_min, lam * lam_rate)

        final_velocity = np.exp(-m)
        final_slowness = np.exp(m)
        final_pred = np.asarray(self.fop.response(pg.Vector(final_slowness)), dtype=float).ravel()

        # Every chi2 recorded in the loop is pre-update, so when the loop ends by
        # exhausting its iterations the last entry describes the model from before
        # the final step, not the one being returned. Close the gap.
        final_residual = self.t_obs - self._to_col(final_pred)
        wd_final = self.Wd_diag.reshape(-1, 1)
        final_chi2 = float(
            (wd_final * final_residual).T.dot(wd_final * final_residual).item()
        ) / max(self.t_obs.size, 1)
        if stop_reason == "iteration_cap":
            result.iteration_chi2.append(final_chi2)
            result.iteration_models.append(final_velocity.copy())
            result.iteration_data_errors.append(final_residual.ravel().copy())

        self.fop.createJacobian(pg.Vector(final_slowness))
        # Use the same coverage definition as PyGIMLi TravelTimeManager:
        # standardizedCoverage = sign(|C^T * C * rayCoverage|).
        if self.mgr is not None:
            try:
                if hasattr(self.fop, "createConstraints"):
                    self.fop.createConstraints()
                coverage = np.asarray(self.mgr.standardizedCoverage(), dtype=float).ravel()
            except Exception:
                # Fallback with the same standardized-coverage formula.
                J = self.fop.jacobian()
                ray_coverage = np.asarray(J.transMult(np.ones(J.rows())), dtype=float).ravel()
                Ctmp = pg.matrix.RSparseMapMatrix()
                self.fop.regionManager().fillConstraints(Ctmp)
                C = pg.utils.sparseMatrix2coo(Ctmp).tocsr()
                coverage = np.sign(np.abs(C.T.dot(C.dot(ray_coverage))))
        else:
            coverage = None

        result.final_model = final_velocity
        result.predicted_data = final_pred
        result.coverage = coverage
        result.mesh = self.fop.paraDomain if hasattr(self.fop, "paraDomain") else self.mesh
        result.meta["inversion_parameters"] = dict(self.parameters)
        result.meta["final_lambda"] = lam
        # Why the loop ended, so a caller driving lambda can tell "this lambda
        # cannot do better" apart from "this run ran out of iterations".
        result.meta["stop_reason"] = stop_reason
        result.meta["iterations"] = len(result.iteration_chi2)
        result.meta["line_search_failures"] = line_search_failures
        result.meta["chi2"] = final_chi2
        result.meta["lambda"] = float(self.parameters["lambda_val"])

        if verbose:
            print("End of inversion")
        return result
