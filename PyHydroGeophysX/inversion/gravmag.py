"""SimPEG gravity and magnetics inversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX._internal.utils import noop as _noop
from PyHydroGeophysX.data_processing import table_io
from PyHydroGeophysX.data_processing.gravmag import (
    regional_residual,
    spatially_balanced_indices,
)

LogFn = Callable[[str], None]

_DEFAULT_NOISE_FLOOR = {"gravity": 0.5, "magnetics": 2.0}


class InversionBackendUnavailable(BackendUnavailable):
    """SimPEG / discretize / a usable solver could not be imported."""


def backend_status() -> Dict[str, Any]:
    """Report whether the SimPEG potential-field inversion stack is available."""
    try:
        import pymatsolver  # noqa: F401
        from discretize import TensorMesh  # noqa: F401
        from simpeg.potential_fields import gravity, magnetics  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - optional numerical backend
        return {"available": False, "error": str(exc)}
    return {"available": True, "error": ""}


def invert_gravmag(x, y, value, kind: str, *, z: Optional[np.ndarray] = None,
                   field: Optional[Dict[str, Any]] = None, detrend: int = 0,
                   n_xy: int = 22, n_z: int = 12, max_iterations: int = 20,
                   beta0_ratio: float = 1.0, max_stations: int = 600,
                   relative_error: float = 0.03, noise_floor: Optional[float] = None,
                   solver: str = "simpeg", auto_beta: bool = True,
                   target_chi2: float = 1.0, chi2_tolerance: float = 0.2,
                   max_beta_trials: int = 6, sensitivity_power: float = 1.0,
                   out_dir: Optional[str] = None, random_seed: Optional[int] = 42,
                   log: LogFn = _noop) -> Dict[str, Any]:
    """Run a SimPEG 3D potential-field inversion under the survey.

    ``gravity`` recovers a density-contrast model (g/cc); ``magnetics`` recovers a
    susceptibility model (SI) and needs ``field`` = {inclination, declination,
    strength_nT}. ``z`` is optional per-station elevation (m, positive upward); a
    missing value falls back to 1 m. ``detrend`` (0..3) removes a polynomial
    regional trend before inversion. The returned grid uses elevation increasing
    upward. ``random_seed`` makes SimPEG's eigenvalue-based beta estimate
    reproducible. Raises :class:`InversionBackendUnavailable` if SimPEG is missing.
    """
    try:
        import pymatsolver
        from discretize import TensorMesh
        from simpeg import (data, data_misfit, directives, inverse_problem,
                            inversion, maps, optimization, regularization)
        is_grav = str(kind).lower().startswith("grav")
        if is_grav:
            from simpeg.potential_fields import gravity as pf
        else:
            from simpeg.potential_fields import magnetics as pf
    except Exception as exc:  # noqa: BLE001
        raise InversionBackendUnavailable(str(exc))

    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    value = np.asarray(value, float).ravel()
    if not (x.size == y.size == value.size):
        raise ValueError("x, y and value must have the same number of stations.")
    if z is None:
        z = np.full(x.size, 1.0, dtype=float)
    else:
        z = np.asarray(z, float).ravel()
        if z.size != x.size:
            raise ValueError("z must be omitted or contain one elevation per station.")
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(value) & np.isfinite(z)
    x, y, value, z = x[good], y[good], value[good], z[good]
    if x.size < 20:
        raise ValueError("Need at least ~20 stations for a stable inversion.")
    if int(detrend) > 0:
        _, value = regional_residual(x, y, value, degree=int(detrend))
        log(f"Removed a degree-{int(detrend)} regional trend before inversion.")
    n_input = int(x.size)
    if x.size > int(max_stations):
        idx = spatially_balanced_indices(x, y, int(max_stations))
        x, y, value, z = x[idx], y[idx], value[idx], z[idx]
    log(f"{kind} inversion: {x.size} stations")

    x0, x1 = float(x.min()), float(x.max()); y0, y1 = float(y.min()), float(y.max())
    csx = (x1 - x0) / int(n_xy); csy = (y1 - y0) / int(n_xy)
    csz = max(csx, csy) * 0.6
    padx, pady = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    nx, ny, nz = int(n_xy) + 4, int(n_xy) + 4, int(n_z)
    ox, oy, oz = x0 - padx - 2 * csx, y0 - pady - 2 * csy, -csz * nz
    mesh = TensorMesh([[(csx, nx)], [(csy, ny)], [(csz, nz)]], origin=[ox, oy, oz])
    actv = np.ones(mesh.n_cells, dtype=bool)
    model_map = maps.IdentityMap(nP=mesh.n_cells)

    rx = pf.receivers.Point(np.c_[x, y, z],
                            components=("gz" if is_grav else "tmi"))
    if is_grav:
        survey = pf.survey.Survey(pf.sources.SourceField(receiver_list=[rx]))
        sim = pf.simulation.Simulation3DIntegral(
            mesh=mesh, survey=survey, rhoMap=model_map, active_cells=actv, engine="geoana")
        m_label, m_cmap = "density (g/cc)", "RdBu_r"
        lower, upper, floor = -1.0, 1.0, 0.5
    else:
        f = field or {}
        src = pf.sources.UniformBackgroundField(
            receiver_list=[rx], amplitude=float(f.get("strength_nT", 50000.0)),
            inclination=float(f.get("inclination", 60.0)),
            declination=float(f.get("declination", 0.0)))
        survey = pf.survey.Survey(src)
        sim = pf.simulation.Simulation3DIntegral(
            mesh=mesh, survey=survey, chiMap=model_map, active_cells=actv, engine="geoana")
        m_label, m_cmap = "susceptibility (SI)", "viridis"
        # Smooth susceptibility "contrast": bound symmetric about 0 so the start
        # model (0) is interior, otherwise ProjectedGNCG is stuck at the 0 lower bound.
        lower, upper, floor = -0.5, 0.5, 2.0
    # scipy LU solver: avoids the Pardiso/MKL native crash in this environment.
    sim.solver = pymatsolver.Solver
    sim.solver_opts = {}

    if noise_floor is None:
        noise_floor = _DEFAULT_NOISE_FLOOR["gravity" if is_grav else "magnetics"]
    std = max(0.0, float(relative_error)) * np.abs(value) + max(0.0, float(noise_floor))
    dmis = data_misfit.L2DataMisfit(
        data=data.Data(survey, dobs=value, standard_deviation=std), simulation=sim)
    reg = regularization.WeightedLeastSquares(mesh, active_cells=actv)
    opt = optimization.ProjectedGNCG(maxIter=int(max_iterations), lower=lower, upper=upper,
                                     maxIterCG=20, tolCG=1e-3)
    beta_report: Dict[str, Any] = {"solver": str(solver)}
    if str(solver).lower() == "linear":
        # Potential-field sensitivities do not depend on the model, so this is a
        # quadratic problem: one linear solve per beta, no Gauss-Newton loop, no
        # beta cooling schedule, and a beta that the result can be attributed to.
        m0 = np.zeros(mesh.n_cells)
        # Weight before estimating beta: the weights change the model term, and
        # a beta scaled against the unweighted one starts in the wrong place.
        weights = apply_sensitivity_weights(sim, dmis, reg, m0,
                                            power=float(sensitivity_power), log=log)
        beta_report["sensitivity_weighted"] = weights is not None
        beta0 = estimate_beta0(dmis, reg, m0, ratio=float(beta0_ratio),
                               seed=int(random_seed or 42))
        log(f"Linear Tikhonov solve ({mesh.n_cells} cells, {value.size} data), "
            f"beta0 {beta0:.4g}")
        if auto_beta:
            swept = sweep_beta_for_chi2(
                dmis, reg, m0, int(value.size), beta0=beta0,
                target_chi2=float(target_chi2), chi2_tolerance=float(chi2_tolerance),
                max_trials=int(max_beta_trials), bounds=(lower, upper), log=log,
            )
            mrec = np.asarray(swept["model"], dtype=float)
            chi2 = float(swept["chi2"])
            convergence = [t["chi2"] for t in swept["trials"]]
            beta_report.update({k: swept[k] for k in
                                ("beta", "beta0", "status", "reason", "trials",
                                 "clipped")})
            log(f"  beta {swept['beta']:.4g} gives chi2 {chi2:.3f} "
                f"({swept['status']})")
        else:
            mrec, info = solve_tikhonov(dmis, reg, m0, beta0,
                                        bounds=(lower, upper))
            mrec = np.asarray(mrec, dtype=float)
            chi2 = float(dmis(mrec)) / max(int(value.size), 1)
            convergence = [chi2]
            beta_report.update({"beta": float(beta0), "beta0": float(beta0),
                                "status": "fixed", "trials": [],
                                "clipped": bool(info["clipped"])})
        return _gravmag_payload(
            kind, mesh, mrec, nx, ny, nz, ox, oy, oz, csx, csy, csz,
            m_label, m_cmap, chi2, value.size, n_input, relative_error,
            noise_floor, convergence, beta_report, out_dir)

    invprob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    # ``on_disk`` was added in newer SimPEG releases. ``save_txt=False`` keeps
    # iteration history in memory and is supported by both 0.24 and newer APIs.
    history_directive = directives.SaveOutputEveryIteration(save_txt=False)
    dlist = [directives.UpdateSensitivityWeights(every_iteration=False),
             directives.BetaEstimate_ByEig(beta0_ratio=float(beta0_ratio),
                                           random_seed=random_seed),
             directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
             history_directive, directives.TargetMisfit()]
    log(f"Running SimPEG inversion ({mesh.n_cells} cells, up to {int(max_iterations)} iters)…")
    mrec = np.asarray(inversion.BaseInversion(invprob, directiveList=dlist).run(np.zeros(mesh.n_cells)), float)
    try:
        phi_d = float(dmis(mrec)); chi2 = phi_d / value.size
    except Exception:  # noqa: BLE001
        chi2 = float("nan")
    convergence = [float(phi_d) / value.size for phi_d in history_directive.phi_d]

    return _gravmag_payload(
        kind, mesh, mrec, nx, ny, nz, ox, oy, oz, csx, csy, csz,
        m_label, m_cmap, chi2, value.size, n_input, relative_error,
        noise_floor, convergence, beta_report, out_dir)


def _gravmag_payload(kind, mesh, mrec, nx, ny, nz, ox, oy, oz, csx, csy, csz,
                     m_label, m_cmap, chi2, n_data, n_input, relative_error,
                     noise_floor, convergence, beta_report, out_dir):
    """Shared result shape for both the SimPEG and the linear solver paths."""
    model3d = mrec.reshape((nx, ny, nz), order="F")
    ex = ox + csx * np.arange(nx + 1)
    ey = oy + csy * np.arange(ny + 1)
    ez = oz + csz * np.arange(nz + 1)
    out: Dict[str, Any] = {
        "kind": kind, "edges": (ex, ey, ez), "model3d": model3d,
        "label": m_label, "cmap": m_cmap, "log_scale": False,
        "chi2": chi2, "n_data": int(n_data), "n_input": n_input,
        "n_cells": int(mesh.n_cells), "relative_error": float(relative_error),
        "noise_floor": float(noise_floor), "convergence": convergence,
        "model_range": [float(np.nanmin(mrec)), float(np.nanmax(mrec))],
        "beta": beta_report,
    }
    if out_dir:
        base = table_io.ensure_dir(Path(out_dir) / "gravmag_inversion")
        np.savez(base / "model_grid.npz", ex=ex, ey=ey, ez=ez, model=model3d)
        try:
            import pyvista as pv
            grid = pv.RectilinearGrid(ex, ey, ez)
            grid.cell_data[m_label] = model3d.flatten(order="F")
            grid.save(str(base / "model.vtr"))
            out["vtk"] = str(base / "model.vtr")
        except Exception:  # noqa: BLE001 - VTK export is optional
            pass
        out["output_dir"] = str(base)
    return out

#: Beta is not lambda. It scales against the data units and the mesh, so its
#: useful range sits far below the ERT/SRT regularization bounds; on a typical
#: gravity survey chi2 = 1 lands near 1e-3. The estimate below makes the search
#: scale free, and these bounds only stop it running away.
BETA_BOUNDS: Tuple[float, float] = (1e-12, 1e8)


def estimate_beta0(dmis, reg, m, *, ratio: float = 1.0, seed: int = 42,
                   n_power: int = 20) -> float:
    """Scale-free starting beta, as SimPEG's ``BetaEstimate_ByEig`` does it.

    Power-iterate both Hessians and take the ratio of their largest eigenvalues,
    so beta starts where the two objective terms are comparable regardless of
    the data units or the mesh size.
    """
    rng = np.random.default_rng(seed)

    def largest_eig(operator) -> float:
        v = rng.normal(size=m.size)
        v /= np.linalg.norm(v)
        value = 0.0
        for _ in range(int(n_power)):
            w = np.asarray(operator(v), dtype=float)
            norm = float(np.linalg.norm(w))
            if norm <= 0:
                return 0.0
            v = w / norm
            value = norm
        return value

    top_d = largest_eig(lambda v: dmis.deriv2(m, v))
    top_m = largest_eig(lambda v: reg.deriv2(m, v))
    if top_m <= 0:
        return float(ratio)
    return float(ratio) * top_d / top_m


def apply_sensitivity_weights(sim, dmis, reg, m: np.ndarray, *,
                              power: float = 1.0, floor: float = 1e-8,
                              log: LogFn = _noop) -> Optional[np.ndarray]:
    """Weight the regularization by each cell's sensitivity, and say if it worked.

    Potential-field sensitivity falls off sharply with depth, so an unweighted
    smallness term buys its misfit reduction most cheaply at the surface: the
    recovered body ends up plastered against the top of the mesh whatever its
    real depth. Weighting the model term by ``sqrt(diag(J^T W^T W J))`` is the
    standard correction (Li and Oldenburg, 1996) and is what SimPEG's
    ``UpdateSensitivityWeights`` directive applies on the iterative path.

    Measured on a 0.7 g/cc block buried 60-160 m: unweighted the peak sits at
    12 m, weighted it sits at 88 m.
    """
    diagonal = None
    try:  # SimPEG can give the diagonal without forming the full Jacobian
        diagonal = np.asarray(sim.getJtJdiag(m, W=dmis.W), dtype=float)
    except Exception:  # noqa: BLE001 - fall back to the dense sensitivities
        try:
            jac = np.asarray(sim.getJ(m), dtype=float)
            weights = dmis.W.diagonal() if hasattr(dmis.W, "diagonal") else dmis.W
            scaled = jac * np.asarray(weights, dtype=float).reshape(-1, 1)
            diagonal = np.einsum("ij,ij->j", scaled, scaled)
        except Exception as exc:  # noqa: BLE001 - weighting is best effort
            log(f"  sensitivity weighting unavailable ({exc}); "
                "the model may collect at the surface")
            return None
    if diagonal is None or diagonal.size != m.size or not np.isfinite(diagonal).any():
        return None
    top = float(np.nanmax(diagonal))
    if top <= 0:
        return None
    values = np.maximum((diagonal / top) ** (float(power) / 2.0), float(floor))
    try:
        reg.set_weights(sensitivity=values)
    except Exception as exc:  # noqa: BLE001 - older regularization API
        log(f"  could not attach sensitivity weights ({exc})")
        return None
    log(f"  sensitivity weights applied: {values.min():.2e} .. {values.max():.2e}")
    return values


def solve_tikhonov(dmis, reg, m_ref: np.ndarray, beta: float, *,
                   bounds: Optional[Tuple[float, float]] = None,
                   cg_maxiter: int = 400, cg_tol: float = 1e-8):
    """Minimize ``phi_d(m) + beta * phi_m(m)`` for a linear forward operator.

    Potential-field sensitivities do not depend on the model, so the objective
    is quadratic and a single Newton step from any point is the exact minimizer;
    there is no Gauss-Newton loop and no line search to run. The step is taken by
    conjugate gradients on SimPEG's own ``deriv``/``deriv2``, which keeps every
    weighting convention identical to the directive-driven path.

    ``bounds`` are applied by clipping. That is a projection onto the box, not a
    constrained optimum, so the returned model is only the true minimizer when it
    lies inside; ``clipped`` in the result says whether the bound bit.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    n = int(m_ref.size)
    gradient = (np.asarray(dmis.deriv(m_ref), dtype=float)
                + float(beta) * np.asarray(reg.deriv(m_ref), dtype=float))

    def matvec(v):
        return (np.asarray(dmis.deriv2(m_ref, v), dtype=float)
                + float(beta) * np.asarray(reg.deriv2(m_ref, v), dtype=float))

    operator = LinearOperator((n, n), matvec=matvec, dtype=float)
    try:  # SciPy renamed the tolerance argument
        step, info = cg(operator, -gradient, rtol=float(cg_tol),
                        maxiter=int(cg_maxiter))
    except TypeError:  # noqa: BLE001 - older SciPy
        step, info = cg(operator, -gradient, tol=float(cg_tol),
                        maxiter=int(cg_maxiter))
    model = m_ref + np.asarray(step, dtype=float)
    clipped = False
    if bounds is not None:
        lower, upper = float(min(bounds)), float(max(bounds))
        limited = np.clip(model, lower, upper)
        clipped = bool(np.any(limited != model))
        model = limited
    return model, {"cg_info": int(info), "clipped": clipped}


def sweep_beta_for_chi2(dmis, reg, m_ref: np.ndarray, n_data: int, *,
                        beta0: float, target_chi2: float = 1.0,
                        chi2_tolerance: float = 0.2, max_trials: int = 6,
                        bounds: Optional[Tuple[float, float]] = None,
                        beta_bounds: Tuple[float, float] = BETA_BOUNDS,
                        cg_maxiter: int = 400, cg_tol: float = 1e-8,
                        log: LogFn = _noop) -> Dict[str, Any]:
    """Find the beta whose chi2 lands on ``target_chi2``.

    Each trial is one linear solve rather than a whole nonlinear inversion, so
    unlike the ERT and travel-time searches this one can afford to be thorough,
    and there is nothing to warm start: the minimizer for a given beta does not
    depend on where the solve began.
    """
    from .lambda_search import search_lambda_for_chi2

    solved: Dict[float, np.ndarray] = {}
    trials: Dict[float, Dict[str, Any]] = {}

    def chi2_of(beta: float) -> float:
        model, info = solve_tikhonov(dmis, reg, m_ref, beta, bounds=bounds,
                                     cg_maxiter=cg_maxiter, cg_tol=cg_tol)
        # SimPEG's L2DataMisfit is ||W (dpred - dobs)||^2 with no half factor,
        # measured directly rather than assumed.
        value = float(dmis(model)) / max(int(n_data), 1)
        solved[float(beta)] = model
        trials[float(beta)] = {"beta": float(beta), "chi2": value,
                               "clipped": bool(info["clipped"])}
        log(f"  beta {beta:.4g} -> chi2 {value:.3f}"
            + ("  (bounds active)" if info["clipped"] else ""))
        return value

    start_chi2 = chi2_of(beta0)
    search = search_lambda_for_chi2(
        chi2_of, start_lambda=float(beta0), start_chi2=start_chi2,
        target_chi2=float(target_chi2), tolerance=float(chi2_tolerance),
        max_trials=int(max_trials), bounds=beta_bounds, log=log,
    )
    best = float(search["lam"])
    ordered = [trials[b] for b in sorted(trials, reverse=True)]
    return {
        "beta": best,
        "beta0": float(beta0),
        "model": solved[best],
        "chi2": float(search["chi2"]),
        "status": str(search["status"]),
        "reason": str(search["reason"]),
        "trials": ordered,
        "clipped": bool(trials[best]["clipped"]),
    }


__all__ = [
    "BETA_BOUNDS",
    "InversionBackendUnavailable",
    "apply_sensitivity_weights",
    "backend_status",
    "estimate_beta0",
    "invert_gravmag",
    "solve_tikhonov",
    "sweep_beta_for_chi2",
]
