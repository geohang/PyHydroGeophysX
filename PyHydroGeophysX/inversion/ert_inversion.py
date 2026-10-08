"""
Single-time ERT inversion functionality.
"""
from dataclasses import dataclass, field, fields
from functools import lru_cache
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pygimli as pg
from pygimli.physics import ert
import scipy.sparse as sp
from scipy.sparse import diags

from PyHydroGeophysX._internal.utils import noop as _noop_log
from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from ..forward.ert_forward import ertforandjac2, ertforward2
from ..solvers.linear_solvers import _SPD_METHODS, StackedSystem, generalized_solver
from .base import InversionBase, InversionResult
from .lambda_search import LAMBDA_BOUNDS, search_lambda_for_chi2
from .metrics import metrics_from_manager
from .model_result import ModelResult  # noqa: F401 - written here first
from .ert_mesh import (  # noqa: F401 - re-exported, see below
    MAX_MESH_QUALITY,
    MESH_SUFFIXES,
    ZONE_INTERFACE_MARKER,
    _ZONE_OUTLINE_MARKER,
    _interface_regions,
    _sensors_outside,
    build_inversion_mesh,
    load_inversion_mesh,
    mark_zone_interfaces,
    mesh_preview,
)

# ``LAMBDA_BOUNDS`` and ``search_lambda_for_chi2`` were defined here first and
# are re-exported so existing imports keep working. They moved to
# :mod:`PyHydroGeophysX.inversion.lambda_search` because the potential-field and
# EM1D paths need them and must not import pygimli to get them.
# The inversion mesh - built, imported, cut at zone outlines, previewed - is
# in :mod:`PyHydroGeophysX.inversion.ert_mesh`; its names are re-exported
# here, where they were defined first.


#: How ``err`` is chosen. ``file`` trusts the ``err`` column the instrument wrote
#: and only estimates where it is missing; ``estimate`` always recomputes from
#: ``relative_error``/``absolute_error``; ``max`` takes the larger of the two per
#: datum, which is the conservative reading when the file's errors look
#: optimistic. ``stack`` adds each reading's stacking spread - the scatter of the
#: potential's samples within the reading, as a fraction of it, which Subsurface
#: Insights records and the ``stack`` token carries - to the estimate in
#: quadrature. That spread is not a reading's uncertainty and usually overstates
#: it, so it is only ever asked for. ``reciprocal`` gives each reading the error
#: of a reciprocal error model ``dR = 10**b * |R|**m`` fitted to the survey's
#: reciprocal pairs (``error_model``: ``m``, ``b``, ``floor``; see
#: ``ert_io.reciprocal_model_errors``). ``file`` is the default because silently
#: discarding measured errors makes chi2 report on an error model the data never
#: had.
ERROR_SOURCES = ("file", "estimate", "max", "stack", "reciprocal")


def _estimate_errors(data, *, relative_error: float, absolute_error: float):
    """Per-datum relative error ``relative + absolute/|R|``.

    The absolute term is a resistance floor in Ohm. It matters when a survey
    spans a wide range of signal levels, because a flat percentage is far too
    optimistic for the weakest readings; with a narrow ``|R|`` range it is close
    to a constant offset and changes little.
    """
    try:
        errors = np.asarray(
            ert.estimateError(data, relativeError=float(relative_error)), dtype=float
        )
    except Exception:  # noqa: BLE001 - backend/version compatibility
        errors = np.full(int(data.size()), float(relative_error))
    if absolute_error > 0:
        if data.haveData("r"):
            resistance = np.abs(np.asarray(data["r"], dtype=float))
        elif data.haveData("rhoa") and data.haveData("k"):
            resistance = np.abs(
                np.asarray(data["rhoa"], dtype=float)
                / np.maximum(np.abs(np.asarray(data["k"], dtype=float)), 1e-12)
            )
        else:
            resistance = None
        if resistance is not None:
            errors = errors + float(absolute_error) / np.maximum(resistance, 1e-12)
    return errors


def _prepare_ert_data(
    data_path: str | Path,
    *,
    relative_error: float,
    instrument: Optional[str],
    log: Callable[[str], None],
    absolute_error: float = 0.0,
    error_source: str = "file",
    error_floor: float = 0.005,
    error_model: Optional[Dict[str, Any]] = None,
):
    """Load an ERT file and fill in geometric factors, rhoa, and per-datum errors.

    Returns ``(data, error_info)``. ``error_info`` records which error model was
    used and its spread, so the caller can tell the user that chi2 is being
    measured against, say, the instrument's 6 % rather than an assumed 5 %.
    ``error_model`` is the reciprocal error model ``error_source="reciprocal"``
    applies (``m``, ``b``, ``floor``).
    """
    from PyHydroGeophysX.data_processing.ert_io import load_ert_container

    data = load_ert_container(str(data_path), instrument=instrument, log=log)
    if not data.haveData("k"):
        data["k"] = ert.createGeometricFactors(data, numerical=False)
    if not data.haveData("rhoa"):
        if data.haveData("r"):
            data["rhoa"] = data["r"] * data["k"]
        elif data.haveData("u") and data.haveData("i"):
            data["rhoa"] = data["u"] / data["i"] * data["k"]

    source = str(error_source).lower()
    if source not in ERROR_SOURCES:
        source = "file"
    estimated = _estimate_errors(
        data, relative_error=relative_error, absolute_error=absolute_error
    )
    from_file = None
    if data.haveData("err"):
        candidate = np.asarray(data["err"], dtype=float)
        if np.any(np.isfinite(candidate) & (candidate > 0)):
            from_file = candidate

    model = dict(error_model or {})
    if source == "reciprocal":
        if "m" in model and "b" in model:
            from PyHydroGeophysX.data_processing.ert_io import (
                container_resistance,
                reciprocal_model_errors,
            )

            resistance = container_resistance(data)
            # A reading the model cannot place (no resistance) is estimated,
            # by the NaN fill below.
            errors = reciprocal_model_errors(
                resistance if resistance is not None else np.full(int(data.size()), np.nan),
                float(model["m"]), float(model["b"]), float(model.get("floor") or 0.0))
            used = (f"reciprocal error model dR = 10^{float(model['b']):.4f} * "
                    f"R^{float(model['m']):.4f}, at least "
                    f"{100.0 * float(model.get('floor') or 0.0):g} %")
        else:
            errors = estimated
            used = "estimate (no reciprocal error model was given)"
    elif source == "stack":
        # Independent errors add in quadrature: the reading's own scatter, and
        # the estimate standing for everything the scatter cannot see.
        if data.haveData("stack"):
            spread = np.asarray(data["stack"], dtype=float)
            spread = np.where(np.isfinite(spread) & (spread > 0), spread, 0.0)
            errors = np.sqrt(estimated ** 2 + spread ** 2)
            used = "stacking spread with the estimate, in quadrature"
        else:
            errors = estimated
            used = "estimate (the data record no stacking spread)"
    elif source == "estimate" or from_file is None:
        errors = estimated
        used = "estimate" if source != "file" or from_file is None else source
        if source == "file" and from_file is None:
            used = "estimate (file had no usable err column)"
    elif source == "max":
        errors = np.maximum(from_file, estimated)
        used = "max(file, estimate)"
    else:
        # Keep the file's numbers where they are usable, estimate the rest.
        errors = np.where(np.isfinite(from_file) & (from_file > 0), from_file, estimated)
        used = "file"

    errors = np.asarray(errors, dtype=float)
    errors[~np.isfinite(errors)] = float(relative_error)
    errors = np.clip(errors, float(error_floor), 1.0)
    data["err"] = errors

    info = {
        "source": used,
        "mean": float(errors.mean()),
        "min": float(errors.min()),
        "max": float(errors.max()),
        "file_mean": float(np.mean(from_file)) if from_file is not None else None,
        "estimate_mean": float(np.mean(estimated)),
    }
    if source == "reciprocal" and "m" in model and "b" in model:
        info["median"] = float(np.median(errors))
        info["model"] = {key: model[key] for key in ("m", "b", "floor", "r2", "r2_raw",
                                                     "pairs", "fitted_over") if key in model}
    log(f"  error model: {used}, mean {info['mean'] * 100:.2f} %")
    return data, info


#: Ratios a wrong geometric factor tends to land on, and what they usually mean.
_K_SUSPECTS = (
    (2.0, "the half-space (2*pi) and full-space (4*pi) conventions are mixed; "
          "check whether the electrodes are being treated as buried"),
    (0.5, "the half-space (2*pi) and full-space (4*pi) conventions are mixed; "
          "check whether the electrodes are being treated as buried"),
    (2.0 * np.pi, "a factor of 2*pi is missing from the geometric factors"),
    (1.0 / (2.0 * np.pi), "an extra factor of 2*pi is in the geometric factors"),
)


def validate_geometric_factors(container, mesh, *, rho0: float = 100.0,
                               tolerance: float = 0.05,
                               log: Callable[[str], None] = _noop_log) -> Dict[str, Any]:
    """Check ``k`` by forward-modelling a homogeneous half space.

    A homogeneous model of resistivity ``rho0`` must return ``rho0`` as the
    apparent resistivity of every configuration. The forward response is
    ``(U/I) * k``, so the returned ratio is exactly ``k / k_true`` for the
    geometry actually being modelled, and it is independent of the field data.

    This catches what chi2 cannot. A geometric factor that is uniformly wrong by
    a factor X forces the inversion to scale the model by 1/X to fit the same
    apparent resistivities, so the section is wrong by X while the fit looks
    perfect. Topography, wrong electrode spacing, and half-space versus
    full-space convention errors all show up here.

    Returns the ratio statistics, an ``ok`` flag, and a ``message`` naming the
    likely cause and the consequence.
    """
    info: Dict[str, Any] = {"checked": False, "ok": True, "message": "",
                            "rho0": float(rho0), "tolerance": float(tolerance)}
    try:
        fop = ert.ERTModelling()
        fop.setData(container)
        fop.setMesh(mesh)
        response = np.asarray(
            fop.response(pg.Vector(fop.paraDomain.cellCount(), float(rho0))),
            dtype=float,
        )
    except Exception as exc:  # noqa: BLE001 - the check must never break a run
        info["message"] = f"geometric-factor check skipped ({exc})"
        return info

    ratio = response / float(rho0)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        info["message"] = "geometric-factor check skipped (no usable response)"
        return info

    median = float(np.median(ratio))
    deviation = np.abs(ratio - 1.0)
    info.update({
        "checked": True,
        "ratio_median": median,
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "deviation_median": float(np.median(deviation)),
        "deviation_max": float(deviation.max()),
        "n_beyond_tolerance": int((deviation > float(tolerance)).sum()),
        "n_data": int(ratio.size),
    })

    scale_off = abs(median - 1.0) > float(tolerance)
    # A median near 1 with a wide spread is a geometry problem rather than a
    # convention problem, and it does not rescale the whole section.
    spread_off = float(np.median(np.abs(ratio / median - 1.0))) > float(tolerance)

    # The homogeneous run only tests k against the geometry. It cannot see a rhoa
    # that was built with a *different* k, which is just as damaging and just as
    # invisible to chi2, so compare against the instrument's own factors when the
    # loader kept them.
    provenance_off = False
    if container.haveData("k_file"):
        k_used = np.asarray(container["k"], dtype=float)
        k_file = np.asarray(container["k_file"], dtype=float)
        good = np.isfinite(k_file) & (np.abs(k_file) > 1e-12) & (np.abs(k_used) > 1e-12)
        if good.any():
            file_ratio = float(np.median(k_used[good] / k_file[good]))
            info["k_file_ratio"] = file_ratio
            provenance_off = abs(file_ratio - 1.0) > float(tolerance)

    info["ok"] = not (scale_off or spread_off or provenance_off)
    if info["ok"]:
        info["message"] = (
            f"geometric factors check out: a homogeneous {rho0:g} model returns "
            f"{median * 100:.1f} % of it (spread {info['deviation_max'] * 100:.1f} % max)")
        return info

    parts: List[str] = []
    if scale_off:
        cause = next((why for value, why in _K_SUSPECTS
                      if abs(median / value - 1.0) < 0.05), "")
        parts.append(
            f"a homogeneous {rho0:g} Ohm-m model returns {median * rho0:.1f} Ohm-m "
            f"instead, so k is off by a factor of {median:.3f}"
            + (f"; {cause}" if cause else ""))
        parts.append(
            f"the inversion will scale the model by {1.0 / median:.3f} to fit the "
            "same apparent resistivities, so the section is wrong by that factor "
            "while chi2 stays perfectly happy")
        info["suspected_factor"] = median
    if spread_off:
        parts.append(
            f"the response also varies by {info['deviation_max'] * 100:.0f} % across "
            "configurations, which points at electrode positions or topography "
            "rather than a single convention error")
    if provenance_off:
        ratio_file = info["k_file_ratio"]
        parts.append(
            f"the factors in the file differ from the ones computed here by "
            f"{ratio_file:.3f}, so the apparent resistivities were formed with a "
            "different convention than the forward run would use; the observation "
            "and the response would be measuring different things")
        info["suspected_factor"] = info.get("suspected_factor", ratio_file)
    info["message"] = "Geometric factors look wrong: " + ". ".join(parts) + "."
    log("  " + info["message"])
    return info


#: What to do about geometric factors before anything else touches the data.
#: ``fix`` validates and, if the check fails, recomputes k numerically on the
#: inversion mesh and rebuilds rhoa from the measured transfer resistance.
GEOMETRIC_FACTOR_POLICIES = ("off", "check", "fix")


def repair_geometric_factors(container, mesh,
                             log: Callable[[str], None] = _noop_log) -> Dict[str, Any]:
    """Recompute k numerically on the inversion mesh and rebuild rhoa from R.

    The transfer resistance R = rhoa/k is what the instrument actually measured;
    k and rhoa are both derived from it. So the repair keeps R fixed, takes k
    from a forward run on the mesh the inversion will use, and rebuilds
    ``rhoa = R * k``. Numerical factors also carry the topography that the
    analytic half-space formula cannot.

    R has to be recovered with the factors that *formed* rhoa, which is the
    file's own ``k_file`` when the loader kept it, and only otherwise the
    container's ``k``. Dividing by the wrong one would bake the discrepancy into
    R and leave the section scaled after the repair rather than before it.

    Modifies ``container`` in place and returns what changed.
    """
    k_old = np.asarray(container["k"], dtype=float)
    rhoa_old = np.asarray(container["rhoa"], dtype=float)
    if container.haveData("k_file"):
        k_source = np.asarray(container["k_file"], dtype=float)
        provenance = "file"
    else:
        k_source = k_old
        provenance = "container"
    with np.errstate(divide="ignore", invalid="ignore"):
        resistance = rhoa_old / np.where(np.abs(k_source) > 1e-12, k_source, np.nan)
    resistance = np.nan_to_num(resistance, nan=0.0, posinf=0.0, neginf=0.0)

    k_new = np.asarray(
        ert.createGeometricFactors(container, numerical=True, mesh=mesh), dtype=float
    )
    container["k"] = k_new
    container["r"] = resistance
    container["rhoa"] = resistance * k_new
    # k_file described the old apparent resistivities; keeping it would make the
    # provenance check fire again on data that is now self-consistent.
    if container.haveData("k_file"):
        container["k_file"] = k_new

    ratio = np.divide(k_new, k_source, out=np.ones_like(k_new),
                      where=np.abs(k_source) > 1e-12)
    return {
        "resistance_from": provenance,
        "k_median_before": float(np.median(k_source)),
        "k_median_after": float(np.median(k_new)),
        "k_ratio_median": float(np.median(ratio)),
        "rhoa_median_before": float(np.median(rhoa_old)),
        "rhoa_median_after": float(np.median(np.asarray(container["rhoa"], dtype=float))),
        "sign_flips": int(np.sum(np.sign(k_new) != np.sign(k_old))),
    }


def ensure_geometric_factors(container, mesh, *, policy: str = "fix",
                             tolerance: float = 0.05, rho0: float = 100.0,
                             log: Callable[[str], None] = _noop_log) -> Dict[str, Any]:
    """Validate k before the data are used, and repair it when asked.

    A repair is kept only if it actually makes the check pass; otherwise the
    original factors are restored, because a failed repair is worse than a
    reported problem. Either way the caller is told what happened.
    """
    policy = str(policy).lower()
    if policy not in GEOMETRIC_FACTOR_POLICIES:
        policy = "fix"
    if policy == "off":
        return {"checked": False, "ok": True, "policy": policy, "repaired": False,
                "message": "geometric-factor check disabled"}

    info = validate_geometric_factors(container, mesh, rho0=rho0,
                                      tolerance=tolerance, log=log)
    info["policy"] = policy
    info["repaired"] = False
    if info["ok"] or not info.get("checked") or policy == "check":
        return info

    log("  recomputing the geometric factors numerically on the inversion mesh…")
    before = {"k": np.asarray(container["k"], dtype=float).copy(),
              "rhoa": np.asarray(container["rhoa"], dtype=float).copy(),
              "r": np.asarray(container["r"], dtype=float).copy()
              if container.haveData("r") else None,
              "k_file": np.asarray(container["k_file"], dtype=float).copy()
              if container.haveData("k_file") else None}
    try:
        change = repair_geometric_factors(container, mesh, log=log)
    except Exception as exc:  # noqa: BLE001 - never let the repair break the run
        info["repair_error"] = str(exc)
        info["message"] += f" The repair could not run ({exc})."
        return info

    after = validate_geometric_factors(container, mesh, rho0=rho0,
                                       tolerance=tolerance, log=_noop_log)
    if not after.get("ok"):
        container["k"] = before["k"]
        container["rhoa"] = before["rhoa"]
        if before["r"] is not None:
            container["r"] = before["r"]
        if before["k_file"] is not None:
            container["k_file"] = before["k_file"]
        info["repair_failed"] = True
        info["message"] += (
            " Recomputing them numerically did not fix it either, so the original "
            "factors were kept; treat the resistivity scale as unverified.")
        log("  the recomputed factors did not pass either; the originals were restored")
        return info

    info["repaired"] = True
    info["repair"] = change
    info["ratio_after"] = after["ratio_median"]
    # The consequence has to come from the correction actually applied, not from
    # the homogeneous ratio: when the file's factors were the problem, that ratio
    # looks clean and would understate the damage.
    applied = float(change["k_ratio_median"])
    averted = 1.0 / applied if abs(applied) > 1e-12 else float("inf")
    info["averted_factor"] = averted
    info["message"] = (
        "Geometric factors disagreed with the geometry being modelled and have been "
        f"recomputed numerically on the inversion mesh: median k "
        f"{change['k_median_before']:.2f} -> {change['k_median_after']:.2f} "
        f"(x{applied:.3f}), with the apparent resistivities rebuilt from the measured "
        f"transfer resistance (median {change['rhoa_median_before']:.2f} -> "
        f"{change['rhoa_median_after']:.2f} Ohm-m). The homogeneous check now returns "
        f"{after['ratio_median']:.4f}. Left alone, the inverted section would have "
        f"been off by {averted:.2f}x with no sign of it in chi2.")
    log("  " + info["message"])
    if change["sign_flips"]:
        log(f"  note: {change['sign_flips']} geometric factor(s) changed sign")
    return info


def _gauss_newton_step(system, rhs, method: str, **solver_options):
    """Solve the stacked Gauss-Newton system ``system @ dm = rhs`` with any method.

    A least-squares method takes the stacked system ``[W_d J; sqrt(lambda) W_m]``
    as it is. An SPD method ('spd_cholesky', 'spd_cg') needs the square normal
    matrix, so it is given ``system^T system`` and ``system^T rhs``: the same
    step, in the form TimeLapseERTInversion solves. Handed the stacked system
    itself, the SPD methods refused it as not square.
    """
    if str(method).lower().strip() in _SPD_METHODS:
        rhs = np.asarray(rhs, dtype=float).reshape(-1, 1)
        if isinstance(system, StackedSystem):
            normal, normal_rhs = system.normal_matrix(), system.T @ rhs
        else:
            system = np.asarray(system, dtype=float)
            normal, normal_rhs = system.T @ system, system.T @ rhs
        # The normal matrix is built for this call alone, so the factorization
        # may work in its buffer.
        return generalized_solver(normal, normal_rhs, method=method,
                                  use_gpu=solver_options.get("use_gpu", False),
                                  overwrite_a=True)
    return generalized_solver(system, rhs, method=method, **solver_options)


def _weighted_residuals(response, data) -> np.ndarray:
    """Data misfit in units of the assumed error, in the log space both engines solve in."""
    observed = np.asarray(data["rhoa"], dtype=float)
    predicted = np.asarray(response, dtype=float)
    errors = np.asarray(data["err"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = (np.log(observed) - np.log(predicted)) / np.log1p(errors)
    return np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class ERTRun:
    """One inversion at one lambda, described the same way by either engine."""

    lam: float
    chi2: float
    iterations: int
    stop: str  # "target" | "plateau" | "iteration_cap"
    convergence: List[float]
    model: np.ndarray
    response: np.ndarray
    mesh: Any
    coverage: Optional[np.ndarray] = None
    manager: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def result(self):
        """A manager-like handle for the viewer and the VTK export."""
        if self.manager is not None:
            return self.manager
        return ModelResult(self.mesh, self.model, self.response, self.coverage)


class _PyHydroEngine:
    """The in-house Gauss-Newton ERT inversion (``ERTInversion``).

    Set up once per data container and re-run at different lambdas: ``setup()``
    builds the forward operator and the dense constraint matrix, which is the
    expensive part and does not depend on lambda.
    """

    name = "pyhydro"

    def __init__(self, container, mesh, *, model_constraints=(1e-2, 1e5),
                 method="cgls", zones=None, log=_noop_log):
        self.container = container
        self._inv = ERTInversion(
            container, mesh=mesh, model_constraints=model_constraints,
            method=method, verbose=False, zones=zones or None,
        )
        self._inv.setup()
        self.mesh = self._inv.fwd_operator.paraDomain
        self.zone_prior = self._inv.zone_prior()

    def reference_model(self):
        """The model a cold start regularizes against: homogeneous, or a priori.

        Warm-started stages must keep pulling toward this, not toward whichever
        model the previous stage ended on, or the penalty silently becomes
        roughness of the change instead of roughness of the model. With zones it
        is the a-priori model, as it is for the cold start.
        """
        background = float(np.median(np.exp(self._inv.rhos1)))
        if self.zone_prior is not None:
            return self.zone_prior.with_background(background)
        return np.full(int(self._inv.fwd_operator.paraDomain.cellCount()), background)

    def fit(self, *, lam, max_iterations, plateau_tolerance, target_chi2,
            start_model=None, reference_model=None) -> ERTRun:
        self._inv.parameters["lambda_val"] = float(lam)
        self._inv.parameters["max_iterations"] = int(max_iterations)
        self._inv.parameters["target_chi_squared"] = float(target_chi2)
        self._inv.parameters["convergence_tolerance"] = float(plateau_tolerance)
        res = self._inv.run(initial_model=start_model, reference_model=reference_model)
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
            coverage=None if res.coverage is None else np.asarray(res.coverage, dtype=float),
            metrics={
                "chi2": float(res.meta.get("chi2", float("nan"))),
                "lambda": float(lam),
                "iterations": int(res.meta.get("iterations", len(history))),
                "n_data": int(self.container.size()),
                "line_search_failures": int(res.meta.get("line_search_failures", 0)),
            },
        )


class _PygimliEngine:
    """PyGIMLi's ``ERTManager``, kept as an alternative to the in-house solver."""

    name = "pygimli"

    def __init__(self, container, mesh, *, model_constraints=None, method=None,
                 zones=None, log=_noop_log):
        self.container = container
        self.mesh = mesh
        # Zones reach PyGIMLi as its starting model; it has no per-cell fix.
        self.zone_prior = None
        self._start = None
        if zones:
            from .ert_zones import zone_prior

            fop = ert.ERTModelling()
            fop.setData(container)
            fop.setMesh(mesh)
            self.zone_prior = zone_prior(fop.paraDomain, zones, bounds=model_constraints)
            rhoa = np.asarray(container["rhoa"], dtype=float)
            background = float(np.median(rhoa[np.isfinite(rhoa) & (rhoa > 0)]))
            self._start = self.zone_prior.with_background(background)

    def reference_model(self):
        """PyGIMLi's smoothness constraint acts on the model itself, so there is
        no separate reference to pin; ``startModel`` alone is the warm start."""
        return None

    def fit(self, *, lam, max_iterations, plateau_tolerance, target_chi2,
            start_model=None, reference_model=None) -> ERTRun:
        manager = ert.ERTManager(self.container)
        kwargs: Dict[str, Any] = {"dPhi": float(plateau_tolerance) * 100.0}
        if start_model is None and self._start is not None:
            start_model = self._start
        if start_model is not None:
            kwargs["startModel"] = start_model
        manager.invert(self.container, mesh=self.mesh, lam=float(lam),
                       maxIter=int(max_iterations), verbose=False, **kwargs)
        metrics, history = metrics_from_manager(
            manager, n_data=int(self.container.size()), lam=float(lam)
        )
        # PyGIMLi records the starting model too, so a full budget is maxIter + 1.
        stop = "iteration_cap" if len(history) > int(max_iterations) else "plateau"
        chi2 = float(metrics.get("chi2", float("nan")))
        if chi2 == chi2 and chi2 < float(target_chi2):
            stop = "target"
        coverage = None
        try:
            coverage = np.asarray(manager.coverage(), dtype=float)
        except Exception:  # noqa: BLE001 - coverage is optional
            coverage = None
        return ERTRun(
            lam=float(lam), chi2=chi2, iterations=max(len(history) - 1, 0), stop=stop,
            convergence=[float(h) for h in history],
            model=np.asarray(manager.model, dtype=float),
            response=np.asarray(manager.inv.response, dtype=float),
            mesh=manager.paraDomain, coverage=coverage, manager=manager,
            metrics=dict(metrics),
        )


_ADTLERT_SOLVERS: Dict[str, str] = {
    "cgls": "pyhydro_cgls",
    "pyhydro_cgls": "pyhydro_cgls",
    "lsqr": "lsqr",
    "normal_cg": "normal_cg",
    "gpu_cgls": "gpu_cgls",
}

# Studio's plateau tolerance measures relative chi-squared improvement.
# ADTLERT's step_tolerance instead measures the RMS update in log-model space;
# treating the former as the latter made the default 0.5% stop ADTLERT after a
# single small model step. Keep ADTLERT's own conservative numerical threshold
# and let _fit_to_plateau apply the chi-squared criterion between runs.
_ADTLERT_STEP_TOLERANCE = 1.0e-4

def _adtlert_solver_name(method: str, *, prefer_gpu: bool = False) -> str:
    if prefer_gpu and str(method).lower() == "cgls":
        try:
            import cupy as cp

            if int(cp.cuda.runtime.getDeviceCount()) > 0:
                return "gpu_cgls"
        except Exception:  # noqa: BLE001 - CPU fallback is intentional
            pass
    solver = _ADTLERT_SOLVERS.get(str(method).lower())
    if solver is None:
        choices = ", ".join(sorted(_ADTLERT_SOLVERS))
        raise ValueError(
            f"Unknown ADTLERT linearized solver {method!r}; "
            f"choose one of {choices}."
        )
    return solver


def _enable_adtlert_float64() -> None:
    """Put ADTLERT's runtime in float64 before anything imports it.

    Float32 is not a speed/accuracy trade here, it simply does not work: the
    2.5D solve returns non-finite total fields and the forward raises
    ``FloatingPointError``. ADTLERT reads this at import time, so it has to be
    set before the first ``import adtlert``, and it is assigned rather than
    defaulted because a stale ``0`` in the environment would otherwise break
    the backend with no way to tell why.
    """
    os.environ["ADTLERT_ENABLE_FLOAT64"] = "1"


def _adtlert_cuda_available() -> bool:
    """Return whether ADTLERT's Torch/CuPy CUDA 12 path is usable."""
    _enable_adtlert_float64()
    try:
        import adtlert  # noqa: F401
        import cupy as cp
        import torch

        if not torch.cuda.is_available():
            return False
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return False
        # Importing CuPy and enumerating devices do not prove that NVRTC and
        # the CUDA headers are usable. Compile one tiny kernel so incomplete
        # Windows installations fall back before an inversion starts.
        probe = cp.arange(1, dtype=cp.float32)
        return float(cp.sum(probe).get()) == 0.0
    except Exception:  # noqa: BLE001 - the original ERT engine is the fallback
        return False


def _adtlert_cudss_available() -> bool:
    """Return whether the cuDSS GPU forward solver is importable and usable."""
    if not _adtlert_cuda_available():
        return False
    try:
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401
    except Exception:  # noqa: BLE001 - the original ERT engine is the fallback
        return False
    return True


def _adtlert_forward_solver_backend() -> str:
    """Require cuDSS so ADTLERT never silently uses its slow SciPy solver."""
    if not _adtlert_cudss_available():
        raise BackendUnavailable(
            "ADTLERT requires its cuDSS CUDA 12 forward solver. Install "
            "`pyhydrogeophysx[adtlert]` or use engine='pyhydro'."
        )
    return "cudss"


def _resolve_ert_engine(name: str, *, log=_noop_log) -> str:
    """Use ADTLERT only with CUDA and cuDSS; otherwise retain PyHydro ERT."""
    requested = str(name).lower()
    if requested == "adtlert" and not _adtlert_cudss_available():
        log(
            "ADTLERT CUDA 12/cuDSS is unavailable; falling back to the "
            "original PyHydro ERT engine."
        )
        return "pyhydro"
    return requested


def _adtlert_survey_supported(container) -> bool:
    """Return whether ADTLERT 0.1 can represent every ABMN electrode."""
    return all(
        not np.any(np.asarray(container[field], dtype=int) < 0)
        for field in ("a", "b", "m", "n")
    )


def _canonicalize_adtlert_polarity(container, *, log=_noop_log):
    """Return an equivalent positive-geometric-factor survey for ADTLERT.

    ADTLERT's fast normal-sensitivity path expects the receiver polarity that
    gives a positive geometric factor.  A negative factor is physically valid,
    so canonicalize only the ADTLERT survey copy by exchanging M and N for the
    affected rows.  Apparent resistivity and errors keep their original order
    and values; the caller's container is never modified.
    """
    n_data = int(container.size())
    geometric_factors = np.asarray(container["k"], dtype=float).reshape(-1)
    if geometric_factors.shape != (n_data,):
        raise ValueError("ADTLERT requires one geometric factor per datum")

    invalid = ~np.isfinite(geometric_factors) | (geometric_factors == 0.0)
    if np.any(invalid):
        count = int(np.count_nonzero(invalid))
        raise ValueError(
            "ADTLERT requires finite, non-zero geometric factors; "
            f"found {count} invalid value(s)"
        )

    negative = geometric_factors < 0.0
    if not np.any(negative):
        return container

    normalized = pg.DataContainerERT(container)
    receiver_m = np.asarray(container["m"], dtype=np.int64).copy()
    receiver_n = np.asarray(container["n"], dtype=np.int64).copy()
    normalized_m = receiver_m.copy()
    normalized_n = receiver_n.copy()
    normalized_m[negative] = receiver_n[negative]
    normalized_n[negative] = receiver_m[negative]
    normalized["m"] = normalized_m
    normalized["n"] = normalized_n

    normalized_k = geometric_factors.copy()
    normalized_k[negative] *= -1.0
    normalized["k"] = normalized_k
    normalized["r"] = np.asarray(container["rhoa"], dtype=float) / normalized_k

    count = int(np.count_nonzero(negative))
    log(
        "  ADTLERT polarity normalization: swapped M/N for "
        f"{count}/{n_data} negative-k measurements; rhoa and err unchanged"
    )
    return normalized


def _build_adtlert_forward(container, mesh, *, log=_noop_log, field_cache_entries=None):
    """Build one shared ADTLERT forward context for single or windowed ERT.

    ``field_cache_entries`` bounds the forward's cache of solved fields, which
    keeps the total fields of the last 8 models by default - one wavenumber x
    electrode x node array per model, 130 MB on a 7473-cell mesh with 56
    electrodes, and a GPU copy of it besides. ``None`` keeps ADTLERT's default.
    """
    _enable_adtlert_float64()
    try:
        import adtlert
        from adtlert.forward import ERTForward2p5D, mesh_to_adtlert, survey_to_adtlert
        from adtlert.inversion import ParameterizedERTForward2p5D
    except ImportError as exc:
        raise BackendUnavailable(
            "The ADTLERT ERT backend is unavailable. Install it with "
            "`pip install \"pyhydrogeophysx[adtlert]\"`."
        ) from exc

    if int(mesh.dim()) != 2:
        raise ValueError(
            "engine='adtlert' currently supports 2D profile meshes only; "
            "use engine='pygimli' for this mesh"
        )

    active_ids = np.asarray(
        [
            int(cell.id())
            for cell in mesh.cells()
            if int(cell.marker()) > 1
        ],
        dtype=np.int32,
    )
    if active_ids.size == 0:
        raise ValueError(
            "the ADTLERT parameter domain has no cells with marker > 1"
        )

    # createMeshByCellIdx preserves the full-mesh order used by active_ids.
    # That same order is used for ADTLERT models and the existing result viewers.
    result_mesh = mesh.createMeshByCellIdx(pg.IVector(active_ids.tolist()))
    forward_mesh = mesh_to_adtlert(mesh)
    parameter_mesh = mesh_to_adtlert(result_mesh)
    adtlert_container = _canonicalize_adtlert_polarity(container, log=log)
    survey = survey_to_adtlert(adtlert_container, dimension=2)
    geometric_mode = (
        "analytic" if bool(forward_mesh.is_flat_surface) else "numerical"
    )
    forward_solver = _adtlert_forward_solver_backend()
    cache_options = {}
    if field_cache_entries is not None and "normal_field_cache_max_entries" in (
            inspect.signature(ERTForward2p5D.from_mesh_survey).parameters):
        cache_options["normal_field_cache_max_entries"] = int(field_cache_entries)
    forward = ParameterizedERTForward2p5D.from_mesh_survey(
        forward_mesh,
        survey,
        active_ids,
        regularization_mesh=parameter_mesh,
        background_mode="pygimli_prolongation",
        topographic_geometric_factor_mode=geometric_mode,
        linear_solver_backend=forward_solver,
        **cache_options,
    )
    # Zone outlines the smoothness must not cross reach ADTLERT the way it
    # takes known structure: one unit label per parameter cell, with no
    # constraint between cells of different units (see _adtlert_spatial_term).
    units = _interface_regions(mesh, active_ids)
    if units is not None:
        forward.structural_prior_cell_ids = units
        forward.structural_cross_weight = 0.0
    released = _release_adtlert_terrain_setup(getattr(forward, "forward_operator", forward))
    version = str(getattr(adtlert, "__version__", ""))
    log(
        f"  ADTLERT {version or '(unknown version)'}: "
        f"{mesh.cellCount()} forward cells, {active_ids.size} parameters, "
        f"{forward_solver} forward solver"
        + ("" if units is None else
           f"; smoothness cut along the zone outlines ({int(units.max()) + 1} parts)")
        + ("; terrain setup meshes released after use" if released else "")
    )
    return forward, result_mesh, active_ids, version


#: What ADTLERT's 2.5D forward builds over a surface with topography to set
#: itself up, and the results that make each one unnecessary afterwards: the
#: setup-only discretization, its intermediate potentials in _derived_cache,
#: the prefix of its solver state in _cudss_state, and the cached result.
_ADTLERT_TERRAIN_SETUP = (
    # Unit primary potentials, solved on the P2 mesh and kept on the solve mesh.
    ("primary_potential_discretization", "primary_potential_sub_potentials",
     "primary_potential_", "auxiliary_sub_potentials", "_auxiliary_sub_potential_stack"),
    # Numerical geometric factors.
    ("geometric_auxiliary_discretization", "auxiliary_geometric_sub_potentials",
     "auxiliary_geometric_", "geometric_factors", "_geometric_factors"),
)


def _release_adtlert_terrain_setup(forward) -> bool:
    """Compute what ADTLERT's forward derives from topography, then drop its scaffolding.

    Over a surface with topography the forward solves the unit primary
    potentials and the numerical geometric factors on a P2 refinement of the
    mesh, once, and keeps the results; every later solve reads those results.
    The two P2 discretizations, their potentials and their cuDSS factorizations
    stayed alive regardless: 1.05 GB of the 3.54 GB the forward held on the
    DAS-1 example (5994 cells), whose solves and Jacobians are the same to
    1e-15 without them. The results are computed here - the first solve's work
    moved, not added: 19 s to build and solve once either way - and each piece
    is released only once the result that replaces it is cached; anything this
    ADTLERT does differently is left alone. Returns whether anything was
    released.

    This reaches into ADTLERT's private state, which offers no public way to
    let go of it; ``adtlert<0.2`` is the version it was written against.
    """
    derived = getattr(forward, "_derived_cache", None)
    solver_state = getattr(forward, "_cudss_state", None)
    if not isinstance(derived, dict) or not isinstance(solver_state, dict):
        return False
    released = False
    for field_name, potentials, prefix, result, compute in _ADTLERT_TERRAIN_SETUP:
        discretization = getattr(forward, field_name, None)
        if discretization is None or not callable(getattr(forward, compute, None)):
            continue
        getattr(forward, compute)()
        if result not in derived:
            continue
        object.__setattr__(forward, field_name, None)  # a frozen dataclass field
        if getattr(forward, "auxiliary_discretization", None) is discretization:
            # Its constructor's fallback when there is no geometric discretization.
            object.__setattr__(forward, "auxiliary_discretization",
                               getattr(forward, "primary_auxiliary_discretization", None))
        derived.pop(potentials, None)
        for key in [key for key in solver_state if key.startswith(prefix)]:
            entry = solver_state.pop(key)
            if key.endswith("_solver_batch_gpu") or key.endswith("_solver_gpu"):
                try:
                    entry.free()  # as the forward's own close() releases a solver
                except Exception:  # noqa: BLE001 - freeing is best effort
                    pass
        released = True
    if released:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - no CUDA cache to empty
            pass
    return released


def _adtlert_spatial_term(name: str, forward):
    """ADTLERT's spatial regularization ``name``, cut at the zone outlines if marked.

    When :func:`_build_adtlert_forward` found zone edges the smoothness must
    not cross, the first-order term becomes ADTLERT's structure-guided one with
    no weight across units, and a robust (L1, Huber) term keeps its own IRLS
    weighting on that same cut operator. Otherwise ``name`` is returned as is.
    """
    if getattr(forward, "structural_prior_cell_ids", None) is None:
        return name
    from adtlert.inversion.regularization import build_spatial_regularization

    term = build_spatial_regularization(name)
    if term.name == "first_order":
        return "structural_prior"
    if term.name in ("first_order_tv", "first_order_huber"):
        return _cut_robust_term(type(term))(
            **{item.name: getattr(term, item.name) for item in fields(term)})
    raise ValueError(f"ADTLERT's {term.name!r} regularization cannot be cut at "
                     "zone outlines; use first-order smoothness, L1 or Huber.")


@lru_cache(maxsize=None)
def _cut_robust_term(base):
    """``base`` (ADTLERT's TV or Huber term) on the structure-guided operator."""
    from adtlert.inversion.regularization import StructuralPriorSpatialRegularization

    class _Cut(base):
        def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0):
            return StructuralPriorSpatialRegularization().matrix(
                forward, n_cells, z_weight=z_weight)

    _Cut.__name__ = _Cut.__qualname__ = f"ZoneCut{base.__name__}"
    return _Cut


class _ADTLertEngine:
    """ADTLERT's differentiable 2.5D inversion behind the common ERT contract.

    PyGIMLi remains responsible for file loading, QC and mesh generation.  The
    numerical solve is delegated to ADTLERT, with the parameter-domain cell
    ordering preserved so the existing viewers and exporters keep working.
    """

    name = "adtlert"

    def __init__(self, container, mesh, *, model_constraints=(1e-2, 1e5),
                 method="cgls", zones=None, log=_noop_log):
        self._forward, self.mesh, active_ids, self._adtlert_version = (
            _build_adtlert_forward(container, mesh, log=log)
        )
        self.container = container
        self._observed = np.asarray(container["rhoa"], dtype=float)
        self._errors = np.log1p(np.asarray(container["err"], dtype=float))
        self._initial_model = np.full(
            active_ids.size, float(np.median(self._observed)), dtype=float
        )
        # Zones become the starting and reference model; ADTLERT inverts every
        # cell, so a fixed zone is not held (the run log says so).
        self.zone_prior = None
        if zones:
            from .ert_zones import zone_prior

            self.zone_prior = zone_prior(self.mesh, zones, bounds=model_constraints)
            self._initial_model = self.zone_prior.with_background(
                float(np.median(self._observed)))
        self._model_constraints = tuple(
            float(value) for value in model_constraints
        )
        self._solver = _adtlert_solver_name(method, prefer_gpu=True)
        log(
            "  ADTLERT paper configuration: "
            "normal_sensitivity=True, "
            "include_robin_boundary_derivative=False, "
            f"linearized_solver={self._solver}, "
            f"step_tolerance={_ADTLERT_STEP_TOLERANCE:g}"
        )

    def reference_model(self):
        return self._initial_model.copy()

    def fit(self, *, lam, max_iterations, plateau_tolerance, target_chi2,
            start_model=None, reference_model=None) -> ERTRun:
        from adtlert.inversion import (
            InversionConfig,
            invert_single_log_resistivity,
        )

        initial = (
            self._initial_model
            if start_model is None
            else np.asarray(start_model, dtype=float)
        )
        # The shared lambda-search contract uses 0.0 to disable target-based
        # stopping while a trial is run to its plateau. ADTLERT represents the
        # same state with None and deliberately rejects non-positive values.
        requested_target = float(target_chi2)
        adtlert_target = requested_target if requested_target > 0.0 else None
        config = InversionConfig(
            max_iterations=int(max_iterations),
            data_std=self._errors,
            regularization=float(lam),
            spatial_regularization=_adtlert_spatial_term("first_order", self._forward),
            model_bounds=self._model_constraints,
            target_chi2=adtlert_target,
            # ``plateau_tolerance`` is a chi-squared improvement threshold in
            # our public API, not ADTLERT's log-model update norm.
            step_tolerance=_ADTLERT_STEP_TOLERANCE,
            linearized_solver=self._solver,
            normal_sensitivity=True,
            include_robin_boundary_derivative=False,
            max_log_step=1.0,
            line_search=True,
        )
        inverted = invert_single_log_resistivity(
            self._forward,
            self._observed,
            initial,
            reference_model=reference_model,
            config=config,
        )
        history = [float(value) for value in inverted.iteration_chi2]
        chi2 = history[-1] if history else float("nan")
        iterations = len(history)
        if (adtlert_target is not None and np.isfinite(chi2)
                and chi2 < adtlert_target):
            stop = "target"
        elif iterations >= int(max_iterations):
            stop = "iteration_cap"
        else:
            stop = "plateau"
        return ERTRun(
            lam=float(lam),
            chi2=float(chi2),
            iterations=iterations,
            stop=stop,
            convergence=history,
            model=np.asarray(inverted.final_model, dtype=float),
            response=np.asarray(inverted.predicted_data, dtype=float),
            mesh=self.mesh,
            coverage=np.asarray(inverted.coverage, dtype=float),
            metrics={
                "backend": "adtlert",
                "backend_version": self._adtlert_version,
                "chi2": float(chi2),
                "lambda": float(lam),
                "iterations": iterations,
                "n_data": int(self.container.size()),
                "sensitivity_profile": "paper",
                "normal_sensitivity": True,
                "include_robin_boundary_derivative": False,
                "linearized_solver": self._solver,
            },
        )


def _E4DEngine(container, mesh, **kwargs):
    """E4D, PNNL's 3-D code, run as an external program (see ``inversion.e4d``)."""
    from .e4d import E4DEngine

    return E4DEngine(container, mesh, **kwargs)


def _R2Engine(container, mesh, **kwargs):
    """Binley's R2, 2-D, run as an external program (see ``inversion.r2``)."""
    from .r2 import R2Engine

    return R2Engine(container, mesh, program="r2", **kwargs)


def _R3tEngine(container, mesh, **kwargs):
    """Binley's R3t, 3-D, run as an external program (see ``inversion.r2``)."""
    from .r2 import R2Engine

    return R2Engine(container, mesh, program="r3t", **kwargs)


ENGINES = {
    "pyhydro": _PyHydroEngine,
    "pygimli": _PygimliEngine,
    "adtlert": _ADTLertEngine,
    "e4d": _E4DEngine,
    "r2": _R2Engine,
    "r3t": _R3tEngine,
}

#: Engines that run an external program on a mesh of their own making, kept
#: for the whole pipeline, with settings of their own.
_EXTERNAL_ENGINES = ("e4d", "r2", "r3t")

#: Engines that choose their regularization weight themselves at every
#: iteration (Occam), so a lambda and the lambda search mean nothing to them.
_SELF_REGULARIZING = ("r2", "r3t")

#: How the external programs are named in logs and reports.
_PROGRAM_NAMES = {"e4d": "E4D", "r2": "R2", "r3t": "R3t"}


def _make_engine(name: str, container, mesh, **kwargs):
    resolved = _resolve_ert_engine(
        name, log=kwargs.get("log", _noop_log)
    )
    factory = ENGINES.get(resolved)
    if factory is None:
        raise ValueError(
            f"Unknown ERT engine {name!r}; choose one of {sorted(ENGINES)}."
        )
    return factory(container, mesh, **kwargs)


def _diverged(run) -> bool:
    """Did this run end worse than it started?

    ``stop`` can otherwise only say target / iteration_cap / plateau, so a
    misfit that climbs every iteration is reported as a plateau and the railed
    model it produced is handed back as a result. Comparing the ends of the
    convergence history is enough to tell the difference, and it costs nothing.
    """
    history = [float(value) for value in (getattr(run, "convergence", None) or [])
               if value == value]
    return len(history) >= 2 and history[-1] > history[0]


def _fit_to_plateau(engine, *, lam: float, max_iterations: int,
                    plateau_tolerance: float, target_chi2: float,
                    max_total_iterations: int, start_model=None,
                    reference_model=None, stop_on_target: bool = True,
                    log: Callable[[str], None] = _noop_log) -> ERTRun:
    """Iterate at a fixed lambda until the misfit stops improving.

    A run that exhausts its iteration budget is continued from its own model,
    with the regularization reference pinned to the original one, until it
    plateaus or the total-iteration ceiling is reached. Only a plateaued chi2 is
    attributable to lambda, so the lambda search must not judge a lambda before
    this returns.

    ``start_model`` warm-starts the stage from a neighbouring lambda's solution.
    ``reference_model`` must then be the homogeneous model the cold start would
    have used, so the penalty stays on model roughness.

    ``stop_on_target`` must be False for a lambda trial. Every engine stops as
    soon as chi2 falls below ``target_chi2``, and that test is made before the
    new lambda has moved anything. A trial warm-started from a model already
    under the target therefore returns on its first iteration with the seed's
    chi2, identically for every lambda, which makes an upward search report no
    improvement while the target was in fact reachable. chi2 is never negative,
    so 0.0 is a target the stopping rule can never meet.
    """
    engine_target = float(target_chi2) if stop_on_target else 0.0
    reference = reference_model
    run = engine.fit(lam=lam, max_iterations=max_iterations,
                     plateau_tolerance=plateau_tolerance, target_chi2=engine_target,
                     start_model=start_model, reference_model=reference)
    if _diverged(run):
        # Continuing a run that is climbing only buys a worse model, and the
        # caller needs to know the number it is being handed is not a fit.
        run.stop = "diverged"
        log(f"    lambda = {lam:g}: the misfit rose from "
            f"{run.convergence[0]:.3f} to {run.chi2:.3f}; this is not a "
            "converged result")
        return run
    while run.stop == "iteration_cap" and run.iterations < int(max_total_iterations):
        extra = min(int(max_iterations), int(max_total_iterations) - run.iterations)
        if extra <= 0:
            break
        log(f"    lambda = {lam:g}: continuing past {run.iterations} iteration(s) "
            f"(chi2 = {run.chi2:.3f}) for up to {extra} more")
        if reference is None:
            reference = engine.reference_model()
        previous, before = run, run.chi2
        nxt = engine.fit(lam=lam, max_iterations=extra,
                         plateau_tolerance=plateau_tolerance, target_chi2=engine_target,
                         start_model=run.model, reference_model=reference)
        nxt.convergence = list(run.convergence) + list(nxt.convergence[1:])
        nxt.iterations = run.iterations + nxt.iterations
        nxt.metrics["iterations"] = nxt.iterations
        # Hitting the iteration cap is not proof that the misfit is still
        # falling. A heavily over-regularized lambda can oscillate instead, and
        # continuing then spends the whole ceiling to end up no better.
        gained = before - nxt.chi2
        if not (before == before and nxt.chi2 == nxt.chi2
                and gained > abs(before) * float(plateau_tolerance)):
            # A continuation is allowed to stop improving. It is not allowed to
            # hand back something worse than what it was given, which is what a
            # diverging backend does on every extra block.
            worse = not (nxt.chi2 == nxt.chi2 and nxt.chi2 <= before)
            run = previous if worse else nxt
            run.convergence = list(nxt.convergence)
            run.stop = "stalled"
            log(f"    lambda = {lam:g}: no further improvement "
                f"({before:.3f} -> {nxt.chi2:.3f}); "
                + ("keeping the better earlier model" if worse
                   else "stopping here"))
            break
        run = nxt
    if run.stop == "iteration_cap":
        log(f"    lambda = {lam:g}: reached the {int(max_total_iterations)}-iteration "
            "ceiling while still improving, so its chi2 is an upper bound")
    return run


def _reject_outliers(engine_factory, container, run: ERTRun, *, threshold: float,
                     passes: int, min_fraction: float, lam: float,
                     max_iterations: int, plateau_tolerance: float,
                     target_chi2: float, max_total_iterations: int,
                     log: Callable[[str], None] = _noop_log):
    """Drop data the converged model cannot explain, then re-invert.

    Each pass removes measurements whose weighted residual exceeds ``threshold``
    and re-inverts at the same lambda. ``min_fraction`` bounds how much of the
    original dataset may be deleted; when more data exceed the threshold than
    that allows, the pass drops the worst offenders up to the limit rather than
    refusing outright. Cancelling the pass would leave the data untouched exactly
    when the fit is worst, which is when rejection is most wanted.

    Returns ``(container, engine, run, info)``.
    """
    n_start = int(container.size())
    floor = int(np.ceil(float(min_fraction) * n_start))
    info: Dict[str, Any] = {
        "enabled": True, "threshold": float(threshold), "n_start": n_start,
        "floor": floor, "passes": [], "dropped": 0, "kept": n_start,
        "stopped_because": "", "limited_by_floor": False,
    }
    engine = None
    for index in range(1, int(passes) + 1):
        allowed = int(container.size()) - floor
        if allowed <= 0:
            info["stopped_because"] = (
                f"at the {int(min_fraction * 100)} % floor of {floor} measurements")
            break
        residual = _weighted_residuals(run.response, container)
        drop = np.abs(residual) > float(threshold)
        n_drop = int(drop.sum())
        if n_drop == 0:
            info["stopped_because"] = "nothing left above the cut"
            break
        if n_drop > allowed:
            # Keep the cut but bound it: drop the worst `allowed` measurements.
            worst = np.argsort(-np.abs(residual))[:allowed]
            drop = np.zeros(int(container.size()), dtype=bool)
            drop[worst] = True
            n_drop = allowed
            info["limited_by_floor"] = True
            log(f"  more data exceed {threshold:g} sigma than the "
                f"{int(min_fraction * 100)} % floor allows; dropping the worst {n_drop}")
        trimmed = container.copy()
        # BVector must be built from the numpy bool array: handing it a Python
        # list of bools builds a mask that silently drops every measurement.
        trimmed.markInvalid(pg.core.BVector(drop))
        trimmed.removeInvalid()
        engine = engine_factory(trimmed)
        run = _fit_to_plateau(
            engine, lam=lam, max_iterations=max_iterations,
            plateau_tolerance=plateau_tolerance, target_chi2=target_chi2,
            max_total_iterations=max_total_iterations, log=log,
        )
        container = trimmed
        info["passes"].append({
            "pass": index, "dropped": n_drop, "kept": int(container.size()),
            "chi2": float(run.chi2),
            "convergence": [float(c) for c in run.convergence],
        })
        log(f"  rejected {n_drop}/{n_drop + int(container.size())} over "
            f"{threshold:g} sigma -> chi2 {run.chi2:.3f}")
        if info["limited_by_floor"]:
            # A bounded cut leaves known-bad data in, so stop rather than nibble
            # at the floor pass after pass without ever clearing the outliers.
            info["stopped_because"] = (
                f"the {int(min_fraction * 100)} % floor capped the cut; data beyond "
                f"{threshold:g} sigma remain")
            break
    info["kept"] = int(container.size())
    info["dropped"] = n_start - info["kept"]
    if not info["stopped_because"]:
        info["stopped_because"] = f"all {int(passes)} pass(es) used"
    return container, engine, run, info


def _export_model_vtk(manager, output_dir: str | Path, filename: str) -> str:
    """Write the manager's model to VTK, returning "" if the export is unavailable."""
    from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

    try:
        mesh = manager.paraDomain
        mesh["resistivity"] = np.asarray(manager.model, dtype=float)
        path = Path(output_dir) / filename
        # exportVTK opens a narrow path, which fails on any folder Windows' ANSI
        # codepage cannot represent; via_ascii_path stages the write when needed.
        via_ascii_path(mesh.exportVTK, path, mode="write")
        return str(path)
    except Exception:  # noqa: BLE001
        return ""


def _export_model_bundle(manager, output_dir: str | Path, stem: str) -> Dict[str, str]:
    """Persist the manager-shaped data needed by a viewer in another process."""
    from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

    out = Path(output_dir)
    paths = {
        "mesh": out / f"{stem}_mesh.bms",
        "model": out / f"{stem}_model.npy",
        "response": out / f"{stem}_response.npy",
        "coverage": out / f"{stem}_coverage.npy",
    }
    via_ascii_path(manager.paraDomain.save, paths["mesh"], mode="write")
    np.save(paths["model"], np.asarray(manager.model, dtype=float))
    response = getattr(manager, "response", None)
    if response is None:
        inverse = getattr(manager, "inv", None)
        response = getattr(inverse, "response", None)
    if response is not None:
        np.save(paths["response"], np.asarray(response, dtype=float))
    else:
        paths.pop("response")
    try:
        np.save(paths["coverage"], np.asarray(manager.coverage(), dtype=float))
    except Exception:  # noqa: BLE001 - coverage is optional for the viewer
        paths.pop("coverage")
    return {key: str(path) for key, path in paths.items()}


def _zone_notes(engine: str, prior, holds_fixed: bool = False) -> List[str]:
    """What a zone list will and will not do on ``engine``, for the run log.

    The in-house engine, R2 and R3t hold a fixed zone (``holds_fixed``); the
    others start from the a-priori values and invert every cell. A zone the
    user fixed that then moves must be said up front, not discovered in the
    result.
    """
    notes = []
    if prior is None:
        return notes
    if prior.report:
        notes.append("  a-priori zones: " + prior.summary())
    from .ert_zones import polygons_cover_nothing

    empty = polygons_cover_nothing(prior)
    if empty:
        notes.append(f"  (zone(s) {', '.join(empty)} cover no cell of the "
                     "parameter domain and have no effect)")
    if prior.any_fixed and engine != "pyhydro" and not holds_fixed:
        notes.append(f"  Note: the {engine} engine cannot hold a zone fixed, so the "
                     "fixed zones are used as starting values and inverted like "
                     "any other cell. Use the in-house Gauss-Newton engine, R2 or "
                     "R3t to keep them fixed.")
    return notes


def run_ert_manager_inversion(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    relative_error: float = 0.03,
    absolute_error: float = 0.0,
    error_source: str = "file",
    error_floor: float = 0.005,
    error_model: Optional[Dict[str, Any]] = None,
    mesh_quality: float = 34.0,
    para_depth: float = 0.0,
    para_max_cell_size: float = 0.0,
    para_boundary: float = 2.0,
    surface_nodes: int = 1,
    outer_width: float = 0.0,
    outer_max_cell_size: float = 0.0,
    mesh_file: str = "",
    zones: Optional[Sequence[Dict[str, Any]]] = None,
    conform_to_zones: bool = False,
    decouple_zones: bool = False,
    lam: float = 50.0,
    max_iterations: int = 20,
    plateau_tolerance: float = 0.005,
    max_total_iterations: int = 60,
    engine: str = "pyhydro",
    model_constraints: Tuple[float, float] = (1e-2, 1e5),
    solver: str = "cgls",
    geometric_factor_policy: str = "fix",
    geometric_factor_tolerance: float = 0.05,
    instrument: Optional[str] = None,
    reject_outliers: bool = False,
    outlier_threshold: float = 3.0,
    outlier_passes: int = 2,
    min_data_fraction: float = 0.5,
    auto_lambda: bool = False,
    target_chi2: float = 1.0,
    chi2_tolerance: float = 0.2,
    max_lambda_trials: int = 6,
    lambda_bounds: Tuple[float, float] = LAMBDA_BOUNDS,
    lambda_warm_start: bool = True,
    lambda_cold_retry_chi2: float = 15.0,
    e4d: Optional[Dict[str, Any]] = None,
    r2: Optional[Dict[str, Any]] = None,
    log: Callable[[str], None] = _noop_log,
) -> Dict[str, Any]:
    """Invert one ERT dataset, in the order that actually lowers chi2.

    The stages run back to back, each optional:

    0. **Geometric factors** (``geometric_factor_policy``). A homogeneous forward
       run must return the model resistivity; if it does not, ``k`` disagrees with
       the geometry being modelled and the whole section is scaled by that factor
       with no trace in chi2. ``fix`` recomputes ``k`` numerically on the inversion
       mesh and rebuilds ``rhoa`` from the measured transfer resistance.
    1. **Error model.** ``error_source`` decides whether the file's own ``err``
       column is trusted, recomputed from ``relative_error``/``absolute_error``,
       combined with that estimate, built from each reading's stacking spread,
       or given by a reciprocal error model (``error_model``; see
       :data:`ERROR_SOURCES`). Overwriting a measured error with an assumed
       one makes chi2 report on an error model the data never had.
    2. **Inversion at the requested lambda, iterated to a plateau.** A run that
       exhausts ``max_iterations`` is continued (up to ``max_total_iterations``)
       rather than being judged where it stopped. This run is always kept, under
       ``fixed_lambda``.
    3. **Outlier rejection** (``reject_outliers``). Measurements the converged
       model cannot explain are dropped and the inversion repeated, at most
       ``outlier_passes`` times and never below ``min_data_fraction`` of the data.
    4. **Lambda search** (``auto_lambda``). Only now, with a plateaued misfit on
       the cleaned data, is lambda allowed to move; every trial is itself
       iterated to a plateau before its chi2 counts. With ``lambda_warm_start``
       each trial continues from the nearest lambda already solved rather than
       restarting from a homogeneous model, which is why ``lam`` should start on
       the smooth side: the sweep is then a relaxation from an over-regularized
       model down to the data, the direction in which continuation is stable.

    ``engine`` selects the solver: ``"pyhydro"`` for the in-house Gauss-Newton
    inversion (``ERTInversion``), ``"pygimli"`` for ``ert.ERTManager``, or
    ``"adtlert"`` for the optional differentiable ADTLERT 2.5D backend when
    Torch, CuPy CUDA 12 and cuDSS are available. Linux and Windows both use
    cuDSS; ADTLERT's slower SciPy forward solver is intentionally disabled.
    Linux remains the recommended platform for the best performance. Without
    CUDA 12 or cuDSS, ``"adtlert"`` falls back to ``"pyhydro"``.

    ADTLERT matches the published real-data branch: fast normal sensitivity,
    no Robin-boundary derivative, line search, a maximum log step of one, and
    GPU CGLS when CUDA is available.

    ``"e4d"`` runs PNNL's E4D, an external 3-D program, with ``e4d`` holding
    its settings (:class:`~PyHydroGeophysX.inversion.e4d.E4DSettings`: the
    launcher, the program, the MPI processes). E4D runs on Linux, on Windows
    inside WSL 2, and on macOS when built from source; elsewhere its run
    folder is written under ``output_dir/e4d`` and the run stops with
    :class:`~PyHydroGeophysX.inversion.e4d.E4DNotRun` saying how to run it.
    It never falls back to another engine. A profile is inverted on a 3-D
    mesh built around the line and read back as the section along it.

    ``"r2"`` and ``"r3t"`` run Andrew Binley's R2 (2-D profiles) and R3t (3-D
    surveys, and profiles on a 3-D mesh around the line), with ``r2`` holding
    their settings (:class:`~PyHydroGeophysX.inversion.r2.R2Settings`: the
    program, the launcher). They are Windows programs, run natively there and
    through Wine on Linux and macOS, found in an installed ResIPy unless named;
    elsewhere their run folder is written under ``output_dir/r2`` (or
    ``r3t``) and the run stops with
    :class:`~PyHydroGeophysX.inversion.r2.R2NotRun`. Both search their
    smoothing weight at every iteration themselves, so ``lam`` and
    ``auto_lambda`` do not apply to them; ``max_iterations`` and
    ``target_chi2`` (as an RMS of ``sqrt(target_chi2)``) do, and they hold
    fixed zones fixed.

    ``zones`` carries a-priori resistivity zones (see
    :mod:`~PyHydroGeophysX.inversion.ert_zones`): polygons whose value the
    inversion starts from and regularizes toward, and which the in-house engine
    can hold fixed. The other engines use the values as the starting model
    only, and the log says so. With ``conform_to_zones`` a generated mesh
    follows the zone outlines, so no cell straddles one; with
    ``decouple_zones`` the smoothness constraint is dropped across them, so the
    model may jump at a zone's edge (every engine honours it).

    The mesh options (``mesh_quality`` to ``outer_max_cell_size``) are those of
    :func:`build_inversion_mesh`.
    """
    requested_engine = str(engine).lower()
    engine = _resolve_ert_engine(requested_engine, log=log)
    data, error_info = _prepare_ert_data(
        data_path, relative_error=relative_error, instrument=instrument, log=log,
        absolute_error=absolute_error, error_source=error_source,
        error_floor=error_floor, error_model=error_model,
    )
    if engine == "adtlert" and not _adtlert_survey_supported(data):
        log(
            "ADTLERT 0.1 cannot represent remote electrodes encoded with "
            "negative ABMN indices; falling back to the original PyHydro "
            "ERT engine."
        )
        engine = "pyhydro"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("Inverting ERT data…")
    from .ert_zones import normalize_zones

    zones = normalize_zones(zones)
    mesh_report: Dict[str, Any] = {}
    inversion_mesh = build_inversion_mesh(
        data, mesh_quality=mesh_quality, para_depth=para_depth,
        para_max_cell_size=para_max_cell_size, para_boundary=para_boundary,
        surface_nodes=surface_nodes, outer_width=outer_width,
        outer_max_cell_size=outer_max_cell_size,
        conform_zones=zones if conform_to_zones else None,
        mesh_file=mesh_file, log=log, report=mesh_report)
    decoupled_edges = 0
    if decouple_zones and zones:
        inversion_mesh, decoupled_edges = mark_zone_interfaces(inversion_mesh, zones)
        log(f"  smoothness dropped across {decoupled_edges} cell edges along the "
            "zone outlines" if decoupled_edges else
            "  (no cell edge lies on a zone outline, so the smoothness is unchanged)")

    # Before anything is fitted, confirm the geometric factors agree with the
    # geometry being modelled, and repair them if they do not. A uniform error
    # here rescales the whole section and leaves chi2 untouched, so it has to be
    # settled up front rather than inferred from the result.
    geometry_info = ensure_geometric_factors(
        data, inversion_mesh, policy=geometric_factor_policy,
        tolerance=geometric_factor_tolerance, log=log,
    )

    engine_options: Dict[str, Any] = {}
    if engine in _EXTERNAL_ENGINES:
        # One mesh and set of run folders for the whole pipeline, kept with
        # the results so a run can be inspected or repeated by hand.
        settings = dict((e4d if engine == "e4d" else r2) or {})
        settings.setdefault("workdir", str(out / engine))
        engine_options = {"options": settings, "outer_width": outer_width}
    external_engine = None

    def build_engine(container):
        nonlocal external_engine
        if external_engine is not None:
            # The external program's mesh does not depend on which data are
            # kept, so an outlier pass reuses it rather than meshing again.
            return external_engine.with_data(container)
        built = _make_engine(engine, container, inversion_mesh,
                             model_constraints=model_constraints, method=solver,
                             zones=zones, log=log, **engine_options)
        if engine in _EXTERNAL_ENGINES:
            external_engine = built
        return built

    active_engine = build_engine(data)
    zone_prior_used = getattr(active_engine, "zone_prior", None)
    holds_fixed = bool(getattr(active_engine, "holds_fixed_zones", engine == "pyhydro"))
    for line in _zone_notes(engine, zone_prior_used, holds_fixed):
        log(line)
    if engine in _SELF_REGULARIZING:
        log(f"  {_PROGRAM_NAMES[engine]} searches its smoothing weight at "
            "every iteration itself (an Occam inversion): the lambda setting does not apply, "
            "and the alpha it settles on is reported.")
    requested_lam = float(lam)
    target = float(target_chi2)
    tol = abs(float(chi2_tolerance))

    def to_plateau(eng, value):
        return _fit_to_plateau(
            eng, lam=value, max_iterations=max_iterations,
            plateau_tolerance=plateau_tolerance, target_chi2=target,
            max_total_iterations=max_total_iterations, log=log,
        )

    # -- stage 2: the requested lambda, iterated until the misfit flattens ----
    fixed_run = to_plateau(active_engine, requested_lam)
    log(f"  lam {requested_lam:g} -> chi2 {fixed_run.chi2:.3f} "
        f"({fixed_run.iterations} it, {fixed_run.stop})")

    result: Dict[str, Any] = {
        "engine": str(engine),
        "engine_requested": requested_engine,
        "adtlert_sensitivity_profile": (
            "paper" if engine == "adtlert" else None
        ),
        "geometric_factors": geometry_info,
        "data_error": error_info,
        "lambda_requested": requested_lam,
        "lambda_used": requested_lam,
        "auto_lambda_status": "off",
        "auto_lambda_note": "",
        "lambda_trials": [],
        "outliers": {"enabled": bool(reject_outliers)},
        "convergence_stop": fixed_run.stop,
        # Which cells each a-priori zone covered, and whether the engine held
        # the fixed ones: a zone that silently moved would look like a result.
        "zones": [dict(entry) for entry in getattr(zone_prior_used, "report", [])],
        "zones_fixed_held": bool(zone_prior_used is not None
                                 and zone_prior_used.any_fixed and holds_fixed),
        # The outline edges the mesh gained, and the cell edges the smoothness
        # no longer crosses: 0 when the option was off or found nothing to do.
        "mesh_zone_outline_edges": int(mesh_report.get("zone_outline_edges", 0)),
        "zone_edges_decoupled": int(decoupled_edges),
    }

    run = fixed_run
    container = data

    # -- stage 3: drop what the converged model cannot explain ---------------
    if reject_outliers:
        container, trimmed_engine, run, outlier_info = _reject_outliers(
            build_engine, data, run, threshold=outlier_threshold,
            passes=outlier_passes, min_fraction=min_data_fraction, lam=requested_lam,
            max_iterations=max_iterations, plateau_tolerance=plateau_tolerance,
            target_chi2=target, max_total_iterations=max_total_iterations, log=log,
        )
        result["outliers"] = outlier_info
        if trimmed_engine is not None:
            active_engine = trimmed_engine
        # The per-pass lines above already say what was dropped; no summary here.

    # -- stage 4: only now may lambda move -----------------------------------
    cleaned_run = run
    search: Optional[Dict[str, Any]] = None
    best: Dict[str, Any] = {}
    trial_detail: Dict[float, ERTRun] = {}
    # Chronological, unlike trial_detail: a lambda revisited by the cold retry
    # overwrites its dict entry, but both attempts belong in the history.
    trial_order: List[Tuple[float, ERTRun]] = []
    if not auto_lambda:
        pass
    elif engine in _SELF_REGULARIZING:
        result["auto_lambda_status"] = "not_applicable"
        result["auto_lambda_note"] = (
            f"Auto-lambda was not run: {_PROGRAM_NAMES[engine]} chooses its own smoothing "
            "weight at every iteration (it settled on alpha = "
            f"{cleaned_run.metrics.get(engine + '_alpha', float('nan')):.4g}), so there is "
            "no lambda to search.")
    elif cleaned_run.chi2 != cleaned_run.chi2:
        result["auto_lambda_status"] = "unavailable"
        result["auto_lambda_note"] = (
            "Auto-lambda was requested but the inversion reported no chi-squared, "
            f"so lambda stays at the requested {requested_lam:g}."
        )
    elif abs(cleaned_run.chi2 - target) <= tol:
        result["auto_lambda_status"] = "already_on_target"
        result["auto_lambda_note"] = (
            f"chi2 = {cleaned_run.chi2:.2f} at the requested lambda = "
            f"{requested_lam:g}, inside the target band {target:g} +/- {tol:g}. "
            "No lambda search needed."
        )
        result["lambda_trials"] = [
            {"lambda": requested_lam, "chi2": cleaned_run.chi2,
             "iterations": cleaned_run.iterations, "stop": cleaned_run.stop}
        ]
    else:
        log(f"  chi2 {cleaned_run.chi2:.2f} outside {target:g} +/- {tol:g}; "
            f"relaxing lambda (max {int(max_lambda_trials)} trials)")
        best.update(run=cleaned_run, chi2=cleaned_run.chi2, lam=requested_lam)
        # Warm start: each new lambda continues from the solution of the nearest
        # lambda already solved, instead of restarting from a homogeneous model.
        # The regularization reference stays pinned to that homogeneous model, so
        # the penalty is still on model roughness and the converged answer is the
        # same problem, just reached from closer.
        solved: Dict[float, np.ndarray] = {requested_lam: cleaned_run.model}
        pinned = active_engine.reference_model() if lambda_warm_start else None
        # A trial is kept for its chi2 and convergence; only the best one, the
        # cleaned and the fixed-lambda runs are ever exported. On the pyGIMLi
        # engine each run also holds its ERTManager - Jacobian, meshes and
        # forward operator - and every trial of a sweep used to stay alive,
        # thirteen with a cold retry. A run nothing can still select drops it;
        # its result is then the model, mesh and response it already carries.
        exported = {id(cleaned_run), id(fixed_run)}
        protected = set()

        def _release(candidate: Optional[ERTRun]) -> None:
            if candidate is not None and id(candidate) not in exported | protected:
                candidate.manager = None

        def _nearest_solved(trial_lam: float):
            if not lambda_warm_start or not solved:
                return None
            nearest = min(solved, key=lambda done: abs(np.log(done / trial_lam)))
            return solved[nearest], nearest

        def _evaluate(trial_lam: float) -> float:
            seed = _nearest_solved(float(trial_lam))
            start = seed[0] if seed else None
            trial = _fit_to_plateau(
                active_engine, lam=float(trial_lam), max_iterations=max_iterations,
                plateau_tolerance=plateau_tolerance, target_chi2=target,
                max_total_iterations=max_total_iterations,
                start_model=start, reference_model=pinned if start is not None else None,
                stop_on_target=False,
                log=log,
            )
            trial_detail[float(trial_lam)] = trial
            trial_order.append((float(trial_lam), trial))
            solved[float(trial_lam)] = trial.model
            origin = f" from {seed[1]:g}" if seed else " cold"
            log(f"  lam {trial_lam:g}{origin} -> chi2 {trial.chi2:.3f} "
                f"({trial.iterations} it)")
            if trial.chi2 == trial.chi2 and abs(trial.chi2 - target) < abs(best["chi2"] - target):
                _release(best.get("run"))
                best.update(run=trial, chi2=trial.chi2, lam=float(trial_lam))
            else:
                _release(trial)
            return trial.chi2

        search = search_lambda_for_chi2(
            _evaluate, start_lambda=requested_lam, start_chi2=cleaned_run.chi2,
            target_chi2=target, tolerance=tol, max_trials=max_lambda_trials,
            bounds=lambda_bounds, log=log,
        )

        # A warm sweep inherits its path. If it ends up badly off target, that may
        # be the continuation having followed the smooth branch into a poor basin
        # rather than the data being unfittable, so try once from scratch and keep
        # whichever came closer.
        if (lambda_warm_start and best.get("chi2", float("inf")) > float(lambda_cold_retry_chi2)
                and search["status"] != "converged"):
            log(f"  chi2 still {best['chi2']:.1f}; retrying the sweep cold")
            warm_best, warm_search = dict(best), search
            protected.add(id(warm_best["run"]))  # it may still win
            # Reset to the common starting point so the cold sweep is judged on
            # its own trials rather than inheriting the warm sweep's best.
            lambda_warm_start = False
            solved.clear()
            best.clear()
            best.update(run=cleaned_run, chi2=cleaned_run.chi2, lam=requested_lam)
            search = search_lambda_for_chi2(
                _evaluate, start_lambda=requested_lam, start_chi2=cleaned_run.chi2,
                target_chi2=target, tolerance=tol, max_trials=max_lambda_trials,
                bounds=lambda_bounds, log=log,
            )
            cold_best = dict(best)
            # Closeness to the target is the criterion, not raw chi2: overshooting
            # below the band is no better than missing above it.
            helped = abs(cold_best["chi2"] - target) < abs(warm_best["chi2"] - target)
            result["cold_retry"] = {
                "warm_chi2": float(warm_best["chi2"]), "warm_lambda": float(warm_best["lam"]),
                "cold_chi2": float(cold_best["chi2"]), "cold_lambda": float(cold_best["lam"]),
                "helped": bool(helped),
            }
            if helped:
                log(f"  cold sweep won: chi2 {warm_best['chi2']:.2f} -> "
                    f"{cold_best['chi2']:.2f}")
                protected.discard(id(warm_best["run"]))
                _release(warm_best["run"])
            else:
                _release(cold_best["run"])
                best.clear()
                best.update(warm_best)
                search = warm_search
                log("  cold sweep did not help; keeping the warm result")
        result["auto_lambda_status"] = search["status"]
        for entry in search["trials"]:
            detail = trial_detail.get(float(entry["lambda"]))
            result["lambda_trials"].append({
                "lambda": float(entry["lambda"]), "chi2": float(entry["chi2"]),
                "iterations": detail.iterations if detail else cleaned_run.iterations,
                "stop": detail.stop if detail else cleaned_run.stop,
            })

    switched = bool(best) and float(best["lam"]) != requested_lam
    if switched:
        run = best["run"]
        result["lambda_used"] = float(best["lam"])
        extra = len(search["trials"]) - 1
        head = (f"Auto-λ: {requested_lam:g} → {best['lam']:g}, "
                f"χ² {cleaned_run.chi2:.2f} → {best['chi2']:.2f} in {extra} trial(s).")
        if search["status"] == "converged":
            result["auto_lambda_note"] = f"{head} Your λ is kept too."
        else:
            result["auto_lambda_note"] = (
                f"{head} Still outside {target:g} ± {tol:g}, and the misfit had "
                "flattened at every λ, so the limit is the data or the error model.")
    elif search is not None:
        result["auto_lambda_status"] = "no_improvement"
        result["auto_lambda_note"] = (
            f"Auto-λ: no λ beat {requested_lam:g} (χ² {cleaned_run.chi2:.2f}).")

    # What error level the residuals imply, which is the honest answer when the
    # target stays out of reach: the data are noisier than the file claims.
    implied = _implied_error(run.response, container)
    result["data_error"] = dict(error_info)
    result["data_error"]["implied_from_residuals"] = implied
    if implied is not None and implied > 1.5 * error_info["mean"]:
        log(f"  residuals imply {implied * 100:.1f} % error, not the "
            f"{error_info['mean'] * 100:.1f} % assumed")

    primary = run.result
    fixed_handle = fixed_run.result
    vtk_path = _export_model_vtk(primary, out, "resistivity_model.vtk")
    keep_fixed = switched or (result["outliers"].get("dropped") or 0) > 0
    fixed_vtk_path = (
        _export_model_vtk(fixed_handle, out, "resistivity_model_fixed_lambda.vtk")
        if keep_fixed else vtk_path
    )
    # An inversion that converged is not lost because its output folder could not
    # be written. The caller still gets the model, the mesh and the convergence
    # history in memory; only the on-disk copy is missing, and `export_error`
    # says so instead of the whole run surfacing as a failure minutes in.
    try:
        model_bundle = _export_model_bundle(primary, out, "resistivity")
        fixed_model_bundle = (
            _export_model_bundle(fixed_handle, out, "resistivity_fixed_lambda")
            if keep_fixed else dict(model_bundle)
        )
    except Exception as exc:  # noqa: BLE001 - the inversion result is still good
        model_bundle = {}
        fixed_model_bundle = {}
        result["export_error"] = str(exc)
        log(f"  the inversion finished, but writing its results to {out} failed: {exc}")

    if result["auto_lambda_note"]:
        log(result["auto_lambda_note"])

    # Every iteration the pipeline ran, in order, tagged with the lambda and the
    # data count in force at the time. One concatenated curve tells the whole
    # story: where lambda relaxed, and where a rejection pass changed the data
    # under the misfit (which is why chi2 can step down discontinuously).
    track: List[Dict[str, Any]] = [{
        "stage": "start", "lambda": requested_lam, "n_data": int(data.size()),
        "chi2": [float(c) for c in fixed_run.convergence],
    }]
    for entry in (result["outliers"].get("passes") or []):
        track.append({
            "stage": f"reject {entry['dropped']}", "lambda": requested_lam,
            "n_data": int(entry["kept"]),
            "chi2": [float(c) for c in entry.get("convergence", [])],
        })
    for trial_lam, detail in trial_order:
        track.append({
            "stage": "lambda", "lambda": float(trial_lam),
            "n_data": int(container.size()),
            "chi2": [float(c) for c in detail.convergence],
        })
    result["convergence_track"] = [seg for seg in track if seg["chi2"]]

    metrics = dict(run.metrics)
    metrics.setdefault("chi2", run.chi2)
    if engine in _SELF_REGULARIZING:
        # No lambda was used: the program's own smoothing weight stands in its
        # place, so a stored run does not report a lambda that never acted.
        metrics.pop("lambda", None)
        metrics["smoothing_alpha"] = float(run.metrics.get(f"{engine}_alpha", float("nan")))
    else:
        metrics["lambda"] = float(result["lambda_used"])
    metrics["n_data"] = int(container.size())
    metrics["iterations"] = run.iterations
    if engine == "e4d":
        # Where E4D's own files are: its log, every sigma.N, the 3-D model.
        result["e4d"] = {"command": str(run.metrics.get("e4d_command", "")),
                         "run_dir": str(run.metrics.get("e4d_run", "")),
                         "elements": int(run.metrics.get("e4d_elements", 0)),
                         "vtk_3d": str(Path(str(run.metrics.get("e4d_run", ""))) /
                                       "resistivity_3d.vtk")}
        log(f"  E4D files and the 3-D model: {result['e4d']['run_dir']}")
    elif engine in _SELF_REGULARIZING:
        # Where R2's or R3t's own files are, and the smoothing weight it chose,
        # which stands in for lambda.
        run_dir = str(run.metrics.get(f"{engine}_run", ""))
        result[engine] = {"command": str(run.metrics.get(f"{engine}_command", "")),
                          "run_dir": run_dir,
                          "elements": int(run.metrics.get(f"{engine}_elements", 0)),
                          "alpha": float(run.metrics.get(f"{engine}_alpha", float("nan"))),
                          "stop_message": str(run.metrics.get(f"{engine}_stop_message", "")),
                          "vtk_3d": str(run.metrics.get("vtk_3d", ""))}
        result["lambda_applies"] = False
        log(f"  {_PROGRAM_NAMES[engine]} files: {run_dir} (final alpha "
            f"{result[engine]['alpha']:.4g})")

    result.update(
        {
            "mgr": primary,
            "chi2": run.chi2,
            "vtk": vtk_path,
            "model_bundle": model_bundle,
            "metrics": metrics,
            "convergence": list(run.convergence),
            "fixed_mgr": fixed_handle if keep_fixed else primary,
            "fixed_lambda": {
                "lambda": requested_lam,
                "chi2": fixed_run.chi2,
                "iterations": fixed_run.iterations,
                "n_data": int(data.size()),
                "stop": fixed_run.stop,
                "vtk": fixed_vtk_path,
            },
            "fixed_metrics": dict(fixed_run.metrics),
            "fixed_convergence": list(fixed_run.convergence),
            "fixed_vtk_path": fixed_vtk_path,
            "fixed_model_bundle": fixed_model_bundle,
            "target_chi2": target,
            "chi2_tolerance": tol,
        }
    )
    return result


def _implied_error(response, container) -> Optional[float]:
    """Relative data error implied by the spread of the log residuals."""
    try:
        observed = np.asarray(container["rhoa"], dtype=float)
        predicted = np.asarray(response, dtype=float)
        spread = float(np.std(np.log(observed) - np.log(predicted)))
        return float(np.expm1(spread))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# ERTInversion
# ---------------------------------------------------------------------------
class ERTInversion(InversionBase):
    """Single-time ERT inversion class."""
    
    def __init__(self, data_file: str, mesh: Optional[pg.Mesh] = None, **kwargs):
        """
        Initialize ERT inversion.
        
        Args:
            data_file: Path to ERT data file
            mesh: Mesh for inversion (created if None)
            **kwargs: Additional parameters including:
                - lambda_val: Regularization parameter
                - method: Solver method. A least-squares method ('cgls', the
                  default, 'lsqr', 'scipy_lsmr', ...) solves the stacked system;
                  'spd_cholesky' or 'spd_cg' solve its normal equations, as the
                  time-lapse inversion does.
                - model_constraints: (min, max) model parameter bounds
                - max_iterations: Maximum iterations
                - absoluteError: Absolute resistance error floor [Ohm] (default 0.0001)
                - relativeError: Relative data error
                - lambda_rate: Lambda reduction rate
                - lambda_min: Minimum lambda value
                - min_relative_error: Minimum error floor (default 0.01)
                - max_relative_error: Maximum error cap (default 0.50)
                - use_gpu: Whether to use GPU acceleration (requires CuPy)
                - parallel: Whether to use parallel CPU computation
                - n_jobs: Number of parallel jobs (-1 for all cores)
                - zones: a-priori resistivity zones, a list of dicts with a
                  ``polygon`` in mesh coordinates, a ``resistivity`` and an
                  optional ``fixed`` (see ``ert_zones``). The run starts from the
                  a-priori model, regularizes toward it, and holds fixed zones.
                - reference_weight: weight of a smallness term that pulls the
                  model toward the reference model, relative to the smoothness
                  term (default 0.0, smoothness alone). First-order smoothness
                  does not see a constant shift of the model, so without this
                  term a homogeneous reference model has no effect on the result.
        """
        # Load ERT data. An already-loaded container is accepted so that callers
        # who have applied their own error model or QC filter (the auto-lambda
        # pipeline does both) do not have to round-trip through a file.
        data = data_file if isinstance(data_file, pg.DataContainer) else ert.load(data_file)

        # Compute geometric factors numerically from electrode positions
        # This ensures accurate K values based on actual electrode geometry
        # Check if K values are missing, all zeros, or all ones (placeholder values)
        if 'k' not in data.dataMap() or len(data['k']) == 0 or np.allclose(data['k'], 0) or np.allclose(data['k'], 1):
            print("   Computing geometric factors from electrode positions...")
            data['k'] = ert.createGeometricFactors(data, numerical=True)
            # Rebuild rhoa only from resistances the file actually carries.
            # pyGIMLi fills an absent 'r' with zeros, so an unconditional r * k
            # wiped out every apparent resistivity of a file that stores rhoa
            # (fielddataline2.dat does) and the run returned 1e-6 Ohm m.
            if data.haveData('r'):
                data['rhoa'] = data['r'] * data['k']
            elif not data.haveData('rhoa') and data.haveData('u') and data.haveData('i'):
                data['rhoa'] = data['u'] / data['i'] * data['k']
        # Call parent initializer
        super().__init__(data, mesh, **kwargs)
        
        # Set ERT-specific default parameters

        ert_defaults = {
            'lambda_val': 10.0,
            'method': 'cgls',
            'absoluteError': 0.0001,
            'relativeError': 0.05,
            'lambda_rate': 1.0,
            'lambda_min': 1.0,
            'min_relative_error': 0.01,
            'max_relative_error': 2.00,
            # Stopping. Both were hard-coded (chi2 < 1.5, dPhi < 0.01) before; the
            # lambda search needs them configurable, because a run that stops at
            # 1.5 cannot tell you whether this lambda could have reached 1.0, and
            # a run that stops at dPhi = 1 % may still be descending.
            'target_chi_squared': 1.0,
            'convergence_tolerance': 0.005,
            'min_iterations': 5,
            'reference_weight': 0.0,
            'verbose': True,
            'use_gpu': False,      # Add GPU acceleration option
            'parallel': False,     # Add parallel computation option
            'n_jobs': -1           # Number of parallel jobs (-1 means all available cores)
        }
        
        # Update parameters with ERT defaults
        for key, value in ert_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize internal variables
        self.fwd_operator = None
        self.Wdert = None  # Data weighting matrix
        self.Wm_r = None   # Model weighting matrix
        self.rhos1 = None  # Log-transformed apparent resistivities
        self._zone_prior = None  # the zones on the parameter mesh, once built

    def zone_prior(self):
        """The a-priori zones on this inversion's parameter mesh, or None.

        Worked out once the forward operator has built the parameter mesh,
        whose cell order is the model vector's. See ``ert_zones``.
        """
        zones = self.parameters.get('zones')
        if not zones:
            return None
        if self._zone_prior is None:
            if self.fwd_operator is None:
                self.setup()
            from .ert_zones import zone_prior

            self._zone_prior = zone_prior(
                self.fwd_operator.paraDomain, zones,
                bounds=self.parameters['model_constraints'])
        return self._zone_prior

    def setup(self):
        """Set up ERT inversion (create operators, matrices, etc.)"""
        # DEBUG: Print electrode positions from data before mesh creation
        sensors = self.data.sensorPositions()
       
        if len(sensors) > 0:
            y_vals = [s.y() for s in sensors]
            

        # Create mesh if not provided
        if self.mesh is None:
            ert_manager = ert.ERTManager(self.data)
            self.mesh = ert_manager.createMesh(data=self.data, quality=34)
            
            y_coords = [n.y() for n in self.mesh.nodes()]

        # Initialize forward operator
        self.fwd_operator = ert.ERTModelling()
        self.fwd_operator.setData(self.data)
        self.fwd_operator.setMesh(self.mesh)
        
        # Prepare data
        rhos = self.data['rhoa']
        rhoa_values = np.asarray(rhos.array(), dtype=float)
        unusable = int(np.count_nonzero(~(np.isfinite(rhoa_values) & (rhoa_values > 0))))
        if unusable:
            # The inversion works on log(rhoa); a zero or negative value becomes
            # -inf or NaN and the run "converges" to garbage without an error.
            raise ValueError(
                f"{unusable} of {rhoa_values.size} readings have no positive apparent "
                "resistivity, which a log-resistivity inversion cannot use. Filter "
                "them out, or supply resistances ('r') or voltage and current "
                "('u', 'i') so rhoa can be computed.")
        self.rhos1 = np.log(rhoa_values)
        self.rhos1 = self.rhos1.reshape(self.rhos1.shape[0], 1)
        
        # Data error matrix
        # Check if error data exists and is valid (non-zero values)
        has_valid_err = False
        if 'err' in self.data.dataMap():
            err_array = self.data['err'].array()
            has_valid_err = np.all(err_array > 0) and np.all(np.isfinite(err_array))

        if has_valid_err:
            # If data has valid error values, use them
            Delta_rhoa_rhoa = self.data['err'].array()
            print(f'   Using provided error estimates (mean: {np.mean(Delta_rhoa_rhoa):.4f}, range: [{np.min(Delta_rhoa_rhoa):.4f}, {np.max(Delta_rhoa_rhoa):.4f}])')
        else:
            # Estimate per-measurement relative error (Seb's resistance-based formula):
            #   err_i = relativeError + absoluteError / |r_i|
            # Because δρa/ρa = δr/r, noise lives in resistance space.
            # r is used directly if available; otherwise reconstructed from rhoa / k.
            abs_e = float(self.parameters['absoluteError'])
            rel_e = float(self.parameters['relativeError'])
            print(f'   No valid error data found, estimating errors '
                  f'(absoluteError={abs_e}, relativeError={rel_e})')

            # haveData, not "in dataMap": pyGIMLi lists 'r' even when the file has
            # none, and |r| = 0 gave every datum the max_relative_error cap.
            if self.data.haveData('r'):
                r_abs = np.abs(self.data['r'].array())
            elif self.data.haveData('k'):
                r_abs = np.abs(rhos.array()) / np.maximum(np.abs(self.data['k'].array()), 1e-10)
            else:
                raise RuntimeError("Cannot estimate error: data must contain 'r' or ('rhoa' + 'k').")

            Delta_rhoa_rhoa = rel_e + abs_e / np.maximum(r_abs, 1e-10)
            print(f'   Estimated error statistics (mean: {np.mean(Delta_rhoa_rhoa):.4f}, '
                  f'range: [{np.min(Delta_rhoa_rhoa):.4f}, {np.max(Delta_rhoa_rhoa):.4f}])')

        # Clip errors: too small → min_relative_error, too large → max_relative_error
        min_err = float(self.parameters.get('min_relative_error', 0.01))
        max_err = float(self.parameters.get('max_relative_error', 2.00))
        Delta_rhoa_rhoa = np.clip(np.asarray(Delta_rhoa_rhoa, dtype=float), min_err, max_err)

        # Data weighting is diagonal, so hold it as a vector and apply it by
        # broadcasting. Materializing it as a D-by-D array cost 0.27 GB at
        # D = 6000 and turned 'Wdert.dot(Jr)' into an O(D^2 P) matmul where
        # O(D P) does the same job. Wdert is kept as a scipy diagonal for any
        # caller that reads the attribute; the hot paths below use the vector.
        self.Wdert_diag = 1.0 / np.log(Delta_rhoa_rhoa + 1)
        self.Wdert = diags(self.Wdert_diag)
        
        # Create model regularization matrix
        rm = self.fwd_operator.regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)
        self.Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
        cw = rm.constraintWeights().array()
        # Left sparse. It holds about two nonzeros per row, so densifying it
        # was roughly 300 MB of mostly zeros for a 5000-cell mesh, kept for the
        # whole run. The one place that needs a dense copy makes it on the spot.
        self.Wm_r = diags(cw).dot(self.Wm_r).tocsr()
    
    def run(self, initial_model: Optional[np.ndarray] = None,
            reference_model: Optional[np.ndarray] = None) -> InversionResult:
        """
        Run ERT inversion.

        Args:
            initial_model: Initial model parameters (if None, a homogeneous model is used)
            reference_model: Model the regularization pulls toward. Defaults to the
                initial model, which is the usual smoothness-from-homogeneous setup.
                Pass it explicitly when continuing an interrupted run. Leaving it to
                default makes the restart point the new reference, so the penalty
                becomes roughness of the *change* since the restart rather than
                roughness of the model; accumulated structure then goes unpenalized
                and the run drifts steadily under-regularized. Measured on the
                Ashton line over a 20 to 0.3 lambda ladder, that reached chi2 5.72
                where the same ladder with the reference pinned reached 6.38.

        Returns:
            InversionResult with inversion results
        """
        # Make sure setup has been called
        if self.fwd_operator is None:
            self.setup()
        
        # Initialize result object
        result = InversionResult()
        
        # Set up initial model if not provided
        if initial_model is None:
            rhomodel = np.median(np.exp(self.rhos1)) * np.ones((self.fwd_operator.paraDomain.cellCount(), 1))
            mr = np.log(rhomodel)
        else:
            if initial_model.ndim == 1:
                initial_model = initial_model.reshape(-1, 1)
            if np.min(initial_model) <= 0:
                # Handle negative/zero values with absolute value + offset (safer than small offset)
                print(f'WARNING: Initial model contains non-positive values (min={np.min(initial_model):.2e}). Using absolute values.')
                mr = np.log(np.abs(initial_model) + 1.0)
            else:
                mr = np.log(initial_model)

        # A-priori zones: the run starts from the a-priori model and regularizes
        # toward it, and a fixed zone keeps its value whatever the run is
        # continued from.
        prior = self.zone_prior()
        prior_mr = fixed = None
        if prior is not None:
            background = float(np.median(np.exp(self.rhos1)))
            prior_mr = np.log(prior.with_background(background)).reshape(-1, 1)
            if initial_model is None:
                mr = prior_mr.copy()
            else:
                mr[prior.fixed] = prior_mr[prior.fixed]
            fixed = prior.fixed if prior.any_fixed else None
            result.meta['zones'] = [dict(entry) for entry in prior.report]

        # Reference model is the initial model (the a-priori one with zones)
        # unless the caller pinned one
        if reference_model is None:
            mr_R = (prior_mr if prior_mr is not None else mr).copy()
        else:
            ref = np.asarray(reference_model, dtype=float)
            if ref.ndim == 1:
                ref = ref.reshape(-1, 1)
            mr_R = np.log(np.maximum(ref, 1e-12))

        # Regularization parameter
        L_mr = np.sqrt(self.parameters['lambda_val'])

        # Smallness, relative to the smoothness. The first-order smoothness rows
        # cancel any constant, so without it a homogeneous reference drops out
        # of the objective and two runs from different homogeneous references
        # converge to the same model. The depth-of-investigation index compares
        # exactly such runs; without this term it measured only how far each
        # had got when it stopped, and on the Ex_sensitivity_analysis survey it
        # was near zero in every cell. Oldenburg and Li (1999) keep the weight
        # small.
        alpha_s = float(self.parameters.get('reference_weight', 0.0))
        if not np.isfinite(alpha_s) or alpha_s < 0:
            raise ValueError(f"reference_weight must be finite and non-negative, got {alpha_s}.")

        # Model constraints
        min_mr, max_mr = self.parameters['model_constraints']
        min_mr = np.log(min_mr)
        max_mr = np.log(max_mr)

        # Apply constraints to initial model immediately
        mr = np.clip(mr, min_mr, max_mr)
        if reference_model is None and prior_mr is None:
            mr_R = mr.copy()  # Update reference model after clipping
        else:
            mr_R = np.clip(mr_R, min_mr, max_mr)

        # Initial setup for the inversion
        delta_mr = (mr - mr_R)
        chi2_ert = 1

        lam_val = float(self.parameters['lambda_val'])
        target_chi2 = float(self.parameters.get('target_chi_squared', 1.0))
        dphi_tol = float(self.parameters.get('convergence_tolerance', 0.005))
        # A flat second iteration is normal while the line search finds its
        # scale, so a plateau only counts as convergence after
        # 'min_iterations'. The clamp keeps the guard from swallowing the
        # plateau test outright when a caller asks for very few iterations.
        min_iterations = min(
            int(self.parameters.get('min_iterations', 5)),
            max(int(self.parameters['max_iterations']) - 2, 0),
        )
        verbose = bool(self.parameters.get('verbose', True))
        stop_reason = 'iteration_cap'
        line_search_failures = 0
        # What the forward operator last solved: (log model, linear response,
        # whether its Jacobian followed). The line search solves the model it
        # accepts, and each iteration used to solve that same model again before
        # building its Jacobian - a third of the run. pyGIMLi's createJacobian
        # reads the potentials of the operator's last response() and does not
        # check the model, so a solve is reused only while this record still
        # describes the operator.
        solved = None

        # Main inversion loop
        for nn in range(self.parameters['max_iterations']):
            if verbose:
                print(f'-------------------Iteration: {nn} ---------------------------')

            # Forward modeling and Jacobian computation; the forward solve is
            # skipped when the operator already holds this model's.
            held = solved is not None and np.array_equal(solved[0], mr)
            dr, Jr, linear = ertforandjac2(
                self.fwd_operator, mr, self.mesh,
                response=solved[1] if held else None, with_response=True)
            solved = (mr, linear, True)
            dr = dr.reshape(dr.shape[0], 1)

            # The regularization pull is toward the reference model, so it has to
            # be recomputed from the current model every iteration. Leaving it at
            # its initial value of zero silently drops that term from the gradient
            # and from the Armijo test, which makes the step inconsistent with the
            # system being solved: the effective regularization is weaker than
            # lambda claims, and the line search starts failing once the model has
            # moved away from the reference.
            delta_mr = mr - mr_R

            # Data misfit calculation
            dataerror_ert = self.rhos1 - dr
            # np.dot does not dispatch to a scipy sparse matrix, and the
            # weighting is diagonal anyway, so apply it by broadcasting.
            # Shape stays (1, 1), which the chi2 conversion below relies on.
            weighted_err = self.Wdert_diag.reshape(-1, 1) * np.asarray(
                dataerror_ert, dtype=float).reshape(-1, 1)
            fdert = weighted_err.T.dot(weighted_err)

            # Model regularization term. The stacked system below solves the normal
            # equations of ||Wd (d - f(m))||^2 + lambda (||Wm (m - m_ref)||^2 +
            # reference_weight ||m - m_ref||^2), so the objective evaluated here
            # must use lambda, not sqrt(lambda).
            wm_r = self.Wm_r * (mr - mr_R)
            fmert = lam_val * (wm_r.T.dot(wm_r) + alpha_s * delta_mr.T.dot(delta_mr))

            # Total objective function
            fc_r = fdert + fmert

            # Compute chi-squared and check convergence
            old_chi2 = chi2_ert
            chi2_ert = fdert / len(dr)
            # Convert to scalar if it's an array (fdert is (1,1) from matrix multiplication)
            if isinstance(chi2_ert, np.ndarray):
                chi2_ert = float(chi2_ert.item())
            dPhi = abs(chi2_ert - old_chi2) / old_chi2 if nn > 0 else 1.0

            if verbose:
                print(f'chi2: {chi2_ert}')
                print(f'dPhi: {dPhi}')

            # Store iterations data
            result.iteration_models.append(np.exp(mr.ravel()))
            result.iteration_chi2.append(chi2_ert)
            result.iteration_data_errors.append(dataerror_ert.ravel())

            # Check for convergence. Both criteria are parameters: stop once the
            # data are fitted to the target, or once the misfit has flattened.
            if chi2_ert < target_chi2:
                stop_reason = 'target'
                break
            if dPhi < dphi_tol and nn > min_iterations:
                stop_reason = 'plateau'
                break

            # System matrix and gradient. 'wd' applies the diagonal data
            # weights by broadcasting instead of through a D-by-D matmul.
            wd = self.Wdert_diag.reshape(-1, 1)
            data_residual = np.asarray(dr - self.rhos1, dtype=float).reshape(-1, 1)
            reg_residual = np.asarray(
                self.Wm_r.dot(delta_mr), dtype=float
            ).reshape(-1, 1)
            gc_r = np.vstack((wd * data_residual, L_mr * reg_residual))
            # The stacked system [W_d J; L W_m; L sqrt(alpha_s) I] is held as
            # its blocks, the Jacobian dense and the rest sparse. It used to be
            # one dense array, built by densifying W_m (about 1.5 n^2 doubles)
            # and np.eye(n), copied again by each vstack: 4.4 GB on top of the
            # run for a 10779-cell mesh. The GPU solvers take it assembled.
            blocks = [wd * Jr, L_mr * self.Wm_r]
            if alpha_s > 0:
                gc_r = np.vstack((gc_r, L_mr * np.sqrt(alpha_s) * delta_mr))
                blocks.append(L_mr * np.sqrt(alpha_s) * sp.identity(Jr.shape[1], format="csr"))
            N11_R = StackedSystem(blocks)
            if self.parameters['use_gpu']:
                N11_R = N11_R.toarray()

            gc_r = np.array(gc_r)
            gc_r = gc_r.reshape(-1, 1)
            
            # Alternative gradient formulation
            # L_mr stays at the first power here, as it was. This expression
            # feeds only the Armijo test below, and rescaling it would change
            # which step lengths are accepted.
            reg_gradient = np.asarray(
                self.Wm_r.T.dot(self.Wm_r.dot(delta_mr)), dtype=float
            ).reshape(-1, 1) + alpha_s * delta_mr
            gc_r1 = Jr.T.dot(wd ** 2 * data_residual) + L_mr * reg_gradient
            
            # Solve for the update: the stacked system for a least-squares
            # method, its normal equations for an SPD one. Fixed cells are left
            # out of the system - their columns removed - so their update is
            # zero, while the smoothness rows that tie them to their neighbours
            # still pull on the free cells.
            solver_options = dict(
                use_gpu=self.parameters['use_gpu'],
                parallel=self.parameters.get('parallel', False),
                n_jobs=self.parameters.get('n_jobs', -1))
            if fixed is None:
                d_mr = _gauss_newton_step(
                    N11_R, -gc_r, self.parameters['method'], **solver_options)
            else:
                free = ~fixed
                d_mr = np.zeros_like(mr)
                d_mr[free] = np.asarray(_gauss_newton_step(
                    N11_R.columns(free) if isinstance(N11_R, StackedSystem) else N11_R[:, free],
                    -gc_r, self.parameters['method'],
                    **solver_options), dtype=float).reshape(-1, 1)
            
            # Line search. Armijo sufficient decrease is
            # f(m + mu d) <= f(m) + c mu (d . g), with d . g < 0 for a descent
            # direction, so the goal lies below f(m). It was written with the
            # sign reversed, which put the goal above f(m) and let a step that
            # did not lower the objective, or raised it, pass. The best trial
            # that did lower it is kept in case none meets the test.
            mu_LS = 1
            iarm = 1
            f_current = float(np.asarray(fc_r).item())
            directional = float(np.asarray(
                d_mr.T.dot(gc_r1.reshape(gc_r1.shape[0], 1))).item())
            best_mr, best_f = None, f_current
            while True:
                mr1 = mr + mu_LS * d_mr

                # Enforce constraints BEFORE forward modeling (critical fix)
                mr1 = np.clip(mr1, min_mr, max_mr)

                # Validate that model is finite before forward modeling
                if not np.all(np.isfinite(mr1)):
                    if verbose:
                        print(f'WARNING: Non-finite values detected in model at iteration {nn}')
                    mr1 = np.nan_to_num(mr1, nan=min_mr, posinf=max_mr, neginf=min_mr)

                dr, linear = ertforward2(self.fwd_operator, mr1, self.mesh, with_response=True)
                solved = (mr1, linear, False)
                dr = dr.reshape(dr.shape[0], 1)

                dataerror_ert = self.rhos1 - dr
                # np.dot does not dispatch to a scipy sparse matrix, and the
                # weighting is diagonal anyway, so apply it by broadcasting.
                weighted_err = self.Wdert_diag.reshape(-1, 1) * np.asarray(
                    dataerror_ert, dtype=float).reshape(-1, 1)
                fdert = weighted_err.T.dot(weighted_err)
                wm_trial = self.Wm_r * (mr1 - mr_R)
                fmert = lam_val * (wm_trial.T.dot(wm_trial)
                                   + alpha_s * (mr1 - mr_R).T.dot(mr1 - mr_R))

                ft_r = float(np.asarray(fdert + fmert).item())
                if ft_r < best_f:
                    best_mr, best_f = mr1, ft_r

                fgoal = f_current + 1e-4 * mu_LS * directional
                #print(f'ft_r: {ft_r}, fgoal: {fgoal}')

                # 'ft_r < f_current' only matters when d . g >= 0 (an inexact
                # solve), where the Armijo goal alone would admit an increase.
                if ft_r <= fgoal and ft_r < f_current:
                    break
                else:
                    iarm = iarm + 1
                    mu_LS = mu_LS / 2

                if iarm > 20:
                    line_search_failures += 1
                    if verbose:
                        print('Line search FAIL EXIT')
                    # Never step uphill: take the best trial that lowered the
                    # objective, or stay put. Before, the last (tiniest) trial
                    # was taken whatever its objective.
                    mr1 = best_mr if best_mr is not None else mr
                    break

            # Update model
            mr = mr1

            # Apply model constraints
            mr = np.clip(mr, min_mr, max_mr)

            # Optional lambda cooling. The default rate of 1.0 leaves lambda fixed,
            # which is what the lambda search assumes: a schedule that moves lambda
            # every iteration makes the final chi2 unattributable to any one value.
            lambda_min = self.parameters['lambda_min']
            if L_mr > np.sqrt(lambda_min):
                L_mr = L_mr * self.parameters['lambda_rate']
                lam_val = float(L_mr ** 2)  # keep the objective consistent with the system

        # Process final model
        final_model = np.exp(mr)

        # Compute final forward response. The loop has usually just solved this
        # model - it ends right after its forward solve, or after the line search
        # accepted it - and that solve is the same one unless the forward
        # wrappers' resistivity clip changed the model they passed.
        clip_free = np.array_equal(np.clip(np.exp(np.clip(mr, -20, 20)), 0.001, 1e6), final_model)
        if solved is not None and clip_free and np.array_equal(solved[0], mr):
            dr = pg.Vector(solved[1])
            jacobian_held = solved[2]
        else:
            dr = self.fwd_operator.response(pg.Vector(final_model.ravel()))
            jacobian_held = False

        # Every chi2 recorded in the loop is pre-update, so when the loop ends
        # on the iteration cap the history, and meta['chi2'] with it, described
        # the model from before the final step, not the one returned
        # (fielddataline2 with max_iterations=1: 3.49 reported, 0.25 actual).
        # Score the returned model the same way the loop does.
        iterations_run = len(result.iteration_chi2)
        if stop_reason == 'iteration_cap':
            final_err = self.rhos1 - np.log(np.asarray(dr, dtype=float)).reshape(-1, 1)
            weighted_final = self.Wdert_diag.reshape(-1, 1) * final_err
            result.iteration_chi2.append(
                float(weighted_final.T.dot(weighted_final).item()) / max(len(final_err), 1))
            result.iteration_models.append(final_model.ravel().copy())
            result.iteration_data_errors.append(final_err.ravel())
        
        # Compute coverage, from the Jacobian of the final model - already built
        # when the loop stopped at the start of an iteration.
        if not jacobian_held:
            self.fwd_operator.createJacobian(pg.Vector(final_model.ravel()))
        covTrans = pg.core.coverageDCtrans(
            self.fwd_operator.jacobian(),
            1.0 / dr,
            1.0 / pg.Vector(final_model.ravel())
        )
        
        paramSizes = np.zeros(len(final_model))
        mesh2 = self.fwd_operator.paraDomain
        
        for c in mesh2.cells():
            paramSizes[c.marker()] += c.size()
            
        FinalJ = np.log10(covTrans / paramSizes)
        
        # Store results
        result.final_model = final_model.ravel()
        result.coverage = FinalJ
        result.predicted_data = dr.array()
        result.mesh = mesh2
        # Why the loop ended, so a caller driving lambda can tell "this lambda
        # cannot do better" apart from "this run ran out of iterations".
        result.meta['stop_reason'] = stop_reason
        result.meta['iterations'] = iterations_run
        result.meta['line_search_failures'] = line_search_failures
        result.meta['chi2'] = float(result.iteration_chi2[-1]) if result.iteration_chi2 else float('nan')
        result.meta['lambda'] = float(self.parameters['lambda_val'])

        if verbose:
            print('End of inversion')
        return result
