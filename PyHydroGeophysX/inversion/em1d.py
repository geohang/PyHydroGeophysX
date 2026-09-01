"""Occam-style 1D FDEM/TDEM inversion helpers."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_LN10 = math.log(10.0)

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX._internal.utils import noop as _noop
from PyHydroGeophysX.forward.em1d import (
    _fdem_config,
    _tdem_config,
    _tdem_geometry,
    model_depth_profile,
)

LogFn = Callable[[str], None]

DEFAULT_INVERSION = {
    "n_layers": 15, "min_thickness": 1.0, "max_thickness": 40.0,
    "starting_resistivity": 100.0, "max_iterations": 30,
    "rel_error": 0.05, "noise_floor": 1e-14, "smoothness": 0.3,
    "lateral_smoothness": 0.0, "lci_passes": 1,
    # Line inversion. ``lci_mode`` picks how neighbouring soundings are tied
    # together: "simultaneous" solves the line as one system, "sequential" runs
    # the older block-coordinate passes, "off" inverts each sounding alone. The
    # mode only takes effect when ``lateral_smoothness`` is positive.
    "lci_mode": "simultaneous", "reference_distance": 10.0,
    # The formal coupled-line solver is bound-aware sparse TRF.  The tolerances
    # below are the balanced full-survey settings validated on trailcreek: the
    # stricter ftol=1e-6 spent all 90 evaluations in the weakly constrained
    # convergence tail without completing robust reweighting.  Gauss-Newton is
    # still available explicitly for fast previews and historical reproduction.
    "lci_solver": "trf", "lci_max_nfev": 90,
    "lci_ftol": 1e-4, "lci_xtol": 1e-6, "lci_gtol": 1e-5,
    # A multiplier on ``lateral_smoothness``, not a second knob. The two meet as
    # a product before the solver sees either, and the product is then squared
    # into the penalty weight, so 1.3 x 2.0 and 2.6 x 1.0 give the same section
    # to the last bit. It is kept at 1.0 and left alone; ``lateral_smoothness``
    # is the one to move, because it is the number the project file itself
    # records and it is one value rather than a product of two.
    "lateral_weight_scale": 1.0, "auto_lambda": True,
    # How the lateral tie between neighbouring soundings falls off with their
    # separation: the penalty scales as
    # ``(reference_distance / separation) ** lateral_distance_power``. 1.0 makes
    # the tie inversely proportional to distance; 0.0 ties every neighbouring
    # pair alike regardless of spacing.
    "lateral_distance_power": 1.0,
    "target_chi2": 1.0, "chi2_tolerance": 0.2, "max_lambda_trials": 5,
    "convergence_tolerance": 0.02, "min_iterations": 2,
    "reject_outliers": False, "outlier_threshold": 3.0,
    # Optional data-only Huber IRLS. Original errors remain available for QC;
    # effective errors grow for large residuals, but no imported gate is lost.
    "robust_errors": False, "robust_threshold": 3.0,
    "robust_passes": 3, "robust_max_error_factor": 10.0,
    "robust_min_unchanged_fraction": 0.0, "robust_target_chi2": 0.0,
    "robust_target_tolerance": 0.25,
    # Optional empirical resistive-background tendency. Weak absolute LM signal
    # can support a resistive half-space/background interpretation, but it does
    # not locate that resistivity in the shallow layers. The historical
    # ``shallow_prior_*`` key prefix remains for saved-config compatibility;
    # the penalty now spans the whole layer vector. A zero explicit resistivity
    # chooses ``effective starting half-space * factor`` instead.
    "shallow_prior_enabled": False, "shallow_prior_depth_m": 0.0,
    "shallow_prior_min_resistivity": 0.0,
    "shallow_prior_resistivity_factor": 2.0,
    # The signal trigger has its own fixed homogeneous reference, so changing
    # the soft model target does not redefine which observations trigger it.
    "shallow_prior_signal_reference_resistivity": 1000.0,
    "shallow_prior_weight": 1.0,
    "shallow_prior_window": 11, "shallow_prior_snr_ratio": 0.6,
    "shallow_prior_mode": "quality_trend", "shallow_prior_reference_gate": 2,
    "shallow_prior_signal_ratio": 0.8, "shallow_prior_noise_ratio_max": 2.0,
    "shallow_prior_signal_threshold": 0.0,
    "outlier_passes": 2, "min_data_fraction": 0.8,
    "min_gates_per_sounding": 3,
    # Data QC. ``min_rel_error`` floors and ``max_rel_error`` caps the stack
    # error a gate arrives with, before ``rel_error`` joins it in quadrature.
    # Both off by default; see :func:`_tdem_uncertainty`.
    "min_rel_error": 0.0, "max_rel_error": None,
    # Model bounds in ohm-m, as the optimiser's box constraint on log10 rho.
    # A layer the data cannot resolve is driven only by the regularisation, so
    # without a bound it walks until the box stops it; the width of the box then
    # sets how far a deep layer can rail, and a section whose deepest cells sit
    # on the bound is reporting the bound rather than the ground. The default
    # pair is wide (1 to 1e5) so it constrains nothing on well-resolved data;
    # narrow it to the range the target geology can plausibly span.
    "rho_min": 1.0, "rho_max": 1e5,
    # How far auto-lambda may scale the smoothness while chasing target_chi2.
    "scale_bounds": (1e-4, 1e4),
    # Cumulative sensitivity below which a cell is reported as unresolved. Named
    # here rather than left to the caller so a preset can move it and a panel can
    # move it back. The value repeats em1d_lci.DOI_SENSITIVITY_THRESHOLD rather
    # than importing it, because that module pulls in SciPy sparse and this one
    # is imported by the CLI and the Qt panel; a test asserts the two agree.
    "doi_threshold": 0.8,
}


#: Named starting points for the inversion settings.
#:
#: :data:`DEFAULT_INVERSION` has to serve everything from a 30 m ground sounding
#: to an airborne line that sees several hundred metres, so it is deliberately
#: unopinionated. A survey type that has been worked through can do better than
#: unopinionated, and that is what a preset carries: the settings a particular
#: kind of data was found to want, in one place, with the reason recorded beside
#: each one.
#:
#: A preset holds only what it changes. :func:`preset_inversion` merges it over
#: the defaults, so a key nobody has an opinion about stays wherever the
#: framework put it.
INVERSION_PRESETS: Dict[str, Dict[str, Any]] = {
    "generic": {},
    "ground_tem": {
        # A ground system's gates run from a few microseconds to a few hundred,
        # which is a hundred metres of diffusion depth, not four hundred. Twenty
        # layers over that range put the finest ones where the early gates
        # actually resolve something.
        "n_layers": 20, "min_thickness": 1.0, "max_thickness": 14.74,
        # A ground sounding keeps a median of four or five gates against twenty
        # layers. Chasing chi-square to 1 on that is fitting noise: the model
        # oscillates to pass through a handful of points. Fixing the
        # regularisation and accepting a higher misfit is the honest trade, and
        # it halves the upper tail of the recovered resistivity.
        "smoothness": 1.5, "lateral_smoothness": 1.3,
        "auto_lambda": False, "scale_bounds": (0.5, 2.0),
        # Preserve sparse LM/HM observations. Large residuals get a bounded
        # increase in effective error instead of being deleted. Legacy hard
        # rejection options remain available for reproducing older runs.
        "reject_outliers": False, "outlier_threshold": 3.0,
        "robust_errors": True, "robust_threshold": 3.0,
        "robust_passes": 3, "robust_max_error_factor": 10.0,
        "robust_min_unchanged_fraction": 0.70, "robust_target_chi2": 1.75,
        "robust_target_tolerance": 0.25,
        "shallow_prior_enabled": True,
        "shallow_prior_mode": "signal_threshold",
        # Weak absolute LM supports only a background-resistivity tendency, not
        # a shallow depth. Twice the effective starting half-space is a modest
        # one-sided target; stronger multipliers too readily manufacture the
        # result this empirical prior assumes.
        "shallow_prior_resistivity_factor": 2.0,
        "outlier_passes": 2, "min_data_fraction": 0.8,
        "min_gates_per_sounding": 3,
        # A stack error is itself estimated from a finite number of repeats, so
        # a gate that happens to stack quietly can report a few tenths of a
        # percent and then outweigh its neighbours by orders of magnitude.
        "min_rel_error": 0.03, "rel_error": 0.06,
        # A deep layer the data cannot resolve walks until the box stops it, so
        # the box decides how far it walks. Weathered bedrock does not reach
        # 1e5 ohm-m, and leaving room to means reporting the bound as if it
        # were a measurement.
        "rho_max": 1.0e4,
        # Cumulative sensitivity is summed from the bottom up and the deepest
        # layer is thick, so a low threshold saturates: on one survey more than
        # half the stations reported a depth of investigation at the very base
        # of the model, which is the metric running out rather than the data.
        "doi_threshold": 6.0,
        "auto_starting_model": True,
    },
}


def preset_inversion(name: str = "generic") -> Dict[str, Any]:
    """Inversion settings for a named survey type, over the framework defaults."""
    key = str(name).strip().lower()
    if key not in INVERSION_PRESETS:
        raise ValueError(
            f"inversion preset must be one of {sorted(INVERSION_PRESETS)}; "
            f"got {name!r}.")
    return {**DEFAULT_INVERSION, **INVERSION_PRESETS[key]}


def _layer_thicknesses(n_layers: int, min_thickness: float, max_thickness: float) -> np.ndarray:
    n_layers = max(2, int(n_layers))
    if n_layers == 2:
        return np.array([float(min_thickness)], dtype=float)
    return np.geomspace(float(min_thickness), float(max_thickness), n_layers - 1)


def _inversion_layer_thicknesses(inv: Dict[str, Any]) -> np.ndarray:
    n_layers = int(inv.get("n_layers", 15))
    explicit = np.asarray(inv.get("layer_thicknesses", []), dtype=float).ravel()
    if (
        explicit.size == n_layers - 1
        and np.all(np.isfinite(explicit))
        and np.all(explicit > 0.0)
    ):
        return explicit
    return _layer_thicknesses(
        n_layers,
        float(inv.get("min_thickness", 1.0)),
        float(inv.get("max_thickness", 40.0)),
    )


def _log_resistivity_bounds(inv: Dict[str, Any]) -> "tuple[float, float]":
    """``(log10 rho_min, log10 rho_max)`` for the optimiser's box constraint.

    Kept in one place so the per-sounding solver and the coupled line solver
    cannot drift apart on what a plausible resistivity is.
    """
    # Only a missing key or an explicit None falls back to the default. ``or``
    # would swallow a caller's 0.0 as well and hand back 1.0, turning a bad
    # setting into a silently different model.
    low = inv.get("rho_min", 1.0)
    high = inv.get("rho_max", 1e5)
    lo = float(1.0 if low is None else low)
    hi = float(1e5 if high is None else high)
    if not (0.0 < lo < hi):
        raise ValueError(
            f"rho_min must be positive and below rho_max; got {lo} and {hi}.")
    return math.log10(lo), math.log10(hi)


def _tdem_uncertainty(observed: np.ndarray, item: Dict[str, Any],
                      rel: float, floor: float,
                      min_rel: float = 0.0,
                      max_rel: Optional[float] = None) -> np.ndarray:
    """Per-gate uncertainty: the recorded stack error with ``rel`` added in quadrature.

    Two independent things go wrong with a gate. Its stack error says how
    repeatably it was measured, and the file records one per gate. Everything
    else, system calibration and the error in representing the ground as 1D
    layers, applies to every gate alike and the stack error knows nothing about
    it; that is what ``rel`` carries.

    Adding them in quadrature is how independent errors combine. It also
    behaves better than taking the larger of the two, which was the previous
    rule: quadrature leaves a gate that is already noisy essentially untouched,
    so the instrument's relative weighting between clean and noisy gates
    survives, where a floor flattens every gate below it to the same weight.

    Where the file carries no stack error for a gate, ``rel`` is the whole
    budget, so a partially populated column is still usable.

    ``rel`` is the size of that uniform term, not an amount to add on top of
    whatever is already there. Some formats store an error that has a uniform
    term folded in already, and a reader that knows this reports how much as
    ``item["uniform_error"]``; only the shortfall is then added in quadrature.
    Without that key the two are the same thing, so nothing changes for a file
    whose stored column is a bare stack error.

    ``min_rel`` and ``max_rel`` clamp the recorded stack error before the
    quadrature step.

    The floor is the half that matters. A stack error is itself estimated from a
    finite number of repeat transients, so it is a random variable in its own
    right: its relative scatter is of order 1/sqrt(2N), and a gate that happens
    to stack quietly can report a few tenths of a percent where the true
    repeatability is nearer a few percent. The gate enters the misfit weighted by
    the reciprocal of its error, so such a gate outweighs its neighbours by two
    orders of magnitude, and on a station carrying four or five gates it alone
    decides the model. Flooring at the repeatability the instrument can resolve
    removes that failure mode and leaves the relative weighting of every gate
    above the floor unchanged.

    The ceiling limits how far a noisy gate may be down-weighted: clipping a
    recorded 42 % error to 25 % makes that gate *more* influential, not less.
    It is useful only when very large reported errors would otherwise remove a
    retained gate from the fit in practice. To reject noisy gates, use the
    reader's ``max_relative_std`` setting instead. Both bounds default to off.
    """
    data_rel = np.asarray(item.get("relative_std", []), dtype=float).ravel()
    if data_rel.size == observed.size:
        data_rel = np.where(np.isfinite(data_rel) & (data_rel > 0.0), data_rel, 0.0)
        # Clamp only where a stack error was actually recorded. A gate whose
        # column is empty carries 0.0 as "unknown", and lifting that to the floor
        # would invent an error the file never claimed.
        recorded = data_rel > 0.0
        if min_rel > 0.0:
            data_rel = np.where(recorded, np.maximum(data_rel, float(min_rel)), data_rel)
        if max_rel is not None:
            data_rel = np.where(recorded, np.minimum(data_rel, float(max_rel)), data_rel)
        # Add only the shortfall. Where a file states that its recorded error
        # already carries a uniform term, that term is part of ``rel`` rather
        # than additional to it, and adding the whole of ``rel`` again would
        # count the same physical error twice. A gate with no recorded error
        # carries no baked-in term either, so ``rel`` is its whole budget.
        baked = float(item.get("uniform_error", 0.0) or 0.0)
        extra = math.sqrt(max(float(rel) ** 2 - max(baked, 0.0) ** 2, 0.0))
        total = np.where(recorded, np.hypot(data_rel, extra), float(rel))
        return total * np.abs(observed) + floor
    return rel * np.abs(observed) + floor


def _occam_1d(forward_vec: Callable[[np.ndarray], np.ndarray], dobs_vec: np.ndarray,
              unc_vec: np.ndarray, n_layers: int, inv: Dict[str, Any], log: LogFn,
              jacobian_vec: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """Smooth fixed-layer fit with optional LCI neighbor regularization.

    ``jacobian_vec`` returns ``d(predicted) / d(sigma)`` shaped
    ``(n_data, n_layers)``. Given one, the optimizer is handed the analytic
    derivative of the whole residual instead of differencing it: SciPy's
    two-point rule costs one extra forward call per layer per step, which on a
    20-layer model is 20 forwards spent to learn what one sensitivity call
    already knows. Without one the numerical route still works, so a forward
    operator that cannot supply a sensitivity keeps running.
    """
    from scipy.optimize import least_squares
    lam = float(inv.get("smoothness", 0.3))
    start_res = float(inv.get("starting_resistivity", 100.0))
    lo, hi = _log_resistivity_bounds(inv)
    start_model = np.asarray(inv.get("starting_model", []), dtype=float).ravel()
    if start_model.size == n_layers and np.all(np.isfinite(start_model)):
        x0 = np.clip(np.log10(np.clip(start_model, 10.0 ** lo, 10.0 ** hi)), lo, hi)
    else:
        x0 = float(np.clip(np.log10(max(start_res, 1.0)), lo, hi)) * np.ones(n_layers)
    lateral_model = np.asarray(inv.get("lateral_reference", []), dtype=float).ravel()
    lateral_weight = float(inv.get("lateral_weight", 0.0))
    if (
        lateral_model.size == n_layers
        and np.all(np.isfinite(lateral_model))
        and lateral_weight > 0.0
    ):
        lateral_log = np.log10(np.clip(lateral_model, 10.0 ** lo, 10.0 ** hi))
    else:
        lateral_log = np.array([], dtype=float)
    unc_vec = np.clip(unc_vec, 1e-30, None)
    from .em1d_priors import shallow_prior_terms
    prior_lower, prior_weights = shallow_prior_terms(inv, _inversion_layer_thicknesses(
        {**inv, "n_layers": n_layers}))
    convergence: List[float] = []

    def residual(logres: np.ndarray) -> np.ndarray:
        sigma = 1.0 / np.power(10.0, logres)
        pred = np.asarray(forward_vec(sigma), dtype=float)
        data_res = (pred - dobs_vec) / unc_vec
        smooth = lam * np.diff(logres)
        lateral = (
            lateral_weight * (logres - lateral_log)
            if lateral_log.size else np.array([], dtype=float)
        )
        prior = prior_weights * np.maximum(prior_lower - logres, 0.)
        return np.concatenate([data_res, smooth, lateral, prior])

    # The two regularization blocks are linear in logres, so their rows are the
    # same matrix at every model and are built once here. Only the data rows
    # need the forward operator.
    smooth_rows = lam * (np.eye(n_layers, k=1) - np.eye(n_layers))[:n_layers - 1]
    lateral_rows = (lateral_weight * np.eye(n_layers) if lateral_log.size
                    else np.zeros((0, n_layers)))

    def jacobian(logres: np.ndarray) -> np.ndarray:
        """d(residual) / d(log10 resistivity), all three blocks.

        The chain rule through ``sigma = 10**(-logres)`` contributes
        ``d sigma / d logres = -ln(10) * sigma``, the same factor the coupled
        line solver applies in :func:`em1d_lci._sensitivity_line`.
        """
        sigma = 1.0 / np.power(10.0, logres)
        jac = np.asarray(jacobian_vec(sigma), dtype=float)
        if jac.shape != (dobs_vec.size, n_layers):
            raise ValueError(
                f"the forward operator returned a Jacobian of shape {jac.shape}, "
                f"expected {(dobs_vec.size, n_layers)}.")
        data_rows = (jac * (-_LN10 * sigma)[None, :]) / unc_vec[:, None]
        prior_rows = np.diag(-prior_weights * (logres < prior_lower))
        return np.vstack([data_rows, smooth_rows, lateral_rows, prior_rows])

    # SciPy's ``max_nfev`` is the number of outer residual evaluations. Numerical
    # Jacobian probes are additional calls, so multiplying by ``n_layers`` here
    # makes a 20-layer line inversion unnecessarily hundreds of evaluations long.
    max_nfev = max(4, int(inv.get("max_iterations", 30)))
    def on_iteration(intermediate) -> None:
        """Record the data-only chi-square from SciPy's outer optimizer."""
        fun = getattr(intermediate, "fun", None)
        if fun is None:
            x = np.asarray(getattr(intermediate, "x", intermediate), dtype=float)
            fun = residual(x)
        data_res = np.asarray(fun, dtype=float).ravel()[:dobs_vec.size]
        convergence.append(float(np.mean(data_res ** 2)))

    kwargs: Dict[str, Any] = {"max_nfev": max_nfev, "xtol": 1e-8, "ftol": 1e-8}
    if jacobian_vec is not None:
        kwargs["jac"] = jacobian
    # SciPy < 1.16 has no callback argument. Keep those supported environments
    # working; their quality page falls back to the final chi-square chart.
    try:
        import inspect
        if "callback" in inspect.signature(least_squares).parameters:
            kwargs["callback"] = on_iteration
    except Exception:  # pragma: no cover - defensive compatibility path
        pass
    sol = least_squares(residual, x0, bounds=(lo, hi), method="trf", **kwargs)
    res = np.power(10.0, sol.x)
    data_res = residual(sol.x)[: dobs_vec.size]
    chi2 = float(np.mean(data_res ** 2))
    if not convergence or not np.isclose(convergence[-1], chi2):
        convergence.append(chi2)
    log(f"  inversion done: {sol.nfev} forward evals, chi2={chi2:.3f}")
    return res, chi2, int(sol.nfev), convergence


def _occam_with_optional_rejection(
    forward_vec: Callable[[np.ndarray], np.ndarray],
    dobs_vec: np.ndarray,
    unc_vec: np.ndarray,
    n_layers: int,
    inv: Dict[str, Any],
    log: LogFn,
    jacobian_vec: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """Run Occam with optional robust errors, or legacy hard rejection.

    Robust errors take precedence and preserve every gate. The simultaneous
    line solver has its own survey-wide reweighting/rejection routines.
    This counterpart covers a single sounding, independent line inversion and
    sequential LCI without changing their solver when rejection is disabled.
    The retained fraction is measured against the sounding's original data and
    every re-fit is warm-started from the previous model.
    """
    dobs_vec = np.asarray(dobs_vec, dtype=float).ravel()
    unc_vec = np.asarray(unc_vec, dtype=float).ravel()
    if dobs_vec.size != unc_vec.size:
        raise ValueError("observed data and uncertainty must have the same length.")
    keep = np.ones(dobs_vec.size, dtype=bool)
    if bool(inv.get("robust_errors", False)):
        from .robust_errors import reweight_errors, robust_error_options

        if inv.get("reject_outliers", False):
            log("Robust errors enabled: hard rejection is bypassed; all gates retained.")
        total_nfev = 0
        histories = []

        def robust_solve(effective, previous):
            nonlocal total_nfev
            settings = dict(inv)
            if previous is not None:
                settings["starting_model"] = previous[0]
            fitted = _occam_1d(forward_vec, dobs_vec, effective, n_layers,
                               settings, log, jacobian_vec)
            total_nfev += int(fitted[2])
            histories.extend(fitted[3])
            return fitted

        fitted, _, robust = reweight_errors(
            dobs_vec, unc_vec, robust_solve,
            lambda fit: forward_vec(1.0 / np.clip(fit[0], 1e-12, None)),
            history=lambda fit: fit[3], log=log, **robust_error_options(inv))
        # Keep the established result layout; report raw-error chi2 as the main
        # metric so increasing errors cannot masquerade as a better raw fit.
        info = {"enabled": False, "n_start": int(keep.size), "kept": int(keep.size),
                "dropped": 0, "passes": [], "robust": robust,
                "stopped_because": "robust error weighting; no hard rejection"}
        return fitted[0], robust["chi2_original"], total_nfev, histories, keep, info
    enabled = bool(inv.get("reject_outliers", False))
    threshold = float(inv.get("outlier_threshold", 3.0))
    passes = max(0, int(inv.get("outlier_passes", 2)))
    fraction = float(np.clip(inv.get("min_data_fraction", 0.8), 0.0, 1.0))
    min_gates = max(0, int(inv.get("min_gates_per_sounding", 3)))
    floor = min(dobs_vec.size, max(
        int(math.ceil(fraction * dobs_vec.size)), min(min_gates, dobs_vec.size)))
    info: Dict[str, Any] = {
        "enabled": enabled,
        "threshold": threshold,
        "n_start": int(dobs_vec.size),
        "floor": int(floor),
        "passes": [],
        "kept": int(dobs_vec.size),
        "dropped": 0,
        "stopped_because": "disabled" if not enabled else "",
    }
    histories: List[float] = []
    total_nfev = 0
    working_inv = dict(inv)

    def solve(mask: np.ndarray):
        def selected_forward(sigma: np.ndarray) -> np.ndarray:
            return np.asarray(forward_vec(sigma), dtype=float).ravel()[mask]

        selected_jacobian = None
        if jacobian_vec is not None:
            def selected_jacobian(sigma: np.ndarray) -> np.ndarray:
                return np.asarray(jacobian_vec(sigma), dtype=float)[mask, :]

        return _occam_1d(
            selected_forward, dobs_vec[mask], unc_vec[mask], n_layers,
            working_inv, log, selected_jacobian)

    res, chi2, nfev, convergence = solve(keep)
    total_nfev += int(nfev)
    histories.extend(convergence)
    if enabled and not passes:
        info["stopped_because"] = "no rejection passes requested"
    if not enabled or not passes or not dobs_vec.size:
        return res, chi2, total_nfev, histories, keep, info

    for pass_index in range(1, passes + 1):
        sigma = 1.0 / np.clip(res, 1e-12, None)
        predicted = np.asarray(forward_vec(sigma), dtype=float).ravel()
        residual = np.abs((predicted - dobs_vec) / np.clip(unc_vec, 1e-30, None))
        candidates = np.flatnonzero(keep & (residual > threshold))
        allowed = int(keep.sum()) - floor
        if not candidates.size:
            info["stopped_because"] = "nothing left above the cut"
            break
        if allowed <= 0:
            info["stopped_because"] = f"at the {fraction:.0%} floor"
            break
        if candidates.size > allowed:
            order = candidates[np.argsort(-residual[candidates])]
            candidates = order[:allowed]
        keep[candidates] = False
        working_inv = {**working_inv, "starting_model": np.asarray(res, dtype=float)}
        res, chi2, nfev, convergence = solve(keep)
        total_nfev += int(nfev)
        histories.extend(convergence)
        info["passes"].append({
            "pass": pass_index,
            "dropped": int(candidates.size),
            "kept": int(keep.sum()),
            "chi2": float(chi2),
        })
        log(
            f"  rejected {candidates.size} gate(s) over {threshold:g} sigma, "
            f"{int(keep.sum())} left -> chi2 {chi2:.3f}")
        if int(keep.sum()) == floor:
            info["stopped_because"] = f"at the {fraction:.0%} floor"
            break
    if not info["stopped_because"]:
        info["stopped_because"] = f"all {passes} pass(es) used"
    info["kept"] = int(keep.sum())
    info["dropped"] = int(dobs_vec.size - keep.sum())
    return res, chi2, total_nfev, histories, keep, info


#: One warmed-up forward operator per worker thread, per distinct survey.
#:
#: ``Simulation1DLayered`` does a substantial one-time setup on its first call
#: and caches it on the instance: measured on a ground TDEM station it is around
#: 8 s the first time and 18 ms afterwards, and the setup grows with the number
#: of receiver times, so modelling the receiver's analog filter makes it an order
#: of magnitude more expensive than it is without.
#:
#: A line inversion re-uses one layer grid and one system geometry for every
#: station, so a fresh instance per station pays that setup hundreds of times
#: over for nothing. It cannot simply be shared, because two threads calling
#: ``getJ`` on one instance corrupt the state it caches. Keeping the instances in
#: thread-local storage gives one per worker rather than one per station, which
#: is the saving without the race: on a 600-station line that is around ten
#: warm-ups instead of twelve hundred.
_MODELER_CACHE = threading.local()

#: How many distinct surveys one thread keeps warm. A joint LM+HM run needs two
#: per distinct geometry, and a bounded number keeps a long survey from holding
#: every simulation it ever built.
#:
#: Eight was enough while a line shared one geometry. It is not enough now that
#: the transmitter-receiver distance is read per station: a survey binned at
#: :data:`~PyHydroGeophysX.workflows.em1d.STATION_DISTANCE_BIN_M` presents about
#: twenty-five distinct distances, so two moments need fifty entries and a cache
#: of eight evicts an operator before it is reused. That is the difference
#: between fourteen seconds and twenty milliseconds per station.
_MODELER_CACHE_SIZE = 64


def _modeler_key(thick: np.ndarray, config: Any) -> tuple:
    """Identity of a forward operator: same key means the instance is reusable.

    Everything the operator was built from has to appear here. Two stations that
    differ only in sensor height need different instances, and silently sharing
    one would return another station's geometry with no sign that it had.
    """
    def _blob(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tobytes()
        if isinstance(value, (list, tuple)):
            return tuple(_blob(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), _blob(v)) for k, v in value.items()))
        return value

    fields = tuple(sorted(
        (name, _blob(getattr(config, name)))
        for name in vars(config)
    ))
    return (np.asarray(thick, dtype=float).tobytes(), fields)


def _support_times(times: np.ndarray) -> np.ndarray:
    """Station gate times, padded so the forward has something to interpolate on.

    SimPEG builds a spline over the receiver times and needs more of them than a
    heavily gated station provides. The padding is forward-only support: the
    residual still uses the station's own gates, and no observed value is
    interpolated or invented.
    """
    times = np.asarray(times, dtype=float).ravel()
    if times.size >= 5:
        return times
    lower = max(float(np.min(times)) * 0.5, 1e-9)
    upper = max(float(np.max(times)) * 2.0, lower * 4.0)
    return np.unique(np.concatenate([times, np.geomspace(lower, upper, 7)]))


def _moment_gate_times(geometry: Dict[str, Any], times: np.ndarray) -> np.ndarray:
    """Every gate centre the moment records, or the station's own as a fallback.

    Modelling the instrument's whole gate set rather than one station's surviving
    subset is what lets a line share a single forward operator: the receiver
    times then depend on the moment, not on which gates that station happened to
    keep. It also gives the analog-filter reconstruction more nodes to work with
    than a three-gate station would, so the shared operator is at least as
    accurate as the per-station one it replaces.
    """
    windows = geometry.get("gate_windows") or {}
    centres = np.asarray(windows.get("centre", []), dtype=float).ravel()
    if centres.size >= 2 and np.all(np.isfinite(centres)) and np.all(centres > 0.0):
        return np.unique(centres)
    return _support_times(times)


def _thread_local_modeler(thick: np.ndarray, geometry: Dict[str, Any],
                          model_times: np.ndarray):
    """A forward operator for this survey, warmed up once per worker thread.

    Call this from the thread that will use the result, not from the thread that
    assembles the work. SimPEG's simulation caches the sensitivity on itself, so
    two threads calling ``getJ`` on one instance race: one of them can read the
    cache mid-write and get ``None`` back where a matrix should be. Building the
    blocks on the main thread and handing the same instance to a pool of workers
    is exactly that race, and it surfaces as a Jacobian with no dimensions.

    The cache is keyed on the whole survey configuration, so a line re-uses one
    warmed-up simulation per moment per thread rather than building one per
    station. A pool of N workers pays the setup N times instead of once, which
    on a 540-station line is still two orders of magnitude fewer builds.
    """
    try:
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    config = _tdem_config(geometry, model_times)
    cache = getattr(_MODELER_CACHE, "items", None)
    if cache is None:
        cache = _MODELER_CACHE.items = {}
    key = _modeler_key(thick, config)
    modeler = cache.get(key)
    if modeler is None:
        modeler = TDEMForwardModeling(thicknesses=thick, survey_config=config)
        if len(cache) >= _MODELER_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[key] = modeler
    return modeler


def tdem_moment_blocks(data: Dict[str, Any], geom: Dict[str, Any],
                       inv: Dict[str, Any], thick: np.ndarray) -> List[Dict[str, Any]]:
    """One forward block per usable moment at a TDEM station.

    Stations carrying separate ``LM`` and ``HM`` gate sets produce one block
    each; a plain single-response station produces one block named ``TDEM``.
    Returning the same shape for both is what lets the per-sounding inversion
    and the coupled line inversion share this assembly instead of each writing
    its own copy of the uncertainty and gate-selection rules.
    """
    try:
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))

    scale = float(inv.get("data_scale", 1.0))
    rel = float(inv.get("rel_error", 0.05))
    floor = float(inv.get("noise_floor", 1e-18)) * scale
    min_rel = float(inv.get("min_rel_error", 0.0))
    max_rel = inv.get("max_rel_error")
    max_rel = float(max_rel) if max_rel is not None else None
    sign = float(geom.get("response_sign", 1.0))
    moments = dict(data.get("moments", {}))
    items = ([(name, dict(moments[name])) for name in ("LM", "HM") if name in moments]
             or [("TDEM", dict(data))])

    blocks: List[Dict[str, Any]] = []
    for name, item in items:
        times = np.asarray(item.get("times", []), dtype=float).ravel()
        observed = np.asarray(item.get("response", []), dtype=float).ravel() * scale
        if not times.size or observed.size != times.size:
            continue
        # The turn-off ramp and the gate windows belong to the moment, not to
        # the station, so the geometry a block needs is the moment's.
        geometry = _tdem_geometry(data, geom, item.get("transmitter"))
        # Model every gate the instrument records, not just the ones this station
        # kept, and select afterwards. The forward operator then depends on the
        # moment and the layer grid alone, so a line re-uses one warmed-up
        # simulation instead of building one per station. See _MODELER_CACHE for
        # what that is worth. Modelling gates a station rejected costs nothing:
        # they are never compared against an observation.
        model_times = _moment_gate_times(geometry, times)
        channel_indices = np.asarray([
            int(np.argmin(np.abs(model_times - value))) for value in times
        ], dtype=int)
        if not np.allclose(model_times[channel_indices], times, rtol=1e-6, atol=0.0):
            # A station whose gates are not on the instrument's own grid, which a
            # re-binned or hand-edited file can produce. Fall back to modelling
            # exactly what it holds rather than selecting the wrong channels.
            model_times = _support_times(times)
            channel_indices = np.asarray([
                int(np.argmin(np.abs(model_times - value))) for value in times
            ], dtype=int)
        blocks.append({
            "name": name,
            "times": times,
            "model_times": model_times,
            "channel_indices": channel_indices,
            "observed": observed,
            "uncertainty": _tdem_uncertainty(observed, item, rel, floor,
                                             min_rel, max_rel),
            "sign": sign,
            # What the operator is, rather than the operator itself. Blocks are
            # assembled here on one thread and evaluated on a pool of workers,
            # so each worker resolves its own instance at call time; see
            # _thread_local_modeler for why sharing one is not safe.
            "thicknesses": thick,
            "geometry": geometry,
        })
    if not blocks:
        raise ValueError("The joint TDEM sounding has no usable LM or HM gates.")
    return blocks


def _block_modeler(item: Dict[str, Any]):
    """The forward operator for one block, resolved on the calling thread."""
    return _thread_local_modeler(
        item["thicknesses"], item["geometry"], item["model_times"])


def _moment_forward(blocks: List[Dict[str, Any]]) -> Callable[[np.ndarray], np.ndarray]:
    """Predicted response for the gates that were actually measured."""
    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        return np.concatenate([
            item["sign"] * np.asarray(
                _block_modeler(item).forward(sigma), dtype=float).ravel()[
                    : item["model_times"].size][item["channel_indices"]]
            for item in blocks
        ])
    return forward_vec


def _moment_jacobian(blocks: List[Dict[str, Any]]) -> Callable[[np.ndarray], np.ndarray]:
    """Analytic d(response)/d(sigma), row-selected to match ``_moment_forward``."""
    def jacobian(sigma: np.ndarray) -> np.ndarray:
        return np.vstack([
            item["sign"] * np.asarray(
                _block_modeler(item).sensitivity(sigma), dtype=float)[
                    : item["model_times"].size][item["channel_indices"], :]
            for item in blocks
        ])
    return jacobian


def _fdem_pieces(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                 thick: np.ndarray):
    """Observed vector, uncertainty, forward, and Jacobian for one FDEM sounding.

    Data are ordered ``[real over frequency, imag over frequency]``, matching
    :func:`fdem_invert`. SimPEG's ``dpred`` interleaves the two, so the same row
    selection is applied to the response and to the Jacobian; that shared
    selection is what keeps the two consistent.
    """
    try:
        from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))

    freqs = np.asarray(data["frequencies"], dtype=float).ravel()
    scale = float(inv.get("data_scale", 1.0))
    obs_r = np.asarray(data["real"], dtype=float).ravel() * scale
    obs_i = np.asarray(data["imag"], dtype=float).ravel() * scale
    rel = float(inv.get("rel_error", 0.05))
    floor = float(inv.get("noise_floor", 1e-14)) * scale
    amp = np.abs(obs_r + 1j * obs_i)
    observed = np.concatenate([obs_r, obs_i])
    uncertainty = np.concatenate([rel * amp + floor, rel * amp + floor])
    modeler = FDEMForwardModeling(
        thicknesses=thick, survey_config=_fdem_config(geom, freqs))
    nf = int(freqs.size)
    rows = np.concatenate([np.arange(0, 2 * nf, 2), np.arange(1, 2 * nf, 2)])

    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        resp = np.asarray(modeler.forward(sigma)).ravel()
        if resp.size == 2 * nf and not np.iscomplexobj(resp):
            resp = resp[0::2] + 1j * resp[1::2]
        resp = np.asarray(resp, dtype=complex).ravel()[:nf]
        return np.concatenate([resp.real, resp.imag])

    def jacobian(sigma: np.ndarray) -> np.ndarray:
        return np.asarray(
            modeler.simulation.getJ(sigma), dtype=float)[rows, :]

    return observed, uncertainty, forward_vec, jacobian, modeler, freqs


def build_sounding_block(data: Dict[str, Any], geom: Dict[str, Any],
                         inv: Dict[str, Any], method: str = "TDEM", *,
                         position: float = 0.0, line: int = 0, label: str = ""):
    """Package one sounding for the coupled line inversion.

    The observed vector, uncertainty, and forward operator are built by the
    same code the per-sounding inversion uses, so a station fits the same data
    whether it is solved alone or as part of a line. The Jacobian comes from
    SimPEG's analytic sensitivity rather than from finite differences.
    """
    from PyHydroGeophysX.inversion.em1d_lci import SoundingBlock

    thick = _inversion_layer_thicknesses(inv)
    if str(method).upper() == "FDEM":
        observed, uncertainty, forward_vec, jacobian, _, _ = _fdem_pieces(
            data, geom, inv, thick)
    else:
        blocks = tdem_moment_blocks(data, geom, inv, thick)
        observed = np.concatenate([item["observed"] for item in blocks])
        uncertainty = np.concatenate([item["uncertainty"] for item in blocks])
        forward_vec = _moment_forward(blocks)
        jacobian = _moment_jacobian(blocks)
    from .em1d_priors import shallow_prior_terms
    prior_lower, prior_weights = shallow_prior_terms(inv, thick)
    return SoundingBlock(
        forward=forward_vec, jacobian=jacobian, dobs=observed,
        uncertainty=uncertainty, position=float(position), line=int(line),
        label=str(label), prior_lower=prior_lower, prior_weights=prior_weights)


def fdem_invert(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert an FDEM sounding for a layered resistivity model (Occam 1D)."""
    n_layers = int(inv.get("n_layers", 15))
    thick = _inversion_layer_thicknesses(inv)
    dobs_vec, unc_vec, forward_vec, jacobian_vec, _, freqs = _fdem_pieces(
        data, geom, inv, thick)
    obs_r, obs_i = dobs_vec[: freqs.size], dobs_vec[freqs.size:]
    log(f"FDEM inversion: {freqs.size} freqs, {n_layers} layers")

    res, chi2, nfev, convergence, fit_mask, outliers = (
        _occam_with_optional_rejection(
            forward_vec, dobs_vec, unc_vec, n_layers, inv, log, jacobian_vec))
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    pred_r, pred_i = pred[: freqs.size], pred[freqs.size:]
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "FDEM", "frequencies": freqs,
            "obs_real": obs_r, "obs_imag": obs_i,
            "pred_real": pred_r, "pred_imag": pred_i,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2,
            "n_data": int(fit_mask.sum()), "fit_mask": fit_mask,
            "outliers": outliers, "robust": outliers.get("robust", {"enabled": False}),
            "chi2_effective": outliers.get("robust", {}).get("chi2_effective", chi2),
            "nfev": nfev, "n_layers": n_layers,
            "convergence": convergence}


def tdem_invert(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert a TDEM sounding for a layered resistivity model (Occam 1D)."""
    try:
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    times = np.asarray(data["times"], dtype=float).ravel()
    scale = float(inv.get("data_scale", 1.0))
    dobs = np.asarray(data["response"], dtype=float).ravel() * scale
    rel = float(inv.get("rel_error", 0.05)); floor = float(inv.get("noise_floor", 1e-18)) * scale
    min_rel = float(inv.get("min_rel_error", 0.0))
    max_rel = inv.get("max_rel_error")
    max_rel = float(max_rel) if max_rel is not None else None
    unc = _tdem_uncertainty(dobs, data, rel, floor, min_rel, max_rel)
    n_layers = int(inv.get("n_layers", 15))
    thick = _inversion_layer_thicknesses(inv)
    geometry = _tdem_geometry(data, geom)
    modeler = TDEMForwardModeling(
        thicknesses=thick, survey_config=_tdem_config(geometry, times))
    log(f"TDEM inversion: {times.size} times, {n_layers} layers")
    sign = float(geometry.get("response_sign", 1.0))

    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        return sign * np.asarray(modeler.forward(sigma), dtype=float).ravel()[: times.size]

    def jacobian_vec(sigma: np.ndarray) -> np.ndarray:
        return sign * np.asarray(modeler.sensitivity(sigma), dtype=float)[: times.size]

    res, chi2, nfev, convergence, fit_mask, outliers = (
        _occam_with_optional_rejection(
            forward_vec, dobs, unc, n_layers, inv, log, jacobian_vec))
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "TDEM", "times": times, "obs": dobs, "pred": pred,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2,
            "n_data": int(fit_mask.sum()), "fit_mask": fit_mask,
            "outliers": outliers, "robust": outliers.get("robust", {"enabled": False}),
            "chi2_effective": outliers.get("robust", {}).get("chi2_effective", chi2),
            "nfev": nfev, "n_layers": n_layers,
            "convergence": convergence}


def tdem_joint_invert(
    data: Dict[str, Any],
    geom: Dict[str, Any],
    inv: Dict[str, Any],
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Invert all available LM/HM gates at one station for one shared 1D model."""
    try:
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    moments = dict(data.get("moments", {}))
    if not moments:
        return tdem_invert(data, geom, inv, log=log)

    n_layers = int(inv.get("n_layers", 15))
    thick = _inversion_layer_thicknesses(inv)
    blocks = tdem_moment_blocks(data, geom, inv, thick)

    observed = np.concatenate([item["observed"] for item in blocks])
    uncertainty = np.concatenate([item["uncertainty"] for item in blocks])
    log(
        "Joint LM+HM inversion: "
        + ", ".join(f"{item['name']}={item['times'].size}" for item in blocks)
        + f" gates, {n_layers} shared layers"
    )

    forward_vec = _moment_forward(blocks)
    res, chi2, nfev, convergence, fit_mask, outliers = (
        _occam_with_optional_rejection(
            forward_vec, observed, uncertainty, n_layers, inv, log,
            _moment_jacobian(blocks)))
    sigma = 1.0 / np.clip(res, 1e-12, None)
    predicted = forward_vec(sigma)
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    offset = 0
    for item in blocks:
        count = int(item["times"].size)
        predictions[item["name"]] = {
            "times": item["times"],
            "obs": item["observed"],
            "pred": predicted[offset:offset + count],
            "fit_mask": fit_mask[offset:offset + count],
        }
        if outliers.get("robust", {}).get("enabled"):
            for key in ("uncertainty_original", "uncertainty_effective", "weights"):
                predictions[item["name"]][key] = np.asarray(
                    outliers["robust"][key][offset:offset + count])
        offset += count
    depth, res_step = model_depth_profile(thick, res)
    return {
        "method": "TDEM",
        "joint_moments": True,
        "moments": predictions,
        "obs": observed,
        "pred": predicted,
        "thickness": thick,
        "resistivity": res,
        "conductivity": sigma,
        "depth": depth,
        "resistivity_step": res_step,
        "chi2": chi2,
        "n_data": int(fit_mask.sum()),
        "fit_mask": fit_mask,
        "outliers": outliers,
        "robust": outliers.get("robust", {"enabled": False}),
        "chi2_effective": outliers.get("robust", {}).get("chi2_effective", chi2),
        "nfev": nfev,
        "n_layers": n_layers,
        "convergence": convergence,
    }

__all__ = [
    "DEFAULT_INVERSION",
    "build_sounding_block",
    "fdem_invert",
    "tdem_invert",
    "tdem_joint_invert",
    "tdem_moment_blocks",
]
