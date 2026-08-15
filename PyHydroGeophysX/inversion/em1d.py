"""Occam-style 1D FDEM/TDEM inversion helpers."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_LN10 = math.log(10.0)

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX._internal.utils import noop as _noop
from PyHydroGeophysX.forward.em1d import (
    _fdem_config,
    _tdem_config,
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
    "lateral_weight_scale": 1.0, "auto_lambda": True,
    "target_chi2": 1.0, "chi2_tolerance": 0.2, "max_lambda_trials": 5,
    "convergence_tolerance": 0.02, "min_iterations": 2,
}


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


def _tdem_uncertainty(observed: np.ndarray, item: Dict[str, Any],
                      rel: float, floor: float) -> np.ndarray:
    """Per-gate uncertainty: the recorded stack error with ``rel`` added in quadrature.

    Two independent things go wrong with a gate. Its stack error says how
    repeatably it was measured, and the file records one per gate. Everything
    else, system calibration and the error in representing the ground as 1D
    layers, applies to every gate alike and the stack error knows nothing about
    it; that is what ``rel`` carries.

    Adding them in quadrature is how independent errors combine, and it is what
    the instrument vendor's own documentation describes ("calculated based on
    the signal-to-noise ratio and with a 3 % uniform uncertainty added"). It also
    behaves better than taking the larger of the two, which was the previous
    rule: quadrature leaves a gate that is already noisy essentially untouched,
    so the instrument's relative weighting between clean and noisy gates
    survives, where a floor flattens every gate below it to the same weight.

    Where the file carries no stack error for a gate, ``rel`` is the whole
    budget, so a partially populated column is still usable.
    """
    data_rel = np.asarray(item.get("relative_std", []), dtype=float).ravel()
    if data_rel.size == observed.size:
        data_rel = np.where(np.isfinite(data_rel) & (data_rel > 0.0), data_rel, 0.0)
        total = np.hypot(data_rel, float(rel))
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
    lo, hi = 0.0, 5.0  # log10 resistivity bounds (1 .. 1e5 ohm-m)
    start_model = np.asarray(inv.get("starting_model", []), dtype=float).ravel()
    if start_model.size == n_layers and np.all(np.isfinite(start_model)):
        x0 = np.clip(np.log10(np.clip(start_model, 1.0, 1e5)), lo, hi)
    else:
        x0 = float(np.clip(np.log10(max(start_res, 1.0)), lo, hi)) * np.ones(n_layers)
    lateral_model = np.asarray(inv.get("lateral_reference", []), dtype=float).ravel()
    lateral_weight = float(inv.get("lateral_weight", 0.0))
    if (
        lateral_model.size == n_layers
        and np.all(np.isfinite(lateral_model))
        and lateral_weight > 0.0
    ):
        lateral_log = np.log10(np.clip(lateral_model, 1.0, 1e5))
    else:
        lateral_log = np.array([], dtype=float)
    unc_vec = np.clip(unc_vec, 1e-30, None)
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
        return np.concatenate([data_res, smooth, lateral])

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
        return np.vstack([data_rows, smooth_rows, lateral_rows])

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
        # SimPEG constructs a spline over receiver times and needs a denser time
        # vector than some heavily gated TEMcompany stations provide. Add forward-
        # only support times, then select the original gates for the residual.
        # No observed values are interpolated or invented.
        model_times = times
        if times.size < 5:
            lower = max(float(np.min(times)) * 0.5, 1e-9)
            upper = max(float(np.max(times)) * 2.0, lower * 4.0)
            model_times = np.unique(np.concatenate([
                times, np.geomspace(lower, upper, 7)
            ]))
        # The turn-off ramp and the gate windows belong to the moment, not to
        # the station, so the geometry a block needs is the moment's.
        geometry = {**geom, **(item.get("transmitter") or {})}
        # One simulation per station. Sharing them between stations would save
        # SimPEG's first-call setup, which is minutes on a long line, but a
        # Simulation1DLayered caches state on the instance and two threads
        # calling getJ on the same one corrupt it; the per-sounding threading is
        # worth more than the setup it would save.
        modeler = TDEMForwardModeling(
            thicknesses=thick, survey_config=_tdem_config(geometry, model_times))
        channel_indices = np.asarray([
            int(np.argmin(np.abs(model_times - value))) for value in times
        ], dtype=int)
        blocks.append({
            "name": name,
            "times": times,
            "model_times": model_times,
            "channel_indices": channel_indices,
            "observed": observed,
            "uncertainty": _tdem_uncertainty(observed, item, rel, floor),
            "sign": sign,
            "modeler": modeler,
        })
    if not blocks:
        raise ValueError("The joint TDEM sounding has no usable LM or HM gates.")
    return blocks


def _moment_forward(blocks: List[Dict[str, Any]]) -> Callable[[np.ndarray], np.ndarray]:
    """Predicted response for the gates that were actually measured."""
    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        return np.concatenate([
            item["sign"] * np.asarray(
                item["modeler"].forward(sigma), dtype=float).ravel()[
                    : item["model_times"].size][item["channel_indices"]]
            for item in blocks
        ])
    return forward_vec


def _moment_jacobian(blocks: List[Dict[str, Any]]) -> Callable[[np.ndarray], np.ndarray]:
    """Analytic d(response)/d(sigma), row-selected to match ``_moment_forward``."""
    def jacobian(sigma: np.ndarray) -> np.ndarray:
        return np.vstack([
            item["sign"] * np.asarray(
                item["modeler"].sensitivity(sigma), dtype=float)[
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
    return SoundingBlock(
        forward=forward_vec, jacobian=jacobian, dobs=observed,
        uncertainty=uncertainty, position=float(position), line=int(line),
        label=str(label))


def fdem_invert(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert an FDEM sounding for a layered resistivity model (Occam 1D)."""
    n_layers = int(inv.get("n_layers", 15))
    thick = _inversion_layer_thicknesses(inv)
    dobs_vec, unc_vec, forward_vec, jacobian_vec, _, freqs = _fdem_pieces(
        data, geom, inv, thick)
    obs_r, obs_i = dobs_vec[: freqs.size], dobs_vec[freqs.size:]
    log(f"FDEM inversion: {freqs.size} freqs, {n_layers} layers")

    res, chi2, nfev, convergence = _occam_1d(
        forward_vec, dobs_vec, unc_vec, n_layers, inv, log, jacobian_vec)
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    pred_r, pred_i = pred[: freqs.size], pred[freqs.size:]
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "FDEM", "frequencies": freqs,
            "obs_real": obs_r, "obs_imag": obs_i,
            "pred_real": pred_r, "pred_imag": pred_i,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2,
            "n_data": int(dobs_vec.size), "nfev": nfev, "n_layers": n_layers,
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
    unc = _tdem_uncertainty(dobs, data, rel, floor)
    n_layers = int(inv.get("n_layers", 15))
    thick = _inversion_layer_thicknesses(inv)
    modeler = TDEMForwardModeling(thicknesses=thick, survey_config=_tdem_config(geom, times))
    log(f"TDEM inversion: {times.size} times, {n_layers} layers")
    sign = float(geom.get("response_sign", 1.0))

    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        return sign * np.asarray(modeler.forward(sigma), dtype=float).ravel()[: times.size]

    def jacobian_vec(sigma: np.ndarray) -> np.ndarray:
        return sign * np.asarray(modeler.sensitivity(sigma), dtype=float)[: times.size]

    res, chi2, nfev, convergence = _occam_1d(
        forward_vec, dobs, unc, n_layers, inv, log, jacobian_vec)
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "TDEM", "times": times, "obs": dobs, "pred": pred,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2,
            "n_data": int(dobs.size), "nfev": nfev, "n_layers": n_layers,
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
    res, chi2, nfev, convergence = _occam_1d(
        forward_vec, observed, uncertainty, n_layers, inv, log,
        _moment_jacobian(blocks))
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
        }
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
        "n_data": int(observed.size),
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
