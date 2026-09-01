"""Empirical resistive-background regularisation, separate from data errors.

A sustained weak absolute LM response may be consistent with a resistive
half-space/background, but it does not identify a shallow layer or its depth.
The optional prior therefore applies a weak, one-sided tendency to the whole
1-D model. It is not independent geological evidence and may also be triggered
by interference, coupling or acquisition changes. It never enters data chi2 or
data-only DOI.

The historical ``shallow_prior_*`` option names remain readable so existing
project files and notebooks do not break. Their current physical interpretation
is the background tendency described above, not a depth-local constraint.
"""
from __future__ import annotations

import numpy as np


def raw_lm_quality_rows(datasets, reference_gate=2):
    """Recover identical diagnostics from live projects and saved Qt inputs.

    The stored STD gives an absolute uncertainty proxy, not an independent
    measurement of ambient noise. Flags do not change this diagnostic gate.
    """
    from PyHydroGeophysX.data_processing.em1d import _reference_gate_signal

    if reference_gate < 0:
        raise ValueError("shallow_prior_reference_gate must be nonnegative.")
    rows = []
    for data in datasets:
        raw = (data or {}).get("raw_lm_quality", {})
        values = _reference_gate_signal(raw.get("response", []),
                                       raw.get("relative_std", []),
                                       raw.get("times", []), reference_gate)
        rows.append(dict(zip(("LM_signal", "LM_noise", "LM_reference_time"), values)))
    return rows


def shallow_signal_thresholds(datasets, geometries, inv):
    """Absolute raw-signal limits, fixed by a manual value or a forward model.

    Automatic mode models a homogeneous half-space at a fixed signal-reference
    resistivity,
    with the same waveform, filters, gate window and station geometry as the fit.
    This calibrates a heuristic trigger, NOT the depth of a resistive layer.
    No inverse data_scale enters: both observed diagnostics and limits are in
    the project's stored normalized-response units. Operator caches are reused.
    """
    from .em1d import tdem_moment_blocks, _moment_forward

    fixed = float(inv.get("shallow_prior_signal_threshold", 0.))
    if not np.isfinite(fixed) or fixed < 0:
        raise ValueError("shallow_prior_signal_threshold must be >=0; 0 means auto forward calibration.")
    if len(geometries) != len(datasets):
        raise ValueError("one geometry per sounding is required for signal calibration.")
    limits = np.full(len(datasets), np.nan)
    reference = int(inv.get("shallow_prior_reference_gate", 2))
    if reference < 0:
        raise ValueError("shallow_prior_reference_gate must be nonnegative.")
    rho = float(inv.get("shallow_prior_signal_reference_resistivity", 1000.))
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("signal-reference resistivity must be positive for forward calibration.")
    for i, (data, geometry) in enumerate(zip(datasets, geometries)):
        raw = (data or {}).get("raw_lm_quality", {})
        times = np.asarray(raw.get("times", []), float)
        if not data or reference >= len(times) or not np.isfinite(times[reference]) or times[reference] <= 0:
            continue
        if fixed > 0:
            limits[i] = fixed
            continue
        if not raw.get("transmitter"):
            continue
        diagnostic = {**data, "moments": {"LM": {
            "times": np.array([times[reference]]), "response": np.zeros(1),
            "transmitter": raw["transmitter"]}}}
        # Equal resistivity in both layers is a homogeneous half-space; the
        # interface location has no physical effect and does not change the fit grid.
        blocks = tdem_moment_blocks(diagnostic, geometry, {**inv, "data_scale": 1.}, np.array([10.]))
        value = float(abs(_moment_forward(blocks)(np.full(2, 1./rho))[0]))
        if np.isfinite(value) and value > 0:
            limits[i] = value
    return limits


def resistive_prior_target(inv):
    """Return ``(reference, target, factor, source)`` in ohm-m.

    The automatic reference is the effective starting half-space supplied by
    the workflow. A warm-started model uses its geometric-median resistivity.
    Standalone callers fall back to ``starting_model`` and then
    ``starting_resistivity``. An explicit positive historical
    ``shallow_prior_min_resistivity`` still overrides the automatic target.
    """
    factor = float(inv.get("shallow_prior_resistivity_factor", 2.0))
    if not np.isfinite(factor) or factor < 1.0:
        raise ValueError("resistive-prior factor must be finite and at least 1.")
    supplied = inv.get("_resistive_prior_reference_resistivity")
    if supplied is not None:
        reference = float(supplied)
    else:
        model = np.asarray(inv.get("starting_model", []), dtype=float).ravel()
        valid = model[np.isfinite(model) & (model > 0.)]
        reference = (float(10. ** np.mean(np.log10(valid))) if valid.size
                     else float(inv.get("starting_resistivity", 100.0)))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("resistive-prior starting reference must be positive.")
    explicit = float(inv.get("shallow_prior_min_resistivity", 0.0))
    if not np.isfinite(explicit) or explicit < 0:
        raise ValueError("explicit resistive-prior target must be >= 0.")
    source = "explicit" if explicit > 0 else "starting_model_factor"
    target = explicit if explicit > 0 else reference * factor
    upper = float(inv.get("rho_max", np.inf))
    if np.isfinite(upper):
        target = min(target, upper)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("resistive-prior target must be positive and finite.")
    return reference, target, factor, source


def shallow_prior_terms(inv, thicknesses):
    """Whole-model lower log10(rho) tendency with grid-normalised weight.

    ``thicknesses`` is accepted to retain the established public function
    signature. Every layer, including the basal half-space, receives the same
    coefficient. Dividing by ``sqrt(n_layers)`` keeps the penalty on a uniform
    model unchanged when the grid is split into more layers.
    """
    thick = np.asarray(thicknesses, dtype=float)
    n = len(thick) + 1
    lower, weights = np.zeros(n), np.zeros(n)
    if not inv.get("shallow_prior_enabled", False):
        return lower, weights
    weight = float(inv.get("shallow_prior_weight", 1.0))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("resistive-background prior needs a nonnegative finite weight.")
    _, rho, _, _ = resistive_prior_target(inv)
    # No spatial context in a standalone fit: only a deliberate manual prior applies.
    score = float(inv.get("_shallow_prior_score",
                          1.0 if inv.get("shallow_prior_mode") == "manual" else 0.0))
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("resistive-background prior score must be in [0, 1].")
    lower[:] = np.log10(rho)
    weights[:] = weight * score / np.sqrt(max(n, 1))
    return lower, weights


def shallow_prior_scores(datasets, positions, lines, inv, quality_rows=None, signal_thresholds=None):
    """Fixed, pre-fit spatial scores from early LM SNR, reset at survey lines.

    Baseline is the median of the first full window in acquisition-distance order.
    Raw diagnostics additionally require signal decline at the same physical gate
    time, without a large uncertainty increase. Without raw diagnostics, missing
    imported LM counts as zero under the user's empirical assumption; this weaker
    fallback is identified in the report. Unreadable stations remain unavailable.
    """
    from .em1d import _tdem_uncertainty

    n = len(datasets)
    scores = np.zeros(n)
    quality = np.full(n, np.nan)
    ratios = np.full(n, np.nan)
    signal_ratio = np.full(n, np.nan)
    noise_ratio = np.full(n, np.nan)
    signal, noise, reference_time = (np.full(n, np.nan) for _ in range(3))
    local_distance = np.full(n, np.nan)
    enabled = bool(inv.get("shallow_prior_enabled", False))
    mode = str(inv.get("shallow_prior_mode", "quality_trend"))
    window = int(inv.get("shallow_prior_window", 11))
    cut = float(inv.get("shallow_prior_snr_ratio", .6))
    if mode not in {"quality_trend", "signal_threshold", "manual"} or window < 3 or not 0 < cut < 1:
        raise ValueError("shallow prior mode must be quality_trend/signal_threshold/manual, window >=3, SNR ratio in (0,1).")
    signal_cut = float(inv.get("shallow_prior_signal_ratio", .8))
    noise_cap = float(inv.get("shallow_prior_noise_ratio_max", 2.))
    if not 0 < signal_cut < 1 or not np.isfinite(noise_cap) or noise_cap <= 0:
        raise ValueError("shallow prior requires signal ratio in (0,1) and positive noise ratio cap.")
    if quality_rows is not None and len(quality_rows) != n:
        raise ValueError("quality_rows must contain one row per sounding.")
    if enabled and mode in {"quality_trend", "signal_threshold"} and quality_rows is not None:
        for i, row in enumerate(quality_rows):
            if not row:
                continue
            signal[i] = float(row.get("LM_signal", np.nan))
            noise[i] = float(row.get("LM_noise", np.nan))
            reference_time[i] = float(row.get("LM_reference_time", np.nan))
            if np.isfinite(signal[i]) and np.isfinite(noise[i]) and noise[i] > 0:
                quality[i] = signal[i] / noise[i]
    elif enabled and mode == "quality_trend":
        for i, data in enumerate(datasets):
            if not data or "moments" not in data:
                continue
            lm = data["moments"].get("LM")
            quality[i] = 0.
            if not lm:
                continue
            times = np.asarray(lm["times"], float)
            observed = np.asarray(lm["response"], float) * float(inv.get("data_scale", 1.0))
            unc = _tdem_uncertainty(
                observed, lm, float(inv.get("rel_error", .05)),
                float(inv.get("noise_floor", 1e-18)) * float(inv.get("data_scale", 1.0)),
                float(inv.get("min_rel_error", 0.)), inv.get("max_rel_error"))
            early = (times <= float(inv.get("shallow_prior_lm_time_max_s", 1e-5))) & (observed > 0)
            if early.any():
                quality[i] = float(np.median(np.abs(observed[early]) / unc[early]))
    limits = (np.asarray(signal_thresholds, float).ravel() if signal_thresholds is not None
              else np.full(n, float(inv.get("shallow_prior_signal_threshold", 0.))))
    if limits.size != n:
        raise ValueError("signal_thresholds must contain one limit per sounding.")
    signal_to_threshold = np.full(n, np.nan)
    if mode == "signal_threshold":
        np.divide(signal, limits, out=signal_to_threshold, where=np.isfinite(limits) & (limits > 0))
    allowed_lines = inv.get("shallow_prior_lines", []) or []
    baselines = {}
    for line in np.unique(lines):
        ix = np.flatnonzero(np.asarray(lines) == line)
        ix = ix[np.argsort(np.asarray(positions)[ix], kind="stable")]
        local_distance[ix] = np.asarray(positions)[ix] - np.min(np.asarray(positions)[ix])
        if not enabled or (allowed_lines and int(line) not in allowed_lines):
            continue
        if mode == "manual":
            scores[ix] = 1.
        elif mode == "signal_threshold":
            for j in range(window-1, len(ix)):
                segment = ix[j-window+1:j+1]
                values = signal_to_threshold[segment]
                # Missing/dummy raw gates are unavailable, not weak signal.
                # Require most stations AND the current station below threshold.
                if (np.isfinite(values).sum() < window*.8 or not np.isfinite(values[-1])
                        or np.count_nonzero(values < 1.) < int(np.ceil(window*.8))):
                    continue
                scores[ix[j]] = np.clip(1. - max(float(np.nanmedian(values)), values[-1]), 0., 1.)
        elif len(ix) >= window:
            first = quality[ix[:window]]
            baseline = float(np.nanmedian(first)) if np.isfinite(first).sum() >= window*.8 else 0.
            baselines[str(int(line))] = baseline
            if baseline > 0:
                base_signal = float(np.nanmedian(signal[ix[:window]])) if quality_rows is not None else np.nan
                base_noise = float(np.nanmedian(noise[ix[:window]])) if quality_rows is not None else np.nan
                base_time = float(np.nanmedian(reference_time[ix[:window]])) if quality_rows is not None else np.nan
                for j in range(window, len(ix)):
                    segment = ix[j-window+1:j+1]
                    values = quality[segment]
                    if np.isfinite(values).sum() < window*.8:
                        continue
                    ratio = float(np.nanmedian(values) / baseline)
                    ratios[ix[j]] = ratio
                    if quality_rows is not None:
                        signal_ratio[ix[j]] = float(np.nanmedian(signal[segment]) / base_signal) if base_signal > 0 else np.nan
                        noise_ratio[ix[j]] = float(np.nanmedian(noise[segment]) / base_noise) if base_noise > 0 else np.nan
                        # Do not create a resistive tendency from noise growth
                        # alone or from comparing different physical gate times.
                        if (not np.allclose(reference_time[segment], base_time, rtol=.01, atol=0.)
                                or not signal_ratio[ix[j]] < signal_cut
                                or not noise_ratio[ix[j]] <= noise_cap):
                            continue
                    scores[ix[j]] = np.clip((cut - ratio) / cut, 0., 1.)
    scores[local_distance < float(inv.get("shallow_prior_distance_min_m", 0.))] = 0.
    end = float(inv.get("shallow_prior_distance_max_m", 0.))
    if end > 0:
        scores[local_distance > end] = 0.
    scores[[not bool(data) for data in datasets]] = 0.
    reference_rho, target_rho, factor, target_source = resistive_prior_target(inv)
    return scores, {"enabled": enabled, "mode": mode, "baseline_snr_by_line": baselines,
                    "active_soundings": int(np.count_nonzero(scores)),
                    "scope": "whole_model_background", "depth_m": None,
                    "reference_resistivity": reference_rho,
                    "target_resistivity": target_rho,
                    "minimum_resistivity": target_rho,
                    "resistivity_factor": factor, "target_source": target_source,
                    "weight": float(inv.get("shallow_prior_weight", 1.)),
                    "window": window, "snr_ratio_threshold": cut,
                    "reference_gate": int(inv.get("shallow_prior_reference_gate", 2)),
                    "signal_threshold": limits.tolist(),
                    "signal_to_threshold": signal_to_threshold.tolist(),
                    "signal_threshold_source": ("manual" if float(inv.get("shallow_prior_signal_threshold", 0.)) > 0
                                                else "homogeneous_halfspace_forward"),
                    "signal_reference_resistivity": float(inv.get(
                        "shallow_prior_signal_reference_resistivity", 1000.)),
                    "signal_ratio_threshold": signal_cut, "noise_ratio_max": noise_cap,
                    "score": scores.tolist(), "early_lm_snr": quality.tolist(),
                    "quality_source": "fixed_raw_LM_gate" if quality_rows is not None else "imported_LM_gates",
                    "signal_ratio": signal_ratio.tolist(), "noise_ratio": noise_ratio.tolist(),
                    "smoothed_snr_ratio": ratios.tolist(), "line_distance_m": local_distance.tolist(),
                    "interpretation": ("weak absolute LM can support a resistive background, "
                                       "but does not locate a shallow layer; empirical and not "
                                       "independent geological evidence")}
