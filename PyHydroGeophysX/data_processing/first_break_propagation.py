"""Carry a few hand-picked first breaks across the rest of a shot gather.

A user picks the first arrival on a handful of traces ("anchors"); this module
predicts it on every other trace and snaps each prediction to the arrival
nearest it. The prediction is, in order of preference:

* a 1D velocity model fitted to the anchors (weight 1) and the automatic picks
  (a small weight), predicted from the source-receiver geometry;
* otherwise the automatic picks shifted onto a single anchor, or the anchors
  interpolated (monotone cubic between them, linear beyond them), each trace
  bounded by the anchors on either side.

With two or more anchors and no velocity model, the picks follow a
continuity-constrained path through a first-break reward map (energy ratio,
modified energy ratio, gradient and an AIC onset), found by dynamic programming,
so neighbouring picks cannot jump between arrivals. Otherwise each trace takes
the best onset inside a window around its prediction.

The algorithm lived inside the Streamlit seismic page, where nothing else could
call it or test it. It needs NumPy; SciPy, when installed, curves the
interpolation between three or more anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Propagation:
    """What :func:`propagate_first_breaks` decided.

    Attributes
    ----------
    manual_count : int
        Usable anchors among the picks.
    updates : list of (int, int)
        ``(trace, sample)`` for every trace that is not an anchor, in trace
        order.
    message : str
        Why nothing was propagated; empty when it was.
    velocity_model : object or None
        The fitted 1D velocity model, carrying the final ``predicted_times``;
        None when none was used.
    model_decided : bool
        True once the inputs passed their checks and the velocity-model choice
        was made, so ``velocity_model`` (possibly None) replaces any earlier one.
    """

    manual_count: int = 0
    updates: List[Tuple[int, int]] = field(default_factory=list)
    message: str = ""
    velocity_model: Any = None
    model_decided: bool = False


def propagate_first_breaks(
    trace_data: Any,
    time_values: Any,
    pick_positions: Any,
    pick_times: Any,
    baseline_times: Any,
    is_manual: Any,
    *,
    search_window_s: float,
    max_time_s: float,
    learning_method: str = "1D velocity model",
    geometry: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Propagation:
    """Predict the first break on every trace from the anchors among the picks.

    Parameters
    ----------
    trace_data : ndarray
        ``(n_samples, n_traces)`` amplitudes.
    time_values : ndarray
        ``(n_samples,)`` sample times in seconds.
    pick_positions : ndarray
        Trace position of each pick (NaN where it matches no trace).
    pick_times : ndarray
        Each pick's time in seconds.
    baseline_times : ndarray
        Each pick's automatic time, the baseline the anchors correct.
    is_manual : ndarray of bool
        Which picks were placed by hand.
    search_window_s : float
        Half-width of the window a prediction is snapped within.
    max_time_s : float
        Latest time a pick may take.
    learning_method : str
        Starts with ``"1d"`` to fit a 1D velocity model; anything else uses
        the anchors alone.
    geometry : callable, optional
        Returns ``(source_x, receiver_x)`` per trace for the velocity model.
        Called only when the model is fitted; an exception from it falls back
        to the anchors.

    Returns
    -------
    Propagation
    """
    from PyHydroGeophysX.data_processing.seismic import (
        fit_velocity_traveltime_model,
        predict_velocity_traveltimes,
    )

    local_positions = np.asarray(pick_positions, dtype=float)
    pick_times = np.asarray(pick_times, dtype=float)
    baseline_values = np.asarray(baseline_times, dtype=float)
    manual_flags = np.asarray(is_manual, dtype=bool)
    manual_mask = (
        manual_flags
        & np.isfinite(local_positions)
        & np.isfinite(pick_times)
        & (pick_times > 0)
    )
    manual_count = int(manual_mask.sum())
    result = Propagation(manual_count=manual_count)
    if manual_count < 1:
        result.message = "Save at least one anchor pick before updating the remaining traces."
        return result

    trace_data = np.asarray(trace_data, dtype=float)
    n_samples, n_traces = trace_data.shape
    time_values = np.asarray(time_values, dtype=float)
    if time_values.size != n_samples:
        result.message = "Trace and time arrays do not have matching sample counts."
        return result

    dt = float(np.nanmedian(np.diff(time_values))) if time_values.size > 1 else 0.001
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.001
    half_window = max(1, int(round(float(search_window_s) / dt)))
    max_sample = int(np.clip(np.searchsorted(time_values, float(max_time_s), side="right") - 1, 0, n_samples - 1))

    baseline_by_trace = np.full(n_traces, np.nan, dtype=float)
    for position, baseline in zip(local_positions, baseline_values):
        if np.isfinite(position) and np.isfinite(baseline) and baseline > 0:
            trace_pos = int(np.clip(round(float(position)), 0, n_traces - 1))
            baseline_by_trace[trace_pos] = float(baseline)
    fallback_by_trace = np.full(n_traces, np.nan, dtype=float)
    for position, pick_time in zip(local_positions, pick_times):
        if np.isfinite(position) and np.isfinite(pick_time) and pick_time > 0:
            trace_pos = int(np.clip(round(float(position)), 0, n_traces - 1))
            fallback_by_trace[trace_pos] = float(pick_time)
    missing_baseline = ~np.isfinite(baseline_by_trace)
    baseline_by_trace[missing_baseline] = fallback_by_trace[missing_baseline]
    known = np.where(np.isfinite(baseline_by_trace))[0]
    if known.size == 0:
        result.message = "No baseline auto-pick times are available for correction."
        return result
    if known.size < n_traces:
        baseline_by_trace = np.interp(np.arange(n_traces, dtype=float), known.astype(float), baseline_by_trace[known])

    manual_x = local_positions[manual_mask].astype(float)
    manual_t = pick_times[manual_mask].astype(float)
    manual_trace_map: Dict[int, float] = {}
    for trace_pos_f, pick_time in zip(manual_x, manual_t):
        if np.isfinite(trace_pos_f) and np.isfinite(pick_time) and pick_time > 0:
            manual_trace_map[int(np.clip(round(float(trace_pos_f)), 0, n_traces - 1))] = float(pick_time)
    if not manual_trace_map:
        result.message = "No valid anchor pick times are available for correction."
        return result

    manual_trace_positions = np.asarray(sorted(manual_trace_map.keys()), dtype=float)
    manual_pick_times = np.asarray([manual_trace_map[int(pos)] for pos in manual_trace_positions], dtype=float)
    x_all = np.arange(n_traces, dtype=float)
    lower_time_bounds = np.full(n_traces, float(time_values[0]), dtype=float)
    upper_time_bounds = np.full(n_traces, float(time_values[max_sample]), dtype=float)

    predicted_times: Optional[np.ndarray] = None
    model_prediction_used = False
    velocity_model = None
    use_velocity_model = str(learning_method or "").strip().lower().startswith("1d")
    if use_velocity_model:
        try:
            if geometry is None:
                raise ValueError("No source-receiver geometry was given for the velocity model.")
            source_x_all, receiver_x_all = geometry()
            hint_weight = 0.15 if manual_count < 2 else 0.04
            fit_source_x: List[float] = []
            fit_receiver_x: List[float] = []
            fit_times: List[float] = []
            fit_weights: List[float] = []
            fit_anchor_mask: List[bool] = []
            for trace_pos_f, pick_time, is_manual_pick in zip(local_positions, pick_times, manual_flags):
                if not (np.isfinite(trace_pos_f) and np.isfinite(pick_time) and pick_time > 0):
                    continue
                trace_pos = int(np.clip(round(float(trace_pos_f)), 0, n_traces - 1))
                weight = 1.0 if is_manual_pick else hint_weight
                if weight <= 0:
                    continue
                fit_source_x.append(float(source_x_all[trace_pos]))
                fit_receiver_x.append(float(receiver_x_all[trace_pos]))
                fit_times.append(float(pick_time))
                fit_weights.append(float(weight))
                fit_anchor_mask.append(bool(is_manual_pick))
            if len(fit_times) < 2:
                raise ValueError("At least two travel-time points are needed for the velocity model.")
            velocity_model = fit_velocity_traveltime_model(
                fit_source_x,
                fit_receiver_x,
                fit_times,
                weights=fit_weights,
                anchor_mask=fit_anchor_mask,
                max_segments=3,
                velocity_bounds=(100.0, 8000.0),
            )
            predicted_times = predict_velocity_traveltimes(velocity_model, source_x_all, receiver_x_all)
            predicted_times = np.asarray(predicted_times, dtype=float)
            if predicted_times.size != n_traces or int(np.isfinite(predicted_times).sum()) < 2:
                raise ValueError("Velocity model did not produce enough valid predictions.")
            missing_pred = ~np.isfinite(predicted_times)
            predicted_times[missing_pred] = baseline_by_trace[missing_pred]
            model_margin = max(float(search_window_s), dt * 4.0)
            lower_time_bounds = np.maximum(float(time_values[0]), predicted_times - model_margin)
            upper_time_bounds = np.minimum(float(time_values[max_sample]), predicted_times + model_margin)
            model_prediction_used = True
        except Exception:  # noqa: BLE001 - any failure falls back to the anchors
            predicted_times = None
            velocity_model = None
    result.model_decided = True

    if predicted_times is None:
        if len(manual_trace_positions) == 1:
            manual_trace = int(manual_trace_positions[0])
            baseline_shift = float(manual_pick_times[0] - baseline_by_trace[manual_trace])
            if not np.isfinite(baseline_shift):
                baseline_shift = 0.0
            predicted_times = baseline_by_trace + baseline_shift
        else:
            predicted_times = np.interp(x_all, manual_trace_positions, manual_pick_times)
            first_x, second_x = float(manual_trace_positions[0]), float(manual_trace_positions[1])
            last_x, penultimate_x = float(manual_trace_positions[-1]), float(manual_trace_positions[-2])
            first_t, second_t = float(manual_pick_times[0]), float(manual_pick_times[1])
            last_t, penultimate_t = float(manual_pick_times[-1]), float(manual_pick_times[-2])
            left_dx = max(abs(second_x - first_x), 1.0)
            right_dx = max(abs(last_x - penultimate_x), 1.0)
            left_of_anchors = x_all < first_x
            right_of_anchors = x_all > last_x
            predicted_times[left_of_anchors] = first_t + ((second_t - first_t) / left_dx) * (x_all[left_of_anchors] - first_x)
            predicted_times[right_of_anchors] = last_t + ((last_t - penultimate_t) / right_dx) * (x_all[right_of_anchors] - last_x)

            if len(manual_trace_positions) >= 3:
                inside_anchors = (x_all >= first_x) & (x_all <= last_x)
                try:
                    from scipy.interpolate import PchipInterpolator

                    pchip = PchipInterpolator(manual_trace_positions, manual_pick_times, extrapolate=False)
                    curved_times = np.asarray(pchip(x_all[inside_anchors]), dtype=float)
                    valid_curved = np.isfinite(curved_times)
                    predicted_times[inside_anchors] = np.where(valid_curved, curved_times, predicted_times[inside_anchors])
                except Exception:  # noqa: BLE001
                    pass

            anchor_margin = max(float(search_window_s), dt * 4.0)
            for trace_pos in range(n_traces):
                right_anchor = int(np.searchsorted(manual_trace_positions, float(trace_pos), side="left"))
                if 0 < right_anchor < len(manual_trace_positions):
                    left_time = float(manual_pick_times[right_anchor - 1])
                    right_time = float(manual_pick_times[right_anchor])
                    lower = min(left_time, right_time) - anchor_margin
                    upper = max(left_time, right_time) + anchor_margin
                    lower_time_bounds[trace_pos] = max(float(time_values[0]), lower)
                    upper_time_bounds[trace_pos] = min(float(time_values[max_sample]), upper)
                    predicted_times[trace_pos] = float(np.clip(predicted_times[trace_pos], lower, upper))
                elif right_anchor == 0 and len(manual_pick_times) > 1:
                    lower = min(first_t, second_t) - 2.0 * anchor_margin
                    upper = max(first_t, second_t) + 2.0 * anchor_margin
                    lower_time_bounds[trace_pos] = max(float(time_values[0]), lower)
                    upper_time_bounds[trace_pos] = min(float(time_values[max_sample]), upper)
                    predicted_times[trace_pos] = float(np.clip(predicted_times[trace_pos], lower, upper))
                elif len(manual_pick_times) > 1:
                    lower = min(penultimate_t, last_t) - 2.0 * anchor_margin
                    upper = max(penultimate_t, last_t) + 2.0 * anchor_margin
                    lower_time_bounds[trace_pos] = max(float(time_values[0]), lower)
                    upper_time_bounds[trace_pos] = min(float(time_values[max_sample]), upper)
                    predicted_times[trace_pos] = float(np.clip(predicted_times[trace_pos], lower, upper))

    for trace_pos, pick_time in manual_trace_map.items():
        predicted_times[int(trace_pos)] = float(pick_time)
    predicted_times = np.clip(predicted_times, float(time_values[0]), float(time_values[max_sample]))
    if model_prediction_used and velocity_model is not None:
        velocity_model.predicted_times = predicted_times.copy()
    result.velocity_model = velocity_model
    manual_traces = set(manual_trace_map.keys())

    def _nearest_first_break_sample(trace: np.ndarray, center: int, lo: int, hi: int) -> int:
        window = np.asarray(trace[lo:hi], dtype=float)
        finite = np.isfinite(window)
        if not finite.any():
            return int(center)
        window_abs = np.abs(window)
        gradient = np.abs(np.gradient(window)) if window.size > 1 else window_abs
        amp_scale = float(np.nanpercentile(window_abs[finite], 90))
        grad_scale = float(np.nanpercentile(gradient[finite], 90))
        amp_scale = max(amp_scale, float(np.nanmedian(window_abs[finite])) * 2.0, 1e-12)
        grad_scale = max(grad_scale, float(np.nanmedian(gradient[finite])) * 2.0, 1e-12)
        onset = (0.75 * window_abs / amp_scale) + (0.25 * gradient / grad_scale)
        sample_numbers = np.arange(lo, hi, dtype=float)
        distance = np.abs(sample_numbers - float(center)) / max(float(half_window), 1.0)
        scores = np.where(finite, onset - (0.65 * distance), -np.inf)
        if window.size > 2:
            scores[0] -= 0.75
            scores[-1] -= 0.75
        if not np.isfinite(scores).any():
            return int(center)
        local_idx = int(np.nanargmax(scores))
        if window.size > 4 and local_idx in {0, window.size - 1}:
            inner_scores = scores[1:-1]
            if np.isfinite(inner_scores).any():
                local_idx = 1 + int(np.nanargmax(inner_scores))
        return int(np.clip(lo + local_idx, 0, max_sample))

    def _forward_mean(values: np.ndarray, window: int) -> np.ndarray:
        window = max(1, int(window))
        padded = np.pad(np.asarray(values, dtype=float), ((0, window - 1), (0, 0)), mode="edge")
        cumulative = np.cumsum(np.vstack([np.zeros((1, padded.shape[1])), padded]), axis=0)
        return (cumulative[window:] - cumulative[:-window]) / float(window)

    def _backward_mean(values: np.ndarray, window: int) -> np.ndarray:
        return np.flipud(_forward_mean(np.flipud(values), window))

    def _normalize_columns(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        normalized = np.zeros_like(values, dtype=float)
        for col in range(values.shape[1]):
            column = values[:, col]
            finite = np.isfinite(column)
            if not finite.any():
                continue
            low, high = np.nanpercentile(column[finite], [10, 95])
            if not np.isfinite(high - low) or high <= low:
                low = float(np.nanmin(column[finite]))
                high = float(np.nanmax(column[finite]))
            scale = max(high - low, 1e-12)
            normalized[:, col] = np.clip((np.nan_to_num(column, nan=low) - low) / scale, 0.0, 1.0)
        return normalized

    def _first_break_reward_map() -> np.ndarray:
        clean_traces = np.nan_to_num(trace_data[: max_sample + 1, :], nan=0.0, posinf=0.0, neginf=0.0)
        abs_traces = np.abs(clean_traces)
        energy = clean_traces**2
        short_window = max(2, int(round(0.002 / dt)))
        long_window = max(short_window * 3, int(round(0.010 / dt)))
        future_energy = _forward_mean(energy, short_window)
        past_energy = _backward_mean(energy, long_window)
        energy_ratio = future_energy / (past_energy + np.nanmedian(past_energy) * 0.05 + 1e-12)
        mer = energy_ratio * (abs_traces + np.nanmedian(abs_traces, axis=0, keepdims=True))
        gradient = np.abs(np.gradient(clean_traces, axis=0))

        aic_reward = np.zeros_like(clean_traces, dtype=float)
        sample_axis = np.arange(clean_traces.shape[0], dtype=float)
        aic_width = max(2.0, float(half_window) * 0.35)
        for trace_pos in range(clean_traces.shape[1]):
            trace = clean_traces[:, trace_pos]
            if trace.size < 8 or not np.isfinite(trace).any():
                continue
            trace = trace - float(np.nanmedian(trace))
            csum = np.cumsum(trace)
            csum2 = np.cumsum(trace**2)
            n = trace.size
            candidates = np.arange(2, n - 2, dtype=int)
            if candidates.size == 0:
                continue
            left_count = candidates.astype(float)
            right_count = (n - candidates).astype(float)
            left_mean = csum[candidates - 1] / left_count
            right_sum = csum[-1] - csum[candidates - 1]
            right_mean = right_sum / right_count
            left_var = (csum2[candidates - 1] / left_count) - left_mean**2
            right_var = ((csum2[-1] - csum2[candidates - 1]) / right_count) - right_mean**2
            left_var = np.maximum(left_var, 1e-12)
            right_var = np.maximum(right_var, 1e-12)
            aic = left_count * np.log(left_var) + right_count * np.log(right_var)
            if not np.isfinite(aic).any():
                continue
            aic_sample = float(candidates[int(np.nanargmin(aic))])
            aic_reward[:, trace_pos] = np.exp(-0.5 * ((sample_axis - aic_sample) / aic_width) ** 2)

        reward = (
            0.45 * _normalize_columns(mer)
            + 0.25 * _normalize_columns(energy_ratio)
            + 0.20 * _normalize_columns(gradient)
            + 0.10 * aic_reward
        )
        return np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _continuity_constrained_path_samples() -> Optional[np.ndarray]:
        if manual_count < 2:
            return None
        reward = _first_break_reward_map()
        n_states = max_sample + 1
        center_samples = np.asarray(
            np.clip(np.searchsorted(time_values, predicted_times), 0, max_sample),
            dtype=int,
        )
        center_diffs = np.abs(np.diff(center_samples)).astype(float)
        max_center_jump = float(np.nanpercentile(center_diffs, 95)) if center_diffs.size else 1.0
        max_jump = int(np.clip(max(max_center_jump + 3.0, half_window * 0.75, 3.0), 3, max_sample))
        center_sigma = max(float(half_window), float(max_jump), 2.0)
        smoothness = 0.85
        center_weight = 0.22
        dp = np.full((n_traces, n_states), -np.inf, dtype=float)
        back = np.full((n_traces, n_states), -1, dtype=int)

        for trace_pos in range(n_traces):
            lower_bound_idx = int(
                np.clip(np.searchsorted(time_values, float(lower_time_bounds[trace_pos]), side="left"), 0, max_sample)
            )
            upper_bound_idx = int(
                np.clip(np.searchsorted(time_values, float(upper_time_bounds[trace_pos]), side="right"), 1, max_sample + 1)
            )
            score = np.full(n_states, -np.inf, dtype=float)
            samples = np.arange(lower_bound_idx, upper_bound_idx, dtype=int)
            if samples.size:
                distance = (samples.astype(float) - float(center_samples[trace_pos])) / center_sigma
                score[samples] = reward[samples, trace_pos] - center_weight * distance**2
            if trace_pos in manual_trace_map:
                anchor_sample = int(
                    np.clip(np.searchsorted(time_values, float(manual_trace_map[trace_pos])), lower_bound_idx, upper_bound_idx - 1)
                )
                score[:] = -np.inf
                score[anchor_sample] = 5.0
            if trace_pos == 0:
                dp[trace_pos, :] = score
                continue
            previous = dp[trace_pos - 1, :]
            for sample in np.where(np.isfinite(score))[0]:
                lo_prev = max(0, int(sample) - max_jump)
                hi_prev = min(n_states, int(sample) + max_jump + 1)
                previous_window = previous[lo_prev:hi_prev]
                if not np.isfinite(previous_window).any():
                    finite_previous = np.where(np.isfinite(previous))[0]
                    if not finite_previous.size:
                        continue
                    jumps = finite_previous.astype(float) - float(sample)
                    transition = previous[finite_previous] - smoothness * (jumps / max(float(max_jump), 1.0)) ** 2
                    best_local = int(np.nanargmax(transition))
                    best_prev = int(finite_previous[best_local])
                    best_value = float(transition[best_local])
                else:
                    prev_samples = np.arange(lo_prev, hi_prev, dtype=float)
                    jumps = prev_samples - float(sample)
                    transition = previous_window - smoothness * (jumps / max(float(max_jump), 1.0)) ** 2
                    best_local = int(np.nanargmax(transition))
                    best_prev = int(lo_prev + best_local)
                    best_value = float(transition[best_local])
                dp[trace_pos, sample] = score[sample] + best_value
                back[trace_pos, sample] = best_prev

        if not np.isfinite(dp[-1, :]).any():
            return None
        path = np.full(n_traces, -1, dtype=int)
        path[-1] = int(np.nanargmax(dp[-1, :]))
        for trace_pos in range(n_traces - 1, 0, -1):
            previous = int(back[trace_pos, path[trace_pos]])
            if previous < 0:
                previous = int(center_samples[trace_pos - 1])
            path[trace_pos - 1] = int(previous)
        path = np.asarray(np.clip(path, 0, max_sample), dtype=int)
        for trace_pos, pick_time in manual_trace_map.items():
            path[int(trace_pos)] = int(np.clip(np.searchsorted(time_values, float(pick_time)), 0, max_sample))
        return path

    path_samples = None if model_prediction_used else _continuity_constrained_path_samples()

    for trace_pos in range(n_traces):
        if trace_pos in manual_traces:
            continue
        predicted = float(predicted_times[trace_pos])
        if not np.isfinite(predicted):
            continue
        predicted = float(np.clip(predicted, float(time_values[0]), float(time_values[max_sample])))
        if path_samples is not None:
            sample_idx = int(path_samples[trace_pos])
        else:
            center = int(np.clip(np.searchsorted(time_values, predicted), 0, max_sample))
            lo = int(max(0, center - half_window))
            hi = int(min(max_sample + 1, center + half_window + 1))
            lower_bound_idx = int(
                np.clip(np.searchsorted(time_values, float(lower_time_bounds[trace_pos]), side="left"), 0, max_sample)
            )
            upper_bound_idx = int(
                np.clip(np.searchsorted(time_values, float(upper_time_bounds[trace_pos]), side="right"), 1, max_sample + 1)
            )
            lo = max(lo, lower_bound_idx)
            hi = min(hi, upper_bound_idx)
            if hi <= lo:
                lo = int(np.clip(center, lower_bound_idx, max(upper_bound_idx - 1, lower_bound_idx)))
                hi = min(max_sample + 1, lo + 1)
            sample_idx = _nearest_first_break_sample(trace_data[:, trace_pos], center, lo, hi)
            if max_sample > 2:
                sample_idx = int(np.clip(sample_idx, 1, max_sample - 1))
        result.updates.append((int(trace_pos), int(sample_idx)))

    return result


__all__ = ["Propagation", "propagate_first_breaks"]
