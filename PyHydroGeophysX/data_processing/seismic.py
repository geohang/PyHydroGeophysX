"""Raw seismic data processing helpers.

SEG-Y reading prefers a robust third-party library and degrades gracefully so
local workflows keep working with no extra installs. The reader order is:
``segyio`` (broadest coverage: endianness, sample-format codes, byte locations,
SEG-Y rev 1/2), then ``ObsPy``, then a small built-in big-endian reader that
handles the common SEG-Y layout used by the bundled example. Install ``segyio``
(``pip install segyio``) for the most reliable reads of arbitrary field data.
"""

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SegyMetadata:
    """Summary metadata from a SEG-Y file."""

    sample_interval_us: int
    samples_per_trace: int
    format_code: int
    trace_count: int
    file_size_bytes: int

    @property
    def sample_interval_s(self) -> float:
        """Sampling interval in seconds."""

        return float(self.sample_interval_us) * 1e-6


@dataclass
class SeismicTraceHeader:
    """Subset of SEG-Y trace-header fields used by the GUI workflow."""

    field_record: int
    trace_number: int
    energy_source_point: int
    source_x: float
    source_y: float
    source_z: float
    receiver_x: float
    receiver_y: float
    receiver_z: float
    offset: float


@dataclass
class SeismicShotGather:
    """One shot gather extracted from a seismic dataset."""

    field_record: int
    trace_indices: np.ndarray
    traces: np.ndarray
    time: np.ndarray
    channels: np.ndarray
    offsets: np.ndarray
    headers: List[SeismicTraceHeader]


@dataclass
class SeismicDataset:
    """In-memory SEG-Y dataset with traces arranged as samples by traces."""

    path: str
    traces: np.ndarray
    time: np.ndarray
    headers: List[SeismicTraceHeader]
    metadata: SegyMetadata

    @property
    def field_records(self) -> List[int]:
        """Sorted shot/field-record identifiers available in this dataset."""

        return sorted({int(header.field_record) for header in self.headers})

    def get_gather(self, field_record: Optional[int] = None) -> SeismicShotGather:
        """Return one shot gather.

        Parameters
        ----------
        field_record : int, optional
            Field-record id. If omitted, the first available record is used.

        Returns
        -------
        SeismicShotGather
            Gather with traces, channel ids, offsets, and headers.
        """

        if not self.headers:
            raise ValueError("Dataset contains no trace headers.")
        if self.traces.size == 0:
            raise ValueError("Dataset was loaded without trace samples.")

        record = field_record if field_record is not None else self.field_records[0]
        indices = np.array(
            [i for i, header in enumerate(self.headers) if int(header.field_record) == int(record)],
            dtype=int,
        )
        if indices.size == 0:
            raise ValueError(f"No traces found for field_record={record}.")

        gather_headers = [self.headers[int(i)] for i in indices]
        return SeismicShotGather(
            field_record=int(record),
            trace_indices=indices,
            traces=self.traces[:, indices],
            time=self.time,
            channels=np.array([h.trace_number for h in gather_headers], dtype=int),
            offsets=np.array([h.offset for h in gather_headers], dtype=float),
            headers=gather_headers,
        )


@dataclass
class FirstBreakPick:
    """One first-break pick."""

    source_id: int
    receiver_id: int
    time_s: float
    source_x: float
    source_z: float
    receiver_x: float
    receiver_z: float
    field_record: int
    trace_number: int
    trace_index: int
    amplitude: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a CSV/JSON friendly representation."""

        return {
            "source_id": self.source_id,
            "receiver_id": self.receiver_id,
            "time_s": self.time_s,
            "source_x": self.source_x,
            "source_z": self.source_z,
            "receiver_x": self.receiver_x,
            "receiver_z": self.receiver_z,
            "field_record": self.field_record,
            "trace_number": self.trace_number,
            "trace_index": self.trace_index,
            "amplitude": self.amplitude,
        }


@dataclass
class TravelTimeModelSegment:
    """One fitted straight travel-time branch in offset space."""

    branch_id: str
    segment_index: int
    x_min: float
    x_max: float
    slope_s_per_m: float
    intercept_s: float
    apparent_velocity_m_s: float
    crossover_offset_m: Optional[float] = None
    intercept_time_s: Optional[float] = None
    depth_estimate_m: Optional[float] = None


@dataclass
class TravelTimeModel:
    """Piecewise-linear 1-D velocity model used for first-arrival guidance."""

    segments: List[TravelTimeModelSegment]
    residual_rms_s: float
    branch_ids: List[str]
    predicted_times: Optional[np.ndarray] = None
    message: str = ""


def _as_1d_float(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def _branch_ids_from_geometry(source_x: np.ndarray, receiver_x: np.ndarray) -> np.ndarray:
    signed_offset = receiver_x - source_x
    has_left = bool(np.any(signed_offset < 0))
    has_right = bool(np.any(signed_offset > 0))
    if has_left and has_right:
        return np.where(signed_offset < 0, "left", "right")
    return np.full(source_x.shape, "all", dtype=object)


def _weighted_line_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    design = np.column_stack([x, np.ones_like(x)])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    slope, intercept = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    predicted = slope * x + intercept
    rmse = math.sqrt(float(np.average((y - predicted) ** 2, weights=weights)))
    return float(slope), float(intercept), float(rmse), predicted


def _fit_anchor_exact_branch_segments(
    branch_id: str,
    x: np.ndarray,
    t: np.ndarray,
    anchor_mask: np.ndarray,
    weights: np.ndarray,
    velocity_bounds: Tuple[float, float],
) -> Optional[Tuple[List[TravelTimeModelSegment], float, str]]:
    """Create piecewise-linear segments that pass through manual anchors."""

    anchor_x = x[anchor_mask]
    anchor_t = t[anchor_mask]
    anchor_w = weights[anchor_mask]
    if anchor_x.size < 2:
        return None

    unique_x: List[float] = []
    unique_t: List[float] = []
    for value in np.unique(anchor_x):
        mask = anchor_x == value
        unique_x.append(float(value))
        unique_t.append(float(np.average(anchor_t[mask], weights=anchor_w[mask])))
    anchor_x = np.asarray(unique_x, dtype=float)
    anchor_t = np.asarray(unique_t, dtype=float)
    order = np.argsort(anchor_x)
    anchor_x = anchor_x[order]
    anchor_t = anchor_t[order]

    if anchor_x.size < 2:
        return None

    min_velocity, max_velocity = velocity_bounds
    min_slope = 1.0 / max(float(max_velocity), 1e-12)
    max_slope = 1.0 / max(float(min_velocity), 1e-12)
    segments: List[TravelTimeModelSegment] = []
    slopes: List[float] = []
    for idx in range(anchor_x.size - 1):
        dx = float(anchor_x[idx + 1] - anchor_x[idx])
        if abs(dx) <= 1e-12:
            continue
        slope = float((anchor_t[idx + 1] - anchor_t[idx]) / dx)
        if not np.isfinite(slope) or not min_slope <= slope <= max_slope:
            raise ValueError(f"Manual anchors for branch {branch_id} are not consistent with positive apparent velocity.")
        intercept = float(anchor_t[idx] - slope * anchor_x[idx])
        slopes.append(slope)
        segments.append(
            TravelTimeModelSegment(
                branch_id=branch_id,
                segment_index=len(segments) + 1,
                x_min=float(anchor_x[idx]),
                x_max=float(anchor_x[idx + 1]),
                slope_s_per_m=slope,
                intercept_s=intercept,
                apparent_velocity_m_s=float(1.0 / slope),
            )
        )

    if not segments:
        return None

    predicted = _predict_branch_times(x, segments)
    rmse = math.sqrt(float(np.average((t - predicted) ** 2, weights=weights)))

    previous_segment: Optional[TravelTimeModelSegment] = None
    for segment in segments:
        if previous_segment is not None:
            denominator = segment.slope_s_per_m - previous_segment.slope_s_per_m
            if abs(denominator) > 1e-12:
                segment.crossover_offset_m = float((previous_segment.intercept_s - segment.intercept_s) / denominator)
            segment.intercept_time_s = float(segment.intercept_s)
            v1 = previous_segment.apparent_velocity_m_s
            v2 = segment.apparent_velocity_m_s
            ti = segment.intercept_s
            if ti > 0 and v2 > v1 > 0:
                segment.depth_estimate_m = float(ti * v1 * v2 / (2.0 * math.sqrt(max(v2 * v2 - v1 * v1, 1e-12))))
        previous_segment = segment

    return segments, float(rmse), f"anchor-exact {len(segments)} slope(s)"


def _fit_branch_segments(
    branch_id: str,
    x: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    max_segments: int,
    velocity_bounds: Tuple[float, float],
) -> Tuple[List[TravelTimeModelSegment], float, str]:
    order = np.argsort(x)
    x = x[order]
    t = t[order]
    weights = weights[order]

    unique_x: List[float] = []
    unique_t: List[float] = []
    unique_w: List[float] = []
    for value in np.unique(x):
        mask = x == value
        w = weights[mask]
        unique_x.append(float(value))
        unique_t.append(float(np.average(t[mask], weights=w)))
        unique_w.append(float(np.sum(w)))
    x = np.asarray(unique_x, dtype=float)
    t = np.asarray(unique_t, dtype=float)
    weights = np.asarray(unique_w, dtype=float)

    if x.size < 2:
        raise ValueError(f"Not enough picks to fit branch {branch_id}.")

    min_velocity, max_velocity = velocity_bounds
    min_slope = 1.0 / max(float(max_velocity), 1e-12)
    max_slope = 1.0 / max(float(min_velocity), 1e-12)
    max_segments = int(np.clip(max_segments, 1, 3))
    max_segments = min(max_segments, max(1, x.size // 2))

    candidates: List[Tuple[int, float, List[TravelTimeModelSegment], np.ndarray]] = []
    n = int(x.size)

    def add_candidate(boundaries: List[Tuple[int, int]]) -> None:
        segments: List[TravelTimeModelSegment] = []
        predicted = np.full_like(t, np.nan, dtype=float)
        slopes: List[float] = []
        for segment_index, (start, stop) in enumerate(boundaries, start=1):
            if stop - start < 2:
                return
            slope, intercept, _rmse, segment_pred = _weighted_line_fit(x[start:stop], t[start:stop], weights[start:stop])
            if not np.isfinite(slope) or not min_slope <= slope <= max_slope:
                return
            slopes.append(slope)
            predicted[start:stop] = segment_pred
            segments.append(
                TravelTimeModelSegment(
                    branch_id=branch_id,
                    segment_index=segment_index,
                    x_min=float(x[start]),
                    x_max=float(x[stop - 1]),
                    slope_s_per_m=float(slope),
                    intercept_s=float(intercept),
                    apparent_velocity_m_s=float(1.0 / slope),
                )
            )
        for left_slope, right_slope in zip(slopes, slopes[1:]):
            if right_slope > left_slope * 1.10:
                return
        rmse = math.sqrt(float(np.average((t - predicted) ** 2, weights=weights)))
        penalty = 0.0015 * max(0, len(boundaries) - 1)
        candidates.append((len(boundaries), rmse + penalty, segments, predicted))

    add_candidate([(0, n)])
    if max_segments >= 2:
        for split in range(2, n - 1):
            add_candidate([(0, split), (split, n)])
    if max_segments >= 3:
        for split_a in range(2, n - 3):
            for split_b in range(split_a + 2, n - 1):
                add_candidate([(0, split_a), (split_a, split_b), (split_b, n)])

    if not candidates:
        raise ValueError(f"No physically valid velocity model for branch {branch_id}.")

    candidates.sort(key=lambda item: (item[1], item[0]))
    best_score = candidates[0][1]
    simplest = min(
        (candidate for candidate in candidates if candidate[1] <= best_score + 0.0015),
        key=lambda item: (item[0], item[1]),
    )
    _n_segments, _score, segments, predicted = simplest
    rmse = math.sqrt(float(np.average((t - predicted) ** 2, weights=weights)))

    previous_segment: Optional[TravelTimeModelSegment] = None
    for segment in segments:
        if previous_segment is not None:
            denominator = segment.slope_s_per_m - previous_segment.slope_s_per_m
            if abs(denominator) > 1e-12:
                segment.crossover_offset_m = float((previous_segment.intercept_s - segment.intercept_s) / denominator)
            segment.intercept_time_s = float(segment.intercept_s)
            v1 = previous_segment.apparent_velocity_m_s
            v2 = segment.apparent_velocity_m_s
            ti = segment.intercept_s
            if ti > 0 and v2 > v1 > 0:
                segment.depth_estimate_m = float(ti * v1 * v2 / (2.0 * math.sqrt(max(v2 * v2 - v1 * v1, 1e-12))))
        previous_segment = segment

    return segments, float(rmse), f"{len(segments)} segment(s)"


def _predict_branch_times(x_abs: np.ndarray, segments: Sequence[TravelTimeModelSegment]) -> np.ndarray:
    result = np.full(x_abs.shape, np.nan, dtype=float)
    if not segments:
        return result
    sorted_segments = sorted(segments, key=lambda item: (item.x_min, item.segment_index))
    for i, x_value in enumerate(x_abs):
        chosen = sorted_segments[-1]
        for segment in sorted_segments:
            if segment.x_min <= x_value <= segment.x_max:
                chosen = segment
                break
            if x_value < segment.x_min:
                chosen = segment
                break
        result[i] = chosen.slope_s_per_m * x_value + chosen.intercept_s
    return result


def fit_velocity_traveltime_model(
    source_x: Sequence[float],
    receiver_x: Sequence[float],
    times_s: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    anchor_mask: Optional[Sequence[bool]] = None,
    max_segments: int = 3,
    velocity_bounds: Tuple[float, float] = (100.0, 8000.0),
) -> TravelTimeModel:
    """Fit a simple piecewise-linear 1-D travel-time model.

    The fitted x-axis is absolute source-receiver offset. If the source lies
    inside a receiver spread, left and right receiver branches are fitted
    independently.
    """

    sx = _as_1d_float(source_x, "source_x")
    rx = _as_1d_float(receiver_x, "receiver_x")
    tt = _as_1d_float(times_s, "times_s")
    if not (sx.size == rx.size == tt.size):
        raise ValueError("source_x, receiver_x, and times_s must have the same length.")
    if weights is None:
        w = np.ones_like(tt, dtype=float)
    else:
        w = _as_1d_float(weights, "weights")
        if w.size != tt.size:
            raise ValueError("weights must match times_s length.")
    if anchor_mask is None:
        anchors = np.zeros_like(tt, dtype=bool)
    else:
        anchors = np.asarray(anchor_mask, dtype=bool).reshape(-1)
        if anchors.size != tt.size:
            raise ValueError("anchor_mask must match times_s length.")

    valid = np.isfinite(sx) & np.isfinite(rx) & np.isfinite(tt) & (tt > 0) & np.isfinite(w) & (w > 0)
    if int(valid.sum()) < 2:
        raise ValueError("At least two valid travel-time picks are required.")

    sx_valid = sx[valid]
    rx_valid = rx[valid]
    tt_valid = tt[valid]
    w_valid = w[valid]
    anchors_valid = anchors[valid]
    branch_ids = _branch_ids_from_geometry(sx_valid, rx_valid)
    x_abs = np.abs(rx_valid - sx_valid)

    segments: List[TravelTimeModelSegment] = []
    rms_values: List[float] = []
    branch_messages: List[str] = []
    for branch_id in sorted(set(str(value) for value in branch_ids)):
        branch_mask = branch_ids == branch_id
        if int(branch_mask.sum()) < 2:
            continue
        exact_result = _fit_anchor_exact_branch_segments(
            branch_id=branch_id,
            x=x_abs[branch_mask],
            t=tt_valid[branch_mask],
            anchor_mask=anchors_valid[branch_mask],
            weights=w_valid[branch_mask],
            velocity_bounds=velocity_bounds,
        )
        if exact_result is None:
            branch_segments, branch_rmse, branch_message = _fit_branch_segments(
                branch_id=branch_id,
                x=x_abs[branch_mask],
                t=tt_valid[branch_mask],
                weights=w_valid[branch_mask],
                max_segments=max_segments,
                velocity_bounds=velocity_bounds,
            )
        else:
            branch_segments, branch_rmse, branch_message = exact_result
        segments.extend(branch_segments)
        rms_values.append(branch_rmse)
        branch_messages.append(f"{branch_id}: {branch_message}")

    if not segments:
        exact_result = _fit_anchor_exact_branch_segments(
            branch_id="all",
            x=x_abs,
            t=tt_valid,
            anchor_mask=anchors_valid,
            weights=w_valid,
            velocity_bounds=velocity_bounds,
        )
        if exact_result is None:
            branch_segments, branch_rmse, branch_message = _fit_branch_segments(
                branch_id="all",
                x=x_abs,
                t=tt_valid,
                weights=w_valid,
                max_segments=max_segments,
                velocity_bounds=velocity_bounds,
            )
        else:
            branch_segments, branch_rmse, branch_message = exact_result
        segments.extend(branch_segments)
        rms_values.append(branch_rmse)
        branch_messages.append(f"all: {branch_message}")

    model = TravelTimeModel(
        segments=segments,
        residual_rms_s=float(np.nanmean(rms_values)) if rms_values else float("nan"),
        branch_ids=sorted({segment.branch_id for segment in segments}),
        message="; ".join(branch_messages),
    )
    model.predicted_times = predict_velocity_traveltimes(model, sx, rx)
    return model


def predict_velocity_traveltimes(
    model: TravelTimeModel,
    source_x: Sequence[float],
    receiver_x: Sequence[float],
) -> np.ndarray:
    """Predict travel times from a fitted 1-D velocity model."""

    sx = _as_1d_float(source_x, "source_x")
    rx = _as_1d_float(receiver_x, "receiver_x")
    if sx.size != rx.size:
        raise ValueError("source_x and receiver_x must have the same length.")
    if not model.segments:
        return np.full(sx.shape, np.nan, dtype=float)

    branch_ids = _branch_ids_from_geometry(sx, rx)
    x_abs = np.abs(rx - sx)
    result = np.full(sx.shape, np.nan, dtype=float)
    segments_by_branch: Dict[str, List[TravelTimeModelSegment]] = {}
    for segment in model.segments:
        segments_by_branch.setdefault(segment.branch_id, []).append(segment)

    for branch_id in sorted(set(str(value) for value in branch_ids)):
        branch_mask = branch_ids == branch_id
        segments = segments_by_branch.get(branch_id) or segments_by_branch.get("all") or model.segments
        result[branch_mask] = _predict_branch_times(x_abs[branch_mask], segments)
    return result


_SEGY_SAMPLE_BYTES = {
    1: 4,  # IBM 32-bit float
    2: 4,  # 32-bit integer
    3: 2,  # 16-bit integer
    5: 4,  # IEEE 32-bit float
    8: 1,  # 8-bit integer
}


def _read_i2(buffer: bytes, start: int) -> int:
    return struct.unpack(">h", buffer[start : start + 2])[0]


def _read_i4(buffer: bytes, start: int) -> int:
    return struct.unpack(">i", buffer[start : start + 4])[0]


def _scalar_multiplier(scalar: int) -> float:
    if scalar > 0:
        return float(scalar)
    if scalar < 0:
        return 1.0 / abs(float(scalar))
    return 1.0


def _ibm_float32_to_native(raw: bytes) -> np.ndarray:
    """Decode big-endian IBM 32-bit floats to native float32."""

    words = np.frombuffer(raw, dtype=">u4").astype(np.uint32)
    if words.size == 0:
        return np.array([], dtype=np.float32)

    sign = np.where((words & 0x80000000) != 0, -1.0, 1.0)
    exponent = ((words >> 24) & 0x7F).astype(np.int32) - 64
    fraction = (words & 0x00FFFFFF).astype(np.float64) / float(0x01000000)
    values = sign * fraction * np.power(16.0, exponent)
    values[words == 0] = 0.0
    return values.astype(np.float32)


def _decode_trace_samples(raw: bytes, format_code: int) -> np.ndarray:
    if format_code == 1:
        return _ibm_float32_to_native(raw)
    if format_code == 2:
        return np.frombuffer(raw, dtype=">i4").astype(np.float32)
    if format_code == 3:
        return np.frombuffer(raw, dtype=">i2").astype(np.float32)
    if format_code == 5:
        return np.frombuffer(raw, dtype=">f4").astype(np.float32)
    if format_code == 8:
        return np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    raise ValueError(f"Unsupported SEG-Y sample format code: {format_code}")


def _parse_trace_header(header: bytes) -> SeismicTraceHeader:
    coord_scale = _scalar_multiplier(_read_i2(header, 70))
    elev_scale = _scalar_multiplier(_read_i2(header, 68))

    source_x = _read_i4(header, 72) * coord_scale
    source_y = _read_i4(header, 76) * coord_scale
    receiver_x = _read_i4(header, 80) * coord_scale
    receiver_y = _read_i4(header, 84) * coord_scale
    source_z = _read_i4(header, 44) * elev_scale
    receiver_z = _read_i4(header, 40) * elev_scale
    offset = _read_i4(header, 36) * coord_scale

    if not np.isfinite(offset) or offset == 0.0:
        dx = receiver_x - source_x
        dy = receiver_y - source_y
        dz = receiver_z - source_z
        offset = math.sqrt(dx * dx + dy * dy + dz * dz)

    return SeismicTraceHeader(
        field_record=_read_i4(header, 8),
        trace_number=_read_i4(header, 12),
        energy_source_point=_read_i4(header, 16),
        source_x=float(source_x),
        source_y=float(source_y),
        source_z=float(source_z),
        receiver_x=float(receiver_x),
        receiver_y=float(receiver_y),
        receiver_z=float(receiver_z),
        offset=float(offset),
    )


def _infer_missing_field_records(headers: List[SeismicTraceHeader]) -> None:
    """Fill zero field records by detecting channel-number resets."""

    if not headers or any(header.field_record != 0 for header in headers):
        return

    record = 1
    previous_trace = None
    for header in headers:
        trace_number = header.trace_number
        if previous_trace is not None and trace_number <= previous_trace:
            record += 1
        header.field_record = record
        previous_trace = trace_number


def _read_segy_builtin(
    file: str,
    max_traces: Optional[int] = None,
    load_traces: bool = True,
) -> SeismicDataset:
    path = Path(file)
    with path.open("rb") as f:
        text_header = f.read(3200)
        binary_header = f.read(400)
        if len(text_header) != 3200 or len(binary_header) != 400:
            raise ValueError(f"{file} is too small to be a SEG-Y file.")

        sample_interval_us = _read_i2(binary_header, 16)
        samples_per_trace = _read_i2(binary_header, 20)
        format_code = _read_i2(binary_header, 24)
        sample_bytes = _SEGY_SAMPLE_BYTES.get(format_code)
        if sample_bytes is None:
            raise ValueError(f"Unsupported SEG-Y sample format code: {format_code}")

        headers: List[SeismicTraceHeader] = []
        traces: List[np.ndarray] = []
        trace_counter = 0
        while True:
            if max_traces is not None and trace_counter >= max_traces:
                break
            trace_header = f.read(240)
            if not trace_header:
                break
            if len(trace_header) != 240:
                raise ValueError("Truncated SEG-Y trace header encountered.")

            ns_trace = _read_i2(trace_header, 114) or samples_per_trace
            sample_raw = f.read(ns_trace * sample_bytes)
            if len(sample_raw) != ns_trace * sample_bytes:
                raise ValueError("Truncated SEG-Y trace samples encountered.")

            headers.append(_parse_trace_header(trace_header))
            if load_traces:
                trace = _decode_trace_samples(sample_raw, format_code)
                if trace.size != samples_per_trace:
                    if trace.size > samples_per_trace:
                        trace = trace[:samples_per_trace]
                    else:
                        trace = np.pad(trace, (0, samples_per_trace - trace.size))
                traces.append(trace)
            trace_counter += 1

    _infer_missing_field_records(headers)
    data = (
        np.column_stack(traces).astype(np.float32, copy=False)
        if traces
        else np.empty((samples_per_trace, 0), dtype=np.float32)
    )
    time = np.arange(samples_per_trace, dtype=float) * float(sample_interval_us) * 1e-6
    metadata = SegyMetadata(
        sample_interval_us=int(sample_interval_us),
        samples_per_trace=int(samples_per_trace),
        format_code=int(format_code),
        trace_count=len(headers),
        file_size_bytes=int(path.stat().st_size),
    )
    return SeismicDataset(str(path), data, time, headers, metadata)


def _read_segy_obspy(file: str, max_traces: Optional[int] = None) -> SeismicDataset:
    from obspy import read

    stream = read(file, format="SEGY")
    if max_traces is not None:
        stream = stream[:max_traces]
    traces = [np.asarray(trace.data, dtype=np.float32) for trace in stream]
    if not traces:
        raise ValueError(f"No traces found in SEG-Y file: {file}")

    n_samples = len(traces[0])
    dt = float(stream[0].stats.delta)
    headers: List[SeismicTraceHeader] = []
    for idx, trace in enumerate(stream):
        segy_header = getattr(trace.stats, "segy", None)
        trace_header = getattr(segy_header, "trace_header", None)
        get = lambda name, default=0: getattr(trace_header, name, default) if trace_header else default
        coord_scale = _scalar_multiplier(int(get("scalar_to_be_applied_to_all_coordinates", 1) or 1))
        elev_scale = _scalar_multiplier(int(get("scalar_to_be_applied_to_all_elevations_and_depths", 1) or 1))
        source_x = float(get("source_coordinate_x", 0)) * coord_scale
        receiver_x = float(get("group_coordinate_x", idx + 1)) * coord_scale
        source_z = float(get("surface_elevation_at_source", 0)) * elev_scale
        receiver_z = float(get("receiver_group_elevation", 0)) * elev_scale
        headers.append(
            SeismicTraceHeader(
                field_record=int(get("original_field_record_number", 1)),
                trace_number=int(get("trace_number_within_the_original_field_record", idx + 1)),
                energy_source_point=int(get("energy_source_point_number", 0)),
                source_x=source_x,
                source_y=float(get("source_coordinate_y", 0)) * coord_scale,
                source_z=source_z,
                receiver_x=receiver_x,
                receiver_y=float(get("group_coordinate_y", 0)) * coord_scale,
                receiver_z=receiver_z,
                offset=float(get("distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group", receiver_x - source_x))
                * coord_scale,
            )
        )

    _infer_missing_field_records(headers)
    return SeismicDataset(
        path=str(Path(file)),
        traces=np.column_stack(traces).astype(np.float32, copy=False),
        time=np.arange(n_samples, dtype=float) * dt,
        headers=headers,
        metadata=SegyMetadata(
            sample_interval_us=int(round(dt * 1e6)),
            samples_per_trace=n_samples,
            format_code=0,
            trace_count=len(headers),
            file_size_bytes=int(Path(file).stat().st_size),
        ),
    )


def _read_segy_segyio(file: str, max_traces: Optional[int] = None) -> SeismicDataset:
    """Read a SEG-Y file with ``segyio`` (robust to endianness, sample format,
    header byte locations, and SEG-Y rev 1/2). Opened with ``ignore_geometry``
    so arbitrary shot gathers (no regular inline/crossline grid) read cleanly.
    """
    import segyio

    # Standard SEG-Y trace-header byte positions (1-indexed); segyio.Field
    # accepts these integer keys directly.
    b_fieldrec, b_traceno, b_esp = 9, 13, 17
    b_offset, b_recv_elev, b_src_elev = 37, 41, 45
    b_elev_scale, b_coord_scale = 69, 71
    b_sx, b_sy, b_gx, b_gy = 73, 77, 81, 85

    with segyio.open(file, "r", ignore_geometry=True) as f:
        f.mmap()
        n_total = int(f.tracecount)
        n = n_total if max_traces is None else min(int(max_traces), n_total)
        n_samples = int(np.asarray(f.samples).size)
        dt_us = int(f.bin[segyio.BinField.Interval]) or 0
        format_code = int(f.format)
        sample_ms = np.asarray(f.samples, dtype=float)
        data = np.zeros((n_samples, n), dtype=np.float32)
        headers: List[SeismicTraceHeader] = []
        for i in range(n):
            data[:, i] = np.asarray(f.trace[i], dtype=np.float32)
            h = f.header[i]
            coord_scale = _scalar_multiplier(int(h[b_coord_scale] or 1))
            elev_scale = _scalar_multiplier(int(h[b_elev_scale] or 1))
            sx = float(h[b_sx]) * coord_scale
            sy = float(h[b_sy]) * coord_scale
            gx = float(h[b_gx]) * coord_scale
            gy = float(h[b_gy]) * coord_scale
            sz = float(h[b_src_elev]) * elev_scale
            gz = float(h[b_recv_elev]) * elev_scale
            offset = float(h[b_offset]) * coord_scale
            if not np.isfinite(offset) or offset == 0.0:
                offset = math.hypot(gx - sx, gy - sy)
            headers.append(
                SeismicTraceHeader(
                    field_record=int(h[b_fieldrec]),
                    trace_number=int(h[b_traceno]),
                    energy_source_point=int(h[b_esp]),
                    source_x=sx, source_y=sy, source_z=sz,
                    receiver_x=gx, receiver_y=gy, receiver_z=gz,
                    offset=offset,
                )
            )

    if dt_us <= 0 and sample_ms.size > 1:
        dt_us = int(round(abs(sample_ms[1] - sample_ms[0]) * 1000.0))  # ms -> us
    dt_s = dt_us * 1e-6 if dt_us > 0 else 1.0
    time = np.arange(n_samples, dtype=float) * dt_s
    _infer_missing_field_records(headers)
    return SeismicDataset(
        path=str(Path(file)),
        traces=data,
        time=time,
        headers=headers,
        metadata=SegyMetadata(
            sample_interval_us=int(dt_us if dt_us > 0 else 0),
            samples_per_trace=n_samples,
            format_code=format_code,
            trace_count=len(headers),
            file_size_bytes=int(Path(file).stat().st_size),
        ),
    )


def read_segy(
    file: str,
    max_traces: Optional[int] = None,
    load_traces: bool = True,
    prefer_obspy: bool = True,
) -> SeismicDataset:
    """Read a SEG-Y file into traces, headers, and metadata.

    Reader order is ``segyio`` -> ``ObsPy`` -> built-in: the most robust library
    available is used, and the built-in conservative reader is the final fallback
    so reads keep working with no third-party SEG-Y dependency installed.

    Parameters
    ----------
    file : str
        SEG-Y path.
    max_traces : int, optional
        Maximum number of traces to read. Useful for responsive GUI previews.
    load_traces : bool, optional
        If False, parse headers but skip sample arrays (built-in reader only).
    prefer_obspy : bool, optional
        When True (default), try ObsPy if ``segyio`` is unavailable or fails.

    Returns
    -------
    SeismicDataset
        Parsed seismic dataset.
    """

    if load_traces:
        # 1) segyio: broadest, most reliable coverage when installed.
        try:
            return _read_segy_segyio(file, max_traces=max_traces)
        except ImportError:
            pass
        except Exception:
            pass
        # 2) ObsPy: broader format support than the built-in reader.
        if prefer_obspy:
            try:
                return _read_segy_obspy(file, max_traces=max_traces)
            except ImportError:
                pass
            except Exception:
                # Fall back to the simple reader for the bundled example and
                # other classic big-endian SEG-Y files.
                pass
    # 3) Built-in conservative big-endian reader (no third-party deps).
    return _read_segy_builtin(file, max_traces=max_traces, load_traces=load_traces)


def _read_u2_le(buffer: bytes, start: int) -> int:
    return struct.unpack("<H", buffer[start : start + 2])[0]


def _read_u4_le(buffer: bytes, start: int) -> int:
    return struct.unpack("<I", buffer[start : start + 4])[0]


def _parse_geometrics_tags(buffer: bytes, start: int, end: int) -> Dict[str, str]:
    """Parse Geometrics length-prefixed text tags from one header block."""

    tags: Dict[str, str] = {}
    position = int(start)
    end = min(int(end), len(buffer))
    while position + 2 <= end:
        record_length = _read_u2_le(buffer, position)
        if record_length < 2 or position + record_length > end:
            break
        raw = buffer[position + 2 : position + record_length].rstrip(b"\x00")
        try:
            text = raw.decode("latin1").strip()
        except UnicodeDecodeError:
            text = ""
        if text:
            parts = text.split(maxsplit=1)
            key = parts[0].strip().upper()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key:
                tags[key] = value
        position += record_length
    return tags


def _float_tag(tags: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = tags.get(key.upper())
    if value is None:
        return float(default)
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return float(default)


def _int_tag(tags: Dict[str, str], key: str, default: int = 0) -> int:
    value = tags.get(key.upper())
    if value is None:
        return int(default)
    try:
        return int(float(str(value).split()[0]))
    except (TypeError, ValueError, IndexError):
        return int(default)


def _numeric_file_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def _read_geometrics_dat_file(
    file: Path,
    field_record_fallback: int,
    max_traces: Optional[int],
    load_traces: bool,
) -> Tuple[List[np.ndarray], List[SeismicTraceHeader], float, int]:
    """Read one Geometrics binary DAT shot gather."""

    raw = file.read_bytes()
    if len(raw) < 64:
        raise ValueError(f"{file} is too small to be a Geometrics DAT file.")

    n_channels = _read_u2_le(raw, 6)
    if n_channels <= 0 or n_channels > 10000:
        raise ValueError(f"Could not infer channel count from Geometrics DAT file: {file}")

    pointer_table_start = 32
    pointer_table_end = pointer_table_start + 4 * n_channels
    if pointer_table_end > len(raw):
        raise ValueError(f"Truncated Geometrics DAT pointer table: {file}")

    trace_offsets = [_read_u4_le(raw, pointer_table_start + 4 * i) for i in range(n_channels)]
    trace_offsets = [offset for offset in trace_offsets if 0 < offset < len(raw)]
    if not trace_offsets:
        raise ValueError(f"No trace blocks found in Geometrics DAT file: {file}")

    traces: List[np.ndarray] = []
    headers: List[SeismicTraceHeader] = []
    dt = 0.001
    n_samples = 0
    trace_limit = len(trace_offsets) if max_traces is None else min(len(trace_offsets), int(max_traces))

    for local_index, trace_offset in enumerate(trace_offsets[:trace_limit]):
        if trace_offset + 32 > len(raw):
            raise ValueError(f"Truncated Geometrics DAT trace header in {file}.")
        header_length = _read_u2_le(raw, trace_offset + 2)
        data_bytes = _read_u4_le(raw, trace_offset + 4)
        samples_per_trace = _read_u4_le(raw, trace_offset + 8)
        bytes_per_sample = _read_u4_le(raw, trace_offset + 12)
        if header_length < 32 or trace_offset + header_length > len(raw):
            raise ValueError(f"Invalid Geometrics DAT trace-header length in {file}.")
        if samples_per_trace <= 0 or bytes_per_sample <= 0:
            raise ValueError(f"Invalid Geometrics DAT sample metadata in {file}.")

        data_start = trace_offset + header_length
        data_stop = data_start + data_bytes
        if data_stop > len(raw):
            raise ValueError(f"Truncated Geometrics DAT trace samples in {file}.")

        tags = _parse_geometrics_tags(raw, trace_offset + 32, trace_offset + header_length)
        dt = _float_tag(tags, "SAMPLE_INTERVAL", dt)
        n_samples = int(samples_per_trace)
        channel_number = _int_tag(tags, "CHANNEL_NUMBER", local_index + 1)
        field_record = _int_tag(tags, "SHOT_SEQUENCE_NUMBER", field_record_fallback)
        source_x = _float_tag(tags, "SOURCE_LOCATION", float(field_record_fallback))
        receiver_x = _float_tag(tags, "RECEIVER_LOCATION", float(local_index + 1))
        offset = receiver_x - source_x

        headers.append(
            SeismicTraceHeader(
                field_record=int(field_record),
                trace_number=int(channel_number),
                energy_source_point=int(field_record),
                source_x=float(source_x),
                source_y=0.0,
                source_z=0.0,
                receiver_x=float(receiver_x),
                receiver_y=0.0,
                receiver_z=0.0,
                offset=float(offset),
            )
        )

        if not load_traces:
            continue
        sample_raw = raw[data_start:data_stop]
        expected_bytes = int(samples_per_trace) * int(bytes_per_sample)
        if len(sample_raw) < expected_bytes:
            raise ValueError(f"Truncated Geometrics DAT sample payload in {file}.")
        sample_raw = sample_raw[:expected_bytes]
        if bytes_per_sample == 4:
            trace = np.frombuffer(sample_raw, dtype="<f4").astype(np.float32)
        elif bytes_per_sample == 2:
            trace = np.frombuffer(sample_raw, dtype="<i2").astype(np.float32)
            trace *= _float_tag(tags, "DESCALING_FACTOR", 1.0)
        elif bytes_per_sample == 1:
            trace = np.frombuffer(sample_raw, dtype=np.int8).astype(np.float32)
            trace *= _float_tag(tags, "DESCALING_FACTOR", 1.0)
        else:
            raise ValueError(f"Unsupported Geometrics DAT sample size ({bytes_per_sample} bytes) in {file}.")
        if trace.size != samples_per_trace:
            raise ValueError(f"Geometrics DAT trace has {trace.size} samples; expected {samples_per_trace}.")
        traces.append(trace)

    if dt <= 0:
        raise ValueError(f"Invalid Geometrics DAT sample interval in {file}.")
    return traces, headers, float(dt), int(n_samples)


def read_geometrics_dat(
    path: str,
    max_traces: Optional[int] = None,
    load_traces: bool = True,
) -> SeismicDataset:
    """Read Geometrics binary DAT seismic records.

    Parameters
    ----------
    path : str
        Either a single Geometrics ``.dat`` file or a directory containing one
        ``.dat`` file per shot gather.
    max_traces : int, optional
        Maximum number of traces to read across all files.
    load_traces : bool, optional
        If False, parse headers but skip sample arrays.

    Returns
    -------
    SeismicDataset
        Dataset with all shots concatenated by trace column and separated by
        ``field_record``.
    """

    root = Path(path).expanduser()
    if root.is_dir():
        files = sorted(root.glob("*.dat"), key=_numeric_file_sort_key)
        if not files:
            raise ValueError(f"No .dat files found in Geometrics directory: {root}")
    elif root.is_file():
        files = [root]
    else:
        raise FileNotFoundError(f"Geometrics DAT path not found: {root}")

    all_traces: List[np.ndarray] = []
    all_headers: List[SeismicTraceHeader] = []
    sample_interval_s: Optional[float] = None
    samples_per_trace: Optional[int] = None
    remaining = None if max_traces is None else int(max_traces)

    for file_index, file in enumerate(files, start=1):
        if remaining is not None and remaining <= 0:
            break
        traces, headers, dt, n_samples = _read_geometrics_dat_file(
            file=file,
            field_record_fallback=file_index,
            max_traces=remaining,
            load_traces=load_traces,
        )
        if sample_interval_s is None:
            sample_interval_s = dt
        elif not math.isclose(sample_interval_s, dt, rel_tol=0.0, abs_tol=max(sample_interval_s * 1e-6, 1e-12)):
            raise ValueError("Geometrics DAT files have inconsistent sample intervals.")
        if samples_per_trace is None:
            samples_per_trace = n_samples
        elif samples_per_trace != n_samples:
            raise ValueError("Geometrics DAT files have inconsistent samples per trace.")
        all_headers.extend(headers)
        all_traces.extend(traces)
        if remaining is not None:
            remaining -= len(headers)

    if not all_headers:
        raise ValueError(f"No traces found in Geometrics DAT input: {root}")
    if sample_interval_s is None or samples_per_trace is None:
        raise ValueError(f"Could not infer Geometrics DAT timing metadata: {root}")

    if load_traces:
        data = np.column_stack(all_traces).astype(np.float32, copy=False) if all_traces else np.empty((samples_per_trace, 0))
    else:
        data = np.empty((int(samples_per_trace), 0), dtype=np.float32)
    time = np.arange(int(samples_per_trace), dtype=float) * float(sample_interval_s)
    metadata = SegyMetadata(
        sample_interval_us=int(round(float(sample_interval_s) * 1e6)),
        samples_per_trace=int(samples_per_trace),
        format_code=200,
        trace_count=len(all_headers),
        file_size_bytes=int(sum(file.stat().st_size for file in files)),
    )
    return SeismicDataset(str(root), data, time, all_headers, metadata)


def normalize_traces(data: np.ndarray, trace_axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Normalize each seismic trace by its maximum absolute amplitude."""

    arr = np.asarray(data, dtype=float)
    max_abs = np.max(np.abs(arr), axis=0 if trace_axis == 1 else 1, keepdims=True)
    max_abs = np.where(max_abs > eps, max_abs, 1.0)
    return np.nan_to_num(arr / max_abs)


def apply_agc(
    data: np.ndarray,
    dt: float,
    window: float = 0.05,
    rms: float = 1.0,
) -> np.ndarray:
    """Apply gate-based automatic gain control to traces.

    The implementation follows the MATLAB example's gate/interpolation logic
    while handling zero-energy windows safely.
    """

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be a 2-D array of samples by traces.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    n_samples, n_traces = arr.shape
    gate = max(1, int(round(float(window) / float(dt))))
    n_gates = max(1, int(math.ceil(n_samples / gate)))
    sample_index = np.arange(n_samples, dtype=float)
    output = np.zeros_like(arr, dtype=float)

    for itrace in range(n_traces):
        trace = arr[:, itrace]
        if np.any(~np.isfinite(trace)):
            continue
        centers: List[float] = []
        gains: List[float] = []
        for igate in range(n_gates):
            start = igate * gate
            stop = min((igate + 1) * gate, n_samples)
            segment = trace[start:stop]
            energy = float(np.mean(segment * segment)) if segment.size else 0.0
            gain = float(rms) / math.sqrt(energy) if energy > 0 else 1.0
            centers.append(0.5 * (start + stop - 1))
            gains.append(gain)

        interp_x = np.array([0.0, *centers, float(n_samples - 1)])
        interp_gain = np.array([gains[0], *gains, gains[-1]])
        gain_curve = np.interp(sample_index, interp_x, interp_gain)
        output[:, itrace] = trace * gain_curve

    return output


def tukey_taper(n_samples: int, taper_samples: int) -> np.ndarray:
    """Return a Tukey-style taper with taper length in samples."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if taper_samples <= 0:
        return np.ones(n_samples, dtype=float)

    taper_samples = min(int(taper_samples), max(1, (n_samples - 1) // 2))
    window = np.ones(n_samples, dtype=float)
    ramp = np.arange(taper_samples, dtype=float)
    left = 0.5 * (1.0 + np.cos(np.pi * (ramp / taper_samples - 1.0)))
    window[:taper_samples] = left
    window[-taper_samples:] = left[::-1]
    return window


def bandpass_filter(
    data: np.ndarray,
    dt: float,
    f1: float,
    f2: float,
    f3: float,
    f4: float,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass using the pass-band edges.

    ``f1`` and ``f4`` are retained for API compatibility with Ormsby-style
    four-corner filters; this implementation uses ``f2`` and ``f3`` as the
    pass-band edges.
    """

    del f1, f4
    arr = np.asarray(data, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    nyquist = 0.5 / dt
    low = max(float(f2), 1e-6) / nyquist
    high = min(float(f3), nyquist * 0.999) / nyquist
    if not 0 < low < high < 1:
        raise ValueError("Invalid bandpass frequencies for the sampling interval.")

    from scipy.signal import butter, filtfilt

    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, arr, axis=0)


def _as_dataset_and_headers(
    source: SeismicDataset | SeismicShotGather | np.ndarray,
    headers: Optional[Sequence[SeismicTraceHeader]] = None,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[SeismicTraceHeader], np.ndarray]:
    if isinstance(source, SeismicDataset):
        return source.traces, source.time, source.headers, np.arange(source.traces.shape[1])
    if isinstance(source, SeismicShotGather):
        return source.traces, source.time, source.headers, source.trace_indices
    arr = np.asarray(source, dtype=float)
    if arr.ndim != 2:
        raise ValueError("raw trace input must be 2-D samples by traces.")
    if headers is None:
        headers = [
            SeismicTraceHeader(
                field_record=1,
                trace_number=i + 1,
                energy_source_point=1,
                source_x=0.0,
                source_y=0.0,
                source_z=0.0,
                receiver_x=float(i + 1),
                receiver_y=0.0,
                receiver_z=0.0,
                offset=float(i + 1),
            )
            for i in range(arr.shape[1])
        ]
    if dt is None:
        raise ValueError("dt is required when passing raw trace arrays.")
    time = np.arange(arr.shape[0], dtype=float) * float(dt)
    return arr, time, list(headers), np.arange(arr.shape[1])


def _fallback_pick_geometry(header: SeismicTraceHeader) -> Tuple[float, float, float, float]:
    source_id = header.energy_source_point or header.field_record or 1
    receiver_id = header.trace_number or 1

    source_x = header.source_x
    receiver_x = header.receiver_x
    if not np.isfinite(source_x):
        source_x = float(source_id)
    if not np.isfinite(receiver_x):
        receiver_x = float(receiver_id)

    if source_x == 0.0 and receiver_x == 0.0:
        source_x = float(source_id)
        receiver_x = float(receiver_id)
    elif receiver_x == source_x and header.offset:
        receiver_x = source_x + float(header.offset)

    return float(source_x), float(header.source_z), float(receiver_x), float(header.receiver_z)


def pick_first_breaks(
    data: SeismicDataset | SeismicShotGather | np.ndarray,
    dt: Optional[float] = None,
    headers: Optional[Sequence[SeismicTraceHeader]] = None,
    threshold: float = 0.2,
    noise_multiplier: float = 5.0,
    min_time: float = 0.0,
    max_time: Optional[float] = None,
    polarity: float = 1.0,
) -> List[FirstBreakPick]:
    """Pick first arrivals using a simple amplitude/noise threshold.

    This assisted picker is intended as a starting point for GUI review rather
    than a final scientific picking algorithm.
    """

    traces, time, trace_headers, trace_indices = _as_dataset_and_headers(data, headers=headers, dt=dt)
    if traces.size == 0:
        return []

    picks: List[FirstBreakPick] = []
    start = int(np.searchsorted(time, min_time, side="left"))
    stop = int(np.searchsorted(time, max_time, side="right")) if max_time is not None else len(time)
    start = max(0, min(start, len(time) - 1))
    stop = max(start + 1, min(stop, len(time)))
    # Noise window must lie entirely before the 'start' sample so that actual
    # first arrivals do not contaminate the noise estimate.
    noise_stop = max(2, min(start, int(max(1, 0.05 * len(time)))))

    for itrace, header in enumerate(trace_headers):
        trace = np.asarray(traces[:, itrace], dtype=float) * float(polarity)
        window = trace[start:stop]
        if window.size == 0 or not np.any(np.isfinite(window)):
            pick_time = float("nan")
            amplitude = float("nan")
        else:
            abs_window = np.abs(np.nan_to_num(window))
            noise = np.nanstd(trace[:noise_stop])
            absolute_threshold = max(float(threshold) * float(np.nanmax(abs_window)), float(noise_multiplier) * float(noise))
            crossings = np.where(abs_window >= absolute_threshold)[0]
            if crossings.size:
                idx = start + int(crossings[0])
            else:
                idx = start + int(np.nanargmax(abs_window))
            pick_time = float(time[idx])
            amplitude = float(trace[idx])

        source_x, source_z, receiver_x, receiver_z = _fallback_pick_geometry(header)
        picks.append(
            FirstBreakPick(
                source_id=int(header.energy_source_point or header.field_record or 1),
                receiver_id=int(header.trace_number or itrace + 1),
                time_s=pick_time,
                source_x=source_x,
                source_z=source_z,
                receiver_x=receiver_x,
                receiver_z=receiver_z,
                field_record=int(header.field_record),
                trace_number=int(header.trace_number or itrace + 1),
                trace_index=int(trace_indices[itrace]),
                amplitude=amplitude,
            )
        )

    return picks


def _pick_from_any(value: FirstBreakPick | Dict[str, Any]) -> FirstBreakPick:
    if isinstance(value, FirstBreakPick):
        return value

    def float_or_default(raw: Any, default: float) -> float:
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    return FirstBreakPick(
        source_id=int(value.get("source_id", value.get("field_record", 1))),
        receiver_id=int(value.get("receiver_id", value.get("trace_number", 1))),
        time_s=float_or_default(value.get("time_s", value.get("time")), np.nan),
        source_x=float_or_default(value.get("source_x"), np.nan),
        source_z=float_or_default(value.get("source_z"), 0.0),
        receiver_x=float_or_default(value.get("receiver_x", value.get("offset")), np.nan),
        receiver_z=float_or_default(value.get("receiver_z"), 0.0),
        field_record=int(value.get("field_record", value.get("source_id", 1))),
        trace_number=int(value.get("trace_number", value.get("receiver_id", 1))),
        trace_index=int(value.get("trace_index", 0)),
        amplitude=float_or_default(value.get("amplitude"), np.nan),
    )


def export_first_breaks(
    picks: Iterable[FirstBreakPick | Dict[str, Any]],
    filename: str,
) -> str:
    """Export first-break picks to CSV."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    pick_list = [_pick_from_any(pick) for pick in picks]
    fieldnames = list(FirstBreakPick(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 0, 0.0).to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pick in pick_list:
            writer.writerow(pick.to_dict())
    return str(path)


def first_breaks_to_traveltime(
    picks: Iterable[FirstBreakPick | Dict[str, Any]],
    filename: str,
    receiver_spacing: float = 1.0,
    shot_spacing: Optional[float] = None,
) -> str:
    """Export first breaks to a PyGIMLi/BERT travel-time ``.dat`` file."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    pick_list = [
        p
        for p in (_pick_from_any(pick) for pick in picks)
        if np.isfinite(p.time_s) and p.time_s > 0
    ]
    if not pick_list:
        raise ValueError("No finite positive first-break picks to export.")

    min_source = min(pick.source_id for pick in pick_list)
    min_receiver = min(pick.receiver_id for pick in pick_list)
    shot_dx = receiver_spacing if shot_spacing is None else shot_spacing

    def coord_pair(pick: FirstBreakPick, role: str) -> Tuple[float, float]:
        if role == "source":
            x = pick.source_x
            z = pick.source_z
            fallback_x = (pick.source_id - min_source) * shot_dx
        else:
            x = pick.receiver_x
            z = pick.receiver_z
            fallback_x = (pick.receiver_id - min_receiver) * receiver_spacing
        if not np.isfinite(x):
            x = fallback_x
        if not np.isfinite(z):
            z = 0.0
        return (round(float(x), 6), round(float(z), 6))

    sensor_index: Dict[Tuple[float, float], int] = {}
    sensors: List[Tuple[float, float]] = []

    def ensure_sensor(coord: Tuple[float, float]) -> int:
        if coord not in sensor_index:
            sensor_index[coord] = len(sensors) + 1
            sensors.append(coord)
        return sensor_index[coord]

    rows: List[Tuple[int, int, float]] = []
    for pick in pick_list:
        sid = ensure_sensor(coord_pair(pick, "source"))
        gid = ensure_sensor(coord_pair(pick, "receiver"))
        if sid == gid:
            continue
        rows.append((sid, gid, float(pick.time_s)))

    if not rows:
        raise ValueError("No valid source-receiver pairs remained after export filtering.")

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(sensors)}\n")
        f.write("# x y\n")
        for x, z in sensors:
            f.write(f"{x:g}\t{z:g}\n")
        f.write(f"{len(rows)}\n")
        f.write("# s g t\n")
        for sid, gid, time_s in rows:
            f.write(f"{sid}\t{gid}\t{time_s:.9g}\n")

    return str(path)


__all__ = [
    "SegyMetadata",
    "SeismicTraceHeader",
    "SeismicShotGather",
    "SeismicDataset",
    "FirstBreakPick",
    "read_segy",
    "apply_agc",
    "normalize_traces",
    "tukey_taper",
    "bandpass_filter",
    "pick_first_breaks",
    "export_first_breaks",
    "first_breaks_to_traveltime",
]
