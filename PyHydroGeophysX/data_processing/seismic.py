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
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SegyMetadata:
    """Summary metadata from a SEG-Y file.

    ``sample_interval_us`` is the header's whole number of microseconds unless
    a truer value was given (:func:`read_segy`'s ``sample_interval_s``), which
    can be fractional: 31.25 for a 32 kHz record.
    """

    sample_interval_us: float
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
    raw_offset = _read_i4(header, 36)
    offset = raw_offset * coord_scale
    # The standard applies the coordinate scalar to bytes 73-88 and 181-188
    # only, so bytes 37-40 hold the offset unscaled; some writers scale it
    # anyway. Where the coordinates give the distance, take whichever reading
    # agrees with it: a Geode line written with scalar -100 otherwise reads its
    # 2 m offset as 0.02 m.
    distance = math.hypot(receiver_x - source_x, receiver_y - source_y)
    if raw_offset and distance > 0 and coord_scale != 1.0:
        if abs(abs(raw_offset) - distance) < abs(abs(offset) - distance):
            offset = float(raw_offset)

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
    sample_interval_s: Optional[float] = None,
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
    sample_interval_s : float, optional
        The sample interval to use instead of the header's. SEG-Y keeps it in
        whole microseconds, so a 32 kHz record is written as 31 us and every
        time read with it is 0.8% short; :func:`record_sample_interval` finds
        the true value in an acquisition record kept beside the file. Without
        it, a SEG-Y rev 2 file's extended sample interval
        (:func:`segy_extended_sample_interval`) is used when it is set.

    Returns
    -------
    SeismicDataset
        Parsed seismic dataset.
    """

    dataset = None
    if load_traces:
        # 1) segyio: broadest, most reliable coverage when installed.
        try:
            dataset = _read_segy_segyio(file, max_traces=max_traces)
        except ImportError:
            pass
        except Exception:
            pass
        # 2) ObsPy: broader format support than the built-in reader.
        if dataset is None and prefer_obspy:
            try:
                dataset = _read_segy_obspy(file, max_traces=max_traces)
            except ImportError:
                pass
            except Exception:
                # Fall back to the simple reader for the bundled example and
                # other classic big-endian SEG-Y files.
                pass
    # 3) Built-in conservative big-endian reader (no third-party deps).
    if dataset is None:
        dataset = _read_segy_builtin(file, max_traces=max_traces, load_traces=load_traces)
    if sample_interval_s:
        set_sample_interval(dataset, sample_interval_s)
    else:
        extended = segy_extended_sample_interval(file)
        header = float(dataset.metadata.sample_interval_us)
        if extended is not None and abs(extended - header) > 1e-9 and (
                header <= 0 or abs(extended - header) < 1.0):
            set_sample_interval(dataset, extended * 1e-6)
    return dataset


def segy_extended_sample_interval(file: str) -> Optional[float]:
    """The SEG-Y rev 2 extended sample interval (us), or None.

    Rev 2 files may store the interval as an IEEE double in binary-header bytes
    3273-3280, which then overrides the whole microseconds of bytes 3217-3218:
    a 32 kHz Geode record keeps 31.25 there and 31 in the integer field, so
    reading only the latter makes every time 0.8 % short. None for an earlier
    revision, a zero or unreadable value, or a file too short to hold it.
    """
    try:
        with open(file, "rb") as fh:
            fh.seek(3200)
            binary = fh.read(400)
    except OSError:
        return None
    if len(binary) < 302:
        return None
    if binary[300] < 2:          # major revision, byte 3501
        return None
    # Byte-order marker 3297-3300: 0x01020304 read in the file's order.
    marker = binary[96:100]
    order = "<" if marker == b"\x04\x03\x02\x01" else ">"
    value = struct.unpack(order + "d", binary[72:80])[0]
    if not math.isfinite(value) or value <= 0 or value > 1e7:
        return None
    return float(value)


def set_sample_interval(dataset: SeismicDataset, sample_interval_s: float) -> SeismicDataset:
    """Give ``dataset`` the sample interval ``sample_interval_s``: its time axis and metadata."""
    dataset.time = np.arange(dataset.time.size, dtype=float) * float(sample_interval_s)
    dataset.metadata.sample_interval_us = round(float(sample_interval_s) * 1e6, 6)
    return dataset


def record_sample_interval(segy_file: str) -> Optional[float]:
    """The sample interval, in seconds, that an acquisition record beside a SEG-Y file states.

    The record is ``<stem>_record.txt`` beside the file, else the one
    ``*record*.txt`` in its folder - the shot log a seismograph writes with
    its export. Read from a line such as ``Sample interval: 31.25 us`` (us,
    µs or ms), else from ``Record length: 128 ms (4096 samples)``. None
    without a record or either line.

    >>> import tempfile, os
    >>> folder = tempfile.mkdtemp()
    >>> _ = open(os.path.join(folder, "L1_record.txt"), "w").write(
    ...     "Record length: 128 ms (4096 samples)\\nSample interval: 31.25 us\\n")
    >>> record_sample_interval(os.path.join(folder, "L1.sgy"))
    3.125e-05
    """
    import re

    path = Path(segy_file)
    candidates = [path.with_name(f"{path.stem}_record.txt")]
    others = sorted(p for p in path.parent.glob("*record*.txt") if p not in candidates)
    if len(others) == 1:
        candidates += others
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"sample\s*interval\s*[:=]?\s*([0-9.]+)\s*(us|µs|μs|ms)\b", text, re.I)
        if match:
            value = float(match.group(1))
            return value * (1e-3 if match.group(2).lower() == "ms" else 1e-6)
        match = re.search(r"record\s*length\s*[:=]?\s*([0-9.]+)\s*ms\s*\(\s*(\d+)\s*samples", text, re.I)
        if match and int(match.group(2)) > 0:
            return float(match.group(1)) * 1e-3 / int(match.group(2))
    return None


def apply_record_interval(dataset: SeismicDataset, segy_file: str,
                          tolerance: float = 0.05) -> Optional[str]:
    """Use the sample interval the acquisition record states, when the header rounded it.

    SEG-Y keeps the interval in whole microseconds: a 32 kHz record's 31.25 us
    is written 31, and every time read with it comes out 0.8% short - every
    velocity 0.8% too high. A record beside the file
    (:func:`record_sample_interval`) that states an interval within
    ``tolerance`` of the header's is the truer value and is given to
    ``dataset``; one that disagrees by more describes something else, and the
    header's is kept.

    Returns a sentence saying which was used and why, or None when there is no
    record or the two agree.
    """
    stated = record_sample_interval(segy_file)
    header = dataset.metadata.sample_interval_s
    if not stated or header <= 0 or abs(stated - header) <= 1e-9 * stated:
        return None
    if abs(stated - header) / stated < tolerance:
        set_sample_interval(dataset, stated)
        return (f"The SEG-Y header gives the sample interval as {header * 1e6:g} us, in "
                f"whole microseconds; the acquisition record beside it gives "
                f"{stated * 1e6:g} us, which was used. With the header's value every time "
                f"would be {100 * abs(stated - header) / stated:.1f}% "
                f"{'short' if header < stated else 'long'}, and every velocity as much off.")
    return (f"The acquisition record beside the SEG-Y file gives a sample interval of "
            f"{stated * 1e6:g} us, the header {header * 1e6:g} us - more than rounding "
            f"apart, so the header's value was kept.")


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


#: SEG-2 data format codes (byte 12 of a trace descriptor block) and the sample
#: type each stands for. Code 3, 20-bit SEG-D floating point, is not read.
_SEG2_SAMPLE_TYPES = {1: "<i2", 2: "<i4", 4: "<f4", 5: "<f8"}


def _read_geometrics_dat_file(
    file: Path,
    field_record_fallback: int,
    max_traces: Optional[int],
    load_traces: bool,
) -> Tuple[List[np.ndarray], List[SeismicTraceHeader], float, int]:
    """Read one Geometrics binary DAT shot gather.

    Geometrics writes SEG-2. Byte 12 of each trace descriptor is the data
    format code, not a size in bytes: read as a size, only float32 (code 4)
    came out right, while 16- and 32-bit integer traces were decoded as 8- and
    16-bit ones - the right number of samples, all of them wrong.
    """

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
    dt: Optional[float] = None
    n_samples: Optional[int] = None
    delay: Optional[float] = None
    trace_limit = len(trace_offsets) if max_traces is None else min(len(trace_offsets), int(max_traces))

    for local_index, trace_offset in enumerate(trace_offsets[:trace_limit]):
        if trace_offset + 32 > len(raw):
            raise ValueError(f"Truncated Geometrics DAT trace header in {file}.")
        header_length = _read_u2_le(raw, trace_offset + 2)
        data_bytes = _read_u4_le(raw, trace_offset + 4)
        samples_per_trace = _read_u4_le(raw, trace_offset + 8)
        format_code = raw[trace_offset + 12]
        if header_length < 32 or trace_offset + header_length > len(raw):
            raise ValueError(f"Invalid Geometrics DAT trace-header length in {file}.")
        if samples_per_trace <= 0:
            raise ValueError(f"Invalid Geometrics DAT sample metadata in {file}.")
        if format_code not in _SEG2_SAMPLE_TYPES:
            raise ValueError(
                f"Unsupported SEG-2 data format code {format_code} in {file} "
                "(read: 1 = int16, 2 = int32, 4 = float32, 5 = float64).")

        data_start = trace_offset + header_length
        data_stop = data_start + data_bytes
        if data_stop > len(raw):
            raise ValueError(f"Truncated Geometrics DAT trace samples in {file}.")

        tags = _parse_geometrics_tags(raw, trace_offset + 32, trace_offset + header_length)
        # One time axis serves the whole record, so every trace must share it:
        # taking whichever trace came last put the others on the wrong clock.
        # A trace without the tag says nothing about its interval either way.
        trace_dt = _float_tag(tags, "SAMPLE_INTERVAL", math.nan)
        if math.isnan(trace_dt):
            pass
        elif dt is None:
            dt = trace_dt
        elif not math.isclose(trace_dt, dt, rel_tol=0.0, abs_tol=max(dt * 1e-6, 1e-12)):
            raise ValueError(
                f"{file}: trace {local_index + 1} is sampled every {trace_dt:g} s, an "
                f"earlier one every {dt:g} s; a record needs one sample interval.")
        if n_samples is None:
            n_samples = int(samples_per_trace)
        elif int(samples_per_trace) != n_samples:
            raise ValueError(
                f"{file}: trace {local_index + 1} has {samples_per_trace} samples, an "
                f"earlier one {n_samples}; a record needs one trace length.")
        trace_delay = _float_tag(tags, "DELAY", math.nan)
        if math.isnan(trace_delay):
            pass
        elif delay is None:
            delay = trace_delay
        elif not math.isclose(trace_delay, delay, rel_tol=0.0,
                              abs_tol=max((dt or 0.001) * 1e-3, 1e-12)):
            raise ValueError(
                f"{file}: trace {local_index + 1} starts {trace_delay:g} s after the "
                f"shot, an earlier one {delay:g} s; a record needs one start time.")
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
        sample_type = np.dtype(_SEG2_SAMPLE_TYPES[format_code])
        expected_bytes = int(samples_per_trace) * sample_type.itemsize
        if len(sample_raw) < expected_bytes:
            raise ValueError(f"Truncated Geometrics DAT sample payload in {file}.")
        sample_raw = sample_raw[:expected_bytes]
        trace = np.frombuffer(sample_raw, dtype=sample_type).astype(np.float32)
        if sample_type.kind == "i":
            # Integer samples are counts; the descaling factor makes them units.
            trace *= _float_tag(tags, "DESCALING_FACTOR", 1.0)
        if trace.size != samples_per_trace:
            raise ValueError(f"Geometrics DAT trace has {trace.size} samples; expected {samples_per_trace}.")
        traces.append(trace)

    if dt is None:
        dt = 0.001
    if dt <= 0:
        raise ValueError(f"Invalid Geometrics DAT sample interval in {file}.")
    return traces, headers, float(dt), int(n_samples or 0)


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


def _fallback_pick_geometry(header: SeismicTraceHeader,
                            has_geometry: bool = False) -> Tuple[float, float, float, float]:
    """A trace's source and receiver positions, from its header or, failing that, its ids.

    Both x at zero means no coordinates were written - unless other traces of
    the file have them (``has_geometry``): then it is a shot standing over
    the geophone at the origin, and replacing it with the ids would move that
    trace to another place entirely.
    """
    source_id = header.energy_source_point or header.field_record or 1
    receiver_id = header.trace_number or 1

    source_x = header.source_x
    receiver_x = header.receiver_x
    if not np.isfinite(source_x):
        source_x = float(source_id)
    if not np.isfinite(receiver_x):
        receiver_x = float(receiver_id)

    if source_x == 0.0 and receiver_x == 0.0 and not has_geometry:
        source_x = float(source_id)
        receiver_x = float(receiver_id)
    elif receiver_x == source_x and header.offset:
        receiver_x = source_x + float(header.offset)

    return float(source_x), float(header.source_z), float(receiver_x), float(header.receiver_z)


def _aic_minimum(segment: np.ndarray) -> int:
    """Where ``segment`` splits best into a quiet part and a loud one (Maeda, 1985).

    ``AIC(k) = k log var(x[:k]) + (n - k - 1) log var(x[k:])``, evaluated for
    every ``k`` from running sums; its minimum is the onset.
    """
    x = np.asarray(segment, dtype=float)
    n = x.size
    if n < 6:
        return n // 2
    k = np.arange(2, n - 2)
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    left_mean, left_sq = c1[k - 1] / k, c2[k - 1] / k
    right_n = n - k
    right_mean = (c1[-1] - c1[k - 1]) / right_n
    right_sq = (c2[-1] - c2[k - 1]) / right_n
    floor = 1e-12 * max(float(np.mean(x * x)), 1e-300)
    left_var = np.maximum(left_sq - left_mean ** 2, floor)
    right_var = np.maximum(right_sq - right_mean ** 2, floor)
    aic = k * np.log(left_var) + (right_n - 1) * np.log(right_var)
    return int(k[np.argmin(aic)])


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
    than a final scientific picking algorithm; :func:`repick_against_curve`
    and :func:`monotonic_pick_check` correct and screen what it picks.
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
    # Whether the headers carry coordinates at all, decided over every trace.
    has_geometry = any((np.isfinite(h.source_x) and h.source_x != 0.0)
                       or (np.isfinite(h.receiver_x) and h.receiver_x != 0.0)
                       for h in trace_headers)

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

        source_x, source_z, receiver_x, receiver_z = _fallback_pick_geometry(header, has_geometry)
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

    order = sorted(range(len(sensors)), key=lambda i: (sensors[i][0], sensors[i][1]))
    remap = {old_index + 1: new_index + 1 for new_index, old_index in enumerate(order)}
    sorted_sensors = [sensors[i] for i in order]
    rows = [(remap[sid], remap[gid], time_s) for sid, gid, time_s in rows]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(sorted_sensors)}\n")
        f.write("# x y\n")
        for x, z in sorted_sensors:
            f.write(f"{x:g}\t{z:g}\n")
        f.write(f"{len(rows)}\n")
        f.write("# s g t\n")
        for sid, gid, time_s in rows:
            f.write(f"{sid}\t{gid}\t{time_s:.9g}\n")

    return str(path)


def _median_isotonic(values: np.ndarray) -> np.ndarray:
    """The non-decreasing sequence closest to ``values`` in absolute deviation.

    Pool-adjacent-violators with block medians, so one wild value moves the
    fit much less than a least-squares fit would let it.
    """
    blocks: List[List[float]] = []
    for value in values:
        blocks.append([float(value)])
        while len(blocks) > 1 and np.median(blocks[-2]) > np.median(blocks[-1]):
            last = blocks.pop()
            blocks[-1].extend(last)
    return np.concatenate([np.full(len(block), np.median(block)) for block in blocks])


def _trimmed_curve(times: np.ndarray, tolerance_s: float, relative: float):
    """Positions kept, and the non-decreasing fit at them, of times sorted by offset.

    While a time lies farther from the closest non-decreasing curve than
    ``tolerance_s`` or ``relative`` of the curve, the one farthest out is set
    aside and the curve fitted again - so a wild pick is not let drag a good
    neighbour out with it. Returns ``(kept, fit)``; fewer than four times are
    all kept as they are.
    """
    kept = list(range(len(times)))
    fit = np.asarray(times, dtype=float)
    while len(kept) >= 4:
        fit = _median_isotonic(times[kept])
        excess = np.abs(times[kept] - fit) / np.maximum(tolerance_s, relative * fit)
        worst = int(np.argmax(excess))
        if excess[worst] <= 1.0:
            break
        kept.pop(worst)
    else:
        fit = np.asarray(times, dtype=float)[kept]
    return kept, fit


def _shot_sides(pick_list: Sequence[FirstBreakPick], failed: bool = False):
    """Each side of each shot: the pick indices, sorted by offset, and the offsets.

    Only the picks with a positive time, unless ``failed``: then also those the
    picker left at time zero or without a time.
    """
    by_shot: Dict[float, List[int]] = {}
    for index, pick in enumerate(pick_list):
        if failed or (np.isfinite(pick.time_s) and pick.time_s > 0):
            by_shot.setdefault(round(float(pick.source_x), 6), []).append(index)
    for shot, members in by_shot.items():
        for side in (-1.0, 1.0):
            mine = [i for i in members if np.sign(pick_list[i].receiver_x - shot) in (side, 0.0)]
            mine.sort(key=lambda i: abs(pick_list[i].receiver_x - shot))
            yield mine, np.array([abs(pick_list[i].receiver_x - shot) for i in mine])


def _aic_repick(pick: FirstBreakPick, trace: np.ndarray, dt: float, expected: float,
                window_fraction: float, window_min_s: float) -> Optional[FirstBreakPick]:
    """``pick`` picked again on ``trace`` by AIC around ``expected``; None if the window will not do."""
    from dataclasses import replace

    half = max(window_min_s, window_fraction * expected)
    a = max(0, int((expected - half) / dt))
    b = min(trace.shape[0], int((expected + half) / dt) + 1)
    if b - a < 8:
        return None
    x = trace - np.median(trace)
    if not np.all(np.isfinite(x[a:b])):
        return None
    onset = a + _aic_minimum(x[a:b])
    return replace(pick, time_s=float(onset * dt), amplitude=float(trace[onset]))


def repick_against_curve(
    picks: Iterable[FirstBreakPick | Dict[str, Any]],
    traces: np.ndarray,
    dt: float,
    window_fraction: float = 0.3,
    window_min_s: float = 0.002,
    tolerance_s: float = 0.0015,
    relative: float = 0.2,
) -> Tuple[List[FirstBreakPick], List[FirstBreakPick]]:
    """Pick again, along its shot's first-arrival curve, each pick that strays from it.

    The curve is the one :func:`monotonic_pick_check` fits to each side of a
    shot. A pick off it is picked again on its trace (``traces[:, trace_index]``,
    samples by traces, best without gain) by the AIC picker (Maeda, 1985) in a
    window of ``window_fraction`` of the curve's time, and at least
    ``window_min_s``, either side of where the curve puts it - beyond the
    curve's last pick, where its last pick puts it. A threshold picker that
    took the noise after time zero for the arrival - on one line one pick in
    seven, on traces whose late surface waves dwarf the first arrival - is
    corrected rather than its trace lost; the picks on the curve are left as
    they are, which held up better than picking every trace again.

    So is a pick left at time zero, or without a time - a picker stopped at
    the first sample of a trace that starts on a DC level - but only once the
    curve has a timed pick nearer the shot than it: right beside the shot the
    curve says nothing of where the arrival is, and such a pick is left
    without a time.

    Returns ``(picks, repicked)``: all the picks, corrected, and the corrected
    ones as they now stand.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> dt, n = 0.0005, 400
    >>> arrival = [0.010 + 0.004 * r for r in range(8)]
    >>> traces = rng.normal(0, 0.01, (n, 8))
    >>> for r, t in enumerate(arrival):
    ...     k = int(t / dt); traces[k:, r] += np.sin(np.arange(n - k) * 0.6)
    >>> make = lambda r, t: {"source_id": 1, "receiver_id": r + 1, "time_s": t,
    ...     "source_x": 0.0, "source_z": 0.0, "receiver_x": float(r + 1), "receiver_z": 0.0,
    ...     "field_record": 1, "trace_number": r + 1, "trace_index": r, "amplitude": 1.0}
    >>> picks = [make(r, {5: 0.0005, 2: 0.0}.get(r, t)) for r, t in enumerate(arrival)]
    >>> fixed, repicked = repick_against_curve(picks, traces, dt)
    >>> [p.receiver_x for p in repicked]
    [3.0, 6.0]
    >>> [abs(fixed[r].time_s - arrival[r]) <= 1.5 * dt for r in (2, 5)]
    [True, True]
    """
    pick_list = [_pick_from_any(pick) for pick in picks]
    arr = np.asarray(traces, dtype=float)
    repicked: List[int] = []
    for mine, offsets in _shot_sides(pick_list, failed=True):
        timed = [k for k, i in enumerate(mine)
                 if np.isfinite(pick_list[i].time_s) and pick_list[i].time_s > 0]
        if len(timed) < 4:
            continue
        times = np.array([pick_list[mine[k]].time_s for k in timed])
        kept_timed, fit = _trimmed_curve(times, tolerance_s, relative)
        kept = [timed[k] for k in kept_timed]
        for position, index in enumerate(mine):
            if position in kept:
                continue
            if position not in timed and offsets[position] < offsets[kept[0]]:
                continue
            column = pick_list[index].trace_index
            if not 0 <= column < arr.shape[1]:
                continue
            expected = float(np.interp(offsets[position], offsets[kept], fit))
            again = _aic_repick(pick_list[index], arr[:, column], dt, expected,
                                window_fraction, window_min_s)
            if again is not None:
                pick_list[index] = again
                repicked.append(index)
    return pick_list, [pick_list[i] for i in repicked]


def _receiver_sides(pick_list: Sequence[FirstBreakPick]):
    """Each side of each receiver: the indices of its timed picks, sorted by offset, and the offsets.

    A common-receiver gather - one geophone's picks from every shot - which
    by reciprocity is a shot gather with the source at the geophone. A shot
    standing on the geophone belongs to neither side.
    """
    by_receiver: Dict[float, List[int]] = {}
    for index, pick in enumerate(pick_list):
        if np.isfinite(pick.time_s) and pick.time_s > 0:
            by_receiver.setdefault(round(float(pick.receiver_x), 6), []).append(index)
    for receiver, members in by_receiver.items():
        for side in (-1.0, 1.0):
            mine = [i for i in members if np.sign(pick_list[i].source_x - receiver) == side]
            mine.sort(key=lambda i: abs(pick_list[i].source_x - receiver))
            yield mine, np.array([abs(pick_list[i].source_x - receiver) for i in mine])


def _shot_residuals(pick_list: Sequence[FirstBreakPick], tolerance_s: float,
                    relative: float) -> Dict[int, float]:
    """How far each timed pick lies from the line through its two nearest shot neighbours.

    In units of the tolerance there, ``max(tolerance_s, relative * time)``;
    the two nearest picks of the same side of the same shot, by offset, so a
    pick at the end of a side is measured against the line on from the two
    before it.
    """
    residuals: Dict[int, float] = {}
    for mine, offsets in _shot_sides(pick_list):
        times = np.array([pick_list[i].time_s for i in mine])
        for k, index in enumerate(mine):
            others = sorted((j for j in range(len(mine)) if j != k),
                            key=lambda j: abs(offsets[j] - offsets[k]))[:2]
            if len(others) < 2:
                continue
            (o1, t1), (o2, t2) = sorted((offsets[j], times[j]) for j in others)
            guess = t1 + (t2 - t1) * (offsets[k] - o1) / (o2 - o1) if o2 > o1 else t1
            residual = abs(times[k] - guess) / max(tolerance_s, relative * guess)
            residuals[index] = max(residuals.get(index, 0.0), residual)
    return residuals


def _neighbour_strays(pick_list: Sequence[FirstBreakPick], tolerance_s: float, relative: float,
                      floor: float) -> List[Tuple[int, float]]:
    """The picks that break their receiver's curve and stray in their own shot's, with where the curve puts them."""
    residuals = _shot_residuals(pick_list, tolerance_s, relative)
    strays: List[Tuple[int, float]] = []
    for mine, offsets in _receiver_sides(pick_list):
        if len(mine) < 4:
            continue
        times = np.array([pick_list[i].time_s for i in mine])
        kept = list(range(len(mine)))
        while len(kept) >= 4:
            fit = _median_isotonic(times[kept])
            excess = np.abs(times[kept] - fit) / np.maximum(tolerance_s, relative * fit)
            off = [k for k, e in zip(kept, excess) if e > 1.0]
            if not off:
                break
            # The receiver's curve says two picks disagree, not which one is
            # wrong: the one its own shot's gather also leaves out of line.
            worst = max(off, key=lambda k: residuals.get(mine[k], 0.0))
            if residuals.get(mine[worst], 0.0) < floor:
                break
            kept.remove(worst)
            strays.append((mine[worst], float(np.interp(offsets[worst], offsets[kept],
                                                        _median_isotonic(times[kept])))))
    return strays


def neighbour_shot_check(
    picks: Iterable[FirstBreakPick | Dict[str, Any]],
    traces: Optional[np.ndarray | Callable[[FirstBreakPick], Optional[np.ndarray]]] = None,
    dt: Optional[float] = None,
    tolerance_s: float = 0.0015,
    relative: float = 0.1,
    window_fraction: float = 0.2,
    window_min_s: float = 0.002,
    floor: float = 1.0,
) -> Tuple[List[FirstBreakPick], List[FirstBreakPick], List[FirstBreakPick]]:
    """Check each pick against the neighbouring shots' picks at the same geophone.

    By reciprocity, one geophone's picks from every shot form a shot gather
    with the source at the geophone, so along each side of it they too can
    only come later farther away. The neighbouring shots, a few metres apart,
    say where a pick belongs more closely than its own shot's curve can: a
    shot off the end of the spread has no reciprocals, and far from any shot
    the curve's own tolerance leaves room for a pick a quarter early. Each
    side of each geophone is fitted as :func:`monotonic_pick_check` fits a
    shot, to within ``tolerance_s`` or ``relative`` of the curve - half the
    shot check's relative tolerance, the neighbours being that close. That
    curve says two picks disagree, not which one is wrong: of the picks off
    it, the one that also lies farthest from the line through its own shot
    neighbours (:func:`_shot_residuals`) is taken, and only if that is at
    least ``floor`` tolerances; a conflict neither shot gather explains is
    left alone.

    With ``traces`` - samples by traces indexed by ``trace_index``, or a
    function giving a pick's trace - such a pick is picked again by the AIC
    picker (Maeda, 1985) in a window of ``window_fraction`` of the curve's
    time, and at least ``window_min_s``, either side of where the neighbours
    put it, and what still strays is left out; without, it is left out.

    Run it after :func:`reciprocal_shot_check`: a shot with a timing error of
    its own breaks every geophone's curve, and is to be left out whole rather
    than corrected pick by pick.

    On one 24-geophone line it picked again 10 of 356 picks - among them two
    of an off-end shot 7-9 ms before the next shot's at the same geophones -
    and left out 3; chi-squared fell from 9.2 to 6.6 and the reciprocal times,
    which it does not look at, came to agree to 1.75 ms on average instead of
    2.4.

    Returns ``(kept, rejected, repicked)``: the picks kept, those left out, and
    those picked again as they now stand.

    Examples
    --------
    >>> make = lambda s, r: {"source_id": 1, "receiver_id": 1, "source_x": float(s),
    ...     "source_z": 0.0, "receiver_x": float(r), "receiver_z": 0.0, "field_record": 1,
    ...     "trace_number": 1, "trace_index": 0, "amplitude": 1.0,
    ...     "time_s": min(abs(s - r) / 300, 0.02 + abs(s - r) / 2000)}
    >>> picks = [make(s, r) for s in range(-2, 13, 2) for r in range(12) if s != r]
    >>> early = next(i for i, p in enumerate(picks) if (p["source_x"], p["receiver_x"]) == (-2, 9))
    >>> picks[early]["time_s"] -= 0.006
    >>> kept, rejected, repicked = neighbour_shot_check(picks)
    >>> [(p.source_x, p.receiver_x) for p in rejected], len(kept)
    ([(-2.0, 9.0)], 89)
    """
    if traces is not None and not dt:
        raise ValueError("dt is required to pick again on the traces.")
    pick_list = [_pick_from_any(pick) for pick in picks]
    if traces is None or callable(traces):
        trace_of = traces
    else:
        arr = np.asarray(traces, dtype=float)

        def trace_of(pick: FirstBreakPick) -> Optional[np.ndarray]:
            return arr[:, pick.trace_index] if 0 <= pick.trace_index < arr.shape[1] else None
    repicked: List[int] = []
    if trace_of is not None:
        for index, expected in _neighbour_strays(pick_list, tolerance_s, relative, floor):
            trace = trace_of(pick_list[index])
            if trace is None:
                continue
            again = _aic_repick(pick_list[index], np.asarray(trace, dtype=float), float(dt),
                                expected, window_fraction, window_min_s)
            if again is not None:
                pick_list[index] = again
                repicked.append(index)
    out = {index for index, _ in _neighbour_strays(pick_list, tolerance_s, relative, floor)}
    return ([p for i, p in enumerate(pick_list) if i not in out],
            [p for i, p in enumerate(pick_list) if i in out],
            [pick_list[i] for i in repicked])


def monotonic_pick_check(
    picks: Iterable[FirstBreakPick | Dict[str, Any]],
    tolerance_s: float = 0.0015,
    relative: float = 0.2,
) -> Tuple[List[FirstBreakPick], List[FirstBreakPick]]:
    """Drop the picks that break a shot's first-arrival curve.

    Moving away from a shot, the first arrival can only come later: each side
    of each shot is fitted with the closest non-decreasing curve (in absolute
    deviation), and while a pick lies farther from it than ``tolerance_s`` or
    ``relative`` of its time, the one farthest out is dropped and the curve
    fitted again - so a wild pick is not let drag a good neighbour out with it. It
    catches an automatic picker that jumped to noise or to another phase - on
    one 24-geophone line, six traces of one shot picked at 10 ms where the
    curve stood at 23 ms - while leaving the curve's own shape, however far
    from straight, alone. On that line dropping eight such picks of 354 took
    the inversion from chi-squared 54 to 5. A side with fewer than four picks
    is kept as it is.

    Returns ``(kept, rejected)``.

    Examples
    --------
    >>> make = lambda r, t: {"source_id": 1, "receiver_id": int(r), "time_s": t,
    ...     "source_x": 0.0, "source_z": 0.0, "receiver_x": float(r), "receiver_z": 0.0,
    ...     "field_record": 1, "trace_number": 1, "trace_index": 0, "amplitude": 1.0}
    >>> times = [0.004, 0.008, 0.011, 0.014, 0.003, 0.019, 0.021, 0.023]
    >>> kept, rejected = monotonic_pick_check([make(r + 1, t) for r, t in enumerate(times)])
    >>> [p.receiver_x for p in rejected], len(kept)
    ([5.0], 7)
    """
    pick_list = [_pick_from_any(pick) for pick in picks]
    rejected_ids = set()
    for mine, _ in _shot_sides(pick_list):
        if len(mine) < 4:
            continue
        times = np.array([pick_list[i].time_s for i in mine])
        kept, _fit = _trimmed_curve(times, tolerance_s, relative)
        rejected_ids.update(index for position, index in enumerate(mine)
                            if position not in kept)
    kept = [p for i, p in enumerate(pick_list) if i not in rejected_ids]
    rejected = [p for i, p in enumerate(pick_list) if i in rejected_ids]
    return kept, rejected


def reciprocal_shot_check(
    picks: Iterable[FirstBreakPick | Dict[str, Any]],
    factor: float = 3.0,
    floor_s: float = 0.002,
    tolerance_m: float = 1e-3,
) -> Tuple[List[FirstBreakPick], List[Dict[str, Any]]]:
    """Drop the shots whose travel times disagree with their reciprocals.

    A travel time is the same whichever end of the path the source is at, so
    where a shot stands on a geophone and another shot's receiver stands on it
    in turn, the two times should agree to within the picking error. A shot
    whose times differ from their reciprocals by the same amount, pair after
    pair, has a timing error of its own - a trigger that fired early or late -
    and every pick it contributes is wrong by that amount. For each shot, the
    median of its reciprocal differences is compared with the survey's
    typical difference: beyond ``factor`` times that, and at least ``floor_s``,
    the shot is dropped. Shots with no reciprocal pair (off the end of the
    spread) cannot be checked and are kept.

    On one 24-geophone line two shots in seventeen came out 14 and 7.7 ms
    early against every reciprocal, while the rest agreed to about 1 ms.

    Returns ``(kept, dropped)``: the picks of the shots kept, and for each
    dropped shot its ``source_x``, ``pairs`` and ``median_difference_ms``.

    Examples
    --------
    >>> make = lambda s, r, t: {"source_id": int(s), "receiver_id": int(r), "time_s": t,
    ...     "source_x": s, "source_z": 0.0, "receiver_x": r, "receiver_z": 0.0,
    ...     "field_record": 1, "trace_number": 1, "trace_index": 0, "amplitude": 1.0}
    >>> xs = range(6)
    >>> picks = [make(s, r, 0.02 + 0.002 * abs(s - r) + 0.0004 * ((s * 7 + r * 3) % 3)
    ...                     - (0.01 if s == 3 else 0.0))
    ...          for s in xs for r in xs if s != r]
    >>> kept, dropped = reciprocal_shot_check(picks)
    >>> [d["source_x"] for d in dropped], len(kept)
    ([3.0], 25)
    """
    pick_list = [_pick_from_any(pick) for pick in picks]

    def key(x: float) -> float:
        return round(float(x) / tolerance_m) * tolerance_m

    times = {(key(p.source_x), key(p.receiver_x)): p.time_s
             for p in pick_list if np.isfinite(p.time_s) and p.time_s > 0}
    shots = sorted({key(p.source_x) for p in pick_list})
    medians: Dict[float, Tuple[int, float, float]] = {}
    for a in shots:
        diffs = [times[(a, b)] - times[(b, a)] for b in shots
                 if b != a and (a, b) in times and (b, a) in times]
        if diffs:
            medians[a] = (len(diffs), float(np.median(diffs)), float(np.median(np.abs(diffs))))
    if not medians:
        return pick_list, []
    typical = float(np.median([spread for _, _, spread in medians.values()]))
    limit = max(factor * typical, floor_s)
    bad = {shot: (pairs, median) for shot, (pairs, median, _) in medians.items()
           if abs(median) > limit}
    kept = [p for p in pick_list if key(p.source_x) not in bad]
    dropped = [{"source_x": float(shot), "pairs": pairs,
                "median_difference_ms": round(1e3 * median, 2)}
               for shot, (pairs, median) in sorted(bad.items())]
    return kept, dropped


def pick_and_correct(
    data: SeismicDataset | SeismicShotGather | np.ndarray,
    dt: Optional[float] = None,
    headers: Optional[Sequence[SeismicTraceHeader]] = None,
    *,
    agc_window: float = 0.05,
    bandpass: Optional[Sequence[float]] = None,
    threshold: float = 0.2,
    noise_multiplier: float = 5.0,
    min_time: float = 0.0,
    max_time: Optional[float] = 0.15,
    polarity: float = 1.0,
    place: Optional[Callable[[List[FirstBreakPick]], List[FirstBreakPick]]] = None,
    repick: bool = True,
) -> Tuple[List[FirstBreakPick], List[FirstBreakPick]]:
    """First-arrival picks, with the ones that stray from their shot's curve picked again.

    The automatic picking the seismic workflow and the studio's Seismic page
    share, in three steps:

    1. :func:`pick_first_breaks` on the traces with automatic gain control
       (:func:`apply_agc`, a window of ``agc_window`` seconds; 0 for none)
       and, given ``bandpass`` as four corner frequencies in Hz, a band-pass
       filter;
    2. ``place``, given, puts the picks on the survey's geometry - a
       coordinate file, or the shot and geophone positions set by hand -
       since the curves of step 3 are functions of offset; without it the
       trace headers' positions are used;
    3. :func:`repick_against_curve` on the traces as recorded, without gain
       (``repick=False`` skips it).

    ``data`` is what :func:`pick_first_breaks` takes, and each pick's
    ``trace_index`` is what it gives: the trace's column in ``data``, or for
    a :class:`SeismicShotGather` its index in the dataset.

    Returns ``(picks, repicked)``: all the picks, and the ones picked again as
    they now stand. :func:`screen_picks` then leaves out what still strays.
    """
    from dataclasses import replace

    traces, time, trace_headers, trace_indices = _as_dataset_and_headers(data, headers=headers, dt=dt)
    if traces.size == 0:
        return [], []
    step = float(dt) if dt else (float(time[1] - time[0]) if time.size > 1 else 0.0)
    if step <= 0:
        raise ValueError("Picking needs the sample interval.")
    gained = np.asarray(traces, dtype=float)
    if agc_window and float(agc_window) > 0:
        gained = apply_agc(gained, dt=step, window=float(agc_window))
    if bandpass:
        f1, f2, f3, f4 = (float(f) for f in bandpass)
        gained = bandpass_filter(gained, dt=step, f1=f1, f2=f2, f3=f3, f4=f4)
    # Picked on the columns as passed, so trace_index reaches the traces below.
    picks = pick_first_breaks(gained, dt=step, headers=trace_headers, threshold=threshold,
                              noise_multiplier=noise_multiplier, min_time=min_time,
                              max_time=max_time, polarity=polarity)
    if place is not None:
        picks = list(place(picks))
    repicked: List[FirstBreakPick] = []
    if repick:
        picks, repicked = repick_against_curve(picks, traces, step)
    if not np.array_equal(trace_indices, np.arange(len(trace_headers))):
        def index(pick: FirstBreakPick) -> FirstBreakPick:
            return replace(pick, trace_index=int(trace_indices[pick.trace_index]))
        picks, repicked = [index(p) for p in picks], [index(p) for p in repicked]
    return picks, repicked


class PickScreen(NamedTuple):
    """What :func:`screen_picks` kept, corrected and left out."""

    kept: List[FirstBreakPick]
    #: single picks that break their shot's first-arrival curve
    rejected: List[FirstBreakPick]
    #: shots with a timing error of their own, as :func:`reciprocal_shot_check` gives them
    dropped: List[Dict[str, Any]]
    #: picks that break their geophone's curve across the neighbouring shots, left out
    neighbour_rejected: List[FirstBreakPick]
    #: and picked again, as they now stand (some may then be left out)
    neighbour_repicked: List[FirstBreakPick]


def screen_picks(
    picks: Iterable[FirstBreakPick | Dict[str, Any]],
    monotonic_check: bool = True,
    reciprocity_check: bool = True,
    neighbour_check: bool = True,
    traces: Optional[np.ndarray | Callable[[FirstBreakPick], Optional[np.ndarray]]] = None,
    dt: Optional[float] = None,
) -> PickScreen:
    """Leave out the picks that break their shot's curve, the shots with a timing error, then check the neighbours.

    :func:`monotonic_pick_check`; :func:`reciprocal_shot_check` on what it
    keeps, so the shot check sees clean curves; then
    :func:`neighbour_shot_check`, after the shot check so that a shot with a
    timing error is left out whole rather than corrected pick by pick - it
    picks again on ``traces`` (as it takes them, with ``dt``) when they are
    given. The last two need every shot of the line: a shot whose reciprocals
    were not picked cannot be checked, and is kept.
    """
    kept = [_pick_from_any(pick) for pick in picks]
    rejected: List[FirstBreakPick] = []
    dropped: List[Dict[str, Any]] = []
    neighbour_rejected: List[FirstBreakPick] = []
    neighbour_repicked: List[FirstBreakPick] = []
    if monotonic_check:
        kept, rejected = monotonic_pick_check(kept)
    if reciprocity_check:
        kept, dropped = reciprocal_shot_check(kept)
    if neighbour_check:
        kept, neighbour_rejected, neighbour_repicked = neighbour_shot_check(kept, traces, dt)
    return PickScreen(kept, rejected, dropped, neighbour_rejected, neighbour_repicked)


def export_traveltime_container(data: Any, filename: str) -> str:
    """Persist a loaded PyGIMLi travel-time container in portable BERT format."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    sensors = np.asarray(data.sensors(), dtype=float)
    sources = np.asarray(data["s"], dtype=int).ravel()
    receivers = np.asarray(data["g"], dtype=int).ravel()
    times = np.asarray(data["t"], dtype=float).ravel()
    if not (sources.size == receivers.size == times.size):
        raise ValueError("Travel-time container fields s, g, and t have different lengths.")
    if sensors.ndim != 2 or sensors.shape[0] == 0:
        raise ValueError("Travel-time container has no sensor positions.")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{sensors.shape[0]}\n")
        stream.write("# x y\n")
        for sensor in sensors:
            x = float(sensor[0])
            z = float(sensor[1]) if sensor.size > 1 else 0.0
            stream.write(f"{x:g}\t{z:g}\n")
        stream.write(f"{times.size}\n")
        stream.write("# s g t\n")
        for source, receiver, time_s in zip(sources, receivers, times):
            stream.write(f"{int(source) + 1}\t{int(receiver) + 1}\t{float(time_s):.9g}\n")
    return str(path)


__all__ = [
    "SegyMetadata",
    "SeismicTraceHeader",
    "SeismicShotGather",
    "SeismicDataset",
    "FirstBreakPick",
    "read_segy",
    "record_sample_interval",
    "apply_record_interval",
    "set_sample_interval",
    "apply_agc",
    "normalize_traces",
    "tukey_taper",
    "bandpass_filter",
    "pick_first_breaks",
    "pick_and_correct",
    "screen_picks",
    "PickScreen",
    "neighbour_shot_check",
    "repick_against_curve",
    "export_first_breaks",
    "export_traveltime_container",
    "first_breaks_to_traveltime",
    "monotonic_pick_check",
    "reciprocal_shot_check",
]
