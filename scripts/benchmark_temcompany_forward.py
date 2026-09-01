"""Measure the studio TDEM forward against a TEMcompany project's own responses.

A TEMcompany project stores, for every station its inversion touched, the model
it recovered and the forward response it computed for that model. Running our
operator on the same model, at the same gate windows and the same per-station
geometry, turns "close to the stored response" into a number.

The comparison is per gate, so it also says where any disagreement sits. The
early gates of the low moment are where the turn-off ramp and the receiver
electronics dominate; the late gates of the high moment are where neither
contributes and only the layered response is left.

Usage::

    python scripts/benchmark_temcompany_forward.py PROJECT_DIR [-n 40]
    python scripts/benchmark_temcompany_forward.py PROJECT_DIR --ablate

``--ablate`` reports the same statistic with one operator switched off at a
time, which is what attributes a percentage to a cause.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PyHydroGeophysX.data_processing.temcompany_reference import (
    load_reference_models,
)
from PyHydroGeophysX.forward.em1d import tdem_forward as _unused  # noqa: F401
from PyHydroGeophysX.forward.tdem_forward import (
    TDEMForwardModeling,
    TDEMSurveyConfig,
)

#: Operator settings each ablation turns off, against the aligned default.
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "aligned": {},
    # An empty, explicitly supplied filter description keeps the native-order
    # waveform/repetition/gate chain active while bypassing only its filters.
    "no-filter": {"analog_lowpass": {}},
    "no-repetition": {"waveform_repetitions": 0},
    "gate-centre": {"gate_window": "centre"},
    "gate-square-window": {"gate_window": "square"},
    "coarse-filter-grid": {"analog_points_per_decade": 20},
    "nominal-offset": {"nominal_offset": True},
}


def _config(station: Dict[str, Any], spec: Dict[str, Any], moment: str,
            block: Dict[str, np.ndarray], overrides: Dict[str, Any]
            ) -> TDEMSurveyConfig:
    """One survey configuration, built the way the aligned pipeline builds it."""
    geometry = station["geometry"]
    offset = float(geometry.get("rx_tx_distance", 0.0))
    if overrides.get("nominal_offset") or offset <= 0.0:
        nominal = np.asarray(spec.get("RxCoilXYZPos", [15.0, 0.0, 0.0]), float)
        offset = float(abs(nominal[0])) if nominal.size else 15.0
    rx_height = float(geometry.get("rx_coil_height", 0.0))
    tx_height = float(geometry.get("tx_coil_height", rx_height))
    lowpass = {"first_order_cutoffs_hz": tuple(
        float(v) for v in spec.get("LPFilter_1order", ()))}
    if "analog_lowpass" in overrides:
        lowpass = overrides["analog_lowpass"]
    period = spec.get(moment + "WaveformPeriod")
    return TDEMSurveyConfig(
        source_location=np.array([0.0, 0.0, tx_height]),
        source_radius=float(np.sqrt(float(spec["TxLoopArea"]) / np.pi)),
        source_turns=int(spec.get("NTurnsTxLoop", 1) or 1),
        source_moment=1.0,
        waveform_times=np.asarray(spec[moment + "WaveformTime"], float),
        waveform_currents=np.asarray(spec[moment + "WaveformAmplitude"], float),
        waveform_period=float(period) if period else None,
        waveform_repetitions=int(overrides.get("waveform_repetitions", 3)),
        gate_open=block["gate_open"],
        gate_close=block["gate_close"],
        gate_window=str(overrides.get("gate_window", "tukey")),
        gate_window_par=float(overrides.get(
            "gate_window_par", spec.get("GateShapePar1", 0.667) or 0.667)),
        analog_lowpass=lowpass,
        analog_points_per_decade=int(
            overrides.get("analog_points_per_decade", 150)),
        analog_model_points_per_decade=int(
            overrides.get("analog_model_points_per_decade", 40)),
        instrument_points_per_decade=int(
            overrides.get("instrument_points_per_decade", 10)),
        instrument_model_points_per_decade=int(
            overrides.get("instrument_model_points_per_decade", 10)),
        gate_quadrature_order=int(overrides.get("gate_quadrature_order", 8)),
        receiver_location=np.array([offset, 0.0, rx_height]),
        receiver_orientation="z",
        receiver_type="dbdt",
        times=block["times"],
        waveform_type="step_off",
    )


def ratios(project: Path, limit: Optional[int], overrides: Dict[str, Any],
           inversion_name: Optional[str] = None) -> Dict[str, List[np.ndarray]]:
    """``modelled / stored`` per gate, keyed by moment, over the sampled stations."""
    reference = load_reference_models(project, inversion_name)
    spec = reference["spec"]
    stations = reference["stations"]
    if limit is not None and limit < len(stations):
        # Spread the sample over the survey rather than taking a prefix: the
        # first stations of a line share a geometry and a resistivity range.
        stations = [stations[i] for i in
                    np.unique(np.linspace(0, len(stations) - 1, limit).astype(int))]
    out: Dict[str, List[np.ndarray]] = {}
    for station in stations:
        for moment, block in station["moments"].items():
            if not block["times"].size:
                continue
            config = _config(station, spec, moment, block, overrides)
            modeler = TDEMForwardModeling(
                thicknesses=station["thickness"], survey_config=config)
            ours = -1.0 * np.asarray(
                modeler.forward(1.0 / station["resistivity"]), float).ravel()
            stored = block["forward"]
            if ours.size != stored.size:
                continue
            out.setdefault(moment, []).append(ours / stored)
    return out


def report(label: str, per_moment: Dict[str, List[np.ndarray]]) -> None:
    """Print the pooled statistic, then the per-gate one."""
    for moment in sorted(per_moment):
        pooled = np.concatenate(per_moment[moment])
        pooled = pooled[np.isfinite(pooled)]
        if not pooled.size:
            continue
        print("  %-20s %-3s n=%-5d median=%.4f  p5=%.4f  p95=%.4f  "
              "max|err|=%.1f%%" % (
                  label, moment, pooled.size, np.median(pooled),
                  np.percentile(pooled, 5), np.percentile(pooled, 95),
                  100.0 * np.max(np.abs(pooled - 1.0))))


def per_gate(per_moment: Dict[str, List[np.ndarray]]) -> None:
    """Median ratio by gate index, which is where a shape shows up."""
    for moment in sorted(per_moment):
        width = max(item.size for item in per_moment[moment])
        columns = [[] for _ in range(width)]
        for item in per_moment[moment]:
            for index, value in enumerate(item):
                if np.isfinite(value):
                    columns[index].append(value)
        cells = " ".join(
            "%.3f" % np.median(column) if column else "   . "
            for column in columns)
        print("    %-3s by gate: %s" % (moment, cells))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("-n", "--max-stations", type=int, default=40)
    parser.add_argument("--inversion-name", default=None)
    parser.add_argument("--ablate", action="store_true")
    args = parser.parse_args(argv)

    cases = ABLATIONS if args.ablate else {"aligned": {}}
    print("Forward against stored ForwardData: %s" % args.project)
    for label, overrides in cases.items():
        started = time.time()
        result = ratios(args.project, args.max_stations, overrides,
                        args.inversion_name)
        report(label, result)
        if label == "aligned":
            per_gate(result)
        print("    (%.1fs)" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
