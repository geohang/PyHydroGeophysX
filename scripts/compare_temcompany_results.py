"""Compare Studio TDEM modeling/inversion with a TEMcompany project result.

The comparison has two deliberately separate parts:

1. Forward-check every TEMcompany model against its archived HM observations,
   allowing one global unit-normalization scalar for the Studio response.
2. Independently invert a spatially representative subset of HM soundings on
   the same 20-layer grid and compare recovered resistivity with TEMcompany's
   joint laterally constrained inversion (LCI).

The first checks format/geometry/response compatibility. The second is expected
to expose differences between independent HM-only 1D inversion and LM+HM LCI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from PyHydroGeophysX.forward.em1d import _tdem_config, _tdem_geometry
from PyHydroGeophysX.workflows import em1d
from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling


def _positive_log_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    mask = (
        np.isfinite(observed) & np.isfinite(predicted)
        & (observed > 0.0) & (predicted > 0.0)
    )
    if np.count_nonzero(mask) < 3:
        return {"n": int(np.count_nonzero(mask)), "scale": np.nan,
                "log_rmse": np.nan, "correlation": np.nan}
    obs = observed[mask]
    pred = predicted[mask]
    scale = float(np.exp(np.mean(np.log(obs / pred))))
    residual = np.log10(scale * pred / obs)
    return {
        "n": int(obs.size),
        "scale": scale,
        "log_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "correlation": float(np.corrcoef(np.log10(obs), np.log10(pred))[0, 1]),
    }


def _dataset(datasets: str, moment: str) -> Dict[str, Any] | None:
    for item in json.loads(datasets):
        if str(item.get("MomentType", "")).upper() == moment:
            return item
    return None


def _representative(rows: List[Dict[str, Any]], per_line: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for line in sorted({int(row["line"]) for row in rows}):
        candidates = [row for row in rows if int(row["line"]) == line]
        indices = np.unique(np.linspace(
            0, len(candidates) - 1, min(per_line, len(candidates))
        ).astype(int))
        selected.extend(candidates[index] for index in indices)
    return selected


def compare(
    project: Path,
    output: Path,
    *,
    moment: str = "HM",
    sample_per_line: int = 3,
    max_iterations: int = 8,
) -> Dict[str, Any]:
    database = project / "project.db"
    if not database.exists():
        raise ValueError(f"TEMcompany database not found: {database}")
    output.mkdir(parents=True, exist_ok=True)

    sounding = em1d.load_sounding(str(project), "TDEM", moment=moment)
    geom = _tdem_geometry(
        sounding, {**sounding["system"], "tem_moment": moment})
    native_scale = float(sounding["system"].get("data_scale", 1.0))
    independent_scale = em1d.estimate_data_scale(
        str(project), "TDEM", geom, max_soundings=8)

    con = sqlite3.connect(database)
    con.row_factory = sqlite3.Row
    model_rows = list(con.execute(
        "SELECT m.AverageDataID, m.LineNumber, m.UTMx, m.UTMy, "
        "m.DataFit, m.Resistivity, m.Thickness, m.Datasets, "
        "s.StationId "
        "FROM InversionModel m "
        "LEFT JOIN StationStackData s ON s.AveragedDataId = m.AverageDataID "
        "ORDER BY m.LineNumber, m.AverageDataID"
    ))
    con.close()

    forward_rows: List[Dict[str, Any]] = []
    forward_logs: List[np.ndarray] = []
    eligible: List[Dict[str, Any]] = []
    for row in model_rows:
        dataset = _dataset(row["Datasets"], moment)
        if not dataset:
            continue
        observed = np.asarray(dataset.get("InputData", []), dtype=float)
        company_forward = np.asarray(dataset.get("ForwardData", []), dtype=float)
        times = np.asarray(dataset.get("Time_Centre", []), dtype=float)
        relative_std = np.asarray(dataset.get("InputSTD", []), dtype=float)
        if min(observed.size, company_forward.size, times.size) < 5:
            continue
        resistivity = np.asarray(json.loads(row["Resistivity"]), dtype=float)
        thickness = np.asarray(json.loads(row["Thickness"]), dtype=float)
        modeler = TDEMForwardModeling(
            thickness, survey_config=_tdem_config(geom, times))
        studio_forward = (
            float(geom.get("response_sign", 1.0))
            * np.asarray(modeler.forward(1.0 / resistivity), dtype=float).ravel()
        )
        studio_metrics = _positive_log_metrics(observed, studio_forward)
        company_metrics = _positive_log_metrics(observed, company_forward)
        mask = (
            np.isfinite(observed) & np.isfinite(studio_forward)
            & (observed > 0.0) & (studio_forward > 0.0)
        )
        if np.any(mask):
            forward_logs.append(
                np.log(observed[mask] / studio_forward[mask]))
        record = {
            "average_data_id": int(row["AverageDataID"]),
            "station_id": str(row["StationId"] or row["AverageDataID"]),
            "line": int(row["LineNumber"]),
            "x": float(row["UTMx"]),
            "y": float(row["UTMy"]),
            "company_data_fit": float(row["DataFit"]),
            "company_shape_log_rmse": company_metrics["log_rmse"],
            "company_shape_correlation": company_metrics["correlation"],
            "studio_shape_scale": studio_metrics["scale"],
            "studio_shape_log_rmse": studio_metrics["log_rmse"],
            "studio_shape_correlation": studio_metrics["correlation"],
            "gate_count": int(observed.size),
        }
        forward_rows.append(record)
        eligible.append({
            **record,
            "observed": observed,
            "times": times,
            "relative_std": relative_std,
            "company_resistivity": resistivity,
            "thickness": thickness,
        })

    global_scale = float(np.exp(np.mean(np.concatenate(forward_logs))))
    inversion_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    for row in _representative(eligible, sample_per_line):
        data = {
            "times": row["times"],
            "response": row["observed"],
            "relative_std": row["relative_std"],
        }
        company_resistivity = row["company_resistivity"]
        thickness = row["thickness"]
        result = em1d.tdem_invert(
            data,
            geom,
            {
                **em1d.DEFAULT_INVERSION,
                "n_layers": int(company_resistivity.size),
                "min_thickness": float(thickness[0]),
                "max_thickness": float(thickness[-1]),
                "max_iterations": int(max_iterations),
                "smoothness": 0.3,
                "data_scale": native_scale,
            },
        )
        recovered = np.asarray(result["resistivity"], dtype=float)
        log_delta = np.log10(recovered / company_resistivity)
        model_correlation = float(np.corrcoef(
            np.log10(company_resistivity), np.log10(recovered))[0, 1])
        inversion_rows.append({
            "average_data_id": row["average_data_id"],
            "station_id": row["station_id"],
            "line": row["line"],
            "gate_count": row["gate_count"],
            "studio_chi2": float(result["chi2"]),
            "model_log_rmse": float(np.sqrt(np.mean(log_delta ** 2))),
            "model_log_correlation": model_correlation,
            "median_resistivity_ratio": float(
                np.median(recovered / company_resistivity)),
        })
        tops = np.concatenate([[0.0], np.cumsum(thickness)])
        for layer, (top, company_value, studio_value) in enumerate(
            zip(tops, company_resistivity, recovered), start=1
        ):
            layer_rows.append({
                "average_data_id": row["average_data_id"],
                "station_id": row["station_id"],
                "line": row["line"],
                "layer": layer,
                "depth_top_m": float(top),
                "company_resistivity_ohm_m": float(company_value),
                "studio_resistivity_ohm_m": float(studio_value),
            })

    def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        records = list(rows)
        if not records:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv(output / "forward_comparison.csv", forward_rows)
    write_csv(output / "inversion_comparison.csv", inversion_rows)
    write_csv(output / "layer_comparison.csv", layer_rows)

    summary = {
        "project": str(project),
        "moment": moment,
        "company_models_compared": len(forward_rows),
        "studio_inversions_compared": len(inversion_rows),
        "native_format_scale_used": native_scale,
        "independent_auto_scale": independent_scale,
        "company_model_global_forward_scale": global_scale,
        "forward_shape": {
            "company_median_log_rmse": float(np.nanmedian([
                row["company_shape_log_rmse"] for row in forward_rows])),
            "studio_median_log_rmse": float(np.nanmedian([
                row["studio_shape_log_rmse"] for row in forward_rows])),
            "company_median_correlation": float(np.nanmedian([
                row["company_shape_correlation"] for row in forward_rows])),
            "studio_median_correlation": float(np.nanmedian([
                row["studio_shape_correlation"] for row in forward_rows])),
        },
        "inversion_model": {
            "median_log_rmse": float(np.nanmedian([
                row["model_log_rmse"] for row in inversion_rows])),
            "median_log_correlation": float(np.nanmedian([
                row["model_log_correlation"] for row in inversion_rows])),
            "median_resistivity_ratio": float(np.nanmedian([
                row["median_resistivity_ratio"] for row in inversion_rows])),
            "median_chi2": float(np.nanmedian([
                row["studio_chi2"] for row in inversion_rows])),
        },
        "interpretation": (
            "Forward decay shapes test format/geometry compatibility after a unit scalar. "
            "Recovered models compare independent HM-only 1D inversion against the "
            "company's LM+HM laterally constrained inversion and are not expected to match."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    _plot_comparison(forward_rows, inversion_rows, layer_rows, output)
    return summary


def _plot_comparison(
    forward_rows: List[Dict[str, Any]],
    inversion_rows: List[Dict[str, Any]],
    layer_rows: List[Dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    company_shape = np.asarray(
        [row["company_shape_log_rmse"] for row in forward_rows], dtype=float)
    studio_shape = np.asarray(
        [row["studio_shape_log_rmse"] for row in forward_rows], dtype=float)
    axes[0].boxplot(
        [
            company_shape[np.isfinite(company_shape)],
            studio_shape[np.isfinite(studio_shape)],
        ],
        tick_labels=["TEMcompany", "Studio\n(company model)"],
        showmeans=True,
    )
    axes[0].set_ylabel("Decay-shape log10 RMSE")
    axes[0].set_title("Forward response")
    axes[0].grid(axis="y", alpha=0.25)

    labels = [row["station_id"] for row in inversion_rows]
    values = [row["model_log_rmse"] for row in inversion_rows]
    axes[1].bar(np.arange(len(values)), values)
    axes[1].set_xticks(np.arange(len(values)), labels, rotation=55, ha="right")
    axes[1].set_ylabel("20-layer log10 resistivity RMSE")
    axes[1].set_title("Independent HM inversion")
    axes[1].grid(axis="y", alpha=0.25)

    chosen = []
    for line in sorted({row["line"] for row in inversion_rows}):
        match = next(row for row in inversion_rows if row["line"] == line)
        chosen.append(match)
    for index, chosen_row in enumerate(chosen):
        records = [
            row for row in layer_rows
            if row["average_data_id"] == chosen_row["average_data_id"]
        ]
        depth = np.asarray([row["depth_top_m"] for row in records])
        company = np.asarray([row["company_resistivity_ohm_m"] for row in records])
        studio = np.asarray([row["studio_resistivity_ohm_m"] for row in records])
        color = f"C{index}"
        line_label = str(int(chosen_row["line"]))
        axes[2].step(company, depth, where="post", color=color,
                     label=f"L{line_label} company")
        axes[2].step(studio, depth, where="post", color=color, linestyle="--",
                     label=f"L{line_label} studio")
    axes[2].set_xscale("log")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Resistivity (ohm m)")
    axes[2].set_ylabel("Depth to layer top (m)")
    axes[2].set_title("Representative profiles")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output / "temcompany_studio_comparison.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/temcompany_comparison"))
    parser.add_argument("--moment", choices=em1d.TEMCOMPANY_MOMENTS, default="HM")
    parser.add_argument("--sample-per-line", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()
    summary = compare(
        args.project,
        args.output,
        moment=args.moment,
        sample_per_line=max(1, args.sample_per_line),
        max_iterations=max(1, args.max_iterations),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
