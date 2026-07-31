"""Compare the Workbench joint LM+HM LCI with archived TEMcompany models."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from PyHydroGeophysX.workflows import em1d


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _lateral_roughness(models: np.ndarray, lines: np.ndarray) -> float:
    changes: List[np.ndarray] = []
    for line in np.unique(lines):
        selected = models[lines == line]
        if selected.shape[0] > 1:
            changes.append(np.abs(np.diff(np.log10(selected), axis=0)).ravel())
    return float(np.nanmedian(np.concatenate(changes))) if changes else float("nan")


def compare(
    project: Path,
    output: Path,
    *,
    max_iterations: int = 8,
    lci_passes: int = 1,
    max_soundings: int = 500,
    warm_start: Path | None = None,
) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sounding = em1d.load_sounding(
        str(project), "TDEM", sounding=0, moment="LM+HM")
    defaults = dict(sounding.get("inversion_defaults", {}))
    inv = {
        **em1d.DEFAULT_INVERSION,
        **defaults,
        "data_scale": 1.0,
        "max_iterations": int(max_iterations),
        "lci_passes": int(lci_passes),
    }
    geom = {**sounding["system"], "tem_moment": "LM+HM"}
    initial_models = None
    if warm_start is not None:
        with np.load(warm_start) as archive:
            # Saved sections are deepest-first; the inversion uses surface-first.
            initial_models = np.asarray(
                archive["resistivity"], dtype=float)[:, ::-1].copy()
    result = em1d.invert_line(
        str(project),
        "TDEM",
        geom,
        inv,
        positions=np.asarray(sounding["positions"], dtype=float),
        heights=np.asarray(sounding["heights"], dtype=float),
        max_soundings=min(int(max_soundings), int(sounding["n_soundings"])),
        doi_blank=False,
        out_dir=output,
        initial_models=initial_models,
        log=print,
    )
    recovered = np.asarray(result["model3d"][:, 0, ::-1], dtype=float)
    average_ids = np.asarray(
        sounding["average_data_ids"], dtype=int)[:recovered.shape[0]]
    station_ids = np.asarray(
        sounding["station_ids"], dtype=str)[:recovered.shape[0]]
    lines = np.asarray(
        sounding["line_numbers"], dtype=int)[:recovered.shape[0]]

    con = sqlite3.connect(project / "project.db")
    con.row_factory = sqlite3.Row
    company_rows = {
        int(row["AverageDataID"]): row
        for row in con.execute(
            "SELECT AverageDataID, LineNumber, DataFit, Resistivity, Thickness "
            "FROM InversionModel"
        )
    }
    con.close()

    comparison_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    company_models: List[np.ndarray] = []
    workbench_models: List[np.ndarray] = []
    compared_lines: List[int] = []
    for index, average_id in enumerate(average_ids):
        row = company_rows.get(int(average_id))
        if row is None or not np.all(np.isfinite(recovered[index])):
            continue
        company = np.asarray(json.loads(row["Resistivity"]), dtype=float)
        workbench = recovered[index]
        if company.size != workbench.size:
            continue
        delta = np.log10(workbench / company)
        correlation = float(np.corrcoef(
            np.log10(company), np.log10(workbench))[0, 1])
        comparison_rows.append({
            "average_data_id": int(average_id),
            "station_id": str(station_ids[index]),
            "line": int(lines[index]),
            "company_data_fit": float(row["DataFit"]),
            "workbench_chi2": float(result["chi2_list"][index]),
            "model_log_rmse": float(np.sqrt(np.mean(delta ** 2))),
            "model_log_correlation": correlation,
            "median_resistivity_ratio": float(np.median(workbench / company)),
        })
        thickness = np.asarray(json.loads(row["Thickness"]), dtype=float)
        tops = np.concatenate([[0.0], np.cumsum(thickness)])
        for layer, (top, company_value, workbench_value) in enumerate(
            zip(tops, company, workbench), start=1
        ):
            layer_rows.append({
                "average_data_id": int(average_id),
                "station_id": str(station_ids[index]),
                "line": int(lines[index]),
                "layer": layer,
                "depth_top_m": float(top),
                "company_resistivity_ohm_m": float(company_value),
                "workbench_resistivity_ohm_m": float(workbench_value),
            })
        company_models.append(company)
        workbench_models.append(workbench)
        compared_lines.append(int(lines[index]))

    company_array = np.asarray(company_models, dtype=float)
    workbench_array = np.asarray(workbench_models, dtype=float)
    line_array = np.asarray(compared_lines, dtype=int)
    summary = {
        "project": str(project),
        "workflow": "LM+HM joint 1D models with same-line L2 lateral constraints",
        "models_compared": len(comparison_rows),
        "n_layers": int(result["n_layers"]),
        "vertical_smoothness": float(inv["smoothness"]),
        "lateral_smoothness": float(inv["lateral_smoothness"]),
        "lateral_weight_scale": float(inv.get("lateral_weight_scale", 1.0)),
        "lci_passes": int(inv["lci_passes"]),
        "max_iterations": int(inv["max_iterations"]),
        "starting_resistivity": float(inv["starting_resistivity"]),
        "model_comparison": {
            "median_log_rmse": float(np.nanmedian([
                row["model_log_rmse"] for row in comparison_rows])),
            "median_log_correlation": float(np.nanmedian([
                row["model_log_correlation"] for row in comparison_rows])),
            "median_resistivity_ratio": float(np.nanmedian([
                row["median_resistivity_ratio"] for row in comparison_rows])),
            "median_chi2": float(np.nanmedian([
                row["workbench_chi2"] for row in comparison_rows])),
        },
        "lateral_roughness": {
            "company_median_adjacent_log_difference": _lateral_roughness(
                company_array, line_array),
            "workbench_median_adjacent_log_difference": _lateral_roughness(
                workbench_array, line_array),
        },
    }
    _write_csv(output / "lci_inversion_comparison.csv", comparison_rows)
    _write_csv(output / "lci_layer_comparison.csv", layer_rows)
    (output / "lci_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    _plot(company_array, workbench_array, comparison_rows, layer_rows, output)
    return summary


def _plot(
    company: np.ndarray,
    workbench: np.ndarray,
    comparison_rows: List[Dict[str, Any]],
    layer_rows: List[Dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].scatter(company.ravel(), workbench.ravel(), s=8, alpha=0.35)
    limits = [
        float(np.nanmin([company.min(), workbench.min()])),
        float(np.nanmax([company.max(), workbench.max()])),
    ]
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[0].set(xscale="log", yscale="log", xlabel="TEMcompany resistivity",
                ylabel="Workbench resistivity", title="All layers")
    axes[0].grid(alpha=0.25)

    line_values = sorted({int(row["line"]) for row in comparison_rows})
    axes[1].boxplot([
        [row["model_log_rmse"] for row in comparison_rows if int(row["line"]) == line]
        for line in line_values
    ], tick_labels=[str(line) for line in line_values], showmeans=True)
    axes[1].set(xlabel="Survey line", ylabel="20-layer log10 RMSE",
                title="LCI model difference")
    axes[1].grid(axis="y", alpha=0.25)

    for color_index, line in enumerate(line_values):
        chosen = next(row for row in comparison_rows if int(row["line"]) == line)
        records = [
            item for item in layer_rows
            if item["average_data_id"] == chosen["average_data_id"]
        ]
        depth = np.asarray([item["depth_top_m"] for item in records])
        company_profile = np.asarray([
            item["company_resistivity_ohm_m"] for item in records])
        workbench_profile = np.asarray([
            item["workbench_resistivity_ohm_m"] for item in records])
        color = f"C{color_index}"
        axes[2].step(company_profile, depth, where="post", color=color,
                     label=f"L{line} company")
        axes[2].step(workbench_profile, depth, where="post", color=color,
                     linestyle="--", label=f"L{line} workbench")
    axes[2].set(xscale="log", xlabel="Resistivity (ohm m)",
                ylabel="Depth to layer top (m)", title="Representative profiles")
    axes[2].invert_yaxis()
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "temcompany_lci_comparison.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/temcompany_lci_comparison"))
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--lci-passes", type=int, default=1)
    parser.add_argument("--max-soundings", type=int, default=500)
    parser.add_argument(
        "--warm-start", type=Path, default=None,
        help="Existing resistivity_section.npz used to skip independent initialization.",
    )
    args = parser.parse_args()
    summary = compare(
        args.project,
        args.output,
        max_iterations=max(4, args.max_iterations),
        lci_passes=max(0, args.lci_passes),
        max_soundings=max(1, args.max_soundings),
        warm_start=args.warm_start,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
