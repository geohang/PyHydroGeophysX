"""
Hello-world agent example with no LLM call.

Loads a bundled ERT file, runs a short deterministic pyGIMLi inversion, and
writes a tiny report. This is intentionally small enough for a first checkout.
"""

from pathlib import Path
import sys
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygimli as pg
from pygimli.physics import ert

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyHydroGeophysX.agents import AgentResult


def run_hello_agent(output_dir: Optional[Union[str, Path]] = None) -> AgentResult:
    """Run a no-LLM ERT inversion on bundled data.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory where the report and figure are written.

    Returns
    -------
    AgentResult
        Dict-like summary with output paths and basic inversion metrics.
    """
    output_path = Path(output_dir) if output_dir is not None else REPO_ROOT / "results" / "hello_agent"
    output_path.mkdir(parents=True, exist_ok=True)
    data_file = REPO_ROOT / "examples" / "data" / "ERT" / "Bert" / "fielddataline2.dat"

    data = ert.load(str(data_file))
    data["k"] = ert.createGeometricFactors(data)
    data["err"] = pg.Vector(data.size(), 0.05)

    manager = ert.ERTManager(data, verbose=False)
    model = manager.invert(lam=20, maxIter=3, verbose=False)
    chi2 = float(manager.inv.chi2())

    figure_path = output_path / "hello_agent_ert.png"
    _, ax = plt.subplots(figsize=(8, 3.5))
    manager.showResult(ax=ax, cMap="Spectral_r")
    ax.set_title("Hello Agent ERT Inversion")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    report_path = output_path / "hello_agent_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Hello Agent Report",
                "",
                f"- Data file: `{data_file}`",
                f"- Sensors: {data.sensorCount()}",
                f"- Measurements: {data.size()}",
                f"- Model cells: {len(model)}",
                f"- Final chi2: {chi2:.3f}",
                f"- Figure: `{figure_path}`",
            ]
        ),
        encoding="utf-8",
    )

    return AgentResult(
        status="success",
        summary="Hello agent ERT inversion completed without an API key.",
        data={
            "data_file": str(data_file),
            "figure": str(figure_path),
            "report": str(report_path),
            "model_cells": int(len(model)),
            "chi2": chi2,
        },
        next_suggested_action="Open the report and compare this deterministic path with the full multi-agent workflow.",
    )


if __name__ == "__main__":
    result = run_hello_agent()
    print(result.summary)
    print(f"Report: {result['report']}")
    print(f"Figure: {result['figure']}")
