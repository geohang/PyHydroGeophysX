"""Rebuild the cached results that the web app shows in Demo mode.

Demo mode in ``examples/app_geophysics_workflow.py`` reads ``ert_demo.json`` and
``joint_demo.json`` from this folder, together with the figures they list. Both
come from offline runs of the workflow engine - no API key and no LLM call - on
data shipped in ``examples/data``, so they show what a first run produces.
Run this again after a change that alters those results:

    python examples/demo_cache/make_demo_cache.py

The two runs take a few minutes together. They work in a temporary folder; only
the JSON files and downscaled copies of the figures are written here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "examples" / "data"
sys.path.insert(0, str(REPO))

#: Stored figure width in pixels; with a 256-colour palette this keeps the
#: whole cache well under 2 MB.
FIGURE_WIDTH = 1000

#: Each demo: the request as a user would type it, its inputs relative to
#: examples/data, and the figures to keep as (file in the run folder, label).
DEMOS: Dict[str, Dict[str, Any]] = {
    "ert_demo": {
        "title": "ERT inversion and water content: DAS-1 example survey",
        "request": "Invert this ERT survey and estimate the water content",
        "inputs": {"data_file": "ERT/DAS/20171105_1418.Data",
                   "electrode_file": "ERT/DAS/electrodes.dat"},
        "config": {"inversion_params": {"lambda": 20.0, "max_iterations": 12}},
        "figures": [
            ("resistivity_model.png", "Inverted resistivity (ohm m)"),
            ("water_content.png", "Water content, Monte Carlo mean"),
            ("water_content_uncertainty.png", "Water content uncertainty, Monte Carlo standard deviation"),
        ],
    },
    "joint_demo": {
        "title": "Joint ERT and seismic refraction: structure-constrained water content",
        "request": ("Invert the ERT line with the seismic refraction structure and "
                    "estimate the water content"),
        "inputs": {"data_file": "ERT/Bert/fielddataline2.dat",
                   "seismic_file": "Seismic/srtfieldline2.dat"},
        "config": {"instrument": "BERT",
                   "inversion_params": {"lambda": 20.0, "max_iterations": 12}},
        "figures": [
            ("seismic/seismic_velocity_model.png", "Seismic refraction velocity model"),
            ("resistivity_model.png", "Resistivity, constrained by the 1200 m/s velocity interface"),
            ("water_content.png", "Water content from the structure-constrained model"),
        ],
    },
}


def _range(values: Any) -> List[float]:
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    return [float(array.min()), float(array.max())] if array.size else []


def _save_figure(source: Path, target: Path) -> None:
    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((FIGURE_WIDTH, FIGURE_WIDTH))
        image.quantize(colors=256).save(target, optimize=True)


def _metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    from PyHydroGeophysX.agents._chi2 import chi2_history

    metrics: Dict[str, Any] = {}
    resistivity = _range(results.get("resistivity_model"))
    if resistivity:
        metrics["resistivity (ohm m)"] = resistivity
    water = _range(results.get("water_content_mean"))
    if water:
        metrics["water content (-)"] = water
    spread = np.asarray(results.get("water_content_std"), dtype=float)
    if spread.size and np.isfinite(spread).any():
        metrics["mean water-content uncertainty"] = f"±{float(np.nanmean(spread)):.3f}"
    values = results.get("chi2_values")
    if values is None:
        values = results.get("chi2")  # the single-survey result: one number
    history = [float(values)] if isinstance(values, (int, float)) else chi2_history(values)
    if history:
        metrics["final chi-squared"] = f"{history[-1]:.2f}"
    score = (results.get("evaluation_results") or {}).get("quality_score")
    if score is not None:
        metrics["quality score"] = f"{float(score):.1f} / 100"
    return metrics


def build(name: str, spec: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
    """Run one demo offline and write its JSON and figures into this folder."""
    from PyHydroGeophysX import __version__
    from PyHydroGeophysX.agents import BaseAgent

    out = workdir / name
    out.mkdir(parents=True, exist_ok=True)
    config: Dict[str, Any] = {key: str(DATA / rel) for key, rel in spec["inputs"].items()}
    config["ert_file"] = config["data_file"]
    config.update(spec["config"])
    config.update({"user_request": spec["request"], "convert_to_water_content": True,
                   "project_dir": str(out), "output_dir": str(out)})

    started = time.time()
    results, plan, interpretation, _files = BaseAgent.run_unified_agent_workflow(
        config, None, None, "openai", str(out))
    seconds = time.time() - started

    figures = []
    for index, (relative, label) in enumerate(spec["figures"], 1):
        source = out / relative
        if not source.exists():
            raise FileNotFoundError(f"{name}: the run wrote no {relative}")
        target = f"{name}_{Path(relative).stem}.png"
        _save_figure(source, HERE / target)
        figures.append({"path": target, "label": label})

    steps = [re.sub(r" \(chosen because: [^)]*\)", "", line[2:]).strip()
             for line in (interpretation or "").splitlines() if line.startswith("- [")]
    status = str(results.get("status") or "unknown")
    demo = {
        "title": spec["title"],
        "caveat": (f"Cached output of an offline run (no API key, no LLM call) of "
                   f"PyHydroGeophysX {__version__} on data shipped in examples/data. "
                   "It is not a result for your data."),
        "summary": (f"Request: \"{spec['request']}\". The run finished with status "
                    f"'{status}' in {seconds:.0f} s: "
                    + " -> ".join(step.get("step", "") for step in plan) + "."),
        "metrics": _metrics(results),
        "figures": figures,
        "steps": steps,
        "warnings": [str(w) for w in results.get("warnings") or []],
        "provenance": {
            "generated_by": "examples/demo_cache/make_demo_cache.py",
            "generated": time.strftime("%Y-%m-%d"),
            "package_version": __version__,
            "inputs": {key: f"examples/data/{rel}" for key, rel in spec["inputs"].items()},
            "request": spec["request"],
            "status": status,
        },
    }
    (HERE / f"{name}.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return demo


def main() -> int:
    # Offline by construction: an agent given no key reads one from these, and a
    # cache that depended on a model's reply would not be reproducible.
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(key, None)
    import matplotlib

    matplotlib.use("Agg")
    workdir = Path(tempfile.mkdtemp(prefix="phgx_demo_"))
    # Some steps write beside the working directory (a results/ folder), so the
    # runs work from the temporary folder, not from the repository.
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        for name, spec in DEMOS.items():
            demo = build(name, spec, workdir)
            print(f"{name}: {demo['provenance']['status']}, {len(demo['figures'])} figures")
    finally:
        os.chdir(previous)
        shutil.rmtree(workdir, ignore_errors=True)
    total = sum(path.stat().st_size for path in HERE.glob("*") if path.suffix in (".json", ".png"))
    print(f"demo cache: {total / 1e6:.2f} MB in {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
