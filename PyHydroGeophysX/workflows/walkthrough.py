"""Readable, step-by-step Python and notebooks from a workflow recipe.

:func:`PyHydroGeophysX.workflows.generate_python` emits a runner: one
``run_workflow(spec)`` call plus a nested-dict blob. That reproduces a run
exactly, which is what CI and batch reruns need, but it teaches nothing and its
parameters cannot really be edited.

This module emits the other half: the same run written the way a person would
write it, with named parameters, the actual domain calls, printed intermediates
and a figure. Both files come from one recipe, so a walkthrough always describes
the run it was generated from, and
``tests/test_walkthroughs.py::test_gravmag_walkthrough_matches_the_engine``
pins the two to identical artifacts.

Each :class:`Walkthrough` is the single prose description of what a workflow
does. Parameters are rendered into named module-level constants, so step code is
plain text that never needs escaping: it refers to those constants by name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .models import ArtifactRef, WorkflowSpec, iter_artifact_refs
from .registry import get_workflow

ParameterRenderer = Callable[[Mapping[str, Any]], Sequence[str]]
InputRenderer = Callable[[WorkflowSpec], Sequence[str]]


@dataclass(frozen=True)
class Step:
    """One numbered section of a walkthrough.

    ``note`` carries the reasoning a reader needs: what the step is for, how to
    tell whether it worked, and which parameter to reach for when it did not.
    ``code`` is emitted verbatim and refers to the constants that the parameter
    block defines.
    """

    title: str
    code: str
    note: str = ""


@dataclass(frozen=True)
class Walkthrough:
    summary: str
    imports: Sequence[str]
    steps: Sequence[Step]
    parameters: ParameterRenderer = lambda _p: ()
    inputs: InputRenderer | None = None
    reading: Sequence[str] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _wrap(text: str, width: int = 76, prefix: str = "# ") -> List[str]:
    """Wrap prose into comment lines without breaking words."""
    lines: List[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if len(prefix) + len(candidate) > width and current:
            lines.append(prefix + current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(prefix + current)
    return lines


def _banner(title: str, width: int = 78) -> str:
    prefix = f"# --- {title} "
    return prefix + "-" * max(3, width - len(prefix))


def _npz_groups(spec: WorkflowSpec) -> Dict[str, Dict[str, str]]:
    """Map an ``npz`` artifact path to ``{variable name: array key}``."""
    groups: Dict[str, Dict[str, str]] = {}
    for name, value in spec.inputs.items():
        refs = list(iter_artifact_refs(value))
        if len(refs) != 1 or refs[0].format != "npz":
            continue
        ref = refs[0]
        groups.setdefault(ref.path, {})[name] = str(
            ref.metadata.get("array_key", name)
        )
    return groups


def _variable(name: str) -> str:
    """A Python identifier for a recipe input name."""
    cleaned = "".join(char if char.isalnum() else "_" for char in name)
    return cleaned if not cleaned[:1].isdigit() else f"_{cleaned}"


def _bundle_directory(spec: WorkflowSpec, key: str, constant: str) -> List[str]:
    """Emit the one directory a file-bundle input resolves to.

    ``geo_hydrology`` and ``hydro_geophysics`` do not take individual files;
    their domain functions take the directory the files share, and the engine
    derives it from the artifact parents.
    """
    refs = list(iter_artifact_refs(spec.inputs.get(key) or {}))
    if not refs:
        return []
    parents = {Path(ref.path).parent.as_posix() for ref in refs}
    if len(parents) != 1:
        raise ValueError(f"{key} artifacts must share one directory, got {parents}.")
    parent = parents.pop()
    suffix = "" if parent in {"", "."} else f' / "{parent}"'
    return [f"{constant} = DATA_DIR{suffix}"]


def default_inputs(spec: WorkflowSpec) -> List[str]:
    """Load array bundles as named variables and other artifacts as paths."""
    lines: List[str] = []
    groups = _npz_groups(spec)
    for path, mapping in sorted(groups.items()):
        stem = _variable(Path(path).stem)
        lines.append(f'{stem} = np.load(DATA_DIR / "{Path(path).name}")')
        for name, key in sorted(mapping.items()):
            lines.append(f'{_variable(name)} = {stem}["{key}"].ravel()')
    handled = set(groups)
    for name, value in sorted(spec.inputs.items()):
        refs = [ref for ref in iter_artifact_refs(value) if ref.path not in handled]
        if len(refs) == 1:
            lines.append(f'{_variable(name)} = DATA_DIR / "{Path(refs[0].path).name}"')
        elif refs:
            joined = ", ".join(f'DATA_DIR / "{Path(r.path).name}"' for r in refs)
            lines.append(f"{_variable(name)} = [{joined}]")
    return lines


def _mentions(name: str, text: str) -> bool:
    """Whether *text* uses *name* as a whole identifier.

    Word boundaries matter here: a substring test would count the ``z`` inside
    ``zip(...)`` as a use of an input named ``z``.
    """
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def _used_input_lines(lines: Sequence[str], body: str) -> List[str]:
    """Drop input assignments whose variable no body step ever mentions.

    A recipe often carries more inputs than one workflow consumes; emitting the
    unused ones invites a reader to wonder what they were for.
    """
    kept: List[str] = []
    for line in lines:
        target = line.split("=", 1)[0].strip()
        if not target or not target.isidentifier():
            kept.append(line)
            continue
        # A bundle handle is kept when any variable extracted from it is used.
        dependents = [
            other for other in lines
            if other is not line and _mentions(target, other.split("=", 1)[-1])
        ]
        if _mentions(target, body) or any(
            _mentions(dep.split("=", 1)[0].strip(), body) for dep in dependents
        ):
            kept.append(line)
    return kept


_HEADER = '''"""{summary}

Generated by PyHydroGeophysX from workflow {workflow_id}.

Every parameter below is a plain module-level variable: change one and re-run.
For a byte-identical rerun of the original workbench run, use the companion
``run_{stem}.py``, which calls the workflow engine directly.
"""

from __future__ import annotations

from pathlib import Path

{imports}

#: Recipe inputs are resolved relative to this file.
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "walkthrough_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
'''


def _plan_for(spec: WorkflowSpec) -> Walkthrough:
    descriptor = get_workflow(spec.workflow_id)
    spec.validate(stochastic=descriptor.stochastic)
    try:
        return WALKTHROUGHS[spec.workflow_id]
    except KeyError as exc:
        raise NotImplementedError(
            f"No walkthrough is defined for {spec.workflow_id!r}. Use "
            f"generate_python() for a runner, or add an entry to "
            f"PyHydroGeophysX.workflows.walkthrough.WALKTHROUGHS."
        ) from exc


def _sections(spec: WorkflowSpec, plan: Walkthrough) -> Dict[str, Any]:
    """Resolve every piece a renderer needs, independent of output format."""
    parameters = dict(spec.parameters)
    body = "\n".join(step.code for step in plan.steps)
    render_inputs = plan.inputs or default_inputs
    input_lines = _used_input_lines(list(render_inputs(spec)), body)
    return {
        "parameters": list(plan.parameters(parameters)),
        "inputs": input_lines,
    }


def generate_walkthrough(spec: WorkflowSpec, path: str | Path) -> Path:
    """Write a readable, runnable transcript of *spec* as a ``.py`` script."""
    plan = _plan_for(spec)
    parts = _sections(spec, plan)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    chunks: List[str] = [
        _HEADER.format(
            summary=plan.summary,
            workflow_id=spec.workflow_id,
            stem=destination.stem.replace("_walkthrough", ""),
            imports="\n".join(plan.imports),
        )
    ]
    if parts["inputs"]:
        chunks.append(_banner("Inputs") + "\n" + "\n".join(parts["inputs"]) + "\n")
    if parts["parameters"]:
        chunks.append(_banner("Parameters") + "\n" + "\n".join(parts["parameters"]) + "\n")
    for index, step in enumerate(plan.steps, start=1):
        block = [_banner(f"Step {index}: {step.title}")]
        if step.note:
            block.extend(_wrap(step.note))
        block.append(step.code.rstrip())
        chunks.append("\n".join(block) + "\n")
    if plan.reading:
        chunks.append(
            _banner("Further reading") + "\n"
            + "\n".join(f"# {line}" for line in plan.reading) + "\n"
        )

    destination.write_text("\n".join(chunks), encoding="utf-8")
    return destination


def _markdown_cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _code_cell(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.rstrip().splitlines(True),
    }


def generate_notebook(spec: WorkflowSpec, path: str | Path) -> Path:
    """Write the same transcript as a Jupyter notebook.

    Prose becomes markdown cells and each step becomes one code cell, so a
    reader can run the steps one at a time and inspect what each produced.
    """
    plan = _plan_for(spec)
    parts = _sections(spec, plan)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.stem.replace("_walkthrough", "")

    cells: List[Dict[str, Any]] = [
        _markdown_cell(
            f"# {spec.workflow_id}\n\n{plan.summary}\n\n"
            f"Generated by PyHydroGeophysX from a workbench run. Every parameter "
            f"below is a plain variable: change one and re-run the cell.\n\n"
            f"For a byte-identical rerun of the original run, use the companion "
            f"`run_{stem}.py`, which calls the workflow engine directly.\n"
        ),
        _code_cell(
            "from pathlib import Path\n\n"
            + "\n".join(line for line in plan.imports if line)
            + "\n\nDATA_DIR = Path.cwd()\n"
            "OUT_DIR = DATA_DIR / \"walkthrough_results\"\n"
            "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        ),
    ]
    if parts["inputs"]:
        cells.append(_markdown_cell("## Inputs\n"))
        cells.append(_code_cell("\n".join(parts["inputs"])))
    if parts["parameters"]:
        cells.append(_markdown_cell("## Parameters\n"))
        cells.append(_code_cell("\n".join(parts["parameters"])))
    for index, step in enumerate(plan.steps, start=1):
        heading = f"## Step {index}: {step.title}\n"
        cells.append(_markdown_cell(f"{heading}\n{step.note}\n" if step.note else heading))
        cells.append(_code_cell(step.code))
    if plan.reading:
        listed = "\n".join(f"- `{line}`" for line in plan.reading)
        cells.append(_markdown_cell(f"## Further reading\n\n{listed}\n"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


# ---------------------------------------------------------------------------
# Parameter blocks
# ---------------------------------------------------------------------------
def _get(parameters: Mapping[str, Any], key: str, default: Any) -> Any:
    value = parameters.get(key)
    return default if value is None else value


def _gravmag_process_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    return (
        f'DETREND = {int(_get(p, "detrend", 1))}'
        "   # polynomial degree of the regional field: 0 = constant, 1 = plane, 2 = quadratic",
        f'NX, NY = {int(_get(p, "nx", 120))}, {int(_get(p, "ny", 120))}'
        "   # output grid size in cells",
    )


def _gravmag_forward_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    return (
        f'KIND = {str(_get(p, "kind", "gravity"))!r}'
        "   # 'gravity' -> mGal from density contrast; 'magnetics' -> nT from susceptibility",
        f'BODIES = {list(_get(p, "bodies", []))!r}',
        f'FIELD = {dict(_get(p, "field", {}))!r}'
        "   # inducing-field geometry; magnetics needs inclination/declination/strength",
    )


def _gravmag_invert_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    known = dict(p)
    kind = str(known.pop("kind", "gravity"))
    return (
        f"KIND = {kind!r}"
        "   # 'gravity' recovers density contrast (g/cc); 'magnetics' recovers susceptibility (SI)",
        f"INVERSION_PARAMETERS = {known!r}"
        "   # passed straight through to invert_gravmag",
    )


def _srt_invert_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    known = dict(p)
    known.pop("receiver_spacing", None)  # geometry bookkeeping, not a solver knob
    return (
        f"INVERSION_PARAMETERS = {known!r}"
        "   # passed straight through to run_srt_manager_inversion",
    )


def _ert_single_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    return (
        f'RELATIVE_ERROR = {float(_get(p, "relative_error", 0.03))}'
        "   # assumed data error; too small makes the inversion chase noise",
        f'MESH_QUALITY = {float(_get(p, "mesh_quality", 34.0))}'
        "   # minimum triangle angle; raise for a smoother mesh at higher cost",
        f'PARA_DEPTH = {float(_get(p, "para_depth", 0.0))}'
        "   # depth of the inverted domain in metres; 0 lets pyGIMLi size it",
        f'PARA_MAX_CELL_SIZE = {float(_get(p, "para_max_cell_size", 0.0))}'
        "   # largest cell in the inverted domain; 0 lets pyGIMLi size it",
        f'LAMBDA = {float(_get(p, "lambda", 50.0))}'
        "   # regularization strength; start on the smooth side and let the search relax it",
        f'MAX_ITERATIONS = {int(_get(p, "max_iterations", 20))}',
        f'INSTRUMENT = {_get(p, "instrument", None)!r}'
        "   # parser hint; None lets the reader auto-detect the format",
        f'ERROR_SOURCE = {str(_get(p, "error_source", "file"))!r}'
        "   # 'file' trusts the instrument's err column, 'estimate' recomputes it",
        f'ABSOLUTE_ERROR = {float(_get(p, "absolute_error", 0.0))}'
        "   # resistance floor in Ohm, added as absolute/|R|; matters at low signal",
        f'ENGINE = {str(_get(p, "engine", "pyhydro"))!r}'
        "   # 'pyhydro' is the in-house Gauss-Newton solver, 'pygimli' the ERTManager",
        f'GEOMETRIC_FACTOR_POLICY = {str(_get(p, "geometric_factor_policy", "fix"))!r}'
        "   # 'fix' recomputes k when a homogeneous forward run does not return rho0",
        f'GEOMETRIC_FACTOR_TOLERANCE = {float(_get(p, "geometric_factor_tolerance", 0.05))}'
        "   # how far that homogeneous response may drift before k is called wrong",
        f'PLATEAU_TOLERANCE = {float(_get(p, "plateau_tolerance", 0.005))}'
        "   # a lambda is done once chi2 improves by less than this fraction per step",
        f'MAX_TOTAL_ITERATIONS = {int(_get(p, "max_total_iterations", 60))}'
        "   # ceiling when a run is continued because it was still descending",
        f'REJECT_OUTLIERS = {bool(_get(p, "reject_outliers", False))}'
        "   # drop data the converged model cannot explain, then re-invert",
        f'OUTLIER_THRESHOLD = {float(_get(p, "outlier_threshold", 3.0))}'
        "   # rejection cut, in units of the assumed error",
        f'OUTLIER_PASSES = {int(_get(p, "outlier_passes", 2))}',
        f'MIN_DATA_FRACTION = {float(_get(p, "min_data_fraction", 0.5))}'
        "   # never reject below this share of the measurements",
        f'AUTO_LAMBDA = {bool(_get(p, "auto_lambda", False))}'
        "   # re-invert at other lambdas when LAMBDA misses the chi2 target",
        f'TARGET_CHI2 = {float(_get(p, "target_chi2", 1.0))}'
        "   # 1.0 means the model explains the data to within its error bars",
        f'CHI2_TOLERANCE = {float(_get(p, "chi2_tolerance", 0.2))}'
        "   # half-width of the accepted chi2 band around TARGET_CHI2",
        f'MAX_LAMBDA_TRIALS = {int(_get(p, "max_lambda_trials", 6))}'
        "   # cap on the extra inversions the lambda search may run",
        f'LAMBDA_WARM_START = {bool(_get(p, "lambda_warm_start", True))}'
        "   # continue each lambda from the nearest one already solved",
    )


def _passthrough_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    return (f"PARAMETERS = {dict(p)!r}",)


def _em_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    known = dict(p)
    method = str(known.pop("method", "TDEM")).upper()
    moment = str(known.pop("moment", "HM"))
    sounding = int(known.pop("sounding", 0))
    geometry = dict(known.pop("geometry", {}))
    return (
        f"METHOD = {method!r}"
        "   # 'FDEM' (frequency domain) or 'TDEM' (time domain)",
        f"MOMENT = {moment!r}"
        "   # TEMcompany transmitter moment: 'LM+HM' inverts both jointly",
        f"SOUNDING = {sounding}"
        "   # 1-based index of the sounding to preview from a multi-station file",
        f"GEOMETRY = {geometry!r}"
        "   # loop radius, flight height, orientation, waveform",
        f"INVERSION_PARAMETERS = {known!r}"
        "   # layer count, thickness range, smoothness, error floor, iteration cap",
    )


def _mesh3d_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    known = dict(p)
    formats = list(known.pop("output_formats", []))
    name = str(known.pop("output_name", "mesh3d"))
    return (
        f"MESH_CONFIG = {known!r}"
        "   # extent, cell sizes, layer markers, electrode layout",
        f"OUTPUT_NAME = {name!r}",
        f"OUTPUT_FORMATS = {formats!r}"
        "   # any of 'bms', 'vtk', 'msh'; empty keeps the mesh in memory only",
    )


def _joint_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    known = dict(p)
    method_a = str(known.pop("method_a", "ERT"))
    method_b = str(known.pop("method_b", "SRT"))
    strategy = str(known.pop("strategy", "cross_gradient_direct"))
    baseline = bool(known.pop("run_baseline", True))
    return (
        f"METHOD_A, METHOD_B = {method_a!r}, {method_b!r}"
        "   # the two datasets to couple",
        f"STRATEGY = {strategy!r}"
        "   # cross-gradient couples structure without assuming a petrophysical law",
        f"RUN_BASELINE = {baseline!r}"
        "   # also invert each method alone, so the joint result has something to beat",
        f"PARAMETERS = {known!r}",
    )


def _hydro_parameters(p: Mapping[str, Any]) -> Sequence[str]:
    known = dict(p)
    methods = list(known.pop("methods", []))
    point1 = known.pop("point1", None)
    point2 = known.pop("point2", None)
    return (
        f"METHODS = {methods!r}"
        "   # geophysical responses to simulate from the same hydrology model",
        f"POINT1, POINT2 = {point1!r}, {point2!r}"
        "   # profile endpoints in model coordinates",
        f"PARAMETERS = {known!r}"
        "   # petrophysical and survey settings shared by the selected methods",
    )


# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------
_NUMPY_IMPORTS = ("import matplotlib.pyplot as plt", "import numpy as np", "")

WALKTHROUGHS: Dict[str, Walkthrough] = {}


WALKTHROUGHS["gravmag.process"] = Walkthrough(
    summary=(
        "Gravity and magnetics station processing: separate the regional trend "
        "from the residual anomaly, grid both, and report QC statistics."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.data_processing import gravmag as gravmag_io",
        "from PyHydroGeophysX.workflows import gravmag",
    ),
    parameters=_gravmag_process_parameters,
    steps=(
        Step(
            title="Separate the regional trend from the residual anomaly",
            note=(
                "A potential field measured at the surface mixes a long-wavelength "
                "regional component, produced by deep or broad structure, with the "
                "short-wavelength residual produced by the shallow target. Fitting and "
                "removing a low-order polynomial surface is the classical separation. "
                "qc_products does the fit, grids all three fields onto a regular map, "
                "and returns per-field statistics."
            ),
            code=(
                "qc = gravmag_io.qc_products(x, y, values, detrend=DETREND, nx=NX, ny=NY)\n"
                "\n"
                "print(f\"{len(qc['x'])} finite stations, detrend degree {qc['detrend']}\")\n"
                'for name, stat in qc["stats"].items():\n'
                "    print(f\"  {name:9s} min={stat['min']:9.3f}  max={stat['max']:9.3f}\"\n"
                "          f\"  mean={stat['mean']:9.3f}  std={stat['std']:8.3f}\")"
            ),
        ),
        Step(
            title="Check the separation before trusting it",
            note=(
                "The residual should be centred on zero and the regional should be "
                "smooth. A residual that still carries a trend means the polynomial "
                "degree is too low; a residual that has lost the target means it is too "
                "high. This is the step where you change DETREND and look again."
            ),
            code=(
                'residual_mean = qc["stats"]["Residual"]["mean"]\n'
                "assert abs(residual_mean) < 1e-6, (\n"
                '    f"residual mean {residual_mean:.3g} is not centred on zero; "\n'
                '    "the regional fit did not converge"\n'
                ")"
            ),
        ),
        Step(
            title="Save each field as npy, CSV and VTK",
            note=(
                "save_grid writes one set of files per field. The CSV is the portable "
                "form, the npy keeps full precision, and the VTK opens in ParaView. VTK "
                "is best-effort and is skipped when pyvista is absent."
            ),
            code=(
                'for label, grid in qc["grids"].items():\n'
                "    written = gravmag.save_grid(grid, OUT_DIR, name=label.lower())\n"
                '    print(f"{label:9s} -> {len(written)} file(s)")'
            ),
        ),
        Step(
            title="Map the three fields",
            note=(
                "Observed, regional and residual each get their own colour scale so the "
                "eye compares shape rather than amplitude. Station positions are "
                "overlaid to show where the grid is interpolating far from data, which "
                "is where apparent anomalies are least trustworthy."
            ),
            code=(
                'fields = ["Observed", "Regional", "Residual"]\n'
                "fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)\n"
                "for ax, label in zip(axes, fields):\n"
                '    grid = qc["grids"][label]\n'
                '    mesh = ax.pcolormesh(grid["xx"], grid["yy"], grid["zz"], shading="auto")\n'
                '    ax.scatter(qc["x"], qc["y"], s=1, c="k", alpha=0.15)\n'
                "    ax.set_title(label)\n"
                '    ax.set_xlabel("x")\n'
                '    ax.set_aspect("equal")\n'
                "    fig.colorbar(mesh, ax=ax)\n"
                'axes[0].set_ylabel("y")\n'
                'fig.savefig(OUT_DIR / "fields.png", dpi=150)\n'
                "print(f\"figure -> {OUT_DIR / 'fields.png'}\")"
            ),
        ),
    ),
    reading=(
        "gravmag_io.regional_residual  - the polynomial fit behind DETREND",
        "gravmag_io.grid_data          - scattered-to-regular interpolation",
        "gravmag_io.extract_profile    - sample a grid along a line",
        "gravmag.invert_gravmag        - 3D inversion of the residual field",
    ),
)


WALKTHROUGHS["gravmag.forward_bodies"] = Walkthrough(
    summary=(
        "Analytic gravity or magnetic response of buried bodies, evaluated at the "
        "station positions."
    ),
    imports=_NUMPY_IMPORTS + ("from PyHydroGeophysX.workflows import gravmag",),
    parameters=_gravmag_forward_parameters,
    steps=(
        Step(
            title="Compute the response of every body",
            note=(
                "Each body is a dict describing a sphere or a right rectangular prism "
                "plus its physical contrast. Gravity uses closed-form expressions "
                "(Nagy 1966 for the prism); magnetics treats a sphere as an induced "
                "dipole. Responses superpose, so the total is the sum over bodies and "
                "you can add or remove one without recomputing the others."
            ),
            code=(
                "response = gravmag.forward_bodies(x, y, KIND, BODIES, field=FIELD)\n"
                "\n"
                'unit = "mGal" if KIND == "gravity" else "nT"\n'
                'print(f"{len(BODIES)} body/bodies at {x.size} stations")\n'
                'print(f"  min={response.min():.4f} {unit}  max={response.max():.4f} {unit}"\n'
                '      f"  mean={response.mean():.4f} {unit}")'
            ),
        ),
        Step(
            title="Save the synthetic response",
            note=(
                "The npy keeps full precision for a later inversion test; the CSV pairs "
                "each station with its value so the file stands alone."
            ),
            code=(
                'np.save(OUT_DIR / f"{KIND}_forward.npy", response)\n'
                "np.savetxt(\n"
                '    OUT_DIR / f"{KIND}_forward.csv",\n'
                "    np.column_stack([x, y, response]),\n"
                '    delimiter=",",\n'
                '    header="x,y,response",\n'
                '    comments="",\n'
                ")"
            ),
        ),
        Step(
            title="Map the synthetic anomaly",
            note=(
                "Plotting the forward response before inverting anything is the cheapest "
                "sanity check available: the anomaly should sit over the body, and its "
                "width should scale with burial depth."
            ),
            code=(
                "fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)\n"
                'scatter = ax.scatter(x, y, c=response, s=12, cmap="viridis")\n'
                'ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")\n'
                'ax.set_title(f"{KIND} forward response ({unit})")\n'
                "fig.colorbar(scatter, ax=ax)\n"
                'fig.savefig(OUT_DIR / f"{KIND}_forward.png", dpi=150)'
            ),
        ),
    ),
    reading=(
        "gravmag.gravity_sphere   - point-mass approximation for a compact body",
        "gravmag.gravity_prism    - Nagy (1966) right rectangular prism",
        "gravmag.magnetic_dipole  - induced magnetization of a sphere",
    ),
)


WALKTHROUGHS["gravmag.invert"] = Walkthrough(
    summary=(
        "SimPEG 3D inversion of a gravity or magnetic field for a subsurface "
        "property model."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.inversion.gravmag import backend_status, invert_gravmag",
    ),
    parameters=_gravmag_invert_parameters,
    steps=(
        Step(
            title="Confirm the potential-field backend is usable",
            note=(
                "The inversion needs SimPEG plus a working sparse solver. Checking first "
                "turns a missing dependency into one clear message instead of a "
                "traceback several minutes into a run."
            ),
            code=(
                "status = backend_status()\n"
                'print(status)\n'
                'assert status.get("available"), status'
            ),
        ),
        Step(
            title="Invert the field for a 3D property model",
            note=(
                "The station values are inverted onto a 3D cell model: gravity recovers "
                "density contrast in g/cc, magnetics recovers susceptibility in SI. "
                "Regularization keeps the result smooth, because potential-field data "
                "alone cannot resolve depth uniquely. Treat depth as the least certain "
                "dimension of the answer."
            ),
            code=(
                "# Station elevation is optional; pass it when the survey recorded it,\n"
                "# because burial depth is measured from the ground, not from z = 0.\n"
                'elevation = {"z": z} if "z" in globals() else {}\n'
                "\n"
                "result = invert_gravmag(\n"
                "    x, y, values, KIND, out_dir=str(OUT_DIR), **elevation, **INVERSION_PARAMETERS\n"
                ")\n"
                "\n"
                'model = np.asarray(result["model"], dtype=float)\n'
                'print(f"recovered {model.size} cells, "\n'
                '      f"range {model.min():.4g} to {model.max():.4g}")'
            ),
        ),
        Step(
            title="Check the data fit",
            note=(
                "A chi-squared near 1 means the model explains the data to within the "
                "assumed noise. Much below 1 means the data error was overstated and the "
                "model is fitting noise; much above 1 means the model cannot reproduce "
                "the observations and the regularization or the mesh needs revisiting."
            ),
            code=(
                'chi2 = result.get("chi2")\n'
                'print(f"chi2 = {chi2}")\n'
                'for key in ("rrms", "iterations", "n_data"):\n'
                "    if key in result:\n"
                '        print(f"  {key} = {result[key]}")'
            ),
        ),
    ),
    reading=(
        "inversion.gravmag.backend_status  - which backend pieces are present",
        "workflows.gravmag.save_grid       - write the observed/residual maps",
    ),
)


WALKTHROUGHS["seismic.srt_inversion"] = Walkthrough(
    summary=(
        "Seismic refraction tomography: invert first-arrival travel times for a "
        "2D velocity model."
    ),
    parameters=_srt_invert_parameters,
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.inversion.srt_inversion import run_srt_manager_inversion",
    ),
    steps=(
        Step(
            title="Invert the travel-time picks",
            note=(
                "The travel-time file pairs each source-receiver pair with a first-break "
                "time. Those picks are the entire input to the tomography, so their "
                "quality sets the ceiling on the result. Manual picks from the workbench "
                "were exported to this file, which is why the run is reproducible "
                "without repeating the picking. With auto_lambda set, the run at the "
                "lambda you chose happens first and is kept; the search only adds "
                "trials when its chi2 misses the target band."
            ),
            code=(
                "result = run_srt_manager_inversion(\n"
                "    traveltime, OUT_DIR, **INVERSION_PARAMETERS)\n"
                "\n"
                'print(f"{result[\'n\']} travel times inverted")\n'
                'print(f"lambda used: {result.get(\'lambda_used\')}")'
            ),
        ),
        Step(
            title="Read the convergence and fit",
            note=(
                "chi2 near 1 means the model reproduces the picks within their assumed "
                "error. Watch the per-iteration history as well: a fit that improves and "
                "then flattens has converged, while one still dropping at the last "
                "iteration was cut short by the iteration cap."
            ),
            code=(
                'for name, value in (result.get("metrics") or {}).items():\n'
                '    print(f"  {name:12s} {value}")\n'
                'convergence = result.get("convergence") or []\n'
                "if convergence:\n"
                '    print(f"  iterations   {len(convergence)}")'
            ),
        ),
        Step(
            title="Plot the velocity model",
            note=(
                "Refraction tomography resolves the shallow subsurface well and loses "
                "resolution with depth. Read the deepest part of the section as a trend "
                "rather than a measurement, and use the ~1200 m/s contour as the usual "
                "regolith/bedrock marker in weathered terrain."
            ),
            code=(
                "manager = result[\"mgr\"]\n"
                "try:\n"
                "    ax, cbar = manager.showResult()\n"
                "    ax.figure.savefig(OUT_DIR / \"velocity_model.png\", dpi=150)\n"
                "except Exception as exc:\n"
                '    print(f"pygimli could not draw the section directly: {exc}")\n'
                '    velocity = np.asarray(manager.velocity, dtype=float)\n'
                '    print(f"velocity range {velocity.min():.0f} to {velocity.max():.0f} m/s")'
            ),
        ),
    ),
    reading=(
        "data_processing.seismic.pick_first_breaks       - assisted first-break picking",
        "data_processing.seismic.first_breaks_to_traveltime - picks to a travel-time file",
        "Geophy_modular.structure_integration           - velocity to a bedrock interface",
    ),
)


WALKTHROUGHS["ert.single_inversion"] = Walkthrough(
    summary=(
        "Single-dataset ERT inversion: apparent resistivity measurements to a 2D "
        "resistivity model."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion",
    ),
    parameters=_ert_single_parameters,
    steps=(
        Step(
            title="Invert the measurements",
            note=(
                "The reader detects the instrument format, builds a mesh around the "
                "electrode layout, and runs a smoothness-constrained Gauss-Newton "
                "inversion. The stages run in the order that actually lowers chi2: fix "
                "the error model first, iterate at LAMBDA until the misfit flattens, "
                "drop the data the converged model cannot explain, and only then let "
                "LAMBDA move. Reversing that order blames the regularization for a bad "
                "error model or for an unfinished descent."
            ),
            code=(
                "result = run_ert_manager_inversion(\n"
                "    data,\n"
                "    OUT_DIR,\n"
                "    instrument=INSTRUMENT,\n"
                "    engine=ENGINE,\n"
                "    geometric_factor_policy=GEOMETRIC_FACTOR_POLICY,\n"
                "    geometric_factor_tolerance=GEOMETRIC_FACTOR_TOLERANCE,\n"
                "    relative_error=RELATIVE_ERROR,\n"
                "    absolute_error=ABSOLUTE_ERROR,\n"
                "    error_source=ERROR_SOURCE,\n"
                "    mesh_quality=MESH_QUALITY,\n"
                "    para_depth=PARA_DEPTH,\n"
                "    para_max_cell_size=PARA_MAX_CELL_SIZE,\n"
                "    lam=LAMBDA,\n"
                "    max_iterations=MAX_ITERATIONS,\n"
                "    plateau_tolerance=PLATEAU_TOLERANCE,\n"
                "    max_total_iterations=MAX_TOTAL_ITERATIONS,\n"
                "    reject_outliers=REJECT_OUTLIERS,\n"
                "    outlier_threshold=OUTLIER_THRESHOLD,\n"
                "    outlier_passes=OUTLIER_PASSES,\n"
                "    min_data_fraction=MIN_DATA_FRACTION,\n"
                "    auto_lambda=AUTO_LAMBDA,\n"
                "    target_chi2=TARGET_CHI2,\n"
                "    chi2_tolerance=CHI2_TOLERANCE,\n"
                "    max_lambda_trials=MAX_LAMBDA_TRIALS,\n"
                "    lambda_warm_start=LAMBDA_WARM_START,\n"
                ")"
            ),
        ),
        Step(
            title="Judge the fit before reading the model",
            note=(
                "Read result['geometric_factors'] before anything else. A geometric "
                "factor that is uniformly wrong rescales the whole section and leaves "
                "chi2 untouched, so a perfect fit is no defence against it. "
                "chi2 near 1 means the model explains the data to within the assumed "
                "error. Well below 1 means the error was too generous and structure in "
                "the image may be fitted noise. Well above 1 has three causes worth "
                "separating, and result tells them apart: implied_from_residuals far "
                "above the assumed error means the error model is wrong; a "
                "convergence_stop of 'iteration_cap' means the run never finished; and "
                "a lambda search that flattens well above the target at every trial "
                "means neither lambda nor iteration count can save it."
            ),
            code=(
                'geom = result["geometric_factors"]\n'
                'if geom.get("repaired") or (geom.get("checked") and not geom["ok"]):\n'
                '    print("GEOMETRIC FACTORS:", geom["message"])\n'
                'for key in ("chi2", "rrms", "iterations", "n_data"):\n'
                "    if key in result[\"metrics\"]:\n"
                '        print(f"  {key:12s} {result[\'metrics\'][key]}")\n'
                'print(f"  lambda       {result[\'lambda_used\']} '
                '(requested {result[\'lambda_requested\']})")\n'
                'print(f"  stopped on   {result[\'convergence_stop\']}")\n'
                'err = result["data_error"]\n'
                'print(f"  error model  {err[\'source\']}, mean {err[\'mean\']:.3%}; '
                'residuals imply {err[\'implied_from_residuals\']:.1%}")\n'
                'if result["outliers"].get("dropped"):\n'
                '    print(f"  rejected     {result[\'outliers\'][\'dropped\']} of '
                '{result[\'outliers\'][\'n_start\']} measurements")\n'
                "if result.get(\"auto_lambda_note\"):\n"
                "    print(result[\"auto_lambda_note\"])\n"
                "for trial in result.get(\"lambda_trials\", []):\n"
                '    print(f"    lambda={trial[\'lambda\']:g}  chi2={trial[\'chi2\']:.3f}  '
                '{trial[\'iterations\']} iters, stopped on {trial[\'stop\']}")'
            ),
        ),
        Step(
            title="Plot the resistivity section",
            note=(
                "ERT sensitivity falls off with depth and away from the electrode line, "
                "so trust the centre of the section most. Where the manager exposes a "
                "coverage array, blank the poorly constrained cells rather than letting "
                "them read as real structure."
            ),
            code=(
                'manager = result.get("mgr")\n'
                "if manager is not None:\n"
                "    try:\n"
                "        ax, cbar = manager.showResult()\n"
                '        ax.figure.savefig(OUT_DIR / "resistivity_model.png", dpi=150)\n'
                "    except Exception as exc:\n"
                '        print(f"pygimli could not draw the section directly: {exc}")'
            ),
        ),
    ),
    reading=(
        "data_processing.ert_io          - readers and edited-container export",
        "inversion.time_lapse            - the same data through time",
        "Geophy_modular.ERT_to_WC        - resistivity to water content",
    ),
)


WALKTHROUGHS["ert.timelapse_inversion"] = Walkthrough(
    summary=(
        "Time-lapse ERT inversion: a sequence of datasets inverted together so that "
        "change through time is resolved rather than re-imaged independently."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.inversion.time_lapse import run_timelapse_ert",
    ),
    parameters=_passthrough_parameters,
    steps=(
        Step(
            title="Invert the whole series at once",
            note=(
                "Inverting each survey separately lets unconstrained noise differ between "
                "time steps, which shows up as spurious change. A time-lapse inversion "
                "adds temporal regularization so the model is allowed to change only "
                "where the data require it. That is why the file list and the measurement "
                "times are passed together rather than looped over."
            ),
            code=(
                "result = run_timelapse_ert(\n"
                "    [str(path) for path in data_files],\n"
                "    list(measurement_times),\n"
                "    PARAMETERS,\n"
                "    str(OUT_DIR),\n"
                ")\n"
                "\n"
                'print(f"{len(data_files)} time steps inverted")'
            ),
        ),
        Step(
            title="Read the per-step fit",
            note=(
                "One badly fitted time step can drag the whole series. Compare chi2 "
                "across steps rather than looking only at the aggregate: an outlier "
                "usually means a survey with bad contacts rather than real change."
            ),
            code=(
                'for key in ("chi2", "rrms", "iterations", "n_data"):\n'
                "    if key in result:\n"
                '        print(f"  {key:12s} {result[key]}")'
            ),
        ),
        Step(
            title="Show change relative to the first survey",
            note=(
                "Time-lapse results are read as ratios or differences against a baseline, "
                "not as absolute sections. A resistivity decrease usually means wetting "
                "and an increase usually means drying, but temperature also shifts "
                "resistivity, so correct for it before converting change to water content."
            ),
            code=(
                'models = result.get("models")\n'
                "if models is not None:\n"
                "    models = np.asarray(models, dtype=float)\n"
                "    baseline = models[0]\n"
                "    for index, model in enumerate(models[1:], start=1):\n"
                "        ratio = model / baseline\n"
                '        print(f"  step {index}: ratio {ratio.min():.3f} to {ratio.max():.3f}")'
            ),
        ),
    ),
    reading=(
        "inversion.windowed              - window the series for long time-lapse runs",
        "Geophy_modular.ERT_to_WC        - resistivity change to water-content change",
    ),
)


WALKTHROUGHS["ert3d.forward"] = Walkthrough(
    summary="3D ERT forward simulation over a prepared mesh and electrode layout.",
    imports=_NUMPY_IMPORTS + (
        "import pandas as pd",
        "",
        "from PyHydroGeophysX.core.mesh_serialization import load_mesh_artifact",
        "from PyHydroGeophysX.forward.ert3d import run_ert3d_forward",
    ),
    parameters=_passthrough_parameters,
    steps=(
        Step(
            title="Load the mesh and its sidecar",
            note=(
                "A pygimli mesh file alone does not preserve cell markers, region "
                "definitions and secondary nodes, so the workbench writes a sidecar "
                "alongside it. Loading both restores the mesh the forward run actually "
                "used; loading the .bms alone would silently lose the layering."
            ),
            code=(
                "mesh = load_mesh_artifact(mesh, mesh_structure)\n"
                "electrodes = pd.read_csv(sensors)\n"
                'print(f"{mesh.cellCount()} cells, {len(electrodes)} electrodes")'
            ),
        ),
        Step(
            title="Simulate the measurements",
            note=(
                "The forward run produces the apparent resistivities the survey would "
                "record over this model. Use it to test whether a planned electrode "
                "layout can resolve the target before going to the field, and to generate "
                "synthetic data for inversion tests where the true model is known."
            ),
            code=(
                "result = run_ert3d_forward(mesh, electrodes, output_dir=str(OUT_DIR), **PARAMETERS)\n"
                "\n"
                'for key, value in result.items():\n'
                "    if isinstance(value, (int, float, str)):\n"
                '        print(f"  {key:16s} {value}")'
            ),
        ),
    ),
    reading=(
        "core.mesh_3d.generate_mesh        - build the mesh this workflow consumes",
        "core.mesh_serialization           - lossless mesh save and load",
    ),
)


WALKTHROUGHS["em.inversion"] = Walkthrough(
    summary=(
        "1D electromagnetic inversion: a frequency- or time-domain sounding to a "
        "layered resistivity model."
    ),
    imports=_NUMPY_IMPORTS + ("from PyHydroGeophysX.workflows import em1d",),
    parameters=_em_parameters,
    steps=(
        Step(
            title="Load the sounding",
            note=(
                "The reader accepts single-sounding CSV exports and complete TEMcompany "
                "project directories. For a dual-moment survey, 'LM+HM' keeps both "
                "moments so they can be inverted jointly: the low moment constrains the "
                "shallow layers and the high moment reaches deeper."
            ),
            code=(
                "data = em1d.load_sounding(str(data), METHOD, sounding=SOUNDING, moment=MOMENT)\n"
                "\n"
                'print(f"method {METHOD}, moments {data.get(\"moments\") or [MOMENT]}")'
            ),
        ),
        Step(
            title="Invert for a layered model",
            note=(
                "The inversion solves for resistivity in a fixed set of layers whose "
                "thicknesses grow with depth, because EM resolution degrades downward. "
                "Smoothness keeps neighbouring layers from oscillating. A 1D result is "
                "only meaningful where the ground is approximately layered beneath the "
                "loop, so check that assumption before interpreting laterally."
            ),
            code=(
                'if METHOD == "FDEM":\n'
                "    result = em1d.fdem_invert(data, GEOMETRY, INVERSION_PARAMETERS)\n"
                'elif data.get("moments"):\n'
                "    result = em1d.tdem_joint_invert(data, GEOMETRY, INVERSION_PARAMETERS)\n"
                "else:\n"
                "    result = em1d.tdem_invert(data, GEOMETRY, INVERSION_PARAMETERS)\n"
                "\n"
                'for key in ("chi2", "rrms", "iterations"):\n'
                "    if key in result:\n"
                '        print(f"  {key:12s} {result[key]}")'
            ),
        ),
        Step(
            title="Plot the depth profile",
            note=(
                "A layered model is drawn as a step profile. Layer boundaries are as "
                "uncertain as the resistivity contrast across them is small, so read a "
                "weak contrast as a gradual transition rather than a sharp interface."
            ),
            code=(
                'thickness = np.asarray(result.get("thickness", []), dtype=float)\n'
                'resistivity = np.asarray(result.get("resistivity", []), dtype=float)\n'
                "if resistivity.size:\n"
                "    depth, rho = em1d.model_depth_profile(thickness, resistivity)\n"
                "    fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)\n"
                "    ax.step(rho, depth, where=\"post\")\n"
                '    ax.set_xscale("log"); ax.invert_yaxis()\n'
                '    ax.set_xlabel("resistivity (ohm-m)"); ax.set_ylabel("depth (m)")\n'
                '    fig.savefig(OUT_DIR / "em_profile.png", dpi=150)'
            ),
        ),
    ),
    reading=(
        "em1d.example_catalog       - documented example soundings and their settings",
        "em1d.calibrate_to_reference - fix the absolute level of normalized airborne data",
        "em1d.invert_line           - the same inversion along a whole line",
    ),
)


WALKTHROUGHS["mesh3d.build"] = Walkthrough(
    summary="Build a 3D mesh with topography and layer markers for forward modelling.",
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.core.mesh_3d import generate_mesh, save_outputs",
    ),
    parameters=_mesh3d_parameters,
    steps=(
        Step(
            title="Generate the mesh",
            note=(
                "Cell size controls both accuracy and cost: too coarse and the forward "
                "response is wrong near electrodes, too fine and the inversion becomes "
                "impractical. A common compromise is cells smaller than half the minimum "
                "electrode spacing near the surface, growing with depth."
            ),
            code=(
                "config = dict(MESH_CONFIG)\n"
                'config["output_dir"] = str(OUT_DIR)\n'
                'config["topography_points"] = np.load(topography_points)\n'
                "result = generate_mesh(config)\n"
                "\n"
                'mesh = result["mesh"]\n'
                'print(f"{mesh.cellCount()} cells, {mesh.nodeCount()} nodes")'
            ),
        ),
        Step(
            title="Write the mesh in the requested formats",
            note=(
                "bms is the pygimli native format and round-trips losslessly with the "
                "sidecar; vtk opens in ParaView for inspection; msh suits external "
                "solvers. Writing more than one costs little and saves a rebuild later."
            ),
            code=(
                "if OUTPUT_FORMATS:\n"
                "    outputs = save_outputs(\n"
                '        result["mesh"], result["electrodes"], OUT_DIR, OUTPUT_NAME, OUTPUT_FORMATS\n'
                "    )\n"
                "    for name, path in outputs.items():\n"
                '        print(f"  {name:6s} -> {path}")'
            ),
        ),
    ),
    reading=(
        "core.mesh_serialization.save_mesh_artifact - mesh plus sidecar, lossless",
        "forward.ert3d.run_ert3d_forward            - simulate over this mesh",
    ),
)


WALKTHROUGHS["seismic3d.build"] = Walkthrough(
    summary=(
        "Interpolate 2D seismic velocity lines into a 3D structural model and "
        "extract the bedrock interface."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.Geophy_modular.structure_integration import build_3d_model",
    ),
    parameters=_passthrough_parameters,
    steps=(
        Step(
            title="Build the 3D model from the 2D lines",
            note=(
                "Each line contributes a velocity section along its own map trace. "
                "Kriging fills the volume between lines, so the result is best near the "
                "lines and increasingly interpolated away from them. The line spacing "
                "therefore sets the real resolution of the volume, whatever grid "
                "resolution is requested."
            ),
            code=(
                "context = {\"output_dir\": str(OUT_DIR)}\n"
                "parameters = dict(PARAMETERS)\n"
                'parameters["output_dir"] = str(OUT_DIR)\n'
                'parameters["lines"] = lines\n'
                "result = build_3d_model(context, parameters)\n"
                "\n"
                'print(f"status {result.get(\"status\")}, '
                '{result.get(\"n_grid_points\")} grid points")'
            ),
        ),
        Step(
            title="Read the bedrock interface",
            note=(
                "The interface is the depth where velocity crosses a threshold, commonly "
                "around 1200 m/s for the regolith to bedrock transition in weathered "
                "terrain. That threshold is site-specific: calibrate it against a "
                "borehole where one exists rather than accepting the default."
            ),
            code=(
                'for key in ("interface_path", "vtk_path", "output_dir"):\n'
                "    if result.get(key):\n"
                '        print(f"  {key:16s} {result[key]}")'
            ),
        ),
    ),
    reading=(
        "Geophy_modular.structure_integration.extract_line_structure - one line to an interface",
        "Geophy_modular.ERT_to_WC.derive_markers_from_interface      - interface to ERT layer markers",
    ),
)


WALKTHROUGHS["geo_hydrology.ert_to_wc"] = Walkthrough(
    summary=(
        "Convert an inverted ERT resistivity model to water content and porosity "
        "with Monte Carlo uncertainty."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.Geophy_modular.ERT_to_WC import run_ert_to_wc",
    ),
    parameters=_passthrough_parameters,
    inputs=lambda spec: _bundle_directory(spec, "model_files", "MODEL_DIR"),
    steps=(
        Step(
            title="Convert resistivity to water content",
            note=(
                "Archie-type petrophysics maps resistivity to saturation, and porosity "
                "turns saturation into water content. Every parameter in that mapping "
                "carries uncertainty, so the conversion is run many times with parameters "
                "drawn from per-layer distributions. The spread across realizations is "
                "the honest error bar; a single deterministic conversion would hide it."
            ),
            code=(
                'context = {"output_dir": str(OUT_DIR), "geo_data_dir": str(MODEL_DIR)}\n'
                "parameters = dict(PARAMETERS)\n"
                'parameters["output_dir"] = str(OUT_DIR)\n'
                'parameters["model_data_dir"] = str(MODEL_DIR)\n'
                "result = run_ert_to_wc(context, parameters)\n"
                "\n"
                "print(f\"status {result.get('status')}, \"\n"
                "      f\"{result.get('n_realizations')} realizations, \"\n"
                "      f\"{result.get('mesh_cells')} cells\")"
            ),
        ),
        Step(
            title="Read the uncertainty, not just the mean",
            note=(
                "The mean map looks like a deterministic answer and invites overreading. "
                "The standard deviation shows where the petrophysical parameters, rather "
                "than the geophysics, control the result. Layers whose parameters were "
                "poorly constrained show wide spread even where the resistivity model is "
                "well resolved."
            ),
            code=(
                'for key, value in result.items():\n'
                '    if key.endswith("_paths") and isinstance(value, (list, tuple)):\n'
                '        print(f"  {key}: {len(value)} file(s)")\n'
                '    elif isinstance(value, (int, float, str)) and key != "status":\n'
                '        print(f"  {key:18s} {value}")'
            ),
        ),
    ),
    reading=(
        "petrophysics.monte_carlo.run_petrophysics_monte_carlo - the seeded sampler",
        "petrophysics.resistivity_models                       - the Archie relations",
        "Geophy_modular.ERT_to_WC.derive_markers_from_interface - layers from seismic structure",
    ),
)


WALKTHROUGHS["hydro_geophysics.forward"] = Walkthrough(
    summary=(
        "Simulate several geophysical responses from one hydrological model, so the "
        "methods can be compared on identical ground truth."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.Hydro_modular.hydro_to_geophysics import run_hydro_forward",
    ),
    parameters=_hydro_parameters,
    inputs=lambda spec: _bundle_directory(spec, "hydro_files", "HYDRO_DIR"),
    steps=(
        Step(
            title="Run the coupled forward simulation",
            note=(
                "Water content and porosity from the hydrological model are converted to "
                "resistivity, seismic velocity and density through rock physics, and each "
                "selected method is then simulated on the resulting property field. "
                "Because every method sees the same subsurface, differences between them "
                "come from method sensitivity rather than from differing assumptions."
            ),
            code=(
                'context = {"output_dir": str(OUT_DIR)}\n'
                "parameters = dict(PARAMETERS)\n"
                'parameters["output_dir"] = str(OUT_DIR)\n'
                'parameters["hydro_data_dir"] = str(HYDRO_DIR)\n'
                "result = run_hydro_forward(context, parameters, METHODS, POINT1, POINT2)\n"
                "\n"
                "print(f\"status {result.get('status')}, methods {METHODS}\")"
            ),
        ),
        Step(
            title="Compare what each method resolved",
            note=(
                "ERT follows water content most directly, seismic velocity responds to "
                "both saturation and matrix stiffness, and gravity responds to bulk "
                "density change. Where two methods disagree over the same ground, the "
                "disagreement is information about which property actually changed."
            ),
            code=(
                'for key, value in sorted(result.items()):\n'
                '    if key.endswith(("_path", "_paths")):\n'
                '        print(f"  {key}: {value}")'
            ),
        ),
    ),
    reading=(
        "Hydro_modular.hydro_to_ert     - resistivity from water content",
        "Hydro_modular.hydro_to_srt     - velocity via Hertz-Mindlin and DEM",
        "Hydro_modular.hydro_to_gravity - density change to gravity response",
    ),
)


WALKTHROUGHS["joint_inversion.run"] = Walkthrough(
    summary=(
        "Joint inversion of two geophysical datasets, coupled so that the recovered "
        "models share structure."
    ),
    imports=_NUMPY_IMPORTS + (
        "from PyHydroGeophysX.data_processing.joint_io import load_joint_observations",
        "from PyHydroGeophysX.inversion.joint import run_joint_inversion",
        "from PyHydroGeophysX.inversion.joint_api import JointInversionRequest",
    ),
    parameters=_joint_parameters,
    steps=(
        Step(
            title="Load both datasets",
            note=(
                "Each method keeps its own reader, because a travel-time file and a "
                "resistivity file share nothing but their geometry. The loader normalizes "
                "them into the observation containers the joint solver expects."
            ),
            code=(
                "observations = {\n"
                "    METHOD_A: load_joint_observations(METHOD_A, data_a),\n"
                "    METHOD_B: load_joint_observations(METHOD_B, data_b),\n"
                "}\n"
                'print(f"loaded {sorted(observations)}")'
            ),
        ),
        Step(
            title="Invert the two datasets together",
            note=(
                "Cross-gradient coupling penalizes models whose gradients point in "
                "different directions, which pushes boundaries to line up without "
                "assuming any relation between resistivity and velocity. That matters "
                "when no reliable petrophysical law links the two properties at the site."
            ),
            code=(
                "request = JointInversionRequest(\n"
                "    method_a=METHOD_A,\n"
                "    method_b=METHOD_B,\n"
                "    strategy=STRATEGY,\n"
                "    data=observations,\n"
                "    parameters=PARAMETERS,\n"
                "    output_dir=OUT_DIR,\n"
                "    run_baseline=RUN_BASELINE,\n"
                ")\n"
                "result = run_joint_inversion(request)"
            ),
        ),
        Step(
            title="Compare against the single-method baselines",
            note=(
                "A joint result is only worth its extra complexity if it beats each "
                "method inverted alone. With RUN_BASELINE on, the separate inversions are "
                "produced in the same run, so the comparison uses identical data, mesh "
                "and regularization. If the joint model does not improve the fit or "
                "sharpen the structure, report that rather than the joint model."
            ),
            code=(
                'for key, value in sorted(vars(result).items() if hasattr(result, "__dict__")\n'
                "                        else dict(result).items()):\n"
                "    if isinstance(value, (int, float, str)):\n"
                '        print(f"  {key:20s} {value}")'
            ),
        ),
    ),
    reading=(
        "inversion.joint_api.get_joint_capabilities - which method pairs and strategies exist",
        "inversion.joint_ert_srt                    - the ERT/SRT cross-gradient solver",
    ),
)


__all__ = [
    "Step",
    "Walkthrough",
    "WALKTHROUGHS",
    "default_inputs",
    "generate_notebook",
    "generate_walkthrough",
]
