"""Initial built-in workflow handlers.

Workflow IDs are registry identifiers. They intentionally do not imply that a
same-named function exists in the scientific module.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .models import ArtifactRef, RunContext, WorkflowRunResult, WorkflowSpec


def _load_array(value: Any, context: RunContext, *, name: str) -> np.ndarray:
    if isinstance(value, ArtifactRef):
        cache_key = context.cache_key(value)
        if cache_key in context.object_cache:
            return np.asarray(context.object_cache[cache_key], dtype=float)
        path = context.resolve_artifact(value)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            key = str(value.metadata.get("array_key") or name)
            if key not in archive.files:
                if len(archive.files) != 1:
                    raise ValueError(
                        f"Artifact {path} has arrays {archive.files}; metadata.array_key is required."
                    )
                key = archive.files[0]
            array = archive[key]
        else:
            delimiter = "," if suffix == ".csv" else None
            array = np.loadtxt(path, delimiter=delimiter)
        context.object_cache[cache_key] = array
        return np.asarray(array, dtype=float)
    return np.asarray(value, dtype=float)


def _artifact(
    path: Path,
    *,
    context: RunContext,
    artifact_id: str,
    kind: str,
    format: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    return ArtifactRef.from_path(
        path,
        artifact_id=artifact_id,
        kind=kind,
        format=format,
        base_dir=context.project_root,
        metadata=metadata,
    )


def run_gravmag_process(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.workflows import gravmag

    x = _load_array(spec.inputs.get("x"), context, name="x").ravel()
    y = _load_array(spec.inputs.get("y"), context, name="y").ravel()
    values = _load_array(spec.inputs.get("values"), context, name="values").ravel()
    if not (x.size == y.size == values.size):
        raise ValueError("gravmag.process requires x, y, and values of equal length.")
    parameters = dict(spec.parameters)
    qc = gravmag.qc_products(
        x,
        y,
        values,
        detrend=int(parameters.get("detrend", 1)),
        nx=int(parameters.get("nx", 120)),
        ny=int(parameters.get("ny", 120)),
    )
    artifacts = []
    for label, grid in qc["grids"].items():
        slug = label.lower()
        paths = gravmag.save_grid(
            grid,
            context.output_dir,
            name=slug,
            log=context.progress,
        )
        artifacts.extend(
            _artifact(
                Path(path),
                context=context,
                artifact_id=f"gravmag:{slug}:{Path(path).suffix.lstrip('.')}",
                kind="gravmag_grid",
                metadata={"field": label},
            )
            for path in paths
        )
    profile = None
    if parameters.get("profile"):
        profile_config = parameters["profile"]
        profile = gravmag.extract_profile(
            qc["grids"][str(profile_config.get("field", "Residual"))],
            profile_config["p1"],
            profile_config["p2"],
            n=int(profile_config.get("n", 200)),
        )
        profile_path = context.output_dir / "profile.csv"
        with profile_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["distance", "x", "y", "value"])
            writer.writerows(zip(
                profile["distance"], profile["x"], profile["y"], profile["value"]
            ))
        artifacts.append(_artifact(
            profile_path,
            context=context,
            artifact_id="gravmag:profile:csv",
            kind="gravmag_profile",
        ))
    return WorkflowRunResult(
        status="ok",
        summary={
            "workflow_id": spec.workflow_id,
            "stations": int(x.size),
            "detrend": int(qc["detrend"]),
            "profile": bool(profile is not None),
        },
        metrics=qc["stats"],
        artifacts=artifacts,
        provenance={"workflow_id": spec.workflow_id, "schema_version": spec.schema_version},
        objects={"qc": qc, "profile": profile},
    )


def run_gravmag_forward_bodies(
    spec: WorkflowSpec,
    context: RunContext,
) -> WorkflowRunResult:
    from PyHydroGeophysX.workflows import gravmag

    x = _load_array(spec.inputs.get("x"), context, name="x").ravel()
    y = _load_array(spec.inputs.get("y"), context, name="y").ravel()
    kind = str(spec.parameters.get("kind", "gravity"))
    bodies = list(spec.parameters.get("bodies") or [])
    field = dict(spec.parameters.get("field") or {})
    response = gravmag.forward_bodies(
        x, y, kind, bodies, field=field, log=context.progress
    )
    npy_path = context.output_dir / f"{kind}_forward.npy"
    np.save(npy_path, response)
    csv_path = context.output_dir / f"{kind}_forward.csv"
    np.savetxt(
        csv_path,
        np.column_stack([x, y, response]),
        delimiter=",",
        header="x,y,response",
        comments="",
    )
    return WorkflowRunResult(
        status="ok",
        summary={
            "workflow_id": spec.workflow_id,
            "kind": kind,
            "stations": int(x.size),
            "bodies": len(bodies),
        },
        metrics={
            "min": float(np.min(response)),
            "max": float(np.max(response)),
            "mean": float(np.mean(response)),
        },
        artifacts=[
            _artifact(
                npy_path,
                context=context,
                artifact_id=f"gravmag:{kind}:forward:npy",
                kind="gravmag_response",
            ),
            _artifact(
                csv_path,
                context=context,
                artifact_id=f"gravmag:{kind}:forward:csv",
                kind="gravmag_response",
            ),
        ],
        provenance={"workflow_id": spec.workflow_id, "schema_version": spec.schema_version},
        objects={"response": response},
    )


def run_srt_inversion(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.inversion.srt_inversion import run_srt_manager_inversion

    travel_time = spec.inputs.get("traveltime")
    if not isinstance(travel_time, ArtifactRef):
        raise ValueError(
            "seismic.srt_inversion requires inputs.traveltime as an ArtifactRef."
        )
    travel_time_path = context.resolve_artifact(travel_time)
    # Only the inversion's own knobs are forwarded; receiver_spacing and any
    # other caller bookkeeping stay in the spec without reaching the solver.
    accepted = ("engine", "lam", "max_iterations", "max_total_iterations",
                "plateau_tolerance", "mesh_quality", "para_depth",
                "para_max_cell_size", "secondary_nodes", "auto_lambda",
                "target_chi2", "chi2_tolerance", "max_lambda_trials",
                "lambda_warm_start")
    options = {key: spec.parameters[key]
               for key in accepted if key in spec.parameters}
    result = run_srt_manager_inversion(
        travel_time_path,
        context.output_dir,
        log=context.progress,
        **options,
    )
    artifacts = []
    vtk = str(result.get("vtk") or "")
    if vtk and Path(vtk).is_file():
        artifacts.append(_artifact(
            Path(vtk),
            context=context,
            artifact_id="seismic:srt:velocity_vtk",
            kind="velocity_model",
        ))
    # The lambda the run settled on, and why, belong with the misfit: a chi2 is
    # not interpretable without knowing which lambda produced it.
    metrics = dict(result.get("metrics") or {})
    metrics["lambda"] = float(result.get("lambda_used", options.get("lam", 0.0)))
    if result.get("convergence_track"):
        metrics["convergence_track"] = result["convergence_track"]
    return WorkflowRunResult(
        status="ok",
        summary={
            "workflow_id": spec.workflow_id,
            "n": int(result["n"]),
            "pick_source": str(spec.metadata.get("pick_source", "uploaded")),
            "lambda_used": float(result.get("lambda_used", 0.0)),
            "auto_lambda_status": str(result.get("auto_lambda_status", "off")),
            "auto_lambda_note": str(result.get("auto_lambda_note", "")),
        },
        metrics=metrics,
        artifacts=artifacts,
        provenance={
            "workflow_id": spec.workflow_id,
            "schema_version": spec.schema_version,
            "inputs": {"traveltime": travel_time.to_dict()["$artifact"]},
        },
        objects={
            "manager": result["mgr"],
            "convergence": result.get("convergence") or [],
            "lambda_trials": result.get("lambda_trials") or [],
        },
    )


__all__ = [
    "run_gravmag_forward_bodies",
    "run_gravmag_process",
    "run_srt_inversion",
]
