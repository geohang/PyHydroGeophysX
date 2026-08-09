"""Adapters from serializable workflow contracts to canonical domain APIs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .models import ArtifactRef, RunContext, WorkflowRunResult, WorkflowSpec


def _materialize(value: Any, context: RunContext) -> Any:
    if isinstance(value, ArtifactRef):
        return str(context.resolve_artifact(value))
    if isinstance(value, Mapping):
        return {str(key): _materialize(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item, context) for item in value]
    return value


def _json_value(value: Any) -> tuple[bool, Any]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True, value
    if isinstance(value, np.generic):
        return True, value.item()
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            ok, encoded = _json_value(item)
            if not ok:
                return False, None
            result[str(key)] = encoded
        return True, result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            ok, encoded = _json_value(item)
            if not ok:
                return False, None
            result.append(encoded)
        return True, result
    if is_dataclass(value):
        return _json_value(asdict(value))
    return False, None


def _legacy_result(
    spec: WorkflowSpec,
    context: RunContext,
    result: Any,
) -> WorkflowRunResult:
    if is_dataclass(result):
        raw = asdict(result)
        objects = {"domain_result": result}
    elif isinstance(result, Mapping):
        raw = dict(result)
        objects = {}
    else:
        raw = {}
        objects = {"domain_result": result}
    status = str(raw.pop("status", "ok"))
    artifacts = []
    summary: Dict[str, Any] = {}
    for key, value in raw.items():
        if key == "output_dir":
            summary[key] = str(value)
            continue
        path_values = []
        if (
            key.endswith("_path") or key in {"vtk", "vtk_combined"}
        ) and isinstance(value, (str, Path)):
            path_values = [value]
        elif key.endswith("_paths") and isinstance(value, (list, tuple)):
            path_values = value
        elif key in {"figure_paths", "data_paths", "artifacts", "outputs"}:
            if isinstance(value, Mapping):
                path_values = list(value.values())
            elif isinstance(value, (list, tuple)):
                path_values = value
        for index, candidate in enumerate(path_values):
            path = Path(str(candidate))
            if path.is_file():
                artifacts.append(ArtifactRef.from_path(
                    path,
                    artifact_id=f"{spec.workflow_id}:{key}:{index}",
                    kind=key.rstrip("s"),
                    base_dir=context.project_root,
                ))
        ok, encoded = _json_value(value)
        if ok:
            summary[key] = encoded
        else:
            objects[key] = value
    metrics = dict(summary.pop("metrics", {}) or {})
    for key in ("chi2", "rrms", "iterations", "n_data"):
        if key in summary:
            metrics.setdefault(key, summary[key])
    return WorkflowRunResult(
        status=status,
        summary=summary,
        metrics=metrics,
        artifacts=artifacts,
        warnings=list(summary.pop("warnings", []) or []),
        provenance={"workflow_id": spec.workflow_id, "schema_version": spec.schema_version},
        objects=objects,
    )


def run_hydro_geophysics(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.Hydro_modular.hydro_to_geophysics import run_hydro_forward

    inputs = _materialize(spec.inputs, context)
    parameters = dict(_materialize(spec.parameters, context))
    parameters["seed"] = int(spec.seed)
    parameters["output_dir"] = str(context.output_dir)
    hydro_files = dict(inputs.get("hydro_files") or {})
    if hydro_files:
        parents = {str(Path(path).parent) for path in hydro_files.values()}
        if len(parents) != 1:
            raise ValueError("Hydrology input artifacts must share one directory.")
        parameters["hydro_data_dir"] = parents.pop()
    domain_context = dict(inputs.get("context") or inputs)
    domain_context.setdefault("output_dir", str(context.output_dir))
    methods = list(parameters.pop("methods", inputs.get("methods", [])))
    point1 = parameters.pop("point1", inputs.get("point1"))
    point2 = parameters.pop("point2", inputs.get("point2"))
    result = run_hydro_forward(
        domain_context, parameters, methods, point1, point2, log=context.progress
    )
    return _legacy_result(spec, context, result)


def run_geo_hydrology(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.Geophy_modular.ERT_to_WC import run_ert_to_wc

    inputs = _materialize(spec.inputs, context)
    parameters = dict(_materialize(spec.parameters, context))
    parameters["seed"] = int(spec.seed)
    parameters["output_dir"] = str(context.output_dir)
    model_files = dict(inputs.get("model_files") or {})
    if model_files:
        parents = {str(Path(path).parent) for path in model_files.values()}
        if len(parents) != 1:
            raise ValueError("ERT model artifacts must share one directory.")
        parameters["model_data_dir"] = parents.pop()
    domain_context = dict(inputs.get("context") or inputs)
    domain_context.setdefault("output_dir", str(context.output_dir))
    return _legacy_result(
        spec, context, run_ert_to_wc(domain_context, parameters, log=context.progress)
    )


def run_seismic3d(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.Geophy_modular.structure_integration import build_3d_model

    inputs = _materialize(spec.inputs, context)
    parameters = dict(_materialize(spec.parameters, context))
    parameters["output_dir"] = str(context.output_dir)
    parameters["lines"] = list(inputs.get("lines") or [])
    domain_context = dict(inputs.get("context") or inputs)
    domain_context.setdefault("output_dir", str(context.output_dir))
    return _legacy_result(
        spec, context, build_3d_model(domain_context, parameters, log=context.progress)
    )


def run_mesh3d(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.core.mesh_3d import generate_mesh, save_outputs

    config = dict(_materialize(spec.parameters, context))
    inputs = _materialize(spec.inputs, context)
    if inputs.get("topography_points"):
        config["topography_points"] = np.load(inputs["topography_points"])
    config["output_dir"] = str(context.output_dir)
    output_formats = list(config.pop("output_formats", []))
    output_name = str(config.pop("output_name", "mesh3d"))
    result = generate_mesh(config, log=context.progress)
    if output_formats:
        # A mesh that generated is worth keeping even when writing it out fails.
        # Letting the export raise here discards it and reports "mesh generation
        # failed", which sends the user looking at meshing parameters for what is
        # really a filesystem or path-encoding problem.
        try:
            result["outputs"] = save_outputs(
                result["mesh"],
                result["electrodes"],
                context.output_dir,
                output_name,
                output_formats,
            )
        except Exception as exc:  # noqa: BLE001 - the mesh itself is still good
            result["outputs"] = {}
            result["output_error"] = str(exc)
            context.progress(
                f"The mesh generated, but saving it to {context.output_dir} failed: {exc}")
    return _legacy_result(spec, context, result)


def run_ert_timelapse(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.inversion.time_lapse import run_timelapse_ert

    inputs = _materialize(spec.inputs, context)
    files = list(inputs.get("data_files") or [])
    times = list(inputs.get("measurement_times") or range(len(files)))
    result = run_timelapse_ert(
        files,
        times,
        dict(_materialize(spec.parameters, context)),
        str(context.output_dir),
        log=context.progress,
    )
    return _legacy_result(spec, context, result)


def run_ert3d_forward(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    import pandas as pd

    from PyHydroGeophysX.core.mesh_serialization import load_mesh_artifact
    from PyHydroGeophysX.forward.ert3d import run_ert3d_forward as simulate

    mesh_ref = spec.inputs.get("mesh")
    mesh_structure_ref = spec.inputs.get("mesh_structure")
    sensors_ref = spec.inputs.get("sensors")
    if (
        not isinstance(mesh_ref, ArtifactRef)
        or not isinstance(mesh_structure_ref, ArtifactRef)
        or not isinstance(sensors_ref, ArtifactRef)
    ):
        raise ValueError(
            "ert3d.forward requires mesh, mesh_structure, and sensors ArtifactRefs."
        )
    structure_path = context.resolve_artifact(mesh_structure_ref)
    mesh = context.load_object(
        mesh_ref,
        lambda path: load_mesh_artifact(path, structure_path),
    )
    sensors = context.load_object(sensors_ref, pd.read_csv)
    parameters = dict(_materialize(spec.parameters, context))
    parameters["seed"] = int(spec.seed)
    parameters["output_dir"] = str(context.output_dir)
    return _legacy_result(
        spec,
        context,
        simulate(mesh, sensors, log=context.progress, **parameters),
    )


def run_ert_single(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion

    source = spec.inputs.get("data")
    if not isinstance(source, ArtifactRef):
        raise ValueError("ert.single_inversion requires inputs.data as an ArtifactRef.")
    parameters = dict(spec.parameters)
    # The desktop workflow has already serialized its edited/QC-filtered
    # container in pygimli's native unified format.  Re-running the original
    # instrument parser here is both redundant and harmful in an isolated
    # process: the BERT parser imports pandas/pyarrow even though no tabular
    # conversion remains to do.  On Windows Store Python, arrow.dll has crashed
    # during that unnecessary second parse.  Raw public workflow inputs still
    # retain the caller-selected instrument parser.
    normalized_input = bool((source.metadata or {}).get("qc_filtered", False))
    instrument = None if normalized_input else parameters.get("instrument")
    result = run_ert_manager_inversion(
        context.resolve_artifact(source),
        context.output_dir,
        relative_error=float(parameters.get("relative_error", 0.03)),
        absolute_error=float(parameters.get("absolute_error", 0.0)),
        error_source=str(parameters.get("error_source", "file")),
        mesh_quality=float(parameters.get("mesh_quality", 34.0)),
        para_depth=float(parameters.get("para_depth", 0.0)),
        para_max_cell_size=float(parameters.get("para_max_cell_size", 0.0)),
        mesh_file=str(parameters.get("mesh_file", "") or ""),
        lam=float(parameters.get("lambda", 50.0)),
        max_iterations=int(parameters.get("max_iterations", 20)),
        plateau_tolerance=float(parameters.get("plateau_tolerance", 0.005)),
        max_total_iterations=int(parameters.get("max_total_iterations", 60)),
        engine=str(parameters.get("engine", "pyhydro")),
        geometric_factor_policy=str(parameters.get("geometric_factor_policy", "fix")),
        geometric_factor_tolerance=float(
            parameters.get("geometric_factor_tolerance", 0.05)),
        instrument=instrument,
        reject_outliers=bool(parameters.get("reject_outliers", False)),
        outlier_threshold=float(parameters.get("outlier_threshold", 3.0)),
        outlier_passes=int(parameters.get("outlier_passes", 2)),
        min_data_fraction=float(parameters.get("min_data_fraction", 0.5)),
        auto_lambda=bool(parameters.get("auto_lambda", False)),
        target_chi2=float(parameters.get("target_chi2", 1.0)),
        chi2_tolerance=float(parameters.get("chi2_tolerance", 0.2)),
        max_lambda_trials=int(parameters.get("max_lambda_trials", 6)),
        lambda_warm_start=bool(parameters.get("lambda_warm_start", True)),
        lambda_cold_retry_chi2=float(parameters.get("lambda_cold_retry_chi2", 15.0)),
        log=context.progress,
    )
    workflow_result = _legacy_result(spec, context, result)
    manager = workflow_result.objects.pop("mgr", None)
    if manager is not None:
        workflow_result.objects["manager"] = manager
    fixed_manager = workflow_result.objects.pop("fixed_mgr", None)
    if fixed_manager is not None:
        workflow_result.objects["fixed_manager"] = fixed_manager
    convergence = list(workflow_result.summary.get("convergence", []) or [])
    workflow_result.objects["convergence"] = convergence
    workflow_result.objects["fixed_convergence"] = list(
        workflow_result.summary.get("fixed_convergence", []) or []
    )
    return workflow_result


def run_em_inversion(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.workflows import em1d

    source = spec.inputs.get("data")
    if not isinstance(source, ArtifactRef):
        raise ValueError("em.inversion requires inputs.data as an ArtifactRef.")
    parameters = dict(spec.parameters)
    method = str(parameters.pop("method", "TDEM")).upper()
    moment = str(parameters.pop("moment", "HM"))
    sounding = int(parameters.pop("sounding", 0))
    data = context.load_object(
        source,
        lambda path: em1d.load_sounding(
            str(path), method, sounding=sounding, moment=moment
        ),
    )
    geometry = dict(parameters.pop("geometry", {}))
    if method == "FDEM":
        result = em1d.fdem_invert(data, geometry, parameters, log=context.progress)
    elif data.get("moments"):
        result = em1d.tdem_joint_invert(
            data, geometry, parameters, log=context.progress
        )
    else:
        result = em1d.tdem_invert(data, geometry, parameters, log=context.progress)
    return _legacy_result(spec, context, result)


def run_joint(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.data_processing.joint_io import load_joint_observations
    from PyHydroGeophysX.inversion.joint import run_joint_inversion
    from PyHydroGeophysX.inversion.joint_api import JointInversionRequest

    parameters = dict(_materialize(spec.parameters, context))
    serialized_data = spec.inputs.get("data")
    if not isinstance(serialized_data, Mapping):
        raise ValueError("joint_inversion.run requires inputs.data artifact mapping.")
    loaded_data: Dict[str, Any] = {}
    for method, reference in serialized_data.items():
        if not isinstance(reference, ArtifactRef):
            raise ValueError(f"Joint input {method!r} must be an ArtifactRef.")
        loaded_data[str(method)] = context.load_object(
            reference,
            lambda path, selected=str(method): load_joint_observations(selected, path),
        )
    run_baseline = bool(parameters.pop("run_baseline", True))
    request = JointInversionRequest(
        method_a=str(parameters.pop("method_a")),
        method_b=str(parameters.pop("method_b")),
        strategy=str(parameters.pop("strategy")),
        data=loaded_data,
        parameters=parameters,
        output_dir=context.output_dir,
        run_baseline=run_baseline,
    )
    return _legacy_result(
        spec,
        context,
        run_joint_inversion(
            request,
            progress=lambda record: context.progress(
                json.dumps(record, sort_keys=True, default=str)
            ),
        ),
    )


def run_gravmag_inversion(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.inversion.gravmag import invert_gravmag
    from .builtin import _load_array

    parameters = dict(spec.parameters)
    kind = str(parameters.pop("kind", "gravity"))
    parameters.setdefault("solver", "simpeg")
    parameters["out_dir"] = str(context.output_dir)
    parameters["random_seed"] = int(spec.seed)
    z_value = spec.inputs.get("z")
    if z_value is not None:
        parameters["z"] = _load_array(z_value, context, name="z")
    result = invert_gravmag(
        _load_array(spec.inputs.get("x"), context, name="x"),
        _load_array(spec.inputs.get("y"), context, name="y"),
        _load_array(spec.inputs.get("values"), context, name="values"),
        kind,
        **parameters,
        log=context.progress,
    )
    return _legacy_result(spec, context, result)


__all__ = [
    "run_em_inversion",
    "run_ert3d_forward",
    "run_ert_single",
    "run_ert_timelapse",
    "run_geo_hydrology",
    "run_gravmag_inversion",
    "run_hydro_geophysics",
    "run_joint",
    "run_mesh3d",
    "run_seismic3d",
]
