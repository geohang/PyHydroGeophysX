"""Adapters from serializable workflow contracts to canonical domain APIs.

One handler per registered workflow: it turns a recipe's inputs into the objects
and paths a domain function takes, calls it, and files what comes back as a
:class:`WorkflowRunResult`. Workflow IDs are registry identifiers; they do not
imply that a same-named function exists in the scientific module.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import fields, is_dataclass
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

import numpy as np

from PyHydroGeophysX.data_processing import run_inputs

from .models import ArtifactRef, RunContext, WorkflowRunResult, WorkflowSpec


@contextmanager
def _input_directory(
    bundle: Optional[str], fallback: Mapping[str, str]
) -> Iterator[Optional[str]]:
    """Yield a directory holding the run's inputs under their original names.

    A run stores these inputs as one compressed bundle rather than as the loose
    files it used to copy, because a time-lapse sequence or a hydrology array
    set is several files that only ever get read together. The readers below are
    PyGIMLi's loaders and ``np.load``, which want a real path, so the bundle is
    expanded for the length of the run and thrown away afterwards; only the
    compressed copy is what the Project keeps.

    ``fallback`` is the older ``{name: path}`` mapping. Runs recorded before the
    bundle existed still carry it, and their copied files are already a
    directory, so those are used where they are.
    """
    if bundle:
        with tempfile.TemporaryDirectory(prefix="phgx_inputs_") as scratch:
            yield str(run_inputs.expand_file_bundle(bundle, scratch))
        return
    if not fallback:
        yield None
        return
    parents = {str(Path(path).parent) for path in fallback.values()}
    if len(parents) != 1:
        raise ValueError("Input artifacts must share one directory.")
    yield parents.pop()


def _materialize(value: Any, context: RunContext) -> Any:
    if isinstance(value, ArtifactRef):
        return str(context.resolve_artifact(value))
    if isinstance(value, Mapping):
        return {str(key): _materialize(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item, context) for item in value]
    return value


def _load_array(value: Any, context: RunContext, *, name: str) -> np.ndarray:
    if isinstance(value, ArtifactRef):
        # One .npz holds several arrays, so the member read is part of the key:
        # keyed by the artifact alone, x, y and values taken from one archive
        # all came back as whichever of them was read first.
        cache_key = f"{context.cache_key(value)}#{value.metadata.get('array_key') or name}"
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


def _keywords_for(function: Callable[..., Any], parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """The recipe parameters ``function`` takes as keywords, typed as its defaults are.

    A parameter the recipe leaves out, or sets to None, is not passed, so the
    function's own default applies and cannot drift from a copy kept here. A key
    the function does not take - page bookkeeping such as ``receiver_spacing`` -
    stays in the recipe without reaching it. A value is converted to the type of
    the default it replaces, as these handlers used to do one keyword at a time,
    so a recipe written with ``20.0`` iterations still runs, and a JSON list
    arrives as the tuple a pair of bounds defaults to. Arguments without a
    default (the data and output paths) and ``log`` are the handler's to pass.
    """
    keywords: Dict[str, Any] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if (name == "log" or parameter.default is inspect.Parameter.empty
                or parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                          inspect.Parameter.KEYWORD_ONLY)):
            continue
        value = parameters.get(name)
        if value is None:
            continue
        default = parameter.default
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, (int, float, str)) and not isinstance(value, bool):
            value = type(default)(value)
        elif isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)            # a pair such as bounds, read back from JSON
        keywords[name] = value
    return keywords


def _json_value(value: Any) -> tuple[bool, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        # Not JSON: in the summary it failed the whole run as it finished (an
        # undefined chi2, an empty range). As an object it comes back as is.
        return False, None
    if value is None or isinstance(value, (str, int, float, bool)):
        return True, value
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
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(_fields(value))
    return False, None


def _fields(value: Any) -> Dict[str, Any]:
    """A dataclass's fields as they are. ``dataclasses.asdict`` deep-copies
    each one, and a PyGIMLi mesh cannot be copied that way: a joint ERT + SRT
    result, which holds its mesh, failed at the end of the run."""
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _legacy_result(
    spec: WorkflowSpec,
    context: RunContext,
    result: Any,
) -> WorkflowRunResult:
    if is_dataclass(result) and not isinstance(result, type):
        raw = _fields(result)
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
    domain_context = dict(inputs.get("context") or inputs)
    domain_context.setdefault("output_dir", str(context.output_dir))
    methods = list(parameters.pop("methods", inputs.get("methods", [])))
    point1 = parameters.pop("point1", inputs.get("point1"))
    point2 = parameters.pop("point2", inputs.get("point2"))
    with _input_directory(
        inputs.get("hydro_bundle"), dict(inputs.get("hydro_files") or {})
    ) as data_dir:
        if data_dir:
            parameters["hydro_data_dir"] = data_dir
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
    domain_context = dict(inputs.get("context") or inputs)
    domain_context.setdefault("output_dir", str(context.output_dir))
    with _input_directory(
        inputs.get("model_bundle"), dict(inputs.get("model_files") or {})
    ) as data_dir:
        if data_dir:
            parameters["model_data_dir"] = data_dir
        result = run_ert_to_wc(domain_context, parameters, log=context.progress)
    return _legacy_result(spec, context, result)


def run_seismic3d(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.Geophy_modular.structure_integration import build_3d_model

    inputs = _materialize(spec.inputs, context)
    parameters = dict(_materialize(spec.parameters, context))
    parameters["output_dir"] = str(context.output_dir)
    lines = list(inputs.get("lines") or [])
    domain_context = dict(inputs.get("context") or inputs)
    domain_context.setdefault("output_dir", str(context.output_dir))
    with _input_directory(inputs.get("line_bundle"), {}) as scratch:
        if scratch:
            # The page stored bundle member names in place of the paths, so
            # point each line back at the file the bundle just wrote.
            lines = [
                {
                    **line,
                    **{
                        field: str(Path(scratch) / str(line[field]))
                        for field in ("mesh", "velocity")
                        if field in line
                    },
                }
                for line in lines
            ]
        parameters["lines"] = lines
        result = build_3d_model(domain_context, parameters, log=context.progress)
    return _legacy_result(spec, context, result)


def run_mesh3d(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.core.mesh_3d import generate_mesh, save_outputs

    config = dict(_materialize(spec.parameters, context))
    inputs = _materialize(spec.inputs, context)
    if inputs.get("topography_points"):
        config["topography_points"] = np.load(inputs["topography_points"])
    if inputs.get("e4d_config"):
        config["e4d_config_path"] = str(inputs["e4d_config"])
    config["output_dir"] = str(context.output_dir)
    output_formats = list(config.pop("output_formats", []))
    output_name = str(config.pop("output_name", "mesh3d"))
    # The E4D engine writes its .cfg, .poly and mesh files under this name.
    config.setdefault("e4d_basename", output_name)
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
    # The E4D engine has already written what E4D runs on beside the mesh.
    for key, path in dict(result.get("e4d_files") or {}).items():
        result.setdefault("outputs", {})[f"e4d_{key}"] = path
    return _legacy_result(spec, context, result)


def run_ert_timelapse(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.inversion.time_lapse import run_timelapse_ert

    inputs = _materialize(spec.inputs, context)
    bundle = inputs.get("data_bundle")
    parameters = dict(_materialize(spec.parameters, context))
    with _input_directory(bundle, {}) as scratch:
        if scratch:
            # Names were written in sequence order, and the order is the whole
            # point of a time-lapse: step n is what step n-1 is compared to.
            files = [
                str(Path(scratch) / name)
                for name in run_inputs.bundle_file_names(bundle)
            ]
        else:
            files = list(inputs.get("data_files") or [])
        times = list(inputs.get("measurement_times") or range(len(files)))
        # The bundle staged the files under generated names, so the acquisition
        # dates survive only in the labels the caller recorded alongside them.
        labels = [str(lbl) for lbl in (inputs.get("time_labels") or [])]
        # Absolute acquisition times, recorded alongside the labels for the same
        # reason: the bundle renamed the files, so this is the only place they
        # survive. They are what the run reports intervals from.
        stamps = [str(value) for value in (inputs.get("timestamps") or [])]
        result = run_timelapse_ert(
            files, times, parameters, str(context.output_dir), log=context.progress,
            time_labels=labels or None,
            time_unit=str(inputs.get("time_unit") or ""),
            timestamps=stamps or None,
            # The electrode positions every survey is placed on, when the run
            # was given an electrode file rather than each file's own header.
            electrode_file=inputs.get("electrodes") or None,
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
    # The page records the regularization weight as "lambda", which cannot be a
    # Python keyword; it is the inversion's ``lam``.
    if "lambda" in parameters:
        parameters["lam"] = parameters.pop("lambda")
    options = _keywords_for(run_ert_manager_inversion, parameters)
    # The desktop workflow has already serialized its edited/QC-filtered
    # container in pygimli's native unified format.  Re-running the original
    # instrument parser here is both redundant and harmful in an isolated
    # process: the BERT parser imports pandas/pyarrow even though no tabular
    # conversion remains to do.  On Windows Store Python, arrow.dll has crashed
    # during that unnecessary second parse.  Raw public workflow inputs still
    # retain the caller-selected instrument parser.
    if bool((source.metadata or {}).get("qc_filtered", False)):
        options.pop("instrument", None)
    result = run_ert_manager_inversion(
        context.resolve_artifact(source),
        context.output_dir,
        log=context.progress,
        **options,
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
    options = _keywords_for(run_srt_manager_inversion, spec.parameters)
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


def run_em_inversion(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.data_processing.em1d import (
        load_sounding,
        save_inversion,
        sounding_options,
    )
    from PyHydroGeophysX.inversion.em1d import fdem_invert, tdem_invert, tdem_joint_invert

    source = spec.inputs.get("data")
    if not isinstance(source, ArtifactRef):
        raise ValueError("em.inversion requires inputs.data as an ArtifactRef.")
    parameters = dict(spec.parameters)
    method = str(parameters.pop("method", "TDEM")).upper()
    moment = str(parameters.pop("moment", "HM"))
    sounding = int(parameters.pop("sounding", 0))
    geometry = dict(parameters.pop("geometry", {}))
    data = context.load_object(
        source,
        lambda path: load_sounding(str(path), method, sounding=sounding, moment=moment,
                                   **sounding_options(geometry)),
    )
    if method == "FDEM":
        result = fdem_invert(data, geometry, parameters, log=context.progress)
    elif data.get("moments"):
        result = tdem_joint_invert(
            data, geometry, parameters, log=context.progress
        )
    else:
        result = tdem_invert(data, geometry, parameters, log=context.progress)
    if (result.get("robust") or {}).get("enabled"):
        # The persisted single-sounding path needs the same weight audit as lines.
        result["data_paths"] = save_inversion(result, context.output_dir)
        result["metrics"] = {"chi2_effective": result["chi2_effective"],
                             "downweighted_gates": result["robust"]["downweighted"]}
    return _legacy_result(spec, context, result)


def run_em_line_inversion(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    """A line of soundings inverted together (:func:`~PyHydroGeophysX.inversion.em1d_line.invert_line`).

    ``inputs.data`` is the stored soundings; ``inputs.positions`` and
    ``inputs.heights``, when the survey has them, the along-line distance and
    the sensor height of each, as arrays in an ``.npz``. The parameters are the
    method, ``geometry``, ``inversion`` and the rest of ``invert_line``'s
    keywords.
    """
    from PyHydroGeophysX.inversion.em1d_line import invert_line

    source = spec.inputs.get("data")
    if not isinstance(source, ArtifactRef):
        raise ValueError("em.line_inversion requires inputs.data as an ArtifactRef.")
    parameters = dict(_materialize(spec.parameters, context))
    method = str(parameters.pop("method", "TDEM")).upper()
    geometry = dict(parameters.pop("geometry", {}))
    inversion = dict(parameters.pop("inversion", {}))
    for key in ("positions", "heights"):
        if spec.inputs.get(key) is not None:
            parameters[key] = _load_array(spec.inputs[key], context, name=key)
    result = invert_line(
        str(context.resolve_artifact(source)), method, geometry, inversion,
        out_dir=context.output_dir, log=context.progress, **parameters,
    )
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


def run_gravmag_process(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.data_processing.gravmag import (
        extract_profile,
        qc_products,
        save_grid,
    )

    x = _load_array(spec.inputs.get("x"), context, name="x").ravel()
    y = _load_array(spec.inputs.get("y"), context, name="y").ravel()
    values = _load_array(spec.inputs.get("values"), context, name="values").ravel()
    if not (x.size == y.size == values.size):
        raise ValueError("gravmag.process requires x, y, and values of equal length.")
    parameters = dict(spec.parameters)
    qc = qc_products(x, y, values, **_keywords_for(qc_products, parameters))
    artifacts = []
    for label, grid in qc["grids"].items():
        slug = label.lower()
        paths = save_grid(
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
        profile = extract_profile(
            qc["grids"][str(profile_config.get("field", "Residual"))],
            profile_config["p1"],
            profile_config["p2"],
            **_keywords_for(extract_profile, profile_config),
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
    from PyHydroGeophysX.forward.gravmag import forward_bodies

    x = _load_array(spec.inputs.get("x"), context, name="x").ravel()
    y = _load_array(spec.inputs.get("y"), context, name="y").ravel()
    kind = str(spec.parameters.get("kind", "gravity"))
    bodies = list(spec.parameters.get("bodies") or [])
    field = dict(spec.parameters.get("field") or {})
    response = forward_bodies(
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


def run_gravmag_inversion(spec: WorkflowSpec, context: RunContext) -> WorkflowRunResult:
    from PyHydroGeophysX.inversion.gravmag import invert_gravmag

    parameters = dict(spec.parameters)
    kind = str(parameters.pop("kind", "gravity"))
    z_value = spec.inputs.get("z")
    if z_value is not None:
        parameters["z"] = _load_array(z_value, context, name="z")
    options = _keywords_for(invert_gravmag, parameters)
    options.update(out_dir=str(context.output_dir), random_seed=int(spec.seed))
    result = invert_gravmag(
        _load_array(spec.inputs.get("x"), context, name="x"),
        _load_array(spec.inputs.get("y"), context, name="y"),
        _load_array(spec.inputs.get("values"), context, name="values"),
        kind,
        **options,
        log=context.progress,
    )
    return _legacy_result(spec, context, result)


__all__ = [
    "run_em_inversion",
    "run_em_line_inversion",
    "run_ert3d_forward",
    "run_ert_single",
    "run_ert_timelapse",
    "run_geo_hydrology",
    "run_gravmag_forward_bodies",
    "run_gravmag_inversion",
    "run_gravmag_process",
    "run_hydro_geophysics",
    "run_joint",
    "run_mesh3d",
    "run_seismic3d",
    "run_srt_inversion",
]
