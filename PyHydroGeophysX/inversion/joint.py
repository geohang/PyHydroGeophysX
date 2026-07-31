"""Capability registry and public dispatch API for multi-method inversion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from PyHydroGeophysX.inversion.joint_api import (
    METHODS,
    JointInversionRequest,
    JointInversionResult,
    JointPairCapability,
    _method_name,
    _json_ready,
    get_joint_capabilities,
    get_joint_capability,
    normalize_joint_pair,
    pair_joint_soundings,
    split_joint_soundings,
    validate_profile_interface,
)


def _data_for(request: JointInversionRequest, method: str) -> Any:
    for key, value in request.data.items():
        try:
            if _method_name(key) == method:
                return value
        except ValueError:
            continue
    raise ValueError(f"Joint inversion requires observed {method} data.")


def _sensor_extent(data: Any) -> Optional[Tuple[float, float]]:
    try:
        positions = data.sensorPositions()
    except Exception:
        return None
    xs = np.asarray([float(position.x()) for position in positions], dtype=float)
    if xs.size == 0 or not np.all(np.isfinite(xs)):
        return None
    return float(xs.min()), float(xs.max())


def _offset_container(data: Any, x_offset: float, z_offset: float) -> Any:
    if not x_offset and not z_offset:
        return data
    import pygimli as pg
    copied = pg.DataContainer(data)
    for index, position in enumerate(copied.sensorPositions()):
        copied.setSensorPosition(index, pg.Pos(position.x() + x_offset, position.y() + z_offset, position.z()))
    return copied


def _write_result_files(request: JointInversionRequest, result: JointInversionResult) -> None:
    out = Path(request.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    request_payload = {
        "method_a": request.method_a,
        "method_b": request.method_b,
        "strategy": request.strategy,
        "parameters": _json_ready({
            key: value for key, value in request.parameters.items()
            if key != "progress_callback"
        }),
        "run_baseline": request.run_baseline,
    }
    request_path = out / "request.json"
    request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
    result.artifacts["request"] = str(request_path)

    array_payload: Dict[str, np.ndarray] = {}
    for group, values in (("model", result.models), ("predicted", result.predicted), ("coverage", result.coverage)):
        for name, value in values.items():
            try:
                array = np.asarray(value)
            except Exception:
                continue
            if array.dtype != object:
                array_payload[f"{group}_{name.lower()}"] = array
    if array_payload:
        arrays_path = out / "joint_arrays.npz"
        np.savez_compressed(arrays_path, **array_payload)
        result.artifacts["arrays"] = str(arrays_path)
        for name, array in array_payload.items():
            array_path = out / f"{name}.npy"
            np.save(array_path, array)
            result.artifacts[f"{name}_npy"] = str(array_path)

    if result.history:
        fields = sorted({key for row in result.history for key in row})
        history_path = out / "iteration_history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in result.history:
                writer.writerow({key: _json_ready(row.get(key)) for key in fields})
        result.artifacts["history"] = str(history_path)

    if result.baseline:
        baseline_arrays: Dict[str, np.ndarray] = {}
        baseline_metrics: Dict[str, Any] = {}

        def collect_baseline(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    collect_baseline(f"{prefix}_{key}" if prefix else str(key), item)
                return
            try:
                array = np.asarray(value)
            except Exception:
                baseline_metrics[prefix] = _json_ready(value)
                return
            if array.dtype != object and array.ndim > 0:
                baseline_arrays[prefix.lower()] = array
            elif array.size == 1:
                baseline_metrics[prefix] = _json_ready(array.item())

        collect_baseline("", result.baseline)
        if baseline_arrays:
            baseline_arrays_path = out / "baseline_arrays.npz"
            np.savez_compressed(baseline_arrays_path, **baseline_arrays)
            result.artifacts["baseline_arrays"] = str(baseline_arrays_path)
        baseline_summary_path = out / "baseline_summary.json"
        baseline_summary_path.write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
        result.artifacts["baseline_summary"] = str(baseline_summary_path)

    summary_path = out / "summary.json"
    result.artifacts["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")


def _ert_srt_baseline(inv: Any, joint: Any, parameters: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    baseline: Dict[str, Any] = {}
    warnings: List[str] = []
    max_iterations = int(parameters.get("baseline_max_iterations", parameters.get("max_iterations", 20)))
    try:
        from pygimli.physics import ert
        manager = ert.ERTManager(inv.ert_data)
        model = manager.invert(
            data=inv.ert_data,
            mesh=joint.mesh,
            lam=float(parameters.get("lambda_ert", 10.0)),
            maxIter=max_iterations,
            verbose=False,
        )
        baseline["ERT"] = {
            "model": np.asarray(model, dtype=float),
            "chi2": float(manager.inv.chi2()),
        }
    except Exception as exc:
        warnings.append(f"ERT independent baseline failed: {exc}")
    try:
        from pygimli.physics import TravelTimeManager
        manager = TravelTimeManager(inv.srt_data)
        model = manager.invert(
            data=inv.srt_data,
            mesh=joint.mesh,
            lam=float(parameters.get("lambda_srt", 10.0)),
            maxIter=max_iterations,
            verbose=False,
        )
        baseline["SRT"] = {
            "model": np.asarray(model, dtype=float),
            "chi2": float(manager.inv.chi2()),
        }
    except Exception as exc:
        warnings.append(f"SRT independent baseline failed: {exc}")
    return baseline, warnings


def _run_ert_srt(request: JointInversionRequest) -> JointInversionResult:
    ert_data = _data_for(request, "ERT")
    srt_data = _data_for(request, "SRT")
    parameters = dict(request.parameters)
    x_ert = float(parameters.pop("ert_x_offset", 0.0))
    z_ert = float(parameters.pop("ert_z_offset", 0.0))
    x_srt = float(parameters.pop("srt_x_offset", 0.0))
    z_srt = float(parameters.pop("srt_z_offset", 0.0))

    from PyHydroGeophysX.inversion.joint_ert_srt import JointERTSRTInversion
    # Paths are loaded by the existing backend before offsets are applied.
    inv = JointERTSRTInversion(ert_data=ert_data, srt_data=srt_data, **parameters)
    inv.ert_data = _offset_container(inv.ert_data, x_ert, z_ert)
    inv.srt_data = _offset_container(inv.srt_data, x_srt, z_srt)
    ert_extent = _sensor_extent(inv.ert_data)
    srt_extent = _sensor_extent(inv.srt_data)
    if ert_extent and srt_extent and min(ert_extent[1], srt_extent[1]) <= max(ert_extent[0], srt_extent[0]):
        raise ValueError(
            "ERT and SRT sensor ranges do not overlap. Check profile coordinates or set explicit offsets."
        )

    if request.strategy == "sequential_structure":
        from PyHydroGeophysX.agents.structure_constraint_agent import StructureConstraintAgent
        progress_callback = parameters.get("progress_callback")
        if callable(progress_callback):
            progress_callback({"stage": "seismic_structure", "message": "Running SRT and extracting interface"})
        ert_params = dict(parameters.get("ert_params", {}))
        ert_params.setdefault("max_iterations", int(parameters.get("max_iterations", 20)))
        ert_params.setdefault("lambda", float(parameters.get("lambda_ert", 10.0)))
        interface_coords = parameters.get("interface_coords")
        if interface_coords is not None:
            ert_positions = inv.ert_data.sensorPositions()
            interface_coords = validate_profile_interface(
                interface_coords,
                [float(position.x()) for position in ert_positions],
                [float(position.y()) for position in ert_positions],
            )
        sequential = StructureConstraintAgent().execute({
            "ert_data": inv.ert_data,
            "seismic_data": inv.srt_data,
            "interface_coords": interface_coords,
            "velocity_threshold": parameters.get("velocity_threshold", 1000.0),
            "seismic_params": parameters.get("seismic_params", {}),
            "inversion_params": ert_params,
            "mesh_params": parameters.get("mesh_params", {}),
            "output_dir": str(request.output_dir),
        })
        seismic_results = sequential.get("seismic_results") or {}
        models = {"ERT": sequential["resistivity_model"]}
        coverage = {"ERT": sequential.get("coverage")}
        if seismic_results.get("velocity_model") is not None:
            models["SRT"] = seismic_results["velocity_model"]
            coverage["SRT"] = seismic_results.get("coverage")
        predicted = {}
        if sequential.get("predicted") is not None:
            predicted["ERT"] = sequential["predicted"]
        if seismic_results.get("predicted") is not None:
            predicted["SRT"] = seismic_results["predicted"]
        chi2 = {}
        sequential_chi2 = float(sequential.get("chi2", float("nan")))
        if np.isfinite(sequential_chi2):
            chi2["ERT"] = sequential_chi2
        seismic_chi2 = float(seismic_results.get("chi2", float("nan")))
        if np.isfinite(seismic_chi2):
            chi2["SRT"] = seismic_chi2
        result = JointInversionResult(
            methods=("ERT", "SRT"),
            strategy=request.strategy,
            models=models,
            predicted=predicted,
            coverage=coverage,
            chi2=chi2,
            meta={
                "interface_coords": sequential.get("interface_coords"),
                "mesh": sequential.get("mesh"),
                "cell_markers": sequential.get("cell_markers"),
                "statistics": sequential.get("statistics"),
                "output_dir": str(request.output_dir),
                "backend": "pygimli",
                "requested_backend": "pygimli",
                "data_counts": {
                    "ERT": int(inv.ert_data.size()),
                    "SRT": int(inv.srt_data.size()),
                },
            },
        )
        if request.run_baseline:
            from types import SimpleNamespace
            baseline_mesh = sequential.get("mesh")
            if baseline_mesh is None:
                baseline_mesh = sequential.get("constrained_mesh")
            result.baseline, baseline_warnings = _ert_srt_baseline(
                inv, SimpleNamespace(mesh=baseline_mesh), parameters
            )
            result.warnings.extend(baseline_warnings)
        if interface_coords is not None:
            result.warnings.append(
                "Reused an existing SRT-derived interface; no new SRT fit was computed for the sequential run."
            )
        if callable(progress_callback):
            progress_callback({"stage": "complete", "message": "Sequential constrained inversion complete"})
        sequential_out = Path(request.output_dir)
        for artifact_name, filename in (
            ("resistivity", "resistivity_model.npy"),
            ("coverage", "coverage.npy"),
            ("cell_markers", "cell_markers.npy"),
            ("mesh", "constrained_mesh.bms"),
        ):
            artifact_path = sequential_out / filename
            if artifact_path.exists():
                result.artifacts[artifact_name] = str(artifact_path)
        try:
            mesh = sequential.get("mesh")
            if mesh is not None:
                mesh["constrained_ert_resistivity"] = np.asarray(
                    sequential["resistivity_model"], dtype=float
                )
                vtk_path = sequential_out / "constrained_ert_model.vtk"
                mesh.exportVTK(str(vtk_path))
                result.artifacts["vtk"] = str(vtk_path)
        except Exception as exc:
            result.warnings.append(f"Could not export constrained ERT VTK: {exc}")
        _write_result_files(request, result)
        return result

    if request.strategy == "cross_gradient_geostatistical":
        inv.parameters.update({
            "regularization_mode": "smoothness",
            "cross_gradient_mode": "spatial",
            "cross_gradient_source": "geostat",
            "lambda_cg_ert": float(parameters.get("lambda_cg_ert", 5000.0)),
            "lambda_cg_srt": float(parameters.get("lambda_cg_srt", 5000.0)),
        })
    joint = inv.run()
    joint_meta = dict(joint.meta)
    if isinstance(joint_meta.get("parameters"), Mapping):
        joint_meta["parameters"] = {
            key: value for key, value in joint_meta["parameters"].items()
            if key != "progress_callback"
        }
    result = JointInversionResult(
        methods=("ERT", "SRT"),
        strategy=request.strategy,
        models={"ERT": joint.ert_resistivity, "SRT": joint.srt_velocity},
        predicted={"ERT": joint.ert_predicted, "SRT": joint.srt_predicted},
        coverage={"ERT": joint.ert_coverage, "SRT": joint.srt_coverage},
        chi2={"ERT": float(joint.chi2_ert), "SRT": float(joint.chi2_srt)},
        history=list(joint.iteration_history),
        meta={
            "mesh": joint.mesh,
            "backend": "pygimli",
            "requested_backend": "pygimli",
            "data_counts": {
                "ERT": int(inv.ert_data.size()),
                "SRT": int(inv.srt_data.size()),
            },
            **joint_meta,
        },
    )
    if request.run_baseline:
        result.baseline, baseline_warnings = _ert_srt_baseline(inv, joint, parameters)
        result.warnings.extend(baseline_warnings)
    out = Path(request.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        mesh_path = out / "joint_mesh.bms"
        joint.mesh.save(str(mesh_path))
        result.artifacts["mesh"] = str(mesh_path)
        joint.mesh["joint_ert_resistivity"] = np.asarray(joint.ert_resistivity, dtype=float)
        joint.mesh["joint_srt_velocity"] = np.asarray(joint.srt_velocity, dtype=float)
        vtk_path = out / "joint_models.vtk"
        joint.mesh.exportVTK(str(vtk_path))
        result.artifacts["vtk"] = str(vtk_path)
    except Exception as exc:
        result.warnings.append(f"Could not save joint mesh: {exc}")
    _write_result_files(request, result)
    return result


def _run_fdem_tdem(request: JointInversionRequest) -> JointInversionResult:
    from PyHydroGeophysX.inversion.joint_fdem_tdem import JointFDEMTDEMInversion

    f_value = _data_for(request, "FDEM")
    t_value = _data_for(request, "TDEM")
    f_soundings, _f_coordinates = split_joint_soundings(f_value)
    t_soundings, _t_coordinates = split_joint_soundings(t_value)
    pairs = pair_joint_soundings(f_value, t_value, request.parameters)
    models: List[np.ndarray] = []
    histories: List[Dict[str, Any]] = []
    predicted_f: List[np.ndarray] = []
    predicted_t: List[np.ndarray] = []
    coverage_f: List[np.ndarray] = []
    coverage_t: List[np.ndarray] = []
    chi2_f: List[float] = []
    chi2_t: List[float] = []
    baselines: Dict[str, Any] = {}
    manifest: List[Dict[str, Any]] = []
    backend_records: List[Dict[str, Any]] = []
    fallback_warnings: List[str] = []
    thicknesses: Optional[np.ndarray] = None
    parameters = dict(request.parameters)
    f_geometry = parameters.pop("fdem_geometry", {})
    t_geometry = parameters.pop("tdem_geometry", {})
    parameters["run_baseline"] = request.run_baseline
    for output_index, (f_index, t_index, pairing_mode) in enumerate(pairs):
        inversion = JointFDEMTDEMInversion(
            fdem_data=f_soundings[f_index],
            tdem_data=t_soundings[t_index],
            fdem_geometry=f_geometry,
            tdem_geometry=t_geometry,
            **parameters,
        )
        joint = inversion.run()
        thicknesses = joint.thicknesses
        models.append(joint.resistivity)
        predicted_f.append(joint.predicted_fdem)
        predicted_t.append(joint.predicted_tdem)
        coverage_f.append(np.asarray(joint.coverage_fdem, dtype=float))
        coverage_t.append(np.asarray(joint.coverage_tdem, dtype=float))
        chi2_f.append(joint.chi2_fdem)
        chi2_t.append(joint.chi2_tdem)
        for row in joint.convergence:
            histories.append({"sounding": output_index + 1, **row})
        if joint.baseline:
            baselines[str(output_index + 1)] = joint.baseline
        backend = str(joint.meta.get("backend", "unknown"))
        backend_records.append({
            "sounding": output_index + 1,
            "backend": backend,
            "version": str(joint.meta.get("backend_version", "")),
        })
        fallback_reason = str(joint.meta.get("fallback_reason", ""))
        if fallback_reason:
            fallback_warnings.append(
                f"Sounding {output_index + 1} used SciPy fallback: {fallback_reason}"
            )
        manifest.append({
            "output_sounding": output_index + 1,
            "fdem_index": f_index,
            "tdem_index": t_index,
            "mode": pairing_mode,
            "backend": backend,
            "backend_version": str(joint.meta.get("backend_version", "")),
        })
        progress_callback = parameters.get("progress_callback")
        if callable(progress_callback):
            progress_callback({
                "sounding": output_index + 1,
                "soundings_total": len(pairs),
                "chi2_fdem": joint.chi2_fdem,
                "chi2_tdem": joint.chi2_tdem,
            })
    model_array = np.asarray(models, dtype=float)
    resolved_backends = sorted({item["backend"] for item in backend_records})
    resolved_backend = resolved_backends[0] if len(resolved_backends) == 1 else "mixed"
    backend_versions = sorted({item["version"] for item in backend_records if item["version"]})
    result = JointInversionResult(
        methods=("FDEM", "TDEM"),
        strategy=request.strategy,
        models={"resistivity": model_array[0] if len(model_array) == 1 else model_array},
        predicted={
            "FDEM": predicted_f[0] if len(predicted_f) == 1 else np.asarray(predicted_f),
            "TDEM": predicted_t[0] if len(predicted_t) == 1 else np.asarray(predicted_t),
        },
        coverage={
            "FDEM": coverage_f[0] if len(coverage_f) == 1 else np.asarray(coverage_f),
            "TDEM": coverage_t[0] if len(coverage_t) == 1 else np.asarray(coverage_t),
        },
        chi2={"FDEM": float(np.mean(chi2_f)), "TDEM": float(np.mean(chi2_t))},
        history=histories,
        baseline=baselines,
        meta={
            "thicknesses": thicknesses,
            "pairing_manifest": manifest,
            "backend": resolved_backend,
            "backend_version": ", ".join(backend_versions),
            "requested_backend": str(parameters.get("backend", "auto")),
            "station_backends": backend_records,
            "n_soundings": len(pairs),
            "data_counts": {
                "FDEM": int(sum(
                    2 * np.asarray(f_soundings[index]["frequencies"]).size
                    for index, _t_index, _mode in pairs
                )),
                "TDEM": int(sum(
                    np.asarray(t_soundings[index]["times"]).size
                    for _f_index, index, _mode in pairs
                )),
            },
        },
    )
    result.warnings.extend(fallback_warnings)
    if len(pairs) < len(f_soundings) or len(pairs) < len(t_soundings):
        result.warnings.append(
            f"Used {len(pairs)} matched sounding pair(s); "
            f"{len(f_soundings) - len(pairs)} FDEM and "
            f"{len(t_soundings) - len(pairs)} TDEM sounding(s) were unmatched."
        )
    out = Path(request.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairing_path = out / "pairing_manifest.csv"
    with pairing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "output_sounding", "fdem_index", "tdem_index", "mode",
                "backend", "backend_version",
            ),
        )
        writer.writeheader()
        writer.writerows(manifest)
    result.artifacts["pairing_manifest"] = str(pairing_path)
    _write_result_files(request, result)
    return result


def _run_gravity_magnetics(request: JointInversionRequest) -> JointInversionResult:
    """Run the native SimPEG gravity--magnetics cross-gradient adapter."""
    from PyHydroGeophysX.inversion.joint_gravity_magnetics import (
        JointGravityMagneticsInversion,
    )

    parameters = dict(request.parameters)
    parameters["run_baseline"] = request.run_baseline
    parameters["output_dir"] = str(request.output_dir)
    joint = JointGravityMagneticsInversion(
        gravity_data=_data_for(request, "Gravity"),
        magnetics_data=_data_for(request, "Magnetics"),
        **parameters,
    ).run()
    shape = tuple(int(value) for value in joint.model_shape)
    density3d = np.asarray(joint.density, dtype=float).reshape(shape, order="F")
    susceptibility3d = np.asarray(joint.susceptibility, dtype=float).reshape(shape, order="F")
    result = JointInversionResult(
        methods=("Gravity", "Magnetics"),
        strategy=request.strategy,
        models={"Gravity": joint.density, "Magnetics": joint.susceptibility},
        predicted={
            "Gravity": joint.predicted_gravity,
            "Magnetics": joint.predicted_magnetics,
        },
        coverage={
            "Gravity": joint.coverage_gravity,
            "Magnetics": joint.coverage_magnetics,
            "cross_gradient": joint.cross_gradient,
        },
        chi2={
            "Gravity": float(joint.chi2_gravity),
            "Magnetics": float(joint.chi2_magnetics),
        },
        history=list(joint.convergence),
        baseline=dict(joint.baseline),
        meta={
            **joint.meta,
            "edges": joint.edges,
            "model_shape": shape,
            "requested_backend": "simpeg",
        },
    )
    result.warnings.extend(joint.meta.get("baseline_warnings", []))
    out = Path(request.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid_path = out / "joint_potential_field_models.npz"
    np.savez_compressed(
        grid_path,
        ex=joint.edges[0], ey=joint.edges[1], ez=joint.edges[2],
        density=density3d, susceptibility=susceptibility3d,
        normalized_cross_gradient=joint.cross_gradient.reshape(shape, order="F"),
    )
    result.artifacts["model_grid"] = str(grid_path)
    try:
        import pyvista as pv
        grid = pv.RectilinearGrid(*joint.edges)
        grid.cell_data["density_g_cc"] = density3d.flatten(order="F")
        grid.cell_data["susceptibility_SI"] = susceptibility3d.flatten(order="F")
        grid.cell_data["normalized_cross_gradient"] = joint.cross_gradient
        vtk_path = out / "joint_potential_field_models.vtr"
        grid.save(str(vtk_path))
        result.artifacts["vtk"] = str(vtk_path)
    except Exception as exc:  # noqa: BLE001 - VTK support is optional
        result.warnings.append(f"Could not export Gravity/Magnetics VTK: {exc}")
    _write_result_files(request, result)
    return result


class _FunctionJointAdapter:
    """Small adapter that binds one capability to its dependency-lazy runner."""

    def __init__(
        self,
        capability: JointPairCapability,
        runner: Callable[[JointInversionRequest], JointInversionResult],
    ) -> None:
        self.capability = capability
        self._runner = runner

    def run(self, request: JointInversionRequest) -> JointInversionResult:
        return self._runner(request)


_ADAPTERS = {
    "ert_srt": _FunctionJointAdapter(get_joint_capability("ERT", "SRT"), _run_ert_srt),
    "fdem_tdem": _FunctionJointAdapter(get_joint_capability("FDEM", "TDEM"), _run_fdem_tdem),
    "gravity_magnetics": _FunctionJointAdapter(
        get_joint_capability("Gravity", "Magnetics"), _run_gravity_magnetics
    ),
}


def run_joint_inversion(
    request: Union[JointInversionRequest, Mapping[str, Any]],
    progress: Optional[Any] = None,
) -> JointInversionResult:
    """Validate and execute a registered joint or cooperative inversion."""
    if isinstance(request, Mapping):
        request = JointInversionRequest(**dict(request))
    if not isinstance(request, JointInversionRequest):
        raise TypeError("request must be JointInversionRequest or a compatible mapping.")
    if progress is not None:
        request = JointInversionRequest(
            method_a=request.method_a,
            method_b=request.method_b,
            strategy=request.strategy,
            data=request.data,
            parameters={**request.parameters, "progress_callback": progress},
            output_dir=request.output_dir,
            run_baseline=request.run_baseline,
        )
    capability = get_joint_capability(request.method_a, request.method_b)
    if not capability.implemented:
        raise NotImplementedError(
            f"Joint inversion for {capability.methods[0]} + {capability.methods[1]} is planned but not implemented."
        )
    if request.strategy not in capability.strategies:
        raise ValueError(
            f"Strategy {request.strategy!r} is unavailable for {capability.methods}; "
            f"choose one of {tuple(capability.strategies)}."
        )
    adapter = _ADAPTERS.get(str(capability.runner or ""))
    if adapter is None:
        raise NotImplementedError(f"No runner is registered for {capability.methods}.")
    return adapter.run(request)


__all__ = [
    "JointInversionRequest",
    "JointInversionResult",
    "JointPairCapability",
    "METHODS",
    "get_joint_capabilities",
    "get_joint_capability",
    "normalize_joint_pair",
    "run_joint_inversion",
]
