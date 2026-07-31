"""SimPEG cross-gradient joint inversion for gravity and magnetic data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from PyHydroGeophysX.data_processing.gravmag import (
    regional_residual,
    spatially_balanced_indices,
)
from PyHydroGeophysX.inversion.gravmag import (
    InversionBackendUnavailable,
    invert_gravmag,
)

ProgressFn = Callable[[Dict[str, Any]], None]


def _projected_gncg_compatible(
    optimization: Any,
    *,
    max_iterations: int,
) -> Any:
    """Construct ProjectedGNCG across old and new SimPEG keyword APIs."""
    common = {
        "maxIter": int(max_iterations),
        "lower": -2.0,
        "upper": 2.0,
        "maxIterLS": 20,
        "tolX": 1e-3,
    }
    try:
        return optimization.ProjectedGNCG(
            **common, cg_maxiter=100, cg_rtol=1e-3
        )
    except Exception as exc:  # SimPEG < 0.25 raises a generic Exception here
        if "attr is not recognized" not in str(exc):
            raise
        return optimization.ProjectedGNCG(
            **common, maxIterCG=100, tolCG=1e-3
        )


def _similarity_history_compatible(directives: Any) -> Any:
    """Construct the in-memory history directive across SimPEG versions."""
    try:
        return directives.SimilarityMeasureSaveOutputEveryIteration(on_disk=False)
    except Exception as exc:  # SimPEG < 0.25 uses save_txt instead of on_disk
        if "attr is not recognized" not in str(exc):
            raise
        return directives.SimilarityMeasureSaveOutputEveryIteration(save_txt=False)


@dataclass
class GravityMagneticsJointResult:
    """Result of a shared-mesh gravity--magnetics inversion."""

    density: np.ndarray
    susceptibility: np.ndarray
    predicted_gravity: np.ndarray
    predicted_magnetics: np.ndarray
    coverage_gravity: np.ndarray
    coverage_magnetics: np.ndarray
    chi2_gravity: float
    chi2_magnetics: float
    convergence: List[Dict[str, Any]]
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray]
    model_shape: Tuple[int, int, int]
    cross_gradient: np.ndarray
    baseline: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def _station_data(value: Mapping[str, Any], name: str) -> Dict[str, np.ndarray]:
    """Validate and normalize one potential-field station mapping."""
    missing = [key for key in ("x", "y", "value") if key not in value]
    if missing:
        raise ValueError(f"{name} data are missing required fields: {', '.join(missing)}.")
    x = np.asarray(value["x"], dtype=float).ravel()
    y = np.asarray(value["y"], dtype=float).ravel()
    observed = np.asarray(value["value"], dtype=float).ravel()
    z_value = value.get("z")
    z = np.ones(x.size, dtype=float) if z_value is None else np.asarray(z_value, dtype=float).ravel()
    if not (x.size == y.size == z.size == observed.size):
        raise ValueError(f"{name} x, y, z and value arrays must have matching lengths.")
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(observed)
    x, y, z, observed = x[good], y[good], z[good], observed[good]
    if x.size < 20:
        raise ValueError(f"{name} requires at least 20 finite stations for 3-D inversion.")
    return {"x": x, "y": y, "z": z, "value": observed}


def _overlap(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    """Return whether two station footprints overlap in both horizontal axes."""
    x_overlap = min(a["x"].max(), b["x"].max()) > max(a["x"].min(), b["x"].min())
    y_overlap = min(a["y"].max(), b["y"].max()) > max(a["y"].min(), b["y"].min())
    return bool(x_overlap and y_overlap)


class JointGravityMagneticsInversion:
    """Jointly recover density and susceptibility with SimPEG cross-gradient coupling.

    Inputs are mappings with ``x``, ``y``, ``value`` and optional ``z`` arrays.
    Gravity values use mGal and magnetic total-field values use nT. Both surveys
    are placed on one tensor mesh; the stacked inversion model is split by a
    :class:`simpeg.maps.Wires` map into density (g/cc) and susceptibility (SI).
    """

    def __init__(
        self,
        gravity_data: Mapping[str, Any],
        magnetics_data: Mapping[str, Any],
        *,
        field: Optional[Mapping[str, Any]] = None,
        n_xy: int = 12,
        n_z: int = 8,
        max_iterations: int = 10,
        max_stations: int = 600,
        gravity_relative_error: float = 0.03,
        magnetics_relative_error: float = 0.03,
        gravity_noise_floor: float = 0.5,
        magnetics_noise_floor: float = 2.0,
        gravity_weight: float = 1.0,
        magnetics_weight: float = 1.0,
        cross_gradient_weight: float = 2e12,
        beta0_ratio: float = 1.0,
        gravity_detrend: int = 0,
        magnetics_detrend: int = 0,
        run_baseline: bool = True,
        baseline_max_iterations: Optional[int] = None,
        random_seed: Optional[int] = 42,
        output_dir: Optional[str] = None,
        progress_callback: Optional[ProgressFn] = None,
    ) -> None:
        self.gravity_data = gravity_data
        self.magnetics_data = magnetics_data
        self.field = dict(field or {})
        self.n_xy = int(n_xy)
        self.n_z = int(n_z)
        self.max_iterations = int(max_iterations)
        self.max_stations = int(max_stations)
        self.gravity_relative_error = float(gravity_relative_error)
        self.magnetics_relative_error = float(magnetics_relative_error)
        self.gravity_noise_floor = float(gravity_noise_floor)
        self.magnetics_noise_floor = float(magnetics_noise_floor)
        self.gravity_weight = float(gravity_weight)
        self.magnetics_weight = float(magnetics_weight)
        self.cross_gradient_weight = float(cross_gradient_weight)
        self.beta0_ratio = float(beta0_ratio)
        self.gravity_detrend = int(gravity_detrend)
        self.magnetics_detrend = int(magnetics_detrend)
        self.run_baseline = bool(run_baseline)
        self.baseline_max_iterations = int(baseline_max_iterations or max_iterations)
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.progress_callback = progress_callback

    def _prepare(self, value: Mapping[str, Any], name: str, detrend: int) -> Dict[str, np.ndarray]:
        stations = _station_data(value, name)
        if detrend > 0:
            _, stations["value"] = regional_residual(
                stations["x"], stations["y"], stations["value"], degree=detrend
            )
        if stations["x"].size > self.max_stations:
            indices = spatially_balanced_indices(
                stations["x"], stations["y"], self.max_stations
            )
            stations = {key: array[indices] for key, array in stations.items()}
        return stations

    def _baseline(
        self,
        gravity_data: Dict[str, np.ndarray],
        magnetics_data: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, Any], List[str]]:
        baseline: Dict[str, Any] = {}
        warnings: List[str] = []
        common = {
            "n_xy": self.n_xy,
            "n_z": self.n_z,
            "max_iterations": self.baseline_max_iterations,
            "max_stations": self.max_stations,
            "random_seed": self.random_seed,
        }
        for method, stations in (("Gravity", gravity_data), ("Magnetics", magnetics_data)):
            try:
                out_dir = None
                if self.output_dir:
                    out_dir = str(Path(self.output_dir) / "baseline" / method.lower())
                result = invert_gravmag(
                    stations["x"], stations["y"], stations["value"], method,
                    z=stations["z"], field=self.field if method == "Magnetics" else None,
                    detrend=0,
                    relative_error=(self.gravity_relative_error if method == "Gravity"
                                    else self.magnetics_relative_error),
                    noise_floor=(self.gravity_noise_floor if method == "Gravity"
                                 else self.magnetics_noise_floor),
                    out_dir=out_dir,
                    **common,
                )
                baseline[method] = {
                    "model": result["model3d"],
                    "chi2": result["chi2"],
                    "edges": result["edges"],
                }
            except Exception as exc:  # noqa: BLE001 - baseline must not discard joint result
                warnings.append(f"{method} independent baseline failed: {exc}")
        return baseline, warnings

    def run(self) -> GravityMagneticsJointResult:
        """Run the native SimPEG similarity-measure inversion."""
        try:
            import pymatsolver
            import simpeg
            from discretize import TensorMesh
            from simpeg import (
                data,
                data_misfit,
                directives,
                inverse_problem,
                inversion,
                maps,
                optimization,
                regularization,
            )
            from simpeg.potential_fields import gravity, magnetics
        except Exception as exc:  # noqa: BLE001 - optional numerical stack
            raise InversionBackendUnavailable(str(exc)) from exc

        if self.n_xy < 4 or self.n_z < 3:
            raise ValueError("Joint Gravity/Magnetics mesh requires n_xy >= 4 and n_z >= 3.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.cross_gradient_weight < 0:
            raise ValueError("cross_gradient_weight cannot be negative.")

        grav = self._prepare(self.gravity_data, "Gravity", self.gravity_detrend)
        mag = self._prepare(self.magnetics_data, "Magnetics", self.magnetics_detrend)
        if not _overlap(grav, mag):
            raise ValueError(
                "Gravity and Magnetics station footprints do not overlap in x/y; "
                "reproject or align them before joint inversion."
            )
        progress = self.progress_callback
        if callable(progress):
            progress({"stage": "mesh", "message": "Building shared potential-field mesh"})

        all_x = np.r_[grav["x"], mag["x"]]
        all_y = np.r_[grav["y"], mag["y"]]
        all_z = np.r_[grav["z"], mag["z"]]
        x0, x1 = float(all_x.min()), float(all_x.max())
        y0, y1 = float(all_y.min()), float(all_y.max())
        x_span = max(x1 - x0, 1.0)
        y_span = max(y1 - y0, 1.0)
        csx, csy = x_span / self.n_xy, y_span / self.n_xy
        csz = max(csx, csy) * 0.6
        nx = self.n_xy + 4
        ny = self.n_xy + 4
        nz = self.n_z
        ox = x0 - 0.15 * x_span - 2 * csx
        oy = y0 - 0.15 * y_span - 2 * csy
        surface_z = float(np.nanmin(all_z) - 0.05 * csz)
        oz = surface_z - csz * nz
        mesh = TensorMesh([[(csx, nx)], [(csy, ny)], [(csz, nz)]], origin=[ox, oy, oz])
        active = np.ones(mesh.n_cells, dtype=bool)
        n_cells = int(active.sum())
        wires = maps.Wires(("density", n_cells), ("susceptibility", n_cells))

        grav_rx = gravity.receivers.Point(
            np.c_[grav["x"], grav["y"], grav["z"]], components="gz"
        )
        grav_survey = gravity.survey.Survey(
            gravity.sources.SourceField(receiver_list=[grav_rx])
        )
        mag_rx = magnetics.receivers.Point(
            np.c_[mag["x"], mag["y"], mag["z"]], components="tmi"
        )
        mag_source = magnetics.sources.UniformBackgroundField(
            receiver_list=[mag_rx],
            amplitude=float(self.field.get("strength_nT", 50000.0)),
            inclination=float(self.field.get("inclination", 60.0)),
            declination=float(self.field.get("declination", 0.0)),
        )
        mag_survey = magnetics.survey.Survey(mag_source)
        grav_sim = gravity.simulation.Simulation3DIntegral(
            mesh=mesh, survey=grav_survey, rhoMap=wires.density,
            active_cells=active, engine="geoana",
        )
        mag_sim = magnetics.simulation.Simulation3DIntegral(
            mesh=mesh, survey=mag_survey, chiMap=wires.susceptibility,
            active_cells=active, engine="geoana", model_type="scalar",
        )
        for simulation in (grav_sim, mag_sim):
            simulation.solver = pymatsolver.Solver
            simulation.solver_opts = {}

        grav_std = (self.gravity_relative_error * np.abs(grav["value"])
                    + self.gravity_noise_floor)
        mag_std = (self.magnetics_relative_error * np.abs(mag["value"])
                   + self.magnetics_noise_floor)
        grav_misfit = data_misfit.L2DataMisfit(
            data=data.Data(grav_survey, dobs=grav["value"], standard_deviation=grav_std),
            simulation=grav_sim,
        )
        mag_misfit = data_misfit.L2DataMisfit(
            data=data.Data(mag_survey, dobs=mag["value"], standard_deviation=mag_std),
            simulation=mag_sim,
        )
        combined_misfit = self.gravity_weight * grav_misfit + self.magnetics_weight * mag_misfit
        grav_reg = regularization.WeightedLeastSquares(
            mesh, active_cells=active, mapping=wires.density
        )
        mag_reg = regularization.WeightedLeastSquares(
            mesh, active_cells=active, mapping=wires.susceptibility
        )
        cross_reg = regularization.CrossGradient(mesh, wires, active_cells=active)
        combined_reg = grav_reg + mag_reg + self.cross_gradient_weight * cross_reg
        optimizer = _projected_gncg_compatible(
            optimization, max_iterations=self.max_iterations
        )
        inverse = inverse_problem.BaseInvProblem(combined_misfit, combined_reg, optimizer)
        history_output = _similarity_history_compatible(directives)

        class _ProgressDirective(directives.InversionDirective):
            def endIter(inner_self) -> None:  # noqa: N802 - SimPEG directive API
                if not callable(progress):
                    return
                values = list(inner_self.invProb.phi_d_list)
                progress({
                    "stage": "inversion",
                    "iteration": int(inner_self.opt.iter),
                    "chi2_gravity": float(values[0]) / grav["value"].size,
                    "chi2_magnetics": float(values[1]) / mag["value"].size,
                })

        directive_list = [
            directives.SimilarityMeasureInversionDirective(),
            directives.UpdateSensitivityWeights(every_iteration=False),
            directives.MovingAndMultiTargetStopping(tol=1e-6),
            directives.PairedBetaEstimate_ByEig(
                beta0_ratio=self.beta0_ratio
            ),
            directives.PairedBetaSchedule(cooling_factor=5, cooling_rate=1),
            history_output,
            directives.UpdatePreconditioner(),
            _ProgressDirective(),
        ]
        starting = np.r_[np.full(n_cells, 1e-6), np.full(n_cells, 1e-6)]
        random_state = np.random.get_state()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        try:
            recovered = np.asarray(
                inversion.BaseInversion(inverse, directiveList=directive_list).run(starting),
                dtype=float,
            )
        finally:
            np.random.set_state(random_state)
        density, susceptibility = (np.asarray(item, dtype=float) for item in wires * recovered)
        predicted_gravity = np.asarray(grav_sim.dpred(recovered), dtype=float)
        predicted_magnetics = np.asarray(mag_sim.dpred(recovered), dtype=float)
        chi2_gravity = float(grav_misfit(recovered)) / grav["value"].size
        chi2_magnetics = float(mag_misfit(recovered)) / mag["value"].size
        try:
            gravity_diag = np.asarray(grav_sim.getJtJdiag(recovered), dtype=float)[:n_cells]
            magnetics_diag = np.asarray(mag_sim.getJtJdiag(recovered), dtype=float)[n_cells:]
            coverage_gravity = np.sqrt(np.maximum(gravity_diag, 0.0))
            coverage_magnetics = np.sqrt(np.maximum(magnetics_diag, 0.0))
        except Exception:  # noqa: BLE001 - sensitivity export is optional
            coverage_gravity = np.asarray([], dtype=float)
            coverage_magnetics = np.asarray([], dtype=float)
        cross_gradient = np.asarray(
            cross_reg.calculate_cross_gradient(recovered, normalized=True), dtype=float
        )
        convergence: List[Dict[str, Any]] = []
        for index, values in enumerate(history_output.phi_d, start=1):
            convergence.append({
                "iteration": index,
                "chi2_gravity": float(values[0]) / grav["value"].size,
                "chi2_magnetics": float(values[1]) / mag["value"].size,
                "cross_gradient": float(history_output.phi_sim[index - 1]),
            })
        baseline: Dict[str, Any] = {}
        baseline_warnings: List[str] = []
        if self.run_baseline:
            if callable(progress):
                progress({"stage": "baseline", "message": "Running independent baselines"})
            baseline, baseline_warnings = self._baseline(grav, mag)
        if callable(progress):
            progress({"stage": "complete", "message": "Gravity/Magnetics joint inversion complete"})

        edges = (
            np.asarray(mesh.nodes_x, dtype=float),
            np.asarray(mesh.nodes_y, dtype=float),
            np.asarray(mesh.nodes_z, dtype=float),
        )
        return GravityMagneticsJointResult(
            density=density,
            susceptibility=susceptibility,
            predicted_gravity=predicted_gravity,
            predicted_magnetics=predicted_magnetics,
            coverage_gravity=coverage_gravity,
            coverage_magnetics=coverage_magnetics,
            chi2_gravity=chi2_gravity,
            chi2_magnetics=chi2_magnetics,
            convergence=convergence,
            edges=edges,
            model_shape=(nx, ny, nz),
            cross_gradient=cross_gradient,
            baseline=baseline,
            meta={
                "backend": "simpeg",
                "backend_version": str(simpeg.__version__),
                "n_cells": n_cells,
                "data_counts": {
                    "Gravity": int(grav["value"].size),
                    "Magnetics": int(mag["value"].size),
                },
                "observed_gravity": grav["value"],
                "observed_magnetics": mag["value"],
                "gravity_locations": np.c_[grav["x"], grav["y"], grav["z"]],
                "magnetics_locations": np.c_[mag["x"], mag["y"], mag["z"]],
                "baseline_warnings": baseline_warnings,
            },
        )


__all__ = ["GravityMagneticsJointResult", "JointGravityMagneticsInversion"]
