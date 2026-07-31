"""Shared-model joint inversion for collocated FDEM and TDEM soundings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from scipy.optimize import least_squares


@dataclass
class JointFDEMTDEMResult:
    """Outputs from a shared-conductivity FDEM-TDEM inversion."""

    resistivity: np.ndarray
    conductivity: np.ndarray
    thicknesses: np.ndarray
    predicted_fdem: np.ndarray
    predicted_tdem: np.ndarray
    chi2_fdem: float
    chi2_tdem: float
    coverage_fdem: Optional[np.ndarray] = None
    coverage_tdem: Optional[np.ndarray] = None
    convergence: List[Dict[str, float]] = field(default_factory=list)
    baseline: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _JointInputs:
    """Validated arrays and simulations shared by both optimizer backends."""

    frequencies: np.ndarray
    times: np.ndarray
    f_obs: np.ndarray
    f_obs_interleaved: np.ndarray
    t_obs: np.ndarray
    f_unc: np.ndarray
    f_unc_interleaved: np.ndarray
    t_unc: np.ndarray
    thicknesses: np.ndarray
    n_layers: int
    f_geom: Dict[str, Any]
    t_geom: Dict[str, Any]
    f_modeler: Any
    t_modeler: Any


class JointFDEMTDEMInversion:
    """Invert one FDEM and one TDEM sounding for a common 1-D conductivity model.

    Both datasets retain their own survey geometry and uncertainty model. Their
    normalized residuals are combined with a common first-difference smoothness
    term, so neither method dominates solely because it has more channels.
    """

    def __init__(
        self,
        fdem_data: Mapping[str, Any],
        tdem_data: Mapping[str, Any],
        fdem_geometry: Optional[Mapping[str, Any]] = None,
        tdem_geometry: Optional[Mapping[str, Any]] = None,
        thicknesses: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        self.fdem_data = dict(fdem_data)
        self.tdem_data = dict(tdem_data)
        self.fdem_geometry = dict(fdem_geometry or {})
        self.tdem_geometry = dict(tdem_geometry or {})
        self.parameters: Dict[str, Any] = {
            "n_layers": 15,
            "min_thickness": 1.0,
            "max_thickness": 40.0,
            "starting_resistivity": 100.0,
            "min_resistivity": 1.0,
            "max_resistivity": 10000.0,
            "max_iterations": 30,
            "smoothness": 0.3,
            "fdem_relative_error": 0.05,
            "tdem_relative_error": 0.05,
            "fdem_noise_floor": 1e-14,
            "tdem_noise_floor": 1e-18,
            "fdem_data_scale": 1.0,
            "tdem_data_scale": 1.0,
            "fdem_weight": 1.0,
            "tdem_weight": 1.0,
            "backend": "auto",
            "run_baseline": True,
        }
        self.parameters.update(kwargs)
        self.thicknesses = None if thicknesses is None else np.asarray(thicknesses, dtype=float).ravel()

    def _layer_thicknesses(self) -> np.ndarray:
        if self.thicknesses is not None:
            if self.thicknesses.size < 1 or np.any(self.thicknesses <= 0):
                raise ValueError("thicknesses must contain positive finite values.")
            return self.thicknesses
        n_layers = max(int(self.parameters["n_layers"]), 2)
        return np.geomspace(
            float(self.parameters["min_thickness"]),
            float(self.parameters["max_thickness"]),
            n_layers - 1,
        )

    @staticmethod
    def _require(data: Mapping[str, Any], fields: tuple, method: str) -> None:
        missing = [name for name in fields if name not in data]
        if missing:
            raise ValueError(f"{method} data is missing required fields: {missing}.")

    @staticmethod
    def _version_tuple(value: str) -> tuple:
        """Return the numeric major/minor/patch prefix of a version string."""
        import re

        values = [int(item) for item in re.findall(r"\d+", str(value))[:3]]
        return tuple((values + [0, 0, 0])[:3])

    @classmethod
    def native_backend_status(cls) -> Dict[str, Any]:
        """Report whether the tested native SimPEG joint path is available."""
        try:
            import simpeg
        except ImportError as exc:
            return {"available": False, "version": "", "reason": str(exc)}
        version = str(getattr(simpeg, "__version__", "0"))
        if cls._version_tuple(version) < (0, 25, 2):
            return {
                "available": False,
                "version": version,
                "reason": "Native joint inversion requires SimPEG >= 0.25.2; SciPy fallback remains available.",
            }
        return {"available": True, "version": version, "reason": ""}

    def _prepare_inputs(self) -> _JointInputs:
        """Validate observations and construct the two native simulations."""
        self._require(self.fdem_data, ("frequencies", "real", "imag"), "FDEM")
        self._require(self.tdem_data, ("times", "response"), "TDEM")

        # Import lazily so the registry and Qt page remain usable without SimPEG.
        from PyHydroGeophysX.forward import em1d as em_forward
        from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling

        frequencies = np.asarray(self.fdem_data["frequencies"], dtype=float).ravel()
        times = np.asarray(self.tdem_data["times"], dtype=float).ravel()
        f_scale = float(self.parameters["fdem_data_scale"])
        t_scale = float(self.parameters["tdem_data_scale"])
        f_real = np.asarray(self.fdem_data["real"], dtype=float).ravel() * f_scale
        f_imag = np.asarray(self.fdem_data["imag"], dtype=float).ravel() * f_scale
        t_obs = np.asarray(self.tdem_data["response"], dtype=float).ravel() * t_scale
        if f_real.size != frequencies.size or f_imag.size != frequencies.size:
            raise ValueError("FDEM frequency, real, and imaginary arrays must have equal length.")
        if t_obs.size != times.size:
            raise ValueError("TDEM time and response arrays must have equal length.")

        thicknesses = self._layer_thicknesses()
        n_layers = thicknesses.size + 1
        f_geom = dict(em_forward.DEFAULT_FDEM)
        f_geom.update(self.fdem_geometry)
        t_geom = dict(em_forward.DEFAULT_TDEM)
        t_geom.update(self.tdem_geometry)
        f_modeler = FDEMForwardModeling(
            thicknesses=thicknesses,
            survey_config=em_forward._fdem_config(f_geom, frequencies),
        )
        t_modeler = TDEMForwardModeling(
            thicknesses=thicknesses,
            survey_config=em_forward._tdem_config(t_geom, times),
        )

        f_obs = np.concatenate((f_real, f_imag))
        f_obs_interleaved = np.column_stack((f_real, f_imag)).ravel()
        f_amp = np.abs(f_real + 1j * f_imag)
        f_unc = np.concatenate((f_amp, f_amp)) * float(self.parameters["fdem_relative_error"])
        f_unc += float(self.parameters["fdem_noise_floor"]) * abs(f_scale)
        f_unc_interleaved = np.repeat(f_amp, 2) * float(self.parameters["fdem_relative_error"])
        f_unc_interleaved += float(self.parameters["fdem_noise_floor"]) * abs(f_scale)
        t_unc = float(self.parameters["tdem_relative_error"]) * np.abs(t_obs)
        t_unc += float(self.parameters["tdem_noise_floor"]) * abs(t_scale)
        f_unc = np.clip(f_unc, 1e-30, None)
        f_unc_interleaved = np.clip(f_unc_interleaved, 1e-30, None)
        t_unc = np.clip(t_unc, 1e-30, None)

        return _JointInputs(
            frequencies=frequencies,
            times=times,
            f_obs=f_obs,
            f_obs_interleaved=f_obs_interleaved,
            t_obs=t_obs,
            f_unc=f_unc,
            f_unc_interleaved=f_unc_interleaved,
            t_unc=t_unc,
            thicknesses=thicknesses,
            n_layers=n_layers,
            f_geom=f_geom,
            t_geom=t_geom,
            f_modeler=f_modeler,
            t_modeler=t_modeler,
        )

    @staticmethod
    def _normalized_coverage(block: np.ndarray) -> np.ndarray:
        sensitivity = np.sqrt(np.sum(np.square(block), axis=0))
        maximum = float(np.max(sensitivity)) if sensitivity.size else 0.0
        return sensitivity / maximum if maximum > 0 else sensitivity

    def _run_scipy(self, inputs: _JointInputs) -> JointFDEMTDEMResult:
        """Run the backwards-compatible SciPy least-squares implementation."""
        import scipy

        def forward_fdem(resistivity: np.ndarray) -> np.ndarray:
            response = np.asarray(inputs.f_modeler.forward(
                1.0 / np.clip(resistivity, 1e-12, None)
            )).ravel()
            if response.size == 2 * inputs.frequencies.size and not np.iscomplexobj(response):
                response = response[0::2] + 1j * response[1::2]
            response = np.asarray(response, dtype=complex).ravel()[: inputs.frequencies.size]
            return np.concatenate((response.real, response.imag))

        def forward_tdem(resistivity: np.ndarray) -> np.ndarray:
            return np.asarray(
                inputs.t_modeler.forward(1.0 / np.clip(resistivity, 1e-12, None)), dtype=float
            ).ravel()[: inputs.times.size]

        f_weight = float(self.parameters["fdem_weight"]) / np.sqrt(max(inputs.f_obs.size, 1))
        t_weight = float(self.parameters["tdem_weight"]) / np.sqrt(max(inputs.t_obs.size, 1))
        smooth = np.sqrt(max(float(self.parameters["smoothness"]), 0.0))
        convergence: List[Dict[str, float]] = []

        def components(log10_resistivity: np.ndarray) -> tuple:
            resistivity = np.power(10.0, log10_resistivity)
            pred_f = forward_fdem(resistivity)
            pred_t = forward_tdem(resistivity)
            raw_f = (pred_f - inputs.f_obs) / inputs.f_unc
            raw_t = (pred_t - inputs.t_obs) / inputs.t_unc
            return pred_f, pred_t, raw_f, raw_t

        def residual(log10_resistivity: np.ndarray) -> np.ndarray:
            _pred_f, _pred_t, raw_f, raw_t = components(log10_resistivity)
            model_term = smooth * np.diff(log10_resistivity)
            return np.concatenate((f_weight * raw_f, t_weight * raw_t, model_term))

        def callback(intermediate: Any) -> None:
            model = np.asarray(getattr(intermediate, "x", intermediate), dtype=float)
            _pred_f, _pred_t, raw_f, raw_t = components(model)
            convergence.append({
                "chi2_fdem": float(np.mean(raw_f**2)),
                "chi2_tdem": float(np.mean(raw_t**2)),
            })
            progress_callback = self.parameters.get("progress_callback")
            if callable(progress_callback):
                progress_callback(dict(convergence[-1]))

        start = np.full(inputs.n_layers, np.log10(float(self.parameters["starting_resistivity"])))
        lower = np.full(inputs.n_layers, np.log10(float(self.parameters["min_resistivity"])))
        upper = np.full(inputs.n_layers, np.log10(float(self.parameters["max_resistivity"])))
        options: Dict[str, Any] = {
            "method": "trf",
            "bounds": (lower, upper),
            "max_nfev": max(40, int(self.parameters["max_iterations"]) * (inputs.n_layers + 1)),
            "xtol": 1e-8,
            "ftol": 1e-8,
        }
        try:
            import inspect
            if "callback" in inspect.signature(least_squares).parameters:
                options["callback"] = callback
        except Exception:
            pass
        solution = least_squares(residual, start, **options)
        resistivity = np.power(10.0, solution.x)
        predicted_fdem, predicted_tdem, raw_fdem, raw_tdem = components(solution.x)
        final_record = {
            "chi2_fdem": float(np.mean(raw_fdem**2)),
            "chi2_tdem": float(np.mean(raw_tdem**2)),
        }
        if not convergence or convergence[-1] != final_record:
            convergence.append(final_record)

        jacobian = np.asarray(solution.jac, dtype=float)

        fdem_rows = inputs.f_obs.size
        tdem_rows = inputs.t_obs.size
        coverage_fdem = self._normalized_coverage(jacobian[:fdem_rows])
        coverage_tdem = self._normalized_coverage(jacobian[fdem_rows:fdem_rows + tdem_rows])

        return JointFDEMTDEMResult(
            resistivity=resistivity,
            conductivity=1.0 / np.clip(resistivity, 1e-12, None),
            thicknesses=inputs.thicknesses,
            predicted_fdem=predicted_fdem,
            predicted_tdem=predicted_tdem,
            chi2_fdem=final_record["chi2_fdem"],
            chi2_tdem=final_record["chi2_tdem"],
            coverage_fdem=coverage_fdem,
            coverage_tdem=coverage_tdem,
            convergence=convergence,
            meta={
                "nfev": int(solution.nfev),
                "iterations": len(convergence),
                "backend_version": str(getattr(scipy, "__version__", "")),
            },
        )

    def _run_simpeg(self, inputs: _JointInputs) -> JointFDEMTDEMResult:
        """Run a native SimPEG multi-data-misfit inversion on one shared model."""
        from scipy.sparse import diags
        import simpeg
        from simpeg import (
            data,
            data_misfit,
            directives,
            inverse_problem,
            inversion,
            maps,
            objective_function,
            optimization,
        )
        from simpeg.electromagnetics import frequency_domain as fdem
        from simpeg.electromagnetics import time_domain as tdem

        model_map = maps.ExpMap(nP=inputs.n_layers)
        f_simulation = fdem.Simulation1DLayered(
            survey=inputs.f_modeler.survey,
            thicknesses=inputs.thicknesses,
            sigmaMap=model_map,
        )
        t_simulation = tdem.Simulation1DLayered(
            survey=inputs.t_modeler.survey,
            thicknesses=inputs.thicknesses,
            sigmaMap=model_map,
        )
        f_data = data.Data(
            inputs.f_modeler.survey,
            dobs=inputs.f_obs_interleaved,
            standard_deviation=inputs.f_unc_interleaved,
        )
        t_data = data.Data(
            inputs.t_modeler.survey,
            dobs=inputs.t_obs,
            standard_deviation=inputs.t_unc,
        )
        f_misfit = data_misfit.L2DataMisfit(data=f_data, simulation=f_simulation)
        t_misfit = data_misfit.L2DataMisfit(data=t_data, simulation=t_simulation)
        f_multiplier = float(self.parameters["fdem_weight"]) ** 2 / max(inputs.f_obs.size, 1)
        t_multiplier = float(self.parameters["tdem_weight"]) ** 2 / max(inputs.t_obs.size, 1)
        joint_misfit = f_multiplier * f_misfit + t_multiplier * t_misfit

        difference = diags(
            (-np.ones(inputs.n_layers - 1), np.ones(inputs.n_layers - 1)),
            (0, 1),
            shape=(inputs.n_layers - 1, inputs.n_layers),
            format="csr",
        )
        smoothness = max(float(self.parameters["smoothness"]), 0.0)
        regularization = objective_function.L2ObjectiveFunction(
            nP=inputs.n_layers,
            W=(np.sqrt(smoothness) / np.log(10.0)) * difference,
        )
        lower = np.full(inputs.n_layers, -np.log(float(self.parameters["max_resistivity"])))
        upper = np.full(inputs.n_layers, -np.log(float(self.parameters["min_resistivity"])))
        optimizer = optimization.ProjectedGNCG(
            maxIter=int(self.parameters["max_iterations"]),
            lower=lower,
            upper=upper,
            cg_maxiter=max(20, 4 * inputs.n_layers),
            cg_atol=1e-3,
            cg_rtol=0.0,
            maxIterLS=20,
            tolX=1e-3,
            tolF=1e-3,
            tolG=1e-3,
        )
        convergence: List[Dict[str, float]] = []

        def components(log_conductivity: np.ndarray) -> tuple:
            f_interleaved = np.asarray(f_simulation.dpred(log_conductivity), dtype=float).ravel()
            f_complex = f_interleaved[0::2] + 1j * f_interleaved[1::2]
            pred_f = np.concatenate((f_complex.real, f_complex.imag))
            pred_t = np.asarray(t_simulation.dpred(log_conductivity), dtype=float).ravel()
            raw_f = (pred_f - inputs.f_obs) / inputs.f_unc
            raw_t = (pred_t - inputs.t_obs) / inputs.t_unc
            return pred_f, pred_t, raw_f, raw_t

        def callback(model: np.ndarray) -> None:
            _pred_f, _pred_t, raw_f, raw_t = components(np.asarray(model, dtype=float))
            record = {
                "iteration": len(convergence) + 1,
                "chi2_fdem": float(np.mean(raw_f**2)),
                "chi2_tdem": float(np.mean(raw_t**2)),
            }
            convergence.append(record)
            progress_callback = self.parameters.get("progress_callback")
            if callable(progress_callback):
                progress_callback(dict(record))

        class _ProgressDirective(directives.InversionDirective):
            def endIter(self) -> None:  # noqa: N802 - SimPEG directive API
                callback(np.asarray(self.invProb.model, dtype=float))

        inverse_problem_object = inverse_problem.BaseInvProblem(
            joint_misfit,
            regularization,
            optimizer,
            beta=1.0,
            print_version=False,
            init_bfgs=False,
        )
        inversion_object = inversion.BaseInversion(
            inverse_problem_object,
            directiveList=[_ProgressDirective()],
        )
        starting_model = np.full(
            inputs.n_layers,
            -np.log(float(self.parameters["starting_resistivity"])),
        )
        recovered_model = np.asarray(inversion_object.run(starting_model), dtype=float)
        predicted_fdem, predicted_tdem, raw_fdem, raw_tdem = components(recovered_model)
        final_record = {
            "iteration": int(getattr(optimizer, "iter", len(convergence))),
            "chi2_fdem": float(np.mean(raw_fdem**2)),
            "chi2_tdem": float(np.mean(raw_tdem**2)),
        }
        if not convergence or not (
            np.isclose(convergence[-1]["chi2_fdem"], final_record["chi2_fdem"])
            and np.isclose(convergence[-1]["chi2_tdem"], final_record["chi2_tdem"])
        ):
            convergence.append(final_record)

        f_jacobian = np.asarray(f_simulation.getJ(recovered_model), dtype=float)
        t_jacobian = np.asarray(t_simulation.getJ(recovered_model), dtype=float)
        f_weighted_jacobian = (
            np.sqrt(f_multiplier) * f_jacobian / inputs.f_unc_interleaved[:, None]
        )
        t_weighted_jacobian = (
            np.sqrt(t_multiplier) * t_jacobian / inputs.t_unc[:, None]
        )
        resistivity = np.exp(-recovered_model)
        return JointFDEMTDEMResult(
            resistivity=resistivity,
            conductivity=np.exp(recovered_model),
            thicknesses=inputs.thicknesses,
            predicted_fdem=predicted_fdem,
            predicted_tdem=predicted_tdem,
            chi2_fdem=final_record["chi2_fdem"],
            chi2_tdem=final_record["chi2_tdem"],
            coverage_fdem=self._normalized_coverage(f_weighted_jacobian),
            coverage_tdem=self._normalized_coverage(t_weighted_jacobian),
            convergence=convergence,
            meta={
                "iterations": int(getattr(optimizer, "iter", len(convergence))),
                "simpeg_version": str(getattr(simpeg, "__version__", "")),
                "backend_version": str(getattr(simpeg, "__version__", "")),
            },
        )

    def _run_baseline(self, inputs: _JointInputs) -> Dict[str, Any]:
        """Run the existing independent single-method comparison models."""
        if not bool(self.parameters.get("run_baseline", True)):
            return {}
        from PyHydroGeophysX.inversion import em1d as em_inversion

        common = {
            "n_layers": inputs.n_layers,
            "min_thickness": float(inputs.thicknesses.min()),
            "max_thickness": float(inputs.thicknesses.max()),
            "starting_resistivity": float(self.parameters["starting_resistivity"]),
            "max_iterations": int(self.parameters["max_iterations"]),
            "smoothness": float(self.parameters["smoothness"]),
        }
        f_inv = {
            **common,
            "rel_error": float(self.parameters["fdem_relative_error"]),
            "noise_floor": float(self.parameters["fdem_noise_floor"]),
            "data_scale": float(self.parameters["fdem_data_scale"]),
        }
        t_inv = {
            **common,
            "rel_error": float(self.parameters["tdem_relative_error"]),
            "noise_floor": float(self.parameters["tdem_noise_floor"]),
            "data_scale": float(self.parameters["tdem_data_scale"]),
        }
        return {
            "FDEM": em_inversion.fdem_invert(self.fdem_data, inputs.f_geom, f_inv),
            "TDEM": em_inversion.tdem_invert(self.tdem_data, inputs.t_geom, t_inv),
        }

    def run(self) -> JointFDEMTDEMResult:
        """Run native SimPEG joint inversion or the compatible SciPy fallback."""
        inputs = self._prepare_inputs()
        requested = str(self.parameters.get("backend", "auto")).strip().lower()
        aliases = {"native": "simpeg", "simpeg_native": "simpeg", "fallback": "scipy"}
        requested = aliases.get(requested, requested)
        if requested not in {"auto", "simpeg", "scipy"}:
            raise ValueError("backend must be 'auto', 'simpeg', or 'scipy'.")

        status = self.native_backend_status()
        fallback_reason = ""
        if requested == "scipy" or (requested == "auto" and not status["available"]):
            resolved = "scipy"
            fallback_reason = str(status["reason"]) if requested == "auto" else ""
            result = self._run_scipy(inputs)
        elif requested == "simpeg" and not status["available"]:
            raise RuntimeError(str(status["reason"]))
        else:
            try:
                result = self._run_simpeg(inputs)
                resolved = "simpeg"
            except Exception as exc:
                if requested == "simpeg" or "cancelled by user" in str(exc).lower():
                    raise
                fallback_reason = f"Native SimPEG joint inversion failed: {exc}"
                result = self._run_scipy(inputs)
                resolved = "scipy"

        result.baseline = self._run_baseline(inputs)
        result.meta.update({
            "backend": resolved,
            "requested_backend": requested,
            "fallback_reason": fallback_reason,
            "parameters": {
                key: value for key, value in self.parameters.items()
                if key != "progress_callback"
            },
        })
        return result


__all__ = ["JointFDEMTDEMInversion", "JointFDEMTDEMResult"]
