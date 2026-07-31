"""Unified geophysical inversion agent for SRT and FDEM workflows."""

import os
from typing import Any, Dict, Optional

import numpy as np

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Geophysical Inversion Agent
# ---------------------------------------------------------------------------
class GeophysicalInversionAgent(BaseAgent):
    """
    Agent for multi-method inversion orchestration.

    Supports SRT and FDEM natively and delegates ERT to ERTInversionAgent.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, llm_provider: str = "openai"):
        super().__init__("geophysical_inversion", api_key, model, llm_provider)
        self.system_message = (
            "You are an expert geophysical inversion assistant. "
            "You configure and execute SRT and FDEM inversions and route ERT jobs "
            "to the dedicated ERT inversion agent."
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        method = str(input_data.get("method", "ert")).lower().strip()

        if method == "ert":
            from .ert_inversion_agent import ERTInversionAgent

            self._log_execution("Delegating ERT task to ERTInversionAgent")
            ert_agent = ERTInversionAgent(
                api_key=self.api_key,
                model=self.model,
                llm_provider=self.llm_provider,
            )
            return ert_agent.execute(input_data)

        if method == "srt":
            return self._run_srt(input_data)

        if method == "fdem":
            return self._run_fdem(input_data)

        if method in {"joint_ert_srt", "joint", "ert_srt_joint"}:
            return self._run_joint_ert_srt(input_data)

        if method == "tdem":
            from .tdem_agent import TDEMAgent

            self._log_execution("Delegating TDEM task to TDEMAgent")
            tdem_agent = TDEMAgent(
                api_key=self.api_key,
                model=self.model,
                llm_provider=self.llm_provider,
            )
            return tdem_agent.execute(input_data)

        raise ValueError("Unsupported method. Use one of: 'ert', 'srt', 'fdem', 'tdem', 'joint_ert_srt'.")

    def _run_joint_ert_srt(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        from PyHydroGeophysX.inversion.joint_ert_srt import JointERTSRTInversion

        output_dir = input_data.get("output_dir", "results/joint_ert_srt_inversion")
        os.makedirs(output_dir, exist_ok=True)

        inversion_params = dict(input_data.get("inversion_params", {}))

        if "ert_data" not in input_data or "srt_data" not in input_data:
            raise ValueError("Joint ERT-SRT inversion requires both 'ert_data' and 'srt_data'.")

        self._log_execution("Running joint ERT-SRT inversion")
        inversion = JointERTSRTInversion(
            ert_data=input_data["ert_data"],
            srt_data=input_data["srt_data"],
            mesh=input_data.get("mesh"),
            **inversion_params,
        )
        result = inversion.run()

        np.save(os.path.join(output_dir, "joint_ert_resistivity.npy"), result.ert_resistivity)
        np.save(os.path.join(output_dir, "joint_srt_velocity.npy"), result.srt_velocity)
        np.save(os.path.join(output_dir, "joint_ert_predicted_log_rhoa.npy"), result.ert_predicted)
        np.save(os.path.join(output_dir, "joint_srt_predicted_time.npy"), result.srt_predicted)

        self.results = {
            "status": "success",
            "method": "joint_ert_srt",
            "ert_resistivity": result.ert_resistivity,
            "srt_velocity": result.srt_velocity,
            "chi2_ert": result.chi2_ert,
            "chi2_srt": result.chi2_srt,
            "iterations": result.iteration_history,
            "output_dir": output_dir,
        }
        return self.results

    def _run_srt(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        from PyHydroGeophysX.inversion.srt_inversion import SRTInversion
        from PyHydroGeophysX.inversion.srt_time_lapse import TimeLapseSRTInversion

        output_dir = input_data.get("output_dir", "results/srt_inversion")
        os.makedirs(output_dir, exist_ok=True)

        inversion_params = dict(input_data.get("inversion_params", {}))

        data_files = input_data.get("data_files")
        is_time_lapse = bool(input_data.get("time_lapse", False))

        if data_files is not None and len(data_files) > 1:
            is_time_lapse = True

        if is_time_lapse:
            if not data_files:
                raise ValueError("time_lapse SRT requires 'data_files'.")

            measurement_times = input_data.get("measurement_times")
            if measurement_times is None:
                measurement_times = list(range(len(data_files)))

            self._log_execution(f"Running time-lapse SRT inversion for {len(data_files)} timesteps")
            inversion = TimeLapseSRTInversion(
                data_files=data_files,
                measurement_times=measurement_times,
                mesh=input_data.get("mesh"),
                **inversion_params,
            )
            result = inversion.run(initial_model=input_data.get("initial_model"))

            np.save(os.path.join(output_dir, "srt_timelapse_models.npy"), result.final_models)
            np.save(os.path.join(output_dir, "srt_timelapse_predicted.npy"), result.predicted_data)

            self.results = {
                "status": "success",
                "method": "srt",
                "mode": "time_lapse",
                "final_models": result.final_models,
                "timesteps": result.timesteps,
                "chi2_history": result.all_chi2,
                "output_dir": output_dir,
            }
            return self.results

        data_file = input_data.get("data_file")
        if data_file is None:
            raise ValueError("single-time SRT requires 'data_file'.")

        self._log_execution(f"Running single-time SRT inversion: {data_file}")
        inversion = SRTInversion(
            data_file=data_file,
            mesh=input_data.get("mesh"),
            **inversion_params,
        )
        result = inversion.run(initial_model=input_data.get("initial_model"))

        np.save(os.path.join(output_dir, "srt_velocity_model.npy"), result.final_model)
        np.save(os.path.join(output_dir, "srt_predicted_data.npy"), result.predicted_data)

        self.results = {
            "status": "success",
            "method": "srt",
            "mode": "single",
            "velocity_model": result.final_model,
            "predicted_data": result.predicted_data,
            "chi2_history": result.iteration_chi2,
            "output_dir": output_dir,
        }
        return self.results

    def _run_fdem(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        from PyHydroGeophysX.inversion.fdem_inversion import FDEMInversion

        output_dir = input_data.get("output_dir", "results/fdem_inversion")
        os.makedirs(output_dir, exist_ok=True)

        required = ["frequencies", "dobs", "uncertainties"]
        missing = [name for name in required if name not in input_data]
        if missing:
            raise ValueError(f"FDEM inversion missing required fields: {missing}")

        inversion_params = dict(input_data.get("inversion_params", {}))

        self._log_execution("Running FDEM inversion")
        inversion = FDEMInversion(
            frequencies=np.asarray(input_data["frequencies"], dtype=float),
            dobs=np.asarray(input_data["dobs"]),
            uncertainties=np.asarray(input_data["uncertainties"]),
            thicknesses=input_data.get("thicknesses"),
            source_location=input_data.get("source_location"),
            source_radius=float(input_data.get("source_radius", 10.0)),
            receiver_location=input_data.get("receiver_location"),
            receiver_orientation=str(input_data.get("receiver_orientation", "z")),
            receiver_component=str(input_data.get("receiver_component", "secondary")),
            waveform_type=str(input_data.get("waveform_type", "dipole")),
            n_layers=int(input_data.get("n_layers", 25)),
            min_thickness=float(input_data.get("min_thickness", 1.0)),
            max_thickness=float(input_data.get("max_thickness", 30.0)),
            **inversion_params,
        )

        result = inversion.run(starting_model=input_data.get("starting_model"))

        np.save(os.path.join(output_dir, "fdem_recovered_conductivity.npy"), result.recovered_conductivity)
        np.save(os.path.join(output_dir, "fdem_recovered_model_log.npy"), result.recovered_model)
        np.save(os.path.join(output_dir, "fdem_predicted_data.npy"), result.predicted_data)

        self.results = {
            "status": "success",
            "method": "fdem",
            "recovered_conductivity": result.recovered_conductivity,
            "predicted_data": result.predicted_data,
            "chi2": result.chi2,
            "thicknesses": result.thicknesses,
            "frequencies": result.frequencies,
            "output_dir": output_dir,
        }
        return self.results

    def _log_execution(self, message: str, level: str = "INFO") -> None:
        print(f"[{self.name}][{level}] {message}")
