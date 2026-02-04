"""
Model Output Agent

Loads hydrological model outputs (MODFLOW and ParFlow) and provides
basic summaries and visualization artifacts.
"""

from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .base_agent import BaseAgent


class ModelOutputAgent(BaseAgent):
    """
    Agent for loading hydrological model outputs.

    Supports:
    - MODFLOW water content (WaterContent) and porosity (via flopy)
    - ParFlow saturation and porosity/mask (PFB files via parflow)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        super().__init__("model_output", api_key, model, llm_provider)
        self.system_message = (
            "You load hydrological model outputs (MODFLOW/ParFlow) and summarize them "
            "for downstream hydrogeophysics workflows."
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load hydrological model outputs based on configuration.

        Args:
            input_data: Dictionary containing:
                - hydro_model: 'modflow', 'parflow', 'both', or 'auto'
                - modflow_dir / parflow_dir / model_directory
                - idomain_file (MODFLOW)
                - model_name (MODFLOW)
                - run_name (ParFlow)
                - timestep (int)
                - parflow_timestep (int)
                - nlay (int)
                - plot_layer (int)
                - output_dir

        Returns:
            Dictionary with loaded arrays, summary stats, and artifact paths.
        """
        self._log_execution("Starting hydrological model output loading")

        output_dir = Path(input_data.get("output_dir", "results/model_output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        hydro_model = (input_data.get("hydro_model") or "auto").lower()
        user_request = (input_data.get("user_request") or "").lower()

        wants_modflow = hydro_model in ["modflow", "both"]
        wants_parflow = hydro_model in ["parflow", "both"]
        if hydro_model == "auto":
            wants_modflow = "modflow" in user_request or input_data.get("modflow_dir") is not None
            wants_parflow = "parflow" in user_request or input_data.get("parflow_dir") is not None

        if not wants_modflow and not wants_parflow:
            raise ValueError("No hydrological model specified. Use 'modflow', 'parflow', or 'both'.")

        results: Dict[str, Any] = {
            "status": "success",
            "output_dir": str(output_dir),
            "modflow": None,
            "parflow": None,
            "warnings": [],
        }

        user_request = (input_data.get("user_request") or "").lower()
        convert_to_resistivity = bool(input_data.get("convert_to_resistivity")) or ("resistivity" in user_request)
        convert_to_velocity = bool(input_data.get("convert_to_velocity")) or ("velocity" in user_request)
        petrophysical_params = input_data.get("petrophysical_params", {}) or {}
        rho_sat = petrophysical_params.get("rho_sat", None)
        n_exp = petrophysical_params.get("n", None)

        if wants_modflow:
            try:
                results["modflow"] = self._load_modflow(input_data, output_dir)
            except Exception as exc:
                results["warnings"].append(f"MODFLOW load failed: {exc}")
                if not wants_parflow:
                    results["status"] = "failed"
                    results["error"] = f"MODFLOW load failed: {exc}"
                    return results

        if wants_parflow:
            try:
                results["parflow"] = self._load_parflow(input_data, output_dir)
            except Exception as exc:
                results["warnings"].append(f"ParFlow load failed: {exc}")
                if not wants_modflow:
                    results["status"] = "failed"
                    results["error"] = f"ParFlow load failed: {exc}"
                    return results

        # Optional resistivity conversion if requested
        if convert_to_resistivity:
            if rho_sat is None or n_exp is None:
                results["warnings"].append("Resistivity conversion requested but rho_sat or n not provided.")
            else:
                porosity_default = petrophysical_params.get("porosity") or petrophysical_params.get("phi")
                self._apply_resistivity_conversion(results, rho_sat, n_exp, output_dir, porosity_default)

        # Optional velocity conversion if requested
        if convert_to_velocity:
            velocity_params = {
                "v_dry": petrophysical_params.get("v_dry", 3500.0),
                "v_sat": petrophysical_params.get("v_sat", 4500.0),
                "velocity_model": petrophysical_params.get("velocity_model", "linear"),
            }
            porosity_default = petrophysical_params.get("porosity") or petrophysical_params.get("phi")
            self._apply_velocity_conversion(results, velocity_params, output_dir, porosity_default)

        if results.get("warnings"):
            self._log_execution("Completed with warnings: " + "; ".join(results["warnings"]), level="WARNING")
        else:
            self._log_execution("Model output loading completed successfully")

        return results

    def _apply_resistivity_conversion(
        self,
        results: Dict[str, Any],
        rho_sat: float,
        n_exp: float,
        output_dir: Path,
        porosity_default: Optional[float] = None
    ) -> None:
        """Convert water content (or saturation) to resistivity when possible."""
        from PyHydroGeophysX.petrophysics.resistivity_models import water_content_to_resistivity

        # MODFLOW conversion
        mod = results.get("modflow")
        if mod:
            try:
                wc_path = mod.get("water_content_file")
                por_path = mod.get("porosity_file")
                if wc_path and os.path.exists(wc_path):
                    water_content = np.load(wc_path)
                    if por_path and os.path.exists(por_path):
                        porosity = np.load(por_path)
                    elif porosity_default is not None:
                        porosity = float(porosity_default)
                    else:
                        results["warnings"].append(
                            "MODFLOW resistivity conversion skipped (porosity missing; provide porosity or enable porosity loading)."
                        )
                        porosity = None

                    if porosity is not None:
                        if isinstance(porosity, np.ndarray) and porosity.ndim == 2 and water_content.ndim == 3:
                            porosity = np.broadcast_to(porosity, water_content.shape)
                        resistivity = water_content_to_resistivity(
                            water_content, rhos=rho_sat, n=n_exp, porosity=porosity
                        )
                        res_path = Path(mod.get("output_dir", output_dir / "modflow"))
                        res_path.mkdir(parents=True, exist_ok=True)
                        res_file = res_path / "resistivity_from_modflow.npy"
                        np.save(res_file, resistivity)
                        mod["resistivity_file"] = str(res_file)
                        mod["resistivity_stats"] = self._array_stats(resistivity)
                else:
                    results["warnings"].append("MODFLOW resistivity conversion skipped (missing water content).")
            except Exception as exc:
                results["warnings"].append(f"MODFLOW resistivity conversion failed: {exc}")

        # ParFlow conversion
        par = results.get("parflow")
        if par:
            try:
                sat_path = par.get("saturation_file")
                por_path = par.get("porosity_file")
                if sat_path and os.path.exists(sat_path):
                    saturation = np.load(sat_path)
                    if por_path and os.path.exists(por_path):
                        porosity = np.load(por_path)
                    elif porosity_default is not None:
                        porosity = float(porosity_default)
                    else:
                        results["warnings"].append(
                            "ParFlow resistivity conversion skipped (porosity missing; provide porosity or enable porosity loading)."
                        )
                        porosity = None

                    if porosity is not None:
                        water_content = saturation * porosity
                        resistivity = water_content_to_resistivity(
                            water_content, rhos=rho_sat, n=n_exp, porosity=porosity
                        )
                        res_path = Path(par.get("output_dir", output_dir / "parflow"))
                        res_path.mkdir(parents=True, exist_ok=True)
                        res_file = res_path / "resistivity_from_parflow.npy"
                        np.save(res_file, resistivity)
                        par["resistivity_file"] = str(res_file)
                        par["resistivity_stats"] = self._array_stats(resistivity)
                else:
                    results["warnings"].append("ParFlow resistivity conversion skipped (missing saturation).")
            except Exception as exc:
                results["warnings"].append(f"ParFlow resistivity conversion failed: {exc}")

    def _apply_velocity_conversion(
        self,
        results: Dict[str, Any],
        velocity_params: Dict[str, Any],
        output_dir: Path,
        porosity_default: Optional[float] = None
    ) -> None:
        """Convert water content (or saturation) to seismic velocity when possible."""
        from PyHydroGeophysX.petrophysics.velocity_models import water_content_to_velocity

        v_dry = velocity_params.get("v_dry", 3500.0)
        v_sat = velocity_params.get("v_sat", 4500.0)
        velocity_model = velocity_params.get("velocity_model", "linear")

        # MODFLOW conversion
        mod = results.get("modflow")
        if mod:
            try:
                wc_path = mod.get("water_content_file")
                por_path = mod.get("porosity_file")
                if wc_path and os.path.exists(wc_path):
                    water_content = np.load(wc_path)
                    if por_path and os.path.exists(por_path):
                        porosity = np.load(por_path)
                    elif porosity_default is not None:
                        porosity = float(porosity_default)
                    else:
                        results["warnings"].append(
                            "MODFLOW velocity conversion skipped (porosity missing; provide porosity or enable porosity loading)."
                        )
                        porosity = None

                    if porosity is not None:
                        if isinstance(porosity, np.ndarray) and porosity.ndim == 2 and water_content.ndim == 3:
                            porosity = np.broadcast_to(porosity, water_content.shape)
                        velocity = water_content_to_velocity(
                            water_content, v_dry=v_dry, v_sat=v_sat, porosity=porosity, model=velocity_model
                        )
                        vel_path = Path(mod.get("output_dir", output_dir / "modflow"))
                        vel_path.mkdir(parents=True, exist_ok=True)
                        vel_file = vel_path / "velocity_from_modflow.npy"
                        np.save(vel_file, velocity)
                        mod["velocity_file"] = str(vel_file)
                        mod["velocity_stats"] = self._array_stats(velocity)
                else:
                    results["warnings"].append("MODFLOW velocity conversion skipped (missing water content).")
            except Exception as exc:
                results["warnings"].append(f"MODFLOW velocity conversion failed: {exc}")

        # ParFlow conversion
        par = results.get("parflow")
        if par:
            try:
                sat_path = par.get("saturation_file")
                por_path = par.get("porosity_file")
                if sat_path and os.path.exists(sat_path):
                    saturation = np.load(sat_path)
                    if por_path and os.path.exists(por_path):
                        porosity = np.load(por_path)
                    elif porosity_default is not None:
                        porosity = float(porosity_default)
                    else:
                        results["warnings"].append(
                            "ParFlow velocity conversion skipped (porosity missing; provide porosity or enable porosity loading)."
                        )
                        porosity = None

                    if porosity is not None:
                        water_content = saturation * porosity
                        velocity = water_content_to_velocity(
                            water_content, v_dry=v_dry, v_sat=v_sat, porosity=porosity, model=velocity_model
                        )
                        vel_path = Path(par.get("output_dir", output_dir / "parflow"))
                        vel_path.mkdir(parents=True, exist_ok=True)
                        vel_file = vel_path / "velocity_from_parflow.npy"
                        np.save(vel_file, velocity)
                        par["velocity_file"] = str(vel_file)
                        par["velocity_stats"] = self._array_stats(velocity)
                else:
                    results["warnings"].append("ParFlow velocity conversion skipped (missing saturation).")
            except Exception as exc:
                results["warnings"].append(f"ParFlow velocity conversion failed: {exc}")

    def _load_modflow(self, input_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent, MODFLOWPorosity

        modflow_dir = input_data.get("modflow_dir") or input_data.get("model_directory")
        if not modflow_dir:
            raise ValueError("MODFLOW model directory not provided.")
        modflow_dir = Path(modflow_dir)
        if not modflow_dir.exists():
            raise FileNotFoundError(f"MODFLOW directory not found: {modflow_dir}")

        load_water_content = input_data.get("load_water_content", True)
        load_porosity = input_data.get("load_porosity", True)

        idomain_file = input_data.get("idomain_file")
        if not idomain_file:
            # Try common filenames
            for candidate in ["id.txt", "idomain.txt", "idomain.dat"]:
                cand_path = modflow_dir / candidate
                if cand_path.exists():
                    idomain_file = str(cand_path)
                    break
        water_content = None
        wc_path = None
        timestep = int(input_data.get("timestep", 1))
        nlay = int(input_data.get("nlay", 3))

        if load_water_content:
            watercontent_file = modflow_dir / "WaterContent"
            if not idomain_file:
                self._log_execution("idomain file not found; skipping MODFLOW water content.", level="WARNING")
            elif not watercontent_file.exists():
                self._log_execution("WaterContent file not found; skipping MODFLOW water content.", level="WARNING")
            else:
                idomain = np.loadtxt(idomain_file)
                if idomain.ndim != 2:
                    raise ValueError(f"idomain must be 2D array, got shape {idomain.shape}")
                wc_processor = MODFLOWWaterContent(str(modflow_dir), idomain)
                water_content = wc_processor.load_timestep(timestep, nlay=nlay)

        modflow_output_dir = output_dir / "modflow"
        modflow_output_dir.mkdir(parents=True, exist_ok=True)

        if water_content is not None:
            wc_path = modflow_output_dir / f"water_content_t{timestep}.npy"
            np.save(wc_path, water_content)

        porosity = None
        porosity_path = None
        model_name = input_data.get("model_name") or self._infer_modflow_model_name(modflow_dir)
        if load_porosity:
            if model_name:
                try:
                    porosity_loader = MODFLOWPorosity(str(modflow_dir), model_name=model_name)
                    porosity = porosity_loader.load_porosity()
                    porosity_path = modflow_output_dir / "porosity.npy"
                    np.save(porosity_path, porosity)
                except Exception as exc:
                    self._log_execution(f"MODFLOW porosity load failed: {exc}", level="WARNING")
            else:
                self._log_execution("MODFLOW model_name not provided; skipping porosity.", level="WARNING")

        plot_layer = int(input_data.get("plot_layer", max(0, min(nlay - 1, nlay // 2))))
        plot_paths = {}
        if water_content is not None:
            plot_paths.update(self._save_layer_plot(
                water_content, plot_layer, "MODFLOW Water Content", modflow_output_dir / "water_content_layer.png"
            ))
        if porosity is not None:
            plot_paths.update(self._save_layer_plot(
                porosity, plot_layer, "MODFLOW Porosity", modflow_output_dir / "porosity_layer.png"
            ))

        stats = {
            "water_content": self._array_stats(water_content),
            "porosity": self._array_stats(porosity) if porosity is not None else None,
        }

        if water_content is None and porosity is None:
            raise ValueError("No MODFLOW outputs loaded. Check files or flags.")

        return {
            "model_directory": str(modflow_dir),
            "idomain_file": str(idomain_file),
            "timestep": timestep,
            "nlay": nlay,
            "model_name": model_name,
            "water_content_file": str(wc_path),
            "porosity_file": str(porosity_path) if porosity_path else None,
            "plots": plot_paths,
            "stats": stats,
            "output_dir": str(modflow_output_dir),
        }

    def _load_parflow(self, input_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        from PyHydroGeophysX.model_output.parflow_output import ParflowSaturation, ParflowPorosity, read_pfb

        parflow_dir = input_data.get("parflow_dir") or input_data.get("model_directory")
        uploaded_files = input_data.get("uploaded_files", {})
        
        # Check if we have direct PFB file uploads
        pfb_files = [f for f in uploaded_files.values() if f.lower().endswith('.pfb')]
        
        parflow_output_dir = output_dir / "parflow"
        parflow_output_dir.mkdir(parents=True, exist_ok=True)
        
        saturation = None
        saturation_path = None
        porosity = None
        porosity_path = None
        mask = None
        mask_path = None
        run_name = input_data.get("run_name")
        parflow_timestep = int(input_data.get("parflow_timestep", 0))
        
        # Strategy 1: Direct PFB file reading from uploaded files
        if pfb_files:
            self._log_execution(f"Found {len(pfb_files)} uploaded PFB file(s), using direct file reading")
            
            for pfb_path in pfb_files:
                pfb_name = os.path.basename(pfb_path).lower()
                try:
                    data = read_pfb(pfb_path)
                    # Replace ParFlow no-data values with NaN
                    data[data < -1e38] = np.nan
                    
                    if ".out.satur." in pfb_name or "satur" in pfb_name:
                        saturation = data
                        # Extract timestep from filename if present
                        import re
                        match = re.search(r'\.(\d+)\.pfb$', pfb_path)
                        if match:
                            parflow_timestep = int(match.group(1))
                        saturation_path = parflow_output_dir / f"saturation_t{parflow_timestep}.npy"
                        np.save(saturation_path, saturation)
                        self._log_execution(f"Loaded saturation from {pfb_name}, shape: {saturation.shape}")
                    elif ".out.porosity" in pfb_name or "porosity" in pfb_name:
                        porosity = data
                        porosity_path = parflow_output_dir / "porosity.npy"
                        np.save(porosity_path, porosity)
                        self._log_execution(f"Loaded porosity from {pfb_name}, shape: {porosity.shape}")
                    elif ".out.mask" in pfb_name or "mask" in pfb_name:
                        mask = data
                        mask_path = parflow_output_dir / "mask.npy"
                        np.save(mask_path, mask)
                        self._log_execution(f"Loaded mask from {pfb_name}, shape: {mask.shape}")
                    else:
                        # Unknown type, assume saturation if we don't have one yet
                        if saturation is None:
                            saturation = data
                            saturation_path = parflow_output_dir / f"saturation_t{parflow_timestep}.npy"
                            np.save(saturation_path, saturation)
                            self._log_execution(f"Loaded data from {pfb_name} as saturation, shape: {saturation.shape}")
                except Exception as exc:
                    self._log_execution(f"Failed to read PFB file {pfb_name}: {exc}", level="WARNING")
        
        # Strategy 2: Standard ParFlow directory structure
        elif parflow_dir:
            parflow_dir = Path(parflow_dir)
            if not parflow_dir.exists():
                raise FileNotFoundError(f"ParFlow directory not found: {parflow_dir}")

            load_saturation = input_data.get("load_saturation", True)
            load_porosity = input_data.get("load_porosity", True)
            load_mask = input_data.get("load_mask", True)

            run_name = run_name or self._infer_parflow_run_name(parflow_dir)
            if not run_name:
                raise ValueError("ParFlow run_name not provided and could not be inferred.")

            if load_saturation:
                try:
                    saturation_proc = ParflowSaturation(str(parflow_dir), run_name=run_name)
                    available = getattr(saturation_proc, "available_timesteps", [])
                    if available:
                        parflow_timestep = max(0, min(parflow_timestep, len(available) - 1))
                    saturation = saturation_proc.load_timestep(parflow_timestep)
                    saturation_path = parflow_output_dir / f"saturation_t{parflow_timestep}.npy"
                    np.save(saturation_path, saturation)
                except Exception as exc:
                    self._log_execution(f"ParFlow saturation load failed: {exc}", level="WARNING")

            if load_porosity:
                try:
                    porosity_proc = ParflowPorosity(str(parflow_dir), run_name=run_name)
                    porosity = porosity_proc.load_porosity()
                    porosity_path = parflow_output_dir / "porosity.npy"
                    np.save(porosity_path, porosity)
                except Exception as exc:
                    self._log_execution(f"ParFlow porosity load failed: {exc}", level="WARNING")

            if load_mask and porosity is not None:
                try:
                    porosity_proc = ParflowPorosity(str(parflow_dir), run_name=run_name)
                    mask = porosity_proc.load_mask()
                    mask_path = parflow_output_dir / "mask.npy"
                    np.save(mask_path, mask)
                except Exception as exc:
                    self._log_execution(f"ParFlow mask load failed: {exc}", level="WARNING")
        else:
            raise ValueError("No ParFlow directory or PFB files provided.")
        
        # Apply mask if available
        if mask is not None:
            if saturation is not None:
                saturation = np.where(mask == 0, np.nan, saturation)
            if porosity is not None:
                porosity = np.where(mask == 0, np.nan, porosity)

        plot_paths = {}
        if saturation is not None:
            nz = saturation.shape[0]
            plot_layer = int(input_data.get("plot_layer", max(0, min(nz - 1, nz // 2))))
            plot_paths.update(self._save_layer_plot(
                saturation, plot_layer, "ParFlow Saturation", parflow_output_dir / "saturation_layer.png"
            ))
        if porosity is not None:
            nz = porosity.shape[0]
            plot_layer = int(input_data.get("plot_layer", max(0, min(nz - 1, nz // 2))))
            plot_paths.update(self._save_layer_plot(
                porosity, plot_layer, "ParFlow Porosity", parflow_output_dir / "porosity_layer.png"
            ))

        stats = {
            "saturation": self._array_stats(saturation),
            "porosity": self._array_stats(porosity),
            "mask": self._array_stats(mask) if mask is not None else None,
        }

        if saturation is None and porosity is None:
            raise ValueError("No ParFlow outputs loaded. Check files or flags.")

        return {
            "model_directory": str(parflow_dir) if parflow_dir else "uploaded_files",
            "run_name": run_name,
            "timestep": parflow_timestep,
            "saturation_file": str(saturation_path) if saturation_path else None,
            "porosity_file": str(porosity_path) if porosity_path else None,
            "mask_file": str(mask_path) if mask_path else None,
            "plots": plot_paths,
            "stats": stats,
            "output_dir": str(parflow_output_dir),
        }

    def _infer_modflow_model_name(self, modflow_dir: Path) -> Optional[str]:
        nam_files = list(modflow_dir.glob("*.nam"))
        if len(nam_files) == 1:
            return nam_files[0].stem
        return None

    def _infer_parflow_run_name(self, parflow_dir: Path) -> Optional[str]:
        # Look for files like <run_name>.out.satur.00001.pfb
        for fname in os.listdir(parflow_dir):
            if ".out.satur." in fname and fname.endswith(".pfb"):
                return fname.split(".out.satur.")[0]
            if ".out.press." in fname and fname.endswith(".pfb"):
                return fname.split(".out.press.")[0]
        return None

    def _save_layer_plot(self, arr: np.ndarray, layer: int, title: str, out_path: Path) -> Dict[str, str]:
        if arr is None:
            return {}
        data = np.array(arr)
        if data.ndim < 2:
            return {}
        if data.ndim == 3:
            layer = max(0, min(data.shape[0] - 1, layer))
            slice_2d = data[layer]
        else:
            slice_2d = data

        plt.figure(figsize=(6, 4))
        plt.imshow(slice_2d, cmap="viridis")
        plt.title(f"{title} (Layer {layer})")
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return {out_path.stem: str(out_path)}

    def _array_stats(self, arr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        if arr is None:
            return None
        data = np.array(arr)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": float(np.nanmin(finite)),
            "max": float(np.nanmax(finite)),
            "mean": float(np.nanmean(finite)),
        }

    def _log_execution(self, message: str, level: str = "INFO"):
        print(f"[{self.name}] [{level}] {message}")
