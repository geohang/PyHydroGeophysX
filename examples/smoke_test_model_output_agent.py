"""
Smoke test for hydrological model output workflow.

Runs MODFLOW and ParFlow model output loading (and resistivity conversion)
using the unified workflow dispatcher.
"""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyHydroGeophysX.agents import BaseAgent  # noqa: E402


def _can_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


def run_case(name: str, config: dict, output_dir: Path):
    print(f"\n=== Running case: {name} ===")
    results, execution_plan, interpretation, report_files = BaseAgent.run_unified_agent_workflow(
        workflow_config=config,
        api_key="",
        llm_model="gpt-4o-mini",
        llm_provider="openai",
        output_dir=output_dir,
    )
    print(f"[{name}] status: {results.get('status')}")
    if results.get("warnings"):
        print(f"[{name}] warnings: {results.get('warnings')}")
    if interpretation:
        print(f"[{name}] interpretation: {interpretation[:200]}")
    if report_files:
        print(f"[{name}] report files:")
        for k, v in report_files.items():
            print(f"  - {k}: {v}")
    
    # Print detailed stats for model outputs
    for model_key in ["modflow", "parflow"]:
        model_data = results.get(model_key)
        if model_data:
            print(f"\n  [{model_key.upper()} Output Details]")
            if model_data.get("saturation_stats"):
                print(f"    saturation_stats: {model_data['saturation_stats']}")
            if model_data.get("water_content_stats"):
                print(f"    water_content_stats: {model_data['water_content_stats']}")
            if model_data.get("porosity_stats"):
                print(f"    porosity_stats: {model_data['porosity_stats']}")
            if model_data.get("resistivity_stats"):
                print(f"    resistivity_stats: {model_data['resistivity_stats']}")
            if model_data.get("velocity_stats"):
                print(f"    velocity_stats: {model_data['velocity_stats']}")


def main():
    output_dir = ROOT / "examples" / "results" / "smoke_model_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    modflow_dir = ROOT / "examples" / "data" / "modflow"
    parflow_dir = ROOT / "examples" / "data" / "parflow" / "test2"

    flopy_ok = _can_import("flopy")
    # Note: parflow package is no longer required - we have a standalone PFB reader

    # MODFLOW-only case (skip porosity if flopy missing)
    modflow_petrophysics = {"rho_sat": 541, "n": 1.24, "v_dry": 3500, "v_sat": 4500, "velocity_model": "linear"}
    if not flopy_ok:
        # Fallback porosity for conversion testing when flopy is missing
        modflow_petrophysics["porosity"] = 0.3

    modflow_config = {
        "hydro_model": "modflow",
        "modflow_dir": str(modflow_dir),
        "idomain_file": str(modflow_dir / "id.txt"),
        "model_name": "TLnewtest2sfb2",
        "timestep": 1,
        "nlay": 3,
        "load_water_content": True,
        "load_porosity": flopy_ok,
        "convert_to_resistivity": True,
        "convert_to_velocity": True,
        "petrophysical_params": modflow_petrophysics,
        "user_request": "Load MODFLOW outputs and convert to resistivity and velocity."
    }
    run_case("modflow", modflow_config, output_dir)

    # ParFlow-only case (now works without parflow package using standalone PFB reader)
    parflow_petrophysics = {"rho_sat": 541, "n": 1.24, "porosity": 0.3, "v_dry": 3500, "v_sat": 4500, "velocity_model": "wyllie"}

    parflow_config = {
        "hydro_model": "parflow",
        "parflow_dir": str(parflow_dir),
        "run_name": "test2",
        "parflow_timestep": 0,
        "load_saturation": True,
        "load_porosity": True,
        "load_mask": True,
        "convert_to_resistivity": True,
        "convert_to_velocity": True,
        "petrophysical_params": parflow_petrophysics,
        "user_request": "Load ParFlow outputs and convert to resistivity and velocity."
    }
    # ParFlow test now runs regardless of parflow package installation
    run_case("parflow", parflow_config, output_dir)


if __name__ == "__main__":
    main()
