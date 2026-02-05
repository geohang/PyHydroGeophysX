"""
Quick smoke tests for PyHydroGeophysX agent workflows (non-Streamlit).

Run:
    conda activate pg  # or your env
    python examples/smoke_test_workflows.py
"""
# sphinx_gallery_thumbnail_path = 'auto_examples/images/smoke_test_workflows_fig_01.png'

from pathlib import Path
import sys
import traceback

# Allow importing the package from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyHydroGeophysX.agents import BaseAgent  # noqa: E402


def run_case(name: str, config: dict, output_dir: Path, api_key: str = ""):
    print(f"\n=== Running case: {name} ===")
    try:
        results, execution_plan, interpretation, report_files = BaseAgent.run_unified_agent_workflow(
            workflow_config=config,
            api_key=api_key,
            llm_model="gpt-4o-mini",
            llm_provider="openai",
            output_dir=output_dir,
        )
        print(f"[{name}] status: {results.get('status')}")
        if interpretation:
            print(f"[{name}] interpretation snippet: {interpretation[:200]}")
        if report_files:
            print(f"[{name}] report artifacts:")
            for k, v in report_files.items():
                print(f"  - {k}: {v}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}] FAILED: {exc}")
        traceback.print_exc()


def main():
    base_out = Path("examples/results/smoke_tests")
    base_out.mkdir(parents=True, exist_ok=True)

    # Case 1: Standard ERT with petrophysics
    case1 = {
        "workflow_type": "direct_ert",
        "ert_file": "examples/data/ERT/DAS/20171105_1418.Data",
        "electrode_file": "examples/data/ERT/DAS/electrodes.dat",
        "instrument": "DAS-1",
        "convert_to_water_content": True,
        "petrophysical_params": {"rho_sat": 541, "porosity": 0.37, "n": 1.24},
        "inversion_params": {"lambda": 15.0, "max_iterations": 8, "method": "cgls"},
    }

    # Case 2: ERT imaging only (skip water content)
    case2 = {
        "workflow_type": "direct_ert",
        "ert_file": "examples/data/ERT/DAS/20171105_1418.Data",
        "electrode_file": "examples/data/ERT/DAS/electrodes.dat",
        "instrument": "DAS-1",
        "convert_to_water_content": False,
        "inversion_params": {"lambda": 12.0, "max_iterations": 6, "method": "cgls"},
    }

    # Case 3: Data fusion (structure-constrained) without petrophysics
    case3 = {
        "workflow_type": "data_fusion",
        "fusion_pattern": "structure_constraint",
        "methods": ["seismic", "ert"],
        "seismic_file": "examples/data/Seismic/srtfieldline2.dat",
        "velocity_threshold": 1000,
        "ert_file": "examples/data/ERT/Bert/fielddataline2.dat",
        "convert_to_water_content": False,
        "output_dir": str(base_out / "case3_fusion"),
    }

    run_case("direct_ert_with_petro", case1, base_out / "case1_direct_petro")
    run_case("direct_ert_imaging_only", case2, base_out / "case2_direct_image")
    run_case("data_fusion_no_petro", case3, base_out / "case3_fusion")


if __name__ == "__main__":
    main()
