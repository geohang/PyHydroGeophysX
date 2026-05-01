from pathlib import Path
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="PyHydroGeophysX agent imports require numpy.",
)


def _agent_exports():
    from PyHydroGeophysX.agents import (
        AgentCoordinator,
        AgentResult,
        ContextInputAgent,
        ERTLoaderAgent,
        InversionEvaluationAgent,
    )

    return AgentCoordinator, AgentResult, ContextInputAgent, ERTLoaderAgent, InversionEvaluationAgent


def test_dry_run_no_llm_calls(monkeypatch, tmp_path):
    AgentCoordinator, _, ContextInputAgent, _, _ = _agent_exports()
    calls = {"count": 0}

    def fail_if_called(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("dry_run must not call an LLM")

    monkeypatch.setattr(ContextInputAgent, "query_llm", fail_if_called, raising=False)

    coordinator = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
    result = coordinator.execute_workflow(
        {"user_request": "Run ERT inversion on missing.ohm"},
        dry_run=True,
    )

    assert calls["count"] == 0
    assert result["status"] == "failed"
    assert "missing.ohm" in result.get("error_fix_hint", "") or result.get("validation_errors")


def test_dry_run_raw_segy_routes_to_preprocessing(tmp_path):
    AgentCoordinator, _, _, _, _ = _agent_exports()
    segy_path = Path("example/example/example_data.sgy")
    if not segy_path.exists():
        pytest.skip("Bundled raw SEG-Y example is not present.")

    coordinator = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
    result = coordinator.execute_workflow(
        {"user_request": f"Run seismic refraction tomography on {segy_path.as_posix()}"},
        dry_run=True,
    )

    config = result.get("workflow_config")
    steps = [step["step"] for step in result.get("execution_plan")]
    assert result["status"] == "success"
    assert config["raw_seismic_file"].endswith("example_data.sgy")
    assert "instrument" not in config
    assert any("Read raw SEG-Y" in step for step in steps)
    assert any("Export PyGIMLi travel-time data" in step for step in steps)


def test_agent_result_dict_compat():
    _, AgentResult, _, _, _ = _agent_exports()
    result = AgentResult(status="success", summary="Loaded data.", data={"value": 2})

    assert result["status"] == "success"
    assert result.get("value") == 2
    assert "value" in result.keys()

    legacy = AgentResult.from_dict(
        {"status": "needs_improvement", "interpretation": "Check manually.", "value": 3}
    )
    assert legacy["status"] == "needs_review"
    assert legacy["value"] == 3
    assert legacy["llm_interpretation"] == "Check manually."

    serialized = AgentResult.from_dict({"status": "success", "summary": "ok", "data": {"nested": 4}})
    assert serialized["nested"] == 4


def test_llm_usage_ledger_records_cost():
    _, _, _, _, _ = _agent_exports()
    from PyHydroGeophysX.agents import BaseAgent

    class DummyAgent(BaseAgent):
        def execute(self, input_data):
            return {}

    agent = DummyAgent("test_agent", api_key=None)
    agent.llm_provider = "openai"
    agent.model = "gpt-4o-mini"

    agent._record_llm_usage("prompt text", "completion text", prompt_tokens=10, completion_tokens=5)

    assert agent.llm_usage_ledger[0]["prompt_tokens"] == 10
    assert agent.llm_usage_ledger[0]["completion_tokens"] == 5
    assert agent.llm_usage_ledger[0]["cost_estimate_usd"] is not None


def test_nl_parse_confirm(monkeypatch):
    _, _, ContextInputAgent, _, _ = _agent_exports()
    agent = ContextInputAgent(api_key=None)

    monkeypatch.setattr(agent, "parse_request", lambda request, available_data=None: {"inversion_mode": "standard"})
    monkeypatch.setattr(agent, "explain_config", lambda config: "Parsed config.")

    result = agent.execute({"user_request": "Run ERT inversion"})

    assert result["status"] == "needs_review"
    assert "data_file or ert_file" in result["error_fix_hint"]
    assert "instrument" in result["error_fix_hint"]


def test_file_validation_missing(tmp_path):
    _, _, _, ERTLoaderAgent, _ = _agent_exports()
    missing_file = tmp_path / "does_not_exist.ohm"
    result = ERTLoaderAgent(api_key=None).execute({"data_file": str(missing_file)})

    assert result["status"] == "failed"
    assert str(missing_file.resolve()) in result["error_fix_hint"]


def test_quality_loop_transparent(monkeypatch):
    _, _, _, _, InversionEvaluationAgent = _agent_exports()
    agent = InversionEvaluationAgent(api_key=None)
    progress_events = []

    def fake_evaluate(results, params, quality_threshold=70):
        return {
            "quality_score": 50.0,
            "is_acceptable": False,
            "metrics": {"data_fit": {"final_chi2": 2.0}},
            "recommendations": ["Increase damping."],
        }

    def fake_adjust(params, metrics, recommendations):
        adjusted = dict(params)
        adjusted["lambda"] = params.get("lambda", 20) + 10
        return adjusted

    monkeypatch.setattr(agent, "_evaluate_quality", fake_evaluate)
    monkeypatch.setattr(agent, "_adjust_parameters", fake_adjust)
    monkeypatch.setattr(agent, "_rerun_inversion", lambda input_data, params: {"status": "success"})

    result = agent.execute(
        {
            "inversion_results": {"status": "success"},
            "inversion_params": {"lambda": 20},
            "max_attempts": 2,
            "quality_threshold": 90,
            "progress_callback": lambda *args: progress_events.append(args),
        }
    )

    assert result["status"] == "needs_review"
    assert "Attempt 2/2" in result["transparent_log"][-1]
    assert progress_events


def test_demo_mode_app():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[2] / "examples" / "app_geophysics_workflow.py"

    app = streamlit_testing.AppTest.from_file(str(app_path))
    app.run(timeout=30)

    assert not app.exception
    page_text = "\n".join(
        str(element.value)
        for collection in [
            app.markdown,
            app.warning,
            app.success,
            app.caption,
            app.subheader,
            app.metric,
        ]
        for element in collection
    )
    assert "Describe your workflow" in page_text
    assert "Raw seismic data to SRT" in page_text
    assert "Local Deployment" in page_text
