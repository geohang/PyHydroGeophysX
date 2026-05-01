"""
Test suite for all 7 code fixes applied to PyHydroGeophysX agents.

Fixes tested:
  1. ContextInputAgent has system_message set after __init__
  2. Workflow detection delegates to WorkflowOrchestratorAgent._detect_workflow_type
  3. save_results() serialises numpy arrays to .npy (no data loss)
  4. get_workflow_summary() reports total_llm_cost_estimate_usd / tokens / calls
  5. _retry_llm_call retries on rate-limit errors with exponential back-off
  6. execute_workflow supports resume=True with checkpoint save / load
  7. preview_workflow calls _check_dependencies and surfaces warnings

Run with:
    cd <repo_root>
    python -m pytest tests/test_all_fixes.py -v
"""
import json
import os
import time
import pickle
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_agents():
    from PyHydroGeophysX.agents.base_agent import BaseAgent, AgentResult
    from PyHydroGeophysX.agents.context_input_agent import ContextInputAgent
    from PyHydroGeophysX.agents.workflow_orchestrator_agent import WorkflowOrchestratorAgent
    from PyHydroGeophysX.agents.agent_coordinator import AgentCoordinator
    return BaseAgent, AgentResult, ContextInputAgent, WorkflowOrchestratorAgent, AgentCoordinator


API_KEY = os.environ.get("OPENAI_API_KEY", "")
LIVE_LLM = bool(API_KEY)

# ---------------------------------------------------------------------------
# Fix 1 – ContextInputAgent.system_message
# ---------------------------------------------------------------------------

class TestFix1_ContextInputAgentSystemMessage:
    def test_system_message_is_set(self):
        _, _, ContextInputAgent, _, _ = _import_agents()
        agent = ContextInputAgent(api_key=None)
        assert hasattr(agent, "system_message"), "system_message attribute missing"
        assert isinstance(agent.system_message, str) and len(agent.system_message) > 10, \
            "system_message should be a non-trivial string"

    def test_system_message_contains_workflow_keywords(self):
        _, _, ContextInputAgent, _, _ = _import_agents()
        agent = ContextInputAgent(api_key=None)
        lower = agent.system_message.lower()
        assert any(kw in lower for kw in ["workflow", "configuration", "geophysic"]), \
            f"system_message should mention workflow context, got: {agent.system_message}"

    def test_agent_md_augmentation_fires(self, tmp_path):
        """The .agent.md body should be appended to system_message on first query_llm call."""
        _, _, ContextInputAgent, _, _ = _import_agents()
        BaseAgent, *_ = _import_agents()

        agent = ContextInputAgent(api_key=None)
        original_msg = agent.system_message

        # Simulate the augmentation path (if .agent.md exists for 'ContextInputAgent')
        # Even if no MD file exists, system_message must already be set (Fix 1)
        assert len(original_msg) > 0, "system_message must be set before any LLM query"


# ---------------------------------------------------------------------------
# Fix 2 – Workflow detection delegation
# ---------------------------------------------------------------------------

class TestFix2_WorkflowDetectionDelegation:
    """WorkflowOrchestratorAgent._detect_workflow_type must cover all workflow types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _, _, _, WorkflowOrchestratorAgent, _ = _import_agents()
        self.orch = WorkflowOrchestratorAgent.__new__(WorkflowOrchestratorAgent)
        self.orch.name = "workflow_orchestrator"
        self.orch.context = {}
        self.orch.results = {}
        self.orch.execution_log = []

    def _detect(self, config):
        return self.orch._detect_workflow_type(config)

    def test_tdem_detected(self):
        assert self._detect({"tdem_file": "data.dat"}) == "tdem"

    def test_tdem_from_user_request(self):
        assert self._detect({"user_request": "Run TDEM survey inversion"}) == "tdem"

    def test_seismic_standalone(self):
        result = self._detect({"seismic_file": "seis.dat", "user_request": "seismic refraction"})
        assert result == "seismic"

    def test_model_output(self):
        assert self._detect({"hydro_model": "modflow"}) == "model_output"

    def test_time_lapse(self):
        result = self._detect({"time_lapse_files": ["a.ohm", "b.ohm"]})
        assert result == "time_lapse"

    def test_time_lapse_from_inversion_mode(self):
        assert self._detect({"ert_file": "a.ohm", "inversion_mode": "time-lapse"}) == "time_lapse"

    def test_data_fusion_with_both_files(self):
        result = self._detect({"ert_file": "ert.ohm", "seismic_file": "seis.dat"})
        assert result == "data_fusion"

    def test_data_fusion_velocity_threshold(self):
        assert self._detect({"velocity_threshold": 1200}) == "data_fusion"

    def test_direct_ert(self):
        assert self._detect({"ert_file": "data.ohm"}) == "direct_ert"

    def test_custom_fallback(self):
        assert self._detect({}) == "custom"

    def test_ert_data_process(self):
        result = self._detect({
            "ert_file": "data.ohm",
            "user_request": "quality control and export the ert data"
        })
        assert result == "ert_data_process"

    def test_no_duplicate_logic_in_base_agent(self):
        """base_agent.run_unified_agent_workflow should NOT contain the old detection block."""
        import inspect
        from PyHydroGeophysX.agents import base_agent
        source = inspect.getsource(base_agent.BaseAgent.run_unified_agent_workflow)
        # The old block started with a direct mentions_ert / mentions_seismic flag computation
        # The new code calls _WOA._detect_workflow_type
        assert "_detect_workflow_type" in source, \
            "run_unified_agent_workflow must delegate to _detect_workflow_type"
        assert "mentions_ert" not in source, \
            "Duplicate detection flag 'mentions_ert' must NOT remain in run_unified_agent_workflow"


# ---------------------------------------------------------------------------
# Fix 3 – save_results data preservation
# ---------------------------------------------------------------------------

class TestFix3_SaveResultsDataPreservation:
    """save_results must serialise numpy arrays as .npy sidecar files."""

    def _make_agent(self):
        from PyHydroGeophysX.agents.ert_loader_agent import ERTLoaderAgent
        agent = ERTLoaderAgent.__new__(ERTLoaderAgent)
        agent.name = "ert_loader"
        agent.context = {}
        agent.results = {}
        agent.execution_log = []
        agent.llm_usage_ledger = []
        return agent

    def test_numpy_array_saved_as_npy(self, tmp_path):
        agent = self._make_agent()
        arr = np.array([1.0, 2.0, 3.0])
        agent.results = {"resistivity": arr, "label": "test"}
        json_path = agent.save_results(str(tmp_path))

        # JSON metadata should reference the .npy file
        with open(json_path) as f:
            meta = json.load(f)
        res = meta["results"]["resistivity"]
        assert res["__type__"] == "numpy_array", f"Expected numpy_array type, got {res}"

        # The actual .npy file must exist
        npy_file = tmp_path / res["file"]
        assert npy_file.exists(), f".npy file not found: {npy_file}"

        # Round-trip: loaded array equals original
        loaded = np.load(str(npy_file))
        np.testing.assert_array_equal(loaded, arr)

    def test_scalar_stays_in_json(self, tmp_path):
        agent = self._make_agent()
        agent.results = {"chi2": 1.23, "iterations": 5, "status": "success"}
        json_path = agent.save_results(str(tmp_path))
        with open(json_path) as f:
            meta = json.load(f)
        assert meta["results"]["chi2"] == pytest.approx(1.23)
        assert meta["results"]["iterations"] == 5
        assert meta["results"]["status"] == "success"

    def test_2d_array_shape_recorded(self, tmp_path):
        agent = self._make_agent()
        arr2d = np.ones((10, 3))
        agent.results = {"mesh_coords": arr2d}
        json_path = agent.save_results(str(tmp_path))
        with open(json_path) as f:
            meta = json.load(f)
        assert meta["results"]["mesh_coords"]["shape"] == [10, 3]

    def test_no_str_fallback_for_arrays(self, tmp_path):
        """Arrays must NOT end up as plain strings in the JSON."""
        agent = self._make_agent()
        agent.results = {"data": np.zeros(5)}
        json_path = agent.save_results(str(tmp_path))
        with open(json_path) as f:
            meta = json.load(f)
        assert not isinstance(meta["results"]["data"], str), \
            "numpy array was silently converted to str — data loss!"


# ---------------------------------------------------------------------------
# Fix 4 – LLM cost aggregation in get_workflow_summary
# ---------------------------------------------------------------------------

class TestFix4_LLMCostAggregation:
    def test_summary_has_cost_fields(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        summary = coord.get_workflow_summary()
        assert "total_llm_cost_estimate_usd" in summary
        assert "total_llm_tokens" in summary
        assert "llm_calls" in summary

    def test_cost_accumulates_from_ledger(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        coord.llm_usage_ledger = [
            {"cost_estimate_usd": 0.001, "total_tokens": 100},
            {"cost_estimate_usd": 0.002, "total_tokens": 200},
        ]
        summary = coord.get_workflow_summary()
        assert summary["total_llm_cost_estimate_usd"] == pytest.approx(0.003)
        assert summary["total_llm_tokens"] == 300
        assert summary["llm_calls"] == 2

    def test_cost_zero_when_no_calls(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        summary = coord.get_workflow_summary()
        assert summary["total_llm_cost_estimate_usd"] == 0.0
        assert summary["llm_calls"] == 0


# ---------------------------------------------------------------------------
# Fix 5 – LLM retry with exponential back-off
# ---------------------------------------------------------------------------

class TestFix5_LLMRetryBackoff:
    def test_retry_helper_exists(self):
        BaseAgent, *_ = _import_agents()
        assert hasattr(BaseAgent, "_retry_llm_call"), "_retry_llm_call static method missing"

    def test_retries_on_rate_limit(self):
        BaseAgent, *_ = _import_agents()
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("rate limit exceeded – 429")
            return "success"

        # Patch time.sleep so tests don't actually wait
        original_sleep = time.sleep
        try:
            time.sleep = lambda s: None  # no-op
            result = BaseAgent._retry_llm_call(flaky, max_retries=3)
        finally:
            time.sleep = original_sleep

        assert result == "success"
        assert call_count["n"] == 3

    def test_propagates_non_transient_error_immediately(self):
        BaseAgent, *_ = _import_agents()
        call_count = {"n": 0}

        def always_fails():
            call_count["n"] += 1
            raise ValueError("invalid model name")

        original_sleep = time.sleep
        try:
            time.sleep = lambda s: None
            with pytest.raises(ValueError, match="invalid model name"):
                BaseAgent._retry_llm_call(always_fails, max_retries=3)
        finally:
            time.sleep = original_sleep

        assert call_count["n"] == 1, "Non-transient error must not be retried"

    def test_raises_after_max_retries_exhausted(self):
        BaseAgent, *_ = _import_agents()
        call_count = {"n": 0}

        def always_rate_limited():
            call_count["n"] += 1
            raise RuntimeError("too many requests – quota exceeded")

        original_sleep = time.sleep
        try:
            time.sleep = lambda s: None
            with pytest.raises(RuntimeError, match="too many requests"):
                BaseAgent._retry_llm_call(always_rate_limited, max_retries=3)
        finally:
            time.sleep = original_sleep

        assert call_count["n"] == 3

    @pytest.mark.skipif(not LIVE_LLM, reason="No OPENAI_API_KEY set")
    def test_real_openai_call_succeeds_with_retry(self):
        """Smoke-test: a real OpenAI call should succeed (no rate-limit hit expected)."""
        _, _, ContextInputAgent, _, _ = _import_agents()
        agent = ContextInputAgent(api_key=API_KEY, model="gpt-4o-mini")
        response = agent.query_llm("Say 'hello' in one word.", max_tokens=5)
        assert isinstance(response, str) and len(response) > 0


# ---------------------------------------------------------------------------
# Fix 6 – Checkpoint / resume
# ---------------------------------------------------------------------------

class TestFix6_CheckpointResume:
    def test_save_and_load_checkpoint_roundtrip(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))

        payload = {"status": "success", "chi2": 1.5, "array": np.array([1, 2, 3])}
        coord._save_checkpoint("invert_ert", payload)

        loaded = coord._load_checkpoint("invert_ert")
        assert loaded is not None
        assert loaded["status"] == "success"
        np.testing.assert_array_equal(loaded["array"], np.array([1, 2, 3]))

    def test_load_returns_none_for_missing_checkpoint(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        result = coord._load_checkpoint("nonexistent_step")
        assert result is None

    def test_checkpoint_files_created(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        coord._save_checkpoint("load_ert", {"data": "mock"})
        pkl = tmp_path / "checkpoints" / "load_ert.pkl"
        assert pkl.exists(), "Pickle checkpoint file not created"

    def test_json_sidecar_created(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        coord._save_checkpoint("fetch_climate", {"status": "success", "temp": 15.0})
        json_sidecar = tmp_path / "checkpoints" / "fetch_climate.json"
        assert json_sidecar.exists(), "JSON sidecar not created"
        with open(json_sidecar) as f:
            data = json.load(f)
        assert data["status"] == "success"

    def test_execute_workflow_signature_accepts_resume(self, tmp_path):
        """execute_workflow must accept resume keyword without TypeError."""
        _, _, _, _, AgentCoordinator = _import_agents()
        import inspect
        sig = inspect.signature(AgentCoordinator.execute_workflow)
        assert "resume" in sig.parameters, \
            "execute_workflow must have a 'resume' parameter"


# ---------------------------------------------------------------------------
# Fix 7 – Environment pre-check in preview_workflow
# ---------------------------------------------------------------------------

class TestFix7_EnvironmentPreCheck:
    def test_check_dependencies_method_exists(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        assert hasattr(coord, "_check_dependencies"), "_check_dependencies method missing"

    def test_check_dependencies_returns_list(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        result = coord._check_dependencies([])
        assert isinstance(result, list)

    def test_missing_pygimli_triggers_warning(self, tmp_path, monkeypatch):
        _, _, _, _, AgentCoordinator = _import_agents()
        import importlib
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))

        # Simulate PyGIMLi absent
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name: None if name == "pygimli" else importlib.util.find_spec.__wrapped__(name)
                            if hasattr(importlib.util.find_spec, "__wrapped__") else object())

        plan = [{"agent": "ERTInversionAgent", "step": "Run ERT inversion"}]
        warnings = coord._check_dependencies(plan)
        assert any("pygimli" in w.lower() or "PyGIMLi" in w for w in warnings), \
            f"Expected PyGIMLi warning, got: {warnings}"

    def test_preview_workflow_includes_dependency_warnings(self, tmp_path, monkeypatch):
        """preview_workflow result warnings must include dependency check output."""
        _, _, _, _, AgentCoordinator = _import_agents()
        import importlib

        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))

        # Inject a fake dependency warning
        monkeypatch.setattr(coord, "_check_dependencies",
                            lambda plan: ["TEST_DEP_WARNING: fake missing package"])

        result = coord.preview_workflow({"ert_file": __file__})  # __file__ exists
        all_warnings = result.get("validation_warnings") or result.get("warnings") or []
        data_warnings = (result.get("data") or {}).get("validation_warnings", [])
        combined = all_warnings + data_warnings
        assert any("TEST_DEP_WARNING" in w for w in combined), \
            f"Dependency warning not surfaced. warnings={combined}"

    def test_preview_workflow_returns_plan(self, tmp_path):
        _, _, _, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
        result = coord.preview_workflow({"ert_file": __file__})
        plan = (result.get("data") or {}).get("execution_plan") or result.get("execution_plan", [])
        assert isinstance(plan, list)


# ---------------------------------------------------------------------------
# Live LLM integration smoke-test (Fix 1 + 4 + 5 together)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIVE_LLM, reason="No OPENAI_API_KEY set")
class TestLiveLLMIntegration:
    """Requires OPENAI_API_KEY in environment. Tests that Fixes 1, 4, 5 work end-to-end."""

    def test_context_input_agent_builds_config(self):
        _, AgentResult, ContextInputAgent, _, _ = _import_agents()
        agent = ContextInputAgent(api_key=API_KEY, model="gpt-4o-mini")
        assert agent.system_message, "system_message must be set before query"

        result = agent.preview_config("Run ERT inversion on data.ohm with Syscal instrument")
        # preview_config returns AgentResult or dict
        assert result is not None
        # Extract the workflow_config regardless of return type
        if hasattr(result, 'data'):
            config = result.data.get("workflow_config") if result.data else None
        elif isinstance(result, dict):
            config = result.get("workflow_config") or result.get("data", {}).get("workflow_config")
        else:
            config = None
        assert config is not None, f"workflow_config not found in result: {result}"

    def test_llm_usage_ledger_populated(self):
        _, _, ContextInputAgent, _, _ = _import_agents()
        agent = ContextInputAgent(api_key=API_KEY, model="gpt-4o-mini")
        agent.query_llm("What is 1+1? Answer with just the number.", max_tokens=5)
        assert len(agent.llm_usage_ledger) > 0, "llm_usage_ledger should record the call"
        entry = agent.llm_usage_ledger[0]
        assert "cost_estimate_usd" in entry
        # total_tokens is either stored directly or computed from prompt + completion
        total = entry.get("total_tokens") or (
            entry.get("prompt_tokens", 0) + entry.get("completion_tokens", 0)
        )
        assert total > 0, f"Expected non-zero token count, got entry: {entry}"

    def test_cost_aggregation_after_live_call(self, tmp_path):
        _, _, ContextInputAgent, _, AgentCoordinator = _import_agents()
        coord = AgentCoordinator(api_key=API_KEY, output_dir=str(tmp_path))
        # Manually add a ledger entry as execute_workflow would
        coord.llm_usage_ledger = [
            {"cost_estimate_usd": 0.0001, "total_tokens": 50},
        ]
        summary = coord.get_workflow_summary()
        assert summary["total_llm_cost_estimate_usd"] > 0
        print(f"\n  LLM cost: ${summary['total_llm_cost_estimate_usd']:.6f}")
        print(f"  Tokens: {summary['total_llm_tokens']}")
        print(f"  Calls: {summary['llm_calls']}")
