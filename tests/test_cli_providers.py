"""CLI protocol / studio integration tests; never use a real account or network."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from PyHydroGeophysX.llm import cli_providers as cli
from PyHydroGeophysX.llm import cli_setup as setup
from PyHydroGeophysX.llm.providers import make_provider, PROVIDER_ORDER, DESKTOP_PROVIDER_ORDER


SPEC = [{"name": "navigate", "description": "Open module", "parameters": {
    "type": "object", "properties": {"module": {"type": "string"}},
    "required": ["module"]}}]


@pytest.mark.parametrize("provider_id", ["codex_cli", "claude_code"])
def test_cli_proposal_round_trip(monkeypatch, provider_id):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-use")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-use")
    prompt_seen = []

    def run(command, **kwargs):
        assert "shell" not in kwargs
        assert kwargs["timeout"] > 0
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        prompt_seen.append(kwargs["input"])
        reply = {"content": "打开 ERT", "tool_calls": [
            {"name": "navigate", "arguments": '{"module":"ert"}'}]}
        if provider_id == "codex_cli":
            assert command[-1] == "-"
            assert command[command.index("--sandbox") + 1] == "read-only"
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text(json.dumps(reply), encoding="utf-8")
            stdout = "progress output does not enter the conversation"
        else:
            assert command[command.index("--tools") + 1] == ""
            assert "--strict-mcp-config" in command
            stdout = json.dumps({"structured_output": reply, "is_error": False})
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", run)
    provider = make_provider(provider_id, api_key="ignored")
    assert provider.available()[0]
    assert provider._api_key is None
    assert not provider.supports_vision()
    history = [{"role": "user", "content": "打开 ERT"},
               {"role": "tool", "id": "old", "name": "navigate",
                "content": {"status": "declined"}}]
    reply = provider.complete("Studio rules", history, SPEC)
    assert reply["tool_calls"][0]["arguments"] == {"module": "ert"}
    assert reply["tool_calls"][0]["id"].startswith("cli_")
    assert "declined" in prompt_seen[0]
    assert "打开 ERT" in prompt_seen[0]
    assert provider_id in DESKTOP_PROVIDER_ORDER and provider_id not in PROVIDER_ORDER


@pytest.mark.parametrize("calls", [
    [{"name": "Bash", "arguments": "{}"}],
    [{"name": "navigate", "arguments": "[]"}],
    [{"name": "navigate", "arguments": "broken"}],
    "not a list",
])
def test_invalid_proposals_are_rejected(calls):
    with pytest.raises((ValueError, TypeError)):
        cli._normalise_reply({"content": "", "tool_calls": calls}, SPEC)


def test_missing_cli_is_actionable(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    ready, reason = make_provider("codex_cli").available()
    monkeypatch.setattr(setup, "find_executable", lambda _: None)
    ready, reason = make_provider("codex_cli").available()
    assert not ready and "automatically" in reason


def test_npm_windows_launcher_uses_node(monkeypatch, tmp_path):
    entry = tmp_path / "node_modules/@openai/codex/bin/codex.js"
    entry.parent.mkdir(parents=True)
    entry.touch()
    wrapper = tmp_path / "codex.cmd"
    monkeypatch.setattr(cli.shutil, "which", lambda name:
                        "node.exe" if name == "node" else str(wrapper))
    monkeypatch.setattr(setup, "find_node", lambda _: "node.exe")
    assert make_provider("codex_cli")._command() == ["node.exe", str(entry)]


@pytest.mark.parametrize("failure", ["timeout", "login", "structured"])
def test_cli_errors(monkeypatch, failure):
    provider = make_provider("claude_code")
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/bin/claude")

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "login":
            return SimpleNamespace(returncode=1, stdout="", stderr="Not logged in")
        return SimpleNamespace(returncode=0, stdout='{"is_error":false,"result":"plain"}', stderr="")

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises((RuntimeError, ValueError), match={
        "timeout": "timed out", "login": "Not logged in", "structured": "structured_output"}[failure]):
        provider.complete("rules", [], [])


def test_workflow_model_access_without_key(monkeypatch):
    from PyHydroGeophysX.agents.context_input_agent import ContextInputAgent
    from PyHydroGeophysX.agents.runtime.entry import make_ask
    seen = []

    def complete(self, system, messages, specs, max_tokens=0):
        seen.append(messages)
        return {"content": '{"why":"ready","tool":"next"}', "tool_calls": []}

    monkeypatch.setattr(cli.CliProvider, "complete", complete)
    for provider_id in ("codex_cli", "claude_code"):
        agent = ContextInputAgent(llm_provider=provider_id)
        assert agent.api_key is None and agent.llm_enabled
        assert '"tool":"next"' in agent.query_llm("choose a step")
        ask = make_ask(None, None, provider_id)
        assert ask is not None
        assert '"tool":"next"' in ask("choose a step")
    assert len(seen) == 4


def test_qt_cli_settings(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AssistantChatPanel
    from PyHydroGeophysX.qt_apps.agent.controller import StudioController
    from PyHydroGeophysX.agents.assistants import get_assistant

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(setup, "check_login", lambda provider: {
        "logged_in": True, "message": "Signed in. Ready to chat."})
    panel = AssistantChatPanel(StudioController(None), provider=make_provider("codex_cli"))
    from PySide6.QtTest import QTest

    def wait_for_setup():
        for _ in range(100):
            if not panel._cli_workers:
                return
            QTest.qWait(20)
        pytest.fail("CLI setup did not finish")

    wait_for_setup()
    assert panel._key_edit.isHidden()
    assert not panel._login_help.isHidden()
    assert panel._provider_combo.findData("claude_code") >= 0
    assert "codex_cli" in get_assistant("aquah").providers
    submitted = []
    monkeypatch.setattr(panel._controller, "run_to_report", lambda text, settings, *a, **kw:
                        submitted.append(settings) or "Workflow started.")
    panel._execution_mode.setCurrentIndex(panel._execution_mode.findData("auto"))
    panel._input.setPlainText("Run ERT inversion")
    panel._on_send()
    assert submitted[0]["provider"] == "codex_cli"
    assert submitted[0]["api_key"] is None
    panel._on_workflow_finished("Done")
    panel._provider_combo.setCurrentIndex(panel._provider_combo.findData("claude_code"))
    wait_for_setup()
    assert not panel._reasoning.isEnabled()
    panel._input.setPlainText("Run ERT inversion")
    panel._on_send()
    assert submitted[1]["provider"] == "claude_code"
    panel._on_workflow_finished("Done")
    panel._provider_combo.setCurrentIndex(panel._provider_combo.findData("openai"))
    assert not panel._key_edit.isHidden()
    assert panel._login_help.isHidden()
    panel.close()
    app.processEvents()
