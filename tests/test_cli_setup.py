"""Automatic local CLI discovery, verified install and subscription login checks."""
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from PyHydroGeophysX.llm import cli_setup as setup
from PyHydroGeophysX.llm.providers import make_provider


@pytest.fixture
def isolated_install(monkeypatch, tmp_path):
    monkeypatch.setenv("PHGX_CLI_DIR", str(tmp_path / "managed"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.delenv("PHGX_CODEX_CLI", raising=False)
    monkeypatch.delenv("PHGX_CLAUDE_CLI", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(setup.shutil, "which", lambda _: None)
    return tmp_path


def test_managed_install_is_discovered_on_restart(isolated_install):
    provider = make_provider("codex_cli")
    executable = setup.managed_executable(provider)
    executable.parent.mkdir(parents=True)
    executable.touch(mode=0o700)
    assert setup.find_executable(provider) == str(executable)
    assert provider._command() == [str(executable)]


@pytest.mark.skipif(setup.os.name != "nt", reason="Windows desktop installation")
def test_desktop_codex_is_found_without_path(isolated_install):
    executable = isolated_install / "local/OpenAI/Codex/bin/bundled/codex.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert make_provider("codex_cli")._command() == [str(executable)]


@pytest.mark.parametrize("provider_id", ["codex_cli", "claude_code"])
@pytest.mark.parametrize("valid_checksum", [True, False])
def test_installs_only_verified_official_binary(monkeypatch, isolated_install, provider_id, valid_checksum):
    provider = make_provider(provider_id)
    payload = b"verified-native-cli"
    checksum = hashlib.sha256(payload).hexdigest() if valid_checksum else "0" * 64
    monkeypatch.setattr(setup, "_platform_names", lambda: ("x86_64-pc-windows-msvc.exe", "win32-x64"))
    monkeypatch.setattr(setup.platform, "system", lambda: "Windows")
    target = setup.managed_executable(provider)
    url = "https://github.com/openai/codex/releases/download/rust-v1/codex-x86_64-pc-windows-msvc.exe"

    def read(source):
        if source == setup.CODEX_RELEASE:
            return json.dumps({"assets": [{"name": "codex-x86_64-pc-windows-msvc.exe",
                                           "digest": "sha256:" + checksum,
                                           "browser_download_url": url}]}).encode()
        if source.endswith("/latest"):
            return b"2.1.268"
        return json.dumps({"platforms": {"win32-x64": {"checksum": checksum}}}).encode()

    def open_download(request, **kwargs):
        assert request.full_url.startswith(("https://github.com/openai/codex/releases/download/",
                                           setup.CLAUDE_DOWNLOADS + "/2.1.268/"))
        response = io.BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload))}
        return response

    monkeypatch.setattr(setup, "_read", read)
    monkeypatch.setattr(setup, "urlopen", open_download)
    checked = []
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kw:
                        checked.append(Path(command[0]).read_bytes()) or
                        SimpleNamespace(returncode=0, stderr="", stdout="CLI 1.0"))
    if not valid_checksum:
        with pytest.raises(RuntimeError, match="checksum did not match"):
            setup.install_cli(provider)
        assert not target.exists() and not checked
    else:
        assert setup.install_cli(provider) == str(target)
        assert target.read_bytes() == payload and checked == [payload]


def test_prepare_reuses_existing_cli(monkeypatch):
    provider = make_provider("codex_cli")
    monkeypatch.setattr(provider, "_command", lambda: ["existing-cli.exe"])
    monkeypatch.setattr(setup, "install_cli", lambda *a: pytest.fail("Unexpected reinstall"))
    monkeypatch.setattr(setup, "check_login", lambda _: {"logged_in": True})
    assert setup.prepare_cli(provider)["logged_in"]


def test_prepare_installs_missing_cli(monkeypatch):
    provider = make_provider("codex_cli")
    commands = iter([None, ["managed-cli.exe"]])

    def command():
        result = next(commands)
        if result is None:
            raise FileNotFoundError()
        return result

    monkeypatch.delenv("PHGX_CODEX_CLI", raising=False)
    monkeypatch.setattr(provider, "_command", command)
    installed = []
    monkeypatch.setattr(setup, "install_cli", lambda p, progress: installed.append(p.id))
    monkeypatch.setattr(setup, "check_login", lambda _: {"logged_in": False})
    assert not setup.prepare_cli(provider)["logged_in"]
    assert installed == ["codex_cli"]


@pytest.mark.parametrize("provider_id,stdout,stderr,code,expected", [
    ("codex_cli", "", "Logged in using ChatGPT", 0, True),
    ("codex_cli", "", "Logged in using an API key", 0, False),
    ("codex_cli", "", "Not logged in", 1, False),
    ("claude_code", '{"authMethod":"claude.ai"}', "", 0, True),
    ("claude_code", '{"authMethod":"api_key"}', "", 0, False),
    ("claude_code", '{"authMethod":"none"}', "", 1, False),
])
def test_check_subscription_login(monkeypatch, provider_id, stdout, stderr, code, expected):
    provider = make_provider(provider_id)
    monkeypatch.setattr(provider, "_command", lambda: ["cli"])
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr))
    assert setup.check_login(provider)["logged_in"] is expected


def test_qt_login_flow_and_retry(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AssistantChatPanel
    from PyHydroGeophysX.qt_apps.agent.controller import StudioController

    app = QApplication.instance() or QApplication([])
    provider = make_provider("codex_cli")
    monkeypatch.setattr(provider, "_command", lambda: ["existing.exe"])
    login = {"logged_in": False, "message": "Click Log In to continue."}
    monkeypatch.setattr(setup, "prepare_cli", lambda p, progress: dict(login))
    monkeypatch.setattr(setup, "check_login", lambda p: dict(login))
    panel = AssistantChatPanel(StudioController(None), provider=provider)

    def wait():
        for _ in range(100):
            if not panel._cli_workers:
                return
            QTest.qWait(20)
        pytest.fail("Setup worker did not finish")

    wait()
    assert not panel._login_btn.isHidden() and panel._login_btn.isEnabled()
    assert not panel._send_btn.isEnabled()
    assert panel._apply_btn.isHidden()
    process = SimpleNamespace(deleteLater=lambda: None)
    panel._cli_logins[provider.id] = process
    login.update(logged_in=True, message="Signed in. Ready to chat.")
    panel._cli_login_finished(provider, process, 0)
    wait()
    assert panel._send_btn.isEnabled()
    assert not panel._login_btn.isHidden() and panel._login_btn.isEnabled()
    assert "Already signed in" in panel._login_btn.toolTip()
    panel._set_busy(True)
    assert not panel._login_btn.isEnabled()
    panel._set_busy(False)
    assert panel._login_btn.isEnabled()
    panel._cli_setup_failed(provider.id, "Connection unavailable")
    assert panel._cli_retry_btn.isEnabled() and not panel._send_btn.isEnabled()
    panel._cli_retry_btn.click()
    wait()
    assert panel._send_btn.isEnabled()
    panel.close()
    app.processEvents()


def test_qt_login_button_runs_cli_and_enables_chat(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-reach-login")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AssistantChatPanel
    from PyHydroGeophysX.qt_apps.agent.controller import StudioController

    app = QApplication.instance() or QApplication([])
    provider = make_provider("codex_cli")
    script = ("import sys,os; assert sys.argv[-1]=='login'; "
              "assert 'OPENAI_API_KEY' not in os.environ; "
              "print('https://auth.openai.com/login?state=test')")
    monkeypatch.setattr(provider, "_command", lambda: [sys.executable, "-c", script])
    monkeypatch.setattr(setup, "prepare_cli", lambda p, progress: {
        "logged_in": False, "message": "Click Log In to continue."})
    monkeypatch.setattr(setup, "check_login", lambda p: {
        "logged_in": True, "message": "Signed in. Ready to chat."})
    panel = AssistantChatPanel(StudioController(None), provider=provider)
    for _ in range(100):
        if not panel._cli_workers:
            break
        QTest.qWait(20)
    panel._login_btn.click()
    assert panel._cli_states[provider.id]["state"] == "signing_in"
    assert not panel._cancel_login_btn.isHidden()
    for _ in range(200):
        if not panel._cli_logins and not panel._cli_workers:
            break
        QTest.qWait(20)
    assert panel._cli_states[provider.id]["state"] == "ready"
    assert panel._send_btn.isEnabled()
    assert panel._cancel_login_btn.isHidden()
    assert not panel._login_btn.isHidden() and panel._login_btn.isEnabled()
    panel.close()
    app.processEvents()
