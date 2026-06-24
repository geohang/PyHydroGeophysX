"""Headless smoke test for the AQUAH agent layer.

Run with no display and no API key::

    QT_QPA_PLATFORM=offscreen python -m PyHydroGeophysX.qt_apps.agent.selftest

It checks the provider adapters (OpenAI / Anthropic message + tool translation,
no network), verifies that every module class exposes the agent command
interface, drives each module's ``get_status`` through the controller, and runs
one approve/execute cycle of the chat panel with a scripted fake provider. Exit
code 0 means all checks passed.

Note: the Mesh 3D module is exercised only statically (its PyVistaQt GL viewer
cannot initialise on a headless/offscreen display), so it is not navigated to.
"""

from __future__ import annotations

import importlib
import os
import sys
import time

# Modules safe to construct on a headless/offscreen display (no GL viewer).
_NAV_SAFE = ["seismic", "ert", "em", "gravmag", "hydro_geophysics", "geo_hydrology", "seismic3d"]


def _build_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv[:1])


class _FakeProvider:
    """Scripted stand-in for a Provider: returns queued complete() replies."""

    id = "fake"

    def __init__(self, scripted):
        self._scripted = list(scripted)

    def available(self):
        return True, "ready"

    def set_api_key(self, *_):
        pass

    def set_model(self, *_):
        pass

    @property
    def model(self):
        return "fake"

    def complete(self, system, messages, specs):
        return self._scripted.pop(0)


def _check(name, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}{(' - ' + detail) if detail else ''}")
    return bool(ok)


def main() -> int:
    app = _build_app()
    from PyHydroGeophysX.qt_apps.agent import providers
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AquahChatPanel
    from PyHydroGeophysX.qt_apps.agent.controller import WorkbenchController
    from PyHydroGeophysX.qt_apps.agent.tools import tool_specs
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXWorkbench
    from PyHydroGeophysX.qt_apps.modules import MODULE_SPECS
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule

    ok = True
    specs = tool_specs()

    # 1. Provider tool-schema translation (offline).
    oa_tools = providers.to_openai_tools(specs)
    an_tools = providers.to_anthropic_tools(specs)
    ok &= _check(
        "tool schema translation",
        oa_tools[0]["type"] == "function" and "input_schema" in an_tools[0]
        and {t["function"]["name"] for t in oa_tools} == {t["name"] for t in an_tools},
    )

    # 2. Provider message translation (offline), incl. a tool round-trip.
    neutral = [
        {"role": "user", "content": "open seismic"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"id": "c1", "name": "navigate", "arguments": {"module": "seismic"}}]},
        {"role": "tool", "id": "c1", "name": "navigate", "content": {"status": "ok"}},
    ]
    oa_msgs = providers.to_openai_messages("SYS", neutral)
    an_msgs = providers.to_anthropic_messages(neutral)
    oa_ok = (
        oa_msgs[0]["role"] == "system"
        and any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in oa_msgs)
        and any(m.get("role") == "assistant" and m.get("tool_calls") for m in oa_msgs)
    )
    an_tail = an_msgs[-1]
    an_ok = (
        an_msgs[0]["role"] == "user"
        and an_tail["role"] == "user"
        and an_tail["content"][0]["type"] == "tool_result"
        and an_tail["content"][0]["tool_use_id"] == "c1"
        and any(
            isinstance(m["content"], list)
            and any(b.get("type") == "tool_use" and b.get("input") == {"module": "seismic"} for b in m["content"])
            for m in an_msgs if m["role"] == "assistant"
        )
    )
    ok &= _check("openai message translation", oa_ok)
    ok &= _check("anthropic message translation", an_ok)

    # 3. Provider registry builds each provider type.
    ok &= _check("provider registry",
                 all(providers.make_provider(pid).id == pid for pid in providers.PROVIDER_ORDER))

    # 4. Every module class overrides the agent interface (static, no widgets).
    missing = []
    for key, (sub, cls_name, _title) in MODULE_SPECS.items():
        mod = importlib.import_module(f"PyHydroGeophysX.qt_apps.modules.{sub}")
        cls = getattr(mod, cls_name)
        if cls.agent_describe is BaseModule.agent_describe or cls.agent_apply is BaseModule.agent_apply:
            missing.append(key)
    ok &= _check("all modules implement agent interface", not missing, f"missing={missing}")

    # 5. Command layer + per-module live drive.
    window = PyHydroGeophysXWorkbench()
    controller = WorkbenchController(window)

    res = controller.dispatch("list_modules", {})
    ok &= _check("list_modules", res.get("status") == "ok"
                 and {m["key"] for m in res.get("modules", [])} >= set(_NAV_SAFE))

    for key in _NAV_SAFE:
        nav = controller.dispatch("navigate", {"module": key})
        desc = controller.dispatch("describe_current_module", {})
        actions = [a.get("name") for a in desc.get("describe", {}).get("actions", [])]
        status = controller.dispatch("apply_action", {"action": "get_status", "args": {}})
        good = (nav.get("status") == "ok" and window.state.selected_module == key
                and "get_status" in actions and status.get("status") == "ok")
        ok &= _check(f"module '{key}'", good, f"actions={len(actions)}")

    # 6. Error paths degrade cleanly (current module = last navigated).
    ok &= _check("unknown action -> failed",
                 controller.dispatch("apply_action", {"action": "nope", "args": {}}).get("status") == "failed")
    ok &= _check("navigate bad module -> failed",
                 controller.dispatch("navigate", {"module": "does_not_exist"}).get("status") == "failed")

    # 6b. Seismic manual-pick-review actions: present in describe + degrade cleanly w/o data.
    controller.dispatch("navigate", {"module": "seismic"})
    sdesc = controller.dispatch("describe_current_module", {})
    sactions = {a.get("name") for a in sdesc.get("describe", {}).get("actions", [])}
    ok &= _check("seismic review actions present",
                 {"review_picks", "set_pick", "delete_pick", "list_picks"} <= sactions,
                 f"actions={sorted(sactions)}")
    rev = controller.dispatch("apply_action", {"action": "review_picks", "args": {}})
    setp = controller.dispatch("apply_action", {"action": "set_pick", "args": {"trace": 0, "time_s": 0.01}})
    lst = controller.dispatch("apply_action", {"action": "list_picks", "args": {}})
    dele = controller.dispatch("apply_action", {"action": "delete_pick", "args": {"trace": 0}})
    ok &= _check(
        "seismic pick-edit actions degrade cleanly with no data",
        rev.get("status") == "failed" and setp.get("status") == "failed"
        and lst.get("status") == "ok" and dele.get("status") == "ok",
        f"review={rev.get('status')} set={setp.get('status')} list={lst.get('status')} del={dele.get('status')}",
    )

    # 7. Chat panel loop with a scripted fake provider (no network, no clicks).
    first_turn = {
        "content": "I'll open the seismic module.",
        "tool_calls": [{"id": "c1", "name": "navigate", "arguments": {"module": "seismic"}}],
    }
    continuation = {"content": "Opened the Seismic Processing module.", "tool_calls": []}
    panel = AquahChatPanel(controller, provider=_FakeProvider([continuation]))
    panel._set_busy(True)
    panel._on_llm_ok(first_turn)
    gated = panel._current_call is not None and panel._busy
    panel._on_approve()
    for _ in range(200):
        if not panel._busy:
            break
        app.processEvents()
        time.sleep(0.01)
    if panel._worker is not None:
        panel._worker.wait(2000)
    ok &= _check(
        "panel approve executes + continues",
        gated and window.state.selected_module == "seismic" and not panel._busy,
        f"gated={gated} module={window.state.selected_module} busy={panel._busy}",
    )

    # 8. Panel pauses on an 'awaiting_user' tool result (ends the turn, no re-request).
    class _StubController:
        def capabilities_summary(self):
            return "stub"

        def dispatch(self, name, args):
            return {"status": "awaiting_user", "message": "Correct picks then continue.",
                    "suspect_traces": [3, 7]}

    pause_turn = {
        "content": "Auto-picked; please review.",
        "tool_calls": [{"id": "p1", "name": "apply_action",
                        "arguments": {"action": "review_picks", "args": {}}}],
    }
    panel2 = AquahChatPanel(_StubController(), provider=_FakeProvider([]))
    panel2._set_busy(True)
    panel2._on_llm_ok(pause_turn)
    gated2 = panel2._current_call is not None
    panel2._on_approve()
    ok &= _check(
        "panel pauses on awaiting_user (no re-request)",
        gated2 and not panel2._busy and not panel2._awaiting_user and panel2._worker is None,
        f"gated={gated2} busy={panel2._busy} awaiting={panel2._awaiting_user}",
    )

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
