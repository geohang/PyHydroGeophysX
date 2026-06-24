"""AQUAH agent chat mode for the PyHydroGeophysX Streamlit app.

This is the web counterpart of the Qt desktop AQUAH assistant. It lets the user
drive the SAME workflow engine the one-click flow uses
(``BaseAgent.run_unified_agent_workflow``) through a natural-language chat, with
provider/model taken from the app sidebar (OpenAI or Claude).

The module is split so the agent logic is import-light and unit-testable:

- :func:`tool_specs` / :data:`SYSTEM_PROMPT` / :class:`WebController` /
  :func:`run_agent_turn` are pure Python (no Streamlit, no heavy imports) and
  reuse the provider adapters in ``PyHydroGeophysX.qt_apps.agent.providers``.
- :func:`render_aquah_chat` is the Streamlit UI; it imports ``streamlit`` and the
  agents package lazily so importing this module stays cheap.

The one-click workflow is untouched; the app offers a mode toggle so the user
chooses between "one-click" and "AQUAH chat".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyHydroGeophysX.qt_apps.agent import providers

# Provider ids in the app sidebar -> provider ids in qt_apps.agent.providers.
_PROVIDER_MAP = {"openai": "openai", "claude": "anthropic"}

SYSTEM_PROMPT = """You are AQUAH, an assistant embedded in the PyHydroGeophysX web app. You help the user run a geophysical workflow by assembling a workflow configuration and running it — you act only through the provided tools, not by writing code.

Tools: set_config, get_config, list_uploaded_files, run_workflow, get_results.

How to work:
- Build the config from the user's request with set_config. Always set "user_request" to a clear one-line description, plus the data files and key parameters.
- Common config fields: user_request; for ERT inversion: data_file (or ert_file), instrument (E4D / BERT / Syscal / DAS-1 / ABEM-Lund / ResInv / ...), electrode_file (optional); for time-lapse ERT: time_lapse_files (an ordered list) + electrode_file; for seismic refraction: raw_seismic_file or seismic_file; for TDEM: tdem_file; for hydrologic-model conversion: hydro_model ("modflow" or "parflow") + modflow_dir / parflow_dir; inversion params: lambda, max_iterations.
- Call list_uploaded_files to see what the user uploaded, then point the right config field at those paths. NEVER invent file paths — only use uploaded files or paths the user gave you. If you have no data, ask the user to upload files (in the panel) or give paths.
- Before running, call get_config and briefly summarize it. Then call run_workflow — this is a heavy step that runs the full pipeline.
- If a tool result has status "failed" or "blocked", read it and adjust or ask the user.
- Finish every turn by telling the user the next step (what you will do, or what they can ask for). run_workflow runs synchronously and returns when done; after it, call get_results to summarize.
"""


def tool_specs() -> List[Dict[str, Any]]:
    """Provider-neutral tool specs for the web workflow command layer."""
    return [
        {
            "name": "set_config",
            "description": "Merge fields into the workflow configuration (user_request, data files, instrument, parameters).",
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {"type": "object", "description": "Config fields to set/overwrite.",
                                "additionalProperties": True}
                },
                "required": ["updates"],
            },
        },
        {
            "name": "get_config",
            "description": "Return the current workflow configuration and a note of likely-missing fields.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "list_uploaded_files",
            "description": "List the files the user has uploaded in this AQUAH panel, with their paths.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "run_workflow",
            "description": "Run the full workflow with the current configuration (heavy). Returns a result summary.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_results",
            "description": "Summarize the most recent workflow result (interpretation, report files).",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


class WebController:
    """Pure command layer: maps tool calls to workflow-config / run operations.

    ``run_fn`` runs a config and returns a JSON-friendly summary; ``list_files_fn``
    returns the uploaded-file list. Both are injected so this class is testable
    without Streamlit or the heavy engine.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        run_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        list_files_fn: Optional[Callable[[], List[Dict[str, str]]]] = None,
        allow_run: bool = True,
    ) -> None:
        self._state = state
        self._state.setdefault("config", {})
        self._state.setdefault("result", None)
        self._run_fn = run_fn
        self._list_files_fn = list_files_fn or (lambda: [])
        self._allow_run = allow_run

    def dispatch(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = args or {}
        try:
            if name == "set_config":
                return self._set_config(args)
            if name == "get_config":
                return self._get_config()
            if name == "list_uploaded_files":
                return {"status": "ok", "files": self._list_files_fn()}
            if name == "run_workflow":
                return self._run_workflow()
            if name == "get_results":
                return self._get_results()
            return {"status": "failed", "error": f"Unknown tool '{name}'."}
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    # -- config --------------------------------------------------------------
    def _set_config(self, args: Dict[str, Any]) -> Dict[str, Any]:
        updates = args.get("updates")
        if not isinstance(updates, dict) or not updates:
            return {"status": "failed", "error": "Provide 'updates' as a non-empty JSON object."}
        self._state["config"].update(updates)
        return {"status": "ok", "config": self._compact_config()}

    def _get_config(self) -> Dict[str, Any]:
        cfg = self._state["config"]
        has_data = any(k in cfg for k in (
            "data_file", "ert_file", "time_lapse_files", "timelapse_files",
            "seismic_file", "raw_seismic_file", "tdem_file", "modflow_dir", "parflow_dir"))
        missing = []
        if not cfg.get("user_request"):
            missing.append("user_request")
        if not has_data:
            missing.append("a data file/dir (data_file, time_lapse_files, seismic_file, modflow_dir, …)")
        return {"status": "ok", "config": self._compact_config(), "missing": missing}

    def _compact_config(self) -> Dict[str, Any]:
        # Drop bulky internal keys for a readable echo.
        drop = {"uploaded_files"}
        return {k: v for k, v in self._state["config"].items() if k not in drop}

    # -- run -----------------------------------------------------------------
    def _run_workflow(self) -> Dict[str, Any]:
        if not self._allow_run:
            return {"status": "blocked",
                    "message": "Auto-run is off. Ask the user to enable 'Let AQUAH run the "
                               "workflow' or click the Run button below."}
        cfg = self._state["config"]
        result = self._run_fn(cfg)
        if isinstance(result, dict):
            self._state["result"] = result
            return result
        self._state["result"] = {"status": "ok", "result": result}
        return self._state["result"]

    def _get_results(self) -> Dict[str, Any]:
        result = self._state.get("result")
        if not result:
            return {"status": "failed", "error": "No workflow has been run yet."}
        return {"status": "ok", "result": result}


def run_agent_turn(
    provider: Any,
    system: str,
    messages: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    controller: WebController,
    max_steps: int = 8,
) -> List[Dict[str, Any]]:
    """Drive one user turn to completion, auto-executing tool calls.

    Mutates ``messages`` in place and returns a list of transcript events:
    ``{"kind": "assistant", "text": ...}`` or
    ``{"kind": "tool", "name", "args", "result"}``.
    """
    events: List[Dict[str, Any]] = []
    for _ in range(max_steps):
        out = provider.complete(system, messages, specs)
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": out.get("content"),
            "tool_calls": out.get("tool_calls") or [],
        }
        if "_anthropic_content" in out:
            assistant["_anthropic_content"] = out["_anthropic_content"]
        messages.append(assistant)
        if assistant["content"]:
            events.append({"kind": "assistant", "text": assistant["content"]})
        calls = assistant["tool_calls"]
        if not calls:
            break
        for call in calls:
            name = call.get("name", "?")
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            result = controller.dispatch(name, args)
            messages.append({"role": "tool", "id": call.get("id"), "name": name, "content": result})
            events.append({"kind": "tool", "name": name, "args": args, "result": result})
    return events


# ---------------------------------------------------------------------------
# Streamlit UI (imports streamlit + the agents engine lazily)
# ---------------------------------------------------------------------------
def render_aquah_chat(sidebar_state: Dict[str, Any]) -> None:
    """Render the AQUAH chat panel inside the Streamlit workflow tab."""
    import streamlit as st

    st.subheader("💬 AQUAH assistant")
    st.caption("Describe your workflow in plain language; AQUAH builds the configuration and runs the "
               "same engine as one-click mode. Provider and model come from the sidebar.")

    app_provider = st.session_state.get("llm_provider", "openai")
    prov_id = _PROVIDER_MAP.get(app_provider)
    if prov_id is None:
        st.info(f"AQUAH chat supports OpenAI and Claude. The sidebar provider is '{app_provider}' — "
                "switch to OpenAI or Claude in the sidebar, or use One-click mode.")
        return

    api_key = (st.session_state.get("api_key") or "").strip()
    model = (st.session_state.get("llm_model") or "").strip() or None
    provider = providers.make_provider(prov_id, model=model, api_key=api_key or None)
    ok, reason = provider.available()
    if not ok:
        st.warning(f"{reason} Set it in the sidebar (provider/model/API key).")
        return
    st.caption(f"Using **{providers.PROVIDER_META[prov_id]['label']} · {provider.model}**")

    # Session state for this panel.
    st.session_state.setdefault("aquah_messages", [])
    st.session_state.setdefault("aquah_config", {})
    st.session_state.setdefault("aquah_uploads", {})

    output_dir = Path(str(sidebar_state.get("output_dir") or "results/streamlit_workflow")).expanduser()
    upload_dir = output_dir / "aquah_uploads"

    cols = st.columns([3, 1])
    with cols[0]:
        uploaded = st.file_uploader(
            "Upload data files (optional)", accept_multiple_files=True,
            type=["ohm", "dat", "data", "txt", "sgy", "segy", "sgt", "pfb", "nam"],
            key="aquah_uploader",
        )
        if uploaded:
            upload_dir.mkdir(parents=True, exist_ok=True)
            for uf in uploaded:
                dest = upload_dir / uf.name
                dest.write_bytes(uf.getbuffer())
                st.session_state.aquah_uploads[uf.name] = str(dest)
    with cols[1]:
        allow_run = st.checkbox("Let AQUAH run", value=True, key="aquah_allow_run",
                                help="If off, AQUAH builds the config but you run it with the button below.")
        if st.button("🧹 New chat", key="aquah_reset", width="stretch"):
            st.session_state.aquah_messages = []
            st.session_state.aquah_config = {}
            st.rerun()

    if st.session_state.aquah_uploads:
        st.caption("Uploaded: " + ", ".join(st.session_state.aquah_uploads.keys()))

    cfg_now = st.session_state.get("aquah_config") or {}
    if cfg_now:
        with st.expander("Current workflow config", expanded=False):
            st.json(cfg_now)
    st.caption(
        "Next: describe your task (or refine it). AQUAH sets the config, then runs when ready; "
        "results appear in the **Results** tab and the bridge panel."
    )

    # Build controller wired to the real engine.
    def _list_files() -> List[Dict[str, str]]:
        return [{"name": n, "path": p} for n, p in st.session_state.aquah_uploads.items()]

    run_fn = _make_run_fn(str(output_dir))
    state = {"config": st.session_state.aquah_config,
             "result": st.session_state.get("aquah_last_summary")}
    controller = WebController(state, run_fn, _list_files, allow_run=allow_run)

    # Render history.
    for msg in st.session_state.aquah_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. run a time-lapse ERT inversion on my uploaded surveys with lambda 30")
    if prompt:
        st.session_state.aquah_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # The neutral message log the providers consume (rebuilt from the visible
        # history is lossy for tool ids, so keep a parallel neutral log).
        neutral = st.session_state.setdefault("aquah_neutral", [])
        neutral.append({"role": "user", "content": prompt})
        system = SYSTEM_PROMPT
        with st.chat_message("assistant"):
            with st.spinner("AQUAH is working…"):
                try:
                    events = run_agent_turn(provider, system, neutral, tool_specs(), controller)
                except Exception as exc:  # noqa: BLE001
                    events = [{"kind": "assistant", "text": f"Model call failed: {exc}"}]
            reply = _render_events(st, events)
        st.session_state.aquah_config = state["config"]
        st.session_state.aquah_last_summary = state.get("result")
        st.session_state.aquah_messages.append({"role": "assistant", "content": reply})

    # Manual run button (covers 'Let AQUAH run' = off).
    st.markdown("---")
    if st.button("▶ Run workflow with the current config", key="aquah_manual_run"):
        cfg = st.session_state.aquah_config
        if not cfg:
            st.warning("No configuration yet — describe your workflow above first.")
        else:
            with st.spinner("Running workflow…"):
                summary = run_fn(cfg)
            st.session_state.aquah_last_summary = summary
            st.success(f"Workflow finished (status: {summary.get('status')}). "
                       "See the Results tab for figures and the full report.")


def _render_events(st_module: Any, events: List[Dict[str, Any]]) -> str:
    """Render transcript events into the chat; return the combined assistant text."""
    texts: List[str] = []
    for ev in events:
        if ev["kind"] == "assistant":
            st_module.markdown(ev["text"])
            texts.append(ev["text"])
        else:
            name, result = ev["name"], ev["result"]
            status = str(result.get("status", "ok")) if isinstance(result, dict) else "ok"
            icon = "✅" if status in ("ok", "started", "success") else (
                "⏳" if status == "blocked" else "⚠️")
            with st_module.expander(f"{icon} {name} → {status}", expanded=False):
                st_module.code(json.dumps(ev["args"], default=str) or "{}", language="json")
                st_module.code(json.dumps(result, default=str)[:1500], language="json")
            if isinstance(result, dict) and status == "blocked" and result.get("message"):
                st_module.info(result["message"])
    if not texts:
        texts.append("(no reply)")
    return "\n\n".join(texts)


def _make_run_fn(output_dir: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build the real run function: runs the engine and stores workflow_result."""
    import streamlit as st

    def run_fn(config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from PyHydroGeophysX.agents import BaseAgent
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Agents engine unavailable: {exc}"}
        cfg = dict(config)
        cfg.setdefault("output_dir", output_dir)
        try:
            results, plan, interp, files = BaseAgent.run_unified_agent_workflow(
                cfg,
                st.session_state.get("api_key"),
                st.session_state.get("llm_model"),
                st.session_state.get("llm_provider", "openai"),
                output_dir,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        # Store under the SAME key the one-click flow uses, so the Results tab shows it.
        st.session_state.workflow_result = {
            "results": results,
            "execution_plan": plan,
            "interpretation": interp,
            "report_files": files,
            "workflow_config": cfg,
        }
        return {
            "status": "ok",
            "interpretation": (interp or "")[:1500] if isinstance(interp, str) else "",
            "report_files": files or {},
            "steps": len(plan or []),
        }

    return run_fn
