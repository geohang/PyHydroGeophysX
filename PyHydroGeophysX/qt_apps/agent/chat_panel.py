"""The AQUAH chat panel: the right-hand dock that drives the workbench in words.

The panel keeps a provider-neutral ``messages`` list and a small per-turn state
machine. The only blocking step (the model call) runs on a ``QThread``; tool
calls are executed on the main thread and, because the user chose per-step
confirmation, each proposed tool call is shown with Approve / Reject buttons
before it runs. The loop is event-driven: it pauses on a button click and
resumes when the user decides. Provider (OpenAI / Claude / OpenAI-compatible)
and model are selectable at the top of the panel and may change mid-conversation.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.agent.controller import WorkbenchController
from PyHydroGeophysX.qt_apps.agent.providers import (
    PROVIDER_META,
    PROVIDER_ORDER,
    make_provider,
)
from PyHydroGeophysX.qt_apps.agent.runtime import LlmCallWorker
from PyHydroGeophysX.qt_apps.agent.tools import tool_specs

_P = theme.PALETTE

SYSTEM_PROMPT = """You are AQUAH, an assistant embedded in the PyHydroGeophysX desktop workbench, a hydrogeophysics tool. You help the user process and model geophysical data by driving the workbench GUI through tools, not by writing code.

Rules:
- You act only through the provided tools: list_modules, navigate, describe_current_module, apply_action, get_workbench_state.
- Pick the module by matching the task to the module purposes listed under "Workbench modules" below — read them before choosing. For 2D-profile forward modeling / synthetic data (ERT/SRT/EM/gravity along a line), use hydro_geophysics. For 3D ERT forward modeling, use mesh3d (first 'generate' the 3D mesh, then 'run_ert_forward'). Do NOT use ert for forward modeling — it inverts field data.
- Every tool call you make is automatically shown to the user with Approve / Reject buttons before it runs, so you do NOT need to ask for permission in words. Never end your turn with a question like "Shall I proceed?" — instead, briefly say what you are about to do and then make the tool call in the same turn.
- Take one action at a time: call a single tool, wait for its result, then decide the next step.
- After you navigate to a module, call describe_current_module before apply_action, so you use the module's real action names and parameter keys.
- Never invent file paths or other inputs. If an action needs a data file and the user did not give you a path, prefer an example-data action (such as use_example_data or use_example) when the module offers one; only call load_data with a path the user actually provided. If neither is possible, ask the user for the file.
- For a processing request, a typical sequence is: navigate to the right module, describe it, load data (use the example data if the user gave none), set parameters, then run.
- If a tool result has status "failed" or "declined", read it and adjust, or ask the user what to do. Do not invent module names, actions, or parameters; discover them with the tools.
- Always finish your turn by telling the user the next step: what you will do next, or what they can ask for. When you START a long job (a tool result with status "started"), say it is running in the background and that the user can ask for its status (for example "is it done?") or say "continue" to proceed once it finishes — then end your turn rather than guessing the job is complete.
- Seismic first-break picking has a mandatory human checkpoint: load_data -> set_geometry -> (set_params) -> auto_pick (each shot) -> review_picks -> run_srt. After auto_pick you MUST call review_picks and then stop; never call run_srt in the same turn as auto_pick.
- When a tool returns status "awaiting_user", the workflow is paused for the user to act in the GUI (for example correcting picks). Relay the message, end your turn, and resume only when the user says to continue. During the pause the user may also ask you to set_pick or delete_pick specific traces.

Workbench modules (key: purpose):
{modules}
"""


def _default_provider_id() -> str:
    """Pick the first provider that is ready (has a key); else OpenAI."""
    for pid in PROVIDER_ORDER:
        try:
            if make_provider(pid).available()[0]:
                return pid
        except Exception:  # noqa: BLE001
            continue
    return "openai"


class AquahChatPanel(QWidget):
    """Conversation panel that turns plain language into approved tool calls."""

    def __init__(
        self,
        controller: WorkbenchController,
        log=None,
        provider: Optional[Any] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._log = log or (lambda *a, **k: None)
        self._tool_specs = tool_specs()

        if provider is not None:
            self._provider = provider
            self._provider_id = getattr(provider, "id", "openai")
        else:
            self._provider_id = _default_provider_id()
            self._provider = make_provider(self._provider_id)

        self._system = SYSTEM_PROMPT.format(modules=self._controller.capabilities_summary())
        self._messages: List[Dict[str, Any]] = []
        self._tool_queue: List[Dict[str, Any]] = []
        self._current_call: Optional[Dict[str, Any]] = None
        self._executed_in_turn = False
        self._awaiting_user = False
        self._busy = False
        self._worker: Optional[LlmCallWorker] = None

        self._build_ui()
        self._sync_settings_widgets()
        self._reset_conversation()

    # -- UI ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>AQUAH Assistant</b>"))
        header.addStretch(1)
        self._new_btn = QPushButton("New chat")
        self._new_btn.setIcon(theme.icon("fa5s.broom"))
        self._new_btn.clicked.connect(self._reset_conversation)
        header.addWidget(self._new_btn)
        root.addLayout(header)

        root.addWidget(self._build_settings())

        self._transcript = QTextBrowser()
        self._transcript.setOpenExternalLinks(True)
        root.addWidget(self._transcript, stretch=1)

        # Pending tool-call confirmation card (hidden until needed).
        self._confirm = QFrame()
        self._confirm.setObjectName("ConfirmCard")
        self._confirm.setStyleSheet(
            f"#ConfirmCard {{ background:{_P['select_bg']}; border:1px solid {_P['border_blue']};"
            f" border-radius:8px; }}"
        )
        clay = QVBoxLayout(self._confirm)
        self._confirm_label = QLabel()
        self._confirm_label.setWordWrap(True)
        self._confirm_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        clay.addWidget(self._confirm_label)
        cbtns = QHBoxLayout()
        cbtns.addStretch(1)
        self._reject_btn = QPushButton("Reject")
        self._reject_btn.setIcon(theme.icon("fa5s.times", color=_P["red"]))
        self._reject_btn.clicked.connect(self._on_reject)
        self._approve_btn = QPushButton("Approve & run")
        self._approve_btn.setProperty("primary", True)
        self._approve_btn.setIcon(theme.icon("fa5s.check", color="#ffffff"))
        self._approve_btn.clicked.connect(self._on_approve)
        cbtns.addWidget(self._reject_btn)
        cbtns.addWidget(self._approve_btn)
        clay.addLayout(cbtns)
        self._confirm.setVisible(False)
        root.addWidget(self._confirm)

        # Input row.
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("e.g. Run an ERT forward model on the example data")
        self._input.returnPressed.connect(self._on_send)
        self._send_btn = QPushButton("Send")
        self._send_btn.setProperty("primary", True)
        self._send_btn.setIcon(theme.icon("fa5s.paper-plane", color="#ffffff"))
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._input, stretch=1)
        input_row.addWidget(self._send_btn)
        root.addLayout(input_row)

    def _build_settings(self) -> QWidget:
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Provider"))
        self._provider_combo = QComboBox()
        for pid in PROVIDER_ORDER:
            self._provider_combo.addItem(PROVIDER_META[pid]["label"], pid)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        row1.addWidget(self._provider_combo, stretch=1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Model"))
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        row2.addWidget(self._model_combo, stretch=1)
        lay.addLayout(row2)

        self._base_url_row = QWidget()
        burl = QHBoxLayout(self._base_url_row)
        burl.setContentsMargins(0, 0, 0, 0)
        burl.addWidget(QLabel("Base URL"))
        self._base_url_edit = QLineEdit()
        self._base_url_edit.setPlaceholderText("https://api.example.com/v1")
        burl.addWidget(self._base_url_edit, stretch=1)
        lay.addWidget(self._base_url_row)

        row3 = QHBoxLayout()
        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.Password)
        self._key_edit.setPlaceholderText("Paste API key for this session")
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply_settings)
        row3.addWidget(self._key_edit, stretch=1)
        row3.addWidget(self._apply_btn)
        lay.addLayout(row3)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color:{_P['muted']};")
        lay.addWidget(self._status_label)
        return box

    # -- settings handling ---------------------------------------------------
    def _sync_settings_widgets(self) -> None:
        """Reflect the active provider into the combos / rows (no signals)."""
        meta = PROVIDER_META.get(self._provider_id, PROVIDER_META["openai"])
        self._provider_combo.blockSignals(True)
        idx = self._provider_combo.findData(self._provider_id)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)
        self._provider_combo.blockSignals(False)

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItems(meta.get("models", []))
        self._model_combo.setCurrentText(self._provider.model)
        self._model_combo.blockSignals(False)

        self._base_url_row.setVisible(bool(meta.get("needs_base_url")))
        self._key_edit.setPlaceholderText(f"Paste {meta['env_key']} for this session")

    def _on_provider_changed(self, _index: int) -> None:
        pid = self._provider_combo.currentData()
        if not pid or pid == self._provider_id:
            return
        self._provider_id = pid
        self._provider = make_provider(pid)
        self._sync_settings_widgets()
        self._render_note(f"Switched provider to <b>{html.escape(PROVIDER_META[pid]['label'])}</b>.")
        self._refresh_ready_state()

    def _on_model_changed(self, text: str) -> None:
        self._provider.set_model(text)
        self._refresh_ready_state()

    def _on_apply_settings(self) -> None:
        meta = PROVIDER_META.get(self._provider_id, PROVIDER_META["openai"])
        if meta.get("needs_base_url"):
            self._provider.set_base_url(self._base_url_edit.text())
        key = self._key_edit.text().strip()
        if key:
            self._provider.set_api_key(key)
            self._key_edit.clear()
        self._provider.set_model(self._model_combo.currentText())
        self._render_note("Applied provider settings for this session.")
        self._refresh_ready_state()

    # -- conversation lifecycle ---------------------------------------------
    def _reset_conversation(self) -> None:
        self._messages = []
        self._tool_queue = []
        self._current_call = None
        self._executed_in_turn = False
        self._confirm.setVisible(False)
        self._transcript.clear()
        self._render_note(
            "AQUAH is ready — describe what you want to do in plain language. Examples:"
            "<br>&#8226; <i>open Hydro &#8594; Geophysics and run an ERT forward model on the example data</i>"
            "<br>&#8226; <i>load ERT data from &lt;path&gt; as E4D and run the inversion with lambda 30</i>"
            "<br>&#8226; <i>build a 3D mesh and run a 3D ERT forward model</i>"
            "<br>Each step is shown with Approve / Reject before it runs."
        )
        self._refresh_ready_state()

    def _refresh_ready_state(self) -> None:
        ok, reason = self._provider.available()
        self._input.setEnabled(ok and not self._busy)
        self._send_btn.setEnabled(ok and not self._busy)
        label = PROVIDER_META.get(self._provider_id, {}).get("label", self._provider_id)
        if ok:
            self._status_label.setText(f"{html.escape(label)} · {html.escape(self._provider.model)} — ready")
        else:
            self._status_label.setText(f"{html.escape(label)}: {html.escape(reason)}")

    # -- sending / receiving -------------------------------------------------
    def _on_send(self) -> None:
        if self._busy:
            return
        ok, reason = self._provider.available()
        if not ok:
            self._render_note(reason)
            self._refresh_ready_state()
            return
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._render_user(text)
        self._messages.append({"role": "user", "content": text})
        self._set_busy(True)
        self._start_request()

    def _start_request(self) -> None:
        self._send_btn.setText("…")
        worker = LlmCallWorker(self._provider, self._system, self._messages, self._tool_specs)
        worker.succeeded.connect(self._on_llm_ok)
        worker.failed.connect(self._on_llm_failed)
        worker.finished.connect(lambda: self._send_btn.setText("Send"))
        self._worker = worker
        worker.start()

    def _on_llm_failed(self, error: str) -> None:
        self._render_note(f"Model call failed: {html.escape(error)}")
        self._set_busy(False)

    def _on_llm_ok(self, out: Dict[str, Any]) -> None:
        # Record the assistant turn (neutral form) so tool ids line up next call.
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": out.get("content"),
            "tool_calls": out.get("tool_calls") or [],
        }
        if "_anthropic_content" in out:
            assistant["_anthropic_content"] = out["_anthropic_content"]
        self._messages.append(assistant)

        if assistant["content"]:
            self._render_assistant(assistant["content"])
        if assistant["tool_calls"]:
            self._tool_queue = list(assistant["tool_calls"])
            self._executed_in_turn = False
            self._process_next_tool()
        else:
            self._set_busy(False)

    # -- tool loop -----------------------------------------------------------
    def _process_next_tool(self) -> None:
        if self._awaiting_user and self._tool_queue:
            # A checkpoint paused the workflow; do not run the rest of this batch,
            # but answer every queued call so tool_call / tool_result stay paired.
            for skipped in self._tool_queue:
                self._answer_tool(skipped, {"status": "skipped",
                                            "message": "Paused for manual pick review."})
            self._tool_queue = []
        if not self._tool_queue:
            if self._awaiting_user:
                self._awaiting_user = False
                self._set_busy(False)  # hand control to the user; resume on "continue"
            elif self._executed_in_turn:
                self._start_request()  # let the model react to the tool results
            else:
                self._set_busy(False)
            return
        call = self._tool_queue.pop(0)
        self._current_call = call
        args = call.get("arguments")
        if not isinstance(args, dict):
            args = {}
        self._show_confirmation(call.get("name", "?"), args)

    def _show_confirmation(self, name: str, args: Dict[str, Any]) -> None:
        self._confirm_label.setText(self._describe_call(name, args))
        self._confirm.setVisible(True)
        self._approve_btn.setEnabled(True)
        self._reject_btn.setEnabled(True)

    def _on_approve(self) -> None:
        call = self._current_call
        if call is None:
            return
        self._confirm.setVisible(False)
        name = call.get("name", "?")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        result = self._controller.dispatch(name, args)
        if isinstance(result, dict) and result.get("status") == "awaiting_user":
            self._awaiting_user = True
        self._answer_tool(call, result)
        self._render_tool_result(name, result)
        self._executed_in_turn = True
        self._current_call = None
        self._process_next_tool()

    def _on_reject(self) -> None:
        call = self._current_call
        if call is None:
            return
        self._confirm.setVisible(False)
        result = {"status": "declined", "message": "The user declined this action."}
        self._answer_tool(call, result)
        self._render_note("You rejected the step.")
        self._executed_in_turn = True
        self._current_call = None
        self._process_next_tool()

    def _answer_tool(self, call: Dict[str, Any], result: Dict[str, Any]) -> None:
        self._messages.append({
            "role": "tool",
            "id": call.get("id"),
            "name": call.get("name"),
            "content": result,
        })

    # -- state / rendering ---------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._input.setEnabled(not busy)
        self._send_btn.setEnabled(not busy)
        if not busy:
            self._refresh_ready_state()
            self._input.setFocus()

    def _describe_call(self, name: str, args: Dict[str, Any]) -> str:
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "(no arguments)"
        return (
            f"<b>AQUAH wants to run:</b> <code>{html.escape(name)}</code><br>"
            f"<span style='color:{_P['muted']}'>{html.escape(arg_str)}</span>"
        )

    def _bubble(self, who: str, text: str, color: str) -> None:
        safe = html.escape(text).replace("\n", "<br>")
        self._transcript.append(
            f"<div style='margin:6px 0;'>"
            f"<span style='color:{color}; font-weight:700;'>{who}</span><br>{safe}</div>"
        )
        self._scroll_to_end()

    def _render_user(self, text: str) -> None:
        self._bubble("You", text, _P["primary"])

    def _render_assistant(self, text: str) -> None:
        self._bubble("AQUAH", text, _P["accent"])

    def _render_note(self, html_text: str) -> None:
        self._transcript.append(
            f"<div style='margin:6px 0; color:{_P['muted']};'>{html_text}</div>"
        )
        self._scroll_to_end()

    def _render_tool_result(self, name: str, result: Dict[str, Any]) -> None:
        import json

        status = str(result.get("status", "ok"))
        color = _P["green"] if status in ("ok", "started", "success", "awaiting_user") else _P["red"]
        compact = {k: v for k, v in result.items() if k != "describe"}
        text = json.dumps(compact, default=str)[:600]
        self._transcript.append(
            f"<div style='margin:4px 0;'>"
            f"<span style='color:{color}; font-weight:700;'>"
            f"{html.escape(name)} &#8594; {html.escape(status)}</span><br>"
            f"<code style='color:{_P['muted']};'>{html.escape(text)}</code></div>"
        )
        # Next-step guidance so the user is never left wondering what to do.
        if status == "started":
            self._render_note(
                "&#9203; This is running in the background. Ask <i>“is it done?”</i> for the "
                "status, or say <i>“continue”</i> / tell AQUAH the next step.")
        elif status == "awaiting_user":
            note = html.escape(str(result.get("message")
                                   or "Paused. Correct the picks, then say “continue”."))
            suspect = result.get("suspect_traces")
            if isinstance(suspect, list) and suspect:
                note += f"<br>Suspect traces to check: <code>{html.escape(str(suspect))}</code>"
            self._render_note(f"&#9208; <b>Paused for you.</b> {note}")
        else:
            hint = result.get("hint") or (result.get("error") if status == "failed" else None)
            if hint:
                self._render_note(f"&#128161; {html.escape(str(hint))}")
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        sb = self._transcript.verticalScrollBar()
        sb.setValue(sb.maximum())
