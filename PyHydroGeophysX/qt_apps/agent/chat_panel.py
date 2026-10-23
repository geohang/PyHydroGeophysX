"""The assistant chat panel: the right-hand dock that drives the studio in words.

Which assistant the user is talking to - AQUAH for hydrogeophysics, GeoSAGE for
geological modelling, or another registered in
:mod:`PyHydroGeophysX.agents.assistants` - is chosen at the top of the panel.
The assistant supplies its name, its part of the system prompt, its example
requests and the workflow "Auto to report" runs; the rules for driving the
studio are this panel's and the same for every assistant.

The panel keeps a provider-neutral ``messages`` list and a small per-turn state
machine. The only blocking step (the model call) runs on a ``QThread``; tool
calls are executed on the main thread and, because the user chose per-step
confirmation, each proposed tool call is shown with Approve / Reject buttons
before it runs. The loop is event-driven: it pauses on a button click and
resumes when the user decides. Provider (OpenAI / Claude / OpenAI-compatible)
and model are selectable at the top of the panel and may change mid-conversation.

When the selected model reads images, ``capture_view`` becomes available and the
panel carries screenshots into the conversation. Those never ride inside the
``tool`` result: they are held until the whole tool batch is answered, then sent
as one follow-up ``user`` message. Keeping the ``tool`` entries contiguous is
what makes a batch of parallel tool calls legal for the Anthropic Messages API,
and a string-only ``tool`` role is what OpenAI chat-completions requires.
"""

from __future__ import annotations

import base64
import html
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, QUrl, Signal, QProcess, QProcessEnvironment
from PySide6.QtGui import QColor, QImage, QTextDocument
from PySide6.QtWidgets import (
    QCheckBox,
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.agent.controller import StudioController
from PyHydroGeophysX.qt_apps.agent.providers import (
    MODEL_TIERS,
    PROVIDER_META,
    DESKTOP_PROVIDER_ORDER as PROVIDER_ORDER,
    CLI_PROVIDER_IDS,
    TIER_CUSTOM,
    TIER_ORDER,
    image_block,
    make_provider,
    prewarm,
    price_label,
    provider_has_tiers,
    text_block,
    tier_effort,
    tier_for_model,
    tier_label,
    tier_model,
)
from PyHydroGeophysX.qt_apps.agent.runtime import LlmCallWorker, CliSetupWorker, ScriptWorker
from PyHydroGeophysX.qt_apps.widgets import ai_presence as presence
from PyHydroGeophysX.agents import assistants as assistant_registry
from PyHydroGeophysX.qt_apps.widgets.ai_presence import AiOrb
from PyHydroGeophysX.agents.skills import (TOOL_NAME as SKILL_TOOL, index_text as skill_index,
                                           run_tool as run_skill_tool,
                                           tool_spec as skill_tool_spec)
from PyHydroGeophysX.agents.studio_tools import tool_specs

_P = theme.PALETTE

# Inputs that, while paused at a checkpoint, resume the workflow with no LLM call.
_CONTINUE_WORDS = frozenset({"continue", "next", "go", "go on", "proceed", "done",
                             "next shot", "继续", "下一炮", "下一个"})

SYSTEM_PROMPT = """You are {name} ({title}), an assistant embedded in the PyHydroGeophysX desktop studio. {summary} You help the user by driving the studio GUI through tools. Where the studio cannot do what is needed, you can propose a short temporary script, which runs only if the user approves it.

Rules:
- You act only through the provided tools: list_modules, navigate, describe_current_module, apply_action, get_studio_state, run_to_report, propose_script.
- When something fails or the studio has no action for what is needed, think about the cause before giving up or handing the problem back. First look for a way with the studio's own actions (another action, other parameters, another module). If there is none, and a few lines of Python would get past it - a file in a layout a module cannot read, one module's result in a format another module has no action to take, a number no module computes - call propose_script. Check the module's actions first: the water-content module, for example, takes the ERT page's model with use_ert_model and opens an inversion's output folder with set_data_dir, so it needs no script for that. A script must never invent layer markers: an inversion mesh's cell markers number cells, they are not layers. In its purpose, say plainly what went wrong, what the script will do and what you will do with its output. The user sees the code and decides; if they reject it, accept that and offer the other options. After it runs, use what it wrote through the studio's actions (for example set_data_dir on the folder it wrote). If it fails, read its error and what it printed, fix the cause (not with a bare except or a made-up value) and call propose_script again with the corrected script and a purpose that says what was wrong; the user approves it again. If a fix does not help either, say so and offer another way rather than a third version. Never propose the same script unchanged. Never present a script's guess as a measured result.
- The user chose to work step by step, and can change their mind at any point. When they ask for the end result rather than the next step - in any words or language, e.g. "give me the final result directly", "just finish it", "use auto report", "直接给我结果" - call run_to_report in that same turn, with ask_user false since they asked for it. It hands the rest of the task to Auto to report, which carries on from where you are now, reuses what is already done, and writes the report. Never tell the user to switch modes themselves; switching is what run_to_report does. In its goal, restate the whole task as the run should read it: what the user asked for, the data and every choice made so far (instrument, geometry, single or time-lapse, inversion settings and the fit obtained), and what is left to do.
- Pick the module by matching the task to the module purposes listed under "Studio modules" below — read them before choosing.
- You decide which calls the user approves, with each call's ask_user argument. Calls with ask_user false run at once and the user sees them in the chat; calls with ask_user true wait for Approve / Reject. Ask where the user's judgement matters: before a long computation (an inversion, a forward run, picking every shot) whose settings they have not confirmed, when you choose a value, data source or method they did not give, and before a change of plan. Do not ask for routine work: looking, opening a module, loading the data they named, applying values they gave, the next step of a plan they already agreed to. A task of ten steps should ask once or twice, at its decisions. Clearing or removing picks, electrodes or inputs, exporting or saving to a path you name, and running a script, always wait for the user whatever you set.
- Never ask for permission in words ("Shall I proceed?"): briefly say what you are about to do and make the tool call in the same turn, with ask_user true if it needs their say.
- Think the whole task through before the first action: what the goal needs, which modules and actions get there, what could go wrong. Say the plan in a sentence or two, then revise it as results come in.
- Take one action at a time: call a single tool, wait for its result, then decide the next step.
- The usual order of a module's actions is a default, not a rule. When the results show a better path - running a step again with other parameters, a check in another module, a product the goal implies that nobody asked for - say what you would do differently, why, and what it should gain, and make that call with ask_user true; the user decides through the approval card. If they reject it, carry on along the usual path.
- navigate returns the opened module's description under "current_module": its actions with their parameter keys, its current state and its views. Use those real action names and parameter keys in apply_action. Do not call describe_current_module right after navigate - it would return the same description again; call it only to refresh the state later (for example after data were loaded or a run finished), or when no navigate result is at hand.
- Never invent file paths or other inputs. If an action needs a data file and the user did not give you a path, prefer an example-data action (such as use_example_data or use_example) when the module offers one, unless a rule below says to ask first; only call load_data with a path the user actually provided. If neither is possible, ask the user for the file.
- For a processing request, a typical sequence is: navigate to the right module (its result describes the module), load data (use the example data if the user gave none), set parameters, then run.
- If a tool result has status "failed" or "declined", read it and adjust, or ask the user what to do. Do not invent module names, actions, or parameters; discover them with the tools.
- A message that starts with "[Studio]" comes from the studio, not the user: it reports something that happened, such as an automatic run that failed. Do what it asks, and address the user in your reply.
- Always finish your turn by telling the user the next step: what you will do next, or what they can ask for. When you START a long job (a tool result with status "started"), say it is running in the background and that the user can ask for its status (for example "is it done?") or say "continue" to proceed once it finishes — then end your turn rather than guessing the job is complete.
- After a background job, refresh the module's status before reporting progress. Read inversion_status, inversion_running, has_model and result_summary when provided. A completed model is available for review in the processing module even before saving it or adding it to Map. Report the returned fit metrics; if a metric is missing, say it is unavailable rather than claiming no model exists. If a new run is running or failed while an earlier model remains, distinguish the latest run from that earlier result.
- When a tool returns status "awaiting_user", the workflow is paused for the user to act in the GUI (for example correcting picks). Relay the message, end your turn, and resume only when the user says to continue.
{persona}{vision_rules}{skill_rules}
Studio modules (key: purpose):
{modules}
"""

#: Appended to the rules when the assistant has written procedures (skills,
#: :mod:`PyHydroGeophysX.agents.skills`): the index only; read_skill reads one.
SKILL_RULES = """- Written procedures (skills) are listed below by name and when they apply. Before a task one of them covers, read it with read_skill (it runs at once and changes nothing), then follow it where it fits the data and the request. A procedure is its author's advice: it never overrides the user, the checkpoints above or what the data show, and it is not a reason to skip asking. Say which procedure you followed.
Written procedures:
{index}
"""

#: Appended to the rules only when the selected model reads images, so a
#: text-only model is never told about a tool it will not be given.
VISION_RULES = """- You can see the studio. capture_view takes a picture of a panel in the current module and shows it to you; the module's description (from navigate or describe_current_module) lists the panel names under "views". Use it to judge what numbers do not show: artefacts or implausible structure in an inversion section, picks that follow noise instead of the first arrival, a mesh with the wrong extent or bad element shape. Say which view you looked at, report what is actually in the picture, and say plainly when the image is too coarse to judge rather than guessing.
- Do not capture after every step. One picture costs about as much context as a long message, so capture when a result is worth judging visually, never to confirm that a parameter was set.
- When a capture_view result carries a "context" field, those numbers are exact and the picture is not. Never read an index, a trace number or a coordinate off the image when "context" gives it to you: dozens of items can share one axis and the marker sits far from its tick label, so counting across a plot is where you will be wrong. Judge quality from the picture, then name the item from the numbers. If the two disagree, trust the numbers for identity and say what the picture showed you.
- If the user asks about pick quality while a first-break review is paused, capture_view the 'gather' view before answering, and cross-check its "context" pick table against what you see.
"""

#: Name of the tool that hands a step-by-step task to Auto to report. The panel
#: runs it itself, not the studio controller: it needs the provider settings,
#: and it ends the step-by-step turn instead of adding a step to it.
HANDOVER_TOOL = "run_to_report"


#: Name of the tool that proposes a temporary script. Run by the panel, after
#: the user has read it and approved, in a process of its own.
SCRIPT_TOOL = "propose_script"

# -- which calls wait for the user ---------------------------------------------
# The model decides, call by call ("ask_user"): routine actions run at once and
# are shown in the chat, decisions wait for the user. Asking before every call
# would make a ten-step task ten clicks, most of them on actions nobody refuses.
# The rules below only guard that judgement and stand in when the model gives
# none.

#: Tools that only look. They run at once, whatever the model says.
READ_ONLY_TOOLS = frozenset({"list_modules", "describe_current_module",
                             "get_studio_state", "capture_view", SKILL_TOOL})
#: Page actions that only look or show, by name.
_LOOK_ACTION = re.compile(r"^(get_|list_|describe|show_|preview_)")
#: Page actions that throw away a set of the user's work (picks corrected by
#: hand, electrodes, inputs): always the user's call. remove_air_wave and the
#: like are filters, not removals, and stay with the model.
_DISCARD_ACTION = re.compile(r"^(clear|discard|reset)|^(remove_input|remove_timelapse_files|"
                             r"delete_electrode)$")
#: Page actions that write files. Where the model names the destination, the
#: user sees it before anything is written there.
_WRITE_ACTION = re.compile(r"^(export|save)")
_DESTINATION_KEY = re.compile(r"path|file|folder|dir|destination|output", re.IGNORECASE)


def _names_destination(args: Any) -> bool:
    """Whether a page action's arguments name where to write.

    >>> _names_destination({"path": "C:/elsewhere/grid.tif"}), _names_destination({"dpi": 200})
    (True, False)
    """
    return isinstance(args, dict) and any(
        _DESTINATION_KEY.search(str(key)) and str(value or "").strip()
        for key, value in args.items())
#: With no "ask_user" from the model: actions that start a long computation ask.
_LONG_ACTION = re.compile(r"^(run|generate|interpolate|process)|invert|forward|auto_pick_all")
#: Calls in a row the assistant may run without the user; the next one asks.
#: A guard against a loop, not a budget: a task normally ends well before it.
MAX_UNASKED = 25
#: The answer to a call the user's Stop came before.
STOPPED = {"status": "skipped", "message": "The user stopped the assistant."}
#: Problems in a row - an automatic run that failed or fell short, a background
#: job that failed - handed to the assistant without the user saying anything
#: in between; after that the chat waits for the user.
MAX_LOOKS = 2
#: What the assistant is told when a job it started in the background failed.
JOB_PROBLEM = (
    "[Studio] The job you started on the {title} page ({module}) failed after your "
    "turn ended. What the page logged:\n{errors}\n\n"
    "Find out why before you answer: describe_current_module (navigate to {module} "
    "first if the user has moved on) shows the page's state and its actions. Then tell "
    "the user in a few sentences what went wrong, and fix what you can: other settings, "
    "other input, another action or module, or a temporary script where the studio has "
    "no way. Ask the user (ask_user true) before running the job again or making a choice "
    "that is theirs. If nothing you can do would help, say plainly what is missing.")
#: What the assistant is told when a page reported an error outside its own calls.
PAGE_PROBLEM = (
    "[Studio] The {title} page ({module}) reported an error that none of your tool calls "
    "returned - most likely something the user did on the page, or work they started "
    "there. What the page reported:\n{errors}\n\n"
    "Look before you answer: navigate to {module} if it is not the current module, and "
    "read its state with describe_current_module. Then tell the user in a few sentences "
    "what went wrong and how to put it right. If it is something they can simply correct "
    "(a wrong file, a missing input, a setting out of range), say what to do. Offer to do "
    "it for them, and ask (ask_user true) before you change their settings or run "
    "anything: this was their operation.")
#: Model replies that could not be used, asked for again before giving up.
MAX_REPLY_RETRIES = 1
#: A failed model call that asking again would not fix: access, quota, limits.
_NO_RETRY = re.compile(r"usage limit|quota|sign in|signed in|log ?in|api key|"
                       r"unauthori[sz]ed|forbidden|\b40[13]\b|billing", re.IGNORECASE)
#: What the assistant is told when an automatic run failed or fell short.
RUN_PROBLEM = (
    "[Studio] The automatic run {how}. What it reported:\n{report}\n\n"
    "Find out why before you answer: apply_action get_status on the Workflow page "
    "(module one_click) gives the run's steps with their status, its warnings and its "
    "output folder, and the module a failed step ran in shows its state. Then tell the "
    "user in a few sentences what went wrong, and fix what you can: correct an input or "
    "a setting with the studio's actions and run the task again with run_to_report, "
    "finish the missing part step by step, or propose a temporary script where the "
    "studio has no way. Ask the user (ask_user true) before running the task again or "
    "making a choice that is theirs. If nothing you can do would help, say plainly what "
    "is missing and what the user could provide.")

ASK_USER_FIELD = {
    "type": "boolean",
    "description": (
        "true to have the user approve this call before it runs; false to run it at "
        "once (it is shown in the chat). Ask where their judgement matters: before a "
        "long computation whose settings they have not confirmed, when you choose a "
        "value, data source or method they did not give, and before a change of plan. "
        "Do not ask for routine actions: opening a module, loading the data they named, "
        "applying values they gave, the next step of a plan they agreed to."),
}


def needs_approval(name: str, args: Dict[str, Any]) -> bool:
    """Whether a tool call waits for the user, from the model's "ask_user" and the guards.

    Examples
    --------
    >>> needs_approval('navigate', {'module': 'ert'})
    False
    >>> needs_approval('apply_action', {'action': 'load_data', 'ask_user': False})
    False
    >>> needs_approval('apply_action', {'action': 'run_timelapse'})     # model silent
    True
    >>> needs_approval('apply_action', {'action': 'run_timelapse', 'ask_user': False})
    False
    >>> needs_approval('apply_action', {'action': 'clear_picks', 'ask_user': False})
    True
    >>> needs_approval('apply_action', {'action': 'remove_input', 'ask_user': False})
    True
    >>> needs_approval('apply_action', {'action': ' delete_electrode', 'ask_user': False})
    True
    >>> needs_approval('apply_action', {'action': 'remove_air_wave', 'ask_user': False})
    False
    >>> needs_approval('apply_action', {'action': 'export_grid', 'ask_user': False,
    ...                                 'args': {'path': 'D:/out/grid.tif'}})
    True
    >>> needs_approval('apply_action', {'action': ' export_grid', 'ask_user': False,
    ...                                 'args': {'path': 'D:/out/grid.tif'}})
    True
    >>> needs_approval('apply_action', {'action': 'export_grid', 'ask_user': False, 'args': {}})
    False
    >>> needs_approval('get_studio_state', {'ask_user': True})
    False
    >>> needs_approval('navigate', {'module': 'ert', 'ask_user': 'true'})
    True
    """
    if name == SCRIPT_TOOL:
        return True
    if name in READ_ONLY_TOOLS:
        return False
    action = str(args.get("action", "")).strip() if name == "apply_action" else ""
    if action and _DISCARD_ACTION.search(action):
        return True
    if action and _WRITE_ACTION.match(action) and _names_destination(args.get("args")):
        return True
    if action and _LOOK_ACTION.match(action):
        return False
    flag = args.get("ask_user")
    if isinstance(flag, str) and flag.strip().lower() in ("true", "false"):
        flag = flag.strip().lower() == "true"
    if isinstance(flag, bool):
        return flag
    return bool(action) and bool(_LONG_ACTION.search(action))


def _with_ask_user(spec: Dict[str, Any]) -> Dict[str, Any]:
    """``spec`` with the "ask_user" argument, for every tool that may need asking."""
    if spec.get("name") in READ_ONLY_TOOLS or spec.get("name") == SCRIPT_TOOL:
        return spec
    params = dict(spec.get("parameters") or {"type": "object", "properties": {}})
    params["properties"] = {**(params.get("properties") or {}), "ask_user": ASK_USER_FIELD}
    return {**spec, "parameters": params}


def _script_spec() -> Dict[str, Any]:
    """The ``propose_script`` tool, with the rules a script must keep."""
    from PyHydroGeophysX.agents.runtime.scratch import SCRIPT_CONTRACT

    return {
        "name": SCRIPT_TOOL,
        "description": (
            "Propose a short temporary Python script to get past a problem the studio's "
            "own actions cannot solve. The user reads the purpose and the code and "
            "decides; it runs only if they approve, in a separate process with a time "
            "limit, reading the inputs you name and writing only to a new folder of its "
            "own. The result (what main returned, the files it wrote with their full "
            "paths, what it printed, or the error) comes back to you. " + SCRIPT_CONTRACT),
        "parameters": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": (
                    "For the user, in two or three plain sentences: what went wrong, "
                    "what the script does about it, and what you will do with its output.")},
                "code": {"type": "string", "description": "The whole script, defining main(inputs, out)."},
                "inputs": {"type": "object", "additionalProperties": True, "description": (
                    "name -> path of each file or folder the script reads; only paths the "
                    "user gave or a tool result reported.")},
            },
            "required": ["purpose", "code"],
        },
    }


def _handover_spec(agent) -> Dict[str, Any]:
    """The ``run_to_report`` tool, with ``agent``'s input roles spelled out."""
    roles = ", ".join(key for _label, key in agent.input_roles)
    ordered = ", ".join(agent.ordered_roles)
    return {
        "name": HANDOVER_TOOL,
        "description": (
            "Hand the rest of the task to Auto to report: the automatic workflow carries "
            "on from where the conversation is now to the final products and the written "
            "report, with no approval at each step. Call it when the user wants the end "
            "result rather than the next step, in whatever words. Data loaded and results "
            "finished in the studio (for example a completed ERT inversion) are carried "
            "over and not redone."),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "The whole task as the automatic run should read it: what the user "
                        "asked for, the data and every choice made so far (instrument, "
                        "geometry, single or time-lapse, inversion settings, the fit "
                        "obtained) and what is left to do (for example water content with "
                        "uncertainty, then the report)."),
                },
                "inputs": {
                    "type": "object",
                    "description": (
                        "Data files to give the run, as {role: path, or a list of paths "
                        f"for {ordered or 'an ordered role'}}}. Roles: {roles}. Use only "
                        "paths the user gave or a tool result reported. Files the user "
                        "attached to the chat are included already."),
                    "additionalProperties": True,
                },
            },
            "required": ["goal"],
        },
    }


class ChatInputEdit(QPlainTextEdit):
    """Multi-line input box that sends on Ctrl/Command+Enter."""

    sendRequested = Signal()
    filesPasted = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedHeight(190)
        self.setAcceptDrops(True)

    def insertFromMimeData(self, source) -> None:
        paths = [url.toLocalFile() for url in source.urls() if url.isLocalFile()] if source.hasUrls() else []
        if not paths and source.hasText():
            lines = [line.strip().strip('"') for line in source.text().splitlines() if line.strip()]
            if lines and all(Path(line).is_absolute() and Path(line).exists() for line in lines):
                paths = lines
        if paths:
            self.filesPasted.emit([str(Path(path)) for path in paths])
            return
        super().insertFromMimeData(source)

    def canInsertFromMimeData(self, source) -> bool:
        return source.hasUrls() or super().canInsertFromMimeData(source)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            self.insertFromMimeData(event.mimeData())
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: D401 - Qt event override
        if (
            event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)
        ):
            self.sendRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _default_provider_id() -> str:
    """Start the desktop with Codex; automatic setup handles missing installs."""
    return "codex_cli"


class AssistantChatPanel(QWidget):
    """Conversation panel that turns plain language into approved tool calls."""

    def __init__(
        self,
        controller: StudioController,
        log=None,
        provider: Optional[Any] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._log = log or (lambda *a, **k: None)

        if provider is not None:
            self._provider = provider
            self._provider_id = getattr(provider, "id", "openai")
        else:
            self._provider_id = _default_provider_id()
            self._provider = make_provider(self._provider_id)

        self._messages: List[Dict[str, Any]] = []
        self._tool_queue: List[Dict[str, Any]] = []
        self._pending_workflow_text: Optional[str] = None
        self._current_call: Optional[Dict[str, Any]] = None
        self._executed_in_turn = False
        self._awaiting_user = False
        # Step by step: calls run since the user last had a say, and whether
        # they pressed Stop during this turn.
        self._unasked = 0
        self._stop_requested = False
        # Until the user touches the Reasoning box, effort follows the level.
        self._effort_is_user_set = False
        self._paused_resume: Optional[Dict[str, Any]] = None  # resume action while paused
        self._resume_seq = 0
        # "guided" (step by step) or "auto" (Auto to report); None until the
        # user answers the choice their first request brings up.
        self._mode: Optional[str] = None
        # The request (text, reference sources) waiting on that choice.
        self._pending_choice: Optional[Tuple[str, tuple]] = None
        # True from the start of an automatic run the chat began to its end.
        self._auto_running = False
        # Problems handed to the assistant since the user last spoke.
        self._looks = 0
        # Background jobs that failed, waiting for the assistant to be free.
        self._job_failures: List[Dict[str, Any]] = []
        # Model replies asked for again in this turn because they could not be used.
        self._reply_retries = 0
        # Errors the pages hit outside the assistant's own calls: the user's own
        # operations on a page, a button whose code raised.
        watch = getattr(self._controller, "watch_page_errors", None)
        if callable(watch):
            watch(self._on_page_failure)
        # A step-by-step task handed to Auto to report: its goal, waiting for
        # the tool batch to be answered, and the studio work it carries over.
        self._pending_handover: Optional[str] = None
        self._handover_work: Optional[Dict[str, Any]] = None
        self._script_worker: Optional[ScriptWorker] = None
        self._busy = False
        self._cli_states = {}
        self._cli_models = {}
        self._cli_model_errors = {}
        self._cli_workers = {}
        self._cli_logins = {}
        self._closing = False
        self._worker: Optional[LlmCallWorker] = None
        # Captures answered this turn, flushed once the tool batch is drained.
        self._pending_images: List[Tuple[str, Dict[str, Any]]] = []
        self._image_seq = 0
        self._vision = False
        self._tool_specs: List[Dict[str, Any]] = []
        self._system = ""
        self._reference_workers = []
        self._conversation_scope = self._controller.attachment_scope()
        # (provider, whether it could send) when its warm-up was last started.
        self._prewarmed: Optional[Tuple[Any, bool]] = None
        self._thinking_started = 0.0
        self._refresh_capabilities()

        self._build_ui()
        self._sync_settings_widgets()
        # Start open only when there is nothing to send with: otherwise a first
        # run would report "no API key" with no visible way to supply one.
        if not self._provider.available()[0] and not assistant_registry.active().offline_workflow:
            self._settings_btn.setChecked(True)
        self._sync_assistant_options()
        self._reset_conversation()

    # -- model capabilities --------------------------------------------------
    def _refresh_capabilities(self) -> None:
        """Re-derive the tool list and system prompt for the selected model.

        Called whenever provider or model changes, since vision support (and so
        the ``capture_view`` tool and its rules) is a per-model property.
        """
        try:
            self._vision = bool(self._provider.supports_vision())
        except Exception:  # noqa: BLE001 - an odd provider simply gets no vision
            self._vision = False
        agent = assistant_registry.active()
        # Read once per assistant and model; a skill added on disk is seen on
        # the next change of either, or the next start.
        self._skills = agent.load_skills()
        self._tool_specs = [_with_ask_user(spec) for spec in
                            tool_specs(vision=self._vision)
                            + ([skill_tool_spec()] if self._skills else [])
                            + [_handover_spec(agent), _script_spec()]]
        self._system = SYSTEM_PROMPT.format(
            name=agent.name, title=agent.title, summary=agent.summary,
            persona=agent.persona,
            modules=self._controller.capabilities_summary(agent.studio_modules),
            vision_rules=VISION_RULES if self._vision else "",
            skill_rules=(SKILL_RULES.format(index=skill_index(self._skills, width=240))
                         if self._skills else ""),
        )

    @staticmethod
    def _name() -> str:
        """The name of the assistant the user is talking to."""
        return assistant_registry.active().name

    # -- UI ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        # Which assistant this is. One that cannot run here - not installed, or
        # still in development - is listed but greyed out, with the reason.
        self._assistant_combo = QComboBox()
        self._assistant_combo.setToolTip("The AI assistant you are talking to")
        # A styled delegate, so a greyed-out entry is drawn grey: the combo's
        # default menu delegate draws every entry alike.
        self._assistant_combo.setItemDelegate(QStyledItemDelegate(self._assistant_combo))
        for agent in assistant_registry.assistants():
            # The name alone: the header has little room, and the domain is in
            # the tooltip, the welcome note and the Workflow page's title.
            ready, why = agent.availability()
            label = agent.name if ready else f"{agent.name} ({agent.status if agent.status != 'ready' else 'unavailable'})"
            self._assistant_combo.addItem(label, agent.key)
            index = self._assistant_combo.count() - 1
            self._assistant_combo.setItemData(
                index, f"{agent.title}\n{agent.summary}" + (f"\n\n{why}" if why else ""),
                Qt.ToolTipRole)
            if not ready:
                item = self._assistant_combo.model().item(index)
                if item is not None:
                    item.setEnabled(False)
                # The combo's own delegate ignores the stylesheet's disabled
                # colour; a grey that reads on both appearances says it instead.
                self._assistant_combo.setItemData(index, QColor("#8e8e93"), Qt.ForegroundRole)
        current = self._assistant_combo.findData(assistant_registry.active().key)
        self._assistant_combo.setCurrentIndex(max(0, current))
        self._assistant_combo.currentIndexChanged.connect(self._on_assistant_changed)
        header.addWidget(self._assistant_combo)
        header.addStretch(1)
        # Provider, model, key and the tool switches are chosen once a session
        # and then only read back, so they collapse behind this button. What
        # stays out is what changes per message: the status line (can I send,
        # and what does it cost).
        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setCheckable(True)
        self._settings_btn.setIcon(theme.icon("fa5s.chevron-right"))
        self._settings_btn.setToolTip("Provider, model, API key, and the RAG / MCP switches")
        self._settings_btn.toggled.connect(self._on_settings_toggled)
        header.addWidget(self._settings_btn)
        self._new_btn = QPushButton("New chat")
        self._new_btn.setIcon(theme.icon("fa5s.broom"))
        self._new_btn.clicked.connect(self._reset_conversation)
        header.addWidget(self._new_btn)
        root.addLayout(header)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        theme.set_tone(self._status_label, "muted")
        root.addWidget(self._status_label)

        self._settings = self._build_settings()
        self._settings.setVisible(False)
        root.addWidget(self._settings)

        self._transcript = QTextBrowser()
        self._transcript.setOpenExternalLinks(True)
        root.addWidget(self._transcript, stretch=1)

        # Shown while a model request is in flight, with how long it has been.
        # The Send button's "…" alone is too little sign of the wait that
        # step-by-step spends most of its time in. The orb is the agent's presence in the
        # studio's one visual language: the same one is on the Workflow page
        # and on every panel an automatic run is working in.
        self._thinking_row = QWidget()
        thinking_layout = QHBoxLayout(self._thinking_row)
        thinking_layout.setContentsMargins(0, 0, 0, 0)
        thinking_layout.setSpacing(8)
        self._thinking_orb = AiOrb(22)
        thinking_layout.addWidget(self._thinking_orb)
        self._thinking = QLabel()
        self._thinking.setObjectName("aquahThinking")
        thinking_layout.addWidget(self._thinking, 1)
        self._thinking_row.setVisible(False)
        root.addWidget(self._thinking_row)
        # What an automatic run is doing right now, while it runs; None when
        # no run is going.
        self._live: Optional[str] = None
        self._thinking_clock = QTimer(self)
        self._thinking_clock.setInterval(1000)
        self._thinking_clock.timeout.connect(self._update_thinking)

        # Pending tool-call confirmation card (hidden until needed).
        self._confirm = QFrame()
        # Styled by the studio stylesheet (#ConfirmCard), so it follows the
        # Light and Dark appearances.
        self._confirm.setObjectName("ConfirmCard")
        clay = QVBoxLayout(self._confirm)
        self._confirm_label = QLabel()
        self._confirm_label.setWordWrap(True)
        self._confirm_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        clay.addWidget(self._confirm_label)
        # A proposed script, in a box that can be read and copied: approving
        # code means having read it. Styled like the Workflow page's (#agentCode).
        self._confirm_code = QPlainTextEdit()
        self._confirm_code.setObjectName("agentCode")
        self._confirm_code.setReadOnly(True)
        self._confirm_code.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._confirm_code.setMaximumHeight(240)
        self._confirm_code.setVisible(False)
        clay.addWidget(self._confirm_code)
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

        # How to carry out a request, asked once it has been sent rather than
        # picked beforehand: the choice is about this task, and a picker set
        # ahead of it is easy to leave on the wrong mode.
        self._choice = QFrame()
        self._choice.setObjectName("ConfirmCard")
        choice_layout = QVBoxLayout(self._choice)
        self._choice_label = QLabel()
        self._choice_label.setWordWrap(True)
        choice_layout.addWidget(self._choice_label)
        choice_btns = QHBoxLayout()
        choice_btns.addStretch(1)
        self._guided_btn = QPushButton("Step by step")
        self._guided_btn.setIcon(theme.icon("fa5s.list-ol"))
        self._guided_btn.clicked.connect(lambda: self._choose_mode("guided"))
        self._auto_btn = QPushButton("Auto to report")
        self._auto_btn.setProperty("primary", True)
        self._auto_btn.setIcon(theme.icon("fa5s.forward", color="#ffffff"))
        self._auto_btn.clicked.connect(lambda: self._choose_mode("auto"))
        choice_btns.addWidget(self._guided_btn)
        choice_btns.addWidget(self._auto_btn)
        choice_layout.addLayout(choice_btns)
        self._choice.setVisible(False)
        root.addWidget(self._choice)

        # Multi-line input row.
        self._attachments_box = QWidget()
        attachments_layout = QHBoxLayout(self._attachments_box)
        attachments_layout.setContentsMargins(0, 0, 0, 0)
        self._attachments = QListWidget()
        self._attachments.setMaximumHeight(82)
        self._attachments.setTextElideMode(Qt.ElideMiddle)
        self._attachments.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._attachments.setToolTip('Files shared with the assistant; RAG references are searched when RAG is enabled.')
        attachments_layout.addWidget(self._attachments, 1)
        self._remove_attachment_btn = QPushButton('Remove')
        self._remove_attachment_btn.clicked.connect(self._remove_attachment)
        attachments_layout.addWidget(self._remove_attachment_btn)
        root.addWidget(self._attachments_box)
        self._attachments_box.hide()
        composer = QVBoxLayout()
        composer.setSpacing(5)
        self._input = ChatInputEdit()
        self._input.setPlaceholderText("e.g. Build a 3D crosshole mesh with 12 sensors per borehole")
        self._input.sendRequested.connect(self._on_send)
        self._input.filesPasted.connect(self._paste_files)
        self._input.textChanged.connect(self._prewarm_provider)
        self._send_btn = QPushButton("Send")
        self._send_btn.setProperty("primary", True)
        self._send_btn.setIcon(theme.icon("fa5s.paper-plane", color="#ffffff"))
        self._send_btn.clicked.connect(self._on_send)
        composer.addWidget(self._input)
        input_actions = QHBoxLayout()
        input_hint = QLabel('Paste or drop files · Ctrl+Enter to send')
        theme.set_tone(input_hint, 'muted')
        input_hint.setWordWrap(True)
        input_actions.addWidget(input_hint)
        input_actions.addStretch(1)
        # Step by step runs routine actions without asking; Stop is the
        # user's way to cut in while it does.
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setIcon(theme.icon("fa5s.stop"))
        self._stop_btn.setToolTip("Stop the assistant after the action in hand. Anything "
                                  "already running on a page carries on there.")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setVisible(False)
        input_actions.addWidget(self._stop_btn)
        input_actions.addWidget(self._send_btn)
        composer.addLayout(input_actions)
        root.addLayout(composer)
        self._attachment_timer = QTimer(self)
        self._attachment_timer.setInterval(1000)
        self._attachment_timer.timeout.connect(self._refresh_attachment_list)
        self._attachment_timer.start()

    def _refresh_attachment_list(self):
        rows = self._controller.chat_attachments()
        signature = [(r['role'], r['path'], r['rag']) for r in rows]
        if signature == getattr(self, '_attachment_signature', None):
            return
        self._attachment_signature = signature
        self._attachments.clear()
        labels = {key: label for label, key in self._controller.chat_attachment_roles()}
        for row in rows:
            label = 'AI will identify the purpose when you send your task' if row['role'] == 'pending' else labels.get(row['role'], row['role'])
            item = QListWidgetItem(Path(row['path']).name)
            item.setToolTip(row['path'] + '\n' + label)
            item.setData(Qt.UserRole, row)
            self._attachments.addItem(item)
        row_height = self._attachments.sizeHintForRow(0) if rows else 0
        self._attachments.setFixedHeight(min(82, max(28, row_height * min(len(rows), 3)
                                                    + 2 * self._attachments.frameWidth())))
        self._attachments_box.setVisible(bool(rows))

    def _paste_files(self, paths):
        if self._busy:
            return
        result = self._controller.queue_chat_files(paths)
        if result.get('status') != 'ok':
            self._render_note(html.escape(result.get('error', 'Could not add files.')))
            return
        self._refresh_attachment_list()
        if result['added']:
            names = ', '.join(Path(path).name for path in result['added'])
            self._render_note('Files received: ' + html.escape(names) + '. Describe your task; I will identify how to use them.')

    def _remove_attachment(self):
        if self._busy or self._attachments.currentItem() is None:
            return
        row = self._attachments.currentItem().data(Qt.UserRole)
        result = self._controller.remove_chat_attachment(row['role'], row['path'])
        if result.get('status') != 'ok':
            self._render_note(html.escape(result.get('error', 'Could not remove the file.')))
        self._refresh_attachment_list()
        if not self._controller.pending_chat_files():
            self._file_goal = ''

    def _build_settings(self) -> QWidget:
        """Build the collapsible configuration block.

        Laid out on a grid rather than a row per control: each of these is a
        short word or a model name, so one per full-width row would spend most
        of the panel's height on empty combo boxes. Pairing them across four
        columns keeps the labels aligned and about halves the block.
        """
        box = QFrame()
        lay = QGridLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(6)
        lay.setVerticalSpacing(4)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        # The cost ladder, on its own full-width row: its entries name a level,
        # a model and a price, which no half row can show. Picking a level sets
        # the model for whichever provider is selected, so switching provider
        # keeps the level you chose.
        self._level_row = QWidget()
        lrow = QHBoxLayout(self._level_row)
        lrow.setContentsMargins(0, 0, 0, 0)
        lrow.setSpacing(6)
        lrow.addWidget(QLabel("Level"))
        self._level_combo = QComboBox()
        for tier_id in TIER_ORDER:
            self._level_combo.addItem(tier_id, tier_id)  # text filled in by _sync
        self._level_combo.addItem("Custom model", TIER_CUSTOM)
        self._level_combo.currentIndexChanged.connect(self._on_level_changed)
        # What the level is for is a tooltip, not a label. Visible, it would wrap
        # to three lines - more height than every other control here put
        # together - to say something that is read once.
        lrow.addWidget(self._level_combo, stretch=1)
        lay.addWidget(self._level_row, 0, 0, 1, 4)

        lay.addWidget(QLabel("Provider"), 1, 0)
        self._provider_combo = QComboBox()
        for pid in PROVIDER_ORDER:
            self._provider_combo.addItem(PROVIDER_META[pid]["label"], pid)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        lay.addWidget(self._provider_combo, 1, 1)

        lay.addWidget(QLabel("Model"), 1, 2)
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        lay.addWidget(self._model_combo, 1, 3)

        self._base_url_row = QWidget()
        burl = QHBoxLayout(self._base_url_row)
        burl.setContentsMargins(0, 0, 0, 0)
        burl.setSpacing(6)
        burl.addWidget(QLabel("Base URL"))
        self._base_url_edit = QLineEdit()
        self._base_url_edit.setPlaceholderText("https://api.example.com/v1")
        burl.addWidget(self._base_url_edit, stretch=1)
        lay.addWidget(self._base_url_row, 2, 0, 1, 4)

        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.Password)
        self._key_edit.setPlaceholderText("Paste API key for this session")
        lay.addWidget(self._key_edit, 3, 0, 1, 3)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply_settings)
        lay.addWidget(self._apply_btn, 3, 3)

        # One line: where sign-in stands, and the one action that fits it -
        # Log in, Log out, Cancel or Retry - as a small button at its end. A
        # full-width "Log In" under "Signed in. Ready to chat." would be easy to
        # click by mistake and say the opposite of the state.
        self._cli_login_row = QWidget()
        login_row = QHBoxLayout(self._cli_login_row)
        login_row.setContentsMargins(0, 0, 0, 0)
        login_row.setSpacing(6)
        self._login_help = QLabel()
        self._login_help.setWordWrap(True)
        self._login_help.setOpenExternalLinks(True)
        login_row.addWidget(self._login_help, 1)
        self._login_btn = QPushButton("Log in")
        self._login_btn.clicked.connect(self._on_cli_login_button)
        self._cli_retry_btn = QPushButton("Retry")
        self._cli_retry_btn.setToolTip("Run the CLI setup again.")
        self._cli_retry_btn.clicked.connect(self._retry_cli_setup)
        self._cancel_login_btn = QPushButton("Cancel")
        self._cancel_login_btn.setToolTip("Stop waiting for the browser sign-in.")
        self._cancel_login_btn.clicked.connect(self._cancel_cli_login)
        for button in (self._login_btn, self._cli_retry_btn, self._cancel_login_btn):
            button.setProperty("compact", True)
            login_row.addWidget(button, 0, Qt.AlignTop)
        lay.addWidget(self._cli_login_row, 5, 0, 1, 4)

        lay.addWidget(QLabel("Reasoning"), 4, 0)
        self._reasoning = QComboBox()
        self._reasoning.addItems(['none', 'low', 'medium', 'high', 'xhigh', 'max'])
        self._reasoning.setCurrentText('medium')
        self._reasoning.setToolTip(
            'How hard the model thinks before answering. Follows the level until '
            'you change it, then stays where you put it. Chat pays this on every '
            'tool call, so a high effort is felt one step at a time. Applies to '
            'reasoning models; not a display of private chain-of-thought.')
        self._reasoning.currentTextChanged.connect(self._on_reasoning_changed)
        lay.addWidget(self._reasoning, 4, 1)

        # Short labels, full sentences in the tooltips: the names are what you
        # scan for, the explanation is what you hover for.
        self._rag = QCheckBox('RAG')
        self._rag.setChecked(True)
        self._rag.setToolTip('Retrieve relevant excerpts from local documentation and selected reference files; record sources in the report.')
        self._mcp = QCheckBox('MCP')
        self._mcp.setToolTip('Connect to the bundled stdio MCP server and retrieve available workflow APIs. Requires the mcp extra.')
        lay.addWidget(self._rag, 4, 2)
        lay.addWidget(self._mcp, 4, 3)
        return box

    # -- assistant -------------------------------------------------------------
    def _on_assistant_changed(self, _index: int) -> None:
        """Talk to another assistant: a new conversation, with its own prompt."""
        key = self._assistant_combo.currentData()
        if not key or key == assistant_registry.active().key:
            return
        if hasattr(self._controller, "set_assistant"):
            self._controller.set_assistant(key)
        else:
            assistant_registry.set_active(key)
        self._refresh_capabilities()
        self._reset_conversation()

    def sync_assistant(self) -> None:
        """Show the active assistant in the picker, e.g. after a restore."""
        index = self._assistant_combo.findData(assistant_registry.active().key)
        if index >= 0 and index != self._assistant_combo.currentIndex():
            self._assistant_combo.blockSignals(True)
            self._assistant_combo.setCurrentIndex(index)
            self._assistant_combo.blockSignals(False)
            self._refresh_capabilities()
            self._reset_conversation()
        self._sync_assistant_options()

    def _sync_assistant_options(self):
        agent = assistant_registry.active()
        for key, widget in (('rag', self._rag), ('mcp', self._mcp)):
            # Host chat retrieval is available even when a plugin does not
            # implement its own workflow retrieval controls.
            available = key == 'rag' or key in getattr(agent, 'retrieval', ('rag', 'mcp'))
            widget.setVisible(available)
            widget.setEnabled(available)
        if agent.offline_workflow:
            self._settings_btn.setChecked(False)
        self._refresh_ready_state()

    def start_workflow(self, text):
        """Start from the task page using the same provider and chat contract."""
        if self._busy:
            # A module action can request an automatic run from inside the tool
            # loop. Finish pairing that call with its result before switching.
            self._pending_workflow_text = str(text)
            return
        ready = self._provider.available()[0]
        if self._provider_id in CLI_PROVIDER_IDS:
            ready = ready and self._cli_states.get(self._provider_id, {}).get('state') == 'ready'
        if not ready:
            self._settings_btn.setChecked(True)
            self._render_note('Configure the AI provider, apply the settings, then start the task again.')
            return
        # Starting from the Workflow page, or handed over from step by step, is
        # itself the choice of Auto to report.
        self._pending_choice = None
        self._choice.setVisible(False)
        self._mode = "auto"
        self._input.setPlainText(text)
        self._on_send()

    # -- settings handling ---------------------------------------------------
    def _on_settings_toggled(self, shown: bool) -> None:
        """Show or hide the configuration block, and point the chevron at it.

        A drawn icon rather than a text arrow: a chevron character disappears
        under font fallback, and a checkable button's pressed look is too quiet
        to read as open/closed on its own.
        """
        self._settings.setVisible(shown)
        self._settings_btn.setIcon(
            theme.icon("fa5s.chevron-down" if shown else "fa5s.chevron-right"))

    def _sync_settings_widgets(self) -> None:
        """Reflect the active provider into the combos / rows (no signals)."""
        meta = PROVIDER_META.get(self._provider_id, PROVIDER_META["openai"])
        self._provider_combo.blockSignals(True)
        idx = self._provider_combo.findData(self._provider_id)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)
        self._provider_combo.blockSignals(False)

        self._populate_model_combo()

        self._sync_level_widgets()
        self._base_url_row.setVisible(bool(meta.get("needs_base_url")))
        # Show the endpoint already set (OPENAI_COMPAT_BASE_URL), so Apply
        # keeps it rather than clearing it with an empty field.
        self._base_url_edit.setText(getattr(self._provider, "_base_url", None) or "")
        self._key_edit.setPlaceholderText(f"Paste {meta['env_key']} for this session")
        cli = self._provider_id in CLI_PROVIDER_IDS
        self._key_edit.clear()
        self._key_edit.setVisible(not cli)
        self._apply_btn.setVisible(not cli)
        self._login_help.setText(meta.get("login_help", ""))
        self._login_help.setVisible(cli)
        self._cli_login_row.setVisible(cli)
        self._reasoning.setEnabled(self._provider_id != "claude_code")
        if cli:
            if self._provider_id not in self._cli_states:
                self._start_cli_setup(self._provider)
            self._refresh_cli_controls()

    def _populate_model_combo(self):
        """Refresh catalog options without applying or changing a model."""
        pid = self._provider_id
        meta = PROVIDER_META.get(pid, PROVIDER_META["openai"])
        models = self._cli_models.get(pid, meta.get("models", []))
        selected = self._provider.model
        was_blocked = self._model_combo.blockSignals(True)
        try:
            self._model_combo.clear()
            self._model_combo.addItems(models)
            if pid in CLI_PROVIDER_IDS:
                index = self._model_combo.findText("default")
                if index >= 0:
                    self._model_combo.setItemData(
                        index, "Follow your CLI's default model configuration.", Qt.ToolTipRole)
            index = self._model_combo.findText(selected)
            if index >= 0:
                self._model_combo.setCurrentIndex(index)
            else:
                self._model_combo.setCurrentText(selected)
        finally:
            self._model_combo.blockSignals(was_blocked)
        tooltip = "Select a model or enter a model ID."
        if pid in CLI_PROVIDER_IDS:
            tooltip += " default follows your CLI configuration."
        if pid == "codex_cli":
            tooltip += " Available models are read automatically from Codex after sign-in."
            if pid in self._cli_model_errors:
                tooltip += "\nCould not refresh models: " + self._cli_model_errors[pid]
        self._model_combo.setToolTip(tooltip)

    def _start_cli_setup(self, provider, check_only=False):
        pid = provider.id
        if pid in self._cli_workers or self._closing:
            return
        self._cli_states[pid] = {"state": "preparing", "message": "Checking login…" if check_only else "Setting up automatically…"}
        app = QApplication.instance()
        worker = CliSetupWorker(provider, check_only, parent=app)
        self._cli_workers[pid] = worker
        app.aboutToQuit.connect(worker.cancel_and_wait)
        worker.progress.connect(lambda text, p=pid: self._cli_setup_progress(p, text))
        worker.succeeded.connect(lambda result, p=pid: self._cli_setup_done(p, result))
        worker.failed.connect(lambda message, p=pid: self._cli_setup_failed(p, message))
        worker.finished.connect(lambda p=pid: self._cli_setup_finished(p))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        if pid == self._provider_id:
            self._refresh_ready_state()

    def _cli_setup_progress(self, pid, message):
        if self._closing:
            return
        self._cli_states[pid] = {"state": "preparing", "message": message}
        if pid == self._provider_id:
            self._refresh_ready_state()

    def _cli_setup_finished(self, pid):
        self._cli_workers.pop(pid, None)
        if not self._closing and pid == self._provider_id:
            self._refresh_ready_state()

    def _cli_setup_done(self, pid, result):
        if self._closing:
            return
        if "models" in result:
            self._cli_models[pid] = list(dict.fromkeys(["default", *result["models"]]))
            self._cli_model_errors.pop(pid, None)
        elif result.get("model_error"):
            self._cli_model_errors[pid] = result["model_error"]
        self._cli_states[pid] = {"state": "ready" if result["logged_in"] else "login",
                                 "message": result["message"]}
        if pid == self._provider_id:
            self._populate_model_combo()
            if not result["logged_in"]:
                self._settings_btn.setChecked(True)
            self._refresh_ready_state()

    def _cli_setup_failed(self, pid, message):
        if self._closing:
            return
        self._cli_states[pid] = {"state": "error", "message": "Setup could not finish: " + message[:600]}
        if pid == self._provider_id:
            self._settings_btn.setChecked(True)
            self._refresh_ready_state()

    def _refresh_cli_controls(self):
        status = self._cli_states.get(self._provider_id, {"state": "preparing", "message": "Setting up automatically…"})
        self._login_help.setText(html.escape(status["message"]))
        signed_in = status["state"] == "ready"
        label = PROVIDER_META.get(self._provider_id, {}).get("label", "the CLI").split(" (")[0]
        self._login_btn.setText("Log out" if signed_in else "Log in")
        self._login_btn.setVisible(status["state"] in ("login", "ready"))
        self._login_btn.setEnabled(not self._busy)
        self._login_btn.setToolTip(
            f"Sign {label} out on this computer. To use another account, log out, "
            "then log in." if signed_in else "Sign in with your subscription account "
                                             "in the browser.")
        self._cli_retry_btn.setVisible(status["state"] == "error")
        self._cli_retry_btn.setEnabled(self._provider_id not in self._cli_workers)
        self._cancel_login_btn.setVisible(status["state"] == "signing_in")
        url = status.get("url")
        if url:
            self._login_help.setText(html.escape(status["message"]) +
                                     f' <a href="{html.escape(url, quote=True)}">Open login page</a>')

    def _retry_cli_setup(self):
        self._start_cli_setup(self._provider)

    def _on_cli_login_button(self):
        """Log out when signed in, log in otherwise: the button says which."""
        state = self._cli_states.get(self._provider_id, {}).get("state")
        if state == "ready":
            self._on_cli_logout()
        else:
            self._on_cli_login()

    def _on_cli_logout(self):
        """Sign the CLI out, after saying that this is not only the studio's login.

        The CLI keeps one sign-in for the computer, so logging out here also
        logs out the same CLI in a terminal. Asked first: the button sits beside
        the status line, and a slip should not end someone's sessions.
        """
        provider = self._provider
        pid = provider.id
        if pid not in CLI_PROVIDER_IDS or pid in self._cli_logins or self._busy:
            return
        label = PROVIDER_META.get(pid, {}).get("label", "the CLI").split(" (")[0]
        answer = QMessageBox.question(
            self, "Log out",
            f"Log out of {label}?\n\nThis signs {label} out on this computer, including "
            "where you use it in a terminal. You can log in again here.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return
        try:
            command = provider._command()
        except FileNotFoundError:
            self._start_cli_setup(provider)
            return
        process = QProcess(self)
        env = QProcessEnvironment()
        for key, value in provider._environment().items():
            env.insert(key, value)
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(str(Path.home()))
        process.setProcessChannelMode(QProcess.MergedChannels)
        self._cli_logins[pid] = process
        self._cli_states[pid] = {"state": "signing_out", "message": "Logging out…"}
        process.errorOccurred.connect(lambda error, p=provider, proc=process: self._cli_login_failed(p, proc))
        # Read the saved sign-in back, whatever the exit code, as after a login.
        process.finished.connect(lambda code, status, p=provider, proc=process: self._cli_login_finished(p, proc, code))
        args = ["logout"] if pid == "codex_cli" else ["auth", "logout"]
        process.start(command[0], command[1:] + args)
        self._refresh_ready_state()

    def _on_cli_login(self):
        provider = self._provider
        pid = provider.id
        if pid not in CLI_PROVIDER_IDS or pid in self._cli_logins or self._busy:
            return
        try:
            command = provider._command()
        except FileNotFoundError:
            self._start_cli_setup(provider)
            return
        process = QProcess(self)
        env = QProcessEnvironment()
        for key, value in provider._environment().items():
            env.insert(key, value)
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(str(Path.home()))
        process.setProcessChannelMode(QProcess.MergedChannels)
        self._cli_logins[pid] = process
        self._cli_states[pid] = {"state": "signing_in", "message": "Complete sign-in in your browser…"}
        process.readyReadStandardOutput.connect(lambda p=pid, proc=process: self._cli_login_output(p, proc))
        process.errorOccurred.connect(lambda error, p=provider, proc=process: self._cli_login_failed(p, proc))
        process.finished.connect(lambda code, status, p=provider, proc=process: self._cli_login_finished(p, proc, code))
        args = ["login"] if pid == "codex_cli" else ["auth", "login"]
        process.start(command[0], command[1:] + args)
        self._refresh_ready_state()

    def _cli_login_output(self, pid, process):
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        status = self._cli_states.get(pid)
        if not status or status["state"] != "signing_in":
            return
        # The CLI opens the browser. Keep an official-link fallback in the UI
        # without copying tokens, running a shell or opening arbitrary URLs.
        output = (status.get("output", "") + output)[-24000:]
        status["output"] = output
        for url in re.findall(r"https://[^\s<>]+", output):
            if urlsplit(url).hostname in ("auth.openai.com", "login.openai.com", "claude.ai"):
                status["url"] = url
                break
        if pid == self._provider_id:
            self._refresh_cli_controls()

    def _cli_login_failed(self, provider, process):
        if self._cli_logins.get(provider.id) is not process:
            return
        self._cli_logins.pop(provider.id, None)
        self._cli_setup_failed(provider.id, "The CLI could not start. " + process.errorString())
        process.deleteLater()

    def _cli_login_finished(self, provider, process, code):
        if self._cli_logins.get(provider.id) is not process:
            return
        self._cli_logins.pop(provider.id, None)
        process.deleteLater()
        # Check saved authentication even on nonzero exit, because the browser
        # may have completed sign-in while the CLI was closing.
        self._start_cli_setup(provider, check_only=True)

    def _cancel_cli_login(self):
        process = self._cli_logins.pop(self._provider_id, None)
        if process is not None:
            process.kill()
            process.waitForFinished(1000)
            process.deleteLater()
            self._start_cli_setup(self._provider, check_only=True)

    def closeEvent(self, event):
        self._closing = True
        for worker in self._reference_workers:
            worker.requestInterruption()
        for process in list(self._cli_logins.values()):
            process.kill()
            process.waitForFinished(1000)
        for worker in list(self._cli_workers.values()):
            worker.requestInterruption()
        super().closeEvent(event)

    def _sync_level_widgets(self) -> None:
        """Relabel the ladder for this provider and select the level in force.

        The level is derived from the model rather than remembered separately, so
        a model set by an environment variable or typed into the model box is
        described correctly instead of leaving the selector claiming a level the
        provider is not on.
        """
        tiered = provider_has_tiers(self._provider_id)
        self._level_row.setVisible(tiered)
        if not tiered:
            return
        self._level_combo.blockSignals(True)
        for i, tier_id in enumerate(TIER_ORDER):
            self._level_combo.setItemText(i, tier_label(tier_id, self._provider_id))
        current = tier_for_model(self._provider_id, self._provider.model)
        pos = self._level_combo.findData(current)
        if pos >= 0:
            self._level_combo.setCurrentIndex(pos)
        self._level_combo.blockSignals(False)
        self._level_combo.setToolTip(self._level_help_text(current))
        self._apply_tier_effort(current)

    def _on_reasoning_changed(self, value: str) -> None:
        """A change the user made here outranks the level from then on."""
        self._effort_is_user_set = True
        self._provider.reasoning_effort = value

    def _apply_tier_effort(self, tier_id: str) -> None:
        """Move reasoning effort with the level, unless the user set it.

        Chat drives the studio one tool call at a time and each call is its own
        request, so effort is paid per step rather than once per question. The
        cheapest level at medium effort would be the slowest thing in the app, so
        each level carries the effort it is for. A value the user chose is
        left alone - the combo stays the override.
        """
        if self._effort_is_user_set:
            return
        effort = tier_effort(tier_id)
        if effort == self._reasoning.currentText():
            return
        self._reasoning.blockSignals(True)
        self._reasoning.setCurrentText(effort)
        self._reasoning.blockSignals(False)
        self._provider.reasoning_effort = effort

    def _level_help_text(self, tier_id: str) -> str:
        """What the selected level is for, and where to go when it falls short."""
        tier = MODEL_TIERS.get(tier_id)
        if tier is None:
            price = price_label(self._provider.model)
            tail = f" — {price}" if price else ""
            return f"Custom model — not one of the three levels{tail}."
        nxt = TIER_ORDER[TIER_ORDER.index(tier_id) + 1] if tier_id != TIER_ORDER[-1] else None
        after = (f" Move up to {MODEL_TIERS[nxt]['name']} when this level cannot finish the job."
                 if nxt else " There is nothing above this level, so use it sparingly.")
        return tier["purpose"] + after

    def _on_level_changed(self, _index: int) -> None:
        tier_id = self._level_combo.currentData()
        model = tier_model(tier_id, self._provider_id) if tier_id else None
        if not model:  # "Custom model" — keep whatever is in the model box
            self._level_combo.setToolTip(self._level_help_text(TIER_CUSTOM))
            return
        self._model_combo.setCurrentText(model)  # _on_model_changed applies it
        self._level_combo.setToolTip(self._level_help_text(tier_id))

    def _on_provider_changed(self, _index: int) -> None:
        pid = self._provider_combo.currentData()
        if not pid or pid == self._provider_id:
            return
        # Carry the chosen level across the switch: a user who picked level 2 for
        # cost reasons means level 2 on the new provider too, not its default.
        wanted = tier_for_model(self._provider_id, self._provider.model)
        self._provider_id = pid
        self._provider = make_provider(pid)
        carried = tier_model(wanted, pid)
        if carried:
            self._provider.set_model(carried)
        self._refresh_capabilities()
        self._sync_settings_widgets()
        self._render_note(f"Switched provider to <b>{html.escape(PROVIDER_META[pid]['label'])}</b>.")
        self._refresh_ready_state()

    def _on_model_changed(self, text: str) -> None:
        self._provider.set_model(text)
        self._refresh_capabilities()
        self._sync_level_widgets()
        self._refresh_ready_state()

    def _on_apply_settings(self) -> None:
        meta = PROVIDER_META.get(self._provider_id, PROVIDER_META["openai"])
        if meta.get("needs_base_url"):
            self._provider.set_base_url(self._base_url_edit.text())
        key = self._key_edit.text().strip()
        if key and self._provider_id not in CLI_PROVIDER_IDS:
            self._provider.set_api_key(key)
            self._key_edit.clear()
        self._provider.set_model(self._model_combo.currentText())
        self._refresh_capabilities()
        self._sync_level_widgets()
        self._render_note("Applied provider settings for this session.")
        self._refresh_ready_state()

    # -- conversation lifecycle ---------------------------------------------
    def _reset_conversation(self) -> None:
        if hasattr(self._controller, 'reset_workflow_request'):
            self._controller.reset_workflow_request()
        self._messages = []
        self._reference_query = ''
        self._file_goal = ''
        self._conversation_scope = self._controller.attachment_scope()
        self._tool_queue = []
        self._pending_workflow_text = None
        self._current_call = None
        self._executed_in_turn = False
        self._unasked = 0
        self._stop_requested = False
        self._pending_images = []
        self._confirm.setVisible(False)
        self._mode = None
        self._pending_choice = None
        self._pending_handover = None
        self._handover_work = None
        self._choice.setVisible(False)
        self._transcript.clear()
        vision_line = (
            "<br>&#8226; <i>capture the inversion section and tell me if it looks reasonable</i>"
            if self._vision else ""
        )
        agent = assistant_registry.active()
        if agent.examples:
            self._input.setPlaceholderText(f"e.g. {agent.examples[0]}")
        examples = "".join(f"<br>&#8226; <i>{html.escape(example)}</i>"
                           for example in agent.examples)
        self._render_note(
            f"<b>{html.escape(agent.name)}</b> · {html.escape(agent.domain)} is ready — "
            "describe what you want to do in plain language."
            + (f" Examples:{examples}" if examples else "")
            + f"{vision_line}"
            "<br>After you send it, choose to work step by step or to run straight "
            "through to a report."
        )
        self._refresh_ready_state()

    def _refresh_ready_state(self) -> None:
        ok, reason = self._provider.available()
        if self._provider_id in CLI_PROVIDER_IDS:
            self._refresh_cli_controls()
            status = self._cli_states.get(self._provider_id, {})
            ok = ok and status.get("state") == "ready"
            reason = status.get("message", "Setting up automatically…")
        # While the mode choice is open the request is waiting on it, not on input.
        typing = ok and not self._busy and self._pending_choice is None
        self._input.setEnabled(typing)
        self._send_btn.setEnabled(typing)
        label = PROVIDER_META.get(self._provider_id, {}).get("label", self._provider_id)
        if ok:
            eye = " · vision" if self._vision else ""
            price = ("" if self._provider_id in CLI_PROVIDER_IDS
                     else price_label(self._provider.model, compact=True))
            cost = f" · {html.escape(price)}" if price else ""
            tier = MODEL_TIERS.get(tier_for_model(self._provider_id, self._provider.model))
            level = f"{html.escape(tier['name'])} · " if tier else ""
            self._status_label.setText(
                f"{level}{html.escape(label)} · {html.escape(self._provider.model)}"
                f" — ready{cost}{eye}")
        else:
            self._status_label.setText('Local numerical tasks are ready · Configure AI only for interpretation'
                                       if assistant_registry.active().offline_workflow else
                                       f"{html.escape(label)}: {html.escape(reason)}")

    # -- sending / receiving -------------------------------------------------
    def _on_send(self) -> None:
        if self._busy:
            return
        if (self._provider_id in CLI_PROVIDER_IDS
                and self._cli_states.get(self._provider_id, {}).get("state") != "ready"):
            self._refresh_ready_state()
            return
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._looks = 0
        if self._conversation_scope != self._controller.attachment_scope():
            self._reset_conversation()
        # Fast-path: while paused at a checkpoint, "continue" runs the module's
        # declared resume action directly — no LLM round-trip, no approval click.
        if self._mode != "auto" and self._paused_resume and text.lower() in _CONTINUE_WORDS:
            self._input.clear()
            self._render_user(text)
            self._resume_paused(text)
            return
        ok, reason = self._provider.available()
        if not ok:
            self._render_note(reason)
            self._refresh_ready_state()
            return
        self._input.clear()
        self._render_user(text)
        if text.lower() not in _CONTINUE_WORDS:
            self._reference_query = text
        pending = self._controller.pending_chat_files()
        if pending:
            # "continue" when asked about a file is not information about it:
            # it hands the decision back. Appended as the word itself, the
            # classifier would read the same evidence and ask the same question.
            clarification = (
                'the user asked you to decide: give every file its most likely role from '
                'the evidence, and use unknown only if no role is plausible at all'
                if text.lower() in _CONTINUE_WORDS else text)
            self._file_goal = (self._file_goal + '\nUser clarification: ' + clarification
                               if self._file_goal else text)
            from .attachments import ClipboardClassificationWorker
            self._provider.reasoning_effort = self._reasoning.currentText()
            worker = ClipboardClassificationWorker(
                self._file_goal, pending,
                {key: label for label, key in self._controller.chat_attachment_roles()},
                self._provider, self._controller.attachment_scope(), QApplication.instance())
            self._track_file_worker(worker, self._on_files_classified)
            self._set_live('Identifying how to use the pasted files', presence.THINKING)
            worker.start()
            return
        self._retrieve_for_text(text)

    def _track_file_worker(self, worker, callback):
        self._reference_workers.append(worker)
        worker.ready.connect(callback)
        worker.finished.connect(lambda w=worker: self._reference_workers.remove(w))
        worker.finished.connect(worker.deleteLater)
        self._set_busy(True)

    def _on_files_classified(self, result):
        if self._closing:
            return
        self._set_live(None)
        if result['scope'] != self._controller.attachment_scope():
            self._set_busy(False)
            self._reset_conversation()
            self._render_note('The Project or assistant changed. Send the task again for the current files.')
            return
        if result.get('error'):
            self._render_note('Could not identify file purposes: ' + html.escape(result['error']))
            self._set_busy(False)
            return
        applied = self._controller.apply_chat_classification(result['files'])
        if applied.get('status') != 'ok':
            message = applied.get('error', 'Tell me more about these files.')
            for row in applied.get('files', []):
                message += '\n' + Path(row['path']).name + ': ' + str(row.get('reason', 'Purpose unclear'))
            message += (f"\nSay what they are, or reply “continue” to let {self._name()} "
                        "decide from the evidence.")
            self._render_note(html.escape(message).replace('\n', '<br>'))
            self._set_busy(False)
            return
        labels = {key: label for label, key in self._controller.chat_attachment_roles()}
        for row in result['files']:
            self._render_note(html.escape(f"{Path(row['path']).name}: {labels.get(row['role'], row['role'])}. {row.get('reason', '')}"))
        if any(row['role'] == 'chat_reference' for row in result['files']):
            self._rag.setChecked(True)
        self._file_goal = ''
        self._reference_query = result['query']
        self._refresh_attachment_list()
        self._retrieve_for_text(result['query'])

    def _retrieve_for_text(self, text):
        paths = list(dict.fromkeys(row['path'] for row in self._controller.chat_attachments() if row['rag']))
        if self._rag.isChecked() and paths:
            from .attachments import ReferenceRetrievalWorker
            worker = ReferenceRetrievalWorker(text, paths, self._controller.attachment_scope(),
                                             QApplication.instance(), search_query=self._reference_query or text)
            self._track_file_worker(worker, self._on_references_ready)
            self._set_live('Searching attached reference documents', presence.THINKING)
            worker.start()
            return
        self._deliver_text(text)

    def _on_references_ready(self, result):
        if self._closing:
            return
        if result['scope'] != self._controller.attachment_scope():
            self._set_live(None)
            self._set_busy(False)
            self._reset_conversation()
            self._render_note('The Project or assistant changed. Send your request again with the current files.')
            return
        for warning in result['diagnostics']:
            self._render_note('Reference: ' + html.escape(warning))
        sources = result['sources']
        if sources:
            lines = []
            for source in sources:
                location = f"page {source['page']}" if 'page' in source else f"line {source['line']}"
                lines.append(html.escape(f"{source['source']} ({location})"))
            self._render_note('<b>RAG sources</b><br>' + '<br>'.join(lines))
        else:
            self._render_note('No matching reference excerpts found for this request.')
        self._set_live(None)
        self._deliver_text(result['query'], sources)

    def _deliver_text(self, text, sources=()):
        if self._mode is None:
            self._offer_modes(text, sources)
            return
        from PyHydroGeophysX.agents.local_knowledge import ATTACHMENTS_HEADER, format_context
        rows = self._controller.chat_attachments()
        manifest = '\n'.join(f"{row['role']}: {row['path']}" for row in rows)
        context = (ATTACHMENTS_HEADER + '\n' + manifest if manifest else '')
        reference_context = format_context(sources)
        if reference_context:
            context += '\n\n' + reference_context
        self._messages.append({"role": "user", "content": text + ('\n\n' + context if context else '')})
        if self._mode == "auto":
            self._start_auto(text, context, sources)
            return
        self._paused_resume = None  # a non-"continue" message takes manual control
        self._unasked = 0           # the user has just had their say
        self._stop_requested = False
        self._set_busy(True)
        self._stop_btn.setEnabled(True)
        self._stop_btn.setVisible(True)
        self._start_request()

    # -- step by step or Auto to report ----------------------------------------
    def _auto_supported(self) -> bool:
        return self._provider_id in set(assistant_registry.active().providers)

    def _offer_modes(self, text: str, sources=()) -> None:
        """Ask how to carry out the request just sent, and hold it until answered."""
        agent = assistant_registry.active()
        self._pending_choice = (text, tuple(sources))
        auto = self._auto_supported()
        self._auto_btn.setEnabled(auto)
        self._auto_btn.setToolTip(
            "Run the whole task and write the report without stopping." if auto else
            f"Auto to report with {agent.name} runs on {', '.join(agent.providers)}; "
            "choose one of those in Settings.")
        self._guided_btn.setToolTip(
            f"{agent.name} does the routine steps itself and asks you at the decisions. "
            "Ask for the final result at any point and the rest runs automatically.")
        self._choice_label.setText(
            f"<b>How should {html.escape(agent.name)} carry this out?</b><br>"
            "Step by step, it does the routine steps itself, asks you at the decisions, "
            "and you can ask for the final result at any point. Auto to report runs "
            "straight through to the report.")
        self._choice.setVisible(True)
        self._set_busy(False)
        self._scroll_to_end()

    def _choose_mode(self, mode: str) -> None:
        pending, self._pending_choice = self._pending_choice, None
        self._choice.setVisible(False)
        if pending is None:
            return
        self._mode = mode
        self._render_note(f"Step by step: {html.escape(self._name())} runs the routine "
                          "steps and asks you at the decisions. Press Stop to cut in."
                          if mode == "guided" else
                          "Auto to report: running straight through to the report.")
        self._deliver_text(*pending)

    def _start_auto(self, text: str, context: str = "", sources=()) -> None:
        """Run ``text`` with the assistant's workflow, through to its report."""
        agent = assistant_registry.active()
        if not self._auto_supported():
            self._render_note(f"Auto to report with {html.escape(agent.name)} supports "
                              f"{', '.join(agent.providers)}. Choose one of those in "
                              "Settings, or work step by step.")
            # Ask again next time rather than leave every message failing here.
            self._mode = None
            self._set_busy(False)
            return
        settings = {"assistant": agent.key,
                    "provider": "claude" if self._provider_id == "anthropic" else self._provider_id,
                    "model": self._provider.model,
                    "api_key": self._provider._api_key,
                    "base_url": getattr(self._provider, "_base_url", None),
                    "reasoning_effort": self._reasoning.currentText(),
                    "use_rag": self._rag.isChecked() and 'rag' in agent.retrieval,
                    "chat_context": context,
                    "chat_reference_sources": list(sources),
                    "use_mcp": self._mcp.isEnabled() and self._mcp.isChecked()}
        if self._handover_work:
            # Work finished step by step, for the run to carry on from. Kept
            # until the run is over: a run that first waits for data or for
            # file roles to be checked starts again on "continue".
            settings["studio_work"] = self._handover_work
        try:
            message = self._controller.run_to_report(
                text, settings, self._on_workflow_finished,
                on_step=self._on_workflow_step)
        except Exception as exc:
            message = f"Could not start workflow: {exc}"
        self._report_workflow(message)
        if message.startswith("Could not start workflow"):
            self._mode = "guided"
            self._look_into_run("could not start", message)
            return
        if message.startswith("Workflow started"):
            self._auto_running = True
            # A run that pauses before each step leaves the chat free meanwhile.
            if message.startswith("Workflow started."):
                self._set_busy(True)
            self._set_live(f"{self._name()} is reading your request", presence.THINKING)

    def _report_workflow(self, message: str) -> None:
        self._set_live(None)
        # The run's warnings arrive a line each; the page shows only their count.
        self._render_note(html.escape(message).replace("\n", "<br>"))
        self._messages.append({"role": "assistant", "content": message})
        self._set_busy(False)

    def _on_workflow_finished(self, message):
        self._report_workflow(message)
        if not self._auto_running or self._controller.workflow_awaiting_continue():
            return
        # The run is over. What follows is a conversation about it: questions,
        # a change to try, another run - which the assistant starts itself
        # when asked, so the chat goes back to step by step.
        self._auto_running = False
        self._handover_work = None
        outcome = getattr(self._controller, "workflow_outcome", None)
        status = str((outcome() if callable(outcome) else {}).get("status") or "")
        problem = {"failed": "failed", "incomplete": "did not complete"}.get(status)
        if self._mode == "auto":
            self._mode = "guided"
            if problem is None:
                self._render_note(f"Ask {html.escape(self._name())} about the results, or say "
                                  "what to change; it can run the task to a report again.")
        if problem is not None:
            # A run that failed or fell short is the assistant's to look into,
            # not a line left in the chat for the user to puzzle over.
            self._look_into_run(problem, message)

    def _look_into_run(self, how: str, report: str) -> None:
        """Hand a run that failed or fell short to the assistant, step by step.

        It reads the run's record, says what went wrong and fixes what it can;
        running the task again and choices that are the user's are put to them.
        """
        self._look_into(RUN_PROBLEM.format(how=how, report=str(report)[:4000]))

    def _on_job_failed(self, failure: Dict[str, Any]) -> None:
        """A background job the assistant started failed: hand it over once it is free."""
        if self._closing:
            return
        self._job_failures.append(failure)
        QTimer.singleShot(0, self._deliver_job_failures)

    def _on_page_failure(self, failure: Dict[str, Any]) -> None:
        """A page reported an error outside the assistant's calls: hand it over once free."""
        self._on_job_failed({**failure, "source": "page"})

    def _deliver_job_failures(self) -> None:
        # Not while a turn, an approval, the step-by-step / Auto choice or an
        # automatic run is going: the failure waits, and _set_busy(False)
        # brings it back here.
        if (self._closing or not self._job_failures or self._busy
                or self._pending_choice is not None or self._mode == "auto"
                or self._auto_running):
            return
        failures, self._job_failures = self._job_failures, []
        mine = [f for f in failures if f.get("source") != "page"]
        for failure in failures:
            title = html.escape(str(failure.get("title") or failure.get("module") or "module"))
            self._render_note(f"The job on the {title} page failed." if failure in mine else
                              f"The {title} page reported an error: "
                              f"{html.escape(str((failure.get('errors') or [''])[-1])[:300])}")
        failure = failures[-1]
        errors = "\n".join(f"- {line}" for item in failures
                           for line in item.get("errors") or [])
        template = JOB_PROBLEM if len(mine) == len(failures) else PAGE_PROBLEM
        self._look_into(template.format(
            title=failure.get("title") or failure.get("module") or "module",
            module=failure.get("module") or "", errors=errors[:4000] or "- (no message)"),
            unavailable=f"Ask {html.escape(self._name())} about it, or say what to change.")

    def _look_into(self, prompt: str, unavailable: str = "") -> None:
        """Have the assistant look into a problem the studio reports, without the user asking.

        After :data:`MAX_LOOKS` such problems without a word from the user, or
        with no model to ask, the chat waits for them instead and says
        ``unavailable``.
        """
        name = html.escape(self._name())
        ready = (self._provider_id not in CLI_PROVIDER_IDS
                 or self._cli_states.get(self._provider_id, {}).get("state") == "ready")
        if self._looks >= MAX_LOOKS or not ready or not self._provider.available()[0]:
            self._render_note(unavailable or f"Ask {name} what went wrong, or say what to change.")
            return
        if self._mode is None:
            # A page's error before any conversation: it opens one, step by step.
            self._mode = "guided"
        self._looks += 1
        self._render_note(f"{name} is looking into what went wrong.")
        self._messages.append({"role": "user", "content": prompt})
        self._paused_resume = None
        self._unasked = 0
        self._stop_requested = False
        self._set_busy(True)
        self._stop_btn.setEnabled(True)
        self._stop_btn.setVisible(True)
        self._start_request()

    def _on_workflow_step(self, event: Dict[str, Any]) -> None:
        """Narrate an automatic run here, a line per step as each one ends.

        Without this the chat would say "Workflow started." and then nothing
        until the report, so the conversation that started the run would go
        quiet for the whole run. Each finished step reports back with what it
        found, and the line under the transcript says what the assistant is
        doing meanwhile.
        """
        label = str(event.get("label") or event.get("tool") or "Step")
        if event.get("phase") == "start":
            self._set_live(f"{self._name()} · {label}", presence.WORKING)
            return
        if event.get("phase") == "paused":
            self._set_live(f"{self._name()} is paused · Resume under the Live timeline",
                           presence.WAITING)
            return
        if event.get("phase") == "resumed":
            self._set_live(f"{self._name()} is carrying on", presence.THINKING)
            return
        if event.get("phase") == "waiting":
            what = f"your approval of {label}" if label else "your answer"
            self._set_live(f"{self._name()} is waiting for {what} in Workflow", presence.WAITING)
            return
        if event.get("phase") != "done":
            return
        status = str(event.get("status") or "ok")
        # Fixed system colours that read on both appearances: these lines stay
        # in the transcript when the appearance changes.
        mark, color = (("&#10003;", "#34c759") if status == "ok" else
                       ("&#8631;", presence.CARET_GRAY) if status == "skipped" else
                       ("&#10007;", "#ff3b30"))
        summary = str(event.get("summary") or "")
        if len(summary) > 280:
            summary = summary[:277].rstrip() + "…"
        self._transcript.append(
            f"<div style='margin:4px 0;'>"
            f"<span style='color:#af52de;'>&#9679;</span> "
            f"<b>{html.escape(label)}</b> "
            f"<span style='color:{color}; font-weight:700;'>"
            f"{mark}</span>"
            + (f"<br><span style='color:{presence.CARET_GRAY};'>{html.escape(summary)}</span>"
               if summary else "")
            + "</div>")
        self._scroll_to_end()
        self._set_live(f"{self._name()} is choosing the next step", presence.THINKING)

    def _set_live(self, text: Optional[str], state: str = presence.WORKING) -> None:
        """Show what an automatic run is doing under the transcript, or clear it."""
        self._live = text
        if text is None:
            if not self._thinking_clock.isActive():
                self._thinking_orb.set_state(presence.IDLE)
                self._thinking_row.setVisible(False)
            return
        self._thinking.setText(text + "…")
        self._thinking_orb.set_state(state)
        self._thinking_row.setVisible(True)

    def _resume_paused(self, text: str) -> None:
        """Run the paused checkpoint's resume action directly (no LLM call), keeping the
        message log valid (user -> assistant tool_call -> tool result) for later calls."""
        spec = self._paused_resume or {}
        action = spec.get("action")
        if not action:
            self._paused_resume = None
            return
        args = {"action": action, "args": spec.get("args", {}) or {}}
        self._resume_seq += 1
        cid = f"resume_{self._resume_seq}"
        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": cid, "name": "apply_action", "arguments": args}]})
        result = self._controller.dispatch("apply_action", args)
        self._answer_tool({"id": cid, "name": "apply_action"}, result)
        self._flush_pending_images()
        self._render_tool_result("apply_action", result)
        self._paused_resume = (result.get("resume")
                               if isinstance(result, dict) and result.get("status") == "awaiting_user"
                               else None)
        self._refresh_ready_state()

    def _start_request(self) -> None:
        self._provider.reasoning_effort = self._reasoning.currentText()
        self._send_btn.setText("…")
        worker = LlmCallWorker(self._provider, self._system, self._messages, self._tool_specs)
        worker.succeeded.connect(self._on_llm_ok)
        worker.failed.connect(self._on_llm_failed)
        worker.finished.connect(lambda: self._send_btn.setText("Send"))
        # Only the request in flight may take the indicator down: a finished
        # signal delivered late must not hide the next request's.
        worker.finished.connect(
            lambda w=worker: self._set_thinking(False) if self._worker is w else None)
        self._worker = worker
        self._set_thinking(True)
        worker.start()

    def _set_thinking(self, on: bool) -> None:
        """Show that a model request is in flight, and for how long; or hide it."""
        if on:
            self._thinking_started = time.monotonic()
            self._update_thinking()
            self._thinking_clock.start()
            self._thinking_orb.set_state(presence.THINKING)
            self._thinking_row.setVisible(True)
            return
        self._thinking_clock.stop()
        if self._live is not None:
            self._set_live(self._live)
            return
        self._thinking_orb.set_state(presence.IDLE)
        self._thinking_row.setVisible(False)

    def _update_thinking(self) -> None:
        seconds = int(time.monotonic() - self._thinking_started)
        self._thinking.setText(f"{self._name()} is thinking…" + (f" {seconds} s" if seconds else ""))

    def _prewarm_provider(self) -> None:
        """Do the provider's first-request setup while the user is still typing.

        Importing the SDK and building its client add to the first request of a
        session, and none of it depends on the message. It runs on a background
        thread from the first keystroke - once per provider, and once more when
        a key makes a provider that could not send able to.
        """
        provider = self._provider
        last = self._prewarmed
        if last is not None and last[0] is provider and last[1]:
            return
        try:
            ready = bool(provider.available()[0])
        except Exception:  # noqa: BLE001 - a provider that cannot say is not ready
            ready = False
        if last is not None and last[0] is provider and last[1] == ready:
            return
        self._prewarmed = (provider, ready)
        prewarm(provider)

    def _on_llm_failed(self, error: str) -> None:
        self._set_thinking(False)
        if (not self._closing and not self._stop_requested
                and self._reply_retries < MAX_REPLY_RETRIES and not _NO_RETRY.search(str(error))):
            # A reply with a tool call that could not be read (an unknown tool,
            # arguments that are not JSON), or a call that timed out, is asked
            # for once more with the reason, rather than ending the turn with
            # the tool results it had not yet answered.
            self._reply_retries += 1
            self._render_note(f"The model's reply could not be used ({html.escape(error)}); "
                              "asking again.")
            if not self._messages or self._messages[-1].get("role") != "user":
                self._messages.append({"role": "user", "content": (
                    f"[Studio] Your last reply could not be used: {error} Answer again. "
                    "Call only the tools you were given, each with a JSON object of its "
                    "parameters.")})
            self._start_request()
            return
        self._render_note(f"Model call failed: {html.escape(error)}")
        self._set_busy(False)

    def _on_llm_ok(self, out: Dict[str, Any]) -> None:
        self._set_thinking(False)
        self._reply_retries = 0
        # Record the assistant turn (neutral form) so tool ids line up next call.
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": out.get("content"),
            "tool_calls": out.get("tool_calls") or [],
        }
        if "_anthropic_content" in out:
            assistant["_anthropic_content"] = out["_anthropic_content"]
        if "_openai_output" in out:
            assistant["_openai_output"] = out["_openai_output"]
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
        if self._stop_requested and self._tool_queue:
            for skipped in self._tool_queue:
                self._answer_tool(skipped, STOPPED)
            self._tool_queue = []
        if (self._awaiting_user or self._pending_handover is not None) and self._tool_queue:
            # A checkpoint paused the workflow, or Auto to report took over; do
            # not run the rest of this batch, but answer every queued call so
            # tool_call / tool_result stay paired.
            why = ("Handed over to Auto to report." if self._pending_handover is not None
                   else "Paused for manual pick review.")
            for skipped in self._tool_queue:
                self._answer_tool(skipped, {"status": "skipped", "message": why})
            self._tool_queue = []
        if not self._tool_queue:
            # Every tool entry for this batch is now recorded, so a user message
            # carrying the captures can be appended without splitting the pairs.
            self._flush_pending_images()
            if self._stop_requested:
                # Nothing further this turn: no hand-over, no model call.
                self._pending_handover = self._handover_work = None
                self._pending_workflow_text = None
                self._awaiting_user = False
                self._render_note(f"Stopped. {html.escape(self._name())} waits for your "
                                  "next message.")
                self._set_busy(False)
            elif self._pending_handover is not None:
                goal, self._pending_handover = self._pending_handover, None
                self._set_busy(False)
                self._hand_over(goal)
            elif self._pending_workflow_text:
                text = self._pending_workflow_text
                self._pending_workflow_text = None
                self._set_busy(False)
                self.start_workflow(text)
            elif self._awaiting_user:
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
        args = {} if args is None or args == "" else args
        name = call.get("name", "?")
        if not isinstance(args, dict):
            # Arguments the provider could not read as JSON. Run with none, the
            # call would do something other than what was meant; back to the model.
            unread = {"status": "failed", "error": (
                f"The arguments of this {name} call could not be read "
                f"({str(args)[:200]!r}). Call it again with a JSON object of its parameters.")}
            self._answer_tool(call, unread)
            self._render_tool_result(name, unread)
            self._executed_in_turn = True
            self._current_call = None
            self._process_next_tool()
            return
        streak = self._unasked >= MAX_UNASKED
        if streak or needs_approval(name, args):
            self._show_confirmation(name, args, streak=streak)
            return
        # Routine: run it now, in the open. The event loop gets a turn first,
        # so the chat repaints and a Stop pressed meanwhile is seen.
        self._unasked += 1
        self._render_note(f"&#9656; {self._describe_call_short(name, args)}")
        QTimer.singleShot(0, lambda c=call: self._run_call(c))

    def _show_confirmation(self, name: str, args: Dict[str, Any], streak: bool = False) -> None:
        intro = (f"{html.escape(self._name())} has taken {self._unasked} steps on its own; "
                 "approve to let it carry on.<br>" if streak else "")
        self._confirm_label.setText(intro + self._describe_call(name, args))
        script = name == SCRIPT_TOOL
        self._confirm_code.setPlainText(str(args.get("code") or "") if script else "")
        self._confirm_code.setVisible(script)
        self._approve_btn.setText("Run this script" if script else "Approve & run")
        self._confirm.setVisible(True)
        self._approve_btn.setEnabled(True)
        self._reject_btn.setEnabled(True)

    def _on_approve(self) -> None:
        call = self._current_call
        if call is None:
            return
        self._confirm.setVisible(False)
        self._unasked = 0
        self._run_call(call)

    def _run_call(self, call: Dict[str, Any]) -> None:
        """Run ``call``, approved or routine, answer it, and go on to the next."""
        if self._closing or call is not self._current_call:
            return  # the conversation was reset while it waited
        if self._stop_requested:
            self._answer_tool(call, STOPPED)
            self._current_call = None
            self._process_next_tool()
            return
        name = call.get("name", "?")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        args = {k: v for k, v in args.items() if k != "ask_user"}
        try:
            if name == SCRIPT_TOOL:
                self._start_script(call, args)  # answers the call when it has run
                return
            result = (self._accept_handover(args) if name == HANDOVER_TOOL
                      else run_skill_tool(getattr(self, "_skills", []), args)
                      if name == SKILL_TOOL
                      else self._controller.dispatch(name, args))
        except Exception as exc:  # noqa: BLE001 - the model reads it and adjusts
            # A failure in the panel's own part of a call would leave the chat
            # busy with nothing answered; the model gets the error instead.
            result = {"status": "failed",
                      "error": f"The studio could not carry out this call ({type(exc).__name__}: {exc})."}
        if not isinstance(result, dict):
            result = {"status": "ok", "result": result}
        if result.get("status") == "awaiting_user":
            self._awaiting_user = True
            self._paused_resume = result.get("resume")
        if result.get("status") == "started":
            # The turn ends while the job runs; if it fails, the assistant hears.
            watch = getattr(self._controller, "watch_job", None)
            if callable(watch):
                try:
                    watch(self._on_job_failed)
                except Exception:  # noqa: BLE001 - following the job is a courtesy
                    pass
        self._answer_tool(call, result)
        self._render_tool_result(name, result)
        self._executed_in_turn = True
        self._current_call = None
        self._process_next_tool()

    def _start_script(self, call: Dict[str, Any], args: Dict[str, Any]) -> None:
        """Run an approved script off the UI thread; its result answers the call."""
        from PyHydroGeophysX.agents.runtime.scratch import check_script, script_folder

        code = str(args.get("code") or "")
        raw = args.get("inputs") if isinstance(args.get("inputs"), dict) else {}
        inputs = {str(k): str(v) for k, v in raw.items() if v}
        problem = check_script(code) or next(
            (f"Input not found: {path}." for path in inputs.values()
             if not Path(path).exists()), "")
        if problem:
            # Back to the model to fix: nothing ran.
            self._finish_script(call, {"status": "failed", "error": problem})
            return
        folder = script_folder(self._controller.scratch_root(), "chat")
        worker = ScriptWorker(code, inputs, folder, str(args.get("purpose") or ""),
                              parent=QApplication.instance())
        self._script_worker = worker
        worker.done.connect(lambda record, c=call: self._finish_script(c, record))
        worker.finished.connect(worker.deleteLater)
        self._set_live("Running the script you approved", presence.WORKING)
        worker.start()

    def _finish_script(self, call: Dict[str, Any], record: Dict[str, Any]) -> None:
        if self._closing:
            return
        self._set_live(None)
        folder = Path(str(record.get("folder") or ""))
        result = {"status": record.get("status", "failed"),
                  "result": record.get("result"),
                  # Full paths, ready for the module actions that take them.
                  "files": [str(folder / f) for f in record.get("files") or []],
                  "folder": str(folder) if record.get("folder") else "",
                  "printed": str(record.get("printed") or "")[-1500:],
                  "error": str(record.get("error") or "")[-1500:],
                  "seconds": record.get("seconds")}
        self._answer_tool(call, result)
        self._render_tool_result(SCRIPT_TOOL, result)
        self._executed_in_turn = True
        self._current_call = None
        self._process_next_tool()

    def _accept_handover(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Take the step-by-step task over to Auto to report, once the batch is answered.

        The inputs the assistant names join the Workflow page's; what the
        studio has finished is collected now, while the modules still hold it.
        """
        goal = str(args.get("goal") or "").strip()
        if not goal:
            return {"status": "failed", "error": "run_to_report needs the goal to run."}
        if not self._auto_supported():
            agent = assistant_registry.active()
            return {"status": "failed", "error": (
                f"Auto to report with {agent.name} runs on {', '.join(agent.providers)}, "
                "not the provider selected now. Tell the user to choose one of those in "
                "Settings, and carry on step by step meanwhile.")}
        inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else {}
        for role, value in inputs.items():
            paths = [str(p) for p in (value if isinstance(value, list) else [value]) if p]
            if not paths:
                continue
            known = {row["path"] for row in self._controller.chat_attachments()
                     if row["role"] == role}
            paths = [p for p in paths if str(Path(p).resolve()) not in known]
            if not paths:
                continue
            added = self._controller.add_chat_attachments(role, paths)
            if added.get("status") != "ok":
                return {"status": "failed", "error": f"{role}: {added.get('error')}"}
        self._handover_work = self._controller.studio_work() or None
        ert = (self._handover_work or {}).get("ert") or {}
        used = str(Path(ert["electrode_file"]).resolve()) if ert.get("electrode_file") else ""
        if used:
            # The ERT page read this file as electrode positions, and that use
            # outranks a classifier's label: a run handed an electrode file
            # labelled "terrain" would refuse it, after the page had used it.
            for row in self._controller.chat_attachments():
                if (row["role"] not in ("electrode_file", "pending", "chat_reference")
                        and str(Path(row["path"]).resolve()) == used):
                    self._controller.remove_chat_attachment(row["role"], row["path"])
            if not any(row["role"] == "electrode_file"
                       for row in self._controller.chat_attachments()):
                added = self._controller.add_chat_attachments("electrode_file", [used])
                if added.get("status") != "ok":
                    return {"status": "failed", "error": f"electrode_file: {added.get('error')}"}
        roles = {row["role"] for row in self._controller.chat_attachments()}
        if ert.get("source_files") and not roles & {"data_file", "time_lapse_files"}:
            # The surveys the studio inverted, when the assistant named none:
            # the run reloads them to evaluate the model it takes over.
            role = "time_lapse_files" if ert.get("kind") == "timelapse" else "data_file"
            fill = {role: ert["source_files"] if role == "time_lapse_files"
                    else ert["source_files"][:1]}
            if ert.get("electrode_file") and "electrode_file" not in roles:
                fill["electrode_file"] = [ert["electrode_file"]]
            for role, paths in fill.items():
                added = self._controller.add_chat_attachments(role, paths)
                if added.get("status") != "ok":
                    return {"status": "failed", "error": f"{role}: {added.get('error')}"}
        self._pending_handover = goal
        carried = sorted((self._handover_work or {}).keys())
        return {"status": "ok", "message": "Auto to report is taking over.",
                "carried_over": carried}

    def _hand_over(self, goal: str) -> None:
        """Run the handed-over goal with Auto to report, from where the chat is."""
        self._mode = "auto"
        self._paused_resume = None
        carried = ", ".join(sorted(self._handover_work or {}))
        self._render_note(
            "<b>Auto to report</b> takes it from here"
            + (f", carrying over the finished {html.escape(carried)} work" if carried else "")
            + f".<br><i>{html.escape(goal)}</i>")
        self._reference_query = goal
        self._retrieve_for_text(goal)

    def _on_reject(self) -> None:
        call = self._current_call
        if call is None:
            return
        self._confirm.setVisible(False)
        self._unasked = 0
        result = {"status": "declined", "message": "The user declined this action."}
        self._answer_tool(call, result)
        self._render_note("You rejected the step.")
        self._executed_in_turn = True
        self._current_call = None
        self._process_next_tool()

    def _on_stop(self) -> None:
        """Stop the step-by-step assistant after the action in hand.

        A model reply still on its way is recorded but not acted on; a call
        waiting on the approval card is answered as stopped. Either way the
        turn ends without another model call.
        """
        if self._stop_requested:
            return
        self._stop_requested = True
        self._stop_btn.setEnabled(False)
        if self._current_call is not None and not self._confirm.isHidden():
            self._confirm.setVisible(False)
            call, self._current_call = self._current_call, None
            self._answer_tool(call, STOPPED)
            self._process_next_tool()
        elif self._busy:
            self._render_note("Stopping after the current action…")

    def _answer_tool(self, call: Dict[str, Any], result: Dict[str, Any]) -> None:
        self._messages.append({
            "role": "tool",
            "id": call.get("id"),
            "name": call.get("name"),
            "content": self._compact_for_context(result),
        })
        image = result.get("_image") if isinstance(result, dict) else None
        if isinstance(image, dict) and image.get("data"):
            self._pending_images.append((str(result.get("view") or "view"), image))

    def _flush_pending_images(self) -> None:
        """Send this turn's captures as one follow-up user message."""
        if not self._pending_images:
            return
        blocks: List[Dict[str, Any]] = []
        for name, image in self._pending_images:
            blocks.append(text_block(f"Captured view '{name}':"))
            blocks.append(image_block(image["data"],
                                      image.get("media_type", "image/png")))
        self._messages.append({"role": "user", "content": blocks})
        self._pending_images = []

    @staticmethod
    def _compact_for_context(result: Any) -> Any:
        """Trim display-only / redundant fields before a result enters the model
        context, so a long multi-step session does not bloat each call."""
        if not isinstance(result, dict):
            return result
        drop = {"message", "records_total", "records_picked", "resume", "suspect_traces",
                # The capture travels as an image block, never as base64 text.
                "_image"}
        return {k: v for k, v in result.items() if k not in drop}

    # -- state / rendering ---------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if self._provider_id in CLI_PROVIDER_IDS:
            self._refresh_cli_controls()
        # One assistant per conversation turn: no switching mid-run.
        self._assistant_combo.setEnabled(not busy)
        self._new_btn.setEnabled(not busy)
        self._remove_attachment_btn.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._send_btn.setEnabled(not busy)
        if not busy:
            self._stop_btn.setVisible(False)
            self._refresh_ready_state()
            self._input.setFocus()
            if getattr(self, "_job_failures", None):
                QTimer.singleShot(0, self._deliver_job_failures)

    def _describe_call(self, name: str, args: Dict[str, Any]) -> str:
        if name == SCRIPT_TOOL:
            inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else {}
            reads = ", ".join(Path(str(p)).name for p in inputs.values() if p)
            return (
                f"<b>{html.escape(self._name())} proposes a temporary script</b><br>"
                f"{html.escape(str(args.get('purpose') or ''))}<br>"
                f"<span style='color:{_P['muted']}'>It reads {html.escape(reads or 'nothing')} "
                "and writes only to a new folder of its own, in a separate process with a "
                "time limit. Your files are not changed. Read it before you decide.</span>")
        if name == HANDOVER_TOOL:
            inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else {}
            files = ", ".join(Path(str(p)).name for value in inputs.values()
                              for p in (value if isinstance(value, list) else [value]))
            return (
                f"<b>{html.escape(self._name())} will finish this with Auto to report</b>, "
                "from where you are now to the report, without stopping for approval.<br>"
                f"<span style='color:{_P['muted']}'>{html.escape(str(args.get('goal') or ''))}"
                + (f"<br>Adds: {html.escape(files)}" if files else "") + "</span>")
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items()
                            if k != "ask_user") or "(no arguments)"
        return (
            f"<b>{html.escape(self._name())} wants to run:</b> <code>{html.escape(name)}</code><br>"
            f"<span style='color:{_P['muted']}'>{html.escape(arg_str)}</span>"
        )

    @staticmethod
    def _describe_call_short(name: str, args: Dict[str, Any]) -> str:
        """One line for a call run without asking: what ran, with what."""
        shown = {k: v for k, v in args.items() if k != "ask_user"}
        if name == "apply_action":
            name = str(shown.get("action") or "?")
            shown = shown.get("args") if isinstance(shown.get("args"), dict) else {}
        detail = ", ".join(f"{k}={v!r}" for k, v in shown.items())
        if len(detail) > 160:
            detail = detail[:157] + "…"
        return (f"<code>{html.escape(name)}</code>"
                + (f" {html.escape(detail)}" if detail else ""))

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
        self._bubble(self._name(), text, _P["accent"])

    def _render_note(self, html_text: str) -> None:
        self._transcript.append(
            f"<div style='margin:6px 0; color:{_P['muted']};'>{html_text}</div>"
        )
        self._scroll_to_end()

    def _render_tool_result(self, name: str, result: Dict[str, Any]) -> None:
        import json

        status = str(result.get("status", "ok"))
        color = _P["green"] if status in ("ok", "started", "success", "awaiting_user") else _P["red"]
        compact = {k: v for k, v in result.items() if k not in ("describe", "_image")}
        described = result.get("describe")
        if isinstance(described, dict):
            # Too long to show; say what the assistant was given instead of "ok".
            compact["module"] = described.get("title") or described.get("module")
            compact["actions"] = len(described.get("actions") or [])
        text = json.dumps(compact, default=str)[:600]
        self._transcript.append(
            f"<div style='margin:4px 0;'>"
            f"<span style='color:{color}; font-weight:700;'>"
            f"{html.escape(name)} &#8594; {html.escape(status)}</span><br>"
            f"<code style='color:{_P['muted']};'>{html.escape(text)}</code></div>"
        )
        image = result.get("_image")
        if isinstance(image, dict) and image.get("data"):
            self._render_image(image["data"])
        # Next-step guidance so the user is never left wondering what to do.
        if status == "started":
            self._render_note(
                "This is running in the background. Ask <i>“is it done?”</i> for the "
                "status, or say <i>“continue”</i> / tell the assistant the next step.")
        elif status == "awaiting_user":
            note = html.escape(str(result.get("message")
                                   or "Paused. Correct the picks, then say “continue”."))
            suspect = result.get("suspect_traces")
            if isinstance(suspect, list) and suspect:
                note += f"<br>Suspect traces to check: <code>{html.escape(str(suspect))}</code>"
            self._render_note(f"<b>Paused for you.</b> {note}")
        else:
            hint = result.get("hint") or (result.get("error") if status == "failed" else None)
            if hint:
                self._render_note(f"Hint: {html.escape(str(hint))}")
        self._scroll_to_end()

    def _render_image(self, data_b64: str) -> None:
        """Show a capture in the transcript, so the user sees what the model saw."""
        try:
            raw = base64.b64decode(data_b64)
        except Exception:  # noqa: BLE001
            return
        image = QImage()
        if not image.loadFromData(raw, "PNG"):
            return
        self._image_seq += 1
        key = f"aquah-capture-{self._image_seq}"
        self._transcript.document().addResource(
            QTextDocument.ImageResource, QUrl(key), image)
        width = min(image.width(), 360)
        self._transcript.append(
            f"<div style='margin:4px 0;'><img src='{key}' width='{width}'></div>")
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        sb = self._transcript.verticalScrollBar()
        sb.setValue(sb.maximum())


#: Alias for callers that still use the AQUAH-specific name.
AquahChatPanel = AssistantChatPanel
