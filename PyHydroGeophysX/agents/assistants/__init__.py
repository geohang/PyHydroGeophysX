"""Domain AI assistants: AQUAH for hydrogeophysics, GeoSAGE for geological modelling, and more.

An *assistant* is one complete AI agent for a domain - what the user talks to
in the studio's chat and what "Auto to report" runs. PyHydroGeophysX used to
have exactly one, AQUAH, written into the desktop app, the workflow runner and
the chat's system prompt. This package makes an assistant a plug-in: each is a
small package that describes itself with an :class:`Assistant`, and the studio
builds its chat, its Workflow page and its run from that description.

What an assistant supplies
--------------------------
- **A workflow** (``workflow="pkg.module:function"``): what "Auto to report"
  runs, in the run's own process. It takes the run's payload and returns its
  result; see :data:`WORKFLOW_CONTRACT`. Most assistants build it on
  :func:`PyHydroGeophysX.agents.runtime.entry.drive`, the controller loop over
  a tool registry, which gives the studio's live timeline, step approval and
  chat narration without any further work.
- **Tools** (``tools="pkg.module:attr"``): the assistant's own registry of
  :class:`~PyHydroGeophysX.agents.runtime.tools.Tool`, each with what it
  requires, what it produces and the studio module that shows its work.
- **What it says and accepts**: a chat persona, example requests, the kinds of
  input file the Workflow page offers, and the colours of its glow.

Adding one
----------
In this repository: add a subpackage beside :mod:`.aquah` and :mod:`.geosage`
exposing ``ASSISTANT`` and list it in :data:`BUILTIN`. From a separate package:
declare an entry point, and nothing here needs editing::

    [project.entry-points."pyhydrogeophysx.assistants"]
    geosage = "geosage.pyhydrogeophysx:ASSISTANT"

The developer guide (``docs/source/agents/adding_an_assistant.rst``) walks
through both, with GeoSAGE as the example.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

#: The assistants that ship with PyHydroGeophysX, as ``module:attribute``.
BUILTIN: Tuple[str, ...] = (
    "PyHydroGeophysX.agents.assistants.aquah:ASSISTANT",
    "PyHydroGeophysX.agents.assistants.geosage:ASSISTANT",
)
#: Where installed packages register further assistants.
ENTRY_POINT_GROUP = "pyhydrogeophysx.assistants"
#: The assistant the studio starts with.
DEFAULT_KEY = "aquah"

#: What a workflow function is given and must return. Written out here because
#: it is the one interface an assistant has to honour exactly.
WORKFLOW_CONTRACT = """\
run(payload, progress, *, approve=None, on_event=None, events=None, **hooks) -> dict

payload   the run as the studio starts it:
            request        the user's goal, in their words
            inputs         {role: path or [paths]}, roles from Assistant.input_roles
            provider, model, api_key, reasoning_effort   model access
            output_dir     where everything the run writes goes
            step_mode      True to pause before each step (pass approve on to drive)
            use_rag, use_mcp
            mode           'classify' with data_folder, when the assistant offers
                           folder classification
progress  progress(step, fraction, details, module='') for the status line
approve   approve(event) -> 'proceed' | 'skip' | 'stop', or a question's option
          id; blocks until the user answers. Pass to drive() as on_step via
          runtime.step_by_step when payload['step_mode'] is set, and as ask_user.
on_event  structured step events; pass straight to drive(on_event=...)
events    list the progress events are being collected in (for an audit)

returns   {'status': 'success' | 'needs_review' | 'incomplete',
           'interpretation': str,         first lines shown in the chat
           'warnings': [str],
           'report_files': {'report_markdown': path, ...},
           'output_dir': str, 'execution_plan': [...], 'workflow_config': {...}}
          or, for mode='classify', {'status': 'classified', 'catalog': {...}}
raises    for a run that cannot start or produced nothing; the message is shown
"""


def _load(ref: str) -> Any:
    """The object a ``module:attribute`` reference names."""
    module, _, attribute = str(ref).partition(":")
    value = importlib.import_module(module)
    for part in attribute.split(".") if attribute else ():
        value = getattr(value, part)
    return value


@dataclass(frozen=True)
class Assistant:
    """One domain AI assistant, described for the studio.

    Parameters
    ----------
    key : str
        Identifier, lower case: ``'aquah'``, ``'geosage'``.
    name : str
        What the user sees: ``'AQUAH'``.
    domain : str
        Its field in a few words: ``'Hydrogeophysics'``.
    title : str
        The name spelt out.
    summary : str
        One or two sentences: what it does and what it needs.
    workflow : str
        ``'module:function'`` of the run, per :data:`WORKFLOW_CONTRACT`.
    tools : str
        ``'module:attribute'`` of its tool registry (a dict of name to Tool, or
        a function returning one). Optional; the studio uses it to put a name
        to the step running.
    persona : str
        The part of the chat's system prompt that is this assistant's: who it
        is, and rules for the studio modules it works with.
    examples : tuple of str
        Requests shown when a chat starts.
    input_roles : tuple of (label, key)
        The kinds of input the Workflow page offers to add by hand. A key
        ending in ``_dir`` takes a folder.
    ordered_roles : tuple of str
        Roles that take several files, kept in the order given (time-lapse
        surveys, say).
    folder_classifier : bool
        Whether the workflow accepts ``mode='classify'`` to sort a chosen
        folder's files into roles with the model.
    studio_modules : tuple of str
        The studio modules its chat may drive (keys of
        ``qt_apps.modules.MODULE_SPECS``); empty for all of them.
    colors : tuple of str
        Four colours for its glow and orb; empty for the studio's own.
    requires_packages : tuple of str
        Importable packages it needs, checked by :meth:`availability`.
    status : str
        ``'ready'``, or ``'in development'`` while it cannot run yet.
    status_note : str
        Shown when it is not available.
    providers : tuple of str
        Model providers its workflow supports.
    offline_workflow : bool
        Whether the workflow can produce useful results without model access.
        Enables an explicit "Run without AI" action in the Workflow page.
    """

    key: str
    name: str
    domain: str
    title: str
    summary: str
    workflow: str
    tools: str = ""
    persona: str = ""
    examples: Tuple[str, ...] = ()
    input_roles: Tuple[Tuple[str, str], ...] = ()
    ordered_roles: Tuple[str, ...] = ()
    folder_classifier: bool = False
    studio_modules: Tuple[str, ...] = ()
    colors: Tuple[str, ...] = ()
    requires_packages: Tuple[str, ...] = ()
    status: str = "ready"
    status_note: str = ""
    providers: Tuple[str, ...] = ("openai", "anthropic")
    offline_workflow: bool = False
    # Optional desktop extensions; headless discovery never imports their UI.
    workflow_setup: str = ""
    retrieval: Tuple[str, ...] = ("rag", "mcp")
    focused_workspace: bool = False
    _cache: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def availability(self) -> Tuple[bool, str]:
        """Whether it can run here, and why not when it cannot."""
        if self.status != "ready":
            return False, self.status_note or f"{self.name} is {self.status}."
        missing = [name for name in self.requires_packages
                   if importlib.util.find_spec(name) is None]
        if missing:
            return False, (f"{self.name} needs {', '.join(missing)}; install "
                           f"{'it' if len(missing) == 1 else 'them'} to use it.")
        return True, ""

    def load_workflow(self) -> Callable[..., Dict[str, Any]]:
        """The workflow function, imported on first use."""
        if "workflow" not in self._cache:
            self._cache["workflow"] = _load(self.workflow)
        return self._cache["workflow"]

    def load_tools(self) -> Dict[str, Any]:
        """Its tool registry, name to Tool; empty when it declares none."""
        if not self.tools:
            return {}
        if "tools" not in self._cache:
            # The registry itself, not a copy, so tools registered later are seen.
            value = _load(self.tools)
            self._cache["tools"] = value() if callable(value) else value
        return self._cache["tools"]

    def tool_for_label(self, label: str) -> str:
        """The name of the tool whose label (or name) is ``label``, or ""."""
        try:
            tools = self.load_tools()
        except Exception:  # noqa: BLE001 - a display lookup must not fail a run
            return ""
        for name, tool in tools.items():
            if (getattr(tool, "label", "") or name) == label:
                return name
        return ""


_REGISTRY: Dict[str, Assistant] = {}
_state = {"loaded": False, "active": DEFAULT_KEY}
_errors: List[str] = []


def register_assistant(assistant: Assistant) -> Assistant:
    """Add ``assistant`` to the registry, replacing one with the same key."""
    if not isinstance(assistant, Assistant):
        raise TypeError(f"Expected an Assistant, got {type(assistant).__name__}.")
    _REGISTRY[assistant.key] = assistant
    return assistant


def _discover() -> None:
    """Load the built-in assistants and any registered by installed packages.

    A plug-in that fails to load is recorded in :func:`load_errors` and left
    out; it must not take the built-in assistants down with it.
    """
    if _state["loaded"]:
        return
    _state["loaded"] = True
    refs = [(ref, ref) for ref in BUILTIN]
    try:
        from importlib.metadata import entry_points

        found = entry_points()
        group = (found.select(group=ENTRY_POINT_GROUP) if hasattr(found, "select")
                 else found.get(ENTRY_POINT_GROUP, ()))
        refs += [(entry.value, entry.name) for entry in group]
    except Exception as exc:  # noqa: BLE001 - discovery is best effort
        _errors.append(f"entry points: {exc}")
    for ref, label in refs:
        try:
            value = _load(ref)
            register_assistant(value() if callable(value) else value)
        except Exception as exc:  # noqa: BLE001 - one bad plug-in, not all
            _errors.append(f"{label}: {type(exc).__name__}: {exc}")


def assistants() -> List[Assistant]:
    """Every registered assistant, built-ins first."""
    _discover()
    return list(_REGISTRY.values())


def get_assistant(key: Optional[str] = None) -> Assistant:
    """The assistant ``key`` (the active one when omitted).

    Raises
    ------
    KeyError
        For a key nobody registered, naming the ones that were.
    """
    _discover()
    key = (key or _state["active"] or DEFAULT_KEY).lower()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"No assistant '{key}'. Registered: "
                       f"{', '.join(_REGISTRY) or 'none'}.") from None


def active() -> Assistant:
    """The assistant the user has chosen in this process."""
    return get_assistant(_state["active"])


def set_active(key: str) -> Assistant:
    """Choose the assistant this process works with; returns it."""
    assistant = get_assistant(key)
    _state["active"] = assistant.key
    return assistant


def load_errors() -> List[str]:
    """Plug-ins that failed to load, with the reason."""
    _discover()
    return list(_errors)


def accepted_hooks(fn: Callable[..., Any], **hooks: Any) -> Dict[str, Any]:
    """The ``hooks`` that ``fn`` takes, by name or through ``**kwargs``.

    A run function that predates them - an older caller, or a stub in a test -
    still runs; it simply cannot pause or ask. One whose signature cannot be
    read is given them all.

    Examples
    --------
    >>> accepted_hooks(lambda a, on_step=None: a, on_step=1, ask_user=2)
    {'on_step': 1}
    >>> accepted_hooks(lambda a, **kw: a, on_step=1, ask_user=2)
    {'on_step': 1, 'ask_user': 2}
    """
    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return dict(hooks)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return dict(hooks)
    names = {p.name for p in parameters
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.KEYWORD_ONLY)}
    return {name: value for name, value in hooks.items() if name in names}


__all__ = ["Assistant", "BUILTIN", "DEFAULT_KEY", "ENTRY_POINT_GROUP",
           "WORKFLOW_CONTRACT", "accepted_hooks", "active", "assistants",
           "get_assistant", "load_errors", "register_assistant", "set_active"]
