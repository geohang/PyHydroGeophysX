"""The desktop studio's pages, as tools an automatic run can use.

An automatic run works headless in a process of its own, and its tools are the
runtime catalog's. The studio's pages hold more: ERT quality control, mesh
zones, other inversion engines, temperature correction, seismic picking,
boreholes, the Project Map. This is how a run reaches them - the same page
actions the user clicks and the chat drives (``agent_describe`` /
``agent_apply``), carried out by the desktop on its own thread
(``qt_apps.agent.controller.StudioController.serve_studio``).

The desktop sets :data:`current` to a function that takes one request and
returns the answer; elsewhere (a script, the web app, a test) it is None and
the run is offered no studio pages. A request is ``{"op": ...}``:

- ``catalog`` - the pages a run may use, with what each is for;
- ``describe`` - one page's actions, their arguments and its state;
- ``apply`` - run one action; a long one (an inversion) is answered when it
  has finished;
- ``handover`` - the page's finished result, written out for the run (the ERT
  page's model, see ``take_studio_inversion`` in the catalog).
"""

from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

#: The desktop's bridge: request -> answer. None when there is no studio.
current: ContextVar[Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]] = ContextVar(
    "studio_bridge", default=None)

_catalog: List[Dict[str, str]] = []


def available() -> bool:
    """Whether this run can use the studio's pages."""
    return current.get() is not None


def call(op: str, **request: Any) -> Dict[str, Any]:
    """Send one request to the studio and return its answer.

    Never raises: no studio, or a bridge that fails, is a failed answer the
    caller reports like any other.

    Examples
    --------
    >>> call('describe', module='ert')['status']
    'failed'
    """
    bridge = current.get()
    if bridge is None:
        return {"status": "failed", "error": "This run has no studio to work in."}
    try:
        answer = bridge({"op": op, **request})
    except Exception as exc:  # noqa: BLE001 - a lost bridge is a failed call
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return answer if isinstance(answer, dict) else {"status": "failed",
                                                    "error": "No answer from the studio."}


def catalog() -> List[Dict[str, str]]:
    """The pages a run may use, asked once per process."""
    if not _catalog and available():
        answer = call("catalog")
        _catalog.extend(m for m in answer.get("modules") or [] if isinstance(m, dict))
    return list(_catalog)


def describe_compact(desc: Any, action: str = "") -> str:
    """A page's description, short enough for the controller to read.

    Every action with its arguments and the first sentence of what it does,
    then the page's state - a page with twenty actions and paragraphs of
    detail on each would otherwise crowd out everything else the model reads.
    With ``action``, that one action's whole description instead.

    Examples
    --------
    >>> page = {'actions': [{'name': 'run', 'args': {'lambda': 'float'},
    ...                      'desc': 'Invert. Long detail.'}], 'state': {'has_model': False}}
    >>> print(describe_compact(page))
    actions: run(lambda: float) - Invert.
    state: {'has_model': False}
    >>> print(describe_compact(page, 'run'))
    run(lambda: float): Invert. Long detail.
    """
    if not isinstance(desc, dict):
        return str(desc)
    actions = [a for a in desc.get("actions") or [] if isinstance(a, dict)]

    def signature(entry):
        args = entry.get("args") or {}
        sig = ", ".join(f"{k}: {v}" for k, v in args.items()) if isinstance(args, dict) else ""
        return f"{entry.get('name')}({sig})"

    if action:
        for entry in actions:
            if entry.get("name") == action:
                return f"{signature(entry)}: {entry.get('desc') or ''}"
        return f"No action '{action}'; the actions are " + ", ".join(
            str(a.get("name")) for a in actions) + "."
    lines = []
    if actions:
        parts = []
        for entry in actions:
            text = str(entry.get("desc") or "")
            first = text.split(". ")[0].rstrip(".")
            parts.append(signature(entry) + (f" - {first}." if first else ""))
        lines.append("actions: " + "; ".join(parts))
    state = desc.get("state")
    if state is not None:
        lines.append(f"state: {state}")
    note = desc.get("note")
    if note:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def result_summary(answer: Dict[str, Any], limit: int = 700) -> str:
    """What a page action came back with, in one line for the transcript."""
    keep = {k: v for k, v in answer.items()
            if k not in ("describe", "_image", "page_state") and v not in (None, "", [], {})}
    text = ", ".join(f"{k}: {v}" for k, v in keep.items())
    state = answer.get("page_state")
    if isinstance(state, dict):
        brief = {k: v for k, v in state.items()
                 if isinstance(v, (bool, int, float, str)) and v not in ("", None)}
        text += f"; page now: {brief}"
    return text[:limit]
