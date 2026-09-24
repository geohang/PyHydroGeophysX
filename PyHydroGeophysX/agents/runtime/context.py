"""The run transcript: what was asked, what was done, and what was found.

One object, passed to every tool, holding two separate things that used to be
conflated in a results dictionary:

- **Artifacts** - the data itself: loaded surveys, recovered models, meshes,
  arrays. Large, and of interest only to the tool that consumes them.
- **Steps** - the record of what happened: which tool ran, why, what it
  concluded, what it could not do. Small, and of interest to everything,
  including the model choosing the next action and the report describing the
  run afterwards.

Keeping them apart is what makes a controller loop affordable. The model reads
:meth:`RunContext.transcript`, which is a few hundred words whatever the run
did; it never sees a mesh. This is the same discipline a tool-using assistant
applies when it summarises a tool result instead of pasting it.

It also removes a whole class of reporting bug. The execution plan is derived
from the steps that ran (:meth:`RunContext.plan`) rather than written by hand
beside them, so a plan cannot describe a step that did not happen - which is
exactly what it used to do.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

#: A step's outcome. ``blocked`` is distinct from ``failed``: the tool never ran
#: because something it needed was absent, which is a fact about the run rather
#: than an error in the tool.
STATUSES = ("ok", "failed", "blocked", "skipped")


@dataclass
class Step:
    """One tool invocation and what came of it."""

    tool: str
    #: Why this step was chosen, in the controller's words. Recorded because a
    #: reader of the finished report asks "why did it do that", and the answer
    #: is otherwise nowhere.
    reason: str = ""
    status: str = "ok"
    #: One or two sentences of what the tool concluded - not its output data.
    summary: str = ""
    error: str = ""
    produced: List[str] = field(default_factory=list)
    seconds: float = 0.0
    agent: str = ""
    description: str = ""
    #: What the tool is for, in its own words. Distinct from ``reason``, which
    #: is why this run chose it; without a model to ask there is no reason, and
    #: the plan then read "next available step" for every row.
    purpose: str = ""

    def line(self) -> str:
        """The step as one line of transcript."""
        mark = {"ok": "done", "failed": "FAILED", "blocked": "blocked",
                "skipped": "skipped"}.get(self.status, self.status)
        parts = [f"[{mark}] {self.tool}"]
        if self.reason:
            parts.append(f"(chosen because: {self.reason})")
        detail = self.summary or self.error
        if detail:
            parts.append(f"-> {detail}")
        return " ".join(parts)


class RunContext:
    """Everything one workflow run knows about itself.

    Parameters
    ----------
    goal : str
        The user's request, in their own words.
    config : dict, optional
        The parsed workflow configuration.
    output_dir : str, optional
        Where artifacts are written.
    settings : dict, optional
        Anything a tool needs that is not configuration - API key, model name,
        provider, progress callback.

    Examples
    --------
    >>> ctx = RunContext('invert this line', {'ert_file': 'a.dat'})
    >>> ctx.put('ert_data', object(), 'Loaded 1 survey, 812 measurements.')
    >>> ctx.has('ert_data')
    True
    >>> ctx.begin('invert_ert', 'the data are loaded').tool
    'invert_ert'
    >>> ctx.finish(summary='Converged to chi-squared 1.6.')
    >>> ctx.steps[-1].status
    'ok'
    """

    def __init__(self, goal: str = "", config: Optional[Dict[str, Any]] = None,
                 output_dir: str = "", settings: Optional[Dict[str, Any]] = None):
        self.goal = str(goal or "")
        self.config: Dict[str, Any] = dict(config or {})
        self.output_dir = str(output_dir or "")
        self.settings: Dict[str, Any] = dict(settings or {})
        self.artifacts: Dict[str, Any] = {}
        #: A one-line description of each artifact, for the transcript. The
        #: artifact itself may be a hundred megabytes; this is what the model
        #: gets to see about it.
        self.descriptions: Dict[str, str] = {}
        self.steps: List[Step] = []
        self.warnings: List[str] = []
        #: Choices the run put to the user and the answers it got. Recorded so
        #: the report can say a result rests on a decision somebody made, which
        #: is otherwise invisible in the numbers.
        self.questions: List[Dict[str, Any]] = []
        self._open: Optional[Step] = None
        self._started = time.monotonic()

    # -- artifacts ---------------------------------------------------------
    def put(self, key: str, value: Any, description: str = "") -> None:
        """Record an artifact and how to describe it."""
        self.artifacts[key] = value
        if description:
            self.descriptions[key] = description
        if self._open is not None and key not in self._open.produced:
            self._open.produced.append(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def has(self, *keys: str) -> bool:
        """True when every named artifact is present and not None."""
        return all(self.artifacts.get(key) is not None for key in keys)

    def note(self, message: str) -> None:
        """A warning the user should see, deduplicated."""
        text = str(message).strip()
        if text and text not in self.warnings:
            self.warnings.append(text)

    def ask(self, question: str, options: List[Dict[str, str]],
            default: str = "") -> str:
        """Put a choice to the user and wait, or take ``default`` if nobody is there.

        For the cases a run genuinely cannot decide for itself: two files that
        disagree about their origin, a conversion with no calibration, a
        product that can be produced two defensible ways. Guessing there is
        worse than asking, and failing is worse than either - the run has
        usually done a lot of work by the time it finds out.

        Parameters
        ----------
        question : str
            What is being asked, in one or two sentences, stating the evidence.
        options : list of dict
            ``{"id", "label", "detail"}`` per choice. ``id`` is what comes
            back; ``detail`` says what that choice will actually do.
        default : str
            The id to use when there is no one to ask - a headless run, a
            script, a test. Defaults to the first option.

        Returns
        -------
        str
            The chosen option's id. Always one of the offered ids: an answer
            that matches nothing falls back to ``default``, because a typo must
            not select behaviour nobody offered.

        Raises
        ------
        None

        Examples
        --------
        >>> ctx = RunContext('goal')
        >>> options = [{'id': 'a', 'label': 'Use the station file'},
        ...            {'id': 'b', 'label': 'Use the SEG-Y headers'}]
        >>> ctx.ask('Which origin is right?', options)   # nobody to ask
        'a'
        >>> ctx.warnings[0].startswith('Nobody was available')
        True
        >>> ctx.settings['ask_user'] = lambda event: 'b'
        >>> ctx.ask('Which origin is right?', options)
        'b'
        """
        valid = [str(o.get("id", "")) for o in options if o.get("id")]
        fallback = str(default or (valid[0] if valid else ""))
        asker = self.settings.get("ask_user")
        if not callable(asker) or not valid:
            if valid:
                self.note(f"Nobody was available to answer: {question} "
                          f"Continued with '{fallback}'.")
            return fallback
        try:
            answer = str(asker({"event": "question", "question": str(question),
                                "options": list(options), "default": fallback}))
        except Exception:  # noqa: BLE001 - an unanswerable question is not a crash
            answer = ""
        chosen = answer if answer in valid else fallback
        self.questions.append({"question": str(question), "answer": chosen,
                               "options": list(options)})
        return chosen

    # -- steps -------------------------------------------------------------
    def begin(self, tool: str, reason: str = "", agent: str = "",
              description: str = "", purpose: str = "") -> Step:
        """Open a step. Only one is open at a time."""
        step = Step(tool=tool, reason=reason, agent=agent, description=description,
                    purpose=purpose)
        step._t0 = time.monotonic()  # type: ignore[attr-defined]
        self.steps.append(step)
        self._open = step
        return step

    def finish(self, summary: str = "", status: str = "ok", error: str = "") -> None:
        """Close the open step with what it concluded."""
        if self._open is None:
            return
        step = self._open
        step.status = status if status in STATUSES else "ok"
        step.summary = str(summary or "")
        step.error = str(error or "")
        step.seconds = time.monotonic() - getattr(step, "_t0", time.monotonic())
        self._open = None

    def ran(self, tool: str) -> bool:
        """True when ``tool`` has already completed successfully."""
        return any(s.tool == tool and s.status == "ok" for s in self.steps)

    def attempted(self, tool: str) -> int:
        """How many times ``tool`` has been tried, successfully or not.

        A ``blocked`` step is not an attempt: the tool never ran, because
        something it needed was missing or it was not on offer. Counting it made
        a premature choice permanent - a conversion named before the evaluation
        existed was never offered again once the evaluation did exist.
        """
        return sum(1 for s in self.steps if s.tool == tool and s.status != "blocked")

    # -- views -------------------------------------------------------------
    def transcript(self, limit: int = 40) -> str:
        """The run so far, as the controller reads it.

        Parameters
        ----------
        limit : int
            Most recent steps to include. A run that loops is the one that most
            needs its recent history; the early steps are represented by the
            artifacts they produced.

        Returns
        -------
        str
            Goal, artifacts on hand, and the step log.
        """
        lines = [f"Goal: {self.goal or '(none stated)'}"]
        if self.artifacts:
            lines.append("Available now:")
            for key in sorted(self.artifacts):
                described = self.descriptions.get(key, "")
                lines.append(f"  - {key}" + (f": {described}" if described else ""))
        else:
            lines.append("Available now: nothing yet.")
        if self.steps:
            lines.append("Steps so far:")
            for step in self.steps[-limit:]:
                lines.append(f"  {step.line()}")
        else:
            lines.append("Steps so far: none.")
        if self.warnings:
            lines.append("Warnings raised:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)

    def plan(self) -> List[Dict[str, Any]]:
        """The execution plan, derived from what actually ran.

        Returns the shape the report and the audit already consume, so a plan
        can no longer disagree with the run it describes.
        """
        return [{"step": step.description or step.tool,
                 "agent": step.agent or step.tool,
                 "description": step.purpose or step.description or "",
                 "reason": step.reason,
                 "status": step.status,
                 "seconds": round(step.seconds, 1),
                 "outputs": list(step.produced)}
                for step in self.steps]

    def unfinished(self) -> List[str]:
        """Tools that were attempted and did not succeed."""
        failed = [s.tool for s in self.steps if s.status in ("failed", "blocked")]
        return [tool for tool in dict.fromkeys(failed) if not self.ran(tool)]

    def results(self) -> Dict[str, Any]:
        """The artifacts, as the dictionary the existing callers expect."""
        out = dict(self.artifacts)
        out["warnings"] = list(self.warnings)
        return out

    def elapsed(self) -> float:
        return time.monotonic() - self._started


def merge_outputs(ctx: RunContext, outputs: Dict[str, Any],
                  describe: Optional[Dict[str, str]] = None,
                  keys: Optional[Iterable[str]] = None) -> List[str]:
    """Store a tool's outputs as artifacts, skipping empties.

    Parameters
    ----------
    ctx : RunContext
        The run to store into.
    outputs : dict
        What the tool returned.
    describe : dict, optional
        Per-key one-line descriptions for the transcript.
    keys : iterable of str, optional
        Restrict to these keys; by default everything that is not None and not
        bookkeeping (``status``, ``error``) is stored.

    Returns
    -------
    list of str
        The keys that were stored.

    Examples
    --------
    >>> ctx = RunContext('x')
    >>> merge_outputs(ctx, {'status': 'success', 'model': [1, 2], 'empty': None})
    ['model']
    """
    stored: List[str] = []
    skip = {"status", "error", "warnings"}
    for key, value in (outputs or {}).items():
        if keys is not None and key not in keys:
            continue
        if key in skip or value is None:
            continue
        ctx.put(key, value, (describe or {}).get(key, ""))
        stored.append(key)
    return stored
