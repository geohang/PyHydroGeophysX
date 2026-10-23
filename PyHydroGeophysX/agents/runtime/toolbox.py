"""What the controller may do besides the next step on the standard route.

The registry's dependency order is the standard route: load, invert, evaluate,
convert, report. It is a good default and it stays the default - a run without
a model follows it exactly, and a run with one follows it unless the evidence
says otherwise. What it cannot do on its own is react to that evidence beyond
the one retry of a failed step: run the inversion again because the fit is
poor, add the climate series that would explain a change, look at a product's
numbers before deciding, or write ten lines of Python for something no tool
does.

This module is that latitude, and its limits:

- **Read-only looks are free.** ``inspect`` summarises a product's numbers into
  the transcript, and ``read_skill`` puts a written procedure there
  (:mod:`PyHydroGeophysX.agents.skills`); neither changes anything, so they
  need nobody's permission.
- **Leaving a step out is the model's call.** A step for data the request never
  named, or a method it did not ask for, is dropped on the model's judgement,
  with its reason recorded - but never one the report or a requested product
  still needs (``controller._skip_steps``).
- **Every other change of course is the user's decision.** A tool the route has
  not scheduled, a step run again with other settings, a changed setting, a
  temporary script, a question: each is put to the user with the model's
  reason as its recommendation (:func:`propose`), and staying on the route is
  the default - the answer when nobody is there.
- **Bounded.** At most :data:`MAX_PROPOSALS` proposals in a run, and a declined
  proposal is not offered again, so a run cannot turn into a negotiation.
"""

from typing import Any, Dict, List, Sequence, Set

import numpy as np

from .context import RunContext

#: Off-route proposals a run may put to the user. Each costs the user an
#: answer; a handful is enough to change course where it matters.
MAX_PROPOSALS = 4

#: Looks a run may take - at a product's numbers, or at a studio page's actions
#: and state - before it must act on what it has seen.
MAX_INSPECTIONS = 10

#: Written procedures (:mod:`PyHydroGeophysX.agents.skills`) a run may read. A
#: procedure read stays in the transcript for every decision after it, so a few
#: is the limit, each cut to :data:`SKILL_READ_CHARS`.
MAX_SKILL_READS = 3
SKILL_READ_CHARS = 4000
#: Longest description of a procedure in the controller's index.
SKILL_INDEX_CHARS = 300


def proposals_left(ctx: RunContext) -> int:
    """How many more off-route proposals this run may make."""
    return max(0, MAX_PROPOSALS - len(ctx.proposals))


def looks_left(ctx: RunContext) -> int:
    """How many more looks - at a product, or at a studio page - this run may take."""
    taken = sum(1 for s in ctx.steps if s.tool == "inspect" or s.tool.startswith("look:"))
    return max(0, MAX_INSPECTIONS - taken)


def studio_pages_allowed(ctx: RunContext) -> Set[str]:
    """The studio pages the user has let this run use."""
    return {p["key"].split(":", 1)[1] for p in ctx.proposals
            if p.get("accepted") and str(p.get("key", "")).startswith("studio:")}


def studio_text(ctx: RunContext) -> List[str]:
    """The studio's pages, as the controller's prompt lists them; [] without a studio."""
    from . import studio
    from .recovery import input_paths

    pages = studio.catalog()
    if not pages:
        return []
    allowed = studio_pages_allowed(ctx)
    lines = ["Studio pages - the desktop studio's own tools, the ones the user clicks:"]
    lines.extend(f"  - {p['key']} ({p['title']}): {p.get('purpose', '')}" for p in pages)
    lines.append(
        "  \"studio\": {\"module\": key, \"action\": \"describe\"} shows a page's actions, "
        "their arguments and its state (a look, without asking); add \"args\": {\"action\": "
        "name} for one action's full description. \"studio\": {\"module\": key, "
        "\"action\": name, \"args\": {...}} runs one action. The first action on a page is "
        "put to the user with your reason; once they agree, the run may use that page. A "
        "long action (an inversion) returns when it has finished. A model made on the ERT "
        "page comes into this run with the step take_studio_inversion.")
    lines.append("  Pages the user has let you use: " + (", ".join(sorted(allowed)) or "none yet"))
    files = input_paths(ctx)
    if files:
        lines.append("  Files of this run a page can open: "
                     + "; ".join(f"{k} = {v}" for k, v in list(files.items())[:12]))
    return lines


def declined(ctx: RunContext) -> Set[str]:
    """Keys of the proposals the user turned down, never to be offered again."""
    return {p["key"] for p in ctx.proposals if not p.get("accepted")}


def extra_tools(ctx: RunContext, tools: Dict[str, Any]) -> List[Any]:
    """Tools whose inputs exist but which the route has not scheduled.

    Their ``when`` gate is closed - the request did not name their product, or
    the configuration does not call for them - and they have not been tried.
    The gate is right for a run that cannot ask; with a user to ask, the model
    may propose one.

    Examples
    --------
    >>> from .tools import Tool
    >>> only = {'climate': Tool('climate', 'Weather.', lambda c: ('', {}),
    ...                         when=lambda c: False)}
    >>> [t.name for t in extra_tools(RunContext('x'), only)]
    ['climate']
    """
    out = []
    refused = declined(ctx)
    for tool in tools.values():
        if (f"extra:{tool.name}" in refused or not getattr(tool, "proposable", True)
                or not ctx.has(*tool.requires)):
            continue
        if ctx.attempted(tool.name) and not tool.repeatable:
            continue
        applies = getattr(tool, "applies", None)
        try:
            if applies is not None and not applies(ctx):
                continue
        except Exception:  # noqa: BLE001 - a broken check offers nothing
            continue
        try:
            scheduled = tool.when is None or bool(tool.when(ctx))
        except Exception:  # noqa: BLE001 - a broken gate counts as scheduled
            scheduled = True
        if not scheduled:
            out.append(tool)
    return out


def redo_candidates(ctx: RunContext, tools: Dict[str, Any],
                    finish: str = "report_files") -> List[Any]:
    """Finished steps that could be run again with other settings.

    Not the step that writes the final product: it is run again by itself
    whenever a step before it is (:func:`downstream`).
    """
    done = {s.tool for s in ctx.steps if s.status == "ok"}
    refused = declined(ctx)
    return [tool for name, tool in tools.items()
            if name in done and tool.produces and finish not in tool.produces
            and ctx.has(*tool.requires) and f"redo:{name}" not in refused]


def downstream(tools: Dict[str, Any], produced: Sequence[str]) -> List[str]:
    """Every tool that reads, directly or through others, what ``produced`` names.

    Examples
    --------
    >>> from .tools import Tool
    >>> reg = {'inv': Tool('inv', '', lambda c: ('', {}), produces=('m',)),
    ...        'ev': Tool('ev', '', lambda c: ('', {}), requires=('m',), produces=('e',)),
    ...        'rep': Tool('rep', '', lambda c: ('', {}), requires=('e',))}
    >>> downstream(reg, ['m'])
    ['ev', 'rep']
    """
    changed = set(produced)
    found: List[str] = []
    grew = True
    while grew:
        grew = False
        for name, tool in tools.items():
            if name in found or not changed.intersection(tool.requires):
                continue
            found.append(name)
            changed.update(tool.produces)
            grew = True
    return found


def skippable(ctx: RunContext, tools: Dict[str, Any], route: Sequence[str],
              finish: str = "report_files") -> List[str]:
    """Steps on the route ahead that may be left out, by name.

    Not the step that writes the final product, and not one already tried. The
    steps after a skipped one that need its products drop off the route by
    themselves. Whether a step is still needed for a requested product is
    decided when it is skipped (``controller._skip_steps``).

    Examples
    --------
    >>> from .tools import Tool
    >>> reg = {'tem': Tool('tem', '', lambda c: ('', {}), produces=('t',)),
    ...        'rep': Tool('rep', '', lambda c: ('', {}), produces=('report_files',))}
    >>> skippable(RunContext('x'), reg, ['tem', 'rep'])
    ['tem']
    """
    out = []
    for name in route:
        tool = tools.get(name)
        if tool is None or finish in tool.produces or ctx.attempted(name):
            continue
        out.append(name)
    return out


def toolbox_text(ctx: RunContext, tools: Dict[str, Any], scripts: bool = True,
                 route: Sequence[str] = (), finish: str = "report_files") -> str:
    """The off-route actions as the controller's prompt lists them.

    ``route`` is the standard route ahead, by step name, for the steps the
    model may propose to leave out.
    """
    left = proposals_left(ctx)
    lines: List[str] = []
    dropped = skippable(ctx, tools, route, finish)
    if dropped:
        lines.append("Your call, without asking: leave out route steps that do not serve "
                     "the goal (name them in \"skip\"): " + ", ".join(dropped))
    if left:
        extras = extra_tools(ctx, tools)
        if extras:
            lines.append("Steps the route has not scheduled (name them in \"tool\"):")
            lines.extend("  " + tool.offer() for tool in extras)
        redo = redo_candidates(ctx, tools)
        if redo:
            lines.append("Steps already done that can run again with changed "
                         "\"settings\" (name them in \"tool\"; the steps after them run "
                         "again on the new result): "
                         + ", ".join(tool.name for tool in redo))
        if scripts:
            lines.append("A temporary Python script, for something no step does "
                         "(\"script\").")
        lines.append("A question to the user, when the choice is theirs (\"ask\").")
        lines.append(f"You may make {left} more off-route proposal(s) in this run.")
    else:
        lines.append("(No more off-route proposals in this run: keep to the route.)")
    looks = looks_left(ctx)
    if looks > 0:
        lines.append("Free, without asking: \"inspect\" a product listed under "
                     f"Available now to see its numbers ({looks} looks left).")
    lines.extend(skills_text(ctx))
    lines.extend(studio_text(ctx))
    refused = sorted(declined(ctx))
    if refused:
        lines.append("The user declined: " + ", ".join(refused) + ". Do not propose "
                     "these again.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# looking at a product
# ---------------------------------------------------------------------------
def describe_value(value: Any, depth: int = 0, budget: int = 1600) -> str:
    """A product's numbers in a few lines: shapes, ranges, the scalars it holds.

    Examples
    --------
    >>> describe_value(np.array([[1.0, 2.0], [3.0, np.nan]]))
    'array (2, 2): 3 finite, min 1, median 2, max 3'
    >>> describe_value({'chi2': 1.5, 'status': 'success'})
    '{chi2: 1.5; status: success}'
    """
    if value is None:
        return "None"
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return f"{float(value):.4g}" if isinstance(value, (float, np.floating)) else str(value)
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            return f"array {array.shape} of {array.dtype}"
        finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array.ravel()
        if finite.size == 0:
            return f"array {array.shape}: no finite values"
        return (f"array {array.shape}: {finite.size} finite, min {finite.min():.4g}, "
                f"median {np.median(finite):.4g}, max {finite.max():.4g}")
    if hasattr(value, "cellCount") and callable(value.cellCount):
        try:
            return f"mesh: {value.cellCount()} cells, {value.nodeCount()} nodes"
        except Exception:  # noqa: BLE001
            return "mesh"
    if depth >= 2:
        return type(value).__name__
    if isinstance(value, dict):
        parts = []
        for key, item in list(value.items())[:24]:
            parts.append(f"{key}: {describe_value(item, depth + 1, budget // 4)}")
            if sum(len(p) for p in parts) > budget:
                parts.append("...")
                break
        return "{" + "; ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "empty list"
        numbers = [v for v in value if isinstance(v, (int, float, np.integer, np.floating))]
        if len(numbers) == len(value):
            return describe_value(np.asarray(value, dtype=float), depth + 1, budget)
        return (f"list of {len(value)}; first: "
                f"{describe_value(value[0], depth + 1, budget // 2)}")
    return type(value).__name__


def inspect_artifact(ctx: RunContext, key: str) -> str:
    """Record a read-only look at product ``key`` as a step, and return what it saw."""
    ctx.begin("inspect", f"look at {key}", agent="Inspect",
              description=f"Inspect {key}", purpose="Read a product's numbers; changes nothing.")
    if not ctx.has(key):
        ctx.finish(status="blocked",
                   error=f"No product named '{key}'. Inspect one listed under Available now.")
        return ""
    text = describe_value(ctx.get(key))[:1800]
    ctx.finish(summary=f"{key}: {text}")
    return text


# ---------------------------------------------------------------------------
# reading a written procedure
# ---------------------------------------------------------------------------
def run_skills(ctx: RunContext) -> List[Any]:
    """The written procedures this run's assistant can read (``settings['skills']``)."""
    return list(ctx.settings.get("skills") or [])


def skills_read(ctx: RunContext) -> List[str]:
    """The procedures this run has read, by name, oldest first."""
    return [s.tool.split(":", 1)[1] for s in ctx.steps
            if s.tool.startswith("skill:") and s.status == "ok"]


def skill_reads_left(ctx: RunContext) -> int:
    """How many more procedures this run may read."""
    taken = sum(1 for s in ctx.steps if s.tool.startswith("skill:"))
    return max(0, MAX_SKILL_READS - taken)


def skills_text(ctx: RunContext) -> List[str]:
    """The procedures not yet read, as the controller's prompt lists them; [] for none.

    Examples
    --------
    >>> from pathlib import Path
    >>> from PyHydroGeophysX.agents.skills import Skill
    >>> ctx = RunContext('x', settings={'skills': [Skill('ert-qc', 'Check ERT data.', Path('s'))]})
    >>> skills_text(ctx)[1]
    '  - ert-qc: Check ERT data.'
    """
    left = skill_reads_left(ctx)
    done = set(skills_read(ctx))
    unread = [s for s in run_skills(ctx) if s.name not in done]
    if not unread or not left:
        return []
    from ..skills import index_text

    return ["Written procedures, free to read without asking when one fits what the run "
            f"is doing (\"read_skill\": name in place of \"tool\"; {left} reads left). "
            "A procedure is advice: the user's goal and the rules here come first.",
            *("  " + line for line in index_text(unread, SKILL_INDEX_CHARS).splitlines())]


def read_skill(ctx: RunContext, name: str, why: str = "") -> str:
    """Record a read of the procedure ``name`` as a step, and return its text.

    The procedure goes into the step's summary, which is how the transcript -
    and so every decision after it - carries what it says.
    """
    from ..skills import find

    skill = find(run_skills(ctx), name)
    ctx.begin(f"skill:{skill.name if skill else name}", why, agent="Skills",
              description=f"Read the procedure {skill.name if skill else name}",
              purpose="Read a written procedure; changes nothing.")
    if skill is None:
        ctx.finish(status="blocked", error=f"No procedure named '{name}'. Read one listed "
                                           "under Written procedures.")
        return ""
    try:
        text = skill.body(limit=SKILL_READ_CHARS)
    except OSError as exc:
        ctx.finish(status="failed", error=f"Could not read it ({type(exc).__name__}: {exc}).")
        return ""
    indented = "\n".join("      " + line if line else "" for line in text.splitlines())
    ctx.finish(summary=f"Procedure {skill.name} (advice to weigh against the evidence, "
                       f"not an instruction; the user's goal comes first):\n{indented}")
    return text


# ---------------------------------------------------------------------------
# asking the user
# ---------------------------------------------------------------------------
def propose(ctx: RunContext, key: str, title: str, why: str, route_next: str,
            name: str = "The assistant", detail: str = "") -> bool:
    """Put an off-route action to the user, recommending it; True if accepted.

    ``key`` identifies the proposal (``extra:<tool>``, ``redo:<tool>``,
    ``settings``, ``script``...) so a declined one is never offered again.
    Staying on the route is the default and the answer when nobody is there.

    Examples
    --------
    >>> ctx = RunContext('x')
    >>> propose(ctx, 'extra:climate', 'Fetch the climate series', 'rain explains it',
    ...         'Write the report')
    False
    >>> ctx.proposals[0]['accepted'], 'extra:climate' in declined(ctx)
    (False, True)
    >>> ctx.settings['ask_user'] = lambda event: 'go'
    >>> propose(ctx, 'redo:invert', 'Invert again with lambda 20', 'chi2 is 39', 'Evaluate')
    True
    """
    question = (f"{name} suggests a change of course: {title}.\n\n"
                f"Why: {why or 'no reason given'}"
                + (f"\n\n{detail}" if detail else "")
                + f"\n\nThe standard next step would be: {route_next or 'to finish the run'}.")
    answer = ctx.ask(question, [
        {"id": "go", "label": "Yes, do it (recommended)",
         "detail": f"Do what {name} suggests; it is recorded in the report."},
        {"id": "route", "label": "No - keep to the standard route",
         "detail": "Carry on with the usual next step."}], default="route")
    accepted = answer == "go"
    ctx.proposals.append({"key": key, "title": title, "why": why, "accepted": accepted})
    return accepted


def ask_choice(ctx: RunContext, question: str, options: Sequence[str],
               recommend: str = "", why: str = "") -> str:
    """Ask the user a question of the model's, and record the answer as guidance.

    Examples
    --------
    >>> ctx = RunContext('x')
    >>> ask_choice(ctx, 'Archie or Waxman-Smits?', ['Archie', 'Waxman-Smits'], 'Archie')
    'Archie'
    >>> print(ctx.guidance[-1])
    You asked: Archie or Waxman-Smits? The user chose: Archie (the default; nobody answered).
    """
    labels = [str(o)[:120] for o in options if str(o).strip()][:4]
    if len(labels) < 2:
        return ""
    best = recommend if recommend in labels else labels[0]
    ids = [f"opt{i}" for i in range(len(labels))]
    answer = ctx.ask(question + (f"\n\nWhy this matters: {why}" if why else ""),
                     [{"id": i, "label": label + (" (recommended)" if label == best else ""),
                       "detail": ""} for i, label in zip(ids, labels)],
                     default=ids[labels.index(best)])
    chosen = labels[ids.index(answer)] if answer in ids else best
    answered = bool(ctx.questions) and ctx.questions[-1].get("question", "").startswith(question)
    ctx.proposals.append({"key": f"ask:{question[:40]}", "title": question,
                          "why": why, "accepted": True})
    ctx.guidance.append(f"You asked: {question} The user chose: {chosen}"
                        + ("." if answered else " (the default; nobody answered)."))
    return chosen
