"""What the loop does about a step that failed.

Recording a failed step and moving on is right for a step nothing can be done
about, and wrong for the ones that fail because a setting was unsuitable rather
than because the work is impossible - an inversion that diverged at the
configured regularisation, a conversion given a threshold no cell meets, a
loader pointed at the wrong column layout. A person looking at the error would
change one number and run it again; this lets the loop do the same. It
generalises the inversion's own retry: evaluate the result, adjust lambda,
invert again.

Three things are deliberately separated:

- **Diagnosis** is the model's, because reading an error message and saying what
  it means is what a model is for. It never runs anything.
- **The decision** is bounded here: the only recovery this performs on its own
  is re-running the same tool with different *configuration*. Changing numbers
  is reversible, inspectable, and recorded in the plan; it cannot reach outside
  the run.
- **Anything beyond configuration** - a generated reader for a coordinate file
  nobody anticipated - is a proposal that goes to the user through
  :meth:`RunContext.ask` with the source in front of them, and runs only on an
  explicit yes. See :mod:`~PyHydroGeophysX.agents.runtime.adapters`. Code a
  model wrote, executed unattended against somebody's data, is not a thing this
  package does on its own initiative: a run with nobody to ask declines.

A recovery is itself a step in the transcript, with the diagnosis as its reason,
so a reader of the report sees what failed, what the run thought about it, what
it changed, and whether that worked.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..._internal.utils import fence_data, parse_json_object
from .context import RunContext

#: How many times one tool may be recovered within a run. One retry catches the
#: unsuitable-setting case; a tool that fails twice with different settings is
#: failing for a reason settings will not fix, and going round again only spends
#: the user's time and tokens.
MAX_RECOVERIES = 1

#: Configuration keys a diagnosis is allowed to change. Everything here is a
#: number or a switch that selects between behaviours already implemented and
#: tested. Absent on purpose: every path key, because a recovery that can point
#: the run at a different file is a recovery that can read anything on the disk,
#: and the user chose those files.
ADJUSTABLE = (
    "inversion_params", "temporal_regularization", "time_lapse_method",
    "quality_threshold", "max_attempts", "auto_adjust",
    "velocity_threshold", "seismic_inversion_params", "align_origin",
    "petrophysical_params", "n_realizations", "coverage_threshold",
    "instrument", "crs", "first_break_params", "tdem_params", "mt_params",
    "gravmag_params", "gravmag_kind", "magnetic_field",
    # Not gpr_velocity: a depth needs a velocity somebody gave, not one a
    # recovery chose.
    "gpr_params",
)

RECOVERY_PROMPT = """A step of a hydrogeophysical workflow failed. Say what to do.

The run so far:
{transcript}

The step that failed: {tool}
What it is for: {purpose}
The error: {error}

The settings it ran with (only these may be changed):
{settings}

Coordinate files in this run that a reader could be written for, if one of
them is what could not be read: {adaptable}

The run's input files and folders, by configuration key: {inputs}

Answer with JSON and nothing else:
{{"cause": "one sentence on why this failed",
  "action": "retry" | "adapt" | "script" | "give_up",
  "changes": {{"<setting>": <new value>}},
  "file_key": "<one of the coordinate files above, for adapt only>",
  "replaces": "<an input key above whose file the script's output stands in for, or null>",
  "purpose": "<for script: what the script will do and why that gets the step past this, in two sentences a scientist can check>",
  "why": "one sentence on why this should work"}}

Rules:
- "retry" only when a different setting could plausibly fix it. Re-running with
  the same settings is never worth doing.
- "adapt" only when a coordinate file above holds the right numbers in a layout
  the reader does not handle. Name it in "file_key". A human will read the
  reader you would write and decide whether to run it, so do not propose this
  for a file whose contents are simply wrong or missing.
- "script" when a short Python script could get past the cause and no setting
  can: convert or repair an input into a form the step reads (name it in
  "replaces"; the step is then run again on the script's output), or work out
  something the step needs. You will be asked for the code next; a person reads
  it and decides whether it runs, and the original files are never changed.
- "give_up" when the failure is about missing data, an unimplemented method, or
  anything none of the above can change. This is a normal answer, not a defeat;
  say the cause plainly so the report can state it.
- Only the settings listed above may appear in "changes". Keep the same types.
- Change as little as possible: one or two values, not a new configuration."""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """The first JSON object in ``text``, or None.

    Models wrap JSON in prose and in code fences however they are asked not to.

    Examples
    --------
    >>> _extract_json('```json\\n{"action": "retry"}\\n```')
    {'action': 'retry'}
    >>> _extract_json('no json here') is None
    True
    """
    if not text:
        return None
    parsed = parse_json_object(str(text))
    return parsed if isinstance(parsed, dict) else None


def _merge(current: Any, change: Any) -> Any:
    """Apply ``change`` to ``current``, merging one level of dict.

    A diagnosis that wants a different lambda says ``{"inversion_params":
    {"lambda": 5}}`` and means "that one key", not "replace every inversion
    setting with this".

    Examples
    --------
    >>> _merge({'lambda': 15, 'max_iterations': 10}, {'lambda': 5})
    {'lambda': 5, 'max_iterations': 10}
    >>> _merge(15.0, 5.0)
    5.0
    """
    if isinstance(current, dict) and isinstance(change, dict):
        merged = dict(current)
        merged.update(change)
        return merged
    return change


def settings_for(ctx: RunContext) -> Dict[str, Any]:
    """The adjustable part of the configuration, as the diagnosis sees it."""
    return {key: ctx.config[key] for key in ADJUSTABLE if key in ctx.config}


def input_paths(ctx: RunContext) -> Dict[str, str]:
    """The run's input files and folders that exist, by configuration key.

    A list (time-lapse surveys) is spelled out one entry per file, as
    ``key[0]``, ``key[1]``... A temporary script reads these and nothing else.

    Examples
    --------
    >>> input_paths(RunContext('goal', {'data_file': 'nowhere.dat', 'lambda': 5}))
    {}
    """
    found: Dict[str, str] = {}
    for key, value in ctx.config.items():
        if not str(key).endswith(("_file", "_files", "_dir")):
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for index, item in enumerate(values):
            if item and Path(str(item)).exists():
                found[f"{key}[{index}]" if isinstance(value, (list, tuple)) else key] = str(item)
    # What the run's steps have written so far, for a script to read: the
    # model bundle, the report, figures named in a product.
    for key, value in ctx.artifacts.items():
        entries = value.items() if isinstance(value, dict) else [(None, value)]
        for sub, item in entries:
            if isinstance(item, str) and item and Path(item).exists():
                found[key if sub is None else f"{key}.{sub}"] = item
    return found


def diagnose(ctx: RunContext, tool: Any, error: str,
             ask: Optional[Callable[[str], str]]) -> Dict[str, Any]:
    """Ask what went wrong and what to change, and keep only what is allowed.

    Parameters
    ----------
    ctx : RunContext
        The run, for the transcript and the current settings.
    tool : Tool or None
        The tool that failed.
    error : str
        Its error message.
    ask : callable or None
        Takes a prompt, returns the model's reply. Without one there is no
        diagnosis: a deterministic run does not guess at causes.

    Returns
    -------
    dict
        ``{"cause", "action", "changes", "why"}``. ``action`` is ``"retry"``
        only when a diagnosis asked for it *and* named at least one allowed
        setting that would actually differ; otherwise ``"give_up"``, so a model
        that answers "retry" with nothing to change cannot loop the run.

    Raises
    ------
    None
        An unreachable or unparseable model gives up, which is what the loop
        did before any of this existed.

    Examples
    --------
    >>> ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    >>> reply = '{"cause": "lambda too high", "action": "retry",' \\
    ...         ' "changes": {"inversion_params": {"lambda": 5}},' \\
    ...         ' "why": "less smoothing"}'
    >>> plan = diagnose(ctx, None, 'diverged', lambda prompt: reply)
    >>> plan['action'], plan['changes']
    ('retry', {'inversion_params': {'lambda': 5}})
    >>> diagnose(ctx, None, 'diverged', None)['action']
    'give_up'
    """
    blank = {"cause": "", "action": "give_up", "changes": {}, "file_key": "",
             "replaces": "", "purpose": "", "why": ""}
    if not callable(ask):
        return blank
    settings = settings_for(ctx)
    from .adapters import ADAPTABLE_FILES

    present = [key for key in ADAPTABLE_FILES if ctx.config.get(key)]
    inputs = input_paths(ctx)
    prompt = RECOVERY_PROMPT.format(
        transcript=ctx.transcript(limit=12),
        tool=getattr(tool, "name", "") or "the step",
        purpose=getattr(tool, "description", "") or "(not described)",
        error=str(error)[:800],
        settings=json.dumps(settings, indent=2, default=str)[:2000],
        adaptable=", ".join(present) or "(none)",
        inputs=json.dumps(inputs, indent=2)[:1500] if inputs else "(none)")
    try:
        parsed = _extract_json(ask(prompt))
    except Exception:  # noqa: BLE001 - an unreachable model is not a crash
        parsed = None
    if not parsed:
        return blank
    # Only the allowed keys, and only values that would actually differ. A
    # "retry" that changes nothing is a re-run of the thing that just failed.
    changes = {key: value for key, value in (parsed.get("changes") or {}).items()
               if key in ADJUSTABLE
               and _merge(ctx.config.get(key), value) != ctx.config.get(key)}
    wanted = str(parsed.get("action"))
    file_key = str(parsed.get("file_key") or "")
    # Only an input the run actually has can be stood in for.
    replaces = str(parsed.get("replaces") or "")
    replaces = replaces if replaces in inputs and "[" not in replaces else ""
    purpose = str(parsed.get("purpose") or "")[:600]
    if wanted == "adapt" and file_key in ADAPTABLE_FILES and ctx.config.get(file_key):
        action = "adapt"
    elif wanted == "retry" and changes:
        action = "retry"
    elif wanted == "script" and purpose:
        action = "script"
    else:
        action = "give_up"
        file_key = ""
    return {"cause": str(parsed.get("cause") or "")[:300],
            "action": action,
            "changes": changes,
            "file_key": file_key,
            "replaces": replaces if action == "script" else "",
            "purpose": purpose if action == "script" else "",
            "why": str(parsed.get("why") or "")[:300]}


def apply_changes(ctx: RunContext, changes: Dict[str, Any]) -> List[str]:
    """Put the diagnosis's settings into the configuration, and say which.

    Returns
    -------
    list of str
        ``"key: old -> new"`` per change, for the transcript. The run's
        configuration is what the report prints, so a number that changed
        mid-run has to be visible as having changed.

    Examples
    --------
    >>> ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    >>> apply_changes(ctx, {'inversion_params': {'lambda': 5}})
    ["inversion_params: {'lambda': 15} -> {'lambda': 5}"]
    """
    applied = []
    for key, value in (changes or {}).items():
        if key not in ADJUSTABLE:
            continue
        before = ctx.config.get(key)
        after = _merge(before, value)
        ctx.config[key] = after
        applied.append(f"{key}: {before} -> {after}")
    return applied


def recover(ctx: RunContext, tool: Any, error: str,
            ask: Optional[Callable[[str], str]],
            run: Callable[[str], str],
            attempts: Optional[Dict[str, int]] = None) -> Tuple[str, str]:
    """Diagnose a failed step and, if a setting can be blamed, run it again.

    Parameters
    ----------
    ctx : RunContext
        The run. Its configuration is what a recovery changes.
    tool : Tool
        The tool that failed.
    error : str
        Its error.
    ask : callable or None
        The model, for the diagnosis.
    run : callable
        Takes the tool's name and runs it again, returning the new status. The
        controller passes its own ``invoke`` so the retry is an ordinary step,
        recorded like any other.
    attempts : dict, optional
        Tool name to recoveries already spent, carried across a run.

    Returns
    -------
    tuple
        ``(status, note)`` - the status after any retry, and a sentence for the
        warnings. ``status`` is the original ``"failed"`` when nothing was tried.

    Raises
    ------
    None

    Examples
    --------
    >>> from .tools import Tool
    >>> ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    >>> tool = Tool('invert', 'Invert.', lambda c: ('', {}),
    ...             label='Run ERT inversion')
    >>> reply = '{"cause": "too smooth", "action": "retry",' \\
    ...         ' "changes": {"inversion_params": {"lambda": 5}}, "why": "ok"}'
    >>> recover(ctx, tool, 'diverged', lambda p: reply, lambda name: 'ok')
    ('ok', "Run ERT inversion failed (diverged). Diagnosis: too smooth. Retried with inversion_params: {'lambda': 15} -> {'lambda': 5}, which worked.")
    """
    spent = attempts if attempts is not None else {}
    name = getattr(tool, "name", "") or ""
    if spent.get(name, 0) >= MAX_RECOVERIES:
        return "failed", ""
    plan = diagnose(ctx, tool, error, ask)
    label = getattr(tool, "label", "") or name or "the step"
    if plan["action"] == "script":
        spent[name] = spent.get(name, 0) + 1
        return script_fix(ctx, tool, error, plan, ask, run)
    if plan["action"] == "adapt":
        # A coordinate file whose layout nobody wrote a reader for. The model
        # writes one, the user reads it and decides, and only then does anything
        # run. Declining is the default and a perfectly good outcome.
        from .adapters import adapt_file

        spent[name] = spent.get(name, 0) + 1
        key = plan["file_key"]
        rewritten = adapt_file(ctx, key, error, ask)
        if not rewritten:
            return "failed", (f"{label} failed ({error}). Diagnosis: "
                              f"{plan['cause']} A reader for {key} was not run, "
                              f"so the step was left undone.")
        original = ctx.config.get(key)
        ctx.config[key] = rewritten
        status = run(name)
        outcome = "which worked" if status == "ok" else "which did not help either"
        return status, (f"{label} failed ({error}). Diagnosis: {plan['cause']} "
                        f"{key} was re-read by approved generated code and the "
                        f"step retried against {rewritten} in place of "
                        f"{original}, {outcome}.")
    if plan["action"] != "retry":
        if plan["cause"]:
            return "failed", (f"{label} failed ({error}). Diagnosis: "
                              f"{plan['cause']} Nothing in the settings would "
                              f"change that, so it was left undone.")
        return "failed", ""
    spent[name] = spent.get(name, 0) + 1
    applied = apply_changes(ctx, plan["changes"])
    ctx.note(f"{label} was retried with changed settings after it failed: "
             f"{'; '.join(applied)}. Reason given: {plan['why'] or plan['cause']}")
    status = run(name)
    outcome = "which worked" if status == "ok" else "which did not help either"
    return status, (f"{label} failed ({error}). Diagnosis: {plan['cause']}. "
                    f"Retried with {'; '.join(applied)}, {outcome}.")


SCRIPT_PROMPT = """A step of a hydrogeophysical workflow failed, and a short temporary
Python script is to get the run past it. A scientist will read the script before
deciding whether it runs.

The step that failed: {tool}
What it is for: {purpose}
The error: {error}
What the script is to do: {plan}
{replaces}
Inputs the script receives in `inputs` (key -> path), with the first lines of
each text file:
{samples}

{contract}

Answer with JSON and nothing else:
{{"purpose": "two sentences: what the script does and why that fixes the step",
  "code": "<the whole script>"}}"""

TASK_SCRIPT_PROMPT = """A hydrogeophysical workflow needs a short temporary Python
script for something none of its steps does. A scientist will read the script
before deciding whether it runs.

The run so far:
{transcript}

What the script is to do: {task}

Inputs the script receives in `inputs` (key -> path): the run's input files and
the files its steps have written, with the first lines of each text file:
{samples}

{contract}

Answer with JSON and nothing else:
{{"purpose": "two sentences: what the script does and what the run gains from it",
  "code": "<the whole script>"}}"""

FIX_SCRIPT_PROMPT = """A temporary Python script written for a hydrogeophysical
workflow failed when it ran. Fix it. A scientist will read the fixed script before
deciding whether it runs.

What the script is for: {purpose}

The script:
{code}

How it failed:
{error}

What it printed before that:
{printed}

Inputs the script receives in `inputs` (key -> path), with the first lines of
each text file:
{samples}

{contract}

Fix the cause the error shows; do not paper over it (a bare except, a made-up
value). Answer with JSON and nothing else:
{{"purpose": "two sentences: what was wrong and what the fixed script does",
  "code": "<the whole fixed script>"}}"""

#: Fixes the model may write for an approved script that failed, each put to
#: the user again before it runs.
MAX_SCRIPT_FIXES = 1

#: The reply cap for a script: none. A capped reply cut every proposal with
#: code in it short, mid-function.
SCRIPT_TOKENS = None


def ask_long(ask: Callable[..., str], prompt: str,
             max_tokens: Optional[int] = SCRIPT_TOKENS) -> str:
    """Ask with no cap on the reply, when ``ask`` lets the caller say so."""
    try:
        takes = "max_tokens" in inspect.signature(ask).parameters
    except (TypeError, ValueError):
        takes = False
    return ask(prompt, max_tokens=max_tokens) if takes else ask(prompt)


def _preview(path: str, lines: int = 12, size: int = 1500) -> str:
    """The first lines of a text file, or what kind of thing ``path`` is."""
    target = Path(path)
    if target.is_dir():
        names = sorted(p.name for p in target.iterdir())[:25]
        return "(folder) " + ", ".join(names)
    try:
        with target.open("rb") as handle:
            head = handle.read(size)
    except OSError:
        return "(unreadable)"
    if b"\0" in head:
        return f"(binary, {target.stat().st_size} bytes)"
    return "\n".join(head.decode("utf-8", errors="replace").splitlines()[:lines])


def _samples(inputs: Dict[str, str], limit: int = 6000) -> str:
    """Each input's path and first lines, fenced as data, in at most ``limit`` characters.

    Examples
    --------
    >>> _samples({})
    '(none)'
    """
    parts: List[str] = []
    used = 0
    for index, (key, path) in enumerate(inputs.items()):
        part = f"- {key}: {path}\n" + fence_data(_preview(path), f"the start of {Path(path).name}")
        if used + len(part) > limit:
            parts.append(f"- ({len(inputs) - index} more inputs, not shown)")
            break
        parts.append(part)
        used += len(part)
    return "\n".join(parts) or "(none)"


def _write_script(ask: Optional[Callable[..., str]], prompt: str) -> Tuple[str, str, str]:
    """The model's ``(purpose, code, problem)``; ``problem`` is "" when it may be offered.

    A draft that breaks a rule of :func:`.scratch.check_script` is sent back once
    with the reason.
    """
    from . import scratch

    if not callable(ask):
        return "", "", "there is no model to write it"
    purpose, code, problem = "", "", "no script was written"
    for _attempt in range(2):
        try:
            purpose, code = scratch.split_answer(ask_long(ask, prompt))
        except Exception as exc:  # noqa: BLE001 - an unreachable model writes nothing
            return "", "", str(exc)
        problem = scratch.check_script(code) if code else "no script was written"
        if not problem:
            break
        prompt += (f"\n\nYour last script was refused: {problem} "
                   "Write it again within the rules.")
    return purpose, code, problem


def _run_approved(ctx: RunContext, intro: str, purpose: str, code: str,
                  inputs: Dict[str, str], label: str, reason: str,
                  ask: Optional[Callable[..., str]] = None) -> Optional[Dict[str, Any]]:
    """Show the script and run it if the user says yes; the record, or None if declined.

    The run is a step of its own (``temporary_script``), so the plan and the
    report show what ran, what it returned and what it wrote. A script that
    fails goes back to the model (``ask``) with its error and what it printed;
    the fix it writes is checked and put to the user again, at most
    :data:`MAX_SCRIPT_FIXES` times. The record returned is the last run's;
    declining a fix keeps the failed one.
    """
    from . import scratch

    record: Optional[Dict[str, Any]] = None
    who = str(ctx.settings.get("assistant_name") or "the assistant")
    for attempt in range(MAX_SCRIPT_FIXES + 1):
        answer = ctx.ask(scratch.describe_for_user(f"{intro} {purpose}", code, inputs), [
            {"id": "decline", "label": "Do not run it",
             "detail": "Carry on without it. The report will say it was offered."},
            {"id": "run", "label": "Run this script",
             "detail": "Run it once for this run, in its own folder with a time limit. "
                       "Your files are not changed, and the script and what it did are "
                       "kept with the run and named in the report."}],
            default="decline")
        if answer != "run":
            return record
        folder = scratch.script_folder(Path(ctx.output_dir) / "scratch", label or "script")
        ctx.begin("temporary_script", reason=reason, agent="Temporary script",
                  description=("Run an approved temporary script" if not attempt else
                               "Run the fixed temporary script"), purpose=purpose)
        record = scratch.run_script(code, inputs, folder, purpose=purpose)
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        found = json.dumps(result, default=str)[:400] if result else ""
        if record["status"] != "ok":
            ctx.finish(summary=f"The approved script failed: {record['error'][-300:]}",
                       status="failed", error=record["error"][-500:])
        else:
            ctx.finish(summary=f"The approved script ran in {record['seconds']} s"
                               + (f" and returned {found}" if found else "")
                               + (f"; it wrote {', '.join(record['files'][:6])}"
                                  if record["files"] else "")
                               + f" (kept in {folder}).", status="ok")
        record["found"] = found
        record["folder"] = str(folder)
        if record["status"] == "ok" or attempt >= MAX_SCRIPT_FIXES or not callable(ask):
            return record
        error = str(record.get("error") or "")
        fixed_purpose, fixed, problem = _write_script(ask, FIX_SCRIPT_PROMPT.format(
            purpose=purpose or "(not given)", code=code, error=error[-1500:],
            printed=fence_data(str(record.get("printed") or "")[-1500:] or "(nothing)",
                               "the script's output"),
            samples=_samples(inputs), contract=scratch.SCRIPT_CONTRACT))
        if problem or fixed.strip() == code.strip():
            ctx.note(f"The approved temporary script failed ({_last_line(error)}), and "
                     f"{who} could not write a fix for it"
                     + (f" ({problem})." if problem else " that differs from it."))
            return record
        intro = (f"The script you approved failed ({_last_line(error)}). "
                 f"{who[:1].upper() + who[1:]} has read the error and fixed it.")
        purpose, code = fixed_purpose or purpose, fixed
    return record


def _last_line(error: str, limit: int = 300) -> str:
    """The line of an error that says what went wrong: a traceback's last.

    >>> _last_line('Traceback (most recent call last):\\n  File "x"\\nValueError: bad row\\n')
    'ValueError: bad row'
    """
    lines = [line.strip() for line in str(error or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "no message")[:limit]


def script_fix(ctx: RunContext, tool: Any, error: str, plan: Dict[str, Any],
               ask: Optional[Callable[..., str]], run: Callable[[str], str]) -> Tuple[str, str]:
    """Have the model write a temporary script, and run it if the user says yes.

    The model writes it; :func:`.scratch.check_script` checks it (once more after
    a fix if the first draft breaks a rule); the user reads it and decides; and
    only then does it run, in a process and folder of its own
    (:func:`.scratch.run_script`). When it stands in for an input
    (``plan["replaces"]``), the configuration points at its output and the step
    runs again. Declining is the default and a perfectly good outcome.

    Returns
    -------
    tuple
        ``(status, note)`` as :func:`recover` returns them.
    """
    from . import scratch

    name = getattr(tool, "name", "") or ""
    label = getattr(tool, "label", "") or name or "the step"
    cause = str(plan.get("cause") or "not given").rstrip(". ") + "."
    head = f"{label} failed ({error}). Diagnosis: {cause}"
    inputs = input_paths(ctx)
    replaces = plan.get("replaces") or ""
    prompt = SCRIPT_PROMPT.format(
        tool=name or "the step",
        purpose=getattr(tool, "description", "") or "(not described)",
        error=str(error)[:800], plan=plan.get("purpose") or "",
        replaces=(f"Write the replacement for inputs['{replaces}'] in `out` and return "
                  f"its name as \"replacement\" in the result dict; the step will be "
                  f"run again on it." if replaces else ""),
        samples=_samples(inputs), contract=scratch.SCRIPT_CONTRACT)
    purpose, code, problem = _write_script(ask, prompt)
    if problem:
        return "failed", f"{head} A temporary script could not be written ({problem})."
    purpose = purpose or plan.get("purpose") or ""
    who = str(ctx.settings.get("assistant_name") or "The assistant")
    record = _run_approved(
        ctx, f"{label} failed: {cause} {who} has written a temporary script to get past it.",
        purpose, code, inputs, name, plan["cause"], ask=ask)
    if record is None:
        ctx.note(f"{head} A temporary script to get past it was offered and declined, "
                 "so the step was left undone.")
        return "failed", ""
    folder = Path(record["folder"])
    if record["status"] != "ok":
        return "failed", (f"{head} The approved temporary script ran and failed "
                          f"({_last_line(record['error'])}); it is kept in {folder}.")
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    replacement = str(result.get("replacement") or "")
    if not replacement and len(record["files"]) == 1:
        replacement = record["files"][0]
    target = (folder / replacement).resolve() if replacement else None
    if (replaces and target is not None and target.exists()
            and folder.resolve() in target.parents):
        original = ctx.config.get(replaces)
        ctx.config[replaces] = str(target)
        status = run(name)
        outcome = "which worked" if status == "ok" else "which did not help either"
        return status, (f"{head} A temporary script you approved wrote {target.name} in "
                        f"place of {Path(str(original)).name} ({replaces}), and the step "
                        f"was run again on it, {outcome}. The script is kept in {folder}; "
                        "your original file is unchanged.")
    # Something worked out rather than a file replaced: the next decision reads it.
    found = record.get("found") or ""
    return "failed", (f"{head} A temporary script you approved ran"
                      + (f" and found {found}" if found else "")
                      + f"; it is kept in {folder}. The step itself was not run again.")


def script_task(ctx: RunContext, task: str, ask: Optional[Callable[..., str]],
                why: str = "") -> str:
    """A temporary script the controller asks for on its own initiative.

    For something no step does - a quantity the tools do not compute, a check
    of a product against the data, a figure the goal needs. The model writes
    it, the user reads it with the reason and decides, and only then does it
    run. Returns a sentence for the transcript.
    """
    from . import scratch

    inputs = input_paths(ctx)
    prompt = TASK_SCRIPT_PROMPT.format(transcript=ctx.transcript(limit=12), task=task,
                                       samples=_samples(inputs),
                                       contract=scratch.SCRIPT_CONTRACT)
    purpose, code, problem = _write_script(ask, prompt)
    key = f"script:{task[:40]}"
    if problem:
        ctx.proposals.append({"key": key, "title": f"a temporary script to {task}",
                              "why": why, "accepted": False})
        return f"A temporary script to {task} could not be written ({problem})."
    who = str(ctx.settings.get("assistant_name") or "The assistant")
    record = _run_approved(ctx, f"{who} suggests a temporary script. Why: {why}",
                           purpose or task, code, inputs, "task", why or task, ask=ask)
    ctx.proposals.append({"key": key, "title": f"a temporary script: {purpose or task}",
                          "why": why, "accepted": record is not None})
    if record is None:
        return f"A temporary script ({purpose or task}) was offered and declined."
    if record["status"] != "ok":
        return (f"The approved temporary script failed ({_last_line(record['error'])}); "
                f"it is kept in {record['folder']}.")
    return ("The approved temporary script ran" + (f" and found {record['found']}"
            if record.get("found") else "") + f"; it is kept in {record['folder']}.")
