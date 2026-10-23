#!/usr/bin/env python3
"""How well a model reads user requests in the PyHydroGeophysX agent.

The agent asks the model one structured question about every request -
``DELIVERABLES_PROMPT`` in ``PyHydroGeophysX/agents/_intent.py`` - and the
answer decides whether water content is produced, maps or sections, feet or
metres, the site name, and which extraction stages run. The tests check that
contract with fixed replies; this runner measures what real models answer.

It sends each case in ``requests.jsonl`` through the agent's own code:
``read_request`` builds the prompt (``DELIVERABLES_PROMPT`` with the request
substituted) and a ``ContextInputAgent``'s ``query_llm`` sends it, with the
agent's own settings: on the API providers temperature 0.7, 1000 reply tokens
and no system message; on the CLI providers the parser's system message.
Replies are read with the agent's parser
(``parse_json_object``, ``_config_from``, ``_aspects_from``), so the score is
what the agent would do with them.

Usage
-----
Run from the repository root with the project environment's Python (the
``pyhydrogeophysx`` conda environment)::

    # What a run would send, and to whom; sends nothing
    python tools/agent_eval/run_eval.py --provider claude --model claude-sonnet-5 --dry-run

    # Score saved replies offline (no API call)
    python tools/agent_eval/run_eval.py --replies tools/agent_eval/sample_replies.jsonl --out <dir>

    # Live runs: these spend credit on the provider's account
    python tools/agent_eval/run_eval.py --provider openai --model <model id> --out <dir>
    python tools/agent_eval/run_eval.py --provider gemini --repeat 5 --limit 20 --out <dir>
    python tools/agent_eval/run_eval.py --provider openai_compatible --model deepseek-chat \
        --api-key-env DEEPSEEK_API_KEY --out <dir>     # with OPENAI_COMPAT_BASE_URL set
    python tools/agent_eval/run_eval.py --provider codex_cli --out <dir>   # saved CLI login

    # Score a live run again later, offline, from the replies it saved
    python tools/agent_eval/run_eval.py --replies <dir>/replies.jsonl --provider openai \
        --model <model id> --out <dir>/rescored

Options
-------
--provider      openai, gemini, claude, openai_compatible, codex_cli or claude_code.
                With --replies it only labels the report.
--model         Model id; by default the agent's own (``OPENAI_MODEL`` and so on).
--api-key-env   Environment variable holding the key; by default the one the
                agent reads (``OPENAI_API_KEY``, ``GEMINI_API_KEY``,
                ``ANTHROPIC_API_KEY``, ``OPENAI_COMPAT_API_KEY``). The CLI
                providers use their saved login.
--cases FILE    The cases (default: ``requests.jsonl`` beside this script).
--limit N       Only the first N cases.
--repeat N      Ask each case N times, to see how stable the answers are.
--dry-run       Print how many prompts would be sent, to which provider and
                model, and a rough cost; send nothing.
--replies FILE  Score saved replies instead of asking a model. One JSON object
                per line: ``{"id": "<case id>", "reply": "<raw reply text>"}``,
                or ``{"id": ..., "error": "..."}`` for a failed call. Several
                lines for one id are scored as repeats. A live run writes its
                replies in this format, so it can be scored again offline.
--out DIR       Where results go: ``results.json`` (every reply and score),
                ``summary.md``, and for a live run ``replies.jsonl``.

The cases
---------
One JSON object per line::

    {"id": "neg-01", "request": "...", "lang": "en" | "zh" | "mixed",
     "expect": {"water_content": false, "resistivity_only": true,
                "aspects": {"tdem": false}},
     "why": "what the case tests",
     "accept": {"site_location": ["Collins"]},       # optional
     "ambiguous": {"resistivity_only": "reason"}}    # optional

``expect`` uses the prompt's own field names and values and holds only the
fields whose answer the prompt makes unambiguous; ``ambiguous`` records the
fields left out and why. The category is the id without its number
(``neg-01`` -> ``neg``). A site name is right when the agent keeps it and it
contains the expected name (or one of ``accept``), ignoring case and
punctuation.

Scoring
-------
Each expected field of each reply is one of:

correct      the agent reads the same value from the reply as from the expectation
wrong        it reads a different one
missing      the reply has no such field (or an aspect that is not true/false),
             so the agent decides it without the model (keyword rules or default)
unparseable  the reply holds no JSON object; the agent falls back for everything
error        the call failed (live runs and saved errors only)

"The agent reads" is ``_config_from`` and ``_aspects_from`` applied to that one
field, so it is the agent's own reading: ``"feet"`` counts as feet, a
``length_unit`` of null and of ``"m"`` both mean metres, a ``water_content``
of ``"no"`` is ignored like null, a bare number pair given as the site is
dropped. Such answers still appear in the off-contract list. The extra row
``convert_to_water_content`` is
the agent's combined decision from ``water_content`` and ``resistivity_only``
- the one that decides whether water content is produced.

The confusion tables count expected value against the agent's reading, except
``length_unit``, which counts the raw answer (ft / m / null / other) because
the agent reads null and "m" alike. Answers outside the prompt's contract
(a string for a boolean, coordinates for a site, a missing aspect) are listed
separately for every field of every reply, whether or not the case expects it.
A case is wrong if any field is, missing if none is wrong and one is missing.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# The working tree's prompt and parser, not whichever copy is installed.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PyHydroGeophysX.agents._intent import (  # noqa: E402
    DELIVERABLES_PROMPT,
    ROUTABLE_ASPECTS,
    _aspects_from,
    _config_from,
    _parse_json_object,
    read_request,
)

PROVIDERS = ("openai", "gemini", "claude", "openai_compatible", "codex_cli", "claude_code")
#: The variable each API provider's key is read from, as ``BaseAgent`` reads it.
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
           "claude": "ANTHROPIC_API_KEY", "openai_compatible": "OPENAI_COMPAT_API_KEY"}
CLI_PROVIDERS = ("codex_cli", "claude_code")

TOP_FIELDS = ("water_content", "resistivity_only", "spatial_map", "length_unit",
              "site_location")
ASPECT_FIELDS = tuple(f"aspects.{name}" for name in ROUTABLE_ASPECTS)
FIELDS = TOP_FIELDS + ASPECT_FIELDS
DECISION = "convert_to_water_content"
OUTCOMES = ("correct", "wrong", "missing", "unparseable", "error")
#: Rough reply length for the dry run's cost estimate.
REPLY_TOKENS_GUESS = 120

_MISSING = object()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def build_prompt(request: str) -> str:
    """The prompt the agent sends for ``request``: ``read_request``'s substitution."""
    return DELIVERABLES_PROMPT.replace("{request}", str(request))


def prompt_the_agent_sends(request: str) -> str:
    """The prompt ``read_request`` itself builds, caught before anything is sent."""
    sent: List[str] = []

    def capture(prompt: str) -> str:
        sent.append(prompt)
        return ""

    read_request(request, capture)
    if not sent:
        raise SystemExit(f"read_request sent no prompt for {request!r}")
    return sent[0]


def _check_value(field: str, value: Any) -> bool:
    if field == "water_content":
        return value is None or isinstance(value, bool)
    if field in ("resistivity_only", "spatial_map") or field.startswith("aspects."):
        return isinstance(value, bool)
    if field == "length_unit":
        return value in ("ft", "m", None)
    if field == "site_location":
        return value is None or (isinstance(value, str) and bool(value.strip()))
    return False


def flatten(expect: Dict[str, Any]) -> Dict[str, Any]:
    """``{"aspects": {"gpr": true}}`` -> ``{"aspects.gpr": true}``."""
    flat = {key: value for key, value in expect.items() if key != "aspects"}
    for name, value in (expect.get("aspects") or {}).items():
        flat[f"aspects.{name}"] = value
    return flat


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """The cases in ``path``, checked; exits with the line number of a bad one."""
    cases, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            case = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{where}: not JSON ({exc})")
        for key in ("id", "request", "lang", "expect", "why"):
            if key not in case:
                raise SystemExit(f"{where}: no {key!r}")
        if case["id"] in seen:
            raise SystemExit(f"{where}: id {case['id']} appears twice")
        seen.add(case["id"])
        if not isinstance(case["expect"].get("aspects", {}), dict):
            raise SystemExit(f"{where}: expect.aspects must be an object")
        flat = flatten(case["expect"])
        for field, value in flat.items():
            if field not in FIELDS:
                raise SystemExit(f"{where}: unknown field {field!r}")
            if not _check_value(field, value):
                raise SystemExit(f"{where}: {field} = {value!r} is not a value the prompt allows")
        for field in case.get("ambiguous", {}):
            if field not in FIELDS or field in flat:
                raise SystemExit(f"{where}: ambiguous field {field!r} is unknown or also expected")
        case["category"] = case["id"].rsplit("-", 1)[0]
        case["flat"] = flat
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Reading a reply as the agent does
# ---------------------------------------------------------------------------

def raw_value(answer: Dict[str, Any], field: str) -> Any:
    """The value the reply gives ``field``, or ``_MISSING``."""
    if field.startswith("aspects."):
        aspects = answer.get("aspects")
        name = field.split(".", 1)[1]
        if not isinstance(aspects, dict) or name not in aspects:
            return _MISSING
        return aspects[name]
    return answer.get(field, _MISSING)


def agent_value(answer: Dict[str, Any], field: str) -> Any:
    """``field`` as the agent reads it, through its own ``_config_from``/``_aspects_from``.

    The field is read alone, so ``resistivity_only`` does not mask
    ``water_content``; :data:`DECISION` is their combination.
    """
    if field.startswith("aspects."):
        return _aspects_from(answer)[field.split(".", 1)[1]]
    one = _config_from({field: answer[field]})
    if field == "water_content":
        return one.get("convert_to_water_content")
    if field == "resistivity_only":
        return one.get("convert_to_water_content") is False
    if field == "spatial_map":
        return one["spatial_maps"]
    if field == "length_unit":
        return one["length_unit"]
    return one.get("site_location")


def expected_answer(field: str, value: Any) -> Dict[str, Any]:
    """The expectation as a reply holding only that field."""
    if field.startswith("aspects."):
        return {"aspects": {field.split(".", 1)[1]: value}}
    return {field: value}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(text).casefold()).split())


def looks_like_coordinates(text: str) -> bool:
    """Two decimal numbers or a degree sign: a remembered position, not a name."""
    return len(re.findall(r"[-+]?\d{1,3}\.\d+", text)) >= 2 or "°" in text


def expected_decision(flat: Dict[str, Any]) -> Any:
    """What the agent should decide about water content, or ``_MISSING`` when open.

    When ``resistivity_only`` is left out as ambiguous, both of its values are
    tried; the decision is scored only if they agree.
    """
    if "water_content" not in flat:
        return _MISSING
    options = ([flat["resistivity_only"]] if "resistivity_only" in flat else [True, False])
    decisions = {_config_from({"water_content": flat["water_content"],
                               "resistivity_only": ro}).get("convert_to_water_content")
                 for ro in options}
    return decisions.pop() if len(decisions) == 1 else _MISSING


def label(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def length_label(raw: Any) -> str:
    """The raw ``length_unit`` for the confusion table: ft, m, null or other."""
    if raw is _MISSING:
        return "missing"
    if raw is None:
        return "null"
    if raw in ("ft", "m"):
        return raw
    return "other"


def off_contract(answer: Dict[str, Any]) -> List[str]:
    """Every answer outside the prompt's contract, for every field of the reply."""
    notes = []
    for field in TOP_FIELDS:
        if field not in answer:
            notes.append(f"{field}: absent")
            continue
        value = answer[field]
        if field == "site_location" and isinstance(value, str) and looks_like_coordinates(value):
            kept = "site_location" in _config_from({"site_location": value})
            notes.append(f"site_location: {value!r} looks like coordinates "
                         f"({'the agent keeps it as a name to look up' if kept else 'the agent drops it'})")
        elif not _check_value(field, value):
            notes.append(f"{field}: {json.dumps(value, ensure_ascii=False)} is not a value the prompt allows")
    aspects = answer.get("aspects")
    if not isinstance(aspects, dict):
        notes.append("aspects: absent or not an object")
    else:
        for name in ROUTABLE_ASPECTS:
            if name not in aspects:
                notes.append(f"aspects.{name}: absent")
            elif not isinstance(aspects[name], bool):
                notes.append(f"aspects.{name}: {json.dumps(aspects[name], ensure_ascii=False)} "
                             "is not true/false")
    return notes


def score_reply(case: Dict[str, Any], reply: Optional[str], error: Optional[str]) -> Dict[str, Any]:
    """One reply to one case, scored field by field."""
    run: Dict[str, Any] = {"reply": reply, "error": error, "fields": {}, "off_contract": []}
    flat = case["flat"]
    scored = list(flat)
    decision_expected = expected_decision(flat)
    if decision_expected is not _MISSING:
        scored.append(DECISION)

    answer = None
    if error is None:
        # The agent's own path: read_request with the reply in place of a model.
        def query(prompt: str) -> str:
            if prompt != build_prompt(case["request"]):
                raise SystemExit(f"{case['id']}: read_request built a different prompt")
            return reply

        config, _aspects = read_request(case["request"], query)
        parsed = _parse_json_object(reply)
        answer = parsed if isinstance(parsed, dict) else None
        run["agent_config"] = config
        run["agent_aspects"] = _aspects
        if answer is not None:
            run["off_contract"] = off_contract(answer)

    for field in scored:
        entry: Dict[str, Any] = {}
        if field == DECISION:
            entry["expected"] = decision_expected
        else:
            entry["expected"] = flat[field]
        if error is not None:
            entry["outcome"] = "error"
            run["fields"][field] = entry
            continue
        if answer is None:
            entry["outcome"] = "unparseable"
            run["fields"][field] = entry
            continue

        if field == DECISION:
            got = run["agent_config"].get("convert_to_water_content")
            entry["read"] = got
            if "water_content" not in answer and "resistivity_only" not in answer:
                entry["outcome"] = "missing"
            else:
                entry["outcome"] = "correct" if got == decision_expected else "wrong"
            run["fields"][field] = entry
            continue

        raw = raw_value(answer, field)
        entry["raw"] = None if raw is _MISSING else raw
        entry["raw_label"] = length_label(raw) if field == "length_unit" else label(raw)
        if raw is _MISSING:
            entry["outcome"] = "missing"
            run["fields"][field] = entry
            continue
        got = agent_value(answer, field)
        entry["read"] = got
        if field.startswith("aspects.") and got is None:
            # Not true/false: the agent treats the model as silent here.
            entry["outcome"] = "missing"
        elif field == "site_location":
            want = flat[field]
            if want is None:
                ok = got is None
            else:
                accepted = case.get("accept", {}).get("site_location") or [want]
                ok = isinstance(got, str) and any(_norm(alt) in _norm(got) for alt in accepted)
            entry["outcome"] = "correct" if ok else "wrong"
        else:
            want = agent_value(expected_answer(field, flat[field]), field)
            entry["outcome"] = "correct" if got == want else "wrong"
        run["fields"][field] = entry

    outcomes = {entry["outcome"] for entry in run["fields"].values()}
    if error is not None:
        run["outcome"] = "error"
    elif answer is None:
        run["outcome"] = "unparseable"
    elif "wrong" in outcomes:
        run["outcome"] = "wrong"
    elif "missing" in outcomes:
        run["outcome"] = "missing"
    else:
        run["outcome"] = "correct"
    return run


# ---------------------------------------------------------------------------
# Asking a model
# ---------------------------------------------------------------------------

def make_agent(provider: str, model: Optional[str], api_key_env: Optional[str]):
    """``(agent, key_env, key_found)``: a ``ContextInputAgent`` as the AQUAH workflow builds one.

    Without ``api_key_env`` the agent reads its provider's own variable. With
    it, that variable alone: an unset one is reported rather than letting the
    agent fall back to another account's key.
    """
    from PyHydroGeophysX.agents.context_input_agent import ContextInputAgent

    if provider in CLI_PROVIDERS:
        agent = ContextInputAgent(api_key=None, model=model, llm_provider=provider)
        return agent, None, True
    env = api_key_env or KEY_ENV[provider]
    # Agents take a key only when given one (BaseAgent's KEY_FROM_ENVIRONMENT).
    key = os.environ.get(env) or None
    agent = ContextInputAgent(api_key=key, model=model, llm_provider=provider)
    return agent, env, bool(key)


def ask(agent, request: str) -> Tuple[Optional[str], Optional[str], str, float]:
    """``(reply, error, prompt, seconds)`` for one request, sent through ``read_request``."""
    record: Dict[str, Any] = {}

    def query(prompt: str) -> str:
        record["prompt"] = prompt
        try:
            record["reply"] = agent.query_llm(prompt)
        except Exception as exc:  # noqa: BLE001 - recorded as this case's error
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        return record["reply"]

    start = time.time()
    read_request(request, query)
    return (record.get("reply"), record.get("error"), record.get("prompt", ""),
            time.time() - start)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "-"


def summarise(cases: List[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Counts for ``results.json`` and the Markdown summary."""
    scored_cases = [case for case in cases if case["runs"]]
    runs = [run for case in scored_cases for run in case["runs"]]

    case_outcomes = collections.Counter(run["outcome"] for run in runs)
    per_field: Dict[str, collections.Counter] = {}
    for field in FIELDS + (DECISION,):
        counter = collections.Counter()
        for run in runs:
            if field in run["fields"]:
                counter[run["fields"][field]["outcome"]] += 1
        if counter:
            per_field[field] = counter

    per_category: Dict[str, collections.Counter] = collections.OrderedDict()
    for case in scored_cases:
        counter = per_category.setdefault(case["category"], collections.Counter())
        counter["cases"] += 1
        for run in case["runs"]:
            counter["runs"] += 1
            counter[run["outcome"]] += 1

    confusion: Dict[str, Dict[str, collections.Counter]] = {}
    for field in FIELDS + (DECISION,):
        table: Dict[str, collections.Counter] = {}
        for run in runs:
            entry = run["fields"].get(field)
            if entry is None:
                continue
            row = label(entry["expected"])
            if entry["outcome"] in ("unparseable", "error", "missing"):
                col = entry["outcome"]
            elif field == "length_unit":
                col = entry["raw_label"]
            else:
                col = label(entry["read"])
            table.setdefault(row, collections.Counter())[col] += 1
        if table and field != "site_location":
            confusion[field] = table

    off = [(case["id"], i, note) for case in scored_cases
           for i, run in enumerate(case["runs"]) for note in run["off_contract"]]

    unstable = []
    for case in scored_cases:
        if len(case["runs"]) < 2:
            continue
        for field in case["runs"][0]["fields"]:
            seen = []
            for run in case["runs"]:
                entry = run["fields"][field]
                value = (entry["outcome"] if entry["outcome"] in ("unparseable", "error", "missing")
                         else label(entry.get("read")))
                seen.append(value)
            if len(set(seen)) > 1:
                unstable.append((case["id"], field, seen))

    problems = []
    for case in scored_cases:
        for i, run in enumerate(case["runs"]):
            for field, entry in run["fields"].items():
                if entry["outcome"] in ("wrong", "missing"):
                    problems.append((case["id"], i, field, entry))

    ambiguous = collections.Counter(field for case in cases for field in case.get("ambiguous", {}))

    summary = {
        "cases_in_file": meta["cases_in_file"],
        "cases_scored": len(scored_cases),
        "cases_without_reply": [case["id"] for case in cases if not case["runs"]],
        "runs": len(runs),
        "case_outcomes": dict(case_outcomes),
        "per_field": {field: dict(counter) for field, counter in per_field.items()},
        "per_category": {cat: dict(counter) for cat, counter in per_category.items()},
        "confusion": {field: {row: dict(cols) for row, cols in table.items()}
                      for field, table in confusion.items()},
        "off_contract": [{"id": cid, "run": i, "note": note} for cid, i, note in off],
        "unstable": [{"id": cid, "field": field, "answers": seen} for cid, field, seen in unstable],
        "left_out_as_ambiguous": dict(ambiguous),
    }

    lines = ["# Request-reading evaluation", ""]
    source = (f"replies file `{meta['replies']}` (offline)" if meta.get("replies")
              else "live model calls")
    lines += [f"- Source: {source}",
              f"- Provider / model: {meta.get('provider') or '-'} / {meta.get('model') or '-'}",
              f"- Prompt: `DELIVERABLES_PROMPT`, sha256 `{meta['prompt_sha256'][:12]}`",
              f"- Cases: {len(scored_cases)} scored of {meta['cases_in_file']} in the file"
              + (f" ({len(summary['cases_without_reply'])} without a reply)"
                 if summary["cases_without_reply"] else "")
              + f"; {len(runs)} replies",
              "- Replies: " + ", ".join(f"{name} {case_outcomes.get(name, 0)}" for name in OUTCOMES
                                         if case_outcomes.get(name) or name != "error")]
    if meta.get("usage"):
        usage = meta["usage"]
        lines.append(f"- Tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out; "
                     f"estimated cost ${usage['cost_estimate_usd']:.4f}")
    lines += ["", "## Per field", "",
              "| field | scored | correct | wrong | missing | unparseable | error | accuracy |",
              "|---|---|---|---|---|---|---|---|"]
    for field, counter in per_field.items():
        total = sum(counter.values())
        name = f"{field} (agent decision)" if field == DECISION else field
        lines.append(f"| {name} | {total} | " + " | ".join(str(counter.get(o, 0)) for o in OUTCOMES)
                     + f" | {_pct(counter.get('correct', 0), total)} |")
    lines += ["", "## Per category (whole replies)", "",
              "| category | cases | replies | correct | wrong | missing | unparseable | error |",
              "|---|---|---|---|---|---|---|---|"]
    for category, counter in per_category.items():
        lines.append(f"| {category} | {counter['cases']} | {counter['runs']} | "
                     + " | ".join(str(counter.get(o, 0)) for o in OUTCOMES) + " |")

    lines += ["", "## Confusion (rows: expected; columns: what the agent read)", ""]
    for field, table in confusion.items():
        columns = sorted({col for cols in table.values() for col in cols},
                         key=lambda c: (["true", "false", "null", "ft", "m", "other", "missing",
                                         "unparseable", "error"] + [c]).index(c))
        note = " (raw answer)" if field == "length_unit" else ""
        lines.append(f"**{field}**{note}: " + "; ".join(
            f"expected {row} -> " + ", ".join(f"{col} {table[row][col]}" for col in columns
                                              if table[row].get(col))
            for row in sorted(table)))
        lines.append("")

    lines += ["## Wrong and missing answers", ""]
    if problems:
        lines += ["| case | reply | field | expected | answered | agent read | outcome |",
                  "|---|---|---|---|---|---|---|"]
        for cid, i, field, entry in problems:
            answered = ("-" if field == DECISION
                        else json.dumps(entry.get("raw"), ensure_ascii=False)
                        if entry.get("raw_label") != "missing" else "(absent)")
            read = label(entry["read"]) if "read" in entry else "-"
            lines.append(f"| {cid} | {i + 1} | {field} | {label(entry['expected'])} | {answered} | "
                         f"{read} | {entry['outcome']} |")
    else:
        lines.append("None.")
    lines += ["", "## Answers outside the prompt's contract", ""]
    lines += [f"- {cid} (reply {i + 1}): {note}" for cid, i, note in off] or ["None."]
    lines += ["", "## Stability (cases asked more than once)", ""]
    if any(len(case["runs"]) > 1 for case in scored_cases):
        lines += [f"- {cid} {field}: {' / '.join(seen)}" for cid, field, seen in unstable] or [
            "Every repeated case got the same reading every time."]
    else:
        lines.append("Each case was asked once (use --repeat N).")
    lines += ["", "## Fields left out of the expectations as ambiguous", "",
              ", ".join(f"{field} {count}" for field, count in sorted(ambiguous.items()))
              + " (reasons in requests.jsonl and results.json).", ""]
    return summary, "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def read_replies(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    replies: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{path.name}:{number}: not JSON ({exc})")
        if "id" not in item or ("reply" not in item and "error" not in item):
            raise SystemExit(f"{path.name}:{number}: needs an id and a reply or an error")
        replies[item["id"]].append(item)
    return replies


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - not a reconfigurable stream
            pass
    parser = argparse.ArgumentParser(
        description="Score how a model reads requests with the agent's DELIVERABLES_PROMPT.")
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--cases", type=Path, default=HERE / "requests.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replies", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.dry_run and args.replies:
        parser.error("--dry-run and --replies do not go together: --replies sends nothing")
    if not args.replies and not args.provider:
        parser.error("--provider is needed unless --replies is given")
    if not args.dry_run and not args.out:
        parser.error("--out is needed")

    all_cases = load_cases(args.cases)
    cases = all_cases[:args.limit] if args.limit else all_cases
    prompt_sha = hashlib.sha256(DELIVERABLES_PROMPT.encode("utf-8")).hexdigest()
    meta: Dict[str, Any] = {
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases_file": str(args.cases), "cases_in_file": len(all_cases),
        "limit": args.limit, "repeat": args.repeat, "prompt_sha256": prompt_sha,
        "provider": args.provider, "model": args.model,
        "replies": str(args.replies) if args.replies else None,
    }

    if args.dry_run:
        for case in cases:
            if prompt_the_agent_sends(case["request"]) != build_prompt(case["request"]):
                raise SystemExit(f"{case['id']}: read_request builds a different prompt")
        agent, env, found = make_agent(args.provider, args.model, args.api_key_env)
        prompts = [build_prompt(case["request"]) for case in cases]
        from PyHydroGeophysX.agents._pricing import estimate_llm_cost_usd, estimate_tokens

        tokens_in = sum(estimate_tokens(p) for p in prompts) * args.repeat
        tokens_out = REPLY_TOKENS_GUESS * len(prompts) * args.repeat
        cost = estimate_llm_cost_usd(args.provider, agent.model, tokens_in, tokens_out)
        if args.provider in CLI_PROVIDERS:
            access = "saved CLI login"
        else:
            source = env if args.api_key_env else f"{env} (the agent's own)"
            access = f"key from {source}: {'set' if found else 'NOT SET'}"
        print("Dry run: nothing is sent.")
        print(f"Would send {len(prompts) * args.repeat} prompts "
              f"({len(prompts)} cases x {args.repeat} repeat{'s' if args.repeat > 1 else ''}) "
              f"to provider {args.provider}, model {agent.model} ({access}).")
        if args.provider == "openai_compatible":
            base = os.environ.get("OPENAI_COMPAT_BASE_URL", "").strip()
            print(f"Endpoint: {base or 'OPENAI_COMPAT_BASE_URL NOT SET'}")
        print(f"Prompt: DELIVERABLES_PROMPT (sha256 {prompt_sha[:12]}), "
              f"{min(map(len, prompts))}-{max(map(len, prompts))} characters each; "
              "each matches the prompt read_request builds.")
        print(f"About {tokens_in} prompt and {tokens_out} reply tokens; "
              + ("counted against the CLI login's plan." if args.provider in CLI_PROVIDERS
                 else f"estimated cost ${cost:.4f} (agents._pricing)."))
        print(f"Cases from {args.cases} ({len(all_cases)} in the file"
              + (f", first {len(cases)}" if args.limit else "") + ").")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    if args.replies:
        saved = read_replies(args.replies)
        unknown = sorted(set(saved) - {case["id"] for case in all_cases})
        if unknown:
            print(f"Replies for unknown case ids ignored: {', '.join(unknown)}", file=sys.stderr)
        if args.repeat > 1:
            print("--repeat is ignored with --replies: lines with the same id are the repeats.",
                  file=sys.stderr)
        for case in cases:
            case["runs"] = [score_reply(case, item.get("reply"), item.get("error"))
                            for item in saved.get(case["id"], [])]
    else:
        agent, env, found = make_agent(args.provider, args.model, args.api_key_env)
        meta["model"] = agent.model
        if not found:
            raise SystemExit(f"No key for {args.provider}: {env} is not set.")
        replies_path = args.out / "replies.jsonl"
        with replies_path.open("w", encoding="utf-8") as handle:
            for number, case in enumerate(cases, 1):
                case["runs"] = []
                for repeat in range(args.repeat):
                    reply, error, prompt, seconds = ask(agent, case["request"])
                    if prompt != build_prompt(case["request"]):
                        raise SystemExit(f"{case['id']}: read_request built a different prompt")
                    handle.write(json.dumps({"id": case["id"], "repeat": repeat, "reply": reply,
                                             "error": error, "seconds": round(seconds, 2)},
                                            ensure_ascii=False) + "\n")
                    handle.flush()
                    run = score_reply(case, reply, error)
                    run["seconds"] = round(seconds, 2)
                    case["runs"].append(run)
                    print(f"[{number}/{len(cases)}] {case['id']} #{repeat + 1}: {run['outcome']}"
                          f" ({seconds:.1f} s)")
        ledger = getattr(agent, "llm_usage_ledger", [])
        meta["usage"] = {
            "calls": len(ledger),
            "prompt_tokens": sum(r["prompt_tokens"] for r in ledger),
            "completion_tokens": sum(r["completion_tokens"] for r in ledger),
            "cost_estimate_usd": sum(r["cost_estimate_usd"] or 0.0 for r in ledger),
        }

    for case in all_cases:
        case.setdefault("runs", [])
    summary, markdown = summarise(cases, meta)
    results = {
        "meta": meta,
        "summary": summary,
        "cases": [{key: case[key] for key in ("id", "category", "lang", "request", "expect", "why",
                                              "runs")}
                  | {key: case[key] for key in ("accept", "ambiguous") if key in case}
                  for case in cases],
    }
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=lambda o: None), encoding="utf-8")
    (args.out / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {args.out / 'results.json'} and {args.out / 'summary.md'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
