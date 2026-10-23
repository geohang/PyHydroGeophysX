"""Does AQUAH's stated confidence mean anything? Score it against judged decisions.

Every decision in a flexible run states a confidence from 0 to 1, kept on the
step it led to: ``execution_plan[i]["confidence"]`` in the run's
``workflow_result.json``. The controller acts on it in one place only - leaving
route steps out below ``LOW_CONFIDENCE`` is put to the user - so the question is
whether a low number marks the decisions that are actually wrong.

Two commands; neither calls a model::

    # 1. a sheet of every decided step from finished runs, to judge
    python tools/agent_eval/decision_confidence.py sheet RUN_DIR [RUN_DIR ...] --out decisions.csv

    # 2. fill the "right" column (1: an expert would have taken this step here,
    #    0: not; leave it empty to skip the row), then
    python tools/agent_eval/decision_confidence.py score decisions.csv

The score gives the area under the ROC curve of confidence against the
judgement (0.5: the number says nothing; 1: every right decision was more
confident than every wrong one), the Brier score, a table by confidence band,
and what the ``LOW_CONFIDENCE`` threshold would have caught: how many decisions
below it were wrong, and how many wrong ones it missed. Checks of this kind -
discrimination and calibration of an agent's stated confidence - are the
uncertainty-quantification skill's in DOE's Genesis Agent Skills catalogue
(ModCon, 2026, https://github.com/AI-ModCon/genesis-skills); this script is
written independently.

A few dozen judged decisions are the least that says anything; the script
prints the count beside every number.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Kept in step with PyHydroGeophysX.agents.runtime.controller.LOW_CONFIDENCE;
#: read from there when the package imports.
DEFAULT_LOW = 0.5
BANDS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001))
COLUMNS = ["run", "index", "step", "status", "confidence", "reason", "right"]


def low_confidence() -> float:
    try:
        from PyHydroGeophysX.agents.runtime.controller import LOW_CONFIDENCE
        return float(LOW_CONFIDENCE)
    except Exception:  # noqa: BLE001 - the script works without the package
        return DEFAULT_LOW


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Area under the ROC curve: the chance a right decision outranks a wrong one.

    Ties count a half (Mann-Whitney). None without both kinds of decision.

    >>> auroc([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0])
    1.0
    >>> auroc([0.5, 0.5], [1, 0])
    0.5
    >>> auroc([0.9], [1]) is None
    True
    """
    right = [s for s, y in zip(scores, labels) if y]
    wrong = [s for s, y in zip(scores, labels) if not y]
    if not right or not wrong:
        return None
    wins = sum(1.0 if r > w else 0.5 if r == w else 0.0 for r in right for w in wrong)
    return wins / (len(right) * len(wrong))


def brier(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Mean squared gap between confidence and outcome (0 is perfect).

    >>> brier([1.0, 0.0], [1, 0])
    0.0
    """
    if not scores:
        return None
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def decided_steps(run_dir: Path) -> Iterable[Dict[str, object]]:
    """The steps of one run that a model decided, with their confidence."""
    path = run_dir / "workflow_result.json" if run_dir.is_dir() else run_dir
    plan = json.loads(path.read_text(encoding="utf-8")).get("execution_plan") or []
    for index, step in enumerate(plan):
        if isinstance(step, dict) and step.get("confidence") is not None:
            yield {"run": str(run_dir), "index": index, "step": step.get("step", ""),
                   "status": step.get("status", ""), "confidence": step["confidence"],
                   "reason": str(step.get("reason", ""))[:400], "right": ""}


def write_sheet(run_dirs: Sequence[str], out: Path) -> int:
    rows = [row for d in run_dirs for row in decided_steps(Path(d))]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_judged(sheet: Path) -> Tuple[List[float], List[int]]:
    scores, labels = [], []
    with sheet.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            judged = str(row.get("right", "")).strip()
            if judged not in ("0", "1"):
                continue
            scores.append(float(row["confidence"]))
            labels.append(int(judged))
    return scores, labels


def report(scores: Sequence[float], labels: Sequence[int], low: float) -> str:
    n, wrong = len(labels), labels.count(0)
    area, error = auroc(scores, labels), brier(scores, labels)
    lines = [f"Judged decisions: {n} ({n - wrong} right, {wrong} wrong)",
             "AUROC: " + (f"{area:.3f}" if area is not None else "n/a (needs right and wrong)"),
             "Brier: " + (f"{error:.3f}" if error is not None else "n/a"),
             "", "| confidence | decisions | right | mean confidence |", "|---|---|---|---|"]
    for lo, hi in BANDS:
        band = [(s, y) for s, y in zip(scores, labels) if lo <= s < hi]
        if band:
            share = sum(y for _, y in band) / len(band)
            mean = sum(s for s, _ in band) / len(band)
            lines.append(f"| {lo:.1f}-{min(hi, 1.0):.1f} | {len(band)} | {share:.0%} | {mean:.2f} |")
    below = [y for s, y in zip(scores, labels) if s < low]
    caught = below.count(0)
    lines += ["", f"Below LOW_CONFIDENCE ({low}): {len(below)} decisions, {caught} of them wrong; "
                  f"it misses {wrong - caught} of the {wrong} wrong decisions."]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sheet = sub.add_parser("sheet", help="write a sheet of decided steps to judge")
    sheet.add_argument("runs", nargs="+", help="run folders (or workflow_result.json files)")
    sheet.add_argument("--out", default="decisions.csv")
    score = sub.add_parser("score", help="score a judged sheet")
    score.add_argument("sheet")
    score.add_argument("--low", type=float, default=None,
                       help="threshold to report on (default: the controller's)")
    args = parser.parse_args(argv)
    if args.command == "sheet":
        count = write_sheet(args.runs, Path(args.out))
        print(f"Wrote {count} decided steps to {args.out}; fill the 'right' column.")
        return 0
    scores, labels = read_judged(Path(args.sheet))
    print(report(scores, labels, low_confidence() if args.low is None else args.low))
    return 0


if __name__ == "__main__":
    sys.exit(main())
