"""Command-line interface for versioned workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .codegen import generate_python
from .models import RunContext
from .recipe import load_recipe
from .registry import list_workflows
from .runner import run_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyhydrogeophysx-workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List registered workflow IDs.")
    validate = subparsers.add_parser("validate", help="Validate a recipe.")
    validate.add_argument("recipe", type=Path)
    run = subparsers.add_parser("run", help="Run a recipe.")
    run.add_argument("recipe", type=Path)
    run.add_argument("--project-root", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--result-file",
        type=Path,
        help="Write the process-safe result JSON here instead of stdout.",
    )
    export = subparsers.add_parser("export-code", help="Generate Python from a recipe.")
    export.add_argument("recipe", type=Path)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for descriptor in list_workflows():
            print(f"{descriptor.workflow_id}\t{descriptor.handler_path}")
        return 0
    spec = load_recipe(args.recipe)
    if args.command == "validate":
        print(f"valid: {spec.workflow_id}")
        return 0
    if args.command == "export-code":
        print(generate_python(spec, args.output))
        return 0
    recipe_dir = args.recipe.resolve().parent
    progress = (
        (lambda message: print(str(message), flush=True))
        if args.result_file is not None
        else None
    )
    context_kwargs = {
        "project_root": (args.project_root or recipe_dir),
        "output_dir": (args.output_dir or recipe_dir / "results"),
    }
    if progress is not None:
        context_kwargs["progress"] = progress
    context = RunContext(
        **context_kwargs,
    )
    # ERT's embedded instrument parsers use pandas but never pyarrow.  Loading
    # the optional Arrow DLL in a QProcess child has produced native access
    # violations on Microsoft Store Python.  Make that unused optional backend
    # look unavailable only for the lifetime of an isolated ERT workflow; this
    # leaves ordinary CLI and Streamlit/Arrow use untouched.
    missing = object()
    previous_pyarrow = missing
    block_pyarrow = (
        args.result_file is not None
        and str(getattr(spec, "workflow_id", "")).startswith("ert.")
    )
    if block_pyarrow:
        previous_pyarrow = sys.modules.get("pyarrow", missing)
        if previous_pyarrow is missing:
            sys.modules["pyarrow"] = None
    try:
        result = run_workflow(spec, context)
    finally:
        if block_pyarrow and previous_pyarrow is missing:
            sys.modules.pop("pyarrow", None)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.result_file is not None:
        destination = args.result_file.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
    else:
        print(payload)
    return 0 if result.status in {"ok", "completed", "success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
