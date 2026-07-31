"""Command-line interface for versioned workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    context = RunContext(
        project_root=(args.project_root or recipe_dir),
        output_dir=(args.output_dir or recipe_dir / "results"),
    )
    result = run_workflow(spec, context)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status in {"ok", "completed", "success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
