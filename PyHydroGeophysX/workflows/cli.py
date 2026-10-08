"""Command-line interface for versioned workflows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any, Sequence
import warnings

from .codegen import generate_python
from .models import RunContext
from .objects import is_plain_data, objects_folder, save_objects
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
    run.add_argument(
        "--objects",
        default="",
        help="With --result-file: the result's live objects (meshes, arrays, "
             "tables), by name and comma-separated, to write beside it for the "
             "caller to read back; '*' for all of them.",
    )
    export = subparsers.add_parser("export-code", help="Generate Python from a recipe.")
    export.add_argument("recipe", type=Path)
    export.add_argument("--output", type=Path, required=True)
    return parser


def _quiet_pygimli_show() -> None:
    """Silence the one warning every workflow that draws with pyGIMLi printed.

    pyGIMLi's first figure calls ``plt.show()`` once, to register a viewer for
    the end of the script. A workflow draws on the non-interactive Agg backend,
    so matplotlib answers with "FigureCanvasAgg is non-interactive, and thus
    cannot be shown", and the warning machinery echoes the offending source
    line, ``plt.show()``, on a line of its own. In the studio's log the pair
    read as something having gone wrong, just as the inversion finished. Only
    that message is filtered; every other warning still reaches the log.
    """
    warnings.filterwarnings(
        "ignore", message="FigureCanvasAgg is non-interactive", category=UserWarning)


#: Set to 0 to let a run with ``--result-file`` outlive the program that started it.
EXIT_WITH_PARENT_ENV = "PHGX_EXIT_WITH_PARENT"


def _is_launcher(parent: Any, me: Any) -> bool:
    """Whether ``parent`` is a Windows virtual environment's ``python.exe``.

    That one is only a launcher: it starts the environment's real interpreter
    as its child with the same arguments, and waits for it.
    """
    try:
        return (os.path.basename(parent.exe()).lower().startswith("python")
                and parent.cmdline()[1:] == me.cmdline()[1:])
    except Exception:  # noqa: BLE001 - psutil.Error, or a process already gone
        return False


def _exit_with_parent() -> bool:
    """End this run, and what it started, once the program that started it is gone.

    The studio runs every workflow in a process of its own and reads the
    result when it ends. A studio that crashed or was ended from the Task
    Manager left that process computing - an inversion holding the GPU and
    gigabytes of memory, for as long as it took, with no window left to show
    it or stop it - because Windows does not end a process's children with it.
    So the run watches the program that started it (past a virtual
    environment's launcher, see :func:`_is_launcher`) on a thread of its own,
    and when that program ends, ends its own children - an E4D or R2 run - and
    then itself.

    Returns whether a watch was started: never without psutil, which the
    desktop studio installs.
    """
    flag = str(os.environ.get(EXIT_WITH_PARENT_ENV, "1")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    try:
        import psutil

        me = psutil.Process()
        parent = me.parent()
        if parent is not None and _is_launcher(parent, me):
            parent = parent.parent()
        if parent is None or not parent.is_running():
            return False
    except Exception:  # noqa: BLE001 - no psutil, or no parent to watch
        return False

    def watch() -> None:
        try:
            parent.wait()
        except Exception:  # noqa: BLE001 - cannot tell that it ended, so carry on
            return
        try:
            for child in me.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    pass
        finally:
            # Not sys.exit: that only ends this thread. Nobody is left to read
            # a result, so there is nothing to finish writing either.
            os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()
    return True


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
    _quiet_pygimli_show()
    if args.result_file is not None:
        # Run for a program that waits for the result file - the studio, the
        # MCP server - so it is pointless, and costly, once that program is gone.
        _exit_with_parent()
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
    if args.result_file is not None:
        # Otherwise the run's own last progress line stays on the studio's
        # progress and status bars while the result is written here and read
        # back there, which on a large result takes a while.
        print("[progress 1/1] Handing the results back", flush=True)
        destination = args.result_file.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = result.to_dict()
        folder = objects_folder(destination)
        shutil.rmtree(folder, ignore_errors=True)     # a previous run's
        wanted = [name.strip() for name in str(args.objects or "").split(",") if name.strip()]
        # The caller reads these back (see workflows.objects), so a page gets
        # what a run in a thread gave it. Heavy objects only when asked for - a
        # mesh it does not show would cost it a load for nothing - and plain
        # data always: it is small, and a value with a NaN in it is one.
        chosen = {name: value for name, value in result.objects.items()
                  if "*" in wanted or name in wanted or is_plain_data(value)}
        if chosen or wanted:
            document["objects"] = save_objects(chosen, folder)
            document["objects"]["skipped"].update({
                name: "the workflow returned no object of this name"
                for name in wanted if name != "*" and name not in result.objects})
        payload = json.dumps(document, indent=2, sort_keys=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
    else:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status in {"ok", "completed", "success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
