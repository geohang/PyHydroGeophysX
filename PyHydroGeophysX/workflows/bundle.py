"""Export a validated recipe together with the code that reproduces it."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .codegen import generate_python
from .models import WorkflowSpec
from .recipe import save_recipe
from .walkthrough import generate_notebook, generate_walkthrough


def export_workflow_bundle(
    spec: WorkflowSpec,
    directory: str | Path,
    *,
    stem: str = "workflow",
    teaching: bool = True,
) -> Tuple[Path, Path]:
    """Write the recipe and the runner, plus a walkthrough script and notebook.

    Four files land in *directory*:

    ``<stem>_recipe.json``
        The run configuration, the single source both generators read from.
    ``run_<stem>.py``
        A runner that calls the workflow engine, for a byte-identical rerun.
    ``<stem>_walkthrough.py`` and ``<stem>_walkthrough.ipynb``
        The same run written as domain-level calls with named parameters and
        prose, for reading, teaching and editing.

    Returns ``(recipe_path, script_path)``. The walkthrough files are written as
    a side effect and a failure to produce them never blocks the reproducible
    pair: a workflow with no walkthrough definition still exports a valid recipe
    and runner.

    Validation occurs before any writer mutates its destination, so a
    non-serializable or seedless stochastic spec cannot produce a runnable-
    looking partial export.
    """
    # ``save_recipe`` and ``generate_python`` both validate, but doing it here
    # first prevents the first file appearing when the second would reject.
    from .registry import get_workflow

    descriptor = get_workflow(spec.workflow_id)
    spec.validate(stochastic=descriptor.stochastic)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    recipe_path = save_recipe(spec, target / f"{stem}_recipe.json")
    script_path = generate_python(spec, target / f"run_{stem}.py")
    if teaching:
        try:
            generate_walkthrough(spec, target / f"{stem}_walkthrough.py")
            generate_notebook(spec, target / f"{stem}_walkthrough.ipynb")
        except NotImplementedError:
            # No walkthrough is defined for this workflow yet. The recipe and
            # the runner are the contract; the teaching copy is a bonus.
            pass
    return recipe_path, script_path


def teaching_paths(script_path: str | Path) -> Tuple[Path | None, Path | None]:
    """Locate the walkthrough pair :func:`export_workflow_bundle` wrote.

    Takes the runner path because that is what a caller already holds, and
    returns ``(walkthrough_py, walkthrough_ipynb)`` with ``None`` for whichever
    is absent. The naming convention lives in this module, so resolving by
    convention here is a lookup rather than a guess.
    """
    runner = Path(script_path)
    if not runner.name.startswith("run_"):
        return None, None
    stem = runner.stem[len("run_"):]
    script = runner.with_name(f"{stem}_walkthrough.py")
    notebook = runner.with_name(f"{stem}_walkthrough.ipynb")
    return (script if script.is_file() else None,
            notebook if notebook.is_file() else None)


__all__ = ["export_workflow_bundle", "teaching_paths"]
