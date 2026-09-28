"""Lightweight, Qt-free I/O helpers for the desktop studio.

The generic numeric-table loaders live in
:mod:`PyHydroGeophysX.data_processing.table_io` and are re-exported here so
existing desktop imports keep
working. Only the JSON helpers used by the Streamlit <-> Qt bridge remain
local: ``write_json`` is atomic (temp file + ``os.replace``) so the bridge
poller never sees a half-written document. The pages' "Use example" buttons
find the source repository's example data through :func:`find_example`, and
say what is missing with :func:`missing_example_message`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from PyHydroGeophysX.data_processing.table_io import (  # noqa: F401
    ensure_dir,
    load_2d_array,
    load_xyz_table,
    read_json,
    write_csv,
    write_json,
)

PathLike = Union[str, Path]

#: Names a clone of the source repository (or its ``examples`` folder), for an
#: installed package, which carries no example data.
EXAMPLES_ENV = "PHGX_EXAMPLES_DIR"
REPOSITORY_URL = "https://github.com/geohang/PyHydroGeophysX"


def example_dirs(project_root: Optional[PathLike] = None) -> List[Path]:
    """Folders that may be a source checkout's ``examples``, most specific first.

    The example data lives in the source repository - ``examples/data``, and
    what ``examples/generate_synthetic_examples.py`` writes under
    ``examples/results`` - and a pip-installed package has none of it. Looked
    for in the Streamlit context's project root, the checkout this package runs
    from, the folder ``PHGX_EXAMPLES_DIR`` names, and the working folder and its
    parents, so a studio started from inside a clone finds the clone's data.
    """
    bases: List[Path] = []
    if project_root:
        bases.append(Path(project_root))
    bases.extend(Path(__file__).resolve().parents)
    configured = os.environ.get(EXAMPLES_ENV, "").strip()
    if configured:
        bases.append(Path(configured).expanduser())
    try:
        cwd = Path.cwd().resolve()
        bases.extend([cwd, *cwd.parents])
    except OSError:
        pass
    found: List[Path] = []
    seen = set()
    for base in bases:
        candidate = base if base.name.lower() == "examples" else base / "examples"
        key = os.path.normcase(str(candidate))
        if key not in seen:
            seen.add(key)
            found.append(candidate)
    return found


def find_example(relative: str, project_root: Optional[PathLike] = None,
                 accept: Optional[Callable[[Path], bool]] = None) -> Optional[Path]:
    """``examples/<relative>`` in the first candidate that has it (or passes ``accept``)."""
    for root in example_dirs(project_root):
        candidate = root / relative
        try:
            ok = accept(candidate) if accept is not None else candidate.exists()
        except OSError:
            ok = False
        if ok:
            return candidate
    return None


def missing_example_message(what: str, relative: str, *, generated: bool = False) -> str:
    """Why an example is not here, and how to get it, for the page to show.

    ``relative`` is the path under ``examples/``; ``generated`` marks data the
    repository does not hold until ``generate_synthetic_examples.py`` has run.
    """
    location = f"examples/{relative}"
    if generated:
        source = (f"{what} is not in this installation. It is generated in a clone of "
                  f"the source repository ({REPOSITORY_URL}) by running "
                  f"'python examples/generate_synthetic_examples.py --quick' there, which "
                  f"writes {location}; the pip package carries no example data.")
    else:
        source = (f"{what} is not in this installation: example data ships with the "
                  f"source repository ({REPOSITORY_URL}, {location}), not with the pip "
                  "package.")
    return (f"{source} To use it, start the studio from inside a clone, or set "
            f"{EXAMPLES_ENV} to the clone's folder and restart the studio; the files "
            "can also be opened from the clone directly.")


__all__ = [
    "EXAMPLES_ENV",
    "PathLike",
    "REPOSITORY_URL",
    "ensure_dir",
    "example_dirs",
    "find_example",
    "load_2d_array",
    "load_xyz_table",
    "missing_example_message",
    "read_json",
    "write_csv",
    "write_json",
]
