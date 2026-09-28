"""Launcher for the 3D Mesh Builder Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#: The app, relative to a source checkout.
APP_RELATIVE_PATH = Path("examples") / "app_mesh3d.py"

#: What a pip install is told. The app lives in examples/, which the wheel does
#: not include, so an installed launcher found nothing to run and ended in a
#: FileNotFoundError traceback.
NOT_BUNDLED_MESSAGE = (
    "The PyHydroGeophysX 3D Mesh Builder app ships with the source repository, "
    "not with the pip package, and no copy was found beside this installation or "
    "in the current folder.\n"
    "To run it locally:\n"
    "  git clone https://github.com/geohang/PyHydroGeophysX.git\n"
    "  cd PyHydroGeophysX\n"
    "  streamlit run examples/app_mesh3d.py\n"
    "or pass the app's path to this launcher:\n"
    "  python -m PyHydroGeophysX.gui_mesh3d path/to/app_mesh3d.py"
)


def _find_app() -> Optional[Path]:
    """Locate the 3D Mesh Builder Streamlit app.

    Returns
    -------
    Path or None
        The app beside the package (a source checkout or an editable install)
        or under the current folder; None when neither has it, as for a
        package installed from a wheel.
    """
    package_root = Path(__file__).resolve().parent
    candidates = [
        package_root.parent / APP_RELATIVE_PATH,
        Path.cwd() / APP_RELATIVE_PATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _split_app_argument(args: List[str]) -> Tuple[Optional[Path], List[str]]:
    """Take an app path given as the first argument, if there is one."""
    if args and args[0].lower().endswith(".py"):
        return Path(args[0]).expanduser(), args[1:]
    return None, args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the 3D Mesh Builder GUI.

    Parameters
    ----------
    argv : sequence of str, optional
        Extra arguments forwarded to ``streamlit run``. A first argument ending
        in ``.py`` is taken as the app to run instead of the one found in a
        source checkout.

    Returns
    -------
    int
        Streamlit exit code.

    Raises
    ------
    SystemExit
        With a message and a non-zero status when Streamlit is not installed
        or no app can be found to run.
    """
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required for the 3D Mesh Builder GUI. "
            "Install with: pip install streamlit plotly"
        ) from exc

    given, extra_args = _split_app_argument(list(argv if argv is not None else sys.argv[1:]))
    if given is not None:
        if not given.is_file():
            raise SystemExit(f"No app file at {given}.\n\n{NOT_BUNDLED_MESSAGE}")
        app_path = given
    else:
        app_path = _find_app()
        if app_path is None:
            raise SystemExit(NOT_BUNDLED_MESSAGE)
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
