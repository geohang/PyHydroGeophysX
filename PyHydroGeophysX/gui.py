"""Local GUI launcher for PyHydroGeophysX."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#: The web app, relative to a source checkout.
APP_RELATIVE_PATH = Path("examples") / "app_geophysics_workflow.py"

#: What a pip install is told. The Streamlit apps live in examples/, which the
#: wheel does not include, so the launcher of an installed package found nothing
#: to run and ended in a FileNotFoundError traceback.
NOT_BUNDLED_MESSAGE = (
    "The PyHydroGeophysX web apps ship with the source repository, not with the "
    "pip package, and no copy was found beside this installation or in the "
    "current folder.\n"
    "To run the web app locally:\n"
    "  git clone https://github.com/geohang/PyHydroGeophysX.git\n"
    "  cd PyHydroGeophysX\n"
    "  streamlit run examples/app_geophysics_workflow.py\n"
    "or pass the app's path to this launcher:\n"
    "  pyhydrogeophysx-gui path/to/app_geophysics_workflow.py\n"
    "The hosted app needs no install: https://pyhydrogeophysx.streamlit.app/"
)


def _find_streamlit_app() -> Optional[Path]:
    """Locate the repository Streamlit app used by the local GUI.

    Returns
    -------
    Path or None
        The app beside the package (a source checkout or an editable
        install) or under the current folder; None when neither has it,
        which is the case for a package installed from a wheel.
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
    """Launch the Streamlit GUI.

    Parameters
    ----------
    argv : sequence of str, optional
        Extra arguments passed to ``streamlit run``. A first argument ending
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
            "Streamlit is required for the local GUI. Install with "
            "`pip install pyhydrogeophysx[webapp]` or install Streamlit in your conda environment."
        ) from exc

    given, extra_args = _split_app_argument(list(argv if argv is not None else sys.argv[1:]))
    if given is not None:
        if not given.is_file():
            raise SystemExit(f"No app file at {given}.\n\n{NOT_BUNDLED_MESSAGE}")
        app_path = given
    else:
        app_path = _find_streamlit_app()
        if app_path is None:
            raise SystemExit(NOT_BUNDLED_MESSAGE)
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
