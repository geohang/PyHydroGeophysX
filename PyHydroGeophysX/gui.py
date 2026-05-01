"""Local GUI launcher for PyHydroGeophysX."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence


def _find_streamlit_app() -> Path:
    """Locate the repository Streamlit app used by the local GUI."""

    package_root = Path(__file__).resolve().parent
    candidates = [
        package_root.parent / "examples" / "app_geophysics_workflow.py",
        Path.cwd() / "examples" / "app_geophysics_workflow.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find examples/app_geophysics_workflow.py. "
        "Run the launcher from a PyHydroGeophysX checkout or install the package in editable mode."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the Streamlit GUI.

    Parameters
    ----------
    argv : sequence of str, optional
        Extra arguments passed to ``streamlit run``.

    Returns
    -------
    int
        Streamlit exit code.
    """

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required for the local GUI. Install with "
            "`pip install pyhydrogeophysx[webapp]` or install Streamlit in your conda environment."
        ) from exc

    app_path = _find_streamlit_app()
    extra_args = list(argv if argv is not None else sys.argv[1:])
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
