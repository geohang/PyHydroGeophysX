"""Launcher for the 3D Mesh Builder Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence


def _find_app() -> Path:
    """Locate the 3D Mesh Builder Streamlit app."""
    package_root = Path(__file__).resolve().parent
    candidates = [
        package_root.parent / "examples" / "app_mesh3d.py",
        Path.cwd() / "examples" / "app_mesh3d.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find examples/app_mesh3d.py. "
        "Run the launcher from a PyHydroGeophysX checkout or install the package in editable mode."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the 3D Mesh Builder GUI.

    Parameters
    ----------
    argv:
        Extra arguments forwarded to ``streamlit run``.

    Returns
    -------
    int
        Streamlit exit code.
    """
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required for the 3D Mesh Builder GUI. "
            "Install with: pip install streamlit plotly"
        ) from exc

    app_path = _find_app()
    extra_args = list(argv if argv is not None else sys.argv[1:])
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
