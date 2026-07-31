"""Installation guidance and shared errors for optional dependencies."""

from __future__ import annotations

from typing import Mapping


INSTALL_HINTS: Mapping[str, str] = {
    "obspy": 'pip install "pyhydrogeophysx[seismic-raw]"',
    "segyio": "pip install segyio",
    "pyvista": "pip install pyvista pyvistaqt",
    "pyvistaqt": "pip install pyvista pyvistaqt",
    "pygimli": "conda install -c gimli pygimli",
    "simpeg": 'pip install "pyhydrogeophysx[geophysics]"',
    "pymatsolver": 'pip install "pyhydrogeophysx[geophysics]"',
    "resipy": "pip install resipy",
    "qtawesome": 'pip install "pyhydrogeophysx[desktop]"',
    "pyqtgraph": 'pip install "pyhydrogeophysx[desktop]"',
}


def missing_dependency_name(exc: BaseException) -> str:
    name = getattr(exc, "name", None)
    return str(name).split(".")[0] if name else ""


def installation_hint(exc: BaseException) -> str:
    name = missing_dependency_name(exc)
    if not name:
        return ""
    return INSTALL_HINTS.get(name, f"pip install {name}")


def optional_import_error(public_name: str, exc: ImportError) -> ImportError:
    package = missing_dependency_name(exc) or "an optional dependency"
    command = installation_hint(exc)
    detail = f" Install it with `{command}`." if command else ""
    return ImportError(
        f"{public_name} is unavailable because {package!r} could not be imported."
        f"{detail}"
    )


class BackendUnavailable(RuntimeError):
    """Raised when an optional numerical backend cannot be used."""


__all__ = [
    "BackendUnavailable",
    "INSTALL_HINTS",
    "installation_hint",
    "missing_dependency_name",
    "optional_import_error",
]
