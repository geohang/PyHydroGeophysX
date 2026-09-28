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


def _internal_module_name(exc: BaseException) -> str:
    """The PyHydroGeophysX module an import error names, or "" for any other."""
    name = str(getattr(exc, "name", None) or "")
    return name if name.split(".")[0] == "PyHydroGeophysX" else ""


def missing_dependency_name(exc: BaseException) -> str:
    """Top-level package an import error names, "" for none or for this package.

    A module of PyHydroGeophysX itself failing to import is a broken install,
    not a missing dependency, and used to be answered with the advice to
    ``pip install PyHydroGeophysX``.
    """
    name = getattr(exc, "name", None)
    if not name or _internal_module_name(exc):
        return ""
    return str(name).split(".")[0]


def installation_hint(exc: BaseException) -> str:
    name = missing_dependency_name(exc)
    if not name:
        return ""
    return INSTALL_HINTS.get(name, f"pip install {name}")


def optional_import_error(public_name: str, exc: ImportError) -> ImportError:
    internal = _internal_module_name(exc)
    if internal:
        return ImportError(
            f"{public_name} is unavailable because the PyHydroGeophysX module "
            f"{internal!r} could not be imported ({exc}). The installation looks "
            "incomplete: use a complete checkout of the repository, or reinstall "
            "PyHydroGeophysX."
        )
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
