"""Shared warnings for the compatibility paths kept through the 0.5 series.

PyHydroGeophysX 0.5.0 is the first release after 0.3.0 - 0.4.0 was never
published - so every path deprecated since 0.3.0 warns from 0.5.0 on and goes
in 0.6.0. Keeping both numbers here keeps every message in step.
"""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.util
import inspect
import sys
import warnings
from typing import Any, Callable, Dict, Mapping, Tuple

#: The release in which the compatibility paths start to warn.
DEPRECATED_IN = "0.5.0"
#: The release that removes them.
REMOVED_IN = "0.6.0"


def warn_legacy_path(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated in PyHydroGeophysX {DEPRECATED_IN}; use {new}. "
        f"The compatibility path will be removed in {REMOVED_IN}.",
        DeprecationWarning,
        stacklevel=3,
    )


def legacy_names(module: str, names: Mapping[str, str]) -> Callable[[str], Any]:
    """A module ``__getattr__`` that still answers to names 0.3.0 had.

    Parameters
    ----------
    module : str
        The module's ``__name__``.
    names : dict
        Each old name, mapped to the full dotted path of what replaced it: an
        attribute of a module, or a module itself.

    Returns
    -------
    callable
        The module's ``__getattr__``. An old name warns with its replacement and
        returns it; any other name raises ``AttributeError``, as it would with
        no hook.

    Examples
    --------
    >>> import os, warnings
    >>> lookup = legacy_names("example", {"SEPARATOR": "os.sep"})
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     value = lookup("SEPARATOR")
    >>> value == os.sep, str(caught[0].message).split(". ")[0]
    (True, 'example.SEPARATOR is deprecated in PyHydroGeophysX 0.5.0; use os.sep')
    """
    def __getattr__(name: str) -> Any:
        new = names.get(name)
        if new is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        warn_legacy_path(f"{module}.{name}", new)
        home, _, attribute = new.rpartition(".")
        parent = importlib.import_module(home)
        try:
            return getattr(parent, attribute)
        except AttributeError:
            # A submodule is an attribute of its package only once imported.
            return importlib.import_module(new)

    return __getattr__


def renamed_keywords(owner: str, **renamed: str) -> Callable[[Callable], Callable]:
    """Accept keyword arguments under the names 0.3.0 gave them.

    Parameters
    ----------
    owner : str
        The callable as the warning names it, such as ``"binaryread"``.
    **renamed : str
        Each old keyword, set to its new name.

    Returns
    -------
    callable
        A decorator. An old keyword warns and is passed on under its new name;
        giving both is a ``TypeError``. The signature the wrapper reports is the
        function's own.

    Examples
    --------
    >>> import warnings
    >>> @renamed_keywords("scale", factor="gain")
    ... def scale(value, gain=1.0):
    ...     return value * gain
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     scaled = scale(2.0, factor=3.0)
    >>> scaled, str(caught[0].message).startswith("scale(factor=...) is deprecated")
    (6.0, True)
    """
    def decorate(func: Callable) -> Callable:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in renamed.items():
                if old not in kwargs:
                    continue
                value = kwargs.pop(old)
                try:
                    given = signature.bind_partial(*args, **kwargs).arguments
                except TypeError:
                    given = {}  # a call that is wrong anyway; the function says how
                if new in given:
                    raise TypeError(
                        f"{owner}() got both {new!r} and {old!r}, its name before "
                        f"PyHydroGeophysX {DEPRECATED_IN}; pass only {new!r}")
                warn_legacy_path(f"{owner}({old}=...)", f"{owner}({new}=...)")
                kwargs[new] = value
            return func(*args, **kwargs)

        return wrapper

    return decorate


class _RemovedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answer imports of modules that were deleted from the package.

    A deleted module fails with a bare ``ModuleNotFoundError`` that says nothing
    about what replaced it, so code written for 0.3.0 broke without a hint. For
    each name registered here the import succeeds instead: it warns with the
    replacement, and ``build`` fills the stand-in module with whatever names the
    old one offered - stand-ins that refuse to run with the same message.

    The finder sits at the end of ``sys.meta_path``, so it only ever sees names
    that no real module answered.
    """

    def __init__(self) -> None:
        self._modules: Dict[str, Tuple[str, Callable]] = {}

    def add(self, name: str, message: str, build: Callable) -> None:
        self._modules[name] = (message, build)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._modules:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        message, build = self._modules[module.__name__]
        # The import machinery's own frames do not count towards stacklevel,
        # so 2 is the line that wrote the import.
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        build(module)


_REMOVED_MODULES = _RemovedModules()


def register_removed_module(name: str, message: str, build: Callable) -> None:
    """Make ``import name`` warn with ``message`` instead of failing.

    Parameters
    ----------
    name : str
        Full dotted name of the deleted module.
    message : str
        What replaced it; the warning says this.
    build : callable
        ``build(module)`` puts the old module's names on the stand-in.

    Examples
    --------
    >>> register_removed_module(
    ...     "PyHydroGeophysX._internal.gone_example", "gone; use nothing",
    ...     lambda module: setattr(module, "ANSWER", 42))
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     from PyHydroGeophysX._internal.gone_example import ANSWER
    >>> ANSWER, str(caught[0].message)
    (42, 'gone; use nothing')
    """
    if _REMOVED_MODULES not in sys.meta_path:
        sys.meta_path.append(_REMOVED_MODULES)
    _REMOVED_MODULES.add(name, message, build)


__all__ = [
    "DEPRECATED_IN",
    "REMOVED_IN",
    "legacy_names",
    "register_removed_module",
    "renamed_keywords",
    "warn_legacy_path",
]
