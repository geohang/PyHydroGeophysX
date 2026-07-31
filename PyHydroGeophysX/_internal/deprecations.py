"""Shared compatibility-window warnings for the 0.4 to 0.5 transition."""

from __future__ import annotations

import warnings


def warn_legacy_path(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated in PyHydroGeophysX 0.4.0; use {new}. "
        "The compatibility path will be removed in 0.5.0.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = ["warn_legacy_path"]
