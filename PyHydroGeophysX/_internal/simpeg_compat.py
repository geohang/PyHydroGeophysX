"""Keyword compatibility across SimPEG releases.

SimPEG 0.25 renamed the conjugate-gradient options of its Gauss-Newton
optimizers (``maxIterCG`` -> ``cg_maxiter``, ``tolCG`` -> ``cg_rtol``, same
meaning) and the in-memory switch of its output directives (``save_txt`` ->
``on_disk``); 0.26 removed the old names, so a call written for one release
raises on the other. The supported range is ``simpeg>=0.24,<1.0``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def cg_keywords(optimizer: type, *, maxiter: int, rtol: float) -> dict[str, Any]:
    """CG keyword arguments for ``optimizer`` under the installed SimPEG."""
    try:
        names = set(inspect.signature(optimizer).parameters)
    except (TypeError, ValueError):
        names = set()
    if "cg_maxiter" in names:
        return {"cg_maxiter": int(maxiter), "cg_rtol": float(rtol)}
    return {"maxIterCG": int(maxiter), "tolCG": float(rtol)}


def in_memory_output(directive: Callable[..., Any]) -> Any:
    """An output directive that keeps its history in memory, under any SimPEG.

    The directives forward their options through ``**kwargs``, so the
    signature does not tell which name is accepted; SimPEG < 0.25 rejects
    ``on_disk`` with a generic ``Exception`` naming the attribute.
    """
    try:
        return directive(on_disk=False)
    except Exception as exc:
        if "on_disk" not in str(exc) and "not recognized" not in str(exc):
            raise
        return directive(save_txt=False)


__all__ = ["cg_keywords", "in_memory_output"]
