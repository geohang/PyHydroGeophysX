"""Shared plotting convention for inverted ERT resistivity models."""

from __future__ import annotations

from typing import Any, Dict


ERT_RESISTIVITY_LABEL = "Resistivity (Ω·m)"


def ert_model_plot_kwargs(show_mesh: bool = True) -> Dict[str, Any]:
    """Return the standard pyGIMLi keyword arguments for an ERT model plot."""
    return {
        "cMap": "Spectral_r",
        "logScale": True,
        "showMesh": bool(show_mesh),
    }
