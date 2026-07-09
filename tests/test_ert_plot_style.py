"""Tests for the shared ERT model plotting convention."""

from PyHydroGeophysX.qt_apps.ert_plot_style import (
    ERT_RESISTIVITY_LABEL,
    ert_model_plot_kwargs,
)


def test_ert_model_plot_kwargs_use_the_standard_ert_style():
    """ERT model figures use the shared logarithmic, meshed convention."""
    assert ERT_RESISTIVITY_LABEL == "Resistivity (Ω·m)"
    assert ert_model_plot_kwargs() == {
        "cMap": "Spectral_r",
        "logScale": True,
        "showMesh": True,
    }
    assert ert_model_plot_kwargs(show_mesh=False)["showMesh"] is False
