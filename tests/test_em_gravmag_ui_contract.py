"""Optional offscreen checks for the EM and Gravity/Magnetics workbench pages."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication, QPushButton

from PyHydroGeophysX.qt_apps.modules.em_processing import EMProcessingModule
from PyHydroGeophysX.qt_apps.modules.gravmag_processing import GravMagProcessingModule
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView


class _State:
    output_dir = ""

    def __init__(self) -> None:
        self.module_results = {}

    def update_module_result(self, key, value) -> None:
        self.module_results[key] = value


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_em_example_action_and_method_specific_controls(app) -> None:
    module = EMProcessingModule(_State(), lambda *_args: None)
    assert "Examples" not in {button.text() for button in module.findChildren(QPushButton)}
    result = module.agent_apply("use_example_data", {"example": "synthetic_fdem"})
    assert result["status"] == "ok"
    assert result["method"] == "FDEM"
    assert not module._tx_rx.isHidden()
    module._method.setCurrentText("TDEM")
    assert module._tx_rx.isHidden()
    assert "backend" in module.agent_apply("get_status", {})


def test_gravmag_example_qc_and_agent_controls(app) -> None:
    module = GravMagProcessingModule(_State(), lambda *_args: None)
    assert "Use example data" not in {button.text() for button in module.findChildren(QPushButton)}
    loaded = module.agent_apply("use_example_data", {"kind": "gravity"})
    assert loaded["status"] == "ok"
    assert loaded["stations"] == 3877
    qc = module.agent_apply("run_qc", {})
    assert qc["status"] == "ok"
    assert set(qc["stats"]) == {"Observed", "Regional", "Residual"}
    params = module.agent_apply("set_params", {"params": {
        "station_elevation": 5.0, "relative_error": 0.05, "noise_floor": 0.7,
        "max_stations": 300,
    }})
    assert params["status"] == "ok"
    forward = module.agent_apply("run_forward_bodies", {"bodies": [{
        "type": "sphere", "x0": 0.0, "y0": 0.0, "z0": 100.0,
        "radius": 50.0, "density_contrast": 300.0,
    }]})
    assert forward["status"] == "ok"


def test_quality_view_draws_final_misfit_without_iteration_history(app) -> None:
    view = InversionQualityView()
    view.show_quality({"chi2": 2.5, "n_data": 30}, convergence=None, title="EM inversion")
    axis = view._fig.axes[0]
    assert axis.get_title() == "Final data misfit"
    assert len(axis.patches) == 1
