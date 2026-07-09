"""Offscreen UI contract for Hydro -> Geophysics parameter and result flow."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication

from PyHydroGeophysX.qt_apps.modules.hydro_geophysics import HydroGeophysicsModule
from PyHydroGeophysX.qt_apps.state import WorkbenchState


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_hydro_agent_confirms_parameters_and_splits_selected_results(app) -> None:
    state = WorkbenchState(project_root=ROOT, hydro_data_dir=ROOT / "examples" / "data")
    module = HydroGeophysicsModule(state, lambda *_args: None)
    selected = module.agent_apply("select_methods", {"methods": ["ERT", "SRT"]})
    assert selected["parameter_confirmation_required"] is True
    assert selected["parameter_defaults"]["ERT"]["electrodes"] == 72
    assert module._current == 3

    # Make the readiness predicate true without starting a real worker.
    module._top = np.zeros((2, 2))
    module._point1, module._point2 = [0.0, 0.0], [1.0, 1.0]
    blocked = module.agent_apply("run", {})
    assert blocked["required_action"] == "confirm_parameters"
    assert module.agent_apply("confirm_parameters", {"mode": "defaults"})["status"] == "ok"

    image = ROOT / "examples" / "data" / "EM" / "EastRiver_VTEM" / "eastriver_area_depthslices.png"
    module._populate_results({
        "status": "ok", "methods": ["ERT", "SRT"], "mesh_cells": 12,
        "display_paths": {
            "ERT": {"model": str(image), "measurement": str(image)},
            "SRT": {"model": str(image), "measurement": str(image)},
        },
        "data_paths": [], "config_path": "",
    })
    assert module._result_method.count() == 2
    assert module._result_method.itemData(0) == "ERT"
    assert module._result_method.itemData(1) == "SRT"
    assert "ERT model" in module._result_model_caption.text()
