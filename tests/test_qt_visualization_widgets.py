"""Focused regressions for the physical meaning of Qt image displays."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import PySide6.QtGui  # noqa: F401
    import pyqtgraph  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"Qt stack unavailable: {exc}", allow_module_level=True)

from PySide6.QtWidgets import QApplication

from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule
from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
from PyHydroGeophysX.qt_apps.state import WorkbenchState
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.em_overview_view import EMOverviewView
from PyHydroGeophysX.qt_apps.widgets.plan_slice_view import PlanSliceView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_ert_pseudosection_has_stable_static_markers_and_physical_colour_scale(
    app, tmp_path: Path
) -> None:
    view = ERTProcessingModule(
        WorkbenchState(output_dir=tmp_path), lambda *_args: None
    )
    view._pseudo = [(0.0, 1.0, 10.0), (1.0, 2.0, 100.0), (2.0, 3.0, 1000.0)]
    view._draw_pseudosection()
    view._pseudo_canvas.draw()
    app.processEvents()

    assert not view._pseudo_legend.isHidden()
    legend_values = [float(label.text()) for label in view._pseudo_scale_labels]
    assert legend_values == sorted(legend_values)
    assert legend_values[0] >= 10.0
    assert legend_values[-1] <= 1000.0
    assert len(view._pseudo_ax.collections) == 1
    offsets = np.asarray(view._pseudo_ax.collections[0].get_offsets(), dtype=float)
    np.testing.assert_allclose(
        offsets, np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    )
    assert view._pseudo_ax.get_xlim() == pytest.approx((-0.06, 2.06))
    assert view._pseudo_ax.get_ylim() == pytest.approx((3.24, 0.0))
    assert view._pseudo_ax.yaxis_inverted()
    view.close()


def test_array_log_colourbar_reports_physical_values_and_resets_extent(app) -> None:
    view = ArrayViewer()
    values = np.geomspace(1.0, 1000.0, 120).reshape(10, 12)
    view.set_array(
        values,
        log=True,
        extent=(100.0, 220.0, -20.0, 0.0),
        value_label="Resistivity (Ω·m)",
    )
    assert not view._hist.axis.logMode
    assert view._hist.axis.tickStrings([0.0, 1.0, 2.0, 3.0], 1.0, 1.0) == [
        "1", "10", "100", "1000"
    ]
    assert view._hist.axis.labelText == "Resistivity (Ω·m)"
    assert view._img.transform().m11() != pytest.approx(1.0)

    view.set_array(np.arange(12.0).reshape(3, 4))
    assert view._img.transform().m11() == pytest.approx(1.0)
    assert view._img.transform().m22() == pytest.approx(1.0)
    view.close()


def test_plan_slice_explains_empty_log_layer_without_empty_colourbar(app) -> None:
    view = PlanSliceView()
    view.show_slices(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.full((2, 2), np.nan),
        np.array([1.0, 2.0]),
        label="Resistivity (Ω·m)",
    )
    view._canvas.draw()
    assert len(view._fig.axes) == 1
    assert "No positive finite values" in view._fig.axes[0].texts[0].get_text()
    view.close()


def test_em_smooth_section_draws_a_single_sounding(app) -> None:
    view = EMOverviewView()
    view.show_result({
        "model3d": np.array([[[200.0, 100.0, 50.0]]]),
        "depth_edges": np.array([0.0, 2.0, 5.0, 10.0]),
        "positions": np.array([0.0]),
        "line_numbers": np.array([0]),
        "chi2_list": np.array([1.0]),
        "log_scale": True,
        "label": "Resistivity (Ω·m)",
        "method": "TDEM",
    })
    view._canvas.draw()
    section = view._fig.axes[0]
    assert section.images, "Smooth mode should render a column for one sounding."
    view.close()


def test_model3d_log_view_explains_nonpositive_model(app, monkeypatch) -> None:
    import PyHydroGeophysX.qt_apps.widgets.model3d_view as model3d

    monkeypatch.setattr(
        model3d, "try_import_pyvista", lambda: (False, None, None, "disabled")
    )
    view = model3d.Model3DView()
    view.show_model(
        (np.arange(3.0), np.arange(2.0), np.arange(3.0)),
        -np.ones((2, 1, 2)),
        log_scale=True,
    )
    view._canvas.draw()
    assert len(view._fig.axes) == 1
    assert "No positive finite values" in view._fig.axes[0].texts[0].get_text()
    view.close()


def test_model_viewer_preserves_resistivity_artifact_semantics(
    app, tmp_path: Path
) -> None:
    path = tmp_path / "apparent_resistivity.npy"
    np.save(path, np.geomspace(1.0, 1000.0, 20).reshape(4, 5))
    state = WorkbenchState(output_dir=tmp_path)
    view = ModelViewerModule(state, lambda *_args: None)
    view._render_numpy(
        path,
        "array",
        {
            "kind": "apparent_resistivity",
            "format": "npy",
            "metadata": {"label": "Apparent resistivity", "units": "Ω·m"},
        },
    )
    rendered = view._visual_layout.itemAt(0).widget()
    assert isinstance(rendered, ArrayViewer)
    assert rendered._log_display
    assert rendered._hist.axis.labelText == "Apparent resistivity (Ω·m)"

    stack_path = tmp_path / "constant_resistivity.npy"
    np.save(stack_path, np.full((2, 4, 5), 100.0))
    view._render_numpy(stack_path, "array_stack", {"kind": "resistivity_model"})
    rendered_stack = view._visual_layout.itemAt(0).widget().findChild(ArrayViewer)
    assert rendered_stack is not None
    assert rendered_stack._img.levels[1] > rendered_stack._img.levels[0]
    view.close()
