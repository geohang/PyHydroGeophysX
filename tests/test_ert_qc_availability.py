import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt stack unavailable: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app


class _ResistanceOnlyData:
    """Minimal BERT-like data: resistance/error/k/rhoa, but no U/I pairs."""

    def __init__(self) -> None:
        self._data = {
            "a": np.array([0, 0]),
            "b": np.array([1, 1]),
            "m": np.array([2, 3]),
            "n": np.array([3, 4]),
            "r": np.array([1.0, 2.0]),
            "err": np.array([0.03, 0.03]),
            "k": np.array([10.0, 20.0]),
            "rhoa": np.array([10.0, 40.0]),
        }

    def haveData(self, token: str) -> bool:  # noqa: N802 - PyGIMLi API
        return token in self._data

    def __getitem__(self, token: str):
        return self._data[token]

    def size(self) -> int:
        return 2


def test_more_checks_explains_fields_missing_from_bert_file(qt_app) -> None:
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule
    from PyHydroGeophysX.qt_apps.state import WorkbenchState

    view = ERTProcessingModule(
        WorkbenchState(output_dir=Path.cwd()),
        lambda *_args: None,
    )
    view._ert_data_full = _ResistanceOnlyData()
    view._refresh_qc_availability()
    view._qc_more.setChecked(True)

    assert not view._qc_min_v.isEnabled()
    assert not view._qc_min_i.isEnabled()
    assert view._qc_max_k.isEnabled()
    assert not view._qc_max_recip.isEnabled()
    explanation = view._qc_support_note.text()
    assert "no voltage column" in explanation
    assert "no current column" in explanation
    assert "no reciprocal pairs" in explanation
    assert "not because the input box is broken" in explanation

    view.stop_workers()
    view.close()
    qt_app.processEvents()
