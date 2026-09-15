"""Loading a file under the wrong Instrument must report itself, not crash.

ResIPy parses happily under a mismatched format and hands back quadrupoles that
cite electrodes the file never defines. The container conversion then yields
None, which used to reach ``[True] * data.size()`` as an AttributeError in a
worker callback — a traceback in the console and nothing in the UI.
"""

import os
from pathlib import Path
from types import SimpleNamespace

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


def _module():
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    return ERTProcessingModule(StudioState(output_dir=Path.cwd()), lambda *_a: None)


def _standard(electrode_ids, quad_ids, app_res=12.0):
    quad = SimpleNamespace(A=quad_ids[0], B=quad_ids[1], M=quad_ids[2], N=quad_ids[3])
    return SimpleNamespace(
        electrodes=[SimpleNamespace(id=i, x=float(i), y=0.0, z=0.0)
                    for i in electrode_ids],
        observations=[SimpleNamespace(quad=quad, app_res=app_res, rel_err=0.03)],
    )


def test_unconvertible_file_warns_instead_of_raising(qt_app) -> None:
    module = _module()
    try:
        messages = []
        module.log = lambda text, level="info": messages.append((level, text))
        note = module._container_failure_reason(
            _standard([1, 2, 3, 4], [101, 102, 103, 104]))

        module._on_ert_loaded("line1.dat", {
            "elec": [(0.0, 0.0), (1.0, 0.0)], "pseudo": [], "nmeas": 1,
            "data": None, "warning": "", "reader": "ResIPy",
            "reader_reason": "", "data_note": note,
        })

        assert module._qc_mask == []
        assert module._ert_data is None
        warnings = [text for level, text in messages if level == "warn"]
        assert len(warnings) == 1
        assert "Instrument setting is probably wrong" in warnings[0]
        # Nothing may claim the load succeeded when it produced no container.
        assert not [text for level, text in messages if level == "success"]
        # The panel says so too; the log alone is easy to miss.
        assert "Not usable for inversion" in module._info.text()
    finally:
        module.stop_workers()
        module.close()
        qt_app.processEvents()


def test_a_good_file_still_reports_success(qt_app) -> None:
    pg = pytest.importorskip("pygimli")
    from pygimli.physics import ert as pg_ert

    module = _module()
    try:
        messages = []
        module.log = lambda text, level="info": messages.append((level, text))
        data = pg_ert.createData(elecs=4, schemeName="dd")
        data["rhoa"] = [100.0] * data.size()

        module._on_ert_loaded("good.dat", {
            "elec": [(float(i), 0.0) for i in range(4)], "pseudo": [],
            "nmeas": int(data.size()), "data": data, "warning": "",
            "reader": "PyGIMLi", "reader_reason": "", "data_note": "",
        })

        assert module._qc_mask == [True] * int(data.size())
        assert not [text for level, text in messages if level == "warn"]
        assert [text for level, text in messages if level == "success"]
        assert "Not usable for inversion" not in module._info.text()
    finally:
        module.stop_workers()
        module.close()
        qt_app.processEvents()


@pytest.mark.parametrize("std, expected", [
    (_standard([1, 2, 3, 4], [101, 102, 103, 104]), "Instrument setting"),
    (_standard([1, 2, 3, 4], [1, 2, 3, 4], app_res=None), "apparent resistivity"),
    (SimpleNamespace(electrodes=[], observations=[]), "no measurements"),
])
def test_failure_reason_names_the_cause(std, expected) -> None:
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule

    assert expected in ERTProcessingModule._container_failure_reason(std)
