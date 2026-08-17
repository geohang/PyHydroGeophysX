"""Regressions in the EM page's line-inversion plumbing."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PyHydroGeophysX.data_processing import em1d, run_inputs


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import PySide6.QtGui  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"Qt stack unavailable: {exc}", allow_module_level=True)

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QSpinBox

from PyHydroGeophysX.qt_apps.modules.em_processing import EMProcessingModule


def test_cpu_thread_control_does_not_replace_worker_registry() -> None:
    app = QApplication.instance() or QApplication([])
    module = EMProcessingModule(None, lambda *_args: None)
    worker = QThread()

    assert isinstance(module._workers, list)
    assert isinstance(module._parallel_workers_spin, QSpinBox)
    module.register_worker(worker)
    assert worker in module._workers

    module._drop_worker(worker)
    module.close()
    app.processEvents()


def _page_stub(source: Path) -> SimpleNamespace:
    """``_persist_source`` reads a handful of attributes; no widget is needed."""
    return SimpleNamespace(
        _source_path=source,
        _data={},
        _tem_moment=SimpleNamespace(currentText=lambda: "HM"),
        _collect_geom=lambda: {"use_project_flags": True},
        log=lambda *_args, **_kwargs: None,
    )


def _survey_file(root: Path, n_soundings: int = 3) -> Path:
    """A plain TDEM table: first column times, one column per sounding."""
    times = np.geomspace(1e-5, 1e-3, 12)
    columns = [times] + [times * float(index + 1) for index in range(n_soundings)]
    survey = root / "trailcreek" / "line.txt"
    survey.parent.mkdir(parents=True)
    np.savetxt(survey, np.column_stack(columns))
    return survey


def test_the_run_keeps_the_soundings_not_the_survey_folder(tmp_path) -> None:
    """A run used to cost another copy of the acquisition it read.

    The trailcreek Project sat on the survey folder, so ``copytree`` walked
    into the duplicate it was still writing and nested it until Windows
    refused the path (WinError 206), after writing 63 MB. Only the soundings
    are ever read back, so only those are stored.
    """
    survey = _survey_file(tmp_path)
    inputs = tmp_path / "project" / "runs" / "20260815-141721_emline_7c96" / "inputs"
    inputs.mkdir(parents=True)

    persisted = EMProcessingModule._persist_source(
        _page_stub(survey), inputs, "TDEM"
    )

    assert persisted.parent == inputs
    assert [item.name for item in inputs.iterdir()] == ["em_soundings.npz"]
    assert run_inputs.is_container(persisted)


def test_every_stored_sounding_comes_back_through_load_sounding(tmp_path) -> None:
    """``invert_line`` reads station ``s`` by index, so the container must too."""
    survey = _survey_file(tmp_path, n_soundings=3)
    inputs = tmp_path / "project" / "inputs"
    inputs.mkdir(parents=True)

    persisted = EMProcessingModule._persist_source(
        _page_stub(survey), inputs, "TDEM"
    )

    for index in range(3):
        direct = em1d.load_sounding(str(survey), "TDEM", sounding=index)
        stored = em1d.load_sounding(str(persisted), "TDEM", sounding=index)
        assert stored["n_soundings"] == 3
        np.testing.assert_allclose(stored["times"], direct["times"])
        np.testing.assert_allclose(stored["response"], direct["response"])


def test_a_container_refuses_the_method_it_was_not_written_for(tmp_path) -> None:
    """The moment and method decide what the gates mean; a silent swap is worse."""
    survey = _survey_file(tmp_path)
    inputs = tmp_path / "project" / "inputs"
    inputs.mkdir(parents=True)

    persisted = EMProcessingModule._persist_source(
        _page_stub(survey), inputs, "TDEM"
    )

    with pytest.raises(ValueError, match="TDEM"):
        em1d.load_sounding(str(persisted), "FDEM")


def test_an_unreadable_survey_leaves_the_run_pointing_at_the_original(tmp_path) -> None:
    """Persistence is bookkeeping; it must not be what kills an inversion."""
    survey = tmp_path / "trailcreek"
    survey.mkdir()
    inputs = tmp_path / "project" / "inputs"
    inputs.mkdir(parents=True)

    persisted = EMProcessingModule._persist_source(_page_stub(survey), inputs, "TDEM")

    assert persisted == survey
