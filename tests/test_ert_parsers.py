"""Tests for the embedded (no-resipy) ERT parser path on bundled sample data."""

from pathlib import Path

import pytest

pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parents[1]
BERT_FILE = REPO_ROOT / "examples" / "data" / "ERT" / "Bert" / "fielddataline2.dat"


@pytest.fixture()
def agent_module():
    return pytest.importorskip("PyHydroGeophysX.data_processing.ert_data_agent")


def test_embedded_bert_parser(agent_module, tmp_path, monkeypatch):
    if not BERT_FILE.exists():
        pytest.skip("BERT sample data not present in this checkout")
    monkeypatch.setattr(agent_module, "_HAS_RESIPY", False)
    std = agent_module.load_ert_resipy(
        project_dir=str(tmp_path / "proj"),
        data_file=str(BERT_FILE),
        instrument="BERT",
    )
    assert len(std.electrodes) == 72
    assert len(std.observations) == 936
    ys = [float(e.y) for e in std.electrodes]
    zs = [float(e.z) for e in std.electrodes]
    relief = max(max(ys) - min(ys), max(zs) - min(zs))
    assert relief > 5.0
    app_res = [o.app_res for o in std.observations if o.app_res is not None]
    assert app_res, "expected apparent-resistivity values in the observations"
