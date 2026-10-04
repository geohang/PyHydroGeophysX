"""External GeoSAGE integration stays optional and preserves honest outcomes."""

import json

import pytest


def test_needs_review_is_not_saved_as_success():
    from PyHydroGeophysX.qt_apps.results_store import normalize_status

    assert normalize_status("needs_review") == "needs_review"


def test_installed_geosage_runs_on_the_studio_contract(tmp_path):
    pytest.importorskip("geosage.pyhydrogeophysx")
    import numpy as np
    from discretize import TensorMesh
    from PyHydroGeophysX.agents.assistants import get_assistant
    from PyHydroGeophysX.qt_apps.agent.one_click_runner import execute

    source = tmp_path / "source"
    (source / "mesh").mkdir(parents=True)
    (source / "inversion_result").mkdir()
    (source / "geology_models").mkdir()
    mesh = TensorMesh([[10] * 3, [10] * 3, [10] * 3], x0=[100, 200, -30])
    mesh.write_UBC(source / "mesh/mesh_core.msh")
    shape = (3, 3, 3)
    np.save(source / "inversion_result/joint_density_core.npy", np.arange(27).reshape(shape) / 100)
    np.save(source / "inversion_result/joint_susceptibility_core.npy", np.ones(shape) / 1000)
    for name in ("unit_id_3d", "geo_id_3d"):
        np.save(source / f"geology_models/{name}.npy", np.ones(shape, dtype=int))
    (source / "geology_models/geo_defs.json").write_text(json.dumps({"1": "Synthetic unit"}))
    assistant = get_assistant("geosage")
    assert assistant.availability()[0]
    events, approved = [], []

    def approve(event):
        approved.append(event["tool"])
        return "proceed"

    result = execute({"assistant": "geosage", "request": "Inspect the synthetic archive",
                      "inputs": {"source_inversion_dir": str(source)},
                      "output_dir": str(tmp_path / "output"), "step_mode": True},
                     lambda *args: None, approve=approve, on_event=events.append)
    assert result["status"] == "needs_review"
    assert result["source_files_unchanged"]
    assert approved == list(assistant.load_tools())
    assert [e["tool"] for e in events if e["phase"] == "done"] == approved
    assert result["report_files"]["report_markdown"].endswith("evidence_summary.md")
