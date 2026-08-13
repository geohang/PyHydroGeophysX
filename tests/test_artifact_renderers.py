import pytest

from PyHydroGeophysX.qt_apps.artifact_renderers import select_renderer


@pytest.mark.parametrize(
    ("artifact", "shape", "expected"),
    [
        ({"kind": "ert_model_bundle"}, None, "mesh_bundle"),
        # A bare PyGIMLi mesh is the 3-D mesh builder's main output.
        ({"path": "forward_mesh.bms"}, None, "mesh"),
        ({"kind": "mesh", "path": "grid.bms"}, None, "mesh"),
        ({"path": "velocity.vtk"}, None, "vtk"),
        ({"path": "section.npy"}, (20, 30), "array"),
        ({"path": "times.npy"}, (12, 20, 30), "array_stack"),
        ({"kind": "gravmag_profile", "path": "line.csv"}, None, "curve"),
        ({"path": "stations.csv"}, None, "table"),
        ({"path": "overview.png"}, None, "image"),
        ({"path": "result.json"}, None, "json"),
    ],
)
def test_renderer_dispatch(artifact, shape, expected):
    assert select_renderer(artifact, shape) == expected
