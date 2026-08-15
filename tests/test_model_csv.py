"""The flat CSV views of inversion models.

These tables are what a collaborator without PyGIMLi reads, so the properties
that matter are that every row carries a coordinate the value really belongs to,
and that a shape the writer cannot interpret is refused rather than guessed.
"""

from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing.model_csv import (
    column_name,
    export_model_csv,
    model_cell_table,
    write_grid_model_csv,
    write_layered_model_csv,
)


def _read(path: Path):
    """Return ``(header, rows)`` without depending on pandas."""
    import csv

    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    return rows[0], rows[1:]


@pytest.fixture(scope="module")
def mesh():
    pg = pytest.importorskip("pygimli")
    return pg.meshtools.createMesh(
        pg.meshtools.createRectangle(start=[0, -20], end=[100, 0], marker=2),
        quality=31, area=25,
    )


# -- units in the header ---------------------------------------------------
@pytest.mark.parametrize(
    ("name", "units", "expected"),
    [
        ("resistivity", "ohm.m", "resistivity_ohm_m"),
        ("resistivity", "Ω·m", "resistivity_ohm_m"),
        ("velocity", "m/s", "velocity_m_per_s"),
        ("density_contrast", "g/cc", "density_contrast_g_per_cc"),
        ("susceptibility", "SI", "susceptibility_SI"),
        ("value", "", "value"),
    ],
)
def test_units_are_spelled_out_in_the_column_name(name, units, expected):
    """A separator must become an underscore, never vanish into 'ohmm'."""
    assert column_name(name, units) == expected


# -- mesh models -----------------------------------------------------------
def test_every_row_carries_the_coordinate_of_its_own_cell(mesh, tmp_path):
    values = np.linspace(50.0, 2000.0, mesh.cellCount())
    header, rows = _read(export_model_csv(
        tmp_path, mesh, values, value_name="resistivity", units="ohm.m")[0])

    assert header[:5] == ["cell", "x", "z", "marker", "cell_area_m2"]
    assert header[5] == "resistivity_ohm_m"
    assert len(rows) == mesh.cellCount()

    centres = np.asarray(mesh.cellCenters())
    for index, row in enumerate(rows):
        assert int(row[0]) == index
        assert float(row[1]) == pytest.approx(centres[index, 0])
        assert float(row[2]) == pytest.approx(centres[index, 1])
        assert float(row[5]) == pytest.approx(values[index])


def test_mesh_geometry_lets_a_reader_rebuild_the_cell_polygons(mesh, tmp_path):
    written = export_model_csv(
        tmp_path, mesh, np.ones(mesh.cellCount()), value_name="v")
    assert [Path(item).name for item in written] == [
        "model_cells.csv", "mesh_nodes.csv", "mesh_cell_nodes.csv"]

    node_header, node_rows = _read(written[1])
    assert node_header == ["node", "x", "z"]
    assert len(node_rows) == mesh.nodeCount()

    cell_header, cell_rows = _read(written[2])
    assert cell_header[:2] == ["cell", "marker"]
    assert len(cell_rows) == mesh.cellCount()
    # Each listed node id must resolve to a real node, which is what makes the
    # pair of files usable as a triangulation.
    for row in cell_rows:
        for value in row[2:]:
            if value:
                assert 0 <= int(value) < mesh.nodeCount()


def test_geometry_can_be_skipped(mesh, tmp_path):
    written = export_model_csv(
        tmp_path, mesh, np.ones(mesh.cellCount()), value_name="v", geometry=False)
    assert len(written) == 1
    assert not (tmp_path / "mesh_nodes.csv").exists()


def test_time_lapse_orientation_is_read_off_the_mesh(mesh, tmp_path):
    """(n_cells, n_steps) and (n_steps, n_cells) must produce the same table."""
    n_cells = mesh.cellCount()
    models = np.column_stack([np.arange(n_cells) + step for step in range(4)])
    by_cell = model_cell_table(mesh, models, value_name="resistivity", units="ohm.m")
    by_step = model_cell_table(mesh, models.T, value_name="resistivity", units="ohm.m")
    assert by_cell == by_step
    assert by_cell[0][5:] == [f"resistivity_ohm_m_t{index:02d}" for index in range(4)]


def test_step_labels_name_the_columns_and_stay_unique(mesh, tmp_path):
    models = np.ones((mesh.cellCount(), 3))
    header, _ = model_cell_table(
        mesh, models, value_name="resistivity", units="ohm.m",
        step_labels=["2024-05-01", "2024-06-01", "2024-06-01"],
    )
    assert header[5:] == [
        "resistivity_ohm_m_2024_05_01",
        "resistivity_ohm_m_2024_06_01",
        "resistivity_ohm_m_2024_06_01_2",
    ]


def test_non_finite_values_become_empty_fields(mesh, tmp_path):
    values = np.ones(mesh.cellCount())
    values[0], values[1] = np.nan, np.inf
    _, rows = _read(export_model_csv(tmp_path, mesh, values, value_name="v")[0])
    assert rows[0][-1] == ""
    assert rows[1][-1] == ""
    assert rows[2][-1] == "1"


def test_a_model_that_does_not_fit_the_mesh_is_refused(mesh):
    with pytest.raises(ValueError, match="does not match the mesh"):
        model_cell_table(mesh, np.ones(mesh.cellCount() - 1), value_name="v")


# -- rectilinear grid models -----------------------------------------------
def test_grid_rows_track_their_own_index_and_extent(tmp_path):
    x_edges = np.linspace(0, 100, 6)
    y_edges = np.linspace(0, 40, 3)
    z_edges = np.linspace(-50, 0, 4)
    shape = (5, 2, 3)
    model = np.arange(np.prod(shape), dtype=float).reshape(shape)

    header, rows = _read(write_grid_model_csv(
        tmp_path / "grid.csv", (x_edges, y_edges, z_edges), model,
        value_name="density_contrast", units="g/cc"))
    assert header[-1] == "density_contrast_g_per_cc"
    assert len(rows) == int(np.prod(shape))

    for row in rows:
        i, j, k = int(row[1]), int(row[2]), int(row[3])
        assert float(row[-1]) == pytest.approx(model[i, j, k])
        for centre, low, high in ((4, 7, 8), (5, 9, 10), (6, 11, 12)):
            assert float(row[low]) <= float(row[centre]) <= float(row[high])


def test_a_fortran_ordered_flat_grid_matches_the_array(tmp_path):
    """SimPEG hands back a flat, Fortran-ordered model; it must land the same."""
    edges = (np.linspace(0, 10, 4), np.linspace(0, 10, 3), np.linspace(-5, 0, 3))
    model = np.arange(3 * 2 * 2, dtype=float).reshape((3, 2, 2))
    from_array = write_grid_model_csv(tmp_path / "a.csv", edges, model, value_name="v")
    from_flat = write_grid_model_csv(
        tmp_path / "b.csv", edges, model.flatten(order="F"), value_name="v")
    assert from_array.read_text(encoding="utf-8") == from_flat.read_text(encoding="utf-8")


def test_a_grid_model_of_the_wrong_size_is_refused(tmp_path):
    edges = (np.linspace(0, 10, 4), np.linspace(0, 10, 3), np.linspace(-5, 0, 3))
    with pytest.raises(ValueError, match="values for a"):
        write_grid_model_csv(tmp_path / "bad.csv", edges, np.ones(5), value_name="v")


# -- layered models --------------------------------------------------------
def test_layer_depths_cumulate_and_the_half_space_stays_open(tmp_path):
    thicknesses = [5.0, 10.0, 20.0]
    model = np.array([[100.0, 30.0, 300.0, 1000.0], [90.0, 25.0, 280.0, 950.0]])
    header, rows = _read(write_layered_model_csv(
        tmp_path / "layered.csv", thicknesses, model,
        value_name="resistivity", units="ohm.m", positions=["S1", "S2"]))

    assert header == ["sounding", "layer", "depth_top_m", "depth_bottom_m",
                      "resistivity_ohm_m"]
    assert len(rows) == 8
    assert [row[2] for row in rows[:4]] == ["0", "5", "15", "35"]
    assert [row[3] for row in rows[:4]] == ["5", "15", "35", ""]
    assert rows[4][0] == "S2"


def test_a_single_sounding_may_be_one_dimensional(tmp_path):
    _, rows = _read(write_layered_model_csv(
        tmp_path / "one.csv", [5.0, 10.0], np.array([100.0, 30.0, 300.0]),
        value_name="resistivity", units="ohm.m"))
    assert len(rows) == 3
    assert rows[0][0] == "0"


def test_too_few_thicknesses_are_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot describe"):
        write_layered_model_csv(
            tmp_path / "bad.csv", [5.0], np.ones((1, 4)), value_name="v")
