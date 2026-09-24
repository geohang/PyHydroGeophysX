"""Regression tests for the hydrologic-model readers and hydro-to-ERT coupling.

What went wrong:

* ``MODFLOWWaterContent.get_timestep_info`` skipped ``nuzfcells * 3`` values
  per record whatever the model's layer count, so for any model that is not
  three layers deep it landed mid-record and read data as headers. The record
  header already states the count (ncol * nrow = UZF cells x layers).
* ``hydro_to_ert`` relabelled every cell of the caller's mesh as marker 2
  before appending the boundary, so the ``model_mesh.bms`` that
  ``run_hydro_forward`` saves afterwards had lost its layer markers.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from PyHydroGeophysX.model_output.water_content import MODFLOWWaterContent

_HEADER = np.dtype([
    ("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
    ("text", "S16"), ("maxbound", "<i4"), ("1", "<i4"), ("11", "<i4"),
])


def _write_water_content(folder, idomain, nlay, n_steps):
    n_cells = int(np.count_nonzero(idomain)) * nlay
    with open(os.path.join(folder, "WaterContent"), "wb") as handle:
        for k in range(n_steps):
            np.array([(k + 1, 1, float(k + 1), 10.0 * (k + 1), b"   WATER-CONTENT",
                       n_cells, 1, 1)], dtype=_HEADER).tofile(handle)
            np.full(n_cells, 0.1 + 0.01 * k).tofile(handle)


@pytest.mark.parametrize("nlay", [1, 2, 5])
def test_timestep_info_follows_the_file_for_any_layer_count(tmp_path, nlay):
    idomain = np.array([[1, 0, 1], [1, 1, 0]])
    _write_water_content(str(tmp_path), idomain, nlay, n_steps=4)
    info = MODFLOWWaterContent(str(tmp_path), idomain).get_timestep_info()
    assert [int(kstp) for kstp, _, _, _ in info] == [1, 2, 3, 4]
    assert [float(totim) for _, _, _, totim in info] == [10.0, 20.0, 30.0, 40.0]


def _write_distinct(folder, idomain, nlay, n_steps):
    """A WaterContent file whose every value says which step, layer and cell it is."""
    n_uzf = int(np.count_nonzero(idomain))
    with open(os.path.join(folder, "WaterContent"), "wb") as handle:
        for k in range(n_steps):
            np.array([(k + 1, 1, float(k + 1), 10.0 * (k + 1), b"   WATER-CONTENT",
                       n_uzf * nlay, 1, 1)], dtype=_HEADER).tofile(handle)
            (1000.0 * k + np.arange(n_uzf * nlay, dtype=float)).tofile(handle)


@pytest.mark.parametrize("nlay", [1, 2, 5])
def test_each_value_lands_on_its_own_layer_and_cell(tmp_path, nlay):
    idomain = np.array([[1, 0, 1], [1, 1, 0]])
    _write_distinct(str(tmp_path), idomain, nlay, n_steps=3)
    reader = MODFLOWWaterContent(str(tmp_path), idomain)
    wc = reader.load_time_range(nlay=nlay)
    assert wc.shape == (3, nlay, 2, 3)
    active = idomain != 0
    for k in range(3):
        for layer in range(nlay):
            # Row-major UZF numbering, a layer's cells after the one above.
            expected = 1000.0 * k + layer * 4 + np.arange(4)
            np.testing.assert_array_equal(wc[k, layer][active], expected)
            assert np.all(np.isnan(wc[k, layer][~active]))
    np.testing.assert_array_equal(reader.load_time_range(1, 3, nlay=nlay), wc[1:3])
    np.testing.assert_array_equal(reader.load_timestep(2, nlay=nlay), wc[2])


def test_a_layer_count_the_file_does_not_hold_is_refused(tmp_path):
    idomain = np.array([[1, 0, 1], [1, 1, 0]])
    _write_distinct(str(tmp_path), idomain, nlay=2, n_steps=2)
    with pytest.raises(ValueError, match="the file has 2 layers"):
        MODFLOWWaterContent(str(tmp_path), idomain).load_time_range()   # nlay=3 default


def test_the_shipped_model_reads_as_the_value_by_value_reader_did():
    folder = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "data", "modflow")
    if not os.path.exists(os.path.join(folder, "WaterContent")):
        pytest.skip("examples/data/modflow is not present")
    reader = MODFLOWWaterContent(folder, np.loadtxt(os.path.join(folder, "id.txt")))
    wc = reader.load_time_range(2, 5)
    # The old reader, one value at a time, for the same three records.
    reference = []
    with open(os.path.join(folder, "WaterContent"), "rb") as handle:
        for record in range(5):
            header = np.fromfile(handle, _HEADER, 1)
            values = np.fromfile(handle, "<f8", int(header["maxbound"][0]))
            if record >= 2:
                grid = np.full((3, reader.nrows, reader.ncols), np.nan)
                for layer in range(3):
                    for n in range(reader.nuzfcells):
                        i, j = reader.iuzno_dict_rev[n]
                        grid[layer, i, j] = values[layer * reader.nuzfcells + n]
                reference.append(grid)
    np.testing.assert_array_equal(wc, np.array(reference))


def test_timestep_info_on_the_shipped_three_layer_model():
    folder = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "data", "modflow")
    if not os.path.exists(os.path.join(folder, "WaterContent")):
        pytest.skip("examples/data/modflow is not present")
    reader = MODFLOWWaterContent(folder, np.loadtxt(os.path.join(folder, "id.txt")))
    info = reader.get_timestep_info()
    assert len(info) == 2922
    assert float(info[-1][3]) == 2922.0


def test_hydro_to_ert_keeps_the_callers_layer_markers():
    pg = pytest.importorskip("pygimli")
    from PyHydroGeophysX.Hydro_modular.hydro_to_ert import hydro_to_ert

    mesh = pg.createGrid(x=np.linspace(0, 24, 25), y=np.linspace(-8, 0, 9))
    depth = -np.array(mesh.cellCenters())[:, 1]
    markers = np.where(depth < 3, 3, np.where(depth < 6, 0, 2))
    mesh.setCellMarkers(markers)
    n = mesh.cellCount()
    profile = SimpleNamespace(L_profile=np.array([0.0, 24.0]),
                              surface_profile=np.array([0.0, 0.0]))
    data, res = hydro_to_ert(
        water_content=np.full(n, 0.2), porosity=np.full(n, 0.35), mesh=mesh,
        profile_interpolator=profile, layer_idx=[0, 1, 2], structure=None,
        marker_labels=[3, 0, 2], rho_parameters={}, electrode_spacing=1.0,
        num_electrodes=24, noise_level=0.0, seed=0)
    np.testing.assert_array_equal(np.asarray(mesh.cellMarkers()), markers)
    assert res.shape == (n,) and np.all(res > 0)
    assert data.size() > 0


def test_a_run_that_needs_simpeg_is_refused_before_anything_is_computed(monkeypatch):
    """Without SimPEG the EM and gravity converters are placeholders, so ERT and
    SRT still import. A run that asks for one must stop at once, not after ERT and
    SRT have been computed for nothing."""
    pytest.importorskip("pygimli")
    import PyHydroGeophysX.Hydro_modular as hydro
    from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
    from PyHydroGeophysX.Hydro_modular import hydro_to_geophysics as pipeline

    missing = ModuleNotFoundError("No module named 'simpeg'", name="simpeg")   # as import raises it
    placeholder = hydro._unavailable("hydro_to_tdem", missing)
    monkeypatch.setattr(hydro, "hydro_to_tdem", placeholder)
    started = []
    monkeypatch.setattr(pipeline, "extract_profile", lambda *a, **k: started.append(a))
    with pytest.raises(BackendUnavailable, match="TDEM needs SimPEG"):
        pipeline.run_hydro_forward({}, {}, ["ERT", "TDEM"], (0.0, 0.0), (1.0, 1.0))
    assert started == [], "the run began before it was refused"
    with pytest.raises(ImportError, match="'simpeg' could not be imported"):
        placeholder()                       # called anyway, it says what is missing
