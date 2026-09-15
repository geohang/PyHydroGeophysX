"""Different survey lengths must work without discarding real observations."""
from pathlib import Path

import numpy as np
import pytest

pg = pytest.importorskip("pygimli")
from pygimli.physics import ert
from PyHydroGeophysX.data_processing.ert_io import (
    align_timelapse_abmn, normalize_for_timelapse,
)
from PyHydroGeophysX.inversion.time_lapse import TimeLapseERTInversion
from PyHydroGeophysX.inversion.windowed import WindowedTimeLapseERTInversion


@pytest.fixture
def surveys(tmp_path):
    scheme = ert.createData(elecs=np.arange(8.), schemeName="dd")
    scheme["k"] = ert.createGeometricFactors(scheme)
    scheme["rhoa"] = np.arange(scheme.size()) + 100.
    scheme["err"] = np.full(scheme.size(), .03)
    scheme["valid"] = np.ones(scheme.size())
    first = pg.DataContainerERT(scheme)
    first.remove(np.arange(first.size()) == 0)
    second = pg.DataContainerERT(scheme)
    second.remove(np.arange(second.size()) < 2)
    paths = []
    for i, data in enumerate([first, second, scheme]):
        path = tmp_path / f"survey_{i}.dat"
        data.save(str(path))
        paths.append(str(path))
    return paths, [first, second, scheme]


def test_native_normalization_and_inversion_accept_unequal_lengths(surveys, tmp_path):
    paths, original = surveys
    root, names, datasets = normalize_for_timelapse(paths, None, str(tmp_path / "out"))
    assert [d.size() for d in datasets] == [d.size() for d in original]
    inv = TimeLapseERTInversion(
        [str(Path(root) / name) for name in names],
        [0., 1., 2.], max_iterations=1, verbose=False,
    )
    inv.setup()
    assert inv.rhos1.size == sum(d.size() for d in original)
    np.testing.assert_allclose(inv.rhos1.ravel(), np.log(np.concatenate([
        np.asarray(d["rhoa"]) for d in original
    ])))
    result = inv.run()
    assert np.all(np.isfinite(result.final_models))


def test_adtlert_union_preserves_data_and_missing_error_through_reload(surveys, tmp_path):
    paths, original = surveys
    root, names, aligned = normalize_for_timelapse(
        paths, None, str(tmp_path / "out"), engine="adtlert", max_error=.1,
    )
    assert [d.size() for d in aligned] == [original[2].size()] * 3
    for data, source in zip(aligned, original):
        lookup = {tuple(int(source[k][i]) for k in "abmn"): i for i in range(source.size())}
        for i in range(data.size()):
            key = tuple(int(data[k][i]) for k in "abmn")
            if key in lookup:
                assert data["rhoa"][i] == source["rhoa"][lookup[key]]
                assert data["err"][i] == pytest.approx(.03)
            else:
                assert data["err"][i] == 1.
    inv = object.__new__(WindowedTimeLapseERTInversion)
    inv.data_dir, inv.ert_files, inv.inversion_params = root, names, {}
    _, observed, errors = inv._load_adtlert_series()
    assert observed.shape == errors.shape == (3, original[2].size())
    assert (errors == 1.).sum() == 3


def test_alignment_reorders_fields_and_rejects_ambiguous_electrodes(surveys):
    _, original = surveys
    source = original[2]
    shuffled = pg.DataContainerERT(source)
    for token in source.dataMap().keys():
        shuffled[str(token)] = np.asarray(source[str(token)])[::-1].copy()
    aligned = align_timelapse_abmn([source, shuffled])
    for token in ("a", "b", "m", "n", "rhoa", "err", "k"):
        np.testing.assert_array_equal(aligned[0][token], aligned[1][token])
    shuffled.setSensorPosition(0, pg.Pos(99., 0.))
    with pytest.raises(ValueError, match="electrode positions"):
        align_timelapse_abmn([source, shuffled])
