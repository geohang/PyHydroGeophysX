import numpy as np
import pytest

from PyHydroGeophysX.data_processing import table_io


@pytest.mark.parametrize("separator", [",", " ", "\t", ";"])
@pytest.mark.parametrize("header", [False, True])
def test_numeric_tables_preserve_rows_and_columns(tmp_path, separator, header):
    path = tmp_path / "survey.txt"
    rows = [["x", "z", "value"]] if header else []
    rows += [["1", "2", "3"], ["4", "5", "6"]]
    path.write_text("\ufeff# survey\n" + "\n".join(separator.join(row) for row in rows), encoding="utf-8")
    np.testing.assert_array_equal(table_io.load_xyz_table(path, 3), [[1, 2, 3], [4, 5, 6]])


def test_bad_cell_is_not_silently_dropped_as_a_column(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_text("x,y,z,value\n1,2,bad,3\n4,5,6,7\n")
    with pytest.raises(ValueError, match="Could not parse numeric table"):
        table_io.load_xyz_table(path, 3)


def test_quoted_csv_numbers_are_preserved(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_text('"x","z","value"\n"1","2","3"\n')
    np.testing.assert_array_equal(table_io.load_xyz_table(path, 3), [[1, 2, 3]])


def test_one_column_is_not_mistaken_for_one_station(tmp_path):
    path = tmp_path / "survey.txt"
    path.write_text("1\n2\n3\n")
    assert table_io.load_2d_array(path).shape == (3, 1)
    with pytest.raises(ValueError, match="at least 3 columns"):
        table_io.load_xyz_table(path, 3)


def test_failed_array_export_keeps_previous_result_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "result.npy"
    table_io.save_npy_atomic(target, [1., 2.])
    def fail(handle, array):
        handle.write(b"incomplete")
        raise ValueError("array cannot be serialized")
    monkeypatch.setattr(table_io.np, "save", fail)
    with pytest.raises(ValueError):
        table_io.save_npy_atomic(target, [3.])
    np.testing.assert_array_equal(np.load(target), [1., 2.])
    assert list(target.parent.iterdir()) == [target]
