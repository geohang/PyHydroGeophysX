"""Tests for the Qt-free desktop I/O helpers."""

import numpy as np
import pytest

from PyHydroGeophysX.qt_apps import io_utils


def test_npy_round_trip(tmp_path):
    arr = np.arange(12, dtype=float).reshape(3, 4)
    p = tmp_path / "a.npy"
    np.save(p, arr)
    assert np.array_equal(io_utils.load_2d_array(p), arr)


def test_npz_reads_stored_array(tmp_path):
    p = tmp_path / "b.npz"
    np.savez(p, only=np.eye(2))
    assert io_utils.load_2d_array(p).shape == (2, 2)


def test_csv_with_header(tmp_path):
    p = tmp_path / "table.csv"
    p.write_text("x,z\n0,100.0\n1,99.5\n2,98.0\n", encoding="utf-8")
    arr = io_utils.load_xyz_table(p, min_cols=2)
    assert arr.shape == (3, 2)
    assert arr[2, 1] == pytest.approx(98.0)


def test_missing_file_raises():
    with pytest.raises(ValueError):
        io_utils.load_2d_array("no_such_file.npy")


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "weird.xyz"
    p.write_text("1 2 3", encoding="utf-8")
    with pytest.raises(ValueError):
        io_utils.load_2d_array(p)


def test_write_json_is_atomic_and_round_trips(tmp_path):
    target = tmp_path / "out" / "result.json"
    io_utils.write_json(target, {"a": 1, "path": tmp_path})
    data = io_utils.read_json(target)
    assert data["a"] == 1
    leftovers = [p for p in target.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_read_json_missing_and_malformed(tmp_path):
    assert io_utils.read_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert io_utils.read_json(bad) is None


def test_write_csv(tmp_path):
    p = io_utils.write_csv(tmp_path / "t.csv", [[1, 2], [3, 4]], header=["a", "b"])
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == "1,2"
    assert lines[2] == "3,4"
