import numpy as np

from PyHydroGeophysX.data_processing import (
    export_results_to_csv,
    read_tem_fast,
)
from PyHydroGeophysX.io import read_tem_fast as legacy_read_tem_fast


class _DummyResult:
    def __init__(self):
        self.final_model = np.array([1.0, 2.0, 3.0])
        self.predicted_data = np.array([0.1, 0.2, 0.3])


def test_data_processing_tem_fast_reader_and_legacy_shim(tmp_path):
    path = tmp_path / "tem_fast.txt"
    np.savetxt(path, np.array([[1.0e-4, 2.0e-8, 1.0e-9], [2.0e-4, 1.5e-8, 1.2e-9]]))

    new_out = read_tem_fast(str(path))
    old_out = legacy_read_tem_fast(str(path))

    assert set(new_out.keys()) == {"time", "data", "uncertainty"}
    assert np.allclose(new_out["time"], old_out["time"])
    assert np.allclose(new_out["data"], old_out["data"])
    assert np.allclose(new_out["uncertainty"], old_out["uncertainty"])


def test_export_results_to_csv_from_data_processing(tmp_path):
    out_file = tmp_path / "result.csv"
    export_results_to_csv(_DummyResult(), str(out_file))

    content = out_file.read_text(encoding="utf-8")
    assert "final_model" in content
    assert "predicted_data" in content
