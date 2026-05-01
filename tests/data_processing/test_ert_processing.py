from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing.ert_data_agent import (
    calculate_reciprocal_errors,
    export_ert_dataset,
    export_for_inversion,
    load_ert_resipy,
    qc_and_visualize,
)


def _das_example_paths() -> tuple[Path, Path]:
    data_dir = Path("examples/data/ERT/DAS")
    return data_dir / "20171105_1418.Data", data_dir / "electrodes.dat"


def test_load_das_example_with_external_electrodes(tmp_path):
    data_file, electrode_file = _das_example_paths()
    if not data_file.exists() or not electrode_file.exists():
        pytest.skip("Bundled DAS ERT example is not present.")

    ert = load_ert_resipy(
        project_dir=str(tmp_path / "resipy_project"),
        data_file=str(data_file),
        instrument="DAS-1",
        electrode_file=str(electrode_file),
    )

    assert ert.instrument == "DAS-1"
    assert len(ert.electrodes) == 56
    assert len(ert.observations) > 900
    assert ert.metadata["app_res_source"] == "resistance"
    assert ert.electrodes[0].x == pytest.approx(21.375)
    assert ert.electrodes[0].y == pytest.approx(2965.981)

    values = np.asarray([obs.app_res for obs in ert.observations], dtype=float)
    assert np.isfinite(values).all()
    assert values.min() > 0


def test_das_qc_and_pygimli_export(tmp_path):
    data_file, electrode_file = _das_example_paths()
    if not data_file.exists() or not electrode_file.exists():
        pytest.skip("Bundled DAS ERT example is not present.")

    ert = load_ert_resipy(
        project_dir=str(tmp_path / "resipy_project"),
        data_file=str(data_file),
        instrument="DAS-1",
        electrode_file=str(electrode_file),
    )

    artifacts = qc_and_visualize(ert, outdir=str(tmp_path / "qc"))
    for path_text in artifacts.values():
        assert Path(path_text).exists()

    export_file = export_for_inversion(
        ert,
        outdir=str(tmp_path / "export"),
        fmt="pgimli",
        export_strategy="legacy",
    )
    export_path = Path(export_file)
    assert export_path.exists()

    lines = export_path.read_text(encoding="utf-8").splitlines()
    assert int(lines[0]) == 56
    measurement_count = int(lines[58])
    assert measurement_count > 800
    assert lines[59].strip() == "# a b m n r rhoa k err"


def test_das_reciprocal_qc_and_multi_format_export(tmp_path):
    data_file, electrode_file = _das_example_paths()
    if not data_file.exists() or not electrode_file.exists():
        pytest.skip("Bundled DAS ERT example is not present.")

    ert = load_ert_resipy(
        project_dir=str(tmp_path / "resipy_project"),
        data_file=str(data_file),
        instrument="DAS-1",
        electrode_file=str(electrode_file),
    )

    reciprocal = calculate_reciprocal_errors(ert)

    assert len(reciprocal) == len(ert.observations)
    assert reciprocal["reciprocal_error_percent"].notna().sum() > 900
    assert reciprocal["reciprocal_error_percent"].median() < 1.0
    assert reciprocal["reciprocal_pair_count"].min() >= 1

    outputs = export_ert_dataset(
        ert,
        outdir=str(tmp_path / "multi_export"),
        formats=[
            "standard_json",
            "observations_csv",
            "electrodes_csv",
            "reciprocal_csv",
        ],
    )

    assert set(outputs) == {
        "standard_json",
        "observations_csv",
        "electrodes_csv",
        "reciprocal_csv",
    }
    for path_text in outputs.values():
        assert Path(path_text).exists()

    observations_csv = Path(outputs["observations_csv"]).read_text(encoding="utf-8")
    assert "reciprocal_error_percent" in observations_csv.splitlines()[0]
