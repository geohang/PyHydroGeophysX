"""Regression tests for EM and potential-field example/QC pipelines."""

from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.qt_apps import em_pipeline, gravmag_pipeline, io_utils


ROOT = Path(__file__).resolve().parents[1]


def test_em_examples_and_companion_geometry_load() -> None:
    """Every catalog entry must point to a parseable workbench data file."""
    catalog = em_pipeline.example_catalog()
    assert set(catalog) == {"east_river_vtem", "skytem_bhmar", "synthetic_fdem"}

    expected = {"east_river_vtem": ("TDEM", 30, 22), "skytem_bhmar": ("TDEM", 21, 5),
                "synthetic_fdem": ("FDEM", 16, 1)}
    for key, (method, channels, soundings) in expected.items():
        spec = catalog[key]
        assert Path(spec["path"]).exists()
        sounding = em_pipeline.load_sounding(str(spec["path"]), method)
        abscissa = sounding.get("frequencies", sounding.get("times"))
        assert int(np.asarray(abscissa).size) == channels
        assert int(sounding["n_soundings"]) == soundings

    geometry = em_pipeline.load_line_geometry(str(catalog["east_river_vtem"]["geometry_path"]))
    assert geometry["n"] == 22
    assert geometry["has_heights"] and geometry["has_xy"]


def test_synthetic_fdem_has_deterministic_response() -> None:
    """The committed FDEM example uses default geometry and non-zero quadratures."""
    spec = em_pipeline.example_catalog()["synthetic_fdem"]
    data = em_pipeline.load_sounding(str(spec["path"]), "FDEM")
    assert data["frequencies"][0] == 100.0
    assert data["frequencies"][-1] == 100000.0
    assert np.all(np.isfinite(data["real"]))
    assert np.all(np.isfinite(data["imag"]))
    assert np.any(np.abs(data["real"]) > 0.0)
    assert np.any(np.abs(data["imag"]) > 0.0)


def test_gravmag_examples_qc_and_spatial_sampling() -> None:
    """Bundled potential-field data must produce deterministic finite QC products."""
    examples = {
        "gravity": ROOT / "examples" / "data" / "Gravity_Magnetics" / "bushveld_gravity_disturbance.csv",
        "magnetics": ROOT / "examples" / "data" / "Gravity_Magnetics" / "britain_aeromagnetic_anomaly.csv",
    }
    for path in examples.values():
        table = io_utils.load_xyz_table(path, min_cols=3)
        qc = gravmag_pipeline.qc_products(table[:, 0], table[:, 1], table[:, 2], detrend=1,
                                           nx=30, ny=20)
        assert set(qc["fields"]) == {"Observed", "Regional", "Residual"}
        assert np.isfinite(qc["grids"]["Residual"]["zz"]).all()
        indices = gravmag_pipeline.spatially_balanced_indices(table[:, 0], table[:, 1], 60)
        assert np.array_equal(indices, gravmag_pipeline.spatially_balanced_indices(table[:, 0], table[:, 1], 60))
        assert indices.size == 60
        assert np.ptp(table[indices, 0]) > 0.5 * np.ptp(table[:, 0])
        assert np.ptp(table[indices, 1]) > 0.5 * np.ptp(table[:, 1])


def test_qc_degree_zero_matches_undetrended_inversion_semantics() -> None:
    """A zero detrend must leave the residual equal to the observed field."""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    value = np.array([2.0, 3.0, 4.0, 5.0])
    qc = gravmag_pipeline.qc_products(x, y, value, detrend=0, nx=2, ny=2)
    assert np.array_equal(qc["fields"]["Residual"], value)
    assert np.array_equal(qc["fields"]["Regional"], np.zeros_like(value))


def test_backend_status_has_actionable_shape() -> None:
    """Optional-backend checks always return a JSON-ready state and error text."""
    em_status = em_pipeline.backend_status()
    assert set(em_status["methods"]) == {"FDEM", "TDEM"}
    for item in em_status["methods"].values():
        assert isinstance(item["available"], bool)
        assert isinstance(item["error"], str)
    gravmag_status = gravmag_pipeline.backend_status()
    assert isinstance(gravmag_status["available"], bool)
    assert isinstance(gravmag_status["error"], str)


def test_simpeg_inversion_returns_per_iteration_history() -> None:
    """When SimPEG is installed, the 3D inversion exposes real chi-square history."""
    if not gravmag_pipeline.backend_status()["available"]:
        pytest.skip("SimPEG potential-field backend is not installed.")
    table = io_utils.load_xyz_table(
        ROOT / "examples" / "data" / "Gravity_Magnetics" / "bushveld_gravity_disturbance.csv",
        min_cols=3,
    )
    result = gravmag_pipeline.invert_gravmag(
        table[:80, 0], table[:80, 1], table[:80, 2], "gravity", detrend=1,
        n_xy=8, n_z=4, max_iterations=3, max_stations=30,
    )
    history = result["convergence"]
    assert len(history) >= 1
    assert np.isfinite(history).all()
    assert history[-1] == pytest.approx(result["chi2"])


def test_em_outer_optimizer_returns_per_iteration_history() -> None:
    """The SciPy optimizer around the EM SimPEG forward model exposes its history."""
    if not em_pipeline.backend_status("FDEM")["available"]:
        pytest.skip("FDEM backend is not installed.")
    data = em_pipeline.load_sounding(
        str(em_pipeline.example_catalog()["synthetic_fdem"]["path"]), "FDEM"
    )
    result = em_pipeline.fdem_invert(
        data, em_pipeline.DEFAULT_FDEM,
        {**em_pipeline.DEFAULT_INVERSION, "n_layers": 5, "max_iterations": 5},
    )
    history = result["convergence"]
    assert len(history) >= 1
    assert np.isfinite(history).all()
    assert history[-1] == pytest.approx(result["chi2"])
