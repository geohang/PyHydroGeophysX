"""Tests for the independent ERT instrument readers and reciprocal analysis.

The DAS-1 tests run against the real acquisitions under ``examples/data/ERT/DAS``
rather than a fixture, because the point of the reader is that it survives what
an instrument actually writes: failed readings carrying a text message where the
numbers belong, multi-cable electrode addressing, and a column layout that moves
with the acquisition mode.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PyHydroGeophysX.data_processing.ert_formats import (
    _columns_from_mode,
    _read_directives,
    parse_das1,
    parse_res2dinv_general,
    parse_tx0,
    reciprocal_errors,
)

DAS_DIR = Path(__file__).resolve().parents[1] / "examples" / "data" / "ERT" / "DAS"
DAS_FILES = sorted(DAS_DIR.glob("*.Data"))

pytestmark = pytest.mark.skipif(not DAS_FILES, reason="DAS-1 sample acquisitions absent")


@pytest.mark.parametrize("path", DAS_FILES, ids=lambda p: p.name)
def test_every_sample_acquisition_reads_consistently(path: Path) -> None:
    """The five repeats of one monitoring array must agree on its geometry."""
    elec, df = parse_das1(path)

    assert elec.shape == (280, 3)
    assert np.isfinite(elec).all()
    assert 940 <= len(df) <= 950
    assert {"a", "b", "m", "n", "resist", "dev", "ip"} <= set(df.columns)


def test_electrode_indices_address_the_electrode_table() -> None:
    """a/b/m/n are 1-based indices into ``elec``, which is what the callers assume."""
    elec, df = parse_das1(DAS_FILES[0])
    quad = df[["a", "b", "m", "n"]].to_numpy()

    assert quad.min() >= 1
    assert quad.max() <= len(elec)
    # A quadrupole never reuses an electrode in two roles.
    assert all(len(set(row)) == 4 for row in quad)


def test_records_whose_reading_failed_are_dropped() -> None:
    """A failed reading writes a message where the numbers go; it is not a datum."""
    path = DAS_FILES[0]
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = next(i for i, l in enumerate(raw) if l.strip().startswith("#data_start"))
    end = next(i for i, l in enumerate(raw) if l.strip().startswith("#data_end"))
    body = [l for l in raw[start + 1:end] if l.strip() and not l.lstrip().startswith("!")]
    failed = [l for l in body if "out of range" in l]

    _, df = parse_das1(path)

    assert failed, "this sample is expected to contain failed readings"
    assert len(df) == len(body) - len(failed)
    assert np.isfinite(df["resist"]).all()


def test_the_mode_flags_reproduce_the_written_column_map() -> None:
    """Two independent routes to the layout have to agree where both are available.

    The format can write its layout as ``#..._col`` directives, and it can be
    inferred from the acquisition-mode flags. A file carrying both is the only
    chance to check the inference, so it is worth spending a test on.
    """
    lines = DAS_FILES[0].read_text(encoding="utf-8", errors="ignore").splitlines()
    written = _read_directives(lines)
    inferred = _columns_from_mode(lines)

    for key in ("data_res_col", "data_std_res_col", "data_amp_col",
                "data_ip_wind_col", "data_std_ip_col",
                "data_i_curr_col", "data_contact_r_col"):
        assert inferred[key] == written[key], key


def test_a_file_with_no_electrode_section_is_refused() -> None:
    with pytest.raises(ValueError, match="elec_start"):
        parse_das1(__file__)


def test_the_stacking_error_is_reported_as_a_fraction_too() -> None:
    """``dev`` is in ohm, so a consumer cannot tell its units from its size.

    The stacking standard deviations here reach a few tenths of an ohm, which a
    magnitude test reads as a fraction and therefore as a several-percent error,
    where the true relative error is under 0.1 %. Emitting ``error`` alongside
    removes the guess.
    """
    _, df = parse_das1(DAS_FILES[0])

    assert "error" in df.columns
    np.testing.assert_allclose(
        df["error"], df["dev"].abs() / df["resist"].abs(), rtol=1e-12)
    assert df["error"].median() < 0.01 < df["dev"].abs().max()


# --- reciprocal analysis -----------------------------------------------------

def _pair(a, b, m, n, resist):
    return pd.DataFrame({"a": a, "b": b, "m": m, "n": n, "resist": resist})


def test_a_strict_reciprocal_pair_scores_zero() -> None:
    df = _pair([1, 3], [2, 4], [3, 1], [4, 2], [100.0, 100.0])
    out = reciprocal_errors(df, 0.05)

    assert out["reciprocalErrRel"].tolist() == [0.0, 0.0]
    assert out["reciprocalMean"].tolist() == [100.0, 100.0]


def test_reversing_one_pair_flips_the_sign_and_still_pairs() -> None:
    """R = V_MN / I_AB, so reversing one pair alone negates R.

    Without the sign correction this pair reads as a 200 % disagreement and is
    thrown away, which is the opposite of what it deserves.
    """
    df = _pair([1, 4], [2, 3], [3, 1], [4, 2], [100.0, -100.0])
    out = reciprocal_errors(df, 0.05)

    assert out["reciprocalErrRel"].tolist() == [0.0, 0.0]
    assert len(out) == 2


def test_reversing_both_pairs_leaves_the_sign_alone() -> None:
    df = _pair([1, 4], [2, 3], [3, 2], [4, 1], [100.0, 100.0])
    out = reciprocal_errors(df, 0.05)

    assert out["reciprocalErrRel"].tolist() == [0.0, 0.0]


def test_a_disagreeing_pair_is_scored_and_removed() -> None:
    df = _pair([1, 3], [2, 4], [3, 1], [4, 2], [100.0, 120.0])
    scored = reciprocal_errors(df, 0.05, drop_failed=False)
    filtered = reciprocal_errors(df, 0.05)

    assert scored["reciprocalErrRel"].iloc[0] == pytest.approx(20.0 / 110.0)
    assert filtered.empty
    # 20 % disagreement passes a 25 % threshold.
    assert len(reciprocal_errors(df, 0.25)) == 2


def test_an_unpaired_measurement_survives_unscored() -> None:
    """A survey measured in one direction only must pass through untouched."""
    df = _pair([1], [2], [3], [4], [100.0])
    out = reciprocal_errors(df, 0.05)

    assert len(out) == 1
    assert np.isnan(out["reciprocalErrRel"].iloc[0])
    assert out["reciprocalMean"].iloc[0] == pytest.approx(100.0)


def test_distinct_quadrupoles_are_never_paired_with_each_other() -> None:
    """The label has to separate measurements that are not reciprocals."""
    df = _pair([1, 1], [2, 3], [3, 2], [4, 4], [100.0, 500.0])
    out = reciprocal_errors(df, 0.05, drop_failed=False)

    assert out["reciprocalErrRel"].isna().all()


def test_the_real_survey_pairs_without_collisions() -> None:
    """Every group in the sample is a lone measurement or exactly one pair."""
    _, df = parse_das1(DAS_FILES[0])
    out = reciprocal_errors(df, 0.05, drop_failed=False)
    paired = out["reciprocalErrRel"].notna().sum()

    assert paired % 2 == 0
    assert 0 < paired <= len(out)
    assert out["reciprocalErrRel"].dropna().max() < 0.10


def test_a_frame_without_the_needed_columns_passes_through() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = reciprocal_errors(df, 0.05)

    assert len(out) == 1
    assert np.isnan(out["reciprocalErrRel"].iloc[0])


# --- Res2DInv general array --------------------------------------------------

RES2DINV_SURFACE = """Demo line
2.0
11
0
Type of measurement (0=app.resistivity,1=resistance)
0
3
0
0
4 0.0 0.0 6.0 0.0 2.0 0.0 4.0 0.0 100.0
4 2.0 0.0 8.0 0.0 4.0 0.0 6.0 0.0 110.0
4 4.0 0.0 10.0 0.0 6.0 0.0 8.0 0.0 120.0
0
0
"""


def test_res2dinv_general_array_recovers_electrodes_from_positions(tmp_path) -> None:
    path = tmp_path / "line.dat"
    path.write_text(RES2DINV_SURFACE, encoding="utf-8")

    elec, df = parse_res2dinv_general(path)

    assert len(df) == 3
    np.testing.assert_allclose(elec[:, 0], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert df["a"].tolist() == [1, 2, 3]
    assert df["b"].tolist() == [4, 5, 6]
    assert df["rhoa"].tolist() == [100.0, 110.0, 120.0]
    assert df.attrs["electrode_spacing"] == pytest.approx(2.0)


def test_a_borehole_layout_is_refused_rather_than_flattened(tmp_path) -> None:
    """Two electrodes can share an x below surface, so position stops identifying them."""
    path = tmp_path / "borehole.dat"
    path.write_text(RES2DINV_SURFACE.replace(
        "4 2.0 0.0 8.0 0.0 4.0 0.0 6.0 0.0 110.0",
        "4 2.0 -5.0 8.0 0.0 4.0 0.0 6.0 0.0 110.0"), encoding="utf-8")

    with pytest.raises(ValueError, match="more than one elevation"):
        parse_res2dinv_general(path)


# --- tx0 (Lippmann 4-Point Light) --------------------------------------------

# The legend is the real one. It names "n" twice: the N electrode at column 4,
# and the stack count ten columns later. That collision is the whole reason this
# fixture exists, so the stack column is given a value (7) that could never be a
# plausible N for these quadrupoles.
TX0 = """* Data file >demo.tx0<  * 1/1/2024
* Measuring device * type = LGM 4-Point Light 10W
* Electrode positions ******************************************************
* Electrode [  1] x y z (m) =       0.000       0.000       0.000
* Electrode [  2] x y z (m) =       1.000       0.000      -0.100
* Electrode [  3] x y z (m) =       2.000       0.000      -0.200
* Electrode [  4] x y z (m) =       3.000       0.000      -0.300
* Electrode [  5] x y z (m) =       4.000       0.000      -0.400
* Electrode [  6] x y z (m) =       5.000       0.000      -0.500
* num    A    B    M    N        I         U      dU       U90    dU90       rho     phi     f   n nAB
*                               mA        mV       %        mV       %      Ohmm    mrad    Hz   -   -
    1    1    4    2    3   0.1000  55.71550   0.500   3.18665   0.079  3500.708  57.195  2.50   7   1
    2    2    5    3    4   0.1000  90.33770   1.000  -1.29470   4.850  5676.085 -14.332  2.50   7   1
    3    3    6    4    5   0.1000  65.22535   2.500   3.56825   1.438  4098.229  54.706  2.50   7   1
* 188 : Sender open I=  1.000 mA
* 188 : U out of range
"""


def test_tx0_reads_geometry_and_measurements(tmp_path) -> None:
    path = tmp_path / "demo.tx0"
    path.write_text(TX0, encoding="utf-8")

    elec, df = parse_tx0(path)

    assert elec.shape == (6, 3)
    np.testing.assert_allclose(elec[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(elec[:, 2], [0.0, -0.1, -0.2, -0.3, -0.4, -0.5])
    assert len(df) == 3
    assert df["rhoa"].tolist() == [3500.708, 5676.085, 4098.229]
    assert df["ip"].tolist() == [57.195, -14.332, 54.706]


def test_tx0_takes_n_from_the_electrode_column_not_the_stack_count(tmp_path) -> None:
    """The legend names "n" twice; the first one is the electrode.

    Reading the later column instead gives every measurement the same "N", which
    parses cleanly and is entirely wrong.
    """
    path = tmp_path / "demo.tx0"
    path.write_text(TX0, encoding="utf-8")

    _, df = parse_tx0(path)

    assert df["n"].tolist() == [3, 4, 5]
    assert df["n"].nunique() == 3


def test_tx0_reports_the_error_as_a_fraction(tmp_path) -> None:
    """dU is a percentage in the file; a consumer needs a fraction."""
    path = tmp_path / "demo.tx0"
    path.write_text(TX0, encoding="utf-8")

    _, df = parse_tx0(path)

    np.testing.assert_allclose(df["error"], [0.005, 0.010, 0.025])


def test_tx0_ignores_the_units_row_and_the_trailing_warnings(tmp_path) -> None:
    """Both begin with '*', which is what separates them from measurements."""
    path = tmp_path / "demo.tx0"
    path.write_text(TX0, encoding="utf-8")

    _, df = parse_tx0(path)

    assert len(df) == 3
    assert np.isfinite(df[["a", "b", "m", "n", "rhoa"]].to_numpy()).all()


def test_tx0_without_an_electrode_block_is_refused(tmp_path) -> None:
    path = tmp_path / "bare.tx0"
    path.write_text("\n".join(l for l in TX0.splitlines()
                              if "Electrode [" not in l), encoding="utf-8")

    with pytest.raises(ValueError, match="Electrode"):
        parse_tx0(path)


# --- a format with no reader must refuse, not fall through -------------------

def test_a_format_with_no_reader_names_resipy() -> None:
    from PyHydroGeophysX.data_processing.ert_data_agent import _needs_resipy

    with pytest.raises(NotImplementedError) as raised:
        _needs_resipy("Syscal")("anything.csv")

    message = str(raised.value)
    assert "Syscal" in message
    assert "pip install" in message
    # It has to say what still works, or the reader is just a dead end.
    assert "DAS-1" in message


def test_the_generic_fallback_does_not_swallow_that_refusal(tmp_path) -> None:
    """A missing reader is a fact about the format, not a parse failure.

    The loader retries a failed parse with the unified reader, which will
    consume almost any whitespace-delimited table and return a quadrupole built
    from the wrong columns. Letting it answer "this format needs ResIPy" turns an
    honest refusal into silent, plausible-looking nonsense, so the refusal has to
    travel through that retry untouched.
    """
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    original = dict(agent._EMBEDDED_PARSER_MAP)
    try:
        agent._EMBEDDED_PARSER_MAP.pop("DAS-1", None)   # pretend it has no reader
        with pytest.raises(NotImplementedError, match="needs ResIPy"):
            agent._load_ert_embedded_parsers(
                data_file=str(DAS_FILES[0]),
                electrode_file=None,
                project_dir=str(tmp_path),
                instrument="DAS-1",
            )
    finally:
        agent._EMBEDDED_PARSER_MAP.clear()
        agent._EMBEDDED_PARSER_MAP.update(original)


def test_the_install_hint_distinguishes_absent_from_broken() -> None:
    """The fix differs: install it, versus repair the environment it is in."""
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    saved = (agent._RESIPY_ERROR, agent._RESIPY_MISSING)
    try:
        agent._RESIPY_ERROR, agent._RESIPY_MISSING = "No module named 'resipy'", True
        absent = agent.resipy_install_hint()
        agent._RESIPY_ERROR, agent._RESIPY_MISSING = "DLL load failed", False
        broken = agent.resipy_install_hint()
    finally:
        agent._RESIPY_ERROR, agent._RESIPY_MISSING = saved

    assert "did not import" not in absent
    assert "DLL load failed" in broken
    assert "pip install" in absent and "pip install" in broken
