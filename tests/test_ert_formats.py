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
    parse_sting,
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


def _general_array(path, *, measurement_type=0, ip=False):
    """Twenty Wenner readings on a 2 m line, written as a Res2DInv general array.

    With measurement type 0 the values are apparent resistivities 100, 101, ...;
    with type 1 they are the transfer resistance of a 100 ohm m half-space at
    a = 2 m, 100 / (4 pi). With ``ip`` each row also carries a chargeability.
    """
    rows = []
    for i in range(20):
        a, m, n, b = (2.0 * (i + k) for k in range(4))
        value = 100.0 / (4 * np.pi) if measurement_type else 100.0 + i
        tail = f" {5.0 + 0.1 * i:.1f}" if ip else ""
        rows.append(f"4 {a} 0 {b} 0 {m} 0 {n} 0 {value:.6f}{tail}")
    header = ["Wenner as general array", "2.0", "11", "0",
              "Type of measurement (0=app. resistivity,1=resistance)",
              str(measurement_type), str(len(rows)), "1", "1" if ip else "0"]
    if ip:
        header += ["Chargeability", "mV/V", "0.12,0.26"]
    path.write_text("\n".join(header + rows + ["0", "0", "0", "0"]) + "\n",
                    encoding="utf-8")
    return path


def test_res2dinv_resistances_come_back_as_resistances(tmp_path) -> None:
    """The line under "Type of measurement" says the values are resistances.

    Read as apparent resistivities regardless, a 100 ohm m ground came back as
    7.96 ohm m - off by each reading's geometric factor - and the loaders pass
    rhoa through as given, so nothing downstream could catch it.
    """
    _, df = parse_res2dinv_general(_general_array(tmp_path / "r.dat", measurement_type=1))

    assert "rhoa" not in df.columns
    np.testing.assert_allclose(df["resist"], 100.0 / (4 * np.pi), rtol=1e-6)


def test_res2dinv_resistances_load_as_the_true_resistivity(tmp_path) -> None:
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.data_processing import ert_data_agent as agent
    from PyHydroGeophysX.data_processing.ert_io import standard_to_pg

    ert = agent._load_ert_embedded_parsers(
        data_file=str(_general_array(tmp_path / "r.dat", measurement_type=1)),
        project_dir=str(tmp_path), instrument="ResInv")
    assert ert.metadata["app_res_source"] == "resistance"
    data = standard_to_pg(ert)      # kept alive: data["rhoa"] is a view into it
    np.testing.assert_allclose(np.asarray(data["rhoa"], dtype=float), 100.0, rtol=1e-5)


def test_res2dinv_ip_values_are_read_as_ip(tmp_path) -> None:
    _, df = parse_res2dinv_general(_general_array(tmp_path / "ip.dat", ip=True))

    assert df["rhoa"].tolist()[:3] == [100.0, 101.0, 102.0]
    np.testing.assert_allclose(df["ip"].iloc[:3], [5.0, 5.1, 5.2])
    assert df.attrs["ip_unit"] == "mV/V"


@pytest.mark.parametrize("ip", [False, True], ids=["dc", "ip"])
@pytest.mark.parametrize("picked", ["ResInv", "ABEM-Lund", "BERT"])
def test_a_general_array_file_is_read_as_one_whatever_was_picked(tmp_path, ip, picked) -> None:
    """Every general-array file carries both marks the ABEM guess looked for.

    The "Type of measurement" header line, and rows opening with 4 - there the
    electrode count. Handed to the ABEM reader, a file with IP came back with
    the chargeability (5.0, 5.1, 5.2) as its apparent resistivity, and one
    without IP loaded on no route at all.
    """
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    path = _general_array(tmp_path / "line.dat", ip=ip)
    assert not agent._looks_like_abem_lund_file(path)

    ert = agent._load_ert_embedded_parsers(
        data_file=str(path), project_dir=str(tmp_path), instrument=picked)

    assert ert.instrument == "ResInv"
    assert [o.app_res for o in ert.observations[:3]] == [100.0, 101.0, 102.0]


def test_the_resipy_route_reads_a_general_array_file_as_one(tmp_path) -> None:
    """With ResIPy installed its own Res2DInv reader takes the file (resistances
    plus k); without it, this package's. Either way the section is the same."""
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.data_processing import ert_data_agent as agent
    from PyHydroGeophysX.data_processing.ert_io import standard_to_pg

    ert = agent.load_ert_resipy(
        project_dir=str(tmp_path / "prj"),
        data_file=str(_general_array(tmp_path / "line.dat", ip=True)),
        instrument="ABEM-Lund")

    assert ert.instrument == "ResInv"
    data = standard_to_pg(ert)      # kept alive: data["rhoa"] is a view into it
    np.testing.assert_allclose(np.asarray(data["rhoa"], dtype=float)[:3],
                               [100.0, 101.0, 102.0], rtol=1e-6)


def test_an_abem_style_export_is_still_recognised(tmp_path) -> None:
    """The guess still stands for the layout the ABEM reader was written for."""
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    path = tmp_path / "abem.dat"
    rows = [f"4 {a} 0 {a + 3} 0 {a + 1} 0 {a + 2} 0 0.1 15.9155 100.0" for a in range(10)]
    path.write_text("Type of measurement\n" + "\n".join(rows) + "\n", encoding="utf-8")

    assert agent._looks_like_abem_lund_file(path)


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


# --- AGI SuperSting .stg -----------------------------------------------------

#: Four dipole-dipole records on a six-electrode, 6 m line, written the way a
#: SuperSting R8 writes them: three header lines, then record number, mode,
#: date, time, V/I, stacking error, current, apparent resistivity, command file,
#: the four x/y/z triplets, and the acquisition settings as key=value.
#: Electrodes sit at 0, 6, 12, 18 and 24 m. Records 3 and 4 are one reciprocal
#: pair, record 2 is a negative reading, and every apparent resistivity is the
#: resistance times the analytic dipole-dipole factor for that quadrupole.
STG = """Advanced Geosciences, Inc. SuperSting R8-IP Resistivity meter. S/N: SS1006019 Type: 3D
Firmware version: 01.23.75E Survey period: 20260811 Records: 4
Unit: meter
   1,USER   ,20260811,10:15:00, 1.00000E-01,   1,488, 1.13097E+01,LINE1    , 1.80000E+01, 0.00000E+00, 0.00000E+00, 2.40000E+01, 0.00000E+00, 0.00000E+00, 1.20000E+01, 0.00000E+00, 0.00000E+00, 6.00000E+00, 0.00000E+00, 0.00000E+00,Cmd=29,HV=221,Cyk=2,MTime=1.2,Gain=10,Ch=1
   2,USER   ,20260811,10:15:00,-5.00000E-02,   3,488,-2.26195E+01,LINE1    , 1.80000E+01, 0.00000E+00, 0.00000E+00, 2.40000E+01, 0.00000E+00, 0.00000E+00, 6.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,Cmd=29,HV=221,Cyk=2,MTime=1.2,Gain=20,Ch=2
   3,USER   ,20260811,10:15:12, 1.00000E-01,  12,488, 1.13097E+01,LINE1    , 1.20000E+01, 0.00000E+00, 0.00000E+00, 1.80000E+01, 0.00000E+00, 0.00000E+00, 6.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,Cmd=37,HV=178,Cyk=2,MTime=1.2,Gain=50,Ch=3
   4,USER   ,20260811,10:15:12, 1.02000E-01,   0,488, 1.15359E+01,LINE1    , 6.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.20000E+01, 0.00000E+00, 0.00000E+00, 1.80000E+01, 0.00000E+00, 0.00000E+00,Cmd=37,HV=178,Cyk=2,MTime=1.2,Gain=50,Ch=4
"""


def _sting_east_west(tmp_path):
    """The fixture as written: a line running along x."""
    path = tmp_path / "reference.stg"
    path.write_text(STG, encoding="utf-8")
    return parse_sting(path)


def test_sting_recovers_the_electrode_table_from_the_measurements(tmp_path) -> None:
    """The format writes positions, never an electrode table, so it is rebuilt."""
    path = tmp_path / "line.stg"
    path.write_text(STG, encoding="utf-8")

    elec, df = parse_sting(path)

    assert elec.shape == (5, 3)
    np.testing.assert_allclose(elec[:, 0], [0.0, 6.0, 12.0, 18.0, 24.0])
    quad = df[["a", "b", "m", "n"]].to_numpy()
    assert quad.min() >= 1 and quad.max() <= len(elec)
    # Record 1 is A=18 m, B=24 m, M=12 m, N=6 m on that table.
    assert quad[0].tolist() == [4, 5, 3, 2]


def test_sting_header_lines_are_not_mistaken_for_records(tmp_path) -> None:
    path = tmp_path / "line.stg"
    path.write_text(STG, encoding="utf-8")

    _, df = parse_sting(path)

    assert len(df) == 4 == df.attrs["records_declared"]
    assert df.attrs["lines_skipped"] == 3
    assert df.attrs["unit"] == "meter"


def test_sting_carries_the_instruments_own_geometric_factor(tmp_path) -> None:
    """k = rhoa / R is what the instrument used, and is not the same claim as
    the factor the electrode positions imply."""
    path = tmp_path / "line.stg"
    path.write_text(STG, encoding="utf-8")

    _, df = parse_sting(path)

    np.testing.assert_allclose(df["k"], df["rhoa"] / df["resist"])
    # For record 1 the analytic dipole-dipole factor is 2*pi / (1/6 - 1/12 - 1/12 + 1/18).
    np.testing.assert_allclose(df["k"].iloc[0], 113.097, rtol=1e-4)


def test_sting_reports_the_stacking_error_as_a_fraction_too(tmp_path) -> None:
    """Column 6 is a percentage; a consumer needs the fraction."""
    path = tmp_path / "line.stg"
    path.write_text(STG, encoding="utf-8")

    _, df = parse_sting(path)

    np.testing.assert_allclose(df["error_percent"], [1.0, 3.0, 12.0, 0.0])
    np.testing.assert_allclose(df["error"], [0.01, 0.03, 0.12, 0.0])


def test_sting_keeps_a_negative_reading(tmp_path) -> None:
    """A reversed or noisy reading is a QC decision, not a parse failure."""
    path = tmp_path / "line.stg"
    path.write_text(STG, encoding="utf-8")

    _, df = parse_sting(path)

    assert (df["resist"] < 0).sum() == 1
    assert (df["rhoa"] < 0).sum() == 1


def test_sting_pairs_a_reciprocal_written_as_positions(tmp_path) -> None:
    """Records 3 and 4 are one reciprocal pair, which only works if both map
    onto the same electrode numbers."""
    path = tmp_path / "line.stg"
    path.write_text(STG, encoding="utf-8")

    _, df = parse_sting(path)

    scored = reciprocal_errors(df, max_reciprocal_error=0.5)
    assert np.isfinite(scored["reciprocalErrRel"]).sum() == 2


def test_sting_reads_ip_windows_without_a_column_map(tmp_path) -> None:
    """An IP acquisition appends decay windows after the geometry; a DC one
    appends nothing, and ``ip`` has to exist in both cases."""
    dc_path = tmp_path / "dc.stg"
    dc_path.write_text(STG, encoding="utf-8")
    ip_lines = []
    for line in STG.splitlines():
        if line.startswith(("Advanced", "Firmware", "Unit")):
            ip_lines.append(line)
            continue
        head, settings = line.split(",Cmd=", 1)
        ip_lines.append(f"{head}, 4.500, 3.200, 2.100,Cmd={settings}")
    (tmp_path / "ip.stg").write_text("\n".join(ip_lines), encoding="utf-8")

    _, dc = parse_sting(dc_path)
    _, ip = parse_sting(tmp_path / "ip.stg")

    assert dc["ip"].isna().all()
    assert "ip_1" not in dc.columns
    np.testing.assert_allclose(ip["ip"], 4.5)
    np.testing.assert_allclose(ip["ip_3"], 2.1)
    # The geometry must survive the extra columns untouched.
    assert ip[["a", "b", "m", "n"]].equals(dc[["a", "b", "m", "n"]])


def test_sting_numbers_electrodes_along_the_line_whatever_its_bearing(tmp_path) -> None:
    """A line laid out north-south must get the same numbering as one laid out
    east-west, because the index is only a handle on a position."""
    swapped = []
    for line in STG.splitlines():
        if line.startswith(("Advanced", "Firmware", "Unit")):
            swapped.append(line)
            continue
        head, rest = line.split(",LINE1    ,", 1)
        coords = [c.strip() for c in rest.split(",") if "=" not in c]
        settings = rest[rest.index("Cmd="):]
        flipped = []
        for i in range(0, 12, 3):
            x, y, z = coords[i:i + 3]
            flipped += [y, x, z]          # put the line on the y axis instead
        swapped.append(f"{head},LINE1    , " + ", ".join(flipped) + "," + settings)
    path = tmp_path / "northsouth.stg"
    path.write_text("\n".join(swapped), encoding="utf-8")

    elec, df = parse_sting(path)
    _, reference = _sting_east_west(tmp_path)

    np.testing.assert_allclose(elec[:, 1], [0.0, 6.0, 12.0, 18.0, 24.0])
    assert df[["a", "b", "m", "n"]].equals(reference[["a", "b", "m", "n"]])


def test_a_file_that_is_not_a_sting_export_is_refused(tmp_path) -> None:
    path = tmp_path / "notsting.stg"
    path.write_text("some,other,file\nwith,two,rows\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no SuperSting records"):
        parse_sting(path)


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


def test_the_four_electrode_columns_share_one_numbering(tmp_path) -> None:
    """fielddataline2.dat numbers its electrodes from 0, and only column A has a 0.

    Deciding 0- or 1-based per column shifted A by one and left B, M and N
    alone, so 69 of the 936 readings came out with coincident electrodes.
    """
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    source = DAS_DIR.parent / "Bert" / "fielddataline2.dat"
    if not source.exists():
        pytest.skip("fielddataline2.dat is not present")
    ert = agent._load_ert_embedded_parsers(
        data_file=str(source), electrode_file=None,
        project_dir=str(tmp_path), instrument="BERT")
    quads = np.array([[o.quad.A, o.quad.B, o.quad.M, o.quad.N] for o in ert.observations])
    assert quads[0].tolist() == [1, 38, 19, 20]        # the file's 0 37 18 19, all shifted
    coincident = ((quads[:, 0] == quads[:, 2]) | (quads[:, 0] == quads[:, 3])
                  | (quads[:, 1] == quads[:, 2]) | (quads[:, 1] == quads[:, 3]))
    assert not coincident.any()


def test_the_pygimli_fallback_reads_resistances_and_percent_errors(tmp_path) -> None:
    """pyGIMLi lists rhoa, err and k even when a file has none, zero-filled.

    The fallback read "rhoa in dataMap" as the file having apparent resistivity,
    so a resistance-only file loaded as all zeros; and it copied the error column
    without the percent check the embedded parsers apply, so fielddataline2.dat
    (errors in percent) came back with a median error of 61 %. It is the route
    that file takes whenever ResIPy is installed, because ResIPy cannot parse it.
    """
    from PyHydroGeophysX.data_processing import ert_data_agent as agent
    from PyHydroGeophysX.data_processing.ert_io import standard_to_pg

    pytest.importorskip("pygimli")
    # Twelve electrodes 1 m apart and Wenner quadrupoles: rhoa = 100 Ohm m is a
    # resistance of 100 / (2 pi).
    quads = [(a, a + 3, a + 1, a + 2) for a in range(1, 10)]
    lines = ["12", "# x y z"] + [f"{i:.1f} 0 0" for i in range(12)]
    lines += [str(len(quads)), "# a b m n r err"]
    lines += [f"{a} {b} {m} {n} {100 / (2 * np.pi):.6f} 0.02" for a, b, m, n in quads]
    resistances = tmp_path / "r_only.dat"
    resistances.write_text("\n".join(lines) + "\n")
    ert = agent._load_ert_pygimli(data_file=str(resistances), project_dir=str(tmp_path))
    # Keep the container alive: data["rhoa"] is a view into it, and indexing a
    # temporary left numpy reading freed memory (an access violation).
    data = standard_to_pg(ert)
    assert np.asarray(data["rhoa"], dtype=float) == pytest.approx(100.0, rel=1e-3)

    source = DAS_DIR.parent / "Bert" / "fielddataline2.dat"
    if source.exists():
        ert = agent._load_ert_pygimli(data_file=str(source), project_dir=str(tmp_path))
        assert np.median([o.rel_err for o in ert.observations]) < 0.05


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


def test_an_electrode_file_for_the_survey_lines_up_with_its_electrodes(tmp_path) -> None:
    """The DAS-1 header lists 280 electrodes on nine cables; the survey uses 56.

    electrodes.dat lists those 56, and the pair loaded under ResIPy but failed
    here ("Length of values (56) does not match length of index") - the pair the
    Streamlit app tells users to load. The file's rows go to the electrodes the
    sequence addresses, in order, which is the numbering ResIPy gives them.
    """
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    listed_path = DAS_DIR / "electrodes.dat"
    if not listed_path.exists():
        pytest.skip("electrodes.dat is not present")
    listed = np.loadtxt(listed_path)
    _, df = parse_das1(DAS_FILES[0])
    assert len(df.attrs["survey_electrodes"]) == len(listed) == 56

    ert = agent._load_ert_embedded_parsers(
        data_file=str(DAS_FILES[0]), electrode_file=str(listed_path),
        project_dir=str(tmp_path), instrument="DAS-1")

    assert len(ert.electrodes) == 56
    np.testing.assert_allclose([[e.x, e.y, e.z] for e in ert.electrodes], listed)
    quads = np.array([[o.quad.A, o.quad.B, o.quad.M, o.quad.N] for o in ert.observations])
    assert len(quads) == len(df)
    assert quads.min() >= 1 and quads.max() <= 56
    # The first good reading, 009,02 009,05 009,03 009,04: electrodes 2 5 3 4.
    assert quads[0].tolist() == [2, 5, 3, 4]


def test_an_electrode_file_that_fits_neither_count_says_both(tmp_path) -> None:
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    short = tmp_path / "ten.dat"
    np.savetxt(short, np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)]))

    with pytest.raises(RuntimeError, match=r"lists 10 electrodes.*280.*56"):
        agent._load_ert_embedded_parsers(
            data_file=str(DAS_FILES[0]), electrode_file=str(short),
            project_dir=str(tmp_path), instrument="DAS-1")


def test_the_unified_reader_keeps_an_exponent_without_a_decimal_point(tmp_path) -> None:
    """"2e-05" is one number; read as 2 and -5 it shifted every later column.

    pyGIMLi writes the decimal point (2.00000000000000e-05), so its own files
    were never affected; other writers of the format do not.
    """
    from PyHydroGeophysX.data_processing.ert_data_agent import _unified_ert_parser

    path = tmp_path / "exponent.dat"
    path.write_text("4\n# x y z\n0 0 0\n1 0 0\n2 0 0\n3 0 0\n"
                    "1\n# a b m n r err\n1 4 2 3 2e-05 3E-2\n", encoding="utf-8")

    _, df = _unified_ert_parser(path)

    assert df[["a", "b", "m", "n"]].iloc[0].tolist() == [1, 4, 2, 3]
    assert df["resist"].iloc[0] == pytest.approx(2e-05)
    assert df["dev"].iloc[0] == pytest.approx(0.03)


# --- ResIPy is handed the file types it dispatches on ------------------------

@pytest.mark.parametrize("instrument, ftype", [
    ("Protocol DC", "ProtocolDC"), ("Protocol IP", "ProtocolIP"),
    ("PRIME/RESIMGR", "BGS Prime"), ("ARES", "ARES"), ("ResInv", "ResInv"),
])
def test_resipy_is_handed_the_file_type_it_dispatches_on(
        tmp_path, monkeypatch, instrument, ftype) -> None:
    """Given its GUI labels ("Protocol DC", "ResInv (2D/3D)", "ARES (beta)"),
    ResIPy raised "not implemented yet", so these never reached its parsers."""
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    seen = []

    class StubProject:
        def __init__(self, *args, **kwargs):
            self.surveys = []

        def createSurvey(self, fname=None, ftype=None, **kwargs):
            seen.append(ftype)
            raise RuntimeError("stub: nothing is parsed here")

    monkeypatch.setattr(agent, "_HAS_RESIPY", True)
    monkeypatch.setattr(agent, "Project", StubProject, raising=False)
    monkeypatch.setattr(agent, "_HAS_PYGIMLI", False)      # keep the fallback local
    path = tmp_path / "survey.txt"
    path.write_text("not a survey\n", encoding="utf-8")

    with pytest.raises(Exception):
        agent.load_ert_resipy(project_dir=str(tmp_path / "prj"), data_file=str(path),
                              instrument=instrument)
    assert seen == [ftype]


def test_every_mapped_file_type_is_one_resipy_dispatches_on(tmp_path) -> None:
    survey_module = pytest.importorskip("resipy.Survey")
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    junk = tmp_path / "junk.txt"
    junk.write_text("1 2 3\n", encoding="utf-8")
    for instrument, ftype in agent._FTYPE_MAP.items():
        if instrument in agent._NOT_RESIPY_FTYPES:
            continue
        try:
            survey_module.Survey(str(junk), ftype=ftype)
        except Exception as error:  # noqa: BLE001 - a parse error is expected
            assert "not implemented" not in str(error), (instrument, ftype)


# --- ABEM-Lund rows give positions --------------------------------------------

def _abem_walk(positions):
    """ABEM-style rows for a Wenner walk along ``positions``."""
    rows = [f"4 {positions[i]} 0 {positions[i + 3]} 0 {positions[i + 1]} 0 "
            f"{positions[i + 2]} 0 0.1 15.9155 100.0" for i in range(len(positions) - 3)]
    return "Type of measurement\n" + "\n".join(rows) + "\n"


@pytest.mark.parametrize("positions", [
    [p for p in range(25) if p != 11],
    list(range(-5, 19)),
    [2 * p for p in range(24)],
], ids=["one-electrode-missing", "left-of-origin", "2m"])
def test_abem_positions_become_the_right_electrodes(tmp_path, positions) -> None:
    """Handed over as positions, whole metres that fit 1..n were read as indices:
    a 1 m line with one electrode missing came back with 20 of 21 wrong."""
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    path = tmp_path / "abem.dat"
    path.write_text(_abem_walk(positions), encoding="utf-8")

    ert = agent._load_ert_embedded_parsers(
        data_file=str(path), project_dir=str(tmp_path), instrument="ABEM-Lund")

    x = {e.id: e.x for e in ert.electrodes}
    assert len(ert.electrodes) == len(positions)
    got = [(x[o.quad.A], x[o.quad.B], x[o.quad.M], x[o.quad.N]) for o in ert.observations]
    want = [(positions[i], positions[i + 3], positions[i + 1], positions[i + 2])
            for i in range(len(positions) - 3)]
    assert got == [tuple(float(v) for v in quad) for quad in want]


def test_the_abem_reader_keeps_signs_and_exponents(tmp_path) -> None:
    """"-3" read as 3, and "2e-05" as the two numbers 2 and 5."""
    from PyHydroGeophysX.data_processing.ert_data_agent import _abem_lund_parser

    path = tmp_path / "abem.dat"
    path.write_text("Type of measurement\n4 -3 0 0 0 -2 0 -1 0 0.1 2e-05 100.0\n",
                    encoding="utf-8")

    elec, df = _abem_lund_parser(path)

    np.testing.assert_allclose(elec[:, 0], [-3.0, -2.0, -1.0, 0.0])
    assert df[["a", "b", "m", "n"]].iloc[0].tolist() == [1, 4, 2, 3]
    assert df["resist"].iloc[0] == pytest.approx(2e-05)
    assert df["app"].iloc[0] == pytest.approx(100.0)


# --- a reader's refusal is the answer -----------------------------------------

def _das_without_data_marker(tmp_path):
    text = DAS_FILES[0].read_text(encoding="utf-8", errors="ignore")
    path = tmp_path / "broken.Data"
    path.write_text(text.replace("#data_start", "#data_begin"), encoding="utf-8")
    return path


def test_a_chosen_reader_that_refuses_is_not_second_guessed(tmp_path) -> None:
    """Retried with the unified reader, this file loaded as one reading on one
    electrode, and the error naming the missing section was lost."""
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    with pytest.raises(ValueError, match="#data_start") as raised:
        agent._load_ert_embedded_parsers(
            data_file=str(_das_without_data_marker(tmp_path)),
            project_dir=str(tmp_path), instrument="DAS-1")
    assert "BERT" not in str(raised.value)


def test_the_resipy_route_reports_the_readers_errors(tmp_path) -> None:
    """PyGIMLi's loader cannot read DAS-1; tried anyway, its "'DataMap' object has
    no attribute 'size'" was all the user saw."""
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    with pytest.raises(ValueError, match="#data_start") as raised:
        agent.load_ert_resipy(project_dir=str(tmp_path / "prj"),
                              data_file=str(_das_without_data_marker(tmp_path)),
                              instrument="DAS-1")
    assert "DataMap" not in str(raised.value)


def test_the_studio_loader_does_not_sweep_past_a_refusal(tmp_path) -> None:
    """Its recovery sweep "auto-recovered" the file above as E4D: one reading."""
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.data_processing.ert_io import load_ert_container

    with pytest.raises(ValueError, match="#data_start"):
        load_ert_container(str(_das_without_data_marker(tmp_path)), instrument="DAS-1")


def test_a_tx0_file_is_not_retried_as_a_converted_table(tmp_path) -> None:
    from PyHydroGeophysX.data_processing.ert_data_agent import _lippmann_parser

    path = tmp_path / "nolegend.tx0"
    path.write_text("\n".join(line for line in TX0.splitlines()
                              if not line.lstrip().startswith("* num")), encoding="utf-8")

    with pytest.raises(ValueError, match="legend"):
        _lippmann_parser(str(path))


def test_the_lippmann_entry_still_reads_a_converted_table(tmp_path) -> None:
    from PyHydroGeophysX.data_processing.ert_data_agent import _lippmann_parser

    path = tmp_path / "converted.dat"
    path.write_text("4\n# x y z\n0 0 0\n1 0 0\n2 0 0\n3 0 0\n1\n# a b m n r\n1 4 2 3 10.0\n",
                    encoding="utf-8")

    _, df = _lippmann_parser(str(path))
    assert df["resist"].tolist() == [10.0]


def _abem_lookalike(path, *, unified_blocks):
    """A file the ABEM guess keys on ("Type of measurement", rows opening with 4)
    whose rows the ABEM reader cannot use; optionally a unified survey too."""
    lines = ["# Type of measurement"]
    if unified_blocks:
        lines += ["4", "# x y z", "0 0 0", "1 0 0", "2 0 0", "3 0 0",
                  "1", "# a b m n r", "1 4 2 3 10.0"]
    lines += ["4 " + " ".join(["x"] * 10)] * 3
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_failed_guess_falls_back_to_the_reader_asked_for(tmp_path) -> None:
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    path = _abem_lookalike(tmp_path / "guess.dat", unified_blocks=True)
    assert agent._looks_like_abem_lund_file(path)

    ert = agent._load_ert_embedded_parsers(
        data_file=str(path), project_dir=str(tmp_path), instrument="BERT")

    assert ert.instrument == "BERT"
    assert ert.metadata["parser_used"] == "BERT (fallback)"
    assert [o.app_res for o in ert.observations] == [10.0]


def test_when_the_guess_and_the_fallback_both_fail_the_guess_is_reported(tmp_path) -> None:
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    path = _abem_lookalike(tmp_path / "guess.dat", unified_blocks=False)

    with pytest.raises(ValueError) as raised:
        agent._load_ert_embedded_parsers(
            data_file=str(path), project_dir=str(tmp_path), instrument="BERT")
    assert str(raised.value).startswith(
        "Failed to parse ERT data with ABEM-Lund: No ABEM-Lund measurement rows")


def test_qc_writes_its_table_as_csv_without_a_parquet_engine(tmp_path, monkeypatch) -> None:
    """Parquet needs pyarrow or fastparquet, neither a dependency, and the whole
    load failed over the format of this one diagnostic table."""
    from PyHydroGeophysX.data_processing import ert_data_agent as agent

    ert = agent._load_ert_embedded_parsers(data_file=str(DAS_FILES[0]),
                                           project_dir=str(tmp_path), instrument="DAS-1")

    def no_engine(self, *args, **kwargs):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)
    artifacts = agent.qc_and_visualize(ert, outdir=str(tmp_path / "qc"))
    assert "observations_parquet" not in artifacts
    assert len(pd.read_csv(artifacts["observations_csv"])) == len(ert.observations)
    assert Path(artifacts["standard_json"]).exists()
