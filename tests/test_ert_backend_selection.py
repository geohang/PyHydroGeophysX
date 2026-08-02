"""Backend selection that does not require the optional ADTLERT package."""

import pytest

pytest.importorskip("pygimli")

from PyHydroGeophysX.inversion import ert_inversion  # noqa: E402


def test_requested_adtlert_falls_back_without_cuda(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(
        ert_inversion, "_adtlert_cudss_available", lambda: False
    )

    resolved = ert_inversion._resolve_ert_engine(
        "adtlert", log=messages.append
    )

    assert resolved == "pyhydro"
    assert messages and "original PyHydro ERT engine" in messages[0]


def test_requested_adtlert_is_retained_with_cuda_and_cudss(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ert_inversion, "_adtlert_cudss_available", lambda: True
    )

    assert ert_inversion._resolve_ert_engine("adtlert") == "adtlert"


def test_adtlert_rejects_remote_electrode_indices() -> None:
    container = {
        "a": [0, -1],
        "b": [1, 2],
        "m": [2, 3],
        "n": [3, 4],
    }

    assert not ert_inversion._adtlert_survey_supported(container)
