"""Backend selection that does not require the optional ADTLERT package."""

import pytest

pytest.importorskip("pygimli")

from PyHydroGeophysX.inversion import ert_inversion  # noqa: E402


def test_requested_adtlert_falls_back_without_cuda(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(
        ert_inversion, "_adtlert_cuda_available", lambda: False
    )

    resolved = ert_inversion._resolve_ert_engine(
        "adtlert", log=messages.append
    )

    assert resolved == "pyhydro"
    assert messages and "original PyHydro ERT engine" in messages[0]


def test_requested_adtlert_is_retained_with_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        ert_inversion, "_adtlert_cuda_available", lambda: True
    )

    assert ert_inversion._resolve_ert_engine("adtlert") == "adtlert"
