import pytest

pytest.importorskip("pygimli")
pytest.importorskip("simpeg")


def test_existing_inversion_imports_remain_available():
    from PyHydroGeophysX.inversion import (
        ERTInversion,
        TDEMInversion,
        TimeLapseERTInversion,
        WindowedTimeLapseERTInversion,
    )

    assert ERTInversion is not None
    assert TimeLapseERTInversion is not None
    assert WindowedTimeLapseERTInversion is not None
    assert TDEMInversion is not None
