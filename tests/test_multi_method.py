import sys
import types

import pytest

pytest.importorskip("pygimli")
pytest.importorskip("simpeg")

from PyHydroGeophysX.inversion.multi_method import GeophysicalInversion


class _DummyEngine:
    def __init__(self, tag: str, **kwargs):
        self.tag = tag
        self.kwargs = kwargs

    def run(self, **kwargs):
        return {"tag": self.tag, "init": self.kwargs, "run": kwargs}


def _module_with_class(class_name: str, tag: str):
    module = types.ModuleType(class_name)

    def _factory(**kwargs):
        return _DummyEngine(tag=tag, **kwargs)

    setattr(module, class_name, _factory)
    return module


def test_geophysical_inversion_dispatch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "PyHydroGeophysX.inversion.ert_inversion",
        _module_with_class("ERTInversion", "ert"),
    )
    monkeypatch.setitem(
        sys.modules,
        "PyHydroGeophysX.inversion.srt_inversion",
        _module_with_class("SRTInversion", "srt"),
    )
    monkeypatch.setitem(
        sys.modules,
        "PyHydroGeophysX.inversion.tdem_inversion",
        _module_with_class("TDEMInversion", "tdem"),
    )
    monkeypatch.setitem(
        sys.modules,
        "PyHydroGeophysX.inversion.fdem_inversion",
        _module_with_class("FDEMInversion", "fdem"),
    )
    monkeypatch.setitem(
        sys.modules,
        "PyHydroGeophysX.inversion.joint_ert_srt",
        _module_with_class("JointERTSRTInversion", "joint_ert_srt"),
    )

    for method in ["ert", "srt", "tdem", "fdem", "joint_ert_srt", "joint"]:
        inv = GeophysicalInversion(method, a=1)
        out = inv.run(b=2)
        expected = "joint_ert_srt" if method == "joint" else method
        assert out["tag"] == expected
        assert out["init"]["a"] == 1
        assert out["run"]["b"] == 2


def test_geophysical_inversion_invalid_method():
    with pytest.raises(ValueError):
        GeophysicalInversion("unknown")
