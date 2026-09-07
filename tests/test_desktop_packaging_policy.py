"""Execute the desktop spec with collection mocked, without freezing engines."""

from pathlib import Path
import runpy
import sys
import types

import pytest


@pytest.mark.parametrize("include_resipy", [False, True])
def test_full_bundle_requires_explicit_resipy_and_existing_data(monkeypatch, include_resipy):
    root = Path(__file__).resolve().parents[1]
    collected = []
    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_submodules = lambda *a, **kw: []
    def collect_all(name):
        collected.append(name)
        return [], [], []
    hooks.collect_all = collect_all
    monkeypatch.setitem(sys.modules, "PyInstaller", types.ModuleType("PyInstaller"))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils", types.ModuleType("PyInstaller.utils"))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    monkeypatch.setenv("PHGX_BUILD_VARIANT", "full")
    monkeypatch.setenv("PHGX_BUNDLE_RESIPY", "1" if include_resipy else "0")
    analysis = types.SimpleNamespace(pure=[], zipped_data=[], scripts=[], binaries=[], zipfiles=[], datas=[])
    def stub(*args, **kwargs):
        return analysis
    state = runpy.run_path(str(root / "packaging/pyinstaller_studio.spec"), init_globals={
        "SPECPATH": str(root / "packaging"), "Analysis": stub,
        "PYZ": stub, "EXE": stub, "COLLECT": stub, "BUNDLE": stub})
    assert ("resipy" in collected) == include_resipy
    assert ("resipy" in state["excludes"]) != include_resipy
    for source, _ in state["datas"]:
        assert Path(source).exists(), source
    assert any(Path(source).name == "LICENSE" for source, _ in state["datas"])
    assert any(Path(source).name == "THIRD_PARTY_NOTICES.md" for source, _ in state["datas"])
