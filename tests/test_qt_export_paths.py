"""Guards on the studio export paths that fail silently when they break.

An export button that raises inside a Qt slot prints to stderr and leaves the
window looking idle, so these failures reach the user as "nothing happened"
rather than as an error. Both cases below shipped that way.
"""

import inspect
import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _require_qt():
    """Skip when the studio pages cannot be imported.

    The pages import PySide6 and pyqtgraph at module scope, so a run without
    the desktop extras cannot reach them at all. Not importorskip: Qt fails
    with a plain ImportError on a machine that has PySide6 but not the GL
    libraries it links against, and that skips only on ModuleNotFoundError.
    """
    try:
        import PySide6.QtGui  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt stack unavailable: {exc}")


def test_the_em_compatibility_shim_reexports_every_public_name():
    """A name missing from the shim is an AttributeError at button-press time.

    ``qt_apps.em_pipeline`` re-exports an explicit list, and the Qt module calls
    through it. ``save_line_csv`` was absent, so exporting an EM *line*
    inversion raised while the single-sounding export beside it worked.
    """
    from PyHydroGeophysX.qt_apps import em_pipeline
    from PyHydroGeophysX.workflows import em1d

    missing = [name for name in em1d.__all__ if not hasattr(em_pipeline, name)]
    assert not missing, f"em_pipeline does not re-export: {', '.join(missing)}"


def test_the_em_module_only_calls_names_the_shim_provides():
    """Every ``em_pipeline.<name>`` the Qt page calls must resolve."""
    import re

    _require_qt()

    from PyHydroGeophysX.qt_apps import em_pipeline
    from PyHydroGeophysX.qt_apps.modules import em_processing

    called = set(re.findall(r"\bem_pipeline\.([A-Za-z_][A-Za-z0-9_]*)",
                            inspect.getsource(em_processing)))
    missing = sorted(name for name in called if not hasattr(em_pipeline, name))
    assert not missing, f"em_processing calls missing names: {', '.join(missing)}"


@pytest.mark.parametrize(
    ("module_path", "function"),
    [
        ("PyHydroGeophysX.qt_apps.modules.ert_processing", "_export_resistivity_model"),
        ("PyHydroGeophysX.qt_apps.modules.seismic_processing", "_export_velocity_model"),
    ],
)
def test_model_exports_stage_pygimli_writes_through_an_ascii_path(module_path, function):
    """A PyGIMLi write on a non-ANSI path must go through ``via_ascii_path``.

    PyGIMLi's writers take a narrow ``std::string``, which Windows resolves
    through the ANSI codepage, so a folder like ``D:\\结果`` is unreachable.
    Calling ``mesh.save`` directly there writes the ``.npy`` and then fails,
    leaving a half-finished export. The seismic module did exactly that while
    the ERT module beside it staged the write.
    """
    import importlib

    _require_qt()

    module = importlib.import_module(module_path)
    page_class = next(
        value for _name, value in vars(module).items()
        if isinstance(value, type) and hasattr(value, function)
    )
    source = inspect.getsource(getattr(page_class, function))
    for call in ("mesh.save", "mesh.exportVTK"):
        assert call not in source.replace("via_ascii_path(mesh.save", "").replace(
            "via_ascii_path(mesh.exportVTK", ""
        ), f"{function} calls {call} directly instead of through via_ascii_path"


def test_the_base_page_answers_the_export_hook_with_nothing():
    """``File > Export Results…`` asks each page what it can write.

    A page without the hook would make the command silently do nothing on that
    module, so the base class supplies the empty answer for every page that has
    no exports of its own. The body ignores ``self``, so it is called unbound
    rather than building a QWidget.
    """
    _require_qt()

    from PyHydroGeophysX.qt_apps.modules.base import BaseModule

    assert BaseModule.export_actions(None) == []


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    [
        ("PyHydroGeophysX.qt_apps.modules.ert_processing", "ERTProcessingModule"),
        ("PyHydroGeophysX.qt_apps.modules.seismic_processing", "SeismicProcessingModule"),
        ("PyHydroGeophysX.qt_apps.modules.em_processing", "EMProcessingModule"),
        ("PyHydroGeophysX.qt_apps.modules.gravmag_processing", "GravMagProcessingModule"),
        ("PyHydroGeophysX.qt_apps.modules.joint_inversion", "JointInversionModule"),
    ],
)
def test_pages_with_exporters_advertise_them(module_path, class_name):
    """Each page that writes model files must offer them through the hook.

    Otherwise its exports stay reachable only from a button on one tab, which is
    the arrangement the single Export command replaced.
    """
    import importlib

    _require_qt()

    from PyHydroGeophysX.qt_apps.modules.base import BaseModule

    page_class = getattr(importlib.import_module(module_path), class_name)
    assert page_class.export_actions is not BaseModule.export_actions, (
        f"{class_name} has export functions but does not advertise them."
    )
    # Every advertised callable must actually exist on the class.
    source = inspect.getsource(page_class.export_actions)
    for name in set(__import__("re").findall(r"self\.(_export[A-Za-z0-9_]*)", source)):
        assert hasattr(page_class, name), f"{class_name}.{name} does not exist."
