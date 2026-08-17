"""The two names every module answers to.

A module is addressed by a navigation key (``ert``) and a result key
(``ert_processing``). The split is deliberate: the first is what the CLI flag,
the agent's ``navigate`` tool, and the Streamlit bridge pass around, and the
second is what a page publishes results under and what stored run records carry.
Nothing was translating between them, so a lookup that received one and needed
the other silently found nothing. These tests pin the mapping and the pairing.
"""

import pytest

from PyHydroGeophysX.workflows.registry import (
    MODULE_DESCRIPTORS,
    module_descriptor_for,
    navigation_key_for,
    result_key_for,
)


def _require_qt():
    """Skip the two tests that reach the Qt pages when Qt is absent.

    The registry itself is pure Python and always tested. The pages import
    PySide6 and pyqtgraph at module scope. Not importorskip: Qt fails with a
    plain ImportError on a machine that has PySide6 but not the GL libraries it
    links against, and that skips only on ModuleNotFoundError.
    """
    try:
        import PySide6.QtGui  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt stack unavailable: {exc}")


@pytest.mark.parametrize("descriptor", list(MODULE_DESCRIPTORS.values()),
                         ids=lambda item: item.navigation_key)
def test_every_spelling_of_a_module_resolves_to_the_same_descriptor(descriptor):
    for spelling in (descriptor.navigation_key, descriptor.result_key,
                     *descriptor.aliases):
        assert module_descriptor_for(spelling) is descriptor
        assert navigation_key_for(spelling) == descriptor.navigation_key
        assert result_key_for(spelling) == descriptor.result_key


def test_the_four_modules_that_use_two_names_translate_both_ways():
    """These are the ones the mismatch actually bit."""
    for navigation, result in (("ert", "ert_processing"),
                               ("seismic", "seismic_processing"),
                               ("em", "em_processing"),
                               ("gravmag", "gravmag_processing")):
        assert result_key_for(navigation) == result
        assert navigation_key_for(result) == navigation


def test_an_unknown_key_passes_through_unchanged():
    """A resolver that invented a key would be worse than one that declines."""
    assert navigation_key_for("not_a_module") == "not_a_module"
    assert result_key_for("not_a_module") == "not_a_module"
    assert module_descriptor_for("not_a_module") is None


def test_no_two_modules_claim_the_same_spelling():
    seen: dict = {}
    for descriptor in MODULE_DESCRIPTORS.values():
        for spelling in (descriptor.navigation_key, descriptor.result_key,
                         *descriptor.aliases):
            owner = seen.setdefault(spelling, descriptor.navigation_key)
            assert owner == descriptor.navigation_key, (
                f"{spelling!r} is claimed by both {owner} and "
                f"{descriptor.navigation_key}"
            )


def test_navigator_keys_match_the_registry():
    """``MODULE_SPECS`` drives the navigator; drift there breaks navigation."""
    _require_qt()

    from PyHydroGeophysX.qt_apps.modules import MODULE_SPECS

    for key in MODULE_DESCRIPTORS:
        assert key in MODULE_SPECS, f"{key} is registered but cannot be navigated to"


@pytest.mark.parametrize(
    ("navigation_key", "module_path", "class_name"),
    [
        ("seismic", "seismic_processing", "SeismicProcessingModule"),
        ("ert", "ert_processing", "ERTProcessingModule"),
        ("em", "em_processing", "EMProcessingModule"),
        ("gravmag", "gravmag_processing", "GravMagProcessingModule"),
        ("mesh3d", "mesh3d_processing", "Mesh3DModule"),
        ("joint_inversion", "joint_inversion", "JointInversionModule"),
        ("hydro_geophysics", "hydro_geophysics", "HydroGeophysicsModule"),
        ("geo_hydrology", "geo_hydrology", "GeoHydrologyModule"),
        ("seismic3d", "seismic3d", "Seismic3DModule"),
    ],
)
def test_each_page_publishes_under_its_registered_result_key(
    navigation_key, module_path, class_name
):
    """A page whose ``module_key`` drifts from the registry loses its results.

    Its results land under a key nothing looks up, which is how
    ``Export Module Result`` came to report "no result" on exactly the four
    modules that had one.
    """
    _require_qt()

    import importlib

    page_class = getattr(
        importlib.import_module(f"PyHydroGeophysX.qt_apps.modules.{module_path}"),
        class_name,
    )
    expected = MODULE_DESCRIPTORS[navigation_key].result_key
    assert page_class.module_key == expected, (
        f"{class_name}.module_key is {page_class.module_key!r}; the registry "
        f"says results for {navigation_key!r} go under {expected!r}"
    )
