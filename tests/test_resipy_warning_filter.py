"""ResIPy's pandas deprecation noise must not reach the Studio log.

``Project.computeFineMeshDepth`` calls ``DataFrame.replace`` and casts straight
to int, so pandas' downcasting FutureWarning says nothing the reader can act on
— but it fires on every load. It is filtered at the ResIPy boundary only, so the
suppression must be narrow in both directions: that one message silenced, every
other warning still delivered.
"""

import warnings

import pytest

from PyHydroGeophysX.data_processing import ert_data_agent as agent


def test_the_filter_silences_only_the_resipy_downcasting_warning() -> None:
    @agent.quiet_resipy
    def noisy() -> str:
        warnings.warn(
            "Downcasting behavior in `replace` is deprecated and will be removed "
            "in a future version.", FutureWarning)
        warnings.warn("an unrelated pandas change", FutureWarning)
        warnings.warn("something the user should see", UserWarning)
        return "returned"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert noisy() == "returned"

    messages = [str(item.message) for item in caught]
    assert not [m for m in messages if "Downcasting behavior" in m]
    assert [m for m in messages if "an unrelated pandas change" in m]
    assert [m for m in messages if "something the user should see" in m]


def test_the_filter_is_not_installed_process_wide() -> None:
    """A library that edits the global filters silences its caller's code too."""
    before = list(warnings.filters)

    @agent.quiet_resipy
    def quiet() -> None:
        pass

    quiet()
    assert warnings.filters == before


def test_the_filter_survives_an_exception() -> None:
    @agent.quiet_resipy
    def raises() -> None:
        raise ValueError("bad file")

    before = list(warnings.filters)
    with pytest.raises(ValueError):
        raises()
    assert warnings.filters == before


@pytest.mark.parametrize("name", ["load_ert_resipy", "export_for_inversion"])
def test_the_resipy_entry_points_stay_wrapped(name: str) -> None:
    """Both functions drive ResIPy, so both have to carry the filter."""
    func = getattr(agent, name)
    assert hasattr(func, "__wrapped__"), f"{name} lost its quiet_resipy decorator"
    assert func.__name__ == name  # functools.wraps kept the identity intact
