"""Plan-view resistivity slices, and the masking that keeps them honest.

A survey of a few lines leaves most of a map with no station anywhere near it.
A triangulated interpolation will fill that space from stations hundreds of
metres away, and the result looks like coverage the survey never had. These
tests hold the two properties that stop it: a requested depth is snapped to a
layer the model actually has rather than interpolated between layers, and
ground further than a stated distance from a station is left blank.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
pytest.importorskip("scipy")

from PyHydroGeophysX.visualization import plot_depth_slices  # noqa: E402
from PyHydroGeophysX.visualization.multi_method import (  # noqa: E402
    _median_station_spacing,
)

#: Layer centres of a twenty-layer geometric grid, as a ground survey uses.
DEPTHS = (0.5, 1.6, 2.8, 4.3, 6.0, 8.0, 10.2, 12.9, 16.0, 19.5)


def cells(n=40, spacing=2.0, depths=DEPTHS, below=None):
    """A straight line of stations, each carrying the same layer stack."""
    x = np.arange(n, dtype=float) * spacing
    rows = {"x": [], "y": [], "depth_center_m": [], "resistivity_ohm_m": [],
            "below_doi": []}
    for i, xi in enumerate(x):
        for k, d in enumerate(depths):
            rows["x"].append(xi)
            rows["y"].append(0.0)
            rows["depth_center_m"].append(float(d))
            rows["resistivity_ohm_m"].append(100.0 * (k + 1))
            rows["below_doi"].append(
                0 if below is None else int(d >= below))
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def test_a_requested_depth_snaps_to_a_layer_the_model_has():
    """Interpolating between layer centres would invent a resolution.

    The panel is titled with the depth used rather than the one asked for, so
    the picture cannot claim a slice the layering does not support.
    """
    fig, axes = plot_depth_slices(cells(), depths=(5.0,), ncols=1)
    try:
        title = axes.ravel()[0].get_title()
        assert title.startswith("4.3 m depth"), title   # nearest layer to 5.0
    finally:
        plt.close(fig)


def test_every_requested_depth_gets_a_panel():
    fig, axes = plot_depth_slices(cells(), depths=(1.0, 5.0, 12.0), ncols=3)
    try:
        drawn = [ax.get_title() for ax in axes.ravel() if ax.get_title()]
        assert len(drawn) == 3
        assert [t.split()[0] for t in drawn] == ["0.5", "4.3", "12.9"]
    finally:
        plt.close(fig)


def test_ground_far_from_any_station_is_left_blank():
    """The mask is the whole point; without it the blank fills from far away."""
    data = cells(n=20, spacing=2.0)
    fig, axes = plot_depth_slices(
        data, depths=(4.3,), max_distance=5.0, ncols=1,
        extent=(-100.0, 140.0, -100.0, 100.0), grid=120)
    try:
        mesh = axes.ravel()[0].collections[0]
        values = np.asarray(mesh.get_array(), dtype=float)
        blank = ~np.isfinite(values)
        assert blank.any(), "nothing was masked, so the mask did nothing"
        # The stations occupy 38 m of a 240 x 200 m map, so most of it must go.
        assert blank.mean() > 0.9
    finally:
        plt.close(fig)


def test_a_wider_reach_colours_more_ground():
    data = cells(n=20, spacing=2.0)
    covered = []
    for reach in (5.0, 25.0):
        fig, axes = plot_depth_slices(
            data, depths=(4.3,), max_distance=reach, ncols=1,
            extent=(-100.0, 140.0, -100.0, 100.0), grid=120)
        values = np.asarray(axes.ravel()[0].collections[0].get_array(),
                            dtype=float)
        covered.append(float(np.isfinite(values).mean()))
        plt.close(fig)
    assert covered[1] > covered[0]


def test_one_mask_width_serves_every_panel():
    """A ribbon that changes width between panels must mean the coverage did.

    The mask is about where a station is, and the survey traces the same track
    at every depth. How far a sounding sees sideways is a resolution question
    and belongs in how the answer is read, not in which ground gets coloured.
    """
    data = cells(n=30, spacing=2.0)
    fig, axes = plot_depth_slices(
        data, depths=(0.5, 19.5), ncols=2,
        extent=(-60.0, 120.0, -60.0, 60.0), grid=140)
    try:
        drawn = [ax for ax in axes.ravel() if ax.get_title()]
        covered = [np.isfinite(np.asarray(ax.collections[0].get_array(),
                                          dtype=float)).sum() for ax in drawn]
        assert covered[0] == covered[1], covered
        # And the figure says which width it used, once.
        notes = [t.get_text() for t in fig.texts if "Coloured within" in t.get_text()]
        assert len(notes) == 1, notes
        assert "9 m" in notes[0] or "10 m" in notes[0], notes[0]
    finally:
        plt.close(fig)


def test_cells_below_the_investigation_depth_are_dropped():
    """A slice past the survey's reach shows a gap, not a colour."""
    data = cells(below=10.0)
    fig, axes = plot_depth_slices(data, depths=(4.3,), ncols=1)
    shallow = axes.ravel()[0].get_title()
    plt.close(fig)
    assert "stations" in shallow, shallow

    # At 16 m every cell is past the investigation depth. The panel has to say
    # so rather than quietly drawing the deepest layer that did survive.
    fig, axes = plot_depth_slices(data, depths=(16.0,), ncols=1)
    try:
        title = axes.ravel()[0].get_title()
        assert title.startswith("16.0 m depth"), title
        assert "nothing resolved" in title, title
    finally:
        plt.close(fig)

    # With the cut off, the same depth draws.
    fig, axes = plot_depth_slices(data, depths=(16.0,), drop_below_doi=False,
                                  ncols=1)
    try:
        title = axes.ravel()[0].get_title()
        assert title.startswith("16.0 m depth") and "stations" in title
    finally:
        plt.close(fig)


def test_a_table_without_the_columns_says_which_one_is_missing():
    data = cells()
    del data["resistivity_ohm_m"]
    with pytest.raises(ValueError, match="resistivity_ohm_m"):
        plot_depth_slices(data, depths=(4.3,))


def test_all_panels_share_one_colour_range():
    """Panels that scaled independently could not be compared to each other."""
    fig, axes = plot_depth_slices(cells(), depths=(0.5, 4.3, 16.0), ncols=3)
    try:
        limits = {tuple(ax.collections[0].get_clim())
                  for ax in axes.ravel() if ax.collections}
        assert len(limits) == 1, limits
    finally:
        plt.close(fig)


def test_the_spacing_is_measured_between_stations_not_between_cells():
    """Every station contributes one row per layer, and the duplicates are 0 m."""
    data = cells(n=10, spacing=3.0)
    assert _median_station_spacing(data["x"], data["y"]) == pytest.approx(3.0)


def test_one_station_does_not_divide_by_a_zero_spacing():
    assert _median_station_spacing(np.array([5.0]), np.array([2.0])) > 0
