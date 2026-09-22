"""Clipping a section to the part of it the data resolve."""

import numpy as np
import pytest

from PyHydroGeophysX.core import section_geometry as sg


class FakeNode:
    def __init__(self, x, y):
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y


class FakeCell:
    def __init__(self, nodes):
        self._nodes = nodes

    def nodes(self):
        return self._nodes


class FakeMesh:
    """The slice of pyGIMLi's mesh interface the geometry helpers use.

    A grid of square cells under a sloping surface, so the tests exercise the
    topography handling without needing a geophysics backend installed.
    """

    def __init__(self, nx=20, nz=8, dx=5.0, dz=2.0, slope=0.5):
        self._centers, self._positions, self._cells = [], [], []
        for i in range(nx):
            x0, x1 = i * dx, (i + 1) * dx
            top0, top1 = slope * x0, slope * x1
            for k in range(nz):
                z0, z1 = -k * dz, -(k + 1) * dz
                corners = [FakeNode(x0, top0 + z0), FakeNode(x1, top1 + z0),
                           FakeNode(x1, top1 + z1), FakeNode(x0, top0 + z1)]
                self._cells.append(FakeCell(corners))
                self._centers.append([0.5 * (x0 + x1),
                                      0.5 * (top0 + top1) + 0.5 * (z0 + z1)])
                self._positions.extend([[n.x(), n.y()] for n in corners])

    def cellCount(self):
        return len(self._cells)

    def cellCenters(self):
        return np.asarray(self._centers, dtype=float)

    def positions(self):
        return np.asarray(self._positions, dtype=float)

    def cells(self):
        return self._cells

    def boundaries(self):
        raise NotImplementedError    # no markers: forces the node-envelope route


@pytest.fixture
def mesh():
    return FakeMesh()


def test_the_surface_follows_the_topography(mesh):
    surface = sg.surface_line(mesh)
    assert sg.surface_elevation_at(0.0, surface) == pytest.approx(0.0, abs=1e-6)
    assert sg.surface_elevation_at(100.0, surface) == pytest.approx(50.0, abs=2.0)


def test_depths_are_measured_below_the_sloping_surface(mesh):
    depths = sg.cell_depths(mesh)
    assert depths.min() > 0.0
    assert depths.max() == pytest.approx(15.0, abs=1.5)
    # The bottom of a cell is deeper than its centre, by half a cell.
    assert sg.cell_bottom_depths(mesh).max() > depths.max()


def test_electrode_positions_define_the_surface_when_given(mesh):
    sensors = np.array([[0.0, 100.0], [100.0, 100.0]])
    assert sg.surface_elevation_at(50.0, sg.surface_line(mesh, sensors)) == pytest.approx(100.0)


def test_the_envelope_fills_the_holes_a_per_cell_cut_leaves(mesh):
    """A shallow cell that happened to fall below the cut is kept, not punched out."""
    depths = sg.cell_depths(mesh)
    coverage = -depths / 10.0                       # falls off with depth
    coverage[np.argmin(depths)] = -99.0             # one bad shallow cell
    raw = sg.coverage_mask(coverage, -1.0)
    clipped = sg.coverage_envelope_mask(mesh, coverage, -1.0)
    assert not raw[np.argmin(depths)]               # the per-cell cut drops it
    assert clipped[np.argmin(depths)]               # the envelope keeps it
    assert clipped.sum() >= raw.sum()


def test_the_envelope_still_cuts_the_deep_part(mesh):
    depths = sg.cell_depths(mesh)
    coverage = -depths / 10.0
    clipped = sg.coverage_envelope_mask(mesh, coverage, -1.0)
    assert not clipped.all()
    assert depths[clipped].max() < depths.max()


def test_the_polygon_is_closed_and_bounded_by_the_surface(mesh):
    depths = sg.cell_depths(mesh)
    polygon = sg.coverage_envelope_polygon(mesh, -depths / 10.0, -1.0)
    assert polygon is not None
    assert polygon[0] == pytest.approx(polygon[-1])      # closed
    surface = sg.surface_line(mesh)
    top = sg.surface_elevation_at(polygon[:, 0], surface)
    assert np.all(polygon[:, 1] <= top + 1e-6)          # never above the ground


def test_a_cut_that_decides_nothing_returns_no_envelope(mesh):
    depths = sg.cell_depths(mesh)
    assert sg.coverage_envelope(mesh, -depths / 10.0, -100.0) is None   # keeps all
    assert sg.coverage_envelope(mesh, -depths / 10.0, 100.0) is None    # keeps none
    # and the mask falls back to the plain per-cell answer rather than raising
    assert sg.coverage_envelope_mask(mesh, -depths / 10.0, -100.0).all()


def test_a_boolean_coverage_is_taken_as_the_mask(mesh):
    depths = sg.cell_depths(mesh)
    rays = depths < 8.0
    assert sg.coverage_mask(rays, 0.0) is rays
    assert sg.coverage_envelope_mask(mesh, rays, 0.0).sum() >= rays.sum()


def test_clipping_an_axes_applies_a_clip_path_and_an_outline(mesh):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from PyHydroGeophysX.visualization.section_clip import clip_section_to_coverage

    depths = sg.cell_depths(mesh)
    centers = sg.cell_centers(mesh)
    fig, ax = plt.subplots()
    drawn = ax.scatter(centers[:, 0], centers[:, 1], c=depths)
    envelope = clip_section_to_coverage(ax, mesh, -depths / 10.0, -1.0, tighten=True)
    assert envelope is not None and "polygon" in envelope
    assert drawn.get_clip_path() is not None
    assert ax.get_ylim()[0] > centers[:, 1].min()       # tightened to the shape
    plt.close(fig)
