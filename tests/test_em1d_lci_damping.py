"""Zeroth-order damping toward a reference model.

The two roughness operators constrain the shape of a model and say nothing about
its level. Where the data stops resolving, nothing holds the level and the model
drifts to wherever the residual gradient leads. These tests use a forward
operator that is blind to the bottom layer by construction, so the drift is not a
matter of degree: without damping the bottom is unconstrained, with it the bottom
rests on the reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from PyHydroGeophysX.inversion.em1d_lci import SoundingBlock, solve_lci

N_LAYERS = 4
TRUTH = np.array([30.0, 30.0, 30.0, 30.0])
RESOLVED = np.tile(TRUTH[:3], (2, 1))

#: ``model_damping`` is a fraction of ``smoothness``, so a ratio that
#: dominates the roughness tie has to be well above one. At the 0.3
#: smoothness these tests use, 10 puts the smallness weight a hundred
#: times the roughness weight, which is what pins the unseen layer.
STRONG = 10.0


def _blind_block(position: float, truth: np.ndarray = TRUTH) -> SoundingBlock:
    """A sounding whose data depend on every layer but the last.

    ``dobs`` is log10 of each resolved layer's resistivity, so the inverse
    problem is linear, exactly determined above the bottom, and completely
    silent about it.
    """
    seen = N_LAYERS - 1

    def forward(sigma: np.ndarray) -> np.ndarray:
        return np.log10(1.0 / np.asarray(sigma, float)[:seen])

    def jacobian(sigma: np.ndarray) -> np.ndarray:
        s = np.asarray(sigma, float)
        out = np.zeros((seen, N_LAYERS))
        out[np.arange(seen), np.arange(seen)] = -1.0 / (s[:seen] * np.log(10.0))
        return out

    return SoundingBlock(
        forward=forward, jacobian=jacobian,
        dobs=np.log10(truth[:seen]), uncertainty=np.full(seen, 0.01),
        position=position, line=0)


def _solve(smoothness: float = 0.3, **kwargs):
    """Solve with a little vertical roughness, as a real run has.

    Zero roughness would leave the bottom column of the normal matrix exactly
    singular, so the comparison would be against a solve that never moved rather
    than against one that drifted.
    """
    blocks = [_blind_block(0.0), _blind_block(10.0)]
    return solve_lci(
        blocks, N_LAYERS, smoothness=smoothness, lateral_smoothness=0.0,
        starting_resistivity=100.0, max_iterations=40, target_chi2=0.0,
        solver="gauss_newton", verbose=False, **kwargs)


def test_without_damping_the_unseen_layer_follows_its_neighbour() -> None:
    """The baseline. Roughness ties the bottom to the layer above it, so with no
    damping the bottom reports the resolved model rather than anything measured,
    and it lands nowhere near the 100 ohm-m the solve started from."""
    out = _solve()

    np.testing.assert_allclose(out.models[:, :N_LAYERS - 1], RESOLVED, rtol=1e-2)
    assert np.all(out.models[:, -1] < 60.0)


def test_damping_rests_the_unseen_layer_on_the_reference() -> None:
    """With damping the bottom sits at the background half-space, by construction."""
    out = _solve(model_damping=STRONG)

    np.testing.assert_allclose(out.models[:, -1], 100.0, rtol=5e-2)


def test_damping_does_not_move_the_layers_the_data_resolves() -> None:
    """The point is to hold unresolved layers, not to bend resolved ones.

    The data here carries an error of 0.01 against a damping weight of 1, so the
    resolved layers are held some four orders of magnitude more tightly by the
    data than by the reference; a term that visibly moved them would be
    reporting the reference back instead of the ground.
    """
    loose = _solve().models[:, :N_LAYERS - 1]
    damped = _solve(model_damping=1.0).models[:, :N_LAYERS - 1]

    np.testing.assert_allclose(damped, RESOLVED, rtol=1e-2)
    np.testing.assert_allclose(damped, loose, rtol=1e-2)


def test_an_explicit_reference_model_wins_over_the_starting_half_space() -> None:
    reference = np.full((2, N_LAYERS), 250.0)

    out = _solve(model_damping=STRONG, reference_model=reference)

    np.testing.assert_allclose(out.models[:, -1], 250.0, rtol=5e-2)


def test_a_reference_of_the_wrong_shape_falls_back_to_the_half_space() -> None:
    """A silently ignored reference would be worse than an obvious fallback."""
    out = _solve(model_damping=STRONG, reference_model=np.full((5, N_LAYERS), 250.0))

    np.testing.assert_allclose(out.models[:, -1], 100.0, rtol=5e-2)


def test_zero_damping_changes_nothing() -> None:
    np.testing.assert_allclose(_solve(model_damping=0.0).models,
                               _solve().models, rtol=1e-12)


@pytest.mark.parametrize("solver", ["gauss_newton", "trf"])
def test_both_solvers_honour_the_damping(solver: str) -> None:
    """The two paths assemble the term differently: one adds it to the normal
    matrix, the other appends residual rows. They have to agree."""
    blocks = [_blind_block(0.0), _blind_block(10.0)]
    out = solve_lci(
        blocks, N_LAYERS, smoothness=0.3, lateral_smoothness=0.0,
        starting_resistivity=100.0, max_iterations=40, target_chi2=0.0,
        solver=solver, verbose=False, model_damping=STRONG)

    assert np.all(out.models[:, -1] > 70.0)          # pulled toward the reference
    np.testing.assert_allclose(out.models[:, :N_LAYERS - 1], RESOLVED, rtol=5e-2)


def test_the_weight_is_a_fraction_of_the_smoothness() -> None:
    """The parameter is a ratio, so the same value has to mean the same balance
    at two different smoothness settings. An absolute weight would not: 1.0
    against a smoothness of 0.3 is eleven times the roughness weight, and
    against 2.0 it is a quarter of it."""
    a = _solve(smoothness=0.3, model_damping=STRONG).models[:, -1]
    b = _solve(smoothness=1.2, model_damping=STRONG).models[:, -1]

    np.testing.assert_allclose(a, b, rtol=5e-2)


def test_a_ratio_of_one_balances_smallness_against_roughness() -> None:
    """1.0 is SimPEG's alpha_s == alpha_x convention. With the two weights equal,
    the unseen layer settles between the neighbour roughness ties it to (30) and
    the reference damping pulls it to (100), rather than reaching either."""
    bottom = _solve(model_damping=1.0).models[:, -1]

    assert np.all(bottom > 35.0)
    assert np.all(bottom < 90.0)


def test_the_shipped_default_is_on_and_below_equal_weighting() -> None:
    """Measured across four ground TDEM surveys, a uniform smallness term stopped
    the model turning conductive below the depth of investigation on all four,
    most clearly where the undamped deep sat at 6 ohm-m under a resolved 59. It
    is under 1.0 because equal weighting reaches past the unresolved layers. The
    comment on the key carries the numbers for both criteria, including where
    this costs accuracy in the resolved part.
    """
    from PyHydroGeophysX.inversion.em1d import DEFAULT_INVERSION

    shipped = float(DEFAULT_INVERSION["model_damping"])
    assert 0.0 < shipped < 1.0
