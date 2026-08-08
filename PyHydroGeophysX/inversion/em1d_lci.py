"""Simultaneous laterally-constrained inversion (LCI) for 1D EM soundings.

Every sounding on a line is solved in one system. The model vector stacks the
per-sounding layer models, the forward operator is block diagonal (a sounding
only sees its own layers), and an explicit coupling operator ties neighbouring
soundings together. That is the same structure as the time-lapse inversion in
:mod:`PyHydroGeophysX.inversion.time_lapse`, with along-line distance in place
of time.

This replaces block-coordinate LCI, where each sounding is re-inverted on its
own against a reference built from its neighbours' models from the *previous*
pass. Under block coordinates the lateral constraint is never enforced while a
sounding is being solved, so the passes chase each other; here it is part of the
system being solved.

Two properties make the simultaneous form affordable:

* SimPEG's ``Simulation1DLayered.getJ`` returns the analytic sensitivity, and
  measures faster than a single forward call. Finite differencing the same
  Jacobian costs one forward per layer, so the analytic route is roughly an
  order of magnitude cheaper per Gauss-Newton iteration.
* The normal matrix is block tridiagonal, so a sparse factorization scales with
  the number of soundings rather than its cube.

Models are parameterized as ``x = log10(resistivity)``, matching the Occam
routine in :mod:`PyHydroGeophysX.inversion.em1d` so that ``smoothness`` and
``lateral_smoothness`` keep the meaning they have there.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop
from PyHydroGeophysX.inversion.lambda_search import search_lambda_for_chi2

LogFn = Callable[[str], None]

#: log10 resistivity bounds, matching ``_occam_1d`` (1 to 1e5 ohm-m).
LOG_RESISTIVITY_BOUNDS: Tuple[float, float] = (0.0, 5.0)

#: Bounds on the smoothness scale the chi-squared search may visit.
SMOOTHNESS_SCALE_BOUNDS: Tuple[float, float] = (1e-4, 1e4)

#: Cumulated sensitivity a depth has to carry to count as investigated.
#:
#: The published value from Christiansen and Auken (2012), who fine-tuned it
#: across ground conductivity meters, DC soundings and airborne TEM and report
#: 0.6 to 1.2 as the range they considered, moving their example by about 15 %.
#: Because their measure lives in logarithmic data and model space it carries no
#: units of its own, which is what lets one number serve every system: at the
#: depth of investigation, moving every layer below it by one e-fold in
#: resistivity shifts the predicted response by 0.8 error bars in total.
#:
#: The threshold is tied to the error model by construction. Doubling the assumed
#: error halves the sensitivity and the section gets shallower, which is the
#: honest response to noisier data. Raise it for a more conservative picture,
#: lower it to see what the deeper part of the model looks like.
#:
#: Vendors do not agree on how conservative to be. On a 71-station ground TDEM
#: survey the TEMcompany software reported depths of investigation about 2.6
#: times shallower than this threshold gives (median 12 m against 37 m); its
#: numbers are reproduced at roughly 8. Christiansen and Auken's value is the
#: default because it is the published one and it travels across systems, but a
#: project that has to line up with an acquisition package's own sections will
#: want to raise it.
DOI_SENSITIVITY_THRESHOLD: float = 0.8

#: Relative chi-squared gain a rougher model has to earn to be preferred.
#: When the target misfit is out of reach — noisy ground data with model error
#: well above the assumed gate errors — the search keeps relaxing the smoothness
#: for gains in the third decimal place, and hands back a railed model that fits
#: no better than the smooth one. Trials within this margin of the best misfit
#: are treated as the same fit, and the smoothest of them wins.
CHI2_EQUIVALENCE: float = 0.02

_LN10 = math.log(10.0)

DEFAULT_LCI = {
    "max_iterations": 20,
    "convergence_tolerance": 0.02,
    "min_iterations": 2,
    "target_chi2": 1.0,
    "chi2_tolerance": 0.2,
    "auto_lambda": True,
    "max_lambda_trials": 5,
    "line_search_steps": 6,
}


@dataclass
class SoundingBlock:
    """One sounding's contribution to the coupled system.

    ``forward(sigma)`` returns the predicted data for a conductivity model and
    ``jacobian(sigma)`` its derivative with respect to conductivity, shaped
    ``(n_data, n_layers)``. Keeping both as callables lets FDEM, single-moment
    TDEM, and joint LM+HM stations share one solver: the caller decides how the
    response is assembled, the solver only needs the pair.
    """

    forward: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray]
    dobs: np.ndarray
    uncertainty: np.ndarray
    position: float = 0.0
    line: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        self.dobs = np.asarray(self.dobs, dtype=float).ravel()
        self.uncertainty = np.clip(
            np.asarray(self.uncertainty, dtype=float).ravel(), 1e-30, None)
        if self.dobs.size != self.uncertainty.size:
            raise ValueError(
                "dobs and uncertainty must have the same length "
                f"({self.dobs.size} vs {self.uncertainty.size}).")


@dataclass
class LCIResult:
    """Outcome of one coupled line inversion."""

    models: np.ndarray                      # (n_soundings, n_layers) resistivity
    chi2: float                             # whole-line, data only
    chi2_per_sounding: np.ndarray
    chi2_history: List[float] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = ""
    smoothness_scale: float = 1.0
    lambda_vertical: float = 0.0
    lambda_lateral: float = 0.0
    n_data: int = 0
    seconds: float = 0.0
    lambda_search: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "models": self.models,
            "chi2": self.chi2,
            "chi2_per_sounding": self.chi2_per_sounding,
            "chi2_history": list(self.chi2_history),
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "smoothness_scale": self.smoothness_scale,
            "lambda_vertical": self.lambda_vertical,
            "lambda_lateral": self.lambda_lateral,
            "n_data": self.n_data,
            "seconds": self.seconds,
            "lambda_search": self.lambda_search,
        }


def lateral_edges(
    positions: Sequence[float],
    lines: Optional[Sequence[int]] = None,
    *,
    reference_distance: float = 10.0,
) -> List[Tuple[int, int, float]]:
    """Return ``(i, j, weight)`` for each pair of neighbouring soundings.

    Soundings are neighbours when they are adjacent in along-line order on the
    same survey line. The weight falls off as ``sqrt(reference_distance / d)``,
    so the penalty it produces scales as ``reference_distance / d``: two
    stations 10 m apart are tied ten times as tightly as two 100 m apart. The
    weight is capped at 1 so that a pair closer than ``reference_distance`` is
    not tied arbitrarily hard.
    """
    pos = np.asarray(positions, dtype=float).ravel()
    n = pos.size
    grp = (np.zeros(n, dtype=int) if lines is None
           else np.asarray(lines, dtype=int).ravel())
    if grp.size != n:
        grp = np.zeros(n, dtype=int)
    ref = max(float(reference_distance), 1e-6)
    edges: List[Tuple[int, int, float]] = []
    for value in np.unique(grp):
        members = np.flatnonzero(grp == value)
        if members.size < 2:
            continue
        members = members[np.argsort(pos[members], kind="stable")]
        for a, b in zip(members[:-1], members[1:]):
            distance = max(abs(float(pos[b] - pos[a])), ref)
            edges.append((int(a), int(b), math.sqrt(ref / distance)))
    return edges


def _vertical_operator(n_soundings: int, n_layers: int):
    """Block-diagonal first difference down each sounding's layer stack."""
    from scipy import sparse

    if n_layers < 2:
        return sparse.csr_matrix((0, n_soundings * n_layers))
    single = sparse.diags(
        [-np.ones(n_layers - 1), np.ones(n_layers - 1)], offsets=[0, 1],
        shape=(n_layers - 1, n_layers), format="csr")
    return sparse.block_diag([single] * n_soundings, format="csr")


def _lateral_operator(edges, n_soundings: int, n_layers: int):
    """One row per (edge, layer): the weighted difference across the edge."""
    from scipy import sparse

    if not edges:
        return sparse.csr_matrix((0, n_soundings * n_layers))
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for e, (a, b, weight) in enumerate(edges):
        for k in range(n_layers):
            row = e * n_layers + k
            rows.extend((row, row))
            cols.extend((a * n_layers + k, b * n_layers + k))
            vals.extend((float(weight), -float(weight)))
    return sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(edges) * n_layers, n_soundings * n_layers))


def _forward_line(blocks: Sequence[SoundingBlock], x: np.ndarray,
                  n_layers: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for s, block in enumerate(blocks):
        sigma = np.power(10.0, -x[s * n_layers:(s + 1) * n_layers])
        out.append(np.asarray(block.forward(sigma), dtype=float).ravel())
    return out


def _sensitivity_line(blocks: Sequence[SoundingBlock], x: np.ndarray,
                      n_layers: int):
    """Block-diagonal d(predicted) / d(log10 resistivity), weighted by 1/sigma_d.

    The chain rule through ``sigma = 10**(-x)`` contributes
    ``d sigma / d x = -ln(10) * sigma``.
    """
    from scipy import sparse

    parts = []
    for s, block in enumerate(blocks):
        xs = x[s * n_layers:(s + 1) * n_layers]
        sigma = np.power(10.0, -xs)
        jac = np.asarray(block.jacobian(sigma), dtype=float)
        if jac.shape != (block.dobs.size, n_layers):
            raise ValueError(
                f"sounding {s} returned a Jacobian of shape {jac.shape}, "
                f"expected {(block.dobs.size, n_layers)}.")
        scaled = (jac * (-_LN10 * sigma)[None, :]
                  ) / block.uncertainty[:, None]
        parts.append(sparse.csr_matrix(scaled))
    return sparse.block_diag(parts, format="csr")


def _misfit(blocks: Sequence[SoundingBlock],
            predicted: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted data residual for the line, plus the per-sounding chi-squared."""
    residual = np.concatenate([
        (pred - block.dobs) / block.uncertainty
        for block, pred in zip(blocks, predicted)
    ]) if blocks else np.zeros(0)
    per = np.asarray([
        float(np.mean(((pred - block.dobs) / block.uncertainty) ** 2))
        if block.dobs.size else float("nan")
        for block, pred in zip(blocks, predicted)
    ], dtype=float)
    return residual, per


def _solve_normal_equations(gram, rhs: np.ndarray) -> np.ndarray:
    from scipy.sparse import linalg as splinalg

    try:
        return np.asarray(splinalg.spsolve(gram.tocsc(), rhs), dtype=float)
    except Exception:  # noqa: BLE001 - fall back to an iterative least squares
        out = splinalg.lsqr(gram, rhs, atol=1e-10, btol=1e-10)[0]
        return np.asarray(out, dtype=float)


def solve_lci(
    blocks: Sequence[SoundingBlock],
    n_layers: int,
    *,
    smoothness: float = 0.3,
    lateral_smoothness: float = 0.3,
    smoothness_scale: float = 1.0,
    reference_distance: float = 10.0,
    initial_model: Optional[np.ndarray] = None,
    starting_resistivity: float = 100.0,
    max_iterations: int = 20,
    convergence_tolerance: float = 0.02,
    min_iterations: int = 2,
    target_chi2: float = 1.0,
    chi2_tolerance: float = 0.2,
    line_search_steps: int = 6,
    target_steps: int = 4,
    bounds: Tuple[float, float] = LOG_RESISTIVITY_BOUNDS,
    verbose: bool = True,
    log: LogFn = _noop,
) -> LCIResult:
    """Solve one coupled line inversion at a fixed smoothness.

    The objective is

    ``||W (F(x) - d)||^2 + lam_v ||Dz x||^2 + lam_l ||Dx x||^2``

    with ``lam_v = (smoothness_scale * smoothness)^2`` and ``lam_l`` likewise
    from ``lateral_smoothness``. Squaring keeps the two knobs numerically
    equivalent to the residual-stacking convention used by the per-sounding
    Occam routine, so an existing ``smoothness=0.3`` setting means the same
    amount of vertical damping here.

    Iteration stops at ``target_chi2``, when the relative chi-squared
    improvement falls below ``convergence_tolerance`` (the plateau rule used
    throughout this package), when the line search cannot find a descent step,
    or at ``max_iterations``. The reason is reported in ``stop_reason``.

    A Gauss-Newton step near the target routinely shoots past it, so a run that
    stopped at the first iterate below ``target_chi2`` would report whatever
    overshoot the step happened to produce. That both fits noise and makes the
    misfit a jumpy function of the smoothness, which the search in
    :func:`invert_lci` then cannot bisect. Set ``target_steps`` above zero and
    an overshooting step is shortened until the misfit lands within
    ``chi2_tolerance`` of the target instead.
    """
    started = time.time()
    blocks = list(blocks)
    if not blocks:
        raise ValueError("solve_lci needs at least one sounding.")
    n_layers = int(n_layers)
    n_soundings = len(blocks)
    n_data = int(sum(block.dobs.size for block in blocks))
    lo, hi = float(min(bounds)), float(max(bounds))
    speak = log if verbose else _noop

    scale = max(float(smoothness_scale), 0.0)
    lam_v = float(scale * max(float(smoothness), 0.0)) ** 2
    lam_l = float(scale * max(float(lateral_smoothness), 0.0)) ** 2

    x = np.empty(n_soundings * n_layers, dtype=float)
    warm = (np.asarray(initial_model, dtype=float)
            if initial_model is not None else None)
    if warm is not None and warm.shape == (n_soundings, n_layers) \
            and np.all(np.isfinite(warm)) and np.all(warm > 0.0):
        x[:] = np.clip(np.log10(warm), lo, hi).ravel()
    else:
        x[:] = float(np.clip(math.log10(max(float(starting_resistivity), 1.0)),
                             lo, hi))

    edges = lateral_edges(
        [block.position for block in blocks],
        [block.line for block in blocks],
        reference_distance=reference_distance)
    Dz = _vertical_operator(n_soundings, n_layers)
    Dx = _lateral_operator(edges, n_soundings, n_layers)
    reg = (lam_v * (Dz.T @ Dz)) + (lam_l * (Dx.T @ Dx))

    def objective(vec: np.ndarray, residual: np.ndarray) -> float:
        return float(residual @ residual) + float(vec @ (reg @ vec))

    predicted = _forward_line(blocks, x, n_layers)
    residual, per_sounding = _misfit(blocks, predicted)
    chi2 = float(residual @ residual) / max(n_data, 1)
    history = [chi2]
    phi = objective(x, residual)
    speak(f"  LCI start: chi2={chi2:.3f}, {n_soundings} soundings, "
          f"{len(edges)} lateral ties, {n_data} data")

    # The discrepancy principle with a tolerance: fitting past the target band
    # is fitting noise. ``target_chi2 <= 0`` disables it and runs to the plateau,
    # which is what a smoothness comparison at a fixed budget wants.
    def at_target(value: float) -> bool:
        return (float(target_chi2) > 0.0
                and value <= float(target_chi2) + abs(float(chi2_tolerance)))

    stop_reason = "max_iterations"
    iterations = 0
    stalls = 0
    for iteration in range(max(1, int(max_iterations))):
        if at_target(chi2):
            stop_reason = "target"
            break
        G = _sensitivity_line(blocks, x, n_layers)
        gram = (G.T @ G) + reg
        gradient = np.asarray(G.T @ residual, dtype=float) + (reg @ x)
        step = _solve_normal_equations(gram, -gradient)
        if not np.all(np.isfinite(step)):
            stop_reason = "singular_system"
            break

        # Armijo backtracking. Clipping happens inside the trial so the accepted
        # decrease belongs to the model that is actually kept.
        slope = float(gradient @ step)
        base_x, base_phi, previous = x, phi, chi2

        def attempt(a: float):
            trial = np.clip(base_x + a * step, lo, hi)
            trial_pred = _forward_line(blocks, trial, n_layers)
            trial_res, trial_per = _misfit(blocks, trial_pred)
            return (trial, trial_pred, trial_res, trial_per,
                    objective(trial, trial_res),
                    float(trial_res @ trial_res) / max(n_data, 1))

        alpha = 1.0
        accepted = None
        for _ in range(max(1, int(line_search_steps))):
            candidate = attempt(alpha)
            if candidate[4] < base_phi + 1e-4 * alpha * min(slope, 0.0):
                accepted = candidate
                break
            alpha *= 0.5
        if accepted is None:
            stop_reason = "line_search"
            break

        # Shorten an overshooting step so the misfit lands near the target
        # rather than wherever the full Gauss-Newton step happened to put it.
        band = abs(float(chi2_tolerance))
        if (int(target_steps) > 0 and float(target_chi2) > 0.0
                and previous > float(target_chi2)
                and accepted[5] < float(target_chi2) - band):
            low, high = 0.0, alpha
            for _ in range(int(target_steps)):
                mid = 0.5 * (low + high)
                probe = attempt(mid)
                if probe[4] < base_phi and \
                        abs(probe[5] - target_chi2) < abs(accepted[5] - target_chi2):
                    accepted, alpha = probe, mid
                if probe[5] > float(target_chi2):
                    low = mid          # still short of the target: step further
                else:
                    high = mid
                if abs(probe[5] - float(target_chi2)) <= band:
                    break

        x, predicted, residual, per_sounding, phi, _ = accepted

        iterations = iteration + 1
        chi2 = float(residual @ residual) / max(n_data, 1)
        history.append(chi2)
        speak(f"  LCI iter {iterations}: chi2={chi2:.3f} (step {alpha:.3g})")
        if at_target(chi2):
            stop_reason = "target"
            break
        # One thin gain is not a plateau. A backtracked step (alpha well under
        # one) often gains little while the model is still moving, so require
        # the improvement to stay small twice running before giving up.
        gained = previous - chi2
        stalls = (stalls + 1
                  if gained <= abs(previous) * float(convergence_tolerance)
                  else 0)
        if iterations >= int(min_iterations) and stalls >= 2:
            stop_reason = "plateau"
            break

    speak(f"  LCI stop: {stop_reason} at chi2={chi2:.3f} "
          f"after {iterations} iteration(s)")
    models = np.power(10.0, x.reshape(n_soundings, n_layers))
    return LCIResult(
        models=models,
        chi2=chi2,
        chi2_per_sounding=per_sounding,
        chi2_history=history,
        iterations=iterations,
        stop_reason=stop_reason,
        smoothness_scale=scale,
        lambda_vertical=lam_v,
        lambda_lateral=lam_l,
        n_data=n_data,
        seconds=time.time() - started,
    )


def invert_lci(
    blocks: Sequence[SoundingBlock],
    n_layers: int,
    *,
    auto_lambda: bool = True,
    target_chi2: float = 1.0,
    chi2_tolerance: float = 0.2,
    max_lambda_trials: int = 5,
    smoothness_scale: float = 1.0,
    scale_bounds: Tuple[float, float] = SMOOTHNESS_SCALE_BOUNDS,
    verbose: bool = True,
    log: LogFn = _noop,
    **kwargs: Any,
) -> LCIResult:
    """Run the coupled line inversion, relaxing the smoothness if needed.

    The fixed-smoothness run happens first and is always kept. When it misses
    ``target_chi2`` by more than ``chi2_tolerance`` and ``auto_lambda`` is set,
    a bracket-and-bisect search over a single scale multiplying both smoothness
    terms looks for a better fit. Each trial warm-starts from the models of the
    nearest scale already solved, which is what keeps the search affordable.

    The returned result is whichever run landed closest to the target. When no
    trial reached the target band, trials whose misfit is within
    ``CHI2_EQUIVALENCE`` of the best are treated as equally good fits and the
    smoothest of them is returned instead of the roughest. The trial record
    lives in ``lambda_search``.
    """
    solved: Dict[float, LCIResult] = {}

    def nearest(scale: float) -> Optional[np.ndarray]:
        if not solved:
            return None
        key = min(solved, key=lambda k: abs(math.log(max(k, 1e-30))
                                            - math.log(max(scale, 1e-30))))
        return solved[key].models

    def run(scale: float, warm: Optional[np.ndarray]) -> LCIResult:
        out = solve_lci(blocks, n_layers, smoothness_scale=scale,
                        initial_model=warm, target_chi2=target_chi2,
                        chi2_tolerance=chi2_tolerance,
                        verbose=verbose, log=log, **kwargs)
        solved[float(scale)] = out
        return out

    base = run(float(smoothness_scale), kwargs.pop("initial_model", None))
    best = base
    if not auto_lambda or abs(base.chi2 - target_chi2) <= abs(chi2_tolerance):
        best.lambda_search = {
            "status": "skipped" if not auto_lambda else "converged",
            "trials": [{"lambda": base.smoothness_scale, "chi2": base.chi2}],
            "reason": "", "fixed_chi2": base.chi2,
            "fixed_scale": base.smoothness_scale,
        }
        return best

    log(f"  Smoothness scale {base.smoothness_scale:g} gives chi2="
        f"{base.chi2:.2f}; searching.")

    def evaluate(scale: float) -> float:
        out = run(scale, nearest(scale))
        nonlocal best
        if abs(out.chi2 - target_chi2) < abs(best.chi2 - target_chi2):
            best = out
        return out.chi2

    search = search_lambda_for_chi2(
        evaluate,
        start_lambda=float(base.smoothness_scale),
        start_chi2=float(base.chi2),
        target_chi2=float(target_chi2),
        tolerance=float(chi2_tolerance),
        max_trials=int(max_lambda_trials),
        bounds=scale_bounds,
        log=_noop,
    )
    search["fixed_chi2"] = float(base.chi2)
    search["fixed_scale"] = float(base.smoothness_scale)
    if search.get("status") == "best_effort" and best.chi2 > float(target_chi2):
        smoothest = _smoothest_equivalent(solved, best)
        if smoothest is not best:
            log(f"  No trial reached the chi2 target, and scale "
                f"{smoothest.smoothness_scale:.3g} (chi2={smoothest.chi2:.3f}) "
                f"fits within {100 * CHI2_EQUIVALENCE:.0f}% of the best trial at "
                f"{best.smoothness_scale:.3g} (chi2={best.chi2:.3f}); keeping the "
                f"smoother model.")
            search["parsimonious_scale"] = float(smoothest.smoothness_scale)
            best = smoothest
    best.lambda_search = search
    if best is not base:
        log(f"  Smoothness scale {best.smoothness_scale:.3g}: chi2="
            f"{best.chi2:.2f} (fixed {base.smoothness_scale:g} gave "
            f"{base.chi2:.2f}).")
    else:
        log(f"  Search kept the fixed smoothness (chi2={base.chi2:.2f}): "
            f"{search.get('reason') or 'no trial did better'}.")
    return best


def mask_sounding_block(block: SoundingBlock, keep: Sequence[bool]) -> SoundingBlock:
    """A copy of *block* carrying only the gates flagged in ``keep``.

    The forward and Jacobian are wrapped rather than rebuilt, so the SimPEG
    simulation behind them is reused as is and dropping a gate costs nothing.
    A block may end up with no data at all: the sounding then contributes no
    residual and its model comes entirely from its neighbours through the
    lateral constraint, which is the honest outcome when every one of its gates
    was rejected.
    """
    index = np.flatnonzero(np.asarray(keep, dtype=bool).ravel())
    forward, jacobian = block.forward, block.jacobian
    return SoundingBlock(
        forward=lambda sigma: np.asarray(
            forward(sigma), dtype=float).ravel()[index],
        jacobian=lambda sigma: np.asarray(
            jacobian(sigma), dtype=float)[index, :],
        dobs=block.dobs[index],
        uncertainty=block.uncertainty[index],
        position=block.position,
        line=block.line,
        label=block.label,
    )


def cumulated_sensitivity(block: SoundingBlock, model: np.ndarray) -> np.ndarray:
    """Cumulated sensitivity after Christiansen and Auken (2012).

    Their construction, equations 2, 3 and 5 of *A global measure for depth of
    investigation*, GEOPHYSICS 77(4), WB171-WB177:

    * ``G_ij = d log(data_i) / d log(rho_j)``, the Jacobian of the final model in
      logarithmic data and model space. Working in logarithms on both sides is
      what makes the resulting number comparable between data types, and so lets
      one absolute threshold serve every system.
    * ``s_j = sum_i |G_ij| / sigma_i``, summed over all N data with ``sigma_i``
      the standard deviation of the *log* datum, which is its relative error.
    * ``S_j = sum_{k >= j} s_k``, cumulated from the bottom layer upward. Entry
      ``j`` is therefore the total information the data carry about layer ``j``
      and everything below it, counted in error bars.

    Their equation 4 divides ``s`` by the layer thickness; the paper uses that
    only for plotting, and the cumulated quantity here is built from equation 3,
    which is also what keeps it independent of the layer grid: split a layer in
    two and its sensitivity splits with it, so the value at a given depth does
    not move (measured at 0.2 to 2.5 % across a 2x refinement on real ground
    TDEM).

    Only the data part of the Jacobian takes part, so a depth that clears the
    threshold is one the measurements reach, not one the lateral or vertical
    constraint filled in. The Jacobian is SimPEG's analytic sensitivity, the
    same one the coupled solver uses, so this costs about one forward
    evaluation per sounding.

    Their step 2, sub-discretizing a few-layer model before differentiating, is
    unnecessary here: the paper skips it for smooth models, and these are solved
    on a fixed grid of a dozen layers or more.
    """
    layers = int(np.size(model))
    if not block.dobs.size or layers == 0:
        return np.zeros(layers, dtype=float)
    resistivity = np.clip(np.asarray(model, dtype=float).ravel(), 1e-12, None)
    conductivity = 1.0 / resistivity
    jacobian = np.asarray(block.jacobian(conductivity), dtype=float)
    if jacobian.shape != (block.dobs.size, layers):
        return np.zeros(layers, dtype=float)
    predicted = np.asarray(block.forward(conductivity), dtype=float).ravel()
    # d log(d)/d log(rho) = -(d(data)/d(sigma)) * sigma / data. A TDEM decay can
    # cross zero at an offset receiver, where the log derivative is undefined;
    # floor the divisor at the datum's own error bar, below which its logarithm
    # means nothing anyway, so such a gate contributes a bounded sensitivity
    # rather than an infinity.
    scale = np.maximum(np.abs(predicted), block.uncertainty)
    relative = np.maximum(block.uncertainty / np.maximum(np.abs(block.dobs), 1e-300),
                          1e-6)
    weighted = (jacobian * conductivity[None, :]) / (scale * relative)[:, None]
    per_layer = np.abs(weighted).sum(axis=0)
    if not np.isfinite(per_layer).all():
        per_layer = np.nan_to_num(per_layer, nan=0.0, posinf=0.0, neginf=0.0)
    return np.cumsum(per_layer[::-1])[::-1]


def sensitivity_doi(block: SoundingBlock, model: np.ndarray,
                    depth_edges: np.ndarray, *,
                    threshold: float = DOI_SENSITIVITY_THRESHOLD) -> float:
    """Depth of investigation: the bottom of the deepest layer still resolved.

    Returns ``0.0`` when even the shallowest layer misses the threshold, which
    is the honest answer for a station whose gates were all rejected: it has no
    depth of investigation, and its column carries only what its neighbours say.
    """
    cumulated = cumulated_sensitivity(block, model)
    edges = np.asarray(depth_edges, dtype=float).ravel()
    resolved = np.flatnonzero(cumulated >= float(threshold))
    if not resolved.size or resolved[-1] + 1 >= edges.size:
        return 0.0 if not resolved.size else float(edges[-1])
    return float(edges[resolved[-1] + 1])


def weighted_residuals(blocks: Sequence[SoundingBlock],
                       models: np.ndarray) -> np.ndarray:
    """``(predicted - observed) / uncertainty`` for every gate on the line."""
    parts: List[np.ndarray] = []
    for block, model in zip(blocks, np.asarray(models, dtype=float)):
        if not block.dobs.size:
            parts.append(np.zeros(0, dtype=float))
            continue
        sigma = 1.0 / np.clip(model, 1e-12, None)
        predicted = np.asarray(block.forward(sigma), dtype=float).ravel()
        parts.append((predicted - block.dobs) / block.uncertainty)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def invert_lci_rejecting_outliers(
    blocks: Sequence[SoundingBlock],
    n_layers: int,
    *,
    threshold: float = 3.0,
    passes: int = 2,
    min_fraction: float = 0.5,
    min_gates: int = 3,
    log: LogFn = _noop,
    **kwargs: Any,
) -> Tuple[LCIResult, List[SoundingBlock], Dict[str, Any]]:
    """Solve the line, then drop the gates the model cannot explain and re-solve.

    This is the EM counterpart of the ERT outlier pass. Each cycle removes the
    gates whose weighted residual exceeds ``threshold`` and solves again, warm
    started from the model just found and at the smoothness the first solve
    settled on, so a rejection pass costs far less than the first one.
    Two floors bound the cut. ``min_fraction`` is the survey-wide one: when more
    gates exceed the threshold than it allows, the pass drops the worst offenders
    up to the limit rather than refusing, because a bad fit is exactly when
    rejection is wanted. ``min_gates`` is the per-sounding one: a station never
    loses so many gates that fewer than this remain (or all of them, if it
    arrived with fewer). Without it a station holding one or two gates loses them
    both on the first pass, and the section then has a hole where only the
    lateral constraint is left holding the model. Its best-fitting gates are the
    ones kept.

    A single noisy gate on a TDEM station carries a lot of weight (a station may
    hold only a handful), so cutting is per gate rather than per sounding.

    Returns ``(outcome, kept_blocks, info)``.
    """
    blocks = list(blocks)
    n_start = int(sum(block.dobs.size for block in blocks))
    floor = int(math.ceil(max(0.0, float(min_fraction)) * n_start))
    info: Dict[str, Any] = {
        "enabled": True, "threshold": float(threshold), "n_start": n_start,
        "floor": floor, "min_gates": int(min_gates), "passes": [], "dropped": 0,
        "kept": n_start, "stopped_because": "", "limited_by_floor": False,
        "floored_soundings": 0,
    }
    outcome = invert_lci(blocks, n_layers, log=log, **kwargs)
    info["initial"] = {
        "chi2": float(outcome.chi2),
        "smoothness_scale": float(outcome.smoothness_scale),
        "convergence": [float(value) for value in outcome.chi2_history],
        "n_data": n_start,
    }
    # Re-solves stay at the smoothness the first solve settled on, so the change
    # between passes is the data set and nothing else. The caller's warm start
    # belongs to the first solve only; later passes start from what it found.
    rerun = {**kwargs, "auto_lambda": False,
             "smoothness_scale": float(outcome.smoothness_scale)}
    rerun.pop("initial_model", None)
    for index in range(1, max(0, int(passes)) + 1):
        current = int(sum(block.dobs.size for block in blocks))
        allowed = current - floor
        if allowed <= 0:
            info["stopped_because"] = (
                f"at the {int(min_fraction * 100)} % floor of {floor} gates")
            break
        residual = weighted_residuals(blocks, outcome.models)
        drop = np.abs(residual) > float(threshold)
        n_drop = int(drop.sum())
        if n_drop == 0:
            info["stopped_because"] = "nothing left above the cut"
            break
        if n_drop > allowed:
            worst = np.argsort(-np.abs(residual))[:allowed]
            drop = np.zeros(residual.size, dtype=bool)
            drop[worst] = True
            n_drop = allowed
            info["limited_by_floor"] = True
            log(f"  more gates exceed {threshold:g} sigma than the "
                f"{int(min_fraction * 100)} % floor allows; dropping the worst {n_drop}")
        keep = ~drop
        # Put back the best-fitting gates of any sounding the cut would have
        # emptied, so no station is left with the lateral constraint alone.
        floored = 0
        offset = 0
        for block in blocks:
            size = block.dobs.size
            local = keep[offset:offset + size]
            wanted = min(int(min_gates), size)
            if size and int(local.sum()) < wanted:
                best = np.argsort(np.abs(residual[offset:offset + size]))[:wanted]
                local[:] = False
                local[best] = True
                floored += 1
            offset += size
        if floored:
            info["floored_soundings"] = int(info["floored_soundings"]) + floored
            log(f"  kept the {min_gates} best gate(s) at {floored} sounding(s) the "
                f"cut would have emptied")
        n_drop = int((~keep).sum())
        if n_drop == 0:
            info["stopped_because"] = (
                f"every gate over the cut is at a sounding already down to its "
                f"{min_gates}-gate floor")
            break
        trimmed: List[SoundingBlock] = []
        offset = 0
        for block in blocks:
            size = block.dobs.size
            trimmed.append(mask_sounding_block(block, keep[offset:offset + size]))
            offset += size
        blocks = trimmed
        outcome = invert_lci(blocks, n_layers, log=log,
                             initial_model=outcome.models, **rerun)
        kept = int(sum(block.dobs.size for block in blocks))
        info["passes"].append({
            "pass": index, "dropped": n_drop, "kept": kept,
            "chi2": float(outcome.chi2),
            "convergence": [float(value) for value in outcome.chi2_history],
        })
        log(f"  rejected {n_drop} gate(s) over {threshold:g} sigma, "
            f"{kept} left -> chi2 {outcome.chi2:.3f}")
        if info["limited_by_floor"]:
            # A bounded cut leaves known-bad gates in, so stop rather than nibble
            # at the floor pass after pass without ever clearing the outliers.
            info["stopped_because"] = (
                f"the {int(min_fraction * 100)} % floor capped the cut; gates beyond "
                f"{threshold:g} sigma remain")
            break
    info["kept"] = int(sum(block.dobs.size for block in blocks))
    info["dropped"] = n_start - info["kept"]
    if not info["stopped_because"]:
        info["stopped_because"] = f"all {int(passes)} pass(es) used"
    return outcome, blocks, info


def _smoothest_equivalent(solved: Dict[float, LCIResult],
                          best: LCIResult) -> LCIResult:
    """The largest smoothness scale that fits as well as ``best``.

    "As well" means within :data:`CHI2_EQUIVALENCE` in relative terms. Used only
    when the chi-squared target was never reached, where the search would
    otherwise trade a fraction of a percent of misfit for a far rougher model.
    """
    margin = float(best.chi2) * (1.0 + CHI2_EQUIVALENCE)
    equivalent = [item for item in solved.values() if item.chi2 <= margin]
    if not equivalent:
        return best
    return max(equivalent, key=lambda item: item.smoothness_scale)


__all__ = [
    "CHI2_EQUIVALENCE",
    "DEFAULT_LCI",
    "DOI_SENSITIVITY_THRESHOLD",
    "LOG_RESISTIVITY_BOUNDS",
    "SMOOTHNESS_SCALE_BOUNDS",
    "LCIResult",
    "SoundingBlock",
    "cumulated_sensitivity",
    "sensitivity_doi",
    "invert_lci",
    "invert_lci_rejecting_outliers",
    "lateral_edges",
    "mask_sounding_block",
    "solve_lci",
    "weighted_residuals",
]
