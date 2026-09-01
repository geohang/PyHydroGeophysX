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

import contextlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

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
#: On sparse ground TDEM the published value can saturate. With a handful of
#: gates per station and a model of twenty layers, the deepest layer often clears
#: 0.8 on its own: the measure cumulates from the bottom up, and that layer is
#: thick. The reported depth then collapses onto the bottom of the
#: parameterisation for much of the survey, which is the measure meeting a coarse
#: deep grid rather than a claim about resolution. Two symptoms identify it: a
#: large share of stations reporting exactly the model bottom, and stations
#: holding three gates reporting the same depth as stations holding ten. Values
#: in the 6 to 8 range keep the reported depth inside the model on such data.
#: Christiansen and Auken's value is the default because it is the published one
#: and it travels across systems.
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
    prior_lower: Optional[np.ndarray] = None
    prior_weights: Optional[np.ndarray] = None

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
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    chi2_median_history: List[float] = field(default_factory=list)

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
            "diagnostics": dict(self.diagnostics),
            "chi2_median_history": list(self.chi2_median_history),
        }


def lateral_edges(
    positions: Sequence[float],
    lines: Optional[Sequence[int]] = None,
    *,
    reference_distance: float = 10.0,
    distance_power: float = 1.0,
) -> List[Tuple[int, int, float]]:
    """Return ``(i, j, weight)`` for each pair of neighbouring soundings.

    Soundings are neighbours when they are adjacent in along-line order on the
    same survey line. The penalty a pair produces scales as
    ``(reference_distance / d) ** distance_power``, so at the default power of
    one, two stations 10 m apart are tied ten times as tightly as two 100 m
    apart. The weight is the square root of that, because the penalty is the
    square of the weighted difference. It is capped at 1 so that a pair closer
    than ``reference_distance`` is not tied arbitrarily hard.

    ``distance_power`` of 0 removes the distance dependence and ties every
    neighbouring pair alike. Values between 0 and 1 loosen the fall-off, which
    suits a line whose station spacing varies enough that the linear rule leaves
    the widely spaced pairs effectively unconstrained.
    """
    pos = np.asarray(positions, dtype=float).ravel()
    n = pos.size
    grp = (np.zeros(n, dtype=int) if lines is None
           else np.asarray(lines, dtype=int).ravel())
    if grp.size != n:
        grp = np.zeros(n, dtype=int)
    ref = max(float(reference_distance), 1e-6)
    half_power = 0.5 * max(float(distance_power), 0.0)
    edges: List[Tuple[int, int, float]] = []
    for value in np.unique(grp):
        members = np.flatnonzero(grp == value)
        if members.size < 2:
            continue
        members = members[np.argsort(pos[members], kind="stable")]
        for a, b in zip(members[:-1], members[1:]):
            distance = max(abs(float(pos[b] - pos[a])), ref)
            edges.append((int(a), int(b), (ref / distance) ** half_power))
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


def resolve_worker_count(n_soundings: int, requested: int = 0) -> int:
    """Threads to run the per-sounding forward and Jacobian on.

    ``requested <= 0`` decides here: one thread per sounding, capped at the
    cores this process may use. A larger number than there are soundings buys
    nothing, and a thread per core is the most the machine can run at once.

    Threads rather than processes. Each sounding's forward operator is a SimPEG
    simulation held in a closure, which does not pickle, and a process would
    also copy the whole model state per worker. The work is NumPy underneath and
    releases the GIL for most of its duration, which is why threads pay at all;
    they do not pay in full, so expect well under linear scaling.
    """
    if n_soundings <= 1:
        return 1
    if int(requested) > 0:
        return max(1, min(int(requested), int(n_soundings)))
    cores = getattr(os, "process_cpu_count", None)
    available = cores() if cores is not None else os.cpu_count()
    return max(1, min(int(n_soundings), int(available or 1)))


@contextlib.contextmanager
def _worker_pool(workers: int) -> Iterator[Optional[ThreadPoolExecutor]]:
    """A pool for one solve, or None when there is nothing to gain from one.

    Held open across the whole solve rather than per iteration: an LCI run makes
    several forward passes per iteration, and building a pool for each of them
    would spend more on thread startup than the pass costs.
    """
    if workers <= 1:
        yield None
        return
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="lci") as executor:
        with _single_threaded_blas():
            yield executor


@contextlib.contextmanager
def _single_threaded_blas() -> Iterator[None]:
    """Keep BLAS to one thread while the soundings run in parallel.

    Otherwise each worker's linear algebra opens its own thread pool and the
    machine ends up with cores^2 threads competing for cores. Scoped to the
    solve, so anything else in the process keeps its own threading. Needs
    ``threadpoolctl``; without it this does nothing, because the environment
    variables that would control it are read when BLAS loads, long before here.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1):
        yield


def _map_soundings(executor: Optional[ThreadPoolExecutor], fn, count: int) -> list:
    """``fn`` over every sounding index, in order, on the pool if there is one.

    ``executor.map`` yields results in submission order, so the caller gets the
    same list either way and nothing downstream has to know which path ran.
    """
    if executor is None or count < 2:
        return [fn(s) for s in range(count)]
    return list(executor.map(fn, range(count)))


def _forward_line(blocks: Sequence[SoundingBlock], x: np.ndarray, n_layers: int,
                  executor: Optional[ThreadPoolExecutor] = None) -> List[np.ndarray]:
    def one(s: int) -> np.ndarray:
        sigma = np.power(10.0, -x[s * n_layers:(s + 1) * n_layers])
        return np.asarray(blocks[s].forward(sigma), dtype=float).ravel()

    return _map_soundings(executor, one, len(blocks))


def _sensitivity_line(blocks: Sequence[SoundingBlock], x: np.ndarray,
                      n_layers: int,
                      executor: Optional[ThreadPoolExecutor] = None):
    """Block-diagonal d(predicted) / d(log10 resistivity), weighted by 1/sigma_d.

    The chain rule through ``sigma = 10**(-x)`` contributes
    ``d sigma / d x = -ln(10) * sigma``.
    """
    from scipy import sparse

    def one(s: int) -> np.ndarray:
        block = blocks[s]
        sigma = np.power(10.0, -x[s * n_layers:(s + 1) * n_layers])
        jac = np.asarray(block.jacobian(sigma), dtype=float)
        if jac.shape != (block.dobs.size, n_layers):
            raise ValueError(
                f"sounding {s} returned a Jacobian of shape {jac.shape}, "
                f"expected {(block.dobs.size, n_layers)}.")
        return (jac * (-_LN10 * sigma)[None, :]) / block.uncertainty[:, None]

    scaled = _map_soundings(executor, one, len(blocks))
    # Assembled here rather than in the workers: building the sparse blocks is
    # cheap next to a forward call, and it keeps SciPy out of the threads.
    return sparse.block_diag([sparse.csr_matrix(part) for part in scaled],
                             format="csr")


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


def _sounding_median(values) -> float:
    """Equal-sounding median of per-sounding mean squared gate residuals.

    Stations with no data report NaN and contribute no artificial zero. This is
    a display statistic only; optimization still uses the full weighted sum.
    """
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _solve_lci_trf(blocks, n_layers, x, Dz, Dx, prior_lower, prior_weights,
                   lo, hi, lam_v, lam_l, scale, executor, *, max_nfev,
                   ftol, xtol, gtol, target_chi2, chi2_tolerance, speak, started):
    """Bound-aware sparse least squares with fixed data errors and penalties.

    Unlike a clipped unconstrained step, TRF accounts for the feasible directions
    while constructing the step. It minimizes the *whole* objective; the data
    target is used by the outer lambda search, not as an inner convergence test.
    Only accepted iterates (Jacobian evaluations) enter the progress history.
    Works with SciPy >= 1.8, without requiring the newer callback API.
    """
    from scipy import sparse
    from scipy.optimize import least_squares

    if int(max_nfev) != max_nfev or int(max_nfev) < 1:
        raise ValueError("lci_max_nfev must be a positive integer.")
    n_data = sum(b.dobs.size for b in blocks)
    R = sparse.vstack([math.sqrt(lam_v) * Dz, math.sqrt(lam_l) * Dx], format="csr")
    prior_index = np.flatnonzero(prior_weights)
    history, objectives, medians = [], [], []
    cache_x, cache_residual, cache_per = None, None, None

    def residual(vec):
        nonlocal cache_x, cache_residual, cache_per
        if cache_x is None or not np.array_equal(vec, cache_x):
            data_res, cache_per = _misfit(blocks, _forward_line(blocks, vec, n_layers, executor))
            prior = prior_weights[prior_index] * np.maximum(prior_lower[prior_index] - vec[prior_index], 0.)
            cache_x = vec.copy()
            cache_residual = np.r_[data_res, R @ vec, prior]
        return cache_residual

    def record(vec):
        r = residual(vec)
        chi2 = float(r[:n_data] @ r[:n_data]) / max(n_data, 1)
        history.append(chi2)
        medians.append(_sounding_median(cache_per))
        objectives.append(float(r @ r))
        speak(f"  LCI TRF accepted {len(history)-1}: median chi2={medians[-1]:.3f}, "
              f"global chi2={chi2:.3f}, objective={objectives[-1]:.6g}")

    last_jac_x = None

    def jacobian(vec):
        nonlocal last_jac_x
        # Reuse the forward at this model (including SimPEG's cached fields).
        record(vec)
        last_jac_x = vec.copy()
        J = _sensitivity_line(blocks, vec, n_layers, executor)
        rows = np.arange(prior_index.size)
        values = -prior_weights[prior_index] * (vec[prior_index] < prior_lower[prior_index])
        P = sparse.csr_matrix((values, (rows, prior_index)), shape=(prior_index.size, vec.size))
        return sparse.vstack([J, R, P], format="csr")

    fitted = least_squares(
        residual, x, jac=jacobian, bounds=(lo, hi), method="trf", loss="linear",
        x_scale="jac", tr_solver="lsmr", tr_options={"atol": 1e-5, "btol": 1e-5},
        max_nfev=int(max_nfev), ftol=float(ftol), xtol=float(xtol), gtol=float(gtol))
    if last_jac_x is None or not np.array_equal(last_jac_x, fitted.x):
        record(fitted.x)
    final_r = residual(fitted.x)
    # A projected gradient is zero at a bound-constrained stationary point.
    # Report this separately: small objective changes do not establish KKT or
    # global optimality, and reaching the evaluation budget is not convergence.
    projected = fitted.x - np.clip(fitted.x - fitted.grad, lo, hi)
    projected_inf = float(np.linalg.norm(projected, ord=np.inf))
    reason = {0: "max_nfev", 1: "gradient", 2: "objective_tolerance",
              3: "step_tolerance", 4: "objective_and_step_tolerance"}.get(fitted.status, "solver_failure")
    chi2 = float(final_r[:n_data] @ final_r[:n_data]) / max(n_data, 1)
    diagnostics = {
        "solver": "trf", "solver_converged": bool(fitted.success),
        "status": int(fitted.status), "message": str(fitted.message),
        "nfev": int(fitted.nfev), "njev": int(fitted.njev or 0),
        "max_nfev": int(max_nfev), "optimality": float(fitted.optimality),
        "projected_gradient_inf": projected_inf, "stationary": projected_inf <= float(gtol),
        "objective": float(final_r @ final_r), "objective_history": objectives,
        "data_objective": float(final_r[:n_data] @ final_r[:n_data]),
        "regularization_objective": float(final_r[n_data:] @ final_r[n_data:]),
        "lower_bound_fraction": float(np.mean(fitted.x <= lo + 1e-6)),
        "upper_bound_fraction": float(np.mean(fitted.x >= hi - 1e-6)),
        "target_band_reached": bool(target_chi2 > 0 and abs(chi2-target_chi2) <= abs(chi2_tolerance)),
        "ftol": float(ftol), "xtol": float(xtol), "gtol": float(gtol),
    }
    speak(f"  LCI TRF stop: {reason}, chi2={chi2:.3f}, {fitted.nfev} evaluations; "
          f"projected gradient={projected_inf:.3g} (not a global-optimum certificate)")
    return LCIResult(
        models=10.**fitted.x.reshape(len(blocks), n_layers), chi2=chi2,
        chi2_per_sounding=cache_per.copy(), chi2_history=history,
        iterations=max(len(history)-1, 0), stop_reason=reason,
        smoothness_scale=scale, lambda_vertical=lam_v, lambda_lateral=lam_l,
        n_data=n_data, seconds=time.time()-started, diagnostics=diagnostics,
        chi2_median_history=medians)


def solve_lci(
    blocks: Sequence[SoundingBlock],
    n_layers: int,
    *,
    smoothness: float = 0.3,
    lateral_smoothness: float = 0.3,
    smoothness_scale: float = 1.0,
    reference_distance: float = 10.0,
    lateral_distance_power: float = 1.0,
    initial_model: Optional[np.ndarray] = None,
    starting_resistivity: float = 100.0,
    max_iterations: int = 20,
    convergence_tolerance: float = 0.02,
    convergence_metric: str = "data",
    solver: str = "trf",
    trf_max_nfev: int = 90,
    trf_ftol: float = 1e-4,
    trf_xtol: float = 1e-6,
    trf_gtol: float = 1e-5,
    min_iterations: int = 2,
    target_chi2: float = 1.0,
    chi2_tolerance: float = 0.2,
    line_search_steps: int = 6,
    target_steps: int = 4,
    bounds: Tuple[float, float] = LOG_RESISTIVITY_BOUNDS,
    parallel_workers: int = 0,
    verbose: bool = True,
    log: LogFn = _noop,
) -> LCIResult:
    """Solve one coupled line inversion at a fixed smoothness.

    Bound-aware sparse trust-region least squares (``solver='trf'``) is the
    formal default. ``solver='gauss_newton'`` selects the fast legacy path.
    Its budget is ``trf_max_nfev`` (forward evaluations, including rejected
    trials), not ``max_iterations``. Its three tolerances apply to the full
    objective, step and gradient, respectively. The legacy ``gauss_newton``
    path retains its iteration/target controls for backwards compatibility.

    The objective is

    ``||W (F(x) - d)||^2 + lam_v ||Dz x||^2 + lam_l ||Dx x||^2``

    with ``lam_v = (smoothness_scale * smoothness)^2`` and ``lam_l`` likewise
    from ``lateral_smoothness``. Squaring keeps the two knobs numerically
    equivalent to the residual-stacking convention used by the per-sounding
    Occam routine, so an existing ``smoothness=0.3`` setting means the same
    amount of vertical damping here.

    On the legacy Gauss-Newton path, iteration stops at ``target_chi2``, when
    the relative chi-squared improvement falls below ``convergence_tolerance``
    (the plateau rule used throughout this package), when the line search cannot
    find a descent step, or at ``max_iterations``. TRF instead uses its full-
    objective tolerances and forward-evaluation budget. Either path reports the
    reason in ``stop_reason``.

    A Gauss-Newton step near the target routinely shoots past it, so a run that
    stopped at the first iterate below ``target_chi2`` would report whatever
    overshoot the step happened to produce. That both fits noise and makes the
    misfit a jumpy function of the smoothness, which the search in
    :func:`invert_lci` then cannot bisect. Set ``target_steps`` above zero and
    an overshooting step is shortened until the misfit lands within
    ``chi2_tolerance`` of the target instead.

    ``parallel_workers`` runs the per-sounding forward and Jacobian on a thread
    pool, ``0`` choosing a count from the machine. Every sounding owns its
    forward operator, and the workers only read the shared model vector, so the
    arithmetic is the same either way; measured against the serial path, the
    models come back bit for bit identical.
    """
    if convergence_metric not in {"data", "objective"}:
        raise ValueError("convergence_metric must be data or objective.")
    if solver not in {"gauss_newton", "trf"}:
        raise ValueError("solver must be gauss_newton or trf.")
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
        reference_distance=reference_distance,
        distance_power=lateral_distance_power)
    Dz = _vertical_operator(n_soundings, n_layers)
    Dx = _lateral_operator(edges, n_soundings, n_layers)
    reg = (lam_v * (Dz.T @ Dz)) + (lam_l * (Dx.T @ Dx))
    prior_lower = np.concatenate([np.zeros(n_layers) if b.prior_lower is None
                                   else np.asarray(b.prior_lower, float) for b in blocks])
    prior_weights = np.concatenate([np.zeros(n_layers) if b.prior_weights is None
                                     else np.asarray(b.prior_weights, float) for b in blocks])
    if (prior_lower.shape != x.shape or prior_weights.shape != x.shape
            or not np.isfinite(prior_lower).all() or not np.isfinite(prior_weights).all()
            or np.any(prior_weights < 0)):
        raise ValueError(
            "resistive-prior arrays must match the model and have finite nonnegative weights.")

    def objective(vec: np.ndarray, residual: np.ndarray) -> float:
        prior = prior_weights * np.maximum(prior_lower - vec, 0.)
        return float(residual @ residual) + float(vec @ (reg @ vec)) + float(prior @ prior)

    workers = resolve_worker_count(n_soundings, parallel_workers)
    if workers > 1:
        speak(f"  LCI running {n_soundings} soundings on {workers} threads")
    with _worker_pool(workers) as executor:
        if solver == "trf":
            return _solve_lci_trf(
                blocks, n_layers, x, Dz, Dx, prior_lower, prior_weights,
                lo, hi, lam_v, lam_l, scale, executor,
                max_nfev=trf_max_nfev, ftol=trf_ftol, xtol=trf_xtol, gtol=trf_gtol,
                target_chi2=target_chi2, chi2_tolerance=chi2_tolerance,
                speak=speak, started=started)
        predicted = _forward_line(blocks, x, n_layers, executor)
        residual, per_sounding = _misfit(blocks, predicted)
        chi2 = float(residual @ residual) / max(n_data, 1)
        history = [chi2]
        median_history = [_sounding_median(per_sounding)]
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
            G = _sensitivity_line(blocks, x, n_layers, executor)
            gram = (G.T @ G) + reg
            gradient = np.asarray(G.T @ residual, dtype=float) + (reg @ x)
            if np.any(prior_weights):
                from scipy import sparse
                active_weight = prior_weights ** 2 * (x < prior_lower)
                gram = gram + sparse.diags(active_weight)
                gradient = gradient + active_weight * (x - prior_lower)
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
                # The line search is the busiest caller: several of these per
                # iteration against one Jacobian, which is why the pool has to
                # reach in here and not only the top of the loop.
                trial_pred = _forward_line(blocks, trial, n_layers, executor)
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
            median_history.append(_sounding_median(per_sounding))
            speak(f"  LCI iter {iterations}: median chi2={median_history[-1]:.3f}, "
                  f"global chi2={chi2:.3f} (step {alpha:.3g})")
            if at_target(chi2):
                stop_reason = "target"
                break
            # One thin gain is not a plateau. A backtracked step (alpha well under
            # one) often gains little while the model is still moving, so require
            # the improvement to stay small twice running before giving up.
            # Reweighting changes the optimum balance: data chi2 can increase
            # while the full objective still improves. Do not mistake that for
            # convergence of the objective actually used by the line search.
            previous_metric = base_phi if convergence_metric == "objective" else previous
            current_metric = phi if convergence_metric == "objective" else chi2
            gained = previous_metric - current_metric
            stalls = (stalls + 1
                      if gained <= abs(previous_metric) * float(convergence_tolerance)
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
            chi2_median_history=median_history,
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
        usable = {k: v for k, v in solved.items()
                  if not v.diagnostics or v.diagnostics.get("solver_converged", False)}
        if not usable:
            return None
        key = min(usable, key=lambda k: abs(math.log(max(k, 1e-30))
                                            - math.log(max(scale, 1e-30))))
        return usable[key].models

    def run(scale: float, warm: Optional[np.ndarray]) -> LCIResult:
        out = solve_lci(blocks, n_layers, smoothness_scale=scale,
                        initial_model=warm, target_chi2=target_chi2,
                        chi2_tolerance=chi2_tolerance,
                        verbose=verbose, log=log, **kwargs)
        solved[float(scale)] = out
        return out

    base = run(float(smoothness_scale), kwargs.pop("initial_model", None))
    best = base
    if base.diagnostics and not base.diagnostics.get("solver_converged", False):
        base.lambda_search = {
            "status": "solver_incomplete", "reason": base.stop_reason,
            "trials": [{"lambda": base.smoothness_scale, "chi2": base.chi2}],
            "fixed_chi2": base.chi2, "fixed_scale": base.smoothness_scale,
        }
        log("  Inner solver incomplete; keeping errors and smoothness fixed.")
        return base
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
        if out.diagnostics and not out.diagnostics.get("solver_converged", False):
            # The search treats NaN as an unusable trial and stops. Do not rank
            # a budget-limited trial as a regularization optimum.
            return float("nan")
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
        usable = {k: v for k, v in solved.items()
                  if not v.diagnostics or v.diagnostics.get("solver_converged", False)}
        smoothest = _smoothest_equivalent(usable, best)
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
        prior_lower=block.prior_lower, prior_weights=block.prior_weights,
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


def invert_lci_with_robust_errors(
    blocks: Sequence[SoundingBlock], n_layers: int, *, threshold: float = 3.0,
    passes: int = 3, max_error_factor: float = 10.0,
    min_unchanged_fraction: float = 0.0, error_target_chi2: float = 0.0,
    target_tolerance: float = 0.25, log: LogFn = _noop, **kwargs: Any,
) -> Tuple[LCIResult, List[SoundingBlock], Dict[str, Any]]:
    """Warm-started error reweighting, preserving all LM/HM data and operators.

    The returned blocks carry effective errors (also used for DOI). Input blocks
    are untouched. The solver outcome uses effective chi2; the report additionally
    supplies original-error scores for honest comparisons between runs.
    """
    from .robust_errors import reweight_errors

    original = list(blocks)
    offsets = np.r_[0, np.cumsum([b.dobs.size for b in original])]
    fitted_blocks = original
    initial_search = {}
    total_seconds = 0.0
    total_iterations = 0

    def solve(effective, previous):
        nonlocal fitted_blocks, initial_search, total_seconds, total_iterations
        fitted_blocks = [replace(b, uncertainty=effective[offsets[i]:offsets[i + 1]].copy())
                         for i, b in enumerate(original)]
        settings = dict(kwargs)
        if previous is not None:
            settings.update(initial_model=previous.models, auto_lambda=False,
                            smoothness_scale=previous.smoothness_scale,
                            convergence_metric="objective")
        fitted = invert_lci(fitted_blocks, n_layers, log=log, **settings)
        if previous is None:
            initial_search = fitted.lambda_search
        total_seconds += fitted.seconds
        total_iterations += fitted.iterations
        return fitted

    def predicted(outcome):
        # Use the existing parallel forward path and reuse cached operators.
        workers = resolve_worker_count(len(original), int(kwargs.get("parallel_workers", 0)))
        with _worker_pool(workers) as pool:
            return np.concatenate(_forward_line(
                original, np.log10(outcome.models).ravel(), n_layers, pool))

    def stage_statistics(fit, original_residual):
        per = [np.mean(original_residual[offsets[i]:offsets[i+1]] ** 2)
               for i in range(len(original)) if offsets[i+1] > offsets[i]]
        return {"convergence_median": list(getattr(fit, "chi2_median_history", [])),
                "chi2_original_median": _sounding_median(per)}

    outcome, _, info = reweight_errors(
        np.concatenate([b.dobs for b in original]),
        np.concatenate([b.uncertainty for b in original]), solve, predicted,
        threshold=threshold, passes=passes, max_error_factor=max_error_factor,
        min_unchanged_fraction=min_unchanged_fraction, target_chi2=error_target_chi2,
        target_tolerance=target_tolerance,
        solver_ready=lambda fit: not getattr(fit, "diagnostics", {}) or bool(
            fit.diagnostics.get("solver_converged", False)),
        history=lambda fit: fit.chi2_history, stage_statistics=stage_statistics, log=log)
    residual = np.asarray(info["residual_original"])
    info["block_offsets"] = offsets.tolist()
    info["initial_lambda_search"] = initial_search
    info["total_iterations"] = total_iterations
    info["solve_seconds"] = total_seconds
    info["chi2_per_sounding_original"] = [
        float(np.mean(residual[offsets[i]:offsets[i + 1]] ** 2))
        if offsets[i + 1] > offsets[i] else float("nan") for i in range(len(original))]
    return outcome, fitted_blocks, info


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
        "convergence_median": list(outcome.chi2_median_history),
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
            "convergence_median": list(outcome.chi2_median_history),
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
    "invert_lci_with_robust_errors",
    "lateral_edges",
    "mask_sounding_block",
    "solve_lci",
    "weighted_residuals",
]
