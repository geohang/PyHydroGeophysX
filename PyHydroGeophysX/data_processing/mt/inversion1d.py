"""Occam's 1D inversion of MT soundings, alone or jointly with a TEM sounding.

The model is ``log10`` resistivity in many thin layers - thicknesses growing
geometrically from a twentieth of the shortest period's skin depth to beyond
the longest's - and Occam's inversion (Constable et al. 1987) looks for the
smoothest such model (least first-difference roughness) that fits the data
to a target RMS misfit. Each iteration linearizes the forward problem
(:func:`.forward1d.sensitivity_1d` gives the exact derivatives), solves the
regularized normal equations for a sweep of trade-off parameters ``mu``, and
takes the ``mu`` with the least misfit while the target is out of reach, then
the largest ``mu`` that still reaches it.

The data are ``log10 rho_a`` and phase (degrees) of the determinant, of Zxy,
of Zyx, or of both modes against one model. A **static shift** per mode can
be estimated with the model: a free offset of that mode's ``log10 rho_a``,
held to zero by a Gaussian prior of ``static_shift_prior`` (log10 units); the
phases fix the model, the TEM (or the prior) the offset.

Joined with a **TEM sounding** (``tem=``), the transient's response is fitted
by the same model through PyHydroGeophysX's TDEM forward operator - the TEM
resolves the near surface that sets the static shift, the MT the depths
below (Meju 1996). That needs SimPEG, as TEM inversion does; MT alone needs
only NumPy.

Constable, S. C., Parker, R. L. & Constable, C. G. (1987). Occam's inversion:
a practical algorithm for generating smooth models from electromagnetic
sounding data. Geophysics, 52(3), 289-300. https://doi.org/10.1190/1.1442303

Meju, M. A. (1996). Joint inversion of TEM and distorted MT soundings: some
effective practical considerations. Geophysics, 61(1), 56-65.
https://doi.org/10.1190/1.1443956
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .forward1d import sensitivity_1d
from .transfer_function import MU0, TransferFunction

_LN10 = np.log(10.0)
_MODES = ("det", "xy", "yx", "both")


@dataclass
class Sounding1D:
    """What a 1D inversion fits: per mode, ``log10 rho_a`` and phase with errors."""

    frequency: np.ndarray
    modes: List[str]
    log_rho: np.ndarray          # (n_modes, n_freq)
    phase: np.ndarray            # degrees, first quadrant
    log_rho_err: np.ndarray
    phase_err: np.ndarray
    station: str = ""

    @property
    def n_data(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.log_rho)) + np.count_nonzero(np.isfinite(self.phase)))


def _mode_impedance(tf: TransferFunction, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "det":
        z = tf.determinant()
        if tf.z_err is not None:
            # First order: Zdet^2 = Zxx Zyy - Zxy Zyx, so dZdet = (Zyy dZxx + Zxx dZyy - Zyx dZxy
            # - Zxy dZyx) / (2 Zdet), the elements' errors independent. A missing diagonal error
            # counts as none where its element is there.
            Z = tf.z
            weight = np.abs(np.stack([Z[:, 1, 1], Z[:, 0, 0], Z[:, 1, 0], Z[:, 0, 1]], axis=1)) ** 2
            var = np.stack([tf.z_err[:, 0, 0], tf.z_err[:, 1, 1], tf.z_err[:, 0, 1], tf.z_err[:, 1, 0]], axis=1) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.sqrt(np.nansum(weight * var, axis=1)) / (2 * np.abs(z))
            err = np.where(np.isfinite(tf.z_err[:, 0, 1]) & np.isfinite(tf.z_err[:, 1, 0]), err, np.nan)
        else:
            err = np.full(z.shape, np.nan)
        return z, err
    i, j = (0, 1) if mode == "xy" else (1, 0)
    z = tf.z[:, i, j] * (1 if mode == "xy" else -1)
    err = tf.z_err[:, i, j] if tf.z_err is not None else np.full(z.shape, np.nan)
    return z, err


def sounding_from_tf(tf: TransferFunction, *, mode: str = "det", error_floor: float = 0.05,
                     period_range: Optional[Tuple[float, float]] = None) -> Sounding1D:
    """The 1D data of a site: apparent resistivity and phase of a mode, with errors.

    ``error_floor`` is the least relative error of the impedance (5%: about
    0.043 in ``log10 rho_a`` and 1.4 degrees of phase). Phases are folded into
    the first quadrant; values outside it are dropped.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}")
    modes = ["xy", "yx"] if mode == "both" else [mode]
    keep = np.ones(tf.n_frequencies, dtype=bool)
    if period_range is not None:
        keep &= (tf.period >= min(period_range)) & (tf.period <= max(period_range))
    omega = tf.angular_frequency[keep]
    rows = {"log_rho": [], "phase": [], "log_rho_err": [], "phase_err": []}
    for name in modes:
        z, err = _mode_impedance(tf, name)
        z, err = z[keep], err[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = err / np.abs(z)
        # No error of its own: the floor stands in; with no floor either, the datum is left out
        # rather than given an infinite weight.
        missing = ~(np.isfinite(relative) & (relative > 0))
        relative = np.where(missing, error_floor, np.fmax(relative, error_floor))
        phase = np.degrees(np.angle(z))
        good = np.isfinite(z) & (phase > 0) & (phase < 90) & (relative > 0)
        rows["log_rho"].append(np.where(good, np.log10(np.abs(z) ** 2 / (omega * MU0)), np.nan))
        rows["phase"].append(np.where(good, phase, np.nan))
        rows["log_rho_err"].append(2 * relative / _LN10)
        rows["phase_err"].append(np.degrees(relative))
    return Sounding1D(tf.frequency[keep], modes, *(np.array(rows[k]) for k in
                                                  ("log_rho", "phase", "log_rho_err", "phase_err")),
                      station=tf.station)


def layer_thicknesses(frequency: Any, n_layers: int = 50, *, resistivity_guess: Any = 100.0,
                      depth_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Geometric thicknesses from a twentieth of the least skin depth to twice the greatest.

    ``resistivity_guess`` is one resistivity or a pair: the apparent
    resistivity at the highest and at the lowest frequency, which set the
    shallowest and the deepest skin depth. One value for both made the top
    layer of a resistive site with a conductive cover tens of metres thick.
    """
    f = np.asarray(frequency, dtype=float)
    shallow, deep = np.broadcast_to(np.asarray(resistivity_guess, dtype=float), (2,))
    skin = lambda rho, freq: 503.0 * np.sqrt(rho / freq)
    top, bottom = depth_range or (skin(shallow, f.max()) / 20.0, 2.0 * skin(deep, f.min()))
    depths = np.geomspace(top, bottom, n_layers - 1)
    return np.diff(np.r_[0.0, depths])


@dataclass
class Occam1DResult:
    resistivity: np.ndarray
    thickness: np.ndarray
    rms: float
    roughness: float
    mu: float
    static_shift: Dict[str, float]
    predicted: Dict[str, np.ndarray]
    sounding: Sounding1D
    iterations: int
    history: List[Dict[str, float]] = field(default_factory=list)
    tem: Optional[Dict[str, Any]] = None
    converged: bool = False

    @property
    def depth(self) -> np.ndarray:
        """Each layer's top, m (the half-space last)."""
        return np.r_[0.0, np.cumsum(self.thickness)]

    def depth_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Step-plot arrays (depth, resistivity), the half-space drawn one layer deep."""
        tops = self.depth
        bottoms = np.r_[tops[1:], tops[-1] + max(self.thickness[-1], 1.0)]
        return np.column_stack([tops, bottoms]).ravel(), np.repeat(self.resistivity, 2)


def _mt_rows(sounding: Sounding1D, log_rho_model: np.ndarray, thickness: np.ndarray,
             shifts: np.ndarray):
    """Residual-ready predictions and derivatives of the MT data (log10 rho_a, phase deg)."""
    Z, d_log_rho, d_phase = sensitivity_1d(10.0 ** log_rho_model, thickness, sounding.frequency)
    omega = 2 * np.pi * sounding.frequency
    log_rho_a = np.log10(np.abs(Z) ** 2 / (omega * MU0))
    phase = np.degrees(np.angle(Z))
    preds, jac_rows, shift_rows, values, errors = [], [], [], [], []
    n_modes = len(sounding.modes)
    for k in range(n_modes):
        for observed, err, model, derivative, shifted in (
                (sounding.log_rho[k], sounding.log_rho_err[k], log_rho_a + shifts[k], d_log_rho * _LN10, True),
                (sounding.phase[k], sounding.phase_err[k], phase, np.degrees(d_phase) * _LN10, False)):
            good = np.isfinite(observed)
            values.append(observed[good])
            errors.append(err[good])
            preds.append(model[good])
            jac_rows.append(derivative[good])
            row = np.zeros((good.sum(), n_modes))
            if shifted:
                row[:, k] = 1.0
            shift_rows.append(row)
    return (np.concatenate(values), np.concatenate(errors), np.concatenate(preds),
            np.vstack(jac_rows), np.vstack(shift_rows), log_rho_a, phase)


def _occam_choice(trials: List[Tuple[float, float]], target: float) -> int:
    """Occam's pick among ``(mu, rms)``: the largest mu reaching the target, else the least rms."""
    reaching = [i for i, (_, rms) in enumerate(trials) if rms <= target]
    if reaching:
        return max(reaching, key=lambda i: trials[i][0])
    return min(range(len(trials)), key=lambda i: trials[i][1])


def _occam_iterations(evaluate: Callable, x: np.ndarray, D: np.ndarray, prior: np.ndarray, n_model: int,
                      bounds: Tuple[float, float], target_rms: float, max_iterations: int, cheap: bool,
                      log: Optional[Callable[[str], None]], roughness: Callable[[np.ndarray], float]):
    """Occam's iterations from ``x``: each linearizes, sweeps ``mu`` and keeps Occam's choice.

    ``evaluate(x)`` gives ``(rms, residual, jacobian, predicted)``; ``D`` holds
    the roughness rows (``mu`` times its square is the regularization), ``prior``
    rows added as they are; the first ``n_model`` entries of ``x`` are log10
    resistivities, kept within ``bounds``. With ``cheap`` false (a TEM forward)
    the true forward judges a coarse sweep of ``mu`` and the neighbours of the
    linearized choice rather than every candidate.
    """
    lo, hi = bounds
    rms, residual, jac, predicted = evaluate(x)
    history = [{"iteration": 0, "rms": rms, "mu": float("nan"), "roughness": 0.0}]
    mus = np.logspace(-4, 6, 41)
    mu_used, converged, iterations = float("nan"), False, 0
    for iterations in range(1, max_iterations + 1):
        rhs_data = residual + jac @ x
        JtJ, Jtd = jac.T @ jac, jac.T @ rhs_data
        candidates = []
        for mu in mus:
            A = JtJ + mu * D.T @ D + prior.T @ prior
            try:
                candidate = np.linalg.solve(A, Jtd)
            except np.linalg.LinAlgError:
                continue
            candidate[:n_model] = np.clip(candidate[:n_model], lo, hi)
            linear = float(np.sqrt(np.mean((rhs_data - jac @ candidate) ** 2)))
            candidates.append((mu, linear, candidate))
        if not candidates:
            break
        # The true forward decides among the candidates - all of them when it is
        # cheap; with a TEM forward, a coarse sweep of mu plus the neighbours of
        # the linearized choice.
        if cheap:
            chosen = range(len(candidates))
        else:
            guess = _occam_choice([(c[0], c[1]) for c in candidates], target_rms)
            chosen = sorted(set(range(0, len(candidates), 5)) | {
                min(max(guess + k, 0), len(candidates) - 1) for k in (-2, -1, 0, 1, 2)})
        trials = [(candidates[i][0], evaluate(candidates[i][2])[0], candidates[i][2]) for i in chosen]
        best = _occam_choice([(t[0], t[1]) for t in trials], target_rms)
        mu_used, new_rms, x_new = trials[best]
        step = 1.0
        while new_rms > rms and step > 1 / 16:
            # Occam's safeguard: a step that makes things worse is shortened.
            step /= 2
            x_try = x + step * (trials[best][2] - x)
            try_rms = evaluate(x_try)[0]
            if try_rms < new_rms:
                new_rms, x_new = try_rms, x_try
        rough_new = roughness(x_new)
        rough_old = roughness(x)
        x = x_new
        rms, residual, jac, predicted = evaluate(x)
        history.append({"iteration": iterations, "rms": rms, "mu": float(mu_used), "roughness": rough_new})
        if log is not None:
            log(f"  Occam iteration {iterations}: RMS {rms:.3f}, mu {mu_used:.3g}")
        if rms <= target_rms * 1.01 and abs(rough_new - rough_old) <= 1e-3 * max(rough_old, 1e-6):
            converged = True
            break
        if len(history) > 3 and rms > target_rms and abs(history[-2]["rms"] - rms) < 1e-3 * rms:
            break
    return x, rms, predicted, history, mu_used, converged, iterations


def _tem_block(tem: Dict[str, Any], thickness: np.ndarray):
    from PyHydroGeophysX.inversion.em1d import DEFAULT_INVERSION, build_sounding_block

    settings = {**DEFAULT_INVERSION, **dict(tem.get("inversion", {}))}
    settings.update(n_layers=thickness.size + 1, layer_thicknesses=thickness)
    return build_sounding_block(tem["data"], tem.get("geometry", {}), settings, "TDEM")


def occam1d(data: Any, *, mode: str = "det", n_layers: int = 50,
            thickness: Optional[Sequence[float]] = None, target_rms: float = 1.0,
            max_iterations: int = 30, starting_resistivity: Optional[float] = None,
            static_shift: bool = False, static_shift_prior: float = 0.3,
            error_floor: float = 0.05, period_range: Optional[Tuple[float, float]] = None,
            tem: Optional[Dict[str, Any]] = None, tem_weight: float = 1.0,
            log: Optional[Callable[[str], None]] = None,
            resistivity_bounds: Tuple[float, float] = (0.01, 1e5)) -> Occam1DResult:
    """Occam's smoothest model fitting a site's 1D data (and a TEM sounding).

    ``data`` is a :class:`TransferFunction` (its ``mode`` is taken, see
    :func:`sounding_from_tf`) or a :class:`Sounding1D`. ``thickness`` fixes the
    layers; otherwise ``n_layers`` are spread over the skin-depth range.
    ``static_shift`` adds one offset of ``log10 rho_a`` per mode.
    ``tem={"data": ..., "geometry": ..., "inversion": ...}`` is a TDEM sounding
    as :func:`PyHydroGeophysX.inversion.em1d.tdem_invert` takes it, fitted with
    weight ``tem_weight``; the RMS target then applies to all data together.
    """
    sounding = data if isinstance(data, Sounding1D) else sounding_from_tf(
        data, mode=mode, error_floor=error_floor, period_range=period_range)
    if sounding.n_data < 2:
        raise ValueError("too few data in the period range")
    log_rho = sounding.log_rho[np.isfinite(sounding.log_rho)]
    start = float(starting_resistivity or 10 ** np.median(log_rho))
    order = np.argsort(sounding.frequency)
    ends = []
    for columns in (order[-3:], order[:3]):            # the highest, then the lowest frequencies
        values = sounding.log_rho[:, columns]
        values = values[np.isfinite(values)]
        ends.append(10 ** float(np.median(values)) if values.size else start)
    h = np.asarray(thickness, dtype=float) if thickness is not None else layer_thicknesses(
        sounding.frequency, n_layers, resistivity_guess=ends)
    n = h.size + 1
    n_modes = len(sounding.modes)
    n_shift = n_modes if static_shift else 0
    lo, hi = np.log10(resistivity_bounds[0]), np.log10(resistivity_bounds[1])
    block = _tem_block(tem, h) if tem is not None else None

    roughness_matrix = np.diff(np.eye(n), axis=0)
    D = np.zeros((n - 1, n + n_shift))
    D[:, :n] = roughness_matrix
    prior = np.zeros((n_shift, n + n_shift))
    if n_shift:
        prior[:, n:] = np.eye(n_shift) / static_shift_prior

    def evaluate(x):
        model, shifts = x[:n], (x[n:] if n_shift else np.zeros(n_modes))
        values, errors, preds, J, S, log_rho_a, phase = _mt_rows(sounding, model, h, shifts)
        residual = (values - preds) / errors
        jac = np.hstack([J, S[:, :n_shift]]) / errors[:, None]
        if block is not None:
            sigma = 10.0 ** (-model)
            tem_pred = np.asarray(block.forward(sigma), dtype=float)
            tem_res = tem_weight * (np.asarray(block.dobs) - tem_pred) / np.asarray(block.uncertainty)
            tem_jac = (np.asarray(block.jacobian(sigma), dtype=float) * (-_LN10 * sigma)[None, :]
                       * tem_weight / np.asarray(block.uncertainty)[:, None])
            residual = np.r_[residual, tem_res]
            jac = np.vstack([jac, np.hstack([tem_jac, np.zeros((tem_jac.shape[0], n_shift))])])
            extra = {"tem_predicted": tem_pred}
        else:
            extra = {}
        rms = float(np.sqrt(np.mean(residual**2)))
        return rms, residual, jac, {"log_rho_a": log_rho_a, "phase": phase, **extra}

    x = np.r_[np.full(n, np.log10(start)), np.zeros(n_shift)]
    x, rms, predicted, history, mu_used, converged, iterations = _occam_iterations(
        evaluate, x, D, prior, n, (lo, hi), target_rms, max_iterations, block is None, log,
        lambda v: float(np.sum(np.diff(v[:n]) ** 2)))
    shifts = {name: float(10 ** x[n + k]) for k, name in enumerate(sounding.modes)} if n_shift else {}
    tem_out = None
    if block is not None:
        tem_out = {"observed": np.asarray(block.dobs), "uncertainty": np.asarray(block.uncertainty),
                   "predicted": predicted.get("tem_predicted")}
    return Occam1DResult(
        resistivity=10.0 ** x[:n], thickness=h, rms=rms,
        roughness=float(np.sum(np.diff(x[:n]) ** 2)), mu=float(mu_used), static_shift=shifts,
        predicted={"log_rho_a": predicted["log_rho_a"], "phase": predicted["phase"]},
        sounding=sounding, iterations=iterations, history=history, tem=tem_out,
        converged=converged or rms <= target_rms * 1.01)


@dataclass
class Anisotropic1DResult:
    """Occam's azimuthally anisotropic layered model: two resistivities per layer on one strike.

    ``resistivity[0]`` is each layer's resistivity along the axis at azimuth
    ``strike`` (degrees clockwise from north), which Zxy fits in that frame;
    ``resistivity[1]`` along the axis a quarter turn on, which Zyx fits.
    """

    strike: float
    resistivity: np.ndarray            # (2, n_layers)
    thickness: np.ndarray
    rms: float
    mu: float
    static_shift: Dict[str, float]
    predicted: Dict[str, Dict[str, np.ndarray]]
    sounding: Sounding1D
    iterations: int
    history: List[Dict[str, float]] = field(default_factory=list)
    converged: bool = False

    @property
    def depth(self) -> np.ndarray:
        """Each layer's top, m (the half-space last)."""
        return np.r_[0.0, np.cumsum(self.thickness)]

    @property
    def anisotropy(self) -> np.ndarray:
        """Each layer's resistivity across the strike over along it."""
        return self.resistivity[1] / self.resistivity[0]

    def axis(self, k: int) -> Occam1DResult:
        """One axis's model as an :class:`Occam1DResult`, for the plots and conversions that take one."""
        mode = self.sounding.modes[k]
        part = Sounding1D(self.sounding.frequency, [mode], *(getattr(self.sounding, name)[k:k + 1] for name in
                                                             ("log_rho", "phase", "log_rho_err", "phase_err")),
                          station=self.sounding.station)
        return Occam1DResult(
            resistivity=self.resistivity[k], thickness=self.thickness, rms=self.rms,
            roughness=float(np.sum(np.diff(np.log10(self.resistivity[k])) ** 2)), mu=self.mu,
            static_shift={m: v for m, v in self.static_shift.items() if m == mode},
            predicted=self.predicted[mode], sounding=part, iterations=self.iterations,
            history=self.history, converged=self.converged)


def _mean_strike(tf: TransferFunction, period_range: Optional[Tuple[float, float]] = None) -> float:
    """The phase tensor's strike over the periods used, in [0, 90): the mean of 4 x strike, so the
    quarter-turn ambiguity does not split it."""
    from .analysis import phase_tensor

    strike = np.asarray(phase_tensor(tf)["strike"], dtype=float)
    keep = np.isfinite(strike)
    if period_range is not None:
        keep &= (tf.period >= min(period_range)) & (tf.period <= max(period_range))
    if not keep.any():
        raise ValueError("no phase-tensor strike in the period range; give the strike")
    mean = np.angle(np.mean(np.exp(4j * np.radians(strike[keep]))))
    return float(np.degrees(mean) / 4.0 % 90.0)


def occam1d_anisotropic(data: TransferFunction, *, strike: Optional[float] = None, n_layers: int = 50,
                        thickness: Optional[Sequence[float]] = None, target_rms: float = 1.0,
                        max_iterations: int = 30, starting_resistivity: Optional[float] = None,
                        static_shift: bool = False, static_shift_prior: float = 0.3,
                        anisotropy_weight: float = 1.0, error_floor: float = 0.05,
                        period_range: Optional[Tuple[float, float]] = None,
                        log: Optional[Callable[[str], None]] = None,
                        resistivity_bounds: Tuple[float, float] = (0.01, 1e5)) -> Anisotropic1DResult:
    """Occam's smoothest azimuthally anisotropic layered model with one strike.

    Each layer conducts differently along two horizontal axes, the same two
    in every layer: along ``strike`` (degrees clockwise from north; by default
    the phase tensor's, averaged over the periods used) and across it. In that
    frame the layered earth's impedance has no diagonal, and Zxy and Zyx are
    each a 1D isotropic response - Zxy of the resistivities along the strike,
    Zyx of those across it (Pek & Santos 2002, the case of horizontal principal
    axes sharing one azimuth). The two models share the layers and are found
    together: Occam's roughness of each, plus ``anisotropy_weight`` times the
    difference of their ``log10`` resistivities layer by layer, so a layer is
    made anisotropic only where the data ask for it.

    Why the modes differ cannot be told from one site: a 2D structure with the
    same strike, or galvanic distortion, splits them too. A static shift moves
    ``rho_a`` and not the phase, so with ``static_shift`` the difference in
    level between the modes is put down to the shifts and the anisotropy comes
    from the phases' split alone.

    Pek, J. & Santos, F. A. M. (2002). Magnetotelluric impedances and
    parametric sensitivities for 1-D anisotropic layered media. Computers &
    Geosciences, 28(8), 939-950. https://doi.org/10.1016/S0098-3004(02)00014-6
    """
    angle = _mean_strike(data, period_range) if strike is None else float(strike) % 180.0
    sounding = sounding_from_tf(data.rotated(angle), mode="both", error_floor=error_floor,
                                period_range=period_range)
    for k, mode in enumerate(sounding.modes):
        if np.count_nonzero(np.isfinite(sounding.log_rho[k])) < 2:
            raise ValueError(f"too few Z{mode} data in the strike frame for an anisotropic model")
    values = sounding.log_rho[np.isfinite(sounding.log_rho)]
    start = float(starting_resistivity or 10 ** np.median(values))
    order = np.argsort(sounding.frequency)
    ends = []
    for columns in (order[-3:], order[:3]):
        picked = sounding.log_rho[:, columns]
        picked = picked[np.isfinite(picked)]
        ends.append(10 ** float(np.median(picked)) if picked.size else start)
    h = np.asarray(thickness, dtype=float) if thickness is not None else layer_thicknesses(
        sounding.frequency, n_layers, resistivity_guess=ends)
    n = h.size + 1
    n_shift = 2 if static_shift else 0
    width = 2 * n + n_shift
    rough = np.diff(np.eye(n), axis=0)
    D = np.zeros((2 * (n - 1) + n, width))
    D[:n - 1, :n] = rough
    D[n - 1:2 * (n - 1), n:2 * n] = rough
    D[2 * (n - 1):, :n] = anisotropy_weight * np.eye(n)
    D[2 * (n - 1):, n:2 * n] = -anisotropy_weight * np.eye(n)
    prior = np.zeros((n_shift, width))
    if n_shift:
        prior[:, 2 * n:] = np.eye(n_shift) / static_shift_prior
    parts = [Sounding1D(sounding.frequency, [mode], *(getattr(sounding, name)[k:k + 1] for name in
                                                      ("log_rho", "phase", "log_rho_err", "phase_err")),
                        station=sounding.station) for k, mode in enumerate(sounding.modes)]

    def evaluate(x):
        shifts = x[2 * n:] if n_shift else np.zeros(2)
        rows, predicted = [], {}
        for k, part in enumerate(parts):
            obs, err, pred, J, S, log_rho_a, phase = _mt_rows(part, x[k * n:(k + 1) * n], h, shifts[k:k + 1])
            jac = np.zeros((obs.size, width))
            jac[:, k * n:(k + 1) * n] = J
            if n_shift:
                jac[:, 2 * n + k] = S[:, 0]
            rows.append(((obs - pred) / err, jac / err[:, None]))
            predicted[part.modes[0]] = {"log_rho_a": log_rho_a, "phase": phase}
        residual = np.concatenate([r for r, _ in rows])
        return float(np.sqrt(np.mean(residual ** 2))), residual, np.vstack([j for _, j in rows]), predicted

    lo, hi = np.log10(resistivity_bounds[0]), np.log10(resistivity_bounds[1])
    x = np.r_[np.full(2 * n, np.log10(start)), np.zeros(n_shift)]
    roughness = lambda v: float(np.sum(np.diff(v[:n]) ** 2) + np.sum(np.diff(v[n:2 * n]) ** 2))
    x, rms, predicted, history, mu_used, converged, iterations = _occam_iterations(
        evaluate, x, D, prior, 2 * n, (lo, hi), target_rms, max_iterations, True, log, roughness)
    shifts = {mode: float(10 ** x[2 * n + k]) for k, mode in enumerate(sounding.modes)} if n_shift else {}
    return Anisotropic1DResult(
        strike=angle, resistivity=10.0 ** x[:2 * n].reshape(2, n), thickness=h, rms=rms, mu=float(mu_used),
        static_shift=shifts, predicted=predicted, sounding=sounding, iterations=iterations, history=history,
        converged=converged or rms <= target_rms * 1.01)


def water_content_profile(result: Occam1DResult, *, rhos: Any, n: Any, porosity: Any,
                          sigma_sur: Any = 0.0) -> Dict[str, np.ndarray]:
    """Volumetric water content of each layer from its resistivity (Waxman-Smits).

    The petrophysical link the rest of PyHydroGeophysX uses
    (:func:`PyHydroGeophysX.petrophysics.resistivity_models.resistivity_to_water_content`):
    ``rhos`` the saturated resistivity, ``n`` the saturation exponent,
    ``porosity`` and ``sigma_sur`` the surface conductivity - each one value
    or one per layer.
    """
    from PyHydroGeophysX.petrophysics.resistivity_models import resistivity_to_water_content

    shape = result.resistivity.shape
    per_layer = [np.broadcast_to(np.asarray(v, dtype=float), shape) for v in (rhos, n, porosity, sigma_sur)]
    theta = np.array([resistivity_to_water_content(float(r), float(a), float(b), float(p), float(s))
                      for r, a, b, p, s in zip(result.resistivity, *per_layer)])
    return {"top": result.depth, "water_content": theta, "resistivity": result.resistivity}
