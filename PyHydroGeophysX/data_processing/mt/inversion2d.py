"""2D MT forward modelling and inversion of a profile, on SimPEG's NSEM simulations.

A profile of sites across a 2D structure is modelled on a tensor mesh in the
vertical plane through it: the profile horizontal, the strike out of the
plane, air above the surface. SimPEG (Cockett et al. 2015) solves Maxwell's
equations there for the two polarizations - **TE**, the electric field along
strike, which is ``Zxy`` in the strike frame, and **TM**, the electric field
across strike, ``Zyx`` - with 1D fields as the side boundary conditions.

:func:`invert_profile` rotates each site's impedance to the strike, fits the
real and imaginary parts of the chosen modes (errors: the impedance's own,
with a relative floor) for the logarithm of conductivity below the surface,
with a smoothness regularization whose weight is cooled until the misfit
reaches its target (Gauss-Newton, conjugate gradients). The mesh is laid out
from the frequencies: cells a tenth of the least skin depth at the top,
growing to three times the greatest below and beside the profile.

SimPEG works in east, north, up; here the strike is SimPEG's y. Its
e^{+i omega t} impedances agree with this package's, and on a layered earth
both modes reproduce :func:`.forward1d.impedance_1d` to within the mesh's
discretization: about 1-2% over most of the band, rising to a few per cent at
the highest frequency (9% for TE at 1 kHz over 30 ohm m on the default mesh;
a finer ``top_cell`` halves that but not the TM error, which depends on how
the surface fields are interpolated). On a synthetic block the inversion
recovers 12 ohm m for 10 inside it and 96 for 100 beside it.

Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A. & Oldenburg, D. W.
(2015). SimPEG: an open source framework for simulation and gradient based
parameter estimation in geophysical applications. Computers & Geosciences,
85, 142-154. https://doi.org/10.1016/j.cageo.2015.09.015

deGroot-Hedlin, C. & Constable, S. (1990). Occam's inversion to generate
smooth, two-dimensional models from magnetotelluric data. Geophysics, 55(12),
1613-1624. https://doi.org/10.1190/1.1442813
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .transfer_function import TransferFunction

_AIR_CONDUCTIVITY = 1e-8
_MODES = ("te", "tm")


def _skin_depth(resistivity: float, frequency: float) -> float:
    return 503.0 * np.sqrt(resistivity / frequency)


@dataclass
class ProfileMesh:
    """A 2D tensor mesh for a profile: ``mesh`` (discretize) and its ground cells."""

    mesh: Any
    active: np.ndarray
    station_x: np.ndarray

    @property
    def ground_centers(self) -> np.ndarray:
        return self.mesh.cell_centers[self.active]


def build_profile_mesh(station_x: Sequence[float], frequencies: Sequence[float], *,
                       resistivity: float = 100.0, core_cell: Optional[float] = None,
                       top_cell: Optional[float] = None, vertical_growth: float = 1.12,
                       padding_growth: float = 1.4) -> ProfileMesh:
    """A mesh fitted to the profile and the band.

    ``core_cell`` (default: half the least station spacing, at most a fifth of
    the least skin depth) spans the stations; ``top_cell`` (default: a tenth
    of the least skin depth) starts the ground, which grows by
    ``vertical_growth`` to three greatest skin depths; padding grows by
    ``padding_growth`` beyond the profile as far, and the air above as high.
    """
    import discretize

    x = np.sort(np.asarray(station_x, dtype=float))
    f = np.asarray(frequencies, dtype=float)
    least, greatest = _skin_depth(resistivity, f.max()), _skin_depth(resistivity, f.min())
    spacing = np.min(np.diff(x)) if x.size > 1 else least
    dx = float(core_cell or max(min(spacing / 2.0, least / 5.0), 1.0))
    dz = float(top_cell or max(least / 10.0, 0.5))
    reach = 3.0 * greatest
    n_core = int(np.ceil((x.max() - x.min() + 4 * dx) / dx))
    pad = []
    while sum(pad) < reach:
        pad.append(dx * padding_growth ** (len(pad) + 1))
    hx = np.r_[pad[::-1], np.full(n_core, dx), pad]
    ground = []
    while sum(ground) < reach:
        ground.append(dz * vertical_growth ** len(ground))
    air = []
    while sum(air) < reach:
        air.append(dz * 1.5 ** len(air))
    hz = np.r_[ground[::-1], air]
    centre = 0.5 * (x.min() + x.max())
    origin = [centre - n_core * dx / 2.0 - sum(pad), -float(sum(ground))]
    mesh = discretize.TensorMesh([hx, hz], origin=origin)
    active = mesh.cell_centers[:, 1] < 0.0
    return ProfileMesh(mesh, active, x)


def _solver():
    try:
        from simpeg.utils import get_default_solver

        return get_default_solver()
    except Exception:  # pragma: no cover - older SimPEG
        return None


def _simulations(profile: ProfileMesh, station_x: np.ndarray, frequencies: np.ndarray,
                 modes: Sequence[str], mapping) -> Dict[str, Any]:
    from simpeg.electromagnetics import natural_source as nsem

    locations = np.c_[station_x, np.zeros(station_x.size)]
    out = {}
    solver = _solver()
    for mode in modes:
        if mode not in _MODES:
            raise ValueError(f"modes must be among {_MODES}")
        orientation = "yx" if mode == "te" else "xy"
        receivers = [nsem.receivers.Impedance(locations, orientation=orientation, component=c)
                     for c in ("real", "imag")]
        sources = [nsem.sources.Planewave(receivers, frequency=float(fr)) for fr in frequencies]
        cls = nsem.simulation.Simulation2DMagneticField if mode == "te" else nsem.simulation.Simulation2DElectricField
        kwargs = {"survey": nsem.Survey(sources), "sigmaMap": mapping}
        if solver is not None:
            kwargs["solver"] = solver
        out[mode] = cls(profile.mesh, **kwargs)
    return out


def _unpack(data: np.ndarray, n_freq: int, n_sta: int) -> np.ndarray:
    d = np.asarray(data).reshape(n_freq, 2, n_sta)
    return d[:, 0] + 1j * d[:, 1]


def _ground_map(profile: ProfileMesh):
    """log conductivity of the ground cells -> conductivity of every cell (air fixed)."""
    from simpeg import maps

    inject = maps.InjectActiveCells(profile.mesh, profile.active, np.log(_AIR_CONDUCTIVITY))
    return maps.ExpMap(profile.mesh) * inject


def forward_profile(profile: ProfileMesh, resistivity: Any, frequencies: Sequence[float], *,
                    station_x: Optional[Sequence[float]] = None,
                    modes: Sequence[str] = _MODES) -> Dict[str, np.ndarray]:
    """TE and TM impedances (ohm, strike frame) at the stations: ``{mode: (n_freq, n_sta)}``.

    ``resistivity`` is one value per ground cell (``profile.active``), or a
    callable of the ground cells' centres ``(x, z)``.
    """
    centers = profile.ground_centers
    rho = resistivity(centers) if callable(resistivity) else np.broadcast_to(
        np.asarray(resistivity, dtype=float), (centers.shape[0],))
    x = profile.station_x if station_x is None else np.asarray(station_x, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    sims = _simulations(profile, x, f, modes, _ground_map(profile))
    model = -np.log(np.asarray(rho, dtype=float))
    return {mode: _unpack(sim.dpred(model), f.size, x.size) for mode, sim in sims.items()}


@dataclass
class ProfileInversionResult:
    profile: ProfileMesh
    resistivity: np.ndarray
    frequencies: np.ndarray
    modes: List[str]
    observed: Dict[str, np.ndarray]
    predicted: Dict[str, np.ndarray]
    uncertainty: Dict[str, np.ndarray]
    rms: float
    history: List[Dict[str, float]] = field(default_factory=list)
    #: Impedance values each site gave the fit (one per frequency and mode).
    data_per_site: List[int] = field(default_factory=list)
    #: A line for each site that gave fewer than the profile asks for.
    warnings: List[str] = field(default_factory=list)

    def section(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x_nodes, z_nodes, resistivity grid)`` of the ground, for ``pcolormesh``."""
        mesh = self.profile.mesh
        full = np.full(mesh.n_cells, np.nan)
        full[self.profile.active] = self.resistivity
        grid = full.reshape(mesh.shape_cells, order="F").T
        ground_rows = mesh.cell_centers_y < 0
        return mesh.nodes_x, mesh.nodes_y[: ground_rows.sum() + 1], grid[ground_rows]


#: A site frequency within 5 % of a profile frequency (``|ln(f_site / f)|``
#: below this) is taken as that frequency, as it always was.
PROFILE_FREQUENCY_TOLERANCE = 0.05
#: Widest gap between two of a site's own frequencies across which its
#: impedance is interpolated to a profile frequency: a factor of 2.25, so a
#: band sampled three times per decade still interpolates. Midway across a gap
#: that wide the interpolation is off by at most 2.3 % on three test earths
#: (100/10/300, 1000/1/1000 and 10/1000 ohm m), under half the 5 % error
#: floor; at 5 per decade by 0.8 %, at 10 by 0.2 %.
PROFILE_INTERPOLATION_SPAN = float(np.log(2.25))


def _site_impedance(frequency: np.ndarray, values: np.ndarray, errors: Optional[np.ndarray],
                    fr: float) -> Tuple[complex, float, bool]:
    """One impedance element of a site at the profile frequency ``fr``.

    ``(value, error, interpolated)``, NaN where the site has nothing there.
    A site frequency within :data:`PROFILE_FREQUENCY_TOLERANCE` is used as it
    is. Otherwise the value is interpolated between the site's two frequencies
    that bracket ``fr`` - never extrapolated past its band - when they are at
    most :data:`PROFILE_INTERPOLATION_SPAN` apart: ``Z / sqrt(f)``, whose
    modulus is ``sqrt(rho_a)`` up to a constant, linearly in ``ln f``.
    Interpolating that rather than ``Z`` itself takes out the ``sqrt(f)``
    every impedance carries, which is what made a nearest-frequency match
    wrong: on a three-layer earth sampled at 7.2 frequencies per decade, a
    site recorded 8 % off the profile's frequencies has its ``|Z|`` off by up
    to 6 % at the nearest one, more than the 5 % error floor, and this
    interpolation by 0.2 % (0.3 % at a 20 % offset). The error is the larger
    relative error of the two, applied to the interpolated value.
    """
    log_ratio = np.log(frequency / fr)
    m = int(np.argmin(np.abs(log_ratio)))
    if abs(log_ratio[m]) < PROFILE_FREQUENCY_TOLERANCE:
        error = errors[m] if errors is not None else np.nan
        return values[m], error, False
    below = np.flatnonzero(log_ratio < 0)
    above = np.flatnonzero(log_ratio > 0)
    if not below.size or not above.size:
        return np.nan + 0j, np.nan, False
    lo = int(below[np.argmax(log_ratio[below])])
    hi = int(above[np.argmin(log_ratio[above])])
    span = float(np.log(frequency[hi] / frequency[lo]))
    if span > PROFILE_INTERPOLATION_SPAN or not (
            np.isfinite(values[lo]) and np.isfinite(values[hi])):
        return np.nan + 0j, np.nan, False
    t = float(np.log(fr / frequency[lo])) / span
    scaled = ((1.0 - t) * values[lo] / np.sqrt(frequency[lo])
              + t * values[hi] / np.sqrt(frequency[hi]))
    value = scaled * np.sqrt(fr)
    relative = np.nan
    if errors is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.nanmax([errors[lo] / abs(values[lo]), errors[hi] / abs(values[hi])])
    return value, float(relative * abs(value)), True


def _profile_data(tfs: Sequence[TransferFunction], frequencies: np.ndarray, strike: float,
                  modes: Sequence[str], error_floor: float):
    """The profile's observed impedances and errors, and what each site gave.

    ``counts[k]`` is ``(used, interpolated)``: how many of site ``k``'s
    impedance values (one per profile frequency and mode) hold data, and how
    many of those were interpolated (:func:`_site_impedance`).
    """
    observed, uncertainty = {}, {}
    counts = [[0, 0] for _ in tfs]
    rotated = [tf.rotated(strike) for tf in tfs]
    for mode in modes:
        i, j = (0, 1) if mode == "te" else (1, 0)
        z = np.full((frequencies.size, len(tfs)), np.nan + 0j)
        err = np.full(z.shape, np.nan)
        for k, site in enumerate(rotated):
            errors = site.z_err[:, i, j] if site.z_err is not None else None
            for n, fr in enumerate(frequencies):
                value, error, interpolated = _site_impedance(
                    site.frequency, site.z[:, i, j], errors, fr)
                if not np.isfinite(value):
                    continue
                z[n, k] = value
                err[n, k] = np.nanmax([error, error_floor * abs(value)])
                counts[k][0] += 1
                counts[k][1] += int(interpolated)
        observed[mode], uncertainty[mode] = z, err
    return observed, uncertainty, [tuple(c) for c in counts]


def _site_warnings(tfs: Sequence[TransferFunction], counts, n_frequencies: int,
                   modes: Sequence[str]) -> List[str]:
    """One line per site that gives the profile fewer values than it asks for."""
    total = int(n_frequencies) * len(modes)
    lines = []
    for k, (tf, (used, interpolated)) in enumerate(zip(tfs, counts)):
        if used >= total:
            continue
        name = tf.station or f"site {k + 1}"
        if used == 0:
            lines.append(f"Site {name} contributes no data: its frequencies cover "
                         f"none of the profile's {n_frequencies}, so it is in the "
                         "section only as a station position.")
            continue
        note = f", {interpolated} of them interpolated between its own frequencies" \
            if interpolated else ""
        lines.append(f"Site {name} contributes {used} of {total} impedance values "
                     f"({'/'.join(m.upper() for m in modes)} at {n_frequencies} "
                     f"frequencies){note}; the rest are outside its band, between "
                     "frequencies too far apart, or have no value.")
    return lines


def invert_profile(tfs: Sequence[TransferFunction], station_x: Sequence[float], *,
                   strike: float = 0.0, modes: Sequence[str] = _MODES,
                   frequencies: Optional[Sequence[float]] = None, error_floor: float = 0.05,
                   starting_resistivity: Optional[float] = None, profile: Optional[ProfileMesh] = None,
                   max_iterations: int = 20, alpha_s: float = 1e-4, beta0_ratio: float = 10.0,
                   cooling_factor: float = 2.0, target_chifact: float = 1.0,
                   seed: int = 0, log: Optional[Callable[[str], None]] = None) -> ProfileInversionResult:
    """Smooth 2D resistivity section that fits a profile's TE and/or TM impedances.

    ``tfs`` are the sites' transfer functions and ``station_x`` their
    distances along the profile (m); ``strike`` (degrees clockwise from
    north) is the strike the impedances are rotated to. ``frequencies``
    defaults to those of the first site. A site recorded at other frequencies
    is interpolated to them within its own band (:func:`_site_impedance`);
    a site that gives fewer values than the profile asks for is named in the
    result's ``warnings``, with ``data_per_site`` counting what each gave.
    """
    from simpeg import data as sdata, data_misfit, directives, inverse_problem, inversion, \
        optimization, regularization

    x = np.asarray(station_x, dtype=float)
    f = np.asarray(frequencies if frequencies is not None else tfs[0].frequency, dtype=float)
    observed, uncertainty, counts = _profile_data(tfs, f, strike, modes, error_floor)
    warnings = _site_warnings(tfs, counts, f.size, modes)
    for line in warnings:
        if log is not None:
            log(line)
    if starting_resistivity is None:
        rho_a = [np.abs(observed[m]) ** 2 / (2 * np.pi * f[:, None] * 4e-7 * np.pi) for m in modes]
        starting_resistivity = float(10 ** np.nanmedian(np.log10(np.concatenate([r.ravel() for r in rho_a]))))
    profile = profile or build_profile_mesh(x, f, resistivity=starting_resistivity)
    mapping = _ground_map(profile)
    sims = _simulations(profile, x, f, modes, mapping)
    misfits = []
    for mode in modes:
        obs, unc = observed[mode], uncertainty[mode]
        dobs = np.stack([obs.real, obs.imag], axis=1).ravel()
        std = np.stack([unc, unc], axis=1).ravel()
        keep = np.isfinite(dobs) & np.isfinite(std) & (std > 0)
        dobs = np.where(keep, dobs, 0.0)
        std = np.where(keep, std, np.inf)
        data_object = sdata.Data(sims[mode].survey, dobs=dobs, standard_deviation=std)
        misfits.append(data_misfit.L2DataMisfit(data=data_object, simulation=sims[mode]))
    dmis = misfits[0]
    for extra in misfits[1:]:
        dmis = dmis + extra
    n_active = int(profile.active.sum())
    reg = regularization.WeightedLeastSquares(
        profile.mesh, active_cells=profile.active, alpha_s=alpha_s,
        reference_model=np.full(n_active, -np.log(starting_resistivity)))
    opt = optimization.InexactGaussNewton(maxIter=max_iterations, maxIterCG=30, tolCG=1e-3)
    problem = inverse_problem.BaseInvProblem(dmis, reg, opt)
    history: List[Dict[str, float]] = []

    class _Record(directives.InversionDirective):
        def endIter(self):
            history.append({"iteration": int(self.opt.iter), "phi_d": float(self.invProb.phi_d),
                            "phi_m": float(self.invProb.phi_m), "beta": float(self.invProb.beta)})
            if log is not None:
                log(f"  2D iteration {int(self.opt.iter)}: phi_d {float(self.invProb.phi_d):.4g}, "
                    f"beta {float(self.invProb.beta):.3g}")

    target = directives.TargetMisfit(chifact=target_chifact)
    steps = [directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, random_seed=seed),
             directives.BetaSchedule(coolingFactor=cooling_factor, coolingRate=1),
             target, _Record()]
    inv = inversion.BaseInversion(problem, directiveList=steps)
    m0 = np.full(n_active, -np.log(starting_resistivity))
    model = inv.run(m0)
    predicted = {mode: _unpack(sims[mode].dpred(model), f.size, x.size) for mode in modes}
    n_data = sum(int(np.isfinite(observed[m]).sum()) * 2 for m in modes)
    phi_d = sum(float(np.nansum(((predicted[m] - observed[m]).real / uncertainty[m]) ** 2
                                + ((predicted[m] - observed[m]).imag / uncertainty[m]) ** 2)) for m in modes)
    return ProfileInversionResult(profile, np.exp(-model), f, list(modes), observed, predicted,
                                  uncertainty, float(np.sqrt(phi_d / max(n_data, 1))), history,
                                  data_per_site=[int(used) for used, _ in counts],
                                  warnings=warnings)


def station_distances(tfs: Sequence[TransferFunction]) -> Tuple[np.ndarray, float]:
    """Each site's distance along its profile (m) and the profile's azimuth (deg).

    The sites' latitudes and longitudes are projected on a local plane
    (equirectangular, about their mean); the profile is their principal
    direction, and distances count from the site at its start.
    """
    lat = np.array([tf.latitude for tf in tfs], dtype=float)
    lon = np.array([tf.longitude for tf in tfs], dtype=float)
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("every site needs a latitude and longitude, or give the positions")
    radius = 6371000.0
    north = np.radians(lat - lat.mean()) * radius
    east = np.radians(lon - lon.mean()) * radius * np.cos(np.radians(lat.mean()))
    points = np.column_stack([north, east])
    if len(tfs) < 2:
        return np.zeros(len(tfs)), 0.0
    _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    direction = vt[0]
    along = points @ direction
    if along[-1] < along[0]:
        direction, along = -direction, -along
    azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0)
    return along - along.min(), azimuth
