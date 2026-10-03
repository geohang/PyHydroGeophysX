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

    def section(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x_nodes, z_nodes, resistivity grid)`` of the ground, for ``pcolormesh``."""
        mesh = self.profile.mesh
        full = np.full(mesh.n_cells, np.nan)
        full[self.profile.active] = self.resistivity
        grid = full.reshape(mesh.shape_cells, order="F").T
        ground_rows = mesh.cell_centers_y < 0
        return mesh.nodes_x, mesh.nodes_y[: ground_rows.sum() + 1], grid[ground_rows]


def _profile_data(tfs: Sequence[TransferFunction], frequencies: np.ndarray, strike: float,
                  modes: Sequence[str], error_floor: float):
    observed, uncertainty = {}, {}
    for mode in modes:
        i, j = (0, 1) if mode == "te" else (1, 0)
        z = np.full((frequencies.size, len(tfs)), np.nan + 0j)
        err = np.full(z.shape, np.nan)
        for k, tf in enumerate(tfs):
            rotated = tf.rotated(strike)
            for n, fr in enumerate(frequencies):
                m = int(np.argmin(np.abs(np.log(rotated.frequency / fr))))
                if abs(np.log(rotated.frequency[m] / fr)) < 0.05:
                    z[n, k] = rotated.z[m, i, j]
                    e = rotated.z_err[m, i, j] if rotated.z_err is not None else np.nan
                    err[n, k] = np.nanmax([e, error_floor * abs(z[n, k])])
        observed[mode], uncertainty[mode] = z, err
    return observed, uncertainty


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
    defaults to those of the first site, which every site should share.
    """
    from simpeg import data as sdata, data_misfit, directives, inverse_problem, inversion, \
        optimization, regularization

    x = np.asarray(station_x, dtype=float)
    f = np.asarray(frequencies if frequencies is not None else tfs[0].frequency, dtype=float)
    observed, uncertainty = _profile_data(tfs, f, strike, modes, error_floor)
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
                                  uncertainty, float(np.sqrt(phi_d / max(n_data, 1))), history)


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
