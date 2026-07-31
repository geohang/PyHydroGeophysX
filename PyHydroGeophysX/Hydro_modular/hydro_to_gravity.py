"""
Hydrologic-to-gravity conversion helpers for 2D profile workflows.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import griddata

from discretize import TensorMesh
from simpeg import maps
from simpeg.potential_fields import gravity


def _validate_profile_inputs(
    water_content: np.ndarray,
    porosity: np.ndarray,
    layer_boundaries: np.ndarray,
    station_positions: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wc = np.asarray(water_content, dtype=float)
    phi = np.asarray(porosity, dtype=float)
    boundaries = np.asarray(layer_boundaries, dtype=float)

    if wc.ndim != 2:
        raise ValueError(
            f"water_content must be 2D (n_layers, n_stations), got shape {wc.shape}."
        )
    if phi.shape != wc.shape:
        raise ValueError(
            f"porosity shape {phi.shape} must match water_content shape {wc.shape}."
        )

    n_layers, n_stations = wc.shape

    if boundaries.ndim == 1:
        if boundaries.size != n_layers + 1:
            raise ValueError(
                "1D layer_boundaries must have n_layers + 1 values."
            )
        boundaries = np.repeat(boundaries[:, np.newaxis], n_stations, axis=1)
    elif boundaries.ndim == 2:
        expected = (n_layers + 1, n_stations)
        if boundaries.shape != expected:
            raise ValueError(
                f"layer_boundaries shape must be {expected}, got {boundaries.shape}."
            )
    else:
        raise ValueError("layer_boundaries must be 1D or 2D.")

    if station_positions is None:
        x = np.arange(n_stations, dtype=float)
    else:
        x = np.asarray(station_positions, dtype=float).ravel()
        if x.size != n_stations:
            raise ValueError(
                f"station_positions must have {n_stations} entries, got {x.size}."
            )

    return wc, phi, boundaries, x


def hydro_to_gravity(
    water_content: np.ndarray,
    porosity: np.ndarray,
    layer_boundaries: np.ndarray,
    station_positions: Optional[np.ndarray] = None,
    rho_matrix: float = 2650.0,
    rho_water: float = 1000.0,
    rho_air: float = 1.225,
    sensor_height: float = 0.5,
    noise_level: float = 0.01,
    seed: Optional[int] = None,
    mesh_nx: int = 80,
    mesh_nz: int = 60,
    model_width_y: float = 12.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate gravity response from a 2D hydrologic profile using SimPEG.

    The 2D profile is interpolated to a thin 3D TensorMesh (single y strip),
    then ``Simulation3DIntegral`` is used to compute ``gz``.

    Args:
        water_content: Water content matrix, shape (n_layers, n_stations).
        porosity: Porosity matrix, same shape as water_content.
        layer_boundaries: Elevation matrix for layer interfaces,
            shape (n_layers + 1, n_stations), or 1D (n_layers + 1).
        station_positions: Profile x coordinates.
        rho_matrix: Grain density (kg/m^3).
        rho_water: Water density (kg/m^3).
        rho_air: Air density (kg/m^3).
        sensor_height: Sensor height above ground (m).
        noise_level: Relative noise level.
        seed: Random seed.
        mesh_nx: Number of mesh cells in x.
        mesh_nz: Number of mesh cells in z.
        model_width_y: Width of the single y strip (m).
        verbose: Print progress.

    Returns:
        noisy_data: Gravity anomaly with noise (mGal), shape (n_stations,).
        clean_data: Noise-free gravity anomaly (mGal), shape (n_stations,).
        uncertainty: Data uncertainty (mGal), shape (n_stations,).
        density_contrast: Profile density contrast model (kg/m^3),
            shape (n_layers, n_stations).
    """
    wc, phi, boundaries, x = _validate_profile_inputs(
        water_content=water_content,
        porosity=porosity,
        layer_boundaries=layer_boundaries,
        station_positions=station_positions,
    )

    n_layers, n_stations = wc.shape
    saturation = np.clip(wc / np.clip(phi, 1e-6, None), 0.0, 1.0)

    rho_bulk = (1.0 - phi) * rho_matrix + phi * (
        saturation * rho_water + (1.0 - saturation) * rho_air
    )
    rho_background = (1.0 - phi) * rho_matrix + phi * rho_air
    density_contrast = rho_bulk - rho_background

    layer_centers = 0.5 * (boundaries[:-1, :] + boundaries[1:, :])

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    z_top = float(np.max(boundaries[0, :]) + sensor_height + 2.0)
    z_bottom = float(np.min(boundaries[-1, :]) - 5.0)

    if x_max <= x_min:
        x_max = x_min + 1.0
    if z_top <= z_bottom:
        z_top = z_bottom + 1.0

    hx = np.full(int(mesh_nx), (x_max - x_min) / float(mesh_nx), dtype=float)
    hy = np.array([float(model_width_y)], dtype=float)
    hz = np.full(int(mesh_nz), (z_top - z_bottom) / float(mesh_nz), dtype=float)
    mesh = TensorMesh([hx, hy, hz], x0=np.array([x_min, -0.5 * model_width_y, z_bottom]))

    x2d = np.repeat(x[np.newaxis, :], n_layers, axis=0)
    points = np.column_stack((x2d.ravel(), layer_centers.ravel()))
    values = density_contrast.ravel()

    centers = mesh.cell_centers
    query = np.column_stack((centers[:, 0], centers[:, 2]))
    rho_lin = griddata(points, values, query, method="linear")
    rho_near = griddata(points, values, query, method="nearest")
    rho_kgm3 = np.asarray(rho_lin, dtype=float)
    nan_mask = ~np.isfinite(rho_kgm3)
    rho_kgm3[nan_mask] = rho_near[nan_mask]

    surface_z = np.interp(centers[:, 0], x, boundaries[0, :])
    above_surface = centers[:, 2] > surface_z
    rho_kgm3[above_surface] = 0.0

    rho_gcc = rho_kgm3 / 1000.0

    rx_locs = np.column_stack((x, np.zeros_like(x), boundaries[0, :] + sensor_height))
    receivers = gravity.receivers.Point(rx_locs, components="gz")
    source = gravity.sources.SourceField(receiver_list=[receivers])
    survey = gravity.survey.Survey(source)

    simulation = gravity.simulation.Simulation3DIntegral(
        mesh=mesh,
        survey=survey,
        rhoMap=maps.IdentityMap(nP=mesh.nC),
        engine="geoana",
    )

    clean_mgal = np.asarray(simulation.dpred(rho_gcc), dtype=float).ravel()
    rng = np.random.default_rng(seed)
    uncertainty = noise_level * np.maximum(np.abs(clean_mgal), 1e-6)
    noisy_mgal = clean_mgal + rng.normal(scale=uncertainty, size=clean_mgal.size)

    if verbose:
        gmin = float(np.nanmin(clean_mgal))
        gmax = float(np.nanmax(clean_mgal))
        print(
            f"Gravity profile simulation complete: stations={n_stations}, "
            f"range={gmin:.4f} to {gmax:.4f} mGal"
        )

    return noisy_mgal, clean_mgal, uncertainty, density_contrast
