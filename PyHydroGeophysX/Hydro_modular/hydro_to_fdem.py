"""
Hydrologic-to-FDEM conversion helpers for 2D profile workflows.
"""

from typing import Optional, Tuple

import numpy as np

from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling


def _validate_profile_inputs(
    water_content: np.ndarray,
    porosity: np.ndarray,
    layer_boundaries: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return wc, phi, boundaries


def hydro_to_fdem(
    water_content: np.ndarray,
    porosity: np.ndarray,
    layer_boundaries: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    sigma_w: float = 0.05,
    m: float = 1.5,
    n: float = 2.0,
    sigma_s: float = 0.0,
    source_location: Optional[np.ndarray] = None,
    receiver_location: Optional[np.ndarray] = None,
    source_radius: float = 10.0,
    receiver_orientation: str = "z",
    receiver_component: str = "secondary",
    waveform_type: str = "dipole",
    noise_level: float = 0.03,
    seed: Optional[int] = None,
    min_thickness: float = 0.1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate pseudo-2D FDEM response from one hydrologic profile.

    A 1D FDEM sounding is simulated at each profile station and stacked into a
    response matrix.

    Args:
        water_content: Water content matrix, shape (n_layers, n_stations).
        porosity: Porosity matrix, same shape as water_content.
        layer_boundaries: Elevation matrix for layer interfaces,
            shape (n_layers + 1, n_stations), or 1D (n_layers + 1).
        frequencies: FDEM frequencies.
        sigma_w: Pore-water conductivity (S/m).
        m: Cementation exponent.
        n: Saturation exponent.
        sigma_s: Surface conductivity (S/m).
        source_location: Source location [x, y, z].
        receiver_location: Receiver location [x, y, z].
        source_radius: Source loop radius (m).
        receiver_orientation: Receiver orientation.
        receiver_component: 'secondary', 'total', or 'both'.
        waveform_type: 'dipole' or 'loop'.
        noise_level: Relative noise level.
        seed: Random seed.
        min_thickness: Lower bound for finite layer thicknesses (m).
        verbose: Print progress.

    Returns:
        noisy_data: Shape (n_stations, n_frequencies), complex.
        clean_data: Shape (n_stations, n_frequencies), complex.
        uncertainty: Shape (n_stations, n_frequencies), float.
        conductivity: Shape (n_layers, n_stations), float.
    """
    wc, phi, boundaries = _validate_profile_inputs(
        water_content=water_content,
        porosity=porosity,
        layer_boundaries=layer_boundaries,
    )

    n_layers, n_stations = wc.shape
    thicknesses_all = np.clip(
        np.abs(np.diff(boundaries, axis=0)),
        min_thickness,
        None,
    )

    if frequencies is None:
        frequencies = np.logspace(1, 4, 16)

    if source_location is None:
        source_location = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        source_location = np.asarray(source_location, dtype=float)

    if receiver_location is None:
        receiver_location = np.array([10.0, 0.0, 0.0], dtype=float)
    else:
        receiver_location = np.asarray(receiver_location, dtype=float)

    clean_matrix = None
    noisy_matrix = None
    unc_matrix = None
    conductivity = np.zeros((n_layers, n_stations), dtype=float)

    base_rng = np.random.default_rng(seed)
    local_seeds = base_rng.integers(0, 2**31 - 1, size=n_stations)

    for j in range(n_stations):
        if n_layers > 1:
            finite_thickness = thicknesses_all[:-1, j]
        else:
            finite_thickness = np.array([], dtype=float)

        noisy_j, clean_j, unc_j, cond_j = FDEMForwardModeling.hydro_to_fdem(
            water_content=wc[:, j],
            porosity=phi[:, j],
            layer_thicknesses=finite_thickness,
            sigma_w=sigma_w,
            m=m,
            n=n,
            sigma_s=sigma_s,
            frequencies=frequencies,
            source_location=source_location,
            receiver_location=receiver_location,
            source_radius=source_radius,
            receiver_orientation=receiver_orientation,
            receiver_component=receiver_component,
            waveform_type=waveform_type,
            noise_level=noise_level,
            seed=int(local_seeds[j]),
        )

        clean_j = np.asarray(clean_j).ravel()
        noisy_j = np.asarray(noisy_j).ravel()
        unc_j = np.asarray(unc_j, dtype=float).ravel()

        if clean_matrix is None:
            n_freq = clean_j.size
            clean_matrix = np.zeros((n_stations, n_freq), dtype=np.complex128)
            noisy_matrix = np.zeros((n_stations, n_freq), dtype=np.complex128)
            unc_matrix = np.zeros((n_stations, n_freq), dtype=float)

        clean_matrix[j, :] = clean_j
        noisy_matrix[j, :] = noisy_j
        unc_matrix[j, :] = unc_j
        conductivity[:, j] = np.asarray(cond_j, dtype=float).ravel()

    if verbose:
        c_min = float(np.nanmin(conductivity))
        c_max = float(np.nanmax(conductivity))
        print(
            f"FDEM profile simulation complete: stations={n_stations}, "
            f"layers={n_layers}, conductivity={c_min:.4e}-{c_max:.4e} S/m"
        )

    return noisy_matrix, clean_matrix, unc_matrix, conductivity
