"""Canonical seeded Monte Carlo conversion for ERT-derived petrophysics."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop

ProgressFn = Callable[[str], None]


def _distribution(layer: Mapping[str, Any], key: str, default: float) -> tuple[float, float]:
    raw = layer.get(key, default)
    if isinstance(raw, Mapping):
        return float(raw.get("mean", default)), float(raw.get("std", 0.0))
    return float(raw), 0.0


#: Archie exponents are bounded by what rock can be, not merely by being
#: positive. Flooring them at 1e-6 let a 100%-standard-deviation prior put 31%
#: of the ensemble below n = 1, where the Waxman-Smits residual evaluates
#: 0**(n-1) and the initial guess evaluates rho**(1/n): a divide-by-zero and an
#: overflow per realization, and a third of the ensemble carrying exponents no
#: rock has. Cementation and saturation exponents both sit near 1.3-2.5 in the
#: literature; this envelope is wide enough not to narrow a legitimate prior.
EXPONENT_BOUNDS = (1.0, 4.0)


def _sample_layer(
    rng: np.random.Generator,
    layer: Mapping[str, Any],
) -> Dict[str, float | bool]:
    direct_rhos = bool(
        layer.get("use_rho_sat")
        or "rho_sat" in layer
        or ("rhos" in layer and "m" not in layer)
    )
    sampled: Dict[str, float | bool] = {"use_rho_sat": direct_rhos}
    parameter_defaults = {
        "n": 2.0,
        "sigma_sur": 0.0,
        "porosity": 0.3,
    }
    if direct_rhos:
        key = "rho_sat" if "rho_sat" in layer else "rhos"
        mean, std = _distribution(layer, key, 100.0)
        sampled["rho_sat"] = max(1e-6, float(rng.normal(mean, std)))
    else:
        for key, default in (("m", 1.5), ("rho_fluid", 20.0)):
            mean, std = _distribution(layer, key, default)
            value = float(rng.normal(mean, std))
            if key == "m":
                value = float(np.clip(value, *EXPONENT_BOUNDS))
            else:
                value = max(1e-6, value)
            sampled[key] = value
    for key, default in parameter_defaults.items():
        mean, std = _distribution(layer, key, default)
        value = float(rng.normal(mean, std))
        if key == "n":
            value = float(np.clip(value, *EXPONENT_BOUNDS))
        elif key == "sigma_sur":
            value = max(0.0, value)
        elif key == "porosity":
            value = float(np.clip(value, 0.01, 0.9))
        sampled[key] = value
    return sampled


def _statistics(values: np.ndarray) -> Dict[str, np.ndarray]:
    percentiles = np.percentile(values, [10, 50, 90], axis=0)
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "p10": percentiles[0],
        "p50": percentiles[1],
        "p90": percentiles[2],
    }


def run_petrophysics_monte_carlo(
    resistivity: np.ndarray,
    markers: np.ndarray,
    layers: Sequence[Mapping[str, Any]],
    *,
    products: Sequence[str] = ("water_content",),
    n_realizations: int = 100,
    seed: int = 7,
    saturation_value: float = 1.0,
    tortuosity_a: float = 1.0,
    timestep_indices: Optional[Sequence[int]] = None,
    progress: ProgressFn = _noop,
    return_realizations: bool = False,
    cell_chunk_size: int = 1024,
) -> Dict[str, Any]:
    """Run reproducible layer-wise petrophysical uncertainty propagation.

    Parameters
    ----------
    resistivity : array_like
        Positive, finite resistivity in ohm-m, shaped (cells, times) or (cells,).
        A (times, cells) input is transposed only when its first axis cannot
        match markers. Square arrays are always interpreted as (cells, times).
    markers : array_like
        Integer cell labels. Every label must have exactly one layer definition.
    layers : sequence of mappings
        Each mapping requires ``marker``. Parameters accept a scalar or a
        normal distribution specified by ``mean`` and nonnegative ``std``.
        Resistivities use ohm-m, sigma_sur uses S/m, porosity is a fraction.
        Positive parameters are floored at 1e-6, sigma_sur at zero, and sampled
        porosity is clipped to [0.01, 0.9]. One draw per layer and realization
        is shared by all that layer's cells and timesteps.
    products : sequence of str
        Nonempty subset of ``water_content`` and ``porosity`` (volume fractions).
    n_realizations : int
        Positive number of independent parameter draws.
    seed : int
        Local random seed (default 7); NumPy's global RNG is never modified.
    saturation_value : float
        Assumed saturation for porosity estimation, in (0, 1].
    tortuosity_a : float
        Positive, dimensionless Archie tortuosity factor.
    timestep_indices : sequence of int, optional
        Selected zero-based columns, in output order; defaults to all columns.
    progress : callable
        Receives progress messages as strings.
    return_realizations : bool
        Also return samples shaped (realizations, cells, selected times).
    cell_chunk_size : int
        Maximum cells processed at once. Exact percentiles and the random
        draws are independent of this setting. Full sample storage is only
        allocated when return_realizations is True.

    Returns
    -------
    dict
        ``statistics`` maps products to mean, population std, p10, p50, p90
        arrays shaped (cells, selected times). Also includes ``params_used``,
        ``seed`` and ``timestep_indices``.

    Raises
    ------
    ValueError
        Invalid shapes, non-finite/nonpositive resistivity, missing or duplicate
        layer labels, invalid distributions, or unsupported products.
    """
    from .resistivity_models import (
        resistivity_to_porosity,
        resistivity_to_saturation,
        resistivity_to_saturation2,
    )

    resistivity_array = np.atleast_2d(np.asarray(resistivity, dtype=float))
    raw_markers = np.asarray(markers, dtype=float).ravel()
    if (not raw_markers.size or not np.all(np.isfinite(raw_markers))
            or np.any(raw_markers != np.floor(raw_markers))):
        raise ValueError("markers must contain finite integer labels.")
    marker_array = raw_markers.astype(np.int64)
    if (resistivity_array.ndim != 2 or not np.all(np.isfinite(resistivity_array))
            or np.any(resistivity_array <= 0)):
        raise ValueError("resistivity must be a positive finite 1D or 2D array.")
    if resistivity_array.shape[0] != marker_array.size:
        if resistivity_array.shape[1] == marker_array.size:
            resistivity_array = resistivity_array.T
        else:
            raise ValueError("resistivity cell dimension must match markers.")
    if not layers:
        raise ValueError("At least one layer distribution is required.")
    count = int(n_realizations)
    if count <= 0 or count != n_realizations:
        raise ValueError("n_realizations must be positive.")
    chunk_size = int(cell_chunk_size)
    if chunk_size <= 0 or chunk_size != cell_chunk_size:
        raise ValueError("cell_chunk_size must be a positive integer.")
    if not np.isfinite(saturation_value) or not 0 < saturation_value <= 1:
        raise ValueError("saturation_value must be in (0, 1].")
    if not np.isfinite(tortuosity_a) or tortuosity_a <= 0:
        raise ValueError("tortuosity_a must be positive and finite.")
    time_indices = (
        list(range(resistivity_array.shape[1]))
        if timestep_indices is None
        else [int(index) for index in timestep_indices]
    )
    if not time_indices:
        raise ValueError("timestep_indices cannot be empty.")
    if timestep_indices is not None and any(i != raw for i, raw in zip(time_indices, timestep_indices)):
        raise ValueError("timestep_indices must contain integers.")
    if min(time_indices) < 0 or max(time_indices) >= resistivity_array.shape[1]:
        raise IndexError("timestep_indices contains an out-of-range index.")

    normalized_layers = [dict(layer) for layer in layers]
    layer_ids = []
    for layer in normalized_layers:
        marker = layer.get("marker")
        if marker is None or not np.isfinite(marker) or int(marker) != marker:
            raise ValueError("Each layer must have a finite integer marker.")
        layer_ids.append(int(marker))
        for key in ("m", "rho_fluid", "rho_sat", "rhos", "n", "sigma_sur", "porosity"):
            if key in layer:
                mean, std = _distribution(layer, key, 0.0)
                if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
                    raise ValueError(f"Layer {marker}: {key} needs a finite mean and nonnegative std.")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("Layer markers must be unique.")
    missing = set(marker_array) - set(layer_ids)
    if missing:
        raise ValueError(f"Missing layer definitions for markers: {sorted(missing)}")
    wanted = set(str(product) for product in products)
    if not wanted or wanted - {"water_content", "porosity"}:
        raise ValueError("products must contain water_content and/or porosity.")
    want_water = "water_content" in wanted
    want_porosity = "porosity" in wanted
    shape = (count, marker_array.size, len(time_indices))
    water_all = np.empty(shape) if want_water and return_realizations else None
    saturation_all = np.empty(shape) if return_realizations else None
    porosity_all = np.empty(shape) if want_porosity and return_realizations else None
    parameter_names = (
        "m", "rho_fluid", "rho_sat", "n", "sigma_sur", "porosity", "use_rho_sat"
    )
    params_used = {
        int(layer["marker"]): {
            name: np.zeros(count, dtype=float) for name in parameter_names
        }
        for layer in normalized_layers
    }

    rng = np.random.default_rng(int(seed))
    samples = []
    for realization in range(count):
        sampled = {
            int(layer["marker"]): _sample_layer(rng, layer)
            for layer in normalized_layers
        }
        samples.append(sampled)
        for marker, parameters in sampled.items():
            for name in parameter_names:
                value = parameters.get(name, 0.0)
                params_used[marker][name][realization] = float(value)

    statistics = {
        product: {key: np.empty(shape[1:]) for key in ("mean", "std", "p10", "p50", "p90")}
        for product in sorted(wanted)
    }
    # Draw parameters before chunking, preserving the historical RNG sequence.
    # Only exact per-cell statistics are reduced; no approximate quantiles.
    for start in range(0, marker_array.size, chunk_size):
        stop = min(start + chunk_size, marker_array.size)
        progress(f"Monte Carlo cells {start + 1}-{stop}/{marker_array.size}")
        block_shape = (count, stop - start, len(time_indices))
        water = np.empty(block_shape) if want_water else None
        porosity_block = np.empty(block_shape) if want_porosity else None
        block_markers = marker_array[start:stop]
        layer_indices = {int(m): np.flatnonzero(block_markers == m) for m in np.unique(block_markers)}
        observations = resistivity_array[start:stop, :][:, time_indices]
        for marker, indices in layer_indices.items():
            rho = observations[indices].ravel()
            local_shape = (indices.size, len(time_indices))
            for realization, sampled in enumerate(samples):
                parameters = sampled[marker]
                if want_water or return_realizations:
                    if bool(parameters["use_rho_sat"]):
                        saturation = resistivity_to_saturation2(
                            rho, float(parameters["rho_sat"]),
                            float(parameters["n"]), float(parameters["sigma_sur"]),
                        )
                    else:
                        saturation = resistivity_to_saturation(
                            rho, float(parameters["porosity"]), float(parameters["m"]),
                            float(parameters["rho_fluid"]), float(parameters["n"]),
                            float(parameters["sigma_sur"]), a=float(tortuosity_a),
                        )
                    saturation = np.asarray(saturation).reshape(local_shape)
                    if saturation_all is not None:
                        saturation_all[realization, start + indices, :] = saturation
                    if water is not None:
                        water[realization, indices, :] = saturation * float(parameters["porosity"])
                if porosity_block is not None:
                    if bool(parameters["use_rho_sat"]):
                        porosity = np.full(local_shape, float(parameters["porosity"]))
                    else:
                        porosity = np.asarray(resistivity_to_porosity(
                            rho, float(saturation_value), float(parameters["m"]),
                            float(parameters["rho_fluid"]), float(parameters["n"]),
                            float(parameters["sigma_sur"]), a=float(tortuosity_a),
                        )).reshape(local_shape)
                    porosity_block[realization, indices, :] = porosity
        for product, values, full in (
            ("water_content", water, water_all),
            ("porosity", porosity_block, porosity_all),
        ):
            if values is not None:
                for key, value in _statistics(values).items():
                    statistics[product][key][start:stop] = value
                if full is not None:
                    full[:, start:stop, :] = values
    result: Dict[str, Any] = {
        "statistics": statistics,
        "params_used": params_used,
        "seed": int(seed),
        "timestep_indices": time_indices,
    }
    if return_realizations:
        result.update({
            "water_content_all": water_all,
            "saturation_all": saturation_all,
            "porosity_all": porosity_all,
        })
    return result


__all__ = ["run_petrophysics_monte_carlo"]
