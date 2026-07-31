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
            sampled[key] = max(1e-6, float(rng.normal(mean, std)))
    for key, default in parameter_defaults.items():
        mean, std = _distribution(layer, key, default)
        value = float(rng.normal(mean, std))
        if key == "n":
            value = max(1e-6, value)
        elif key == "sigma_sur":
            value = max(0.0, value)
        elif key == "porosity":
            value = float(np.clip(value, 0.01, 0.9))
        sampled[key] = value
    return sampled


def _statistics(values: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "p10": np.percentile(values, 10, axis=0),
        "p50": np.percentile(values, 50, axis=0),
        "p90": np.percentile(values, 90, axis=0),
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
) -> Dict[str, Any]:
    """Run reproducible layer-wise petrophysical uncertainty propagation.

    ``seed`` is mandatory at the workflow boundary and this function never
    reads or mutates NumPy's global RNG state.
    """
    from .resistivity_models import (
        resistivity_to_porosity,
        resistivity_to_saturation,
        resistivity_to_saturation2,
    )

    resistivity_array = np.atleast_2d(np.asarray(resistivity, dtype=float))
    marker_array = np.asarray(markers, dtype=int).ravel()
    if resistivity_array.shape[0] != marker_array.size:
        if resistivity_array.shape[1] == marker_array.size:
            resistivity_array = resistivity_array.T
        else:
            raise ValueError("resistivity cell dimension must match markers.")
    if not layers:
        raise ValueError("At least one layer distribution is required.")
    count = int(n_realizations)
    if count <= 0:
        raise ValueError("n_realizations must be positive.")
    time_indices = (
        list(range(resistivity_array.shape[1]))
        if timestep_indices is None
        else [int(index) for index in timestep_indices]
    )
    if not time_indices:
        raise ValueError("timestep_indices cannot be empty.")
    if min(time_indices) < 0 or max(time_indices) >= resistivity_array.shape[1]:
        raise IndexError("timestep_indices contains an out-of-range index.")

    normalized_layers = [dict(layer) for layer in layers]
    layer_masks = {
        int(layer["marker"]): marker_array == int(layer["marker"])
        for layer in normalized_layers
    }
    wanted = set(str(product) for product in products)
    want_water = "water_content" in wanted
    want_porosity = "porosity" in wanted
    shape = (count, marker_array.size, len(time_indices))
    water_all = np.zeros(shape, dtype=float) if want_water else None
    saturation_all = np.zeros(shape, dtype=float)
    porosity_all = np.zeros(shape, dtype=float) if want_porosity else None
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
    for realization in range(count):
        if realization % max(1, count // 10) == 0:
            progress(f"Monte Carlo realization {realization + 1}/{count}")
        sampled = {
            int(layer["marker"]): _sample_layer(rng, layer)
            for layer in normalized_layers
        }
        porosity_cells = np.zeros(marker_array.size, dtype=float)
        for marker, parameters in sampled.items():
            mask = layer_masks[marker]
            porosity_cells[mask] = float(parameters["porosity"])
            for name in parameter_names:
                value = parameters.get(name, 0.0)
                params_used[marker][name][realization] = float(value)

        for output_column, time_index in enumerate(time_indices):
            resistivity_time = resistivity_array[:, time_index]
            for marker, mask in layer_masks.items():
                if not np.any(mask):
                    continue
                parameters = sampled[marker]
                if bool(parameters["use_rho_sat"]):
                    saturation = resistivity_to_saturation2(
                        resistivity_time[mask],
                        float(parameters["rho_sat"]),
                        float(parameters["n"]),
                        float(parameters["sigma_sur"]),
                    )
                else:
                    saturation = resistivity_to_saturation(
                        resistivity=resistivity_time[mask],
                        porosity=float(parameters["porosity"]),
                        m=float(parameters["m"]),
                        rho_fluid=float(parameters["rho_fluid"]),
                        n=float(parameters["n"]),
                        sigma_sur=float(parameters["sigma_sur"]),
                        a=float(tortuosity_a),
                    )
                saturation_all[realization, mask, output_column] = np.asarray(
                    saturation, dtype=float
                )
                if want_porosity:
                    if bool(parameters["use_rho_sat"]):
                        porosity = np.full(
                            int(mask.sum()), float(parameters["porosity"]), dtype=float
                        )
                    else:
                        porosity = resistivity_to_porosity(
                            resistivity=resistivity_time[mask],
                            saturation=float(saturation_value),
                            m=float(parameters["m"]),
                            rho_fluid=float(parameters["rho_fluid"]),
                            n=float(parameters["n"]),
                            sigma_sur=float(parameters["sigma_sur"]),
                            a=float(tortuosity_a),
                        )
                    porosity_all[realization, mask, output_column] = np.asarray(
                        porosity, dtype=float
                    )
            if want_water:
                water_all[realization, :, output_column] = (
                    saturation_all[realization, :, output_column] * porosity_cells
                )

    statistics: Dict[str, Dict[str, np.ndarray]] = {}
    if want_water and water_all is not None:
        statistics["water_content"] = _statistics(water_all)
    if want_porosity and porosity_all is not None:
        statistics["porosity"] = _statistics(porosity_all)
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
