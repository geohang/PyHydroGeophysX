"""Canonical seeded Monte Carlo conversion for ERT-derived petrophysics."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop

ProgressFn = Callable[[str], None]

#: A layer has to hold enough cells for its own parameters to mean anything.
#: Below this a "layer" is a handful of cells whose statistics are noise.
MIN_CELLS_PER_LAYER = 10


def resolve_layers(cell_markers, n_cells: int,
                   min_cells: int = MIN_CELLS_PER_LAYER):
    """Group cells into geological layers, or into one unit when there is none.

    Cell markers mean different things depending on where the mesh came from. A
    mesh built with regions carries a handful of markers that genuinely separate
    soil from weathered rock from bedrock. An inversion parameter mesh often
    carries one marker per cell, which identifies cells and says nothing about
    geology. Reading the second as layering gives every cell its own
    petrophysical parameter set estimated from one sample.

    The test applied here is whether the markers *group* the mesh: at least two
    values, each covering enough cells to estimate parameters from. When they do
    not, the section is one unit - which is the honest description of a model
    with no structural information, not a failure to find any.

    Parameters
    ----------
    cell_markers : array-like
        One marker per cell, as the mesh reports them.
    n_cells : int
        Number of model cells.
    min_cells : int, optional
        Fewest cells a layer may hold and still be treated as one.

    Returns
    -------
    tuple
        ``(markers, unique_layers, description)`` - the markers to use (all
        zeros for a single unit), their distinct values, and a sentence for the
        log and the report saying which case this is and why.

    Raises
    ------
    None

    Examples
    --------
    >>> import numpy as np
    >>> markers, unique, why = resolve_layers(np.arange(832), 832)
    >>> len(unique), 'single unit' in why
    (1, True)
    >>> markers, unique, why = resolve_layers(np.r_[np.zeros(400), np.ones(432)], 832)
    >>> len(unique), 'layers' in why
    (2, True)
    """
    markers = np.asarray(cell_markers).ravel()
    if markers.size != n_cells:
        markers = np.zeros(n_cells, dtype=int)
        return markers, np.array([0]), (
            "Cell markers did not match the model, so the section is treated as a "
            "single unit with one petrophysical parameter set.")

    unique, counts = np.unique(markers, return_counts=True)
    if unique.size < 2:
        return (markers.astype(int), unique,
                "The mesh reports one region, so the section is a single unit with "
                "one petrophysical parameter set.")
    if counts.min() >= min_cells:
        sizes = ", ".join(str(int(c)) for c in counts)
        return (markers.astype(int), unique,
                f"Using {unique.size} geological layers from the mesh markers "
                f"(cells per layer: {sizes}).")

    # Markers exist but do not group the mesh - typically one per cell, which
    # identifies cells rather than geology.
    detail = (f"{unique.size} markers for {n_cells} cells"
              if unique.size > n_cells // 2 else
              f"the smallest of {unique.size} marker groups holds {int(counts.min())} "
              f"cell(s)")
    return (np.zeros(n_cells, dtype=int), np.array([0]),
            f"The cell markers do not describe geological structure ({detail}), so the "
            f"section is treated as a single unit with one petrophysical parameter set. "
            f"Supplying a layered model is what would let parameters vary with depth.")


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
    if np.isnan(values).any():
        return _statistics_of_solved_draws(values)
    percentiles = np.percentile(values, [10, 50, 90], axis=0)
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "p10": percentiles[0],
        "p50": percentiles[1],
        "p90": percentiles[2],
    }


def _statistics_of_solved_draws(values: np.ndarray) -> Dict[str, np.ndarray]:
    """Statistics over the draws that have a value; NaN where no draw has one.

    A porosity draw with no Waxman-Smits root is NaN. ``np.percentile`` would
    make the whole cell NaN, and ``np.nanpercentile`` loops over cells in
    Python, so the percentiles come from the sorted draws (NaN sorts last),
    with NumPy's default linear interpolation.
    """
    count = np.sum(~np.isnan(values), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(values, axis=0) / count
        std = np.sqrt(np.nansum((values - mean) ** 2, axis=0) / count)
    ordered = np.sort(values, axis=0)
    quantiles = np.array([0.1, 0.5, 0.9]).reshape((-1,) + (1,) * (values.ndim - 1))
    position = np.maximum(count - 1, 0) * quantiles
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, np.maximum(count - 1, 0))
    low = np.take_along_axis(ordered, lower, axis=0)
    high = np.take_along_axis(ordered, upper, axis=0)
    percentiles = np.where(count > 0, low + (high - low) * (position - lower), np.nan)
    return {
        "mean": mean,
        "std": std,
        "p10": percentiles[0],
        "p50": percentiles[1],
        "p90": percentiles[2],
    }


def _no_porosity_warning(marker: int, failed: Mapping[str, Any], params: Mapping[str, np.ndarray],
                         saturation: float, cells_without: int) -> str:
    """One sentence group per layer for the draws whose porosity has no root."""
    draws = sorted(failed["draws"])
    sigma_sur, n = params["sigma_sur"][draws], params["n"][draws]
    limit = 1.0 / (sigma_sur * saturation ** (n - 1.0))

    def span(values) -> str:
        low, high = float(np.min(values)), float(np.max(values))
        return f"{low:.4g}" if np.isclose(low, high) else f"{low:.4g}-{high:.4g}"

    share = 100.0 * failed["count"] / failed["evaluated"]
    return (
        f"Porosity, layer {marker}: {failed['count']} of {failed['evaluated']} cell-survey "
        f"values over the draws ({share:.3g}%) have no Waxman-Smits porosity at the assumed "
        f"saturation S = {saturation:g}. In {len(draws)} of {len(params['n'])} draws the "
        f"surface conductivity (sigma_sur {span(sigma_sur)} S/m, n {span(n)}) alone explains "
        f"resistivity only up to {span(limit)} ohm m, below the measured "
        f"{span(failed['rho'])} ohm m, and porosity only adds conduction; the sigma_sur prior "
        f"is too high for these resistivities. The porosity statistics use the draws with a "
        f"solution (the share without one is statistics['porosity']['no_root_fraction']); "
        f"{cells_without} cell-survey value(s) with no solution in any draw are NaN.")


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
    log_resistivity_std: Optional[np.ndarray] = None,
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
    log_resistivity_std : array_like, optional
        The resistivity's own uncertainty: a standard deviation of
        ln(resistivity) per cell, shaped (cells,) or (cells, times), such as an
        inversion's linearised posterior. Each realization then multiplies the
        resistivity by exp(std * e), e standard normal, drawn independently per
        cell and time from a stream of its own (so the draws do not depend on
        ``cell_chunk_size``), and the statistics carry both sources. Without it
        the resistivity is taken as exact.

    Returns
    -------
    dict
        ``statistics`` maps products to mean, population std, p10, p50, p90
        arrays shaped (cells, selected times). Porosity draws with no
        Waxman-Smits root (surface conduction alone exceeds the measured
        conductivity) are NaN, left out of the porosity statistics, and
        counted in ``statistics["porosity"]["no_root_fraction"]``; a cell
        with no solved draw is NaN. ``warnings`` lists one message per layer
        that had such draws (each also issued as a RuntimeWarning). Also
        includes ``params_used``, ``seed`` and ``timestep_indices``.

    Raises
    ------
    ValueError
        Invalid shapes, non-finite/nonpositive resistivity, missing or duplicate
        layer labels, invalid distributions, or unsupported products.
    """
    from .resistivity_models import (
        _waxman_smits_porosity,
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
    log_std = None
    if log_resistivity_std is not None:
        log_std = np.asarray(log_resistivity_std, dtype=float)
        log_std = log_std.reshape(-1, 1) if log_std.ndim == 1 else log_std
        if (log_std.ndim != 2 or log_std.shape[0] != marker_array.size
                or log_std.shape[1] not in (1, resistivity_array.shape[1])
                or not np.all(np.isfinite(log_std)) or np.any(log_std < 0)):
            raise ValueError("log_resistivity_std needs one finite, non-negative value per "
                             "cell (and per time, or one for all times).")
        if log_std.shape[1] == 1:
            log_std = np.repeat(log_std, resistivity_array.shape[1], axis=1)
        if not np.any(log_std > 0):
            log_std = None

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
    if want_porosity:
        statistics["porosity"]["no_root_fraction"] = np.empty(shape[1:])
    # Per layer: values evaluated, values with no porosity root, the draws
    # they came from and their resistivity range, for one warning per layer.
    no_root: Dict[int, Dict[str, Any]] = {}
    # Draw parameters before chunking, so the RNG sequence does not depend on
    # the chunk size.
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
        # Multiplicative resistivity draws, one stream per cell.
        factors = None
        if log_std is not None:
            block_std = log_std[start:stop, :][:, time_indices]
            factors = np.zeros(block_shape)
            for offset in np.flatnonzero(np.any(block_std > 0, axis=1)):
                stream = np.random.default_rng([int(seed) & 0xFFFFFFFF, 1, start + int(offset)])
                factors[:, offset, :] = (stream.standard_normal((count, len(time_indices)))
                                         * block_std[offset])
            np.exp(factors, out=factors)
        for marker, indices in layer_indices.items():
            exact = observations[indices].ravel()
            local_shape = (indices.size, len(time_indices))
            for realization, sampled in enumerate(samples):
                rho = (exact if factors is None
                       else (observations[indices] * factors[realization, indices, :]).ravel())
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
                        porosity, missing_root = _waxman_smits_porosity(
                            rho, float(saturation_value), float(parameters["m"]),
                            float(parameters["rho_fluid"]), float(parameters["n"]),
                            float(parameters["sigma_sur"]), a=float(tortuosity_a),
                        )
                        porosity = porosity.reshape(local_shape)
                        failed = no_root.setdefault(marker, {
                            "evaluated": 0, "count": 0, "draws": set(),
                            "rho": [np.inf, -np.inf]})
                        failed["evaluated"] += rho.size
                        if missing_root.any():
                            failed["count"] += int(missing_root.sum())
                            failed["draws"].add(realization)
                            unsolved = rho[missing_root]
                            failed["rho"] = [min(failed["rho"][0], unsolved.min()),
                                             max(failed["rho"][1], unsolved.max())]
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
        if porosity_block is not None:
            statistics["porosity"]["no_root_fraction"][start:stop] = (
                np.isnan(porosity_block).mean(axis=0))
    messages = []
    for marker, failed in sorted(no_root.items()):
        if failed["count"]:
            fraction = statistics["porosity"]["no_root_fraction"][marker_array == marker]
            messages.append(_no_porosity_warning(
                marker, failed, params_used[marker], float(saturation_value),
                int(np.sum(fraction == 1.0))))
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    result: Dict[str, Any] = {
        "statistics": statistics,
        "params_used": params_used,
        "seed": int(seed),
        "timestep_indices": time_indices,
        "warnings": messages,
        # Whether the draws varied the resistivity as well as the parameters.
        "resistivity_drawn": log_std is not None,
    }
    if return_realizations:
        result.update({
            "water_content_all": water_all,
            "saturation_all": saturation_all,
            "porosity_all": porosity_all,
        })
    return result


__all__ = ["run_petrophysics_monte_carlo"]
