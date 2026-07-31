"""Dependency-free public types and capability registry for joint inversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

METHODS: Tuple[str, ...] = ("ERT", "SRT", "FDEM", "TDEM", "Gravity", "Magnetics")


@dataclass(frozen=True)
class JointPairCapability:
    """Describe the strategies available for one normalized method pair."""

    methods: Tuple[str, str]
    strategies: Mapping[str, str]
    dimension: str
    model_parameter: str
    implemented: bool
    dependencies: Tuple[str, ...] = ()
    description: str = ""
    runner: Optional[str] = None
    backends: Tuple[str, ...] = ()


@dataclass
class JointInversionRequest:
    """Input contract for a registered joint inversion runner."""

    method_a: str
    method_b: str
    strategy: str
    data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_dir: Union[str, Path] = "results/joint_inversion"
    run_baseline: bool = True


@dataclass
class JointInversionResult:
    """Method-neutral result returned by all registered joint runners."""

    methods: Tuple[str, str]
    strategy: str
    models: Dict[str, Any] = field(default_factory=dict)
    predicted: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    chi2: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    baseline: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"

    def summary(self) -> Dict[str, Any]:
        """Return a compact JSON-ready representation without model arrays."""
        return {
            "status": self.status,
            "methods": list(self.methods),
            "strategy": self.strategy,
            "chi2": dict(self.chi2),
            "iterations": len(self.history),
            "warnings": list(self.warnings),
            "artifacts": dict(self.artifacts),
            "meta": _json_ready(self.meta),
        }


class JointMethodAdapter(Protocol):
    """Interface implemented by a capability-specific request runner."""

    capability: JointPairCapability

    def run(self, request: JointInversionRequest) -> JointInversionResult:
        """Execute a validated request and return method-neutral results."""


_IMPLEMENTED: Dict[Tuple[str, str], JointPairCapability] = {
    ("ERT", "SRT"): JointPairCapability(
        methods=("ERT", "SRT"),
        strategies={
            "cross_gradient_direct": "Direct cross-gradient",
            "cross_gradient_geostatistical": "Spatial/geostatistical cross-gradient",
            "sequential_structure": "Sequential SRT structure → constrained ERT",
        },
        dimension="2-D profile",
        model_parameter="resistivity + velocity",
        implemented=True,
        dependencies=("pygimli",),
        description="Collocated ERT and seismic-refraction travel-time data.",
        runner="ert_srt",
        backends=("pygimli",),
    ),
    ("FDEM", "TDEM"): JointPairCapability(
        methods=("FDEM", "TDEM"),
        strategies={"shared_conductivity": "Shared conductivity"},
        dimension="1-D sounding / matched line",
        model_parameter="electrical conductivity",
        implemented=True,
        dependencies=("simpeg",),
        description="Collocated FDEM and TDEM soundings with a common layered model.",
        runner="fdem_tdem",
        backends=("simpeg", "scipy"),
    ),
    ("Gravity", "Magnetics"): JointPairCapability(
        methods=("Gravity", "Magnetics"),
        strategies={"cross_gradient": "Cross-gradient structural coupling"},
        dimension="3-D shared mesh",
        model_parameter="density contrast + magnetic susceptibility",
        implemented=True,
        dependencies=("simpeg", "discretize", "pymatsolver"),
        description=(
            "Simultaneous SimPEG inversion of overlapping gravity and total-field "
            "magnetic surveys with cross-gradient structural coupling."
        ),
        runner="gravity_magnetics",
        backends=("simpeg",),
    ),
}


def _method_name(method: str) -> str:
    aliases = {
        "ert": "ERT", "srt": "SRT", "seismic": "SRT", "fdem": "FDEM", "tdem": "TDEM",
        "gravity": "Gravity", "magnetic": "Magnetics", "magnetics": "Magnetics",
    }
    normalized = aliases.get(str(method).strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown geophysical method {method!r}. Supported methods: {METHODS}.")
    return normalized


def normalize_joint_pair(method_a: str, method_b: str) -> Tuple[str, str]:
    """Return a stable pair key and reject duplicate methods."""
    a, b = _method_name(method_a), _method_name(method_b)
    if a == b:
        raise ValueError("Joint inversion requires two different geophysical methods.")
    order = {name: index for index, name in enumerate(METHODS)}
    return (a, b) if order[a] < order[b] else (b, a)


def get_joint_capabilities(include_planned: bool = True) -> List[JointPairCapability]:
    """List implemented capabilities and, optionally, planned method pairs."""
    capabilities: List[JointPairCapability] = []
    for pair in combinations(METHODS, 2):
        if pair in _IMPLEMENTED:
            capabilities.append(_IMPLEMENTED[pair])
        elif include_planned:
            capabilities.append(JointPairCapability(
                methods=pair, strategies={}, dimension="mixed / not defined",
                model_parameter="not defined", implemented=False,
                description="Planned: no scientifically validated joint runner is registered.",
            ))
    return capabilities


def get_joint_capability(method_a: str, method_b: str) -> JointPairCapability:
    """Return one capability, including a planned placeholder if unsupported."""
    pair = normalize_joint_pair(method_a, method_b)
    if pair in _IMPLEMENTED:
        return _IMPLEMENTED[pair]
    return JointPairCapability(
        methods=pair, strategies={}, dimension="mixed / not defined", model_parameter="not defined",
        implemented=False, description="Planned: no scientifically validated joint runner is registered.",
    )


def split_joint_soundings(value: Any) -> Tuple[List[Mapping[str, Any]], Optional[np.ndarray]]:
    """Normalize one EM sounding or a line-of-soundings container."""
    if isinstance(value, Mapping) and "soundings" in value:
        entries = [dict(item) for item in value["soundings"]]
        coordinates = value.get("coordinates")
        return entries, None if coordinates is None else np.asarray(coordinates, dtype=float)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        return [dict(item) for item in value], None
    if isinstance(value, Mapping):
        return [dict(value)], None
    raise TypeError("EM data must be a sounding mapping or a sequence of sounding mappings.")


def pair_joint_soundings(
    f_value: Any,
    t_value: Any,
    parameters: Mapping[str, Any],
) -> List[Tuple[int, int, str]]:
    """Pair FDEM/TDEM soundings by coordinate or, when safe, by index."""
    f_soundings, f_coordinates = split_joint_soundings(f_value)
    t_soundings, t_coordinates = split_joint_soundings(t_value)
    if not f_soundings or not t_soundings:
        raise ValueError("FDEM and TDEM must each contain at least one sounding.")
    explicit = parameters.get("pairing_table")
    if explicit is not None:
        pairs: List[Tuple[int, int, str]] = []
        used_f, used_t = set(), set()
        for row in explicit:
            if isinstance(row, Mapping):
                f_index, t_index = int(row["fdem_index"]), int(row["tdem_index"])
            else:
                f_index, t_index = int(row[0]), int(row[1])
            if not 0 <= f_index < len(f_soundings) or not 0 <= t_index < len(t_soundings):
                raise ValueError("Pairing-table indices fall outside the available sounding ranges.")
            if f_index in used_f or t_index in used_t:
                raise ValueError("Pairing table must be one-to-one; duplicate indices were found.")
            used_f.add(f_index); used_t.add(t_index)
            pairs.append((f_index, t_index, "table"))
        if not pairs:
            raise ValueError("Pairing table is empty.")
        return pairs
    if f_coordinates is None or t_coordinates is None:
        if len(f_soundings) != len(t_soundings):
            raise ValueError(
                "FDEM and TDEM sounding counts differ and coordinates are unavailable; provide a pairing table."
            )
        return [(index, index, "index") for index in range(len(f_soundings))]
    if f_coordinates.ndim != 2 or t_coordinates.ndim != 2 or f_coordinates.shape[1] < 2 or t_coordinates.shape[1] < 2:
        raise ValueError("EM line coordinates must have shape (n_soundings, 2 or more).")
    if f_coordinates.shape[0] != len(f_soundings) or t_coordinates.shape[0] != len(t_soundings):
        raise ValueError("Coordinate counts must match their sounding counts.")
    distances = np.linalg.norm(f_coordinates[:, None, :2] - t_coordinates[None, :, :2], axis=2)
    tolerance = parameters.get("pairing_tolerance")
    if tolerance is None:
        samples: List[float] = []
        for coordinates in (f_coordinates, t_coordinates):
            if len(coordinates) > 1:
                delta = np.linalg.norm(np.diff(coordinates[:, :2], axis=0), axis=1)
                samples.extend(delta[np.isfinite(delta) & (delta > 0)].tolist())
        tolerance = 0.5 * float(np.median(samples)) if samples else 1e-6
    tolerance = float(tolerance)
    available = set(range(len(t_soundings)))
    pairs: List[Tuple[int, int, str]] = []
    for f_index in range(len(f_soundings)):
        candidates = sorted(available, key=lambda t_index: distances[f_index, t_index])
        if not candidates or distances[f_index, candidates[0]] > tolerance:
            continue
        t_index = candidates[0]
        available.remove(t_index)
        pairs.append((f_index, t_index, "coordinate"))
    if not pairs:
        raise ValueError(f"No FDEM/TDEM sounding pairs fall within the {tolerance:g} m tolerance.")
    return pairs


def validate_profile_interface(
    interface: Any,
    sensor_x: Any,
    sensor_z: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a 2-D structural interface against an ERT surface profile."""
    if isinstance(interface, tuple) and len(interface) == 2:
        interface_x = np.asarray(interface[0], dtype=float).ravel()
        interface_z = np.asarray(interface[1], dtype=float).ravel()
    else:
        points = np.asarray(interface, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("Structure interface must be (x, z) arrays or an (n, 2) table.")
        interface_x, interface_z = points[:, 0], points[:, 1]
    surface_x = np.asarray(sensor_x, dtype=float).ravel()
    surface_z = np.asarray(sensor_z, dtype=float).ravel()
    if interface_x.size < 2 or interface_x.size != interface_z.size:
        raise ValueError("Structure interface requires at least two matching x/z coordinates.")
    if surface_x.size < 2 or surface_x.size != surface_z.size:
        raise ValueError("ERT surface profile requires at least two matching x/z coordinates.")
    if not all(np.all(np.isfinite(values)) for values in (
        interface_x, interface_z, surface_x, surface_z
    )):
        raise ValueError("Structure interface and ERT surface coordinates must be finite.")
    overlap = ((interface_x >= surface_x.min()) & (interface_x <= surface_x.max()))
    if np.count_nonzero(overlap) < 2:
        raise ValueError("Structure interface does not overlap the ERT profile in x.")
    order = np.argsort(surface_x)
    sorted_x, unique_indices = np.unique(surface_x[order], return_index=True)
    sorted_z = surface_z[order][unique_indices]
    surface_at_interface = np.interp(interface_x[overlap], sorted_x, sorted_z)
    if np.any(interface_z[overlap] >= surface_at_interface):
        raise ValueError(
            "Structure interface must lie below the ERT surface in the same elevation coordinate system."
        )
    return interface_x, interface_z


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "JointInversionRequest", "JointInversionResult", "JointMethodAdapter",
    "JointPairCapability", "METHODS",
    "get_joint_capabilities", "get_joint_capability", "normalize_joint_pair",
    "pair_joint_soundings", "split_joint_soundings", "validate_profile_interface",
]
