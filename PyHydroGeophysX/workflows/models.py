"""Serializable workflow contracts shared by the GUI, CLI, and Python API."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional


SCHEMA_VERSION = "1"
_JSON_SCALARS = (str, int, float, bool, type(None))


class WorkflowValidationError(ValueError):
    """A recipe cannot cross a process boundary safely."""


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Return a stable checksum for an artifact on disk."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"


@dataclass(frozen=True)
class ArtifactRef:
    """A serializable reference to a workflow input or output artifact."""

    artifact_id: str
    kind: str
    path: str
    format: str
    checksum: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "$artifact": {
                "artifact_id": self.artifact_id,
                "kind": self.kind,
                "path": self.path,
                "format": self.format,
                "checksum": self.checksum,
                "metadata": _encode_json(self.metadata, "artifact.metadata"),
            }
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        payload = value.get("$artifact", value)
        if not isinstance(payload, Mapping):
            raise WorkflowValidationError("Artifact payload must be a JSON object.")
        return cls(
            artifact_id=str(payload.get("artifact_id", "")),
            kind=str(payload.get("kind", "")),
            path=str(payload.get("path", "")),
            format=str(payload.get("format", "")),
            checksum=str(payload.get("checksum", "")),
            metadata=dict(payload.get("metadata") or {}),
        )

    def resolve(self, base_dir: Path) -> Path:
        candidate = Path(self.path)
        return candidate if candidate.is_absolute() else base_dir / candidate

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        artifact_id: str,
        kind: str,
        format: Optional[str] = None,
        base_dir: Optional[Path] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        checksum: bool = True,
    ) -> "ArtifactRef":
        resolved = path.resolve()
        stored = resolved
        if base_dir is not None:
            try:
                stored = resolved.relative_to(base_dir.resolve())
            except ValueError:
                pass
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            path=stored.as_posix(),
            format=format or resolved.suffix.lstrip(".").lower(),
            checksum=file_checksum(resolved) if checksum and resolved.is_file() else "",
            metadata=dict(metadata or {}),
        )


def _encode_json(value: Any, location: str = "value") -> Any:
    if isinstance(value, ArtifactRef):
        return value.to_dict()
    if isinstance(value, _JSON_SCALARS):
        if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
            raise WorkflowValidationError(f"{location} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        encoded: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkflowValidationError(f"{location} contains a non-string mapping key.")
            encoded[key] = _encode_json(item, f"{location}.{key}")
        return encoded
    if isinstance(value, (list, tuple)):
        return [_encode_json(item, f"{location}[{index}]") for index, item in enumerate(value)]
    raise WorkflowValidationError(
        f"{location} contains non-serializable {type(value).__name__}; "
        "materialize it as an ArtifactRef first."
    )


def _decode_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "$artifact" in value:
            return ArtifactRef.from_dict(value)
        return {str(key): _decode_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    return value


def iter_artifact_refs(value: Any) -> Iterable[ArtifactRef]:
    """Yield every artifact nested in a workflow input/parameter structure."""
    if isinstance(value, ArtifactRef):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_artifact_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_artifact_refs(item)


@dataclass(frozen=True)
class WorkflowSpec:
    """Everything required to reproduce one workflow in another process."""

    workflow_id: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "inputs": _encode_json(self.inputs, "inputs"),
            "parameters": _encode_json(self.parameters, "parameters"),
            "seed": self.seed,
            "dependencies": _encode_json(self.dependencies, "dependencies"),
            "metadata": _encode_json(self.metadata, "metadata"),
        }
        json.dumps(payload, allow_nan=False)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowSpec":
        version = str(value.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise WorkflowValidationError(
                f"Unsupported workflow schema {version!r}; expected {SCHEMA_VERSION!r}."
            )
        seed = value.get("seed")
        return cls(
            schema_version=version,
            workflow_id=str(value.get("workflow_id", "")),
            inputs=_decode_json(value.get("inputs") or {}),
            parameters=_decode_json(value.get("parameters") or {}),
            seed=None if seed is None else int(seed),
            dependencies=[str(item) for item in value.get("dependencies") or []],
            metadata=_decode_json(value.get("metadata") or {}),
        )

    def validate(self, *, stochastic: bool = False) -> None:
        if not self.workflow_id.strip():
            raise WorkflowValidationError("workflow_id is required.")
        if stochastic and self.seed is None:
            raise WorkflowValidationError(
                f"Workflow {self.workflow_id!r} is stochastic and requires an explicit seed."
            )
        self.to_dict()


ProgressFn = Callable[[str], None]
CancelledFn = Callable[[], bool]


def _noop_progress(_message: str) -> None:
    return None


def _never_cancelled() -> bool:
    return False


@dataclass
class RunContext:
    """Process-local execution settings that are deliberately absent from recipes."""

    project_root: Path = field(default_factory=Path.cwd)
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "workflow_results")
    progress: ProgressFn = _noop_progress
    cancelled: CancelledFn = _never_cancelled
    object_cache: MutableMapping[str, Any] = field(default_factory=dict)

    def prepare(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache_key(ref: ArtifactRef) -> str:
        return f"{ref.artifact_id}:{ref.checksum or ref.path}"

    def resolve_artifact(self, ref: ArtifactRef) -> Path:
        path = ref.resolve(self.project_root).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Workflow artifact {ref.artifact_id!r} does not exist: {path}"
            )
        if ref.checksum and path.is_file():
            actual = file_checksum(path, ref.checksum.partition(":")[0] or "sha256")
            if actual != ref.checksum:
                raise WorkflowValidationError(
                    f"Workflow artifact {ref.artifact_id!r} checksum changed: "
                    f"expected {ref.checksum}, got {actual}."
                )
        return path

    def load_object(self, ref: ArtifactRef, loader: Callable[[Path], Any]) -> Any:
        """Load and cache an artifact by ``artifact_id + checksum``."""
        key = self.cache_key(ref)
        if key not in self.object_cache:
            self.object_cache[key] = loader(self.resolve_artifact(ref))
        return self.object_cache[key]


@dataclass
class WorkflowRunResult:
    """Public serializable result plus an optional same-process object channel."""

    status: str
    summary: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactRef] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    objects: MutableMapping[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return the process-safe view. ``objects`` is intentionally excluded."""
        payload = {
            "status": str(self.status),
            "summary": _encode_json(self.summary, "result.summary"),
            "metrics": _encode_json(self.metrics, "result.metrics"),
            "artifacts": [artifact.to_dict()["$artifact"] for artifact in self.artifacts],
            "warnings": _encode_json(self.warnings, "result.warnings"),
            "provenance": _encode_json(self.provenance, "result.provenance"),
        }
        json.dumps(payload, allow_nan=False)
        return payload

    def legacy_payload(self) -> Dict[str, Any]:
        """Return a same-process mapping for existing viewers during migration.

        Unlike :meth:`to_dict`, this deliberately merges the live ``objects``
        channel back into the summary.  It must never be persisted or emitted
        by CLI/bridge code.
        """
        payload = dict(self.summary)
        for key, value in self.metrics.items():
            payload.setdefault(key, value)
        payload.update(self.objects)
        payload.setdefault("status", self.status)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowRunResult":
        return cls(
            status=str(value.get("status", "")),
            summary=_decode_json(value.get("summary") or {}),
            metrics=_decode_json(value.get("metrics") or {}),
            artifacts=[
                ArtifactRef.from_dict({"$artifact": item})
                for item in value.get("artifacts") or []
            ],
            warnings=[str(item) for item in value.get("warnings") or []],
            provenance=_decode_json(value.get("provenance") or {}),
        )


__all__ = [
    "ArtifactRef",
    "RunContext",
    "SCHEMA_VERSION",
    "WorkflowRunResult",
    "WorkflowSpec",
    "WorkflowValidationError",
    "file_checksum",
    "iter_artifact_refs",
]
