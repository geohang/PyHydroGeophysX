"""
ERT Field Data Processing with RESIPY
======================================

This example demonstrates how to load, quality-control, and export field ERT data
using PyHydroGeophysX's data processing functions.

It now also exposes reusable helper functions so other examples (for example
single-file inversion) can call the same data-processing workflow directly.
"""
# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_ERT_data_process_fig_01.png'

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

# Setup package path for development
try:
    # For regular Python scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # For Jupyter notebooks
    current_dir = os.getcwd()

# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PyHydroGeophysX.data_processing.ert_data_agent import (
    LocalRef,
    export_for_inversion,
    load_ert_resipy,
    qc_and_visualize,
)

SUPPORTED_ERT_EXTENSIONS = (
    ".ohm",
    ".data",
    ".dat",
    ".stg",
    ".ares",
    ".pro",
    ".inv",
    ".txt",
    ".csv",
)

_EXTENSION_TO_INSTRUMENT = {
    ".ohm": "E4D",
    ".data": "DAS-1",
    ".stg": "Sting",
    ".ares": "ARES",
    ".pro": "Protocol DC",
    ".inv": "ResInv",
}


def _resolve_path(path: str | Path) -> Path:
    """Resolve a user path against the current working directory if needed."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    return resolved


def _read_header_text(data_file: Path, max_lines: int = 40) -> str:
    """Read the first few lines of a text file in a robust way."""
    lines = []
    try:
        with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip().lower())
    except OSError:
        return ""
    return "\n".join(lines)


def detect_ert_instrument(data_file: str | Path) -> str:
    """
    Guess instrument type from file extension and simple header checks.

    This is intentionally conservative and can be overridden by passing
    ``instrument=...`` to :func:`process_ert_input`.
    """
    path = _resolve_path(data_file)
    suffix = path.suffix.lower()

    if suffix in _EXTENSION_TO_INSTRUMENT:
        return _EXTENSION_TO_INSTRUMENT[suffix]

    header_text = _read_header_text(path)

    if suffix in (".txt", ".csv"):
        if "protocol" in header_text:
            return "Protocol DC"
        return "Syscal"

    if suffix == ".dat":
        # .dat is commonly BERT/pyGIMLi format.
        return "BERT"

    # Default fallback for unknown text-like ERT files.
    return "BERT"


def find_ert_data_file(
    input_path: str | Path,
    extensions: Iterable[str] = SUPPORTED_ERT_EXTENSIONS,
) -> Path:
    """
    Find an ERT file from either a direct file path or a folder path.

    If a folder contains multiple candidates, the first sorted file is used.
    """
    resolved = _resolve_path(input_path)

    if resolved.is_file():
        return resolved

    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")

    ignored_dirs = {"invdir", "results", "res", "__pycache__", ".git"}
    ignored_name_tokens = ("acknow", "readme", "license")
    preferred = []
    fallback = []
    valid_ext = {ext.lower() for ext in extensions}
    ext_priority = {ext.lower(): i for i, ext in enumerate(extensions)}

    for candidate in sorted(resolved.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in valid_ext:
            continue
        name_lower = candidate.name.lower()
        if any(token in name_lower for token in ignored_name_tokens):
            continue
        is_ignored = any(part.lower() in ignored_dirs for part in candidate.parts)
        if is_ignored:
            fallback.append(candidate)
        else:
            preferred.append(candidate)

    candidates = preferred or fallback
    candidates = sorted(
        candidates,
        key=lambda p: (ext_priority.get(p.suffix.lower(), len(ext_priority)), str(p)),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported ERT files found under: {resolved}\n"
            f"Supported extensions: {', '.join(extensions)}"
        )

    if len(candidates) > 1:
        print(f"Multiple ERT files found under '{resolved}'. Using: {candidates[0]}")

    return candidates[0]


def process_ert_input(
    input_path: str | Path,
    instrument: Optional[str] = None,
    outdir: str | Path = "results/ert_data_process",
    project_dir: Optional[str | Path] = None,
    crs: str = "local",
    epsg: Optional[int] = None,
    local_ref: Optional[LocalRef] = None,
    use_source_error: bool = False,
) -> Dict[str, object]:
    """
    End-to-end processing helper:
    find file -> detect format -> load/QC -> export BERT file.
    """
    data_file = find_ert_data_file(input_path)
    resolved_instrument = instrument or detect_ert_instrument(data_file)

    if local_ref is None:
        local_ref = LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)

    if project_dir is None:
        project_dir = data_file.parent

    ert = load_ert_resipy(
        project_dir=str(project_dir),
        data_file=str(data_file),
        instrument=resolved_instrument,
        crs=crs,
        epsg=epsg,
        local_ref=local_ref,
    )

    artifacts = qc_and_visualize(ert, outdir=str(outdir))
    bert_path = export_for_inversion(
        ert,
        outdir=str(outdir),
        fmt="pgimli",
        use_source_error=use_source_error,
    )

    return {
        "ert": ert,
        "data_file": str(data_file),
        "instrument": resolved_instrument,
        "project_dir": str(project_dir),
        "use_source_error": bool(use_source_error),
        "artifacts": artifacts,
        "bert_path": bert_path,
    }


def run_default_example() -> Dict[str, object]:
    """Run the original E4D processing example."""
    input_file = Path(current_dir) / "data" / "ERT" / "E4D" / "2021-10-08_1400.ohm"
    output_dir = Path(current_dir) / "results" / "ert_data_process"

    result = process_ert_input(
        input_path=input_file,
        instrument="E4D",
        outdir=output_dir,
        project_dir=input_file.parent,
        crs="local",
        local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0),
    )

    ert = result["ert"]
    print(f"Loaded {len(ert.electrodes)} electrodes")
    print(f"Loaded {len(ert.observations)} measurements")
    print(f"Coordinate system: {ert.crs}")
    print(f"Detected instrument: {result['instrument']}")
    print(f"Input file: {result['data_file']}")

    print("\nGenerated QC artifacts:")
    for artifact_type, filepath in result["artifacts"].items():
        print(f"  {artifact_type}: {filepath}")

    print(f"\nExported to pyGIMLi/BERT format: {result['bert_path']}")
    print("\nReady for inversion workflow.")
    return result


if __name__ == "__main__":
    run_default_example()
