"""Execute one example notebook without modifying the source notebook.

This helper intentionally uses ``nbclient`` instead of ``nbconvert`` so the
project's notebook smoke tests need only a kernel and the small execution
stack.  The executed notebook and a machine-readable report are written to a
separate output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-timeout", type=int, default=300)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    workdir = (args.workdir or notebook_path.parent).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / notebook_path.name
    report_path = output_dir / f"{notebook_path.stem}.json"

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    last_cell = -1

    def on_cell_start(cell=None, cell_index: int = -1, **_kwargs) -> None:
        nonlocal last_cell
        last_cell = int(cell_index)
        print(f"NOTEBOOK_CELL_START={last_cell}", flush=True)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    client = NotebookClient(
        notebook,
        timeout=args.cell_timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
        on_cell_start=on_cell_start,
    )
    started = time.monotonic()
    status = "passed"
    error_type = ""
    error = ""
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001 - preserve notebook exception
        status = "failed"
        error_type = type(exc).__name__
        error = str(exc)
    elapsed = time.monotonic() - started

    nbformat.write(notebook, output_path)
    report = {
        "notebook": str(notebook_path),
        "workdir": str(workdir),
        "status": status,
        "seconds": round(elapsed, 3),
        "last_cell": last_cell,
        "error_type": error_type,
        "error": error,
        "executed_notebook": str(output_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
