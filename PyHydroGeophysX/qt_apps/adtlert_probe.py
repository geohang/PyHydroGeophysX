"""Isolated ADTLERT GPU preflight used by the Qt workbench.

This module intentionally imports no Qt package. It is launched with
``python -m`` so Torch, CuPy, cuDSS, and their native DLLs are loaded in a fresh
process rather than in the long-lived GUI process.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def probe_adtlert_runtime() -> Dict[str, Any]:
    """Return the usable ADTLERT CUDA path in this clean interpreter."""
    import cupy as cp
    import torch
    from nvmath.sparse.advanced import DirectSolver  # noqa: F401

    from PyHydroGeophysX.inversion.ert_inversion import (
        _adtlert_cudss_available,
        _adtlert_solver_name,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot see a CUDA-capable GPU")
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("CuPy cannot see a CUDA-capable GPU")
    if not _adtlert_cudss_available():
        raise RuntimeError("ADTLERT CUDA 12/cuDSS probe failed")
    device = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device, bytes):
        device = device.decode(errors="replace")
    return {
        "device": str(device),
        "forward_solver": "cudss",
        "linearized_solver": _adtlert_solver_name("cgls", prefer_gpu=True),
    }


def main() -> int:
    """Print one machine-readable result for :class:`ProcessProbeWorker`."""
    try:
        payload: Dict[str, Any] = {
            "ok": True,
            "result": probe_adtlert_runtime(),
        }
    except Exception as exc:  # noqa: BLE001 - report optional/native failures
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(payload, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a child process
    raise SystemExit(main())
