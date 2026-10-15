"""Check EM native imports in the same isolation used for inversion jobs."""

import json


def main() -> int:
    """Emit the ProcessProbeWorker protocol; never import SimPEG in the GUI."""
    try:
        from PyHydroGeophysX.inversion.em1d_line import backend_status

        payload = {"ok": True, "result": backend_status()}
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(payload, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
