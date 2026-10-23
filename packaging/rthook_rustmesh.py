"""PyInstaller runtime hook for desktop Full builds, which bundle rustmesh: the
bundled libgimli has no Triangle, so pyGIMLi's 2-D meshing goes to rustmesh
through PyHydroGeophysX/core/rustmesh_backend.py from start-up.

PHGX_RUSTMESH_SELFTEST=<file> makes the program check this instead of starting:
the core refuses Triangle, rustmesh meshes a half-space and pyGIMLi models it.
The results go to <file>; the exit code is 0 when every check passes. The
desktop build workflow runs it on the built archive.
"""
import os
import sys

os.environ.setdefault("PYHYDROGEOPHYSX_MESHER", "rustmesh")


def _selftest(out):
    lines, failures = [], []
    try:
        import numpy as np

        import PyHydroGeophysX  # noqa: F401 - installs the backend (PYHYDROGEOPHYSX_MESHER)
        import pygimli as pg
        import pygimli.meshtools as mt
        from pygimli.physics import ert

        from PyHydroGeophysX.core import rustmesh_backend

        lines.append(f"frozen: {getattr(sys, 'frozen', False)}; core {pg.core.versionStr()}")
        world = mt.createWorld(start=[-60.0, 0.0], end=[83.0, -60.0], worldMarker=True)
        scheme = ert.createData(elecs=np.linspace(0.0, 23.0, 24), schemeName="wa")
        for p in scheme.sensors():
            world.createNode(p)
            world.createNode(p - [0.0, 0.1])
        rustmesh_backend.disable()
        try:
            mt.createMesh(world, quality=34)
            failures.append("Triangle: the core made a mesh")
        except RuntimeError as exc:
            lines.append(f"Triangle: the core refuses ({str(exc).strip()[-60:]})")
        rustmesh_backend.enable()
        mesh = mt.createMesh(world, quality=34)
        report = rustmesh_backend.last_report() or {}
        lines.append(f"mesh by rustmesh: {mesh.cellCount()} cells, angle target {report.get('min_angle_target')}, "
                     f"cells below it {report.get('requested_angle_violations')}")
        if report.get("cells") != mesh.cellCount():
            failures.append("rustmesh: the mesh is not the one rustmesh reported")
        data = ert.simulate(mesh, scheme=scheme, res=100.0, noiseLevel=0.0, noiseAbs=0.0, verbose=False)
        rhoa = np.array(data["rhoa"])
        dev = float(np.max(np.abs(rhoa / 100.0 - 1.0)))
        lines.append(f"ERT forward, homogeneous 100 ohm-m: {len(rhoa)} readings, largest deviation {dev:.2%}")
        if not (np.all(np.isfinite(rhoa)) and dev < 0.03):
            failures.append(f"ERT forward: largest deviation {dev:.2%}")
    except Exception as exc:  # noqa: BLE001 - reported in the file
        import traceback
        failures.append(f"{type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
    lines += [f"FAIL: {f}" for f in failures] or ["all checks passed"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 1 if failures else 0


if os.environ.get("PHGX_RUSTMESH_SELFTEST"):
    sys.exit(_selftest(os.environ["PHGX_RUSTMESH_SELFTEST"]))
