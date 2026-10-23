"""Compare the rebuilt pyGIMLi core (rustsolve, no Triangle) with the original
in the same conda environment. The desktop Full build runs, in order:

    python packaging/rustsolve/verify_core.py original verify/orig  # before apply.py install
    python packaging/rustsolve/verify_core.py rebuilt verify/r1     # after it, two processes
    python packaging/rustsolve/verify_core.py rebuilt verify/r2
    python packaging/rustsolve/verify_core.py compare verify --record verify-core.txt

The case is line 2 of the example data (topography, so the primary potentials
come from a P2 mesh) with a 1e-3 ohm-m body in 1e4 ohm-m (contrast 1e7) and a
homogeneous 100 ohm-m model. ``original`` makes the mesh once; every run reads
it. ``compare`` checks:

* responses and Jacobians against the original (CHOLMOD) within the bounds
  below: the rebuilt core changes the solver, not the physics;
* the rebuilt core's primary potentials against its own LinSolver's solution
  of the same P2 system at the P2 nodes: bit for bit, because the gimli patch
  takes those node values instead of interpolating them;
* the two rebuilt runs, in separate processes: bit for bit.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LINE2 = REPO / "examples" / "data" / "ERT" / "Bert" / "fielddataline2.dat"

#: largest relative response difference and largest Jacobian difference over
#: max |J|, rebuilt against original. A 1e7 contrast amplifies the solvers'
#: rounding in the primary potentials (1e-13) through the secondary source.
BOUNDS = {"homogeneous": (1e-8, 1e-10), "contrast": (1e-5, 1e-3)}


def load():
    import pygimli as pg
    from pygimli.physics import ert
    data = ert.load(str(LINE2))
    data["k"] = ert.createGeometricFactors(data, numerical=False)
    return pg, ert, data


def models(fop, data):
    pd = fop.paraDomain
    c = np.array([[cc.center().x(), cc.center().y()] for cc in pd.cells()])
    xs = np.array([q.x() for q in data.sensors()])
    ys = np.array([q.y() for q in data.sensors()])
    x0 = 0.5 * (xs.min() + xs.max())
    y0 = np.interp(x0, xs, ys) - 6.0
    contrast = np.full(pd.cellCount(), 1e4)
    contrast[(c[:, 0] - x0) ** 2 + (c[:, 1] - y0) ** 2 < 16] = 1e-3
    return {"contrast": contrast, "homogeneous": np.full(pd.cellCount(), 100.0)}


def node_hashes(matrix):
    a = np.array(matrix)
    return [hashlib.blake2b(np.ascontiguousarray(a[:, j]).tobytes(), digest_size=8).hexdigest()
            for j in range(a.shape[1])]


def run(out, make_mesh):
    pg, ert, data = load()
    out.mkdir(parents=True, exist_ok=True)
    mesh_file = out.parent / "line2_mesh.bms"
    if make_mesh:
        sys.path.insert(0, str(REPO))
        from PyHydroGeophysX.inversion.ert_mesh import build_inversion_mesh
        build_inversion_mesh(data).save(str(mesh_file))
    mesh = pg.load(str(mesh_file))
    record = {"core": pg.core.versionStr()}
    keep = []
    for case, model in models(_fop(ert, data, mesh), data).items():
        f = _fop(ert, data, mesh)
        resp = np.array(f.response(model))
        f.createJacobian(model)
        jac = np.array(pg.utils.gmat2numpy(f.jacobian()))
        np.savez(out / f"{case}.npz", response=resp, jacobian=jac)
        keep.append(f)
        if case == "contrast":
            record.update(primary(pg, f))
    (out / "record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "primary_hash"}))


def _fop(ert, data, mesh):
    f = ert.ERTModelling()
    f.setData(data)
    f.setMesh(mesh)
    return f


def primary(pg, f):
    """The primary potentials the core computed for ``f``, against the node
    values of the same P2 system solved by the core's own LinSolver (the core
    makes the same P2 mesh at unit resistivity; the forward mesh's nodes are
    its first nodes)."""
    P = pg.Matrix(f._core.primaryPotential())
    if P.rows() == 0:
        raise RuntimeError("no numerical primary potentials: the mesh has no topography")
    fmesh = pg.Mesh(f._core.mesh())
    p2 = fmesh.createP2()
    n = fmesh.nodeCount()
    if not np.array_equal(np.array(p2.positions())[:n], np.array(fmesh.positions())):
        raise RuntimeError("the P2 mesh's first nodes are not the forward mesh's nodes")
    p2.setCellAttributes(1.0)
    dc = pg.DataContainerERT(f._core.data())
    fp = pg.core.DCMultiElectrodeModelling(dc, False)
    fp.setMesh(p2)
    sub = pg.Matrix()
    fp.collectSubPotentials(sub)
    fp.calculate(pg.core.DataMap())
    A, U = np.array(P), np.array(sub)[:, :n]
    d = np.abs(A - U)
    return {"primary_rows": int(A.shape[0]), "primary_nodes": n,
            "primary_nodes_differ": int((d > 0).any(axis=0).sum()),
            "primary_max_rel": float(d.max() / np.abs(U).max()),
            "primary_hash": node_hashes(P)}


def compare(root, record_file):
    lines, failures = [], []
    orig = json.loads((root / "orig" / "record.json").read_text())
    runs = [json.loads((root / r / "record.json").read_text()) for r in ("r1", "r2")]
    lines.append(f"original core {orig['core']}; rebuilt core {runs[0]['core']}")
    lines.append(f"original: primary potentials against its P2 node values: {orig['primary_nodes_differ']} "
                 f"of {orig['primary_nodes']} nodes differ (largest {orig['primary_max_rel']:.1e}; it interpolates)")
    for case, (rbound, jbound) in BOUNDS.items():
        o = np.load(root / "orig" / f"{case}.npz")
        r = [np.load(root / name / f"{case}.npz") for name in ("r1", "r2")]
        dr = float(np.max(np.abs(r[0]["response"] / o["response"] - 1.0)))
        dj = float(np.max(np.abs(r[0]["jacobian"] - o["jacobian"])) / np.max(np.abs(o["jacobian"])))
        lines.append(f"{case}: {o['response'].size} readings; rebuilt against original: response {dr:.1e} "
                     f"(bound {rbound:.0e}), Jacobian {dj:.1e} of max |J| (bound {jbound:.0e})")
        if not (np.isfinite(dr) and dr <= rbound and np.isfinite(dj) and dj <= jbound):
            failures.append(f"{case}: rebuilt against original outside the bounds")
        same = all(np.array_equal(r[0][k], r[1][k]) for k in ("response", "jacobian"))
        lines.append(f"{case}: the two rebuilt runs are {'bit-identical' if same else 'DIFFERENT'}")
        if not same:
            failures.append(f"{case}: the two rebuilt runs differ")
    for name, rec in zip(("r1", "r2"), runs):
        lines.append(f"rebuilt {name}: primary potentials against its own P2 node values: "
                     f"{rec['primary_nodes_differ']} of {rec['primary_nodes']} nodes differ")
        if rec["primary_nodes_differ"]:
            failures.append(f"rebuilt {name}: primary potentials are not the P2 node values")
    if runs[0]["primary_hash"] != runs[1]["primary_hash"]:
        failures.append("the rebuilt primary potentials differ between the two runs")
    lines.append("rebuilt primary potentials: "
                 + ("bit-identical in both runs" if runs[0]["primary_hash"] == runs[1]["primary_hash"]
                    else "DIFFERENT between the runs"))
    lines += [f"FAIL: {f}" for f in failures] or ["all checks passed"]
    text = "\n".join(lines)
    print(text)
    if record_file:
        with open(record_file, "a", encoding="utf-8") as f:
            f.write("\n[verify_core]\n" + text + "\n")
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["original", "rebuilt", "compare"])
    p.add_argument("dir")
    p.add_argument("--record", help="append the comparison to this file")
    a = p.parse_args()
    if a.stage == "compare":
        return compare(Path(a.dir), a.record)
    run(Path(a.dir), make_mesh=a.stage == "original")
    return 0


if __name__ == "__main__":
    sys.exit(main())
