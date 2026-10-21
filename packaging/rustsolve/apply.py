"""Put rustsolve in place of pgcore's SuiteSparse solvers in a conda environment.

The desktop Full build (.github/workflows/build-desktop.yml) runs, in order:

    python packaging/rustsolve/apply.py spec          # conda specs to install
    python packaging/rustsolve/apply.py install --prefix "$CONDA_PREFIX"
    python packaging/rustsolve/apply.py check         # in that environment

``install`` replaces pgcore's libgimli with the one in this folder, adds
rustsolve's C interface beside it and removes the SuiteSparse libraries pgcore
carries itself (Windows). It refuses to run unless the environment holds the
pgcore build these files were verified with, every file it replaces or removes
is that package's own, and every file it adds has the SHA-256 that
manifest.json records. ``check`` loads the result: the core reports
"+rustsolve", a small system solves through rustsolve, a homogeneous ERT forward
gives the half-space's resistivity, and no SuiteSparse library is loaded.

Needs only the standard library, except ``check`` (numpy, pygimli).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def platform_key():
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    sys.exit(f"no rustsolve build for {sys.platform}")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_spec(_args):
    spec = MANIFEST["platforms"][platform_key()]
    pg = spec["pgcore"]
    print(f"pgcore={pg['version']}={pg['build']} pygimli={spec['pygimli']}")
    return 0


def cmd_install(args):
    plat = platform_key()
    spec = MANIFEST["platforms"][plat]
    prefix = Path(args.prefix).resolve()
    metas = sorted((prefix / "conda-meta").glob("pgcore-*.json"))
    if len(metas) != 1:
        sys.exit(f"expected one pgcore package in {prefix}, found {[m.name for m in metas]}")
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    want = spec["pgcore"]
    got = {k: meta.get(k) for k in ("version", "build", "md5")}
    if got != {k: want[k] for k in got}:
        sys.exit(f"pgcore in {prefix} is {got}; these files were verified with {want}")
    # SHA-256 of every file the package installed, as conda recorded it
    recorded = {p["_path"]: p.get("sha256") for p in meta.get("paths_data", {}).get("paths", [])}

    def packaged(rel):
        """The installed file is pgcore's own, unchanged."""
        path = prefix / rel
        if rel not in recorded:
            sys.exit(f"{rel} is not a file of pgcore {want['version']}")
        if not path.is_file() or sha256_of(path) != recorded[rel]:
            sys.exit(f"{path} is not the file pgcore {want['version']} installed")
        return path

    lines = [f"rustsolve {MANIFEST['rustsolve']['version']} (commit {MANIFEST['rustsolve']['commit']}) "
             f"in place of pgcore's SuiteSparse solvers: {plat}, pgcore {want['version']} build {want['build']} "
             f"(MD5 {want['md5']}), prefix {prefix}"]
    for entry in spec["files"]:
        src = HERE / entry["file"]
        if sha256_of(src) != entry["sha256"]:
            sys.exit(f"{src} does not have the SHA-256 manifest.json records")
        dest = prefix / entry["install"]
        if entry["install"] in recorded:
            old = sha256_of(packaged(entry["install"]))
            what = f"replaced (was SHA-256 {old})"
        elif dest.exists():
            sys.exit(f"{dest} exists but pgcore did not install it")
        else:
            what = "added"
        shutil.copyfile(src, dest)
        if os.name != "nt":
            os.chmod(dest, 0o755)
        lines.append(f"  {entry['install']}: {what}; now SHA-256 {sha256_of(dest)}")
    for rel in spec["remove"]:
        packaged(rel).unlink()
        lines.append(f"  {rel}: removed")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.record:
        Path(args.record).write_text(text, encoding="utf-8")
    return 0


def loaded_modules():
    """Full paths of the native modules loaded in this process."""
    import ctypes as C
    if sys.platform == "darwin":
        dyld = C.CDLL("/usr/lib/libSystem.B.dylib")
        dyld._dyld_image_count.restype = C.c_uint32
        dyld._dyld_get_image_name.restype = C.c_char_p
        dyld._dyld_get_image_name.argtypes = [C.c_uint32]
        return [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    from ctypes import wintypes as W
    k32, psapi = C.WinDLL("kernel32"), C.WinDLL("psapi")
    k32.GetCurrentProcess.restype = W.HANDLE
    psapi.EnumProcessModules.argtypes = [W.HANDLE, C.POINTER(W.HMODULE), W.DWORD, C.POINTER(W.DWORD)]
    k32.GetModuleFileNameW.argtypes = [W.HMODULE, W.LPWSTR, W.DWORD]
    mods, need = (W.HMODULE * 4096)(), W.DWORD()
    psapi.EnumProcessModules(k32.GetCurrentProcess(), mods, C.sizeof(mods), C.byref(need))
    buf = C.create_unicode_buffer(2048)
    out = []
    for i in range(need.value // C.sizeof(W.HMODULE)):
        k32.GetModuleFileNameW(mods[i], buf, 2048)
        out.append(buf.value)
    return out


def cmd_check(args):
    import numpy as np
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert

    spec = MANIFEST["platforms"][platform_key()]
    failures, lines = [], []
    version = pg.core.versionStr()
    lines.append(f"pyGIMLi {pg.__version__}, core {version}")
    if version != spec["version_string"]:
        failures.append(f"core version {version}, expected {spec['version_string']}")

    # Symmetric, positive definite and well conditioned (condition number < 3).
    n = 200
    A = pg.core.RSparseMapMatrix(n, n)
    for i in range(n):
        A.setVal(i, i, 4.0)
        if i:
            A.setVal(i, i - 1, -1.0)
            A.setVal(i - 1, i, -1.0)
    S = pg.core.RSparseMatrix(A)
    b = np.sin(np.linspace(0.0, 3.0, n))
    solver = pg.core.LinSolver(S)
    x = np.array(solver.solve(pg.Vector(b)))
    dense = np.diag(np.full(n, 4.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    relres = float(np.linalg.norm(dense @ x - b) / np.linalg.norm(b))
    lines.append(f"LinSolver: {solver.solverName()}, relative residual {relres:.2e}")
    if solver.solverName() != "rustsolve" or not relres < 1e-12:
        failures.append(f"LinSolver: solver {solver.solverName()}, relative residual {relres:.2e}")

    # Homogeneous 100 ohm-m half-space, Wenner on 24 electrodes.
    scheme = ert.createData(elecs=np.linspace(0.0, 23.0, 24), schemeName="wa")
    world = mt.createWorld(start=[-60.0, 0.0], end=[83.0, -60.0], worldMarker=True)
    for p in scheme.sensors():
        world.createNode(p)
        world.createNode(p - [0.0, 0.1])
    mesh = mt.createMesh(world, quality=34)
    data = ert.simulate(mesh, scheme=scheme, res=100.0, noiseLevel=0.0, noiseAbs=0.0, verbose=False)
    rhoa = np.array(data["rhoa"])
    dev = float(np.max(np.abs(rhoa / 100.0 - 1.0)))
    lines.append(f"ERT forward, homogeneous 100 ohm-m: {len(rhoa)} apparent resistivities, "
                 f"largest deviation {dev:.2%}")
    if not (np.all(np.isfinite(rhoa)) and dev < args.rhoa_tolerance):
        failures.append(f"ERT forward: largest deviation {dev:.2%} from the half-space")

    mods = loaded_modules()
    suitesparse = [m for m in mods if any(k in Path(m).name.lower() for k in ("cholmod", "umfpack"))]
    rust = [m for m in mods if "rustsolve" in Path(m).name.lower()]
    lines.append(f"loaded: {', '.join(rust) or 'no rustsolve module'}; SuiteSparse: {', '.join(suitesparse) or 'none'}")
    if suitesparse or not rust:
        failures.append("modules: " + lines[-1])

    lines += [f"FAIL: {f}" for f in failures] or ["all checks passed"]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.record:
        with open(args.record, "a", encoding="utf-8") as f:
            f.write(text)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spec")
    ins = sub.add_parser("install")
    ins.add_argument("--prefix", required=True, help="the conda environment")
    ins.add_argument("--record", help="write what was done to this file")
    chk = sub.add_parser("check")
    chk.add_argument("--record", help="append the results to this file")
    chk.add_argument("--rhoa-tolerance", type=float, default=0.03,
                     help="largest relative deviation from the half-space's resistivity")
    args = parser.parse_args(argv)
    return {"spec": cmd_spec, "install": cmd_install, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
