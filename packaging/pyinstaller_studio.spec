# PyInstaller spec for the PyHydroGeophysX Qt desktop studio.
#
# Build (from the repository root, in an environment with the desktop deps):
#     pip install pyinstaller pillow
#     python packaging/make_icons.py     # generates app_icon.ico / app_icon.icns
#     pyinstaller --noconfirm packaging/pyinstaller_studio.spec
#
# Or use the helper scripts, which do all three steps and zip the result:
#     scripts/build_studio_exe.ps1 [light|full]     (Windows)
#     bash scripts/build_studio_exe.sh [light|full] (macOS / Linux)
#
# Variants -- selected with the PHGX_BUILD_VARIANT environment variable:
#
#   light (default)
#       Excludes the heavy engines (pygimli, SimPEG, resipy, pyvista/vtk) so the
#       bundle stays smaller. Features that require an excluded engine remain
#       unavailable; individual panels report their missing dependencies.
#
#   full
#       Collects the supported engines listed below when they are installed
#       in the build environment (pygimli/pgcore, SimPEG, pyvista/vtk, ...).
#       ResIPy requires explicit PHGX_BUNDLE_RESIPY=1 and the tetgen package
#       PHGX_BUNDLE_TETGEN=1; see THIRD_PARTY_NOTICES.md.
#       Much larger output. Engines missing from the build environment are
#       skipped with a warning and degrade gracefully at runtime, same as light.
#
# Notes
# -----
# * The result is dist/PyHydroGeophysX-Studio/ (plus a .app bundle on macOS).
# * upx is disabled: UPX-compressed Qt DLLs are a known source of broken
#   Windows builds.
# * Set ``console=True`` below while debugging to see tracebacks in a terminal.

import os
import re
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH (injected by PyInstaller) is the directory CONTAINING this spec file.
spec_dir = os.path.abspath(SPECPATH)  # noqa: F821 - .../packaging
repo_root = os.path.dirname(spec_dir)
launcher = os.path.join(repo_root, "PyHydroGeophysX", "qt_apps", "launcher.py")

variant = os.environ.get("PHGX_BUILD_VARIANT", "light").strip().lower()
if variant not in ("light", "full"):
    raise SystemExit(f"PHGX_BUILD_VARIANT must be 'light' or 'full', got {variant!r}")
print(f"[studio spec] building the '{variant}' variant")

hiddenimports = []
# The whole package, in both variants. Much of it is imported by name at run
# time, out of PyInstaller's sight: the assistants (BUILTIN in
# agents/assistants), their workflows and tools ("module:attribute"), and the
# package's lazy exports. Engines a module needs stay out through `excludes`.
#
# collect_submodules imports the package to list it, in a child process that
# starts from this sys.path. The build environment need not have the package
# installed (CI installs only its dependencies), so put the checkout first;
# otherwise the list comes back empty without a warning.
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
hiddenimports += collect_submodules("PyHydroGeophysX")
# Stop here rather than ship a studio without its assistants.
_must_bundle = ("PyHydroGeophysX.agents.assistants.aquah",
                "PyHydroGeophysX.agents.assistants.aquah.workflow")
_missing = [name for name in _must_bundle if name not in hiddenimports]
if _missing:
    raise SystemExit("[studio spec] PyHydroGeophysX submodules were not collected "
                     f"(missing {', '.join(_missing)}); is {repo_root} importable?")
# pyqtgraph.examples initializes Qt when imported, which aborts PyInstaller's
# isolated collection child on headless CI runners; the app uses neither the
# examples nor pyqtgraph.opengl, so keep both out of the collection.
hiddenimports += collect_submodules(
    "pyqtgraph",
    filter=lambda name: not name.startswith(("pyqtgraph.examples", "pyqtgraph.opengl")),
)

# Runtime data: the branded logo (theme._logo_path() checks <bundle root>/logo.png)
# and the per-module input-format docs (read via Path(__file__).with_name(...)).
datas = [(os.path.join(repo_root, "logo.png"), "."),
         (os.path.join(repo_root, "LICENSE"), "."),
         (os.path.join(repo_root, "THIRD_PARTY_NOTICES.md"), "."),
         # LGPL-3.0 and the GPL-3.0 it builds on, for the bundled Qt and PySide6.
         # The release workflow also puts these at the archive root.
         (os.path.join(spec_dir, "licenses", "LGPL-3.0-only.txt"), "LICENSES"),
         (os.path.join(spec_dir, "licenses", "GPL-3.0-only.txt"), "LICENSES")]
_compact_examples = os.path.join(repo_root, "PyHydroGeophysX", "data", "qt_examples")
if os.path.isdir(_compact_examples):
    datas.append((_compact_examples, "PyHydroGeophysX/data/qt_examples"))
else:
    print("[studio spec] compact Qt examples absent; use external data in those panels")
# Compact observations used by the two Joint Inversion tutorials.
for _source, _target in (
    (("examples", "data", "ERT", "Bert", "fielddataline2.dat"), "examples/data/ERT/Bert"),
    (("examples", "data", "Seismic", "srtfieldline2.dat"), "examples/data/Seismic"),
    (("examples", "data", "EM", "joint_synthetic_fdem.csv"), "examples/data/EM"),
    (("examples", "data", "EM", "joint_synthetic_tdem.csv"), "examples/data/EM"),
    (("examples", "data", "EM", "synthetic_tem_lci"), "examples/data/EM/synthetic_tem_lci"),
):
    datas.append((os.path.join(repo_root, *_source), _target))
_modules_dir = os.path.join(repo_root, "PyHydroGeophysX", "qt_apps", "modules")
for _name in sorted(os.listdir(_modules_dir)):
    if _name.endswith("_input_format.md"):
        datas.append((os.path.join(_modules_dir, _name), "PyHydroGeophysX/qt_apps/modules"))

binaries = []

# Always excluded: alternative Qt bindings and dead weight. The Google/grpc
# stack gets traced in from optional integrations of the LLM SDKs (openai /
# anthropic) when it happens to be installed in the build environment; the
# desktop app never uses it, and googleapiclient's discovery-cache JSON files
# have names long enough to break Windows MAX_PATH during COLLECT.
excludes = [
    "tkinter", "matplotlib.tests", "PyQt5", "PyQt6", "PySide2",
    "googleapiclient", "google", "grpc",
]

# Installing an optional GPL package into the build environment must not
# silently change the contents of the distributed desktop app.
_bundle_resipy = os.environ.get("PHGX_BUNDLE_RESIPY", "0") == "1"
if not _bundle_resipy:
    excludes.append("resipy")
# The same for the tetgen package, which compiles in TetGen (AGPL-3.0). Without
# it, E4D-style meshes run a TetGen program found on the machine, or Gmsh.
_bundle_tetgen = os.environ.get("PHGX_BUNDLE_TETGEN", "0") == "1"
if not _bundle_tetgen:
    excludes.append("tetgen")
# Gmsh (GPL-2.0-or-later with a limited exception) is run as a separate program
# the user supplies; its Python API must not ride along from the build
# environment either (THIRD_PARTY_NOTICES.md, "Gmsh").
excludes += ["gmsh", "pygmsh"]

# Qt modules the studio does not use, kept out of every archive. Five are
# available under GPL-3.0 (or a commercial licence) only, with no LGPL option:
# Qt Data Visualization, Qt Charts, Qt Graphs, Qt Quick 3D and Qt Virtual
# Keyboard (the "Licenses" section of each module's Qt 6 documentation). Qt PDF
# is LGPL-3.0 but brings PDFium. PySide6's hooks pull some of them in through
# plugins, so besides excluding the bindings here, their libraries, plugins and
# QML folders are dropped from the collected files after Analysis below.
_QT_LEFT_OUT = ("DataVisualization", "Charts", "Graphs", "GraphsWidgets",
                "Quick3D", "VirtualKeyboard", "Pdf", "PdfWidgets")
excludes += [f"PySide6.Qt{_name}" for _name in _QT_LEFT_OUT]
# Windows' opengl32sw.dll is Qt's Mesa llvmpipe software OpenGL fallback. The
# studio draws its Qt views without OpenGL (VTK uses the system's OpenGL), so
# it stays out too.
#
# Matched against each component of a collected file's path, case-insensitively:
# Qt6Charts.dll, QtCharts.framework, libQt6Charts.so, QtCharts.pyd, the qml
# folders (QtCharts, QtQuick/VirtualKeyboard, QtQuick/Pdf), the plugins
# qtvirtualkeyboardplugin and imageformats/qpdf, and opengl32sw.dll.
_LEFT_OUT_PATHS = [re.compile(
    r"^(lib)?qt6?(datavisualization|charts|graphs|quick3d|virtualkeyboard|pdf)"
    r"|^(virtualkeyboard|pdf)$|^qpdf\.|^opengl32sw\.dll$",
    re.IGNORECASE,
)]
# The Universal CRT (ucrtbase.dll and its api-ms-win-*.dll forwarders) is part
# of Windows 10 and later, which always loads its own copy; a conda
# environment carries one for older systems. Qt 6, and so the studio, needs
# Windows 10 or later, so those files stay out.
if sys.platform == "win32":
    _LEFT_OUT_PATHS.append(re.compile(r"^(ucrtbase|api-ms-win-[a-z0-9-]+)\.dll$", re.IGNORECASE))

if variant == "light":
    excludes += ["pygimli", "pgcore", "simpeg", "resipy", "pyvista", "pyvistaqt", "vtk", "vtkmodules"]
else:
    # VTK's FFmpeg reader/writer is the only part of the full stack that loads
    # FFmpeg, and conda-forge's FFmpeg links x264 and x265 (GPL-2.0-or-later).
    # Neither the studio, PyVista nor pyGIMLi uses it. The release workflow
    # installs vtk-base, which leaves it out; should an environment have it,
    # the module and FFmpeg's own libraries stay out of the archive anyway.
    excludes += ["vtkmodules.vtkIOFFMPEG"]
    _LEFT_OUT_PATHS.append(re.compile(
        r"^(lib)?(vtkIOFFMPEG|avcodec|avformat|avutil|avdevice|avfilter|swscale"
        r"|swresample|postproc|x264|x265|openh264|SvtAv1)",
        re.IGNORECASE,
    ))
    # Bundle whichever engines the build environment provides. collect_all picks
    # up their compiled extensions and data files; anything not installed is
    # skipped (the app shows its usual install hint for that engine).
    for _pkg in (
        "pygimli", "pgcore",
        "simpeg", "discretize", "pymatsolver", "geoana",
        "pyvista", "pyvistaqt", "vtk", "vtkmodules",
        "resipy",
    ):
        if _pkg == "resipy" and not _bundle_resipy:
            continue
        try:
            _d, _b, _h = collect_all(_pkg)
        except Exception as _exc:  # noqa: BLE001 - engine absent from the build env
            print(f"[studio spec] full variant: skipping '{_pkg}' ({_exc})")
            continue
        datas += _d
        binaries += _b
        hiddenimports += _h

# Application icon (generated by packaging/make_icons.py from the repo logo.png).
if sys.platform == "win32":
    _icon_candidate = os.path.join(spec_dir, "app_icon.ico")
elif sys.platform == "darwin":
    _icon_candidate = os.path.join(spec_dir, "app_icon.icns")
else:
    _icon_candidate = None
icon = _icon_candidate if _icon_candidate and os.path.exists(_icon_candidate) else None
if _icon_candidate and icon is None:
    print(f"[studio spec] no icon at {_icon_candidate}; run packaging/make_icons.py first")

block_cipher = None

a = Analysis(
    [launcher],
    pathex=[repo_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


def _left_out(entry):
    return any(pattern.match(part) for pattern in _LEFT_OUT_PATHS
               for part in re.split(r"[\\/]", entry[0]))


# PyInstaller resolves a library a binary imports through PATH, which can find
# a copy outside the build environment first: Git's OpenSSL in place of the one
# beside Python's _ssl.pyd, say. Such a copy has no package record to trace its
# version, licence or source. It is swapped for the build environment's file
# of the same name, or left out when the environment has none (Qt's optional
# OpenSSL backend, for one; Qt then uses the system's TLS, as by default).
_ENV_ROOTS = {os.path.normcase(os.path.realpath(p)) for p in (sys.prefix, sys.base_prefix, repo_root)}
_ENV_LIB_DIRS = [sys.base_prefix, os.path.join(sys.base_prefix, "DLLs"), sys.prefix,
                 os.path.join(sys.prefix, "DLLs"), os.path.join(sys.prefix, "Library", "bin"),
                 os.path.join(sys.prefix, "lib")]


def _outside_env(entry):
    src = entry[1] if len(entry) > 1 else None
    if not src:
        return False
    path = os.path.normcase(os.path.realpath(src))
    return not any(path == root or path.startswith(root + os.sep) for root in _ENV_ROOTS)


def _env_copy(name):
    return next((os.path.join(d, name) for d in _ENV_LIB_DIRS if os.path.isfile(os.path.join(d, name))), None)


_dropped = sorted({entry[0] for entry in list(a.binaries) + list(a.datas) if _left_out(entry)})
a.binaries = [entry for entry in a.binaries if not _left_out(entry)]
a.datas = [entry for entry in a.datas if not _left_out(entry)]
print(f"[studio spec] left out {len(_dropped)} files: "
      + ", ".join(os.path.basename(name) for name in _dropped[:20])
      + (" ..." if len(_dropped) > 20 else ""))
_kept, _swapped, _foreign = [], [], []
for _entry in a.binaries:
    if not _outside_env(_entry):
        _kept.append(_entry)
        continue
    _alt = _env_copy(os.path.basename(_entry[1]))
    if _alt:
        _kept.append((_entry[0], _alt, *_entry[2:]))
        _swapped.append(f"{os.path.basename(_entry[0])} ({os.path.dirname(_entry[1])} -> {os.path.dirname(_alt)})")
    else:
        _foreign.append(f"{os.path.basename(_entry[0])} ({os.path.dirname(_entry[1])})")
a.binaries = _kept
_foreign += [f"{os.path.basename(e[0])} ({os.path.dirname(e[1])})" for e in a.datas if _outside_env(e)]
a.datas = [entry for entry in a.datas if not _outside_env(entry)]
print(f"[studio spec] taken from the build environment instead of PATH: {', '.join(_swapped) or 'none'}")
print(f"[studio spec] left out, from outside the build environment: {', '.join(_foreign) or 'none'}")
_still_in = [name for name, *_ in a.pure if any(
    name.startswith(f"PySide6.Qt{_name}") for _name in _QT_LEFT_OUT)]
if _still_in:
    raise SystemExit(f"[studio spec] Qt modules meant to stay out were bundled: {_still_in}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PyHydroGeophysX-Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # set True to debug
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PyHydroGeophysX-Studio",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PyHydroGeophysX-Studio.app",
        icon=icon,
        bundle_identifier="io.github.geohang.pyhydrogeophysx.studio",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
