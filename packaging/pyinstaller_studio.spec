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
#       bundle stays small and reliable. The app degrades gracefully: loading,
#       viewing, QC, picking, geometry editing, and export work in every module;
#       forward modeling, inversion, and the 3D mesh viewer show an install
#       message instead.
#
#   full
#       Bundles every engine that is installed in the build environment
#       (collect_all over pygimli/pgcore, SimPEG, pyvista/vtk, resipy, ...).
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
hiddenimports += collect_submodules("PyHydroGeophysX.qt_apps")
# pyqtgraph.examples initializes Qt when imported, which aborts PyInstaller's
# isolated collection child on headless CI runners; the app uses neither the
# examples nor pyqtgraph.opengl, so keep both out of the collection.
hiddenimports += collect_submodules(
    "pyqtgraph",
    filter=lambda name: not name.startswith(("pyqtgraph.examples", "pyqtgraph.opengl")),
)

# Runtime data: the branded logo (theme._logo_path() checks <bundle root>/logo.png)
# and the per-module input-format docs (read via Path(__file__).with_name(...)).
datas = [(os.path.join(repo_root, "logo.png"), ".")]
datas.append((
    os.path.join(repo_root, "PyHydroGeophysX", "data", "qt_examples"),
    "PyHydroGeophysX/data/qt_examples",
))
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

if variant == "light":
    excludes += ["pygimli", "pgcore", "simpeg", "resipy", "pyvista", "pyvistaqt", "vtk", "vtkmodules"]
else:
    # Bundle whichever engines the build environment provides. collect_all picks
    # up their compiled extensions and data files; anything not installed is
    # skipped (the app shows its usual install hint for that engine).
    hiddenimports += collect_submodules("PyHydroGeophysX")
    for _pkg in (
        "pygimli", "pgcore",
        "simpeg", "discretize", "pymatsolver", "geoana",
        "pyvista", "pyvistaqt", "vtk", "vtkmodules",
        "resipy",
    ):
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
