# PyInstaller spec for the PyHydroGeophysX Qt desktop workbench.
#
# Build (from the repository root, in an environment with the desktop deps):
#     pip install pyinstaller
#     pyinstaller packaging/pyinstaller_workbench.spec
#
# The result is dist/PyHydroGeophysX-Workbench/ containing the launcher executable.
#
# Notes
# -----
# * The heavy optional engines (pygimli, SimPEG, resipy, pyvista/vtk) are EXCLUDED to
#   keep the bundle buildable and small. The app degrades gracefully without them:
#   the Mesh 3D page shows an install message and Hydro to Geophysics exports a survey
#   configuration JSON instead of running the real forward model. To ship a build that
#   runs forward modeling, remove those names from ``excludes`` and expect a much larger
#   bundle plus platform-specific native-library handling.
# * Set ``console=True`` below while debugging to see tracebacks in a terminal.

import os

from PyInstaller.utils.hooks import collect_submodules

spec_dir = os.path.dirname(os.path.abspath(SPECPATH))  # noqa: F821 - injected by PyInstaller
repo_root = os.path.dirname(spec_dir)
launcher = os.path.join(repo_root, "PyHydroGeophysX", "qt_apps", "launcher.py")

hiddenimports = []
hiddenimports += collect_submodules("PyHydroGeophysX.qt_apps")
hiddenimports += collect_submodules("pyqtgraph")

block_cipher = None

a = Analysis(
    [launcher],
    pathex=[repo_root],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pygimli", "simpeg", "resipy", "pyvista", "pyvistaqt", "vtk",
        "tkinter", "matplotlib.tests", "PyQt5",
    ],
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
    name="PyHydroGeophysX-Workbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # set True to debug
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PyHydroGeophysX-Workbench",
)
