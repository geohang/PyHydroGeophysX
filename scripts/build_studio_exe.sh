#!/usr/bin/env bash
# Build the PyHydroGeophysX Qt desktop studio bundle on macOS / Linux.
#
# Usage (from any directory, inside a Python environment):
#     bash scripts/build_studio_exe.sh [light|full]
#
# Steps: install build dependencies, generate the app icons, run PyInstaller
# with the requested variant, and zip the bundle to
# dist/PyHydroGeophysX-Studio-<os>-<variant>.zip
#
# The "full" variant bundles whichever geophysics engines (pygimli, SimPEG,
# pyvista/vtk, resipy) are installed in the current environment; install them
# first if you want them included.
set -euo pipefail

variant="${1:-light}"
case "$variant" in
  light|full) ;;
  *) echo "usage: $0 [light|full]" >&2; exit 1 ;;
esac

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== PyHydroGeophysX studio build: $variant =="

python -m pip install -r "$repo/requirements-desktop.txt" pyinstaller pillow
python "$repo/packaging/make_icons.py"

export PHGX_BUILD_VARIANT="$variant"
pyinstaller --noconfirm \
  --distpath "$repo/dist" \
  --workpath "$repo/build" \
  "$repo/packaging/pyinstaller_studio.spec"

os_name="linux"
if [ "$(uname -s)" = "Darwin" ]; then os_name="macos"; fi
out="$repo/dist/PyHydroGeophysX-Studio-${os_name}-${variant}.zip"
rm -f "$out"

if [ "$os_name" = "macos" ] && [ -d "$repo/dist/PyHydroGeophysX-Studio.app" ]; then
  # ditto preserves the .app structure, symlinks, and permissions
  ditto -c -k --keepParent "$repo/dist/PyHydroGeophysX-Studio.app" "$out"
else
  (cd "$repo/dist" && zip -qr "$(basename "$out")" "PyHydroGeophysX-Studio")
fi

echo "Built $out"
echo "Smoke test: QT_QPA_PLATFORM=offscreen \"$repo/dist/PyHydroGeophysX-Studio/PyHydroGeophysX-Studio\" --self-test"
