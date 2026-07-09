<#
Build the PyHydroGeophysX Qt desktop workbench bundle on Windows.

Usage (from any directory, inside a Python environment):
    powershell -ExecutionPolicy Bypass -File scripts\build_workbench_exe.ps1 [light|full]

Steps: install build dependencies, generate the app icons, run PyInstaller with
the requested variant, and zip the bundle to
dist\PyHydroGeophysX-Workbench-windows-<variant>.zip

The "full" variant bundles whichever geophysics engines (pygimli, SimPEG,
pyvista/vtk, resipy) are installed in the current environment; install them
first if you want them included.
#>
param(
    [ValidateSet("light", "full")]
    [string]$Variant = "light"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

Write-Host "== PyHydroGeophysX workbench build: $Variant =="

python -m pip install -r (Join-Path $repo "requirements-desktop.txt") pyinstaller pillow
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

python (Join-Path $repo "packaging\make_icons.py")
if ($LASTEXITCODE -ne 0) { throw "icon generation failed" }

$env:PHGX_BUILD_VARIANT = $Variant
pyinstaller --noconfirm `
    --distpath (Join-Path $repo "dist") `
    --workpath (Join-Path $repo "build") `
    (Join-Path $repo "packaging\pyinstaller_workbench.spec")
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

$bundle = Join-Path $repo "dist\PyHydroGeophysX-Workbench"
$zip = Join-Path $repo "dist\PyHydroGeophysX-Workbench-windows-$Variant.zip"
Compress-Archive -Path $bundle -DestinationPath $zip -Force

Write-Host "Built $zip"
Write-Host "Smoke test: `$env:QT_QPA_PLATFORM='offscreen'; & '$bundle\PyHydroGeophysX-Workbench.exe' --self-test"
