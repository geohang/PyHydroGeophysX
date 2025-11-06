@echo off
echo ========================================
echo Building PyHydroGeophysX Documentation
echo ========================================

REM Install requirements
pip install sphinx sphinx-rtd-theme sphinx-gallery

REM Create all necessary directories
mkdir docs\source\_static 2>nul
mkdir docs\build\html 2>nul
mkdir docs\build\html\auto_examples 2>nul
mkdir docs\build\html\auto_examples\images 2>nul

REM Check for figures
if exist docs\source\auto_examples\images\*.png (
    echo ✅ Found figures - copying to build locations
    
    REM Copy to build directory (where Sphinx expects them)
    xcopy docs\source\auto_examples\images\*.png docs\build\html\auto_examples\images\ /Y >nul 2>&1
    
    REM Also copy to _static for direct access
    xcopy docs\source\auto_examples\images\*.png docs\source\_static\ /Y >nul 2>&1
    
    echo Figures copied successfully
) else (
    echo ⚠️  No figures found in docs\source\auto_examples\images\
)

REM Generate API documentation
sphinx-apidoc -f -o docs\source\api PyHydroGeophysX

REM Build documentation
cd docs
sphinx-build -b html source build\html --keep-going -v
cd ..

REM Copy figures again after build (in case Sphinx overwrites)
if exist docs\source\auto_examples\images\*.png (
    xcopy docs\source\auto_examples\images\*.png docs\build\html\auto_examples\images\ /Y >nul 2>&1
    echo Final figure copy completed
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Open: docs\build\html\index.html
echo Gallery: docs\build\html\auto_examples\index.html

pause
start docs\build\html\auto_examples\index.html