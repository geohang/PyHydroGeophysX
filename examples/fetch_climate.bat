@echo off
REM Quick script to fetch climate data using the separate environment

if "%~1"=="" (
    echo Usage: fetch_climate.bat config_file.json
    echo   or: fetch_climate.bat --coords lon lat --dates start end --output file.csv
    exit /b 1
)

echo ====================================================================
echo Fetching Climate Data
echo ====================================================================
echo.

REM Activate climate fetch environment
call conda activate climate_fetch
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Climate fetch environment not found.
    echo Please run setup_climate_env.bat first.
    exit /b 1
)

REM Run the fetcher script
python fetch_climate_data.py %*

REM Deactivate environment
call conda deactivate

echo.
echo ====================================================================
echo Climate data fetch completed!
echo ====================================================================
