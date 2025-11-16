@echo off
echo ========================================
echo Starting PyHydroGeophysX Web Interface
echo ========================================
echo.
echo The web app will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
C:\Users\hchen117\.conda\envs\pg\python.exe -m streamlit run app_geophysics_workflow.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

pause
