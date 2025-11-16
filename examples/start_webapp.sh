#!/bin/bash
# PyHydroGeophysX Web Application Launcher (Linux/Mac)
# This script launches the Streamlit web interface

echo "========================================"
echo "PyHydroGeophysX Web Application"
echo "========================================"
echo ""
echo "Starting Streamlit server..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "[ERROR] Streamlit is not installed!"
    echo "Please install it with: pip install streamlit"
    echo ""
    exit 1
fi

# Launch the app
streamlit run app_geophysics_workflow.py

