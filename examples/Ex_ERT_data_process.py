"""
ERT Field Data Processing with RESIPY
======================================

This example demonstrates how to load, quality control, and export field ERT data
using PyHydroGeophysX's data_processing module with RESIPY integration.

The workflow includes:
1. Loading ERT data from commercial instruments (E4D example)
2. Automatic quality control with diagnostic plots
3. Export to pyGIMLi/BERT format for inversion

Supports 14+ commercial instruments including E4D, Syscal, ABEM-Lund, Sting, ARES,
Protocol DC/IP, BERT, DAS-1, Electra, and more.
"""
# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_ERT_data_process_fig_01.png'

# %%
# Import Required Modules
# -----------------------
# Import the ERT data processing functions and data structures.

import os
import sys

# Setup package path for development
try:
    # For regular Python scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # For Jupyter notebooks
    current_dir = os.getcwd()

# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PyHydroGeophysX.data_processing.ert_data_agent import (
    load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
)

# %%
# Load ERT Field Data
# -------------------
# Load ERT data from E4D instrument format. The function automatically handles:
# - Coordinate reference systems (local, projected, geographic)
# - Windows/OneDrive permission issues
# - Flexible column name detection
# - Data validation and error checking

ert = load_ert_resipy(
    project_dir="data/ERT/E4D",
    data_file="data/ERT/E4D/2021-10-08_1400.ohm",
    instrument="E4D",
    crs="local",
    local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
)

print(f"Loaded {len(ert.electrodes)} electrodes")
print(f"Loaded {len(ert.observations)} measurements")
print(f"Coordinate system: {ert.crs}")

# %%
# Quality Control and Visualization
# ----------------------------------
# Generate diagnostic plots and statistical summaries:
# - Apparent resistivity histogram (log-scale)
# - Pseudosection visualization
# - Summary statistics (JSON)

artifacts = qc_and_visualize(ert, outdir="results/ert_data_process")

print("\nGenerated QC artifacts:")
for artifact_type, filepath in artifacts.items():
    print(f"  {artifact_type}: {filepath}")

# %%
# Export for Inversion
# --------------------
# Export to pyGIMLi/BERT format with:
# - Electrode coordinates (x, y, z)
# - Measurement data (13 columns including geometric factors, resistance, validity)
# - Compatible with pyGIMLi and BERT inversion codes

bert_path = export_for_inversion(
    ert,
    outdir="results/ert_data_process",
    fmt="pgimli",
    filename="bert_data.dat"
)

print(f"\nExported to pyGIMLi/BERT format: {bert_path}")
print("\nReady for inversion workflow!")

# %%
# Alternative: UTM Coordinates
# ----------------------------
# For data with projected coordinates (e.g., UTM):
#
# .. code-block:: python
#
#     ert_utm = load_ert_resipy(
#         project_dir="data/ERT/Syscal",
#         data_file="data/ERT/Syscal/survey.txt",
#         instrument="Syscal",
#         crs="EPSG:32615",  # UTM Zone 15N
#         epsg=32615
#     )

# %%
# Time-Lapse Processing
# ---------------------
# For time-lapse monitoring, process each timestep separately:
#
# .. code-block:: python
#
#     from pathlib import Path
#     from datetime import datetime
#
#     data_files = [
#         "2021-10-08_1400.ohm",
#         "2021-10-09_1400.ohm",
#         "2021-10-10_1400.ohm",
#     ]
#
#     for data_file in data_files:
#         ert = load_ert_resipy(
#             project_dir="data/ERT/E4D",
#             data_file=f"data/ERT/E4D/{data_file}",
#             instrument="E4D",
#             crs="local",
#             local_ref=LocalRef(0, 0, 90)
#         )
#
#         timestamp = datetime.strptime(Path(data_file).stem, "%Y-%m-%d_%H%M")
#         bert_path = export_for_inversion(
#             ert,
#             outdir="results/time_lapse",
#             filename=f"survey_{timestamp.strftime('%Y%m%d_%H%M')}.dat"
#         )

# %%
# Supported Instruments
# ---------------------
# The following instruments are supported:
#
# - **E4D**: E4D resistivity format (.ohm)
# - **Syscal**: Iris Instruments Syscal (.txt)
# - **ABEM-Lund**: ABEM and Lund Imaging systems
# - **Sting**: AGI Sting systems (.stg)
# - **ARES**: GF Instruments ARES systems (.ares)
# - **Protocol DC/IP**: Iris Instruments Protocol (.pro)
# - **BERT**: pyGIMLi/BERT format (.dat)
# - **DAS-1**: DAS-1 systems
# - **Electra**: Electra systems
# - **ResInv**: ResInv format (.inv)
# - **PRIME/RESIMGR**: Prime/Resimgr format
# - **Lippmann**: Lippmann systems
# - **Custom**: User-defined formats
# - **Merged**: Combined datasets
#
# For complete documentation, see:
# https://geohang.github.io/PyHydroGeophysX/api/data_processing.html
