"""Backward-compatible field format imports.

This module keeps old ``PyHydroGeophysX.io`` import paths working while the
canonical location is now ``PyHydroGeophysX.data_processing.io``.
"""

from PyHydroGeophysX.data_processing.io.field_formats import (
    export_results_to_csv,
    export_to_vtk,
    read_seg2_seismic,
    read_tem_fast,
)

__all__ = [
    "read_seg2_seismic",
    "read_tem_fast",
    "export_to_vtk",
    "export_results_to_csv",
]
