"""
Geophysical data processing module for watershed monitoring.
"""

try:
    from PyHydroGeophysX.Geophy_modular.seismic_processor import (
        extract_velocity_structure,
        process_seismic_tomography,
        seismic_velocity_classifier
    )
except ImportError:
    # Handle missing dependencies gracefully
    extract_velocity_structure = None
    process_seismic_tomography = None
    seismic_velocity_classifier = None

try:
    from PyHydroGeophysX.Geophy_modular.structure_integration import (
        create_ert_mesh_with_structure,
        integrate_velocity_interface,
        create_joint_inversion_mesh
    )
except ImportError:
    create_ert_mesh_with_structure = None
    integrate_velocity_interface = None
    create_joint_inversion_mesh = None

try:
    from PyHydroGeophysX.Geophy_modular.ERT_to_WC import (
        ERTtoWC,
        plot_time_series
    )
except ImportError:
    ERTtoWC = None
    plot_time_series = None

__all__ = [
    # Seismic processing functions
    'extract_velocity_structure',
    'process_seismic_tomography',
    'seismic_velocity_classifier',
    
    # Structure integration functions
    'create_ert_mesh_with_structure',
    'integrate_velocity_interface',
    'create_joint_inversion_mesh',

    # ERT to water content conversion
    'ERTtoWC',
    'plot_time_series'
]