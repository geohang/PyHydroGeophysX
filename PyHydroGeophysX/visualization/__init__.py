"""Visualization utilities for PyHydroGeophysX."""

# --- Multi-method panels (existing) ---
from .multi_method import (
    plot_cross_section_with_wells,
    plot_hydro_vs_geophys,
    plot_multi_method_panel,
    plot_em_data_fit,
    plot_em_residuals,
    plot_em_fit_and_residuals,
    plot_time_lapse_panel,
    plot_petrophysical_scatter,
    plot_layered_profiles,
)

# --- 2-D plotting ---
from .plotting import (
    plot_model_section,
    plot_timelapse_snapshots,
    plot_difference_map,
    plot_convergence,
    plot_pseudosection_matrix,
    plot_electrode_layout,
    plot_topography,
    plot_monitoring_timeseries,
    plot_coverage,
)

# --- Animations ---
from .animation import (
    create_timelapse_gif,
    create_timelapse_mp4,
    create_difference_gif,
)

# --- VTK / ParaView export ---
from .vtk_export import (
    export_mesh_to_vtk,
    export_structured_vtk,
    export_structured_vtk_multi,
    export_timelapse_vtk,
    export_timelapse_structured_vtk,
    export_points_to_vtk,
)

__all__ = [
    # Multi-method (existing)
    "plot_multi_method_panel",
    "plot_hydro_vs_geophys",
    "plot_cross_section_with_wells",
    "plot_em_data_fit",
    "plot_em_residuals",
    "plot_em_fit_and_residuals",
    "plot_time_lapse_panel",
    "plot_petrophysical_scatter",
    "plot_layered_profiles",
    # 2-D plotting
    "plot_model_section",
    "plot_timelapse_snapshots",
    "plot_difference_map",
    "plot_convergence",
    "plot_pseudosection_matrix",
    "plot_electrode_layout",
    "plot_topography",
    "plot_monitoring_timeseries",
    "plot_coverage",
    # Animations
    "create_timelapse_gif",
    "create_timelapse_mp4",
    "create_difference_gif",
    # VTK export
    "export_mesh_to_vtk",
    "export_structured_vtk",
    "export_structured_vtk_multi",
    "export_timelapse_vtk",
    "export_timelapse_structured_vtk",
    "export_points_to_vtk",
]
