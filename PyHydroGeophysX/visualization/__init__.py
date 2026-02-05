"""Multi-method visualization utilities."""

from .multi_method import (
    plot_cross_section_with_wells,
    plot_hydro_vs_geophys,
    plot_multi_method_panel,
)

__all__ = [
    "plot_multi_method_panel",
    "plot_hydro_vs_geophys",
    "plot_cross_section_with_wells",
]

from .multi_method import (
    plot_em_data_fit,
    plot_em_residuals,
    plot_em_fit_and_residuals,
    plot_time_lapse_panel,
    plot_petrophysical_scatter,
    plot_layered_profiles,
)

__all__ += [
    "plot_em_data_fit",
    "plot_em_residuals",
    "plot_em_fit_and_residuals",
    "plot_time_lapse_panel",
    "plot_petrophysical_scatter",
    "plot_layered_profiles",
]
