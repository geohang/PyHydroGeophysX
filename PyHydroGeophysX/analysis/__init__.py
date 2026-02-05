"""Sensitivity and resolution analysis utilities."""

from .sensitivity import (
    compute_cumulative_sensitivity,
    compute_depth_of_investigation,
    compute_resolution_matrix,
    plot_sensitivity_map,
)

__all__ = [
    "compute_resolution_matrix",
    "compute_depth_of_investigation",
    "compute_cumulative_sensitivity",
    "plot_sensitivity_map",
]
