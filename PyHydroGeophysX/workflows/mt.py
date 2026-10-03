"""Magnetotelluric workflows: the public names of :mod:`PyHydroGeophysX.data_processing.mt`.

The registered workflows are ``mt.process`` (time series to transfer
functions), ``mt.invert_1d`` (Occam 1D, with static shift and an optional TEM
sounding) and ``mt.invert_profile`` (2D on SimPEG); this module gathers what
their recipes and walkthroughs call.
"""

from __future__ import annotations

from PyHydroGeophysX.data_processing.mt import (
    ProcessingConfig,
    TimeSeriesRun,
    TransferFunction,
    apply_static_shift,
    estimate_static_shift,
    load_runs,
    niblett_bostick,
    occam1d,
    phase_tensor,
    process_mt,
    read_band_setup,
    read_timeseries,
    read_transfer_function,
    save_runs,
    water_content_profile,
    write_edi,
    write_emtf_xml,
)

# The 2D names import SimPEG, so they load on first use.
_PROFILE_NAMES = ("build_profile_mesh", "forward_profile", "invert_profile", "station_distances")


def __getattr__(name: str):
    if name in _PROFILE_NAMES:
        from PyHydroGeophysX.data_processing.mt import inversion2d

        return getattr(inversion2d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProcessingConfig",
    "TimeSeriesRun",
    "TransferFunction",
    "apply_static_shift",
    "build_profile_mesh",
    "estimate_static_shift",
    "forward_profile",
    "invert_profile",
    "load_runs",
    "niblett_bostick",
    "occam1d",
    "phase_tensor",
    "process_mt",
    "read_band_setup",
    "read_timeseries",
    "read_transfer_function",
    "save_runs",
    "station_distances",
    "water_content_profile",
    "write_edi",
    "write_emtf_xml",
]
