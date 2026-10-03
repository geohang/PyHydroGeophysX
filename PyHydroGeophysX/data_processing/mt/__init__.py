"""Magnetotelluric (MT) data: transfer functions, instrument time series and processing.

Everything here is written for PyHydroGeophysX on NumPy and SciPy alone.

- :mod:`.transfer_function` - :class:`TransferFunction`, one site's impedance
  and tipper with their errors, in SI units and e^{+i omega t};
- :mod:`.edi`, :mod:`.emtf_xml`, :mod:`.zfile`, :mod:`.jfile` - the exchange
  formats; :func:`read_transfer_function` picks the reader;
- :mod:`.timeseries` - :class:`TimeSeriesRun` and :class:`Channel`, recorded
  samples with their instrument responses;
- :mod:`.phoenix`, :mod:`.phoenix_legacy`, :mod:`.metronix`, :mod:`.zonge`,
  :mod:`.lemi` - the instruments' own files (Phoenix MTU-5C/5P/8A and
  MTU-5A/V8, Metronix ADU, Zonge ZEN, LEMI-424); :func:`read_timeseries`
  picks the reader;
- :mod:`.processing` (with :mod:`.spectra` and :mod:`.robust`) -
  :func:`process_mt`, robust single-site and remote-reference estimation of
  the impedance and tipper from time series;
- :mod:`.analysis` - phase tensor, skews, strike, induction arrows,
  Niblett-Bostick depths and static shift;
- :mod:`.forward1d` - the layered-earth impedance and its derivatives;
- :mod:`.inversion1d` - Occam's 1D inversion, with static shift and jointly
  with a TEM sounding, and the water content of the result;
- :mod:`.inversion2d` - 2D forward modelling and inversion of a profile on
  SimPEG (an optional dependency, imported when used).
"""

from __future__ import annotations

from .analysis import (
    apply_static_shift,
    bahr_skew,
    estimate_static_shift,
    induction_arrows,
    niblett_bostick,
    phase_tensor,
    static_shift_from_layers,
    swift_skew,
    swift_strike,
)
from .edi import is_edi_file, read_edi, write_edi
from .emtf_xml import is_emtf_xml_file, read_emtf_xml, write_emtf_xml
from .forward1d import apparent_resistivity_1d, impedance_1d, sensitivity_1d
from .inversion1d import Occam1DResult, Sounding1D, occam1d, sounding_from_tf, water_content_profile
from .io import (
    TIMESERIES_FORMATS,
    TRANSFER_FUNCTION_PATTERNS,
    is_transfer_function_file,
    read_timeseries,
    read_transfer_function,
    read_transfer_functions,
    timeseries_format,
)
from .jfile import is_jfile, read_jfile
from .lemi import read_lemi424
from .metronix import read_ats_header, read_metronix, read_metronix_calibration
from .phoenix import read_phoenix, read_phoenix_calibration, read_phoenix_header
from .phoenix_legacy import read_phoenix_legacy, read_phoenix_table
from .processing import ProcessingConfig, process_mt
from .robust import RegressionConfig, robust_regression
from .spectra import Band, default_bands, read_band_setup
from .timeseries import (
    Channel,
    ResponseStage,
    TimeSeriesRun,
    gps_leap_seconds,
    gps_seconds_to_utc,
    load_runs,
    read_response_table,
    runs_from_payload,
    runs_to_payload,
    save_runs,
)
from .transfer_function import (
    FIELD_TO_OHM,
    MU0,
    TransferFunction,
    from_apparent_resistivity,
    rotation_matrix,
)
from .zfile import is_zfile, read_zfile
from .zonge import read_z3d_header, read_zonge

__all__ = [
    "Band",
    "Channel",
    "FIELD_TO_OHM",
    "MU0",
    "Occam1DResult",
    "ProcessingConfig",
    "RegressionConfig",
    "ResponseStage",
    "Sounding1D",
    "TIMESERIES_FORMATS",
    "TRANSFER_FUNCTION_PATTERNS",
    "TimeSeriesRun",
    "TransferFunction",
    "apparent_resistivity_1d",
    "apply_static_shift",
    "bahr_skew",
    "build_profile_mesh",
    "default_bands",
    "estimate_static_shift",
    "forward_profile",
    "from_apparent_resistivity",
    "gps_leap_seconds",
    "gps_seconds_to_utc",
    "impedance_1d",
    "induction_arrows",
    "invert_profile",
    "is_edi_file",
    "is_emtf_xml_file",
    "is_jfile",
    "is_transfer_function_file",
    "is_zfile",
    "load_runs",
    "niblett_bostick",
    "occam1d",
    "phase_tensor",
    "process_mt",
    "read_ats_header",
    "read_band_setup",
    "read_edi",
    "read_emtf_xml",
    "read_jfile",
    "read_lemi424",
    "read_metronix",
    "read_metronix_calibration",
    "read_phoenix",
    "read_phoenix_calibration",
    "read_phoenix_header",
    "read_phoenix_legacy",
    "read_phoenix_table",
    "read_response_table",
    "read_timeseries",
    "read_transfer_function",
    "read_transfer_functions",
    "read_z3d_header",
    "read_zfile",
    "read_zonge",
    "robust_regression",
    "rotation_matrix",
    "runs_from_payload",
    "runs_to_payload",
    "save_runs",
    "sensitivity_1d",
    "sounding_from_tf",
    "static_shift_from_layers",
    "station_distances",
    "swift_skew",
    "swift_strike",
    "timeseries_format",
    "water_content_profile",
    "write_edi",
    "write_emtf_xml",
]


_LAZY = {"build_profile_mesh", "forward_profile", "invert_profile", "station_distances", "ProfileMesh",
         "ProfileInversionResult"}


def __getattr__(name):
    # The 2D tools import SimPEG only when they run; the module itself is light.
    if name in _LAZY:
        from . import inversion2d

        return getattr(inversion2d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
