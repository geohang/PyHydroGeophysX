"""The 1D electromagnetic API in one namespace.

Nothing is implemented here. The readers, the example catalogue and the result
writers are in :mod:`PyHydroGeophysX.data_processing.em1d`, the forward models in
:mod:`PyHydroGeophysX.forward.em1d`, single-sounding inversion in
:mod:`PyHydroGeophysX.inversion.em1d`, and line inversion with its amplitude
calibration in :mod:`PyHydroGeophysX.inversion.em1d_line`. The examples, the
generated walkthroughs and older scripts import them from here.
"""

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX.data_processing.em1d import (
    TEMCOMPANY_MOMENTS,
    build_em_config,
    example_catalog,
    gate_report,
    is_temcompany_source,
    is_ttem_source,
    load_line_geometry,
    load_sounding,
    load_sounding_container,
    load_temcompany_sounding,
    load_ttem_sounding,
    save_inversion,
    save_line_csv,
    save_sounding_container,
    survey_summary,
)
from PyHydroGeophysX.forward.em1d import (
    DEFAULT_FDEM,
    DEFAULT_MODEL,
    DEFAULT_TDEM,
    fdem_forward,
    model_arrays,
    model_depth_profile,
    tdem_forward,
)
from PyHydroGeophysX.inversion.em1d import (
    DEFAULT_INVERSION,
    INVERSION_PRESETS,
    fdem_invert,
    preset_inversion,
    tdem_invert,
    tdem_joint_invert,
)
from PyHydroGeophysX.inversion.em1d_line import (
    FDEM_BOTH_REFUSAL,
    METHODS,
    STATION_DISTANCE_BIN_M,
    backend_status,
    calibrate_to_reference,
    estimate_data_scale,
    invert_line,
)

__all__ = [
    "BackendUnavailable",
    "METHODS",
    "TEMCOMPANY_MOMENTS",
    "DEFAULT_MODEL",
    "DEFAULT_FDEM",
    "DEFAULT_TDEM",
    "DEFAULT_INVERSION",
    "INVERSION_PRESETS",
    "FDEM_BOTH_REFUSAL",
    "STATION_DISTANCE_BIN_M",
    "backend_status",
    "example_catalog",
    "model_arrays",
    "model_depth_profile",
    "is_temcompany_source",
    "is_ttem_source",
    "load_temcompany_sounding",
    "load_ttem_sounding",
    "load_sounding",
    "load_sounding_container",
    "gate_report",
    "survey_summary",
    "save_sounding_container",
    "load_line_geometry",
    "fdem_forward",
    "tdem_forward",
    "fdem_invert",
    "tdem_invert",
    "tdem_joint_invert",
    "preset_inversion",
    "estimate_data_scale",
    "calibrate_to_reference",
    "invert_line",
    "build_em_config",
    "save_inversion",
    "save_line_csv",
]
