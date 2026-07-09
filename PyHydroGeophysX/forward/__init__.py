"""Forward-modeling utilities with independently optional method backends.

Importing an EM submodule must not require the ERT/PyGIMLi stack.  Each public
method is therefore exported only when its own optional dependencies import
successfully; direct imports of individual submodules remain the preferred
interface for applications that need a specific backend.
"""

from __future__ import annotations

__all__ = []


try:  # ERT depends on PyGIMLi.
    from PyHydroGeophysX.forward.ert_forward import (
        ERTForwardModeling,
        ertforandjac,
        ertforandjac2,
        ertforward,
        ertforward2,
    )

    __all__ += [
        "ERTForwardModeling",
        "ertforward",
        "ertforward2",
        "ertforandjac",
        "ertforandjac2",
    ]
except Exception:  # pragma: no cover - depends on optional native backend
    pass

try:
    from PyHydroGeophysX.forward.srt_forward import SeismicForwardModeling

    __all__.append("SeismicForwardModeling")
except Exception:  # pragma: no cover - depends on optional backend
    pass

try:  # TDEM depends on SimPEG/discretize.
    from PyHydroGeophysX.forward.tdem_forward import (
        TDEMForwardModeling,
        TDEMSurveyConfig,
        create_tdem_survey,
        hydro_to_tdem,
    )

    __all__ += [
        "TDEMForwardModeling",
        "TDEMSurveyConfig",
        "create_tdem_survey",
        "hydro_to_tdem",
    ]
except Exception:  # pragma: no cover - depends on optional backend
    pass

try:  # FDEM depends on SimPEG.
    from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling, FDEMSurveyConfig

    __all__ += ["FDEMForwardModeling", "FDEMSurveyConfig"]
except Exception:  # pragma: no cover - depends on optional backend
    pass
