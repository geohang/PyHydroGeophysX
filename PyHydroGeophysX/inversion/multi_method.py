"""Unified multi-method geophysical inversion interface."""


class GeophysicalInversion:
    """
    Unified factory for multi-method geophysical inversion.

    Dispatches to the correct inversion engine.
    """

    SUPPORTED = {"ert", "srt", "tdem", "fdem", "joint_ert_srt", "joint"}

    def __init__(self, method: str, **kwargs):
        method = method.lower().strip()

        if method == "ert":
            from .ert_inversion import ERTInversion

            self._engine = ERTInversion(**kwargs)
        elif method == "srt":
            from .srt_inversion import SRTInversion

            self._engine = SRTInversion(**kwargs)
        elif method == "tdem":
            from .tdem_inversion import TDEMInversion

            self._engine = TDEMInversion(**kwargs)
        elif method == "fdem":
            from .fdem_inversion import FDEMInversion

            self._engine = FDEMInversion(**kwargs)
        elif method in {"joint_ert_srt", "joint"}:
            from .joint_ert_srt import JointERTSRTInversion

            self._engine = JointERTSRTInversion(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Supported: {self.SUPPORTED}")

        self.method = method

    def run(self, **kwargs):
        return self._engine.run(**kwargs)

    def setup(self, **kwargs):
        if hasattr(self._engine, "setup"):
            return self._engine.setup(**kwargs)
        return None

    @property
    def engine(self):
        return self._engine
