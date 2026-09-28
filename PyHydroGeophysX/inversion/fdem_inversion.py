"""
Inversion utilities for Frequency-Domain Electromagnetic (FDEM) data.

Provides 1D layered-earth FDEM inversion using SimPEG.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import simpeg.electromagnetics.frequency_domain as fdem
from discretize import TensorMesh
from simpeg import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
)

from PyHydroGeophysX.forward.fdem_forward import FDEMSurveyConfig


# ---------------------------------------------------------------------------
# FDEMInversion Result
# ---------------------------------------------------------------------------
@dataclass
class FDEMInversionResult:
    """Container for FDEM inversion results."""

    recovered_model: np.ndarray = None
    recovered_conductivity: np.ndarray = None
    l2_model: np.ndarray = None
    l2_conductivity: np.ndarray = None
    predicted_data: np.ndarray = None
    mesh: Any = None
    thicknesses: np.ndarray = None
    chi2: float = None
    frequencies: np.ndarray = None


# ---------------------------------------------------------------------------
# FDEMInversion
# ---------------------------------------------------------------------------
class FDEMInversion:
    """
    1D FDEM inversion using SimPEG.

    Follows the same pattern used in TDEMInversion.
    Source, receiver and ``waveform_type`` are checked, and an omitted receiver
    placed, as in FDEMSurveyConfig: the origin, unless that is directly below
    or above a dipole source (then 10 m along x from it) or away from a loop's
    centre (then the centre on z = 0). Explicit zero horizontal dipole offsets
    are rejected rather than producing a non-finite prediction. Receivers at
    (n, 3) locations take complex ``dobs`` ordered as FDEMForwardModeling
    returns them: the n receivers of each frequency together.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        dobs: np.ndarray,
        uncertainties: np.ndarray,
        thicknesses: Optional[np.ndarray] = None,
        source_location: Optional[np.ndarray] = None,
        source_radius: float = 10.0,
        receiver_location: Optional[np.ndarray] = None,
        receiver_orientation: str = "z",
        receiver_component: str = "secondary",
        waveform_type: str = "dipole",
        n_layers: int = 25,
        min_thickness: float = 1.0,
        max_thickness: float = 30.0,
        **kwargs,
    ):
        self.frequencies = np.asarray(frequencies, dtype=float).ravel()
        self.dobs = np.asarray(dobs)
        self.uncertainties = np.asarray(uncertainties)

        geometry = FDEMSurveyConfig(source_location=source_location,
                                    receiver_location=receiver_location,
                                    waveform_type=waveform_type)
        self.source_location = geometry.source_location
        self.receiver_location = geometry.receiver_location
        self.source_radius = float(source_radius)
        self.receiver_orientation = receiver_orientation
        self.receiver_component = receiver_component
        self.waveform_type = waveform_type

        self.n_layers = int(n_layers)
        self.min_thickness = float(min_thickness)
        self.max_thickness = float(max_thickness)
        self.thicknesses = None if thicknesses is None else np.asarray(thicknesses, dtype=float).ravel()

        self.parameters = {
            "starting_conductivity": 0.01,
            "alpha_s": 0.01,
            "alpha_x": 1.0,
            "max_iterations": 100,
            "use_irls": True,
            "irls_norms": [1, 0],
            "beta0_ratio": 1e2,
            "max_irls_iterations": 30,
            "cg_maxiter": 30,
            "lower_bound": 1e-6,
            "upper_bound": 10.0,
            "verbose": True,
        }
        self.parameters.update(kwargs)

        self.survey = None
        self.mesh = None
        self.model_mapping = None
        self.simulation = None
        self._setup_complete = False

    @staticmethod
    def _receiver_groups(values: np.ndarray, n_receivers: int) -> np.ndarray:
        """One row per (frequency, field), one column per receiver location."""
        values = np.asarray(values).ravel()
        n = max(int(n_receivers), 1)
        return values.reshape(-1, n if values.size % n == 0 else 1)

    @staticmethod
    def _to_simpeg_vector(arr: np.ndarray, n_receivers: int = 1) -> np.ndarray:
        arr = np.asarray(arr)
        if np.iscomplexobj(arr):
            # SimPEG wants the real parts of every receiver location, then the
            # imaginary parts, per (frequency, field) block.
            groups = FDEMInversion._receiver_groups(arr, n_receivers)
            return np.stack((groups.real, groups.imag), axis=1).ravel()
        return arr.ravel().astype(float)

    @staticmethod
    def _pack_complex(response: np.ndarray, n_receivers: int = 1) -> np.ndarray:
        response = np.asarray(response, dtype=float).ravel()
        block = 2 * max(int(n_receivers), 1)
        if response.size % block != 0:
            return response.astype(np.complex128)
        pairs = response.astype(np.complex128).reshape(-1, 2, block // 2)
        return (pairs[:, 0, :] + 1j * pairs[:, 1, :]).ravel()

    @staticmethod
    def _expand_uncertainties(uncertainties: np.ndarray, target_size: int,
                              n_receivers: int = 1) -> np.ndarray:
        unc = np.asarray(uncertainties)
        if np.iscomplexobj(unc):
            unc = np.abs(unc)
        unc = unc.ravel().astype(float)

        if unc.size == target_size:
            return np.clip(unc, 1e-12, None)

        if unc.size * 2 == target_size:
            groups = FDEMInversion._receiver_groups(unc, n_receivers)
            expanded = np.stack((groups, groups), axis=1).ravel()
            return np.clip(expanded, 1e-12, None)

        if unc.size == 1:
            return np.full(target_size, np.clip(float(unc[0]), 1e-12, None), dtype=float)

        raise ValueError(
            f"Uncertainty vector length {unc.size} cannot be mapped to target size {target_size}."
        )

    @staticmethod
    def _receiver_factory(receiver_cls, location, orientation, component):
        try:
            return receiver_cls(
                locations=location,
                orientation=orientation,
                component=component,
            )
        except TypeError:
            return receiver_cls(
                location,
                orientation=orientation,
                component=component,
            )

    @staticmethod
    def _loop_source_factory(receiver_list, frequency, location, radius):
        try:
            return fdem.sources.CircularLoop(
                receiver_list=receiver_list,
                frequency=frequency,
                location=location,
                radius=radius,
                current=1.0,
            )
        except TypeError:
            return fdem.sources.CircularLoop(
                receiver_list=receiver_list,
                frequency=frequency,
                location=location,
                radius=radius,
            )

    @staticmethod
    def _dipole_source_factory(receiver_list, frequency, location):
        try:
            return fdem.sources.MagDipole(
                receiver_list=receiver_list,
                frequency=frequency,
                location=location,
                orientation="z",
                moment=1.0,
            )
        except TypeError:
            return fdem.sources.MagDipole(
                receiver_list=receiver_list,
                frequency=frequency,
                location=location,
            )

    def setup(self) -> None:
        if self.thicknesses is None:
            self.thicknesses = np.logspace(
                np.log10(self.min_thickness),
                np.log10(self.max_thickness),
                max(self.n_layers - 1, 1),
            )

        if self.thicknesses.ndim != 1:
            raise ValueError("thicknesses must be one-dimensional.")

        self.n_layers = self.thicknesses.size + 1

        config = FDEMSurveyConfig(
            source_location=self.source_location,
            source_radius=self.source_radius,
            receiver_location=self.receiver_location,
            receiver_orientation=self.receiver_orientation,
            receiver_component=self.receiver_component,
            frequencies=self.frequencies,
            waveform_type=self.waveform_type,
        )

        receiver_list = []
        mode = str(config.receiver_component).lower()
        location = np.atleast_2d(config.receiver_location)

        if mode in {"secondary", "both"}:
            receiver_list.extend(
                [
                    self._receiver_factory(
                        fdem.receivers.PointMagneticFieldSecondary,
                        location,
                        config.receiver_orientation,
                        "real",
                    ),
                    self._receiver_factory(
                        fdem.receivers.PointMagneticFieldSecondary,
                        location,
                        config.receiver_orientation,
                        "imag",
                    ),
                ]
            )
        if mode in {"total", "both"}:
            receiver_list.extend(
                [
                    self._receiver_factory(
                        fdem.receivers.PointMagneticField,
                        location,
                        config.receiver_orientation,
                        "real",
                    ),
                    self._receiver_factory(
                        fdem.receivers.PointMagneticField,
                        location,
                        config.receiver_orientation,
                        "imag",
                    ),
                ]
            )

        if not receiver_list:
            raise ValueError("receiver_component must be 'secondary', 'total', or 'both'.")

        source_list = []
        for freq in self.frequencies:
            if str(config.waveform_type).lower() == "loop":
                src = self._loop_source_factory(
                    receiver_list=receiver_list,
                    frequency=float(freq),
                    location=config.source_location,
                    radius=config.source_radius,
                )
            else:
                src = self._dipole_source_factory(
                    receiver_list=receiver_list,
                    frequency=float(freq),
                    location=config.source_location,
                )
            source_list.append(src)

        self.survey = fdem.Survey(source_list)

        self.mesh = TensorMesh([np.r_[self.thicknesses, self.thicknesses[-1]]], "0")
        self.model_mapping = maps.ExpMap()

        self.simulation = fdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=self.model_mapping,
        )
        # The default Pardiso solver can terminate the Windows process with a
        # delay-load exception when MKL/Pardiso is unavailable.  Use the
        # portable SciPy-backed pymatsolver implementation explicitly.
        import pymatsolver

        self.simulation.solver = pymatsolver.Solver
        self.simulation.solver_opts = {}

        self._setup_complete = True

        if self.parameters["verbose"]:
            print("FDEM inversion setup complete")
            print(f"  - {self.n_layers} layers")
            print(f"  - frequency range: {self.frequencies.min():.2f} to {self.frequencies.max():.2f} Hz")

    def run(self, starting_model: Optional[np.ndarray] = None) -> FDEMInversionResult:
        if not self._setup_complete:
            self.setup()

        verbose = bool(self.parameters["verbose"])

        if starting_model is None:
            sigma0 = float(self.parameters["starting_conductivity"])
            starting_model = np.log(np.clip(sigma0, 1e-12, None)) * np.ones(self.mesh.nC)
        else:
            starting_model = np.asarray(starting_model, dtype=float).ravel()
            if starting_model.size != self.mesh.nC:
                raise ValueError(
                    f"starting_model must have {self.mesh.nC} entries, got {starting_model.size}."
                )

        n_receivers = np.atleast_2d(self.receiver_location).shape[0]
        dobs_vec = self._to_simpeg_vector(self.dobs, n_receivers)
        if dobs_vec.size != self.survey.nD:
            raise ValueError(
                f"Observed data size ({dobs_vec.size}) does not match survey data count ({self.survey.nD})."
            )

        uncertainty_vec = self._expand_uncertainties(self.uncertainties, dobs_vec.size, n_receivers)

        data_object = data.Data(
            self.survey,
            dobs=dobs_vec,
            standard_deviation=uncertainty_vec,
        )

        dmis = data_misfit.L2DataMisfit(simulation=self.simulation, data=data_object)
        dmis.W = 1.0 / np.clip(uncertainty_vec, 1e-12, None)

        reg_map = maps.IdentityMap(nP=self.mesh.nC)
        reg = regularization.Sparse(
            self.mesh,
            mapping=reg_map,
            alpha_s=float(self.parameters["alpha_s"]),
            alpha_x=float(self.parameters["alpha_x"]),
        )
        reg.reference_model = starting_model

        if self.parameters["use_irls"]:
            reg.norms = self.parameters["irls_norms"]

        opt = optimization.ProjectedGNCG(
            maxIter=int(self.parameters["max_iterations"]),
            maxIterLS=20,
            maxIterCG=int(self.parameters["cg_maxiter"]),
            tolCG=1e-3,
        )

        lower_bound = max(float(self.parameters["lower_bound"]), 1e-12)
        upper_bound = max(float(self.parameters["upper_bound"]), lower_bound * 10)
        opt.lower = np.log(lower_bound) * np.ones(self.mesh.nC)
        opt.upper = np.log(upper_bound) * np.ones(self.mesh.nC)

        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        directives_list = [
            directives.UpdateSensitivityWeights(),
            directives.BetaEstimate_ByEig(beta0_ratio=float(self.parameters["beta0_ratio"])),
            directives.SaveOutputEveryIteration(save_txt=False),
        ]

        if self.parameters["use_irls"]:
            directives_list.append(
                directives.UpdateIRLS(
                    max_irls_iterations=int(self.parameters["max_irls_iterations"]),
                    irls_cooling_factor=1.5,
                )
            )

        directives_list.append(directives.UpdatePreconditioner())

        inv = inversion.BaseInversion(inv_prob, directives_list)

        if verbose:
            print("Running FDEM inversion...")

        recovered_model = inv.run(starting_model)

        result = FDEMInversionResult()
        result.recovered_model = recovered_model
        result.recovered_conductivity = self.model_mapping * recovered_model

        if hasattr(inv_prob, "l2model") and inv_prob.l2model is not None:
            result.l2_model = inv_prob.l2model
            result.l2_conductivity = self.model_mapping * inv_prob.l2model

        pred_vec = np.asarray(self.simulation.dpred(recovered_model), dtype=float).ravel()
        result.predicted_data = self._pack_complex(pred_vec, n_receivers)
        result.mesh = self.mesh
        result.thicknesses = self.thicknesses
        result.frequencies = self.frequencies

        residual = (dobs_vec - pred_vec) / np.clip(uncertainty_vec, 1e-12, None)
        result.chi2 = float(np.sum(residual**2) / residual.size)

        if verbose:
            print(f"FDEM inversion complete. Final chi2: {result.chi2:.3f}")

        return result
