"""
Forward modeling utilities for Frequency-Domain Electromagnetic (FDEM) data.

Uses SimPEG's frequency-domain module for 1D layered-earth simulations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from simpeg import maps
import simpeg.electromagnetics.frequency_domain as fdem


@dataclass
class FDEMSurveyConfig:
    """Configuration for FDEM survey geometry."""

    source_location: np.ndarray = None
    source_radius: float = 10.0
    receiver_location: np.ndarray = None
    receiver_orientation: str = "z"
    receiver_component: str = "secondary"
    frequencies: np.ndarray = None
    waveform_type: str = "dipole"

    def __post_init__(self) -> None:
        if self.source_location is None:
            self.source_location = np.array([0.0, 0.0, 0.0], dtype=float)
        if self.receiver_location is None:
            self.receiver_location = np.array([0.0, 0.0, 0.0], dtype=float)
        if self.frequencies is None:
            self.frequencies = np.logspace(1, 4, 16)


class FDEMForwardModeling:
    """
    Forward modeling of Frequency-Domain EM data using SimPEG.

    Supports 1D layered-earth conductivity models.
    """

    def __init__(
        self,
        thicknesses: np.ndarray,
        survey_config: Optional[FDEMSurveyConfig] = None,
        survey: Optional[fdem.Survey] = None,
    ):
        self.thicknesses = np.asarray(thicknesses, dtype=float).ravel()
        self.n_layers = self.thicknesses.size + 1

        if survey is not None:
            self.survey = survey
            self.survey_config = survey_config
        else:
            self.survey_config = survey_config or FDEMSurveyConfig()
            self.survey = self._create_survey(self.survey_config)

        self.model_mapping = maps.IdentityMap(nP=self.n_layers)
        self.simulation = fdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=self.model_mapping,
        )

    @staticmethod
    def _receiver(real_or_imag: str, config: FDEMSurveyConfig, secondary: bool = True):
        location = np.atleast_2d(np.asarray(config.receiver_location, dtype=float))
        orientation = str(config.receiver_orientation).lower()

        if secondary:
            receiver_cls = fdem.receivers.PointMagneticFieldSecondary
        else:
            receiver_cls = fdem.receivers.PointMagneticField

        try:
            return receiver_cls(
                locations=location,
                orientation=orientation,
                component=real_or_imag,
            )
        except TypeError:
            return receiver_cls(
                location,
                orientation=orientation,
                component=real_or_imag,
            )

    def _create_receiver_list(self, config: FDEMSurveyConfig):
        mode = str(config.receiver_component).lower()
        receivers = []

        if mode in {"secondary", "both"}:
            receivers.extend([
                self._receiver("real", config, secondary=True),
                self._receiver("imag", config, secondary=True),
            ])

        if mode in {"total", "both"}:
            receivers.extend([
                self._receiver("real", config, secondary=False),
                self._receiver("imag", config, secondary=False),
            ])

        if not receivers:
            raise ValueError(
                "receiver_component must be 'secondary', 'total', or 'both'."
            )

        return receivers

    def _create_source(self, receiver_list, frequency: float, config: FDEMSurveyConfig):
        waveform = str(config.waveform_type).lower()
        location = np.asarray(config.source_location, dtype=float)

        if waveform == "loop":
            try:
                return fdem.sources.CircularLoop(
                    receiver_list=receiver_list,
                    frequency=float(frequency),
                    location=location,
                    radius=float(config.source_radius),
                    current=1.0,
                )
            except TypeError:
                return fdem.sources.CircularLoop(
                    receiver_list=receiver_list,
                    frequency=float(frequency),
                    location=location,
                    radius=float(config.source_radius),
                )

        try:
            return fdem.sources.MagDipole(
                receiver_list=receiver_list,
                frequency=float(frequency),
                location=location,
                orientation="z",
                moment=1.0,
            )
        except TypeError:
            return fdem.sources.MagDipole(
                receiver_list=receiver_list,
                frequency=float(frequency),
                location=location,
            )

    def _create_survey(self, config: FDEMSurveyConfig) -> fdem.Survey:
        frequencies = np.asarray(config.frequencies, dtype=float).ravel()
        receiver_list = self._create_receiver_list(config)
        sources = [
            self._create_source(receiver_list, freq, config)
            for freq in frequencies
        ]
        return fdem.Survey(sources)

    @staticmethod
    def _pack_complex_response(response: np.ndarray) -> np.ndarray:
        response = np.asarray(response)
        if np.iscomplexobj(response):
            return response

        flat = response.ravel()
        if flat.size % 2 != 0:
            return flat.astype(np.complex128)

        return flat[0::2].astype(np.complex128) + 1j * flat[1::2].astype(np.complex128)

    def forward(self, conductivity: np.ndarray) -> np.ndarray:
        """Compute FDEM response for a given conductivity model."""
        sigma = np.asarray(conductivity, dtype=float).ravel()
        if sigma.size != self.n_layers:
            raise ValueError(
                f"conductivity must have {self.n_layers} entries, got {sigma.size}."
            )

        dpred = self.simulation.dpred(sigma)
        return self._pack_complex_response(np.asarray(dpred))

    def forward_with_noise(
        self,
        conductivity: np.ndarray,
        noise_level: float = 0.05,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute noisy and clean FDEM responses with data uncertainties."""
        clean = self.forward(conductivity)

        rng = np.random.default_rng(seed)
        uncertainty = noise_level * np.maximum(np.abs(clean), 1e-12)
        noise = rng.normal(scale=uncertainty) + 1j * rng.normal(scale=uncertainty)

        noisy = clean + noise
        return noisy, clean, uncertainty

    @staticmethod
    def hydro_to_fdem(
        water_content: np.ndarray,
        porosity: np.ndarray,
        layer_thicknesses: np.ndarray,
        **petro_params,
    ):
        """Convert hydrological properties to FDEM response via petrophysics."""
        from PyHydroGeophysX.petrophysics.resistivity_models import WS_Model

        water_content = np.asarray(water_content, dtype=float).ravel()
        porosity = np.asarray(porosity, dtype=float).ravel()
        if water_content.size != porosity.size:
            raise ValueError("water_content and porosity must have the same length.")

        n_layers = water_content.size

        sigma_w = petro_params.get("sigma_w", 0.05)
        m = petro_params.get("m", 1.5)
        n = petro_params.get("n", 2.0)
        sigma_s = petro_params.get("sigma_s", 0.0)

        if np.isscalar(sigma_w):
            sigma_w = np.full(n_layers, sigma_w, dtype=float)
        if np.isscalar(m):
            m = np.full(n_layers, m, dtype=float)
        if np.isscalar(n):
            n = np.full(n_layers, n, dtype=float)
        if np.isscalar(sigma_s):
            sigma_s = np.full(n_layers, sigma_s, dtype=float)

        saturation = np.clip(water_content / np.clip(porosity, 1e-6, None), 0.001, 1.0)

        resistivity = np.zeros(n_layers, dtype=float)
        for i in range(n_layers):
            resistivity[i] = WS_Model(
                saturation[i],
                porosity[i],
                sigma_w[i],
                m[i],
                n[i],
                sigma_s[i],
            )

        conductivity = 1.0 / np.clip(resistivity, 1e-12, None)

        config = FDEMSurveyConfig(
            source_location=np.asarray(
                petro_params.get("source_location", np.array([0.0, 0.0, 0.0])),
                dtype=float,
            ),
            source_radius=float(petro_params.get("source_radius", 10.0)),
            receiver_location=np.asarray(
                petro_params.get("receiver_location", np.array([0.0, 0.0, 0.0])),
                dtype=float,
            ),
            receiver_orientation=str(petro_params.get("receiver_orientation", "z")),
            receiver_component=str(petro_params.get("receiver_component", "secondary")),
            frequencies=np.asarray(
                petro_params.get("frequencies", np.logspace(1, 4, 16)),
                dtype=float,
            ),
            waveform_type=str(petro_params.get("waveform_type", "dipole")),
        )

        modeler = FDEMForwardModeling(
            thicknesses=np.asarray(layer_thicknesses, dtype=float).ravel(),
            survey_config=config,
        )

        noisy, clean, uncertainty = modeler.forward_with_noise(
            conductivity=conductivity,
            noise_level=float(petro_params.get("noise_level", 0.05)),
            seed=petro_params.get("seed", None),
        )

        return noisy, clean, uncertainty, conductivity
