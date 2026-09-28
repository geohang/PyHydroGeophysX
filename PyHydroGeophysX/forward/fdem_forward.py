"""
Forward modeling utilities for Frequency-Domain Electromagnetic (FDEM) data.

Uses SimPEG's frequency-domain module for 1D layered-earth simulations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg import maps


# ---------------------------------------------------------------------------
# FDEMSurvey Config
# ---------------------------------------------------------------------------
#: Accepted ``waveform_type`` spellings, compared ignoring case, spaces, "_" and
#: "-". Releases up to 0.4 modelled every name except "loop" as a dipole, so
#: the dipole names that were in use keep working.
_WAVEFORM_ALIASES = {
    "dipole": "dipole", "magdipole": "dipole", "magneticdipole": "dipole",
    "vmd": "dipole", "verticalmagneticdipole": "dipole",
    "loop": "loop", "circularloop": "loop",
}


def _waveform_kind(name) -> str:
    """The source type, ``"dipole"`` or ``"loop"``, that ``name`` spells."""
    kind = _WAVEFORM_ALIASES.get("".join(ch for ch in str(name).lower() if ch.isalnum()))
    if kind is None:
        raise ValueError(f"waveform_type {name!r} is not recognised; use 'dipole' "
                         "(also 'magdipole' or 'vmd') or 'loop'.")
    return kind


@dataclass
class FDEMSurveyConfig:
    """Configuration for FDEM survey geometry.

    ``source_location`` is one XYZ point, shape (3,) or (1, 3), and is stored
    as (3,). ``receiver_location`` is one point, (3,), or several, (n, 3); it
    keeps the shape it was given. Several receivers share every frequency.

    An omitted receiver sits at the origin, as in earlier releases, except
    where the origin cannot be modelled: below or above a dipole source (zero
    horizontal offset, a non-finite response) it goes 10 m along x from the
    source, and for a loop away from the origin it goes to the loop centre on
    z = 0, the only receiver position the layered-earth loop kernel supports.
    An explicit zero horizontal dipole offset is rejected before simulation.
    """

    source_location: np.ndarray = None
    source_radius: float = 10.0
    receiver_location: np.ndarray = None
    receiver_orientation: str = "z"
    receiver_component: str = "secondary"
    frequencies: np.ndarray = None
    waveform_type: str = "dipole"

    def __post_init__(self) -> None:
        self.waveform_type = _waveform_kind(self.waveform_type)
        if self.source_location is None:
            self.source_location = np.array([0.0, 0.0, 0.0], dtype=float)
        source = np.asarray(self.source_location, dtype=float)
        if source.shape not in {(3,), (1, 3)} or not np.isfinite(source).all():
            raise ValueError("source_location must be one finite XYZ point, shape (3,) or (1, 3).")
        self.source_location = source.reshape(3)
        if self.receiver_location is None:
            self.receiver_location = self._default_receiver()
        self.receiver_location = np.asarray(self.receiver_location, dtype=float)
        receivers = np.atleast_2d(self.receiver_location)
        if (receivers.ndim != 2 or receivers.shape[1] != 3 or receivers.shape[0] == 0
                or not np.isfinite(receivers).all()):
            raise ValueError("receiver_location must be finite XYZ points, shape (3,) or (n, 3).")
        if self.waveform_type == "dipole" and np.any(
                np.all(receivers[:, :2] == self.source_location[:2], axis=1)):
            raise ValueError("A dipole survey requires nonzero horizontal source-receiver offset.")
        if self.frequencies is None:
            self.frequencies = np.logspace(1, 4, 16)

    def _default_receiver(self) -> np.ndarray:
        """Receiver used when none is given; see the class docstring."""
        source = self.source_location
        at_origin = not np.any(source[:2])
        if self.waveform_type == "dipole":
            return source + [10.0, 0.0, 0.0] if at_origin else np.zeros(3)
        return np.array([source[0], source[1], 0.0])


# ---------------------------------------------------------------------------
# FDEMForward Modeling
# ---------------------------------------------------------------------------
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
        waveform = _waveform_kind(config.waveform_type)
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
    def _pack_complex_response(response: np.ndarray, n_receivers: int = 1) -> np.ndarray:
        """Pair SimPEG's real and imaginary receivers into complex values.

        SimPEG orders data by source, receiver object, then location, so each
        (frequency, field) block holds the real parts of all ``n_receivers``
        locations followed by their imaginary parts. Pairing neighbouring
        values, as this did before, is right only for a single receiver.
        """
        response = np.asarray(response)
        if np.iscomplexobj(response):
            return response

        flat = response.ravel()
        block = 2 * max(int(n_receivers), 1)
        if flat.size % block != 0:
            return flat.astype(np.complex128)

        pairs = flat.astype(np.complex128).reshape(-1, 2, block // 2)
        return (pairs[:, 0, :] + 1j * pairs[:, 1, :]).ravel()

    def _locations_per_receiver(self) -> int:
        """Locations in each SimPEG receiver when all share one count, else 1."""
        counts = {np.atleast_2d(rx.locations).shape[0]
                  for src in self.survey.source_list for rx in src.receiver_list}
        return counts.pop() if len(counts) == 1 else 1

    def forward(self, conductivity: np.ndarray) -> np.ndarray:
        """Compute FDEM response for a given conductivity model.

        One complex value per frequency, field and receiver, in that order:
        with receivers at (n, 3) locations the n values of a frequency (and
        field, for ``receiver_component="both"``) are consecutive.
        """
        sigma = np.asarray(conductivity, dtype=float).ravel()
        if sigma.size != self.n_layers:
            raise ValueError(
                f"conductivity must have {self.n_layers} entries, got {sigma.size}."
            )

        dpred = self.simulation.dpred(sigma)
        return self._pack_complex_response(np.asarray(dpred), self._locations_per_receiver())

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
            source_location=petro_params.get("source_location"),
            source_radius=float(petro_params.get("source_radius", 10.0)),
            receiver_location=petro_params.get("receiver_location"),
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
