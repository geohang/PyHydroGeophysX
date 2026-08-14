"""
Forward modeling utilities for Time-Domain Electromagnetic (TDEM) simulations.

This module provides classes for 1D TDEM forward modeling using SimPEG,
including support for layered Earth models derived from hydrological data.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import simpeg.electromagnetics.time_domain as tdem

# SimPEG imports
from discretize import TensorMesh
from simpeg import maps
from simpeg.utils import mkvc


# ---------------------------------------------------------------------------
# TDEMSurvey Config
# ---------------------------------------------------------------------------
@dataclass
class TDEMSurveyConfig:
    """Configuration for TDEM survey geometry.
    
    Attributes:
        source_location: [x, y, z] location of source center (m)
        source_radius: Radius of circular loop source (m)
        source_current: Peak current amplitude (A)
        receiver_location: [x, y, z] location of receiver (m)
        receiver_orientation: Component to measure ('x', 'y', or 'z')
        times: Time channels for measurement (s)
        waveform_type: Type of waveform ('step_off', 'ramp_off', 'custom')
    """
    source_location: np.ndarray = None
    source_radius: float = 10.0
    source_current: float = 1.0
    source_turns: int = 1
    #: Transmitter moment in A m^2, overriding ``current * turns * area``. Set it
    #: to 1.0 for data already normalized by the transmitter moment, which is how
    #: TEMcompany instruments report dB/dt (V/A/m^4): dividing the measurement by
    #: the moment and then modelling with that moment counts it twice.
    source_moment: Optional[float] = None
    #: Turn-off waveform as (times, currents) nodes. A real ramp is not a step,
    #: and the earliest gates of a ground system sit only microseconds after it
    #: ends, which is exactly where the difference bites.
    waveform_times: Optional[np.ndarray] = None
    waveform_currents: Optional[np.ndarray] = None
    #: Gate windows, for averaging the response over each one instead of reading
    #: it at the gate centre.
    gate_open: Optional[np.ndarray] = None
    gate_close: Optional[np.ndarray] = None
    gate_samples: int = 1
    #: Analog receiver electronics parsed from a GEX file.  The supported
    #: fields are ``receiver_damping`` / ``receiver_cutoff_hz`` for the
    #: receiver-coil two-pole filter and ``tib_order`` / ``tib_cutoff_hz`` for
    #: the transmitter-interface-board low-pass filter.
    analog_lowpass: Optional[dict] = None
    analog_filter_samples: int = 48
    receiver_location: np.ndarray = None
    receiver_orientation: str = "z"
    receiver_type: str = "b"
    times: np.ndarray = None
    waveform_type: str = "step_off"

    def __post_init__(self):
        if self.source_location is None:
            self.source_location = np.array([0.0, 0.0, 0.0])
        if self.receiver_location is None:
            self.receiver_location = np.array([0.0, 0.0, 0.0])
        if self.times is None:
            self.times = np.logspace(-5, -2, 31)


def _gate_sampling(config: "TDEMSurveyConfig"):
    """Times to model, and the matrix that averages them back onto the gates.

    A gate is an integral over a window, not a reading at its centre, and the
    windows are log-spaced so the late ones are wide. Sampling each window at a
    few points and averaging is the honest version. Returns ``(times, None)``
    when no windows are given or a single sample is asked for, which is the
    plain gate-centre behaviour.
    """
    times = np.asarray(config.times, dtype=float).ravel()
    n_samples = max(1, int(config.gate_samples))
    opens = config.gate_open
    closes = config.gate_close
    if n_samples == 1 or opens is None or closes is None:
        return times, None
    opens = np.asarray(opens, dtype=float).ravel()
    closes = np.asarray(closes, dtype=float).ravel()
    if opens.size != times.size or closes.size != times.size:
        return times, None
    # Gauss-Legendre in log time: the response is close to a power law across a
    # window, so a few nodes spaced in log integrate it far better than in time.
    nodes, weights = np.polynomial.legendre.leggauss(n_samples)
    sample = np.empty(times.size * n_samples, dtype=float)
    matrix = np.zeros((times.size, times.size * n_samples), dtype=float)
    for gate, (low, high) in enumerate(zip(opens, closes)):
        low = max(float(low), 1e-12)
        high = max(float(high), low * (1.0 + 1e-9))
        mid = 0.5 * (np.log(high) + np.log(low))
        half = 0.5 * (np.log(high) - np.log(low))
        block = slice(gate * n_samples, (gate + 1) * n_samples)
        sample[block] = np.exp(mid + half * nodes)
        # d t = t d(ln t), so integrating in log time carries a factor of t.
        share = weights * sample[block]
        matrix[gate, block] = share / share.sum()
    return sample, matrix


def _valid_positive(value, default=0.0) -> float:
    """Return a finite positive float or *default*."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) and parsed > 0.0 else float(default)


def _analog_parameters(spec: Optional[dict]) -> "tuple[float, float, int, float]":
    """Canonical GEX analog-filter parameters for caching and modelling."""
    values = dict(spec or {})
    damping = _valid_positive(values.get("receiver_damping"))
    receiver_cutoff = _valid_positive(values.get("receiver_cutoff_hz"))
    tib_cutoff = _valid_positive(values.get("tib_cutoff_hz"))
    try:
        tib_order = max(0, int(round(float(values.get("tib_order", 0)))))
    except (TypeError, ValueError):
        tib_order = 0
    if not (damping and receiver_cutoff):
        damping = receiver_cutoff = 0.0
    if not tib_cutoff:
        tib_order = 0
    return damping, receiver_cutoff, tib_order, tib_cutoff


def _foh_stage_matrix(times: np.ndarray, a: np.ndarray, b: np.ndarray,
                      c: np.ndarray) -> np.ndarray:
    """Map irregular samples through a continuous state-space low-pass stage.

    The input is linearly interpolated between samples (first-order hold), then
    integrated exactly over each interval with a matrix exponential.  This is
    important on logarithmic TDEM time grids: a zero-order hold would introduce
    artificial steps that are much larger than the microsecond filter constants.
    """
    from scipy.linalg import expm

    sample_times = np.asarray(times, dtype=float).ravel()
    n_samples = sample_times.size
    n_states = int(a.shape[0])
    operator = np.zeros((n_samples, n_samples), dtype=float)
    state = np.zeros((n_states, n_samples), dtype=float)
    identity = np.eye(n_samples, dtype=float)
    for index in range(1, n_samples):
        dt = float(sample_times[index] - sample_times[index - 1])
        # Augment x' = A x + B u with u' = slope and slope' = 0.  The top
        # blocks of exp(M dt) give exact coefficients for u[k-1] and u[k].
        augmented = np.zeros((n_states + 2, n_states + 2), dtype=float)
        augmented[:n_states, :n_states] = a
        augmented[:n_states, n_states] = b.ravel()
        augmented[n_states, n_states + 1] = 1.0
        transition = expm(augmented * dt)
        ad = transition[:n_states, :n_states]
        from_value = transition[:n_states, n_states]
        from_slope = transition[:n_states, n_states + 1]
        g1 = from_slope / dt
        g0 = from_value - g1
        state = (
            ad @ state
            + g0[:, None] * identity[index - 1][None, :]
            + g1[:, None] * identity[index][None, :]
        )
        operator[index] = (c @ state).ravel()
    return operator


@lru_cache(maxsize=32)
def _analog_filter_matrix_cached(
    times: tuple, damping: float, receiver_cutoff: float,
    tib_order: int, tib_cutoff: float,
) -> np.ndarray:
    """Causal GEX analog-response matrix on one internal time grid."""
    sample_times = np.asarray(times, dtype=float)
    operator = np.eye(sample_times.size, dtype=float)
    if receiver_cutoff > 0.0:
        omega = 2.0 * np.pi * receiver_cutoff
        # Standard unity-DC-gain second-order low pass specified by natural
        # frequency and damping ratio, matching RxCoilLPFilter in a GEX file.
        a = np.array([[0.0, 1.0], [-omega ** 2, -2.0 * damping * omega]])
        b = np.array([[0.0], [omega ** 2]])
        c = np.array([[1.0, 0.0]])
        operator = _foh_stage_matrix(sample_times, a, b, c) @ operator
    if tib_order > 0 and tib_cutoff > 0.0:
        omega = 2.0 * np.pi * tib_cutoff
        a = np.array([[-omega]])
        b = np.array([[omega]])
        c = np.array([[1.0]])
        for _ in range(tib_order):
            operator = _foh_stage_matrix(sample_times, a, b, c) @ operator
    operator.setflags(write=False)
    return operator


def _analog_sampling(config: "TDEMSurveyConfig"):
    """Return SimPEG times and the combined analog-filter/gate operator."""
    targets, gate_weights = _gate_sampling(config)
    damping, receiver_cutoff, tib_order, tib_cutoff = _analog_parameters(
        config.analog_lowpass
    )
    if receiver_cutoff <= 0.0 and tib_order <= 0:
        return targets, gate_weights

    targets = np.asarray(targets, dtype=float).ravel()
    positive = targets[np.isfinite(targets) & (targets > 0.0)]
    if positive.size != targets.size:
        raise ValueError("Analog-filtered TDEM receiver times must be finite and positive.")
    # Start well before both the first gate and the fastest filter time constant.
    # Early linear support resolves the microsecond electronics without making
    # late-time modelling unnecessarily dense.
    highest_cutoff = max(receiver_cutoff, tib_cutoff if tib_order else 0.0)
    filter_scale = 1.0 / (2.0 * np.pi * highest_cutoff)
    # Simulation1DLayered loses useful accuracy extremely close to t=0 while
    # these electronics have 0.2--0.4 microsecond time constants.  Starting at
    # 0.05 microsecond is still well inside both filters and avoids making the
    # Hankel/time transform chase an irrelevant nanosecond singularity.
    start = max(5e-8, min(float(positive.min()) / 100.0, filter_scale / 40.0))
    # Dense support is only needed until the first receiver sample (or twelve
    # time constants, whichever is later).  Beyond that point the fast analog
    # stages have settled and the existing gate quadrature nodes supply the
    # local first-order-hold slopes.  Extending the dense grid to the last gate
    # makes a line inversion slower without resolving any extra electronics.
    slowest_cutoff = min(
        value for value in (receiver_cutoff, tib_cutoff if tib_order else 0.0)
        if value > 0.0
    )
    settled = 12.0 / (2.0 * np.pi * slowest_cutoff)
    support_end = min(
        float(positive.max()), max(float(positive.min()), settled)
    )
    count = max(24, int(config.analog_filter_samples))
    internal_times = np.unique(np.concatenate([
        # The electronics have a fixed (not logarithmic) time constant, so a
        # linear grid spends its nodes where the convolution needs them instead
        # of oversampling the very start and undersampling the first gate.
        np.linspace(start, support_end, count), positive
    ]))
    lookup = np.searchsorted(internal_times, targets)
    selection = np.zeros((targets.size, internal_times.size), dtype=float)
    selection[np.arange(targets.size), lookup] = 1.0
    analog = _analog_filter_matrix_cached(
        tuple(float(value) for value in internal_times),
        damping, receiver_cutoff, tib_order, tib_cutoff,
    )
    reduction = selection @ analog
    if gate_weights is not None:
        reduction = gate_weights @ reduction
    return internal_times, reduction


# ---------------------------------------------------------------------------
# TDEMForward Modeling
# ---------------------------------------------------------------------------
class TDEMForwardModeling:
    """Class for forward modeling of Time-Domain Electromagnetic (TDEM) data.
    
    This class provides functionality for 1D layered Earth TDEM forward modeling
    using SimPEG's time_domain module.
    
    Example:
        >>> # Define layer model
        >>> thicknesses = np.array([10.0, 30.0])
        >>> conductivity = np.array([0.01, 0.1, 0.001])  # S/m
        >>> 
        >>> # Create forward modeler
        >>> fwd = TDEMForwardModeling(thicknesses=thicknesses)
        >>> 
        >>> # Compute response
        >>> response = fwd.forward(conductivity)
    """
    
    def __init__(
        self,
        thicknesses: np.ndarray,
        survey_config: Optional[TDEMSurveyConfig] = None,
        survey: Optional[tdem.Survey] = None
    ):
        """
        Initialize TDEM forward modeling.
        
        Args:
            thicknesses: Layer thicknesses (m), N-1 values for N layers
            survey_config: Survey configuration (creates survey if survey not provided)
            survey: Pre-defined SimPEG TDEM survey (optional)
        """
        self.thicknesses = np.asarray(thicknesses)
        self.n_layers = len(thicknesses) + 1
        
        # Create or use provided survey
        if survey is not None:
            self.survey = survey
        else:
            if survey_config is None:
                survey_config = TDEMSurveyConfig()
            self.survey_config = survey_config
            self.survey = self._create_survey(survey_config)
        
        # Create simulation
        self.model_mapping = maps.IdentityMap(nP=self.n_layers)
        self.simulation = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=self.model_mapping,
        )
    
    def _create_survey(self, config: TDEMSurveyConfig) -> tdem.Survey:
        """Create TDEM survey from configuration.
        
        Args:
            config: Survey configuration
            
        Returns:
            SimPEG TDEM Survey object
        """
        # Create receiver
        receiver_cls = (
            tdem.receivers.PointMagneticFluxTimeDerivative
            if str(config.receiver_type).lower() in {"dbdt", "db/dt", "time_derivative"}
            else tdem.receivers.PointMagneticFluxDensity
        )
        self._sample_times, self._gate_weights = _analog_sampling(config)
        receiver_list = [
            receiver_cls(
                config.receiver_location,
                self._sample_times,
                orientation=config.receiver_orientation
            )
        ]

        # Create waveform. Measured turn-off nodes win over the named shapes:
        # a step is only a stand-in for a ramp nobody recorded.
        nodes = np.asarray(config.waveform_times if config.waveform_times is not None
                           else [], dtype=float).ravel()
        currents = np.asarray(config.waveform_currents if config.waveform_currents
                              is not None else [], dtype=float).ravel()
        if nodes.size >= 2 and nodes.size == currents.size:
            waveform = tdem.sources.PiecewiseLinearWaveform(
                times=nodes, currents=currents)
        elif config.waveform_type == "ramp_off":
            waveform = tdem.sources.RampOffWaveform()
        else:
            waveform = tdem.sources.StepOffWaveform()

        # SimPEG's 1D CircularLoop implementation only supports a central-loop
        # receiver. Ground TEM systems such as TEM2Go use a small transmitter loop
        # and an offset receiver; represent that loop by its equivalent magnetic
        # dipole moment (I * N * area) when an offset is present.
        offset = np.linalg.norm(
            np.asarray(config.receiver_location, dtype=float)[:2]
            - np.asarray(config.source_location, dtype=float)[:2]
        )
        if offset > 1e-9:
            magnetic_moment = (
                float(config.source_moment)
                if config.source_moment is not None
                else float(config.source_current)
                * max(1, int(config.source_turns))
                * np.pi
                * float(config.source_radius) ** 2
            )
            source = tdem.sources.MagDipole(
                receiver_list=receiver_list,
                location=config.source_location,
                waveform=waveform,
                moment=magnetic_moment,
                orientation=config.receiver_orientation,
            )
        else:
            source = tdem.sources.CircularLoop(
                receiver_list=receiver_list,
                location=config.source_location,
                waveform=waveform,
                current=config.source_current,
                radius=config.source_radius,
                n_turns=max(1, int(config.source_turns)),
            )
        source_list = [source]
        
        return tdem.Survey(source_list)
    
    def forward(
        self,
        conductivity: np.ndarray,
        log_input: bool = False
    ) -> np.ndarray:
        """
        Compute forward response for a given conductivity model.
        
        Args:
            conductivity: Conductivity values for each layer (S/m)
            log_input: If True, conductivity is log-transformed
            
        Returns:
            Forward response (magnetic flux density, T)
        """
        if log_input:
            sigma = np.exp(conductivity)
        else:
            sigma = np.asarray(conductivity)

        predicted = np.asarray(self.simulation.dpred(sigma), dtype=float).ravel()
        weights = getattr(self, "_gate_weights", None)
        return predicted if weights is None else weights @ predicted

    def sensitivity(self, conductivity: np.ndarray) -> np.ndarray:
        """Analytic d(response)/d(conductivity), averaged over the gate windows.

        The same reduction the forward applies has to be applied to the
        Jacobian, or the two describe different data.
        """
        jacobian = np.asarray(self.simulation.getJ(np.asarray(conductivity)),
                              dtype=float)
        weights = getattr(self, "_gate_weights", None)
        return jacobian if weights is None else weights @ jacobian
    
    def forward_with_noise(
        self,
        conductivity: np.ndarray,
        noise_level: float = 0.05,
        seed: Optional[int] = None,
        log_input: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute forward response with added Gaussian noise.
        
        Args:
            conductivity: Conductivity values for each layer (S/m)
            noise_level: Relative noise level (default 5%)
            seed: Random seed for reproducibility
            log_input: If True, conductivity is log-transformed
            
        Returns:
            Tuple of (noisy_data, clean_data, uncertainties)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Compute clean response
        clean_data = self.forward(conductivity, log_input=log_input)
        
        # Add noise
        noise = noise_level * np.abs(clean_data) * np.random.randn(len(clean_data))
        noisy_data = clean_data + noise
        
        # Compute uncertainties
        uncertainties = noise_level * np.abs(noisy_data)
        
        return noisy_data, clean_data, uncertainties
    
    def get_times(self) -> np.ndarray:
        """Get the time channels from the survey."""
        return self.survey_config.times if hasattr(self, 'survey_config') else None
    
    @property
    def n_data(self) -> int:
        """Number of data points."""
        if hasattr(self, "survey_config"):
            return int(np.asarray(self.survey_config.times).size)
        return self.survey.nD


# ---------------------------------------------------------------------------
# create tdem survey
# ---------------------------------------------------------------------------
def create_tdem_survey(
    times: np.ndarray,
    source_radius: float = 10.0,
    source_current: float = 1.0,
    source_location: Optional[np.ndarray] = None,
    receiver_location: Optional[np.ndarray] = None,
    receiver_orientation: str = "z",
    waveform_type: str = "step_off"
) -> tdem.Survey:
    """
    Create a TDEM survey for 1D sounding.
    
    Args:
        times: Time channels (s)
        source_radius: Loop radius (m)
        source_current: Peak current (A)
        source_location: Source center [x, y, z] (m)
        receiver_location: Receiver position [x, y, z] (m)
        receiver_orientation: Measurement component ('x', 'y', 'z')
        waveform_type: Waveform type ('step_off', 'ramp_off')
        
    Returns:
        SimPEG TDEM Survey object
    """
    if source_location is None:
        source_location = np.array([0.0, 0.0, 0.0])
    if receiver_location is None:
        receiver_location = np.array([0.0, 0.0, 0.0])
    
    config = TDEMSurveyConfig(
        source_location=source_location,
        source_radius=source_radius,
        source_current=source_current,
        receiver_location=receiver_location,
        receiver_orientation=receiver_orientation,
        times=times,
        waveform_type=waveform_type
    )
    
    fwd = TDEMForwardModeling(thicknesses=np.array([1.0]), survey_config=config)
    return fwd.survey


# ---------------------------------------------------------------------------
# hydro to tdem
# ---------------------------------------------------------------------------
def simulate_tdem_sounding_from_hydro(
    water_content: np.ndarray,
    porosity: np.ndarray,
    layer_thicknesses: np.ndarray,
    sigma_w: Union[float, np.ndarray] = 0.05,
    m: Union[float, np.ndarray] = 1.5,
    n: Union[float, np.ndarray] = 2.0,
    sigma_s: Union[float, np.ndarray] = 0.0,
    times: Optional[np.ndarray] = None,
    source_radius: float = 10.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert hydrological properties to TDEM response.
    
    This function takes water content and porosity from hydrological models
    and computes the expected TDEM response using petrophysical relationships.
    
    Args:
        water_content: Water content for each layer (-)
        porosity: Porosity for each layer (-)
        layer_thicknesses: Thickness of each layer except bottom (m)
        sigma_w: Pore water conductivity (S/m)
        m: Cementation exponent
        n: Saturation exponent
        sigma_s: Surface conductivity (S/m)
        times: Time channels (s), default is logspace(-5, -2, 31)
        source_radius: Loop radius (m)
        noise_level: Relative noise level for synthetic data
        seed: Random seed for reproducibility
        verbose: Print progress information
        
    Returns:
        Tuple of (noisy_data, clean_data, uncertainties, conductivity)
    """
    from PyHydroGeophysX.petrophysics.resistivity_models import WS_Model
    
    n_layers = len(water_content)
    
    # Ensure arrays
    water_content = np.atleast_1d(water_content)
    porosity = np.atleast_1d(porosity)
    
    if np.isscalar(sigma_w):
        sigma_w = np.full(n_layers, sigma_w)
    if np.isscalar(m):
        m = np.full(n_layers, m)
    if np.isscalar(n):
        n = np.full(n_layers, n)
    if np.isscalar(sigma_s):
        sigma_s = np.full(n_layers, sigma_s)
    
    # Calculate saturation
    saturation = water_content / porosity
    
    if verbose:
        print(f"Computing conductivity for {n_layers} layers...")
    
    # Convert to conductivity using Waxman-Smits model
    resistivity = np.zeros(n_layers)
    for i in range(n_layers):
        resistivity[i] = WS_Model(
            saturation[i],
            porosity[i],
            sigma_w[i],
            m[i],
            n[i],
            sigma_s[i]
        )
    
    conductivity = 1.0 / resistivity
    
    if verbose:
        print(f"Conductivity range: {conductivity.min():.4f} - {conductivity.max():.4f} S/m")
    
    # Set up times
    if times is None:
        times = np.logspace(-5, -2, 31)
    
    # Create survey configuration
    survey_config = TDEMSurveyConfig(
        times=times,
        source_radius=source_radius
    )
    
    # Create forward modeler
    fwd = TDEMForwardModeling(
        thicknesses=layer_thicknesses,
        survey_config=survey_config
    )
    
    # Compute response
    noisy_data, clean_data, uncertainties = fwd.forward_with_noise(
        conductivity,
        noise_level=noise_level,
        seed=seed
    )
    
    if verbose:
        print(f"Generated {len(noisy_data)} data points")
    
    return noisy_data, clean_data, uncertainties, conductivity


def hydro_to_tdem(*args, **kwargs):
    """Deprecated alias for :func:`simulate_tdem_sounding_from_hydro`."""
    import warnings

    warnings.warn(
        "forward.tdem_forward.hydro_to_tdem is deprecated; use "
        "simulate_tdem_sounding_from_hydro for a single column or "
        "Hydro_modular.hydro_to_tdem for a profile. This compatibility shim is "
        "deprecated in 0.4.0 and will be removed in 0.5.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return simulate_tdem_sounding_from_hydro(*args, **kwargs)
