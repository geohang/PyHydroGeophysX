"""Pore-fluid properties with explicit SI inputs and empirical-model units."""

import re
from typing import NamedTuple

import numpy as np


class BrineProperties(NamedTuple):
    """Bulk modulus in GPa and density in kg/m3, with broadcast input shape."""

    bulk_modulus: np.ndarray
    density: np.ndarray


# Limiting equivalent (NOT molar) conductivities at 25 deg C, S m2 / eq.
# Kanciruk et al. (1986), ORNL/TM-10153, Table 8, CONCAL, printed p. 30:
# https://archive.epa.gov/emap/archive-emap/web/pdf/elsi_dd-2.pdf#page=40
# Its S cm2/eq coefficients are converted here by 1e-4. In particular,
# Ca++ is 59.47, not the molar value 118.94; CO3-- is 69.3, not 138.6.
LIMITING_EQUIVALENT_CONDUCTIVITIES = {
    ion: value * 1e-4 for ion, value in {
        "H+": 349.65, "OH-": 198.0, "Na+": 50.08, "Cl-": 76.31,
        "Ca++": 59.47, "HCO3-": 44.5, "CO3--": 69.3, "Mg++": 53.0,
        "K+": 73.48, "NH4+": 73.5, "SO4--": 80.0, "NO3-": 71.42,
        "F-": 55.4,
    }.items()
}


def _nonnegative(value, name):
    value = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} must contain finite nonnegative values.")
    return value


def ion_charge(name):
    """Return signed charge from repeated trailing signs (Ca++=2, CO3--=-2).

    Only the repeated-sign convention is accepted. Formula digits belong to
    the species (NH4+ has charge +1); numeric charge notation is not parsed.
    Missing or mixed trailing signs raise ValueError.
    """
    if not isinstance(name, str):
        raise ValueError("An ion name must be a string with trailing signs.")
    match = re.fullmatch(r"([^+\-\s]+)(\++|-+)", name)
    if match is None:
        raise ValueError("Use an ion name with repeated trailing signs, e.g. Ca++.")
    signs = match.group(2)
    return len(signs) * (1 if signs[0] == "+" else -1)


def pore_fluid_conductivity(
    conc, equivalent_conductivities=None, return_contributions=True, *,
    ionic_strength_correction=False, kohlrausch_coefficient=None,
):
    """Calculate pore-water conductivity at 25 deg C in S/m.

    Parameters
    ----------
    conc : mapping
        Ion names to dissolved-species concentrations in mol/L. Values may
        be scalars or broadcast-compatible arrays; supply speciated ions,
        not total analytical concentrations or neutral salts.
    equivalent_conductivities : mapping, optional
        Overrides/additions to the limiting table, in S m2/eq. Do not supply
        molar conductivities: sigma_i = abs(z_i) * lambda_i * c_i * 1000.
        The default table is Kanciruk et al. (1986), ORNL/TM-10153, Table 8,
        CONCAL (p. 30). Unknown ions require an explicit coefficient.
    return_contributions : bool
        If True return (sigma_w, {ion: sigma_i}), otherwise only sigma_w.
        An empty composition returns zero (no implicit water autoionization).
    ionic_strength_correction : bool
        Off by default. Apply a calibrated Kohlrausch-type factor
        1 - a * sqrt(I), where I = 0.5 * sum(c_i * z_i**2) in mol/L,
        to every contribution. Intended only for dilute strong electrolytes
        of the composition used to calibrate a; no speciation is performed.
    kohlrausch_coefficient : float, optional
        Required when correction is enabled, in (mol/L)**(-1/2). There is no
        universal default. For a fixed electrolyte, Kohlrausch's
        Lambda = Lambda0 - A * sqrt(c_salt) gives
        a = A / (Lambda0 * sqrt(I/c_salt)). See Metrohm, *Conductometry -
        Conductivity Measurement*, monograph 8.109.5021, p. 14.
        A negative resulting factor raises ValueError rather than clipping.

    Notes
    -----
    Correction is an empirical fixed-composition approximation, not a
    general mixed-electrolyte transport model. Contributions under this
    approximation retain their limiting relative proportions.
    """
    if not isinstance(ionic_strength_correction, (bool, np.bool_)):
        raise ValueError("ionic_strength_correction must be boolean.")
    if ionic_strength_correction:
        if kohlrausch_coefficient is None:
            raise ValueError("Supply a calibrated kohlrausch_coefficient.")
        coefficient = _nonnegative(kohlrausch_coefficient, "kohlrausch_coefficient")
        if coefficient.ndim:
            raise ValueError("kohlrausch_coefficient must be a scalar.")
    elif kohlrausch_coefficient is not None:
        raise ValueError("Enable ionic_strength_correction to use its coefficient.")

    table = dict(LIMITING_EQUIVALENT_CONDUCTIVITIES)
    if equivalent_conductivities is not None:
        table.update(equivalent_conductivities)
    names = list(conc)
    charges = [ion_charge(name) for name in names]
    concentrations = np.broadcast_arrays(*[
        _nonnegative(conc[name], f"concentration of {name}") for name in names
    ]) if names else []
    contributions = {}
    strength = 0.0
    for name, charge, concentration in zip(names, charges, concentrations):
        if name not in table:
            raise ValueError(f"No equivalent conductivity for {name}; supply one.")
        conductivity = _nonnegative(table[name], f"equivalent conductivity of {name}")
        if conductivity.ndim:
            raise ValueError("Equivalent conductivity coefficients must be scalars.")
        contributions[name] = abs(charge) * conductivity * concentration * 1000
        strength = strength + 0.5 * charge**2 * concentration
    if ionic_strength_correction:
        factor = 1 - coefficient * np.sqrt(strength)
        if np.any(factor < 0):
            raise ValueError("Kohlrausch correction is outside its calibrated dilute range.")
        contributions = {name: value * factor for name, value in contributions.items()}
    sigma_w = sum(contributions.values(), 0.0)
    return (sigma_w, contributions) if return_contributions else sigma_w


def bulk_cec_to_qv(cec, porosity):
    """Convert CEC in eq/m3 bulk to Qv in meq/cm3 pore.

    Qv = CEC / porosity * 1e-3. This accepts bulk-volume CEC, not
    mass-specific laboratory CEC (e.g. meq/100 g). Inputs broadcast.
    Porosity must be in (0, 1]; CEC must be finite and nonnegative.
    """
    cec, porosity = np.broadcast_arrays(
        _nonnegative(cec, "cec"), _nonnegative(porosity, "porosity"))
    if np.any((porosity <= 0) | (porosity > 1)):
        raise ValueError("porosity must be in (0, 1].")
    return cec / porosity * 1e-3


def waxman_smits_surface_conductivity(sigma_w, qv):
    """Return the Waxman-Thomas surface coefficient in S/m at 25 deg C.

    sigma_s = 4.6 * (1 - 0.6 * exp(-sigma_w / 1.3)) * qv,
    with sigma_w in S/m and qv in meq/cm3 pore. Thus 1.3 has units S/m
    and 4.6 has units (S/m)/(meq/cm3). Inputs broadcast.

    Pass the result as sigma_s to WS_Model, which applies the formation
    factor. It is NOT the unscaled sigma_sur in the other converters.
    Reference: Waxman and Thomas (1974), doi:10.2118/4094-PA.
    """
    sigma_w, qv = np.broadcast_arrays(
        _nonnegative(sigma_w, "sigma_w"), _nonnegative(qv, "qv"))
    return 4.6 * (1 - 0.6 * np.exp(-sigma_w / 1.3)) * qv


def batzle_wang_brine(p, T, salinity):
    """Return NaCl-brine bulk modulus (GPa) and density (kg/m3).

    p is pore-fluid pressure in Pa (not effective grain-contact pressure);
    T is temperature in deg C; salinity is NaCl mass fraction in kg/kg of
    solution. Scalars and arrays broadcast together. The empirical liquid
    model is restricted here to 0--100 MPa, 0--100 deg C and 0--0.35 kg/kg;
    the caller must ensure the fluid is liquid and salt remains dissolved.

    Implements the density and acoustic-velocity polynomials of Batzle and
    Wang (1992), *Geophysics* 57, 1396--1408, doi:10.1190/1.1443207.
    The published polynomial uses MPa and g/cm3 internally.
    """
    p, temperature, s = np.broadcast_arrays(
        _nonnegative(p, "p"), _nonnegative(T, "T"),
        _nonnegative(salinity, "salinity"))
    if np.any(p > 100e6) or np.any(temperature > 100) or np.any(s > .35):
        raise ValueError("Use p <= 100 MPa, T <= 100 deg C, salinity <= 0.35.")
    pressure = p / 1e6
    t = temperature
    rho_water = 1 + 1e-6 * (
        -80*t - 3.3*t**2 + .00175*t**3 + 489*pressure - 2*t*pressure
        + .016*t**2*pressure - 1.3e-5*t**3*pressure
        - .333*pressure**2 - .002*t*pressure**2)
    density = 1000 * (rho_water + s * (.668 + .44*s + 1e-6 * (
        300*pressure - 2400*pressure*s
        + t*(80 + 3*t - 3300*s - 13*pressure + 47*pressure*s))))
    coefficients = np.array([
        [1402.85, 1.524, 3.437e-3, -1.197e-5],
        [4.871, -1.11e-2, 1.739e-4, -1.628e-6],
        [-4.783e-2, 2.747e-4, -2.135e-6, 1.237e-8],
        [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
        [-2.197e-7, 7.987e-10, 5.23e-11, -4.614e-13],
    ])
    velocity = np.polynomial.polynomial.polyval2d(t, pressure, coefficients)
    velocity = (velocity + s * (
        1170 - 9.6*t + .055*t**2 - 8.5e-5*t**3 + 2.6*pressure
        - .0029*t*pressure - .0476*pressure**2)
        + s**1.5 * (780 - 10*pressure + .16*pressure**2) - 820*s**2)
    return BrineProperties(density * velocity**2 / 1e9, density)
