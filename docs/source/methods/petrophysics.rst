.. _petrophysics-fluid-properties:

Petrophysics and pore-fluid properties
======================================

Capabilities and inputs
-----------------------

Use these Python helpers to connect pore-fluid chemistry, clay exchange
capacity and effective grain-contact pressure to electrical resistivity and
seismic velocity. They require NumPy and SciPy through the package, with no
optional geophysical engine. Inputs may be scalars or broadcast-compatible
arrays. These options are available through Python; the desktop controls do
not yet expose them. For workflows, start with :doc:`/tutorials/index`;
for signatures, see :doc:`/api/petrophysics`.

The defaults remain unchanged: Hertz-Mindlin coordination number 4, water
modulus 2.0 GPa, liquid density 1000 kg/m3, and effective pressure estimated
as ``(mineral_density - 1000) * 9.8 * depth`` in Pa. A coordination number
around 9 is a common idealized random-packing choice, not a universal
recommendation; calibrate it to the material (Mavko et al., 2009, chapter 5).
Set it through ``HertzMindlinModel(coordination_number=9)`` when appropriate.

Brine and seismic velocity
--------------------------

``batzle_wang_brine(p, T, salinity)`` uses pressure in Pa, temperature in
degrees Celsius and NaCl mass fraction in kg/kg of solution. It returns an
object with ``bulk_modulus`` in GPa and ``density`` in kg/m3. The implementation
uses the Batzle-Wang (1992) liquid polynomials with a conservative input
domain of 0--100 MPa, 0--100 degrees Celsius and 0--0.35 kg/kg. These bounds
do not check boiling or salt precipitation; ensure a liquid solution exists.

At 5 MPa and 25 degrees Celsius, fresh water gives 2.25843 GPa and
998.19925 kg/m3. The latter is the empirical polynomial result, not a
separate high-accuracy water equation of state. Increasing salinity at
these conditions raises both density and modulus.

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.petrophysics import HertzMindlinModel, batzle_wang_brine

   brine = batzle_wang_brine(p=5e6, T=25, salinity=0.035)
   vp_high, vp_low, elastic = HertzMindlinModel().calculate_velocity(
       porosity=np.array([0.2, 0.3]), saturation=1.0,
       bulk_modulus=36.6, shear_modulus=45.0, mineral_density=2650,
       effective_pressure=2e6,
       fluid_modulus=brine.bulk_modulus,
       fluid_density=brine.density,
       return_moduli=True,
   )

Pore-fluid pressure and effective grain-contact pressure are different
physical inputs; do not equate them automatically. An explicit
``effective_pressure`` overrides depth, including at zero. Changing
``fluid_density`` affects bulk density only, retaining the historical
pressure estimator unless an effective pressure is supplied.

``fluid_modulus`` replaces the liquid modulus in the existing Brie mixture
(gas modulus 0.01 GPa, exponent 3). An explicit value at or below
1e-12 GPa instead selects the dry-frame limit at every saturation:
``K_sat = K_dry``. Density still follows the supplied saturation and densities.
This permits dry-frame diagnostics without silently changing bulk mass.

With ``return_moduli=False``, the return value remains ``(Vp_high, Vp_low)``.
When True, a third item contains ``Vs_low``, ``Vs_high``, ``K_dry_low``,
``K_dry_high``, ``G_dry_low``, ``G_dry_high``, ``K_sat_low``, ``K_sat_high`` and
``density``. Moduli are GPa, velocities m/s and density kg/m3. The relations
are ``Vp**2 = (K_sat + 4*G_dry/3)*1e9/density`` and
``Vs**2 = G_dry*1e9/density``.

.. _hertz-mindlin-shear-fix:

Hertz-Mindlin shear correction and legacy results
-------------------------------------------------

The Hertz-Mindlin contact shear modulus is

.. math::

   G_{\mathrm{HM}} =
   \frac{5-4\nu}{5(2-\nu)}
   \left[\frac{3 C^2 (1-\phi_c)^2 G_m^2 P}
   {2\pi^2(1-\nu)^2}\right]^{1/3}.

The prefactor is outside the cube root. Use consistent units for pressure
and mineral moduli: this implementation converts the Pa input to GPa.
The formula is verified against the authors' `Rock Physics Handbook
companion routines
<https://pangea.stanford.edu/research/srb/books/RPHII/RPHtools/>`_
(``hertzmind.m``, supplied with Mavko, Mukerji and Dvorkin, 2009), and
`rockphypy GM.hertzmindlin
<https://rockphypy.readthedocs.io/en/latest/_modules/rockphypy/GM.html#GM.hertzmindlin>`_
at ``f=1``. rockphypy expects pressure in MPa, so 6.5502 MPa is supplied
there as 6.5502 and to this API as 6.5502e6.

Affected revisions before ``962a9ced`` used ``10 - 2*nu`` in the denominator
and put the class method's prefactor inside the cube root. The older
standalone ``vel_porous`` had the denominator error only. Those are bugs,
not alternative modeling defaults. The equations were corrected in that
commit; ``vel_porous`` now delegates to ``HertzMindlinModel.calculate_velocity``
so it cannot drift from the two-value and ``return_moduli=True`` paths.
No option restores the incorrect formula.

For the dt_framework notebook 01 section 7.2 baseline at x=2.51 m,
use phi=0.2, S=1, Km=40.39369 GPa, Gm=43.10014 GPa,
mineral density=2657.5 kg/m3, C=9, phi_c=0.4,
effective pressure=6.5502 MPa, fluid modulus=2.260256 GPa and
fluid density=998.4548 kg/m3. These are explicit supplied fluid properties,
not values recomputed by a separate water-property model.

.. list-table:: Lower-bound baseline at identical fluid properties
   :header-rows: 1

   * - Quantity
     - Historical class equation
     - Standard equation
   * - Dry bulk modulus (GPa)
     - 5.691664
     - 4.663967
   * - Dry shear modulus (GPa)
     - 7.614628
     - 5.103210
   * - Saturated bulk modulus (GPa)
     - 12.733917
     - 12.084909
   * - Vp (m/s)
     - 3137.011
     - 2849.907
   * - Vs (m/s)
     - 1809.457
     - 1481.309

Both columns use bulk density 2325.69096 kg/m3. The historical function
did not accept fluid overrides: its dry moduli were evaluated from the
original source and substituted with the supplied fluids for this comparison.
Running the actual old API with its fixed water properties gives lower Vp
3088.660 m/s versus 2794.058 m/s from the corrected legacy API, at the same
equivalent depth of 403.250531 m.

Results change even when the default coordination number remains 4.
For quartz (Km=36.6 GPa, Gm=45 GPa, mineral density=2650 kg/m3) with
phi=0.2, S=1 and depth=5 m, the class's lower Vp decreases from
2109.732 to 2039.173 m/s (3.344%). At phi=0.3, S=0.5 and depth=5 m,
it decreases from 1038.908 to 939.168 m/s (9.600%). Historical vel_porous
instead increases slightly for the first case, from 2036.809 to 2039.173 m/s.
Recompute outputs and review calibrations made with the affected equations.

Ionic conductivity and charge
-----------------------------

``pore_fluid_conductivity`` takes a dictionary of dissolved ion concentrations
in mol/L, with repeated trailing charge signs, such as ``Ca++`` and ``CO3--``.
``ion_charge`` returns 2 and -2 for these names. Missing or mixed signs are
rejected; formula digits are not charge numbers (``NH4+`` has charge +1).

The coefficient convention is **equivalent conductivity**, in S m2/eq:

.. math::

   \sigma_w = 1000 \sum_i |z_i|\lambda_i c_i.

The following table shows the original values in S cm2/eq; multiply by
``1e-4`` for the optional ``equivalent_conductivities`` argument. This mapping
overrides selected defaults and can add ions. Unknown ions without a
coefficient are rejected. Source: Kanciruk et al. (1986), ORNL/TM-10153,
Table 8, CONCAL, printed page 30.

.. list-table:: Limiting equivalent conductivities at 25 degrees Celsius
   :header-rows: 1

   * - Ion
     - S cm2/eq
   * - H+
     - 349.65
   * - OH-
     - 198.0
   * - Na+
     - 50.08
   * - Cl-
     - 76.31
   * - Ca++
     - 59.47
   * - HCO3-
     - 44.5
   * - CO3--
     - 69.3

The default table also includes Mg++, K+, NH4+, SO4--, NO3- and F-.
A molar coefficient already includes the charge contribution: for Ca++
118.94 S cm2/mol corresponds to 59.47 S cm2/eq. Supplying the former
as an equivalent coefficient would double-count calcium's charge.

.. code-block:: python

   from PyHydroGeophysX.petrophysics import pore_fluid_conductivity

   sigma_w, contributions = pore_fluid_conductivity({"Na+": 0.01, "Cl-": 0.01})
   # sigma_w = 0.12639 S/m; contributions sum to sigma_w

Use ``return_contributions=False`` to return only the total. Concentrations
are dissolved species, not neutral salt totals; no speciation, ion pairing
or automatic water dissociation is computed.

Optional concentration correction
---------------------------------

The default is the limiting-ion approximation. It overestimates the
conductivity of finite-concentration salt solutions. For example,
0.005 M CaCl2 gives 0.13578 S/m, about 6.2% above the 0.1279 S/m
conductometry example in Girault (2004).

An optional calibrated Kohlrausch-type correction uses ionic strength
``I = 0.5 * sum(c_i*z_i**2)`` in mol/L and multiplies every contribution
by ``1 - a*sqrt(I)``. Enable ``ionic_strength_correction=True`` and provide
``kohlrausch_coefficient=a`` in (mol/L)**(-1/2). There is no universal default
coefficient. For a fixed electrolyte with ``I = q*c_salt``, the published
Kohlrausch form ``Lambda = Lambda0 - A*sqrt(c_salt)`` corresponds to
``a = A/(Lambda0*sqrt(q))``. See Metrohm's conductometry monograph, page 14.

This is a dilute, fixed-composition empirical correction. Calibrate the
coefficient against data for the intended electrolyte and concentration
range; it is not a predictive mixed-brine transport model. Corrected
contributions retain their limiting proportions and sum to the corrected
total. An absent coefficient or a negative correction factor raises
``ValueError``.

CEC, surface conduction and clipping
------------------------------------

``bulk_cec_to_qv`` converts CEC in eq/m3 **bulk material** to Qv in
meq/cm3 **pore volume**: ``Qv = CEC/porosity * 1e-3``. Thus CEC 10
and porosity 0.2 give Qv 0.05. Mass-specific CEC in meq/100 g requires
a separate conversion using the appropriate bulk mass per volume.

``waxman_smits_surface_conductivity(sigma_w, qv)`` implements the specified
25-degree-Celsius Waxman-Thomas relation:

.. math::

   \sigma_s = 4.6 [1 - 0.6\exp(-\sigma_w/1.3)] Q_v.

Here ``sigma_w`` and ``sigma_s`` are S/m and Qv is meq/cm3; 1.3 has
units S/m and 4.6 has units (S/m)/(meq/cm3). Pass the result as
``sigma_s`` to ``WS_Model``, which scales both fluid and surface terms
by the formation factor. Other converters' ``sigma_sur`` is outside
that scaling, so the coefficients are not interchangeable.

.. code-block:: python

   from PyHydroGeophysX.petrophysics import (
       bulk_cec_to_qv, waxman_smits_surface_conductivity, WS_Model,
   )

   qv = bulk_cec_to_qv(10.0, porosity=0.2)
   sigma_s = waxman_smits_surface_conductivity(sigma_w, qv)
   resistivity = WS_Model(
       saturation=0.8, porosity=0.2, sigma_w=sigma_w, m=2, n=2, sigma_s=sigma_s,
   )

``WS_Model`` retains saturation clipping to [0.001, 1], a conductivity
floor of 1e-6 S/m, and resistivity clipping to [0.1, 1e6] ohm-m.
It emits one ``RuntimeWarning`` per call when any of these limits activates,
including the conductivity floor. Normal inputs emit no clipping warning.
NaN values propagate. These limits are numerical safeguards, not inferred
physical bounds on your sample.

References
----------

Full BibTeX records are in :doc:`/citation`. Relevant sources are Batzle
and Wang (1992), Waxman and Thomas (1974), Kanciruk et al. (1986),
Metrohm's *Conductometry - Conductivity Measurement*, Girault (2004),
and Mavko, Mukerji and Dvorkin (2009).
