petrophysics package
====================

The relationships that connect a hydrological state to a geophysical property.
Each one is a stated model with its own assumptions, so the reference for the
model you use is given alongside it.

Submodules
----------

PyHydroGeophysX.petrophysics.resistivity\_models module
-------------------------------------------------------

Water content and saturation to electrical resistivity, and back.

.. rubric:: References

- Archie, G. E. (1942). The electrical resistivity log as an aid in determining
  some reservoir characteristics. *Transactions of the AIME*, 146(1), 54-62.
  The saturation and resistivity relation used by
  ``water_content_to_resistivity`` and its inverses.
- Waxman, M. H. and Smits, L. J. M. (1968). Electrical conductivities in
  oil-bearing shaly sands. *Society of Petroleum Engineers Journal*, 8(2),
  107-122. The surface-conduction term in ``WS_Model``, which matters where
  clay contributes to the measured conductivity.

.. automodule:: PyHydroGeophysX.petrophysics.resistivity_models
   :members:
   :undoc-members:
   :show-inheritance:

PyHydroGeophysX.petrophysics.velocity\_models module
----------------------------------------------------

Porosity, saturation and mineralogy to seismic velocity.

.. rubric:: References

- Mindlin, R. D. (1949). Compliance of elastic bodies in contact. *Journal of
  Applied Mechanics*, 16, 259-268. The grain-contact theory behind
  ``HertzMindlinModel``.
- Gassmann, F. (1951). Über die Elastizität poröser Medien.
  *Vierteljahrsschrift der Naturforschenden Gesellschaft in Zürich*, 96, 1-23.
  Fluid substitution, applied in ``satK``.
- Hashin, Z. and Shtrikman, S. (1963). A variational approach to the theory of
  the elastic behaviour of multiphase materials. *Journal of the Mechanics and
  Physics of Solids*, 11(2), 127-140. The bounds used by the effective-medium
  models, alongside the Voigt-Reuss-Hill average in ``VRHModel``.
- Brie, A., Pampuri, F., Marsala, A. F., and Meazza, O. (1995). Shear sonic
  interpretation in gas-bearing sands. *SPE Annual Technical Conference*,
  SPE 30595. The fluid-mixing law in ``BrieModel``.

.. automodule:: PyHydroGeophysX.petrophysics.velocity_models
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: PyHydroGeophysX.petrophysics
   :members:
   :undoc-members:
   :show-inheritance:
