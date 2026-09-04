Petrophysics
============

Petrophysical models translate hydrologic properties into geophysical parameters.
PyHydroGeophysX provides Archie, Waxman-Smits, DEM, and Hertz-Mindlin models.

Key Assumptions
---------------

- Archie: clean sands, negligible surface conductivity.
- Waxman-Smits: surface conductivity included; needs clay or CEC terms.
- DEM and Hertz-Mindlin: rock-physics models for velocity conversion.

Inputs and Outputs
------------------

- Inputs: water content, saturation, porosity, mineral properties.
- Outputs: resistivity or seismic velocity fields ready for forward modeling.

Example Usage
-------------

.. code-block:: python

   from PyHydroGeophysX.petrophysics import water_content_to_resistivity

   rho = water_content_to_resistivity(
       water_content=wc,
       rhos=100.0,
       n=2.0,
       porosity=0.3
   )

Examples
--------

- :doc:`/auto_examples/Ex_ERT_workflow`
- :doc:`/auto_examples/Ex_MC_Hydro`
