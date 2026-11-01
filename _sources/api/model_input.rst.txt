model\_input package
====================

Writes interpreted geophysical information back into hydrological model inputs.
This is the return leg of the coupling: ``model_output`` reads a simulation,
and ``model_input`` writes a new one. The conversion step is explicit, so the
relationship used to turn resistivity into a hydrological quantity stays a
stated choice rather than a hidden default.

See :doc:`/tutorials/hydrological_input_updates` for the workflow these
functions belong to.

Submodules
----------

PyHydroGeophysX.model\_input.conversion module
-----------------------------------------------

Maps an interpreted quantity onto a hydrological grid and prepares the update.

.. automodule:: PyHydroGeophysX.model_input.conversion
   :members:
   :undoc-members:
   :show-inheritance:

PyHydroGeophysX.model\_input.writers module
---------------------------------------------

Exports the mapped fields into a new MODFLOW 6 or ParFlow simulation directory,
leaving the source model untouched.

.. automodule:: PyHydroGeophysX.model_input.writers
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: PyHydroGeophysX.model_input
   :members:
   :undoc-members:
   :show-inheritance:
