MODFLOW and ParFlow
===================

The hydrological end of the connection. Every other method in this section
measures the subsurface; MODFLOW and ParFlow are what those measurements are
compared against, and what an interpreted result is written back into.

What you can do
---------------

- Read water content, saturation and porosity out of a MODFLOW or ParFlow run,
  for one timestep or a whole series.
- Carry those states through petrophysics into a geophysical property, then
  forward model the survey that property would produce.
- Map an interpreted geophysical quantity back onto the hydrological grid.
- Write MODFLOW 6 or ParFlow inputs carrying that update, as a copy. Neither
  writer modifies the source model and neither runs the solver.
- Compare hydrological responses with and without geophysically informed
  structure.

Required inputs
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - What is read
     - Notes
   * - MODFLOW 6
     - UZF water content, porosity
     - ``MODFLOWWaterContent`` takes the simulation workspace and a 2D
       ``idomain``; nonzero cells receive consecutive UZF indices in row-major
       order. FloPy reads and writes the simulation.
   * - ParFlow
     - Saturation, porosity, domain mask
     - ``ParflowSaturation`` takes the run directory and the run name, and
       discovers the available timesteps itself.

For writing, you supply the update as an array matching the model grid exactly,
in the model's own length and time units.

Typical outputs
---------------

Going out of the model: arrays of water content, saturation or porosity on the
model grid, ready for the petrophysical conversion.

Going back in: a copy of the simulation with the mapped fields replaced.
``write_modflow6_inputs`` accepts ``hydraulic_conductivity``,
``vertical_conductivity``, ``specific_storage``, ``specific_yield``,
``initial_head`` and ``bottom_elevation``; packages, stress periods and boundary
conditions are left as configured.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Your goal
     - Start here
   * - Understand how the two directions connect
     - :doc:`/tutorials/hydro_geophysical_interaction`
   * - Load model outputs and inspect them
     - :doc:`/auto_examples/Ex_model_output`
   * - Predict an ERT survey from a model state
     - :doc:`/tutorials/hydrology_to_ert`
   * - Predict an EM sounding from a model state
     - :doc:`/tutorials/hydrology_to_tdem`
   * - Compare several methods over one profile
     - :doc:`/auto_examples/Ex_hydro_to_multigeophys`
   * - Write an interpreted result back into the model
     - :doc:`/tutorials/hydrological_input_updates`
   * - Run MODFLOW with geophysically informed structure
     - :doc:`/tutorials/modflow_feedback`

Desktop support
---------------

:doc:`Desktop Studio </agents/desktop_studio>` carries the hydrological side in
its project tree: **Hydro → Geophysics** for the forward direction, in several
guided stages, and **ERT → Water Content** for the interpretation that feeds
back.

Python examples
---------------

- :doc:`/auto_examples/Ex_model_output`: read both model families and plot what
  came out.
- :doc:`/auto_examples/Ex_ERT_workflow`: the full hydrology-to-ERT chain.
- :doc:`/auto_examples/Ex_MODFLOW_geophysics_feedback`: run and compare
  hydrological responses with a uniform and an interpreted interface. This one
  actually runs the solver.

Reading a timestep out of each model family:

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.model_output import MODFLOWWaterContent, ParflowSaturation

   idomain = np.loadtxt("examples/data/modflow/id.txt")
   mf = MODFLOWWaterContent(model_directory="examples/data/modflow", idomain=idomain)
   water_content = mf.load_timestep(1)

   pf = ParflowSaturation(model_directory="examples/data/parflow/test2", run_name="test2")
   saturation = pf.load_timestep(0)

``Hydro_modular`` then carries a state straight through to a survey response:
``hydro_to_ert``, ``hydro_to_srt``, ``hydro_to_tdem``, ``hydro_to_fdem`` and
``hydro_to_gravity``. Each needs the mesh, the profile interpolation and the
petrophysical parameters alongside the model arrays, so
:doc:`/tutorials/hydrology_to_ert` walks one of them end to end rather than
compressing it here.

Relevant API
------------

- :doc:`/api/model_output` for ``MODFLOWWaterContent``, ``MODFLOWPorosity``,
  ``ParflowSaturation`` and ``ParflowPorosity``.
- :doc:`/api/Hydro_modular` for the conversions from a hydrological state to
  each geophysical method.
- :doc:`/api/model_input` for ``write_modflow6_inputs``,
  ``write_parflow_inputs`` and the grid mapping that prepares an update.

Limitations and optional dependencies
-------------------------------------

MODFLOW support goes through FloPy and ParFlow support through pftools, both
installed by the ``geophysics`` extra:

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

The ParFlow readers are imported conditionally, so the package still loads
without pftools; only the ParFlow classes are unavailable. Writing inputs is
not running a model: the writers produce a copy for you to run yourself, and a
MODFLOW 6 groundwater-flow model is required.
