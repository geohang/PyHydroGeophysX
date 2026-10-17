Write updates back to hydrological models
=========================================

Use ``PyHydroGeophysX.model_input`` to export explicitly mapped hydrological
parameters or initial states into a new simulation directory. This supplies the
input-writing step between geophysical interpretation and another hydrological
run. It does not infer a conductivity–resistivity relationship or automatically
calibrate a hydrological model.

Prepare a defensible update
----------------------------

1. Convert the interpreted geophysical information to a hydrological quantity
   using a stated relationship, calibration or assimilation method.
2. Map it to the hydrological grid and native units. Preserve inactive-cell
   values where appropriate; the writers require finite full-grid arrays.
3. Export into a new directory, inspect the inputs and run the hydrological
   solver separately. Compare the new outputs against observations.

Convert and map recovered resistivity
-------------------------------------

``prepare_hydro_updates`` combines explicit petrophysics, a field transform and
distance-limited nearest-sample transfer. Supply one XYZ cell centre per recovered
resistivity value. Both models must use the same metric coordinate system and
elevation datum. Section depth must first be converted to elevation and the
profile placed in map coordinates; the function cannot infer survey placement.

.. code-block:: python

   import numpy as np
   from parflow.tools.io import read_pfidb, read_pfb
   from PyHydroGeophysX.model_input import (
       HydroGrid, prepare_hydro_updates, write_parflow_inputs,
   )

   config = read_pfidb("baseline_pf/model.pfidb")
   grid = HydroGrid.from_parflow(config, active=np.load("active_cells.npy"))
   mapped = prepare_hydro_updates(
       np.load("recovered_cell_xyz.npy"),  # (N, 3), elevations in metres
       np.load("recovered_resistivity.npy"),  # (N,), ohm-m
       grid,
       # Example assumptions: replace with site-calibrated values.
       petrophysics={"saturation": 1., "rho_fluid": 10., "m": 2., "n": 2.},
       field_transforms={"porosity": "porosity"},
       baselines={"porosity": read_pfb("baseline_pf/porosity.pfb")},
       source_valid=np.load("above_doi.npy"),
       max_distance=5.,  # metres; choose based on survey resolution and support
   )
   print("Updated cells:", mapped["porosity"].updated.sum())
   write_parflow_inputs(config, "baseline_pf", "updated_pf",
                        {name: field.values for name, field in mapped.items()})

Supply **known saturation to estimate porosity**, or **known porosity to estimate
saturation and water content**; one resistivity measurement cannot determine both
independently. ``interpret_resistivity`` uses the package's unscaled
``sigma_sur`` convention (S/m); zero surface conductivity reduces to Archie.
Invalid parameter combinations raise errors instead of silently clipping states.
All petrophysical parameters may be scalars or arrays in source-cell order.

For MODFLOW, construct the target grid with ``HydroGrid.from_modflow(model)``.
For nonuniform / terrain-following ParFlow grids, construct ``HydroGrid`` with
explicit XYZ centres and an active mask. ``from_parflow`` supports uniform grids.

Pressure and conductivity updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``saturation_to_pressure`` converts inferred saturation to unsaturated pressure
head with explicit van Genuchten ``alpha``, ``n`` and ``residual_saturation``.
Use it as a field transform:

.. code-block:: python

   from PyHydroGeophysX.model_input import saturation_to_pressure

   transforms = {
       "initial_pressure": lambda state: saturation_to_pressure(
           state["saturation"], alpha=0.5, n=2., residual_saturation=0.1,
       ),
   }
   # Pass transforms as field_transforms, known porosity in petrophysics,
   # and the original pressure field in baselines.

The numbers above are illustrative. Alpha uses inverse metres for metre-based
pressure head. Fully saturated cells need an explicit ``saturated_pressure``:
saturation alone does not identify positive pressure. MODFLOW initial head is
pressure head **plus elevation**, not pressure head alone.

Hydraulic conductivity is not uniquely determined by resistivity. To use a
site-calibrated relationship, supply a callable under ``hydraulic_conductivity``
(MODFLOW) or ``permeability`` (ParFlow), returning source-cell values in model units.
The callable receives the interpreted ``porosity``, ``saturation`` and
``water_content`` arrays. No conductivity relationship is assumed by default.

Mapping safeguards
~~~~~~~~~~~~~~~~~~

Each returned ``MappedField`` contains ``values``, ``updated``, nearest-source
``distance`` and ``source_index``. Inactive cells and cells beyond
``max_distance`` retain their baseline. Exclude unreliable source cells explicitly
with ``source_valid``. This is nearest-sample transfer, not a conservative volume
remapping or an uncertainty estimate; the distance limit bounds the influence of
a profile but does not establish a geological correlation length.

MODFLOW 6
---------

Requires FloPy and an existing GWF simulation. This adapter does not support
legacy MODFLOW-2005 / NWT input decks.

.. code-block:: python

   import flopy
   import numpy as np
   from PyHydroGeophysX.model_input import write_modflow6_inputs

   sim = flopy.mf6.MFSimulation.load(sim_ws="baseline_mf6", verbosity_level=0)
   updated = write_modflow6_inputs(
       sim, "updated_mf6",
       {"hydraulic_conductivity": np.load("mapped_k.npy"),
        "initial_head": np.load("mapped_head.npy")},
       model_name="flow",
   )
   # Inspect and run the exported simulation with a separately installed MF6 executable.
   check = flopy.mf6.MFSimulation.load(sim_ws=str(updated), verbosity_level=0)

Supported fields: ``hydraulic_conductivity`` (NPF K), ``vertical_conductivity``
(NPF K33), ``specific_storage`` (STO SS), ``specific_yield`` (STO SY), and
``initial_head`` (IC STRT). ``bottom_elevation`` updates structured DIS layer
interfaces; every layer must retain strictly positive thickness, including
inactive columns. See :doc:`modflow_feedback` for a runnable example using
geophysically interpreted interfaces. Their packages must already exist. Absolute K33
updates are rejected when the NPF ``k33overk`` ratio option is enabled.

Arrays must match the model grid exactly. Structured-grid order is
``(layer, row, column)``, top layer first. Use the model's length and time units.
External numerical arrays are internalized before export. Ancillary inputs such
as external time-series / observation files may require separate relocation;
verify their references before running. This is not a universal model packager.

ParFlow
-------

Requires ParFlow's Python tools (``pftools``). Supply an existing workspace and a
``Run`` object or the flat dictionary read from a PFIDB definition.

.. code-block:: python

   import numpy as np
   from parflow.tools.io import read_pfidb
   from PyHydroGeophysX.model_input import write_parflow_inputs

   config = read_pfidb("baseline_pf/model.pfidb")
   updated = write_parflow_inputs(
       config, "baseline_pf", "updated_pf",
       {"porosity": np.load("mapped_porosity.npy"),
        "initial_pressure": np.load("mapped_pressure_head.npy")},
       domain="domain",
   )

The export contains ``updated_model.pfidb`` and ``hydro_updates/*.pfb`` alongside
copies of the source workspace files. Existing forcing, grid and tensor settings
are preserved. Supported full-domain fields are ``permeability``, ``porosity`` and
``initial_pressure``. File references must be relative to the source workspace;
symbolic links are rejected. Start from a workspace without ``hydro_updates``.

PFB arrays use ``(z, y, x)``, bottom layer first. Grid origin, spacing and process
topology come from the definition. Retain its native physical units. In particular,
pressure head is not saturation or water content; converting these requires the
model's constitutive relationship. Do not reverse a MODFLOW array blindly:
horizontal orientation and spatial coordinates must also agree.

Verification and scope
----------------------

Both writers refuse an existing destination, preserve the supplied model object,
and publish the new directory only after writing completes. ``hydro_update.json``
records the updated fields and that the simulation has not been executed.
Tests reload MODFLOW input arrays and ParFlow PFB/PFIDB files. Solver convergence,
mass balance and predictive improvement must be checked in a subsequent run.

Reference interfaces: `FloPy simulation I/O
<https://flopy.readthedocs.io/en/latest/source/flopy.mf6.mfsimbase.html>`_
and `ParFlow PFB tools
<https://parflow.readthedocs.io/en/latest/python/tutorials/pfb.html>`_.
