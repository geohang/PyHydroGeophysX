Tutorials & Workflows
=====================

This section provides task-oriented tutorials for common PyHydroGeophysX workflows.
Each tutorial focuses on a specific use case and includes working code examples.


Getting Started Tutorials
-------------------------

If you're new to PyHydroGeophysX, start here:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tutorial
     - Description
   * - :doc:`/quickstart`
     - Quick overview of all major features
   * - :doc:`/auto_examples/Ex_model_output`
     - Load MODFLOW/ParFlow hydrological model outputs
   * - :doc:`/auto_examples/Ex_ERT_data_process`
     - Load and QC field ERT data from commercial instruments


Workflow 1: Field ERT Data Processing
-------------------------------------

**Goal**: Load field ERT data, perform quality control, and export for inversion.

**When to use**: You have field ERT data from instruments like Syscal, ABEM, E4D, or others.

**Key example**: :doc:`/auto_examples/Ex_ERT_data_process`

.. code-block:: python

   from PyHydroGeophysX.data_processing.ert_data_agent import (
       load_ert_resipy, qc_and_visualize, export_for_inversion
   )

   # Load data from commercial instrument
   ert = load_ert_resipy(
       project_dir="data/ERT",
       data_file="field_data.ohm",
       instrument="E4D"
   )

   # Quality control
   qc_and_visualize(ert, outdir="results/qc")

   # Export for inversion
   export_for_inversion(ert, outdir="results/inversion", fmt="pgimli")


Workflow 2: Hydrology → Geophysics Forward Modeling
---------------------------------------------------

**Goal**: Convert hydrological model outputs to geophysical responses.

**When to use**: You have MODFLOW/ParFlow outputs and want to simulate ERT, SRT, or TDEM data.

**Key examples**:

- :doc:`/auto_examples/Ex_ERT_workflow` - Complete ERT forward modeling
- :doc:`/auto_examples/EX_SRT_forward` - Seismic refraction forward modeling  
- :doc:`/auto_examples/Ex_TDEM_workflow` - TDEM forward modeling and inversion

**Steps**:

1. Load hydrological model output (water content, porosity)
2. Apply petrophysical model (Archie, Waxman-Smits, etc.)
3. Create mesh and survey geometry
4. Run forward modeling

.. code-block:: python

   from PyHydroGeophysX.model_output import MODFLOWWaterContent
   from PyHydroGeophysX.petrophysics import water_content_to_resistivity
   from PyHydroGeophysX.forward import ERTForwardModeling

   # 1. Load hydrology
   processor = MODFLOWWaterContent("modflow_dir", idomain)
   wc = processor.load_timestep(0)

   # 2. Convert to resistivity
   res = water_content_to_resistivity(wc, rho_sat=100, n=2.0, porosity=0.3)

   # 3 & 4. Forward model
   fwd = ERTForwardModeling(mesh, scheme)
   data = fwd.simulate(res)


Workflow 3: Time-Lapse ERT Inversion
------------------------------------

**Goal**: Invert time-series ERT data to monitor subsurface changes.

**When to use**: You have multiple ERT datasets at different times and want to track changes.

**Key examples**:

- :doc:`/auto_examples/Ex_Time_lapse_measurement` - Create synthetic time-lapse data
- :doc:`/auto_examples/Ex_TL_inversion` - Time-lapse inversion techniques

.. code-block:: python

   from PyHydroGeophysX.inversion import TimeLapseERTInversion

   inversion = TimeLapseERTInversion(
       data_files=["t0.dat", "t1.dat", "t2.dat"],
       measurement_times=[0, 30, 60],  # days
       lambda_val=50.0,  # Spatial regularization
       alpha=10.0        # Temporal regularization
   )
   result = inversion.run()


Workflow 4: Structure-Constrained Inversion
-------------------------------------------

**Goal**: Use seismic velocity interfaces to constrain ERT inversion.

**When to use**: You have seismic data that defines geological boundaries.

**Key examples**:

- :doc:`/auto_examples/Ex_SRT_inv` - SRT inversion and interface extraction
- :doc:`/auto_examples/Ex_Structure_resinv` - Structure-constrained ERT inversion
- :doc:`/auto_examples/Ex_structure_TLresinv` - Structure-constrained time-lapse inversion


Workflow 5: Uncertainty Quantification
--------------------------------------

**Goal**: Estimate uncertainty in water content from resistivity.

**When to use**: You need confidence bounds on hydrological interpretations.

**Key example**: :doc:`/auto_examples/Ex_MC_Hydro`


Advanced Topics
---------------

- :doc:`/auto_examples/Ex_3D_ERT_forward` - 3D ERT modeling with MODFLOW integration
- :doc:`/auto_examples/Ex_TDEM_workflow` - TDEM forward modeling and inversion
- :doc:`/agents/index` - Multi-Agent AI system for automated workflows


.. toctree::
   :hidden:

   /quickstart
   /auto_examples/Ex_model_output
   /auto_examples/Ex_ERT_data_process
   /auto_examples/Ex_ERT_workflow
   /auto_examples/EX_SRT_forward
   /auto_examples/Ex_Time_lapse_measurement
   /auto_examples/Ex_TL_inversion
   /auto_examples/Ex_SRT_inv
   /auto_examples/Ex_Structure_resinv
   /auto_examples/Ex_structure_TLresinv
   /auto_examples/Ex_MC_Hydro
