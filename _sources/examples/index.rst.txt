Examples by task
================

**Tutorials explain decisions; examples provide the complete scripts.**
Choose a task below, open its example page, then use that page’s Python or notebook
download. Run from a source checkout with the required example data and optional
engines installed; a rendered gallery page is not evidence that the script ran
on your machine. Check the paths and settings at the beginning of each script.

The organizing question
------------------------

**How do geophysical data and hydrological models inform one another?**
Read :doc:`/tutorials/hydro_geophysical_interaction` for the model–data connection,
then choose a direction below. Individual geophysical methods are tools within
that connection, not the endpoint of the learning route.

Recommended first route
-----------------------

**Run the model–data connection:** :doc:`/tutorials/modflow_feedback` is a small
MODFLOW comparison using supplied geophysical structure, with actual solver runs.

1. :doc:`/auto_examples/Ex_model_output`: understand the hydrological inputs.
2. :doc:`/auto_examples/Ex_ERT_workflow`: connect water content to a measurement.
3. :doc:`/auto_examples/Ex_hydro_to_multigeophys`: compare complementary methods.
4. Choose monitoring, joint inversion or uncertainty below for your question.

For field data, start directly with the single-method examples. For mouse-driven
processing and project maps, use :doc:`/agents/desktop_studio`.

Hydrology to geophysical responses
----------------------------------

Connect model states to measurements through petrophysics. Start here to understand the package’s hydrological focus.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Main outcome
   * - :doc:`Load MODFLOW / ParFlow outputs </auto_examples/Ex_model_output>`
     - Hydrological arrays for downstream modeling
   * - :doc:`Hydrology → ERT </auto_examples/Ex_ERT_workflow>`
     - Resistivity models and synthetic ERT responses
   * - :doc:`Hydrology → TDEM </auto_examples/Ex_TDEM_workflow>`
     - Layered time-domain EM modeling and inversion
   * - :doc:`Hydrology → FDEM </auto_examples/Ex_FDEM_workflow>`
     - Frequency-domain EM responses and recovered models
   * - :doc:`One profile → multiple methods </auto_examples/Ex_hydro_to_multigeophys>`
     - Compare ERT, seismic, EM and gravity responses from a shared hydrological profile
   * - :doc:`3D ERT with topography </auto_examples/Ex_3D_ERT_forward>`
     - Extend hydrology-driven forward modeling to a 3D mesh

Hydrological interpretation and uncertainty
-------------------------------------------

Use recovered geophysical properties to estimate hydrological quantities under explicit petrophysical assumptions. Quantify uncertainty before comparing these estimates with hydrological model states.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Main outcome
   * - :doc:`Geophysical structure → MODFLOW </auto_examples/Ex_MODFLOW_geophysics_feedback>`
     - Run and compare hydrological responses with uniform and interpreted interfaces
   * - :doc:`Monte Carlo hydrological interpretation </auto_examples/Ex_MC_Hydro>`
     - Water-content and porosity estimates with uncertainty from petrophysical parameters

Survey processing and single-method inversion
---------------------------------------------

Prepare trustworthy observations and physical-property models for the hydrological comparison. These are supporting workflows; a single-method inversion alone does not update a hydrological simulation.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Main outcome
   * - :doc:`Field ERT inversion </auto_examples/Ex_ERT_single_inversion>`
     - Recover a resistivity section from survey data
   * - :doc:`Seismic forward modeling </auto_examples/EX_SRT_forward>`
     - Generate travel-time data
   * - :doc:`Seismic inversion </auto_examples/Ex_SRT_inv>`
     - Recover a velocity section
   * - :doc:`Airborne EM line </auto_examples/Ex_EM_line_section>`
     - Calibrate and invert a line of soundings
   * - :doc:`Ground TEM: LM + HM </auto_examples/Ex_TEM_LMHM_LCI>`
     - Fit both moments with line constraints using the bundled nine-station project
   * - :doc:`Gravity and magnetics </auto_examples/Ex_gravity_magnetics_inversion>`
     - Inspect anomalies, forward model and run compact 3D inversion

Time-lapse monitoring
---------------------

Connect evolving hydrological states with repeated geophysical observations. Generate measurements from model states, recover temporal property changes, and inspect whether their timing and magnitude support the hydrological interpretation.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Main outcome
   * - :doc:`1. Generate monitoring data </auto_examples/Ex_Time_lapse_measurement>`
     - Synthetic ERT measurements at multiple times
   * - :doc:`2. Invert the time series </auto_examples/Ex_TL_inversion>`
     - Temporally constrained resistivity changes
   * - :doc:`3. Scale to longer series </auto_examples/Ex_TL_inversion_memory>`
     - Compare standard and memory-oriented inversion workflows
   * - :doc:`4. Add structural information </auto_examples/Ex_structure_TLresinv>`
     - Time-lapse inversion with geological constraints

Structural constraints and joint inversion
------------------------------------------

Use complementary information while keeping each method’s data fit visible.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Main outcome
   * - :doc:`Structure-constrained ERT </auto_examples/Ex_Structure_resinv>`
     - Use an existing structural interpretation to guide resistivity inversion
   * - :doc:`Joint ERT + seismic </auto_examples/Ex_joint_inversion>`
     - Recover coupled resistivity and velocity models with cross-gradient / geostatistical constraints

AI-assisted operation
---------------------

:doc:`/tutorials/agent_workflows` explains how to use agents with these scientific
workflows. :doc:`/agent_install` covers installation only; it is not an inversion
tutorial. Browser apps and example-data generators are launch / preparation tools,
not additional scientific examples.

Browse all figures
------------------

Use the :doc:`complete gallery </auto_examples/index>` for thumbnails and downloads.
The categories above cover every scientific Python example currently in the gallery.
