Connecting geophysical data and hydrological models
===================================================

The central question is **what geophysical observations tell us about a
hydrological model, and what that model predicts we should measure**.
Petrophysical relationships and spatial mapping connect the two.

Two directions of information
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Direction
     - Information passed
     - Start with
   * - Hydrological model → geophysical observations
     - Map water content, saturation and porosity to physical properties;
       simulate measurements for a specified survey geometry.
     - :doc:`hydrology_to_ert`, :doc:`hydrology_to_tdem` and
       :doc:`/auto_examples/Ex_hydro_to_multigeophys`
   * - Geophysical observations → hydrological interpretation
     - Recover physical-property models, then estimate water content or
       porosity using explicit petrophysical assumptions and uncertainty.
     - :doc:`field_ert_data_qc` and :doc:`/auto_examples/Ex_MC_Hydro`
   * - Repeated observations ↔ changing model states
     - Compare predicted and recovered changes across aligned observation times.
     - :doc:`time_lapse_monitoring`
   * - Complementary surveys → structural information
     - Use velocity and resistivity information to interpret subsurface boundaries
       and evaluate structural assumptions relevant to hydrological modeling.
     - :doc:`seismic_structure_constraints` and :doc:`joint_inversion`

A practical comparison workflow
-------------------------------

1. **Define the hydrological question.** For example: can the proposed change in
   water content explain a measured change in resistivity?
2. **Prepare the model states.** Read the relevant MODFLOW / ParFlow outputs,
   preserve their time ordering and map them onto the geophysical mesh or profile.
3. **Specify the petrophysical link.** Record porosity, fluid properties and
   constitutive parameters. Different choices can explain similar measurements.
4. **Predict measurements.** Use the actual survey geometry and matching units.
   Compare predicted data with observed data, rather than comparing only images.
5. **Interpret the observations.** Inspect inversion fit and sensitivity before
   converting a recovered property model to hydrological quantities.
6. **Evaluate the hydrological hypothesis.** Compare states at compatible spatial
   and temporal scales, propagate relevant uncertainty, and decide whether the
   model assumptions need revisiting.

What the examples demonstrate
------------------------------

The forward examples demonstrate the model-to-measurement connection. The
Monte Carlo example demonstrates petrophysical interpretation and parameter
uncertainty. Time-lapse and joint examples supply additional observational
constraints. These are complementary steps: running a geophysical inversion
does **not** by itself calibrate or update a MODFLOW / ParFlow simulation.
An automatic hydrological parameter-update loop requires a separately configured
calibration or assimilation workflow; it is not implied by these examples.
For the explicit write-back step, see :doc:`hydrological_input_updates`: export
mapped parameters or initial states to MODFLOW 6 and ParFlow input files, then
run and evaluate the hydrological simulation separately.

Recommended reading route
-------------------------

:doc:`/auto_examples/Ex_model_output` →
:doc:`/auto_examples/Ex_ERT_workflow` →
:doc:`/auto_examples/Ex_Time_lapse_measurement` →
:doc:`/auto_examples/Ex_TL_inversion` →
:doc:`/auto_examples/Ex_MC_Hydro`.

These examples illustrate successive concepts; check each example's dataset,
mesh and output format before transferring files between them. They are not
an automatically connected pipeline.
