Workflows
=========

Tutorials explain **what to prepare, which decisions to make and how to inspect
the result**. For complete scripts and figures, use :doc:`/examples/index`.
Install first using :doc:`/installation`; short API snippets below individual
tutorials are building blocks, not substitutes for their full examples.

Start with the interaction
--------------------------

Read :doc:`hydro_geophysical_interaction` first: **hydrological models predict
geophysical observations; geophysical data constrain hydrological interpretation**.
Use the tutorials below to build the relevant direction of that connection.

Start with your data
--------------------

.. list-table::
   :header-rows: 1
   :widths: 28 37 35

   * - You have
     - Start here
     - Continue with
   * - MODFLOW / ParFlow outputs
     - :doc:`hydrology_to_ert` or :doc:`hydrology_to_tdem`
     - Compare methods from one hydrological profile in :doc:`/auto_examples/Ex_hydro_to_multigeophys`
   * - A field ERT survey
     - :doc:`field_ert_data_qc`
     - :doc:`/auto_examples/Ex_ERT_single_inversion`
   * - Repeated ERT surveys
     - :doc:`time_lapse_monitoring`
     - Inspect temporal changes and compare inversion strategies
   * - ERT and seismic information
     - :doc:`seismic_structure_constraints`
     - :doc:`joint_inversion` when both datasets should be inverted together
   * - Recovered resistivity and petrophysical information
     - :doc:`/auto_examples/Ex_MC_Hydro`
     - Estimate hydrological properties and assess parameter uncertainty

What connects these workflows
-----------------------------

- **Hydrology → petrophysics → measurements:** carry water-content and porosity
  information into geophysical forward models.
- **Complementary methods:** compare responses to the same subsurface, then use
  structural or joint constraints where justified.
- **Monitoring:** recover change through time with temporal constraints.
- **Hydrological interpretation:** examine how petrophysical uncertainty affects
  the quantities inferred from geophysical models.

Use :doc:`agent_workflows` for AI-assisted operation or
:doc:`/agents/desktop_studio` for the Qt interface. These are ways to operate
the workflows, alongside Python scripts.

If you would rather start from the measurement than from the question, the
:doc:`method pages </methods/index>` list what each one needs as input and which
workflow it feeds.

.. toctree::
   :caption: Model–data interaction
   :maxdepth: 1

   hydro_geophysical_interaction
   hydrological_input_updates
   modflow_feedback
   hydrology_to_ert
   hydrology_to_tdem

.. toctree::
   :caption: Measurements and monitoring
   :maxdepth: 1

   field_ert_data_qc
   time_lapse_monitoring

.. toctree::
   :caption: Combining methods
   :maxdepth: 1

   seismic_structure_constraints
   joint_inversion

.. toctree::
   :caption: AI-assisted operation
   :maxdepth: 1

   agent_workflows
