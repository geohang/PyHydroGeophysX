ERT
===

Electrical resistivity tomography. Of the methods here it is the one most
directly tied to water content, which is why most of the hydrological workflows
run through it.

What you can do
---------------

- Forward modelling from a resistivity model, in 2D and 3D, with topography.
- Single-survey inversion.
- Time-lapse inversion of a monitoring series, including a windowed variant for
  series too long to invert at once.
- Structure-constrained inversion, where seismic structure guides the result.
- Joint ERT and seismic inversion against a shared structure.
- Field-data quality control with reciprocal error estimates.

Required inputs
---------------

Either a resistivity model on a mesh together with an electrode layout, for
forward modelling, or a measured dataset in one of the formats below. For
hydrology-driven work you supply water content and porosity instead, and the
petrophysical relationship produces the resistivity model.

The readers return an electrode array alongside a table of measurements, so
nothing downstream depends on which instrument wrote the file.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Format
     - Notes
   * - DAS-1 ``.Data``
     - Self-describing; the column map is read from each file's own header.
   * - ``.tx0``
     - Lippmann 4-Point Light and GeoTest exports, with the electrode table.
   * - Res2DInv ``.dat``
     - General-array format.
   * - AGI SuperSting ``.stg``
     - SuperSting and Sting R1. The electrode table is rebuilt from the records,
       and the instrument's own geometric factor is carried alongside so a
       nominal-versus-surveyed geometry mismatch becomes visible.
   * - ``.ohm``, ``.dat``
     - E4D and pyGIMLi/BERT files. ResIPy adds further instrument formats when
       it is installed.

:doc:`Data and processing </data_and_processing>` covers the readers and the
quality-control tests in full.

Typical outputs
---------------

A resistivity model on the inversion mesh, the data fit that produced it, and
for a monitoring series one model per acquisition time. Estimated water content
follows from the recovered resistivity through the petrophysical relationship,
with the parameter uncertainty propagated alongside it.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Your goal
     - Start here
   * - Predict ERT data from a hydrological model
     - :doc:`/tutorials/hydrology_to_ert`
   * - Check and prepare field measurements
     - :doc:`/tutorials/field_ert_data_qc`
   * - Invert one survey
     - :doc:`/auto_examples/Ex_ERT_single_inversion`
   * - Analyse repeated surveys
     - :doc:`/tutorials/time_lapse_monitoring`
   * - Add structural constraints
     - :doc:`/tutorials/seismic_structure_constraints`
   * - Combine ERT and seismic
     - :doc:`/tutorials/joint_inversion`
   * - Estimate water content from a recovered section
     - :doc:`/auto_examples/Ex_MC_Hydro`

Desktop support
---------------

:doc:`Desktop Studio </agents/desktop_studio>` has an ERT module covering
import, quality control, mesh generation, inversion and display, including the
time-lapse series.

Python examples
---------------

- :doc:`/auto_examples/Ex_ERT_workflow`: hydrology to a resistivity model to a
  simulated survey.
- :doc:`/auto_examples/Ex_ERT_single_inversion`: recover a section from field
  data.
- :doc:`/auto_examples/Ex_3D_ERT_forward`: forward modelling on a 3D mesh with
  topography.
- :doc:`/auto_examples/Ex_Time_lapse_measurement` and
  :doc:`/auto_examples/Ex_TL_inversion`: generate and invert a monitoring
  series.
- :doc:`/auto_examples/Ex_TL_inversion_memory`: the windowed variant for long
  series.
- :doc:`/auto_examples/Ex_Structure_resinv` and
  :doc:`/auto_examples/Ex_structure_TLresinv`: structural constraints, single
  and time-lapse.
- :doc:`/auto_examples/Ex_joint_inversion`: joint ERT and seismic.

A single inversion, in the shortest form it takes:

.. code-block:: python

   from PyHydroGeophysX.inversion import ERTInversion

   inv = ERTInversion(data_file="examples/data/ERT/Bert/fielddataline2.dat")
   result = inv.run()
   print("cells:", result.final_model.size, "final chi2:", result.iteration_chi2[-1])

Relevant API
------------

- :doc:`/api/forward` for ``ERTForwardModeling``.
- :doc:`/api/inversion` for ``ERTInversion``, ``TimeLapseERTInversion``,
  ``WindowedTimeLapseERTInversion``, ``StructuralConstraint`` and
  ``JointERTSRTInversion``.
- :doc:`/api/data_processing` for the readers and quality-control functions.
- :doc:`/api/petrophysics` for the resistivity relationships.

Limitations and optional dependencies
-------------------------------------

Forward modelling and inversion run on pyGIMLi, which the ``geophysics`` extra
installs:

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

ResIPy is a separate optional install that widens the set of readable
instrument formats. A differentiable 2.5D backend, selected with
``engine="adtlert"``, runs on the GPU and needs Python 3.11 or newer; it falls
back to the built-in engine when CUDA is unavailable, and the engine that
actually ran is reported. See :doc:`/installation` for both.
