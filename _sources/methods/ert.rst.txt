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
   * - Subsurface Insights ``.csv``
     - The ``results_processed_*`` export, read by column name across firmware
       versions. Electrode positions are rebuilt from the instrument's own
       geometric factors when they describe a straight, evenly spaced line;
       otherwise the surveyed electrode file supplies them.
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

E4D, PNNL's parallel 3-D ERT code, is an external engine (``engine="e4d"``),
for single surveys and for time-lapse series in E4D's own time-lapse mode. It
is not installed with the package and is built from source
(https://github.com/pnnl/E4D) with gfortran, PETSc and MPI. Where it runs:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Platform
     - E4D
     - How PyHydroGeophysX reaches it
   * - Linux (workstation, cluster node)
     - Supported; E4D's own platform
     - ``e4d`` and ``mpirun`` on PATH, ``PYHYDRO_E4D`` / ``PYHYDRO_MPIRUN``, or
       paths in the settings (``launcher="local"``)
   * - Windows
     - Only inside WSL 2; there is no Windows build
     - E4D built in the WSL Linux, ``launcher="wsl"``; PyHydroGeophysX stays on
       Windows
   * - macOS
     - When built from source (not in E4D's instructions; untested)
     - As on Linux
   * - Anywhere else, or a cluster with a scheduler
     - Elsewhere
     - ``launcher="files"`` writes the complete run folder; run
       ``mpirun -np N e4d`` in it, then read it back with
       ``inversion.e4d.read_e4d_run`` or ``read_e4d_time_lapse``

``python -m PyHydroGeophysX.inversion.e4d`` reports what the current machine
offers. E4D needs at least two MPI processes and no more workers than
electrodes. When it cannot run, the run writes its folder and stops with a
message giving the path; it never falls back to another engine.

R2 (2-D) and R3t (3-D), Andrew Binley's inversion codes and the ones ResIPy
runs, are external engines too (``engine="r2"``, ``engine="r3t"``), for single
surveys and for time-lapse series (their difference inversion against the
first survey). They are not installed with the package; they are free for
non-commercial use, and commercial use needs the author's permission
(http://www.es.lancs.ac.uk/people/amb/Freeware/R2/R2.htm). Where they run:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Platform
     - R2 / R3t
     - How PyHydroGeophysX reaches them
   * - Windows
     - Supported natively (``R2.exe``, ``R3t.exe``)
     - An installed ResIPy's copy, found without importing ResIPy; or the
       program in the settings, ``PYHYDRO_R2`` / ``PYHYDRO_R3T``, or PATH
   * - Linux
     - Through Wine, as their manuals and ResIPy run them
     - ``wine`` or ``wine64`` on PATH and the same ``.exe`` files
   * - macOS
     - Through Wine (untested here)
     - As on Linux
   * - Anywhere else
     - Elsewhere
     - ``launcher="files"`` writes the complete run folder; run the program in
       it, then read it back with ``inversion.r2.read_r2_run``

``python -m PyHydroGeophysX.inversion.r2 --program r2`` (or ``r3t``) reports
what the current machine offers. R2 inverts a profile on its own mesh; R3t a
3-D survey on its tetrahedral mesh, or a profile on a 3-D mesh around the line.
Both search their smoothing weight at every iteration (an Occam inversion), so
``lam`` and ``auto_lambda`` do not apply to them and the weight they settled on
is reported instead; they stop at an RMS misfit of ``sqrt(target_chi2)``. They
hold a fixed a-priori zone at its value.
