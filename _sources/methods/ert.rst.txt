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

External engines: E4D, R2 and R3t
---------------------------------

These programs are not installed with PyHydroGeophysX. Their engines write the
files each program reads, run it, and read its results back, so QC, the error
model, the geometric-factor check, outlier rejection and every viewer and
export work as with the other engines.

.. _ert-e4d:

E4D
~~~

E4D, PNNL's parallel 3-D ERT code (BSD-style licence), is an external engine
(``engine="e4d"``), for single surveys and for time-lapse series in E4D's own
time-lapse mode. It is a Fortran/MPI program built from source
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

E4D needs at least two MPI processes (one master, one or more workers) and no
more workers than electrodes. ``PYHYDRO_E4D_COMMAND`` replaces
``mpirun -np N e4d`` with any command line (``{processes}`` is filled in), for
``srun`` or a container. When E4D cannot run, the run writes its folder and
stops with a message giving the path; it never falls back to another engine.
To see what the current machine offers:

.. code-block:: bash

   python -m PyHydroGeophysX.inversion.e4d

.. code-block:: python

   from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion

   result = run_ert_manager_inversion(
       "survey.dat", "output", engine="e4d", lam=20,
       e4d={"launcher": "auto", "processes": 8},   # or "local", "wsl", "files"
   )
   print(result["e4d"]["run_dir"])   # E4D's e4d.log, sigma.N and resistivity_3d.vtk

E4D models in 3-D. A 3-D survey on an imported tetrahedral mesh is inverted on
that mesh. A 2-D profile is inverted on a 3-D mesh built around the line the
way E4D users build one (a fine zone under the electrodes, reaching half the
inverted depth to either side, inside an outer zone reaching four electrode
spreads beyond it; the ground along the line carried across it), and the
section along the line is read back onto the profile's own mesh. Each run is
E4D's mode ERT3 at a fixed λ (E4D's β), which the λ search can choose:
nearest-neighbour smoothness, update option 3 so it stops once the objective
stops falling, and stopped after the iteration limit, since E4D has none of its
own. E4D's χ² is that of the transfer resistances, with a standard deviation of
``err·|R|``.

A time-lapse series runs as E4D's own time-lapse inversion (ERT4) through
``run_timelapse_ert(..., {"engine": "e4d", ...})`` or the Time-lapse panel: the
first survey is the baseline, each later one starts from the solution before
it, and the smoothness acts on the change from it. The measurements common to
every survey are inverted, as E4D requires the same ones in each. The temporal
weight α, the interval weighting and auto-λ belong to the in-house joint
inversion and are not used; the log says so.

.. _ert-r2-r3t:

R2 and R3t
~~~~~~~~~~

R2 (2-D profiles) and R3t (3-D surveys) are Andrew Binley's finite-element
inversion programs (Lancaster University) and the ones ResIPy runs. They are
external engines too (``engine="r2"``, ``engine="r3t"``), for single surveys
and for time-lapse series (their difference inversion against the first
survey). They are free for non-commercial use, and commercial use needs the
author's permission (http://www.es.lancs.ac.uk/people/amb/Freeware/R2/R2.htm).
Where they run:

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

To see what the current machine offers:

.. code-block:: bash

   python -m PyHydroGeophysX.inversion.r2 --program r2    # or r3t

.. code-block:: python

   result = run_ert_manager_inversion(
       "survey.dat", "output", engine="r2",          # or "r3t"
       r2={"launcher": "auto"},                       # or {"executable": r"C:\R2\R2.exe"}
   )
   print(result["r2"]["run_dir"], result["r2"]["alpha"])   # R2.out, f001_res.dat; its alpha

R2 inverts a profile on the profile's own mesh, every element a parameter. R3t
inverts a 3-D survey on its tetrahedral mesh, and a profile on a 3-D mesh built
around the line as for E4D, read back as the section along it. That is far
slower than R2 (the 72-electrode BERT example took about 16 minutes for ten
iterations on 44 000 elements, R2 seconds), and the ground off the line is held
only by the smoothness, so R2 is the engine for profiles. R3t leaves out of its
fit any reading it cannot use (no signal over uniform ground, with M and N
symmetric about A and B, or a polarity its model does not reproduce); the log
says how many, χ² is over the readings it fitted, and a forward run gives the
others' predictions.

Both are Occam inversions that search their smoothing weight α at every
iteration themselves, so ``lam`` and ``auto_lambda`` do not apply: the run
reports the α it settled on instead. They stop at an RMS misfit of
``sqrt(target_chi2)``, when the misfit stops improving, or at the iteration
limit. The data are transfer resistances with their own standard deviations
``err·|R|``, inverted as logarithms, so their RMS² is the same χ² the other
engines report; their own reweighting is off because the pipeline rejects
outliers itself. A fixed a-priori zone is held at its value (R2's
``param = 0``), the zone outlines become R2/R3t zones that the smoothness does
not cross, and a remote electrode is placed on the mesh node farthest from the
array.

A time-lapse series is their difference inversion (LaBrecque and Yang, 2001):
the first survey is inverted, then every later one from that model with the
smoothness on the change from it, using the readings it shares with the first.
R2 forms the difference data ``d − d0 + f(m0)`` itself; for R3t they are
written from the baseline's predicted data, as its manual describes.
