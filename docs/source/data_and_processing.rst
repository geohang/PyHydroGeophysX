Data and processing
===================

What the package can read, how it checks a dataset, and what it can invert.
This covers stages 2 and 3 of :doc:`the loop <index>`: turning an acquisition
into a dataset, and a dataset into a model.

Reading field data
------------------

Instrument formats are read directly, so a file off the instrument needs no
conversion step. Every reader returns the same contract, an electrode or
geometry array alongside a table of measurements, so the rest of the package
does not care which instrument produced the survey.

.. list-table::
   :header-rows: 1
   :widths: 26 24 50

   * - Method
     - Format
     - Notes
   * - ERT
     - DAS-1 ``.Data``
     - Self-describing. The column map is read from each file's own header
       rather than assumed, so a non-standard layout still loads.
   * - ERT
     - ``.tx0``
     - Lippmann 4-Point Light and GeoTest exports, including the explicit
       electrode table.
   * - ERT
     - Res2DInv ``.dat``
     - General-array format.
   * - ERT
     - ``.ohm``, ``.dat``
     - E4D and pyGIMLi/BERT files, plus ResIPy for processing.
   * - Seismic
     - SEG-2
     - Through ObsPy when it is installed, with a text fallback for arrays
       that were already exported.
   * - TDEM
     - TEM2Go project, ``.tiw`` or ``.db``
     - Station stacks, gate flags, errors, geometry and saved inversion
       settings. Both project layouts are detected automatically.
   * - TDEM
     - TEM2Go acquisition folder, ``.stb``
     - A folder copied straight off the instrument, with no project and no
       export. Gate times, calibration and loop geometry come from the raw
       stream itself.
   * - TDEM
     - ``_StationData.xyz``, ``_RawData.xyz``
     - Self-describing exports. Station stacks are preferred for inversion.
   * - TDEM
     - tTEM raw, ``.skb`` with ``.sps``
     - Uses the system GEX for geometry, waveform, gates and calibration, and
       the TFI import filter when both are present.
   * - Gravity, magnetics
     - Table formats
     - Column files with the survey geometry alongside.

Checking a dataset
------------------

Quality control runs before inversion rather than as an afterthought. Reciprocal
measurements give the error estimate, following the definition in the
induced-polarisation literature (Slater et al. 2000; Binley and Kemna 2005;
Binley and Slater 2020, chapter 6). Gate-level tests for electromagnetic data
apply the acquisition protocol's own thresholds, so a survey is filtered on the
terms it was recorded under.

The readers were written from the published file formats and from the standard
DC-resistivity literature.

Recovering a model
------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Inversion
     - What it is for
   * - Single survey
     - One acquisition, 2D or 3D, ERT or seismic refraction.
   * - Time-lapse
     - A monitoring series inverted together, so the recovered change is
       constrained rather than differenced after the fact.
   * - Windowed time-lapse
     - Long series inverted in overlapping windows, which keeps memory bounded
       on a survey that would not fit at once.
   * - Structure-constrained
     - Seismic structure guides a resistivity inversion where the two respond
       to the same boundary.
   * - Joint ERT and seismic
     - Complementary measurements inverted against a shared structure.
   * - Laterally constrained, EM
     - Neighbouring soundings tied along a line, with both moments fitted to
       one layered model.
   * - Gravity and magnetics
     - Potential-field inversion on the same mesh infrastructure.

Solvers and uncertainty
-----------------------

A Gauss-Newton normal matrix is symmetric positive definite by construction,
and the solvers treat it as one: ``spd_cholesky`` and ``spd_cg`` are available
alongside the least-squares methods, and ``generalized_solver`` reports a
mismatch between the method and the system it was handed.

An optional differentiable 2.5D ERT backend runs on the GPU through CuPy and
cuDSS. It falls back to the built-in engine when CUDA is unavailable, and the
engine that actually ran is reported, so a fallback is visible rather than
silent.

For uncertainty, the package provides a linearized posterior covariance, model
resolution, ensemble Kalman updating, and Monte Carlo propagation of
petrophysical parameter uncertainty into the recovered quantity.

Where to go next
----------------

- :doc:`tutorials/field_ert_data_qc` walks a field dataset through checking and
  export.
- :doc:`tutorials/time_lapse_monitoring` builds and inverts a monitoring series.
- :doc:`api/index` documents every function named above.
