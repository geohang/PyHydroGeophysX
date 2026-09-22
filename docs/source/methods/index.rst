Methods
=======

A method is a tool inside a workflow, not the workflow itself. If you already
know which measurement or which hydrological model you are working with, start
from its page below. If you are choosing, start from
:doc:`Workflows </tutorials/index>` instead and let the scientific question pick
the method.

Every method page follows the same outline: what you can do, what it needs as
input, what it produces, which workflow to open, whether the desktop
application supports it, the Python examples, and the optional dependencies it
requires.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: phgx-cards

   .. grid-item-card:: ERT
      :link: ert
      :link-type: doc

      Electrical resistivity tomography: forward modelling, single-survey and
      time-lapse inversion, structural constraints, field-data QC.

   .. grid-item-card:: Seismic refraction
      :link: seismic
      :link-type: doc

      Travel-time tomography for velocity structure, and the structural
      information it contributes to a resistivity inversion.

   .. grid-item-card:: TDEM and FDEM
      :link: em
      :link-type: doc

      Time- and frequency-domain electromagnetics: layered forward modelling,
      laterally constrained inversion along a line.

   .. grid-item-card:: Gravity and magnetics
      :link: potential_fields
      :link-type: doc

      Potential-field anomalies, forward modelling of bodies, and compact 3D
      inversion.

   .. grid-item-card:: MODFLOW and ParFlow
      :link: hydrological_models
      :link-type: doc

      The hydrological end: read model states, predict what a survey would
      record, and write an interpreted result back as model input.

What each geophysical method needs and gives back
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 26 22

   * - Method
     - Property it responds to
     - Recovers
     - Hydrological use
   * - ERT
     - Electrical resistivity
     - 2D or 3D resistivity section
     - Water content and saturation through a petrophysical relationship
   * - Seismic refraction
     - P-wave velocity
     - 2D velocity section
     - Weathering and bedrock boundaries, porosity
   * - TDEM / FDEM
     - Electrical conductivity
     - Layered conductivity soundings along a line
     - Depth to conductive layers over larger areas than ERT covers
   * - Gravity / magnetics
     - Density, magnetic susceptibility
     - 3D property distribution
     - Basin and structural geometry

:doc:`MODFLOW and ParFlow <hydrological_models>` sit on the other side of that
table. They supply the water content, saturation and porosity the first column
is predicted from, and they are where an interpreted result is written back.

One call across methods
-----------------------

``GeophysicalInversion`` dispatches to whichever method you name, so a script
can switch method without switching API.

.. code-block:: python

   from PyHydroGeophysX.inversion import GeophysicalInversion

   inv = GeophysicalInversion("srt", data_file="examples/data/Seismic/srtfieldline2.dat")
   result = inv.run()

   # Also accepts: "ert", "tdem", "fdem", "joint_ert_srt"

Comparing methods over the same subsurface
------------------------------------------

:doc:`/auto_examples/Ex_hydro_to_multigeophys` takes one hydrological profile
and predicts what ERT, seismic, EM and gravity would each record over it, which
is the fastest way to see what a given method can and cannot resolve for your
site.

.. toctree::
   :maxdepth: 1
   :hidden:

   ert
   seismic
   em
   potential_fields
   hydrological_models
