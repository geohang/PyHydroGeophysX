ATS and PFLOTRAN outputs to geophysics
======================================

Install the optional HDF5 reader dependency:

.. code-block:: bash

   pip install "pyhydrogeophysx[hydrology]"

These adapters implement the same ``HydroModelOutput`` methods as the existing
MODFLOW and ParFlow readers: ``load_timestep``, ``load_time_range`` and
``get_timestep_info``. They read visualization/snapshot output; the simulator
itself does not need to be installed. Model execution and parameter write-back
are outside these adapters' scope.

Reading a state
---------------

.. code-block:: python

   from PyHydroGeophysX import (
       ATSSaturation, ATSWaterContent,
       PFLOTRANSaturation, PFLOTRANWaterContent,
   )

   ats = ATSSaturation("outputs/ats")             # ats_vis_data.h5
   pflotran = PFLOTRANSaturation("outputs/pflotran", run_name="simulation")

   saturation = ats.load_timestep(0)        # (n_cells,)
   series = pflotran.load_time_range(0, 3)  # (n_times, *native_spatial_shape)
   print(ats.get_timestep_info(), ats.time_unit)
   print(pflotran.get_timestep_info(), pflotran.time_unit)

ATS supports fields grouped by numeric cycle, named as ATS names them: plain
(``saturation_liquid``), with a component suffix (``saturation_liquid.cell.0``)
or with the domain's prefix (``domain-saturation_liquid.cell.0``). The domain
is read from the filename - ``ats_vis_data.h5`` is the subsurface,
``ats_vis_surface_data.h5`` the surface. The ``Time`` attribute is read from
each cycle dataset. Auto-detection accepts exactly one of ``ats_vis_data.h5``,
``ats_vis_domain_data.h5`` or ``visdump_data.h5``. Explicit filenames handle
other output bases and domains. The file's ``time unit`` attribute is
preserved; if absent, ``time_unit=None`` unless you supply it explicitly.
Missing times are NaN, not cycle numbers.

PFLOTRAN supports ``Time: <number> <unit>`` snapshot groups across a single
``simulation.h5`` or the numbered files ``simulation-001.h5``,
``simulation-002.h5``, ... of its multiple-file output; another run whose
name begins the same way (``simulation-hires.h5``) is not taken in.
Snapshots are sorted numerically by time. Duplicate times and mixed time units
are rejected. Use ``filename="chosen_file.h5"`` to isolate a particular file
or restart. Neither reader converts time units.

``ATSPorosity`` and ``PFLOTRANPorosity`` read porosity. ``ATSWaterContent`` and
``PFLOTRANWaterContent`` compute **volumetric liquid water content** as
``porosity * liquid_saturation``. ATS's native ``water_content`` can include
molar density and cell volume, so it is not used for this calculation. Ice
content is not included. If porosity is not saved, supply a scalar or an array
matching the saturation shape and cell order:

.. code-block:: python

   water = ATSWaterContent("outputs/ats")
   theta = water.load_timestep(0, porosity=0.35)

Missing fields or cycles raise errors. Arrays keep their original values,
including NaN; they are not clipped. ATS scalar fields must have shape
``(n_cells,)`` or ``(n_cells, 1)``. PFLOTRAN retains HDF5 shape and axis
order - ``(nx, ny, nz)`` for a structured grid, as PFLOTRAN writes it; it is
not reshaped to ParFlow's ``(nz, ny, nx)`` convention.
For custom names, map canonical names to exact HDF5 keys:

.. code-block:: python

   water = ATSWaterContent(
       "outputs/ats", filename="custom_data.h5",
       variable_map={"saturation": "domain-saturation_liquid", "porosity": "domain-porosity"},
   )
   pressure = water.read_field("pressure", 0)

``ATSOutput`` and ``PFLOTRANOutput`` also expose ``read_field`` for other scalar
variables, including temperature, pressure or a species concentration using
its exact HDF5 name. Concentrations are returned in the original units.

Mapping to a geophysical mesh
-----------------------------

The source cell centres come from the simulator's own mesh, in exactly the
order of ``reader.load_timestep(i).ravel()``:

- ATS: the ``*_mesh.h5`` file it writes beside the data (``ats_vis_mesh.h5``
  for ``ats_vis_data.h5``); a deforming mesh is read at each timestep's cycle.
- PFLOTRAN: the ``Coordinates`` group of a structured grid (cell edges along
  x, y and z) or the ``Domain`` group of an unstructured one (vertices and
  XDMF cells).

``reader.output_cell_centers(i)`` returns them; ``interpolate_timestep`` uses
them whenever no ``cell_centers`` are given. An element's centre is the mean
of its nodes, as ATS's own tools compute it. ``axes`` picks the coordinates to
interpolate in - ``axes="xz"`` for a vertical ERT section of a run that is one
cell wide in y. Source and target coordinates must use the same coordinate
system, elevation datum and units; pass ``cell_centers`` yourself (in the
output's cell order) for a mesh exported elsewhere.

.. code-block:: python

   import numpy as np
   from pygimli.physics import ert
   from PyHydroGeophysX import ATSWaterContent
   from PyHydroGeophysX.inversion.ert_mesh import build_inversion_mesh
   from PyHydroGeophysX.petrophysics.resistivity_models import water_content_to_resistivity

   reader = ATSWaterContent("outputs/ats")             # ats_vis_data.h5 + ats_vis_mesh.h5
   survey = ert.load("survey.dat")                      # a profile in x and elevation
   mesh = build_inversion_mesh(survey, para_depth=10)
   target = np.asarray(mesh.cellCenters())[:, :2]       # (x, z) of every cell

   theta = reader.interpolate_timestep(0, target, axes="xz", method="linear")
   inside = np.isfinite(theta)                          # NaN outside the ATS domain
   resistivity = water_content_to_resistivity(theta[inside], rhos=100., n=2., porosity=0.35)

The same mapping works with ``PFLOTRANWaterContent``. ``linear`` interpolation
returns NaN outside the convex hull; ``nearest`` extrapolates explicitly.
Missing source values are excluded. Interpolation is not conservative and
does not respect material boundaries automatically. A 2D survey embedded in
a 3D domain should be sampled using its actual 3D cell coordinates, i.e.
target centres with all three coordinates and no ``axes``. The triangulation
is rebuilt on every call, which for a large 3-D mesh takes noticeable time.

The returned resistivity vector follows the target cell order and can be used
with the existing :doc:`hydrology_to_ert` forward-model setup. Other
petrophysical models can consume saturation, porosity and additional fields
to prepare seismic or EM properties in the same way.

Format references
-----------------

- `ATS visualization configuration <https://amanzi.github.io/ats/dev/input_spec/io/visualization.html>`_
- `ATS HDF5 postprocessing implementation <https://github.com/amanzi/ats/blob/master/tools/utils/ats_xdmf.py>`_
- `ATS liquid water content definition <https://amanzi.github.io/ats/dev/input_spec/process_kernels/physical/transport.html>`_
- `PFLOTRAN snapshot output configuration <https://documentation.pflotran.org/user_guide/cards/subsurface/output_card.html>`_
