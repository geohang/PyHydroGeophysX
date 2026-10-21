# Third-party code, data and desktop distributions

PyHydroGeophysX's own code is licensed under Apache-2.0; see LICENSE.
Third-party packages and data keep their own licences, set out below.

## Geophysics-informed MODFLOW example data

`examples/data/modflow_informed/structure.npz` repackages four text arrays from
Hang Chen's [Geophysics_informed_models](https://github.com/geohang/Geophysics_informed_models),
revision `a23fff3c00c0033f479064cf3651e23b1d5bea07`, under Apache-2.0.
The original license is retained beside the data; `provenance.json` records paths,
hashes and the lossless repackaging. The new example follows the S4 notebook's
interface construction but uses a simplified, illustrative groundwater model.
The upstream project acknowledges the USGS MODFLOW Sagehen example; its complex
SFR/UZF/MVR code is not copied into this compact example.

## E4D

PyHydroGeophysX does not include or redistribute E4D. The `e4d` inversion
engine writes E4D's input files and runs an E4D program the user has built
from [pnnl/E4D](https://github.com/pnnl/E4D) (Copyright 2014, Battelle Memorial
Institute; a BSD-style licence whose terms ask publications based on work
performed with E4D to cite Johnson et al., 2010, or Johnson and Wellman, 2015).
E4D's own build pulls in PETSc, MPI, Triangle and TetGen under their licences.

## R2 and R3t

PyHydroGeophysX does not include or redistribute R2 or R3t. The `r2` and `r3t`
inversion engines write the input files these programs read and run an
`R2.exe` / `R3t.exe` the user already has: the copy inside a ResIPy
installation, or one downloaded from Andrew Binley's page
(http://www.es.lancs.ac.uk/people/amb/Freeware/R2/R2.htm). Their manuals state
that R2 and R3t are provided for non-commercial use and that commercial users
should contact the author (Andrew Binley, Lancaster University); those terms
apply to every run, whichever program launched it. Locating ResIPy's copy does
not import ResIPy.

## TetGen

PyHydroGeophysX does not include or redistribute TetGen. E4D-style meshes
(`core.e4d_mesh`) run a `tetgen` program found on the machine, or the optional
`tetgen` Python package (`pip install tetgen`), whose wrapper is MIT-licensed
but whose compiled-in TetGen 1.6 library is AGPL-3.0. Without either, the same
geometry is meshed with Gmsh.

## Gmsh

The meshing fallback mentioned above uses Gmsh, which is licensed under
GPL-2.0-or-later with an upstream exception for specified third-party projects.
That exception is not a general permission to combine Gmsh with arbitrary
software under any licence. See the
[Gmsh licence](https://gmsh.info/LICENSE.txt).

Gmsh's terms apply independently of the Apache-2.0 licence on PyHydroGeophysX's
own code. Any distribution of its executable, Python API or native libraries
requires a review of the actual integration and applicable source-code and
redistribution obligations. This notice does not establish that Gmsh is bundled;
that must be determined from the individual archive and its build records.

## geoana

`forward/_tdem_kernels.py` holds numba versions of the TE reflection
coefficient of a layered earth and its conductivity gradient, `_rTE_forward`
and `_rTE_gradient` in [geoana](https://github.com/simpeg/geoana)
(`geoana/kernels/tranverse_electric_reflections.py`), Copyright (c) 2017 SimPEG
Team, MIT License. The licence text is kept at the top of that file, so it
travels with the code in every distribution, including the wheel, which does
not carry this document.

## Desktop builds

The desktop studio is built with `packaging/pyinstaller_studio.spec` in two
variants. **Light** contains the package, its Qt interface (PySide6, pyqtgraph,
qtawesome) and its scientific Python dependencies. **Full** adds pyGIMLi,
SimPEG (with discretize, pymatsolver and geoana) and PyVista/VTK.

The requirements below apply to official desktop releases. They describe the
release policy, not a completed audit of a particular archive. Before
publication, each archive must carry these items at its root:

* `LICENSE` and this document;
* `LICENSES/`, containing the applicable third-party licence texts, copyright
  notices and required attributions;
* `BUILD_INFO.txt`, identifying the platform, architecture, application version,
  actual source commit and build tools, and listing the bundled components,
  their versions or build identifiers, licences and notice-file locations;
* `SOURCE_CODE.md`, giving release-specific source downloads and instructions
  for rebuilding and replacing the LGPL-covered libraries.

The component inventory must describe the files actually distributed, including
native libraries, plugins, QML files and bundled assets. Components that are
only installed in the build environment must not be presented as bundled.

### Optional and excluded components

The standard release configuration excludes ResIPy and the optional `tetgen`
Python package, including its compiled-in TetGen library. The standalone E4D,
R2, R3t and TetGen programs described above remain user-supplied tools.

`PHGX_BUNDLE_RESIPY=1` and `PHGX_BUNDLE_TETGEN=1` are custom-build options, not
approved official-release configurations. Enabling either option does not
change the licences of the components or, by itself, satisfy their distribution
requirements. Any such build requires a separate review of the applicable
licences, source-code obligations and licence compatibility before redistribution.

Gmsh requires a separate review as well. Its Python API, native libraries and
executable must not be included in an official archive until that review is
complete. The Gmsh fallback described above does not establish whether Gmsh is
present in any particular desktop archive.

### Qt, PySide6 and Shiboken6

The official desktop interface uses LGPLv3-licensed Qt and PySide6 components,
including the Shiboken6 runtime. Their copyright notices must be retained.
`LICENSES/` must include the unmodified GNU LGPL v3 and GNU GPL v3 texts as
`LGPL-3.0-only.txt` and `GPL-3.0-only.txt`. LGPLv3 incorporates GPLv3 with
additional permissions; including both texts does not, by itself, change the
licence of PyHydroGeophysX's own code.

Release builds must exclude Qt components whose available open-source licence
is GPL-only. This includes Qt Charts, Qt Data Visualization, Qt Graphs, Qt Quick
3D and Qt Virtual Keyboard; this list is not exhaustive. The review must use
the licensing information for the actual Qt version in each archive and cover
bindings, native libraries, plugins, QML files and auxiliary tools. Qt's
[licensing overview](https://doc.qt.io/qt-6/licensing.html) provides the starting
point; the version-specific component notices govern the distributed files.

The studio uses a directory-based build intended to keep the Qt, PySide6 and
Shiboken6 shared libraries replaceable. This layout alone does not establish
compliance. The released application must support replacement with
interface-compatible modified libraries. Users may modify the LGPL-covered
libraries and reverse engineer the combined work for debugging those
modifications, as permitted by LGPLv3. PyHydroGeophysX imposes no additional
restriction on those rights.

`SOURCE_CODE.md` must identify the corresponding source for the actual LGPL
binaries, including applicable downstream patches and the build information
needed for those libraries. It must also give platform-specific replacement
instructions and identify the application source commit used for the archive.
For downloadable releases, complete corresponding library source must be
available at no additional charge through clearly identified release-specific
downloads alongside, or clearly linked from, the binary downloads. The
redistributor remains responsible for the required source availability.
Generic upstream download or repository homepages are not a substitute for
identifying and providing the corresponding source.

The governing texts are the
[GNU LGPL v3](https://www.gnu.org/licenses/lgpl-3.0.html) and the incorporated
[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.html). The Qt project also
publishes [LGPL guidance](https://www.qt.io/development/open-source-lgpl-obligations).

### Code and assets included by dependencies

Qt, PySide6 and the scientific Python packages can include separately licensed
code and assets. Required notices must be collected for the actual bundled
versions, including embedded or statically linked dependencies and any fonts
or icons. A package's top-level licence does not replace its dependencies'
licences. The component inventory must map these notices to their locations in
`LICENSES/`.

Qt documents these components in
[Third-Party Code Used in Qt](https://doc.qt.io/qt-6/licenses-used-in-qt.html)
and [Licenses Used in Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html).
Use the versions corresponding to the distributed binaries. Links to these
pages supplement, and do not replace, licence texts and notices that must
accompany the distribution.

### Windows runtime and software rendering

Windows archives may include Microsoft Visual C++ runtime DLLs, such as
`MSVCP140*.dll` and `VCRUNTIME140*.dll`. These remain subject to the applicable
Microsoft terms, not PyHydroGeophysX's Apache-2.0 licence. Their exact versions,
origins and applicable redistribution terms must be recorded before release;
their presence in an upstream package is not, by itself, a redistribution
assessment. Microsoft's
[redistribution documentation](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files)
is a starting point for that review.

When the archive includes Qt's `opengl32sw.dll` software renderer, the notices
for that particular Mesa llvmpipe build and its dependencies must be retained
in `LICENSES/`. Qt identifies its distributed renderer under the MIT License
and Boost Software License 1.0 in its
[Mesa llvmpipe attribution](https://doc.qt.io/qt-6/qt-attribution-llvmpipe.html).

### Full builds: pyGIMLi, Triangle and visualization libraries

pyGIMLi's 2-D mesh generator wraps Jonathan Shewchuk's Triangle. Triangle has
its own redistribution conditions, independent of pyGIMLi's top-level licence.
Its copyright notices must be retained, and sale or inclusion in commercial
products requires the author's permission. Modified copies must also satisfy
the conditions in the original notice for modifications and source/object-code
distribution. The exact Triangle notice accompanying the bundled version must
be included in `LICENSES/`; this summary does not replace it. See the
[author's Triangle page](https://www.cs.cmu.edu/~quake/triangle.html).

PyVista and VTK also retain their own licences. The published projects use MIT
and BSD-3-Clause, respectively; the notices for the actual distributed versions
and their bundled dependencies must be preserved. See
[PyVista's licence](https://github.com/pyvista/pyvista/blob/main/LICENSE)
and [VTK's licensing information](https://docs.vtk.org/en/latest/about.html).

Full builds require an inventory of their additional native dependencies as
well as their Python packages. They must not be published until the applicable
notices, source obligations and redistribution conditions have been checked.

### Full builds: rustsolve in place of CHOLMOD and UMFPACK

pgcore, pyGIMLi's compiled core, solves sparse systems with SuiteSparse CHOLMOD
and UMFPACK. UMFPACK, and CHOLMOD's Supernodal, Modify and MatrixOps modules,
are GPL-2.0-or-later. In Full builds these are replaced by rustsolve, a
sparse-solver library on faer. The prebuilt files are in `packaging/rustsolve/`;
its `README.txt` gives their origin and SHA-256. The build installs them with
`packaging/rustsolve/apply.py`:

* **rustsolve's C interface** (`rustsolve_capi.dll`, `librustsolve_capi.dylib`).
  rustsolve 0.1.0 is MIT OR Apache-2.0. It is built from
  https://github.com/geohang/rustsolve, a private repository. The Rust crates
  compiled with it are listed in `packaging/rustsolve/README.txt`: faer 0.24.4
  and others under MIT, Apache-2.0, BSD-2-Clause, Zlib or Unlicense terms.
  faer's licence files also cover code ported into faer from SuiteSparse AMD
  and COLAMD (BSD-3-Clause), LAPACK (BSD) and Eigen (MPL-2.0). The
  Eigen-derived code, a divide-and-conquer SVD, is not compiled into these
  files. The Rust standard library is compiled in. The Windows file also
  contains MinGW-w64 runtime code and GCC's unwinder (GPL-3.0-or-later with
  the GCC Runtime Library Exception 3.1).
* **libgimli, modified** (`libgimli.dll`, `libgimli.dylib`). This is pyGIMLi's
  core library (Apache-2.0), rebuilt from pyGIMLi's sources with rustsolve's
  changes:
  * Windows: commit 5991b6ac, the source of pgcore 1.5.0;
  * macOS: v1.5.4, the source of pgcore 1.5.4.

  As section 4(b) of the Apache License asks, the archive's
  `pygimli/MODIFICATIONS.txt` names the changed files. It is in the
  component's folder in `LICENSES/`. The library's version string ends in
  `+rustsolve`. It still contains Triangle, as pyGIMLi's own build does.
* On Windows, the seven SuiteSparse DLLs carried by pgcore 1.5.0 are left
  out: AMD, CAMD, CCOLAMD, CHOLMOD, COLAMD, SuiteSparse_config and UMFPACK.

`packaging/release_audit.py` identifies these files by their SHA-256. It lists
them in `BUILD_INFO.txt` as their own components, with their licence texts
and notices in `LICENSES/`.

## Example data

`examples/data/README.md` lists every example dataset with its source and
licence, and each dataset keeps its attribution beside the files. Data taken
from public releases include:

* `examples/data/EM/skytem_bhmar_*.csv`: five soundings of Brodie, R.C. (2019),
  *Broken Hill Managed Aquifer Recharge (BHMAR) SkyTEM Airborne Electromagnetic
  Survey, NSW, 2009*, Geoscience Australia,
  https://doi.org/10.26186/5d314c5de8040, under CC BY 4.0; the subset was taken
  from the example in Geoscience Australia's ga-aem repository.
* `examples/data/EM/EastRiver_VTEM/`: a subset of Zamudio, K.D., Minsley, B.J.
  and Ball, L.B. (2021), *Airborne electromagnetic, magnetic, and radiometric
  survey, upper East River and surrounding watersheds near Crested Butte,
  Colorado, 2017*, U.S. Geological Survey data release,
  https://doi.org/10.5066/P949ZCZ8, a U.S. Government work in the public domain.
* `examples/data/ERT/E4D/`: ten ERT acquisitions (October 2021 to June 2022)
  from Uhlemann, S., Dafflon, B., Chou, C., Wielandt, S., Wu, Y. and
  Williams, K. (2024), *Geophysical and Environmental Monitoring Data, and
  Subsurface Flow Modelling Results for Chicken Bone Meadow, Mt. Snodgrass,
  Crested Butte, CO*, ESS-DIVE, https://doi.org/10.15485/2283406, under
  CC BY 4.0.
* `examples/data/ERT/DAS/`: five DAS-1 acquisitions (5 to 9 November 2017) from
  the time-lapse ERT monitoring of the No-Name hillslope, Snowy Range,
  Wyoming, by Andrew Parsekian's group at the University of Wyoming. The group
  publishes this monitoring series as Parsekian, A. (2020),
  *NoName_Snowy_Range_MBNF_Hydrogeophysics*, HydroShare,
  http://www.hydroshare.org/resource/3357f6ea99ca48a4b418a0be14fb653b, under
  CC BY 4.0.
* `examples/data/climate/`: Daymet Version 4 R1 daily weather at one point,
  https://doi.org/10.3334/ORNLDAAC/2129, which the ORNL DAAC asks users to cite.
