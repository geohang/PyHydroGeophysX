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

## ResIPy

[ResIPy](https://gitlab.com/hkex/resipy) is licensed under GPL-3.0. It is an
optional dependency (`pip install "PyHydroGeophysX[ert]"`): the wheel and the
desktop archives do not contain it, and PyHydroGeophysX imports it only when
the user has installed it. When it imports, `data_processing/ert_data_agent.py`
defines `_FastSurvey`, a subclass of ResIPy's `Survey`, and reads ERT files
through ResIPy. Three layout details of the BGS PRIME reader in
`data_processing/ert_formats.py` are handled as ResIPy's `primeParserTab`
expects them (Blanchy et al., 2020); `parse_bgs_prime` names them.

## TetGen

PyHydroGeophysX does not include or redistribute TetGen. Meshes from E4D
configurations (`core.e4d_mesh`) run a `tetgen` program found on the machine, or the optional
`tetgen` Python package (`pip install tetgen`), whose wrapper is MIT-licensed
but whose compiled-in TetGen 1.6 library is AGPL-3.0. Without either, the same
geometry is meshed with Gmsh.

## Gmsh

The meshing fallback mentioned above runs a `gmsh` program found on the
machine. PyHydroGeophysX does not include or redistribute Gmsh and does not
import its Python API. Gmsh is licensed under GPL-2.0-or-later with an upstream
exception for specified third-party projects; the exception is not a general
permission to combine Gmsh with software under any licence. See the
[Gmsh licence](https://gmsh.info/LICENSE.txt). Gmsh's terms apply independently
of the Apache-2.0 licence on PyHydroGeophysX's own code.

## geoana

`forward/_tdem_kernels.py` holds numba versions of the TE reflection
coefficient of a layered earth and its conductivity gradient, `_rTE_forward`
and `_rTE_gradient` in [geoana](https://github.com/simpeg/geoana)
(`geoana/kernels/tranverse_electric_reflections.py`), Copyright (c) 2017 SimPEG
Team, MIT License. The licence text is kept at the top of that file, so it
travels with the code in every distribution.

## Phoenix receiver hardware rules

`data_processing/mt/phoenix.py` (`_board_gain`, `_low_pass`) turns the
hardware-configuration bytes of a Phoenix time-series header into a board's
gain, attenuator and low-pass corner. The bit assignments and gain values are
those of two MIT-licensed readers, which implement the same rules:

* [PhoenixGeoPy](https://github.com/torresolmx/PhoenixGeoPy), Copyright (c)
  2021 Phoenix Geophysics Ltd.;
* [mth5](https://github.com/kujaku11/mth5) (`mth5/io/phoenix/readers`),
  Copyright (c) 2020 JP.

The licence text is kept at the top of that file, so it travels with the code
in every distribution.

## Desktop builds

The desktop studio is built with `packaging/pyinstaller_studio.spec` for
Windows and macOS, in two variants. **Light** contains the package, its Qt
interface (PySide6, pyqtgraph, qtawesome) and its scientific Python
dependencies. **Full** adds pyGIMLi (with pgcore), SimPEG (with discretize,
pymatsolver and geoana) and PyVista/VTK. Full builds take their compiled
engines from the gimli and conda-forge channels, use OpenBLAS in place of
Intel MKL, and use conda-forge's `vtk-base`, which leaves out VTK's FFmpeg
module (FFmpeg links x264 and x265, GPL-2.0-or-later).

### What an archive contains

Each archive holds these at its root, beside the `PyHydroGeophysX-Studio`
folder (Windows) or `PyHydroGeophysX-Studio.app` (macOS):

* `LICENSE` and this document;
* `LICENSES/`: the GNU LGPL v3 and GPL v3 texts (`LGPL-3.0-only.txt`,
  `GPL-3.0-only.txt`) and the licence texts, copyright notices and
  attributions of the bundled components;
* `BUILD_INFO.txt`: the platform, architecture, application version, source
  commit and build tools, and the bundled components with their versions or
  build identifiers, licences and notice-file locations;
* `SOURCE_CODE.md`: the source of the LGPL-covered libraries, the application
  commit the archive was built from, and instructions for replacing those
  libraries.

`BUILD_INFO.txt` lists the files in the archive (native libraries, plugins, QML
files and assets as well as Python packages), each traced to the package that
supplied it; components that exist only in the build environment are not
listed.

### Excluded and optional components

The standard build leaves out:

* ResIPy and the `tetgen` Python package, including its compiled-in TetGen
  library;
* Gmsh's Python API, native libraries and executable (`gmsh`, `pygmsh`);
* Python's `readline` module and conda's GNU Readline libraries (GPL-3.0);
* Qt modules available under the GPL only, among them Qt Data Visualization,
  Qt Charts, Qt Graphs, Qt Quick 3D and Qt Virtual Keyboard;
* Qt PDF, which is LGPL-3.0 but brings PDFium;
* Qt's `opengl32sw.dll` software renderer (Windows);
* Full builds also leave out VTK's FFmpeg module and FFmpeg's libraries.

The standalone E4D, R2, R3t, TetGen and Gmsh programs described above are tools
the user supplies.

`PHGX_BUNDLE_RESIPY=1` and `PHGX_BUNDLE_TETGEN=1` are custom-build switches that
add ResIPy (GPL-3.0) or the `tetgen` package (AGPL-3.0) to the archive. They are
not part of the standard build. Neither switch changes the licence of the
component it adds.

### Qt, PySide6 and Shiboken6

The desktop interface uses Qt and PySide6, including the Shiboken6 runtime,
under the GNU LGPL v3. `LICENSES/` holds the unmodified LGPL v3 and the GPL v3
text it incorporates, `LGPL-3.0-only.txt` and `GPL-3.0-only.txt`, with the
copyright notices of these components. Including both texts does not change the
licence of PyHydroGeophysX's own code. When an archive is built, the licence of
each bundled Qt module is read from the sources of that Qt version.

The studio is a one-folder build, so the Qt, PySide6 and Shiboken6 shared
libraries are separate files. The build checks that they can be replaced: it
swaps in separately obtained copies of the same release, alters one of them and
runs the studio's self-test. Users may modify the LGPL-covered libraries and
reverse engineer the combined work for debugging those modifications, as LGPLv3
permits. PyHydroGeophysX imposes no additional restriction on those rights.

The governing texts are the
[GNU LGPL v3](https://www.gnu.org/licenses/lgpl-3.0.html) and the incorporated
[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.html). The Qt project also
publishes [LGPL guidance](https://www.qt.io/development/open-source-lgpl-obligations)
and its [licensing overview](https://doc.qt.io/qt-6/licensing.html).

### Code and assets included by dependencies

Qt, PySide6 and the scientific Python packages include separately licensed
code and assets, among them embedded or statically linked dependencies, fonts
and icons. Their notices are collected for the bundled versions into
`LICENSES/`, and `BUILD_INFO.txt` gives their locations.

Qt documents these components in
[Third-Party Code Used in Qt](https://doc.qt.io/qt-6/licenses-used-in-qt.html)
and [Licenses Used in Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html).
Those pages describe each Qt version; the notices in `LICENSES/` are those of
the distributed binaries.

### Windows runtime

Windows archives may include Microsoft Visual C++ runtime DLLs, such as
`MSVCP140*.dll` and `VCRUNTIME140*.dll`. They are subject to Microsoft's terms,
not to PyHydroGeophysX's Apache-2.0 licence; `BUILD_INFO.txt` lists each file
with its version and the package that carried it. Microsoft's
[redistribution documentation](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files)
describes those terms. The Universal CRT files (`ucrtbase.dll` and the
`api-ms-win-*.dll` forwarders) are part of Windows 10 and later and are not
included.

Qt's `opengl32sw.dll` is Mesa llvmpipe; Qt identifies its distributed renderer
under the MIT License and Boost Software License 1.0 in its
[Mesa llvmpipe attribution](https://doc.qt.io/qt-6/qt-attribution-llvmpipe.html).
The archives do not include it.

### Full builds: pyGIMLi, Triangle and visualization libraries

pyGIMLi's 2-D mesh generator wraps Jonathan Shewchuk's Triangle, which has its
own redistribution conditions, independent of pyGIMLi's top-level licence. They
require its copyright notices to be kept, require the author's permission for
sale or inclusion in commercial products, and set conditions for modified
copies and for source and object-code distribution. See the
[author's Triangle page](https://www.cs.cmu.edu/~quake/triangle.html).
The libgimli in Full builds is built without Triangle and contains none of its
code; pyGIMLi's 2-D meshes come from rustmesh instead (next section).

PyVista and VTK retain their own licences, MIT and BSD-3-Clause for the
published projects. See
[PyVista's licence](https://github.com/pyvista/pyvista/blob/main/LICENSE)
and [VTK's licensing information](https://docs.vtk.org/en/latest/about.html).
The native libraries of the Full archives are listed in their `BUILD_INFO.txt`
with the Python packages.

### Full builds: rustsolve and rustmesh in place of SuiteSparse and Triangle

pgcore, pyGIMLi's compiled core, solves sparse systems with SuiteSparse CHOLMOD
and UMFPACK, and makes 2-D meshes with Triangle. UMFPACK, and CHOLMOD's
Supernodal, Modify and MatrixOps modules, are GPL-2.0-or-later. In Full builds
they are replaced by rustsolve, a sparse-solver library on faer, and rustmesh,
a mesh generator. The prebuilt files are in `packaging/rustsolve/`; its
`README.txt` gives their origin and `manifest.json` their SHA-256. The build
installs them with `packaging/rustsolve/apply.py`:

* **rustsolve's C interface** (`rustsolve_capi.dll`, `librustsolve_capi.dylib`).
  rustsolve 0.1.0 is MIT OR Apache-2.0, from
  https://github.com/geohang/rustsolve at commit c83f4e9. The Rust crates
  compiled with it are faer 0.24.4 and others under MIT, Apache-2.0,
  BSD-2-Clause, Zlib or Unlicense terms. faer's licence files also cover code
  ported into faer from SuiteSparse AMD and COLAMD (BSD-3-Clause), LAPACK (BSD)
  and Eigen (MPL-2.0). The Eigen-derived code, a divide-and-conquer SVD, is not
  compiled into these files. The Rust standard library is compiled in. The
  Windows file also contains MinGW-w64 runtime code and GCC's unwinder
  (GPL-3.0-or-later with the GCC Runtime Library Exception 3.1).
* **rustmesh** (the `rustmesh` Python package, with `_rustmesh.pyd` or
  `_rustmesh.abi3.so`). rustmesh 0.1.0 is MIT OR Apache-2.0, from
  https://github.com/geohang/rustmesh at commit e332d94. It contains a modified
  copy of spade 2.15.1 (MIT OR Apache-2.0; the changes are listed in its
  `PATCHES.md`) and Rust crates under MIT, Apache-2.0, BSD-2-Clause or Zlib
  terms, with the Rust standard library and, on Windows, the same MinGW-w64
  and GCC runtime code. PyHydroGeophysX's rustmesh backend
  (`PyHydroGeophysX/core/rustmesh_backend.py`) makes pyGIMLi's 2-D meshes with
  it.
* **libgimli, modified** (`libgimli.dll`, `libgimli.dylib`). This is pyGIMLi's
  core library (Apache-2.0), rebuilt from pyGIMLi's sources with rustsolve's
  changes (`pgcore/apply_rustsolve.py`) and without Triangle
  (`packaging/rustsolve/apply_notriangle.py`):
  * Windows: commit 5991b6ac, the source of pgcore 1.5.0;
  * macOS: v1.5.4, the source of pgcore 1.5.4.

  As section 4(b) of the Apache License asks, each changed source file
  carries a modification notice, and the archive's `pygimli/MODIFICATIONS.txt`
  names the changed files. It is in the component's folder in `LICENSES/`.
  The library's version string ends in `+rustsolve+notriangle`.
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
