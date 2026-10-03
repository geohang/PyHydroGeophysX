# Third-party code, data and desktop distributions

PyHydroGeophysX's own code is licensed under Apache-2.0; see LICENSE.
Third-party packages and data retain their own licenses. This document does not
relicense them or establish permissions that have not been verified.

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

## Desktop builds

The default light/full builds exclude ResIPy, even if installed in the build
environment. A distributor can explicitly enable it for a full build with
`PHGX_BUNDLE_RESIPY=1`, after reviewing the GPL obligations of that distribution.
An optional dependency declaration alone does not resolve those obligations.
The `tetgen` Python package is excluded the same way, and only
`PHGX_BUNDLE_TETGEN=1` bundles it, after reviewing the AGPL obligations.

The full build collects pyGIMLi, whose 2-D mesh generator wraps Jonathan
Shewchuk's Triangle. Triangle may be redistributed free of charge with its
copyright notices kept; distributing it as part of a commercial system needs
its author's permission.

Qt/PySide6 and other bundled libraries also require their applicable license
texts and distribution obligations to be satisfied. Before releasing an archive,
inspect the actual bundled modules, versions and library licenses, retain the
required notices, and document the applicable source/library replacement route.
PyInstaller hooks may collect some notices; their presence must be verified in
the final archive. Shipping this document alone is not sufficient.

References: [GNU GPL FAQ](https://www.gnu.org/licenses/gpl-faq.html.en),
[Qt LGPL obligations](https://www.qt.io/development/open-source-lgpl-obligations).

## Example data

`examples/data/README.md` lists every example dataset with its source and
licence. Data taken from public releases keep their attribution beside the
files, among them:

* `examples/data/EM/skytem_bhmar_*.csv`: five soundings of Brodie, R.C. (2019),
  *Broken Hill Managed Aquifer Recharge (BHMAR) SkyTEM Airborne Electromagnetic
  Survey, NSW, 2009*, Geoscience Australia,
  https://doi.org/10.26186/5d314c5de8040, under CC BY 4.0; the subset was taken
  from the example in Geoscience Australia's ga-aem repository.
* `examples/data/climate/`: Daymet Version 4 R1 daily weather at one point,
  https://doi.org/10.3334/ORNLDAAC/2129, which the ORNL DAAC asks users to cite.

## Sources requiring provenance verification

| Material | Existing evidence | Verification still needed |
| --- | --- | --- |
| Log-time Hermite interpolation in `forward/tdem_forward.py`, the same rule as [TEM1D](https://github.com/hydrogeophysicsgroup/TEM1D) | Upstream URL and MIT identification | Whether implementation was independently written or adapted; retain upstream copyright/license if applicable |
| ERT DAS and E4D examples | Paper citations in dataset acknowledgements | Data release URL/version and redistribution license |
| East River VTEM subset | USGS data DOI and processing description | Rights statement of the specific release and any third-party exceptions |
| Repository images | Local images and generated example figures | Author/source/license or generating script for each distributed asset |

Gravity/Magnetics acknowledgements already record data sources, licenses and
transformations; retain these alongside redistributed data. Preserve all dataset
acknowledgements when packaging subsets. Unverified provenance is a tracking
item, not an allegation of infringement.
