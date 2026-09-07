# Third-party code, data and desktop distributions

PyHydroGeophysX's own code is licensed under Apache-2.0; see LICENSE.
Third-party packages and data retain their own licenses. This document does not
relicense them or establish permissions that have not been verified.

## Desktop builds

The default light/full builds exclude ResIPy, even if installed in the build
environment. A distributor can explicitly enable it for a full build with
`PHGX_BUNDLE_RESIPY=1`, after reviewing the GPL obligations of that distribution.
An optional dependency declaration alone does not resolve those obligations.

Qt/PySide6 and other bundled libraries also require their applicable license
texts and distribution obligations to be satisfied. Before releasing an archive,
inspect the actual bundled modules, versions and library licenses, retain the
required notices, and document the applicable source/library replacement route.
PyInstaller hooks may collect some notices; their presence must be verified in
the final archive. Shipping this document alone is not sufficient.

References: [GNU GPL FAQ](https://www.gnu.org/licenses/gpl-faq.html.en),
[Qt LGPL obligations](https://www.qt.io/development/open-source-lgpl-obligations).

## Sources requiring provenance verification

| Material | Existing evidence | Verification still needed |
| --- | --- | --- |
| TEM1D interpolation reference in `forward/tdem_forward.py` | Upstream URL and MIT identification | Whether implementation was independently written or adapted; retain upstream copyright/license if applicable |
| ERT DAS and E4D examples | Paper citations in dataset acknowledgements | Data release URL/version and redistribution license |
| SkyTEM BHMAR subset | Geoscience Australia ga-aem source and copyright attribution | License applicable to the specific data files, separately from software |
| East River VTEM subset | USGS data DOI and processing description | Rights statement of the specific release and any third-party exceptions |
| Repository images | Local images and generated example figures | Author/source/license or generating script for each distributed asset |

Gravity/Magnetics acknowledgements already record data sources, licenses and
transformations; retain these alongside redistributed data. Preserve all dataset
acknowledgements when packaging subsets. Unverified provenance is a tracking
item, not an allegation of infringement.
