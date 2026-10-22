# Changelog

Key changes to PyHydroGeophysX, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Before 1.0, minor releases may change the API.

## [Unreleased]

### Changed

- **Third-party notices:** the DAS-1 and E4D example data now name their data releases (HydroShare, ESS-DIVE) and CC BY 4.0 licence, and the desktop-build notes state what each archive bundles.
- **Desktop archives:** each archive now carries `BUILD_INFO.txt` (the components it bundles, with versions, origins and licences), `SOURCE_CODE.md` (the source of its LGPL libraries and how to replace Qt) and `LICENSES/` at its root. `packaging/release_audit.py` writes them and checks every archive (Qt module and plugin licences, excluded engines such as Gmsh and TetGen, bundled third-party libraries, source links, a Qt library swap) before the workflow attaches it to a draft release. Full builds take Qt, PySide6 and VTK from conda-forge (one Qt build, no FFmpeg) and use OpenBLAS.
- **Desktop Full builds:** pyGIMLi's core solves its systems with rustsolve instead of SuiteSparse CHOLMOD and UMFPACK, and is built without Triangle; its 2-D meshes come from rustmesh through `PyHydroGeophysX/core/rustmesh_backend.py`. The prebuilt files and their checks are in `packaging/rustsolve/`.

## [0.5.0] - 2026-10-15

First release after 0.3.0; 0.4.0 was not published. These notes cover every change since 0.3.0.

### Added

- **MT/AMT:** instrument readers, robust remote-reference processing, isotropic and anisotropic 1D inversion, TEM static-shift correction, and 2D profile inversion.
- **GPR:** file readers, signal processing, migration, velocity fitting, and time/depth/elevation views in Studio and AQUAH workflows.
- **EM:** laterally constrained line inversion, joint FDEM/TDEM inversion, and TEMcompany, TEM2Go and tTEM readers.
- **ERT:** a-priori resistivity zones, imported meshes, E4D-style mesh generation, automatic regularization search, reciprocal pairing, averaging and error models, and BGS PRIME/RESIMGR support.
- **ERT engines:** external E4D, R2 and R3t engines, and optional ADTLERT GPU inversion.
- **Hydrology:** ATS and PFLOTRAN output readers.
- **Hydrogeophysics:** hydrological-model input export, joint gravity/magnetic inversion, resistivity temperature correction, and Monte Carlo water-content conversion.
- **Petrophysics:** pore-fluid properties, ionic and surface conductivity helpers, and configurable pressure and fluids for Hertz–Mindlin models.
- **AQUAH:** unified tool-driven runtime, model routing, local reference retrieval, live-panel capture, and request-aware reports.
- **Assistants:** plug-in support, Codex/Claude CLI access, OpenAI-compatible inference endpoints, reusable tool loops, and approved temporary scripts.
- **Studio:** integrated workflows, joint inversion, project maps, saved results, mesh previews, and process-isolated runs.
- **Run monitoring:** live progress, pause/resume, replay, cost tracking, processing records, and unsaved-run recovery.
- **Workflows:** versioned recipes, a CLI, optional MCP access, and reproducible Python scripts and notebooks.
- **Examples:** EM sections, gravity/magnetics, hydrological feedback, sensitivity, assimilation, and posterior uncertainty.

### Changed

- **Studio:** consistent ERT-style pages, contextual controls, pinned result actions, light/dark themes, plot navigation, colour limits, and selectable length units.
- **Studio responsiveness:** deferred numerical-engine imports, background EM checks, cached slice plots, batched logs, and no refresh of hidden quality/property panels.
- **AQUAH:** shared step-by-step and automatic workflows, adaptive routes, reuse of Studio results, and model-first request parsing.
- **Water content:** conversions always include uncertainty and identify uncalibrated parameter assumptions.
- **Solvers:** time-lapse ERT and seismic solvers use Cholesky by default; time-lapse ERT adds interval-weighted temporal regularization.
- **Inversion:** updated stopping rules, EM starting-model selection, regularization search, and ERT quality controls.
- **Performance:** faster MT/TEM processing and Studio navigation; lower memory use in ERT and faster joint ERT+SRT inversion.
- **ERT reading:** E4D, DAS-1 and BERT files load 3–7x faster with identical data (a 3.7k-reading E4D survey: about 300 → 40 ms), so a long time-lapse series spends seconds rather than minutes reading.
- **Package layout:** scientific pipelines moved out of the Qt package; optional dependency groups updated.
- **Climate data:** Open-Meteo ERA5 replaces Daymet.
- **Monte Carlo example:** `Ex_MC_Hydro` estimates porosity with the package's Monte Carlo and says where porosity has no solution and where the S = 1 assumption fails.
- **Documentation:** concise README and changelog, with README images that also show on PyPI; detailed methods, references and setup guidance on the documentation website.

### Upgrade notes

- **Python 3.10+ is required.**
- **Workbench becomes Studio:** use `pyhydrogeophysx-studio` and the renamed Studio classes and launch scripts.
- **Syscal and Protocol DC/IP files require optional ResIPy.**
- Legacy agent classes and structural-mesh helpers were removed; import errors or warnings identify replacements.
- Compatibility module paths and renamed MODFLOW reader arguments are deprecated for removal in 0.6.0.
- Corrected petrophysics, FDEM receiver ordering, temporal weighting, geometry and uncertainty calculations can change results from 0.3.0.

### Fixed

- **Hertz–Mindlin:** corrected the shear-modulus formula and unified legacy and class APIs; results from affected revisions change, while default parameters are retained.
- **Numerical operators:** corrected weighted cross-gradients, rotation invariance, gravity/magnetic broadcasting, and interpolation at large map coordinates.
- **Porosity from resistivity:** where surface conduction alone exceeds the measured conductivity, porosity is now NaN with a warning instead of the solver's starting guess (~0.05); the Monte Carlo leaves those draws out of the statistics and reports their share per cell. The conversion is now closed-form and about 60x faster on the Treeline example.
- **Uncertainty:** corrected resolution and posterior covariance calculations and ES-MDA inflation validation; supplied variances are preserved.
- **Coupling:** stabilized Gassmann endpoints, corrected hydro-profile geometry, hydro-to-geophysics outputs and seismic-to-water-content model selection, and flagged assumed MODFLOW porosity.
- **ERT:** corrected geometric factors and reciprocal errors; improved format validation, electrode geometry handling, time-lapse QC, and solver memory use.
- **ERT auto-detection:** a DAS-1 file is no longer read as a one-reading E4D survey, and trying a file with PyGIMLi no longer leaves `invalid.data` in the working folder.
- **MT:** corrected interference removal, burst processing, calibration, channel directions, saturated-data handling, error propagation, and inversion data selection.
- **Seismic and TEM:** corrected SEG-2 decoding, acquisition timing, shot geometry, picking, sounding signs, and failed-sounding reporting; SRT now honors requested solver settings.
- **Agent workflows:** corrected method selection, file classification, missing products, background-job failures, recovery, and reporting of the parameters actually used.
- **Projects and reports:** improved workflow reports, project portability, Unicode paths, run cleanup, and example reproducibility.
- **Documentation:** the website builds without its 67 long-standing Sphinx warnings: example-gallery text blocks, docstring formatting, duplicate API entries, and an ambiguous `mesh` link.

## [0.3.0] - 2026-07-09

- Desktop Workbench downloads for Windows and macOS, with light and full builds.
- Automatic documentation-site deployment.
