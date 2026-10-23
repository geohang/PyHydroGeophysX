# Changelog

Key changes to PyHydroGeophysX, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Before 1.0, minor releases may change the API.

## [Unreleased]

### Added

- **Studio 3D Mesh Builder:** sensor positions from a file (`Cable,Elec,X,Y,Z` and the like), and "Build mesh from sensor file…" to read one and generate the mesh in one step; "Use for ERT inversion" sends the mesh to the ERT page.
- **Studio ERT Mesh tab:** 3D meshes shown in an interactive 3D view; "Build a 3D mesh…" opens the 3D Mesh Builder with the page's electrodes.
- **ERT:** the in-house, PyGIMLi and ADTLERT engines invert a profile on an imported 3D mesh; the section along the line is shown and the whole model saved as `resistivity_3d.vtk`. Time-lapse series too, each step's volume saved beside its section.
- **ERT 3D surveys:** electrodes covering an area (a grid from an `X,Y,Z` electrode file) keep their y; the Data tab shows the readings in plan, a prism mesh is built under the array, and the in-house, PyGIMLi and ADTLERT engines (on the prisms split into tetrahedra, with quadratic elements) invert single surveys and time-lapse series in 3D. Studio and Saved Results show 3D models in an interactive 3D view and correct them for temperature; Add to Map takes them as depth slices.
- **MT:** `TimeSeriesRun.with_dipole_length` reads an electric channel with another dipole length, for a line that changed during a recording, such as one arm left measuring against the ground electrode after an electrode broke off.
- **Assistants:** written procedures (skills) in the open Agent Skills format. AQUAH ships its own, a group adds more with `PHGX_SKILLS_PATH`, and the assistant reads one when it fits: in Auto to report as a free look, in the chat with a `read_skill` tool.
- **Uncertainty:** `uncertainty.interval_coverage` checks a reported mean +/- standard deviation against a known truth (synthetic models, cores, logs): coverage at 1 and 2 sigma, z statistics and bias, overall and per group such as depth bins.
- **Examples:** `Ex_GPR_workflow` runs the GPR tools on the line simulated over the MODFLOW model: processing, a check of time zero and of the velocity against the CRIM permittivity that made the line (a first-break time zero against peak-timed reflections reads 27 % slow), depth and elevation sections, and the saved line. It is linked from the README, the GPR guide and the examples page.
- **Uncertainty:** `uncertainty.regularized_posterior_std` gives each cell's linearised posterior standard deviation of a regularised Gauss-Newton model; `ERTInversion(posterior_std=True)` returns it as `result.model_log_std`, and `run_petrophysics_monte_carlo(log_resistivity_std=...)` draws the resistivity with it.
- **ADTLERT 3D:** `engine="adtlert"` inverts 3D surveys on tetrahedral meshes (`mesh_file`), in single and windowed time-lapse runs, with ADTLERT's 3D forward.
- **ADTLERT remote electrodes:** pole-dipole and pole-pole surveys (negative ABMN indices) run on ADTLERT, on flat and topographic 2D profiles and on 3D meshes; they no longer fall back to PyHydro. adtlert 0.3 applies the far-field mixed condition on 2.5D terrain meshes, which also changes ADTLERT's 2.5D terrain results.

### Changed

- **3D Mesh Builder:** field meshes (prism, Gmsh, structured grid) reach 4 spreads beyond their domain, sideways and down, through a coarse outer region that is never inverted, as the ERT page's own mesh does ("Outer region"; 0 leaves it out, a Box mesh has none). On a two-layer ground the error fell from 1.0 to 0.45 % (prism), 1.3 to 0.55 % (Gmsh) and 2.2 to 0.39 % (structured grid), 90th percentile. The prism mesh's narrow ring around its domain is no longer inverted, and the view hides the outer region unless "Outer region" above it is ticked.
- **ADTLERT backend:** requires `adtlert>=0.3,<0.4`. The `adtlert` extra no longer installs CuPy, or cuDSS and nvmath-python separately on Windows; ADTLERT brings cuDSS and compiles its kernel with NVRTC. GPU availability is checked with Torch and an NVRTC compile instead of CuPy, and `gpu_cgls` runs on Torch.
- **ADTLERT backend:** surveys are passed with their recorded receiver polarity; ADTLERT 0.2.3 handles negative geometric factors, so the M/N exchange that canonicalized them is gone. Over topography the forward still frees its P2 setup systems (potentials, assembly templates and cuDSS factorizations) once their results are cached, now from adtlert 0.3's caches: on the DAS-1 example its GPU memory after the first solves falls from 1.7 to 0.6 GB, with the same solves.
- **3D mesh, prism engine:** a single electrode line no longer gives a zero-width domain.
- **AQUAH:** every decision in Auto to report states a confidence, kept on the step in the run's plan; leaving route steps out with low confidence is put to the user instead of decided alone.
- **AQUAH:** file previews, reference-document excerpts and a script's output reach the model fenced as data, so text in them that reads like an instruction is not taken as one.
- **Water content:** a single-survey ERT run now draws the resistivity with the inversion's linearised posterior as well as the petrophysical parameters, so the +/- carries both. Reports say what the +/- contains and that it is still too narrow at depth and where coverage is low.
- **Agents:** an agent uses an API key only when given one (`api_key="env"` reads the provider's variable). It used to read `OPENAI_API_KEY` and the like by itself, so a library call, or `api_key=None` in the documentation's no-model examples, could send requests and spend credit. When a key is set but not passed, the agent warns once.
- **AQUAH chat:** removing inputs or electrodes, and exporting or saving to a path the assistant names, now always wait for the user's approval.
- **AQUAH:** the request reader treats rain or weather named as a cause, and data of unstated type, as cases to mark true. A site given as coordinates ("41.37 N, 106.24 W") is no longer taken for a place name.
- **MT reading:** a Phoenix recording read with `start` and `end` now opens only the 24 kHz and 150 Hz files within them, as it already did for native files, so an hour of a day-long recording reads in a fraction of a second rather than loading gigabytes of bursts.
- **Third-party notices:** the DAS-1 and E4D example data now name their data releases (HydroShare, ESS-DIVE) and CC BY 4.0 licence, ResIPy and the MIT-licensed Phoenix receiver rules (PhoenixGeoPy, mth5) have entries, and the desktop-build notes state what each archive bundles and where its notices are.
- **ERT mesh settings:** `ERTInversion` builds its mesh with the `quality`, `paraDX`, `paraMaxCellSize` and `paraDepth` it is given (it always used quality 34 and pyGIMLi's defaults). The ERT agent passes only the settings a request gives, so default runs are unchanged. The time-lapse mesh keeps its defaults unless the request gives its own.
- **Desktop archives:** each archive now carries `BUILD_INFO.txt` (the components it bundles, with versions, origins and licences), `SOURCE_CODE.md` (the source of its LGPL libraries and how to replace Qt) and `LICENSES/` at its root. `packaging/release_audit.py` writes them and checks every archive (Qt module and plugin licences, excluded engines such as Gmsh and TetGen, bundled third-party libraries, source links, a Qt library swap) before the workflow attaches it to a draft release. Full builds take Qt, PySide6 and VTK from conda-forge (one Qt build, no FFmpeg) and use OpenBLAS.
- **Desktop Full builds:** pyGIMLi's core solves its systems with rustsolve instead of SuiteSparse CHOLMOD and UMFPACK, and is built without Triangle; its 2-D meshes come from rustmesh through `PyHydroGeophysX/core/rustmesh_backend.py`. The prebuilt files and their checks are in `packaging/rustsolve/`.
- **README:** the method list, Features, install notes, package structure and examples table now cover GPR, gravity and magnetics, laterally constrained TEM lines, joint FDEM+TDEM and gravity+magnetics inversion, and writing results back to MODFLOW 6 or ParFlow. The examples table gains the eight examples it was missing.
- **README citations:** the README keeps the two PyHydroGeophysX papers and the core packages and engines (pyGIMLi, SimPEG, ADTLERT, ResIPy, E4D, R2/R3t, FloPy/MODFLOW, ParFlow, ATS, PFLOTRAN). The papers behind individual methods (ERT reciprocal errors, shallow seismic and GPR processing, MT, laterally constrained EM inversion, petrophysics and pore fluids) and the Gmsh reference moved, with their BibTeX, to the new Method references section of the website's citation page. The web app's reference list follows the README.
- **Extras:** the README's extras table now matches `pyproject.toml`: it adds the `ert` and `mcp` groups and the packages each group was missing, and names the groups `all` leaves out (`adtlert`, `ert`, `mcp`). `all` now also installs `pyproj`, as `desktop` does.
- **Citations:** the laterally constrained EM inversion (`inversion.em1d_lci`) now credits the method it follows, Auken and Christiansen (2004), and its depth of investigation, Christiansen and Auken (2012), on the docs' citation page (with BibTeX under Method references) and in the module docstring.
- **Citations:** the PyHydroGeophysX software paper is now cited as published, SoftwareX 34, 102710 (https://doi.org/10.1016/j.softx.2026.102710), in the README, the docs' citation page, the agents README and the web app's reference list, instead of "in press" with the SSRN preprint.
- **Wording:** docstrings and comments across the package state what the code does and the rule it keeps, without the history of how it was found; the code, the Studio, the README and the website describe each feature by what it does, such as 3D meshes from E4D mesh configurations.
- **Documentation:** the GPR pages credit GPRPy, readgssi, ImpDAR and RGPR for the standard processing sequence, not for the readers; the petrophysics page names the release (0.5.0) that corrected the Hertz-Mindlin shear prefactor; the desktop studio page gives the PyInstaller commands the build workflow runs.

### Fixed

- **3D ERT meshes from elsewhere:** PyGIMLi reads the outer faces' markers as boundary conditions, and the 3D Mesh Builder's Gmsh, structured-grid and E4D meshes (or any imported mesh) left them no flow, an insulating box: on a two-layer ground the structured grid missed by 25 % (90th percentile). The Mesh Builder and the ERT page's mesh import now mark the ground no flow and every other outer face far-field (2.4 %).
- **3D Mesh Builder, Gmsh:** electrodes on the ground were embedded in the volume, which left each one twice in the mesh with flat cells between; neither ERT engine could use it. They are now embedded in the ground surface.
- **ADTLERT 3D memory:** ADTLERT built its background map as a dense cells x parameters array (3.4 GB on an E4D mesh of 35 644 cells) and padded its assembly to 64 entries (over 7.5 GB with quadratic elements); both are now built lean in PyHydroGeophysX (2.0 GB, same results).
- **Studio assistant:** an ERT run the page declines (ADTLERT's GPU check not yet passed, say) was reported to the assistant as started; it is now reported as not started, with the page's reason.
- **Studio seismic picking:** a Ctrl+drag line pick that ended with Ctrl released before the mouse button left the dashed line on the gather and started the next pick from the old start; whether a drag draws a line is now settled when it starts. A drag no longer redraws every wiggle of the gather (about 7 ms per drag), picks made or undone by hand keep the zoom, and clicks above time zero or past the end of the record make no picks.
- **Water content (AgentCoordinator):** `WaterContentAgent` no longer reads an inversion mesh's per-cell markers as layers in index order or drops petrophysical parameters given by name. It converts through `PetrophysicsAgent`, as AQUAH does: one unit unless the markers are layers, the user's parameters applied, and the not-reliable statement when they are missing. On a synthetic with the truth known, cells within one sigma rose from 27 % to 69 %. Distributions given for every marker of the mesh are still used as given, the inversion's posterior is drawn on both routes, and the result keeps `saturation_all`.
- **ERT loader:** the data-quality figures read a field the observations do not have, so they were never computed. They now come from each reading's apparent resistivity.
- **GPR migration:** `gpr.migrate` (F-K Stolt) drops each trace's mean, and with the record zero-padded before time zero a trace whose first samples cut through a strong arrival kept a constant offset down its whole length, a pale block under the cut once gained. Time now goes through a sine transform, which extends the record to negative times as its own negative, so the offset is gone. The studio's GPR page and the AQUAH workflow get the fix through `gpr.process`, and `Ex_GPR_workflow` no longer band-passes a second time after migration.
- **GPR migration late in the record:** the Stolt mapping read the spectrum between frequency samples by linear interpolation, which lost amplitude of dipping energy late in the record (a diffraction at 90 ns in a 100 ns record focused 14 % low). It now uses a 10-tap Kaiser-windowed sinc, in the same memory and about 1.5 times the run time.
- **GPR band-pass after time zero:** `gpr.process` cut the record at time zero and then band-passed it, so every trace started partway through the direct wave and the filter's low corner rang on that edge, drawing arcs under the shallowest reflection. When a band-pass follows, the samples before time zero are now dropped after it; the time-zero pick and the line's length are unchanged, and the band-pass line of the history says so. The studio's GPR page and the AQUAH workflow get the fix through `gpr.process`; `gpr.bandpass` and `gpr.set_time_zero` called one after the other still cut first, and their docstrings say to band-pass before cutting.
- **ERT forward with a NaN or infinite cell:** pyGIMLi's `ERTModelling` models a NaN cell without a word and only prints a warning for an infinite one, and a NaN water content (a masked or inactive hydrologic cell) keeps its NaN through the petrophysical clip, so `hydro_to_ert` could make plausible but wrong synthetic data. `ERTForwardModeling`, `ertforward`, `ertforandjac`, `ertforward2`, `ertforandjac2`, `hydro_to_ert` (and `run_ert3d_forward` through `ERTForwardModeling`) and the final models of the ERT and time-lapse inversions now go through one check, `forward.ert_forward.check_resistivity_model`, which refuses a NaN, infinite, zero or negative cell before the forward call and says how many there are and where. The single-survey ERT inversion used to set a NaN from a failed update solve to the lower resistivity bound, which the line search could then accept; it now stops with that error.
- **`ertforward`:** the cells outside marker 2 were passed to the forward operator as ln(resistivity), so a 100 ohm-m background was modelled at 4.6 ohm-m, and the model was built with `pg.matrix.RVector`, which pyGIMLi 1.5.5 does not have (it is `pg.core.RVector`), so every call failed. Those cells now keep the given resistivity in ohm-m, and only the cells of marker 2 take `exp(xr)`, as the docstring says.

## [0.5.0] - 2026-10-15

First release after 0.3.0; 0.4.0 was not published. These notes cover every change since 0.3.0.

### Added

- **MT/AMT:** instrument readers, robust remote-reference processing, isotropic and anisotropic 1D inversion, TEM static-shift correction, and 2D profile inversion.
- **GPR:** file readers, signal processing, migration, velocity fitting, and time/depth/elevation views in Studio and AQUAH workflows.
- **EM:** laterally constrained line inversion, joint FDEM/TDEM inversion, and TEMcompany, TEM2Go and tTEM readers.
- **ERT:** a-priori resistivity zones, imported meshes, mesh generation from E4D configurations, automatic regularization search, reciprocal pairing, averaging and error models, and BGS PRIME/RESIMGR support.
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
