# Changelog

All notable changes to PyHydroGeophysX are recorded in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0, a
minor release can change the API.

## [Unreleased]

### Added

- Magnetotelluric data, `data_processing.mt`, on NumPy and SciPy alone - no
  new dependency. `TransferFunction` holds one site's impedance and tipper in
  ohms and e^{+iωt}, with their errors and, where known, the full covariance,
  and rotates them; `read_transfer_function` reads EDI (impedance, apparent
  resistivity and phase, and spectra sections), EMTF XML, Egbert Z-files
  (`.zmm`, `.zrr`, `.zss`) and J-files, and `write_edi` / `write_emtf_xml`
  write them back. EDI and J-files do not state their sign convention, so it is
  decided from the phases. `read_timeseries` reads the instruments' own files
  into `TimeSeriesRun`s of `Channel`s, each with its response chain (sensor,
  dipole, receiver filters) so that `calibrated()` gives mV/km and nT: Phoenix
  MTU-5C/5P/8A recordings (native `.bin`, decimated `.td_*`, with `rxcal.json`
  and `scal.json`), legacy Phoenix MTU-5A/V8 sites (`.TS2`-`.TS5` with `.TBL`),
  Metronix ATS (with the measurement XML's or a text file's coil calibration),
  Zonge Z3D (with the coil table stored in the file) and LEMI-424 text files.
  Times come out in UTC.
- MT processing: `mt.process_mt(runs, remote=...)` estimates the impedance and
  tipper from any instrument's runs - cascade decimation, windowed and
  prewhitened Fourier coefficients in field units, Huber then redescending
  regression, remote reference, EMTF's band set by default (or an EMTF
  band-setup file) - with EMTF's full error covariance. The errors allow for
  the correlation the taper puts between a band's harmonics; on synthetic data
  they are calibrated, and on EMTF's own test sites the impedances match
  EMTF's to about 1%. Runs at several sample rates merge into one site.
  `mt.phase_tensor`, `swift_skew`, `bahr_skew`, `swift_strike`,
  `induction_arrows` and `niblett_bostick` describe dimensionality and depth;
  `estimate_static_shift`, `static_shift_from_layers` (for instance against a
  TEM sounding's layered model) and `apply_static_shift` remove static shift.
  `mt.impedance_1d` and `sensitivity_1d` give a layered earth's impedance and
  its derivatives.
- MT inversion. `mt.occam1d` finds the smoothest layered model that fits a
  site's apparent resistivity and phase (determinant, either mode, or both
  against one model) to a target misfit, with exact derivatives; it can solve
  a static shift per mode, and with `tem=` it fits a TDEM sounding jointly
  through the package's TDEM forward operator, which fixes the static shift
  the MT alone cannot (on a synthetic site shifted by 0.5 it returns 0.51 and
  0.50, and the true model). `mt.water_content_profile` turns the layers into
  water content with the Waxman-Smits link the rest of the package uses.
  `mt.invert_profile` inverts a profile's TE and TM impedances for a 2D
  section on SimPEG's NSEM simulations (`build_profile_mesh` lays the mesh out
  from the band, `forward_profile` models a section); SimPEG is imported only
  when they run. The 1D layering starts from the apparent resistivity at the
  sounding's highest frequency and ends at its lowest, so a resistive site
  under a conductive cover no longer gets a top layer tens of metres thick.
- MT across the package. Three workflows, `mt.process`, `mt.invert_1d` (with
  an optional TEM sounding and water content) and `mt.invert_profile`, run
  through `run_workflow` and export a recipe, a runner and a walkthrough like
  the others; `workflows.mt` gathers their names. The studio has a
  **Magnetotellurics** page laid out like the Seismic page: the Time series,
  Sounding, Dimensionality, 1D model, Phase tensors and 2D section tabs each
  bring up the side panel for their step - reading and processing (with a
  remote reference and a calibration override), the sites, the 1D inversion,
  the profile and its 2D inversion; every run is recorded in the Project, a
  model can be added to Project Map, and AQUAH can drive the page.
  The agent runtime has `invert_mt` and `convert_mt_water_content`: a request
  that names `.edi` files (or says MT and names XML or J-files), or a
  configuration with `mt_files`, inverts the sites in 1D and a located line of
  three or more in 2D; the folder classifier knows MT sites. The Streamlit app
  lists the page. `visualization.plot_mt_dimensionality` draws a site's phase
  tensor against period. New example `Ex_MT_workflow` (a MODFLOW column to AMT
  time series and back to an EDI file, the static shift fixed by TEM, water
  content, a 2D profile, and USMTArray station NMX20, shipped in
  `examples/data/MT` under CC BY 4.0) and a methods page, with the references
  in the README and on the citation page.
- Figures can show lengths in feet. `visualization.axis_units` holds the
  choice: `set_length_unit("ft")` sets it for every figure, `length_unit("ft")`
  sets it for one block, and the section, map and profile functions in
  `visualization` take `length_unit="m"` or `"ft"` for one call. Models, meshes
  and exported files stay in metres; only the axis ticks and labels change, so
  switching units cannot round anything off. The studio offers the same choice
  under View > Length Units, remembered between sessions and applied to every
  page. Agent reports read it from `config["figure_style"]["length_unit"]`, which
  the request parser fills in when a request asks for feet; without a language
  model, a request that says "feet", "ft" or "英尺" is still honoured.
- `visualization.plot_model_section`, `plot_timelapse_snapshots`,
  `plot_difference_map`, `plot_coverage`, the pseudosection plots and the
  animations take `vertical="auto"`, `"elevation"` or `"depth"`.
- E4D as an ERT engine (`engine="e4d"`, and **E4D 3D (PNNL, external)** on the
  studio's ERT page). `inversion.e4d` writes the files PNNL's E4D reads - mesh,
  survey, starting conductivity, inversion and output options, `e4d.inp` - runs
  E4D under MPI, and reads back its `sigma.N`, simulated data and `e4d.log`, so
  an E4D run goes through the same QC, error model, outlier rejection, λ search,
  viewers and exports as the other engines. A profile is inverted on a 3D mesh
  built around the line and read back as its section; a 3D survey on an
  imported tetrahedral mesh is inverted on that mesh. A time-lapse series runs
  as E4D's own time-lapse inversion (ERT4): each survey starts from the solution
  before it and the change from it is smoothed. E4D is not bundled: it runs on
  Linux, on Windows only inside WSL 2, and on macOS when built from source; the
  `files` launcher writes the complete run folder for a cluster or another
  machine, read back with `read_e4d_run` / `read_e4d_time_lapse`, and
  `python -m PyHydroGeophysX.inversion.e4d` reports what the current machine
  offers. A run that cannot reach E4D stops with the folder's path; it never
  falls back to another engine.
- `core.e4d_mesh.write_e4d_mesh` writes any tetrahedral mesh as E4D's
  `.node`/`.ele`/`.face`/`.neigh`/`.trn` files, with the boundary flags E4D's
  own mesh build gives (2 on the outer walls and bottom, 1 on the ground);
  `e4d_config_from_electrodes` takes the fine-zone padding as an (x, y) pair.
- ATS and PFLOTRAN output readers (`ATSSaturation`, `ATSPorosity`,
  `ATSWaterContent`, `PFLOTRANSaturation`, `PFLOTRANPorosity`,
  `PFLOTRANWaterContent`), with the same `load_timestep` / `load_time_range` /
  `get_timestep_info` methods as the MODFLOW and ParFlow readers. They read
  ATS visualization files (cycles in any order, plain, `.cell.0` or
  domain-prefixed names) and PFLOTRAN snapshot files (one file, or the numbered
  files of one run - not another run named alike), keep the simulator's cell
  order, axes and time units, and give volumetric water content as porosity
  times liquid saturation. `output_cell_centers` reads the cell centres from
  the simulator's own mesh - ATS's `*_mesh.h5`, PFLOTRAN's `Coordinates` or
  `Domain` group - and `interpolate_timestep` maps a state onto a geophysical
  mesh with them (`axes="xz"` for a section). They need the new `hydrology`
  extra (h5py), and are Python API only: the studio, the hydro bundle and the
  agents still read MODFLOW and ParFlow alone.
- R2 and R3t as ERT engines (`engine="r2"` / `"r3t"`, and **R2 2D (Binley,
  external)** / **R3t 3D (Binley, external)** on the studio's ERT page).
  `inversion.r2` writes the files Andrew Binley's programs read - `mesh.dat` or
  `mesh3d.dat`, `protocol.dat`, the starting model, `R2.in` / `R3t.in` - runs
  them, and reads back `f001_res.dat`, `f001_err.dat`, the sensitivity map and
  the `.out` log, so a run goes through the same QC, error model, outlier
  rejection, viewers and exports as the other engines. R2 inverts a profile on
  its own mesh; R3t a 3D survey on its tetrahedral mesh, or a profile on a 3D
  mesh built around the line. Both choose their smoothing weight at every
  iteration, so λ and auto-λ do not apply: the run says so and reports the
  weight they chose (`smoothing_alpha`). Fixed a-priori zones are held fixed,
  zone outlines become zones the smoothness does not cross, and a time-lapse
  series is their difference inversion against the first survey. The programs
  are not bundled: on Windows they run natively, found in an installed ResIPy
  unless named; on Linux and macOS through Wine; the `files` launcher writes
  the run folder for elsewhere, read back with `read_r2_run`, and
  `python -m PyHydroGeophysX.inversion.r2` reports what the machine offers.
- The desktop studio shows when the assistant is the one working
  (`qt_apps.widgets.ai_presence`). During an automatic run a soft glow of blue,
  purple, pink and orange flows round the central area (steady orange while it
  waits for you, a green or red flash as it ends), the Workflow page carries a
  banner with the assistant's animated orb (breathing while it decides, a
  turning ring while a step runs), what it is doing and a clock, and a new
  **Live** tab lists each step as a card: the controller's reason for choosing
  it, the module, the time taken, what it found (written out as it arrives) and
  the figures it wrote, with a card for the assistant choosing the next step in
  between.
  Approvals and questions are asked in that card. The chat reports each
  finished step. `run_workflow` takes an `on_event` hook that receives each
  step's start and end as fields, and the desktop runner passes them across as
  `step` events.
- Light and dark appearances for the desktop studio, in View > Appearance
  (Match System, Light, Dark) and the toolbar's day/night switch; the choice is
  remembered, and Match System follows the operating system as it changes.
- AI assistants are plug-ins (`agents.assistants`), so a domain assistant can
  sit beside AQUAH: GeoSAGE, for geological modelling from gravity and
  magnetic data, is being ported this way. An `Assistant` describes one - name
  and domain, chat persona and examples, the input files it takes, its glow
  colours, the packages it needs - and names its workflow and tools, which load
  only when a run starts. Assistants are found in the package and through the
  `pyhydrogeophysx.assistants` entry point, so one can ship in its own
  package; one that fails to import is reported by `load_errors()` and left
  out. The studio's assistant panel has a picker: choosing an assistant starts
  a new conversation with it, gives the Workflow page its input roles and the
  glow its colours, and is remembered. One that is not ready, or whose
  packages are missing, is listed greyed out with the reason. GeoSAGE is
  registered as *in development*, with its six stages laid out as tools after
  the paper's agents (data, petrology, joint inversion, quasi-geology, report,
  review). `docs/source/agents/adding_an_assistant.rst` is the guide to adding
  one.
- The **Live** tab shows the run as a route, its newest figure, and its
  outcome. Across the top, the steps taken are drawn in the assistant's
  colours, the running one with a turning ring and, while the assistant
  decides, a light travelling to the next stop; the steps still ahead are
  hollow stops on a dashed line. The route ahead is the run's own projection,
  `agents.runtime.controller.route_ahead`, recomputed after every step and sent
  to the studio as `phase="route"` events, so it follows the controller rather
  than promising a plan; the banner counts "about N to go" from it. Projected
  artifacts are `context.PROJECTED`, which `RunContext.projected(key)` tests,
  so the ERT gates count the configured surveys instead of guessing. Beside the
  steps, each figure the run writes is shown large the moment it appears,
  revealed behind a band of the assistant's colours, with the earlier ones in a
  strip. A run ends with a card - how it went, how long it worked, its steps,
  figures and files, what to check before relying on it - offering **Read the
  report** and **Open output folder**; the page stays on the Live tab rather
  than switching to the report.
- A run's reasoning streams as it is written. The controller's model call
  streams its reply (`BaseAgent.query_llm(..., on_text=...)`, OpenAI and Claude),
  the decision prompt asks for the reasoning first, and `drive` sends it on as
  `phase="thought"` events; the Live tab shows it under "choosing the next step"
  and the step card does not type it out a second time.
- Steering a running workflow. **Pause after this step** holds the run before
  its next step (the running step finishes) and **Resume** lets it go on; a note
  typed under the timeline is read before the next decision, added to the
  transcript (`RunContext.guidance`), and that decision may change the run's
  adjustable settings to follow it (`recovery.ADJUSTABLE` - never a file). The
  note appears in the timeline as the user's, then with what the assistant made
  of it and which settings changed; a run without a model says it cannot read
  notes. Notes, pause and resume travel through `steering.jsonl` in the run's
  folder, which the run reads between steps (`agents.runtime.steering`,
  `ControlFile`) and which records what the user told the run; answers to
  questions still go over stdin. (A thread reading stdin for them froze runs on
  Windows: a pending pipe read stalls process creation in the same process.)
- Token and cost counter. Every model call's tokens and estimated cost reach a
  live total (`llm.runtime_options.add_usage_listener`), shown in the Workflow
  banner and on the finish card, priced from the provider's list prices.
- Run replay. The Live tab's views are recorded with their timing and saved
  beside the results as `live_replay.json` (`qt_apps.widgets.run_replay`);
  **Replay this run** and **Replay a run…** play it back with play, pause,
  speed and a scrub slider, long waits shortened and the run's own clock shown,
  and **Save frame** writes the Live tab to a PNG. Nothing is recomputed.
- `agents.runtime.entry.drive(ctx, tools=..., prompt=..., finish=...)` runs
  the controller loop over any set of tools with the step events, approvals
  and progress the studio uses, for an assistant's own workflow.
  `run_controller` takes the decision prompt and the product that ends a run
  (`finish`, by default `report_files`) the same way.


### Changed

- Water content from geophysics always states its uncertainty and whose
  petrophysical relationship it used. When the user gave none, the conversion
  still runs, but a warning says the result is not reliable and names every
  parameter drawn with its range - 5-95% of the actual Monte Carlo draws, for
  example "cementation exponent m 1.0-3.5, pore-fluid resistivity 20 Ω·m
  (fixed), saturation exponent n 1.0-4.0, porosity 0.01-0.9" - and asks for the
  site's relationship; when only part of it was given, the warning names the
  parameters that took defaults. The step's own summary carries the mean and
  its ± and the "not reliable" flag, so it is seen while the run goes, and the
  report prints the same statement (`_uncertainty.describe_prior`). This holds
  for ERT (single and time-lapse), TDEM and MT conversions.
- A water-content conversion uses at least 50 Monte Carlo draws, saying so when
  fewer were asked for; one draw reported a standard deviation of exactly zero.
  The CLI coordinator (`AgentCoordinator`) now always runs the uncertainty
  analysis rather than a single estimate.
- Parameters a user leaves out of a partly supplied relationship take the
  default with a generous spread (half its value), not the 5% a supplied value
  gets; `rho_fluid` is read when given, and a `[low, high]` range is accepted
  for any parameter.
- Petrophysical values the request parser produced that the request does not
  contain are dropped, with a warning, so a value from the parser's own example
  is never reported as the user's (`context_input_agent.drop_unstated_petrophysics`).
- AQUAH's desktop workflow moved to `agents.assistants.aquah.workflow`; the
  studio's run process (`qt_apps.agent.one_click_runner`) runs whichever
  assistant the request names and no longer holds AQUAH's code. The chat
  panel is `AssistantChatPanel` (`AquahChatPanel` is kept as an alias), and its
  system prompt is assembled from the active assistant's description.

- An automatic run no longer switches the studio from module to module; it is
  followed in the **Live** tab on the Workflow page. **Also bring each module to
  the front as it runs** (off by default) restores the old behaviour.
- The desktop studio has a new look: neutral greys and white (near-black in
  the dark appearance), one blue accent, green, orange and red for success,
  caution and failure, segmented tabs, rounded cards, thin scrollbars and
  sliders, and a log that is no longer a black console. The saturated brand
  blue is gone. Plots stay on a light canvas in either appearance, so black
  traces and colormaps keep their meaning; their default line colours match
  the accent and status colours. Status text is coloured through the stylesheet (`theme.set_tone`),
  so it follows the appearance.
- A section without elevations is labelled Depth, not Elevation. A survey read
  without elevations puts every electrode at z = 0, and its vertical axis now
  reads "Depth (m)", positive downward, in the library plots, the studio and the
  agent reports; a section with topography keeps "Elevation (m)". The reports
  used to label such a section "Elevation (m)" and print negative numbers, and
  the studio sections labelled "Elevation (m)" over pyGIMLi's positive depth
  ticks. The pseudosection plots read "Pseudo-depth" in the same case.
- The `xlabel` and `ylabel` defaults of `plot_model_section`,
  `plot_apparent_resistivity_pseudosection` and `create_timelapse_gif`/`_mp4`
  are now None, meaning the labels above; a label passed explicitly is used as
  it is.

### Fixed

- The assistant could navigate to the Workflow page, the Project Map and the
  Model Viewer but not read or drive them: none implemented the agent
  interface, and the studio's agent self-test failed on it. The Workflow page
  now reports the run (steps, steps ahead, tokens and cost, how the last run
  ended and its report) and takes inputs by role, options, tabs, pause,
  resume, a note, stop and replay; the Project Map selects surveys, shows or
  hides them, chooses slices and surface settings, interpolates, exports the
  map and the grid, renames, and imports point products (with the CRS given,
  without the placement dialog); the Model Viewer lists, filters, selects and
  compares runs, reads a run's overview, shows its artifacts, labels it and
  saves it to the Project. Removing a survey or deleting a run stays with the
  user.
- The studio window fits a 1920 px screen again. Its minimum width had grown to
  2211 px because a row of view controls cannot be narrower than all of them
  side by side, and that sum became the page's minimum and then the window's.
  The section view shared by the ERT and seismic pages contributed 1125 px, the
  seismic gather bar 817, the Project Map's surface row about 990, the EM
  section bar 784 and the sounding plot bar 757. These rows now wrap onto a
  second line in a narrow panel (`qt_apps.widgets.flow_layout.FlowLayout`),
  with a label kept on the same line as the control it names, so the window's
  minimum is now about 1650 to 1800 px, depending on fonts. The Project Map's
  layer panel no longer has a fixed 200 px minimum that clipped its Rename,
  Remove and Refresh buttons at that width.
- `PetrophysicsAgent` asked a model for parameters, discarded its reply and
  replaced the parameters the user had given with defaults whenever a
  geological context was set; that branch is gone.
- The single-survey report said "Default Archie parameters were used based on
  geological layer type" for what were generic guesses; it now states the
  parameters and ranges drawn, and whose they were.
- The Workflow page's report tab read "Results _report": its "&" was taken as
  a keyboard mnemonic. It now reads "Results & report".
- The ADTLERT windowed time-lapse inversion no longer runs out of memory on a
  long series. ADTLERT's window driver kept every survey's dense Jacobian it
  computed, up to 84 of them, until the run ended, and its forward kept the
  solved fields of the last eight models; on the DAS-1 example (942 readings,
  6024 cells, five surveys) the two held 3 GB and were reused for 4 of 54
  Jacobians and none of 59 field solves. The windows are now inverted one at a
  time through `invert_timelapse_log_resistivity`, with the same windows,
  stitching and progress, and the forward keeps no fields: the example's peak
  falls from 9.2 GB to 6.7 GB and stays flat however many windows follow, for
  the same models (they differ by 1e-4, as two ADTLERT runs do) and run time.
- An ADTLERT forward over topography no longer keeps what it needs only to set
  itself up. It solves the unit primary potentials and the numerical geometric
  factors on a P2 refinement of the mesh once, and kept both P2 meshes, their
  potentials and their solver factorizations for the rest of the run. They
  are released once their results are cached: the DAS-1 forward (5994 cells)
  holds 2.5 GB instead of 3.5 GB in the same time, with identical solves and
  Jacobians, and the windowed example above peaks at 5.6 GB. This applies to
  the single-survey ADTLERT inversion as well.
- The in-house time-lapse ERT inversion builds its spatial and temporal
  regularization operators sparse whatever `save_memory` says. The default mode
  stored them dense - (N C) x (N n) and (N - 1) n x N n arrays of almost only
  zeros, 5.6 GB at the 15000 unknowns below which the studio keeps that mode -
  although they only ever multiply vectors. Five DAS-1 surveys on 1181 cells
  now peak at 1.6 GB instead of 2.2 GB, with the same models to 1e-12.
- The in-house single-survey ERT inversion no longer makes its stacked
  Gauss-Newton system one dense array. It densified the smoothness operator
  (about 1.5 n^2 values) and, with a reference weight, an n x n identity on
  every iteration; the system is now held as its blocks
  (`solvers.linear_solvers.StackedSystem`), the Jacobian dense and the rest
  sparse, and an SPD solver's normal matrix is summed block by block. On the
  BERT example refined to 10779 cells the run needs 1.7 GB instead of 4.4 GB
  and 120 s instead of 160 s, with the same chi2. The GPU solvers still receive
  the assembled array.
- An auto-lambda sweep on the pyGIMLi engine keeps the ERTManager - Jacobian,
  meshes, forward operator - only of the runs that can still be exported: the
  fixed-lambda run, the best trial, and during a cold retry the warm sweep's
  best. Every trial used to keep its own; a five-trial sweep on the BERT
  example peaked at 4.4 GB and now at 2.9 GB, with the same lambda and chi2.
- `generalized_solver` takes a `LinearOperator`, and `scipy_lsqr`,
  `scipy_lsmr` and `precond_lsmr` no longer copy a dense system into CSR, 1.5
  times its size, before solving it.
- `ertforandjac2` scales the Jacobian in place, rather than through two more
  full copies of it per survey and iteration.
- The joint ERT+SRT inversion keeps its structural operators sparse. The
  cross-gradient neighbourhood matrix and the linearized blocks `B1`, `B2`
  were dense n x n arrays, and each of `B1`'s n rows was assembled from an
  n x n weight matrix, O(n^3) per Gauss-Newton step; each row now uses only
  its neighbours. The SRT ray Jacobian stays sparse, the stacked system is
  held as its blocks, an SPD solver gets its normal matrix (it used to fall
  back to LSQR, refusing the stacked system as not square), and the chi2
  checks no longer build Jacobians they throw away. On the shipped BERT and
  refraction lines the run takes 144 s instead of 1866 s at 2661 cells and
  127 s instead of 4565 s at 5515 cells, with peak memory 1.5 GB and 2.5 GB
  instead of 1.8 GB and 3.6 GB. The matrices are the same entry for entry
  (the cross-gradient blocks to 1e-10); the models differ only as LSMR's
  300-iteration truncation does under a change of summation order (chi2 to
  4e-4).

## [0.5.0] - 2026-09-28

0.5.0 is the first release since 0.3.0. Version 0.4.0 was never published, so
these notes cover everything since the `v0.3.0` tag (2026-07-09) and are written
for users upgrading from 0.3.0. Module paths below are relative to
`PyHydroGeophysX`.

### Highlights

- The AQUAH agent workflows run as a controller loop: it reads a shared run
  transcript, chooses one tool from a registry that wraps the existing agents,
  observes the result and chooses again, and the report names any requested
  product the run did not deliver.
- The desktop application is now the PyHydroGeophysX Studio, with new pages for
  request-to-report runs, joint inversion, a project map and saved results.
- EM line inversion can solve a whole line as one laterally constrained system,
  with a chi²-targeted regularization search, robust data errors and readers
  for TEMcompany projects, TEM2Go raw folders and tTEM acquisitions.
- ERT inversions accept a-priori resistivity zones and imported meshes
  (PyGIMLi, Gmsh, VTK, E4D), and an E4D-style tetrahedral mesh builder is
  included.
- Versioned workflow recipes run the same way from Python, the new
  `pyhydrogeophysx-workflow` command and the studio, and export as a runnable
  script plus a step-by-step walkthrough script and notebook.
- Time-lapse ERT, SRT and time-lapse SRT inversions solve the Gauss-Newton step
  with an exact Cholesky factorisation by default; time-lapse ERT adds
  interval-weighted temporal regularization and an optional temperature
  correction.

### Upgrading from 0.3.0

#### What breaks

These no longer work in 0.5.0. Where Python can report it, the error names
what to use instead.

- Python 3.10 or later is required. 0.3.0 declared 3.8, but the package no
  longer imports on 3.8 or 3.9, both past end of life; pip keeps those
  interpreters on 0.3.0.
- The desktop application is now the Studio, and nothing under the Workbench
  name remains. The launcher `pyhydrogeophysx-workbench` is now
  `pyhydrogeophysx-studio`, and upgrading removes the old command, so update
  shortcuts and scripts that start the application. In `qt_apps`,
  `main_window.PyHydroGeophysXWorkbench`, `WorkbenchController` and
  `WorkbenchState` are now `PyHydroGeophysXStudio`, `StudioController` and
  `StudioState`. The result file the web app reads is `full_studio_result.json`
  (0.3.0: `full_workbench_result.json`), and the window layout 0.3.0 saved is
  not carried over.
- `Geophy_modular.structure_integration.create_ert_mesh_with_structure`,
  `create_joint_inversion_mesh` and `integrate_velocity_interface` were removed,
  because none has a drop-in replacement. Importing one, through
  `Geophy_modular` or `structure_integration`, raises an `ImportError` naming
  the replacement, which is built on `core.mesh_utils.add_velocity_interface`.
- The `agents.fetch_climate_data` script module still imports, with a warning,
  but its `main()` raises and names `ClimateDataAgent.execute()`. Climate data
  now come from the Open-Meteo historical-weather API (ERA5), which needs
  neither a second conda environment nor a key; the `climate` extra no longer
  installs `pydaymet` or `xarray`.
- Without ResIPy installed, Syscal and Protocol DC/IP files no longer load. The
  built-in readers that 0.3.0 adapted from ResIPy were removed because of their
  GPL-3.0 licence, and independently written readers cover BERT/unified, E4D,
  ARES, DAS-1, Res2DInv, Sting, ABEM-Lund, Lippmann and Subsurface Insights
  files. The error says so and how to install the optional ResIPy
  (`pip install resipy`, GPL-3.0).

#### What warns

Everything else written for 0.3.0 keeps working in 0.5.0 and emits a
`DeprecationWarning` naming the replacement; the old names go in 0.6.0.

- The MODFLOW reader's keywords were renamed: `MODFLOWWaterContent(sim_ws=)` is
  `model_directory=`, `load_timestep()` and `load_time_range()` take `nlay=`
  instead of `nlay_uzf=`, and `model_output.water_content.binaryread()` takes
  `file=` instead of `file_obj=`. The old keywords are accepted with a warning;
  giving both is a `TypeError`. The reader's `sim_ws` attribute is
  `model_directory`, and `binaryread` returns what it returned in 0.3.0 (one
  `numpy.void` record for a structured dtype, `EOFError` at the end of the
  file).
- `ClimateDataAgent.fetch_climate_data_with_conda()` passes its configuration to
  `execute()`, writes `data/climate/climate_data.csv` and returns the 0.3.0
  result dictionary; the data come from ERA5, and variables ERA5 does not have
  (`srad`, `vp`, `dayl`) are named in its message.
- Names that 0.3.0 modules re-exported from elsewhere are importable from them
  again: `resistivity_to_saturation` from `Geophy_modular.ERT_to_WC` (it is
  `petrophysics.resistivity_models.resistivity_to_saturation2`, as in 0.3.0),
  `HydroModelOutput` from `model_output.modflow_output`, `DEFAULT_TL`,
  `INVERSION_TYPES`, `ert_load`, `ert_plot_style` and `io_utils` from
  `qt_apps.ert_timelapse`, `io_utils` from `qt_apps.em_pipeline` and
  `qt_apps.gravmag_pipeline`, and `gpu_available` from `solvers.solver`. The
  studio page modules in `qt_apps.modules` answer to the helpers 0.3.0 pages
  imported (`metrics_from_manager`, `pick_first_breaks`, `gmp`, `em_pipeline`,
  `PlanSliceView`, `TaskWorker`). The pages' own worker classes, such as
  `ERTInversionWorker`, were internal and are gone: runs now go through a
  separate workflow process.

- `WorkflowOrchestratorAgent`, `CodeGenerationAgent` and
  `GeophysicalInversionAgent` were removed. Importing them from `agents`, or from
  their 0.3.0 modules `agents.workflow_orchestrator_agent`,
  `agents.code_generation_agent` and `agents.geophysical_inversion_agent`, still
  works and emits a `DeprecationWarning` that names the replacement; creating one
  raises `RuntimeError` with the same message. The replacements are
  `BaseAgent.run_unified_agent_workflow()` (or `AgentCoordinator` for the fixed
  ERT pipeline), `workflows.export_workflow_bundle()`, and `ERTInversionAgent`,
  `SeismicAgent` or `TDEMAgent`, or `SRTInversion`, `TimeLapseSRTInversion`,
  `FDEMInversion` and `JointERTSRTInversion` called directly.
- These module paths still import, emit a `DeprecationWarning`, and will be
  removed in 0.6.0:

  | 0.3.0 module | Use instead |
  | --- | --- |
  | `qt_apps.em_pipeline` | `workflows.em1d` |
  | `qt_apps.gravmag_pipeline` | `workflows.gravmag` |
  | `qt_apps.ert_timelapse` | `inversion.time_lapse` |
  | `qt_apps.ert_load` | `data_processing.ert_io` |
  | `qt_apps.ert3d_pipeline` | `forward.ert3d` |
  | `qt_apps.ert_plot_style` | `visualization.ert_style` |
  | `qt_apps.geo_pipeline` | `Geophy_modular.ERT_to_WC` |
  | `qt_apps.hydro_pipeline` | `Hydro_modular.hydro_to_geophysics` |
  | `qt_apps.mesh3d_builder` | `core.mesh_3d` |
  | `qt_apps.raster` | `visualization.raster` |
  | `qt_apps.seismic3d_pipeline` | `Geophy_modular.structure_integration` |
  | `model_output.modflow_output` | `model_output.water_content` |

- These functions warn when called and will be removed in 0.6.0:
  `forward.tdem_forward.hydro_to_tdem` (also reachable as
  `forward.hydro_to_tdem`), replaced by `forward.simulate_tdem_sounding_from_hydro`
  for one column or `Hydro_modular.hydro_to_tdem` for a profile; and
  `solvers.solver.generalized_solver`, replaced by
  `solvers.linear_solvers.generalized_solver`, which `solvers` exports.
- Paths that appeared on the development branch after 0.3.0 and moved before
  this release warn the same way until 0.6.0: the top-level modules `em1d`,
  `gravmag`, `joint_api`, `table_io`, `_utils`, `_deprecations` and
  `_optional_dependencies`, and the helper names that `agents.base_agent`
  re-exports.

#### Numbers that change

Results computed with 0.3.0 defaults differ in these cases.

- `TimeLapseERTInversion` solves each Gauss-Newton step with
  `method='spd_cholesky'` (0.3.0: `'cgls'`), no longer lowers lambda between
  iterations (`lambda_rate` 1.0, was 0.8), and stops at chi² < 1.0 (was 1.5) or
  when the relative misfit change falls below 0.01 after at least 5 iterations.
  The temporal constraint is weighted by survey interval
  (`temporal_weighting='interval'`: the median interval over the pair's
  interval, capped at 10x either way); evenly spaced series are unaffected, and
  `temporal_weighting='uniform'` restores 0.3.0. With `method='cgls'` (or
  `lsqr`, `rrlsqr`, `rrls`) the least-squares solver keeps 0.3.0's limit of 2000
  iterations.
- `ERTInversion` keeps CGLS, but stops at chi² < 1.0 (was 1.5) or at a relative
  misfit change below 0.005 (was 0.01), and the plateau test waits for 5
  iterations.
- `SRTInversion` uses `spd_cholesky` (was `cgls`) and a plateau threshold of
  0.005 (was 0.01) after at least 5 iterations. `TimeLapseSRTInversion` uses
  `spd_cholesky` and keeps its 1.5 and 0.01 thresholds, with a new 5-iteration
  minimum.
- `WindowedTimeLapseERTInversion` normalises the temporal pair weights over the
  whole series and slices them into each window, and a run without a mesh builds
  one mesh from the electrodes of every survey and uses it for all windows
  (0.3.0 meshed each window from its first survey).
- `ESMDA` refuses inflation schedules whose sum of 1/alpha differs from 1 by
  more than 1e-3, such as `[1, 1, 1, 1]`, which assimilated the data four times.
  The default schedule is unchanged.
- `Hydro_modular.hydro_to_gravity`: mesh cells below the lowest interface carry
  no density contrast; 0.3.0 extrapolated the deepest layer into them. On a
  synthetic test profile, gz changed by up to 9.4 %.
- Petrophysics: `resistivity_to_saturation` and `resistivity_to_water_content`
  clip the saturation exponent `n` to [1, 4], and `resistivity_to_saturation`
  clips the cementation exponent `m` to [1, 4]; `resistivity_to_saturation2`
  keeps the caller's `n`. With surface conduction, n = 1 is solved exactly, and
  for 0 < n < 1 `resistivity_to_saturation2` returns the larger physical root,
  or NaN with a `RuntimeWarning` when no root lies in [0, 1]. The dry-end floor
  on the Archie estimate (0.001, and 0.01 in `resistivity_to_saturation2`) is
  gone. Other inputs give the 0.3.0 values.
- `analysis.compute_resolution_matrix` forms the data term as (Wd J)ᵀ(Wd J),
  which differs from 0.3.0's Jᵀ Wd Wd J when `Wd` is not symmetric.
  `analysis.compute_depth_of_investigation` starts its two inversions from
  `reference_resistivity` times `scale_low` and `scale_high` (default: the
  median apparent resistivity) rather than from 0.8 and 1.2 ohm-m.
  `uncertainty.linearized_posterior` uses Cholesky solves and returns a
  symmetric covariance.
- EM inversions pick their starting model from the data
  (`auto_starting_model=True`), and `invert_line` re-solves at other smoothness
  weights to reach `target_chi2` (`auto_lambda=True`); set both to False to fix
  the starting resistivity and the smoothness.

- `SRTInversion` applies `zWeight`. In 0.3.0 the value was set but never
  reached the constraint matrix, so every run smoothed isotropically whatever
  it was given. The default is now 1.0, which is that isotropic smoothing, so a
  default run is unchanged; a run that passes another value, such as 0.2, now
  gets anisotropic smoothing and a different model. `TimeLapseSRTInversion`
  always applied its `zWeight` and keeps its default of 0.2, and the agents'
  seismic steps, which call pyGIMLi's `TravelTimeManager` directly, are
  unaffected. `TimeLapseSRTInversion`'s line search now
  scores the same objective as its update, including the reference-model term,
  so runs that stopped early continue (in a test case chi² went from 6.40 to
  1.88 at lambda = 500).
- `PetrophysicalCoupling.compare_inversions_to_hydro` converts SRT velocities
  with the configured velocity model, default Hertz–Mindlin, which needs a
  calibrated `velocity_inverse`. Without one the result's `"srt"` entry is
  `{"error": ...}` with a warning, and ERT and EM are still compared. 0.3.0
  always applied a linear `v_dry`–`v_sat` mapping; pass
  `velocity_model="linear"` for it. Cells outside the model's range are now NaN
  and counted in `stats["n_masked"]` instead of being clipped to dry or
  saturated.
- FDEM forward modelling and `FDEMInversion` with several receivers pair each
  receiver's real and imaginary parts; 0.3.0 paired the real parts of
  neighbouring receivers, so multi-receiver data change. Single-receiver results
  are unchanged. The default receiver stays at the origin, except where 0.3.0
  gave no answer. A dipole source directly above or below the origin, as in the
  all-default survey, now puts the receiver 10 m along x (0.3.0 returned NaN),
  and a loop away from the origin puts it at the loop centre (0.3.0 raised).
  Unknown `waveform_type` names raise instead of silently meaning a dipole, and
  `CircularLoop` means a loop.
- In time-lapse ERT, `absoluteUError` is a voltage error in volts on both
  engines, as in pyGIMLi; the current is the file's `i`, or the new
  `absoluteCurrent` (default 0.1 A). Numbers change only for runs that set it.
- Reciprocal errors pair only readings whose dipoles are exchanged; repeats in
  the same direction are averaged as stacks, so the reciprocal-error filter no
  longer drops them. QC results change for surveys with repeated readings.

- If you used the temperature correction from the development branch: the
  seasonal `start_day` now dates the first survey rather than the origin of the
  numeric clock, and `simulate_temperature_1d` holds the surface node at the
  record from the first time step.

### Added

#### Agents and the unified runtime

- `agents.runtime`: a controller loop over a registry of tools that wrap the
  existing agents, a run transcript that keeps data and findings apart, recovery
  that retries a step which failed because of an unsuitable setting, and one
  loop that drives both "Auto to report" and step-by-step runs.
  `BaseAgent.run_unified_agent_workflow()` keeps its signature and its
  four-value return.
- The request parser asks the model which products the user wants; fuzzy
  English and Chinese phrase matching remains as the fallback when no model is
  available, and a report names any requested product that the run did not
  produce.
- Three-level model routing (Level 1 to 3, one model per provider at each level)
  in the studio chat settings and the Streamlit sidebar.
- The studio assistant can capture a live panel as an image for vision-capable
  models.
- Reports use one figure style and a figure set chosen from the request;
  desktop reports carry a processing record with input hashes and an
  uncertainty appendix.
- A read-only catalogue of selected data folders, local text retrieval over the
  package documentation and chosen folders with source paths, a generated
  reader for an unrecognised text table that runs only after approval, and a
  client for the bundled MCP server.

#### Desktop studio

- New pages: Data & reports (request, run and report in one workspace), Joint
  Inversion (guided two-method joint inversions), Map (a project-wide survey map
  with satellite or street basemaps and kriging, inverse-distance or
  triangulation surfaces) and Saved Results (a browser for the runs the project
  keeps).
- Every run is recorded in the active project's result store.
- Workflows run in their own process and can be paused and resumed; a
  pre-started standby process removes the interpreter start-up from each run.
- ERT page: a mesh tab that previews the inversion mesh and its zones before a
  run, a time-lapse series view with change from the baseline, a
  temperature-correction panel and an ADTLERT GPU check.
- EM page: a survey map, a gate-by-gate sounding view and a line overview. The
  3D Mesh Builder takes zones as boxes of known resistivity.
- ERT page: the Data QC checks start at working values instead of zero, which
  switched every one of them off: max error 20 %, and under More checks ρa ≤ 0
  dropped, |V| ≥ 10 µV, |I| ≥ 0.1 mA, contact resistance ≤ 30 kΩ, stacking
  spread ≤ 50 % and reciprocal error ≤ 5 %. The |k| cap follows each file
  loaded, at 20 times its median |k|, until it is set by hand, since |k| grows
  with the electrode spacing. None of them acts until Apply filter is pressed,
  and the log now says what each check dropped.
- ERT page, for data that record each reading's stacking spread (Subsurface
  Insights exports): a Max stacking spread check under More checks, and an
  optional error model, "Stacking spread with the estimate", which adds the
  spread to the estimate in quadrature (`error_source='stack'` in
  `run_ert_manager_inversion`). The spread within one reading usually overstates
  its uncertainty, so the default error model is unchanged. The BERT readers
  carry the spread back from the saved inversion input, so a walkthrough script
  reproduces such a run.

#### EM line inversion

- `inversion.em1d_lci`: laterally constrained inversion that solves every
  sounding on a line in one system; `invert_line` uses it when
  `lateral_smoothness` is positive (`lci_mode` selects `simultaneous`,
  `sequential` or `off`).
- Line inversion options: a chi²-targeted smoothness search, starting and
  reference models from neighbouring soundings, outlier rejection, bounded
  Huber reweighting of data errors (`inversion.robust_errors`), an optional
  resistive-background prior (`inversion.em1d_priors`) and blanking below the
  depth of investigation.
- Readers for TEMcompany projects in both project-file layouts, the reference
  inversions stored in them, TEM2Go raw `.stb` folders and tTEM `SKB`/`SPS`
  acquisitions.
- `JointFDEMTDEMInversion`: one model fitted to collocated FDEM and TDEM
  soundings.

#### ERT zones and meshes

- A-priori resistivity zones (`inversion.ert_zones`): polygons with a
  resistivity and an optional fixed flag. The inversion starts from and
  regularizes toward the zone model and holds fixed zones; `conform_to_zones`
  fits the mesh to the zone outlines and `decouple_zones` drops the smoothness
  constraint across them.
- Imported inversion meshes (`inversion.ert_mesh.load_inversion_mesh`):
  `.bms`, `.msh`, `.vtk`, `.vtu`, `.poly`, E4D `.node`/`.ele` meshes and E4D
  `.cfg` configurations, with `mesh_preview` to inspect a mesh before a run.
- `core.e4d_mesh`: reads and writes E4D mesh configuration files, builds the
  mesh the way E4D does (TetGen, or Gmsh), and reads E4D `.node`/`.ele` meshes.
- An optional ADTLERT backend for GPU time-lapse and windowed ERT inversion
  (`engine='adtlert'`, `adtlert` extra, CUDA 12).
- A chi²-targeted regularization search (`inversion.lambda_search`), switched on
  with `auto_lambda` for single and time-lapse ERT, SRT and the EM line
  inversion.

#### Sensitivity, resolution, assimilation and uncertainty

- Worked examples for sensitivity and resolution analysis, ensemble
  assimilation (EnKF and ES-MDA) and linearized posterior uncertainty, run on
  the Treeline catchment MODFLOW model shipped with the examples. The
  functions shipped in 0.3.0; how their results change is listed under
  Upgrading.
- `analysis.compute_depth_of_investigation(reference_resistivity=...)`.
- `ERTInversion(reference_weight=...)`: a smallness term pulling the model
  toward the reference model, weighted relative to the smoothness (default 0,
  smoothness alone). `compute_depth_of_investigation` needs it with
  `ERTInversion`: first-order smoothness ignores a homogeneous reference, so
  without it the two runs converge to one model and the index says nothing
  about the reference.
- `petrophysics.run_petrophysics_monte_carlo`: a seeded Monte Carlo conversion
  of ERT models to water content.

#### Workflows, CLI and code export

- `workflows`: versioned JSON recipes (`WorkflowSpec`, `load_recipe`,
  `save_recipe`), `run_workflow`, and a registry of 14 workflows covering single
  and time-lapse ERT, SRT, EM sounding and line inversion, gravity and
  magnetics, joint inversion, 3D ERT forward modelling, 3D meshes,
  hydrology-to-geophysics forward modelling, ERT to water content and seismic
  structure.
- `pyhydrogeophysx-workflow` with the subcommands `list`, `validate`, `run` and
  `export-code`.
- Code export: `generate_python` for an exact rerun, `generate_walkthrough` and
  `generate_notebook` for readable step-by-step code, and
  `export_workflow_bundle`, which writes the recipe, the runner and both
  walkthrough files.
- `pyhydrogeophysx-mcp`: an optional local stdio MCP server over the workflow
  registry (`mcp` extra).

#### Hydrology and other methods

- `model_input`: maps inverted properties to a hydrological grid and writes
  MODFLOW 6 or ParFlow inputs without modifying the source model.
- `JointGravityMagneticsInversion`: SimPEG cross-gradient joint inversion of
  gravity and magnetic data.
- `run_joint_inversion` and `get_joint_capabilities` dispatch registered
  two-method joint inversions.
- `petrophysics.temperature`: temperature correction of resistivity from a
  constant, a depth profile, a surface record (1D heat conduction) or a seasonal
  wave, applied to time-lapse ERT through `temperature_correction`.
- Plan-view gridding with ordinary kriging, inverse distance and triangulation
  (`core.plan_interpolation`); CSV export of mesh models
  (`data_processing.model_csv`); acquisition times read from file names
  (`data_processing.survey_timing`); and first-break propagation from a few
  hand picks (`data_processing.first_break_propagation`).

#### Examples

- `Ex_EM_line_section`, `Ex_TEM_LMHM_LCI`, `Ex_gravity_magnetics_inversion`,
  `Ex_MODFLOW_geophysics_feedback`, `Ex_sensitivity_analysis`,
  `Ex_ensemble_assimilation` and `Ex_posterior_uncertainty`, each as a script and
  a notebook; a notebook for `Ex_FDEM_workflow` and a script for
  `Ex_TL_inversion_memory`.

### Changed

- The desktop application is called the Studio throughout (see What breaks).
- `forward.TDEMSurveyConfig` keeps 0.3.0's fields first, in their 0.3.0 order,
  and the fields added since follow them, so 0.3.0 positional construction
  works.
- Qt-free code that 0.3.0 kept under `qt_apps` (the 1D EM, gravity and
  magnetics, time-lapse ERT, 3D mesh, 3D ERT forward, hydrology-to-geophysics
  and ERT-to-water-content pipelines) moved into the scientific packages, so
  scripts no longer import the desktop package; see the table under What warns.
- Default language models: Anthropic `claude-haiku-4-5` (0.3.0:
  `claude-sonnet-5`) and OpenAI `gpt-5.6-luna` (0.3.0: `gpt-4.1`), as Level 1 of
  the routing. Gemini uses the `google-genai` SDK; the older
  `google-generativeai` is still used where it is the only one installed.
- Climate data come from Open-Meteo (ERA5, 0.1 to 0.25 degrees) instead of
  Daymet (1 km, North America only), so climate values in reports differ.
- The DAS-1 reader was rewritten, and readers for Lippmann `.tx0`, Res2DInv,
  Sting and Subsurface Insights files were added (see What breaks for Syscal
  and Protocol files).
- `MeshCreator.create_from_layers` and `create_mesh_from_layers` take
  `bottom_elevation` (`bottom_depth` stays as an alias with the same elevation
  meaning) and `markers`, and reject invalid boundaries.
- Dependencies: `palettable` is required; the `gpu` extra installs
  `cupy-cuda12x`; `desktop` adds `pyproj` and `psutil`; new extras are `ert`
  (ResIPy), `adtlert`, `desktop-3d` (PyVista, VTK) and `mcp`.

### Deprecated

- The compatibility paths listed under What warns, to be removed in 0.6.0.

### Removed

- `WorkflowOrchestratorAgent`, `CodeGenerationAgent` and
  `GeophysicalInversionAgent`; their names still import, with a warning.
- `create_ert_mesh_with_structure`, `create_joint_inversion_mesh` and
  `integrate_velocity_interface` (importing one raises an `ImportError` naming
  the replacement), and the `agents.fetch_climate_data` script, whose `main()`
  now raises; `ClimateDataAgent.fetch_climate_data_with_conda()` still works,
  with a warning.
- The built-in Syscal and Protocol DC/IP readers adapted from ResIPy; with
  ResIPy installed those files load as before.
- The examples `Ex_ERT_data_process` and `Ex_cross_constraints.py` and the
  climate helper scripts `fetch_climate.bat` and `setup_climate_env.bat`. The
  launch scripts moved: `examples/start_webapp.bat`/`.sh` are now
  `PyHydroGeophysX/start_webapp.bat`/`.sh`, and `scripts/start_qt_workbench.bat`/
  `.sh` are replaced by `start_studio.bat`/`.sh` in `examples/` and the package
  folder.

### Fixed

- `ERTForwardModeling.create_synthetic_data` declared `cls` without
  `@classmethod`, so calling it on the class needed a dummy first argument; it
  is a classmethod now.
- `ERTInversion` no longer erases the apparent resistivities of a file that
  stores `rhoa` without `r` when it recomputes geometric factors; such a run
  returned 1e-6 ohm-m.
- The Geometrics DAT (SEG-2) reader decodes samples by the trace descriptor's
  data format code. 0.3.0 read that byte as a sample size, which decoded 16- and
  32-bit integer traces wrongly; a record whose traces disagree on sample
  interval, length or delay now raises instead of taking the last trace's
  values.
- `InversionEvaluationAgent` had its chi² ranges for "poor" and "overfit"
  swapped.
- `uncertainty.propagate_petro_uncertainty` always returns `"cov"` as a 2-D
  matrix.

#### Fixes from the 0.5.0 review

Agents and reports:

- `AgentCoordinator` no longer fails after the inversion when a step's result
  cannot be pickled (it holds a pyGIMLi mesh). Checkpoints are best-effort, and
  a truncated checkpoint is re-run rather than loaded.
- A failed seismic step is no longer used as a structure constraint or counted
  as done; the run warns and the report lists it.
- `run_unified_agent_workflow` returns `status="incomplete"` when a step failed
  and was not recovered, even after writing a report. Reports gain a "Not
  delivered" section giving the reason for each failed step and each requested
  product the run did not produce, climate data included.
- Reports leave out empty fields instead of printing "None" or "N/A", give chi²
  to three significant figures, name the inversion method and solver that ran,
  compute the time-series recommendation from the actual series, and say a
  language model wrote the text only when one did.
- Seismic settings from the request (`seismic_params`, and the seismic stage's
  `lam`, `z_weight` and the like) reach `SeismicAgent`.
- Without an API key, or when the model call fails,
  `ContextInputAgent.parse_request` falls back to the deterministic parser with
  a warning. The unified workflow runs the files a request names when the
  configuration names none.
- A water-content request on TDEM data alone is answered from the sounding's
  layered model, with an uncertainty caveat; a missing water content states its
  reason.
- Instrument names in requests and data-file headers match as whole words, so
  "existing", "Albert" or "Canadas" no longer select the Sting, BERT or DAS-1
  reader, and "Terrameter" is recognised.
- The agents no longer force `method='cgls'`; the library's solver default
  applies unless the configuration names one. Lambda trials no longer ask the
  model for an interpretation; only the kept model is interpreted.
- Folder classification in the studio routes the chosen provider (Claude,
  OpenAI, OpenAI-compatible) correctly, and refuses an unsupported one clearly.

Web apps:

- The No-LLM quick modes produce water content when the request asks for it,
  in any wording the runtime recognises, typos and Chinese included, and the
  confirm form has a checkbox for it. They no longer preset DAS-1 or E4D, so an
  instrument the request does not name is read from the file header.
- A failed re-run no longer leaves the previous run's "Workflow complete.",
  plan and downloads on the page; the closing message follows the run's own
  status, and the results summary shows the resistivity and water-content
  ranges.
- Demo mode ships its cached results in `examples/demo_cache/` (rebuilt with
  `python examples/demo_cache/make_demo_cache.py`) and reports a missing cache
  plainly.
- `pyhydrogeophysx-gui` and `python -m PyHydroGeophysX.gui_mesh3d` explain,
  from a pip install, that the web apps ship with the source repository, and
  accept an app path as their first argument.

Desktop studio:

- Saved Results opens ERT results in a project whose path the Windows codepage
  cannot represent, such as a Chinese folder name; `InversionResult` and
  `TimeLapseInversionResult` save and reload their mesh in such folders, and the
  seismic structure helpers read their meshes there.
- A moved or renamed project reopens its runs: records store run-relative
  paths, and older records are rebased onto the run's current folder.
- The ERT page's reciprocal error uses the library's sign-aware pairing; a pair
  written as (M,N,B,A) no longer scores 200 %.
- Geophone files with a station column are ordered by station, so the first
  position and the spacing are right when rows are out of order.
- The seismic page's geophone spacing and geophone-0 x now move the picks
  already made, and the Travel-time plot follows. A pick kept the position
  stamped when it was made, so a spacing set afterwards reached neither the
  plot, the exported travel times nor the SRT inversion: picks taken at the
  default 1 m stayed 1 m apart after the spacing was set to 0.5 m. The plot also
  marks the geophones, so it shows the line before there are any picks.
- The Seismic → Structure example lines are no longer mirrored; the interface
  had been placed 3 to 12 m from the 3-D truth.
- A TDEM line section keeps the map coordinates of its geometry file, so Add to
  Map offers "Coordinates carried by this result".
- A workflow process that crashes natively or runs out of memory (for example
  in CHOLMOD) is reported in plain words with what to try, and the last lines
  of each run's output are kept in `logs/workflow_output.log`.
- When example data is absent, "Use example" says it ships with the source
  repository; the studio also finds a clone it was started from, or one named
  by the new `PHGX_EXAMPLES_DIR` environment variable.
- Saved Results lists each file once and names a single inversion as the ERT
  page does. The window icon is included in the wheel.
- An electrode file now places the electrodes of a time-lapse run. The run
  read each survey's own header while the page previewed the electrode file's
  positions; the file is now kept with the run, and the walkthrough script
  passes it on. The time-lapse walkthrough also defines the survey files and
  times it uses, which it never did for a studio run, so it stopped at a
  NameError.
- An electrode file loaded after the data reads the data again through the same
  reader as loading it first, and a filter already applied is applied again. Laid
  over the loaded data, a file one row short silently dropped that electrode and
  every reading on it; it is now refused, as the readers refuse it.
- Discarding a run no longer fails with WinError 32. The next run's standby
  process waited in the folder of the run that started it, which Windows will
  not delete; it now waits in the package's own folder. A discard that Windows
  refuses part way no longer leaves the run to be saved without its files: the
  run is discarded, and what is left of its folder is offered for removal when
  the project next opens.

Library:

- An electrode table with two electrodes at one position no longer shifts the
  later electrodes onto wrong positions with wrong geometric factors; a reading
  that uses both is refused with a clear error.
- Electrodes moved by an electrode file, or in the studio's electrode table,
  keep each reading's measured transfer resistance: k and the apparent
  resistivity follow the electrodes. A file that reports apparent resistivity
  kept the values formed on its header's positions, so doubling the spacing
  doubled k and halved the resistance the data implied. `load_ert_container`
  places the electrodes from `electrode_file` when PyGIMLi's own reader reads the
  file, and refuses a file listing a different number of electrodes; it used to
  return the header's positions. `normalize_for_timelapse` and
  `run_timelapse_ert` take `electrode_file`.
- `interpolate_to_mesh` accepts any number of layers, not only 14.
- The Waxman–Smits inverses reject a negative `sigma_sur` instead of returning
  a number, and `resistivity_to_saturation` accepts 2-D input.
- `hydro_to_ert(save_path=...)` writes the synthetic data it documents, and
  `hydro_to_ert` and `hydro_to_srt` no longer print the whole layer-ID array on
  every call; with `verbose=True` they print the layer markers.
- `Hydro_modular` imports without PyGIMLi; `hydro_to_ert` and `hydro_to_srt`
  load it on first use.
- An import error caused by a missing module of the package itself names that
  module instead of suggesting `pip install PyHydroGeophysX`.
- ERT instrument detection, the removed agents' 0.3.0 module paths and the
  deprecation messages were brought in line with this release (see Upgrading).

Examples and documentation:

- `EX_SRT_forward` writes to `examples/results/srt_example/` and no longer
  overwrites the shipped `data/Seismic/synthetic_seismic_data.dat`, which was
  refreshed to match the current code (its travel times had drifted by a median
  6 %, up to 58 %).
- `Ex_3D_ERT_forward` renders its 3-D views off screen, so it runs with PyVista
  installed; `Ex_model_output.ipynb` loads a timestep that exists; the
  `Ex_TL_inversion`, `Ex_structure_TLresinv`, `Ex_Time_lapse_measurement` and
  `Ex_MC_Hydro` notebooks read the same files as their scripts; and the seismic
  part of `Ex_multi_agent_workflow.ipynb` runs on the shipped line.
- Examples that need another example's output, or long run times or much
  memory, say so in their header and in the gallery index.
- The README, installation page and agent guides no longer point to examples
  and scripts that do not exist, say that examples, their data and the web apps
  come with the source repository rather than the pip package, and describe
  demo mode as it is. The API snippets for the 3D mesh builder,
  `qc_and_visualize`, `export_for_inversion` and `TDEMSurveyConfig` match the
  code, and the new examples' gallery pages show their figures.

## [0.3.0] - 2026-07-09

- Desktop Workbench downloads for Windows and macOS, in light and full builds,
  and automatic deployment of the documentation site.
