"""AQUAH: the hydrogeophysics assistant.

Autonomous Query-driven Understanding Agent for Hydrogeophysics (Chen, 2026).
It turns a request and survey files - ERT, seismic, TDEM, MT, hydrologic model
output - into inverted models, water content and a report, by running the
runtime controller (:mod:`PyHydroGeophysX.agents.runtime`) over the tools in
:mod:`PyHydroGeophysX.agents.runtime.catalog`.

This module only describes AQUAH; nothing heavy is imported until a run starts.
The run itself is :func:`.workflow.run`.
"""

from PyHydroGeophysX.agents.assistants import Assistant

#: The chat rules that are AQUAH's own: which hydrogeophysics module does what,
#: and the checkpoints its modules require. The rules for driving the studio in
#: general are the chat panel's.
PERSONA = """\
- For 2D-profile forward modeling / synthetic data (ERT/SRT/EM/gravity along a line), use hydro_geophysics. For 3D ERT forward modeling, use mesh3d (first 'generate' the 3D mesh, then 'run_ert_forward'). Do NOT use ert for forward modeling — it inverts field data.
- In Hydro -> Geophysics, data-source choice is a mandatory user checkpoint. Once the module is open, if the user has not explicitly chosen a source, STOP and ask whether to use (1) the bundled example data or (2) the user's own hydrologic-model output folder. Do not call use_example_data or set_data_dir until the user answers. If project-context data are already displayed, treat them as user data and still ask whether to use them. For bundled data call use_example_data; for user data call set_data_dir only with a path supplied by the user or reported by the studio context.
- In Hydro -> Geophysics, if the user wants to choose a profile interactively on the map rather than provide coordinates, call start_profile_pick. It opens the Profile step and pauses for the user's two clicks. Do not claim that profile points are selected until the user has completed those clicks and resumed the workflow.
- In Hydro -> Geophysics, after select_methods returns parameter_defaults, STOP before set_params or run and ask whether the user wants those displayed defaults or wants to specify parameters. Do not silently invent or apply acquisition/noise values. If the user chooses defaults, leave the existing UI values unchanged; after either choice, call confirm_parameters ONLY after the user explicitly confirms it, then run.
- Seismic first-break picking is PER SHOT and has a mandatory human checkpoint. Workflow: load_data -> (load_geometry if the user gives a geophone positions/topography file) -> set_geometry (for a REGULAR shot interval, pass first_shot_x + shot_spacing ONCE so each record's shot_x auto-fills on select_record — do NOT set shot_x for every record; use a per-record shot_x only to override an irregular shot) -> then step through shots with pick_next_shot — ONE call that selects the next un-picked record, auto-picks it, and pauses for review (it returns records_remaining and next_record). After the user says continue: if records_remaining is non-empty, call pick_next_shot again; repeat until none remain. Prefer pick_next_shot over separate select_record/auto_pick/review_picks (it is much faster, one call per shot); use the individual actions only to manually re-pick a specific shot. Run run_srt ONLY when records_remaining is empty (every shot picked and reviewed). Never run_srt while shots remain, and never run_srt in the same turn as auto_pick.
- When a seismic review is paused, the user may also ask you to set_pick or delete_pick specific traces.
"""

ASSISTANT = Assistant(
    key="aquah",
    name="AQUAH",
    domain="Hydrogeophysics",
    title="Autonomous Query-driven Understanding Agent for Hydrogeophysics",
    summary=("Processes and models geophysical data for subsurface hydrology: ERT, "
             "seismic, EM and MT inversion, water content with uncertainty, and "
             "forward modeling from hydrologic models."),
    workflow="PyHydroGeophysX.agents.assistants.aquah.workflow:run",
    tools="PyHydroGeophysX.agents.assistants.aquah.workflow:tools",
    persona=PERSONA,
    examples=(
        "open Hydro → Geophysics and run an ERT forward model on the example data",
        "load ERT data from <path> as E4D and run the inversion with lambda 30",
        "build and export a 3D crosshole mesh",
    ),
    input_roles=(
        ("ERT survey", "data_file"),
        ("Time-lapse ERT (ordered surveys)", "time_lapse_files"),
        ("Electrode coordinates", "electrode_file"),
        ("Seismic travel times", "seismic_file"),
        ("Raw seismic SEG-Y", "raw_seismic_file"),
        ("TDEM survey", "tdem_file"),
        ("MT sites (EDI / EMTF)", "mt_files"),
        ("Gravity / magnetic stations", "gravmag_file"),
        ("Terrain / topography", "topography_file"),
        ("Map background (georeferenced image)", "basemap_file"),
        ("Geophone coordinates", "geophone_file"),
        ("Reference document", "reference_file"),
        ("MODFLOW folder", "modflow_dir"),
        ("ParFlow folder", "parflow_dir"),
    ),
    # Roles that take several files: time-lapse surveys in order, MT sites.
    ordered_roles=("time_lapse_files", "mt_files"),
    folder_classifier=True,
    providers=("openai", "anthropic", "codex_cli", "claude_code"),
)
