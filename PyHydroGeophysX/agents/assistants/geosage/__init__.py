"""GeoSAGE: geological reasoning from joint gravity and magnetic inversion.

Geological Semantic Analysis for Geophysical Exploration (Fang et al., in
review; source at https://github.com/ZhengyangFang/GeoSAGE, MIT licence). A
user states an exploration objective; GeoSAGE prepares the gravity and
magnetic data, compiles geological and petrophysical priors, runs a
deterministic joint inversion (SimPEG, cross-gradient coupled), turns the
density and susceptibility volumes into a quasi-geological model (prior-guided
units and Geo groups, or GMM/BIC when priors are thin), ranks targets, writes
the interpretation and has it reviewed before it is final.

**Status: being ported.** This package is the scaffold the port fills in. The
pipeline is laid out as tools in :mod:`.tools` - one per stage of the paper's
Figure 1, with what each requires and produces - and :mod:`.workflow` runs
them on the same controller as AQUAH, so the studio's chat, live timeline,
step approval and report view work for GeoSAGE as soon as the tools do. Until
then it is listed in the studio's assistant menu but cannot be selected; set
``status="ready"`` below once the tools run.

The paper's seven agents map onto this package as follows:

=================  ====================================================
ContextAgent       :func:`.workflow.configure` (request -> configuration)
DataAgent          ``prepare_data`` tool
PetrologyAgent     ``compile_priors`` tool, and unit/group naming inside
                   ``build_quasi_geology``
InversionAgent     ``run_joint_inversion`` tool (settings), SimPEG backend
GeoAgent           ``build_quasi_geology`` tool (grouping settings)
ReportAgent        ``write_report`` tool (draft, and revision on review)
ReviewAgent        ``review_report`` tool; its approval is the final report
=================  ====================================================
"""

from PyHydroGeophysX.agents.assistants import Assistant

PERSONA = """\
- GeoSAGE works from gravity and magnetic data toward a geological interpretation: data preparation, joint gravity–magnetic inversion for density and susceptibility, a quasi-geological model, target ranking and a reviewed report. Use gravmag to load and inspect gravity/magnetic data, joint_inversion for the joint inversion, and mesh3d or model_viewer to look at 3D models.
- Keep the physics and the interpretation apart: inversion settings change the density and susceptibility models; geological grouping and target ranking change only the interpretation of them. Say which one a request affects.
- Never state a depth, volume or physical-property value that a tool result did not report. Depths are below the topographic surface unless a result says otherwise; do not read a mesh-layer index as a depth.
"""

ASSISTANT = Assistant(
    key="geosage",
    name="GeoSAGE",
    domain="Geological modeling",
    title="Geological Semantic Analysis for Geophysical Exploration",
    summary=("Turns joint gravity–magnetic inversion models into a quasi-geological "
             "model, ranked exploration targets and a reviewed interpretation report."),
    workflow="PyHydroGeophysX.agents.assistants.geosage.workflow:run",
    tools="PyHydroGeophysX.agents.assistants.geosage.tools:TOOLS",
    persona=PERSONA,
    examples=(
        "invert the gravity and magnetic grids jointly and build a quasi-geological model",
        "find serpentinite targets for natural hydrogen from this gravity and TMI data",
        "rank Cu–Ni–PGE targets in the mafic intrusion and write a reviewed report",
    ),
    input_roles=(
        ("Gravity data", "gravity_file"),
        ("Magnetic data (TMI)", "magnetic_file"),
        ("Terrain / topography", "topography_file"),
        ("Borehole lithology / petrophysics", "borehole_file"),
        ("Geological reference document", "reference_file"),
    ),
    studio_modules=("one_click", "gravmag", "joint_inversion", "mesh3d",
                    "model_viewer", "project_map"),
    # Earth and mineral rather than water: orange, yellow, green, teal.
    colors=("#ff9500", "#ffcc00", "#34c759", "#30b0c7"),
    requires_packages=("simpeg",),
    status="in development",
    status_note=("GeoSAGE is being ported into PyHydroGeophysX; its tools are not "
                 "implemented yet. See the developer guide, 'Adding an assistant'."),
)
