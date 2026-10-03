"""GeoSAGE's tools: one per stage of the workflow, with what each needs and makes.

Each tool is offered to the controller only once what it ``requires`` exists,
so the order below is a consequence of the data, not a script. The handlers
are placeholders for the port from https://github.com/ZhengyangFang/GeoSAGE;
each says what it has to do and what it must return. A handler takes the
:class:`~PyHydroGeophysX.agents.runtime.context.RunContext` and returns
``(summary, outputs)``: one sentence a user can read, and a dict of the
artifacts named in ``produces``. ``ctx.config`` holds the configuration from
:func:`.workflow.configure`; ``ctx.get(key)`` reads an earlier tool's artifact;
``ctx.output_dir`` is where files go; ``ctx.settings`` has the model access for
the LLM-based agents (``api_key``, ``model``, ``llm_provider``).

The artifact keys are the contract between the stages:

``field_data``          gridded gravity and magnetic data on the study area,
                        with bounds, fields and uncertainties (DataAgent)
``geological_priors``   lithologic classes, physical-property ranges and
                        grouping rules, or None when none are available
                        (PetrologyAgent)
``property_models``     density and susceptibility on the inversion mesh,
                        with fit statistics and the mesh (InversionAgent)
``geo_model``           unit and Geo-group labels per cell, their statistics,
                        and the figures (GeoAgent, PetrologyAgent)
``draft_report``        the interpretation and target ranking (ReportAgent)
``report_files``        the reviewed report, ``{"report_markdown": path}``;
                        its existence is what lets the run finish (ReviewAgent)
"""

from typing import Any, Dict, Tuple

from PyHydroGeophysX.agents.runtime.context import RunContext
from PyHydroGeophysX.agents.runtime.tools import Tool

GUIDE = "docs/source/agents/adding_an_assistant.rst"


def _not_ported(stage: str):
    def handler(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError(
            f"GeoSAGE's {stage} has not been ported into PyHydroGeophysX yet; "
            f"see {GUIDE}.")
    handler.__name__ = stage
    return handler


TOOLS: Dict[str, Tool] = {tool.name: tool for tool in (
    Tool("prepare_data",
         "Read the gravity and magnetic data, set the study-area bounds and data "
         "fields, and grid them with their uncertainties for inversion.",
         _not_ported("prepare_data"), produces=("field_data",),
         agent="DataAgent", label="Prepare gravity and magnetic data",
         module="gravmag"),
    Tool("compile_priors",
         "Compile geological and petrophysical prior information for the study "
         "area: lithologic classes, their density and susceptibility ranges, and "
         "rules for grouping them. Produces None when nothing reliable is known, "
         "in which case grouping falls back to GMM/BIC.",
         _not_ported("compile_priors"), produces=("geological_priors",),
         agent="PetrologyAgent", label="Compile geological priors"),
    Tool("run_joint_inversion",
         "Jointly invert gravity and magnetic data for density and magnetic "
         "susceptibility on a common mesh (deterministic, SimPEG, cross-gradient "
         "coupling), and report the data fit.",
         _not_ported("run_joint_inversion"), requires=("field_data",),
         produces=("property_models",), agent="InversionAgent",
         label="Run joint gravity-magnetic inversion", module="joint_inversion"),
    Tool("build_quasi_geology",
         "Partition the density and susceptibility models into units (from the "
         "priors, or GMM/BIC when there are none), name them and merge them into "
         "a few Geo groups, then clean the label model for spatial coherence. "
         "Changes the labels only, never the inverted properties.",
         _not_ported("build_quasi_geology"), requires=("property_models",),
         produces=("geo_model",), agent="GeoAgent",
         label="Build quasi-geological model", module="mesh3d"),
    Tool("write_report",
         "Write the geological interpretation: the anomaly systems, the Geo "
         "groups, ranked exploration targets with depths and evidence, and the "
         "uncertainties.",
         _not_ported("write_report"), requires=("geo_model",),
         produces=("draft_report",), agent="ReportAgent",
         label="Write interpretation report"),
    Tool("review_report",
         "Check the draft report against the numerical outputs - numbers, units, "
         "coordinates, evidence, overstatement, terminology - have the report "
         "revised where it fails, and issue the reviewed report.",
         _not_ported("review_report"), requires=("draft_report",),
         produces=("report_files",), agent="ReviewAgent",
         label="Review the report", module="one_click"),
)}
