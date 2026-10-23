"""The agent workflows run what was asked, and say so when they cannot.

* AgentCoordinator: the preview lists the steps the run makes, with the files
  and parameters the run uses; a time-lapse configuration inverts every survey;
  what the pipeline has no step for is refused by preview and run alike; and
  what a request asks for that the run does not do is reported, not dropped.
* The deterministic request parser, which the coordinator runs: the files a
  request names, the instruments and the methods it asks for or rules out.
* Water content from a layered model: parameters given per named layer are
  used, what a request leaves out is filled in and said, and a conversion that
  fails is reported as failed rather than as not requested.
* Agent names and module paths removed in 0.5.0 warn with their replacement.
"""

import importlib
import sys

import numpy as np
import pytest

from PyHydroGeophysX.agents import AgentCoordinator
from PyHydroGeophysX.agents._intent import names_tdem
from PyHydroGeophysX.agents.context_input_agent import ContextInputAgent


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    # These tests never ask a model; agents take a key only when given one, and
    # an empty environment keeps the once-a-process key warning out of the way.
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------
# AgentCoordinator
# --------------------------------------------------------------------------

class _Recorder:
    """A stand-in agent that records what it was given."""

    def __init__(self, name, calls, reply):
        self.name, self.calls, self.reply = name, calls, reply
        self.llm_usage_ledger = []

    def execute(self, input_data):
        self.calls.append((self.name, "execute", input_data))
        return self.reply(input_data)


class _Report(_Recorder):
    def generate_timelapse_report(self, input_data):
        self.calls.append((self.name, "generate_timelapse_report", input_data))
        return {"status": "success", "report_file": "time_lapse_report.md"}


def _coordinator(tmp_path, calls, seismic=False):
    n_cells = 6

    def load(data):
        return {"status": "success", "ert_data": f"survey:{data['data_file']}"}

    def invert(data):
        if data.get("inversion_mode") == "time-lapse":
            models = np.full((n_cells, len(data["time_lapse_data"])), 100.0)
            return {"status": "success", "inversion_mode": "time-lapse",
                    "final_models": models, "mesh": "mesh",
                    "coverage": [np.ones(n_cells)] * models.shape[1]}
        return {"status": "success", "resistivity_model": np.full(n_cells, 100.0),
                "mesh": "mesh", "coverage": np.ones(n_cells)}

    def water_content(data):
        model = np.asarray(data["inversion_results"]["resistivity_model"])
        return {"status": "success", "water_content_mean": np.full(model.shape, 0.2),
                "water_content_std": np.full(model.shape, 0.01), "output_dir": "wc"}

    coordinator = AgentCoordinator(api_key=None, output_dir=str(tmp_path))
    coordinator.register_agent("ert_loader", _Recorder("ert_loader", calls, load))
    coordinator.register_agent("ert_inversion", _Recorder("ert_inversion", calls, invert))
    coordinator.register_agent("water_content", _Recorder("water_content", calls, water_content))
    coordinator.register_agent("report", _Report("report", calls, lambda data: {
        "status": "success", "report_file": "workflow_report.md"}))
    if seismic:
        coordinator.register_agent("seismic_processor", _Recorder(
            "seismic_processor", calls, lambda data: {"status": "success"}))
    return coordinator


_PLAN_AGENT = {"ERTLoaderAgent": "ert_loader", "ERTInversionAgent": "ert_inversion",
               "WaterContentAgent": "water_content", "ReportAgent": "report",
               "SeismicAgent": "seismic_processor"}


def _inputs(calls, agent):
    return [data for name, _, data in calls if name == agent]


@pytest.fixture
def no_file_checks(monkeypatch):
    # The preview checks that the files exist; the run's stand-ins do not care.
    monkeypatch.setattr(AgentCoordinator, "_validate_preview_files", lambda self, cfg: ([], []))


def _preview_and_run(tmp_path, config, seismic=False):
    """Preview and run one configuration; the run makes exactly the planned steps."""
    calls = []
    coordinator = _coordinator(tmp_path, calls, seismic=seismic)
    preview = coordinator.preview_workflow(dict(config))
    result = coordinator.execute_workflow(dict(config))
    planned = [_PLAN_AGENT[step["agent"]] for step in preview["execution_plan"]]
    assert planned == list(dict.fromkeys(name for name, _, _ in calls))   # a time-lapse load repeats
    return preview, result, calls


def test_time_lapse_inverts_every_survey(tmp_path):
    files = ["2021-10-08_1400.ohm", "2021-11-08_1400.ohm", "2021-12-08_1400.ohm"]
    # ContextInputAgent's shape: data_file is the baseline of the list.
    _, result, calls = _preview_and_run(tmp_path, {
        "data_file": files[0], "time_lapse_files": files, "inversion_mode": "time-lapse"})

    assert result["status"] == "success", result.get("error")
    assert [data["data_file"] for data in _inputs(calls, "ert_loader")] == files
    inversion = _inputs(calls, "ert_inversion")[0]
    assert inversion["time_lapse_data"] == [f"survey:{f}" for f in files]
    # The water-content agent receives cells x surveys, the report one entry per survey.
    assert np.asarray(_inputs(calls, "water_content")[0]["inversion_results"]
                      ["resistivity_model"]).shape == (6, 3)
    method, report = next((m, d) for name, m, d in calls if name == "report")
    assert method == "generate_timelapse_report"
    assert len(report["inversion_results"]["time_lapse_water_content"]) == 3


@pytest.mark.parametrize("config, water_content", [
    ({"data_file": "a.ohm"}, True),
    ({"data_file": "a.ohm", "user_request": "estimate the water conent"}, True),
    ({"data_file": "a.ohm", "user_request": "帮我算含水量"}, True),
    ({"data_file": "a.ohm", "user_request": "ERT inversion only"}, False),
    ({"data_file": "a.ohm", "convert_to_water_content": False}, False),
    ({"time_lapse_files": ["a.ohm", "b.ohm"], "user_request": "resistivity only"}, False),
    ({"time_lapse_files": ["a.ohm", "b.ohm"]}, True),
])
def test_the_run_makes_the_steps_the_preview_lists(tmp_path, no_file_checks, config, water_content):
    preview, result, calls = _preview_and_run(tmp_path, config)
    assert result["status"] == "success", result.get("error")
    assert ("water_content" in {name for name, _, _ in calls}) is water_content
    resolved = preview["workflow_config"]
    assert _inputs(calls, "ert_inversion")[0]["inversion_params"] == resolved.get("inversion_params", {})


@pytest.mark.parametrize("config", [
    {"tdem_file": "sounding.csv"},
    {"data_file": "a.ohm", "em_file": "sounding.csv"},
    {"data_file": "a.ohm", "seismic_file": "picks.dat"},
    {"data_file": "a.ohm", "raw_seismic_file": "line.sgy"},
    {"data_file": "a.ohm", "mt_files": ["site.edi"]},
    {"user_request": "Invert the TDEM sounding in sounding.csv"},
    {"data_file": "a.ohm", "user_request": "Invert a.ohm and the TEM sounding"},
    {"user_request": "Invert the seismic travel times in srt.dat"},
    {"data_file": "a.ohm", "user_request": "Invert a.ohm and pick the first breaks in line.sgy"},
])
def test_what_the_pipeline_cannot_run_is_refused_by_preview_and_run(
        tmp_path, no_file_checks, config):
    preview, result, calls = _preview_and_run(tmp_path, config)
    assert preview["status"] == "failed" and preview["execution_plan"] == []
    assert "run_unified_agent_workflow" in preview["validation_errors"][-1]
    assert result["status"] == "failed" and "run_unified_agent_workflow" in result["error"]
    assert calls == []


@pytest.mark.parametrize("config", [
    # The documented seismic constraint, with the request naming its file.
    {"data_file": "a.ohm", "use_seismic": True, "seismic_data": "srt.dat",
     "user_request": "Invert a.ohm with the seismic structure constraint from srt.dat"},
    # The request alone names it.
    {"data_file": "a.ohm",
     "user_request": "Invert a.ohm with the seismic structure constraint from srt.dat"},
    # The ERT survey is a .dat file too, and is not taken for the seismic one.
    {"data_file": "line2.dat",
     "user_request": "Invert the ERT survey line2.dat with the seismic velocities in srt.dat"},
])
def test_a_request_that_names_the_seismic_file_runs_the_seismic_constraint(
        tmp_path, no_file_checks, config):
    # The request parser's guess at a seismic file used to refuse the whole run.
    _, result, calls = _preview_and_run(tmp_path, config, seismic=True)
    assert result["status"] == "success", result.get("error")
    assert _inputs(calls, "seismic_processor")[0]["seismic_data"] == "srt.dat"
    assert _inputs(calls, "ert_inversion")[0]["use_structure_constraint"] is True


def test_the_request_parsers_defaults_do_not_replace_the_runs(tmp_path, no_file_checks):
    # The parser fills in lambda 20, 10 iterations and an uncertainty run for a
    # preview; taken into the run, they replaced the time-lapse defaults.
    _, result, calls = _preview_and_run(tmp_path, {
        "time_lapse_files": ["a.ohm", "b.ohm"],
        "user_request": "time-lapse ERT of the two surveys and their water content"})
    assert result["status"] == "success", result.get("error")
    assert _inputs(calls, "ert_inversion")[0]["inversion_params"] == {}
    # A water content always carries its uncertainty, whatever the parser said.
    assert _inputs(calls, "water_content")[0]["uncertainty_analysis"] is True


def test_a_survey_the_loader_refuses_stops_the_run_with_its_reason(tmp_path):
    calls = []
    coordinator = _coordinator(tmp_path, calls)
    coordinator.register_agent("ert_loader", _Recorder("ert_loader", calls, lambda data: {
        "status": "needs_review",
        "summary": "The declared ERT instrument does not match the file header.",
        "error_fix_hint": "Change instrument to 'Syscal'."}))
    result = coordinator.execute_workflow({"data_file": "a.ohm", "instrument": "Sting"})
    # It used to fail one step later with a bare KeyError 'ert_data'.
    assert result["status"] == "failed" and [name for name, _, _ in calls] == ["ert_loader"]
    assert "does not match the file header" in result["error"]
    assert "Change instrument to 'Syscal'" in result["error"]


def test_what_a_request_asks_for_and_the_run_does_not_do_is_reported(tmp_path, no_file_checks):
    preview, result, _ = _preview_and_run(tmp_path, {
        "data_file": "a.ohm", "user_request": "invert a.ohm and compare with rainfall"})
    assert result["status"] == "success", result.get("error")
    [warning] = result["warnings"]
    assert "climate_config" in warning and warning in preview["warnings"]


def test_a_failed_seismic_step_is_not_used_as_a_constraint(tmp_path, no_file_checks):
    calls = []
    coordinator = _coordinator(tmp_path, calls, seismic=True)
    coordinator.register_agent("seismic_processor", _Recorder("seismic_processor", calls, lambda data: {
        "status": "failed", "error": "seismic_file or seismic_data is required"}))
    result = coordinator.execute_workflow({"data_file": "a.ohm", "use_seismic": True,
                                           "seismic_data": None})
    assert result["status"] == "success", result.get("error")
    inversion = _inputs(calls, "ert_inversion")[0]
    assert "use_structure_constraint" not in inversion and "seismic_structure" not in inversion
    assert "process_seismic" not in coordinator.workflow_state["completed_steps"]
    assert any("seismic step failed" in warning for warning in result["warnings"])
    report = _inputs(calls, "report")[0]["workflow_data"]
    assert report["not_delivered"][0][0] == "Seismic structure constraint"


# --------------------------------------------------------------------------
# The deterministic request parser
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text, files", [
    # Chinese words run onto a name are not part of it...
    ("反演2021-10-08_1400.ohm", ["2021-10-08_1400.ohm"]),
    ("对2021-10-08_1400.ohm和2021-11-08_1400.ohm做时移反演",
     ["2021-10-08_1400.ohm", "2021-11-08_1400.ohm"]),
    ("帮我处理ERT数据2021-10-08_1400.ohm", ["2021-10-08_1400.ohm"]),
    ("TDEM数据就不做了，只要a.ohm", ["a.ohm"]),
    # ...but a Chinese file or folder name is kept whole.
    ("反演 测线1.ohm", ["测线1.ohm"]),
    ("反演 数据/测线1.ohm", ["数据/测线1.ohm"]),
    ("反演D:/野外数据2021/line1.ohm", ["D:/野外数据2021/line1.ohm"]),
    ("反演2021年10月8日.ohm", ["2021年10月8日.ohm"]),
    # Punctuation, quotes, drive letters and lists.
    ("请反演“2021-10-08_1400.ohm”，然后算含水量", ["2021-10-08_1400.ohm"]),
    ("反演a.ohm～c.ohm", ["a.ohm", "c.ohm"]),
    ("invert C:/surveys/a.ohm", ["C:/surveys/a.ohm"]),
    # A time-lapse series in the order given, baseline first.
    ("Baseline 2021-10-08.ohm, monitors:\n- 2021-11-08.ohm\n- 2021-12-08.ohm",
     ["2021-10-08.ohm", "2021-11-08.ohm", "2021-12-08.ohm"]),
    # Not surveys: a file to write, a mesh, a dated seismic file.
    ("Save the model as model.bin after inverting a.ohm and b.ohm", ["a.ohm", "b.ohm"]),
    ("Use the mesh from mesh.bin to invert a.ohm", ["a.ohm"]),
    ("Use seismic travel times srt_2021-10-08.dat to constrain ERT 2021-10-08_1400.ohm",
     ["2021-10-08_1400.ohm"]),
])
def test_the_files_a_request_names(text, files):
    assert ContextInputAgent(api_key=None)._extract_files_regex(text) == files


@pytest.mark.parametrize("text, asked", [
    ("只做ERT反演，不需要TDEM", False),
    ("Invert a.ohm. TDEM is not needed.", False),
    ("ERT inversion of a.ohm instead of TDEM", False),
    # A negation of something else, or of what follows, is not a refusal.
    ("No ERT, TDEM only: sounding.csv", True),
    ("TDEM不需要单独处理，和ERT一起反演：tem.csv, a.ohm", True),
    ("Don't forget the TDEM sounding tem.csv", True),
    ("除了TDEM之外，还要反演ERT数据a.ohm", True),
    ("Invert a.ohm and the TEM sounding", True),
    # The instrument by name, which is not the word "TEM".
    ("现在agent识别不了TEM2go数据", True),
    ("不要TEM2Go，只做ERT", False),
])
def test_a_request_asks_for_tdem_or_rules_it_out(text, asked):
    assert names_tdem(text) is asked


@pytest.mark.parametrize("text, sites", [
    ("Invert the MT sites a.edi and b.edi", ["a.edi", "b.edi"]),
    ("大地电磁反演 nmx20.xml", ["nmx20.xml"]),
    # An XML file is an MT site only when the request says MT, and "Mt." is a mountain.
    ("Invert a.ohm and read meta.xml", None),
    ("Invert a.ohm from the survey on Mt. Hood, settings in run.xml", None),
])
def test_a_request_names_its_mt_sites_or_none(text, sites):
    assert ContextInputAgent(api_key=None).request_inputs(text).get("mt_files") == sites


@pytest.mark.parametrize("text, instrument", [
    # "existing" holds "sting", "compares" "ares", "Albert" "bert".
    ("invert the existing survey a.ohm", None), ("compares the two lines", None),
    ("a.ohm from the site near Albert Lake", None),
    ("SuperSting R8 data", "Sting"), ("用Syscal测的数据", "Syscal"),
    ("ABEM Terrameter LS", "ABEM-Lund"), ("the ARES II survey", "ARES"),
])
def test_an_instrument_is_named_by_a_whole_word(text, instrument):
    assert ContextInputAgent(api_key=None)._infer_instrument_from_text(text) == instrument


# --------------------------------------------------------------------------
# Water content from a layered model
# --------------------------------------------------------------------------

LAYERS = {"regolith": {"rho_sat_range": [50, 250], "n_range": [1.3, 2.2],
                       "porosity_range": [0.25, 0.5]},
          "fractured_bedrock": {"rho_sat_range": [165, 350], "n_range": [2.0, 2.2],
                                "porosity_range": [0.1, 0.3]}}
#: Above and below a seismic interface, with enough cells each to be layers.
MARKERS = np.repeat([2, 3], 20)


def _convert(tmp_path, layer_params):
    """Run the runtime's conversion step, with the real agent, on a two-layer model."""
    from PyHydroGeophysX.agents.runtime import catalog
    from PyHydroGeophysX.agents.runtime.context import RunContext

    ctx = RunContext("convert", config={"layer_params": layer_params, "n_realizations": 50},
                     output_dir=str(tmp_path))
    ctx.put("inversion_results", {"resistivity_model": np.linspace(80.0, 900.0, MARKERS.size),
                                  "cell_markers": MARKERS})
    _, produced = catalog._convert_water_content(ctx)
    return ctx, produced["water_content"][0]


def test_a_structure_constrained_model_gets_the_layer_parameters(tmp_path):
    ctx, step = _convert(tmp_path, LAYERS)
    used = step["layer_params_used"]
    assert used[2]["rho_sat"]["mean"] == 150.0 and used[3]["rho_sat"]["mean"] == 257.5
    assert np.isfinite(step["water_content_mean"]).all()
    assert not ctx.warnings


@pytest.mark.parametrize("layer_params, said", [
    ({name: {key: value for key, value in params.items() if key != "porosity_range"}
      for name, params in LAYERS.items()}, "gave no usable porosity"),
    ({**LAYERS, "regolith": {**LAYERS["regolith"], "rho_sat_range": [None, None]}},
     "is not a pair of numbers"),
    ({"regolith": LAYERS["regolith"]}, "No layer parameters were given for layer 3"),
])
def test_an_incomplete_request_still_converts_and_says_what_it_filled_in(
        tmp_path, layer_params, said):
    # A parsed request that left a range out used to stop the conversion, and
    # the water content it asked for was lost.
    ctx, step = _convert(tmp_path, layer_params)
    assert step["status"] == "success" and np.isfinite(step["water_content_mean"]).all()
    assert set(step["layer_params_used"]) == {2, 3}
    assert any(said in warning for warning in ctx.warnings), ctx.warnings


def test_a_failed_conversion_is_reported_as_failed_not_as_not_requested(tmp_path):
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    from PyHydroGeophysX.agents.runtime import catalog
    from PyHydroGeophysX.agents.runtime.context import RunContext

    ctx = RunContext("estimate water content", config={"convert_to_water_content": True},
                     output_dir=str(tmp_path))
    ctx.begin("convert_water_content")
    ctx.finish(status="failed", error="ValueError: no model to convert")
    data = catalog._survey_report_input(ctx, {})["workflow_data"]
    assert data["skip_petrophysics"] is False
    section = ReportAgent(api_key=None)._generate_wc_summary(data)
    assert "requested, but the conversion did not complete" in section
    assert "no model to convert" in section and "not requested" not in section
    timelapse = ReportAgent(api_key=None)._generate_timelapse_water_content_section(
        {"water_content_failed": "ValueError: no model to convert"})
    assert timelapse.startswith("## Water Content") and "did not complete" in timelapse


def test_water_content_always_states_its_uncertainty_and_whose_parameters(tmp_path):
    # A hydrologic property read from geophysics always carries its spread, and
    # one computed without the user's petrophysical relationship is flagged as
    # unreliable while the run goes, naming the parameters drawn and their ranges.
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    from PyHydroGeophysX.agents.runtime import catalog
    from PyHydroGeophysX.agents.runtime.context import RunContext

    model = 10 ** np.random.default_rng(0).uniform(1.3, 2.7, 200)

    def convert(config):
        ctx = RunContext("estimate water content", dict(config, request="water content"),
                         output_dir=str(tmp_path))
        ctx.put("inversion_results", {"resistivity_model": model, "mesh": None})
        summary, produced = catalog._convert_water_content(ctx)
        for key, value in produced.items():
            ctx.put(key, value)
        return ctx, summary

    ctx, summary = convert({"n_realizations": 1})
    assert "±" in summary and "Not reliable" in summary
    assert any("raised from 1 to 50" in w for w in ctx.warnings)
    stated = next(w for w in ctx.warnings if "it is not reliable" in w)
    for parameter in ("cementation exponent m", "saturation exponent n", "porosity",
                      "pore-fluid resistivity 20 Ω·m (fixed)"):
        assert parameter in stated
    report = ReportAgent(api_key=None)._generate_wc_summary(
        catalog._survey_report_input(ctx, ctx.get("inversion_results"))["workflow_data"])
    assert "Not Calibrated" in report and stated in report

    ctx, _ = convert({"petrophysical_params": {"rho_sat": 541}})
    assert any("saturation exponent n and porosity were not given" in w for w in ctx.warnings)

    ctx, summary = convert({"petrophysical_params": {"m": 1.6, "n": 2.0, "porosity": 0.35,
                                                     "rho_fluid": 25}})
    assert "±" in summary and not any("default" in w for w in ctx.warnings)


# --------------------------------------------------------------------------
# The unified runtime: what it delivers, and what it says it did not
# --------------------------------------------------------------------------

_LAYERED = {"status": "success", "recovered_resistivity": np.array([40.0, 80, 150, 400, 900]),
            "thicknesses": np.array([1.0, 2.0, 3.0, 5.0])}


class _CellNumbered:
    """An inversion mesh as the conversion sees it: one marker per cell."""

    def __init__(self, cells):
        self.cells = cells

    def cellMarkers(self):
        return list(range(self.cells))


@pytest.mark.parametrize("run_uncertainty", [True, False])
@pytest.mark.parametrize("params, relationship, region", [
    ({}, "default", False),
    ({"rho_sat": [200.0, 300.0], "n": 2.0, "porosity": [0.2, 0.3]}, "user", False),
    ({2: {}}, "default", False),
    ({2: {"rhos": {"mean": 50.0, "std": 0.0}, "n": {"mean": 2.0, "std": 0.0},
          "porosity": {"mean": 0.3, "std": 0.0}, "sigma_sur": {"mean": 0.0, "std": 0.0}}},
     "user", True),
])
def test_the_coordinators_water_content_takes_cell_numbers_as_one_unit(tmp_path, params,
                                                                        relationship, region,
                                                                        run_uncertainty):
    # WaterContentAgent read an inversion mesh's cell numbers as layers in index
    # order - regolith for the first cell, bedrock for the last, fractured
    # bedrock for the rest - and dropped parameters given by name: on a
    # synthetic with the truth known, 27 % of cells within one sigma. One
    # genuine region with its own distributions still converts with them, and
    # the result keeps its saturation_all.
    from types import SimpleNamespace

    from PyHydroGeophysX.agents.water_content_agent import WaterContentAgent

    mesh = SimpleNamespace(cellMarkers=lambda: [2] * 40) if region else _CellNumbered(40)
    inputs = {
        "inversion_results": {"mesh": mesh,
                              "resistivity_model": np.linspace(100.0, 900.0, 40)},
        "petrophysical_params": params, "uncertainty_analysis": run_uncertainty,
        "n_realizations": 50, "output_dir": str(tmp_path)}
    out = WaterContentAgent(api_key="").execute(inputs)
    assert out["saturation_all"].shape == ((50, 40, 1) if run_uncertainty else (40, 1))
    assert out["inversion_uncertainty"] is False
    if region:
        expected = 0.3 * (50.0 / inputs["inversion_results"]["resistivity_model"]) ** 0.5
        np.testing.assert_allclose(out["water_content_mean"][:, 0], expected)
        np.testing.assert_allclose(out["water_content_std"], 0.0, atol=1e-15)
        if run_uncertainty:
            for name, distribution in params[2].items():
                np.testing.assert_allclose(out["params_used"][2][name], distribution["mean"])
    else:
        assert out["n_layers"] == 1 and out["petrophysical_relationship"] == relationship
        assert np.all(np.asarray(out["water_content_std"]) > 0)
        if relationship == "user":
            assert 200.0 <= np.median(out["params_used"][0]["rho_sat"]) <= 300.0

    # The inversion's own uncertainty, when it came with the model, is drawn too.
    inputs["inversion_results"]["log_resistivity_std"] = np.full(40, 0.5)
    uncertain = WaterContentAgent(api_key="").execute(inputs)
    assert uncertain["inversion_uncertainty"] is True
    assert uncertain["saturation_all"].shape == out["saturation_all"].shape
    if run_uncertainty:
        assert np.mean(uncertain["water_content_std"]) > np.mean(out["water_content_std"])


def _tdem_stubs(name):
    """TDEM steps that fail to read ``broken``, find no model in ``empty``, or invert."""
    def load(ctx):
        if "broken" in name:
            raise ValueError("Failed to load TDEM data from broken.csv: no time column")
        return "Read.", {"tdem_data": {"source_file": name, "n_soundings": 1}}

    def invert(ctx):
        return "Inverted.", {"tdem_results": {"status": "success"} if "empty" in name
                             else dict(_LAYERED)}
    return {"load_tdem_data": load, "invert_tdem": invert}


@pytest.mark.parametrize("config, status, said", [
    # A broken TDEM file beside good ERT data came back "success", and the
    # report never mentioned TDEM.
    ({"data_file": "a.ohm", "tdem_file": "broken.csv",
      "user_request": "invert a.ohm and the TDEM sounding broken.csv"},
     "incomplete", "Load TDEM data did not complete"),
    # A request alone: the files it names are the run's inputs.
    ({"user_request": "invert a.ohm and estimate water content"},
     "success", "the run uses the file the request names: a.ohm"),
    # A TDEM sounding alone converts to water content, layer by layer...
    ({"tdem_file": "s.csv", "user_request": "invert the TDEM sounding s.csv and "
                                            "estimate the water content"},
     "success", None),
    # ...and when it cannot be converted, the missing water content says why.
    ({"tdem_file": "empty.csv", "user_request": "invert the TDEM sounding "
                                                "empty.csv and estimate the water content"},
     "incomplete", "produced none: The TDEM inversion returned no layered resistivity model"),
])
def test_the_runtime_delivers_or_names_what_it_did_not(tmp_path, monkeypatch, config,
                                                       status, said):
    from types import SimpleNamespace

    from PyHydroGeophysX.agents.report_agent import ReportAgent
    from PyHydroGeophysX.agents.runtime.entry import run_workflow
    from PyHydroGeophysX.agents.runtime.tools import TOOLS

    loaded = []

    def load(ctx):
        loaded.append(ctx.config["data_file"])
        survey = SimpleNamespace(electrodes=[0] * 4, observations=[0] * 10)
        return "Loaded 1 survey.", {"ert_data": [survey], "ert_files": ["a.ohm"]}

    stubs = {
        "load_ert_surveys": load,
        "invert_ert": lambda ctx: ("Inverted.", {"inversion_results": {
            "status": "success", "resistivity_model": np.linspace(50.0, 500.0, 40),
            "chi2": 1.6, "iterations": 7}}),
        "evaluate_inversion": lambda ctx: ("Scored.", {"evaluation_results": {
            "status": "success", "quality_score": 80.0, "summary": "Acceptable."}}),
        **_tdem_stubs(config.get("tdem_file", "")),
    }
    for name, handler in stubs.items():
        monkeypatch.setattr(TOOLS[name], "handler", handler)
    monkeypatch.setattr(ReportAgent, "_save_pdf_report", lambda self, *a, **k: None)

    results, plan, text, files = run_workflow(dict(config), None, None, "openai", tmp_path)
    assert results["status"] == status, results["warnings"]
    if said:
        assert any(said in warning for warning in results["warnings"]), results["warnings"]
    if "data_file" in config or "a.ohm" in config["user_request"]:
        assert loaded == ["a.ohm"]
    if status == "incomplete" and files:
        # The written report names what it does not contain, with the reason.
        report = open(files["report_markdown"], encoding="utf-8").read()
        reason = ("no time column" if "broken" in config.get("tdem_file", "")
                  else "no layered resistivity model")
        assert "Not Delivered" in report and reason in report
    if "data_file" not in config and config.get("tdem_file"):
        # A sounding alone used to end "success" with no report at all.
        assert files["report_markdown"].endswith("tdem_report.md")
    if config.get("tdem_file") == "s.csv":
        assert len(results["water_content"]) == 1
        assert results["tdem_results"]["water_content_mean"].shape == (5,)
        assert (tmp_path / "tdem" / "water_content_by_layer.csv").exists()
        report = open(files["report_markdown"], encoding="utf-8").read()
        assert "Water Content" in report and "±" in report


@pytest.mark.parametrize("config, expected", [
    ({"seismic_params": {"lam": 5, "zWeight": 1.0}}, {"lam": 5, "zWeight": 1.0}),
    ({"seismic_params": {"lam": 5}, "seismic_inversion_params": {"lam": 30}},
     {"lam": 30}),
    ({"lam": 12, "z_weight": 0.5}, {"lam": 12, "zWeight": 0.5}),
])
def test_the_seismic_settings_a_request_gives_reach_the_seismic_agent(tmp_path, monkeypatch,
                                                                     config, expected):
    from PyHydroGeophysX.agents.runtime import catalog
    from PyHydroGeophysX.agents.runtime.context import RunContext
    from PyHydroGeophysX.agents.seismic_agent import SeismicAgent

    seen = []
    monkeypatch.setattr(SeismicAgent, "execute", lambda self, data: seen.append(data) or {
        "status": "failed", "error": "stopped here"})
    ctx = RunContext("invert srt.dat", {"seismic_file": "srt.dat", **config},
                     output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="stopped here"):
        catalog._invert_seismic(ctx)
    assert seen[0]["inversion_params"] == expected


@pytest.mark.parametrize("api_key", [None, "sk-test"])
def test_a_request_is_parsed_without_a_model_and_the_solver_is_left_to_the_library(
        monkeypatch, api_key):
    agent = ContextInputAgent(api_key=api_key)

    def unreachable(*args, **kwargs):
        raise RuntimeError("Error querying openai LLM: connection refused")

    # No key used to raise "OPENAI API key not found"; a failing call, the same.
    monkeypatch.setattr(agent, "query_llm", unreachable)
    config = agent.parse_request("Invert 2021-10-08_1400.ohm and estimate water content")
    assert config["data_file"] == "2021-10-08_1400.ohm"
    # A parser default of 'cgls' overrode the time-lapse solver's spd_cholesky.
    assert "method" not in config["inversion_params"]


def test_a_tdem_project_is_not_also_taken_as_ert_data(monkeypatch):
    import json

    from PyHydroGeophysX.agents.runtime.context import RunContext
    from PyHydroGeophysX.agents.runtime.controller import route_ahead
    from PyHydroGeophysX.agents.runtime.entry import adopt_request_inputs

    # The ERT stage runs on every request and was told its data file is
    # required: it gave the TEM2Go project the user selected as TDEM data as
    # data_file too, and the studio's route showed three ERT steps after the
    # TDEM inversion.
    project = "C:/survey/Oct02/project1.tiw"
    agent = ContextInputAgent(api_key="sk-test")
    monkeypatch.setattr(agent, "query_llm", lambda prompt, **kwargs: (
        json.dumps({"data_file": project, "instrument": "Custom"})
        if "configuring an ERT" in prompt else '{"aspects": {"tdem": true}}'))
    request = "help me process TEM2go data and give me the spatial resistivity"
    config = agent.parse_request(request, available_data={"tdem_file": project})
    assert "data_file" not in config and "ert_file" not in config
    config, _ = adopt_request_inputs({**config, "tdem_file": project, "data_file": project})
    route = [step["tool"] for step in route_ahead(RunContext(request, config, "."))]
    assert route == ["load_tdem_data", "invert_tdem", "evaluate_tdem_inversion",
                     "write_survey_report"]


@pytest.mark.parametrize("request_text, velocity", [
    ("Process the GPR line", None),
    ("Process the GPR line at 0.1 m/ns", 0.1),
])
def test_a_gpr_run_makes_its_previewed_steps_and_states_its_velocity(
        tmp_path, monkeypatch, request_text, velocity):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from pathlib import Path

    from PyHydroGeophysX.agents.context_input_agent import gpr_from_reply
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    from PyHydroGeophysX.agents.runtime.context import RunContext
    from PyHydroGeophysX.agents.runtime.controller import route_ahead
    from PyHydroGeophysX.agents.runtime.entry import run_workflow
    from PyHydroGeophysX.agents.runtime.tools import TOOLS

    example = Path(__file__).resolve().parents[1] / "examples" / "data"
    if not (example / "Watercontent.npy").is_file():
        pytest.skip("the MODFLOW example model is not in this checkout")
    monkeypatch.setattr(ReportAgent, "_save_pdf_report", lambda self, *a, **k: None)
    config = {"user_request": request_text, "gpr_file": str(example)}
    route = [step["tool"] for step in route_ahead(RunContext(request_text, dict(config), "."))]
    assert route == ["load_gpr_line", "process_gpr_line", "write_survey_report"]

    results, plan, _, files = run_workflow(dict(config), None, None, "openai", tmp_path)
    assert results["status"] == "success", results["warnings"]
    assert [step["step"] for step in plan] == [TOOLS[name].label for name in route]
    gpr = results["gpr_results"]
    report = open(files["report_markdown"], encoding="utf-8").read()
    assert "](gpr/example_line_radargram.png)" in report
    assert (tmp_path / "gpr" / "example_line_radargram.png").is_file()
    # A depth is v t / 2: only a velocity the request gives is used, and its
    # absence is said, never filled in - by the run or by a model's reading.
    assert gpr["velocity_m_per_ns"] == velocity
    if velocity is None:
        assert gpr["max_depth_m"] is None and "gives no depth" in report
        assert any("No radar velocity was given" in w for w in results["warnings"])
        assert "gpr_velocity" not in gpr_from_reply({"gpr_velocity": 0.1}, request_text)[0]
    else:
        assert gpr["max_depth_m"] == pytest.approx(0.5 * velocity * gpr["time_window_ns"])
        assert "0.1 m/ns (given for this run)" in report


# --------------------------------------------------------------------------
# Names removed or moved in 0.5.0
# --------------------------------------------------------------------------

_REMOVED = [("WorkflowOrchestratorAgent", "workflow_orchestrator_agent"),
            ("CodeGenerationAgent", "code_generation_agent"),
            ("GeophysicalInversionAgent", "geophysical_inversion_agent")]


@pytest.mark.parametrize("name, module", _REMOVED)
@pytest.mark.parametrize("through", ["package", "module"])
def test_a_removed_agent_names_what_replaced_it(name, module, through, monkeypatch):
    if through == "package":
        source = "PyHydroGeophysX.agents"
        said = f"{name} was removed in PyHydroGeophysX 0.5.0: "
    else:
        # 0.3.0 code imported the module, not the package attribute.
        monkeypatch.delitem(sys.modules, f"PyHydroGeophysX.agents.{module}", raising=False)
        source = f"PyHydroGeophysX.agents.{module}"
        said = f"agents.{module} was removed in PyHydroGeophysX 0.5.0: "
    namespace = {}
    with pytest.warns(DeprecationWarning, match=said):
        exec(f"from {source} import {name}", namespace)
    # Importing the name is harmless; using it says the same thing, loudly.
    with pytest.raises(RuntimeError, match=f"{name} was removed"):
        namespace[name]()


@pytest.mark.parametrize("name, home", [
    ("IMPLEMENTED_SCHEME", "_method"), ("climate_blocker", "_intent"),
    ("wants_climate", "_intent"), ("wants_water_content", "_intent"),
    ("coords_from_config", "_geocode"), ("geocode_place", "_geocode"),
])
def test_a_moved_helper_still_imports_from_base_agent(name, home):
    from PyHydroGeophysX.agents import base_agent

    with pytest.warns(DeprecationWarning, match=f"PyHydroGeophysX.agents.{home}.{name}"):
        value = getattr(base_agent, name)
    assert value is getattr(importlib.import_module(f"PyHydroGeophysX.agents.{home}"), name)


def test_an_agent_takes_the_environments_key_only_when_asked(monkeypatch):
    # An agent used to read OPENAI_API_KEY whenever it was given no key, so a
    # library call - a conversion, a header check, the documentation's no-model
    # examples - sent requests and spent credit nobody asked for.
    import warnings

    from PyHydroGeophysX.agents import base_agent
    from PyHydroGeophysX.agents.context_input_agent import ContextInputAgent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-environment")
    monkeypatch.setattr(base_agent, "_KEYS_REPORTED", set())
    with pytest.warns(UserWarning, match="OPENAI_API_KEY"):
        assert not ContextInputAgent(api_key=None).llm_enabled
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not ContextInputAgent(api_key="").llm_enabled
        assert ContextInputAgent(api_key="env").api_key == "sk-test-environment"


# --------------------------------------------------------------------------
# Assistants: AQUAH, GeoSAGE and later ones plug into one runtime
# --------------------------------------------------------------------------

def test_each_assistant_registers_with_its_own_tools_and_workflow():
    from PyHydroGeophysX.agents.assistants import assistants, get_assistant
    from PyHydroGeophysX.agents.runtime.tools import TOOLS

    assert [a.key for a in assistants()][:2] == ["aquah", "geosage"]
    aquah = get_assistant("aquah")
    assert aquah.availability() == (True, "")
    assert aquah.load_tools() is TOOLS and "write_report" in TOOLS
    assert callable(aquah.load_workflow())
    # GeoSAGE is listed while it is ported, but cannot be chosen, and its tools
    # never mix with AQUAH's, even where a name is the same.
    from PyHydroGeophysX.agents.assistants.geosage import ASSISTANT as geosage
    ready, why = geosage.availability()
    assert not ready and why
    own = geosage.load_tools()
    assert own is not TOOLS and own["write_report"] is not TOOLS["write_report"]
    assert "run_joint_inversion" not in TOOLS
    assert geosage.tool_for_label("Run joint gravity-magnetic inversion") == "run_joint_inversion"


def test_an_assistants_skills_ship_with_it_and_a_group_can_replace_one(tmp_path, monkeypatch):
    # Skills (agents.skills) are the plug-in's own folder, found inside its
    # package, then PHGX_SKILLS_PATH, whose skill of the same name wins. A read
    # never leaves the skill's folder.
    from PyHydroGeophysX.agents.assistants import get_assistant
    from PyHydroGeophysX.agents.skills import ENV_VAR, find, run_tool

    aquah = get_assistant("aquah")
    monkeypatch.delenv(ENV_VAR, raising=False)
    shipped = aquah.load_skills()
    assert find(shipped, "ert-data-quality") is not None
    assert all(skill.description and skill.body() for skill in shipped)
    ours = tmp_path / "lab" / "ert-data-quality"
    ours.mkdir(parents=True)
    (ours / "SKILL.md").write_text("---\nname: ert-data-quality\ndescription: Our lab's QC."
                                   "\n---\nDrop readings above 3 % reciprocal error.\n",
                                   encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "lab"))
    skills = aquah.load_skills()
    assert len(skills) == len(shipped)
    assert find(skills, "ert-data-quality").body() == "Drop readings above 3 % reciprocal error."
    (tmp_path / "lab" / "secret.txt").write_text("not a skill's", encoding="utf-8")
    assert run_tool(skills, {"name": "ert-data-quality", "file": "../secret.txt"})["status"] == "failed"


def test_an_assistant_runs_its_own_tools_to_its_own_final_product(tmp_path, monkeypatch):
    # GeoSAGE's chain on the controller AQUAH uses, with stand-in handlers: the
    # data dependencies alone order the steps, and the run ends with the
    # reviewed report rather than at the first report-like product.
    from PyHydroGeophysX.agents.assistants.geosage import tools, workflow

    made = {"prepare_data": {"field_data": 1}, "compile_priors": {"geological_priors": None},
            "run_joint_inversion": {"property_models": 1}, "build_quasi_geology": {"geo_model": 1},
            "write_report": {"draft_report": {"summary": "Group 5 is the primary target."}},
            "review_report": {"report_files": {"report_markdown": "report.md"}}}
    for name, outputs in made.items():
        monkeypatch.setattr(tools.TOOLS[name], "handler",
                            lambda ctx, outputs=outputs: ("Done.", dict(outputs)))
    events = []
    result = workflow.run({"request": "find serpentinite targets", "output_dir": str(tmp_path)},
                          lambda *a: None, on_event=events.append)
    assert [e["tool"] for e in events if e["phase"] == "done"] == list(made)
    # The route the studio draws: projected in full before the first step,
    # one stop shorter after each, and empty once the final product exists.
    routes = [[step["tool"] for step in e["ahead"]] for e in events if e["phase"] == "route"]
    assert routes[0] == list(made)
    assert routes[1] == list(made)[1:] and routes[-1] == []
    assert result["status"] == "success"
    assert result["report_files"] == {"report_markdown": "report.md"}
    assert result["interpretation"] == "Group 5 is the primary target."
