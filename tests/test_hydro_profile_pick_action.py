"""Static contract tests for AQUAH's interactive hydro-profile action.

The Hydro wizard imports optional Qt dependencies, so this test checks its public
agent contract without requiring a desktop backend.
"""

import ast
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "PyHydroGeophysX"
    / "qt_apps"
    / "modules"
    / "hydro_geophysics.py"
)
CHAT_PANEL_PATH = MODULE_PATH.parents[1] / "agent" / "chat_panel.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


def test_interactive_profile_action_opens_the_picker_and_pauses_aquah():
    """Manual profile selection must expose a GUI checkpoint to AQUAH."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    describe = ast.dump(_function(tree, "agent_describe"))
    handlers = ast.dump(_function(tree, "agent_apply"))
    picker = ast.dump(_function(tree, "_agent_start_profile_pick"))

    assert "start_profile_pick" in describe
    assert "start_profile_pick" in handlers
    assert "_go_to" in picker
    assert "setChecked" in picker
    assert "awaiting_user" in picker


def test_agent_requires_explicit_parameter_confirmation_before_hydro_run():
    """AQUAH must expose defaults, then require a user-confirmation action."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    describe = ast.dump(_function(tree, "agent_describe"))
    handlers = ast.dump(_function(tree, "agent_apply"))
    select = ast.dump(_function(tree, "_agent_select_methods"))
    run = ast.dump(_function(tree, "_agent_run"))
    prompt = CHAT_PANEL_PATH.read_text(encoding="utf-8")

    assert "confirm_parameters" in describe
    assert "confirm_parameters" in handlers
    assert "parameter_defaults" in select
    assert "_agent_parameters_confirmed" in run
    assert "confirm_parameters ONLY after the user explicitly confirms" in prompt


def test_results_are_method_selected_model_and_measurement_panels():
    """Hydro Results must no longer be a single stacked gallery of composite PNGs."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    builder = ast.dump(_function(tree, "_build_results_step"))
    populate = ast.dump(_function(tree, "_populate_results"))
    selected = ast.dump(_function(tree, "_show_selected_result"))

    assert "Model" in builder and "Measurements" in builder
    assert "_result_method" in builder
    assert "display_paths" in populate
    assert "_result_model_view" in selected and "_result_measurement_view" in selected
