"""Static contract tests for visible and agent-only Mesh 3D features."""

import ast
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "PyHydroGeophysX"
    / "qt_apps"
    / "modules"
    / "mesh3d_processing.py"
)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


def test_mesh3d_hides_forward_controls_but_keeps_the_agent_action():
    """3D ERT forward modeling remains agent-only, not part of the mesh UI."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    controls = ast.dump(_function(tree, "_build_controls"))
    describe = ast.dump(_function(tree, "agent_describe"))
    handlers = ast.dump(_function(tree, "agent_apply"))

    assert "_build_ert_forward_group" not in controls
    assert "_agent_run_ert_forward" in ast.dump(tree)
    assert "run_ert_forward" in describe
    assert "run_ert_forward" in handlers


def test_mesh3d_stacked_panels_follow_the_active_page_height():
    """Small pages must not reserve space for a taller stacked sibling."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    helper = ast.dump(_function(tree, "_fit_stack_to_current_page"))
    visibility = ast.dump(_function(tree, "_update_visibility"))

    assert "currentWidget" in helper
    assert "setFixedHeight" in helper
    assert visibility.count("_fit_stack_to_current_page") == 2
