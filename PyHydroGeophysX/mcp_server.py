"""Optional local stdio MCP facade over the existing workflow registry and CLI."""
import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import uuid


def contained(root, path):
    candidate = (root / path).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def workflow_catalog():
    from PyHydroGeophysX.workflows.registry import list_workflows
    return [{'id': d.workflow_id, 'description': d.description, 'handler': d.handler_path}
            for d in list_workflows()]


def create_server(root, allow_run=False):
    from mcp.server.fastmcp import FastMCP
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('--root must be a directory')
    server = FastMCP('PyHydroGeophysX')

    @server.tool()
    def list_workflows() -> list:
        """List existing ERT, seismic, EM, hydro and other registered workflow APIs."""
        return workflow_catalog()

    @server.tool()
    def scan_data_folder(folder: str = '.') -> dict:
        """Read a bounded file inventory and previews under the configured project root."""
        from PyHydroGeophysX.agents.folder_catalog import scan_folder
        return scan_folder(contained(root, folder))

    @server.tool()
    def search_docs(query: str) -> list:
        """Retrieve local project documentation with source paths and line numbers."""
        from PyHydroGeophysX.agents.local_knowledge import retrieve
        return retrieve(query)

    @server.tool()
    def validate_recipe(recipe: str) -> dict:
        """Validate an existing workflow JSON recipe under the project root."""
        from PyHydroGeophysX.workflows.recipe import load_recipe
        from PyHydroGeophysX.workflows.registry import get_workflow
        with contextlib.redirect_stdout(sys.stderr):
            spec = load_recipe(contained(root, recipe))
            descriptor = get_workflow(spec.workflow_id)
            spec.validate(stochastic=descriptor.stochastic)
        return {'workflow_id': spec.workflow_id, 'valid': True}

    if allow_run:
        @server.tool()
        def run_recipe(recipe: str) -> dict:
            """Execute a trusted project recipe through the existing isolated workflow CLI.

            This can use substantial compute and writes a new mcp_runs directory.
            Only exposed when the server owner enables --allow-run. Recipes and
            their referenced inputs must be trusted by the server owner.
            """
            source = contained(root, recipe)
            validate_recipe(recipe)
            output = root / 'mcp_runs' / uuid.uuid4().hex
            output.mkdir(parents=True)
            result_path = output / 'result.json'
            with (output / 'activity.log').open('wb') as log:
                child = subprocess.run([sys.executable, '-m', 'PyHydroGeophysX.workflows.cli',
                    'run', str(source), '--project-root', str(root), '--output-dir', str(output),
                    '--result-file', str(result_path)], stdout=log, stderr=log, timeout=86400)
            return {'exit_code': child.returncode, 'output_dir': str(output),
                    'result': json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None}
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--allow-run', action='store_true', help='Expose computation tools to trusted local MCP clients')
    args = parser.parse_args()
    create_server(args.root, args.allow_run).run(transport='stdio')


if __name__ == '__main__':
    main()
