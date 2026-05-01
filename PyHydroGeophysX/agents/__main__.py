"""Command-line entry points for PyHydroGeophysX agents."""

import argparse
import json
from typing import Any, Dict

from .agent_coordinator import AgentCoordinator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m PyHydroGeophysX.agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser(
        "preview",
        help="Preview an agent workflow without running LLM calls or inversions.",
    )
    preview.add_argument("--request", default="", help="Natural-language workflow request.")
    preview.add_argument("--config-json", default="", help="Inline JSON workflow config.")
    preview.add_argument("--provider", default="openai", help="LLM provider used for cost estimates.")
    preview.add_argument("--output-dir", default="results/agents", help="Output directory.")
    return parser


def main(argv: Any = None) -> int:
    """Run the agent CLI.

    Parameters
    ----------
    argv : sequence, optional
        Command-line arguments. Defaults to ``sys.argv`` via argparse.

    Returns
    -------
    int
        Process exit code.

    Raises
    ------
    json.JSONDecodeError
        Raised if ``--config-json`` is invalid JSON.

    Examples
    --------
    >>> main(["preview", "--request", "Run ERT inversion on demo.ohm"]) in (0, 2)
    True
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "preview":
        config: Dict[str, Any] = {}
        if args.config_json:
            config.update(json.loads(args.config_json))
        if args.request:
            config["user_request"] = args.request
        coordinator = AgentCoordinator(
            api_key=None,
            output_dir=args.output_dir,
            llm_provider=args.provider,
        )
        result = coordinator.execute_workflow(config, dry_run=True)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result["status"] != "failed" else 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
