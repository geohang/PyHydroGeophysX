"""Connect the desktop worker to the bundled MCP server over standard stdio."""
import asyncio
import sys
import os
from datetime import timedelta


async def _catalog(root):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable,
        args=['-m', 'PyHydroGeophysX.mcp_server', '--root', str(root)],
        env={'PYTHONPATH': os.environ.get('PYTHONPATH', ''), 'PYTHONUTF8': '1'})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
            await session.initialize()
            result = await session.call_tool('list_workflows', {})
            if result.isError:
                raise RuntimeError('Local MCP workflow catalog failed.')
            return '\n'.join(item.text for item in result.content if hasattr(item, 'text'))


def catalog(root):
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('MCP is optional. Install pyhydrogeophysx[mcp] in the Studio environment, or turn off the MCP option.') from exc
    return asyncio.run(_catalog(root))
