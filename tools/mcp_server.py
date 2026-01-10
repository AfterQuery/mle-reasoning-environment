"""MCP Server exposing MLE tools via FastMCP."""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tools import get_all_tools

load_dotenv()

mcp = FastMCP("mle-agent")
_tools = get_all_tools()


@mcp.tool()
async def read_file(path: str) -> str:
    """Read contents of a file at the given path."""
    result = await _tools["read_file"]({"path": path})
    return result["result"]


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    result = await _tools["write_file"]({"path": path, "content": content})
    return result["result"]


@mcp.tool()
async def list_files(path: str = ".") -> str:
    """List files and directories at the given path."""
    result = await _tools["list_files"]({"path": path})
    return result["result"]


@mcp.tool()
async def run_python(script_path: str, timeout: int = 60) -> str:
    """Execute a Python script file and return stdout/stderr."""
    result = await _tools["run_python"](
        {"script_path": script_path, "timeout": timeout}
    )
    return result["result"]


@mcp.tool()
async def bash_exec(command: str, timeout: int = 60, cwd: str = None) -> str:
    """Execute a bash command and return stdout/stderr."""
    result = await _tools["bash_exec"](
        {"command": command, "timeout": timeout, "cwd": cwd}
    )
    return result["result"]


def run_server(port: int = 8000):
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
