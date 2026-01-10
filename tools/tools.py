"""MLE-specific tools for file operations, Python execution, and shell commands."""

import os
import shutil
import subprocess
import sys
from typing import Any, Dict


class Tool:
    """Base class for tools."""

    name: str
    description: str
    parameters: dict

    def get_tool_json(self, provider: str = "openai", strict: bool = True) -> dict:
        schema = {
            "type": "object",
            "properties": self.parameters,
            "required": list(self.parameters.keys()),
            "additionalProperties": False,
        }
        if provider == "anthropic":
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": schema,
            }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
                "strict": strict,
            },
        }


class ReadFile(Tool):
    name = "read_file"
    description = "Read contents of a file at the given path"
    parameters = {"path": {"type": "string", "description": "File path to read"}}

    async def __call__(self, args: dict) -> Dict[str, Any]:
        path = args.get("path", "")
        try:
            with open(path, "r") as f:
                return {"success": True, "result": f.read()}
        except Exception as e:
            return {"success": False, "result": f"Error reading file: {e}"}


class WriteFile(Tool):
    name = "write_file"
    description = "Write content to a file at the given path"
    parameters = {
        "path": {"type": "string", "description": "File path to write to"},
        "content": {"type": "string", "description": "Content to write"},
    }

    async def __call__(self, args: dict) -> Dict[str, Any]:
        path, content = args.get("path", ""), args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return {"success": True, "result": f"Successfully wrote to {path}"}
        except Exception as e:
            return {"success": False, "result": f"Error writing file: {e}"}


class ListFiles(Tool):
    name = "list_files"
    description = "List files and directories at the given path"
    parameters = {"path": {"type": "string", "description": "Directory path to list"}}

    async def __call__(self, args: dict) -> Dict[str, Any]:
        path = args.get("path", ".")
        try:
            items = os.listdir(path)
            result = []
            for item in items:
                full = os.path.join(path, item)
                result.append(f"{'[DIR] ' if os.path.isdir(full) else ''}{item}")
            return {"success": True, "result": "\n".join(result)}
        except Exception as e:
            return {"success": False, "result": f"Error listing directory: {e}"}


class RunPython(Tool):
    name = "run_python"
    description = "Execute a Python script file and return stdout/stderr"
    parameters = {
        "script_path": {
            "type": "string",
            "description": "Path to Python script to run",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 60)",
        },
    }

    def _get_python_executable(self) -> str:
        """Find the Python executable, trying 'python' first then 'python3'."""
        # In Docker, 'python' usually works; on macOS, 'python3' is needed
        if shutil.which("python"):
            return "python"
        elif shutil.which("python3"):
            return "python3"
        return sys.executable  # Fall back to current interpreter

    async def __call__(self, args: dict) -> Dict[str, Any]:
        script_path = args.get("script_path", "")
        timeout = args.get("timeout", 60)
        python_exe = self._get_python_executable()
        try:
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(script_path) or ".",
            )
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nReturn code: {result.returncode}"
            return {"success": result.returncode == 0, "result": output}
        except subprocess.TimeoutExpired:
            return {"success": False, "result": f"Script timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "result": f"Error running script: {e}"}


class BashExec(Tool):
    name = "bash_exec"
    description = "Execute a bash command and return stdout/stderr"
    parameters = {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 60)",
        },
        "cwd": {"type": "string", "description": "Working directory (optional)"},
    }

    async def __call__(self, args: dict) -> Dict[str, Any]:
        command = args.get("command", "")
        timeout = args.get("timeout", 60)
        cwd = args.get("cwd", None)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nReturn code: {result.returncode}"
            return {"success": result.returncode == 0, "result": output}
        except subprocess.TimeoutExpired:
            return {"success": False, "result": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "result": f"Error executing command: {e}"}


def get_all_tools() -> Dict[str, Tool]:
    return {
        "read_file": ReadFile(),
        "write_file": WriteFile(),
        "list_files": ListFiles(),
        "run_python": RunPython(),
        "bash_exec": BashExec(),
    }
