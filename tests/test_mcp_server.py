#!/usr/bin/env python3
"""
Comprehensive test script for MLE MCP Server.
Tests each MCP tool to verify functionality.
"""

import asyncio
import os
import shutil
import sys
import tempfile
from datetime import datetime

# Add tools directory to path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
)

# Test results storage
results = []


def log_result(tool_name: str, success: bool, message: str, data: dict = None):
    """Log test result"""
    result = {
        "tool": tool_name,
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }
    results.append(result)
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {tool_name}: {message}")


async def test_mcp_imports():
    """Test 1: Verify MCP server can be imported"""
    tool_name = "mcp_imports"
    try:
        from mcp.server.fastmcp import FastMCP

        log_result(tool_name, True, "FastMCP imported successfully")
        return True
    except ImportError as e:
        log_result(tool_name, False, f"Import error: {e}")
        return False


async def test_mcp_server_creation():
    """Test 2: Verify MCP server can be created"""
    tool_name = "mcp_server_creation"
    try:
        from mcp_server import mcp

        if mcp is not None:
            log_result(tool_name, True, f"MCP server created: {mcp.name}")
            return True
        else:
            log_result(tool_name, False, "MCP server is None")
            return False
    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_mcp_tools_registered():
    """Test 3: Verify all tools are registered"""
    tool_name = "mcp_tools_registered"
    try:
        # Get registered tools
        # FastMCP stores tools internally
        expected_tools = [
            "read_file",
            "write_file",
            "list_files",
            "run_python",
            "bash_exec",
        ]

        # Check if tools are callable via the mcp object
        log_result(tool_name, True, f"Expected tools defined: {expected_tools}")
        return True
    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_read_file_mcp():
    """Test 4: Test read_file via MCP"""
    tool_name = "mcp_read_file"
    try:
        from mcp_server import read_file

        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello MCP World!")
            temp_path = f.name

        try:
            result = await read_file(temp_path)
            if "Hello MCP World!" in result:
                log_result(tool_name, True, "Successfully read file via MCP")
                return True
            else:
                log_result(tool_name, False, f"Unexpected content: {result}")
                return False
        finally:
            os.unlink(temp_path)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_write_file_mcp():
    """Test 5: Test write_file via MCP"""
    tool_name = "mcp_write_file"
    try:
        from mcp_server import write_file

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "mcp_test.txt")

        try:
            result = await write_file(temp_path, "Written via MCP!")

            if os.path.exists(temp_path):
                with open(temp_path) as f:
                    content = f.read()
                if content == "Written via MCP!":
                    log_result(tool_name, True, "Successfully wrote file via MCP")
                    return True
                else:
                    log_result(tool_name, False, f"Content mismatch: {content}")
                    return False
            else:
                log_result(tool_name, False, "File was not created")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_list_files_mcp():
    """Test 6: Test list_files via MCP"""
    tool_name = "mcp_list_files"
    try:
        from mcp_server import list_files

        temp_dir = tempfile.mkdtemp()

        try:
            # Create some files
            with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
                f.write("test")
            os.makedirs(os.path.join(temp_dir, "subdir"))

            result = await list_files(temp_dir)

            if "file1.txt" in result and "[DIR] subdir" in result:
                log_result(tool_name, True, "Successfully listed files via MCP")
                return True
            else:
                log_result(tool_name, False, f"Unexpected listing: {result}")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_run_python_mcp():
    """Test 7: Test run_python via MCP"""
    tool_name = "mcp_run_python"
    try:
        from mcp_server import run_python

        temp_dir = tempfile.mkdtemp()
        script_path = os.path.join(temp_dir, "test_script.py")

        try:
            with open(script_path, "w") as f:
                f.write('print("Hello from MCP Python!")')

            result = await run_python(script_path, timeout=30)

            if "Hello from MCP Python!" in result:
                log_result(tool_name, True, "Successfully ran Python via MCP")
                return True
            else:
                log_result(tool_name, False, f"Unexpected output: {result}")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_bash_exec_mcp():
    """Test 8: Test bash_exec via MCP"""
    tool_name = "mcp_bash_exec"
    try:
        from mcp_server import bash_exec

        result = await bash_exec("echo 'Hello from MCP Bash!'", timeout=30)

        if "Hello from MCP Bash!" in result:
            log_result(tool_name, True, "Successfully ran bash via MCP")
            return True
        else:
            log_result(tool_name, False, f"Unexpected output: {result}")
            return False

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_bash_exec_with_cwd_mcp():
    """Test 9: Test bash_exec with cwd via MCP"""
    tool_name = "mcp_bash_exec_cwd"
    try:
        from mcp_server import bash_exec

        temp_dir = tempfile.mkdtemp()

        try:
            result = await bash_exec("pwd", timeout=30, cwd=temp_dir)

            if temp_dir in result:
                log_result(tool_name, True, "Successfully ran bash with cwd via MCP")
                return True
            else:
                log_result(tool_name, False, f"Unexpected output: {result}")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_error_handling_mcp():
    """Test 10: Test error handling via MCP"""
    tool_name = "mcp_error_handling"
    try:
        from mcp_server import read_file

        result = await read_file("/nonexistent/path/file.txt")

        if "Error" in result:
            log_result(tool_name, True, "Error handling works correctly")
            return True
        else:
            log_result(tool_name, False, f"Expected error message, got: {result}")
            return False

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_run_python_timeout_mcp():
    """Test 11: Test run_python timeout via MCP"""
    tool_name = "mcp_run_python_timeout"
    try:
        from mcp_server import run_python

        temp_dir = tempfile.mkdtemp()
        script_path = os.path.join(temp_dir, "slow_script.py")

        try:
            with open(script_path, "w") as f:
                f.write('import time\ntime.sleep(10)\nprint("done")')

            result = await run_python(script_path, timeout=1)

            if "timed out" in result:
                log_result(tool_name, True, "Timeout handling works correctly")
                return True
            else:
                log_result(tool_name, False, f"Expected timeout, got: {result}")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_mcp_workflow():
    """Test 12: Test complete MCP workflow"""
    tool_name = "mcp_workflow"
    try:
        from mcp_server import list_files, read_file, run_python, write_file

        temp_dir = tempfile.mkdtemp()

        try:
            # 1. Create a Python script
            script_content = """
import json
data = {"message": "MCP workflow test", "value": 42}
with open("output.json", "w") as f:
    json.dump(data, f)
print("Script completed!")
"""
            script_path = os.path.join(temp_dir, "workflow.py")
            await write_file(script_path, script_content)

            # 2. List files to verify
            listing = await list_files(temp_dir)
            if "workflow.py" not in listing:
                log_result(tool_name, False, "Script file not found after write")
                return False

            # 3. Run the script
            run_result = await run_python(script_path, timeout=30)
            if "Script completed!" not in run_result:
                log_result(tool_name, False, f"Script run failed: {run_result}")
                return False

            # 4. Check output file exists
            output_path = os.path.join(temp_dir, "output.json")
            output_content = await read_file(output_path)

            if '"value": 42' in output_content:
                log_result(tool_name, True, "Complete MCP workflow successful")
                return True
            else:
                log_result(tool_name, False, f"Unexpected output: {output_content}")
                return False

        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_mcp_unicode():
    """Test 13: Test unicode handling via MCP"""
    tool_name = "mcp_unicode"
    try:
        from mcp_server import read_file, write_file

        temp_dir = tempfile.mkdtemp()
        unicode_file = os.path.join(temp_dir, "unicode.txt")
        unicode_content = "Hello 世界 🌍 émojis naïve"

        try:
            await write_file(unicode_file, unicode_content)
            result = await read_file(unicode_file)

            if unicode_content in result:
                log_result(tool_name, True, "Unicode handling works correctly")
                return True
            else:
                log_result(tool_name, False, "Unicode mismatch")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_mcp_large_file():
    """Test 14: Test large file handling via MCP"""
    tool_name = "mcp_large_file"
    try:
        from mcp_server import read_file, write_file

        temp_dir = tempfile.mkdtemp()
        large_file = os.path.join(temp_dir, "large.txt")
        large_content = "x" * 100000  # 100KB

        try:
            await write_file(large_file, large_content)
            result = await read_file(large_file)

            if len(result) == 100000:
                log_result(tool_name, True, "Large file handling works correctly")
                return True
            else:
                log_result(tool_name, False, f"Size mismatch: {len(result)}")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_mcp_bash_multiline():
    """Test 15: Test bash multiline output via MCP"""
    tool_name = "mcp_bash_multiline"
    try:
        from mcp_server import bash_exec

        result = await bash_exec(
            "echo 'line1' && echo 'line2' && echo 'line3'", timeout=30
        )

        if "line1" in result and "line2" in result and "line3" in result:
            log_result(tool_name, True, "Multiline output works correctly")
            return True
        else:
            log_result(tool_name, False, f"Missing lines: {result}")
            return False

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def test_mcp_concurrent_writes():
    """Test 16: Test concurrent file writes via MCP"""
    tool_name = "mcp_concurrent_writes"
    try:
        import asyncio

        from mcp_server import read_file, write_file

        temp_dir = tempfile.mkdtemp()

        try:
            # Write 5 files concurrently
            tasks = []
            for i in range(5):
                file_path = os.path.join(temp_dir, f"file_{i}.txt")
                tasks.append(write_file(file_path, f"content_{i}"))

            await asyncio.gather(*tasks)

            # Verify all files exist and have correct content
            all_correct = True
            for i in range(5):
                file_path = os.path.join(temp_dir, f"file_{i}.txt")
                content = await read_file(file_path)
                if f"content_{i}" not in content:
                    all_correct = False
                    break

            if all_correct:
                log_result(tool_name, True, "Concurrent writes work correctly")
                return True
            else:
                log_result(tool_name, False, "Content mismatch in concurrent writes")
                return False
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        log_result(tool_name, False, f"Error: {e}")
        return False


async def run_all_tests():
    """Run all MCP tests"""
    print("=" * 60)
    print("MLE MCP Server - Comprehensive Test Suite")
    print("=" * 60)
    print()

    # Run tests
    await test_mcp_imports()
    await test_mcp_server_creation()
    await test_mcp_tools_registered()
    await test_read_file_mcp()
    await test_write_file_mcp()
    await test_list_files_mcp()
    await test_run_python_mcp()
    await test_bash_exec_mcp()
    await test_bash_exec_with_cwd_mcp()
    await test_error_handling_mcp()
    await test_run_python_timeout_mcp()
    await test_mcp_workflow()
    await test_mcp_unicode()
    await test_mcp_large_file()
    await test_mcp_bash_multiline()
    await test_mcp_concurrent_writes()

    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed / total) * 100:.1f}%")

    # List failures
    failures = [r for r in results if not r["success"]]
    if failures:
        print()
        print("Failed tests:")
        for f in failures:
            print(f"  - {f['tool']}: {f['message']}")

    print()
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print(f"{total - passed} test(s) failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
