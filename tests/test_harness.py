#!/usr/bin/env python3
"""Comprehensive test suite for the MLE Reasoning Environment harness."""

import asyncio
import json
import os
import shutil
import sys
import tempfile

# Add tools directory to path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
)

from tools import BashExec, ListFiles, ReadFile, RunPython, WriteFile, get_all_tools


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        msg = f" - {self.message}" if self.message else ""
        return f"[{status}] {self.name}{msg}"


class HarnessTests:
    def __init__(self):
        self.results = []
        self.temp_dir = None

    def setup(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp(prefix="mle_test_")
        print(f"\nTest directory: {self.temp_dir}")

    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def add_result(self, name: str, passed: bool, message: str = ""):
        self.results.append(TestResult(name, passed, message))

    # ==================== Tool Tests ====================

    async def test_read_file(self):
        """Test ReadFile tool."""
        tool = ReadFile()

        # Create a test file
        test_file = os.path.join(self.temp_dir, "test_read.txt")
        test_content = "Hello, World!\nLine 2\nLine 3"
        with open(test_file, "w") as f:
            f.write(test_content)

        # Test successful read
        result = await tool({"path": test_file})
        if result["success"] and result["result"] == test_content:
            self.add_result("ReadFile - success", True)
        else:
            self.add_result("ReadFile - success", False, f"Got: {result}")

        # Test reading non-existent file
        result = await tool({"path": "/nonexistent/file.txt"})
        if not result["success"] and "Error" in result["result"]:
            self.add_result("ReadFile - nonexistent", True)
        else:
            self.add_result(
                "ReadFile - nonexistent", False, "Should fail for nonexistent file"
            )

    async def test_write_file(self):
        """Test WriteFile tool."""
        tool = WriteFile()

        # Test successful write
        test_file = os.path.join(self.temp_dir, "test_write.txt")
        content = "Test content"
        result = await tool({"path": test_file, "content": content})

        if result["success"] and os.path.exists(test_file):
            with open(test_file) as f:
                if f.read() == content:
                    self.add_result("WriteFile - success", True)
                else:
                    self.add_result("WriteFile - success", False, "Content mismatch")
        else:
            self.add_result("WriteFile - success", False, f"Got: {result}")

        # Test write to nested directory
        nested_file = os.path.join(self.temp_dir, "nested", "dir", "file.txt")
        result = await tool({"path": nested_file, "content": "nested"})
        if result["success"] and os.path.exists(nested_file):
            self.add_result("WriteFile - nested", True)
        else:
            self.add_result(
                "WriteFile - nested", False, "Failed to create nested directories"
            )

    async def test_list_files(self):
        """Test ListFiles tool."""
        tool = ListFiles()

        # Create some test files and directories
        os.makedirs(os.path.join(self.temp_dir, "subdir"))
        with open(os.path.join(self.temp_dir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(self.temp_dir, "file2.py"), "w") as f:
            f.write("test")

        result = await tool({"path": self.temp_dir})
        if result["success"]:
            listing = result["result"]
            if (
                "file1.txt" in listing
                and "file2.py" in listing
                and "[DIR] subdir" in listing
            ):
                self.add_result("ListFiles - success", True)
            else:
                self.add_result(
                    "ListFiles - success", False, f"Missing items: {listing}"
                )
        else:
            self.add_result("ListFiles - success", False, f"Got: {result}")

        # Test listing non-existent directory
        result = await tool({"path": "/nonexistent/dir"})
        if not result["success"]:
            self.add_result("ListFiles - nonexistent", True)
        else:
            self.add_result("ListFiles - nonexistent", False, "Should fail")

    async def test_run_python(self):
        """Test RunPython tool."""
        tool = RunPython()

        # Create a simple Python script
        script_path = os.path.join(self.temp_dir, "test_script.py")
        with open(script_path, "w") as f:
            f.write('print("Hello from Python!")\nimport sys\nsys.exit(0)')

        result = await tool({"script_path": script_path, "timeout": 30})
        if result["success"] and "Hello from Python!" in result["result"]:
            self.add_result("RunPython - success", True)
        else:
            self.add_result("RunPython - success", False, f"Got: {result}")

        # Test script with error
        error_script = os.path.join(self.temp_dir, "error_script.py")
        with open(error_script, "w") as f:
            f.write('raise ValueError("Test error")')

        result = await tool({"script_path": error_script, "timeout": 30})
        if not result["success"] and "ValueError" in result["result"]:
            self.add_result("RunPython - error handling", True)
        else:
            self.add_result("RunPython - error handling", False, f"Got: {result}")

        # Test script timeout (create script that sleeps)
        timeout_script = os.path.join(self.temp_dir, "timeout_script.py")
        with open(timeout_script, "w") as f:
            f.write('import time\ntime.sleep(10)\nprint("done")')

        result = await tool({"script_path": timeout_script, "timeout": 1})
        if not result["success"] and "timed out" in result["result"]:
            self.add_result("RunPython - timeout", True)
        else:
            self.add_result("RunPython - timeout", False, f"Got: {result}")

    async def test_bash_exec(self):
        """Test BashExec tool."""
        tool = BashExec()

        # Test simple command
        result = await tool({"command": "echo 'Hello World'", "timeout": 30})
        if result["success"] and "Hello World" in result["result"]:
            self.add_result("BashExec - echo", True)
        else:
            self.add_result("BashExec - echo", False, f"Got: {result}")

        # Test command with working directory
        result = await tool({"command": "pwd", "cwd": self.temp_dir, "timeout": 30})
        if result["success"] and self.temp_dir in result["result"]:
            self.add_result("BashExec - cwd", True)
        else:
            self.add_result("BashExec - cwd", False, f"Got: {result}")

        # Test failing command
        result = await tool({"command": "exit 1", "timeout": 30})
        if not result["success"] and "Return code: 1" in result["result"]:
            self.add_result("BashExec - failure", True)
        else:
            self.add_result("BashExec - failure", False, f"Got: {result}")

        # Test timeout
        result = await tool({"command": "sleep 10", "timeout": 1})
        if not result["success"] and "timed out" in result["result"]:
            self.add_result("BashExec - timeout", True)
        else:
            self.add_result("BashExec - timeout", False, f"Got: {result}")

    async def test_get_all_tools(self):
        """Test get_all_tools function."""
        tools = get_all_tools()
        expected = ["read_file", "write_file", "list_files", "run_python", "bash_exec"]

        if all(name in tools for name in expected):
            self.add_result("get_all_tools - all present", True)
        else:
            missing = [n for n in expected if n not in tools]
            self.add_result("get_all_tools - all present", False, f"Missing: {missing}")

        # Test tool JSON generation
        for name, tool in tools.items():
            try:
                json_def = tool.get_tool_json(provider="openai")
                if "function" in json_def and json_def["function"]["name"] == name:
                    self.add_result(f"Tool JSON - {name} (openai)", True)
                else:
                    self.add_result(
                        f"Tool JSON - {name} (openai)", False, "Invalid format"
                    )
            except Exception as e:
                self.add_result(f"Tool JSON - {name} (openai)", False, str(e))

            try:
                json_def = tool.get_tool_json(provider="anthropic")
                if "name" in json_def and json_def["name"] == name:
                    self.add_result(f"Tool JSON - {name} (anthropic)", True)
                else:
                    self.add_result(
                        f"Tool JSON - {name} (anthropic)", False, "Invalid format"
                    )
            except Exception as e:
                self.add_result(f"Tool JSON - {name} (anthropic)", False, str(e))

    # ==================== Integration Tests ====================

    async def test_ml_workflow(self):
        """Test a simulated ML workflow."""
        tools = get_all_tools()

        # 1. Create a data directory
        data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(data_dir)

        # 2. Write a simple dataset
        csv_content = "x,y\n1,2\n2,4\n3,6\n4,8"
        result = await tools["write_file"](
            {"path": os.path.join(data_dir, "train.csv"), "content": csv_content}
        )
        if not result["success"]:
            self.add_result("ML workflow - write data", False)
            return
        self.add_result("ML workflow - write data", True)

        # 3. Write a Python script that reads and processes the data
        script_content = """
import pandas as pd
import os

# Read training data
df = pd.read_csv('./data/train.csv')
print(f"Loaded {len(df)} samples")
print(f"Columns: {list(df.columns)}")
print(f"Mean y: {df['y'].mean():.2f}")
"""
        script_path = os.path.join(self.temp_dir, "analyze.py")
        result = await tools["write_file"](
            {"path": script_path, "content": script_content}
        )
        if not result["success"]:
            self.add_result("ML workflow - write script", False)
            return
        self.add_result("ML workflow - write script", True)

        # 4. Run the script
        result = await tools["run_python"]({"script_path": script_path, "timeout": 30})
        if result["success"] and "Loaded 4 samples" in result["result"]:
            self.add_result("ML workflow - run analysis", True)
        else:
            self.add_result("ML workflow - run analysis", False, f"Got: {result}")

        # 5. Create output directory and write results
        result = await tools["bash_exec"](
            {"command": f"mkdir -p {os.path.join(self.temp_dir, 'output')}"}
        )
        if result["success"]:
            self.add_result("ML workflow - create output dir", True)
        else:
            self.add_result("ML workflow - create output dir", False)

        # 6. Write a report
        report = {"analysis": "completed", "samples": 4, "mean_y": 5.0}
        result = await tools["write_file"](
            {
                "path": os.path.join(self.temp_dir, "output", "report.json"),
                "content": json.dumps(report, indent=2),
            }
        )
        if result["success"]:
            self.add_result("ML workflow - write report", True)
        else:
            self.add_result("ML workflow - write report", False)

        # 7. Verify the report
        result = await tools["read_file"](
            {"path": os.path.join(self.temp_dir, "output", "report.json")}
        )
        if result["success"]:
            loaded = json.loads(result["result"])
            if loaded["samples"] == 4:
                self.add_result("ML workflow - verify report", True)
            else:
                self.add_result("ML workflow - verify report", False, "Data mismatch")
        else:
            self.add_result("ML workflow - verify report", False)

    # ==================== Edge Case Tests ====================

    async def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        tools = get_all_tools()

        # Test 1: File with spaces in path
        space_file = os.path.join(self.temp_dir, "file with spaces.txt")
        result = await tools["write_file"](
            {"path": space_file, "content": "spaces work"}
        )
        if result["success"] and os.path.exists(space_file):
            self.add_result("Edge - file with spaces", True)
        else:
            self.add_result("Edge - file with spaces", False)

        # Test 2: Very long content
        long_content = "x" * 500000  # 500KB
        long_file = os.path.join(self.temp_dir, "long.txt")
        result = await tools["write_file"]({"path": long_file, "content": long_content})
        read_result = await tools["read_file"]({"path": long_file})
        if result["success"] and len(read_result["result"]) == 500000:
            self.add_result("Edge - 500KB file", True)
        else:
            self.add_result("Edge - 500KB file", False)

        # Test 3: Special characters in content
        special_content = (
            "Line with \"quotes\" and 'apostrophes' and $variables and `backticks`"
        )
        special_file = os.path.join(self.temp_dir, "special.txt")
        result = await tools["write_file"](
            {"path": special_file, "content": special_content}
        )
        read_result = await tools["read_file"]({"path": special_file})
        if result["success"] and read_result["result"] == special_content:
            self.add_result("Edge - special chars in content", True)
        else:
            self.add_result("Edge - special chars in content", False)

        # Test 4: Bash with pipes and redirects
        result = await tools["bash_exec"](
            {"command": "echo 'hello' | tr 'h' 'H' | tr 'e' 'E'"}
        )
        if result["success"] and "HEllo" in result["result"]:
            self.add_result("Edge - bash pipes", True)
        else:
            self.add_result("Edge - bash pipes", False)

        # Test 5: Python script with multiple outputs
        multi_script = os.path.join(self.temp_dir, "multi.py")
        with open(multi_script, "w") as f:
            f.write("import sys\nprint('stdout')\nprint('stderr', file=sys.stderr)")
        result = await tools["run_python"]({"script_path": multi_script, "timeout": 30})
        if (
            result["success"]
            and "stdout" in result["result"]
            and "stderr" in result["result"]
        ):
            self.add_result("Edge - python stdout+stderr", True)
        else:
            self.add_result("Edge - python stdout+stderr", False)

        # Test 6: Deep nested directory creation
        deep_path = os.path.join(
            self.temp_dir, "a", "b", "c", "d", "e", "f", "deep.txt"
        )
        result = await tools["write_file"]({"path": deep_path, "content": "deep"})
        if result["success"] and os.path.exists(deep_path):
            self.add_result("Edge - deep nested dirs", True)
        else:
            self.add_result("Edge - deep nested dirs", False)

        # Test 7: Overwrite file multiple times
        overwrite_file = os.path.join(self.temp_dir, "overwrite.txt")
        for i in range(5):
            await tools["write_file"](
                {"path": overwrite_file, "content": f"version {i}"}
            )
        result = await tools["read_file"]({"path": overwrite_file})
        if result["success"] and result["result"] == "version 4":
            self.add_result("Edge - multiple overwrites", True)
        else:
            self.add_result("Edge - multiple overwrites", False)

        # Test 8: Empty directory listing
        empty_dir = os.path.join(self.temp_dir, "empty_dir")
        os.makedirs(empty_dir)
        result = await tools["list_files"]({"path": empty_dir})
        if result["success"] and result["result"] == "":
            self.add_result("Edge - empty dir listing", True)
        else:
            self.add_result("Edge - empty dir listing", False)

        # Test 9: Bash command with environment variable
        result = await tools["bash_exec"](
            {"command": "TEST_VAR=hello && echo $TEST_VAR"}
        )
        if result["success"] and "hello" in result["result"]:
            self.add_result("Edge - bash env vars", True)
        else:
            self.add_result("Edge - bash env vars", False)

        # Test 10: Python script that creates files
        creator_script = os.path.join(self.temp_dir, "creator.py")
        output_from_script = os.path.join(self.temp_dir, "created_by_script.txt")
        with open(creator_script, "w") as f:
            f.write(
                f'with open("{output_from_script}", "w") as f: f.write("created")\nprint("done")'
            )
        result = await tools["run_python"](
            {"script_path": creator_script, "timeout": 30}
        )
        if result["success"] and os.path.exists(output_from_script):
            self.add_result("Edge - script creates file", True)
        else:
            self.add_result("Edge - script creates file", False)

    # ==================== Run All Tests ====================

    async def run_all(self):
        """Run all tests."""
        print("=" * 60)
        print("MLE Reasoning Environment Harness - Test Suite")
        print("=" * 60)

        self.setup()

        try:
            print("\n--- Tool Tests ---")
            await self.test_read_file()
            await self.test_write_file()
            await self.test_list_files()
            await self.test_run_python()
            await self.test_bash_exec()
            await self.test_get_all_tools()

            print("\n--- Edge Case Tests ---")
            await self.test_edge_cases()

            print("\n--- Integration Tests ---")
            await self.test_ml_workflow()

        finally:
            self.teardown()

        # Print results
        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        for result in self.results:
            print(result)

        print("\n" + "-" * 60)
        print(f"Total: {len(self.results)} | Passed: {passed} | Failed: {failed}")

        if failed == 0:
            print("\nAll tests passed!")
            return 0
        else:
            print(f"\n{failed} test(s) failed!")
            return 1


async def main():
    tests = HarnessTests()
    return await tests.run_all()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
