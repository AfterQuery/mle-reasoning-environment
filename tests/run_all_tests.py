#!/usr/bin/env python3
"""
Run all MLE Reasoning Environment test suites.
This script executes all test files and provides a combined summary.
"""

import os
import subprocess
import sys


def run_test(test_file: str, description: str) -> bool:
    """Run a single test file and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, test_file], cwd=os.path.dirname(os.path.abspath(__file__))
    )

    return result.returncode == 0


def main():
    """Run all test suites."""
    print("=" * 60)
    print("MLE REASONING ENVIRONMENT - COMPLETE TEST SUITE")
    print("=" * 60)

    tests = [
        ("test_harness.py", "Tool Tests (40 tests)"),
        ("test_mcp_server.py", "MCP Server Tests (16 tests)"),
        ("test_comprehensive.py", "Comprehensive Tests (61 tests)"),
    ]

    # Check if we have API key for evaluator tests
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    if has_api_key:
        tests.append(
            ("test_evaluator.py", "Evaluator Tests (34 tests) - requires API key")
        )
    else:
        print("\nNote: Skipping test_evaluator.py (requires OPENAI_API_KEY)")

    results = []
    for test_file, description in tests:
        success = run_test(test_file, description)
        results.append((test_file, description, success))

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, _, s in results if s)
    failed = sum(1 for _, _, s in results if not s)

    for test_file, description, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {description}")

    print(f"\nTest Suites: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\nAll test suites passed!")
        return 0
    else:
        print(f"\n{failed} test suite(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
