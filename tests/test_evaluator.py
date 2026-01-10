#!/usr/bin/env python3
"""
Comprehensive test suite for the MLE Evaluator.
Tests rubric parsing, file checks, and LLM-based evaluation.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

# Add tools directory to path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
)

results = []


def log_result(test: str, passed: bool, message: str = ""):
    results.append((test, passed, message))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test}" + (f" - {message}" if message else ""))


# ============================================================================
# SECTION 1: RUBRIC PARSING TESTS
# ============================================================================


async def test_rubric_parsing():
    """Test rubric parsing with various formats"""
    print("\n" + "=" * 60)
    print("SECTION 1: RUBRIC PARSING TESTS")
    print("=" * 60)

    from evaluator import RubricEvaluator

    evaluator = RubricEvaluator()

    # Test 1: Basic format
    print("\n--- Basic Format ---")
    rubric1 = """+5|TYPE1|Criterion one
+3|TYPE2|Criterion two
+10|TYPE3|Criterion three"""

    criteria = evaluator.parse_rubric_md(rubric1)
    log_result("parse basic format", len(criteria) == 3)
    log_result(
        "correct points", criteria[0]["points"] == 5 and criteria[1]["points"] == 3
    )
    log_result(
        "correct types",
        criteria[0]["type"] == "TYPE1" and criteria[1]["type"] == "TYPE2",
    )

    # Test 2: With backticks in type
    print("\n--- Backticks in Type ---")
    rubric2 = """+5|`NO_FILE`|report.json exists at response/report.json
+5|`IDENTIFY_ISSUE_TYPE`|The report identifies DATA_LEAKAGE"""

    criteria = evaluator.parse_rubric_md(rubric2)
    log_result("parse backticks", len(criteria) == 2)
    log_result(
        "type with backticks",
        "`NO_FILE`" in criteria[0]["type"] or "NO_FILE" in criteria[0]["type"],
    )

    # Test 3: Negative points (penalties)
    print("\n--- Negative Points ---")
    rubric3 = """+5|GOOD|Good thing
-3|BAD|Penalty for bad thing
+2|OK|Another good thing"""

    criteria = evaluator.parse_rubric_md(rubric3)
    log_result("parse negative points", len(criteria) == 3)
    log_result("negative value", criteria[1]["points"] == -3)

    # Test 4: With empty lines and comments
    print("\n--- Empty Lines and Non-criteria ---")
    rubric4 = """
# This is a comment

+5|TYPE1|Criterion one

Some random text that should be ignored

+3|TYPE2|Criterion two
"""

    criteria = evaluator.parse_rubric_md(rubric4)
    log_result("ignore non-criteria lines", len(criteria) == 2)

    # Test 5: With justification column
    print("\n--- With Justification ---")
    rubric5 = """+5|TYPE1|Criterion one|This is the justification
+3|TYPE2|Criterion two|Another justification"""

    criteria = evaluator.parse_rubric_md(rubric5)
    log_result("parse justification", len(criteria) == 2)
    log_result(
        "has justification",
        criteria[0].get("justification") == "This is the justification",
    )

    # Test 6: Real rubric from task
    print("\n--- Real Task Rubric ---")
    real_rubric = """+5|`NO_FILE`|report.json exists at `../tasking/response/report.json`
+5|`IDENTIFY_ISSUE_TYPE`|The report explicitly indicates the error code, `DATA_LEAKAGE`, and only uses codes including `DATA_LEAKAGE`, `OUTLIER`, `SCALING_ERROR`, `ENCODING_ERROR`, `IMBALANCE`, `OVERFITTING`, `UNDERFITTING`, or `NONE`.
+5|`LOCATE_ISSUES`|The report explicitly indicates the location for the issue is around line 40-48 and/or `nearest_neighbor` function.
+5|`LOCATE_ISSUES`|The report explains that the original code uses TF-IDF fit on the entire training set before cross validation, causing data leakage.
+5|`EXPLAIN_EFFECT`|The report explains that the data leakage issues lead to inflated validation results.
+5|`PROPOSE_FIX`|The report proposes a way to use TF-IDF fit on the training fold only during cross validation."""

    criteria = evaluator.parse_rubric_md(real_rubric)
    log_result("parse real rubric", len(criteria) == 6)
    total_points = sum(c["points"] for c in criteria)
    log_result("total points correct", total_points == 30)


# ============================================================================
# SECTION 2: FILE EXISTENCE CHECKS
# ============================================================================


async def test_file_checks():
    """Test file existence evaluation"""
    print("\n" + "=" * 60)
    print("SECTION 2: FILE EXISTENCE CHECKS")
    print("=" * 60)

    from evaluator import RubricEvaluator

    evaluator = RubricEvaluator()

    temp_dir = tempfile.mkdtemp()

    try:
        # Create test files
        os.makedirs(os.path.join(temp_dir, "response"))
        with open(os.path.join(temp_dir, "response", "report.json"), "w") as f:
            json.dump({"test": "data"}, f)

        print("\n--- File Exists ---")
        criterion = {
            "points": 5,
            "type": "`NO_FILE`",
            "criterion": "report.json exists at response/report.json",
        }
        result = await evaluator.evaluate_criterion(criterion, temp_dir, "")
        log_result("detect existing file", result["passed"] == True)
        log_result("correct points", result["points"] == 5)

        print("\n--- File Missing ---")
        criterion2 = {
            "points": 5,
            "type": "`NO_FILE`",
            "criterion": "missing.json exists at response/missing.json",
        }
        result = await evaluator.evaluate_criterion(criterion2, temp_dir, "")
        log_result("detect missing file", result["passed"] == False)

        print("\n--- Nested Path ---")
        os.makedirs(os.path.join(temp_dir, "deep", "nested", "dir"))
        with open(
            os.path.join(temp_dir, "deep", "nested", "dir", "file.txt"), "w"
        ) as f:
            f.write("test")

        criterion3 = {
            "points": 3,
            "type": "`NO_FILE`",
            "criterion": "file.txt exists at deep/nested/dir/file.txt",
        }
        result = await evaluator.evaluate_criterion(criterion3, temp_dir, "")
        log_result("detect nested file", result["passed"] == True)

        print("\n--- Path with ../tasking/ prefix ---")
        # The real rubrics use ../tasking/response/report.json format
        criterion4 = {
            "points": 5,
            "type": "`NO_FILE`",
            "criterion": "report.json exists at `../tasking/response/report.json`",
        }
        result = await evaluator.evaluate_criterion(criterion4, temp_dir, "")
        log_result("handle ../tasking/ prefix", result["passed"] == True)

    finally:
        shutil.rmtree(temp_dir)


# ============================================================================
# SECTION 3: LLM-BASED EVALUATION
# ============================================================================


async def test_llm_evaluation():
    """Test LLM-based criterion evaluation"""
    print("\n" + "=" * 60)
    print("SECTION 3: LLM-BASED EVALUATION")
    print("=" * 60)

    from evaluator import RubricEvaluator

    evaluator = RubricEvaluator()

    temp_dir = tempfile.mkdtemp()

    try:
        print("\n--- Correct Answer Detection ---")
        criterion = {
            "points": 5,
            "type": "IDENTIFY_ISSUE",
            "criterion": "The answer identifies DATA_LEAKAGE as the primary error type",
        }
        agent_answer = """
After analyzing the code, I found that the primary issue is DATA_LEAKAGE.
The TF-IDF vectorizer is fitted on the entire training set before cross-validation,
which means validation data information leaks into the training process.
"""
        result = await evaluator.evaluate_criterion(criterion, temp_dir, agent_answer)
        log_result("identify correct answer", result["passed"] == True)
        log_result("has reasoning", len(result.get("reasoning", "")) > 0)

        print("\n--- Incorrect Answer Detection ---")
        criterion2 = {
            "points": 5,
            "type": "IDENTIFY_ISSUE",
            "criterion": "The answer identifies OVERFITTING as the primary error type",
        }
        # Same answer about DATA_LEAKAGE should fail OVERFITTING criterion
        result = await evaluator.evaluate_criterion(criterion2, temp_dir, agent_answer)
        log_result("detect incorrect answer", result["passed"] == False)

        print("\n--- Numeric Value Check ---")
        criterion3 = {
            "points": 3,
            "type": "CALCULATION",
            "criterion": "The answer correctly states that the validation score is approximately 0.85",
        }
        agent_answer2 = "The validation score achieved was 0.847, which rounds to 0.85."
        result = await evaluator.evaluate_criterion(criterion3, temp_dir, agent_answer2)
        log_result("check numeric value", result["passed"] == True)

        print("\n--- Code Location Check ---")
        criterion4 = {
            "points": 5,
            "type": "LOCATE_ISSUES",
            "criterion": "The report indicates the issue is around line 40-48 or in the nearest_neighbor function",
        }
        agent_answer3 = """
The bug is located in the nearest_neighbor function, specifically around lines 42-47.
The function searches through all training examples including the validation fold.
"""
        result = await evaluator.evaluate_criterion(criterion4, temp_dir, agent_answer3)
        log_result("check code location", result["passed"] == True)

        print("\n--- Empty Answer ---")
        criterion5 = {
            "points": 5,
            "type": "CONTENT",
            "criterion": "The answer provides a detailed explanation of the fix",
        }
        result = await evaluator.evaluate_criterion(criterion5, temp_dir, "")
        log_result("handle empty answer", result["passed"] == False)

    finally:
        shutil.rmtree(temp_dir)


# ============================================================================
# SECTION 4: FULL EVALUATION
# ============================================================================


async def test_full_evaluation():
    """Test complete evaluation workflow"""
    print("\n" + "=" * 60)
    print("SECTION 4: FULL EVALUATION")
    print("=" * 60)

    from evaluator import RubricEvaluator

    evaluator = RubricEvaluator()

    temp_dir = tempfile.mkdtemp()

    try:
        # Set up files
        os.makedirs(os.path.join(temp_dir, "response"))
        report = {
            "error_type": "DATA_LEAKAGE",
            "locations": "lines 40-48, nearest_neighbor function",
            "explanation": "TF-IDF is fitted on all data before CV, causing information leakage",
            "proposed_fix": "Fit TF-IDF only on training fold during each CV iteration",
        }
        with open(os.path.join(temp_dir, "response", "report.json"), "w") as f:
            json.dump(report, f, indent=2)

        with open(os.path.join(temp_dir, "response", "revised_code.py"), "w") as f:
            f.write("# Fixed code here")

        rubric = """+5|`NO_FILE`|report.json exists at `../tasking/response/report.json`
+3|`NO_FILE`|revised_code.py exists at `../tasking/response/revised_code.py`
+5|`IDENTIFY_ISSUE_TYPE`|The report identifies DATA_LEAKAGE as the error type
+5|`LOCATE_ISSUES`|The report indicates the location around line 40-48 or nearest_neighbor function
+5|`PROPOSE_FIX`|The report proposes fitting TF-IDF only on training data during CV"""

        agent_answer = """
I analyzed the code and found a DATA_LEAKAGE issue.

The problem is in the nearest_neighbor function around lines 40-48. The TF-IDF
vectorizer is fitted on the entire training set before cross-validation begins,
which means information from validation folds leaks into the training process.

FINAL ANSWER:
- Error type: DATA_LEAKAGE
- Location: lines 40-48, nearest_neighbor function
- Fix: Fit TF-IDF only on the training fold during each CV iteration
"""

        print("\n--- Full Evaluation ---")
        result = await evaluator.evaluate(rubric, temp_dir, agent_answer)

        log_result("evaluation completes", "score" in result)
        log_result("has total_possible", "total_possible" in result)
        log_result("has percentage", "percentage" in result)
        log_result("has results array", isinstance(result.get("results"), list))

        print(
            f"\n  Score: {result['score']}/{result['total_possible']} ({result['percentage']:.1f}%)"
        )

        # Check individual results
        passed_count = sum(1 for r in result["results"] if r["passed"])
        total_count = len(result["results"])
        log_result(
            f"criteria evaluated ({passed_count}/{total_count} passed)",
            total_count == 5,
        )

        print("\n  Individual results:")
        for r in result["results"]:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"    [{status}] {r['criterion'][:50]}...")

        # Score should be reasonable (at least files should pass)
        log_result(
            "reasonable score", result["score"] >= 8
        )  # At least file checks should pass

    finally:
        shutil.rmtree(temp_dir)


# ============================================================================
# SECTION 5: EDGE CASES AND ERROR HANDLING
# ============================================================================


async def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("SECTION 5: EDGE CASES AND ERROR HANDLING")
    print("=" * 60)

    from evaluator import RubricEvaluator

    evaluator = RubricEvaluator()

    temp_dir = tempfile.mkdtemp()

    try:
        print("\n--- Empty Rubric ---")
        result = await evaluator.evaluate("", temp_dir, "Some answer")
        log_result(
            "handle empty rubric",
            result["score"] == 0 and result["total_possible"] == 0,
        )

        print("\n--- No Matching Criteria ---")
        rubric = """This is not a valid rubric format
Just some random text
No criteria here"""
        result = await evaluator.evaluate(rubric, temp_dir, "Answer")
        log_result("handle no criteria", len(result["results"]) == 0)

        print("\n--- Very Long Answer ---")
        long_answer = "A" * 20000  # 20KB answer
        criterion = {
            "points": 5,
            "type": "TEST",
            "criterion": "Check something in the answer",
        }
        result = await evaluator.evaluate_criterion(criterion, temp_dir, long_answer)
        log_result("handle long answer", "reasoning" in result)  # Should not crash

        print("\n--- Special Characters in Criterion ---")
        criterion2 = {
            "points": 5,
            "type": "SPECIAL",
            "criterion": "Check for `code` and 'quotes' and \"double quotes\" and $pecial chars",
        }
        result = await evaluator.evaluate_criterion(
            criterion2, temp_dir, "Some answer with code"
        )
        log_result("handle special chars", "reasoning" in result)

        print("\n--- Unicode in Answer ---")
        unicode_answer = "The error is データリーケージ (data leakage) 🔍"
        criterion3 = {
            "points": 5,
            "type": "UNICODE",
            "criterion": "The answer mentions data leakage",
        }
        result = await evaluator.evaluate_criterion(
            criterion3, temp_dir, unicode_answer
        )
        log_result("handle unicode", "reasoning" in result)

    finally:
        shutil.rmtree(temp_dir)


# ============================================================================
# MAIN
# ============================================================================


async def main():
    print("=" * 60)
    print("MLE EVALUATOR - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")

    await test_rubric_parsing()
    await test_file_checks()
    await test_llm_evaluation()
    await test_full_evaluation()
    await test_edge_cases()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)
    total = len(results)

    print(
        f"\nTotal: {total} | Passed: {passed} | Failed: {failed} | Rate: {passed / total * 100:.1f}%"
    )

    if failed > 0:
        print("\nFailed tests:")
        for test, p, msg in results:
            if not p:
                print(f"  - {test}: {msg}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
