#!/usr/bin/env python3
"""CLI for running the MLE agent on tasks."""

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Dict

from agent import Agent
from dotenv import load_dotenv
from evaluator import RubricEvaluator
from llm import LLM
from tools import get_all_tools

load_dotenv()


async def run_task(
    task_json_path: str,
    model_name: str,
    temperature: float = 0.0,
    evaluate: bool = True,
    judge_model: str = "openai/gpt-4o-mini",
) -> Dict:
    """Run agent on a single task."""
    with open(task_json_path, "r") as f:
        task = json.load(f)

    # Extract prompt
    prompt_parts = []
    for msg in task.get("prompt", []):
        if msg.get("type") == "text":
            prompt_parts.append(msg.get("content", ""))
    prompt = "\n".join(prompt_parts)

    # Setup task directory if task_files present
    task_dir = task.get("task_dir", "/app")
    if task.get("task_files"):
        os.makedirs(task_dir, exist_ok=True)
        for path, content in task["task_files"].items():
            full_path = os.path.join(task_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Prepend working directory info to the prompt
        prompt = f"All project files are located in the directory: {task_dir}\nWhen accessing files, use paths like {task_dir}/src/code.py\n\n{prompt}"

    # Run agent
    llm = LLM(model_name, temperature=temperature)
    tools = get_all_tools()

    log_context = {
        "task_id": task.get("task_id"),
        "task_file": task_json_path,
        "model": model_name,
        "temperature": temperature,
        "judge_model": judge_model,
    }

    agent = Agent(
        tools,
        llm,
        max_turns=task.get("max_turns", 50),
        log_context=log_context,
    )

    print(f"Running task: {task.get('task_id', 'unknown')}")
    answer, metadata, trace = await agent.run(prompt)

    result = {
        "task_id": task.get("task_id"),
        "model": model_name,
        "temperature": temperature,
        "judge_model": judge_model,
        "answer": answer,
        "metadata": metadata,
        "timestamp": datetime.now().isoformat(),
        "trace": trace,
    }

    # Evaluate if rubric present
    if evaluate and task.get("rubric_text"):
        evaluator = RubricEvaluator(judge_model=judge_model)
        eval_result = await evaluator.evaluate(task["rubric_text"], task_dir, answer)
        eval_result["judge_model"] = judge_model
        result["evaluation"] = eval_result
        print(
            f"Score: {eval_result['score']}/{eval_result['total_possible']} ({eval_result['percentage']:.1f}%)"
        )

    return result


async def run_all_tasks(
    tasks_dir: str,
    model_name: str,
    output_dir: str,
    temperature: float = 0.0,
    judge_model: str = "openai/gpt-4o-mini",
    evaluate: bool = True,
):
    """Run agent on all tasks in directory."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    task_files = sorted([f for f in os.listdir(tasks_dir) if f.endswith(".json")])
    for task_file in task_files:
        task_path = os.path.join(tasks_dir, task_file)
        try:
            result = await run_task(
                task_path,
                model_name,
                temperature=temperature,
                evaluate=evaluate,
                judge_model=judge_model,
            )
            results.append(result)

            # Save individual result
            out_path = os.path.join(
                output_dir,
                f"result_{task_file}",
            )
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Error on {task_file}: {e}")
            results.append({"task_file": task_file, "error": str(e)})

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "temperature": temperature,
                "judge_model": judge_model,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="MLE Agent Runner")
    parser.add_argument("--task", type=str, help="Path to single task JSON")
    parser.add_argument(
        "--tasks-dir", type=str, default="tasks", help="Directory of task JSONs"
    )
    parser.add_argument(
        "--model", type=str, default="openai/gpt-4o-mini", help="Model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="LLM to use for rubric evaluation",
    )
    args = parser.parse_args()

    if args.task:
        result = asyncio.run(
            run_task(
                args.task,
                args.model,
                temperature=args.temperature,
                evaluate=not args.no_eval,
                judge_model=args.judge_model,
            )
        )
        print(json.dumps(result, indent=2))
    else:
        asyncio.run(
            run_all_tasks(
                args.tasks_dir,
                args.model,
                args.output_dir,
                temperature=args.temperature,
                judge_model=args.judge_model,
                evaluate=not args.no_eval,
            )
        )


if __name__ == "__main__":
    main()
