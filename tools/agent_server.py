#!/usr/bin/env python3
"""Reasoning Environments agent wrapper - receives prompts and runs the MLE agent."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agent import Agent
from dotenv import load_dotenv
from llm import LLM

from tools import get_all_tools

load_dotenv()


def read_task_prompt():
    """Read task prompt from various sources."""
    if "TASK_PROMPT" in os.environ:
        return os.environ["TASK_PROMPT"]
    prompt_file = os.environ.get("PROMPT_FILE", "/workspace/task_prompt.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            return f.read().strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


async def run_agent_with_prompt(
    prompt: str, model_name: str, task_json_path: str = None, temperature: float = 0.0
):
    """Run agent with prompt and save output."""
    llm = LLM(model_name, temperature=temperature)
    tools = get_all_tools()
    log_context = {
        "model": model_name,
        "temperature": temperature,
        "task_file": task_json_path,
    }
    agent = Agent(tools, llm, max_turns=50, log_context=log_context)

    answer, metadata, trace = await agent.run(prompt)

    output = {
        "prompt": prompt,
        "model": model_name,
        "temperature": temperature,
        "answer": answer,
        "metadata": metadata,
        "trace": trace,
    }
    with open("/tmp/agent_output.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save timestamped result
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = Path(task_json_path).stem if task_json_path else "agent"
    with open(results_dir / f"result_{task_name}_{ts}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== FINAL ANSWER ===\n{answer}")
    return answer


def main():
    parser = argparse.ArgumentParser(description="Reasoning Environments Agent Wrapper")
    parser.add_argument("-p", "--prompt", help="Direct prompt")
    parser.add_argument("-f", "--file", help="File containing prompt")
    parser.add_argument(
        "-m", "--model", default="openai/gpt-4o-mini", help="Model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model",
    )
    parser.add_argument("--task-json", help="Path to task JSON file")
    args = parser.parse_args()

    prompt = None
    if args.prompt:
        prompt = args.prompt
    elif args.file:
        with open(args.file, "r") as f:
            prompt = f.read().strip()
    elif args.task_json:
        with open(args.task_json, "r") as f:
            task = json.load(f)
        prompt = "\n".join(
            m.get("content", "")
            for m in task.get("prompt", [])
            if m.get("type") == "text"
        )
    else:
        prompt = read_task_prompt()

    if not prompt:
        print(
            "No prompt provided. Use --prompt, --file, --task-json, or TASK_PROMPT env var."
        )
        return 1

    asyncio.run(
        run_agent_with_prompt(
            prompt, args.model, args.task_json, temperature=args.temperature
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
