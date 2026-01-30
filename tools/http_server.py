#!/usr/bin/env python3
"""
HTTP Server for MLE Reasoning Environment
==========================================

A FastAPI-based HTTP server that wraps the MLE agent for direct API access.
Deploy to Cloud Run for a simple POST /run endpoint.

Usage:
  uvicorn http_server:app --host 0.0.0.0 --port 8080

Endpoints:
  POST /run     - Run agent on a single task, return results
  GET /health   - Health check
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import Agent
from evaluator import RubricEvaluator
from llm import LLM
from tools import get_all_tools

load_dotenv()

app = FastAPI(
    title="MLE Reasoning Environment API",
    description="Direct API for running MLE agent tasks",
    version="1.0.0",
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class PromptItem(BaseModel):
    type: str = "text"
    content: str


class TaskRequest(BaseModel):
    """Docker Worlds format task request."""

    task_id: str = Field(..., description="Unique task identifier")
    prompt: List[PromptItem] = Field(..., description="Task prompt")
    rubric_text: str = Field(default="", description="Pipe-delimited rubric text")
    task_files: Dict[str, str] = Field(
        default={}, description="Files to create: {path: content}"
    )
    task_dir: str = Field(default="/app", description="Working directory for task")
    max_turns: int = Field(default=50, description="Max agent turns")

    # Execution options
    model: str = Field(default="openai/gpt-4o-mini", description="Model to use")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    judge_model: str = Field(
        default="openai/gpt-4o-mini", description="Model for evaluation"
    )
    run_evaluation: bool = Field(default=True, description="Whether to run evaluation")

    # Metadata for tracking (passed through to response)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom metadata to include in response"
    )


class EvaluationResult(BaseModel):
    score: int
    total_possible: int
    percentage: float
    results: List[Dict[str, Any]]


class TaskResponse(BaseModel):
    """Response from running a task."""

    task_id: str
    status: str  # "completed", "failed", "timeout"
    answer: Optional[str] = None
    metadata: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    evaluation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: str
    completed_at: str
    execution_time_seconds: float


# ============================================================================
# TASK EXECUTION
# ============================================================================


async def setup_task_files(task_dir: str, task_files: Dict[str, str]) -> None:
    """Create task files in the working directory."""
    os.makedirs(task_dir, exist_ok=True)
    for rel_path, content in task_files.items():
        full_path = os.path.join(task_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)


async def run_task(request: TaskRequest) -> TaskResponse:
    """Execute a single task and return results."""
    started_at = datetime.utcnow()
    task_dir = None

    try:
        # Create temporary working directory
        task_dir = tempfile.mkdtemp(prefix=f"mle_task_{request.task_id}_")

        # Setup task files
        if request.task_files:
            await setup_task_files(task_dir, request.task_files)

        # Build prompt text
        prompt_parts = []
        for item in request.prompt:
            if item.type == "text":
                prompt_parts.append(item.content)
        prompt = "\n".join(prompt_parts)

        # Prepend working directory info
        if request.task_files:
            prompt = f"All project files are located in the directory: {task_dir}\nWhen accessing files, use paths like {task_dir}/src/code.py\n\n{prompt}"

        # Initialize agent
        llm = LLM(request.model, temperature=request.temperature)
        tools = get_all_tools()

        log_context = {
            "task_id": request.task_id,
            "model": request.model,
            "temperature": request.temperature,
            "judge_model": request.judge_model,
        }

        agent = Agent(
            tools,
            llm,
            max_turns=request.max_turns,
            log_context=log_context,
        )

        # Run agent
        answer, agent_metadata, trace = await agent.run(prompt)

        # Run evaluation if requested
        evaluation = None
        if request.run_evaluation and request.rubric_text:
            evaluator = RubricEvaluator(judge_model=request.judge_model)
            evaluation = await evaluator.evaluate(
                request.rubric_text, task_dir, answer or ""
            )
            evaluation["judge_model"] = request.judge_model

        completed_at = datetime.utcnow()

        # Merge custom metadata with agent metadata
        response_metadata = {**agent_metadata}
        if request.metadata:
            response_metadata["custom"] = request.metadata

        return TaskResponse(
            task_id=request.task_id,
            status="completed",
            answer=answer,
            metadata=response_metadata,
            trace=trace,
            evaluation=evaluation,
            started_at=started_at.isoformat() + "Z",
            completed_at=completed_at.isoformat() + "Z",
            execution_time_seconds=(completed_at - started_at).total_seconds(),
        )

    except asyncio.TimeoutError:
        completed_at = datetime.utcnow()
        return TaskResponse(
            task_id=request.task_id,
            status="timeout",
            error="Task execution timed out",
            started_at=started_at.isoformat() + "Z",
            completed_at=completed_at.isoformat() + "Z",
            execution_time_seconds=(completed_at - started_at).total_seconds(),
        )

    except Exception as e:
        completed_at = datetime.utcnow()
        return TaskResponse(
            task_id=request.task_id,
            status="failed",
            error=str(e),
            started_at=started_at.isoformat() + "Z",
            completed_at=completed_at.isoformat() + "Z",
            execution_time_seconds=(completed_at - started_at).total_seconds(),
        )

    finally:
        # Cleanup temp directory
        if task_dir and os.path.exists(task_dir):
            shutil.rmtree(task_dir, ignore_errors=True)


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.post("/run", response_model=TaskResponse)
async def run_task_endpoint(request: TaskRequest):
    """
    Run an MLE agent task.

    Accepts a Docker Worlds format task JSON, runs the agent,
    optionally evaluates the result, and returns the response.
    """
    return await run_task(request)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
