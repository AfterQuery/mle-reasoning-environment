#!/usr/bin/env python3
"""
HTTP Server for MLE Reasoning Environment
==========================================

A FastAPI-based HTTP server that wraps the MLE agent for direct API access.
Deploy to Cloud Run for a simple POST /run endpoint.

Usage:
  uvicorn http_server:app --host 0.0.0.0 --port 8080

Endpoints:
  POST /run        - Run agent synchronously (legacy, may timeout)
  POST /run-async  - Run agent asynchronously, returns job_id immediately
  POST /run-sync   - Run agent synchronously (for Cloud Tasks integration)
  GET /jobs/{id}   - Check job status
  GET /health      - Health check
"""

import asyncio
import httpx
import ipaddress
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from agent import Agent
from evaluator import RubricEvaluator
from llm import LLM
from tools import get_all_tools

load_dotenv()

app = FastAPI(
    title="MLE Reasoning Environment API",
    description="Direct API for running MLE agent tasks",
    version="2.0.0",
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Allowed callback domains (comma-separated). If empty, all non-private domains allowed.
ALLOWED_CALLBACK_DOMAINS = os.environ.get("ALLOWED_CALLBACK_DOMAINS", "").strip()

# Job store settings
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))  # 1 hour default
MAX_JOBS = int(os.environ.get("MAX_JOBS", "1000"))  # Max jobs in store

# Whether to require HTTPS for callbacks (default True in production)
REQUIRE_HTTPS_CALLBACKS = os.environ.get("REQUIRE_HTTPS_CALLBACKS", "true").lower() == "true"


# ============================================================================
# JOB STORE
# ============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, task_id: str, callback_url: Optional[str] = None):
        self.job_id = job_id
        self.task_id = task_id
        self.callback_url = callback_url
        self.status = JobStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None


# In-memory job store (sufficient for Cloud Run since jobs complete within instance lifetime)
jobs: Dict[str, Job] = {}


# ============================================================================
# SECURITY: CALLBACK URL VALIDATION
# ============================================================================

def validate_callback_url(url: str) -> bool:
    """
    Validate callback URL for SSRF safety.

    Returns True if URL is safe, False otherwise.
    Rejects:
    - Non-HTTPS URLs (when REQUIRE_HTTPS_CALLBACKS is True)
    - localhost/loopback addresses (127.0.0.1, ::1, localhost)
    - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
    - Link-local addresses (169.254.x.x)
    - Cloud metadata service (169.254.169.254)
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
    except Exception:
        print(f"[callback-validation] Failed to parse URL: {url}")
        return False

    # Check scheme
    if REQUIRE_HTTPS_CALLBACKS:
        if parsed.scheme != "https":
            print(f"[callback-validation] Rejected non-HTTPS URL: {url}")
            return False
    else:
        if parsed.scheme not in ("http", "https"):
            print(f"[callback-validation] Rejected invalid scheme: {url}")
            return False

    # Check hostname exists
    hostname = parsed.hostname
    if not hostname:
        print(f"[callback-validation] Rejected URL with no hostname: {url}")
        return False

    # Check against domain allowlist if configured
    if ALLOWED_CALLBACK_DOMAINS:
        allowed = [d.strip().lower() for d in ALLOWED_CALLBACK_DOMAINS.split(",") if d.strip()]
        if hostname.lower() not in allowed:
            print(f"[callback-validation] Rejected URL not in allowlist: {url}")
            return False

    # Check for localhost variants first (before IP parsing)
    lower_hostname = hostname.lower()
    if lower_hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        print(f"[callback-validation] Rejected localhost: {url}")
        return False

    # Check for metadata service IP (cloud SSRF target)
    if lower_hostname == "169.254.169.254":
        print(f"[callback-validation] Rejected metadata service IP: {url}")
        return False

    # Check for private/loopback/link-local IPs
    try:
        # Try to parse as IP address directly
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            print(f"[callback-validation] Rejected private/loopback/link-local IP: {url}")
            return False
    except ValueError:
        # Not an IP address, it's a hostname - that's fine
        pass

    return True


# ============================================================================
# JOB STORE CLEANUP
# ============================================================================

def cleanup_old_jobs() -> int:
    """
    Remove completed jobs older than TTL and enforce max size.

    Returns the number of jobs removed.
    """
    now = datetime.now(timezone.utc)
    removed = 0

    # First pass: remove completed jobs older than TTL
    jobs_to_remove = []
    for job_id, job in jobs.items():
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            if job.completed_at:
                age_seconds = (now - job.completed_at).total_seconds()
                if age_seconds > JOB_TTL_SECONDS:
                    jobs_to_remove.append(job_id)

    for job_id in jobs_to_remove:
        del jobs[job_id]
        removed += 1

    # Second pass: if still over max size, remove oldest completed jobs (LRU)
    if len(jobs) > MAX_JOBS:
        # Sort completed jobs by completed_at (oldest first)
        completed_jobs = [
            (job_id, job) for job_id, job in jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED) and job.completed_at
        ]
        completed_jobs.sort(key=lambda x: x[1].completed_at or now)

        # Remove oldest until under limit
        while len(jobs) > MAX_JOBS and completed_jobs:
            job_id, _ = completed_jobs.pop(0)
            if job_id in jobs:
                del jobs[job_id]
                removed += 1

    if removed > 0:
        print(f"[job-cleanup] Removed {removed} old jobs, {len(jobs)} remaining")

    return removed


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


class AsyncTaskRequest(TaskRequest):
    """Task request for async execution with webhook callback."""

    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results when job completes"
    )
    callback_secret: Optional[str] = Field(
        default=None,
        description="Secret to include in callback for verification"
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


class AsyncTaskResponse(BaseModel):
    """Response from submitting an async task."""

    job_id: str
    task_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status check."""

    job_id: str
    task_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WebhookPayload(BaseModel):
    """Payload sent to callback URL when job completes."""

    job_id: str
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    secret: Optional[str] = None


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
    started_at = datetime.now(timezone.utc)
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

        completed_at = datetime.now(timezone.utc)

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
        completed_at = datetime.now(timezone.utc)
        return TaskResponse(
            task_id=request.task_id,
            status="timeout",
            error="Task execution timed out",
            started_at=started_at.isoformat() + "Z",
            completed_at=completed_at.isoformat() + "Z",
            execution_time_seconds=(completed_at - started_at).total_seconds(),
        )

    except Exception as e:
        completed_at = datetime.now(timezone.utc)
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


async def run_task_async(job_id: str, request: AsyncTaskRequest) -> None:
    """
    Execute a task asynchronously and update job status.
    Called as a background task.
    """
    job = jobs.get(job_id)
    if not job:
        return

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)

    try:
        # Run the task
        result = await run_task(request)

        # Update job with results
        job.status = JobStatus.COMPLETED if result.status == "completed" else JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.result = result.model_dump()
        if result.error:
            job.error = result.error

    except Exception as e:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error = str(e)

    # Send webhook callback if configured
    if request.callback_url:
        await send_webhook_callback(job, request.callback_url, request.callback_secret)


async def send_webhook_callback(
    job: Job,
    callback_url: str,
    callback_secret: Optional[str] = None
) -> None:
    """Send job completion notification to callback URL."""
    payload = WebhookPayload(
        job_id=job.job_id,
        task_id=job.task_id,
        status=job.status.value,
        result=job.result,
        error=job.error,
        secret=callback_secret,
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                callback_url,
                json=payload.model_dump(),
                headers={"Content-Type": "application/json"},
            )
            if response.status_code >= 400:
                print(f"[webhook] Callback to {callback_url} failed with status {response.status_code}")
            else:
                print(f"[webhook] Callback to {callback_url} succeeded")
    except Exception as e:
        print(f"[webhook] Failed to send callback to {callback_url}: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "active_jobs": len([j for j in jobs.values() if j.status == JobStatus.RUNNING]),
        "total_jobs": len(jobs),
    }


@app.post("/run", response_model=TaskResponse)
async def run_task_endpoint(request: TaskRequest):
    """
    Run an MLE agent task synchronously.

    WARNING: This endpoint may timeout for long-running tasks.
    For production use, prefer /run-async.
    """
    return await run_task(request)


@app.post("/run-async", response_model=AsyncTaskResponse)
async def run_task_async_endpoint(
    request: AsyncTaskRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit an MLE agent task for async execution.

    Returns immediately with a job_id. The task runs in the background.
    When complete, results are POSTed to callback_url if provided.
    Use GET /jobs/{job_id} to check status.
    """
    # Cleanup old jobs before accepting new ones
    cleanup_old_jobs()

    # Validate callback URL if provided (SSRF protection)
    if request.callback_url:
        if not validate_callback_url(request.callback_url):
            raise HTTPException(
                status_code=400,
                detail="Invalid callback_url: must be HTTPS and not target private/local addresses"
            )

    job_id = str(uuid.uuid4())

    # Create job record
    job = Job(
        job_id=job_id,
        task_id=request.task_id,
        callback_url=request.callback_url,
    )
    jobs[job_id] = job

    # Schedule background task
    background_tasks.add_task(run_task_async, job_id, request)

    return AsyncTaskResponse(
        job_id=job_id,
        task_id=request.task_id,
        status=JobStatus.PENDING.value,
        message="Task submitted successfully. Use GET /jobs/{job_id} to check status.",
    )


@app.post("/run-sync", response_model=TaskResponse)
async def run_task_sync_endpoint(request: TaskRequest):
    """
    Run an MLE agent task synchronously (blocking).

    Designed for use with Cloud Tasks or other task queue systems that
    make HTTP requests and wait for completion. The HTTP connection stays
    open until the task completes, which keeps Cloud Run instances alive.

    Accepts the same payload as /run-async but ignores callback fields.
    Returns the full TaskResponse with evaluation results.

    Recommended Cloud Run settings when using this endpoint:
    - --timeout=3600 (1 hour max)
    - --concurrency=1 (one task per instance)
    - --memory=4Gi --cpu=2
    """
    print(f"[run-sync] Starting synchronous run for task {request.task_id}")
    if request.metadata:
        # Log only keys, not values (may contain PII)
        print(f"[run-sync] Metadata keys: {list(request.metadata.keys())}")

    result = await run_task(request)

    print(f"[run-sync] Completed task {request.task_id} with status {result.status}")
    return result


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of an async job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        task_id=job.task_id,
        status=job.status.value,
        created_at=job.created_at.isoformat() + "Z",
        started_at=job.started_at.isoformat() + "Z" if job.started_at else None,
        completed_at=job.completed_at.isoformat() + "Z" if job.completed_at else None,
        result=job.result,
        error=job.error,
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
