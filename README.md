# MLE Reasoning Environment

A harness for evaluating ML agents on Machine Learning Error analysis tasks.

## Overview

This harness enables testing AI agents on their ability to:
- Diagnose ML-specific errors in Python code (data leakage, scaling errors, encoding issues, etc.)
- Localize issues to specific code locations
- Propose and implement fixes
- Verify their fixes work correctly

## Architecture

```
mle-reasoning-environment/
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── files/                  # Static files mounted into container
├── tasks/                  # Task JSONs
└── tools/
    ├── tools.py            # Core tools (file I/O, Python exec, bash)
    ├── agent.py            # Agent loop with tool calling
    ├── llm.py              # LiteLLM abstraction for multiple providers
    ├── evaluator.py        # Rubric-based scoring
    ├── run_agent.py        # CLI for running single/batch tasks
    ├── agent_server.py     # Integration wrapper
    ├── mcp_server.py       # MCP server exposing tools
    └── test_harness.py     # Comprehensive test suite
```

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `read_file` | Read contents of a file |
| `write_file` | Write content to a file (creates directories) |
| `list_files` | List directory contents |
| `run_python` | Execute a Python script with timeout |
| `bash_exec` | Execute bash commands with timeout |

## Task Format

Tasks are in JSON format:

```json
{
  "task_id": "error-analysis-1",
  "prompt": [{"type": "text", "content": "...task description..."}],
  "rubrics": [
    {
      "name": "IDENTIFY_ISSUE_TYPE",
      "weight": 5,
      "score": {"type": "discrete", "outcomes": [...]},
      "messages": [{"type": "text", "content": "...criterion..."}]
    }
  ],
  "rubric_text": "+5|IDENTIFY_ISSUE_TYPE|...",
  "use_docker": true,
  "task_dir": "/app",
  "task_files": {"src/code.py": "...", "data/train.csv": "..."},
  "max_turns": 50
}
```

## Quick Start

### Prerequisites

1. Docker installed and running
2. API keys for your LLM provider:
   ```bash
   export OPENAI_API_KEY="your-key"
   # or
   export ANTHROPIC_API_KEY="your-key"
   ```

### Build the Container

```bash
cd mle-reasoning-environment
docker build -t mle-reasoning-environment .
```


### Copy tasks into /tasks

One task has been included in this repo for an example run (`error-analysis-1.json`)

```bash
cp -r path/to/your/task/dir/* mle-reasoning-environment/tasks/
```

### Run a Single Task

```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/tasks:/workspace/tasks \
  -v $(pwd)/results:/workspace/results \
  mle-reasoning-environment \
  python run_agent.py --task /workspace/tasks/task_error-analysis-1-dev.json
```

### Run All Tasks

```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/tasks:/workspace/tasks \
  -v $(pwd)/results:/workspace/results \
  mle-reasoning-environment \
  python run_agent.py --tasks-dir /workspace/tasks --output-dir /workspace/results
```

### Local Development (without Docker)

```bash
cd tools
pip install -r ../requirements.txt
python run_agent.py --task ../tasks/task_error-analysis-1-dev.json --model openai/gpt-4o-mini
```

## Testing

Run the comprehensive test suite:

```bash
cd tools
python test_harness.py
```

Tests cover:
- All file operations (read, write, list)
- Python script execution with timeout
- Bash command execution
- Tool JSON schema generation
- End-to-end ML workflow simulation

## Error Types Detected

| Code | Description |
|------|-------------|
| `DATA_LEAKAGE` | Model sees validation/test data during training |
| `OUTLIER` | Extreme samples not handled properly |
| `SCALING_ERROR` | Features scaled with wrong statistics |
| `ENCODING_ERROR` | Categorical variables encoded incorrectly |
| `IMBALANCE` | Class distribution skewed and not handled |
| `OVERFITTING` | Good train, poor validation performance |
| `UNDERFITTING` | Poor performance on both splits |
| `NONE` | No ML-specific errors present |

## Evaluation

The evaluator uses rubric-based scoring:

1. **File checks**: Verify required output files exist
2. **Content analysis**: LLM-judged criteria for report quality
3. **Code verification**: Check that fixes are implemented correctly

Scoring breakdown for a typical task (90 total points):
- Issue identification: 15 points
- Issue localization: 20 points
- Explanation quality: 5 points
- Fix proposal: 20 points
- Fix implementation: 12 points
- Format requirements: 8 points
- Output file checks: 13 points

## Provider Support

The harness supports multiple LLM providers via LiteLLM:

| Provider | Model Format |
|----------|-------------|
| OpenAI | `openai/gpt-4o-mini`, `openai/gpt-4o` |
| Anthropic | `anthropic/claude-3-5-sonnet-latest` |
| Google | `google/gemini-1.5-pro` |

## MCP Server

For MCP-compatible clients:

```bash
python mcp_server.py
```

Exposes all tools via the Model Context Protocol.

## Output Format

Agent results are saved as JSON:

```json
{
  "task_id": "error-analysis-1-dev",
  "model": "openai/gpt-4o-mini",
  "answer": "FINAL ANSWER: ...",
  "metadata": {
    "turns": 12,
    "tool_calls": 25,
    "errors": []
  },
  "evaluation": {
    "score": 75,
    "total_possible": 90,
    "percentage": 83.3,
    "results": [...]
  }
}
```

## Troubleshooting

### Docker daemon not running
```bash
# macOS: Start Docker Desktop
open -a Docker
```

### Missing API keys
```bash
# Check if set
echo $OPENAI_API_KEY
# Set if missing
export OPENAI_API_KEY="sk-..."
```

### Python not found in container
The harness auto-detects `python` or `python3` - this should work automatically.
