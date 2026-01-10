# MLE Reasoning environment - Testing & Usage Guide

## What Is This Project?

**MLE Reasoning environment** is a benchmark harness for evaluating AI agents on **Machine Learning Error (MLE) analysis tasks**.

The benchmark tests an agent's ability to:
1. **Diagnose** ML-specific bugs in Python code (data leakage, scaling errors, encoding issues, etc.)
2. **Localize** issues to specific code locations
3. **Explain** why the issue causes problems
4. **Propose and implement** fixes
5. **Verify** the fixes work correctly

---

## Project Structure

```
mle-reasoning-environment/
├── Dockerfile              # Container definition (tools only)
├── requirements.txt        # Python dependencies
├── README.md               # Quick start guide
├── TESTING_GUIDE.md        # This file
│
├── tools/                  # Core harness code
│   ├── tools.py            # 5 agent tools (read, write, list, python, bash)
│   ├── agent.py            # Agent loop with LLM integration
│   ├── llm.py              # LiteLLM wrapper for multiple providers
│   ├── evaluator.py        # Rubric-based scoring with LLM judge
│   ├── run_agent.py        # CLI for running tasks
│   ├── mcp_server.py       # MCP server exposing tools
│   └── agent_server.py     # Reasoning environments integration
│
├── tests/                  # Test suites
│   ├── run_all_tests.py    # Run all test suites
│   ├── test_harness.py     # Tool tests (30 tests)
│   ├── test_mcp_server.py  # MCP server tests (12 tests)
│   ├── test_comprehensive.py # Full harness tests (66 tests)
│   └── test_evaluator.py   # Evaluator tests (34 tests, requires API key)
│
├── tasks/                  # Task JSON files (Reasoning environments format)
│   └── task_*.json
│
└── files/                  # Static files for container
    └── README.md
```

---

## Tools Available to Agents

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read file contents | `path` |
| `write_file` | Write/create files (auto-creates directories) | `path`, `content` |
| `list_files` | List directory contents | `path` |
| `run_python` | Execute Python script with timeout | `script_path`, `timeout` |
| `bash_exec` | Execute bash command with timeout | `command`, `timeout`, `cwd` |

---

## Running Tests

### Prerequisites

```bash
# Install dependencies
cd mle-reasoning-environment
pip install -r requirements.txt

# For evaluator tests, set API key
export OPENAI_API_KEY="your-key"
```

### Quick Test Commands

```bash
cd mle-reasoning-environment/tests

# Run individual test suites
python test_harness.py         # 30 tool tests (no API key needed)
python test_mcp_server.py      # 12 MCP tests (no API key needed)
python test_comprehensive.py   # 66 comprehensive tests (no API key needed)
python test_evaluator.py       # 34 evaluator tests (requires API key)

# Run all tests at once
python run_all_tests.py
```

### Test Suites Explained

| Test File | Tests | API Key? | What It Tests |
|-----------|-------|----------|---------------|
| `test_harness.py` | 40 | No | All 5 tools with edge cases |
| `test_mcp_server.py` | 16 | No | MCP server and tool exposure |
| `test_comprehensive.py` | 61 | No | Tools, agent, schema, integration |
| `test_evaluator.py` | 34 | Yes | Rubric parsing, file checks, LLM grading |

**Total: 151 tests**

---

## Running with Docker

### Build the Container

```bash
cd mle-reasoning-environment
docker build -t mle-reasoning-environment .
```

### Run a Task

```bash
# Mount your tasks and run
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/tasks:/workspace/tasks \
  mle-reasoning-environment \
  python run_agent.py \
  --task /workspace/tasks/task_error-analysis-1-dev.json \
  --model openai/gpt-4o-mini
```

### Run Tests in Docker

```bash
# Mount tests folder and run
docker run --rm \
  -v $(pwd)/tests:/workspace/tests \
  -v $(pwd)/tasks:/workspace/tasks \
  mle-reasoning-environment \
  python /workspace/tests/test_comprehensive.py
```

### Run Without Evaluation (Faster)

```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/tasks:/workspace/tasks \
  mle-reasoning-environment \
  python run_agent.py \
  --task /workspace/tasks/task_error-analysis-1-dev.json \
  --model openai/gpt-4o-mini \
  --no-eval
```

---

## Local Development (No Docker)

```bash
cd mle-reasoning-environment/tools

# Install dependencies
pip install -r ../requirements.txt

# Run a task
python run_agent.py \
  --task ../tasks/task_error-analysis-1-dev.json \
  --model openai/gpt-4o-mini

# Run all tasks
python run_agent.py \
  --tasks-dir ../tasks \
  --output-dir ../results \
  --model openai/gpt-4o-mini
```

---

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Path to single task JSON | - |
| `--tasks-dir` | Directory of task JSONs | `tasks` |
| `--model` | LLM model (provider/model) | `openai/gpt-4o-mini` |
| `--output-dir` | Where to save results | `results` |
| `--no-eval` | Skip evaluation (faster) | False |

---

## Supported Models

| Provider | Model Examples |
|----------|----------------|
| OpenAI | `openai/gpt-4o-mini`, `openai/gpt-4o` |
| Anthropic | `anthropic/claude-3-5-sonnet-latest` |
| Google | `google/gemini-1.5-pro` |

---

## Task JSON Format

```json
{
  "task_id": "error-analysis-1-dev",
  "prompt": [{"type": "text", "content": "...task description..."}],
  "rubrics": [
    {
      "name": "IDENTIFY_ISSUE_TYPE",
      "weight": 5,
      "messages": [{"type": "text", "content": "...criterion..."}]
    }
  ],
  "rubric_text": "+5|IDENTIFY_ISSUE_TYPE|...",
  "task_files": {
    "src/code.py": "...code content...",
    "data/train.csv": "...data..."
  },
  "task_dir": "/app",
  "use_docker": true,
  "max_turns": 50
}
```

---

## MCP Server

For MCP-compatible clients:

```bash
cd mle-reasoning-environment/tools
python mcp_server.py
```

This exposes all 5 tools via the Model Context Protocol on stdio.

--- 

## Expected Output

A successful task run produces:

```json
{
  "task_id": "error-analysis-1-dev",
  "model": "openai/gpt-4o-mini",
  "answer": "FINAL ANSWER: ...",
  "metadata": {
    "turns": 14,
    "tool_calls": 16,
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

---

## Error Types in Tasks

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

---

## Troubleshooting

### Docker not running
```bash
open -a Docker  # macOS
docker ps       # verify running
```

### API key issues
```bash
echo $OPENAI_API_KEY | head -c 10  # verify set
export OPENAI_API_KEY="sk-..."     # set if missing
```

### Import errors when running tests
```bash
# Make sure you're in the right directory
cd mle-reasoning-environment/tests
python test_harness.py

# Or set PYTHONPATH manually
export PYTHONPATH=/path/to/mle-reasoning-environment/tools
```

---

## Test Validation Checklist

- [ ] `test_harness.py` - 40/40 pass
- [ ] `test_mcp_server.py` - 16/16 pass
- [ ] `test_comprehensive.py` - 61/61 pass
- [ ] `test_evaluator.py` - 34/34 pass (with API key)
- [ ] Docker builds successfully
- [ ] Real task completes with evaluation
