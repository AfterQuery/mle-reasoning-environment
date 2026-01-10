# MLE Reasoning Environment

ML Error Analysis benchmark.

## Quick Start

```bash
# Build Docker image
docker build -t mle-reasoning-environment .

# Run single task
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY mle-world \
  python run_agent.py --task /workspace/tasks/task_error-analysis-1.json

# Run all tasks
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY mle-world \
  python run_agent.py --tasks-dir /workspace/tasks --model openai/gpt-4o-mini
```

## Task Format

Tasks are JSON files with the following format:
- `prompt`: Task description
- `rubrics`: Evaluation criteria
- `task_files`: Files to set up in /app
- `use_docker`: true
- `mcp_ports`: [8000]
