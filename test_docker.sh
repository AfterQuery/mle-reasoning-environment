#!/bin/bash
# Comprehensive test script for MLE Docker World

set -e

IMAGE_NAME="mle-harness"
TASK_FILE="task_error-analysis-1-dev.json"

echo "=============================================="
echo "MLE Reasoning Environment - Test Suite"
echo "=============================================="

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set"
    echo "Set it with: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Step 1: Run local unit tests
echo ""
echo "[1/4] Running local unit tests..."
cd tools
python3 test_harness.py
cd ..

# Step 2: Build Docker image
echo ""
echo "[2/4] Building Docker image..."
docker build -t $IMAGE_NAME .

# Step 3: Verify container starts
echo ""
echo "[3/4] Verifying container..."
docker run --rm $IMAGE_NAME python --version
docker run --rm $IMAGE_NAME ls /workspace/tasks/

# Step 4: Run a test task (dry run without evaluation to save API costs)
echo ""
echo "[4/4] Running test task..."
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  $IMAGE_NAME \
  python run_agent.py \
  --task /workspace/tasks/$TASK_FILE \
  --model openai/gpt-4o-mini \
  --no-eval

echo ""
echo "=============================================="
echo "All tests completed successfully!"
echo "=============================================="
echo ""
echo "To run with evaluation:"
echo "  docker run --rm -e OPENAI_API_KEY=\$OPENAI_API_KEY $IMAGE_NAME \\"
echo "    python run_agent.py --task /workspace/tasks/$TASK_FILE"
