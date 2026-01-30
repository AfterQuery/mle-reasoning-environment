FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl jq \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tools/ /workspace/tools/
COPY files/ /workspace/files/

# n.b. yudrew - not sure if this is needed, commenting out for now
# ENV PYTHONPATH=/workspace/tools:$PYTHONPATH
WORKDIR /workspace/tools

# Expose port for Cloud Run
EXPOSE 8080
ENV PORT=8080

# Default: CLI mode. For HTTP server: python http_server.py
CMD ["python", "run_agent.py"]
