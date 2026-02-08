#!/bin/bash

echo "Starting Ollama in background..."
ollama serve &> ollama.log &
sleep 5  # give Ollama time to boot

echo "Starting Axeon server in background..."
cd /Users/eddie/Documents/axeon-multi-llm-orchestrator || exit 1
uvicorn axeon_server:app --host 0.0.0.0 --port 8000 --reload &> axeon.log &

sleep 5

echo "Starting Open WebUI in Docker..."
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

echo "All started!"
echo "- Ollama: check http://localhost:11434 (should say running)"
echo "- Axeon: check http://localhost:8000/health"
echo "- WebUI: open http://localhost:3000"
echo "- Logs: ollama.log and axeon.log in current folder"