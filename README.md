# Axeon-Full

A multi-LLM orchestrator for clinical AI assistance, using Ollama, FastAPI, and ChatDev-style agents.

## Setup
1. Install dependencies: `pip install fastapi uvicorn requests pydantic`
2. Run Ollama: `ollama serve` and pull models: `ollama pull qwen` `ollama pull deepseek-coder`
3. Configure `config.json` (ollama_host, models, etc.)
4. Delete or clear `budget_state.json` if exists.
5. Run the server: `python axeon_server.py` or use `start-all.sh` for full stack.

## Usage
- Connect to http://127.0.0.1:8000/v1 in Open WebUI.
- Chat with Axeon: Simple queries use writer model; complex tasks (e.g., "build a clinic app") engage agent swarm.
- Endpoints: /health, /models, /chat/completions (supports stream=true).

## Troubleshooting
- Check logs for Ollama connection issues.
- If crash: Ensure models are pulled and host is correct.
- For streaming: Use in compatible UIs like Open WebUI.

Enjoy your upgraded Axeon!