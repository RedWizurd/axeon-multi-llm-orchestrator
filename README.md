# Axeon-Full

A multi-LLM orchestrator using Ollama + FastAPI, now with configurable parallel swarm execution, bounded tool-calling loops, trace logs, and optional web-AI fallback routing.

## Setup
1. Install dependencies:
   - `pip install fastapi uvicorn requests pydantic python-dotenv`
   - Optional for web fallback scraping: `pip install selenium`
2. Run Ollama and pull models:
   - `ollama serve`
   - `ollama pull qwen2.5:7b-instruct-q4_K_M`
   - `ollama pull deepseek-coder:6.7b-instruct-q5_K_M`
3. Configure `config.json`.
4. Run server:
   - `python axeon_server.py`

## Security
API keys are loaded from `.env` (never commit this file).
Copy `.env.example` to `.env` and fill in your keys.

Example:
```bash
GEMINI_API_KEY=AIza...
OPENWEATHER_API_KEY=abc123...
AXEON_API_KEY=mysecret
```

## New Config Options
- `swarm_mode`: `"sequential"` (safe default) or `"parallel"`.
- `max_iterations`: iterative self-improvement loop cap for self-improvement tasks.
- `include_swarm_trace`: include `swarm_trace` in non-stream responses by default.
- `history.max_chars` / `history.keep_recent`: context compression controls.
- `tools.enabled`, `tools.max_tool_calls_per_agent`, `tools.allowed_roots`: bounded ReAct tool loop safety controls.
- `web_fallback.enabled`, `web_fallback.preferred_for`: route matching tasks to web AI first, fallback local on failure.
- `web_fallback.api_url` / `web_fallback.selenium.*`: API-first or Selenium scraping fallback options.
- `rate_limit.enabled`, `rate_limit.requests_per_min`: per-IP request limits.
- `AXEON_API_KEY` (in `.env`): optional bearer token auth (`Authorization: Bearer <key>`).

## Endpoints
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (`stream=true` supported)
- `GET /logs` (recent in-memory swarm traces)

## How To Test Upgrades
1. Tool loop test (calculator/search trigger):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "axeon",
    "include_swarm_trace": true,
    "messages": [
      {"role": "user", "content": "Use tools if needed: calculate (42*19)+7 and briefly explain the result."}
    ]
  }'
```

2. Parallel swarm test (`config.json` set `"swarm_mode": "parallel"`):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "axeon",
    "temperature": 0.4,
    "include_swarm_trace": true,
    "messages": [
      {"role": "user", "content": "Design and implement a Python rate-limited API client with tests and edge-case review."}
    ]
  }'
```

3. Logs endpoint:

```bash
curl -s "http://127.0.0.1:8000/logs?limit=20"
```

## Notes
- Streaming remains OpenAI-style SSE (`data: {choices:[{delta:{content:...}}]}`).
- If web fallback fails (missing login/session/driver), Axeon automatically falls back to local Ollama path.
- Self-improvement tasks can produce patch/diff output, but no auto-apply is performed.
