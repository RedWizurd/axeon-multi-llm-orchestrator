#!/usr/bin/env python3
"""OpenAI-compatible API wrapper for the Axeon orchestrator."""
from __future__ import annotations

# Production notes:
# - Set API_KEY to a strong value and expose only via HTTPS (use nginx reverse proxy + certbot).
# - For persistent rate limiting, replace in-memory dict with Redis.
# - Consider adding request timeouts via middleware.
# - Monitor logs for budget_state.json interactions from orchestrator.
#
# Maintenance quick reference:
# - Toggle FULL_HISTORY_MODE if handle_turn needs/doesn't need context
# - Set MAX_INPUT_CHARS lower if Ollama crashes on large inputs
# - Enable ENABLE_RATE_LIMIT / API_KEY for remote exposure
# - Disable ENABLE_SWAGGER in production for security

import asyncio
import inspect
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import uvicorn
import axeon_orchestrator
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("axeon_server")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
FALLBACK_CONFIG_PATH = BASE_DIR / "config.fallback.json"
MODEL_ID = "axeon"
FULL_HISTORY_MODE = True
API_KEY = None
MAX_INPUT_CHARS = 8000
ENABLE_CORS = True
ENABLE_RATE_LIMIT = False
RATE_LIMIT_MAX_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 60
ENABLE_SWAGGER = True
PORT = int(os.getenv("PORT", "8000"))

FALLBACK_CONFIG: dict[str, Any] = {
    "models": {
        "writer": "qwen2.5:7b-instruct-q4_K_M",
        "consult": "deepseek-coder:6.7b-instruct-q5_K_M",
    },
    "budgets": {
        "max_consult_ratio": 0.2,
        "max_consults_per_day": 20,
        "max_tokens_per_day": 20000,
    },
    "adapters": {"chatdev": {"enabled": False}},
    "ollama": {"host": "http://127.0.0.1:11434"},
}

RATE_LIMIT_STATE: dict[str, list[float]] = {}
RATE_LIMIT_LOCK = asyncio.Lock()

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

def estimate_tokens(text: str) -> int:
    if not text:
        return 1
    words = len(text.split())
    chars = len(text)
    estimate = int(round((words * 1.3) + (chars / 4)))
    return max(1, estimate)

def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content)

def extract_latest_user_message(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            content = normalize_content(msg.content).strip()
            if content:
                return content
    raise HTTPException(status_code=400, detail="No valid user message found.")

def build_full_history(messages: list[ChatMessage]) -> str:
    history = "\n".join([f"{msg.role.upper()}: {normalize_content(msg.content)}" for msg in messages]).strip()
    if not history:
        raise HTTPException(status_code=400, detail="No valid conversation content.")
    return history

def truncate_for_log(text: str, max_len: int = 240) -> str:
    cleaned = text.replace("\n", "\\n")
    if len(cleaned) > 500:
        cleaned = f"{cleaned[:200]} [...] {cleaned[-200:]}"
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."

def verify_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    provided = (authorization or "").strip().lower()
    expected = f"bearer {API_KEY.lower()}"
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")

def load_config() -> tuple[dict[str, Any], Path]:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
        LOGGER.info(f'Loaded configuration from {CONFIG_PATH}')
        return config, CONFIG_PATH
    except Exception as exc:
        LOGGER.error(f'Failed to load {CONFIG_PATH} ({exc}). Falling back.')
        fallback = dict(FALLBACK_CONFIG)
        try:
            FALLBACK_CONFIG_PATH.write_text(json.dumps(fallback, indent=2) + "\n", encoding="utf-8")
            LOGGER.info(f'Wrote fallback config to {FALLBACK_CONFIG_PATH}')
            return fallback, FALLBACK_CONFIG_PATH
        except Exception as write_exc:
            LOGGER.error(f'Could not persist fallback: {write_exc}')
            return fallback, CONFIG_PATH

def build_orchestrator_runner(config_path: Path, config: dict[str, Any]) -> Callable[[str, float], str]:
    try:
        if hasattr(axeon_orchestrator, "process_query") and callable(axeon_orchestrator.process_query):
            process_query_fn = axeon_orchestrator.process_query
            signature = inspect.signature(process_query_fn)
            def run_with_process_query(query: str, temperature: float) -> str:
                kwargs = {}
                if "query" in signature.parameters:
                    kwargs["query"] = query
                elif signature.parameters:
                    first = next(iter(signature.parameters))
                    kwargs[first] = query
                if "config" in signature.parameters:
                    kwargs["config"] = config
                if "temperature" in signature.parameters:
                    kwargs["temperature"] = temperature
                result = process_query_fn(**kwargs)
                return str(result)
            return run_with_process_query

        orchestrator = axeon_orchestrator.AxeonOrchestrator(config_path=config_path)
        if not hasattr(orchestrator, "handle_turn") or not callable(orchestrator.handle_turn):
            raise RuntimeError("No compatible entry point found.")
        handle_turn_fn = orchestrator.handle_turn
        handle_turn_signature = inspect.signature(handle_turn_fn)
        handle_turn_accepts_temperature = "temperature" in handle_turn_signature.parameters
        def run_with_handle_turn(query: str, temperature: float) -> str:
            if handle_turn_accepts_temperature:
                result = handle_turn_fn(query, temperature=temperature)
                return str(result)
            result = handle_turn_fn(query)
            return str(result)
        return run_with_handle_turn
    except Exception as e:
        LOGGER.error(f"Orchestrator init failed: {e}. Using dummy runner.")
        def dummy_runner(query: str, temperature: float) -> str:
            return f"Axeon dummy response (orchestrator failed to init: {e})"
        return dummy_runner

# Safe load (wrapped to prevent import crash)
try:
    loaded_config, runtime_config_path = load_config()
    orchestrator_runner = build_orchestrator_runner(runtime_config_path, loaded_config)
except Exception as e:
    LOGGER.error(f"Module init failed: {e}")
    orchestrator_runner = lambda q, t: f"Init error: {e}"

app = FastAPI(
    title="Axeon OpenAI-Compatible API",
    version="1.0.0",
    description="Self-hosted Axeon multi-LLM orchestrator with self-healing and local fallback",
    docs_url="/docs" if ENABLE_SWAGGER else None,
)

@app.on_event("startup")
async def startup_event() -> None:
    LOGGER.info(f'Axeon server started on port {PORT}')

@app.on_event("shutdown")
async def shutdown_event() -> None:
    LOGGER.info('Axeon server shutting down. Cleanup complete.')

@app.middleware("http")
async def optional_rate_limit_middleware(request: Request, call_next: Callable[..., Any]) -> Any:
    if not ENABLE_RATE_LIMIT:
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    async with RATE_LIMIT_LOCK:
        timestamps = RATE_LIMIT_STATE.get(client_ip, [])
        timestamps = [ts for ts in timestamps if ts > window_start]
        if not timestamps and client_ip in RATE_LIMIT_STATE:
            del RATE_LIMIT_STATE[client_ip]
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            RATE_LIMIT_STATE[client_ip] = timestamps
            LOGGER.warning(f'Rate limit exceeded ip={client_ip} path={request.url.path}')
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})
        timestamps.append(now)
        RATE_LIMIT_STATE[client_ip] = timestamps
    return await call_next(request)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}

@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_api_key(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "redwizurd",
            }
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_api_key(authorization)
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not implemented yet.")
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")

    try:
        if FULL_HISTORY_MODE:
            orchestrator_input = build_full_history(request.messages)
        else:
            orchestrator_input = extract_latest_user_message(request.messages)

        if MAX_INPUT_CHARS and len(orchestrator_input) > MAX_INPUT_CHARS:
            orchestrator_input = orchestrator_input[-MAX_INPUT_CHARS:] + "\n[truncated]"

        if REQUEST_TIMEOUT_SECONDS is None:
            response_text = await run_in_threadpool(
                orchestrator_runner,
                orchestrator_input,
                request.temperature,
            )
        elif sys.version_info >= (3, 11):
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                response_text = await run_in_threadpool(
                    orchestrator_runner,
                    orchestrator_input,
                    request.temperature,
                )
        else:
            response_text = await asyncio.wait_for(
                run_in_threadpool(
                    orchestrator_runner,
                    orchestrator_input,
                    request.temperature,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        response_text = response_text.strip()
        created = int(time.time())
        prompt_tokens = estimate_tokens(orchestrator_input)
        completion_tokens = estimate_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens

        result = {
            "id": f"chatcmpl-axeon-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        LOGGER.info(
            f'Chat completion successful model={request.model} '
            f'temperature={request.temperature:.2f} '
            f'query={truncate_for_log(orchestrator_input)} '
            f'response_chars={len(response_text)}'
        )
        return result
    except TimeoutError:
        LOGGER.error(f'Request timed out after {REQUEST_TIMEOUT_SECONDS}s path=/v1/chat/completions')
        raise HTTPException(status_code=504, detail="Request timed out.")
    except HTTPException:
        raise
    except ValueError as exc:
        LOGGER.error(f'Bad chat request: {exc}')
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
    except Exception as exc:
        LOGGER.error(f'Chat completion failed: {exc}', exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Test coverage structure (non-executable reference):
# - def test_health_endpoint_returns_healthy_status(client):
#     response = client.get('/health')
#     assert response.json()['status'] == 'healthy'
# ... (rest of your test comments)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)