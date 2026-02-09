#!/usr/bin/env python3
"""OpenAI-compatible API wrapper for the Axeon orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from axeon_orchestrator import get_recent_traces, handle_turn_with_meta, load_config

LOGGER = logging.getLogger("axeon_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_ID = "axeon"
MAX_INPUT_CHARS = 12000
ENABLE_CORS = True
ENABLE_SWAGGER = True
PORT = int(os.getenv("PORT", "8000"))

# Load from .env first, fallback to config.
load_dotenv(BASE_DIR / ".env")

_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_LOCK = Lock()


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    include_swarm_trace: Optional[bool] = None


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip() if content is not None else ""


def extract_history(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
    history = []
    user_message = ""
    for msg in messages[:-1]:
        if msg.role in ["user", "assistant"]:
            text = normalize_content(msg.content)
            if text:
                history.append({"role": msg.role, "content": text})

    if messages and messages[-1].role == "user":
        user_message = normalize_content(messages[-1].content)

    return user_message, history


def truncate_for_log(text: str, max_len: int = 240) -> str:
    cleaned = (text or "").replace("\n", "\\n")
    if len(cleaned) > max_len:
        return f"{cleaned[:max_len-3]}..."
    return cleaned


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_api_key(config: dict[str, Any], authorization: Optional[str]) -> None:
    # Load from .env first, fallback to config.
    expected_key = os.getenv("AXEON_API_KEY") or config.get("api_key")
    auth_enabled = bool(config.get("api_auth", {}).get("enabled", False) or expected_key)
    if not auth_enabled:
        return
    if not expected_key:
        LOGGER.warning("API auth enabled but AXEON_API_KEY is missing; allowing requests (dev fallback).")
        return

    if not authorization or not authorization.startswith("Bearer "):
        LOGGER.warning("API key violation: missing/invalid Authorization header.")
        raise HTTPException(status_code=401, detail="Missing API key.")

    provided = authorization.split(" ", 1)[1].strip()
    if provided != expected_key:
        LOGGER.warning("API key violation: provided bearer token mismatch.")
        raise HTTPException(status_code=401, detail="Invalid API key.")


def enforce_rate_limit(config: dict[str, Any], ip: str) -> None:
    rl_cfg = config.get("rate_limit", {})
    if not rl_cfg.get("enabled", False):
        return

    requests_per_min = int(rl_cfg.get("requests_per_min", 60))
    if requests_per_min <= 0:
        return

    now = time.time()
    cutoff = now - 60

    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS[ip]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= requests_per_min:
            LOGGER.warning(
                "Rate limit violation: ip=%s limit=%s/min current=%s",
                ip,
                requests_per_min,
                len(bucket),
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
        bucket.append(now)


app = FastAPI(
    title="Axeon OpenAI-Compatible API",
    version="1.1.0",
    description="Self-hosted Axeon multi-LLM orchestrator",
    docs_url="/docs" if ENABLE_SWAGGER else None,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    LOGGER.info("%s %s - %s - %.3fs", request.method, request.url.path, response.status_code, duration)
    return response


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)):
    config = load_config(str(CONFIG_PATH))
    enforce_api_key(config, authorization)
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


@app.get("/logs")
async def logs(limit: int = 50, authorization: Optional[str] = Header(default=None)):
    config = load_config(str(CONFIG_PATH))
    enforce_api_key(config, authorization)
    safe_limit = max(1, min(limit, 200))
    return {
        "object": "list",
        "data": get_recent_traces(safe_limit),
    }


async def stream_response(response_text: str):
    """SSE streaming compatible with Open WebUI."""
    chunks = [response_text[i : i + 120] for i in range(0, len(response_text), 120)]
    for chunk in chunks:
        chunk_data = {
            "choices": [
                {
                    "delta": {"content": chunk},
                    "index": 0,
                }
            ]
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    raw_request: Request,
    authorization: Optional[str] = Header(default=None),
    include_swarm_trace: Optional[bool] = None,
):
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")

    config = load_config(str(CONFIG_PATH))
    enforce_api_key(config, authorization)
    enforce_rate_limit(config, get_client_ip(raw_request))

    user_message, history = extract_history(body.messages)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found.")

    if MAX_INPUT_CHARS and len(user_message) > MAX_INPUT_CHARS:
        user_message = user_message[-MAX_INPUT_CHARS:] + "\n[truncated]"

    include_trace_effective = (
        body.include_swarm_trace
        if body.include_swarm_trace is not None
        else include_swarm_trace
        if include_swarm_trace is not None
        else bool(config.get("include_swarm_trace", False))
    )

    LOGGER.info(
        "Processing: %s | history_len: %s | temp: %.2f",
        truncate_for_log(user_message),
        len(history),
        body.temperature,
    )

    try:
        result = await run_in_threadpool(
            handle_turn_with_meta,
            user_message,
            history,
            str(CONFIG_PATH),
            body.temperature,
            include_trace_effective,
        )
        response_text = (result.get("response") or "No response generated.").strip()

        created = int(time.time())
        prompt_tokens = len(user_message.split()) + sum(len(h["content"].split()) for h in history)
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens

        if body.stream:
            return StreamingResponse(
                stream_response(response_text),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        payload: dict[str, Any] = {
            "id": f"chatcmpl-axeon-{uuid.uuid4().hex[:10]}",
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

        if include_trace_effective:
            payload["swarm_trace"] = result.get("swarm_trace", [])
            payload["meta"] = {
                "mode": result.get("mode"),
                "used_swarm": result.get("used_swarm", False),
                "used_web": result.get("used_web", False),
            }

        return payload

    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Chat completion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(exc)}")


if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    uvicorn.run(
        "axeon_server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info",
    )
