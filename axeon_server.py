#!/usr/bin/env python3
"""OpenAI-compatible API wrapper for the Axeon orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from axeon_orchestrator import handle_turn, load_config

LOGGER = logging.getLogger("axeon_server")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_ID = "axeon"
MAX_INPUT_CHARS = 12000
ENABLE_CORS = True
ENABLE_SWAGGER = True
PORT = int(os.getenv("PORT", "8000"))

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

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
        if msg.role in ["user", "assistant"] and msg.content:
            history.append({"role": msg.role, "content": normalize_content(msg.content)})
    if messages and messages[-1].role == "user":
        user_message = normalize_content(messages[-1].content)
    return user_message, history

def truncate_for_log(text: str, max_len: int = 240) -> str:
    cleaned = text.replace("\n", "\\n")[:max_len]
    if len(cleaned) > max_len:
        return f"{cleaned[:max_len-3]}..."
    return cleaned

app = FastAPI(
    title="Axeon OpenAI-Compatible API",
    version="1.0.0",
    description="Self-hosted Axeon multi-LLM orchestrator",
    docs_url="/docs" if ENABLE_SWAGGER else None,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    LOGGER.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "redwizurd",
        }]
    }

async def stream_response(response_text: str):
    """SSE streaming compatible with Open WebUI."""
    chunks = [response_text[i:i+120] for i in range(0, len(response_text), 120)]
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "choices": [{
                "delta": {"content": chunk},
                "index": 0
            }]
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")

    user_message, history = extract_history(request.messages)
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found.")
    
    if MAX_INPUT_CHARS and len(user_message) > MAX_INPUT_CHARS:
        user_message = user_message[-MAX_INPUT_CHARS:] + "\n[truncated]"

    LOGGER.info(f"Processing: {truncate_for_log(user_message)} | history_len: {len(history)}")

    try:
        response_text = await run_in_threadpool(handle_turn, user_message, history)
        response_text = (response_text or "No response generated.").strip()

        created = int(time.time())
        prompt_tokens = len(user_message.split()) + sum(len(h['content'].split()) for h in history)
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens

        if request.stream:
            return StreamingResponse(
                stream_response(response_text),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        return {
            "id": f"chatcmpl-axeon-{uuid.uuid4().hex[:10]}",
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    except Exception as exc:
        LOGGER.error(f"Chat completion failed: {exc}", exc_info=True)
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
        log_level="info"
    )