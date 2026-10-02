"""FastAPI entrypoint for the Cross-Border Data Router agent.

Serves two UIs and an API:
  /        → ADK built-in playground (tool calls expanded natively)
  /chat    → Custom chat playground (styled cards)
  /chat/completions → OpenAI-compatible API
  /health  → Health check
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from os import getenv
from pathlib import Path

from app.agent import create_agent
from app.app_utils.telemetry import enable_tracing
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

APP_NAME = "cross-border-data-router"
USER_ID = "api_user"

runner = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str | None = None
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global runner
    enable_tracing()
    agent = create_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent, session_service=session_service, app_name=APP_NAME
    )
    logger.info("Agent runner initialised")
    yield
    runner = None


# ── Main app: ADK playground at / ────────────────────────────────────
# get_fast_api_app creates a full ADK server with the built-in web UI.
# It reads the agent from the current directory (app/__init__.py).
_AGENTS_DIR = str(Path(__file__).resolve().parent)

app = get_fast_api_app(
    agents_dir=_AGENTS_DIR,
    web=True,
    host="0.0.0.0",
    port=int(getenv("PORT", "8080")),
    lifespan=lifespan,
    allow_origins=["*"],
)


# ── OpenAI-compatible API endpoints ──────────────────────────────────

def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _extract_user_message(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(status_code=400, detail="No user message found")


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if runner is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    user_content = _extract_user_message(request.messages)
    model_id = request.model or getenv("MODEL_NAME", "cross-border-data-router")

    if request.stream:
        return await _handle_stream(user_content, model_id)
    return await _handle_chat(user_content, model_id)


async def _handle_chat(user_content: str, model_id: str) -> dict:
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )
    msg = types.Content(
        role="user", parts=[types.Part.from_text(text=user_content)]
    )

    final_text = ""
    context_messages = []

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=msg
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                context_messages.append({
                    "role": "assistant",
                    "content": f"Calling tool: {part.function_call.name}",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(
                                dict(part.function_call.args)
                                if part.function_call.args else {}
                            ),
                        },
                    }],
                })
            elif hasattr(part, "function_response") and part.function_response:
                context_messages.append({
                    "role": "tool",
                    "name": part.function_response.name,
                    "content": json.dumps(
                        dict(part.function_response.response)
                        if part.function_response.response else {}
                    ),
                })
            elif part.text:
                role = event.content.role or "model"
                context_messages.append({
                    "role": "assistant" if role == "model" else role,
                    "content": part.text,
                })
                if role == "model":
                    final_text = part.text

    return {
        "id": _make_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_text},
            "finish_reason": "stop",
        }],
        "context": context_messages,
        "usage": None,
    }


async def _handle_stream(user_content: str, model_id: str) -> StreamingResponse:
    completion_id = _make_id()
    created = int(time.time())

    async def event_generator() -> AsyncIterator[str]:
        try:
            session = await runner.session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID
            )
            msg = types.Content(
                role="user", parts=[types.Part.from_text(text=user_content)]
            )
            async for event in runner.run_async(
                user_id=USER_ID, session_id=session.id, new_message=msg
            ):
                if not event.content or not event.content.parts:
                    continue
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {
                                "role": "assistant",
                                "tool_calls": [{"index": 0, "type": "function", "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(
                                        dict(part.function_call.args)
                                        if part.function_call.args else {}
                                    ),
                                }}],
                            }, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    elif hasattr(part, "function_response") and part.function_response:
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {
                                "role": "tool",
                                "content": json.dumps(
                                    dict(part.function_response.response)
                                    if part.function_response.response else {}
                                ),
                                "name": part.function_response.name,
                            }, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    elif part.text:
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"content": part.text}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': {'message': 'Internal server error'}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Custom chat playground at /chat ──────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent
_PLAYGROUND_HTML = _BASE_DIR / "playground" / "templates" / "index.html"


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def custom_playground():
    """Serve the custom chat playground UI."""
    return FileResponse(
        _PLAYGROUND_HTML,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(getenv("PORT", "8080")))
