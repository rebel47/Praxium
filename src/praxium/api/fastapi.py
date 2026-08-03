"""Optional FastAPI application and OpenAI-compatible transport."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from praxium.core import Conversation, FrameworkModel, Message, Role
from praxium.graph import Graph

from .application import Application, result_text


class RunRequest(FrameworkModel):
    input: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = "default"


class OpenAIMessage(FrameworkModel):
    role: Role
    content: str
    name: str | None = None


class ChatCompletionRequest(FrameworkModel):
    model: str
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    user: str | None = None


class ResponsesRequest(FrameworkModel):
    model: str
    input: str | list[OpenAIMessage]
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingsRequest(FrameworkModel):
    model: str
    input: str | list[str]
    dimensions: int | None = Field(default=None, ge=1)


def create_fastapi_app(application: Application, *, title: str = "Praxium") -> FastAPI:
    """Create an HTTP app without introducing FastAPI into core imports."""

    app = FastAPI(title=title, version="0.1.0")

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    async def ready() -> dict[str, Any]:
        return {"status": "ready", "components": len(application.describe())}

    @app.get("/components", tags=["framework"])
    async def components() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in application.describe()]

    @app.post("/runs/{name}", tags=["framework"])
    async def run_component(name: str, request: RunRequest) -> dict[str, Any]:
        try:
            value = await application.run(
                name,
                request.input,
                metadata=request.metadata,
                tenant_id=request.tenant_id,
            )
            if isinstance(value, FrameworkModel):
                return value.model_dump(mode="json")
            return {"result": value}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runs/{name}/stream", tags=["framework"])
    async def stream_component(name: str, request: RunRequest) -> StreamingResponse:
        try:
            component = application.get(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not isinstance(component, Graph):
            raise HTTPException(
                status_code=400, detail="event streaming currently requires a graph"
            )

        async def events() -> AsyncIterator[str]:
            state = request.input if isinstance(request.input, dict) else {"input": request.input}
            async for event in application.runtime.stream(
                component,
                state,
                metadata=request.metadata,
                tenant_id=request.tenant_id,
            ):
                yield _sse(event.model_dump(mode="json"), event=event.kind.value)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/models", tags=["openai-compatible"])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": item.name, "object": "model", "created": 0, "owned_by": "praxium"}
                for item in application.describe()
            ],
        }

    @app.post("/v1/chat/completions", tags=["openai-compatible"])
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        if request.tools:
            raise HTTPException(
                status_code=400,
                detail=(
                    "request-defined tools are not supported; register tools on the target agent"
                ),
            )
        conversation = Conversation(
            messages=[
                Message.text(message.role, message.content, name=message.name)
                for message in request.messages
            ]
        )
        if request.stream:
            return StreamingResponse(
                _stream_chat(application, request.model, conversation),
                media_type="text/event-stream",
            )
        value = await _run_or_http_error(application, request.model, conversation)
        text = result_text(value)
        completion_id = f"chatcmpl_{uuid4().hex}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage(value),
        }

    @app.post("/v1/responses", tags=["openai-compatible"])
    async def responses(request: ResponsesRequest) -> Any:
        normalized: str | Conversation
        if isinstance(request.input, str):
            normalized = request.input
        else:
            normalized = Conversation(
                messages=[
                    Message.text(item.role, item.content, name=item.name) for item in request.input
                ]
            )
        if request.stream:
            return StreamingResponse(
                _stream_response(application, request.model, normalized, request.metadata),
                media_type="text/event-stream",
            )
        value = await _run_or_http_error(application, request.model, normalized, request.metadata)
        text = result_text(value)
        return {
            "id": f"resp_{uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": request.model,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": _usage(value),
        }

    @app.post("/v1/embeddings", tags=["openai-compatible"])
    async def embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
        inputs = [request.input] if isinstance(request.input, str) else request.input
        try:
            value = await application.embed(request.model, inputs)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        vectors = value.embeddings
        if request.dimensions is not None:
            vectors = [vector[: request.dimensions] for vector in vectors]
        return {
            "object": "list",
            "model": request.model,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": value.usage.input_tokens,
                "total_tokens": value.usage.total_tokens,
            },
        }

    return app


async def _run_or_http_error(
    application: Application,
    model: str,
    input: Any,
    metadata: dict[str, Any] | None = None,
) -> Any:
    try:
        return await application.run(model, input, metadata=metadata)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _stream_chat(
    application: Application,
    model: str,
    conversation: Conversation,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl_{uuid4().hex}"
    created = int(time.time())
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    value = await application.run(model, conversation)
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": result_text(value)}, "finish_reason": None}
            ],
        }
    )
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


async def _stream_response(
    application: Application,
    model: str,
    input: Any,
    metadata: dict[str, Any],
) -> AsyncIterator[str]:
    response_id = f"resp_{uuid4().hex}"
    yield _sse(
        {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}
    )
    value = await application.run(model, input, metadata=metadata)
    yield _sse(
        {
            "type": "response.output_text.delta",
            "response_id": response_id,
            "output_index": 0,
            "content_index": 0,
            "delta": result_text(value),
        }
    )
    yield _sse(
        {"type": "response.completed", "response": {"id": response_id, "status": "completed"}}
    )
    yield "data: [DONE]\n\n"


def _usage(value: Any) -> dict[str, int]:
    usage = getattr(value, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _sse(payload: Any, *, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, separators=(',', ':'))}\n\n"
