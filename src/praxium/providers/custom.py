"""Callable adapter for provider APIs with non-standard protocols."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

from praxium.core import Response, ResponseDelta, UnsupportedFeatureError
from praxium.models import EmbeddingRequest, EmbeddingResponse, ModelRequest


class CustomModelProvider:
    """Wrap application-defined callables in the ``ModelProvider`` protocol.

    Use this for a proprietary transport that cannot be configured as an
    OpenAI-compatible endpoint. Synchronous handlers run in a worker thread.
    """

    def __init__(
        self,
        *,
        name: str,
        complete: Callable[[ModelRequest], Any],
        stream: Callable[[ModelRequest], Any] | None = None,
        embed: Callable[[EmbeddingRequest], Any] | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("provider name cannot be empty")
        self._name = name
        self._complete_handler = complete
        self._stream_handler = stream
        self._embed_handler = embed

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ModelRequest) -> Response:
        result = await _invoke(self._complete_handler, request)
        if not isinstance(result, Response):
            raise TypeError("custom complete handler must return Response")
        return result

    async def stream(self, request: ModelRequest) -> AsyncIterator[ResponseDelta]:
        if self._stream_handler is None:
            response = await self.complete(request)
            if response.message.text_content:
                yield ResponseDelta(index=0, text=response.message.text_content)
            yield ResponseDelta(
                index=1,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )
            return
        result = await _invoke(self._stream_handler, request)
        if hasattr(result, "__aiter__"):
            async for delta in result:
                if not isinstance(delta, ResponseDelta):
                    raise TypeError("custom stream handler must yield ResponseDelta")
                yield delta
            return
        if not isinstance(result, Iterable):
            raise TypeError("custom stream handler must return an async or sync iterable")
        for delta in result:
            if not isinstance(delta, ResponseDelta):
                raise TypeError("custom stream handler must yield ResponseDelta")
            yield delta

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if self._embed_handler is None:
            raise UnsupportedFeatureError(f"custom provider {self.name!r} has no embedding handler")
        result = await _invoke(self._embed_handler, request)
        if not isinstance(result, EmbeddingResponse):
            raise TypeError("custom embed handler must return EmbeddingResponse")
        return result


async def _invoke(handler: Callable[[Any], Any], value: Any) -> Any:
    result = await asyncio.to_thread(handler, value)
    return await result if inspect.isawaitable(result) else result
