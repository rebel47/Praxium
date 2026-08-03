"""Composable asynchronous middleware for framework boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from praxium.core import ExecutionId, FrameworkModel


class Boundary(StrEnum):
    RUN = "run"
    NODE = "node"
    MODEL = "model"
    TOOL = "tool"
    MEMORY = "memory"


class MiddlewareContext(FrameworkModel):
    boundary: Boundary
    operation: str
    payload: Any = None
    execution_id: ExecutionId | None = None
    tenant_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


NextHandler = Callable[[MiddlewareContext], Awaitable[Any]]


class Middleware(Protocol):
    name: str

    async def __call__(self, context: MiddlewareContext, call_next: NextHandler) -> Any: ...


MiddlewareFunction = Callable[[MiddlewareContext, NextHandler], Awaitable[Any]]


class FunctionMiddleware(FrameworkModel):
    name: str
    handler: MiddlewareFunction = Field(exclude=True, repr=False)

    async def __call__(self, context: MiddlewareContext, call_next: NextHandler) -> Any:
        return await self.handler(context, call_next)


class MiddlewarePipeline:
    """Executes middleware in registration order and unwinds in reverse."""

    def __init__(self, middleware: Sequence[Middleware] = ()) -> None:
        names = [item.name for item in middleware]
        if len(names) != len(set(names)):
            raise ValueError("middleware names must be unique within a pipeline")
        self.middleware = tuple(middleware)

    async def execute(self, context: MiddlewareContext, terminal: NextHandler) -> Any:
        call_next = terminal
        for item in reversed(self.middleware):
            downstream = call_next

            async def invoke(
                current: MiddlewareContext,
                *,
                middleware_item: Middleware = item,
                next_handler: NextHandler = downstream,
            ) -> Any:
                return await middleware_item(current, next_handler)

            call_next = invoke
        return await call_next(context)


class TimeoutMiddleware(FrameworkModel):
    name: str = "timeout"
    timeout_seconds: float = Field(gt=0)

    async def __call__(self, context: MiddlewareContext, call_next: NextHandler) -> Any:
        async with asyncio.timeout(self.timeout_seconds):
            return await call_next(context)


class RedactionMiddleware(FrameworkModel):
    """Redact sensitive mapping keys before and after a boundary call."""

    name: str = "redaction"
    keys: set[str] = Field(
        default_factory=lambda: {
            "api_key",
            "authorization",
            "cookie",
            "password",
            "secret",
            "token",
        }
    )
    replacement: str = "[REDACTED]"

    async def __call__(self, context: MiddlewareContext, call_next: NextHandler) -> Any:
        safe_context = context.model_copy(
            update={"payload": _redact(context.payload, self.keys, self.replacement)}
        )
        result = await call_next(safe_context)
        return _redact(result, self.keys, self.replacement)


def _redact(value: Any, keys: set[str], replacement: str) -> Any:
    if isinstance(value, Mapping):
        return {
            key: replacement if str(key).lower() in keys else _redact(item, keys, replacement)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, keys, replacement) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, keys, replacement) for item in value)
    return value
