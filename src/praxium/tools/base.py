"""Typed tools with explicit permissions and bounded execution."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model, model_validator

from praxium.core import (
    CancellationToken,
    ExecutionId,
    FrameworkModel,
    PermissionDeniedError,
    ToolCallId,
    ToolExecutionError,
)
from praxium.models import ToolDefinition

ToolHandler = Callable[..., Any | Awaitable[Any]]
InputModelT = TypeVar("InputModelT", bound=BaseModel)


class ToolContext(FrameworkModel):
    execution_id: ExecutionId
    call_id: ToolCallId = Field(default_factory=ToolCallId.new)
    tenant_id: str = "default"
    granted_permissions: set[str] = Field(default_factory=set)
    cancellation: CancellationToken = Field(default_factory=CancellationToken, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(FrameworkModel):
    call_id: ToolCallId
    tool_name: str
    status: Literal["success", "error", "denied", "timed_out"]
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    duration_seconds: float = Field(default=0, ge=0)

    @property
    def is_error(self) -> bool:
        return self.status != "success"


class Tool(FrameworkModel):
    """Serializable tool definition plus an excluded executable adapter."""

    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    required_permissions: set[str] = Field(default_factory=set)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_output_bytes: int = Field(default=1_000_000, ge=1)
    sensitive_output: bool = False
    handler: ToolHandler | None = Field(default=None, exclude=True, repr=False)
    input_model: type[BaseModel] | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_schema(self) -> Tool:
        if self.input_schema.get("type") not in (None, "object"):
            raise ValueError("tool input schema must describe an object")
        return self

    @classmethod
    def from_callable(
        cls,
        handler: ToolHandler,
        *,
        name: str | None = None,
        description: str | None = None,
        required_permissions: set[str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> Tool:
        """Create a validated tool definition from a typed Python callable."""

        signature = inspect.signature(handler)
        type_hints = get_type_hints(handler)
        fields: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError("tool callables cannot use *args or **kwargs")
            annotation = type_hints.get(parameter.name, parameter.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            default = parameter.default
            if default is inspect.Parameter.empty:
                default = ...
            fields[parameter.name] = (annotation, default)
        model_name = "".join(piece.title() for piece in re.split(r"[_-]", name or handler.__name__))
        input_model = create_model(f"{model_name}Input", **fields)
        tool_description = (
            description or inspect.getdoc(handler) or f"Call {name or handler.__name__}."
        )
        return cls(
            name=name or handler.__name__,
            description=tool_description,
            input_schema=input_model.model_json_schema(),
            required_permissions=required_permissions or set(),
            timeout_seconds=timeout_seconds,
            handler=handler,
            input_model=input_model,
        )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        started = asyncio.get_running_loop().time()
        missing = self.required_permissions - context.granted_permissions
        if missing:
            permission_error = PermissionDeniedError(
                f"tool {self.name!r} requires permissions: {', '.join(sorted(missing))}",
                context={"tool": self.name, "missing_permissions": sorted(missing)},
            )
            return _error_result(self, context, "denied", permission_error, started)
        if self.handler is None:
            missing_handler_error = ToolExecutionError(
                f"tool {self.name!r} has no executable handler"
            )
            return _error_result(self, context, "error", missing_handler_error, started)
        try:
            context.cancellation.raise_if_cancelled()
            validated = self.input_model.model_validate(arguments) if self.input_model else None
            kwargs = validated.model_dump() if validated is not None else arguments
            async with asyncio.timeout(self.timeout_seconds):
                if inspect.iscoroutinefunction(self.handler):
                    output = await self.handler(**kwargs)
                else:
                    output = await asyncio.to_thread(self.handler, **kwargs)
            serialized = json.dumps(output, default=repr).encode("utf-8")
            if len(serialized) > self.max_output_bytes:
                raise ToolExecutionError(
                    f"tool output exceeded {self.max_output_bytes} bytes",
                    context={"tool": self.name, "output_bytes": len(serialized)},
                )
            return ToolResult(
                call_id=context.call_id,
                tool_name=self.name,
                status="success",
                output="[REDACTED]" if self.sensitive_output else output,
                duration_seconds=asyncio.get_running_loop().time() - started,
            )
        except TimeoutError:
            timeout_error = ToolExecutionError(
                f"tool {self.name!r} timed out after {self.timeout_seconds} seconds"
            )
            return _error_result(self, context, "timed_out", timeout_error, started)
        except Exception as exc:
            execution_error = (
                exc if isinstance(exc, ToolExecutionError) else ToolExecutionError(str(exc))
            )
            return _error_result(self, context, "error", execution_error, started)


def _error_result(
    tool: Tool,
    context: ToolContext,
    status: Literal["error", "denied", "timed_out"],
    error: ToolExecutionError | PermissionDeniedError,
    started: float,
) -> ToolResult:
    return ToolResult(
        call_id=context.call_id,
        tool_name=tool.name,
        status=status,
        error_code=error.code,
        error_message=error.message,
        duration_seconds=asyncio.get_running_loop().time() - started,
    )


class ToolRegistry:
    """Explicit conflict-safe tool registry."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool {name!r} is not registered") from exc

    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[name].definition() for name in sorted(self._tools)]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))
