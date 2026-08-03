"""Typed framework exception hierarchy."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import FrameworkModel


class ErrorDetail(FrameworkModel):
    """Serializable safe error representation."""

    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class FrameworkError(Exception):
    """Base exception carrying a stable machine-readable error code."""

    code = "framework_error"
    retryable = False

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def to_detail(self) -> ErrorDetail:
        return ErrorDetail(
            code=self.code,
            message=self.message,
            context=self.context,
            retryable=self.retryable,
        )


class ConfigurationError(FrameworkError):
    code = "configuration_error"


class ValidationError(FrameworkError):
    code = "validation_error"


class GraphValidationError(ValidationError):
    code = "graph_validation_error"


class StateConflictError(FrameworkError):
    code = "state_conflict"


class ExecutionError(FrameworkError):
    code = "execution_error"


class ExecutionCancelledError(ExecutionError):
    code = "execution_cancelled"


class ExecutionTimeoutError(ExecutionError):
    code = "execution_timeout"
    retryable = True


class NodeExecutionError(ExecutionError):
    code = "node_execution_error"


class ToolExecutionError(ExecutionError):
    code = "tool_execution_error"


class ModelProviderError(ExecutionError):
    """A model provider request failed or returned an invalid response."""

    code = "model_provider_error"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        safe_context = {"provider": provider, **(context or {})}
        if status_code is not None:
            safe_context["status_code"] = status_code
        super().__init__(message, context=safe_context)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class PermissionDeniedError(FrameworkError):
    code = "permission_denied"


class CheckpointError(FrameworkError):
    code = "checkpoint_error"


class CheckpointConflictError(CheckpointError):
    code = "checkpoint_conflict"


class UnsupportedFeatureError(FrameworkError):
    code = "unsupported_feature"
