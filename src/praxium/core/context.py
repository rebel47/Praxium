"""Execution context and cooperative cancellation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from pydantic import Field, PrivateAttr

from .base import FrameworkModel
from .errors import ExecutionCancelledError
from .ids import ExecutionId, GraphId


class CancellationToken(FrameworkModel):
    """A serializable cancellation view backed by a private event."""

    reason: str | None = None
    _event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ExecutionCancelledError(self.reason or "execution cancelled")


class ExecutionContext(FrameworkModel):
    execution_id: ExecutionId
    graph_id: GraphId
    node_id: str | None = None
    attempt: int = Field(default=1, ge=1)
    deadline: datetime | None = None
    tenant_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
    cancellation: CancellationToken = Field(default_factory=CancellationToken, exclude=True)

    @property
    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max((self.deadline - datetime.now(UTC)).total_seconds(), 0.0)

    def for_node(self, node_id: str, attempt: int) -> ExecutionContext:
        return self.model_copy(update={"node_id": node_id, "attempt": attempt})
