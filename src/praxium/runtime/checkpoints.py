"""Checkpoint models and atomic in-memory persistence."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Protocol

from pydantic import Field

from praxium.core import (
    CheckpointConflictError,
    CheckpointId,
    ExecutionId,
    FrameworkModel,
    GraphId,
    State,
    utc_now,
)


class Checkpoint(FrameworkModel):
    id: CheckpointId = Field(default_factory=CheckpointId.new)
    execution_id: ExecutionId
    graph_id: GraphId
    graph_version: int = Field(ge=1)
    graph_fingerprint: str
    state: State
    next_node: str | None = None
    visit_counts: dict[str, int] = Field(default_factory=dict)
    completed_nodes: list[str] = Field(default_factory=list)
    output: Any = None
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class CheckpointStore(Protocol):
    async def save(
        self,
        checkpoint: Checkpoint,
        *,
        expected_revision: int | None = None,
    ) -> Checkpoint: ...

    async def load(self, checkpoint_id: CheckpointId) -> Checkpoint: ...

    async def list_for_execution(self, execution_id: ExecutionId) -> list[Checkpoint]: ...

    async def delete(self, checkpoint_id: CheckpointId) -> None: ...


class InMemoryCheckpointStore:
    """Concurrency-safe reference checkpoint store."""

    def __init__(self) -> None:
        self._items: dict[CheckpointId, Checkpoint] = {}
        self._lock = asyncio.Lock()

    async def save(
        self,
        checkpoint: Checkpoint,
        *,
        expected_revision: int | None = None,
    ) -> Checkpoint:
        async with self._lock:
            current = self._items.get(checkpoint.id)
            if current is None:
                if expected_revision not in (None, 0):
                    raise CheckpointConflictError(
                        "checkpoint does not exist at expected revision",
                        context={"checkpoint_id": str(checkpoint.id)},
                    )
                stored = checkpoint.model_copy(deep=True)
            else:
                if expected_revision is None or current.revision != expected_revision:
                    raise CheckpointConflictError(
                        "checkpoint revision conflict",
                        context={
                            "checkpoint_id": str(checkpoint.id),
                            "expected_revision": expected_revision,
                            "actual_revision": current.revision,
                        },
                    )
                stored = checkpoint.model_copy(update={"revision": current.revision + 1}, deep=True)
            self._items[stored.id] = stored
            return stored.model_copy(deep=True)

    async def load(self, checkpoint_id: CheckpointId) -> Checkpoint:
        async with self._lock:
            try:
                return self._items[checkpoint_id].model_copy(deep=True)
            except KeyError as exc:
                raise KeyError(f"checkpoint {checkpoint_id} was not found") from exc

    async def list_for_execution(self, execution_id: ExecutionId) -> list[Checkpoint]:
        async with self._lock:
            items = [
                checkpoint.model_copy(deep=True)
                for checkpoint in self._items.values()
                if checkpoint.execution_id == execution_id
            ]
        return sorted(items, key=lambda checkpoint: checkpoint.created_at)

    async def delete(self, checkpoint_id: CheckpointId) -> None:
        async with self._lock:
            self._items.pop(checkpoint_id, None)
