"""Event sinks and per-execution ordered emitters."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import Field, PrivateAttr

from praxium.core import EventKind, ExecutionEvent, ExecutionId, FrameworkModel, GraphId


class EventSink(Protocol):
    async def handle(self, event: ExecutionEvent) -> None: ...


class InMemoryEventSink(FrameworkModel):
    """Collector useful for tests, replay, and local debugging."""

    events: list[ExecutionEvent] = Field(default_factory=list)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def handle(self, event: ExecutionEvent) -> None:
        async with self._lock:
            self.events.append(event)


class EventBus:
    """Creates run emitters and isolates failures in telemetry sinks."""

    def __init__(self, sinks: Sequence[EventSink] = ()) -> None:
        self.sinks = tuple(sinks)

    def emitter(self, execution_id: ExecutionId, graph_id: GraphId) -> EventEmitter:
        return EventEmitter(execution_id, graph_id, self.sinks)


class EventEmitter:
    def __init__(
        self,
        execution_id: ExecutionId,
        graph_id: GraphId,
        sinks: Sequence[EventSink],
    ) -> None:
        self.execution_id = execution_id
        self.graph_id = graph_id
        self.sinks = tuple(sinks)
        self.events: list[ExecutionEvent] = []
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def emit(
        self,
        kind: EventKind,
        *,
        node_id: str | None = None,
        attempt: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        async with self._lock:
            self._sequence += 1
            event = ExecutionEvent(
                execution_id=self.execution_id,
                graph_id=self.graph_id,
                sequence=self._sequence,
                kind=kind,
                node_id=node_id,
                attempt=attempt,
                payload=payload or {},
            )
            self.events.append(event)
        if self.sinks:
            await asyncio.gather(
                *(self._safe_handle(sink, event) for sink in self.sinks),
            )
        return event

    @staticmethod
    async def _safe_handle(sink: EventSink, event: ExecutionEvent) -> None:
        try:
            await sink.handle(event)
        except Exception:
            # Telemetry is deliberately non-authoritative. Production adapters
            # should report their own health outside the execution path.
            return


class QueueEventSink:
    """Internal sink used by Runtime.stream."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue()

    async def handle(self, event: ExecutionEvent) -> None:
        await self.queue.put(event)
