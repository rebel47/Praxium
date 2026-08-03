"""Tenant-aware storage contracts and a transactional in-memory reference."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Protocol

from pydantic import Field

from praxium.core import (
    Conversation,
    ConversationId,
    EventId,
    ExecutionEvent,
    ExecutionId,
    ExecutionStatus,
    FrameworkModel,
    utc_now,
)


class TenantContext(FrameworkModel):
    tenant_id: str = Field(min_length=1)
    project_id: str = Field(default="default", min_length=1)


class StoredExecution(FrameworkModel):
    execution_id: ExecutionId
    tenant: TenantContext
    status: ExecutionStatus
    graph_id: str
    state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class AuditRecord(FrameworkModel):
    id: EventId = Field(default_factory=EventId.new)
    tenant: TenantContext
    action: str
    principal_id: str
    resource_type: str
    resource_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class Storage(Protocol):
    async def save_conversation(self, tenant: TenantContext, value: Conversation) -> None: ...

    async def load_conversation(
        self, tenant: TenantContext, conversation_id: ConversationId
    ) -> Conversation | None: ...

    async def save_execution(self, value: StoredExecution) -> None: ...

    async def load_execution(
        self, tenant: TenantContext, execution_id: ExecutionId
    ) -> StoredExecution | None: ...

    async def append_events(self, tenant: TenantContext, events: list[ExecutionEvent]) -> None: ...

    async def append_audit(self, record: AuditRecord) -> None: ...


class InMemoryStorage:
    """Atomic copy-on-read/write storage for development and contract tests."""

    def __init__(self) -> None:
        self._conversations: dict[tuple[str, str, ConversationId], Conversation] = {}
        self._executions: dict[tuple[str, str, ExecutionId], StoredExecution] = {}
        self._events: dict[tuple[str, str, ExecutionId], list[ExecutionEvent]] = {}
        self._audit: dict[tuple[str, str], list[AuditRecord]] = {}
        self._lock = asyncio.Lock()

    async def save_conversation(self, tenant: TenantContext, value: Conversation) -> None:
        async with self._lock:
            self._conversations[_key(tenant, value.id)] = value.model_copy(deep=True)

    async def load_conversation(
        self, tenant: TenantContext, conversation_id: ConversationId
    ) -> Conversation | None:
        async with self._lock:
            value = self._conversations.get(_key(tenant, conversation_id))
            return value.model_copy(deep=True) if value else None

    async def save_execution(self, value: StoredExecution) -> None:
        async with self._lock:
            self._executions[_key(value.tenant, value.execution_id)] = value.model_copy(
                update={"updated_at": utc_now()}, deep=True
            )

    async def load_execution(
        self, tenant: TenantContext, execution_id: ExecutionId
    ) -> StoredExecution | None:
        async with self._lock:
            value = self._executions.get(_key(tenant, execution_id))
            return value.model_copy(deep=True) if value else None

    async def append_events(self, tenant: TenantContext, events: list[ExecutionEvent]) -> None:
        async with self._lock:
            for event in events:
                key = _key(tenant, event.execution_id)
                existing = self._events.setdefault(key, [])
                if existing and event.sequence <= existing[-1].sequence:
                    raise ValueError("event sequences must be strictly increasing")
                existing.append(event.model_copy(deep=True))

    async def list_events(
        self, tenant: TenantContext, execution_id: ExecutionId
    ) -> list[ExecutionEvent]:
        async with self._lock:
            return [
                event.model_copy(deep=True)
                for event in self._events.get(_key(tenant, execution_id), [])
            ]

    async def append_audit(self, record: AuditRecord) -> None:
        async with self._lock:
            self._audit.setdefault((record.tenant.tenant_id, record.tenant.project_id), []).append(
                record.model_copy(deep=True)
            )

    async def list_audit(self, tenant: TenantContext) -> list[AuditRecord]:
        async with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self._audit.get((tenant.tenant_id, tenant.project_id), [])
            ]


def _key(tenant: TenantContext, object_id: Any) -> tuple[str, str, Any]:
    return tenant.tenant_id, tenant.project_id, object_id
