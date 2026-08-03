"""Tenant-isolated memory records and in-memory hybrid retrieval."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field, model_validator

from praxium.core import FrameworkModel, MemoryId, utc_now


class MemoryKind(StrEnum):
    WORKING = "working"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    ENTITY = "entity"
    TOOL = "tool"
    SUMMARY = "summary"


class RetentionPolicy(FrameworkModel):
    ttl_seconds: int | None = Field(default=None, ge=1)
    max_versions: int = Field(default=10, ge=1)


class MemoryRecord(FrameworkModel):
    id: MemoryId = Field(default_factory=MemoryId.new)
    tenant_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    kind: MemoryKind
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    branch: str = "main"
    parent_id: MemoryId | None = None
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_expiration(self) -> MemoryRecord:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        return self

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= datetime.now(UTC)


class MemoryQuery(FrameworkModel):
    tenant_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    text: str | None = None
    kinds: set[MemoryKind] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)
    branch: str = "main"
    limit: int = Field(default=10, ge=1, le=1000)
    include_expired: bool = False


class MemorySearchResult(FrameworkModel):
    record: MemoryRecord
    score: float = Field(ge=0)


class MemoryStore(Protocol):
    async def put(
        self,
        record: MemoryRecord,
        retention: RetentionPolicy | None = None,
    ) -> MemoryRecord: ...

    async def query(self, query: MemoryQuery) -> list[MemorySearchResult]: ...

    async def delete(self, tenant_id: str, memory_id: MemoryId) -> bool: ...

    async def branch(self, tenant_id: str, memory_id: MemoryId, branch: str) -> MemoryRecord: ...


class InMemoryMemoryStore:
    """Concurrency-safe reference store with lexical ranking and version history."""

    def __init__(self) -> None:
        self._records: dict[MemoryId, list[MemoryRecord]] = {}
        self._lock = asyncio.Lock()

    async def put(
        self,
        record: MemoryRecord,
        retention: RetentionPolicy | None = None,
    ) -> MemoryRecord:
        policy = retention or RetentionPolicy()
        async with self._lock:
            versions = self._records.setdefault(record.id, [])
            if versions and versions[-1].tenant_id != record.tenant_id:
                raise ValueError("memory ID already belongs to a different tenant")
            expires_at = record.expires_at
            if expires_at is None and policy.ttl_seconds is not None:
                expires_at = datetime.now(UTC) + timedelta(seconds=policy.ttl_seconds)
            stored = record.model_copy(
                update={"version": len(versions) + 1, "expires_at": expires_at},
                deep=True,
            )
            versions.append(stored)
            if len(versions) > policy.max_versions:
                del versions[: len(versions) - policy.max_versions]
            return stored.model_copy(deep=True)

    async def query(self, query: MemoryQuery) -> list[MemorySearchResult]:
        async with self._lock:
            current = [versions[-1].model_copy(deep=True) for versions in self._records.values()]
        query_terms = _terms(query.text or "")
        results: list[MemorySearchResult] = []
        for record in current:
            if record.tenant_id != query.tenant_id or record.namespace != query.namespace:
                continue
            if record.branch != query.branch:
                continue
            if record.expired and not query.include_expired:
                continue
            if query.kinds and record.kind not in query.kinds:
                continue
            if any(record.metadata.get(key) != value for key, value in query.metadata.items()):
                continue
            record_terms = _terms(record.content)
            score = _jaccard(query_terms, record_terms) if query_terms else 1.0
            if query_terms and score == 0:
                continue
            results.append(MemorySearchResult(record=record, score=score))
        results.sort(
            key=lambda item: (-item.score, -item.record.created_at.timestamp(), str(item.record.id))
        )
        return results[: query.limit]

    async def delete(self, tenant_id: str, memory_id: MemoryId) -> bool:
        async with self._lock:
            versions = self._records.get(memory_id)
            if not versions or versions[-1].tenant_id != tenant_id:
                return False
            del self._records[memory_id]
            return True

    async def branch(self, tenant_id: str, memory_id: MemoryId, branch: str) -> MemoryRecord:
        async with self._lock:
            versions = self._records.get(memory_id)
            if not versions or versions[-1].tenant_id != tenant_id:
                raise KeyError(f"memory {memory_id} was not found")
            source = versions[-1]
            branched = source.model_copy(
                update={
                    "id": MemoryId.new(),
                    "version": 1,
                    "branch": branch,
                    "parent_id": source.id,
                    "created_at": utc_now(),
                },
                deep=True,
            )
            self._records[branched.id] = [branched]
            return branched.model_copy(deep=True)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+", text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
