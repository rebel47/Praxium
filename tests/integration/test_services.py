from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from praxium import (
    Document,
    InMemoryMemoryStore,
    InMemoryRagIndex,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    RetrievalQuery,
    TextChunker,
)
from praxium.middleware import (
    Boundary,
    MiddlewareContext,
    MiddlewarePipeline,
    RedactionMiddleware,
    TimeoutMiddleware,
)
from praxium.plugins import (
    Plugin,
    PluginCapability,
    PluginManifest,
    PluginRegistration,
    PluginRegistry,
)


async def test_memory_is_tenant_isolated_versioned_and_branchable() -> None:
    store = InMemoryMemoryStore()
    record = MemoryRecord(
        tenant_id="tenant-a",
        namespace="assistant",
        kind=MemoryKind.LONG_TERM,
        content="Ada likes analytical explanations",
    )
    first = await store.put(record)
    second = await store.put(
        record.model_copy(update={"content": "Ada likes concise explanations"})
    )

    assert first.version == 1
    assert second.version == 2
    assert await store.query(MemoryQuery(tenant_id="tenant-b", namespace="assistant")) == []
    matches = await store.query(
        MemoryQuery(tenant_id="tenant-a", namespace="assistant", text="concise")
    )
    assert matches[0].record.version == 2

    branch = await store.branch("tenant-a", record.id, "experiment")
    assert branch.parent_id == record.id
    assert branch.branch == "experiment"


async def test_memory_expiration_is_excluded_by_default() -> None:
    store = InMemoryMemoryStore()
    created = datetime.now(UTC) - timedelta(seconds=10)
    expired = MemoryRecord(
        tenant_id="tenant",
        namespace="temp",
        kind=MemoryKind.WORKING,
        content="old",
        created_at=created,
        expires_at=created + timedelta(seconds=1),
    )
    await store.put(expired)
    assert await store.query(MemoryQuery(tenant_id="tenant", namespace="temp")) == []


async def test_rag_preserves_offsets_and_replaces_idempotently() -> None:
    content = "Agent frameworks need explicit state. " * 20
    document = Document(
        tenant_id="tenant",
        namespace="docs",
        content=content,
        source_uri="https://example.test/design",
        metadata={"title": "Design"},
    )
    index = InMemoryRagIndex()
    chunker = TextChunker(chunk_size=100, overlap=10)
    first = await index.index(document, chunker=chunker)
    second = await index.index(document, chunker=chunker)

    assert first.chunks_indexed > 1
    assert second.replaced_chunks == first.chunks_indexed
    results = await index.query(
        RetrievalQuery(tenant_id="tenant", namespace="docs", text="explicit state")
    )
    assert results
    top = results[0]
    assert content[top.citation.start : top.citation.end] == top.chunk.text
    assert top.citation.source_uri == document.source_uri
    assert (
        await index.query(
            RetrievalQuery(tenant_id="other", namespace="docs", text="explicit state")
        )
        == []
    )


async def test_middleware_redacts_both_sides_and_timeout_is_enforced() -> None:
    pipeline = MiddlewarePipeline([RedactionMiddleware()])

    async def echo(context: MiddlewareContext) -> dict[str, object]:
        return {"received": context.payload, "token": "outbound"}

    result = await pipeline.execute(
        MiddlewareContext(
            boundary=Boundary.TOOL, operation="echo", payload={"password": "inbound"}
        ),
        echo,
    )
    assert result == {"received": {"password": "[REDACTED]"}, "token": "[REDACTED]"}

    async def slow(_context: MiddlewareContext) -> None:
        await asyncio.sleep(1)

    timeout = MiddlewarePipeline([TimeoutMiddleware(timeout_seconds=0.01)])
    try:
        await timeout.execute(MiddlewareContext(boundary=Boundary.RUN, operation="slow"), slow)
    except TimeoutError:
        pass
    else:
        raise AssertionError("timeout middleware did not time out")


def test_plugin_registry_rejects_component_conflicts() -> None:
    registry = PluginRegistry()
    first = Plugin(
        manifest=PluginManifest(
            name="first-plugin",
            version="0.1.0",
            description="first",
            capabilities={PluginCapability.TOOL},
        ),
        registrations=[
            PluginRegistration(capability=PluginCapability.TOOL, name="search", component=object())
        ],
    )
    second = Plugin(
        manifest=PluginManifest(
            name="second-plugin",
            version="0.1.0",
            description="second",
            capabilities={PluginCapability.TOOL},
        ),
        registrations=[
            PluginRegistration(capability=PluginCapability.TOOL, name="search", component=object())
        ],
    )
    registry.register(first)
    try:
        registry.register(second)
    except ValueError as exc:
        assert "conflict" in str(exc)
    else:
        raise AssertionError("plugin conflict was not rejected")
