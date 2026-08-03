from __future__ import annotations

from praxium.core import (
    Conversation,
    EventKind,
    ExecutionEvent,
    ExecutionId,
    GraphId,
    Message,
)
from praxium.storage import InMemoryStorage, TenantContext


async def test_storage_is_copy_safe_and_tenant_isolated() -> None:
    storage = InMemoryStorage()
    tenant = TenantContext(tenant_id="one", project_id="project")
    other = TenantContext(tenant_id="two", project_id="project")
    conversation = Conversation(messages=[Message.user("hello")])

    await storage.save_conversation(tenant, conversation)
    loaded = await storage.load_conversation(tenant, conversation.id)

    assert loaded == conversation
    assert loaded is not conversation
    assert await storage.load_conversation(other, conversation.id) is None


async def test_storage_rejects_non_increasing_event_sequences() -> None:
    storage = InMemoryStorage()
    tenant = TenantContext(tenant_id="one")
    execution_id = ExecutionId.new()
    graph_id = GraphId.new()
    event = ExecutionEvent(
        execution_id=execution_id,
        graph_id=graph_id,
        sequence=1,
        kind=EventKind.RUN_STARTED,
    )
    await storage.append_events(tenant, [event])

    try:
        await storage.append_events(tenant, [event])
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("duplicate event sequence was accepted")
