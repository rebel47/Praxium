from __future__ import annotations

import pytest

from praxium.core import CheckpointConflictError, ExecutionId, GraphId, State
from praxium.runtime import Checkpoint, InMemoryCheckpointStore


async def test_checkpoint_store_copy_list_update_and_delete() -> None:
    store = InMemoryCheckpointStore()
    checkpoint = Checkpoint(
        execution_id=ExecutionId.new(),
        graph_id=GraphId.new(),
        graph_version=1,
        graph_fingerprint="fingerprint",
        state=State(data={"step": 1}),
    )
    saved = await store.save(checkpoint)
    loaded = await store.load(saved.id)
    assert loaded == saved
    assert loaded is not saved
    assert await store.list_for_execution(saved.execution_id) == [saved]

    updated = await store.save(
        saved.model_copy(update={"state": State(data={"step": 2})}),
        expected_revision=1,
    )
    assert updated.revision == 2
    with pytest.raises(CheckpointConflictError):
        await store.save(updated, expected_revision=1)

    await store.delete(saved.id)
    with pytest.raises(KeyError):
        await store.load(saved.id)


async def test_checkpoint_create_rejects_nonzero_expected_revision() -> None:
    store = InMemoryCheckpointStore()
    checkpoint = Checkpoint(
        execution_id=ExecutionId.new(),
        graph_id=GraphId.new(),
        graph_version=1,
        graph_fingerprint="fingerprint",
        state=State(),
    )
    with pytest.raises(CheckpointConflictError):
        await store.save(checkpoint, expected_revision=3)
