from __future__ import annotations

import asyncio

from praxium import (
    DeterministicModelProvider,
    ExecutionId,
    Message,
    Model,
    ModelRequest,
    Tool,
    ToolContext,
    ToolRegistry,
    text_response,
)
from praxium.models import EmbeddingRequest


async def test_deterministic_provider_streams_characters_and_embeds() -> None:
    provider = DeterministicModelProvider(responses=[text_response("ok")], stream_by_character=True)
    request = ModelRequest(
        model=Model(name="fake", provider="deterministic"),
        messages=[Message.user("hello")],
    )
    deltas = [delta async for delta in provider.stream(request)]
    assert [delta.text for delta in deltas[:-1]] == ["o", "k"]
    assert deltas[-1].finish_reason == "stop"

    embedded = await provider.embed(
        EmbeddingRequest(model=request.model, inputs=["hello"], dimensions=4)
    )
    assert len(embedded.embeddings[0]) == 4


async def test_tool_reports_validation_timeout_size_and_missing_handler() -> None:
    def typed(value: int) -> int:
        return value

    tool = Tool.from_callable(typed)
    invalid = await tool.execute(
        {"value": "not-an-int"}, ToolContext(execution_id=ExecutionId.new())
    )
    assert invalid.status == "error"

    async def slow() -> None:
        await asyncio.sleep(1)

    timed = await Tool.from_callable(slow, timeout_seconds=0.01).execute(
        {}, ToolContext(execution_id=ExecutionId.new())
    )
    assert timed.status == "timed_out"

    large = tool.model_copy(update={"max_output_bytes": 1})
    oversized = await large.execute({"value": 123}, ToolContext(execution_id=ExecutionId.new()))
    assert oversized.status == "error"
    assert "exceeded" in (oversized.error_message or "")

    missing = Tool(name="missing", description="Missing.", input_schema={"type": "object"})
    absent = await missing.execute({}, ToolContext(execution_id=ExecutionId.new()))
    assert absent.error_code == "tool_execution_error"


def test_tool_registry_is_sorted_and_conflict_safe() -> None:
    first = Tool(name="zeta", description="Z.", input_schema={"type": "object"})
    second = Tool(name="alpha", description="A.", input_schema={"type": "object"})
    registry = ToolRegistry([first, second])
    assert registry.names() == ("alpha", "zeta")
    assert [definition.name for definition in registry.definitions()] == ["alpha", "zeta"]
    try:
        registry.register(first)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate tool registration was accepted")
