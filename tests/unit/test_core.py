from __future__ import annotations

import pytest

from praxium.core import (
    Conversation,
    MergePolicy,
    MergeStrategy,
    Message,
    Role,
    State,
    StateConflictError,
    StatePatch,
    ToolCallPart,
    ToolResultPart,
)


def test_identifiers_and_messages_round_trip_as_json() -> None:
    call = ToolCallPart(tool_name="weather", arguments={"city": "Berlin"})
    conversation = Conversation(
        messages=[
            Message.user("What is the weather?"),
            Message(role=Role.ASSISTANT, parts=[call]),
            Message(
                role=Role.TOOL,
                parts=[
                    ToolResultPart(
                        call_id=call.call_id,
                        tool_name="weather",
                        output={"temperature": 22},
                    )
                ],
            ),
        ]
    )

    restored = Conversation.model_validate_json(conversation.model_dump_json())
    assert restored == conversation
    assert restored.messages[0].text_content == "What is the weather?"


def test_conversation_rejects_orphan_tool_result() -> None:
    with pytest.raises(ValueError, match="unknown call"):
        Conversation(
            messages=[
                Message(
                    role=Role.TOOL,
                    parts=[
                        ToolResultPart(
                            call_id=ToolCallPart(tool_name="x").call_id,
                            tool_name="x",
                        )
                    ],
                )
            ]
        )


def test_state_is_immutable_by_convention_and_tracks_changes() -> None:
    initial = State(data={"count": 1})
    changed = initial.apply(StatePatch(values={"count": 2, "name": "Ada"}))

    assert initial.data == {"count": 1}
    assert changed.version == 1
    assert [change.path for change in changed.history] == ["count", "name"]


def test_parallel_state_requires_explicit_conflict_policy() -> None:
    state = State()
    with pytest.raises(StateConflictError):
        state.merge([{"items": [1]}, {"items": [2]}])

    merged = state.merge(
        [{"items": [1]}, {"items": [2]}],
        {"items": MergePolicy(strategy=MergeStrategy.APPEND)},
    )
    assert merged.data["items"] == [1, 2]


def test_recursive_merge_rejects_leaf_conflicts() -> None:
    state = State()
    with pytest.raises(StateConflictError, match=r"settings\.theme"):
        state.merge(
            [{"settings": {"theme": "light"}}, {"settings": {"theme": "dark"}}],
            {"settings": MergePolicy(strategy=MergeStrategy.RECURSIVE)},
        )
