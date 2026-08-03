"""Opaque, JSON-friendly framework identifiers."""

from __future__ import annotations

from typing import ClassVar, Self
from uuid import uuid4

from pydantic import ConfigDict, RootModel, model_validator


class Identifier(RootModel[str]):
    """An immutable prefixed identifier serialized as a JSON string."""

    model_config = ConfigDict(frozen=True)
    prefix: ClassVar[str] = "id"

    @model_validator(mode="after")
    def validate_identifier(self) -> Self:
        expected = f"{self.prefix}_"
        if not self.root.startswith(expected) or len(self.root) <= len(expected):
            raise ValueError(f"identifier must start with {expected!r}")
        return self

    @classmethod
    def new(cls) -> Self:
        """Create a collision-resistant identifier."""

        return cls(root=f"{cls.prefix}_{uuid4().hex}")

    def __str__(self) -> str:
        return self.root

    def __hash__(self) -> int:
        return hash((type(self), self.root))


class AgentId(Identifier):
    prefix = "agt"


class CheckpointId(Identifier):
    prefix = "chk"


class ConversationId(Identifier):
    prefix = "con"


class EventId(Identifier):
    prefix = "evt"


class ExecutionId(Identifier):
    prefix = "run"


class GraphId(Identifier):
    prefix = "grp"


class MessageId(Identifier):
    prefix = "msg"


class MemoryId(Identifier):
    prefix = "mem"


class DocumentId(Identifier):
    prefix = "doc"


class ChunkId(Identifier):
    prefix = "chkpart"


class ToolCallId(Identifier):
    prefix = "call"
