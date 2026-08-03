"""Provider-neutral message, prompt, response, and usage models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from .base import FrameworkModel, utc_now
from .enums import FinishReason, Role
from .ids import ConversationId, MessageId, ToolCallId


class TextPart(FrameworkModel):
    model_config = ConfigDict(str_strip_whitespace=False)
    type: Literal["text"] = "text"
    text: str


class JsonPart(FrameworkModel):
    type: Literal["json"] = "json"
    data: Any


class ToolCallPart(FrameworkModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: ToolCallId = Field(default_factory=ToolCallId.new)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultPart(FrameworkModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: ToolCallId
    tool_name: str = Field(min_length=1)
    output: Any = None
    is_error: bool = False


class ReferencePart(FrameworkModel):
    type: Literal["reference"] = "reference"
    uri: str = Field(min_length=1)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ContentPart = Annotated[
    TextPart | JsonPart | ToolCallPart | ToolResultPart | ReferencePart,
    Field(discriminator="type"),
]


class Message(FrameworkModel):
    id: MessageId = Field(default_factory=MessageId.new)
    role: Role
    parts: list[ContentPart] = Field(min_length=1)
    name: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text(cls, role: Role, text: str, **kwargs: Any) -> Self:
        return cls(role=role, parts=[TextPart(text=text)], **kwargs)

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> Self:
        return cls.text(Role.USER, text, **kwargs)

    @classmethod
    def assistant(cls, text: str, **kwargs: Any) -> Self:
        return cls.text(Role.ASSISTANT, text, **kwargs)

    @property
    def text_content(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))


class Conversation(FrameworkModel):
    id: ConversationId = Field(default_factory=ConversationId.new)
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_tool_relationships(self) -> Self:
        known_calls: dict[ToolCallId, str] = {}
        for message in self.messages:
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    known_calls[part.call_id] = part.tool_name
                elif isinstance(part, ToolResultPart):
                    expected_name = known_calls.get(part.call_id)
                    if expected_name is None:
                        raise ValueError(f"tool result references unknown call {part.call_id}")
                    if expected_name != part.tool_name:
                        raise ValueError("tool result name does not match originating call")
        return self

    def append(self, message: Message) -> Self:
        return self.model_copy(update={"messages": [*self.messages, message]})


class Prompt(FrameworkModel):
    name: str | None = None
    system: str | None = None
    messages: list[Message] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(FrameworkModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        costs = [value for value in (self.cost_usd, other.cost_usd) if value is not None]
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=sum(costs) if costs else None,
        )


class Response(FrameworkModel):
    message: Message
    finish_reason: FinishReason = FinishReason.STOP
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseDelta(FrameworkModel):
    model_config = ConfigDict(str_strip_whitespace=False)
    index: int = Field(ge=0)
    text: str = ""
    tool_call: ToolCallPart | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
