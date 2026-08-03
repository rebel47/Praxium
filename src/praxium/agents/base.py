"""Agent definitions and bounded model/tool execution loop."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from praxium.core import (
    AgentId,
    Conversation,
    ExecutionId,
    ExecutionLimits,
    FinishReason,
    FrameworkModel,
    Message,
    Role,
    ToolResultPart,
    Usage,
)
from praxium.models import Model, ModelProviderRegistry, ModelRequest
from praxium.tools import Tool, ToolContext, ToolResult


class Agent(FrameworkModel):
    """Serializable composition of instructions, a model, tools, and limits."""

    id: AgentId = Field(default_factory=AgentId.new)
    name: str = Field(min_length=1)
    instructions: str
    model: Model
    tools: list[Tool] = Field(default_factory=list)
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    required_permissions: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(FrameworkModel):
    agent_id: AgentId
    execution_id: ExecutionId
    conversation: Conversation
    response: Message
    finish_reason: FinishReason
    usage: Usage = Field(default_factory=Usage)
    steps: int = Field(ge=1)
    tool_results: list[ToolResult] = Field(default_factory=list)


class AgentRunner:
    """Runs a provider-neutral agent until it answers or reaches a bound."""

    def __init__(self, providers: ModelProviderRegistry) -> None:
        self.providers = providers

    async def run(
        self,
        agent: Agent,
        input: str | Message | Conversation,
        *,
        execution_id: ExecutionId | None = None,
        granted_permissions: set[str] | None = None,
    ) -> AgentResult:
        run_id = execution_id or ExecutionId.new()
        conversation = _normalize_input(agent, input)
        provider = self.providers.get(agent.model.provider)
        tool_map = {tool.name: tool for tool in agent.tools}
        usage = Usage()
        results: list[ToolResult] = []
        tool_calls = 0

        for step in range(1, agent.limits.max_steps + 1):
            response = await provider.complete(
                ModelRequest(
                    model=agent.model,
                    messages=conversation.messages,
                    tools=[tool.definition() for tool in agent.tools],
                )
            )
            conversation = conversation.append(response.message)
            usage = usage + response.usage
            calls = [part for part in response.message.parts if part.type == "tool_call"]
            if not calls:
                return AgentResult(
                    agent_id=agent.id,
                    execution_id=run_id,
                    conversation=conversation,
                    response=response.message,
                    finish_reason=response.finish_reason,
                    usage=usage,
                    steps=step,
                    tool_results=results,
                )

            for call in calls:
                tool_calls += 1
                if tool_calls > agent.limits.max_tool_calls:
                    raise RuntimeError("agent exceeded its tool-call limit")
                tool = tool_map.get(call.tool_name)
                if tool is None:
                    result = ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status="error",
                        error_code="unknown_tool",
                        error_message=f"tool {call.tool_name!r} is not available",
                    )
                else:
                    result = await tool.execute(
                        call.arguments,
                        ToolContext(
                            execution_id=run_id,
                            call_id=call.call_id,
                            granted_permissions=granted_permissions or set(),
                        ),
                    )
                results.append(result)
                conversation = conversation.append(
                    Message(
                        role=Role.TOOL,
                        name=result.tool_name,
                        parts=[
                            ToolResultPart(
                                call_id=result.call_id,
                                tool_name=result.tool_name,
                                output=result.output or result.error_message,
                                is_error=result.is_error,
                            )
                        ],
                    )
                )
        raise RuntimeError("agent exceeded its maximum step count")


def _normalize_input(agent: Agent, input: str | Message | Conversation) -> Conversation:
    if isinstance(input, Conversation):
        conversation = input
    elif isinstance(input, Message):
        conversation = Conversation(messages=[input])
    else:
        conversation = Conversation(messages=[Message.user(input)])
    if agent.instructions and not any(
        message.role == Role.SYSTEM for message in conversation.messages
    ):
        conversation = conversation.model_copy(
            update={
                "messages": [Message.text(Role.SYSTEM, agent.instructions), *conversation.messages]
            }
        )
    return conversation
