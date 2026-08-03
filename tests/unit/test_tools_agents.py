from __future__ import annotations

from praxium import (
    Agent,
    AgentRunner,
    DeterministicModelProvider,
    ExecutionId,
    Message,
    Model,
    ModelProviderRegistry,
    Response,
    Role,
    Tool,
    ToolCallPart,
    ToolContext,
)


async def test_tool_validates_permissions_and_executes_typed_callable() -> None:
    def add(left: int, right: int = 1) -> int:
        """Add two numbers."""

        return left + right

    tool = Tool.from_callable(add, required_permissions={"math"})
    denied = await tool.execute(
        {"left": 2},
        ToolContext(execution_id=ExecutionId.new()),
    )
    assert denied.status == "denied"

    result = await tool.execute(
        {"left": 2, "right": 3},
        ToolContext(execution_id=ExecutionId.new(), granted_permissions={"math"}),
    )
    assert result.status == "success"
    assert result.output == 5
    assert tool.input_schema["required"] == ["left"]


async def test_agent_performs_correlated_tool_loop() -> None:
    def add(left: int, right: int) -> int:
        """Add numbers."""

        return left + right

    tool = Tool.from_callable(add)
    call = ToolCallPart(tool_name="add", arguments={"left": 2, "right": 3})
    provider = DeterministicModelProvider(
        responses=[
            Response(message=Message(role=Role.ASSISTANT, parts=[call])),
            Response(message=Message.assistant("The answer is 5.")),
        ]
    )
    runner = AgentRunner(ModelProviderRegistry([provider]))
    agent = Agent(
        name="calculator",
        instructions="Use tools for arithmetic.",
        model=Model(name="fake", provider="deterministic"),
        tools=[tool],
    )

    result = await runner.run(agent, "What is 2 + 3?")

    assert result.response.text_content == "The answer is 5."
    assert result.steps == 2
    assert result.tool_results[0].call_id == call.call_id
    assert any(message.role == Role.TOOL for message in result.conversation.messages)
