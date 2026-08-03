"""Offline tool-using agent example with no provider API key."""

from __future__ import annotations

import asyncio

from praxium import (
    Agent,
    AgentRunner,
    DeterministicModelProvider,
    Message,
    Model,
    ModelProviderRegistry,
    Response,
    Role,
    Tool,
    ToolCallPart,
)


def add(left: int, right: int) -> int:
    """Add two integers."""

    return left + right


async def main() -> None:
    tool = Tool.from_callable(add)
    call = ToolCallPart(
        tool_name="add",
        arguments={"left": 2, "right": 3},
    )
    provider = DeterministicModelProvider(
        responses=[
            Response(message=Message(role=Role.ASSISTANT, parts=[call])),
            Response(message=Message.assistant("The answer is 5.")),
        ]
    )
    runner = AgentRunner(ModelProviderRegistry([provider]))
    agent = Agent(
        name="calculator",
        instructions="Use the calculator tool for arithmetic.",
        model=Model(name="offline", provider="deterministic"),
        tools=[tool],
    )

    result = await runner.run(agent, "What is 2 + 3?")

    print(result.response.text_content)
    print(f"steps={result.steps}, tool_output={result.tool_results[0].output}")


if __name__ == "__main__":
    asyncio.run(main())
