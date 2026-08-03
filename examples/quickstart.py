"""A dependency-free conditional graph with streamed lifecycle events."""

from __future__ import annotations

import asyncio

from praxium import GraphBuilder, NodeKind, NodeResult, Runtime, State, StatePatch


async def classify(state: State, _context: object) -> NodeResult:
    temperature = float(state.data["temperature"])
    route = "hot" if temperature >= 25 else "cold"
    return NodeResult(route=route)


async def hot_message(_state: State, _context: object) -> StatePatch:
    return StatePatch(values={"message": "It is warm outside."})


async def cold_message(_state: State, _context: object) -> dict[str, str]:
    return {"message": "Bring a jacket."}


async def main() -> None:
    graph = (
        GraphBuilder("weather-advice")
        .add_node("classify", classify, kind=NodeKind.CONDITION)
        .add_node("hot", hot_message)
        .add_node("cold", cold_message)
        .add_conditional_edges("classify", {"hot": "hot", "cold": "cold"})
        .set_entrypoint("classify")
        .set_finish_point("hot")
        .set_finish_point("cold")
        .build()
    )

    runtime = Runtime()
    result = await runtime.run(graph, {"temperature": 29})
    print(result.state.data["message"])
    print(f"status={result.status}, events={len(result.events)}")
    print(graph.to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
