"""Local HTTP server target: praxium serve examples.server:application."""

from __future__ import annotations

from praxium import Application, GraphBuilder, State


async def echo(state: State, _context: object) -> dict[str, object]:
    return {"output": state.data.get("input")}


graph = (
    GraphBuilder("echo")
    .add_node("echo", echo)
    .set_entrypoint("echo")
    .set_finish_point("echo")
    .build()
)

application = Application().register(graph)
