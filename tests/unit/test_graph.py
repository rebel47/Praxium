from __future__ import annotations

import pytest

from praxium import GraphBuilder, NodeKind, State
from praxium.core import GraphValidationError


async def no_op(_state: State, _context: object) -> None:
    return None


def test_builder_validates_and_exports_mermaid() -> None:
    graph = (
        GraphBuilder("router")
        .add_node("route", no_op, kind=NodeKind.CONDITION)
        .add_node("left", no_op)
        .add_node("right", no_op)
        .add_conditional_edges("route", {"left": "left", "right": "right"})
        .set_entrypoint("route")
        .set_finish_point("left")
        .set_finish_point("right")
        .build()
    )

    assert graph.entrypoint == "route"
    assert "route -->|left| left" in graph.to_mermaid()
    assert len(graph.fingerprint) == 64


def test_builder_collects_unreachable_and_missing_target_errors() -> None:
    builder = (
        GraphBuilder("invalid")
        .add_node("start", no_op)
        .add_node("orphan", no_op)
        .add_edge("start", "missing")
        .set_entrypoint("start")
        .set_finish_point("orphan")
    )
    with pytest.raises(GraphValidationError) as captured:
        builder.build()

    codes = {item["code"] for item in captured.value.context["issues"]}
    assert {"missing_edge_target", "unreachable_node"} <= codes


def test_unbounded_cycles_are_rejected() -> None:
    builder = (
        GraphBuilder("cycle")
        .add_node("one", no_op)
        .add_node("two", no_op)
        .add_edge("one", "two")
        .add_edge("two", "one")
        .set_entrypoint("one")
        .set_finish_point("two")
    )
    with pytest.raises(GraphValidationError, match="unbounded_cycle"):
        builder.build()


def test_bounded_cycle_is_valid() -> None:
    graph = (
        GraphBuilder("bounded-cycle")
        .add_node("one", no_op, max_visits=2)
        .add_node("two", no_op)
        .add_conditional_edges("one", {"again": "two", "done": "finish"})
        .add_node("finish", no_op)
        .add_edge("two", "one")
        .set_entrypoint("one")
        .set_finish_point("finish")
        .build()
    )
    assert graph.name == "bounded-cycle"
