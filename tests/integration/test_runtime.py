from __future__ import annotations

import asyncio

from praxium import (
    EventKind,
    ExecutionId,
    ExecutionLimits,
    ExecutionStatus,
    GraphBuilder,
    NodeKind,
    NodeResult,
    RetryPolicy,
    Runtime,
    RuntimeConfig,
    State,
    StatePatch,
    Suspension,
)


async def test_conditional_execution_produces_ordered_events_and_checkpoints() -> None:
    async def route(state: State, _context: object) -> NodeResult:
        return NodeResult(route="yes" if state.data["enabled"] else "no")

    async def yes(_state: State, _context: object) -> dict[str, str]:
        return {"answer": "enabled"}

    async def no(_state: State, _context: object) -> dict[str, str]:
        return {"answer": "disabled"}

    graph = (
        GraphBuilder("conditional")
        .add_node("route", route, kind=NodeKind.CONDITION)
        .add_node("yes", yes)
        .add_node("no", no)
        .add_conditional_edges("route", {"yes": "yes", "no": "no"})
        .set_entrypoint("route")
        .set_finish_point("yes")
        .set_finish_point("no")
        .build()
    )

    result = await Runtime().run(graph, {"enabled": True})

    assert result.status == ExecutionStatus.COMPLETED
    assert result.state.data["answer"] == "enabled"
    assert [run.node_id for run in result.node_runs] == ["route", "yes"]
    assert result.last_checkpoint_id is not None
    assert [event.sequence for event in result.events] == list(range(1, len(result.events) + 1))
    assert result.events[0].kind == EventKind.RUN_STARTED
    assert result.events[-1].kind == EventKind.RUN_COMPLETED


async def test_node_retries_before_succeeding() -> None:
    attempts = 0

    async def unreliable(_state: State, _context: object) -> dict[str, bool]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary")
        return {"ok": True}

    graph = (
        GraphBuilder("retry")
        .add_node(
            "work",
            unreliable,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay_seconds=0),
        )
        .set_entrypoint("work")
        .set_finish_point("work")
        .build()
    )
    result = await Runtime().run(graph)

    assert result.status == ExecutionStatus.COMPLETED
    assert result.node_runs[0].attempts == 3
    assert sum(event.kind == EventKind.NODE_RETRYING for event in result.events) == 2


async def test_node_timeout_returns_typed_terminal_result() -> None:
    async def slow(_state: State, _context: object) -> None:
        await asyncio.sleep(1)

    graph = (
        GraphBuilder("timeout")
        .add_node("slow", slow, timeout_seconds=0.01)
        .set_entrypoint("slow")
        .set_finish_point("slow")
        .build()
    )
    result = await Runtime().run(graph)

    assert result.status == ExecutionStatus.TIMED_OUT
    assert result.error is not None
    assert result.error.code == "execution_timeout"
    assert result.events[-1].kind == EventKind.RUN_TIMED_OUT


async def test_external_cancellation_stops_active_handler() -> None:
    started = asyncio.Event()

    async def wait_forever(_state: State, _context: object) -> None:
        started.set()
        await asyncio.Event().wait()

    graph = (
        GraphBuilder("cancel")
        .add_node("wait", wait_forever)
        .set_entrypoint("wait")
        .set_finish_point("wait")
        .build()
    )
    runtime = Runtime()
    run_id = ExecutionId.new()
    task = asyncio.create_task(runtime.run(graph, execution_id=run_id))
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await runtime.cancel(run_id, "test cancellation") is True
    result = await asyncio.wait_for(task, timeout=1)

    assert result.status == ExecutionStatus.CANCELLED
    assert result.error is not None
    assert result.error.message == "test cancellation"


async def test_suspension_checkpoint_resumes_after_committed_node() -> None:
    approval_calls = 0

    async def approval(_state: State, _context: object) -> NodeResult:
        nonlocal approval_calls
        approval_calls += 1
        return NodeResult(
            patch=StatePatch(values={"approved": True}),
            suspension=Suspension(reason="confirm deployment", payload={"environment": "prod"}),
        )

    async def deploy(_state: State, _context: object) -> dict[str, bool]:
        return {"deployed": True}

    graph = (
        GraphBuilder("approval")
        .add_node("approval", approval, kind=NodeKind.APPROVAL)
        .add_node("deploy", deploy)
        .add_edge("approval", "deploy")
        .set_entrypoint("approval")
        .set_finish_point("deploy")
        .build()
    )
    runtime = Runtime()
    suspended = await runtime.run(graph)

    assert suspended.status == ExecutionStatus.SUSPENDED
    assert suspended.last_checkpoint_id is not None
    resumed = await runtime.resume(graph, suspended.last_checkpoint_id)

    assert resumed.status == ExecutionStatus.COMPLETED
    assert resumed.state.data == {"approved": True, "deployed": True}
    assert approval_calls == 1
    assert [item.node_id for item in resumed.node_runs] == ["deploy"]


async def test_stream_yields_complete_ordered_run() -> None:
    async def finish(_state: State, _context: object) -> dict[str, int]:
        return {"value": 1}

    graph = (
        GraphBuilder("stream")
        .add_node("finish", finish)
        .set_entrypoint("finish")
        .set_finish_point("finish")
        .build()
    )
    events = [event async for event in Runtime().stream(graph)]

    assert events[0].kind == EventKind.RUN_STARTED
    assert events[-1].kind == EventKind.RUN_COMPLETED
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))


async def test_run_limit_prevents_unbounded_work_even_for_bounded_graph_cycle() -> None:
    async def loop(_state: State, _context: object) -> NodeResult:
        return NodeResult(route="again")

    async def back(_state: State, _context: object) -> None:
        return None

    async def finish(_state: State, _context: object) -> None:
        return None

    graph = (
        GraphBuilder("step-limit")
        .add_node("loop", loop, max_visits=10)
        .add_node("back", back, max_visits=10)
        .add_node("finish", finish)
        .add_conditional_edges("loop", {"again": "back", "done": "finish"})
        .add_edge("back", "loop")
        .set_entrypoint("loop")
        .set_finish_point("finish")
        .build()
    )
    runtime = Runtime(config=RuntimeConfig(limits=ExecutionLimits(max_steps=3)))
    result = await runtime.run(graph)

    assert result.status == ExecutionStatus.FAILED
    assert result.error is not None
    assert "exceeded 3 steps" in result.error.message
