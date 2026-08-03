"""Suspend a graph for approval and resume after its committed checkpoint."""

from __future__ import annotations

import asyncio

from praxium import (
    ExecutionContext,
    GraphBuilder,
    NodeKind,
    NodeResult,
    Runtime,
    State,
    StatePatch,
    Suspension,
)


async def request_approval(
    _state: State,
    _context: ExecutionContext,
) -> NodeResult:
    return NodeResult(
        patch=StatePatch(values={"approval_requested": True}),
        suspension=Suspension(
            reason="Approve production deployment",
            payload={"environment": "production"},
        ),
    )


async def deploy(_state: State, context: ExecutionContext) -> dict[str, str]:
    return {
        "deployment": "completed",
        "approved_by": str(context.metadata.get("approved_by", "unknown")),
    }


async def main() -> None:
    graph = (
        GraphBuilder("deployment")
        .add_node("approval", request_approval, kind=NodeKind.APPROVAL)
        .add_node("deploy", deploy)
        .add_edge("approval", "deploy")
        .set_entrypoint("approval")
        .set_finish_point("deploy")
        .build()
    )
    runtime = Runtime()
    suspended = await runtime.run(graph)
    print(f"first_status={suspended.status}")

    checkpoint_id = suspended.last_checkpoint_id
    if checkpoint_id is None:
        raise RuntimeError("suspended run did not produce a checkpoint")

    # In an application, call resume only after an authorized user confirms.
    resumed = await runtime.resume(
        graph,
        checkpoint_id,
        metadata={"approved_by": "user-123"},
    )
    print(f"resumed_status={resumed.status}")
    print(resumed.state.data)


if __name__ == "__main__":
    asyncio.run(main())
