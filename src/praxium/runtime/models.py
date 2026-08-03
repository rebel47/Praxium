"""Execution result and runtime configuration models."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from praxium.core import (
    CheckpointId,
    ErrorDetail,
    ExecutionEvent,
    ExecutionId,
    ExecutionLimits,
    ExecutionStatus,
    FrameworkModel,
    GraphId,
    NodeStatus,
    State,
)


class RuntimeConfig(FrameworkModel):
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    checkpoint_after_each_node: bool = True
    retain_events_in_result: bool = True


class NodeRun(FrameworkModel):
    node_id: str
    status: NodeStatus
    attempts: int = Field(ge=1)
    state_version_before: int = Field(ge=0)
    state_version_after: int = Field(ge=0)
    output: Any = None
    route: str | None = None
    duration_seconds: float = Field(ge=0)
    error: ErrorDetail | None = None


class ExecutionResult(FrameworkModel):
    execution_id: ExecutionId
    graph_id: GraphId
    status: ExecutionStatus
    state: State
    output: Any = None
    node_runs: list[NodeRun] = Field(default_factory=list)
    events: list[ExecutionEvent] = Field(default_factory=list)
    last_checkpoint_id: CheckpointId | None = None
    error: ErrorDetail | None = None
