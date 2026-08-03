"""Serializable graph schema and executable node adapters."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from praxium.core import (
    ExecutionContext,
    FrameworkModel,
    GraphId,
    RetryPolicy,
    State,
    StatePatch,
)

NodeHandler = Callable[[State, ExecutionContext], Any]


class NodeKind(StrEnum):
    TASK = "task"
    CONDITION = "condition"
    APPROVAL = "approval"
    SUBGRAPH = "subgraph"


class Suspension(FrameworkModel):
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)


class NodeResult(FrameworkModel):
    """Normalized result returned by every executed node."""

    patch: StatePatch = Field(default_factory=StatePatch)
    output: Any = None
    route: str | None = None
    suspension: Suspension | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def normalize(cls, value: Any) -> NodeResult:
        if isinstance(value, cls):
            return value
        if isinstance(value, StatePatch):
            return cls(patch=value)
        if isinstance(value, State):
            return cls(patch=StatePatch(values=value.data))
        if isinstance(value, dict):
            return cls(patch=StatePatch(values=value))
        if value is None:
            return cls()
        return cls(output=value)


class Node(FrameworkModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,127}$")
    name: str = Field(min_length=1)
    kind: NodeKind = NodeKind.TASK
    handler: NodeHandler | None = Field(default=None, exclude=True, repr=False)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_visits: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Edge(FrameworkModel):
    source: str
    target: str
    route: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Graph(FrameworkModel):
    id: GraphId = Field(default_factory=GraphId.new)
    name: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    nodes: dict[str, Node]
    edges: list[Edge] = Field(default_factory=list)
    entrypoint: str
    finish_points: set[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_node_keys(self) -> Graph:
        mismatched = [key for key, node in self.nodes.items() if key != node.id]
        if mismatched:
            raise ValueError(f"node map keys must match node IDs: {mismatched}")
        return self

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump_json(
            exclude={"fingerprint"},
            exclude_none=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def outgoing(self, node_id: str) -> list[Edge]:
        return sorted(
            (edge for edge in self.edges if edge.source == node_id),
            key=lambda edge: (-edge.priority, edge.route or "", edge.target),
        )

    def incoming(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            label = re.sub(r'["\n]', " ", node.name)
            shape = f'{{{{"{label}"}}}}' if node.kind == NodeKind.CONDITION else f'["{label}"]'
            lines.append(f"    {node_id}{shape}")
        for edge in self.edges:
            label = f"|{edge.route}|" if edge.route else ""
            lines.append(f"    {edge.source} -->{label} {edge.target}")
        return "\n".join(lines)
