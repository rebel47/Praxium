"""Fluent graph construction that validates before returning a graph."""

from __future__ import annotations

from typing import Any

from praxium.core import GraphId, GraphValidationError, RetryPolicy

from .models import Edge, Graph, Node, NodeHandler, NodeKind
from .validation import validate_graph


class GraphBuilder:
    def __init__(self, name: str, *, graph_id: GraphId | None = None, version: int = 1) -> None:
        self.name = name
        self.graph_id = graph_id or GraphId.new()
        self.version = version
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._entrypoint: str | None = None
        self._finish_points: set[str] = set()
        self._metadata: dict[str, Any] = {}

    def add_node(
        self,
        node_id: str,
        handler: NodeHandler | None = None,
        *,
        name: str | None = None,
        kind: NodeKind = NodeKind.TASK,
        retry_policy: RetryPolicy | None = None,
        timeout_seconds: float | None = None,
        max_visits: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> GraphBuilder:
        if node_id in self._nodes:
            raise GraphValidationError(f"node {node_id!r} is already defined")
        self._nodes[node_id] = Node(
            id=node_id,
            name=name or node_id.replace("_", " ").title(),
            kind=kind,
            handler=handler,
            retry_policy=retry_policy or RetryPolicy(),
            timeout_seconds=timeout_seconds,
            max_visits=max_visits,
            metadata=metadata or {},
        )
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        route: str | None = None,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> GraphBuilder:
        self._edges.append(
            Edge(
                source=source,
                target=target,
                route=route,
                priority=priority,
                metadata=metadata or {},
            )
        )
        return self

    def add_conditional_edges(self, source: str, routes: dict[str, str]) -> GraphBuilder:
        for route, target in routes.items():
            self.add_edge(source, target, route=route)
        return self

    def set_entrypoint(self, node_id: str) -> GraphBuilder:
        self._entrypoint = node_id
        return self

    def set_finish_point(self, node_id: str) -> GraphBuilder:
        self._finish_points.add(node_id)
        return self

    def set_metadata(self, **metadata: Any) -> GraphBuilder:
        self._metadata.update(metadata)
        return self

    def build(self) -> Graph:
        if self._entrypoint is None:
            raise GraphValidationError("graph has no entrypoint")
        if not self._finish_points:
            raise GraphValidationError("graph has no finish point")
        graph = Graph(
            id=self.graph_id,
            name=self.name,
            version=self.version,
            nodes=dict(self._nodes),
            edges=list(self._edges),
            entrypoint=self._entrypoint,
            finish_points=set(self._finish_points),
            metadata=dict(self._metadata),
        )
        report = validate_graph(graph)
        if not report.valid:
            summary = "; ".join(f"{issue.code}: {issue.message}" for issue in report.errors)
            raise GraphValidationError(
                f"graph validation failed: {summary}",
                context={"issues": [issue.model_dump(mode="json") for issue in report.errors]},
            )
        return graph
