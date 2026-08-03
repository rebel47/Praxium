"""Whole-graph validation with actionable issue collection."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from praxium.core import FrameworkModel

from .models import Graph


class GraphValidationIssue(FrameworkModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    node_id: str | None = None
    edge_index: int | None = Field(default=None, ge=0)


class GraphValidationReport(FrameworkModel):
    issues: list[GraphValidationIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[GraphValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


def validate_graph(graph: Graph) -> GraphValidationReport:
    issues: list[GraphValidationIssue] = []
    if graph.entrypoint not in graph.nodes:
        issues.append(_error("missing_entrypoint", "entrypoint does not reference a node"))
    for finish in sorted(graph.finish_points):
        if finish not in graph.nodes:
            issues.append(
                _error("missing_finish_point", f"finish point {finish!r} is missing", finish)
            )

    seen_edges: set[tuple[str, str, str | None]] = set()
    for index, edge in enumerate(graph.edges):
        if edge.source not in graph.nodes:
            issues.append(
                _edge_error("missing_edge_source", f"source {edge.source!r} is missing", index)
            )
        if edge.target not in graph.nodes:
            issues.append(
                _edge_error("missing_edge_target", f"target {edge.target!r} is missing", index)
            )
        identity = (edge.source, edge.target, edge.route)
        if identity in seen_edges:
            issues.append(_edge_error("duplicate_edge", "duplicate edge", index))
        seen_edges.add(identity)

    for node_id in graph.nodes:
        outgoing = graph.outgoing(node_id)
        routes = [edge.route for edge in outgoing]
        if len(outgoing) > 1 and (None in routes or len(set(routes)) != len(routes)):
            issues.append(
                _error(
                    "ambiguous_routing",
                    "multiple outgoing edges require unique route labels",
                    node_id,
                )
            )
        if node_id in graph.finish_points and outgoing:
            issues.append(
                GraphValidationIssue(
                    severity="warning",
                    code="finish_has_outgoing_edges",
                    message="finish point has outgoing edges that will not be followed",
                    node_id=node_id,
                )
            )

    if graph.entrypoint in graph.nodes:
        reachable = _reachable(graph, graph.entrypoint)
        for node_id in sorted(set(graph.nodes) - reachable):
            issues.append(
                _error("unreachable_node", "node is unreachable from entrypoint", node_id)
            )

    for cycle in _cycles(graph):
        if not any(graph.nodes[node_id].max_visits > 1 for node_id in cycle):
            issues.append(
                _error(
                    "unbounded_cycle",
                    f"cycle {sorted(cycle)} has no node with max_visits > 1",
                    sorted(cycle)[0],
                )
            )

    return GraphValidationReport(issues=issues)


def _reachable(graph: Graph, start: str) -> set[str]:
    visited: set[str] = set()
    pending = [start]
    while pending:
        node_id = pending.pop()
        if node_id in visited or node_id not in graph.nodes:
            continue
        visited.add(node_id)
        pending.extend(edge.target for edge in graph.outgoing(node_id))
    return visited


def _cycles(graph: Graph) -> list[set[str]]:
    """Return strongly connected components that form cycles."""

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(node_id: str) -> None:
        nonlocal index
        indices[node_id] = index
        lowlinks[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for edge in graph.outgoing(node_id):
            target = edge.target
            if target not in graph.nodes:
                continue
            if target not in indices:
                visit(target)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target])
            elif target in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indices[target])
        if lowlinks[node_id] == indices[node_id]:
            component: set[str] = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node_id:
                    break
            self_loop = any(edge.target == node_id for edge in graph.outgoing(node_id))
            if len(component) > 1 or self_loop:
                components.append(component)

    for node_id in graph.nodes:
        if node_id not in indices:
            visit(node_id)
    return components


def _error(code: str, message: str, node_id: str | None = None) -> GraphValidationIssue:
    return GraphValidationIssue(severity="error", code=code, message=message, node_id=node_id)


def _edge_error(code: str, message: str, index: int) -> GraphValidationIssue:
    return GraphValidationIssue(severity="error", code=code, message=message, edge_index=index)
