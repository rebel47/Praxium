"""Cancellation-aware sequential and conditional graph runtime."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from praxium.core import (
    CancellationToken,
    CheckpointId,
    ErrorDetail,
    EventKind,
    ExecutionCancelledError,
    ExecutionContext,
    ExecutionError,
    ExecutionId,
    ExecutionStatus,
    ExecutionTimeoutError,
    FrameworkError,
    NodeExecutionError,
    NodeStatus,
    State,
)
from praxium.graph import Graph, Node, NodeResult, validate_graph
from praxium.observability import EventBus, EventEmitter, EventSink
from praxium.observability.events import QueueEventSink

from .checkpoints import Checkpoint, CheckpointStore, InMemoryCheckpointStore
from .models import ExecutionResult, NodeRun, RuntimeConfig


class Runtime:
    """Executes validated graphs with deterministic routing and durable boundaries."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        checkpoint_store: CheckpointStore | None = None,
        event_sinks: Sequence[EventSink] = (),
    ) -> None:
        self.config = config or RuntimeConfig()
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self.event_sinks = tuple(event_sinks)
        self._active: dict[ExecutionId, CancellationToken] = {}
        self._active_lock = asyncio.Lock()

    async def run(
        self,
        graph: Graph,
        initial_state: State | dict[str, Any] | None = None,
        *,
        execution_id: ExecutionId | None = None,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
        _start_node: str | None = None,
        _visit_counts: dict[str, int] | None = None,
        _completed_nodes: list[str] | None = None,
        _extra_sinks: Sequence[EventSink] = (),
    ) -> ExecutionResult:
        report = validate_graph(graph)
        if not report.valid:
            details = "; ".join(f"{item.code}: {item.message}" for item in report.errors)
            raise ExecutionError(f"cannot execute invalid graph: {details}")

        run_id = execution_id or ExecutionId.new()
        state = (
            initial_state if isinstance(initial_state, State) else State(data=initial_state or {})
        )
        token = CancellationToken()
        emitter = EventBus((*self.event_sinks, *_extra_sinks)).emitter(run_id, graph.id)
        deadline = _deadline(self.config.limits.timeout_seconds)
        base_context = ExecutionContext(
            execution_id=run_id,
            graph_id=graph.id,
            deadline=deadline,
            tenant_id=tenant_id,
            metadata=metadata or {},
            cancellation=token,
        )
        node_runs: list[NodeRun] = []
        visits = dict(_visit_counts or {})
        completed = list(_completed_nodes or [])
        current: str | None = _start_node if _start_node is not None else graph.entrypoint
        output: Any = None
        last_checkpoint: Checkpoint | None = None
        status = ExecutionStatus.RUNNING
        error_detail: ErrorDetail | None = None

        async with self._active_lock:
            if run_id in self._active:
                raise ExecutionError(f"execution {run_id} is already active")
            self._active[run_id] = token

        await emitter.emit(
            EventKind.RUN_STARTED,
            payload={
                "graph_name": graph.name,
                "graph_version": graph.version,
                "state_version": state.version,
                "resumed": _start_node is not None,
            },
        )

        try:
            steps = 0
            while current is not None:
                token.raise_if_cancelled()
                if base_context.remaining_seconds == 0:
                    raise ExecutionTimeoutError("execution deadline was reached")
                steps += 1
                if steps > self.config.limits.max_steps:
                    raise ExecutionError(f"execution exceeded {self.config.limits.max_steps} steps")
                node = graph.nodes[current]
                visits[current] = visits.get(current, 0) + 1
                if visits[current] > node.max_visits:
                    raise ExecutionError(
                        f"node {current!r} exceeded max_visits={node.max_visits}",
                        context={"node_id": current, "visits": visits[current]},
                    )

                node_result, node_run = await self._execute_node(
                    node,
                    state,
                    base_context,
                    emitter,
                )
                node_runs.append(node_run)
                state = state.apply(node_result.patch)
                node_run.state_version_after = state.version
                output = node_result.output if node_result.output is not None else output
                completed.append(current)
                next_node = self._select_next(graph, current, node_result)
                if next_node is not None:
                    await emitter.emit(
                        EventKind.EDGE_SELECTED,
                        node_id=current,
                        payload={"target": next_node, "route": node_result.route},
                    )

                if self.config.checkpoint_after_each_node or node_result.suspension is not None:
                    last_checkpoint = await self._save_checkpoint(
                        graph,
                        run_id,
                        state,
                        next_node,
                        visits,
                        completed,
                        output,
                        emitter,
                    )

                if node_result.suspension is not None:
                    status = ExecutionStatus.SUSPENDED
                    await emitter.emit(
                        EventKind.APPROVAL_REQUESTED,
                        node_id=current,
                        payload=node_result.suspension.model_dump(mode="json"),
                    )
                    await emitter.emit(
                        EventKind.RUN_SUSPENDED,
                        node_id=current,
                        payload={
                            "checkpoint_id": str(last_checkpoint.id) if last_checkpoint else None,
                            "next_node": next_node,
                        },
                    )
                    break

                current = next_node
            else:
                status = ExecutionStatus.COMPLETED

            if status == ExecutionStatus.RUNNING:
                status = ExecutionStatus.COMPLETED
            if status == ExecutionStatus.COMPLETED:
                await emitter.emit(
                    EventKind.RUN_COMPLETED,
                    payload={"state_version": state.version, "steps": len(node_runs)},
                )
        except ExecutionCancelledError as exc:
            status = ExecutionStatus.CANCELLED
            error_detail = exc.to_detail()
            await emitter.emit(
                EventKind.RUN_CANCELLED, payload=error_detail.model_dump(mode="json")
            )
        except ExecutionTimeoutError as exc:
            status = ExecutionStatus.TIMED_OUT
            error_detail = exc.to_detail()
            await emitter.emit(
                EventKind.RUN_TIMED_OUT, payload=error_detail.model_dump(mode="json")
            )
        except Exception as exc:
            status = ExecutionStatus.FAILED
            wrapped = exc if isinstance(exc, FrameworkError) else ExecutionError(str(exc))
            error_detail = wrapped.to_detail()
            await emitter.emit(EventKind.RUN_FAILED, payload=error_detail.model_dump(mode="json"))
        finally:
            async with self._active_lock:
                self._active.pop(run_id, None)

        return ExecutionResult(
            execution_id=run_id,
            graph_id=graph.id,
            status=status,
            state=state,
            output=output,
            node_runs=node_runs,
            events=list(emitter.events) if self.config.retain_events_in_result else [],
            last_checkpoint_id=last_checkpoint.id if last_checkpoint else None,
            error=error_detail,
        )

    async def resume(
        self,
        graph: Graph,
        checkpoint_id: CheckpointId,
        *,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> ExecutionResult:
        """Resume at the node after the checkpoint without repeating committed work."""

        checkpoint = await self.checkpoint_store.load(checkpoint_id)
        if checkpoint.graph_id != graph.id or checkpoint.graph_version != graph.version:
            raise ExecutionError("checkpoint belongs to a different graph version")
        if checkpoint.graph_fingerprint != graph.fingerprint:
            raise ExecutionError("graph structure has changed since the checkpoint")
        if checkpoint.next_node is None:
            return ExecutionResult(
                execution_id=ExecutionId.new(),
                graph_id=graph.id,
                status=ExecutionStatus.COMPLETED,
                state=checkpoint.state,
                output=checkpoint.output,
                last_checkpoint_id=checkpoint.id,
            )
        resume_metadata = {**(metadata or {}), "resumed_from": str(checkpoint_id)}
        return await self.run(
            graph,
            checkpoint.state,
            metadata=resume_metadata,
            tenant_id=tenant_id,
            _start_node=checkpoint.next_node,
            _visit_counts=checkpoint.visit_counts,
            _completed_nodes=checkpoint.completed_nodes,
        )

    async def stream(
        self,
        graph: Graph,
        initial_state: State | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Yield ordered events while a run executes."""

        sink = QueueEventSink()
        task = asyncio.create_task(self.run(graph, initial_state, _extra_sinks=(sink,), **kwargs))
        try:
            while True:
                if task.done() and sink.queue.empty():
                    await task
                    break
                get_event = asyncio.create_task(sink.queue.get())
                done, _ = await asyncio.wait({get_event, task}, return_when=asyncio.FIRST_COMPLETED)
                if get_event in done:
                    yield get_event.result()
                else:
                    get_event.cancel()
                    await asyncio.gather(get_event, return_exceptions=True)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def cancel(self, execution_id: ExecutionId, reason: str = "cancelled by caller") -> bool:
        async with self._active_lock:
            token = self._active.get(execution_id)
            if token is None:
                return False
            token.cancel(reason)
            return True

    async def _execute_node(
        self,
        node: Node,
        state: State,
        context: ExecutionContext,
        emitter: EventEmitter,
    ) -> tuple[NodeResult, NodeRun]:
        started = asyncio.get_running_loop().time()
        before_version = state.version
        last_error: FrameworkError | None = None

        for attempt in range(1, node.retry_policy.max_attempts + 1):
            node_context = context.for_node(node.id, attempt)
            await emitter.emit(EventKind.NODE_STARTED, node_id=node.id, attempt=attempt)
            try:
                value = await _invoke_with_control(
                    node,
                    state.model_copy(deep=True),
                    node_context,
                    self.config.limits.node_timeout_seconds,
                )
                result = NodeResult.normalize(value)
                duration = asyncio.get_running_loop().time() - started
                await emitter.emit(
                    EventKind.NODE_COMPLETED,
                    node_id=node.id,
                    attempt=attempt,
                    payload={"route": result.route, "has_output": result.output is not None},
                )
                return result, NodeRun(
                    node_id=node.id,
                    status=NodeStatus.SUSPENDED if result.suspension else NodeStatus.COMPLETED,
                    attempts=attempt,
                    state_version_before=before_version,
                    state_version_after=before_version,
                    output=result.output,
                    route=result.route,
                    duration_seconds=duration,
                )
            except (ExecutionCancelledError, ExecutionTimeoutError):
                raise
            except Exception as exc:
                last_error = (
                    exc
                    if isinstance(exc, FrameworkError)
                    else NodeExecutionError(
                        f"node {node.id!r} failed: {exc}",
                        context={"node_id": node.id, "exception_type": type(exc).__name__},
                    )
                )
                await emitter.emit(
                    EventKind.NODE_FAILED,
                    node_id=node.id,
                    attempt=attempt,
                    payload=last_error.to_detail().model_dump(mode="json"),
                )
                if not node.retry_policy.should_retry(last_error.code, attempt):
                    break
                delay = node.retry_policy.delay_for(attempt)
                await emitter.emit(
                    EventKind.NODE_RETRYING,
                    node_id=node.id,
                    attempt=attempt,
                    payload={"delay_seconds": delay, "next_attempt": attempt + 1},
                )
                await _cancellable_sleep(delay, context.cancellation)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _select_next(graph: Graph, current: str, result: NodeResult) -> str | None:
        if current in graph.finish_points:
            return None
        outgoing = graph.outgoing(current)
        if not outgoing:
            raise ExecutionError(f"non-terminal node {current!r} has no outgoing edge")
        if result.route is None:
            if len(outgoing) == 1 and outgoing[0].route is None:
                return outgoing[0].target
            raise ExecutionError(f"node {current!r} did not choose a required route")
        matches = [edge for edge in outgoing if edge.route == result.route]
        if not matches:
            raise ExecutionError(
                f"node {current!r} selected unknown route {result.route!r}",
                context={"available_routes": [edge.route for edge in outgoing]},
            )
        return matches[0].target

    async def _save_checkpoint(
        self,
        graph: Graph,
        execution_id: ExecutionId,
        state: State,
        next_node: str | None,
        visits: dict[str, int],
        completed: list[str],
        output: Any,
        emitter: EventEmitter,
    ) -> Checkpoint:
        checkpoint = await self.checkpoint_store.save(
            Checkpoint(
                execution_id=execution_id,
                graph_id=graph.id,
                graph_version=graph.version,
                graph_fingerprint=graph.fingerprint,
                state=state,
                next_node=next_node,
                visit_counts=visits,
                completed_nodes=completed,
                output=output,
            )
        )
        await emitter.emit(
            EventKind.CHECKPOINT_SAVED,
            payload={"checkpoint_id": str(checkpoint.id), "next_node": next_node},
        )
        return checkpoint


async def _invoke_with_control(
    node: Node,
    state: State,
    context: ExecutionContext,
    default_timeout: float | None,
) -> Any:
    handler = node.handler
    if handler is None:
        raise NodeExecutionError(f"node {node.id!r} has no executable handler")

    async def invoke() -> Any:
        if inspect.iscoroutinefunction(handler):
            return await handler(state, context)
        value = await asyncio.to_thread(handler, state, context)
        if inspect.isawaitable(value):
            return await value
        return value

    timeout_candidates = [
        value
        for value in (node.timeout_seconds, default_timeout, context.remaining_seconds)
        if value is not None
    ]
    timeout = min(timeout_candidates) if timeout_candidates else None
    work = asyncio.create_task(invoke())
    cancelled = asyncio.create_task(context.cancellation.wait())
    try:
        done, _ = await asyncio.wait(
            {work, cancelled},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if work in done:
            return await work
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        if cancelled in done:
            raise ExecutionCancelledError(context.cancellation.reason or "execution cancelled")
        raise ExecutionTimeoutError(
            f"node {node.id!r} exceeded its timeout",
            context={"node_id": node.id, "timeout_seconds": timeout},
        )
    finally:
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)


async def _cancellable_sleep(delay: float, token: CancellationToken) -> None:
    if delay <= 0:
        token.raise_if_cancelled()
        return
    sleep = asyncio.create_task(asyncio.sleep(delay))
    cancelled = asyncio.create_task(token.wait())
    try:
        done, _ = await asyncio.wait({sleep, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        if cancelled in done:
            sleep.cancel()
            await asyncio.gather(sleep, return_exceptions=True)
            raise ExecutionCancelledError(token.reason or "execution cancelled")
    finally:
        for task in (sleep, cancelled):
            if not task.done():
                task.cancel()
        await asyncio.gather(sleep, cancelled, return_exceptions=True)


def _deadline(timeout_seconds: float | None) -> datetime | None:
    if timeout_seconds is None:
        return None
    return datetime.now(UTC) + timedelta(seconds=timeout_seconds)
