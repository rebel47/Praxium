"""Stable enums shared by framework packages."""

from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SUSPENDED = "suspended"


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SUSPENDED = "suspended"


class EventKind(StrEnum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_TIMED_OUT = "run.timed_out"
    RUN_SUSPENDED = "run.suspended"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    NODE_RETRYING = "node.retrying"
    NODE_STREAM = "node.stream"
    EDGE_SELECTED = "edge.selected"
    CHECKPOINT_SAVED = "checkpoint.saved"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    MODEL_STARTED = "model.started"
    MODEL_STREAM = "model.stream"
    MODEL_COMPLETED = "model.completed"
    MODEL_FAILED = "model.failed"
    MEMORY_READ = "memory.read"
    MEMORY_WRITTEN = "memory.written"
    APPROVAL_REQUESTED = "approval.requested"


class MergeStrategy(StrEnum):
    REPLACE = "replace"
    APPEND = "append"
    ADD = "add"
    SET_UNION = "set_union"
    RECURSIVE = "recursive"
    CUSTOM = "custom"
