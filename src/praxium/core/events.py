"""Ordered execution event envelope."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from .base import FrameworkModel, utc_now
from .enums import EventKind
from .ids import EventId, ExecutionId, GraphId


class ExecutionEvent(FrameworkModel):
    id: EventId = Field(default_factory=EventId.new)
    execution_id: ExecutionId
    graph_id: GraphId
    sequence: int = Field(ge=1)
    kind: EventKind
    timestamp: datetime = Field(default_factory=utc_now)
    node_id: str | None = None
    attempt: int | None = Field(default=None, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
