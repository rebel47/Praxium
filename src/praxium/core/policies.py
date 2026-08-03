"""Reusable execution limits and retry policies."""

from __future__ import annotations

import random

from pydantic import Field

from .base import FrameworkModel


class RetryPolicy(FrameworkModel):
    max_attempts: int = Field(default=1, ge=1, le=100)
    initial_delay_seconds: float = Field(default=0.0, ge=0)
    multiplier: float = Field(default=2.0, ge=1)
    max_delay_seconds: float = Field(default=60.0, ge=0)
    jitter: float = Field(default=0.0, ge=0, le=1)
    retryable_errors: set[str] = Field(default_factory=set)

    def delay_for(self, failed_attempt: int, *, random_value: float | None = None) -> float:
        """Return the delay before the next attempt."""

        base = min(
            self.initial_delay_seconds * (self.multiplier ** max(failed_attempt - 1, 0)),
            self.max_delay_seconds,
        )
        if not base or not self.jitter:
            return base
        value = random.random() if random_value is None else random_value
        return base * (1 - self.jitter + (2 * self.jitter * value))

    def should_retry(self, error_code: str, failed_attempt: int) -> bool:
        if failed_attempt >= self.max_attempts:
            return False
        return not self.retryable_errors or error_code in self.retryable_errors


class ExecutionLimits(FrameworkModel):
    timeout_seconds: float | None = Field(default=None, gt=0)
    node_timeout_seconds: float | None = Field(default=None, gt=0)
    max_steps: int = Field(default=100, ge=1)
    max_tool_calls: int = Field(default=25, ge=0)
    max_output_bytes: int = Field(default=1_000_000, ge=1)
    concurrency: int = Field(default=10, ge=1)
