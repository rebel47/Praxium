"""Shared Pydantic model configuration and time helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class FrameworkModel(BaseModel):
    """Base for public framework value objects."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class FrozenFrameworkModel(FrameworkModel):
    """Base for immutable public value objects."""

    model_config = ConfigDict(frozen=True)
