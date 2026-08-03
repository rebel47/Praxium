"""Versioned state and deterministic merge behavior."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import Field

from .base import FrameworkModel, utc_now
from .enums import MergeStrategy
from .errors import StateConflictError

MergeResolver = Callable[[list[Any]], Any]


class StateChange(FrameworkModel):
    path: str
    before: Any = None
    after: Any = None
    version: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)


class StatePatch(FrameworkModel):
    values: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None


class MergePolicy(FrameworkModel):
    strategy: MergeStrategy = MergeStrategy.REPLACE
    resolver: MergeResolver | None = Field(default=None, exclude=True, repr=False)


class State(FrameworkModel):
    data: dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"
    version: int = Field(default=0, ge=0)
    history: list[StateChange] = Field(default_factory=list)

    def apply(self, patch: StatePatch | Mapping[str, Any]) -> State:
        """Return a new state with a single sequential patch applied."""

        values = patch.values if isinstance(patch, StatePatch) else dict(patch)
        next_version = self.version + 1
        next_data = {**self.data, **values}
        changes = [
            StateChange(
                path=key,
                before=self.data.get(key),
                after=value,
                version=next_version,
            )
            for key, value in values.items()
            if key not in self.data or self.data[key] != value
        ]
        return self.model_copy(
            update={
                "data": next_data,
                "version": next_version,
                "history": [*self.history, *changes],
            }
        )

    def merge(
        self,
        patches: Sequence[StatePatch | Mapping[str, Any]],
        policies: Mapping[str, MergePolicy] | None = None,
    ) -> State:
        """Merge parallel patches, rejecting undeclared conflicting writes."""

        normalized = [
            patch.values if isinstance(patch, StatePatch) else dict(patch) for patch in patches
        ]
        grouped: dict[str, list[Any]] = {}
        for patch in normalized:
            for key, value in patch.items():
                grouped.setdefault(key, []).append(value)

        resolved: dict[str, Any] = {}
        for key, values in grouped.items():
            if len(values) == 1 or all(value == values[0] for value in values[1:]):
                resolved[key] = values[-1]
                continue
            policy = (policies or {}).get(key)
            if policy is None:
                raise StateConflictError(
                    f"parallel writes conflict at {key!r}",
                    context={"path": key, "write_count": len(values)},
                )
            resolved[key] = _resolve_values(key, values, policy)
        return self.apply(resolved)


def _resolve_values(key: str, values: list[Any], policy: MergePolicy) -> Any:
    if policy.strategy == MergeStrategy.REPLACE:
        return values[-1]
    if policy.strategy == MergeStrategy.APPEND:
        result: list[Any] = []
        for value in values:
            result.extend(value if isinstance(value, list) else [value])
        return result
    if policy.strategy == MergeStrategy.ADD:
        try:
            return sum(values)
        except TypeError as exc:
            raise StateConflictError(f"values at {key!r} cannot be added") from exc
    if policy.strategy == MergeStrategy.SET_UNION:
        result_set: set[Any] = set()
        for value in values:
            result_set.update(value if isinstance(value, (list, set, tuple)) else [value])
        return sorted(result_set, key=repr)
    if policy.strategy == MergeStrategy.RECURSIVE:
        result_dict: dict[str, Any] = {}
        for value in values:
            if not isinstance(value, Mapping):
                raise StateConflictError(f"values at {key!r} are not all mappings")
            result_dict = _recursive_merge(result_dict, value, key)
        return result_dict
    if policy.strategy == MergeStrategy.CUSTOM and policy.resolver is not None:
        return policy.resolver(values)
    raise StateConflictError(f"merge policy for {key!r} has no usable resolver")


def _recursive_merge(
    left: Mapping[str, Any], right: Mapping[str, Any], path: str
) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        child_path = f"{path}.{key}"
        if key not in result or result[key] == value:
            result[key] = value
        elif isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _recursive_merge(result[key], value, child_path)
        else:
            raise StateConflictError(f"recursive merge conflict at {child_path!r}")
    return result
