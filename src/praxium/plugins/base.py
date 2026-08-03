"""Explicit, namespaced plugin discovery and registration."""

from __future__ import annotations

import importlib.metadata
import re
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from praxium._version import __version__
from praxium.core import FrameworkModel


class PluginCapability(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    NODE = "node"
    MIDDLEWARE = "middleware"
    MEMORY = "memory"
    STORAGE = "storage"
    DEPLOYMENT = "deployment"
    PROVIDER = "provider"
    CLI = "cli"
    VISUAL = "visual"


class PluginManifest(FrameworkModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:[-+][a-zA-Z0-9.-]+)?$")
    description: str
    author: str | None = None
    min_framework_version: str = "0.1.0"
    max_framework_major: int | None = Field(default=None, ge=0)
    capabilities: set[PluginCapability] = Field(default_factory=set)
    permissions: set[str] = Field(default_factory=set)


class PluginRegistration(FrameworkModel):
    capability: PluginCapability
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,127}$")
    component: Any = Field(exclude=True, repr=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plugin(FrameworkModel):
    manifest: PluginManifest
    registrations: list[PluginRegistration] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_capabilities(self) -> Plugin:
        undeclared = {
            item.capability
            for item in self.registrations
            if item.capability not in self.manifest.capabilities
        }
        if undeclared:
            raise ValueError(f"plugin uses undeclared capabilities: {sorted(undeclared)}")
        return self


class DiscoveredPlugin(FrameworkModel):
    name: str
    value: str
    group: str
    distribution: str | None = None


class PluginRegistry:
    """Registers compatible plugins without implicit replacement."""

    def __init__(self, *, framework_version: str = __version__) -> None:
        self.framework_version = framework_version
        self._plugins: dict[str, Plugin] = {}
        self._components: dict[tuple[PluginCapability, str], tuple[str, Any]] = {}

    def register(self, plugin: Plugin) -> None:
        name = plugin.manifest.name
        if name in self._plugins:
            raise ValueError(f"plugin {name!r} is already registered")
        _check_compatibility(plugin.manifest, self.framework_version)
        pending = [(item.capability, item.name) for item in plugin.registrations]
        conflicts = [key for key in pending if key in self._components]
        if conflicts:
            rendered = ", ".join(
                f"{capability.value}:{component}" for capability, component in conflicts
            )
            raise ValueError(f"plugin registrations conflict: {rendered}")
        self._plugins[name] = plugin
        for registration in plugin.registrations:
            self._components[(registration.capability, registration.name)] = (
                name,
                registration.component,
            )

    def get(self, capability: PluginCapability, name: str) -> Any:
        try:
            return self._components[(capability, name)][1]
        except KeyError as exc:
            raise KeyError(f"plugin component {capability.value}:{name} is not registered") from exc

    def plugins(self) -> tuple[PluginManifest, ...]:
        return tuple(self._plugins[name].manifest for name in sorted(self._plugins))


class PluginLoader:
    """Discovers metadata without importing code; loading is always explicit."""

    def __init__(self, group: str = "praxium.plugins") -> None:
        self.group = group

    def discover(self) -> list[DiscoveredPlugin]:
        points = importlib.metadata.entry_points()
        selected = points.select(group=self.group)
        return [
            DiscoveredPlugin(
                name=point.name,
                value=point.value,
                group=point.group,
                distribution=point.dist.name if point.dist else None,
            )
            for point in sorted(selected, key=lambda item: item.name)
        ]

    def load(self, name: str, registry: PluginRegistry) -> Plugin:
        points = importlib.metadata.entry_points().select(group=self.group, name=name)
        matches = list(points)
        if not matches:
            raise KeyError(f"plugin entry point {name!r} was not found")
        loaded = matches[0].load()
        plugin = loaded() if isinstance(loaded, type) or callable(loaded) else loaded
        if not isinstance(plugin, Plugin):
            raise TypeError("plugin entry point must return a Plugin model")
        registry.register(plugin)
        return plugin


def _check_compatibility(manifest: PluginManifest, framework_version: str) -> None:
    framework = _version_tuple(framework_version)
    minimum = _version_tuple(manifest.min_framework_version)
    if framework < minimum:
        raise ValueError(
            f"plugin {manifest.name!r} requires framework >= {manifest.min_framework_version}"
        )
    if manifest.max_framework_major is not None and framework[0] > manifest.max_framework_major:
        raise ValueError(
            f"plugin {manifest.name!r} supports framework major <= {manifest.max_framework_major}"
        )


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"invalid semantic version {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]
