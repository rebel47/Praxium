"""Framework application registry independent of any web server."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from praxium.agents import Agent, AgentRunner
from praxium.core import Conversation, FrameworkModel, Message
from praxium.graph import Graph
from praxium.models import EmbeddingRequest, EmbeddingResponse, Model
from praxium.runtime import ExecutionResult, Runtime


class ComponentKind(StrEnum):
    AGENT = "agent"
    GRAPH = "graph"
    EMBEDDING = "embedding"


class ComponentDescription(FrameworkModel):
    name: str
    kind: ComponentKind
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Application:
    """Registers executable components and exposes a uniform invocation API."""

    def __init__(
        self,
        *,
        runtime: Runtime | None = None,
        agent_runner: AgentRunner | None = None,
    ) -> None:
        self.runtime = runtime or Runtime()
        self.agent_runner = agent_runner
        self._components: dict[str, Agent | Graph] = {}
        self._embedding_models: dict[str, Model] = {}

    def register(self, component: Agent | Graph, *, name: str | None = None) -> Application:
        public_name = name or component.name
        if public_name in self._components or public_name in self._embedding_models:
            raise ValueError(f"component {public_name!r} is already registered")
        self._components[public_name] = component
        return self

    def register_embedding_model(self, name: str, model: Model) -> Application:
        if name in self._components or name in self._embedding_models:
            raise ValueError(f"component {name!r} is already registered")
        self._embedding_models[name] = model
        return self

    def describe(self) -> list[ComponentDescription]:
        descriptions = [
            ComponentDescription(
                name=name,
                kind=ComponentKind.AGENT if isinstance(value, Agent) else ComponentKind.GRAPH,
                description=value.instructions if isinstance(value, Agent) else None,
                metadata=value.metadata,
            )
            for name, value in self._components.items()
        ]
        descriptions.extend(
            ComponentDescription(name=name, kind=ComponentKind.EMBEDDING)
            for name in self._embedding_models
        )
        return sorted(descriptions, key=lambda value: value.name)

    def get(self, name: str) -> Agent | Graph:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(f"component {name!r} is not registered") from exc

    async def run(
        self,
        name: str,
        input: Any,
        *,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> Any:
        component = self.get(name)
        if isinstance(component, Graph):
            state = input if isinstance(input, dict) else {"input": input}
            return await self.runtime.run(
                component,
                state,
                metadata=metadata,
                tenant_id=tenant_id,
            )
        if self.agent_runner is None:
            raise RuntimeError("an AgentRunner is required to execute registered agents")
        normalized = input
        if not isinstance(input, (str, Message, Conversation)):
            normalized = str(input)
        return await self.agent_runner.run(component, normalized)

    async def embed(self, name: str, inputs: list[str]) -> EmbeddingResponse:
        try:
            model = self._embedding_models[name]
        except KeyError as exc:
            raise KeyError(f"embedding model {name!r} is not registered") from exc
        if self.agent_runner is None:
            raise RuntimeError("an AgentRunner provider registry is required for embeddings")
        provider = self.agent_runner.providers.get(model.provider)
        return await provider.embed(EmbeddingRequest(model=model, inputs=inputs))


def result_text(value: Any) -> str:
    """Extract a compatibility-friendly text result from an application result."""

    if hasattr(value, "response") and isinstance(value.response, Message):
        return value.response.text_content
    if isinstance(value, ExecutionResult):
        if isinstance(value.output, str):
            return value.output
        if "output" in value.state.data:
            return str(value.state.data["output"])
        return str(value.state.data)
    if isinstance(value, Message):
        return value.text_content
    return str(value)
