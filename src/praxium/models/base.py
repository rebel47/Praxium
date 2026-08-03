"""Provider-neutral model contracts and deterministic test provider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, PrivateAttr

from praxium.core import (
    FrameworkModel,
    Message,
    Response,
    ResponseDelta,
    TextPart,
    Usage,
)


class ModelCapabilities(FrameworkModel):
    streaming: bool = True
    tools: bool = True
    structured_output: bool = True
    json_mode: bool = True
    embeddings: bool = False
    vision: bool = False


class Model(FrameworkModel):
    """Serializable model selection and generation configuration."""

    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(FrameworkModel):
    name: str = Field(min_length=1)
    description: str
    input_schema: dict[str, Any]


class ModelRequest(FrameworkModel):
    model: Model
    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(FrameworkModel):
    model: Model
    inputs: list[str] = Field(min_length=1)
    dimensions: int | None = Field(default=None, ge=1)


class EmbeddingResponse(FrameworkModel):
    embeddings: list[list[float]]
    model: str
    usage: Usage = Field(default_factory=Usage)


@runtime_checkable
class ModelProvider(Protocol):
    """Contract implemented by model provider adapters."""

    @property
    def name(self) -> str: ...

    async def complete(self, request: ModelRequest) -> Response: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ResponseDelta]: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


class ModelProviderRegistry:
    """Conflict-safe registry of explicitly supplied provider instances."""

    def __init__(self, providers: Sequence[ModelProvider] = ()) -> None:
        self._providers: dict[str, ModelProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ModelProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"model provider {provider.name!r} is already registered")
        self._providers[provider.name] = provider

    def get(self, name: str) -> ModelProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"model provider {name!r} is not registered") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))


class DeterministicModelProvider(FrameworkModel):
    """A dependency-free provider for tests, examples, and offline development."""

    provider_name: str = "deterministic"
    responses: list[Response] = Field(min_length=1)
    stream_by_character: bool = False
    _index: int = PrivateAttr(default=0)

    @property
    def name(self) -> str:
        return self.provider_name

    async def complete(self, request: ModelRequest) -> Response:
        del request
        index = min(self._index, len(self.responses) - 1)
        self._index += 1
        return self.responses[index].model_copy(deep=True)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ResponseDelta]:
        response = await self.complete(request)
        text = response.message.text_content
        pieces = list(text) if self.stream_by_character else ([text] if text else [])
        for index, piece in enumerate(pieces):
            yield ResponseDelta(index=index, text=piece)
        yield ResponseDelta(
            index=len(pieces),
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        dimensions = request.dimensions or 8
        vectors: list[list[float]] = []
        for text in request.inputs:
            buckets = [0.0] * dimensions
            for index, character in enumerate(text.encode("utf-8")):
                buckets[index % dimensions] += character / 255
            magnitude = sum(value * value for value in buckets) ** 0.5 or 1.0
            vectors.append([value / magnitude for value in buckets])
        return EmbeddingResponse(
            embeddings=vectors,
            model=request.model.name,
            usage=Usage(input_tokens=sum(len(text.split()) for text in request.inputs)),
        )


def text_response(text: str, *, model: str = "deterministic") -> Response:
    """Convenience factory for deterministic providers."""

    return Response(message=Message.assistant(text), model=model)


def response_text_parts(response: Response) -> list[TextPart]:
    return [part for part in response.message.parts if isinstance(part, TextPart)]
