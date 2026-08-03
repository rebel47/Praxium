from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import pytest

from praxium import Message, Model, ModelProviderError, ModelRequest, Role
from praxium.core import (
    ConfigurationError,
    JsonPart,
    ReferencePart,
    ToolCallPart,
    ToolResultPart,
    UnsupportedFeatureError,
)
from praxium.models import EmbeddingRequest, ToolDefinition
from praxium.providers import (
    BedrockProvider,
    HTTPResponse,
    ProviderFactory,
    SSEEvent,
    VertexAIProvider,
    bedrock_messages,
)


class RecordingTransport:
    def __init__(
        self,
        responses: list[HTTPResponse] | None = None,
        events: list[SSEEvent] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.events = list(events or [])
        self.requests: list[dict[str, Any]] = []

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPResponse:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "query": dict(query),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.responses.pop(0)

    async def stream_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> AsyncIterator[SSEEvent]:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "query": dict(query),
                "timeout_seconds": timeout_seconds,
            }
        )
        for event in self.events:
            yield event


def http_response(payload: Any, status: int = 200) -> HTTPResponse:
    return HTTPResponse(
        status_code=status,
        body=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


def model_request(
    provider: str,
    *,
    name: str = "vendor/model-v7",
    output_schema: dict[str, Any] | None = None,
) -> ModelRequest:
    return ModelRequest(
        model=Model(
            name=name,
            provider=provider,
            temperature=0.3,
            max_output_tokens=321,
            timeout_seconds=17,
        ),
        messages=[Message.text(Role.SYSTEM, "Be exact."), Message.user("Hello")],
        tools=[
            ToolDefinition(
                name="weather",
                description="Read the weather.",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
        output_schema=output_schema,
    )


def openai_answer(text: str = "ready") -> HTTPResponse:
    return http_response(
        {
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
    )


async def test_azure_openai_preset_normalizes_v1_and_renews_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        ProviderFactory.azure_openai(api_key="key")

    issued = 0

    async def token_provider() -> str:
        nonlocal issued
        issued += 1
        return f"token-{issued}"

    transport = RecordingTransport([openai_answer(), openai_answer("again")])
    provider = ProviderFactory.azure_openai(
        endpoint="https://sample-resource.openai.azure.com/",
        token_provider=token_provider,
        transport=transport,
    )
    request = model_request("azure-openai", name="my-production-deployment")
    assert (await provider.complete(request)).message.text_content == "ready"
    assert (await provider.complete(request)).message.text_content == "again"
    assert transport.requests[0]["url"] == (
        "https://sample-resource.openai.azure.com/openai/v1/chat/completions"
    )
    assert transport.requests[0]["payload"]["model"] == "my-production-deployment"
    assert [item["headers"]["Authorization"] for item in transport.requests] == [
        "Bearer token-1",
        "Bearer token-2",
    ]

    already_v1 = ProviderFactory.azure_openai(
        endpoint="https://sample-resource.openai.azure.com/openai/v1",
        api_key="key",
        transport=RecordingTransport(),
    )
    assert "openai/v1/openai/v1" not in repr(already_v1)


@pytest.mark.parametrize(
    ("provider_name", "factory", "expected_url"),
    [
        (
            "groq",
            lambda transport: ProviderFactory.groq(api_key="key", transport=transport),
            "https://api.groq.com/openai/v1/chat/completions",
        ),
        (
            "together",
            lambda transport: ProviderFactory.together(api_key="key", transport=transport),
            "https://api.together.xyz/v1/chat/completions",
        ),
        (
            "openrouter",
            lambda transport: ProviderFactory.openrouter(
                api_key="key",
                site_url="https://praxium.example",
                app_name="Praxium",
                transport=transport,
            ),
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    ],
)
async def test_openai_compatible_named_presets_are_one_line_and_keep_any_model_id(
    provider_name: str,
    factory: Any,
    expected_url: str,
) -> None:
    transport = RecordingTransport([openai_answer()])
    provider = factory(transport)
    result = await provider.complete(model_request(provider_name))
    assert result.message.text_content == "ready"
    assert transport.requests[0]["url"] == expected_url
    assert transport.requests[0]["payload"]["model"] == "vendor/model-v7"
    assert transport.requests[0]["headers"]["Authorization"] == "Bearer key"
    if provider_name == "groq":
        assert transport.requests[0]["payload"]["max_completion_tokens"] == 321
        with pytest.raises(UnsupportedFeatureError):
            await provider.embed(
                EmbeddingRequest(model=Model(name="embed", provider="groq"), inputs=["one"])
            )
    if provider_name == "openrouter":
        assert transport.requests[0]["headers"]["HTTP-Referer"] == "https://praxium.example"
        assert transport.requests[0]["headers"]["X-OpenRouter-Title"] == "Praxium"


async def test_together_and_openrouter_presets_support_embeddings() -> None:
    cases: list[tuple[str, Any, RecordingTransport]] = []
    together_transport = RecordingTransport()
    cases.append(
        (
            "together",
            ProviderFactory.together(api_key="key", transport=together_transport),
            together_transport,
        )
    )
    openrouter_transport = RecordingTransport()
    cases.append(
        (
            "openrouter",
            ProviderFactory.openrouter(api_key="key", transport=openrouter_transport),
            openrouter_transport,
        )
    )
    for name, provider, transport in cases:
        transport.responses.append(http_response({"data": [{"index": 0, "embedding": [0.1, 0.2]}]}))
        result = await provider.embed(
            EmbeddingRequest(model=Model(name="any/embed", provider=name), inputs=["hello"])
        )
        assert result.embeddings == [[0.1, 0.2]]
        assert transport.requests[0]["url"].endswith("/embeddings")


class CloseableEvents(Iterator[dict[str, Any]]):
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = iter(events)
        self.closed = False

    def __iter__(self) -> CloseableEvents:
        return self

    def __next__(self) -> dict[str, Any]:
        return next(self._events)

    def close(self) -> None:
        self.closed = True


class FakeBedrockClient:
    def __init__(
        self,
        *,
        completion: Any = None,
        events: CloseableEvents | None = None,
        error: Exception | None = None,
    ) -> None:
        self.completion = completion
        self.events = events
        self.error = error
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def converse(self, **payload: Any) -> Any:
        self.complete_calls.append(payload)
        if self.error:
            raise self.error
        return self.completion

    def converse_stream(self, **payload: Any) -> Any:
        self.stream_calls.append(payload)
        if self.error:
            raise self.error
        return {"stream": self.events}


async def test_bedrock_native_converse_tools_structured_output_and_usage() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    client = FakeBedrockClient(
        completion={
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": '{"answer":42}'},
                        {
                            "toolUse": {
                                "toolUseId": "tooluse-aws-1",
                                "name": "weather",
                                "input": {"city": "Berlin"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 8, "outputTokens": 4, "cacheReadInputTokens": 3},
            "metrics": {"latencyMs": 75},
        }
    )
    provider = BedrockProvider(client=client)
    result = await provider.complete(model_request("bedrock", output_schema=schema))
    assert isinstance(result.message.parts[0], JsonPart)
    assert result.message.parts[0].data == {"answer": 42}
    assert isinstance(result.message.parts[1], ToolCallPart)
    assert result.message.parts[1].arguments == {"city": "Berlin"}
    assert result.finish_reason == "tool_calls"
    assert result.usage.total_tokens == 12
    assert result.usage.cached_input_tokens == 3
    assert result.metadata["latency_ms"] == 75

    payload = client.complete_calls[0]
    assert payload["modelId"] == "vendor/model-v7"
    assert payload["system"] == [{"text": "Be exact."}]
    assert payload["inferenceConfig"] == {"temperature": 0.3, "maxTokens": 321}
    assert payload["toolConfig"]["tools"][0]["toolSpec"]["name"] == "weather"
    json_schema = payload["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
    assert json.loads(json_schema["schema"]) == schema

    with pytest.raises(UnsupportedFeatureError, match="embedding request formats vary"):
        await provider.embed(
            EmbeddingRequest(model=Model(name="embed", provider="bedrock"), inputs=["hello"])
        )


async def test_bedrock_native_stream_maps_text_tools_finish_usage_and_closes() -> None:
    events = CloseableEvents(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "tooluse-2", "name": "weather"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '{"city":'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '"Rome"}'}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "guardrail_intervened"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ]
    )
    client = FakeBedrockClient(events=events)
    provider = ProviderFactory.bedrock(client=client, region_name="eu-central-1")
    deltas = [delta async for delta in provider.stream(model_request("bedrock"))]
    assert deltas[0].text == "Hello"
    assert deltas[1].tool_call is not None
    assert deltas[1].tool_call.arguments == {"city": "Rome"}
    assert deltas[-1].finish_reason == "content_filter"
    assert deltas[-1].usage and deltas[-1].usage.total_tokens == 7
    assert events.closed is True
    assert "eu-central-1" in repr(provider)


def test_bedrock_message_mapping_preserves_calls_results_json_and_references() -> None:
    call = ToolCallPart(tool_name="lookup", arguments={"id": 7})
    system, messages = bedrock_messages(
        [
            Message(
                role=Role.SYSTEM,
                parts=[JsonPart(data={"policy": True}), ReferencePart(uri="urn:policy")],
            ),
            Message(
                role=Role.ASSISTANT,
                parts=[JsonPart(data={"plan": "lookup"}), call],
            ),
            Message(
                role=Role.TOOL,
                parts=[
                    ToolResultPart(
                        call_id=call.call_id,
                        tool_name="lookup",
                        output={"error": "missing"},
                        is_error=True,
                    )
                ],
            ),
        ]
    )
    assert '{"policy":true}' in system
    assert "[urn:policy](urn:policy)" in system
    tool_use = messages[0]["content"][1]["toolUse"]
    assert tool_use["input"] == {"id": 7}
    tool_result = messages[1]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == str(call.call_id)
    assert tool_result["content"] == [{"json": {"error": "missing"}}]
    assert tool_result["status"] == "error"


class FakeAWSError(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "ThrottlingException", "Message": "capacity is busy"},
            "ResponseMetadata": {"HTTPStatusCode": 503},
        }


async def test_bedrock_errors_are_typed_retryable_and_secret_safe() -> None:
    provider = BedrockProvider(client=FakeBedrockClient(error=FakeAWSError()))
    with pytest.raises(ModelProviderError, match="capacity is busy") as caught:
        await provider.complete(model_request("bedrock"))
    assert caught.value.retryable is True
    assert caught.value.status_code == 503
    assert caught.value.context["provider_code"] == "ThrottlingException"
    assert "credential" not in caught.value.to_detail().model_dump_json().lower()

    invalid = BedrockProvider(client=FakeBedrockClient(completion=[]))
    with pytest.raises(ModelProviderError, match="must be an object"):
        await invalid.complete(model_request("bedrock"))


class ValidCredentials:
    valid = True
    token = "adc-token"


async def test_vertex_native_completion_uses_regional_url_and_adc_token() -> None:
    transport = RecordingTransport(
        [
            http_response(
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "vertex"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
                }
            )
        ]
    )
    provider = ProviderFactory.vertex_ai(
        project="my project",
        location="europe-west4",
        credentials=ValidCredentials(),
        transport=transport,
    )
    result = await provider.complete(model_request("vertex-ai", name="gemini-next-preview"))
    assert result.message.text_content == "vertex"
    assert result.usage.total_tokens == 6
    assert transport.requests[0]["url"] == (
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/my%20project/locations/"
        "europe-west4/publishers/google/models/gemini-next-preview:generateContent"
    )
    assert transport.requests[0]["headers"]["Authorization"] == "Bearer adc-token"
    assert transport.requests[0]["query"] == {}
    assert provider.project == "my project"
    assert provider.location == "europe-west4"


async def test_vertex_native_stream_and_embeddings_use_vertex_contract() -> None:
    stream_transport = RecordingTransport(
        events=[
            SSEEvent(
                data=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "Hi"}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 1,
                            "candidatesTokenCount": 1,
                        },
                    }
                )
            )
        ]
    )
    streaming = VertexAIProvider(
        project="project-1",
        location="global",
        access_token="static-token",
        transport=stream_transport,
    )
    deltas = [delta async for delta in streaming.stream(model_request("vertex-ai"))]
    assert deltas[0].text == "Hi"
    assert deltas[-1].usage and deltas[-1].usage.total_tokens == 2
    assert stream_transport.requests[0]["url"].startswith("https://aiplatform.googleapis.com/")
    assert stream_transport.requests[0]["query"] == {"alt": "sse"}

    embedding_transport = RecordingTransport(
        [
            http_response(
                {
                    "predictions": [
                        {
                            "embeddings": {
                                "values": [0.1, 0.2],
                                "statistics": {"token_count": 3},
                            }
                        }
                    ]
                }
            ),
            http_response(
                {
                    "predictions": [
                        {
                            "embeddings": {
                                "values": [0.3, 0.4],
                                "statistics": {"token_count": 2},
                            }
                        }
                    ]
                }
            ),
        ]
    )
    embedding = VertexAIProvider(
        project="project-1",
        access_token="static-token",
        transport=embedding_transport,
    )
    result = await embedding.embed(
        EmbeddingRequest(
            model=Model(name="text-embedding-any", provider="vertex-ai"),
            inputs=["one", "two"],
            dimensions=2,
        )
    )
    assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert result.usage.input_tokens == 5
    assert all(
        request["url"].endswith("/models/text-embedding-any:predict")
        for request in embedding_transport.requests
    )
    assert all(
        request["payload"]["parameters"]["outputDimensionality"] == 2
        for request in embedding_transport.requests
    )


async def test_vertex_validates_configuration_and_embedding_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    with pytest.raises(ConfigurationError, match="GOOGLE_CLOUD_PROJECT"):
        VertexAIProvider(access_token="token")

    provider = VertexAIProvider(
        project="project-1",
        access_token="token",
        transport=RecordingTransport([http_response({"predictions": []})]),
    )
    with pytest.raises(ModelProviderError, match="missing predictions"):
        await provider.embed(
            EmbeddingRequest(model=Model(name="embed", provider="vertex-ai"), inputs=["one"])
        )
