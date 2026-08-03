from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from praxium import (
    Agent,
    AgentRunner,
    Message,
    Model,
    ModelProviderError,
    ModelProviderRegistry,
    ModelRequest,
    Response,
    ResponseDelta,
    Role,
    Tool,
    ToolCallPart,
    ToolResultPart,
)
from praxium.core import ConfigurationError, JsonPart, UnsupportedFeatureError
from praxium.models import EmbeddingRequest, EmbeddingResponse, ToolDefinition
from praxium.providers import (
    AnthropicProvider,
    CustomModelProvider,
    GeminiProvider,
    HTTPResponse,
    OpenAICompatibleProvider,
    ProviderFactory,
    SSEEvent,
    anthropic_messages,
    gemini_contents,
    openai_messages,
)


class FakeTransport:
    def __init__(
        self,
        *,
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
                "timeout": timeout_seconds,
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
                "timeout": timeout_seconds,
            }
        )
        for event in self.events:
            yield event


def response(payload: Any, status: int = 200) -> HTTPResponse:
    return HTTPResponse(
        status_code=status,
        body=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


def request(provider: str = "test", *, output_schema: dict[str, Any] | None = None) -> ModelRequest:
    return ModelRequest(
        model=Model(
            name="model-any-slug",
            provider=provider,
            temperature=0.2,
            max_output_tokens=200,
            timeout_seconds=12,
            parameters={"top_p": 0.9},
        ),
        messages=[Message.text(Role.SYSTEM, "Be concise."), Message.user("Hello")],
        tools=[
            ToolDefinition(
                name="weather",
                description="Get weather.",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
        output_schema=output_schema,
    )


async def test_openai_compatible_completion_tools_agent_loop_and_embeddings() -> None:
    transport = FakeTransport(
        responses=[
            response(
                {
                    "id": "answer-1",
                    "model": "vendor/model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "vendor-call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"left":2,"right":3}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                }
            ),
            response(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "The answer is 5."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 5},
                }
            ),
            response(
                {
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ],
                    "model": "embed-anything",
                    "usage": {"prompt_tokens": 2},
                }
            ),
        ]
    )
    provider = OpenAICompatibleProvider(
        provider_name="vendor",
        base_url="https://vendor.example/v1",
        api_key="secret",
        transport=transport,
    )

    def add(left: int, right: int) -> int:
        return left + right

    runner = AgentRunner(ModelProviderRegistry([provider]))
    result = await runner.run(
        Agent(
            name="calculator",
            instructions="Use tools.",
            model=Model(name="no-allowlist/model:latest", provider="vendor"),
            tools=[Tool.from_callable(add)],
        ),
        "2 + 3",
    )
    assert result.response.text_content == "The answer is 5."
    assert result.usage.total_tokens == 20
    assert result.tool_results[0].output == 5
    second_messages = transport.requests[1]["payload"]["messages"]
    assert second_messages[-1]["tool_call_id"] == "vendor-call-1"
    assert "secret" not in repr(provider)

    embedded = await provider.embed(
        EmbeddingRequest(
            model=Model(name="embed-anything", provider="vendor"),
            inputs=["one", "two"],
            dimensions=2,
        )
    )
    assert embedded.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert transport.requests[-1]["payload"]["dimensions"] == 2


async def test_openai_compatible_structured_output_streaming_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete_transport = FakeTransport(
        responses=[
            response(
                {
                    "choices": [
                        {
                            "message": {"content": '{"answer":42}'},
                            "finish_reason": "length",
                        }
                    ]
                }
            )
        ]
    )
    provider = OpenAICompatibleProvider(
        provider_name="compatible",
        base_url="https://example.test/v1",
        api_key="key",
        max_tokens_field="max_completion_tokens",
        strict_tools=True,
        transport=complete_transport,
    )
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    completed = await provider.complete(request("compatible", output_schema=schema))
    assert isinstance(completed.message.parts[0], JsonPart)
    assert completed.message.parts[0].data == {"answer": 42}
    payload = complete_transport.requests[0]["payload"]
    assert payload["model"] == "model-any-slug"
    assert payload["max_completion_tokens"] == 200
    assert payload["tools"][0]["function"]["strict"] is True
    assert payload["response_format"]["json_schema"]["schema"] == schema

    stream_transport = FakeTransport(
        events=[
            SSEEvent(data='{"choices":[{"delta":{"content":"Hi "}}]}'),
            SSEEvent(
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "stream-call",
                                            "function": {
                                                "name": "weather",
                                                "arguments": '{"city":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            ),
            SSEEvent(
                data='{"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
                '{"arguments":"\\"Berlin\\"}"}}]},"finish_reason":"tool_calls"}]}'
            ),
            SSEEvent(data='{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}'),
            SSEEvent(data="[DONE]"),
        ]
    )
    streaming = OpenAICompatibleProvider(
        provider_name="compatible",
        base_url="https://example.test/v1",
        api_key="key",
        transport=stream_transport,
    )
    deltas = [delta async for delta in streaming.stream(request("compatible"))]
    assert deltas[0].text == "Hi "
    assert deltas[1].tool_call is not None
    assert deltas[1].tool_call.arguments == {"city": "Berlin"}
    assert deltas[-1].finish_reason == "tool_calls"
    assert deltas[-1].usage and deltas[-1].usage.total_tokens == 9

    monkeypatch.delenv("MISSING_VENDOR_KEY", raising=False)
    missing = OpenAICompatibleProvider(
        provider_name="missing",
        base_url="https://example.test/v1",
        api_key_env="MISSING_VENDOR_KEY",
        transport=FakeTransport(),
    )
    with pytest.raises(ConfigurationError, match="MISSING_VENDOR_KEY"):
        await missing.complete(request("missing"))

    failing = OpenAICompatibleProvider(
        provider_name="failing",
        base_url="https://example.test/v1",
        api_key="key",
        transport=FakeTransport(responses=[response({"error": {"message": "bad request"}}, 429)]),
    )
    with pytest.raises(ModelProviderError) as caught:
        await failing.complete(request("failing"))
    assert caught.value.status_code == 429
    assert caught.value.retryable is True
    assert "key" not in caught.value.to_detail().model_dump_json()


def test_openai_message_mapping_preserves_calls_results_json_and_references() -> None:
    call = ToolCallPart(tool_name="lookup", arguments={"id": 3})
    messages = [
        Message(
            role=Role.ASSISTANT,
            parts=[JsonPart(data={"note": True}), call],
        ),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id=call.call_id,
                    tool_name="lookup",
                    output={"value": "ok"},
                )
            ],
        ),
    ]
    converted = openai_messages(messages)
    assert converted[0]["tool_calls"][0]["function"]["arguments"] == '{"id":3}'
    assert converted[1]["content"] == '{"value":"ok"}'


async def test_anthropic_completion_messages_stream_and_unsupported_embeddings() -> None:
    transport = FakeTransport(
        responses=[
            response(
                {
                    "id": "msg_1",
                    "model": "claude-any",
                    "content": [
                        {"type": "text", "text": "Checking."},
                        {
                            "type": "tool_use",
                            "id": "toolu_native",
                            "name": "weather",
                            "input": {"city": "Berlin"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {
                        "input_tokens": 5,
                        "cache_read_input_tokens": 2,
                        "output_tokens": 4,
                    },
                }
            )
        ]
    )
    provider = AnthropicProvider(api_key="secret", transport=transport)
    result = await provider.complete(request("anthropic"))
    assert result.finish_reason == "tool_calls"
    assert result.message.text_content == "Checking."
    call = result.message.parts[1]
    assert isinstance(call, ToolCallPart)
    assert call.arguments == {"city": "Berlin"}
    assert transport.requests[0]["payload"]["system"] == "Be concise."
    assert transport.requests[0]["payload"]["tools"][0]["input_schema"]["type"] == "object"

    system, converted = anthropic_messages(
        [
            Message.text(Role.SYSTEM, "System"),
            result.message,
            Message(
                role=Role.TOOL,
                parts=[
                    ToolResultPart(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        output={"temp": 20},
                    )
                ],
            ),
        ]
    )
    assert system == "System"
    assert converted[-1]["content"][0]["tool_use_id"] == "toolu_native"

    stream = AnthropicProvider(
        api_key="secret",
        transport=FakeTransport(
            events=[
                SSEEvent(
                    event="message_start",
                    data='{"type":"message_start","message":{"usage":{"input_tokens":3}}}',
                ),
                SSEEvent(
                    data='{"type":"content_block_delta","index":0,"delta":'
                    '{"type":"text_delta","text":"Hello"}}'
                ),
                SSEEvent(
                    data='{"type":"content_block_start","index":1,"content_block":'
                    '{"type":"tool_use","id":"toolu_2","name":"weather","input":{}}}'
                ),
                SSEEvent(
                    data='{"type":"content_block_delta","index":1,"delta":'
                    '{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Paris\\"}"}}'
                ),
                SSEEvent(data='{"type":"content_block_stop","index":1}'),
                SSEEvent(
                    data='{"type":"message_delta","delta":{"stop_reason":"tool_use"},'
                    '"usage":{"output_tokens":5}}'
                ),
            ]
        ),
    )
    deltas = [delta async for delta in stream.stream(request("anthropic"))]
    assert deltas[0].text == "Hello"
    assert deltas[1].tool_call and deltas[1].tool_call.arguments == {"city": "Paris"}
    assert deltas[-1].usage and deltas[-1].usage.total_tokens == 8

    with pytest.raises(UnsupportedFeatureError):
        await provider.embed(
            EmbeddingRequest(model=Model(name="none", provider="anthropic"), inputs=["hello"])
        )


async def test_gemini_completion_contents_stream_and_embeddings() -> None:
    transport = FakeTransport(
        responses=[
            response(
                {
                    "modelVersion": "gemini-any",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Let me check."},
                                    {
                                        "functionCall": {
                                            "id": "gem-call",
                                            "name": "weather",
                                            "args": {"city": "Berlin"},
                                        }
                                    },
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3},
                }
            ),
            response(
                {
                    "embeddings": [
                        {"values": [0.1, 0.2]},
                        {"values": [0.3, 0.4]},
                    ]
                }
            ),
        ]
    )
    provider = GeminiProvider(api_key="secret", transport=transport)
    result = await provider.complete(request("gemini"))
    assert result.message.text_content == "Let me check."
    assert isinstance(result.message.parts[1], ToolCallPart)
    assert ":generateContent" in transport.requests[0]["url"]
    assert transport.requests[0]["payload"]["systemInstruction"]["parts"][0]["text"] == (
        "Be concise."
    )

    call = result.message.parts[1]
    assert isinstance(call, ToolCallPart)
    system, contents = gemini_contents(
        [
            Message.text(Role.SYSTEM, "System"),
            result.message,
            Message(
                role=Role.TOOL,
                parts=[
                    ToolResultPart(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        output={"temperature": 20},
                    )
                ],
            ),
        ]
    )
    assert system == "System"
    assert contents[-1]["parts"][0]["functionResponse"]["id"] == "gem-call"

    embedded = await provider.embed(
        EmbeddingRequest(
            model=Model(name="text-embedding/model", provider="gemini"),
            inputs=["a", "b"],
            dimensions=2,
        )
    )
    assert embedded.embeddings[1] == [0.3, 0.4]
    assert ":batchEmbedContents" in transport.requests[-1]["url"]
    assert transport.requests[-1]["payload"]["requests"][0]["outputDimensionality"] == 2

    streaming = GeminiProvider(
        api_key="secret",
        transport=FakeTransport(
            events=[
                SSEEvent(
                    data='{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],'
                    '"usageMetadata":{"promptTokenCount":2}}'
                ),
                SSEEvent(
                    data='{"candidates":[{"content":{"parts":[{"functionCall":'
                    '{"name":"weather","args":{"city":"Rome"}}}]},'
                    '"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,'
                    '"candidatesTokenCount":1}}'
                ),
            ]
        ),
    )
    deltas = [delta async for delta in streaming.stream(request("gemini"))]
    assert deltas[0].text == "Hi"
    assert deltas[1].tool_call and deltas[1].tool_call.tool_name == "weather"
    assert deltas[-1].usage and deltas[-1].usage.total_tokens == 3
    assert deltas[-1].finish_reason == "stop"


async def test_factories_custom_provider_and_model_registry_support_any_slug() -> None:
    presets = [
        ProviderFactory.openai(api_key="x", transport=FakeTransport()),
        ProviderFactory.anthropic(api_key="x", transport=FakeTransport()),
        ProviderFactory.gemini(api_key="x", transport=FakeTransport()),
        ProviderFactory.kimi(api_key="x", transport=FakeTransport()),
        ProviderFactory.glm(api_key="x", transport=FakeTransport()),
        ProviderFactory.ollama(transport=FakeTransport()),
        ProviderFactory.huggingface(api_key="x", transport=FakeTransport()),
        ProviderFactory.azure_openai(
            endpoint="https://resource.openai.azure.com",
            api_key="x",
            transport=FakeTransport(),
        ),
        ProviderFactory.groq(api_key="x", transport=FakeTransport()),
        ProviderFactory.together(api_key="x", transport=FakeTransport()),
        ProviderFactory.openrouter(api_key="x", transport=FakeTransport()),
        ProviderFactory.bedrock(client=object()),
        ProviderFactory.vertex_ai(
            project="project-1",
            access_token="x",
            transport=FakeTransport(),
        ),
        ProviderFactory.openai_compatible(
            provider_name="private-cloud",
            base_url="https://llm.private/v1",
            api_key="x",
            transport=FakeTransport(),
        ),
    ]
    assert [provider.name for provider in presets] == [
        "openai",
        "anthropic",
        "gemini",
        "kimi",
        "glm",
        "ollama",
        "huggingface",
        "azure-openai",
        "groq",
        "together",
        "openrouter",
        "bedrock",
        "vertex-ai",
        "private-cloud",
    ]
    assert Model(name="org/unknown-model:quantized", provider="private-cloud").name == (
        "org/unknown-model:quantized"
    )

    async def complete(_request: ModelRequest) -> Response:
        return Response(message=Message.assistant("custom"))

    def stream(_request: ModelRequest) -> list[ResponseDelta]:
        return [ResponseDelta(index=0, text="custom"), ResponseDelta(index=1, finish_reason="stop")]

    def embed(request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[[1.0] for _ in request.inputs], model=request.model.name
        )

    custom = CustomModelProvider(name="proprietary", complete=complete, stream=stream, embed=embed)
    assert (await custom.complete(request("proprietary"))).message.text_content == "custom"
    custom_deltas = [delta async for delta in custom.stream(request("proprietary"))]
    assert custom_deltas[0].text == "custom"
    embedded = await custom.embed(
        EmbeddingRequest(model=Model(name="anything", provider="proprietary"), inputs=["one"])
    )
    assert embedded.embeddings == [[1.0]]
