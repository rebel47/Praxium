from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import pytest

from praxium import Message, Model, ModelProviderError, ModelRequest, Response, ResponseDelta, Role
from praxium.core import JsonPart, ReferencePart, UnsupportedFeatureError
from praxium.models import EmbeddingRequest, EmbeddingResponse
from praxium.providers import (
    AnthropicProvider,
    CustomModelProvider,
    GeminiProvider,
    HTTPResponse,
    HTTPXTransport,
    OpenAICompatibleProvider,
    SSEEvent,
    anthropic_messages,
    gemini_contents,
    openai_messages,
)
from praxium.providers._http import HTTPStatusError, HTTPTransportError
from praxium.providers._mapping import (
    decode_external_call_id,
    encode_external_call_id,
    map_finish_reason,
    parse_object,
    parse_response_json,
    status_error,
    translate_transport_error,
    usage_from_openai,
)


class EdgeTransport:
    def __init__(
        self,
        *,
        buffered: HTTPResponse | None = None,
        events: list[SSEEvent] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.buffered = buffered
        self.events = events or []
        self.error = error

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPResponse:
        del url, headers, payload, query, timeout_seconds
        if self.error:
            raise self.error
        if self.buffered is None:
            raise AssertionError("no buffered response")
        return self.buffered

    async def stream_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> AsyncIterator[SSEEvent]:
        del url, headers, payload, query, timeout_seconds
        if self.error:
            raise self.error
        for event in self.events:
            yield event


def http_response(value: Any, status: int = 200, *, raw: bool = False) -> HTTPResponse:
    body = str(value).encode() if raw else json.dumps(value).encode()
    return HTTPResponse(status_code=status, body=body, headers={})


def bare_request(provider: str, *, output_schema: dict[str, Any] | None = None) -> ModelRequest:
    return ModelRequest(
        model=Model(name="any/model", provider=provider),
        messages=[Message.user("hello")],
        output_schema=output_schema,
    )


async def test_httpx_transport_buffers_parses_sse_and_translates_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-test"] == "yes"
        if request.url.path.endswith("/stream"):
            return httpx.Response(
                200,
                text=(
                    'event: message\ndata: {"one":1}\n\ndata: first\ndata: second\n\ndata: final'
                ),
            )
        return httpx.Response(200, json={"ok": True})

    transport = HTTPXTransport(transport=httpx.MockTransport(handler))
    buffered = await transport.post_json(
        "https://example.test/json",
        headers={"x-test": "yes"},
        payload={"input": "hello"},
        query={"version": "1"},
        timeout_seconds=2,
    )
    assert buffered.json() == {"ok": True}
    assert '"ok":true' in buffered.text.replace(" ", "").lower()
    events = [
        event
        async for event in transport.stream_sse(
            "https://example.test/stream",
            headers={"x-test": "yes"},
            payload={},
            query={},
            timeout_seconds=2,
        )
    ]
    assert events == [
        SSEEvent(event="message", data='{"one":1}'),
        SSEEvent(data="first\nsecond"),
        SSEEvent(data="final"),
    ]

    status_transport = HTTPXTransport(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503, json={"error": "down"}))
    )
    with pytest.raises(HTTPStatusError) as status_caught:
        _ = [
            event
            async for event in status_transport.stream_sse(
                "https://example.test/stream",
                headers={},
                payload={},
                query={},
                timeout_seconds=2,
            )
        ]
    assert status_caught.value.retryable is True

    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    timeout_transport = HTTPXTransport(transport=httpx.MockTransport(timeout))
    with pytest.raises(HTTPTransportError, match="timed out"):
        await timeout_transport.post_json(
            "https://example.test/json",
            headers={},
            payload={},
            query={},
            timeout_seconds=1,
        )
    with pytest.raises(HTTPTransportError, match="timed out"):
        _ = [
            event
            async for event in timeout_transport.stream_sse(
                "https://example.test/stream",
                headers={},
                payload={},
                query={},
                timeout_seconds=1,
            )
        ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("length", "length"),
        ("MAX_TOKENS", "length"),
        ("tool_use", "tool_calls"),
        ("safety", "content_filter"),
        ("network_error", "error"),
        ("unknown", "stop"),
    ],
)
def test_mapping_helpers_cover_protocol_variations(raw: str, expected: str) -> None:
    assert map_finish_reason(raw) == expected


def test_mapping_helpers_validate_json_usage_ids_and_safe_errors() -> None:
    external = encode_external_call_id("toolu_123")
    assert decode_external_call_id(external) == "toolu_123"
    assert decode_external_call_id("call_local") == "call_local"
    assert parse_object({"a": 1}, provider="p", field="f") == {"a": 1}
    assert parse_object("", provider="p", field="f") == {}
    assert parse_object('{"a":1}', provider="p", field="f") == {"a": 1}
    for invalid in ([], "[]", "{"):
        with pytest.raises(ModelProviderError):
            parse_object(invalid, provider="p", field="f")

    usage = usage_from_openai(
        {
            "input_tokens": 4,
            "output_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        }
    )
    assert usage.total_tokens == 7
    assert usage.cached_input_tokens == 2
    assert usage.reasoning_tokens == 1
    assert usage_from_openai(None).total_tokens == 0

    assert parse_response_json(http_response({"ok": True}), provider="p") == {"ok": True}
    with pytest.raises(ModelProviderError, match="invalid JSON"):
        parse_response_json(http_response("not-json", raw=True), provider="p")
    with pytest.raises(ModelProviderError, match="JSON object"):
        parse_response_json(http_response([1]), provider="p")

    json_error = status_error(
        "p", http_response({"error": {"message": "safe message"}}, status=400)
    )
    assert str(json_error) == "safe message"
    assert json_error.retryable is False
    assert str(status_error("p", http_response("gateway", 502, raw=True))) == (
        "model provider returned HTTP 502"
    )
    translated = translate_transport_error("p", HTTPTransportError("network", retryable=True))
    assert translated.retryable is True
    translated_status = translate_transport_error(
        "p", HTTPStatusError(http_response({"error": "busy"}, 429))
    )
    assert translated_status.status_code == 429
    assert str(translate_transport_error("p", ValueError("secret details"))) == (
        "model provider request failed"
    )


async def test_openai_compatible_edge_failures_and_disabled_embedding() -> None:
    with pytest.raises(ValueError, match="provider_name"):
        OpenAICompatibleProvider(provider_name=" ", base_url="https://x", api_key="x")
    with pytest.raises(ValueError, match="HTTP"):
        OpenAICompatibleProvider(provider_name="x", base_url="file:///x", api_key="x")

    invalid_choices = OpenAICompatibleProvider(
        provider_name="x",
        base_url="https://x/v1",
        api_key="x",
        transport=EdgeTransport(buffered=http_response({"choices": []})),
    )
    with pytest.raises(ModelProviderError, match="choices"):
        await invalid_choices.complete(bare_request("x"))

    invalid_stream = OpenAICompatibleProvider(
        provider_name="x",
        base_url="https://x/v1",
        api_key="x",
        transport=EdgeTransport(events=[SSEEvent(data="not-json")]),
    )
    with pytest.raises(ModelProviderError, match="invalid JSON"):
        _ = [delta async for delta in invalid_stream.stream(bare_request("x"))]

    disabled = OpenAICompatibleProvider(
        provider_name="chat-only",
        base_url="https://x/v1",
        api_key="x",
        embeddings_path=None,
    )
    with pytest.raises(UnsupportedFeatureError):
        await disabled.embed(
            EmbeddingRequest(model=Model(name="none", provider="chat-only"), inputs=["hello"])
        )

    bad_vector = OpenAICompatibleProvider(
        provider_name="x",
        base_url="https://x/v1",
        api_key="x",
        transport=EdgeTransport(
            buffered=http_response({"data": [{"index": 0, "embedding": ["bad"]}]})
        ),
    )
    with pytest.raises(ModelProviderError, match="invalid vector"):
        await bad_vector.embed(
            EmbeddingRequest(model=Model(name="embed", provider="x"), inputs=["hello"])
        )


def test_all_message_mappers_handle_json_references_and_error_results() -> None:
    message = Message(
        role=Role.USER,
        parts=[
            JsonPart(data={"id": 1}),
            ReferencePart(uri="https://example.test", title="source"),
        ],
    )
    assert '{"id":1}' in openai_messages([message])[0]["content"]
    assert (
        "[source](https://example.test)"
        in anthropic_messages([message])[1][0]["content"][1]["text"]
    )
    assert "[source](https://example.test)" in gemini_contents([message])[1][0]["parts"][1]["text"]


async def test_anthropic_structured_missing_key_invalid_response_and_stream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    provider = AnthropicProvider(
        api_key="x",
        transport=EdgeTransport(
            buffered=http_response(
                {
                    "content": [{"type": "text", "text": '{"answer":1}'}],
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            )
        ),
    )
    result = await provider.complete(bare_request("anthropic", output_schema=schema))
    assert isinstance(result.message.parts[0], JsonPart)
    assert result.finish_reason == "length"

    monkeypatch.delenv("NO_ANTHROPIC_KEY", raising=False)
    missing = AnthropicProvider(
        api_key_env="NO_ANTHROPIC_KEY",
        transport=EdgeTransport(buffered=http_response({})),
    )
    with pytest.raises(Exception, match="NO_ANTHROPIC_KEY"):
        await missing.complete(bare_request("anthropic"))

    invalid = AnthropicProvider(
        api_key="x", transport=EdgeTransport(buffered=http_response({"content": "wrong"}))
    )
    with pytest.raises(ModelProviderError, match="content"):
        await invalid.complete(bare_request("anthropic"))

    stream_error = AnthropicProvider(
        api_key="x",
        transport=EdgeTransport(
            events=[SSEEvent(data='{"type":"error","error":{"message":"overloaded"}}')]
        ),
    )
    with pytest.raises(ModelProviderError, match="overloaded"):
        _ = [delta async for delta in stream_error.stream(bare_request("anthropic"))]


async def test_gemini_structured_blocked_missing_key_invalid_and_stream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    structured = GeminiProvider(
        api_key="x",
        transport=EdgeTransport(
            buffered=http_response(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": '{"answer":2}'}]},
                            "finishReason": "MAX_TOKENS",
                        }
                    ]
                }
            )
        ),
    )
    result = await structured.complete(bare_request("gemini", output_schema=schema))
    assert isinstance(result.message.parts[0], JsonPart)
    assert result.finish_reason == "length"

    blocked = GeminiProvider(
        api_key="x",
        transport=EdgeTransport(
            buffered=http_response({"promptFeedback": {"blockReason": "SAFETY"}})
        ),
    )
    blocked_result = await blocked.complete(bare_request("gemini"))
    assert blocked_result.finish_reason == "content_filter"

    monkeypatch.delenv("NO_GEMINI_KEY", raising=False)
    missing = GeminiProvider(
        api_key_env="NO_GEMINI_KEY", transport=EdgeTransport(buffered=http_response({}))
    )
    with pytest.raises(Exception, match="NO_GEMINI_KEY"):
        await missing.complete(bare_request("gemini"))

    invalid = GeminiProvider(
        api_key="x",
        transport=EdgeTransport(buffered=http_response({"candidates": [{}]})),
    )
    with pytest.raises(ModelProviderError, match="content parts"):
        await invalid.complete(bare_request("gemini"))

    stream = GeminiProvider(
        api_key="x", transport=EdgeTransport(events=[SSEEvent(data="not-json")])
    )
    with pytest.raises(ModelProviderError, match="invalid JSON"):
        _ = [delta async for delta in stream.stream(bare_request("gemini"))]


async def test_custom_provider_fallback_async_stream_and_contract_errors() -> None:
    provider = CustomModelProvider(
        name="fallback", complete=lambda _request: Response(message=Message.assistant("answer"))
    )
    deltas = [delta async for delta in provider.stream(bare_request("fallback"))]
    assert [delta.text for delta in deltas] == ["answer", ""]
    with pytest.raises(UnsupportedFeatureError):
        await provider.embed(
            EmbeddingRequest(model=Model(name="x", provider="fallback"), inputs=["x"])
        )

    async def async_stream(_request: ModelRequest) -> AsyncIterator[ResponseDelta]:
        yield ResponseDelta(index=0, text="async")

    async_provider = CustomModelProvider(
        name="async",
        complete=lambda _request: Response(message=Message.assistant("ok")),
        stream=async_stream,
    )
    assert [delta.text async for delta in async_provider.stream(bare_request("async"))] == ["async"]

    invalid_complete = CustomModelProvider(name="bad", complete=lambda _request: "wrong")
    with pytest.raises(TypeError, match="Response"):
        await invalid_complete.complete(bare_request("bad"))
    invalid_stream = CustomModelProvider(
        name="bad-stream",
        complete=lambda _request: Response(message=Message.assistant("ok")),
        stream=lambda _request: ["wrong"],
    )
    with pytest.raises(TypeError, match="ResponseDelta"):
        _ = [delta async for delta in invalid_stream.stream(bare_request("bad-stream"))]
    invalid_embed = CustomModelProvider(
        name="bad-embed",
        complete=lambda _request: Response(message=Message.assistant("ok")),
        embed=lambda _request: "wrong",
    )
    with pytest.raises(TypeError, match="EmbeddingResponse"):
        await invalid_embed.embed(
            EmbeddingRequest(model=Model(name="x", provider="bad-embed"), inputs=["x"])
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        CustomModelProvider(name=" ", complete=lambda _request: None)

    factory_custom = CustomModelProvider(
        name="embed",
        complete=lambda _request: Response(message=Message.assistant("ok")),
        embed=lambda request: EmbeddingResponse(
            embeddings=[[1.0] for _ in request.inputs], model=request.model.name
        ),
    )
    assert (
        await factory_custom.embed(
            EmbeddingRequest(model=Model(name="x", provider="embed"), inputs=["x"])
        )
    ).embeddings == [[1.0]]
