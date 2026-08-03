"""Native Anthropic Messages API provider."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from praxium.core import (
    ConfigurationError,
    JsonPart,
    Message,
    ModelProviderError,
    ReferencePart,
    Response,
    ResponseDelta,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UnsupportedFeatureError,
    Usage,
)
from praxium.models import EmbeddingRequest, EmbeddingResponse, ModelRequest

from ._http import HTTPTransport, HTTPXTransport
from ._mapping import (
    decode_external_call_id,
    encode_external_call_id,
    invalid_response,
    map_finish_reason,
    parse_object,
    parse_response_json,
    translate_transport_error,
)


class AnthropicProvider:
    """Translate Praxium calls to Anthropic's native Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        beta_features: Sequence[str] = (),
        default_max_tokens: int = 4096,
        timeout_seconds: float = 60,
        headers: Mapping[str, str] | None = None,
        transport: HTTPTransport | None = None,
        provider_name: str = "anthropic",
    ) -> None:
        self._name = provider_name
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
        self._anthropic_version = anthropic_version
        self._beta_features = tuple(beta_features)
        self._default_max_tokens = default_max_tokens
        self._timeout_seconds = timeout_seconds
        self._headers = dict(headers or {})
        self._transport = transport or HTTPXTransport()

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider_name={self.name!r})"

    async def complete(self, request: ModelRequest) -> Response:
        try:
            response = await self._transport.post_json(
                f"{self._base_url}/messages",
                headers=self._request_headers(),
                payload=self._payload(request, stream=False),
                query={},
                timeout_seconds=request.model.timeout_seconds or self._timeout_seconds,
            )
            data = parse_response_json(response, provider=self.name)
        except (ModelProviderError, ConfigurationError):
            raise
        except Exception as exc:
            raise translate_transport_error(self.name, exc) from exc
        return self._parse_completion(data, request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ResponseDelta]:
        output_index = 0
        usage = Usage()
        finish_reason = None
        tool_buffers: dict[int, dict[str, str]] = {}
        try:
            async for event in self._transport.stream_sse(
                f"{self._base_url}/messages",
                headers=self._request_headers(),
                payload=self._payload(request, stream=True),
                query={},
                timeout_seconds=request.model.timeout_seconds or self._timeout_seconds,
            ):
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError as exc:
                    raise invalid_response(self.name, "stream contained invalid JSON") from exc
                if not isinstance(data, Mapping):
                    continue
                event_type = str(data.get("type") or event.event or "")
                if event_type == "error":
                    error = data.get("error")
                    message = error.get("message") if isinstance(error, Mapping) else None
                    raise ModelProviderError(
                        str(message or "Anthropic stream returned an error"),
                        provider=self.name,
                    )
                if event_type == "message_start":
                    message = data.get("message")
                    if isinstance(message, Mapping):
                        usage = _usage(message.get("usage"))
                elif event_type == "content_block_start":
                    index = int(data.get("index", 0))
                    block = data.get("content_block")
                    if isinstance(block, Mapping) and block.get("type") == "tool_use":
                        tool_buffers[index] = {
                            "id": str(block.get("id") or f"anthropic-{index}"),
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        }
                elif event_type == "content_block_delta":
                    index = int(data.get("index", 0))
                    delta = data.get("delta")
                    if not isinstance(delta, Mapping):
                        continue
                    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        yield ResponseDelta(index=output_index, text=str(delta["text"]))
                        output_index += 1
                    elif delta.get("type") == "input_json_delta":
                        buffer = tool_buffers.get(index)
                        if buffer is not None:
                            partial = str(delta.get("partial_json") or "")
                            if partial:
                                if buffer["arguments"] == "{}":
                                    buffer["arguments"] = ""
                                buffer["arguments"] += partial
                elif event_type == "content_block_stop":
                    index = int(data.get("index", 0))
                    buffer = tool_buffers.pop(index, None)
                    if buffer is not None:
                        yield ResponseDelta(
                            index=output_index,
                            tool_call=ToolCallPart(
                                call_id=encode_external_call_id(buffer["id"]),
                                tool_name=buffer["name"],
                                arguments=parse_object(
                                    buffer["arguments"],
                                    provider=self.name,
                                    field="tool arguments",
                                ),
                            ),
                        )
                        output_index += 1
                elif event_type == "message_delta":
                    delta = data.get("delta")
                    if isinstance(delta, Mapping):
                        finish_reason = map_finish_reason(delta.get("stop_reason"))
                    delta_usage = _usage(data.get("usage"))
                    usage = Usage(
                        input_tokens=usage.input_tokens,
                        output_tokens=max(usage.output_tokens, delta_usage.output_tokens),
                        cached_input_tokens=usage.cached_input_tokens,
                    )
        except (ModelProviderError, ConfigurationError):
            raise
        except Exception as exc:
            raise translate_transport_error(self.name, exc) from exc
        yield ResponseDelta(
            index=output_index,
            finish_reason=finish_reason or map_finish_reason("stop"),
            usage=usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        del request
        raise UnsupportedFeatureError(
            "Anthropic does not expose embeddings through the Messages API; "
            "register a separate embedding provider"
        )

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        system, messages = anthropic_messages(request.messages)
        payload = {**request.model.parameters}
        payload.update(
            {
                "model": request.model.name,
                "messages": messages,
                "max_tokens": request.model.max_output_tokens or self._default_max_tokens,
                "stream": stream,
            }
        )
        if system:
            payload["system"] = system
        if request.model.temperature is not None:
            payload["temperature"] = request.model.temperature
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        if request.output_schema is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": request.output_schema}
            }
        return payload

    def _parse_completion(self, data: Mapping[str, Any], request: ModelRequest) -> Response:
        content = data.get("content")
        if not isinstance(content, list):
            raise invalid_response(self.name, "Anthropic response is missing content")
        parts: list[Any] = []
        for index, block in enumerate(content):
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = str(block["text"])
                if request.output_schema is not None:
                    parts.append(
                        JsonPart(data=parse_object(text, provider=self.name, field="content"))
                    )
                else:
                    parts.append(TextPart(text=text))
            elif block.get("type") == "tool_use":
                name = block.get("name")
                if not isinstance(name, str):
                    raise invalid_response(self.name, "Anthropic tool use is missing a name")
                parts.append(
                    ToolCallPart(
                        call_id=encode_external_call_id(
                            str(block.get("id") or f"anthropic-{index}")
                        ),
                        tool_name=name,
                        arguments=parse_object(
                            block.get("input", {}), provider=self.name, field="tool arguments"
                        ),
                    )
                )
        if not parts:
            parts.append(TextPart(text=""))
        return Response(
            message=Message(role=Role.ASSISTANT, parts=parts),
            finish_reason=map_finish_reason(data.get("stop_reason")),
            model=str(data.get("model") or request.model.name),
            usage=_usage(data.get("usage")),
            metadata={
                "provider": self.name,
                **({"response_id": str(data["id"])} if data.get("id") else {}),
            },
        )

    def _request_headers(self) -> dict[str, str]:
        api_key = self._api_key or os.getenv(self._api_key_env)
        if not api_key:
            raise ConfigurationError(f"Anthropic API key is missing; set {self._api_key_env}")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self._anthropic_version,
            **self._headers,
        }
        if self._beta_features:
            headers["anthropic-beta"] = ",".join(self._beta_features)
        return headers


def anthropic_messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert messages into Anthropic system text and content blocks."""

    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            system_parts.extend(_block_text(part) for part in message.parts)
            continue
        blocks: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, JsonPart):
                blocks.append({"type": "text", "text": _json_text(part.data)})
            elif isinstance(part, ReferencePart):
                blocks.append({"type": "text", "text": f"[{part.title or part.uri}]({part.uri})"})
            elif isinstance(part, ToolCallPart):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": decode_external_call_id(part.call_id),
                        "name": part.tool_name,
                        "input": part.arguments,
                    }
                )
            elif isinstance(part, ToolResultPart):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": decode_external_call_id(part.call_id),
                        "content": _json_text(part.output),
                        "is_error": part.is_error,
                    }
                )
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        role = "assistant" if message.role == Role.ASSISTANT else "user"
        converted.append({"role": role, "content": blocks})
    return "\n\n".join(part for part in system_parts if part), converted


def _usage(value: Any) -> Usage:
    if not isinstance(value, Mapping):
        return Usage()
    cached = _nonnegative_int(value.get("cache_read_input_tokens"))
    created = _nonnegative_int(value.get("cache_creation_input_tokens"))
    uncached = _nonnegative_int(value.get("input_tokens"))
    return Usage(
        input_tokens=uncached + cached + created,
        output_tokens=_nonnegative_int(value.get("output_tokens")),
        cached_input_tokens=cached,
    )


def _nonnegative_int(value: Any) -> int:
    return int(value) if isinstance(value, int | float) and value >= 0 else 0


def _block_text(part: Any) -> str:
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, JsonPart):
        return _json_text(part.data)
    if isinstance(part, ReferencePart):
        return f"[{part.title or part.uri}]({part.uri})"
    return ""


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
