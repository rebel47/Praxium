"""Universal adapter for OpenAI-compatible chat and embedding endpoints."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
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
    Usage,
)
from praxium.models import EmbeddingRequest, EmbeddingResponse, ModelRequest, ToolDefinition

from ._http import HTTPTransport, HTTPXTransport
from ._mapping import (
    decode_external_call_id,
    encode_external_call_id,
    invalid_response,
    map_finish_reason,
    parse_object,
    parse_response_json,
    translate_transport_error,
    usage_from_openai,
)


class OpenAICompatibleProvider:
    """Connect Praxium to any endpoint implementing OpenAI Chat Completions.

    The model identifier is passed through unchanged. No provider or model
    allowlist is used. Authentication and endpoint paths are configurable so the
    same adapter works with hosted routers and local inference servers.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_key_provider: Callable[[], str | Awaitable[str]] | None = None,
        require_api_key: bool = True,
        chat_path: str = "/chat/completions",
        embeddings_path: str | None = "/embeddings",
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, str] | None = None,
        default_parameters: Mapping[str, Any] | None = None,
        max_tokens_field: str = "max_tokens",
        strict_tools: bool = False,
        include_stream_usage: bool = True,
        timeout_seconds: float = 60,
        transport: HTTPTransport | None = None,
    ) -> None:
        if not provider_name.strip():
            raise ValueError("provider_name cannot be empty")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        self._name = provider_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._api_key_provider = api_key_provider
        self._require_api_key = require_api_key
        self._chat_path = _path(chat_path)
        self._embeddings_path = _path(embeddings_path) if embeddings_path else None
        self._auth_header = auth_header
        self._auth_scheme = auth_scheme
        self._headers = dict(headers or {})
        self._query = dict(query or {})
        self._default_parameters = dict(default_parameters or {})
        self._max_tokens_field = max_tokens_field
        self._strict_tools = strict_tools
        self._include_stream_usage = include_stream_usage
        self._timeout_seconds = timeout_seconds
        self._transport = transport or HTTPXTransport()

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider_name={self.name!r}, base_url={self._base_url!r})"

    async def complete(self, request: ModelRequest) -> Response:
        payload = self._completion_payload(request, stream=False)
        try:
            response = await self._transport.post_json(
                self._url(self._chat_path),
                headers=await self._request_headers(),
                payload=payload,
                query=self._query,
                timeout_seconds=self._timeout(request),
            )
            data = parse_response_json(response, provider=self.name)
        except (ModelProviderError, ConfigurationError):
            raise
        except Exception as exc:
            raise translate_transport_error(self.name, exc) from exc
        return self._parse_completion(data, request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ResponseDelta]:
        payload = self._completion_payload(request, stream=True)
        text_index = 0
        tool_buffers: dict[int, dict[str, str]] = {}
        emitted_tools: set[int] = set()
        usage = Usage()
        finish_reason = None
        try:
            async for event in self._transport.stream_sse(
                self._url(self._chat_path),
                headers=await self._request_headers(),
                payload=payload,
                query=self._query,
                timeout_seconds=self._timeout(request),
            ):
                if event.data.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError as exc:
                    raise invalid_response(self.name, "stream contained invalid JSON") from exc
                if not isinstance(data, dict):
                    raise invalid_response(self.name, "stream chunk must be a JSON object")
                if isinstance(data.get("usage"), Mapping):
                    usage = usage_from_openai(data.get("usage"))
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, Mapping):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield ResponseDelta(index=text_index, text=content)
                        text_index += 1
                    self._update_tool_buffers(tool_buffers, delta.get("tool_calls"))
                if choice.get("finish_reason") is not None:
                    finish_reason = map_finish_reason(choice.get("finish_reason"))
                    for index, call in self._finished_tool_calls(tool_buffers, emitted_tools):
                        emitted_tools.add(index)
                        yield ResponseDelta(index=text_index, tool_call=call)
                        text_index += 1
        except (ModelProviderError, ConfigurationError):
            raise
        except Exception as exc:
            raise translate_transport_error(self.name, exc) from exc

        for index, call in self._finished_tool_calls(tool_buffers, emitted_tools):
            emitted_tools.add(index)
            yield ResponseDelta(index=text_index, tool_call=call)
            text_index += 1
        yield ResponseDelta(
            index=text_index,
            finish_reason=finish_reason or map_finish_reason("stop"),
            usage=usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if self._embeddings_path is None:
            from praxium.core import UnsupportedFeatureError

            raise UnsupportedFeatureError(f"{self.name} adapter does not expose embeddings")
        payload: dict[str, Any] = {
            "model": request.model.name,
            "input": request.inputs,
            "encoding_format": "float",
        }
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions
        try:
            response = await self._transport.post_json(
                self._url(self._embeddings_path),
                headers=await self._request_headers(),
                payload=payload,
                query=self._query,
                timeout_seconds=self._timeout_seconds,
            )
            data = parse_response_json(response, provider=self.name)
        except (ModelProviderError, ConfigurationError):
            raise
        except Exception as exc:
            raise translate_transport_error(self.name, exc) from exc
        raw_embeddings = data.get("data")
        if not isinstance(raw_embeddings, list):
            raise invalid_response(self.name, "embedding response is missing data")
        ordered = sorted(
            (item for item in raw_embeddings if isinstance(item, Mapping)),
            key=lambda item: int(item.get("index", 0)),
        )
        embeddings: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not all(
                isinstance(value, int | float) for value in vector
            ):
                raise invalid_response(self.name, "embedding response contains an invalid vector")
            embeddings.append([float(value) for value in vector])
        if len(embeddings) != len(request.inputs):
            raise invalid_response(self.name, "embedding count does not match input count")
        return EmbeddingResponse(
            embeddings=embeddings,
            model=str(data.get("model") or request.model.name),
            usage=usage_from_openai(data.get("usage")),
        )

    def _completion_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        payload = {**self._default_parameters, **request.model.parameters}
        payload.update(
            {
                "model": request.model.name,
                "messages": openai_messages(request.messages),
                "stream": stream,
            }
        )
        if request.model.temperature is not None:
            payload["temperature"] = request.model.temperature
        if request.model.max_output_tokens is not None:
            payload[self._max_tokens_field] = request.model.max_output_tokens
        if request.tools:
            payload["tools"] = [self._tool_definition(tool) for tool in request.tools]
        if request.output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "praxium_response",
                    "strict": True,
                    "schema": request.output_schema,
                },
            }
        if stream and self._include_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _tool_definition(self, tool: ToolDefinition) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        if self._strict_tools:
            function["strict"] = True
        return {"type": "function", "function": function}

    def _parse_completion(self, data: Mapping[str, Any], request: ModelRequest) -> Response:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise invalid_response(self.name, "completion response is missing choices")
        choice = choices[0]
        raw_message = choice.get("message")
        if not isinstance(raw_message, Mapping):
            raise invalid_response(self.name, "completion response is missing a message")
        parts: list[Any] = []
        content = raw_message.get("content")
        if isinstance(content, str) and content:
            if request.output_schema is not None:
                parts.append(
                    JsonPart(data=parse_object(content, provider=self.name, field="content"))
                )
            else:
                parts.append(TextPart(text=content))
        tool_calls = raw_message.get("tool_calls")
        if isinstance(tool_calls, list):
            for index, raw_call in enumerate(tool_calls):
                parts.append(self._parse_tool_call(raw_call, index))
        if not parts:
            parts.append(TextPart(text=""))
        return Response(
            message=Message(role=Role.ASSISTANT, parts=parts),
            finish_reason=map_finish_reason(choice.get("finish_reason")),
            model=str(data.get("model") or request.model.name),
            usage=usage_from_openai(data.get("usage")),
            metadata={
                "provider": self.name,
                **({"response_id": str(data["id"])} if data.get("id") else {}),
            },
        )

    def _parse_tool_call(self, raw_call: Any, index: int) -> ToolCallPart:
        if not isinstance(raw_call, Mapping):
            raise invalid_response(self.name, "tool call must be an object")
        function = raw_call.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            raise invalid_response(self.name, "tool call is missing a function name")
        external_id = str(raw_call.get("id") or f"{self.name}-{index}")
        return ToolCallPart(
            call_id=encode_external_call_id(external_id),
            tool_name=str(function["name"]),
            arguments=parse_object(
                function.get("arguments", {}), provider=self.name, field="tool arguments"
            ),
        )

    def _update_tool_buffers(self, buffers: dict[int, dict[str, str]], value: Any) -> None:
        if not isinstance(value, list):
            return
        for fallback_index, raw_call in enumerate(value):
            if not isinstance(raw_call, Mapping):
                continue
            index = int(raw_call.get("index", fallback_index))
            buffer = buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if raw_call.get("id"):
                buffer["id"] += str(raw_call["id"])
            function = raw_call.get("function")
            if isinstance(function, Mapping):
                if function.get("name"):
                    buffer["name"] += str(function["name"])
                if function.get("arguments"):
                    buffer["arguments"] += str(function["arguments"])

    def _finished_tool_calls(
        self, buffers: Mapping[int, Mapping[str, str]], emitted: set[int]
    ) -> Sequence[tuple[int, ToolCallPart]]:
        calls: list[tuple[int, ToolCallPart]] = []
        for index in sorted(buffers):
            if index in emitted:
                continue
            buffer = buffers[index]
            if not buffer["name"]:
                continue
            calls.append(
                (
                    index,
                    ToolCallPart(
                        call_id=encode_external_call_id(buffer["id"] or f"{self.name}-{index}"),
                        tool_name=buffer["name"],
                        arguments=parse_object(
                            buffer["arguments"], provider=self.name, field="tool arguments"
                        ),
                    ),
                )
            )
        return calls

    async def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._headers}
        key = self._api_key or (os.getenv(self._api_key_env) if self._api_key_env else None)
        if key is None and self._api_key_provider is not None:
            provided = await asyncio.to_thread(self._api_key_provider)
            key = await provided if inspect.isawaitable(provided) else provided
        if self._require_api_key and not key:
            variable = self._api_key_env or "the configured environment variable"
            raise ConfigurationError(f"{self.name} API key is missing; set {variable}")
        if key:
            value = f"{self._auth_scheme} {key}".strip()
            headers[self._auth_header] = value
        return headers

    def _timeout(self, request: ModelRequest) -> float:
        return request.model.timeout_seconds or self._timeout_seconds

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"


def openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert Praxium messages to OpenAI-compatible chat messages."""

    converted: list[dict[str, Any]] = []
    for message in messages:
        results = [part for part in message.parts if isinstance(part, ToolResultPart)]
        if results:
            for result in results:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": decode_external_call_id(result.call_id),
                        "content": _json_text(result.output),
                    }
                )
            continue
        item: dict[str, Any] = {"role": message.role.value}
        if message.name:
            item["name"] = message.name
        text = "".join(
            _content_text(part) for part in message.parts if not isinstance(part, ToolCallPart)
        )
        calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
        item["content"] = text or (None if calls and message.role == Role.ASSISTANT else "")
        if calls:
            item["tool_calls"] = [
                {
                    "id": decode_external_call_id(call.call_id),
                    "type": "function",
                    "function": {
                        "name": call.tool_name,
                        "arguments": json.dumps(call.arguments, separators=(",", ":")),
                    },
                }
                for call in calls
            ]
        converted.append(item)
    return converted


def _content_text(part: Any) -> str:
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, JsonPart):
        return _json_text(part.data)
    if isinstance(part, ReferencePart):
        label = part.title or part.uri
        return f"[{label}]({part.uri})"
    return ""


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _path(value: str) -> str:
    return value if value.startswith("/") else f"/{value}"
