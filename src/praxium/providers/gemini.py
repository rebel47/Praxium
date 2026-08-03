"""Native Google Gemini API provider."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

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


class GeminiProvider:
    """Translate Praxium calls to Gemini generateContent APIs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        access_token: str | None = None,
        access_token_provider: Callable[[], str | Awaitable[str]] | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 60,
        headers: Mapping[str, str] | None = None,
        transport: HTTPTransport | None = None,
        provider_name: str = "gemini",
    ) -> None:
        self._name = provider_name
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._access_token = access_token
        self._access_token_provider = access_token_provider
        self._base_url = base_url.rstrip("/")
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
                self._model_url(request.model.name, "generateContent"),
                headers=await self._request_headers(),
                payload=self._payload(request),
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
        finish_reason = None
        usage = Usage()
        try:
            async for event in self._transport.stream_sse(
                self._model_url(request.model.name, "streamGenerateContent"),
                headers=await self._request_headers(),
                payload=self._payload(request),
                query={"alt": "sse"},
                timeout_seconds=request.model.timeout_seconds or self._timeout_seconds,
            ):
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError as exc:
                    raise invalid_response(self.name, "stream contained invalid JSON") from exc
                if not isinstance(data, Mapping):
                    continue
                usage = _usage(data.get("usageMetadata"))
                candidates = data.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    continue
                candidate = candidates[0]
                if not isinstance(candidate, Mapping):
                    continue
                if candidate.get("finishReason"):
                    finish_reason = map_finish_reason(candidate.get("finishReason"))
                content = candidate.get("content")
                parts = content.get("parts") if isinstance(content, Mapping) else None
                if not isinstance(parts, list):
                    continue
                for part_index, part in enumerate(parts):
                    if not isinstance(part, Mapping):
                        continue
                    if isinstance(part.get("text"), str) and part["text"]:
                        yield ResponseDelta(index=output_index, text=str(part["text"]))
                        output_index += 1
                    function_call = part.get("functionCall")
                    if isinstance(function_call, Mapping):
                        name = function_call.get("name")
                        if not isinstance(name, str):
                            raise invalid_response(self.name, "Gemini function call has no name")
                        external_id = str(
                            function_call.get("id") or f"gemini-{output_index}-{part_index}"
                        )
                        yield ResponseDelta(
                            index=output_index,
                            tool_call=ToolCallPart(
                                call_id=encode_external_call_id(external_id),
                                tool_name=name,
                                arguments=parse_object(
                                    function_call.get("args", {}),
                                    provider=self.name,
                                    field="tool arguments",
                                ),
                            ),
                        )
                        output_index += 1
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
        model_path = self._model_path(request.model.name)
        embedding_requests: list[dict[str, Any]] = []
        for text in request.inputs:
            item: dict[str, Any] = {
                "model": model_path,
                "content": {"parts": [{"text": text}]},
            }
            if request.dimensions is not None:
                item["outputDimensionality"] = request.dimensions
            embedding_requests.append(item)
        try:
            response = await self._transport.post_json(
                self._model_url(request.model.name, "batchEmbedContents"),
                headers=await self._request_headers(),
                payload={"requests": embedding_requests},
                query={},
                timeout_seconds=request.model.timeout_seconds or self._timeout_seconds,
            )
            data = parse_response_json(response, provider=self.name)
        except (ModelProviderError, ConfigurationError):
            raise
        except Exception as exc:
            raise translate_transport_error(self.name, exc) from exc
        raw_embeddings = data.get("embeddings")
        if not isinstance(raw_embeddings, list):
            raise invalid_response(self.name, "Gemini embedding response is missing embeddings")
        embeddings: list[list[float]] = []
        for item in raw_embeddings:
            values = item.get("values") if isinstance(item, Mapping) else None
            if not isinstance(values, list) or not all(
                isinstance(value, int | float) for value in values
            ):
                raise invalid_response(self.name, "Gemini returned an invalid embedding")
            embeddings.append([float(value) for value in values])
        if len(embeddings) != len(request.inputs):
            raise invalid_response(self.name, "embedding count does not match input count")
        return EmbeddingResponse(embeddings=embeddings, model=request.model.name)

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        system, contents = gemini_contents(request.messages)
        payload = {**request.model.parameters, "contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        generation: dict[str, Any] = {}
        if request.model.temperature is not None:
            generation["temperature"] = request.model.temperature
        if request.model.max_output_tokens is not None:
            generation["maxOutputTokens"] = request.model.max_output_tokens
        if request.output_schema is not None:
            generation.update(
                {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": request.output_schema,
                }
            )
        if generation:
            existing = payload.get("generationConfig")
            payload["generationConfig"] = {
                **(dict(existing) if isinstance(existing, Mapping) else {}),
                **generation,
            }
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parametersJsonSchema": tool.input_schema,
                        }
                        for tool in request.tools
                    ]
                }
            ]
        return payload

    def _parse_completion(self, data: Mapping[str, Any], request: ModelRequest) -> Response:
        candidates = data.get("candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not isinstance(candidates[0], Mapping)
        ):
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, Mapping) and prompt_feedback.get("blockReason"):
                return Response(
                    message=Message.assistant(""),
                    finish_reason=map_finish_reason("blocked"),
                    model=request.model.name,
                    usage=_usage(data.get("usageMetadata")),
                    metadata={"provider": self.name},
                )
            raise invalid_response(self.name, "Gemini response is missing candidates")
        candidate = candidates[0]
        content = candidate.get("content")
        raw_parts = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(raw_parts, list):
            raise invalid_response(self.name, "Gemini response is missing content parts")
        parts: list[Any] = []
        for index, raw_part in enumerate(raw_parts):
            if not isinstance(raw_part, Mapping):
                continue
            if isinstance(raw_part.get("text"), str):
                text = str(raw_part["text"])
                if request.output_schema is not None:
                    parts.append(
                        JsonPart(data=parse_object(text, provider=self.name, field="content"))
                    )
                else:
                    parts.append(TextPart(text=text))
            function_call = raw_part.get("functionCall")
            if isinstance(function_call, Mapping):
                name = function_call.get("name")
                if not isinstance(name, str):
                    raise invalid_response(self.name, "Gemini function call has no name")
                parts.append(
                    ToolCallPart(
                        call_id=encode_external_call_id(
                            str(function_call.get("id") or f"gemini-{index}")
                        ),
                        tool_name=name,
                        arguments=parse_object(
                            function_call.get("args", {}),
                            provider=self.name,
                            field="tool arguments",
                        ),
                    )
                )
        if not parts:
            parts.append(TextPart(text=""))
        return Response(
            message=Message(role=Role.ASSISTANT, parts=parts),
            finish_reason=map_finish_reason(candidate.get("finishReason")),
            model=str(data.get("modelVersion") or request.model.name),
            usage=_usage(data.get("usageMetadata")),
            metadata={"provider": self.name},
        )

    async def _request_headers(self) -> dict[str, str]:
        token = self._access_token
        if token is None and self._access_token_provider is not None:
            provided = await asyncio.to_thread(self._access_token_provider)
            token = await provided if inspect.isawaitable(provided) else provided
        if token:
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                **self._headers,
            }
        api_key = self._api_key or os.getenv(self._api_key_env)
        if not api_key:
            raise ConfigurationError(f"Gemini API key is missing; set {self._api_key_env}")
        return {"Content-Type": "application/json", "x-goog-api-key": api_key, **self._headers}

    def _model_url(self, model: str, method: str) -> str:
        model_name = model.removeprefix("models/")
        return f"{self._base_url}/models/{quote(model_name, safe='-._~')}:{method}"

    def _model_path(self, model: str) -> str:
        return f"models/{model.removeprefix('models/')}"


def gemini_contents(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert Praxium messages into Gemini system text and contents."""

    system: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            system.extend(_part_text(part) for part in message.parts)
            continue
        parts: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                parts.append({"text": part.text})
            elif isinstance(part, JsonPart):
                parts.append({"text": _json_text(part.data)})
            elif isinstance(part, ReferencePart):
                parts.append({"text": f"[{part.title or part.uri}]({part.uri})"})
            elif isinstance(part, ToolCallPart):
                parts.append(
                    {
                        "functionCall": {
                            "id": decode_external_call_id(part.call_id),
                            "name": part.tool_name,
                            "args": part.arguments,
                        }
                    }
                )
            elif isinstance(part, ToolResultPart):
                response = part.output if isinstance(part.output, dict) else {"result": part.output}
                if part.is_error:
                    response = {"error": response}
                parts.append(
                    {
                        "functionResponse": {
                            "id": decode_external_call_id(part.call_id),
                            "name": part.tool_name,
                            "response": response,
                        }
                    }
                )
        if not parts:
            parts.append({"text": ""})
        contents.append(
            {"role": "model" if message.role == Role.ASSISTANT else "user", "parts": parts}
        )
    return "\n\n".join(part for part in system if part), contents


def _usage(value: Any) -> Usage:
    if not isinstance(value, Mapping):
        return Usage()
    return Usage(
        input_tokens=_nonnegative_int(value.get("promptTokenCount")),
        output_tokens=_nonnegative_int(value.get("candidatesTokenCount")),
        cached_input_tokens=_nonnegative_int(value.get("cachedContentTokenCount")),
        reasoning_tokens=_nonnegative_int(value.get("thoughtsTokenCount")),
    )


def _nonnegative_int(value: Any) -> int:
    return int(value) if isinstance(value, int | float) and value >= 0 else 0


def _part_text(part: Any) -> str:
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
