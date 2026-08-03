"""Native Amazon Bedrock Converse provider using the AWS credential chain."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from praxium.core import (
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

from ._mapping import (
    decode_external_call_id,
    encode_external_call_id,
    invalid_response,
    map_finish_reason,
    parse_object,
)

_STREAM_END = object()
_STREAM_ERRORS = {
    "internalServerException",
    "modelStreamErrorException",
    "serviceUnavailableException",
    "throttlingException",
    "validationException",
}


class BedrockProvider:
    """Run message-capable Bedrock models through the native Converse API.

    Authentication, refresh, profiles, roles, and SigV4 signing are delegated to
    Boto3's standard AWS credential chain. Model IDs and inference-profile ARNs are
    passed through unchanged.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any = None,
        provider_name: str = "bedrock",
    ) -> None:
        self._name = provider_name
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider_name={self.name!r}, region_name={self._region_name!r})"
        )

    async def complete(self, request: ModelRequest) -> Response:
        payload = self._payload(request)
        try:
            data = await asyncio.to_thread(self._converse, payload)
        except ModelProviderError:
            raise
        except Exception as exc:
            raise _aws_error(self.name, exc) from exc
        if not isinstance(data, Mapping):
            raise invalid_response(self.name, "Bedrock response must be an object")
        return self._parse_completion(data, request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ResponseDelta]:
        payload = self._payload(request)
        try:
            response = await asyncio.to_thread(self._converse_stream, payload)
        except Exception as exc:
            raise _aws_error(self.name, exc) from exc
        raw_stream = response.get("stream") if isinstance(response, Mapping) else None
        if raw_stream is None:
            raise invalid_response(self.name, "Bedrock stream response is missing stream")
        iterator = iter(raw_stream)
        output_index = 0
        tool_buffers: dict[int, dict[str, str]] = {}
        finish_reason = None
        usage = Usage()
        try:
            while True:
                try:
                    event = await asyncio.to_thread(_next_or_end, iterator)
                except Exception as exc:
                    raise _aws_error(self.name, exc) from exc
                if event is _STREAM_END:
                    break
                if not isinstance(event, Mapping):
                    continue
                error_key = next((key for key in _STREAM_ERRORS if key in event), None)
                if error_key:
                    error = event.get(error_key)
                    message = error.get("message") if isinstance(error, Mapping) else None
                    raise ModelProviderError(
                        str(message or f"Bedrock stream failed with {error_key}"),
                        provider=self.name,
                        retryable=error_key
                        in {
                            "internalServerException",
                            "modelStreamErrorException",
                            "serviceUnavailableException",
                            "throttlingException",
                        },
                    )
                start_event = event.get("contentBlockStart")
                if isinstance(start_event, Mapping):
                    index = int(start_event.get("contentBlockIndex", 0))
                    start = start_event.get("start")
                    tool_use = start.get("toolUse") if isinstance(start, Mapping) else None
                    if isinstance(tool_use, Mapping):
                        tool_buffers[index] = {
                            "id": str(tool_use.get("toolUseId") or f"bedrock-{index}"),
                            "name": str(tool_use.get("name") or ""),
                            "arguments": "",
                        }
                delta_event = event.get("contentBlockDelta")
                if isinstance(delta_event, Mapping):
                    index = int(delta_event.get("contentBlockIndex", 0))
                    delta = delta_event.get("delta")
                    if isinstance(delta, Mapping) and isinstance(delta.get("text"), str):
                        yield ResponseDelta(index=output_index, text=str(delta["text"]))
                        output_index += 1
                    tool_use = delta.get("toolUse") if isinstance(delta, Mapping) else None
                    if isinstance(tool_use, Mapping) and index in tool_buffers:
                        tool_buffers[index]["arguments"] += str(tool_use.get("input") or "")
                stop_event = event.get("contentBlockStop")
                if isinstance(stop_event, Mapping):
                    index = int(stop_event.get("contentBlockIndex", 0))
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
                message_stop = event.get("messageStop")
                if isinstance(message_stop, Mapping):
                    finish_reason = map_finish_reason(message_stop.get("stopReason"))
                metadata = event.get("metadata")
                if isinstance(metadata, Mapping):
                    usage = _bedrock_usage(metadata.get("usage"))
        finally:
            close = getattr(raw_stream, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        yield ResponseDelta(
            index=output_index,
            finish_reason=finish_reason or map_finish_reason("stop"),
            usage=usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        del request
        raise UnsupportedFeatureError(
            "Bedrock embedding request formats vary by model; register a dedicated "
            "embedding provider or a CustomModelProvider handler"
        )

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        system, messages = bedrock_messages(request.messages)
        payload = {**request.model.parameters}
        payload.update({"modelId": request.model.name, "messages": messages})
        if system:
            payload["system"] = [{"text": system}]
        inference = payload.get("inferenceConfig")
        inference_config = dict(inference) if isinstance(inference, Mapping) else {}
        if request.model.temperature is not None:
            inference_config["temperature"] = request.model.temperature
        if request.model.max_output_tokens is not None:
            inference_config["maxTokens"] = request.model.max_output_tokens
        if inference_config:
            payload["inferenceConfig"] = inference_config
        if request.tools:
            existing = payload.get("toolConfig")
            tool_config = dict(existing) if isinstance(existing, Mapping) else {}
            tool_config["tools"] = [
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
                for tool in request.tools
            ]
            payload["toolConfig"] = tool_config
        if request.output_schema is not None:
            payload["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": "praxium_response",
                            "schema": json.dumps(request.output_schema, separators=(",", ":")),
                        }
                    },
                }
            }
        return payload

    def _parse_completion(self, data: Mapping[str, Any], request: ModelRequest) -> Response:
        output = data.get("output")
        message = output.get("message") if isinstance(output, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            raise invalid_response(self.name, "Bedrock response is missing output message content")
        parts: list[Any] = []
        for index, block in enumerate(content):
            if not isinstance(block, Mapping):
                continue
            if isinstance(block.get("text"), str):
                text = str(block["text"])
                if request.output_schema is not None:
                    parts.append(
                        JsonPart(data=parse_object(text, provider=self.name, field="content"))
                    )
                else:
                    parts.append(TextPart(text=text))
            tool_use = block.get("toolUse")
            if isinstance(tool_use, Mapping):
                name = tool_use.get("name")
                if not isinstance(name, str):
                    raise invalid_response(self.name, "Bedrock tool use is missing a name")
                parts.append(
                    ToolCallPart(
                        call_id=encode_external_call_id(
                            str(tool_use.get("toolUseId") or f"bedrock-{index}")
                        ),
                        tool_name=name,
                        arguments=parse_object(
                            tool_use.get("input", {}),
                            provider=self.name,
                            field="tool arguments",
                        ),
                    )
                )
        if not parts:
            parts.append(TextPart(text=""))
        metrics = data.get("metrics")
        metadata: dict[str, Any] = {"provider": self.name}
        if isinstance(metrics, Mapping) and isinstance(metrics.get("latencyMs"), int | float):
            metadata["latency_ms"] = int(metrics["latencyMs"])
        return Response(
            message=Message(role=Role.ASSISTANT, parts=parts),
            finish_reason=map_finish_reason(data.get("stopReason")),
            model=request.model.name,
            usage=_bedrock_usage(data.get("usage")),
            metadata=metadata,
        )

    def _converse(self, payload: Mapping[str, Any]) -> Any:
        return self._get_client().converse(**dict(payload))

    def _converse_stream(self, payload: Mapping[str, Any]) -> Any:
        return self._get_client().converse_stream(**dict(payload))

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    import boto3
                except ImportError as exc:  # pragma: no cover - installation dependent
                    raise RuntimeError(
                        'Amazon Bedrock requires: pip install "praxium[aws]"'
                    ) from exc
                options = {}
                if self._region_name:
                    options["region_name"] = self._region_name
                if self._endpoint_url:
                    options["endpoint_url"] = self._endpoint_url
                self._client = boto3.client("bedrock-runtime", **options)
        return self._client


def bedrock_messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert Praxium messages into Bedrock Converse content blocks."""

    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            system_parts.extend(_part_text(part) for part in message.parts)
            continue
        blocks: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                blocks.append({"text": part.text})
            elif isinstance(part, JsonPart):
                blocks.append({"text": _json_text(part.data)})
            elif isinstance(part, ReferencePart):
                blocks.append({"text": f"[{part.title or part.uri}]({part.uri})"})
            elif isinstance(part, ToolCallPart):
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": decode_external_call_id(part.call_id),
                            "name": part.tool_name,
                            "input": _json_value(part.arguments),
                        }
                    }
                )
            elif isinstance(part, ToolResultPart):
                result_content = (
                    {"text": part.output}
                    if isinstance(part.output, str)
                    else {"json": _json_value(part.output)}
                )
                blocks.append(
                    {
                        "toolResult": {
                            "toolUseId": decode_external_call_id(part.call_id),
                            "content": [result_content],
                            "status": "error" if part.is_error else "success",
                        }
                    }
                )
        if not blocks:
            blocks.append({"text": ""})
        role = "assistant" if message.role == Role.ASSISTANT else "user"
        converted.append({"role": role, "content": blocks})
    return "\n\n".join(value for value in system_parts if value), converted


def _bedrock_usage(value: Any) -> Usage:
    if not isinstance(value, Mapping):
        return Usage()
    return Usage(
        input_tokens=_nonnegative_int(value.get("inputTokens")),
        output_tokens=_nonnegative_int(value.get("outputTokens")),
        cached_input_tokens=_nonnegative_int(value.get("cacheReadInputTokens")),
    )


def _aws_error(provider: str, exc: Exception) -> ModelProviderError:
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, Mapping) else None
    metadata = response.get("ResponseMetadata") if isinstance(response, Mapping) else None
    code = str(error.get("Code") or "") if isinstance(error, Mapping) else ""
    message = error.get("Message") if isinstance(error, Mapping) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    status_code = int(status) if isinstance(status, int | float) else None
    retryable = status_code in {408, 409, 425, 429} if status_code is not None else False
    retryable = retryable or (status_code is not None and status_code >= 500)
    retryable = retryable or code in {
        "InternalServerException",
        "ModelNotReadyException",
        "ModelStreamErrorException",
        "ServiceUnavailableException",
        "ThrottlingException",
    }
    return ModelProviderError(
        str(message or "Amazon Bedrock request failed")[:500],
        provider=provider,
        status_code=status_code,
        retryable=retryable,
        context={"provider_code": code} if code else None,
    )


def _next_or_end(iterator: Iterator[Any]) -> Any:
    return next(iterator, _STREAM_END)


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


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
