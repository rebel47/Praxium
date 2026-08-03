"""Provider-neutral mapping helpers shared by adapters."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from praxium.core import FinishReason, ModelProviderError, ToolCallId, Usage

from ._http import HTTPResponse, HTTPStatusError, HTTPTransportError

_EXTERNAL_CALL_PREFIX = "call_ext_"


def encode_external_call_id(value: str) -> ToolCallId:
    encoded = base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")
    return ToolCallId(root=f"{_EXTERNAL_CALL_PREFIX}{encoded}")


def decode_external_call_id(value: ToolCallId | str) -> str:
    raw = str(value)
    if not raw.startswith(_EXTERNAL_CALL_PREFIX):
        return raw
    encoded = raw[len(_EXTERNAL_CALL_PREFIX) :]
    encoded += "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded).decode()


def parse_object(value: Any, *, provider: str, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise invalid_response(provider, f"{field} must be a JSON object")
    if not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise invalid_response(provider, f"{field} contains invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise invalid_response(provider, f"{field} must decode to an object")
    return decoded


def parse_response_json(response: HTTPResponse, *, provider: str) -> dict[str, Any]:
    if not 200 <= response.status_code < 300:
        raise status_error(provider, response)
    try:
        payload = response.json()
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise invalid_response(provider, "provider returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise invalid_response(provider, "provider response must be a JSON object")
    return payload


def translate_transport_error(provider: str, exc: Exception) -> ModelProviderError:
    if isinstance(exc, HTTPStatusError):
        return status_error(provider, exc.response)
    if isinstance(exc, HTTPTransportError):
        return ModelProviderError(str(exc), provider=provider, retryable=exc.retryable)
    return ModelProviderError("model provider request failed", provider=provider)


def status_error(provider: str, response: HTTPResponse) -> ModelProviderError:
    message = _safe_error_message(response)
    return ModelProviderError(
        message,
        provider=provider,
        status_code=response.status_code,
        retryable=response.status_code in {408, 409, 425, 429} or response.status_code >= 500,
    )


def invalid_response(provider: str, message: str) -> ModelProviderError:
    return ModelProviderError(message, provider=provider)


def map_finish_reason(value: Any) -> FinishReason:
    reason = str(value or "stop").lower()
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        return FinishReason.LENGTH
    if reason in {"tool_calls", "tool_use", "function_call"}:
        return FinishReason.TOOL_CALLS
    if reason in {
        "blocked",
        "content_filter",
        "content_filtered",
        "guardrail_intervened",
        "recitation",
        "safety",
    }:
        return FinishReason.CONTENT_FILTER
    if reason in {"error", "network_error"}:
        return FinishReason.ERROR
    return FinishReason.STOP


def usage_from_openai(value: Any) -> Usage:
    if not isinstance(value, Mapping):
        return Usage()
    prompt_details = value.get("prompt_tokens_details")
    completion_details = value.get("completion_tokens_details")
    return Usage(
        input_tokens=_integer(value.get("prompt_tokens", value.get("input_tokens", 0))),
        output_tokens=_integer(value.get("completion_tokens", value.get("output_tokens", 0))),
        cached_input_tokens=_nested_integer(prompt_details, "cached_tokens"),
        reasoning_tokens=_nested_integer(completion_details, "reasoning_tokens"),
    )


def _safe_error_message(response: HTTPResponse) -> str:
    try:
        payload = response.json()
    except (UnicodeDecodeError, json.JSONDecodeError):
        return f"model provider returned HTTP {response.status_code}"
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            return str(error["message"])[:500]
        if isinstance(error, str):
            return error[:500]
        if isinstance(payload.get("message"), str):
            return str(payload["message"])[:500]
    return f"model provider returned HTTP {response.status_code}"


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, int | float) and value >= 0 else 0


def _nested_integer(value: Any, key: str) -> int:
    return _integer(value.get(key)) if isinstance(value, Mapping) else 0
