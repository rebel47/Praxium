"""Small injectable async HTTP boundary used by provider adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    """Buffered HTTP response with a decoded JSON convenience method."""

    status_code: int
    body: bytes
    headers: Mapping[str, str]

    def json(self) -> Any:
        return json.loads(self.body)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One server-sent event."""

    data: str
    event: str | None = None


class HTTPTransportError(Exception):
    """Internal transport error safe to translate at the provider boundary."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class HTTPStatusError(HTTPTransportError):
    """Internal non-success HTTP response."""

    def __init__(self, response: HTTPResponse) -> None:
        super().__init__(f"HTTP {response.status_code}", retryable=_retryable(response.status_code))
        self.response = response


class HTTPTransport(Protocol):
    """Injectable protocol so adapters can be tested without live services."""

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPResponse: ...

    def stream_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> AsyncIterator[SSEEvent]: ...


class HTTPXTransport:
    """HTTP transport backed by the optional ``httpx`` dependency."""

    def __init__(self, *, transport: Any = None) -> None:
        """Accept an optional httpx transport for proxies and deterministic tests."""

        self._httpx_transport = transport

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPResponse:
        httpx = _load_httpx()
        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds, transport=self._httpx_transport
            ) as client:
                response = await client.post(
                    url,
                    headers=dict(headers),
                    params=dict(query),
                    json=dict(payload),
                )
        except httpx.TimeoutException as exc:
            raise HTTPTransportError("model provider request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise HTTPTransportError(
                "model provider network request failed", retryable=True
            ) from exc
        return HTTPResponse(
            status_code=response.status_code,
            body=response.content,
            headers=dict(response.headers),
        )

    async def stream_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        query: Mapping[str, str],
        timeout_seconds: float,
    ) -> AsyncIterator[SSEEvent]:
        httpx = _load_httpx()
        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds, transport=self._httpx_transport
            ) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=dict(headers),
                    params=dict(query),
                    json=dict(payload),
                ) as response:
                    if not 200 <= response.status_code < 300:
                        body = await response.aread()
                        raise HTTPStatusError(
                            HTTPResponse(
                                status_code=response.status_code,
                                body=body,
                                headers=dict(response.headers),
                            )
                        )
                    event_name: str | None = None
                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if not line:
                            if data_lines:
                                yield SSEEvent(data="\n".join(data_lines), event=event_name)
                            event_name = None
                            data_lines = []
                        elif line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    if data_lines:
                        yield SSEEvent(data="\n".join(data_lines), event=event_name)
        except HTTPStatusError:
            raise
        except httpx.TimeoutException as exc:
            raise HTTPTransportError("model provider stream timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise HTTPTransportError("model provider stream failed", retryable=True) from exc


def _load_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            'Provider adapters require the optional dependency: pip install "praxium[providers]"'
        ) from exc
    return httpx


def _retryable(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500
