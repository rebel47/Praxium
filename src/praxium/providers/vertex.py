"""Google Vertex AI provider using Gemini's native REST protocol and ADC."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from praxium.core import ConfigurationError, ModelProviderError, Usage
from praxium.models import EmbeddingRequest, EmbeddingResponse

from ._http import HTTPTransport
from ._mapping import invalid_response, parse_response_json, translate_transport_error
from .gemini import GeminiProvider


class VertexAIProvider(GeminiProvider):
    """Run Google publisher models on Vertex AI with renewable OAuth credentials."""

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        access_token: str | None = None,
        credentials: Any = None,
        transport: HTTPTransport | None = None,
        provider_name: str = "vertex-ai",
        timeout_seconds: float = 60,
    ) -> None:
        resolved_project = (
            project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
        )
        if not resolved_project:
            raise ConfigurationError(
                "Vertex AI project is missing; pass project or set GOOGLE_CLOUD_PROJECT"
            )
        resolved_location = location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
        host = (
            "aiplatform.googleapis.com"
            if resolved_location == "global"
            else f"{resolved_location}-aiplatform.googleapis.com"
        )
        project_path = quote(resolved_project, safe="-._~")
        location_path = quote(resolved_location, safe="-._~")
        base_url = (
            f"https://{host}/v1/projects/{project_path}/locations/{location_path}/publishers/google"
        )
        self.project = resolved_project
        self.location = resolved_location
        self._google_credentials = credentials
        self._credential_lock = threading.Lock()
        super().__init__(
            access_token=access_token,
            access_token_provider=None if access_token else self._credential_token,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
            provider_name=provider_name,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        calls = [self._embed_one(request, text) for text in request.inputs]
        results = await asyncio.gather(*calls)
        return EmbeddingResponse(
            embeddings=[vector for vector, _tokens in results],
            model=request.model.name,
            usage=Usage(input_tokens=sum(tokens for _vector, tokens in results)),
        )

    async def _embed_one(self, request: EmbeddingRequest, text: str) -> tuple[list[float], int]:
        parameters: dict[str, Any] = {"autoTruncate": True}
        if request.dimensions is not None:
            parameters["outputDimensionality"] = request.dimensions
        try:
            response = await self._transport.post_json(
                self._model_url(request.model.name, "predict"),
                headers=await self._request_headers(),
                payload={"instances": [{"content": text}], "parameters": parameters},
                query={},
                timeout_seconds=request.model.timeout_seconds or self._timeout_seconds,
            )
            data = parse_response_json(response, provider=self.name)
        except (ModelProviderError, ConfigurationError):
            raise
        except Exception as exc:
            raise translate_transport_error(self.name, exc) from exc
        predictions = data.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise invalid_response(self.name, "Vertex embedding response is missing predictions")
        prediction = predictions[0]
        embeddings = prediction.get("embeddings") if isinstance(prediction, Mapping) else None
        values = embeddings.get("values") if isinstance(embeddings, Mapping) else None
        if not isinstance(values, list) or not all(
            isinstance(value, int | float) for value in values
        ):
            raise invalid_response(self.name, "Vertex returned an invalid embedding")
        statistics = embeddings.get("statistics") if isinstance(embeddings, Mapping) else None
        token_count = statistics.get("token_count") if isinstance(statistics, Mapping) else 0
        tokens = (
            int(token_count) if isinstance(token_count, int | float) and token_count >= 0 else 0
        )
        return [float(value) for value in values], tokens

    def _credential_token(self) -> str:
        with self._credential_lock:
            credentials = self._google_credentials
            if credentials is None:
                try:
                    import google.auth
                except ImportError as exc:  # pragma: no cover - installation dependent
                    raise RuntimeError('Vertex AI requires: pip install "praxium[gcp]"') from exc
                credentials, _project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self._google_credentials = credentials
            if not bool(getattr(credentials, "valid", False)):
                try:
                    from google.auth.transport.requests import Request
                except ImportError as exc:  # pragma: no cover - installation dependent
                    raise RuntimeError('Vertex AI requires: pip install "praxium[gcp]"') from exc
                credentials.refresh(Request())
            token = getattr(credentials, "token", None)
            if not isinstance(token, str) or not token:
                raise ConfigurationError("Google credentials did not provide an access token")
            return token
