"""Convenient, secret-safe factories for common model providers."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ._http import HTTPTransport
from .anthropic import AnthropicProvider
from .bedrock import BedrockProvider
from .custom import CustomModelProvider
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider
from .vertex import VertexAIProvider


class ProviderFactory:
    """Create configured providers without restricting available model IDs."""

    @staticmethod
    def openai(
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url=base_url,
            api_key=api_key,
            api_key_env="OPENAI_API_KEY",
            max_tokens_field="max_completion_tokens",
            strict_tools=True,
            transport=transport,
            **options,
        )

    @staticmethod
    def anthropic(
        *, api_key: str | None = None, transport: HTTPTransport | None = None, **options: Any
    ) -> AnthropicProvider:
        return AnthropicProvider(api_key=api_key, transport=transport, **options)

    @staticmethod
    def gemini(
        *, api_key: str | None = None, transport: HTTPTransport | None = None, **options: Any
    ) -> GeminiProvider:
        return GeminiProvider(api_key=api_key, transport=transport, **options)

    @staticmethod
    def kimi(
        *,
        api_key: str | None = None,
        provider_name: str = "kimi",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url="https://api.moonshot.ai/v1",
            api_key=api_key,
            api_key_env="MOONSHOT_API_KEY",
            embeddings_path=None,
            transport=transport,
            **options,
        )

    @staticmethod
    def glm(
        *,
        api_key: str | None = None,
        provider_name: str = "glm",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=api_key,
            api_key_env="ZHIPUAI_API_KEY",
            embeddings_path=None,
            transport=transport,
            **options,
        )

    @staticmethod
    def ollama(
        *,
        base_url: str = "http://localhost:11434/v1",
        provider_name: str = "ollama",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url=base_url,
            api_key="ollama",
            require_api_key=False,
            transport=transport,
            **options,
        )

    @staticmethod
    def huggingface(
        *,
        api_key: str | None = None,
        provider_name: str = "huggingface",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
            api_key_env="HF_TOKEN",
            embeddings_path=None,
            transport=transport,
            **options,
        )

    @staticmethod
    def azure_openai(
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        token_provider: Callable[[], str | Awaitable[str]] | None = None,
        provider_name: str = "azure-openai",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        resolved_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not resolved_endpoint:
            from praxium.core import ConfigurationError

            raise ConfigurationError(
                "Azure OpenAI endpoint is missing; pass endpoint or set AZURE_OPENAI_ENDPOINT"
            )
        base_url = resolved_endpoint.rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url = f"{base_url}/openai/v1"
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url=base_url,
            api_key=api_key,
            api_key_env="AZURE_OPENAI_API_KEY",
            api_key_provider=token_provider,
            max_tokens_field="max_completion_tokens",
            strict_tools=True,
            transport=transport,
            **options,
        )

    @staticmethod
    def groq(
        *,
        api_key: str | None = None,
        provider_name: str = "groq",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            api_key_env="GROQ_API_KEY",
            embeddings_path=None,
            max_tokens_field="max_completion_tokens",
            transport=transport,
            **options,
        )

    @staticmethod
    def together(
        *,
        api_key: str | None = None,
        provider_name: str = "together",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            api_key_env="TOGETHER_API_KEY",
            transport=transport,
            **options,
        )

    @staticmethod
    def openrouter(
        *,
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        headers: Mapping[str, str] | None = None,
        provider_name: str = "openrouter",
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        request_headers = dict(headers or {})
        if site_url:
            request_headers.setdefault("HTTP-Referer", site_url)
        if app_name:
            request_headers.setdefault("X-OpenRouter-Title", app_name)
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            api_key_env="OPENROUTER_API_KEY",
            headers=request_headers,
            transport=transport,
            **options,
        )

    @staticmethod
    def bedrock(
        *,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any = None,
        provider_name: str = "bedrock",
    ) -> BedrockProvider:
        return BedrockProvider(
            region_name=region_name,
            endpoint_url=endpoint_url,
            client=client,
            provider_name=provider_name,
        )

    @staticmethod
    def vertex_ai(
        *,
        project: str | None = None,
        location: str | None = None,
        access_token: str | None = None,
        credentials: Any = None,
        transport: HTTPTransport | None = None,
        provider_name: str = "vertex-ai",
        **options: Any,
    ) -> VertexAIProvider:
        return VertexAIProvider(
            project=project,
            location=location,
            access_token=access_token,
            credentials=credentials,
            transport=transport,
            provider_name=provider_name,
            **options,
        )

    @staticmethod
    def openai_compatible(
        *,
        provider_name: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_key_provider: Callable[[], str | Awaitable[str]] | None = None,
        headers: Mapping[str, str] | None = None,
        transport: HTTPTransport | None = None,
        **options: Any,
    ) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_provider=api_key_provider,
            headers=headers,
            transport=transport,
            **options,
        )

    @staticmethod
    def custom(
        *, name: str, complete: Any, stream: Any = None, embed: Any = None
    ) -> CustomModelProvider:
        return CustomModelProvider(name=name, complete=complete, stream=stream, embed=embed)
