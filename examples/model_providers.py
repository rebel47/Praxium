"""Select any built-in model provider; defaults to deterministic offline mode."""

from __future__ import annotations

import asyncio
import os

from praxium import (
    Agent,
    AgentRunner,
    DeterministicModelProvider,
    Model,
    ModelProviderRegistry,
    Tool,
    text_response,
)
from praxium.models import ModelProvider
from praxium.providers import ProviderFactory


def multiply(left: int, right: int) -> int:
    """Multiply two integers."""

    return left * right


def select_provider(name: str) -> ModelProvider:
    factories = {
        "openai": ProviderFactory.openai,
        "anthropic": ProviderFactory.anthropic,
        "gemini": ProviderFactory.gemini,
        "azure-openai": ProviderFactory.azure_openai,
        "bedrock": ProviderFactory.bedrock,
        "vertex-ai": ProviderFactory.vertex_ai,
        "groq": ProviderFactory.groq,
        "together": ProviderFactory.together,
        "openrouter": ProviderFactory.openrouter,
        "kimi": ProviderFactory.kimi,
        "glm": ProviderFactory.glm,
        "ollama": ProviderFactory.ollama,
        "huggingface": ProviderFactory.huggingface,
    }
    if name == "deterministic":
        return DeterministicModelProvider(responses=[text_response("Offline provider is ready.")])
    try:
        return factories[name]()
    except KeyError as exc:
        supported = ", ".join(["deterministic", *sorted(factories)])
        raise ValueError(f"unknown provider {name!r}; choose one of: {supported}") from exc


async def main() -> None:
    provider_name = os.getenv("PRAXIUM_PROVIDER", "deterministic")
    model_name = os.getenv("PRAXIUM_MODEL", "offline")
    provider = select_provider(provider_name)
    runner = AgentRunner(ModelProviderRegistry([provider]))
    agent = Agent(
        name="portable-calculator",
        instructions="Use the multiplication tool when arithmetic is requested.",
        model=Model(name=model_name, provider=provider.name),
        tools=[Tool.from_callable(multiply)],
    )
    result = await runner.run(agent, "What is 37 multiplied by 19?")
    print(result.response.text_content)


if __name__ == "__main__":
    asyncio.run(main())
