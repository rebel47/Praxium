# Praxium

[![PyPI version](https://img.shields.io/pypi/v/praxium.svg)](https://pypi.org/project/praxium/)
[![Python versions](https://img.shields.io/pypi/pyversions/praxium.svg)](https://pypi.org/project/praxium/)
[![CI](https://github.com/rebel47/Praxium/actions/workflows/ci.yml/badge.svg)](https://github.com/rebel47/Praxium/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/praxium.svg)](https://github.com/rebel47/Praxium/blob/main/LICENSE)

Praxium is a typed, asynchronous Python framework for AI agents and graph
workflows. It provides provider-neutral models and tools, deterministic routing,
structured events, retries, timeouts, cancellation, and checkpoints without
locking application code to one model vendor.

> Praxium `0.1.x` is an alpha release. Its public API is usable and tested, but may
> evolve before `1.0`.

## Installation

Install the framework from PyPI:

```bash
python -m pip install praxium
```

The base installation includes graphs, agents, custom tools, deterministic test
models, in-memory storage, memory, and retrieval. Install only the integrations
your application needs:

```bash
# OpenAI, Anthropic, Gemini, Ollama, Groq, Together, OpenRouter, Kimi,
# GLM, Hugging Face, and arbitrary OpenAI-compatible HTTP endpoints
python -m pip install "praxium[providers]"

# Amazon Bedrock
python -m pip install "praxium[aws]"

# Google Vertex AI
python -m pip install "praxium[gcp,providers]"

# Azure OpenAI with optional Microsoft Entra authentication
python -m pip install "praxium[azure]"

# All cloud-provider dependencies
python -m pip install "praxium[cloud-providers]"

# FastAPI service support
python -m pip install "praxium[api]"
```

Praxium requires Python 3.11 or newer.

## Quick start: run a graph

This example needs no API key or model provider. Save it as `quickstart.py` and run
`python quickstart.py`:

```python
import asyncio

from praxium import GraphBuilder, NodeKind, NodeResult, Runtime, State, StatePatch


async def classify(state: State, _context: object) -> NodeResult:
    route = "warm" if float(state.data["temperature"]) >= 25 else "cold"
    return NodeResult(route=route)


async def warm(_state: State, _context: object) -> StatePatch:
    return StatePatch(values={"advice": "It is warm outside."})


async def cold(_state: State, _context: object) -> StatePatch:
    return StatePatch(values={"advice": "Bring a jacket."})


async def main() -> None:
    graph = (
        GraphBuilder("weather-advice")
        .add_node("classify", classify, kind=NodeKind.CONDITION)
        .add_node("warm", warm)
        .add_node("cold", cold)
        .add_conditional_edges("classify", {"warm": "warm", "cold": "cold"})
        .set_entrypoint("classify")
        .set_finish_point("warm")
        .set_finish_point("cold")
        .build()
    )

    result = await Runtime().run(graph, {"temperature": 29})
    print(result.state.data["advice"])
    print(result.status)


asyncio.run(main())
```

Output:

```text
It is warm outside.
completed
```

## Run an agent with a custom Python tool

Praxium converts typed Python callables into model tool definitions, validates the
arguments, executes the tool, and returns its result to the model. Set
`OPENAI_API_KEY` and `PRAXIUM_MODEL` to values available to your OpenAI account,
then run this file:

```python
import asyncio
import os

from praxium import Agent, AgentRunner, Model, ModelProviderRegistry, Tool
from praxium.providers import ProviderFactory


def multiply(left: int, right: int) -> int:
    """Multiply two integers."""

    return left * right


async def main() -> None:
    provider = ProviderFactory.openai()
    runner = AgentRunner(ModelProviderRegistry([provider]))

    agent = Agent(
        name="calculator",
        instructions="Use the multiplication tool for arithmetic.",
        model=Model(
            name=os.environ["PRAXIUM_MODEL"],
            provider=provider.name,
        ),
        tools=[Tool.from_callable(multiply)],
    )

    result = await runner.run(agent, "What is 37 multiplied by 19?")
    print(result.response.text_content)
    print(result.tool_results)


asyncio.run(main())
```

Tools are provider-neutral. The same callable and agent loop work with every
adapter that supports tool calling; only the provider and model configuration
change.

## Model providers

Praxium does not maintain a model-name allowlist. `Model.name` is passed to the
selected provider unchanged, so newly released, fine-tuned, namespaced, routed,
quantized, and local model IDs do not require a framework update.

| Provider | Factory | Configuration |
|---|---|---|
| OpenAI / GPT | `ProviderFactory.openai()` | `OPENAI_API_KEY` |
| Anthropic / Claude | `ProviderFactory.anthropic()` | `ANTHROPIC_API_KEY` |
| Google Gemini | `ProviderFactory.gemini()` | `GEMINI_API_KEY` |
| Azure OpenAI | `ProviderFactory.azure_openai()` | `AZURE_OPENAI_ENDPOINT` and key/token |
| Amazon Bedrock | `ProviderFactory.bedrock()` | Standard AWS credential chain |
| Google Vertex AI | `ProviderFactory.vertex_ai()` | Google Application Default Credentials |
| Groq | `ProviderFactory.groq()` | `GROQ_API_KEY` |
| Together AI | `ProviderFactory.together()` | `TOGETHER_API_KEY` |
| OpenRouter | `ProviderFactory.openrouter()` | `OPENROUTER_API_KEY` |
| Moonshot / Kimi | `ProviderFactory.kimi()` | `MOONSHOT_API_KEY` |
| Zhipu / GLM | `ProviderFactory.glm()` | `ZHIPUAI_API_KEY` |
| Ollama | `ProviderFactory.ollama()` | Local server; no API key by default |
| Hugging Face router | `ProviderFactory.huggingface()` | `HF_TOKEN` |

For vLLM, LM Studio, LocalAI, a private gateway, or any compatible endpoint:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.openai_compatible(
    provider_name="private-models",
    base_url="https://models.example.com/v1",
    api_key_env="PRIVATE_MODEL_API_KEY",
)

model = Model(
    name="team/fine-tuned-model:latest",
    provider=provider.name,
)
```

APIs with completely different protocols can be integrated with
`CustomModelProvider` callables or by implementing the public `ModelProvider`
protocol. See the
[provider guide](https://github.com/rebel47/Praxium/blob/main/docs/PROVIDERS.md)
for authentication, streaming, structured output, embeddings, custom providers,
and complete agent examples.

## Core capabilities

- Typed messages, multipart content, model requests, responses, and errors
- Async agents with bounded model/tool loops and user-defined Python tools
- Sequential and conditional graphs with whole-graph validation
- Cancellation, deadlines, retries, checkpoints, suspension, and resume
- Ordered execution events and injectable observability sinks
- Provider-neutral streaming, structured output, tool calls, and embeddings
- Tenant-aware in-memory storage, memory, text chunking, and hybrid retrieval
- Plugin, middleware, multi-agent, FastAPI, and OpenAI-compatible service surfaces
- Deterministic offline providers for tests and local development

## Command line

The dependency-free CLI is installed with Praxium:

```bash
praxium --version
praxium doctor
```

Use `praxium --help` to see graph, plugin, and server commands.

## Documentation

- [User guide with examples](https://github.com/rebel47/Praxium/blob/main/docs/USER_GUIDE.md)
- [Model providers and usage examples](https://github.com/rebel47/Praxium/blob/main/docs/PROVIDERS.md)
- [Architecture](https://github.com/rebel47/Praxium/blob/main/docs/ARCHITECTURE.md)
- [PostgreSQL and pgvector design](https://github.com/rebel47/Praxium/blob/main/docs/POSTGRESQL.md)
- [Deployment guide](https://github.com/rebel47/Praxium/blob/main/docs/DEPLOYMENT.md)
- [Delivered and deferred scope](https://github.com/rebel47/Praxium/blob/main/docs/DELIVERY_STATUS.md)
- [Implementation roadmap](https://github.com/rebel47/Praxium/blob/main/docs/IMPLEMENTATION_PLAN.md)

## Development

Editable installs are only needed when contributing to Praxium itself:

```bash
git clone https://github.com/rebel47/Praxium.git
cd Praxium
python -m pip install -e ".[dev,api]"

ruff format --check .
ruff check .
mypy src
pytest --cov=praxium --cov-branch
```

See [CONTRIBUTING.md](https://github.com/rebel47/Praxium/blob/main/CONTRIBUTING.md)
and [SECURITY.md](https://github.com/rebel47/Praxium/blob/main/SECURITY.md) before
opening a pull request or reporting a vulnerability.

## License

Praxium is available under the
[Apache License 2.0](https://github.com/rebel47/Praxium/blob/main/LICENSE).
