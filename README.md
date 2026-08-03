# Praxium

Praxium is a typed, asynchronous foundation for AI agents and graph
workflows. It is designed around explicit state, deterministic routing, structured
events, cancellation, retries, timeouts, checkpoints, and provider-neutral
extension interfaces.

> Status: `0.1.0` alpha. The executable core is being built against the tracked
> [implementation plan](https://github.com/rebel47/Praxium/blob/main/docs/IMPLEMENTATION_PLAN.md).

## Quick start

```bash
python -m pip install -e .
python examples/quickstart.py
```

Install production model adapters:

```bash
python -m pip install -e ".[providers]"
```

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.ollama()
model = Model(name="qwen3:8b", provider=provider.name)
```

Praxium includes native OpenAI, Anthropic, Gemini, Amazon Bedrock, and Google Vertex
AI protocol support plus named presets for Azure OpenAI, Groq, Together,
OpenRouter, Kimi, GLM, Ollama, and Hugging Face. Arbitrary OpenAI-compatible
endpoints and completely custom provider callables are supported without a
model-name allowlist.

```python
from praxium import GraphBuilder, Runtime, State


async def greet(state: State, _context):
    return {"greeting": f"Hello, {state.data['name']}!"}


graph = (
    GraphBuilder("hello")
    .add_node("greet", greet)
    .set_entrypoint("greet")
    .set_finish_point("greet")
    .build()
)

result = await Runtime().run(graph, State(data={"name": "Ada"}))
print(result.state.data["greeting"])
```

## Design principles

- Everything crossing a public boundary is typed.
- Every execution produces correlated, ordered events.
- State and checkpoints are serializable and reproducible.
- The runtime is async and cancellation-aware.
- Provider, database, web, and CLI dependencies are optional adapters.
- Extension points use explicit protocols and conflict-safe registries.

## Development

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy src
pytest --cov=praxium --cov-branch
```

The architecture, milestones, acceptance criteria, and v0.1 boundary are recorded
in [docs/IMPLEMENTATION_PLAN.md](https://github.com/rebel47/Praxium/blob/main/docs/IMPLEMENTATION_PLAN.md).
Repository-wide engineering constraints live in
[AGENTS.md](https://github.com/rebel47/Praxium/blob/main/AGENTS.md).

## Documentation

- [User guide with examples](https://github.com/rebel47/Praxium/blob/main/docs/USER_GUIDE.md)
- [Model providers and usage examples](https://github.com/rebel47/Praxium/blob/main/docs/PROVIDERS.md)
- [Architecture](https://github.com/rebel47/Praxium/blob/main/docs/ARCHITECTURE.md)
- [PostgreSQL and pgvector design](https://github.com/rebel47/Praxium/blob/main/docs/POSTGRESQL.md)
- [Deployment guide](https://github.com/rebel47/Praxium/blob/main/docs/DEPLOYMENT.md)
- [Delivered and deferred scope](https://github.com/rebel47/Praxium/blob/main/docs/DELIVERY_STATUS.md)
- [Complete implementation roadmap](https://github.com/rebel47/Praxium/blob/main/docs/IMPLEMENTATION_PLAN.md)
