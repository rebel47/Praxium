# Praxium User Guide

Praxium is a typed Python framework for building AI agents and deterministic graph
workflows. This guide focuses on working examples: defining state, composing
graphs, running and observing them, using tools and model providers, storing
memory, retrieving documents, exposing APIs, and coordinating agent teams.

## 1. Installation

Praxium requires Python 3.11 or newer.

From this repository:

```bash
python -m pip install -e .
```

Install the optional HTTP server dependencies:

```bash
python -m pip install -e ".[api]"
```

Install the development toolchain:

```bash
python -m pip install -e ".[dev,api]"
```

Verify the environment:

```bash
praxium doctor
```

The root package exports the most commonly used types:

```python
from praxium import GraphBuilder, Runtime, State
```

Specialized types remain available from packages such as `praxium.memory`,
`praxium.middleware`, and `praxium.rag`.

## 2. Your first workflow

A node receives the current `State` and an `ExecutionContext`. It may return a
dictionary, `StatePatch`, `NodeResult`, another `State`, a plain output value, or
`None`. A returned dictionary is interpreted as a state patch.

```python
import asyncio

from praxium import ExecutionContext, GraphBuilder, Runtime, State


async def greet(state: State, context: ExecutionContext) -> dict[str, str]:
    name = str(state.data["name"])
    return {
        "greeting": f"Hello, {name}!",
        "execution_id": str(context.execution_id),
    }


async def main() -> None:
    graph = (
        GraphBuilder("hello")
        .add_node("greet", greet)
        .set_entrypoint("greet")
        .set_finish_point("greet")
        .build()
    )

    result = await Runtime().run(graph, {"name": "Ada"})

    print(result.status)  # completed
    print(result.state.data["greeting"])  # Hello, Ada!
    print(result.state.version)  # 1


asyncio.run(main())
```

Praxium copies state before invoking a node. A failed node therefore cannot commit
changes by mutating the provided state object. Only the returned patch is committed.

## 3. Sequential workflows

Connect nodes with ordinary edges:

```python
import asyncio

from praxium import ExecutionContext, GraphBuilder, Runtime, State


async def normalize(state: State, _context: ExecutionContext) -> dict[str, str]:
    return {"name": str(state.data["name"]).strip().title()}


async def compose(state: State, _context: ExecutionContext) -> dict[str, str]:
    return {"message": f"Welcome, {state.data['name']}"}


graph = (
    GraphBuilder("welcome")
    .add_node("normalize", normalize)
    .add_node("compose", compose)
    .add_edge("normalize", "compose")
    .set_entrypoint("normalize")
    .set_finish_point("compose")
    .build()
)


async def main() -> None:
    result = await Runtime().run(graph, {"name": "  ada lovelace "})
    print(result.state.data["message"])


asyncio.run(main())
```

Every non-terminal node needs an outgoing edge. A finish point stops execution even
if an outgoing edge was accidentally declared; validation reports that situation as
a warning.

## 4. Conditional routing

Use route-labelled edges and return a matching route from `NodeResult`:

```python
import asyncio

from praxium import (
    ExecutionContext,
    GraphBuilder,
    NodeKind,
    NodeResult,
    Runtime,
    State,
)


async def classify(state: State, _context: ExecutionContext) -> NodeResult:
    score = float(state.data["score"])
    return NodeResult(route="accept" if score >= 0.8 else "review")


async def accept(_state: State, _context: ExecutionContext) -> dict[str, str]:
    return {"decision": "accepted"}


async def review(_state: State, _context: ExecutionContext) -> dict[str, str]:
    return {"decision": "manual_review"}


graph = (
    GraphBuilder("decision")
    .add_node("classify", classify, kind=NodeKind.CONDITION)
    .add_node("accept", accept)
    .add_node("review", review)
    .add_conditional_edges(
        "classify",
        {"accept": "accept", "review": "review"},
    )
    .set_entrypoint("classify")
    .set_finish_point("accept")
    .set_finish_point("review")
    .build()
)


async def main() -> None:
    result = await Runtime().run(graph, {"score": 0.91})
    print(result.state.data["decision"])


asyncio.run(main())
```

Selecting a route that has no matching edge produces a failed `ExecutionResult`
with a typed error instead of silently choosing another path.

## 5. Bounded loops

Cycles must declare an explicit visit bound. Runtime step limits provide a second
safety boundary.

```python
import asyncio

from praxium import (
    ExecutionContext,
    ExecutionLimits,
    GraphBuilder,
    NodeKind,
    NodeResult,
    Runtime,
    RuntimeConfig,
    State,
    StatePatch,
)


async def decide(state: State, _context: ExecutionContext) -> NodeResult:
    attempt = int(state.data.get("attempt", 0))
    next_attempt = attempt + 1
    return NodeResult(
        patch=StatePatch(values={"attempt": next_attempt}),
        route="again" if next_attempt < 3 else "done",
    )


async def work(state: State, _context: ExecutionContext) -> dict[str, list[int]]:
    previous = list(state.data.get("observed_attempts", []))
    return {"observed_attempts": [*previous, int(state.data["attempt"])]}


async def finish(_state: State, _context: ExecutionContext) -> dict[str, bool]:
    return {"finished": True}


graph = (
    GraphBuilder("bounded-loop")
    .add_node("decide", decide, kind=NodeKind.CONDITION, max_visits=3)
    .add_node("work", work, max_visits=2)
    .add_node("finish", finish)
    .add_conditional_edges("decide", {"again": "work", "done": "finish"})
    .add_edge("work", "decide")
    .set_entrypoint("decide")
    .set_finish_point("finish")
    .build()
)


async def main() -> None:
    runtime = Runtime(config=RuntimeConfig(limits=ExecutionLimits(max_steps=10)))
    result = await runtime.run(graph)
    print(result.state.data)


asyncio.run(main())
```

An unbounded cycle is rejected by `GraphBuilder.build()` before execution.

## 6. State patches, history, and merge policies

State is versioned and returns a new object after every committed patch:

```python
from praxium import State, StatePatch

initial = State(data={"count": 1})
updated = initial.apply(StatePatch(values={"count": 2, "status": "ready"}))

assert initial.data == {"count": 1}
assert updated.version == 1
assert [change.path for change in updated.history] == ["count", "status"]
```

Parallel state merge primitives are available even though v0.1 graph scheduling is
sequential:

```python
from praxium.core import MergePolicy, MergeStrategy, State

state = State()
merged = state.merge(
    [{"items": ["a"]}, {"items": ["b"]}],
    {"items": MergePolicy(strategy=MergeStrategy.APPEND)},
)

assert merged.data["items"] == ["a", "b"]
```

Without a merge policy, incompatible writes raise `StateConflictError`. Supported
strategies are replace, append, numeric addition, set union, recursive mapping
merge, and a custom resolver.

## 7. Retries, timeouts, and run limits

Retries belong to individual nodes:

```python
import asyncio

from praxium import (
    ExecutionContext,
    ExecutionLimits,
    GraphBuilder,
    RetryPolicy,
    Runtime,
    RuntimeConfig,
    State,
)

attempts = 0


async def unreliable(_state: State, _context: ExecutionContext) -> dict[str, bool]:
    global attempts
    attempts += 1
    if attempts < 3:
        raise OSError("temporary failure")
    return {"ok": True}


graph = (
    GraphBuilder("resilient")
    .add_node(
        "work",
        unreliable,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.1,
            multiplier=2,
            max_delay_seconds=2,
            jitter=0.1,
        ),
        timeout_seconds=5,
    )
    .set_entrypoint("work")
    .set_finish_point("work")
    .build()
)


async def main() -> None:
    runtime = Runtime(
        config=RuntimeConfig(
            limits=ExecutionLimits(
                timeout_seconds=30,
                node_timeout_seconds=10,
                max_steps=20,
            )
        )
    )
    result = await runtime.run(graph)
    print(result.status, result.node_runs[0].attempts)


asyncio.run(main())
```

The effective node timeout is the smallest of the node timeout, default node
timeout, and remaining run deadline. Retry waits are cancellation-aware.

## 8. Streaming execution events

`Runtime.stream()` yields ordered lifecycle events while execution is active:

```python
import asyncio

from praxium import EventKind, ExecutionContext, GraphBuilder, Runtime, State


async def task(_state: State, _context: ExecutionContext) -> dict[str, int]:
    return {"value": 42}


graph = (
    GraphBuilder("events")
    .add_node("task", task)
    .set_entrypoint("task")
    .set_finish_point("task")
    .build()
)


async def main() -> None:
    async for event in Runtime().stream(graph):
        print(event.sequence, event.kind, event.node_id, event.payload)
        if event.kind == EventKind.RUN_COMPLETED:
            print("run completed")


asyncio.run(main())
```

Events include run, node, edge, retry, checkpoint, model, tool, memory, and approval
kinds. Each run starts at sequence 1 and emits strictly increasing sequence numbers.

To retain events outside the returned result, register a sink:

```python
from praxium import Runtime
from praxium.observability import InMemoryEventSink

sink = InMemoryEventSink()
runtime = Runtime(event_sinks=[sink])
```

Exporter failures are isolated and do not change workflow results.

## 9. Cancelling an active execution

Supply an execution ID before starting the run so another task can cancel it:

```python
import asyncio

from praxium import ExecutionContext, ExecutionId, GraphBuilder, Runtime, State

started = asyncio.Event()


async def long_task(_state: State, _context: ExecutionContext) -> None:
    started.set()
    await asyncio.Event().wait()


graph = (
    GraphBuilder("cancellable")
    .add_node("wait", long_task)
    .set_entrypoint("wait")
    .set_finish_point("wait")
    .build()
)


async def main() -> None:
    runtime = Runtime()
    execution_id = ExecutionId.new()
    running = asyncio.create_task(runtime.run(graph, execution_id=execution_id))

    await started.wait()
    cancelled = await runtime.cancel(execution_id, "cancelled by user")
    result = await running

    print(cancelled)  # True
    print(result.status)  # cancelled


asyncio.run(main())
```

Async handlers are cancelled and awaited. A synchronous handler runs in a worker
thread; Python cannot forcibly terminate that thread, so untrusted or potentially
blocking tools require an external process or container sandbox.

## 10. Human approval and checkpoint resume

A node can commit a patch and suspend the run. Praxium checkpoints the next node,
so resuming does not repeat the committed approval node.

```python
import asyncio

from praxium import (
    ExecutionContext,
    GraphBuilder,
    NodeKind,
    NodeResult,
    Runtime,
    State,
    StatePatch,
    Suspension,
)


async def request_approval(
    _state: State,
    _context: ExecutionContext,
) -> NodeResult:
    return NodeResult(
        patch=StatePatch(values={"approval_requested": True}),
        suspension=Suspension(
            reason="Approve production deployment",
            payload={"environment": "production"},
        ),
    )


async def deploy(_state: State, _context: ExecutionContext) -> dict[str, str]:
    return {"deployment": "completed"}


graph = (
    GraphBuilder("deployment")
    .add_node("approval", request_approval, kind=NodeKind.APPROVAL)
    .add_node("deploy", deploy)
    .add_edge("approval", "deploy")
    .set_entrypoint("approval")
    .set_finish_point("deploy")
    .build()
)


async def main() -> None:
    runtime = Runtime()
    suspended = await runtime.run(graph)

    print(suspended.status)  # suspended
    checkpoint_id = suspended.last_checkpoint_id

    # Call resume only after the user or external approval system confirms.
    assert checkpoint_id is not None
    resumed = await runtime.resume(
        graph,
        checkpoint_id,
        metadata={"approved_by": "user-123"},
    )

    print(resumed.status)  # completed
    print(resumed.state.data["deployment"])  # completed


asyncio.run(main())
```

The runnable version is [examples/approval_resume.py](../examples/approval_resume.py).
Resume checks the graph ID, version, and structural fingerprint before executing.

## 11. Defining and calling tools

Praxium generates a JSON Schema from a typed Python callable:

```python
import asyncio

from praxium import ExecutionId, Tool, ToolContext


def add(left: int, right: int = 1) -> int:
    """Add two integers."""

    return left + right


async def main() -> None:
    tool = Tool.from_callable(
        add,
        required_permissions={"calculator:use"},
        timeout_seconds=5,
    )

    context = ToolContext(
        execution_id=ExecutionId.new(),
        granted_permissions={"calculator:use"},
    )
    result = await tool.execute({"left": 4, "right": 5}, context)

    print(tool.input_schema)
    print(result.status, result.output)  # success 9


asyncio.run(main())
```

Tool execution includes:

- Pydantic input validation;
- declared permission checks;
- timeout enforcement;
- synchronous execution outside the event loop;
- maximum serialized output size;
- optional sensitive-output redaction;
- typed success, error, denied, and timed-out results.

Use a registry when tools are assembled dynamically:

```python
from praxium import ToolRegistry

registry = ToolRegistry([tool])
same_tool = registry.get("add")
definitions = registry.definitions()
```

Duplicate names are rejected instead of silently replacing an existing tool.

## 12. Running a tool-using agent

An `Agent` combines instructions, model configuration, tools, and execution limits.
The provider performs model calls; `AgentRunner` manages the bounded tool loop.

This offline example uses `DeterministicModelProvider` so it requires no API key:

```python
import asyncio

from praxium import (
    Agent,
    AgentRunner,
    DeterministicModelProvider,
    Message,
    Model,
    ModelProviderRegistry,
    Response,
    Role,
    Tool,
    ToolCallPart,
)


def add(left: int, right: int) -> int:
    """Add two integers."""

    return left + right


tool = Tool.from_callable(add)
call = ToolCallPart(
    tool_name="add",
    arguments={"left": 2, "right": 3},
)
provider = DeterministicModelProvider(
    responses=[
        Response(message=Message(role=Role.ASSISTANT, parts=[call])),
        Response(message=Message.assistant("The answer is 5.")),
    ]
)
runner = AgentRunner(ModelProviderRegistry([provider]))
agent = Agent(
    name="calculator",
    instructions="Use the calculator tool for arithmetic.",
    model=Model(name="offline", provider="deterministic"),
    tools=[tool],
)


async def main() -> None:
    result = await runner.run(agent, "What is 2 + 3?")
    print(result.response.text_content)
    print(result.steps)
    print(result.tool_results[0].output)


asyncio.run(main())
```

See [examples/agent_tools.py](../examples/agent_tools.py) for the runnable version.

### Connecting real model providers

Install the HTTP provider extra:

```bash
python -m pip install -e ".[providers]"
```

Praxium includes presets for OpenAI/GPT, Anthropic/Claude, Google Gemini, Azure
OpenAI, Amazon Bedrock, Google Vertex AI, Groq, Together, OpenRouter, Moonshot/Kimi,
Zhipu/GLM, Ollama, and the Hugging Face inference router:

```python
from praxium import Model, ModelProviderRegistry
from praxium.providers import ProviderFactory

provider = ProviderFactory.openai()  # reads OPENAI_API_KEY when called
providers = ModelProviderRegistry([provider])
model = Model(name="your-enabled-model-id", provider=provider.name)
```

Switch providers without changing agent or tool code:

```python
provider = ProviderFactory.anthropic()  # ANTHROPIC_API_KEY
provider = ProviderFactory.gemini()  # GEMINI_API_KEY
provider = ProviderFactory.azure_openai()  # AZURE_OPENAI_ENDPOINT + key/token
provider = ProviderFactory.bedrock(region_name="eu-central-1")  # AWS credential chain
provider = ProviderFactory.vertex_ai(project="my-project")  # Google ADC
provider = ProviderFactory.groq()  # GROQ_API_KEY
provider = ProviderFactory.together()  # TOGETHER_API_KEY
provider = ProviderFactory.openrouter()  # OPENROUTER_API_KEY
provider = ProviderFactory.kimi()  # MOONSHOT_API_KEY
provider = ProviderFactory.glm()  # ZHIPUAI_API_KEY
provider = ProviderFactory.ollama()  # local http://localhost:11434/v1
provider = ProviderFactory.huggingface()  # HF_TOKEN
```

For vLLM, LM Studio, LocalAI, a hosted gateway, or any other compatible endpoint:

```python
provider = ProviderFactory.openai_compatible(
    provider_name="private-models",
    base_url="https://models.example.com/v1",
    api_key_env="PRIVATE_MODEL_API_KEY",
)
model = Model(
    name="team/new-or-fine-tuned-model:latest",
    provider="private-models",
)
```

Praxium does not keep a model allowlist. It sends `Model.name` unchanged, so new,
local, namespaced, quantized, and fine-tuned identifiers work without a framework
release. The selected model must still support the requested tools, structured
output, vision, or embedding capability.

For a completely proprietary protocol, use `CustomModelProvider` callables or
implement the public `ModelProvider` protocol. Provider secrets belong inside the
adapter and must not be stored in the serializable `Model` configuration.

See [Model Providers](PROVIDERS.md) for the capability matrix and complete examples
for agents, custom tools, streaming, structured output, embeddings, multiple
providers, error handling, and proprietary adapters.

## 13. Memory

The in-memory backend supports tenant isolation, namespaces, record kinds, metadata
filters, version history, expiration, lexical ranking, and branches.

```python
import asyncio

from praxium import (
    InMemoryMemoryStore,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
)
from praxium.memory import RetentionPolicy


async def main() -> None:
    memory = InMemoryMemoryStore()
    record = MemoryRecord(
        tenant_id="tenant-1",
        namespace="assistant",
        kind=MemoryKind.LONG_TERM,
        content="Ada prefers concise technical explanations.",
        metadata={"source": "profile"},
    )
    stored = await memory.put(
        record,
        RetentionPolicy(ttl_seconds=3600, max_versions=5),
    )

    matches = await memory.query(
        MemoryQuery(
            tenant_id="tenant-1",
            namespace="assistant",
            text="concise explanations",
            kinds={MemoryKind.LONG_TERM},
            metadata={"source": "profile"},
        )
    )

    print(matches[0].record.content)
    print(matches[0].score)

    experiment = await memory.branch(
        "tenant-1",
        stored.id,
        "experiment",
    )
    print(experiment.branch, experiment.parent_id)


asyncio.run(main())
```

Every query requires both tenant and namespace. Expired records are excluded unless
`include_expired=True` is explicitly requested.

## 14. Document indexing and RAG retrieval

The reference RAG index preserves exact source offsets and produces citations:

```python
import asyncio

from praxium import Document, InMemoryRagIndex, RetrievalQuery, TextChunker


async def main() -> None:
    index = InMemoryRagIndex()
    document = Document(
        tenant_id="tenant-1",
        namespace="documentation",
        collection="guides",
        content=(
            "Praxium workflows use explicit versioned state. "
            "Nodes return patches that commit after successful execution. "
        )
        * 10,
        source_uri="https://example.test/praxium/state",
        metadata={"title": "State Guide", "language": "en"},
    )

    indexed = await index.index(
        document,
        chunker=TextChunker(chunk_size=150, overlap=20),
    )
    print(indexed.chunks_indexed)

    results = await index.query(
        RetrievalQuery(
            tenant_id="tenant-1",
            namespace="documentation",
            collection="guides",
            text="versioned state patches",
            metadata={"language": "en"},
            top_k=3,
        )
    )

    for result in results:
        print(result.score, result.chunk.text)
        print(result.citation.source_uri)
        print(result.citation.start, result.citation.end)


asyncio.run(main())
```

Calling `index()` again with the same document ID replaces the previous chunks
atomically in the reference backend. Supply an object implementing the async
`Embedder` protocol to store dense vectors and pass `RetrievalQuery.embedding` to
combine dense cosine similarity with lexical similarity.

The in-memory index is intended for development and contract tests. The production
PostgreSQL/pgvector schema is described in [POSTGRESQL.md](POSTGRESQL.md).

## 15. Middleware

Middleware wraps run, node, model, tool, or memory boundaries:

```python
import asyncio

from praxium.middleware import (
    Boundary,
    MiddlewareContext,
    MiddlewarePipeline,
    RedactionMiddleware,
    TimeoutMiddleware,
)


async def terminal(context: MiddlewareContext) -> dict[str, object]:
    return {
        "received": context.payload,
        "authorization": "outbound-secret",
    }


async def main() -> None:
    pipeline = MiddlewarePipeline(
        [
            TimeoutMiddleware(timeout_seconds=2),
            RedactionMiddleware(),
        ]
    )
    result = await pipeline.execute(
        MiddlewareContext(
            boundary=Boundary.TOOL,
            operation="send",
            payload={"password": "inbound-secret", "message": "hello"},
        ),
        terminal,
    )
    print(result)


asyncio.run(main())
```

Middleware runs in registration order and unwinds in reverse. Names must be unique
inside one pipeline.

## 16. Plugins

Plugins declare compatibility and capabilities before registering components:

```python
from praxium import Tool
from praxium.plugins import (
    Plugin,
    PluginCapability,
    PluginManifest,
    PluginRegistration,
    PluginRegistry,
)


def ping() -> str:
    """Return a health signal."""

    return "pong"


plugin = Plugin(
    manifest=PluginManifest(
        name="example-tools",
        version="0.1.0",
        description="Example Praxium tools.",
        min_framework_version="0.1.0",
        capabilities={PluginCapability.TOOL},
    ),
    registrations=[
        PluginRegistration(
            capability=PluginCapability.TOOL,
            name="ping",
            component=Tool.from_callable(ping),
        )
    ],
)

registry = PluginRegistry()
registry.register(plugin)
ping_tool = registry.get(PluginCapability.TOOL, "ping")
```

Installed plugins use the `praxium.plugins` Python entry-point group. Discovery
reads metadata without importing plugin code:

```bash
praxium plugins
```

Loading is explicit because importing a plugin executes third-party Python code.
Incompatible versions and duplicate component names are rejected.

## 17. Application registry and generated HTTP API

Register graphs and agents on an `Application`:

```python
from praxium import Application

application = Application().register(graph)
```

Execute through the registry:

```python
result = await application.run(
    "decision",
    {"score": 0.91},
    tenant_id="tenant-1",
    metadata={"request_id": "request-123"},
)
```

Create an optional FastAPI application:

```python
from praxium.api import create_fastapi_app

web_app = create_fastapi_app(application, title="My Praxium Service")
```

Run the included server target:

```bash
praxium serve examples.server:application --host 127.0.0.1 --port 8000
```

Generated framework endpoints include:

- `GET /health`
- `GET /ready`
- `GET /components`
- `POST /runs/{name}`
- `POST /runs/{name}/stream`

Call a graph:

```bash
curl -X POST http://127.0.0.1:8000/runs/echo \
  -H "Content-Type: application/json" \
  -d '{"input":{"message":"hello"},"tenant_id":"tenant-1"}'
```

Stream graph events:

```bash
curl -N -X POST http://127.0.0.1:8000/runs/echo/stream \
  -H "Content-Type: application/json" \
  -d '{"input":{"message":"hello"}}'
```

FastAPI automatically exposes OpenAPI at `/openapi.json` and Swagger UI at `/docs`.
Install authentication, authorization, tenant resolution, rate limiting, and TLS
before exposing the alpha server publicly.

## 18. OpenAI-compatible endpoints

The FastAPI factory also exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`

The `model` field is the component name registered on `Application`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "assistant",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Streaming uses server-sent events:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"assistant","input":"Hello","stream":true}'
```

To expose embeddings, register a model configuration:

```python
from praxium import Application, Model, ModelCapabilities

application.register_embedding_model(
    "embedding-small",
    Model(
        name="provider-embedding-model",
        provider="my-provider",
        capabilities=ModelCapabilities(embeddings=True),
    ),
)
```

An `AgentRunner` containing the matching provider registry must be supplied to the
application. Request-defined OpenAI tools are not accepted in v0.1; register tools
on the target Praxium agent instead.

## 19. Multi-agent teams

A team is a bounded sequence of ordinary agent runs. Strategies explicitly choose
delegations; the included strategy visits each member in order.

```python
import asyncio

from praxium import (
    Agent,
    AgentRunner,
    DeterministicModelProvider,
    Model,
    ModelProviderRegistry,
    Team,
    TeamMember,
    TeamPolicy,
    TeamRole,
    TeamRunner,
    text_response,
)

provider = DeterministicModelProvider(
    responses=[
        text_response("Research findings"),
        text_response("Reviewed final answer"),
    ]
)
agent_runner = AgentRunner(ModelProviderRegistry([provider]))
model = Model(name="offline", provider="deterministic")
team = Team(
    name="editorial",
    policy=TeamPolicy(
        max_rounds=1,
        max_delegations=2,
        max_depth=2,
    ),
    members=[
        TeamMember(
            name="researcher",
            role=TeamRole.RESEARCHER,
            agent=Agent(
                name="researcher",
                instructions="Research the subject.",
                model=model,
            ),
        ),
        TeamMember(
            name="reviewer",
            role=TeamRole.REVIEWER,
            agent=Agent(
                name="reviewer",
                instructions="Review the response.",
                model=model,
            ),
        ),
    ],
)


async def main() -> None:
    result = await TeamRunner(agent_runner).run(team, "Explain typed workflows")
    print([turn.member for turn in result.turns])
    print(result.final_response.text_content)


asyncio.run(main())
```

Implement `DelegationStrategy` to select members dynamically. The runner enforces
maximum rounds, delegation count, and recursion depth regardless of strategy output.

## 20. Storage

The reference storage backend copies values at read/write boundaries and isolates
data by tenant and project:

```python
import asyncio

from praxium import Conversation, InMemoryStorage, Message, TenantContext


async def main() -> None:
    storage = InMemoryStorage()
    tenant = TenantContext(tenant_id="tenant-1", project_id="project-1")
    conversation = Conversation(messages=[Message.user("Hello")])

    await storage.save_conversation(tenant, conversation)
    restored = await storage.load_conversation(tenant, conversation.id)

    assert restored == conversation
    assert restored is not conversation


asyncio.run(main())
```

The same backend stores execution summaries, ordered events, and append-only audit
records. It is a single-process development backend, not durable production
storage. See [POSTGRESQL.md](POSTGRESQL.md) and
[`migrations/0001_core.sql`](../migrations/0001_core.sql) for the production schema.

## 21. Serialization and graph visualization

All public values are Pydantic models:

```python
json_payload = result.model_dump_json(indent=2)
restored = type(result).model_validate_json(json_payload)
```

Export a graph as Mermaid:

```python
print(graph.to_mermaid())
```

Or use the CLI with an import target:

```bash
praxium graph examples.server:graph --output mermaid
praxium graph examples.server:graph --output json
```

Node handlers are intentionally excluded from serialized graphs. JSON preserves the
graph structure, metadata, limits, and routes; application code or a trusted plugin
must restore executable handlers.

## 22. Testing applications

Use the deterministic provider and in-memory backends to keep tests offline:

```python
import pytest

from praxium import ExecutionStatus, Runtime


@pytest.mark.asyncio
async def test_workflow() -> None:
    result = await Runtime().run(graph, {"score": 0.9})

    assert result.status == ExecutionStatus.COMPLETED
    assert result.state.data["decision"] == "accepted"
    assert [run.node_id for run in result.node_runs] == ["classify", "accept"]
```

Recommended project checks:

```bash
ruff format --check .
ruff check .
mypy src
pytest --cov=praxium --cov-branch
```

Praxium's own suite includes cancellation, retry, timeout, checkpoint resume,
tenant-isolation, API, plugin, RAG, tool, and multi-agent contract tests.

## 23. Deployment

Build and run the reference container:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Or build only the API image:

```bash
docker build -f deploy/Dockerfile -t praxium:0.1.0 .
docker run --rm -p 8000:8000 praxium:0.1.0
```

The reference image runs as a non-root user and exposes health/readiness endpoints.
See [DEPLOYMENT.md](DEPLOYMENT.md) for PostgreSQL, Redis, Kubernetes, cloud, secrets,
shutdown, and production-hardening guidance.

## 24. Current v0.1 boundaries

The following distinctions prevent surprises:

- Graph execution is sequential and conditional. Parallel fan-out/fan-in and
  distributed workers are roadmap items.
- In-memory checkpoint, storage, memory, and RAG implementations are development
  and contract-test backends.
- Provider feature support still depends on the selected model and account; Praxium
  does not pretend a text-only model supports tools, vision, or embeddings.
- HTTP authentication and enterprise authorization adapters are not built in yet.
- Synchronous Python cannot be forcibly cancelled; sandbox untrusted work outside
  the process.
- Visual workflow studio and enterprise identity/billing are later milestones.

See [DELIVERY_STATUS.md](DELIVERY_STATUS.md) for the exact delivered/deferred matrix
and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full roadmap.

## 25. Where to go next

- Run [examples/quickstart.py](../examples/quickstart.py).
- Run [examples/agent_tools.py](../examples/agent_tools.py).
- Run [examples/model_providers.py](../examples/model_providers.py) offline, or set
  `PRAXIUM_PROVIDER` and `PRAXIUM_MODEL` to exercise a configured production provider.
- Run [examples/approval_resume.py](../examples/approval_resume.py).
- Start [examples/server.py](../examples/server.py) through `praxium serve`.
- Read [ARCHITECTURE.md](ARCHITECTURE.md) before implementing provider or storage
  adapters.
- Use `praxium doctor` when a development or deployment environment behaves
  differently from local tests.
