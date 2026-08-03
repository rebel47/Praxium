# Architecture

## Runtime view

Praxium separates persisted data contracts from executable services.
Pydantic models describe messages, state, graphs, tools, model requests, events,
memory, documents, plugins, and results. Services such as `Runtime`, registries,
stores, and provider adapters own locks, queues, SDK clients, and other live
resources.

```text
Application / FastAPI / OpenAI compatibility / CLI
                       |
        Agents / multi-agent / plugins
                       |
       Graph -------- Runtime -------- Checkpoints
          \              |                  /
           Core models --+-- ordered events
                       |
        Memory / storage / RAG protocols
```

Core imports no optional framework. Importing `praxium` therefore does not
require FastAPI, a database driver, OpenTelemetry, or a provider SDK. The HTTP
factory imports FastAPI only when called.

## Core data lifecycle

1. An application registers an `Agent` or validated `Graph` under a unique name.
2. A request is normalized into a conversation or initial `State`.
3. The runtime creates an execution ID, cancellation token, deadline, and ordered
   event emitter.
4. Before invoking user code, the runtime copies state so a failed node cannot
   mutate committed state by reference.
5. A node result is normalized into a `StatePatch`, optional output, optional route,
   and optional suspension.
6. State applies the patch into a new version and records field-level changes.
7. The selected edge and new state are checkpointed after the node commits.
8. Completion, failure, timeout, cancellation, or suspension is emitted as a
   terminal run event and returned as a typed `ExecutionResult`.

## Graph rules

- A graph has one entrypoint and one or more finish points.
- All nodes must be reachable and edge endpoints must exist.
- More than one outgoing edge requires unique route labels.
- Cycles are allowed only when at least one member declares `max_visits > 1`.
- Runtime step limits remain authoritative even for structurally bounded cycles.
- Finish points do not follow outgoing edges.
- Handlers are excluded from serialization. A serialized graph captures structure;
  executable handlers must be restored by application code or a plugin registry.

The v0.1 scheduler is deterministic and follows one selected edge at a time.
Parallel branch isolation/fan-in is planned for v0.2, after merge semantics and
checkpoint boundaries receive concurrency stress testing.

## Failure semantics

- Invalid graphs fail before a run starts.
- Handler exceptions become `NodeExecutionError` with a safe type/code/context.
- Retries are bounded, evented, exponentially delayed, jitterable, and
  cancellation-aware.
- Node or run deadlines cancel and await active handler tasks before returning.
- External cancellation stops admission of new work and interrupts an active
  async handler.
- Telemetry sink failures are isolated from execution results.
- Checkpoint resume verifies graph ID, version, and structural fingerprint.
- Resume begins after the last committed node, preventing duplicate committed
  side effects.

Synchronous handlers and tools run in `asyncio.to_thread`. Python cannot forcibly
stop a running thread, so security-sensitive or untrusted synchronous tools still
require a process/container sandbox supplied by an integration.

## Extension contracts

### Models

`ModelProvider` supports completion, streaming, and embeddings. Requests use a
provider-neutral capability and tool schema. Production adapters translate OpenAI-
compatible, Anthropic, Gemini/Vertex, and Bedrock Converse wire protocols; named
presets cover Azure OpenAI, Groq, Together, OpenRouter, Kimi, GLM, Ollama, and
Hugging Face. `CustomModelProvider` handles non-standard APIs, while
`DeterministicModelProvider` supplies offline fixtures and deterministic embeddings.

### Tools

`Tool.from_callable` resolves type annotations into JSON Schema. Execution validates
inputs, checks declared permissions, enforces timeout and maximum serialized output,
moves synchronous work off-loop, and can redact sensitive output.

### Middleware

`MiddlewarePipeline` wraps a typed context in registration order. Each middleware
can inspect or replace the request/result only through `call_next`. Redaction and
timeout middleware are included.

### Memory and RAG

All operations require tenant and namespace. The memory reference backend supports
version history, TTL, filtering, lexical retrieval, branches, and copy isolation.
The RAG backend preserves exact source offsets, metadata, collection/namespace,
deterministic chunk IDs, idempotent replacement, lexical/dense fusion, and
citations.

### Plugins

Discovery reads entry-point metadata without importing plugin code. Loading is an
explicit trust action. Manifests declare framework compatibility, capabilities,
and permissions; registry conflicts never overwrite silently.

## HTTP surfaces

`Application.register()` exposes a uniform execution surface. The optional FastAPI
factory adds:

- `GET /health`, `GET /ready`, and `GET /components`;
- `POST /runs/{name}` and graph event streaming at `/runs/{name}/stream`;
- OpenAI-shaped `/v1/models`, `/v1/chat/completions`, `/v1/responses`, and
  `/v1/embeddings` routes.

The v0.1 generated inbound compatibility endpoints use server-sent events but
coalesce an agent's final response into one text delta. Outbound model providers
preserve provider token chunks. Inbound request-defined tool schemas are scheduled
for v0.4 contract work and currently return an explicit unsupported-input error.

## Persistence boundaries

`InMemoryCheckpointStore`, `InMemoryStorage`, `InMemoryMemoryStore`, and
`InMemoryRagIndex` are reference backends and contract-test fixtures. They are
concurrency-safe inside one process and return copies, but are not durable or
distributed. PostgreSQL tables and indexes are specified in
`migrations/0001_core.sql`; the async PostgreSQL adapter is a v0.3 deliverable.

## Security boundaries

- Tenant identifiers are explicit at storage, memory, RAG, API, and execution
  boundaries.
- Secret configuration belongs in private provider adapter fields and is excluded
  from serialization, representation, safe errors, and event metadata.
- Tools need explicit permissions and untrusted code needs an external sandbox.
- HTTP authentication and authorization hooks must be installed before an operator
  exposes the server publicly.
- Alpha components are not a security boundary by themselves.
