# Praxium: Implementation Plan

Status: approved baseline for greenfield implementation  
Last updated: 2026-08-03

## 1. Product definition

Build a typed, asynchronous, observable, reproducible, deployable, and extensible
Python framework for creating and operating AI agents and graph workflows. The
framework combines four concerns without coupling them:

1. A small typed domain model for agents, messages, tools, models, memory, state,
   graphs, and executions.
2. A deterministic graph runtime with streaming, retries, cancellation,
   checkpointing, parallelism, and human-in-the-loop suspension.
3. Integration layers for storage, retrieval, model providers, APIs, plugins, and
   telemetry.
4. Production surfaces: CLI, FastAPI application, OpenAI-compatible endpoints,
   deployment assets, evaluation, and operational documentation.

The initial implementation is a coherent production-oriented foundation, not an
attempt to implement every cloud connector or visual editor in one release.
Provider-specific features are exposed through stable protocols so they can be
added independently.

## 2. Non-negotiable engineering rules

- Python 3.11+ and `asyncio` are the execution baseline.
- Public domain objects and public input/output schemas are Pydantic models.
- `core` has no dependency on FastAPI, database drivers, provider SDKs, or the CLI.
- Runtime behavior is accessed through protocols; integrations depend inward.
- Every execution has a stable ID and every emitted event has a sequence number.
- User-visible execution state can be serialized and replayed.
- Parallel writes are resolved explicitly; silent last-writer-wins behavior is not
  allowed for conflicting values.
- Cancellation, timeout, retry, and cleanup paths are first-class behavior.
- Provider SDKs, databases, and web frameworks are optional dependencies.
- Secrets never appear in model representations, logs, traces, or serialized
  configuration.
- Public APIs are documented, typed, and covered by tests.
- Compatibility guarantees are defined before the first stable release.

## 3. Architecture and dependency direction

```text
                     CLI             HTTP / OpenAI API
                      |                    |
                      +------ application-+
                                   |
 providers  plugins  deployment -> integrations
      \        |                       /
       +-------+----------------------+
                       |
            runtime / graph / agents
                       |
          middleware / memory / RAG
                       |
              observability ports
                       |
                     core
```

Dependencies only point downward. `core` defines data and protocols. The graph
package validates workflow structure. The runtime owns execution. Higher layers
adapt these primitives to model SDKs, databases, HTTP, CLI, and deployment
environments.

### Threading and async model

- The event loop owns scheduling and orchestration.
- Nodes, tools, memory, storage, and model calls expose async interfaces.
- Blocking adapters must use an injected executor and never block the event loop.
- Each run has a cancellation scope, deadline, concurrency semaphore, and event
  stream.
- Parallel branches receive isolated state snapshots and merge at a join.
- Framework code does not create unmanaged background threads.

### Event model

Events use a common envelope containing `event_id`, `execution_id`,
`sequence`, `timestamp`, `kind`, and typed payload. Core event kinds include run,
node, edge, tool, model, memory, checkpoint, retry, stream, approval, and error
lifecycle events. Event sinks are non-authoritative: failure to export telemetry
must not corrupt a run.

### Configuration model

Configuration loads in this precedence order:

1. Explicit Python arguments.
2. Environment variables prefixed with `PRAXIUM_`.
3. A project configuration file.
4. Typed defaults.

Configuration objects are immutable after application startup. Secret values use
Pydantic secret types and are redacted during serialization.

## 4. Target repository layout

```text
.
|-- AGENTS.md
|-- pyproject.toml
|-- README.md
|-- CHANGELOG.md
|-- LICENSE
|-- src/praxium/
|   |-- core/
|   |-- graph/
|   |-- runtime/
|   |-- agents/
|   |-- models/
|   |-- tools/
|   |-- middleware/
|   |-- memory/
|   |-- storage/
|   |-- rag/
|   |-- observability/
|   |-- plugins/
|   |-- api/
|   `-- cli/
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- contract/
|   `-- performance/
|-- examples/
|-- benchmarks/
|-- migrations/
|-- deploy/
`-- docs/
```

## 5. Implementation slices

Each slice must leave the repository runnable and tested. A slice is complete only
when its code, tests, documentation, error behavior, and exported API agree.

### Slice 01: repository constitution and tooling

Deliver:

- `AGENTS.md` with dependency, API, security, testing, and documentation rules.
- `pyproject.toml` with minimal core dependencies and optional dependency groups.
- Source layout, package metadata, semantic version source, lint, type-check, and
  test configuration.
- Public README, contribution workflow, security policy, and changelog.

Acceptance:

- Editable installation succeeds on Python 3.11+.
- Importing the root package does not require optional integrations.
- Test, lint, and type-check commands are documented.

### Slice 02: shared types and identifiers

Deliver:

- UTC timestamp helpers and opaque IDs for executions, graphs, nodes, messages,
  checkpoints, conversations, tools, and events.
- Enums for roles, statuses, event types, route decisions, and merge strategies.
- Base model configuration for strict validation, serialization, and safe reprs.
- Error hierarchy with stable machine-readable error codes.

Acceptance:

- IDs round-trip through JSON.
- All time values are timezone-aware.
- Errors serialize without leaking secret or provider internals.

### Slice 03: messages, prompts, responses, and usage

Deliver:

- Typed message parts for text, JSON, tool calls, tool results, and references.
- `Message`, `Conversation`, `Prompt`, `Response`, token usage, finish reason, and
  streaming delta models.
- Provider-neutral structured output metadata.

Acceptance:

- Conversation order and tool-call relationships validate.
- Responses and stream chunks share a compatible aggregation contract.

### Slice 04: state and execution context

Deliver:

- Serializable state container with version, namespace, data, and change history.
- Immutable execution context carrying run metadata, dependencies, deadlines, and
  cancellation information.
- Merge policies: replace, append, numeric add, set union, recursive merge, and
  custom resolver.
- Conflict detection and structured state diffs.

Acceptance:

- Parallel state changes merge deterministically.
- Conflicting writes fail with an actionable error unless a policy is supplied.

### Slice 05: model abstraction

Deliver:

- `Model` configuration and `ModelProvider` protocol.
- Request, response, stream, embedding, tool-call, structured-output, and
  capability models.
- Deterministic fake model for tests and examples.

Acceptance:

- Non-streaming and streaming provider contracts are interchangeable.
- Capability checks fail before a provider call.

### Slice 06: tool abstraction and security

Deliver:

- Typed `Tool`, `ToolContext`, `ToolResult`, schema inference, and execution
  protocol.
- Permission declarations, confirmation policy, timeout, output-size limit, and
  sensitive-output redaction.
- Callable adapter and tool registry.

Acceptance:

- Inputs validate from generated JSON Schema.
- Permission denial and timeout produce typed tool results and events.

### Slice 07: agent composition

Deliver:

- Agent definition linking instructions, model, tools, memory policy, output type,
  middleware, limits, and dependencies.
- Agent request/result models and a testable agent runner contract.
- Dependency injection by typed key rather than global state.

Acceptance:

- An agent can perform a model/tool loop with a configured maximum step count.
- Tool results are correlated with their originating calls.

### Slice 08: graph schema and builder

Deliver:

- Typed nodes, edges, routes, entry/terminal definitions, subgraph references, and
  graph metadata.
- Fluent builder plus explicit declarative construction.
- Validation for unreachable nodes, missing targets, invalid joins, duplicate IDs,
  unbounded cycles, and incompatible state contracts.
- Mermaid export.

Acceptance:

- Invalid graphs fail before execution with all discovered validation issues.
- Valid graphs serialize and recreate equivalent structure.

### Slice 09: scheduler and node execution

Deliver:

- Async scheduler with ready queue, priority, dependency counting, concurrency
  limits, and node executors.
- Sequential, conditional, loop, dynamic, and subgraph execution.
- Per-node/run timeouts and cooperative cancellation.

Acceptance:

- Scheduling is deterministic for equal-priority ready nodes.
- Cancellation prevents new work and cleans up active work.

### Slice 10: parallelism and coordination

Deliver:

- Fan-out/fan-in, map/reduce, branch isolation, join policy, and bounded parallel
  maps.
- Message passing channels for supervisor/worker patterns.
- Deadlock and non-progress detection.

Acceptance:

- Parallel work respects concurrency limits.
- Branch output ordering is stable regardless of completion order.

### Slice 11: resilience

Deliver:

- Retry policy with exponential backoff, jitter, retryable classifications, and
  attempt limits.
- Circuit-breaker and rate-limit primitives.
- Compensating actions for rollback-capable nodes.
- Backpressure for event and stream consumers.

Acceptance:

- Cancellation interrupts retry waits.
- Compensation executes in reverse successful-node order.

### Slice 12: checkpoints and replay

Deliver:

- Checkpoint model and store protocol.
- In-memory checkpoint store and atomic save/load/list/delete semantics.
- Resume, fork, replay, time-travel, and human-approval suspension tokens.

Acceptance:

- A suspended run resumes without repeating committed nodes.
- Incompatible graph/checkpoint versions fail safely.

### Slice 13: events, traces, and metrics

Deliver:

- Ordered event bus and sink protocol.
- In-memory trace collector, structured logging adapter, metric instruments, and
  OpenTelemetry adapter boundary.
- Cost, tokens, latency, retry, queue, and failure measurements.

Acceptance:

- Every node lifecycle has paired start/terminal events.
- Exporter failure cannot change execution results.

### Slice 14: middleware pipeline

Deliver:

- Ordered async middleware for run, node, model, and tool boundaries.
- Built-ins for logging, metrics, retry, cache, budgets, redaction, output
  validation, injection heuristics, and circuit breaking.
- Short-circuiting and stream interception contracts.

Acceptance:

- Ordering is explicit and observable.
- Middleware can alter a response only through its declared boundary contract.

### Slice 15: memory system

Deliver:

- Memory record, query, result, namespace, version, retention, and policy models.
- Working, conversation, long-term, semantic, entity, tool, summary, and
  hierarchical memory interfaces.
- In-memory store, hybrid ranking hooks, compression/summarization hooks,
  expiration, branch, and replay.

Acceptance:

- Tenant and namespace isolation are mandatory in every memory query.
- Expired or superseded records cannot leak into default retrieval.

### Slice 16: storage and PostgreSQL specification

Deliver:

- Unit-of-work and repository protocols for conversations, executions, events,
  checkpoints, memory, and audit records.
- In-memory transactional reference implementation.
- PostgreSQL schema, migrations, indexes, JSONB boundaries, optimistic versioning,
  retention, partition, backup, and scaling documentation.
- Optional async PostgreSQL adapter.

Acceptance:

- Storage contracts pass against both in-memory and configured PostgreSQL backends.
- Concurrent checkpoint writes use optimistic concurrency.

### Slice 17: retrieval and indexing

Deliver:

- Documents, chunks, collections, namespaces, metadata filters, citations, and
  index job models.
- Chunker, embedder, sparse/dense retriever, fusion, reranker, compressor, and
  citation protocols.
- In-memory keyword/vector implementation and incremental upsert/delete/version.
- pgvector schema and adapter boundary.

Acceptance:

- Re-indexing is idempotent.
- Retrieved context preserves source and character/page provenance.

### Slice 18: plugin system

Deliver:

- Plugin manifest, capability, compatibility, dependency, lifecycle, and security
  models.
- Explicit registry and Python entry-point discovery.
- Namespaced registration for models, tools, nodes, middleware, memory, storage,
  deployers, providers, CLI commands, and visual metadata.

Acceptance:

- Importing a plugin cannot silently replace an existing registration.
- Incompatible plugins are reported without preventing core startup.

### Slice 19: application registry and generated API

Deliver:

- Framework `Application` with `register(agent_or_graph)`.
- Optional FastAPI factory with run, stream, schema, health, readiness, and metrics
  endpoints.
- Authn/authz, error mapping, request IDs, validation, versioning, rate-limit
  hooks, and server-sent event streaming.

Acceptance:

- Core imports work when FastAPI is absent.
- OpenAPI accurately represents registered inputs, outputs, and errors.

### Slice 20: OpenAI compatibility

Deliver:

- `/v1/models`, `/v1/chat/completions`, `/v1/responses`, and `/v1/embeddings`.
- Streaming events, function/tool calls, JSON mode, structured output, usage, and
  compatibility middleware.
- Strict request translation and provider-neutral response translation.

Acceptance:

- Contract tests cover representative OpenAI SDK request shapes.
- Unsupported parameters return explicit compatibility errors.

### Slice 21: CLI and developer experience

Deliver:

- Commands: `create`, `init`, `serve`, `dev`, `deploy`, `graph`, `dashboard`,
  `benchmark`, `test`, `evaluate`, `memory`, `plugins`, and `doctor`.
- Safe templates, configuration discovery, useful diagnostics, and machine-readable
  output mode.

Acceptance:

- `doctor` detects Python, configuration, plugin, and optional-dependency issues.
- CLI failures have stable exit codes and no stack trace unless debug is enabled.

### Slice 22: multi-agent runtime

Deliver:

- Supervisor, planner, worker, reviewer, critic, and executor role patterns using
  ordinary graphs and agents.
- Shared/private memory scopes, typed delegation messages, budgets, depth limits,
  voting, consensus, negotiation, and reflection policies.

Acceptance:

- Delegation cannot exceed run budget or recursion depth.
- Private memory is inaccessible to peers unless explicitly shared.

### Slice 23: evaluation and testing framework

Deliver:

- Dataset, case, scorer, evaluator, experiment, result, and baseline models.
- Exact, semantic, tool-selection, citation, latency, cost, and custom scorers.
- Golden snapshots, deterministic model/tool doubles, trace replay, regression
  comparison, and CI output.

Acceptance:

- Evaluation runs are reproducible from serialized inputs and model fixtures.
- Regressions can fail CI using declared thresholds.

### Slice 24: enterprise boundaries

Deliver:

- Organization, project, principal, role, permission, API key, quota, usage,
  audit, and tenant context models.
- Auth provider protocols for OAuth/OIDC/SAML and a policy enforcement point.
- Envelope-encryption and secret-provider boundaries; no home-grown cryptography.
- Billing-event boundary and usage aggregation.

Acceptance:

- Tenant identity is required at all storage and execution boundaries.
- Security-relevant mutations produce immutable audit events.

### Slice 25: deployment and operations

Deliver:

- Production container, local Compose stack, Kubernetes reference manifests, and
  cloud deployment notes for AWS, Azure, GCP, Railway, Fly.io, Render, and
  DigitalOcean.
- API/worker split, Redis-compatible queue boundary, PostgreSQL, graceful shutdown,
  autoscaling signals, liveness/readiness, migrations, backups, and CI/CD gates.

Acceptance:

- Container runs as a non-root user and has a health check.
- Shutdown stops admission, drains work, checkpoints resumable runs, and exits
  within a configurable grace period.

### Slice 26: visual workflow protocol

Deliver:

- Framework-neutral graph document schema for nodes, ports, edges, layouts,
  subgraphs, breakpoints, and visual metadata.
- Import/export between the document schema and Python graph definitions.
- Debugger protocol for live events, pause, resume, step, inspect, and replay.
- A separate web application is deferred until the protocol stabilizes.

Acceptance:

- Python -> document -> Python preserves executable semantics.
- Unknown visual metadata survives round trips.

### Slice 27: examples, benchmarks, and documentation

Deliver:

- Quickstart agent, conditional graph, parallel research workflow, RAG, human
  approval, resume, API server, plugin, and multi-agent examples.
- Scheduler, state merge, serialization, retrieval, and middleware benchmarks.
- Concept, tutorial, how-to, API reference, operations, and security docs.

Acceptance:

- Examples run in CI and avoid paid external services by default.
- Benchmark results include environment and version metadata.

## 6. Milestones and development order

| Milestone | Slices | Outcome | Exit gate |
|---|---:|---|---|
| M0 Foundation | 01-04 | Installable typed core | Unit tests, lint, type-check |
| M1 Executable Core | 05-09 | Agents and graphs execute | Deterministic runtime tests |
| M2 Durable Runtime | 10-14 | Parallel, resumable, observable runs | Failure/recovery integration tests |
| M3 Knowledge Layer | 15-17 | Memory, persistence, RAG | Backend contract tests |
| M4 Extensible Service | 18-21 | Plugins, HTTP, OpenAI API, CLI | API/CLI contract tests |
| M5 Advanced Platform | 22-26 | Multi-agent, eval, enterprise, deploy, studio protocol | Security and end-to-end tests |
| M6 Release | 27 | Examples, benchmarks, complete docs | Release checklist and compatibility review |

## 7. Testing order and quality gates

For each slice:

1. Model validation and serialization unit tests.
2. Protocol contract tests using deterministic fakes.
3. Happy-path component tests.
4. Timeout, cancellation, retry, invalid-input, and partial-failure tests.
5. Concurrency and ordering tests where relevant.
6. Integration tests for optional adapters.
7. Documentation/example smoke tests.

Repository gates:

- Ruff formatting and linting.
- Strict mypy for framework modules, with documented exceptions only at third-party
  adapter boundaries.
- Pytest with branch coverage; target 95% for core/graph/runtime and 85% overall.
- Dependency vulnerability and secret scans.
- Build wheel/sdist and test installation in a clean environment.
- Python 3.11, 3.12, and 3.13 CI matrix.
- PostgreSQL integration jobs are isolated from unit tests.
- Benchmark regression thresholds are advisory until v1.0, enforced thereafter.

## 8. Documentation order

1. Architecture decision records and terminology.
2. Public object and protocol reference alongside implementation.
3. Quickstart as soon as M1 is usable.
4. Runtime behavior, failure modes, state merge, and checkpoint guides in M2.
5. Integration guides in M3-M4.
6. Security, tenancy, operations, and deployment before any production claim.
7. Migration and compatibility guides for every breaking release.

## 9. Version roadmap

### v0.1: typed executable foundation

- Slices 01-09.
- In-memory defaults and deterministic testing utilities.
- Sequential and conditional workflows, bounded loops, basic streaming, retries,
  timeouts, and events.

### v0.2: durable concurrency

- Slices 10-14.
- Parallelism, checkpoint/resume, middleware, trace collection, and robust
  cancellation.

### v0.3: knowledge and persistence

- Slices 15-17.
- Memory, storage protocols, PostgreSQL/pgvector adapters, and RAG pipeline.

### v0.4: application platform

- Slices 18-21.
- Plugins, generated FastAPI, OpenAI compatibility, and CLI.

### v0.5: multi-agent and evaluation

- Slices 22-23.
- Reusable collaboration patterns and reproducible evaluation.

### v0.6: enterprise and deployment preview

- Slices 24-26.
- Tenant/policy boundaries, production deployment references, and visual protocol.

### v1.0: stable core

- Slice 27 and all accumulated hardening.
- Stable public core, graph, runtime, tool, model, memory, event, and plugin APIs.
- Published compatibility, security, support, and deprecation policies.

### v1.x

- Additional first-party model, tool, vector-store, observability, cloud, and
  identity adapters.
- Performance improvements that do not break public contracts.
- Visual workflow builder based on the stabilized document/debugger protocols.

### v2.0

- Changes requiring evidence from v1 production use: distributed scheduling,
  cross-region durable execution, or major state/checkpoint format evolution.
- v1 migration tooling and at least one full minor-release deprecation window.

## 10. Cross-cutting acceptance matrix

| Concern | Required behavior |
|---|---|
| Typing | Public data crosses boundaries through Pydantic models |
| Async | No blocking I/O on the event loop |
| Observability | Ordered lifecycle events with correlation IDs |
| Reproducibility | Inputs, versions, config fingerprint, events, and state are serializable |
| Security | Least privilege, redaction, explicit tool permissions, tenant isolation |
| Reliability | Timeout, cancellation, retry, cleanup, and idempotency are tested |
| Extensibility | Protocol plus registry; optional integration dependencies stay isolated |
| Deployability | Health, readiness, graceful shutdown, config, migrations, and metrics |
| Compatibility | Versioned schemas and explicit unsupported-feature errors |
| Documentation | Every public surface has a reference and at least one runnable example |

## 11. Primary risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creates shallow implementations | Ship vertical milestones; no placeholder behavior in exported APIs |
| Core becomes coupled to providers | Enforce package dependency tests and optional adapter groups |
| Parallel state is nondeterministic | Isolated snapshots, explicit merge policies, stable ordering |
| Resume duplicates side effects | Node idempotency keys and commit/checkpoint boundaries |
| Plugin import is unsafe | Explicit discovery, compatibility checks, conflict rejection, trust policy |
| Telemetry slows or breaks runs | Bounded queues, backpressure policy, exporter isolation |
| API compatibility drifts | Fixture-based contract tests against documented request/response schemas |
| Tenant data crosses boundaries | Mandatory tenant context and backend contract tests |
| Provider churn leaks inward | Provider-neutral capability model and translation adapters |
| Visual editor dictates runtime | Stabilize a neutral document/debugger protocol first |

## 12. Initial build boundary

The first coding pass will deliver the v0.1 foundation end to end: repository
tooling, core models, model/tool/agent abstractions, graph builder and validation,
an async event-producing runtime with conditional routing, bounded looping,
retries, timeouts, cancellation, streaming, an in-memory checkpoint boundary,
tests, and runnable examples. Later milestones remain specified here and receive
package boundaries now, but are not represented by fake production implementations.

