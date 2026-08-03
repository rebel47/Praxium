# Delivery status

This file reconciles the current repository against `IMPLEMENTATION_PLAN.md`.
“Foundation” means a working typed contract/reference implementation exists, but
the production adapters or advanced behavior remain assigned to a later milestone.

## Delivered in v0.1

| Slice | Status | Evidence |
|---|---|---|
| 01 Repository/tooling | Complete | Packaging, constitution, CI, docs, CLI doctor |
| 02 Shared types/IDs | Complete | Opaque IDs, UTC, enums, typed safe errors |
| 03 Messages/prompts/responses | Complete | Multipart messages, tool correlation, usage, deltas |
| 04 State/context | Complete | Versioned state, history, merge policies, conflicts, cancellation |
| 05 Model abstraction | Complete | Completion/stream/embed protocol, registry, deterministic provider, compatible universal adapter, native Anthropic/Gemini/Bedrock/Vertex, Azure/Groq/Together/OpenRouter/Kimi/GLM/Ollama/Hugging Face presets, custom callable adapter |
| 06 Tool abstraction | Complete | Schema inference, validation, permissions, timeout, size limit, redaction |
| 07 Agent composition | Complete | Bounded model/tool loop and correlated results |
| 08 Graph schema/builder | Complete | Fluent builder, whole-graph validation, cycles, Mermaid, fingerprint |
| 09 Scheduler/node execution | Complete for v0.1 | Deterministic sequential/conditional routing, loops, retries, deadlines |
| 12 Checkpoints/replay | v0.1 complete | Atomic memory store, save after commit, suspension and compatible resume |
| 13 Events/traces | v0.1 complete | Ordered event envelopes, isolated sinks, memory collector, stream transport |
| 14 Middleware | Foundation | Ordered pipeline, function adapter, timeout and recursive redaction |
| 15 Memory | Foundation | Tenant/namespace isolation, versions, TTL, lexical search, branching |
| 16 Storage | Foundation | Tenant-aware protocols/reference store and PostgreSQL/pgvector migration |
| 17 Retrieval | Foundation | Offset-preserving chunks, idempotent replacement, hybrid search, citations |
| 18 Plugins | Foundation | Manifest compatibility, metadata-only discovery, explicit load, conflicts |
| 19 Generated API | Foundation | Registry, health/readiness/components, runs, graph SSE, OpenAPI generation |
| 20 OpenAI compatibility | Foundation | Models, chat, responses, embeddings, SSE shapes and explicit limitations |
| 21 CLI | Foundation | Doctor, graph render/export, plugin discovery, and serve |
| 22 Multi-agent | Foundation | Typed teams, roles, delegation strategy, budgets, depth, deterministic runner |
| 25 Deployment | Foundation | Non-root image, Compose topology, Kubernetes baseline, operations guide |
| 27 Tests/docs/benchmarks | v0.1 complete | 67 tests, 86% branch-aware coverage, examples, benchmark, architecture docs |

## Deliberately deferred to roadmap milestones

- Parallel branch isolation, fan-out/fan-in, map/reduce, dynamic nodes, subgraphs,
  compensation rollback, distributed queues, circuit breaking, and backpressure
  are v0.2 runtime work (slices 10-11).
- PostgreSQL/pgvector async adapters, migrations runner, sparse BM25 provider,
  reranking, compression, and incremental background indexing are v0.3 work.
- Request-defined OpenAI tools, authentication, authorization, rate-limit
  middleware, metrics export, and the remaining CLI command
  families are v0.4 work.
- Evaluation datasets/scorers/experiments are slice 23.
- Organization/RBAC/SSO/SAML/OIDC, quotas, billing, immutable production audit
  adapters, and envelope-encryption integrations are slice 24.
- A distributed worker drain controller and production cloud modules remain in
  slice 25.
- The visual workflow document/debugger protocol and web studio are slice 26.

No exported method silently pretends to provide these deferred capabilities. The
OpenAI adapter rejects request-defined tools explicitly, the CLI exposes only
implemented commands, and documentation labels reference stores as single-process
development/contract backends.

## Verification snapshot

Verified on Python 3.11.8:

- `ruff format --check .`
- `ruff check .`
- `mypy src` (strict; 56 source files)
- `pytest --cov=praxium --cov-branch` (67 passed, 86% total coverage)
- dependency-free quickstart and CLI doctor

The CI matrix repeats these checks on Python 3.11, 3.12, and 3.13 and builds the
wheel/sdist in a clean hosted environment.
