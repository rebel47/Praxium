# Engineering Constitution

This repository implements a provider-neutral AI agent framework.

## Architecture

- Dependencies point inward: integrations -> runtime/agents -> graph and services
  -> core. Core must not import web, CLI, database, or provider SDK packages.
- Public boundary values are Pydantic models. Runtime-only synchronization objects
  may be private attributes or internal classes.
- All I/O APIs are asynchronous. Synchronous user callables are moved to a worker
  thread by adapters.
- Optional integrations must remain importable only when their extras are installed.
- Registries are explicit; importing a module must not mutate a global registry.

## Runtime

- Every execution, node attempt, event, checkpoint, tool call, and model call is
  correlated by stable IDs.
- State transitions are copied, versioned, serializable, and deterministic.
- Conflicting state writes require a declared merge policy.
- Cancellation and timeout code paths must cancel and await child tasks.
- Retry behavior must be bounded and cancellation-aware.
- User code exceptions are wrapped in typed framework errors without discarding
  the original exception chain.

## Security

- Never log or serialize secrets, credentials, authorization headers, or raw
  provider configuration.
- Tools declare permissions and run within timeout and output-size limits.
- Storage and memory APIs carry tenant and namespace context.
- Do not implement cryptography; use vetted provider boundaries.

## Quality

- New behavior requires tests, including failure paths.
- Core, graph, and runtime target 95% branch coverage; the repository targets 85%.
- Use Ruff formatting/linting and strict mypy for framework-owned code.
- Public APIs require docstrings and must be re-exported deliberately.
- Examples must be deterministic and must not require paid services by default.
- Do not add placeholder methods that silently succeed. Raise a typed
  `UnsupportedFeatureError` when a contract is deliberately not implemented.

## Compatibility

- Public models use explicit schema versions where persisted.
- Breaking public API or serialized-schema changes require a changelog entry and
  migration guidance.
- Provider-specific fields live in metadata or adapter packages, never in core
  contracts.

