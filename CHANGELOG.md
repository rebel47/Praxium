# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

## [0.1.1] - 2026-08-03

### Changed

- Rewrote the README as a PyPI-first guide with installation extras, runnable
  graph and tool-using agent examples, provider configuration, CLI usage, and a
  contributor-only development section.
- The generated FastAPI application now reads its displayed version from the
  package version module instead of duplicating a release number.

## [0.1.0] - 2026-08-03

### Added

- Greenfield architecture and implementation plan.
- Initial typed core, graph, runtime, extension interfaces, tests, and examples.
- Universal model provider layer with unrestricted model identifiers, an
  OpenAI-compatible adapter, native Anthropic, Gemini, Amazon Bedrock, and Google
  Vertex AI adapters, Azure OpenAI/Groq/Together/OpenRouter/Kimi/GLM/Ollama/Hugging
  Face presets, renewable OAuth tokens, custom provider callables, token streaming,
  tool calls, structured output, embeddings, typed failures, documentation, and
  contract tests.
- Streaming text now preserves leading and trailing whitespace.
