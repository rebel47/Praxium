# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

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
