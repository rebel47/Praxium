# Model Providers

Praxium can run an agent with any model whose provider can be reached through one
of four integration paths:

1. `OpenAICompatibleProvider` for any OpenAI Chat Completions compatible endpoint;
2. native `AnthropicProvider`, `GeminiProvider`, `BedrockProvider`, and
   `VertexAIProvider` adapters for APIs with different authentication, message, and
   tool protocols;
3. named one-line presets for Azure OpenAI, Groq, Together, OpenRouter, Kimi, GLM,
   Ollama, and the Hugging Face router;
4. `CustomModelProvider` or the public `ModelProvider` protocol for every other
   proprietary API.

Praxium never keeps a model-name allowlist. `Model.name` is sent to the configured
provider unchanged. This means newly released, fine-tuned, namespaced, local, and
quantized model identifiers do not require a framework release.

Provider support and model capability are different. An adapter may support tool
calling while a particular model served by that provider does not. The provider is
the API connection; the model is the remote or local model identifier.

## Install

The provider package uses a small HTTP adapter rather than requiring every vendor
SDK:

```bash
python -m pip install "praxium[providers]"
```

For native AWS and Google Cloud authentication, install the matching extra or all
cloud extras:

```bash
python -m pip install "praxium[aws]"             # Bedrock + Boto3
python -m pip install "praxium[gcp,providers]"   # Vertex AI + ADC + HTTP
python -m pip install "praxium[cloud-providers]" # Azure Identity, AWS, GCP, HTTP
```

Provider modules are safe to import without making a network request. API keys are
read when a call is made and are never stored in the serializable `Model` object.

## Support matrix

| Provider | Factory | Chat | Stream | Tools | Structured output | Embeddings |
|---|---|---:|---:|---:|---:|---:|
| OpenAI / GPT | `ProviderFactory.openai()` | Yes | Yes | Yes | Yes, model-dependent | Yes |
| Anthropic / Claude | `ProviderFactory.anthropic()` | Yes | Yes | Yes, native blocks | Yes, model-dependent | Use another provider |
| Google Gemini | `ProviderFactory.gemini()` | Yes | Yes | Yes, native calls | Yes, model-dependent | Yes |
| Azure OpenAI | `ProviderFactory.azure_openai()` | Yes | Yes | Yes | Yes, deployment-dependent | Yes |
| Amazon Bedrock | `ProviderFactory.bedrock()` | Yes, Converse models | Yes | Yes, native blocks | Yes, model-dependent | Use a dedicated adapter |
| Google Vertex AI | `ProviderFactory.vertex_ai()` | Yes, Google publisher models | Yes | Yes, native calls | Yes, model-dependent | Yes |
| Groq | `ProviderFactory.groq()` | Yes | Yes | Yes, model-dependent | Yes, model-dependent | Not exposed by this preset |
| Together AI | `ProviderFactory.together()` | Yes | Yes | Yes, model-dependent | Yes, model-dependent | Yes |
| OpenRouter | `ProviderFactory.openrouter()` | Yes | Yes | Yes, model-dependent | Yes, model-dependent | Yes |
| Moonshot / Kimi | `ProviderFactory.kimi()` | Yes | Yes | Yes, model-dependent | Model-dependent | Not exposed by this preset |
| Zhipu / GLM | `ProviderFactory.glm()` | Yes | Yes | Yes, model-dependent | Model-dependent | Not exposed by this preset |
| Ollama | `ProviderFactory.ollama()` | Yes | Yes | Yes, model-dependent | Model-dependent | Yes |
| Hugging Face router | `ProviderFactory.huggingface()` | Yes | Yes | Model-dependent | Model-dependent | Chat router does not expose it |
| vLLM, LM Studio, LocalAI, or another compatible server | `openai_compatible(...)` | Yes | Yes | Server/model-dependent | Server/model-dependent | Endpoint-dependent |
| Any non-standard or proprietary API | `CustomModelProvider(...)` | User adapter | User adapter | User adapter | User adapter | User adapter |

“Model-dependent” means Praxium sends and parses that part of the protocol, but the
selected model and account must support it.

## The provider/model relationship

The provider's `name` must equal `Model.provider`. The model slug is otherwise
unrestricted:

```python
from praxium import Model, ModelProviderRegistry
from praxium.providers import ProviderFactory

provider = ProviderFactory.ollama()
providers = ModelProviderRegistry([provider])

model = Model(
    name="qwen3:8b",  # Passed to Ollama unchanged
    provider="ollama",  # Selects the registered provider
)
```

You can register multiple accounts, regions, or endpoints by assigning distinct
provider names:

```python
from praxium.providers import ProviderFactory

eu = ProviderFactory.openai(
    provider_name="openai-eu",
    api_key="...",
    base_url="https://eu-gateway.example/v1",
)
us = ProviderFactory.openai(
    provider_name="openai-us",
    api_key="...",
    base_url="https://us-gateway.example/v1",
)

providers = ModelProviderRegistry([eu, us])
```

## Run an agent with a custom Python tool

Tools are already provider-neutral. Praxium generates their JSON Schema, sends it
using the selected provider's format, executes the call, and translates the result
back for the next model turn.

```python
import asyncio
import os

from praxium import Agent, AgentRunner, Model, ModelProviderRegistry, Tool
from praxium.providers import ProviderFactory


def multiply(left: int, right: int) -> int:
    """Multiply two integers."""

    return left * right


async def main() -> None:
    provider = ProviderFactory.openai(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    runner = AgentRunner(ModelProviderRegistry([provider]))
    agent = Agent(
        name="calculator",
        instructions="Use the arithmetic tool instead of calculating mentally.",
        model=Model(name="your-enabled-gpt-model", provider="openai"),
        tools=[Tool.from_callable(multiply)],
    )

    result = await runner.run(agent, "What is 37 multiplied by 19?")
    print(result.response.text_content)
    print(result.tool_results)


asyncio.run(main())
```

The same `Agent`, `Tool`, and `AgentRunner` code works when only the provider and
model configuration changes.

## OpenAI and GPT models

Set `OPENAI_API_KEY`, create the provider, and use any model available to the
account:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.openai()
model = Model(name="your-enabled-gpt-model", provider=provider.name)
```

The OpenAI preset uses `/v1/chat/completions`, function tools, streamed SSE chunks,
JSON Schema response format, and `/v1/embeddings`. Use `Model.parameters` for
provider parameters that Praxium does not model directly:

```python
model = Model(
    name="your-enabled-gpt-model",
    provider="openai",
    temperature=0.2,
    max_output_tokens=2000,
    parameters={"seed": 7, "top_p": 0.9},
)
```

## Anthropic and Claude models

Set `ANTHROPIC_API_KEY`:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.anthropic()
model = Model(name="your-enabled-claude-model", provider="anthropic")
```

This is a native adapter. It maps system instructions to Anthropic's top-level
system field, function definitions to `input_schema`, assistant calls to
`tool_use`, results to `tool_result`, and partial tool arguments from streaming
events. Anthropic's Messages API has no embeddings endpoint, so register a second
provider for RAG embeddings.

## Google Gemini

Set `GEMINI_API_KEY`:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.gemini()
chat_model = Model(name="your-enabled-gemini-model", provider="gemini")
embedding_model = Model(name="your-gemini-embedding-model", provider="gemini")
```

The native adapter handles `generateContent`, SSE `streamGenerateContent`, Gemini
function calls and responses, JSON Schema generation configuration, and batched
`embedContent` requests.

## Azure OpenAI

Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`. `Model.name` must be your
Azure deployment name, not merely the underlying model family:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.azure_openai()
model = Model(name="my-production-deployment", provider=provider.name)
```

The factory accepts either the resource root
(`https://RESOURCE.openai.azure.com`) or the complete `/openai/v1` base URL. For
Microsoft Entra ID instead of an API key, install `praxium[azure]` and supply a
renewable token provider:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from praxium.providers import ProviderFactory

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default",
)
provider = ProviderFactory.azure_openai(token_provider=token_provider)
```

Praxium invokes the provider for every request, so refreshed Entra tokens are used
without rebuilding the agent or provider.

## Amazon Bedrock

Install `praxium[aws]`, configure the standard AWS credential chain, and create the
native Converse adapter:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.bedrock(region_name="eu-central-1")
model = Model(name="your-converse-model-or-inference-profile-id", provider=provider.name)
```

The adapter delegates credential discovery, profiles, role credentials, refresh,
and SigV4 signing to Boto3. It maps Praxium messages, tools, tool results, JSON
Schema output, streaming events, usage, stop reasons, and safe AWS errors to the
provider-neutral contracts. Model IDs and inference-profile ARNs pass through
unchanged.

Bedrock embeddings are intentionally not presented as one universal preset:
`InvokeModel` request and response bodies differ between embedding model families.
Register a model-specific `CustomModelProvider` embedding handler instead. Chat
models not supported by Bedrock Converse require the same model-specific approach.

## Google Vertex AI

Install `praxium[gcp,providers]`, configure Application Default Credentials, then
select the project and region:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.vertex_ai(
    project="my-gcp-project",
    location="europe-west4",
)
model = Model(name="your-google-publisher-model", provider=provider.name)
```

`project` can instead come from `GOOGLE_CLOUD_PROJECT`; location defaults to
`GOOGLE_CLOUD_LOCATION` and then `us-central1`. The adapter refreshes ADC tokens,
uses Vertex `generateContent`/`streamGenerateContent`, maps native function calls,
and uses the Vertex `predict` contract for text embeddings. It targets Google
publisher models; non-Google publisher endpoints can be implemented through the
public provider protocol or `CustomModelProvider`.

## Groq, Together AI, and OpenRouter

These hosted OpenAI-compatible services now have direct one-line presets:

```python
from praxium.providers import ProviderFactory

groq = ProviderFactory.groq()  # GROQ_API_KEY
together = ProviderFactory.together()  # TOGETHER_API_KEY
openrouter = ProviderFactory.openrouter()  # OPENROUTER_API_KEY
```

OpenRouter also accepts the optional attribution headers without manual adapter
configuration:

```python
openrouter = ProviderFactory.openrouter(
    site_url="https://your-app.example",
    app_name="Your App",
)
```

Groq's named preset is chat-only. Together and OpenRouter expose the compatible
embedding endpoint. Tools and structured output still depend on the selected model
and provider route.

## Kimi

Set `MOONSHOT_API_KEY`:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.kimi()
model = Model(name="your-enabled-kimi-model", provider="kimi")
```

The preset targets `https://api.moonshot.ai/v1` and uses Kimi's compatible Chat
Completions and tool format. The model identifier is not hardcoded.

## GLM

Set `ZHIPUAI_API_KEY`:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.glm()
model = Model(name="your-enabled-glm-model", provider="glm")
```

The preset targets Zhipu's `/api/paas/v4` compatible surface. Provider-specific
features remain available through `Model.parameters`:

```python
model = Model(
    name="your-enabled-glm-model",
    provider="glm",
    parameters={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
)
```

## Ollama and local models

Start Ollama and pull the model before running the agent:

```bash
ollama pull qwen3:8b
```

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.ollama()
model = Model(name="qwen3:8b", provider="ollama")
```

For a remote Ollama daemon or reverse proxy:

```python
provider = ProviderFactory.ollama(base_url="http://model-host:11434/v1")
```

Ollama exposes compatible chat, tools, JSON output, streaming, and embeddings, but
each local model has its own capabilities. Praxium does not download models or
silently replace a requested model.

## Hugging Face Inference Providers

Set `HF_TOKEN` to a fine-grained token with inference permission:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.huggingface()
model = Model(
    name="organization/model-name:fastest",
    provider="huggingface",
)
```

Hugging Face can route a model to the fastest, cheapest, preferred, or explicitly
selected inference provider by using the corresponding model suffix. Its
OpenAI-compatible router is chat-only; use a custom provider or another embedding
provider for non-chat inference tasks.

## Any OpenAI-compatible endpoint

Use the generic factory for vLLM, LM Studio, LocalAI, hosted gateways, fine-tuned
deployments, or a provider released after this version of Praxium:

```python
from praxium import Model
from praxium.providers import ProviderFactory

provider = ProviderFactory.openai_compatible(
    provider_name="private-llm",
    base_url="https://models.example.com/v1",
    api_key_env="PRIVATE_LLM_API_KEY",
)
model = Model(
    name="team/fine-tuned-model-2026-08",
    provider="private-llm",
)
```

The lower-level constructor exposes custom paths, headers, authentication schemes,
query parameters, defaults, and feature switches:

```python
from praxium.providers import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    provider_name="enterprise-models",
    base_url="https://gateway.example.com",
    api_key_env="ENTERPRISE_LLM_KEY",
    auth_header="x-api-key",
    auth_scheme="",
    chat_path="/inference/chat/completions",
    embeddings_path=None,
    query={"api-version": "2026-01-01"},
    headers={"x-tenant": "tenant-1"},
    default_parameters={"top_p": 0.9},
    include_stream_usage=False,
)
```

Do not put authorization secrets in `headers` when configuration might be logged by
application code. Prefer `api_key` or `api_key_env`; provider `repr` output omits
credentials.

## A completely proprietary API

If an API is not covered by the compatible, Anthropic, Gemini/Vertex, or Bedrock
adapters, wrap its SDK or HTTP client. The complete handler receives a typed
`ModelRequest` and returns a typed `Response`:

```python
from collections.abc import AsyncIterator

from praxium import Message, Response, ResponseDelta
from praxium.models import EmbeddingRequest, EmbeddingResponse, ModelRequest
from praxium.providers import CustomModelProvider


async def complete(request: ModelRequest) -> Response:
    vendor_result = await proprietary_client.generate(
        model=request.model.name,
        prompt=request.messages[-1].text_content,
    )
    return Response(
        message=Message.assistant(vendor_result.text),
        model=request.model.name,
    )


async def stream(request: ModelRequest) -> AsyncIterator[ResponseDelta]:
    index = 0
    async for token in proprietary_client.stream(model=request.model.name):
        yield ResponseDelta(index=index, text=token)
        index += 1
    yield ResponseDelta(index=index, finish_reason="stop")


async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    vectors = await proprietary_client.embed(request.model.name, request.inputs)
    return EmbeddingResponse(embeddings=vectors, model=request.model.name)


provider = CustomModelProvider(
    name="proprietary",
    complete=complete,
    stream=stream,
    embed=embed,
)
```

Synchronous handlers are run in a worker thread. For a larger integration, implement
the public `ModelProvider` protocol directly instead of using callable wrappers.

## Structured output

Call a provider directly with a JSON Schema:

```python
from praxium import Message, Model, ModelRequest

response = await provider.complete(
    ModelRequest(
        model=Model(name="your-model", provider=provider.name),
        messages=[Message.user("Extract a city and country from: Berlin, Germany")],
        output_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        },
    )
)

structured = response.message.parts[0]
print(structured.data)
```

Praxium requests the provider's schema mode and parses the result into `JsonPart`.
An invalid JSON result raises `ModelProviderError` instead of returning malformed
structured data.

## Token streaming

```python
from praxium import Message, Model, ModelRequest

request = ModelRequest(
    model=Model(name="your-model", provider=provider.name),
    messages=[Message.user("Write one paragraph.")],
)

async for delta in provider.stream(request):
    if delta.text:
        print(delta.text, end="", flush=True)
    if delta.tool_call:
        print("tool:", delta.tool_call.tool_name, delta.tool_call.arguments)
    if delta.finish_reason:
        print("\nfinish:", delta.finish_reason, "usage:", delta.usage)
```

Text deltas preserve whitespace. Partial function arguments are buffered until they
form one valid provider-neutral `ToolCallPart`.

## Embeddings

```python
from praxium import Model
from praxium.models import EmbeddingRequest

embedding = await provider.embed(
    EmbeddingRequest(
        model=Model(name="your-embedding-model", provider=provider.name),
        inputs=["first document", "second document"],
        dimensions=512,
    )
)

assert len(embedding.embeddings) == 2
```

The dimensions option is sent when the endpoint supports it. Chat-only presets
raise `UnsupportedFeatureError` immediately rather than issuing a known-invalid
request.

## Error handling

Provider failures use a stable typed error:

```python
from praxium import ModelProviderError

try:
    response = await provider.complete(request)
except ModelProviderError as exc:
    print(exc.provider, exc.status_code, exc.retryable)
    safe_payload = exc.to_detail().model_dump()
```

HTTP 408, 409, 425, 429, and 5xx responses are marked retryable. Error details
include only the provider name, status, and a bounded provider message; API keys,
authorization headers, raw request configuration, and response bodies are excluded.

Use `Model.timeout_seconds` to override the provider's default timeout for one
model configuration.

## Testing without API calls

Use `DeterministicModelProvider` for application tests. Provider adapter contract
tests can inject an object implementing `HTTPTransport`, as Praxium's own tests do,
so no live key or paid request is needed.

```python
from praxium import DeterministicModelProvider, ModelProviderRegistry, text_response

provider = DeterministicModelProvider(responses=[text_response("offline answer")])
registry = ModelProviderRegistry([provider])
```

The production adapters and deterministic provider implement the same
`ModelProvider` protocol.
