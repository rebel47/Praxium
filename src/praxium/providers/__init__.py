"""Production model provider adapters.

Install the HTTP dependency with ``pip install "praxium[providers]"``. Importing
this package itself does not create clients, read secrets, or perform network I/O.
"""

from ._http import HTTPResponse, HTTPTransport, HTTPXTransport, SSEEvent
from .anthropic import AnthropicProvider, anthropic_messages
from .bedrock import BedrockProvider, bedrock_messages
from .custom import CustomModelProvider
from .factory import ProviderFactory
from .gemini import GeminiProvider, gemini_contents
from .openai_compatible import OpenAICompatibleProvider, openai_messages
from .vertex import VertexAIProvider

__all__ = [
    "AnthropicProvider",
    "BedrockProvider",
    "CustomModelProvider",
    "GeminiProvider",
    "HTTPResponse",
    "HTTPTransport",
    "HTTPXTransport",
    "OpenAICompatibleProvider",
    "ProviderFactory",
    "SSEEvent",
    "VertexAIProvider",
    "anthropic_messages",
    "bedrock_messages",
    "gemini_contents",
    "openai_messages",
]
