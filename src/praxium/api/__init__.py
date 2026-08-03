"""Application registry and optional HTTP adapters."""

from typing import Any

from .application import Application, ComponentDescription, ComponentKind, result_text


def create_fastapi_app(application: Application, *, title: str = "Praxium") -> Any:
    """Load the optional FastAPI adapter only when called."""

    from .fastapi import create_fastapi_app as factory

    return factory(application, title=title)


__all__ = [
    "Application",
    "ComponentDescription",
    "ComponentKind",
    "create_fastapi_app",
    "result_text",
]
