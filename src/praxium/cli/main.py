"""Dependency-free command line interface for core developer workflows."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
from typing import Any

from praxium._version import __version__
from praxium.graph import Graph
from praxium.plugins import PluginLoader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="praxium", description="Praxium CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="check the development environment")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    graph = subcommands.add_parser("graph", help="render a Graph object as Mermaid")
    graph.add_argument("target", help="Python target in module:attribute form")
    graph.add_argument("--output", choices=["mermaid", "json"], default="mermaid")

    plugins = subcommands.add_parser("plugins", help="list discovered plugin entry points")
    plugins.add_argument("--json", action="store_true", dest="as_json")

    serve = subcommands.add_parser("serve", help="serve an Application target with FastAPI")
    serve.add_argument("target", help="Python target in module:attribute form")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.as_json)
    if args.command == "graph":
        graph = _load_target(args.target)
        if not isinstance(graph, Graph):
            raise SystemExit(f"target {args.target!r} is not a Graph")
        print(graph.to_mermaid() if args.output == "mermaid" else graph.model_dump_json(indent=2))
        return 0
    if args.command == "plugins":
        values = [item.model_dump(mode="json") for item in PluginLoader().discover()]
        if args.as_json:
            print(json.dumps(values, indent=2))
        elif not values:
            print("No Praxium plugins discovered.")
        else:
            for value in values:
                print(f"{value['name']}\t{value['value']}\t{value.get('distribution') or '-'}")
        return 0
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("serve requires: pip install 'praxium[api]'") from exc
        from praxium.api import Application, create_fastapi_app

        application = _load_target(args.target)
        if not isinstance(application, Application):
            raise SystemExit(f"target {args.target!r} is not an Application")
        uvicorn.run(create_fastapi_app(application), host=args.host, port=args.port)
        return 0
    return 2


def _doctor(as_json: bool) -> int:
    checks = {
        "python": {
            "ok": sys.version_info >= (3, 11),
            "value": platform.python_version(),
            "required": ">=3.11",
        },
        "pydantic": _module_check("pydantic", required=True),
        "fastapi": _module_check("fastapi", required=False),
        "uvicorn": _module_check("uvicorn", required=False),
        "asyncpg": _module_check("asyncpg", required=False),
        "opentelemetry": _module_check("opentelemetry", required=False),
    }
    required_ok = checks["python"]["ok"] and checks["pydantic"]["ok"]
    if as_json:
        print(json.dumps({"ok": required_ok, "checks": checks}, indent=2))
    else:
        for name, check in checks.items():
            status = "OK" if check["ok"] else ("MISSING" if not check.get("required") else "ERROR")
            print(f"{status:7} {name:16} {check.get('value', '')}")
    return 0 if required_ok else 1


def _module_check(name: str, *, required: bool) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    value = None
    if spec is not None:
        module = importlib.import_module(name)
        value = getattr(module, "__version__", "installed")
    return {"ok": spec is not None, "value": value, "required": required}


def _load_target(target: str) -> Any:
    if ":" not in target:
        raise SystemExit("target must use module:attribute syntax")
    module_name, attribute = target.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise SystemExit(f"module {module_name!r} has no attribute {attribute!r}") from exc
